"""
Model Architecture Configurations

This module defines pre-configured model architectures of different sizes.
Each configuration specifies the dimensions of a transformer-based LLM.

Educational Notes:
- vocab_size: Number of unique tokens the model can understand
- max_sequence_length: Maximum context window (in tokens)
- dim: Model dimensionality (hidden size)
- num_heads: Number of attention heads
- num_layers: Number of transformer blocks stacked
- hidden_dim: Feed-forward network dimension (usually 4x dim for SwiGLU)
- num_kv_heads: Reserved for future grouped-query attention (GQA) support.
                  NOTE: the current attention module (src/model/attention.py)
                  implements standard multi-head attention only. This field is
                  validated but not yet wired into the model, so leave it None.

Parameter Estimation Formula (approximate):
params ≈ 12 * num_layers * dim^2  # For standard transformers
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for a transformer-based language model.

    Attributes:
        vocab_size: Size of the vocabulary
        max_sequence_length: Maximum sequence length for training/inference
        dim: Model dimension (hidden size)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        hidden_dim: Feed-forward network hidden dimension
        num_kv_heads: Reserved for future grouped-query attention (GQA).
                      Not yet used by the attention module; keep as None.
        use_rope: Whether to use Rotary Position Embeddings
        rope_theta: RoPE base frequency (10,000 is common)
        use_tied_embeddings: Whether to tie input and output embeddings
        dropout: Dropout rate (0.0 = no dropout)
    """

    vocab_size: int
    max_sequence_length: int
    dim: int
    num_heads: int
    num_layers: int
    hidden_dim: Optional[int] = None
    num_kv_heads: Optional[int] = None
    use_rope: bool = True
    rope_theta: float = 10000.0
    use_tied_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        """
        Validate and compute derived dimensions after initialization.

        Educational note: This ensures the configuration is valid and
        automatically computes hidden_dim if not explicitly set.
        """
        # Validate dimensions are divisible
        assert (
            self.dim % self.num_heads == 0
        ), f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"

        # Validate GQA settings
        if self.num_kv_heads is not None:
            assert (
                self.num_heads % self.num_kv_heads == 0
            ), f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

        # Compute hidden_dim if not provided (typically 4x dim for SwiGLU)
        if self.hidden_dim is None:
            # For SwiGLU activation, hidden_dim is typically 4/3 * 4 * dim
            # But we use simpler 4x dim approximation
            self.hidden_dim = 4 * self.dim

    @property
    def head_dim(self) -> int:
        """
        Dimension of each attention head.

        Educational note: In multi-head attention, the model dimension is
        split across multiple heads, each working with head_dim dimensions.
        This allows the model to learn different types of relationships
        in different heads in parallel.
        """
        return self.dim // self.num_heads

    @property
    def num_params(self) -> int:
        """
        Estimate the total number of parameters in the model.

        Educational note: This is a rough estimate used for understanding
        model size. The actual count will differ slightly due to layer
        normalization parameters and embedding layers.
        """
        # Attention parameters: Q, K, V projections and output projection
        attn_params = 4 * self.dim * self.dim * self.num_layers

        # Feed-forward parameters
        ffn_params = 2 * self.dim * self.hidden_dim * self.num_layers

        # Embedding parameters (tied embeddings = counted once)
        emb_params = self.dim * self.vocab_size if self.use_tied_embeddings else 2 * self.dim * self.vocab_size

        # Layer normalization parameters (small, ~8 * dim per layer)
        ln_params = 8 * self.dim * self.num_layers

        return attn_params + ffn_params + emb_params + ln_params


# Pre-configured model sizes
# These are designed for educational purposes with different trade-offs

MODEL_CONFIGS = {
    "pico": ModelConfig(
        vocab_size=32000,
        max_sequence_length=256,
        dim=64,
        num_heads=4,
        num_layers=4,
        # ~1M parameters - Minimal viable model for quick testing (~30 seconds on CPU)
    ),
    "nano": ModelConfig(
        vocab_size=32000,
        max_sequence_length=256,
        dim=128,
        num_heads=8,
        num_layers=6,
        # ~3M parameters - Very small but functional, good for CPU testing
    ),
    "micro": ModelConfig(
        vocab_size=32000,
        max_sequence_length=512,
        dim=192,
        num_heads=12,
        num_layers=8,
        # ~6M parameters - Small but usable, good for fast iteration
    ),
    "tiny": ModelConfig(
        vocab_size=32000,
        max_sequence_length=512,
        dim=256,
        num_heads=8,
        num_layers=6,
        # ~10M parameters - Fastest for experimentation and debugging
    ),
    "small": ModelConfig(
        vocab_size=32000,
        max_sequence_length=512,
        dim=384,
        num_heads=12,
        num_layers=8,
        # ~25M parameters - Recommended starting point for education
    ),
    "medium": ModelConfig(
        vocab_size=32000,
        max_sequence_length=512,
        dim=512,
        num_heads=16,
        num_layers=12,
        # ~50M parameters - More realistic, demonstrates scaling
    ),
    "large": ModelConfig(
        vocab_size=32000,
        max_sequence_length=512,
        dim=640,
        num_heads=20,
        num_layers=16,
        # ~100M parameters - Upper limit for 6GB VRAM
    ),
}


def get_config(size: str = "small") -> ModelConfig:
    """
    Get a pre-configured model by size name.

    Args:
        size: One of 'pico', 'nano', 'micro', 'tiny', 'small', 'medium', 'large'

    Returns:
        ModelConfig object with the specified architecture

    Raises:
        ValueError: If size is not recognized

    Example:
        >>> config = get_config("small")
        >>> print(f"Model size: {config.num_params:,} parameters")
        Model size: 25,000,000 parameters
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Choose from: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[size]

    # Print model info for educational purposes
    print(f"Loaded '{size}' model configuration:")
    print(f"  Parameters: ~{config.num_params:,}")
    print(f"  Architecture: {config.num_layers} layers x {config.dim} dim x {config.num_heads} heads")
    print(f"  Max sequence length: {config.max_sequence_length}")
    print(f"  Vocabulary size: {config.vocab_size}")

    return config


def print_all_configs():
    """
    Print all available model configurations for comparison.

    Educational note: This function helps understand the scaling properties
    of transformer models - how parameter count scales with dimensions.
    """
    print("\nAvailable Model Configurations:")
    print("=" * 70)

    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Parameters: ~{config.num_params:,}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Dimension: {config.dim}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Sequence length: {config.max_sequence_length}")

    print("=" * 70)


if __name__ == "__main__":
    # Educational: Show all available configurations
    print_all_configs()

    # Educational: Demonstrate getting a specific config
    print("\n" + "=" * 70)
    config = get_config("small")
    print("=" * 70)
