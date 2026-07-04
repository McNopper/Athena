"""
Main LLM Model

This module implements the complete language model, assembling all components
into a functional transformer-based LLM.

Educational Notes:
- This is the main model class that users interact with
- Assembles: token embeddings → transformer stack → output projection
- Can be used for both training and inference
- Follows the architecture of modern LLMs (LLaMA, GPT, etc.)

Complete Architecture:
    Input (token IDs)
        ↓
    Token Embeddings (learned vector for each token)
        ↓
    [Repeated N times]
    → Transformer Block (Attention + FFN)
        ↓
    RMSNorm (final normalization)
        ↓
    Output Projection (vocab scores for next token)
        ↓
    Output (logits over vocabulary)

Model Sizes:
- tiny: 10M params (256 dim, 8 heads, 6 layers) - Fastest
- small: 25M params (384 dim, 12 heads, 8 layers) - Recommended
- medium: 50M params (512 dim, 16 heads, 12 layers)
- large: 100M params (640 dim, 20 heads, 16 layers)

Training vs Inference:
- Training: Compute loss over entire sequence
- Inference: Generate token by token (autoregressive)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..config.model_config import ModelConfig
from .embeddings import TokenEmbedding, OutputProjection
from .transformer import TransformerStack
from .layer_norm import RMSNorm
from ..inference.kv_cache import KVCache


class LLM(nn.Module):
    """
    Complete Language Model for training and inference.

    Educational notes:
    - Assembles all components into a functional LLM
    - Architecture: Embeddings → Transformer Stack → Output Projection
    - Can be used for both training and inference
    - Follows modern LLM design (LLaMA-style)

    Forward pass:
        Input tokens → Embeddings → Transformer → Norm → Output → Logits

    Args:
        config: ModelConfig object with architecture specifications

    Example:
        >>> from src.config.model_config import get_config
        >>> config = get_config("small")  # 25M parameters
        >>> model = LLM(config)
        >>> tokens = torch.randint(0, config.vocab_size, (32, 128))
        >>> logits = model(tokens)
        >>> print(logits.shape)
        torch.Size([32, 128, 32000])
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Educational note: Token embeddings
        # Converts discrete token IDs to continuous vectors
        # Shape: (vocab_size, dim)
        self.token_embeddings = TokenEmbedding(
            vocab_size=config.vocab_size,
            dim=config.dim,
        )

        # Educational note: Transformer stack
        # The core of the model - N transformer blocks stacked
        # Each block: Attention + FFN with residual connections
        self.transformer = TransformerStack(
            num_layers=config.num_layers,
            dim=config.dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            max_sequence_length=config.max_sequence_length,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        # Educational note: Final normalization
        # Applied after transformer stack before output projection
        self.norm = RMSNorm(config.dim)

        # Educational note: Output projection
        # Projects from hidden dimension back to vocabulary size
        # Can use tied embeddings (share weights with input embeddings)
        self.output_projection = OutputProjection(
            vocab_size=config.vocab_size,
            dim=config.dim,
            tied_embeddings=config.use_tied_embeddings,
            input_embeddings=self.token_embeddings.token_embedding if config.use_tied_embeddings else None,
        )

        # Educational note: Weight initialization matters a lot. PyTorch's default
        # nn.Embedding initialization is N(0, 1), which — combined with tied
        # embeddings used as the output projection — produces very large initial
        # logits and an implausibly high starting loss. Modern LLMs (GPT-2,
        # LLaMA) initialize weights from a small normal, N(0, 0.02), so the
        # initial loss is close to ln(vocab_size) and training is stable.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize Linear and Embedding weights from N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through the language model.

        Educational notes:
        - Converts tokens to embeddings
        - Passes through transformer stack
        - Projects to vocabulary logits
        - Used for both training and inference

        Args:
            token_ids: Input token IDs
                      Shape: (batch_size, seq_len)
            mask: Optional key-padding mask
                  Shape: (batch_size, seq_len)
            kv_cache: Optional KV cache for efficient incremental decoding.
            start_pos: Absolute position of the first token in ``token_ids``
                       (non-zero when continuing generation with a KV cache).

        Returns:
            Logits over vocabulary
            Shape: (batch_size, seq_len, vocab_size)

        Example:
            >>> model = LLM(config)
            >>> tokens = torch.randint(0, config.vocab_size, (32, 128))
            >>> logits = model(tokens)
            >>> print(logits.shape)
            torch.Size([32, 128, 32000])
        """
        # Step 1: Convert token IDs to embeddings
        # Educational note: Look up learned vector for each token
        # Shape: (batch, seq_len, dim)
        x = self.token_embeddings(token_ids)

        # Step 2: Pass through transformer stack
        # Educational note: Each layer refines the representations
        # Shape: (batch, seq_len, dim)
        x = self.transformer(x, mask=mask, kv_cache=kv_cache, start_pos=start_pos)

        # Step 3: Apply final normalization
        # Educational note: Stabilizes output before projection
        # Shape: (batch, seq_len, dim)
        x = self.norm(x)

        # Step 4: Project to vocabulary logits
        # Educational note: Compute scores for each possible next token
        # Shape: (batch, seq_len, vocab_size)
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively (inference mode).

        Educational notes:
        - Generates token by token (autoregressive)
        - Each new token becomes input for next prediction
        - Sampling strategies: greedy, top-k, top-p, temperature
        - Uses a KV cache so each step only processes the newly added token,
          turning generation from O(n^2) into O(n)

        Args:
            prompt: Input prompt tokens
                   Shape: (batch_size, prompt_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, sample from top k tokens
            top_p: If set, use nucleus sampling
            do_sample: If True, use sampling; if False, use greedy decoding

        Returns:
            Generated tokens (prompt + generated)
            Shape: (batch_size, prompt_len + generated_len)

        Example:
            >>> model = LLM(config)
            >>> prompt = torch.tensor([[1, 2, 3]])  # Start tokens
            >>> generated = model.generate(prompt, max_new_tokens=50)
            >>> print(generated.shape)
            torch.Size([1, 53])
        """
        self.eval()  # Set to evaluation mode

        # Robustness: temperature <= 0 is commonly used to mean "greedy" — honour
        # that instead of dividing logits by zero. top_k <= 0 disables top-k.
        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = 1.0
        if top_k is not None and top_k <= 0:
            top_k = None

        with torch.no_grad():  # No need for gradients during inference
            # Educational note: One KV cache holds keys/values for every layer.
            kv_cache = KVCache(
                batch_size=prompt.shape[0],
                num_heads=self.config.num_heads,
                head_dim=self.config.dim // self.config.num_heads,
                max_sequence_length=self.config.max_sequence_length,
                dtype=next(self.parameters()).dtype,
                device=prompt.device,
            )

            # Start with prompt
            generated = prompt
            next_input = prompt
            start_pos = 0

            for _ in range(max_new_tokens):
                # Forward pass. First step processes the whole prompt; later
                # steps process only the single new token, reusing the cache.
                logits = self(next_input, kv_cache=kv_cache, start_pos=start_pos)

                # Get logits for last position only (next token prediction)
                next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    k = min(top_k, next_token_logits.shape[-1])
                    values, _ = torch.topk(next_token_logits, k)
                    min_values = values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_values,
                        torch.full_like(next_token_logits, float("-inf")),
                        next_token_logits,
                    )

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # Scatter to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)

                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Next step: feed only the new token at its absolute position.
                start_pos = generated.shape[1] - 1
                next_input = next_token

        return generated

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"vocab_size={self.config.vocab_size}, dim={self.config.dim}, num_layers={self.config.num_layers}"

    @property
    def num_parameters(self) -> int:
        """
        Total number of trainable parameters.

        Educational note: Useful for understanding model size and capacity.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_llm():
    """
    Test LLM implementation.

    Educational note: Tests verify:
    1. Model can be instantiated
    2. Forward pass produces correct shapes
    3. Generation works
    4. Parameter count is reasonable
    """
    print("Testing LLM implementation...")

    # Test 1: Model instantiation
    from src.config.model_config import get_config

    config = get_config("tiny")  # Use tiny config for fast testing

    model = LLM(config)
    print(f"✓ Test 1 passed: Model instantiated with {model.num_parameters:,} parameters")

    # Test 2: Forward pass
    batch_size = 4
    seq_len = 64
    vocab_size = config.vocab_size

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(token_ids)

    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Wrong shape: {logits.shape} != {expected_shape}"
    print(f"✓ Test 2 passed: Forward pass produces correct shape {logits.shape}")

    # Test 3: Generation
    prompt_len = 10
    prompt = torch.randint(0, vocab_size, (2, prompt_len))

    generated = model.generate(prompt, max_new_tokens=20, do_sample=False)

    expected_len = prompt_len + 20
    assert generated.shape == (2, expected_len), f"Wrong generation shape: {generated.shape}"
    assert torch.equal(generated[:, :prompt_len], prompt), "Prompt was modified"
    print(f"✓ Test 3 passed: Generation works (generated {expected_len - prompt_len} tokens)")

    # Test 4: Gradient flow
    loss = logits.sum()
    loss.backward()

    assert model.token_embeddings.token_embedding.weight.grad is not None, "No gradients on embeddings"
    assert model.transformer.blocks[0].attention.qkv_proj.weight.grad is not None, "No gradients on attention"
    assert not torch.isnan(model.token_embeddings.token_embedding.weight.grad).any(), "Gradients contain NaN"
    print("✓ Test 4 passed: Gradients flow correctly")

    # Test 5: Parameter count
    params = model.num_parameters
    print(f"✓ Test 5 passed: Model has {params:,} parameters")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_llm()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Language Model Demonstration")
    print("=" * 70)

    # Create model
    from src.config.model_config import get_config

    config = get_config("tiny")
    model = LLM(config)

    print(f"\nModel Architecture:")
    print(model)

    # Create sample input
    batch_size = 2
    seq_len = 64
    vocab_size = config.vocab_size

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput token IDs shape: {token_ids.shape}")

    # Forward pass
    logits = model(token_ids)

    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Generation demo
    print("\n" + "=" * 70)
    print("Text Generation Demo")
    print("=" * 70)

    prompt_len = 5
    prompt = torch.randint(0, vocab_size, (1, prompt_len))

    print(f"\nPrompt length: {prompt_len}")

    # Greedy generation
    generated_greedy = model.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"Greedy generation: {generated_greedy.shape}")

    # Sampling generation
    generated_sampled = model.generate(prompt, max_new_tokens=10, do_sample=True, temperature=0.8)
    print(f"Sampled generation: {generated_sampled.shape}")

    # Parameter breakdown
    params_emb = sum(p.numel() for p in model.token_embeddings.parameters())
    params_transformer = sum(p.numel() for p in model.transformer.parameters())
    params_output = sum(p.numel() for p in model.output_projection.parameters())
    params_total = model.num_parameters

    print(f"\nParameter breakdown:")
    print(f"  Token embeddings: {params_emb:,} ({params_emb/params_total*100:.1f}%)")
    print(f"  Transformer stack: {params_transformer:,} ({params_transformer/params_total*100:.1f}%)")
    print(f"  Output projection: {params_output:,} ({params_output/params_total*100:.1f}%)")
    print(f"  Total: {params_total:,}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- LLM combines embeddings, transformer, and output projection")
    print("- Training: Compute loss over entire sequence")
    print("- Inference: Generate token by token (autoregressive)")
    print("- Deeper models (more layers) learn more complex patterns")
    print("- Tied embeddings save parameters (share input/output weights)")
    print("- This architecture powers GPT, LLaMA, Claude, and other modern LLMs")
    print("=" * 70)
