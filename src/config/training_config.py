"""
Training Configuration

This module defines training hyperparameters and settings for LLM training.

Educational Notes:
- batch_size: Number of sequences processed in parallel
- gradient_accumulation: How many steps to accumulate gradients before updating
                Effective batch size = batch_size × gradient_accumulation × num_gpus
- learning_rate: Step size for optimizer updates
- warmup_steps: Gradually increase LR at start for training stability
- max_steps: Total training iterations (or use num_epochs)
- weight_decay: L2 regularization to prevent overfitting
- max_sequence_length: Context window (should match model config)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration for training a language model.

    Attributes:
        # Data settings
        batch_size: Number of sequences per batch
        gradient_accumulation: Steps to accumulate before optimizer update
        max_sequence_length: Maximum token sequence length

        # Optimization settings
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup steps
        max_steps: Maximum training steps
        num_epochs: Alternative to max_steps - train for N epochs
        weight_decay: L2 regularization coefficient
        gradient_clip: Clip gradients at this norm (prevents exploding)

        # Learning rate schedule
        min_lr: Minimum learning rate (after decay)
        lr_decay_type: 'cosine' or 'linear' decay

        # Precision and memory
        use_fp16: Use mixed precision (FP16) for faster training
        dtype: Data type for model ('float32', 'float16', 'bfloat16')

        # Checkpointing
        checkpoint_every: Save checkpoint every N steps
        save_total_limit: Keep only last N checkpoints

        # Validation
        validation_every: Run validation every N steps
        validation_steps: Number of validation steps

        # Logging
        log_every: Log training metrics every N steps

        # Reproducibility
        seed: Random seed for reproducibility
    """

    # Data settings
    batch_size: int = 32
    gradient_accumulation: int = 4
    max_sequence_length: int = 512

    # Optimization
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    max_steps: Optional[int] = 50000
    num_epochs: Optional[int] = None
    weight_decay: float = 0.1
    gradient_clip: float = 1.0

    # Learning rate schedule
    min_lr: float = 1e-5
    lr_decay_type: str = "cosine"  # 'cosine' or 'linear'

    # Precision
    use_fp16: bool = True
    dtype: str = "float16"  # Will be overridden if use_fp16 is True

    # Checkpointing
    checkpoint_every: int = 5000
    save_total_limit: int = 3

    # Validation
    validation_every: int = 1000
    validation_steps: int = 100

    # Logging
    log_every: int = 100

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """
        Validate configuration after initialization.

        Educational note: Ensures settings are consistent and valid.
        """
        # Override dtype if fp16 is enabled
        if self.use_fp16:
            self.dtype = "float16"

        # Validate learning rate decay type
        assert self.lr_decay_type in [
            "cosine",
            "linear",
        ], f"lr_decay_type must be 'cosine' or 'linear', got {self.lr_decay_type}"

        # Ensure either max_steps or num_epochs is set
        assert (
            self.max_steps is not None or self.num_epochs is not None
        ), "Must set either max_steps or num_epochs"

    @property
    def effective_batch_size(self) -> int:
        """
        The effective batch size after gradient accumulation.

        Educational note: Gradient accumulation allows training with larger
        effective batch sizes than GPU memory would normally allow.
        We accumulate gradients over multiple steps before updating.

        Formula: batch_size × gradient_accumulation × num_gpus
        """
        return self.batch_size * self.gradient_accumulation

    @property
    def total_steps(self) -> int:
        """
        Total training steps (either from max_steps or computed from epochs).

        Educational note: Training can be controlled by either total steps
        or number of epochs through the dataset. This property unifies both.
        """
        if self.max_steps is not None:
            return self.max_steps
        # If num_epochs is set, it will be computed from dataset size during training
        return self.num_epochs  # type: ignore


# Pre-configured training presets for different model sizes

TRAINING_CONFIGS = {
    "pico": TrainingConfig(
        batch_size=16,
        gradient_accumulation=1,
        learning_rate=1e-3,
        warmup_steps=100,
        max_steps=5000,
        # For quick testing (5K steps = ~5 minutes on GPU)
    ),
    "nano": TrainingConfig(
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        warmup_steps=200,
        max_steps=10000,
        # For fast iteration (10K steps = ~15 minutes on GPU)
    ),
    "micro": TrainingConfig(
        batch_size=32,
        gradient_accumulation=2,
        learning_rate=1e-3,
        warmup_steps=500,
        max_steps=20000,
        # Small but usable (20K steps = ~30 minutes on GPU)
    ),
    "tiny": TrainingConfig(
        batch_size=32,
        gradient_accumulation=2,
        learning_rate=1e-3,
        warmup_steps=500,
        max_steps=20000,
    ),
    "small": TrainingConfig(
        batch_size=32,
        gradient_accumulation=4,
        learning_rate=1e-3,
        warmup_steps=1000,
        max_steps=50000,
        # Recommended config for 25M model
    ),
    "medium": TrainingConfig(
        batch_size=16,
        gradient_accumulation=8,
        learning_rate=8e-4,
        warmup_steps=2000,
        max_steps=100000,
        # Larger model needs smaller batch size but more accumulation
    ),
    "large": TrainingConfig(
        batch_size=16,
        gradient_accumulation=8,
        learning_rate=5e-4,
        warmup_steps=2000,
        max_steps=150000,
        # Largest model needs conservative settings
    ),
}


def get_training_config(model_size: str = "small") -> TrainingConfig:
    """
    Get training configuration appropriate for a given model size.

    Args:
        model_size: One of 'pico', 'nano', 'micro', 'tiny', 'small', 'medium', 'large'

    Returns:
        TrainingConfig object with appropriate hyperparameters

    Example:
        >>> config = get_training_config("small")
        >>> print(f"Effective batch size: {config.effective_batch_size}")
        Effective batch size: 128
    """
    if model_size not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(TRAINING_CONFIGS.keys())}")

    config = TRAINING_CONFIGS[model_size]

    print(f"Loaded training config for '{model_size}' model:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Mixed precision: {config.use_fp16}")

    return config


def print_all_configs():
    """
    Print all available training configurations for comparison.
    """
    print("\nAvailable Training Configurations:")
    print("=" * 70)

    for name, config in TRAINING_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation}")
        print(f"  Effective batch size: {config.effective_batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Max steps: {config.max_steps}")
        print(f"  Mixed precision: {config.use_fp16}")

    print("=" * 70)


if __name__ == "__main__":
    # Educational: Show all configurations
    print_all_configs()

    # Educational: Demonstrate getting a specific config
    print("\n" + "=" * 70)
    config = get_training_config("small")
    print("=" * 70)
