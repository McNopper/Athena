#!/usr/bin/env python3
"""
Training Script

This script trains a language model on tokenized data.

Usage:
    python scripts/train.py --config small --data-dir data/processed/

Educational Notes:
- Complete training pipeline
- Loads configuration, data, and model
- Runs training loop with validation
- Saves checkpoints and logs

Examples:
    # Quick test run
    python scripts/train.py --config pico --steps 100 --test-run

    # Full training
    python scripts/train.py --config small --data-dir data/processed/

    # Resume from checkpoint
    python scripts/train.py --config small --resume checkpoints/best.pt
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config.model_config import get_config
from src.config.training_config import get_training_config
from src.model.llm import LLM
from src.utils.device import get_device, set_seed, print_device_info
from src.training.dataset import load_sequences_from_file, create_dataloader
from src.training.optimizer import get_optimizer, get_cosine_schedule_with_warmup, OptimizerScheduler
from src.training.checkpoint import CheckpointManager
from src.training.trainer import LLMTrainer
from src.utils.logger import TrainingLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument("--config", type=str, default="small",
                       choices=["pico", "nano", "micro", "tiny", "small", "medium", "large"],
                       help="Model configuration")
    parser.add_argument("--config-custom", type=str, default=None,
                       help="Path to custom model config (JSON)")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Directory containing processed training data")
    parser.add_argument("--train-file", type=str, default="train_sequences.json",
                       help="Training sequences file")
    parser.add_argument("--val-file", type=str, default="val_sequences.json",
                       help="Validation sequences file")
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Vocabulary size (if not in config)")

    # Training settings
    parser.add_argument("--steps", type=int, default=None,
                       help="Maximum training steps (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--gradient-accumulation", type=int, default=None,
                       help="Gradient accumulation steps (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                       help="Warmup steps (overrides config)")
    parser.add_argument("--weight-decay", type=float, default=None,
                       help="Weight decay (overrides config)")
    parser.add_argument("--max-gradient-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")

    # Precision
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable mixed precision training")

    # Logging and checkpointing
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory to save logs")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log-every", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--validation-every", type=int, default=1000,
                       help="Validate every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                       help="Maximum number of checkpoints to keep")

    # Resumption
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    # Testing
    parser.add_argument("--test-run", action="store_true",
                       help="Quick test run with minimal steps")
    parser.add_argument("--no-validation", action="store_true",
                       help="Skip validation")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Misc
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "cuda", "xpu", "mps", "directml", "cpu"],
                       help="Compute backend / GPU vendor to use")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("LLM Training Script")
    print("=" * 70)

    # Set seed
    set_seed(args.seed)

    # Print device info
    print_device_info()

    # Get device
    device = get_device(backend=args.backend)

    # Load model configuration
    print("\n" + "-" * 70)
    print("Loading Model Configuration")
    print("-" * 70)

    model_config = get_config(args.config)

    # Override vocab size if specified
    if args.vocab_size:
        model_config.vocab_size = args.vocab_size

    print(f"Model config: {args.config}")
    print(f"Parameters: ~{model_config.num_params:,}")
    print(f"Architecture: {model_config.num_layers} layers x {model_config.dim} dim x {model_config.num_heads} heads")

    # Load training configuration
    print("\n" + "-" * 70)
    print("Loading Training Configuration")
    print("-" * 70)

    training_config = get_training_config(args.config)

    # Override with command line arguments
    if args.steps:
        training_config.max_steps = args.steps
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.gradient_accumulation:
        training_config.gradient_accumulation = args.gradient_accumulation
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.warmup_steps:
        training_config.warmup_steps = args.warmup_steps
    if args.weight_decay:
        training_config.weight_decay = args.weight_decay

    # Test run overrides
    if args.test_run:
        print("Test run: Using minimal configuration")
        training_config.max_steps = 100
        training_config.checkpoint_every = 50
        args.log_every = 10
        args.validation_every = 50

    print(f"Training config:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation}")
    print(f"  Effective batch size: {training_config.effective_batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Warmup steps: {training_config.warmup_steps}")
    print(f"  Max steps: {training_config.max_steps}")

    # Load data
    print("\n" + "-" * 70)
    print("Loading Training Data")
    print("-" * 70)

    train_path = os.path.join(args.data_dir, args.train_file)
    val_path = os.path.join(args.data_dir, args.val_file)

    if not os.path.exists(train_path):
        print(f"Error: Training file not found: {train_path}")
        print("Please run data preparation first:")
        print("  python scripts/prepare_data.py --data-dir data/raw/ --output-dir data/processed/")
        return 1

    print(f"Loading training data from {train_path}")
    train_sequences = load_sequences_from_file(train_path)
    print(f"Loaded {len(train_sequences)} training sequences")

    # Load validation data
    val_sequences = None
    if not args.no_validation and os.path.exists(val_path):
        print(f"Loading validation data from {val_path}")
        val_sequences = load_sequences_from_file(val_path)
        print(f"Loaded {len(val_sequences)} validation sequences")

    # Create dataloaders
    train_loader = create_dataloader(
        train_sequences,
        batch_size=training_config.batch_size,
        max_sequence_length=model_config.max_sequence_length,
        shuffle=True,
    )

    val_loader = None
    if val_sequences and not args.no_validation:
        val_loader = create_dataloader(
            val_sequences,
            batch_size=training_config.batch_size,
            max_sequence_length=model_config.max_sequence_length,
            shuffle=False,
        )

    # Create model
    print("\n" + "-" * 70)
    print("Creating Model")
    print("-" * 70)

    model = LLM(model_config)
    model = model.to(device)

    print(f"Model created and moved to {device}")

    # Create optimizer and scheduler
    print("\n" + "-" * 70)
    print("Creating Optimizer and Scheduler")
    print("-" * 70)

    optimizer = get_optimizer(
        model.parameters(),
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )

    opt_sched = OptimizerScheduler(optimizer, scheduler)

    print(f"Optimizer: AdamW (lr={training_config.learning_rate})")
    print(f"Scheduler: Cosine with warmup")

    # Create checkpoint manager
    print("\n" + "-" * 70)
    print("Creating Checkpoint Manager")
    print("-" * 70)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_mgr = CheckpointManager(
        args.checkpoint_dir,
        save_total_limit=args.save_total_limit,
    )

    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        metadata = checkpoint_mgr.load(args.resume, model, optimizer, scheduler)
        if metadata:
            start_step = metadata.get("step", 0)
            start_epoch = metadata.get("epoch", 0)
            print(f"Resumed from step {start_step}")

    # Create logger
    print("\n" + "-" * 70)
    print("Creating Logger")
    print("-" * 70)

    os.makedirs(args.log_dir, exist_ok=True)
    logger = TrainingLogger(
        args.log_dir,
        log_every=args.log_every,
        use_tensorboard=True,
    )

    # Create trainer
    print("\n" + "-" * 70)
    print("Creating Trainer")
    print("-" * 70)

    trainer = LLMTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        opt_sched=opt_sched,
        checkpoint_mgr=checkpoint_mgr,
        logger=logger,
        gradient_accumulation=training_config.gradient_accumulation,
        max_gradient_norm=args.max_gradient_norm,
        use_fp16=not args.no_fp16 and training_config.use_fp16,
        log_every=args.log_every,
        validation_every=args.validation_every if not args.no_validation else None,
        checkpoint_every=args.checkpoint_every,
    )

    # Restore training progress so a resumed run continues counting steps/epochs
    # instead of restarting at 0 (which would break max_steps, logging cadence
    # and checkpoint naming).
    trainer.global_step = start_step
    trainer.current_epoch = start_epoch

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    trainer.train(max_steps=training_config.max_steps)

    # Save final checkpoint
    print("\n" + "-" * 70)
    print("Saving Final Checkpoint")
    print("-" * 70)

    checkpoint_mgr.save(
        model,
        optimizer,
        scheduler,
        step=trainer.global_step,
        loss=0.0,  # placeholder; final.pt is not tracked as "best"
        epoch=trainer.current_epoch,
        config=model_config,
        filename="final.pt",
        track_best=False,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nFinal model saved to {args.checkpoint_dir}")
    print(f"Logs saved to {args.log_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
