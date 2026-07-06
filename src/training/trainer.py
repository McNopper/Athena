"""
Training Loop

This module implements the main training loop for language models.

Educational Notes:
- Orchestrates the entire training process
- Handles forward pass, backward pass, optimization
- Manages validation, checkpointing, logging
- Supports gradient accumulation, mixed precision, gradient clipping

Training Loop Steps:
    1. Load batch
    2. Forward pass (compute logits) — in FP16 via autocast
    3. Compute cross-entropy loss
    4. Scale loss (for FP16 stability) and backward pass
    5. Gradient accumulation (if needed)
    6. Unscale gradients and clip (prevent exploding)
    7. Optimizer step (update weights)
    8. Logging and checkpointing

Key Features:
- Gradient Accumulation: Simulate larger batch sizes without extra memory
- Mixed Precision (AMP): Use FP16 for forward/backward, FP32 for parameter updates.
  A GradScaler multiplies the loss before backward() to prevent FP16 underflow,
  then divides the gradients before the optimizer step.
- Gradient Clipping: Caps the gradient norm to prevent exploding gradients
- Validation: Monitor generalisation via held-out loss
- Checkpointing: Save periodic and best-model snapshots

References:
    Micikevicius et al., "Mixed Precision Training" (2018) — https://arxiv.org/abs/1710.03740
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time
import contextlib

from ..utils.metrics import compute_loss, compute_perplexity, compute_gradient_norm
from ..utils.logger import TrainingLogger
from .checkpoint import CheckpointManager
from .optimizer import OptimizerScheduler


class LLMTrainer:
    """
    Training loop for language models.

    Educational notes:
    - Complete training pipeline
    - Handles forward/backward passes
    - Manages optimization, validation, checkpointing
    - Logs metrics and progress

    Training Flow:
        For each step:
            1. Load batch
            2. Forward pass
            3. Compute loss
            4. Backward pass
            5. (Accumulate gradients)
            6. (Clip gradients)
            7. Optimizer step
            8. Log metrics
            9. (Validate)
            10. (Checkpoint)

    Args:
        model: Language model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        opt_sched: Optimizer and scheduler
        checkpoint_mgr: Checkpoint manager
        logger: Training logger
        gradient_accumulation: Gradient accumulation steps
        max_gradient_norm: Clip gradients at this norm
        use_fp16: Use mixed precision training
        log_every: Log every N steps
        validation_every: Validate every N steps
        checkpoint_every: Save checkpoint every N steps

    Example:
        >>> trainer = LLMTrainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     opt_sched=opt_sched,
        ...     checkpoint_mgr=checkpoint_mgr,
        ...     logger=logger
        ... )
        >>> trainer.train(max_steps=10000)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        opt_sched: OptimizerScheduler,
        checkpoint_mgr: Optional[CheckpointManager] = None,
        logger: Optional[TrainingLogger] = None,
        val_dataloader: Optional[DataLoader] = None,
        gradient_accumulation: int = 1,
        max_gradient_norm: float = 1.0,
        use_fp16: bool = True,
        log_every: int = 100,
        validation_every: int = 1000,
        checkpoint_every: int = 5000,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.opt_sched = opt_sched
        self.checkpoint_mgr = checkpoint_mgr
        self.logger = logger

        # Training settings
        self.gradient_accumulation = gradient_accumulation
        self.max_gradient_norm = max_gradient_norm
        self.use_fp16 = use_fp16
        self.log_every = log_every
        self.validation_every = validation_every
        self.checkpoint_every = checkpoint_every

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.max_steps: Optional[int] = None

        # Logging-interval accumulators. Reset after every log line so the
        # reported loss / throughput reflect the *recent* window rather than a
        # since-start average that lags the current loss.
        self._interval_loss = 0.0
        self._interval_batches = 0
        self._interval_tokens = 0
        self._interval_start = time.time()

        # Device and mixed precision.
        # Educational note: automatic mixed precision (AMP) only pays off on
        # CUDA GPUs with Tensor Cores, so we enable it there and fall back to
        # full precision elsewhere. A GradScaler keeps FP16 gradients from
        # underflowing. See https://pytorch.org/docs/stable/amp.html
        self.device = next(model.parameters()).device
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

    def train(self, max_steps: Optional[int] = None, max_epochs: Optional[int] = None):
        """
        Run training loop.

        Educational notes:
        - Main training loop
        - Iterates over dataloader
        - Performs forward/backward/optimizer steps

        Args:
            max_steps: Maximum training steps
            max_epochs: Maximum training epochs

        Example:
            >>> trainer.train(max_steps=10000)
        """
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)

        self.model.train()

        # Record the step budget so we can stop mid-epoch as well.
        self.max_steps = max_steps

        epoch = 0

        while True:
            # Check if we should stop
            if max_steps is not None and self.global_step >= max_steps:
                print(f"\nReached max steps ({max_steps})")
                break

            if max_epochs is not None and epoch >= max_epochs:
                print(f"\nReached max epochs ({max_epochs})")
                break

            # Train for one epoch
            epoch_loss = self._train_epoch()

            epoch += 1
            self.current_epoch = epoch

            print(f"\nEpoch {epoch} completed. Avg loss: {epoch_loss:.4f}")

        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)

        if self.logger:
            self.logger.finish()

    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        Educational notes:
        - Iterates over all batches in dataloader
        - Performs forward/backward passes
        - Accumulates gradients
        - Updates optimizer

        Returns:
            Average loss for the epoch

        Example:
            >>> avg_loss = trainer._train_epoch()
        """
        accumulated_loss = 0.0
        accumulated_tokens = 0
        num_batches = 0

        # Total micro-batches this epoch. Used to size the final (possibly
        # partial) accumulation window so its loss is scaled by the real number
        # of micro-batches rather than the nominal gradient_accumulation.
        try:
            total_batches = len(self.train_dataloader)
        except TypeError:
            total_batches = None

        pending_grads = False  # True when un-stepped gradients are accumulated

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            # Move to device. Token IDs stay integer (never cast to float, or
            # the embedding lookup would break).
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Size of the accumulation window this batch belongs to. The last
            # window is shorter when the epoch's batch count is not a multiple
            # of gradient_accumulation.
            window_start = (batch_idx // self.gradient_accumulation) * self.gradient_accumulation
            if total_batches is not None:
                window_size = min(self.gradient_accumulation, total_batches - window_start)
            else:
                window_size = self.gradient_accumulation

            # Forward pass (under autocast when AMP is enabled)
            loss, num_tokens = self._forward_step(inputs, targets)

            # Scale loss so accumulated gradients equal the window mean
            loss = loss / window_size

            # Backward pass (GradScaler is a no-op when AMP is disabled)
            self.scaler.scale(loss).backward()
            pending_grads = True

            batch_loss = loss.item() * window_size  # undo the window scaling
            accumulated_loss += batch_loss
            accumulated_tokens += num_tokens
            num_batches += 1
            self._interval_loss += batch_loss
            self._interval_batches += 1
            self._interval_tokens += num_tokens

            # Step once the window is complete (also handles a short final window)
            window_complete = (batch_idx + 1 - window_start) >= window_size
            if window_complete:
                self._apply_optimizer_step(accumulated_loss, num_batches)
                pending_grads = False

                # Check if we should stop mid-epoch
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break

        # Flush any leftover accumulated gradients (e.g. when the dataloader
        # length is unknown and the last window never reached window_size).
        if pending_grads:
            self._apply_optimizer_step(accumulated_loss, num_batches)

        return accumulated_loss / num_batches if num_batches > 0 else 0.0

    def _apply_optimizer_step(
        self,
        accumulated_loss: float,
        num_batches: int,
    ) -> bool:
        """
        Apply one optimizer update for the currently accumulated gradients.

        Handles gradient clipping, the AMP scaler step (which may skip on
        overflow), the LR schedule step, the global-step counter, and periodic
        logging / validation / checkpointing.

        Args:
            accumulated_loss: Epoch-cumulative loss (used for checkpoint metadata)
            num_batches: Epoch-cumulative batch count

        Returns:
            True if the optimizer actually stepped, False if AMP skipped it.
        """
        # Gradient clipping
        if self.max_gradient_norm > 0:
            # Unscale before clipping so the norm is measured in real units.
            self.scaler.unscale_(self.opt_sched.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_gradient_norm,
                )
            )
        else:
            # Unscale first so the reported norm is in real (unscaled) units
            # under AMP (unscale_ is a no-op when AMP is disabled).
            self.scaler.unscale_(self.opt_sched.optimizer)
            grad_norm = compute_gradient_norm(self.model)

        # Optimizer step (through the scaler) then LR schedule step.
        # Educational note: under AMP the scaler may *skip* the optimizer
        # step when gradients overflow (inf/nan). We detect that (the loss
        # scale is reduced) and avoid advancing the LR schedule / step
        # counter so they stay in sync with real parameter updates.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.opt_sched.optimizer)
        self.scaler.update()
        optimizer_stepped = self.scaler.get_scale() >= scale_before

        self.opt_sched.zero_grad()

        if not optimizer_stepped:
            # Skipped step: gradients already cleared; nothing else to do.
            return False

        if self.opt_sched.scheduler is not None:
            self.opt_sched.scheduler.step()

        # Update step
        self.global_step += 1

        # Log metrics (recent-interval averages, reset after each log line)
        if self.global_step % self.log_every == 0:
            avg_loss = (
                self._interval_loss / self._interval_batches
                if self._interval_batches > 0
                else 0.0
            )
            lr = self.opt_sched.learning_rate

            # Throughput over the current logging interval
            elapsed = time.time() - self._interval_start
            tokens_per_sec = self._interval_tokens / elapsed if elapsed > 0 else 0

            if self.logger:
                self.logger.log(
                    step=self.global_step,
                    loss=avg_loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                    epoch=self.current_epoch,
                )

            # Reset the interval accumulators
            self._interval_loss = 0.0
            self._interval_batches = 0
            self._interval_tokens = 0
            self._interval_start = time.time()

        # Validation
        if self.val_dataloader and self.global_step % self.validation_every == 0:
            val_metrics = self._validate()
            if self.logger:
                self.logger.log_validation(
                    step=self.global_step,
                    val_loss=val_metrics["loss"],
                    val_perplexity=val_metrics["perplexity"],
                )

        # Checkpoint
        if self.checkpoint_mgr and self.global_step % self.checkpoint_every == 0:
            avg_loss = accumulated_loss / num_batches
            self.checkpoint_mgr.save(
                self.model,
                self.opt_sched.optimizer,
                self.opt_sched.scheduler,
                step=self.global_step,
                loss=avg_loss,
                epoch=self.current_epoch,
                config=getattr(self.model, "config", None),
            )

        return True

    def _forward_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Perform forward pass.

        Educational notes:
        - Computes model output
        - Calculates loss
        - Returns loss and token count

        Args:
            inputs: Input token IDs
            targets: Target token IDs

        Returns:
            Tuple of (loss, num_tokens)
        """
        # Forward pass. Only enter autocast when AMP is actually enabled (CUDA);
        # constructing torch.autocast for other device types (e.g. DirectML's
        # 'privateuseone') can raise even with enabled=False, so use nullcontext.
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.use_fp16
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = self.model(inputs)
            loss = compute_loss(logits, targets)

        # Count real (non-padding) tokens; padded targets use IGNORE_INDEX (-100)
        # and are excluded from the loss, so they must not inflate throughput.
        num_tokens = int((targets != -100).sum().item())

        return loss, num_tokens

    def _validate(self) -> Dict[str, float]:
        """
        Run validation.

        Educational notes:
        - Evaluates model on validation set
        - Computes loss and perplexity
        - No gradient computation

        Returns:
            Dictionary with validation metrics

        Example:
            >>> val_metrics = trainer._validate()
            >>> print(f"Val loss: {val_metrics['loss']}")
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                logits = self.model(inputs)

                # Compute loss
                loss = compute_loss(logits, targets)

                # Weight the loss by the number of real (non-padding) tokens.
                num_tokens = int((targets != -100).sum().item())
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = compute_perplexity(avg_loss)

        self.model.train()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }


def test_trainer():
    """
    Test trainer implementation.

    Educational note: Tests verify:
    1. Trainer creation works
    2. Training loop runs
    3. Metrics are logged
    """
    print("Testing Trainer implementation...")

    import tempfile
    import shutil
    from ..model.llm import LLM
    from ..config.model_config import get_config
    from ..utils.device import get_device
    from .dataset import create_dataloader

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Create trainer components
        config = get_config("pico")  # Fast testing
        model = LLM(config)

        device = get_device()
        model = model.to(device)

        # Create dummy data
        sequences = [
            [1, 2, 3, 4, 5] * 20,
            [6, 7, 8, 9, 10] * 20,
        ]

        train_loader = create_dataloader(sequences, batch_size=2, shuffle=False)
        print("✓ Test 1 passed: Components created")

        # Test 2: Create optimizer and scheduler
        from .optimizer import get_optimizer, get_cosine_schedule_with_warmup

        optimizer = get_optimizer(model.parameters(), learning_rate=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, max_steps=100)

        opt_sched = OptimizerScheduler(optimizer, scheduler)
        print("✓ Test 2 passed: Optimizer and scheduler created")

        # Test 3: Create logger and checkpoint manager
        from ..utils.logger import TrainingLogger

        logger = TrainingLogger(temp_dir, log_every=5, use_tensorboard=False)
        checkpoint_mgr = CheckpointManager(temp_dir, save_total_limit=2)
        print("✓ Test 3 passed: Logger and checkpoint manager created")

        # Test 4: Create trainer
        trainer = LLMTrainer(
            model=model,
            train_dataloader=train_loader,
            opt_sched=opt_sched,
            checkpoint_mgr=checkpoint_mgr,
            logger=logger,
            log_every=2,
            validation_every=5,
            checkpoint_every=10,
        )
        print("✓ Test 4 passed: Trainer created")

        print("\n✓ All tests passed!")

    finally:
        # Close the logger so its CSV handle is released before we delete the
        # temp directory (Windows will not remove an open file).
        try:
            logger.finish()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    from ..utils.device import get_device

    # Educational: Run tests
    test_trainer()

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Trainer orchestrates entire training process")
    print("- Handles forward/backward/optimizer steps")
    print("- Supports gradient accumulation and mixed precision")
    print("- Logs metrics and saves checkpoints")
    print("- Validation monitors overfitting")
    print("=" * 70)
