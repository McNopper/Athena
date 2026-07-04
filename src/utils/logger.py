"""
Training Logger

This module handles logging during training for monitoring and debugging.

Educational Notes:
- Logs training metrics (loss, learning rate, gradient norm)
- Outputs to console and optionally to TensorBoard
- Tracks training progress over time
- Essential for debugging and monitoring

What to Log:
- Loss: Training and validation
- Learning rate: Current LR (changes with scheduler)
- Gradient norm: Check for exploding gradients
- Throughput: Tokens/second, samples/second
- Timing: Step time, epoch time

Log Destinations:
- Console: Real-time monitoring
- File: Persistent record
- TensorBoard: Visualization and analysis
"""

import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


class TrainingLogger:
    """
    Logger for training metrics and progress.

    Educational notes:
    - Logs metrics to console and CSV file
    - Tracks timing information
    - Provides progress updates
    - Creates persistent training record

    Args:
        log_dir: Directory to save logs
        log_every: Log every N steps
        use_tensorboard: Also log to TensorBoard (if available)

    Example:
        >>> logger = TrainingLogger("logs", log_every=100)
        >>> logger.log(step=100, loss=2.5, lr=0.001)
        >>> logger.finish()
    """

    def __init__(self, log_dir: str, log_every: int = 100, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_every = log_every
        self.use_tensorboard = use_tensorboard

        # Create directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV logging
        self.csv_file = open(self.log_dir / "training_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        self.csv_writer.writerow(["step", "timestamp", "loss", "lr", "grad_norm", "tokens_per_sec"])
        self.csv_file.flush()

        # Track metrics for summary
        self.step_history: List[int] = []
        self.loss_history: List[float] = []

        # Timing
        self.last_log_time = None
        self.step_start_time = None

        # TensorBoard (optional)
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")

    def log(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        epoch: Optional[int] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log training metrics.

        Educational notes:
        - Called every N steps during training
        - Writes to console, CSV, and TensorBoard
        - Tracks history for analysis

        Args:
            step: Current training step
            loss: Current loss value
            lr: Current learning rate
            grad_norm: Gradient norm (optional)
            tokens_per_sec: Training throughput (optional)
            epoch: Current epoch (optional)
            extra_metrics: Additional metrics to log (optional)

        Example:
            >>> logger.log(
            ...     step=1000,
            ...     loss=2.5,
            ...     lr=0.001,
            ...     grad_norm=1.2,
            ...     tokens_per_sec=50000
            ... )
        """
        # Calculate timing
        current_time = time.time()

        if self.last_log_time is not None:
            elapsed = current_time - self.last_log_time
        else:
            elapsed = 0

        self.last_log_time = current_time

        # Log to console
        log_str = f"Step {step:6d}"

        if epoch is not None:
            log_str += f" | Epoch {epoch:3d}"

        log_str += f" | Loss: {loss:.4f} | LR: {lr:.6f}"

        if grad_norm is not None:
            log_str += f" | Grad Norm: {grad_norm:.4f}"

        if tokens_per_sec is not None:
            log_str += f" | Tokens/s: {tokens_per_sec:.0f}"

        if elapsed > 0:
            log_str += f" | Time: {elapsed:.2f}s"

        print(log_str)

        # Log to CSV
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([
            step,
            timestamp,
            f"{loss:.4f}",
            f"{lr:.6f}",
            f"{grad_norm:.4f}" if grad_norm else "",
            f"{tokens_per_sec:.0f}" if tokens_per_sec else "",
        ])
        self.csv_file.flush()

        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/train", loss, step)
            self.tb_writer.add_scalar("LearningRate", lr, step)

            if grad_norm is not None:
                self.tb_writer.add_scalar("Gradients/norm", grad_norm, step)

            if tokens_per_sec is not None:
                self.tb_writer.add_scalar("Throughput/tokens_per_sec", tokens_per_sec, step)

            if extra_metrics:
                for name, value in extra_metrics.items():
                    self.tb_writer.add_scalar(f"Metrics/{name}", value, step)

        # Track history
        self.step_history.append(step)
        self.loss_history.append(loss)

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_perplexity: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log validation metrics.

        Educational notes:
        - Called after validation pass
        - Tracks model performance on validation set
        - Perplexity = exp(loss) (standard language model metric)

        Args:
            step: Current training step
            val_loss: Validation loss
            val_perplexity: Validation perplexity (optional)
            extra_metrics: Additional metrics (optional)

        Example:
            >>> logger.log_validation(
            ...     step=1000,
            ...     val_loss=2.3,
            ...     val_perplexity=10.0
            ... )
        """
        log_str = f"Validation | Step {step:6d} | Loss: {val_loss:.4f}"

        if val_perplexity is not None:
            log_str += f" | Perplexity: {val_perplexity:.2f}"

        if extra_metrics:
            for name, value in extra_metrics.items():
                log_str += f" | {name}: {value:.4f}"

        print(log_str)

        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/validation", val_loss, step)

            if val_perplexity is not None:
                self.tb_writer.add_scalar("Perplexity/validation", val_perplexity, step)

            if extra_metrics:
                for name, value in extra_metrics.items():
                    self.tb_writer.add_scalar(f"Validation/{name}", value, step)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.

        Educational notes:
        - Returns statistics over training history
        - Useful for final report

        Returns:
            Dictionary with summary statistics

        Example:
            >>> summary = logger.get_summary()
            >>> print(f"Final loss: {summary['final_loss']}")
        """
        if not self.loss_history:
            return {}

        return {
            "total_steps": self.step_history[-1] if self.step_history else 0,
            "initial_loss": self.loss_history[0],
            "final_loss": self.loss_history[-1],
            "min_loss": min(self.loss_history),
            "max_loss": max(self.loss_history),
        }

    def finish(self):
        """
        Finish logging and close files.

        Educational note:
        - Call this at end of training
        - Closes CSV file and TensorBoard writer
        """
        # Close CSV
        if self.csv_file:
            self.csv_file.close()

        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()


def test_logger():
    """
    Test logger implementation.

    Educational note: Tests verify:
    1. Logger creation works
    2. Logging metrics works
    3. CSV file is created
    4. Summary is correct
    """
    print("Testing Logger implementation...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Create logger
        logger = TrainingLogger(temp_dir, log_every=10, use_tensorboard=False)
        assert logger.log_dir.exists(), "Log directory not created"
        print("✓ Test 1 passed: Logger created")

        # Test 2: Log metrics
        logger.log(step=10, loss=2.5, lr=0.001, grad_norm=1.0)
        logger.log(step=20, loss=2.3, lr=0.001, grad_norm=0.9)

        # Check CSV file
        csv_path = logger.log_dir / "training_log.csv"
        assert csv_path.exists(), "CSV file not created"
        print("✓ Test 2 passed: Metrics logged")

        # Test 3: Validation logging
        logger.log_validation(step=20, val_loss=2.4, val_perplexity=11.0)
        print("✓ Test 3 passed: Validation logging works")

        # Test 4: Summary
        summary = logger.get_summary()
        assert summary["final_loss"] == 2.3, "Summary incorrect"
        print("✓ Test 4 passed: Summary correct")

        # Test 5: Finish
        logger.finish()
        print("✓ Test 5 passed: Logger finished")

    finally:
        shutil.rmtree(temp_dir)

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_logger()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Training Logger Demonstration")
    print("=" * 70)

    import tempfile
    import shutil
    from math import exp

    temp_dir = tempfile.mkdtemp()

    try:
        logger = TrainingLogger(temp_dir, log_every=100, use_tensorboard=False)

        print(f"\nLogger created:")
        print(f"  Log directory: {temp_dir}")
        print(f"  Log every: {logger.log_every} steps")

        print(f"\nSimulating training:")

        # Simulate training
        for step in range(0, 1000, 100):
            loss = 3.0 - (step / 1000) * 1.5  # Decreasing loss
            lr = 0.001 * (1 - step / 1000)  # Decreasing LR
            grad_norm = 1.0 + (step / 1000) * 0.5

            logger.log(
                step=step,
                loss=loss,
                lr=lr,
                grad_norm=grad_norm,
                tokens_per_sec=50000,
            )

            # Validation every 500 steps
            if step % 500 == 0 and step > 0:
                logger.log_validation(
                    step=step,
                    val_loss=loss + 0.2,
                    val_perplexity=exp(loss + 0.2),
                )

        # Get summary
        summary = logger.get_summary()

        print(f"\nTraining summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Initial loss: {summary['initial_loss']:.4f}")
        print(f"  Final loss: {summary['final_loss']:.4f}")
        print(f"  Min loss: {summary['min_loss']:.4f}")

        logger.finish()

        print("\n" + "=" * 70)
        print("Educational notes:")
        print("- Logger tracks training metrics over time")
        print("- Outputs to console, CSV, and TensorBoard")
        print("- Essential for monitoring and debugging")
        print("- Provides persistent training record")
        print("- Check TensorBoard for visualization")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir)
