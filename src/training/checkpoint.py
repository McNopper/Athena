"""
Checkpoint Manager

This module handles saving and loading model checkpoints during training.

Educational Notes:
- Checkpoints: Saved model states (can resume training later)
- Best Model: Save based on validation loss
- Regular Saves: Save at intervals (resume if crash)
- Components to save: Model, optimizer, scheduler, training state

What to Save:
1. Model state_dict (weights)
2. Optimizer state_dict (momentum, Adam statistics)
3. Scheduler state (current step, learning rate)
4. Training metadata (epoch, step, best loss)

Checkpoint Types:
- Best checkpoint: Model with lowest validation loss
- Regular checkpoint: Periodic saves (every N steps)
- Final checkpoint: End of training

Usage:
    Training → Save checkpoint → Continue/Resume training → Load checkpoint

Why Checkpoints?
- Resume after crash/interruption
- Track best model during training
- Debug training issues
- Deploy best model for inference
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints.

    Educational notes:
    - Saves complete training state (model, optimizer, scheduler)
    - Tracks best model based on validation loss
    - Supports resuming training from checkpoints
    - Automatic cleanup of old checkpoints

    Saved Components:
    - model_state_dict: Model weights
    - optimizer_state_dict: Optimizer state (Adam statistics)
    - scheduler_state_dict: LR scheduler state
    - training_metadata: Step, epoch, best loss, etc.

    Args:
        save_dir: Directory to save checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        metric_name: Metric to track for "best" model (lower is better)

    Example:
        >>> checkpoint_mgr = CheckpointManager("checkpoints", save_total_limit=3)
        >>> # Save checkpoint
        >>> checkpoint_mgr.save(
        ...     model, optimizer, scheduler,
        ...     step=1000, loss=2.5, epoch=5
        ... )
        >>> # Load best model
        >>> best = checkpoint_mgr.load_best()
    """

    def __init__(
        self,
        save_dir: str,
        save_total_limit: int = 3,
        metric_name: str = "val_loss",
    ):
        self.save_dir = Path(save_dir)
        self.save_total_limit = save_total_limit
        self.metric_name = metric_name

        # Create directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track best metric
        self.best_metric = float("inf")
        self.best_checkpoint_path = None

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        config: Optional[object] = None,
        filename: Optional[str] = None,
        track_best: bool = True,
    ) -> str:
        """
        Save training checkpoint.

        Educational notes:
        - Saves model + optimizer + scheduler state
        - Tracks if this is the best model so far
        - Automatically manages checkpoint limit (removes oldest)

        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: LR scheduler (optional)
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Force save as best (optional)

        Returns:
            Path to saved checkpoint

        Example:
            >>> checkpoint_mgr.save(
            ...     model, optimizer, scheduler,
            ...     step=1000, loss=2.5, metrics={"accuracy": 0.8}
            ... )
        """
        # Create checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {},
        }

        # Educational note: Persist the model configuration so the model can be
        # rebuilt for inference without guessing its architecture.
        if config is not None:
            checkpoint_data["model_config"] = config

        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Determine if this is best. When track_best is False (e.g. a plain
        # "final" snapshot with a placeholder loss), never promote to best.pt.
        if track_best:
            current_metric = metrics.get(self.metric_name, loss) if metrics else loss
            is_best = is_best or (current_metric < self.best_metric)
        else:
            is_best = False

        # Generate filename
        if is_best:
            filename = "best.pt"
            self.best_metric = current_metric
            self.best_checkpoint_path = self.save_dir / filename
        elif filename is None:
            filename = f"checkpoint_step_{step}.pt"

        # Save checkpoint
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint_data, checkpoint_path)

        # Cleanup old checkpoints (except best)
        if not is_best:
            self._cleanup_checkpoints()

        return str(checkpoint_path)

    def load(self, checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.

        Educational notes:
        - Loads model weights
        - Restores optimizer state (Adam statistics)
        - Restores scheduler state (learning rate)
        - Returns training metadata (step, loss, etc.)

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to restore state (optional)
            scheduler: Scheduler to restore state (optional)

        Returns:
            Dictionary with training metadata

        Example:
            >>> metadata = checkpoint_mgr.load("checkpoints/best.pt", model)
            >>> print(f"Resuming from step {metadata['step']}")
        """
        # Educational note: checkpoints are produced by this project (trusted),
        # and store the model config object, so we must disable the newer
        # weights_only default to unpickle the full checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return metadata
        metadata = {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", 0.0),
            "metrics": checkpoint.get("metrics", {}),
        }

        return metadata

    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint.

        Educational notes:
        - Loads checkpoint with best validation metric
        - Typically used for inference or final model

        Args:
            model: Model to load weights into
            optimizer: Optimizer to restore (optional)
            scheduler: Scheduler to restore (optional)

        Returns:
            Training metadata, or None if no best checkpoint exists

        Example:
            >>> metadata = checkpoint_mgr.load_best(model)
            >>> if metadata:
            ...     print(f"Best model loss: {metadata['loss']}")
        """
        if self.best_checkpoint_path is None or not self.best_checkpoint_path.exists():
            return None

        return self.load(str(self.best_checkpoint_path), model, optimizer, scheduler)

    def load_latest(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Optional[Dict[str, Any]]:
        """
        Load most recent checkpoint.

        Educational notes:
        - Loads the most recently saved checkpoint
        - Used for resuming training after interruption

        Args:
            model: Model to load weights into
            optimizer: Optimizer to restore (optional)
            scheduler: Scheduler to restore (optional)

        Returns:
            Training metadata, or None if no checkpoint exists
        """
        # Find all checkpoint files
        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))

        if not checkpoints:
            return None

        # Sort by step number (most recent last)
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

        # Load most recent
        latest = checkpoints[-1]
        return self.load(str(latest), model, optimizer, scheduler)

    def _cleanup_checkpoints(self):
        """
        Remove old checkpoints, keeping only save_total_limit.

        Educational note:
        - Keeps disk space manageable
        - Always keeps best checkpoint
        - Removes oldest regular checkpoints first
        """
        # Get all regular checkpoints (not best)
        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

        # Remove oldest if we have too many
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.

        Educational note:
        - Returns count and paths of checkpoints
        - Useful for logging and debugging

        Returns:
            Dictionary with checkpoint information
        """
        checkpoints = list(self.save_dir.glob("*.pt"))

        return {
            "save_dir": str(self.save_dir),
            "total_checkpoints": len(checkpoints),
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "best_metric": self.best_metric,
        }


def test_checkpoint_manager():
    """
    Test checkpoint manager implementation.

    Educational note: Tests verify:
    1. Checkpoint saving works
    2. Checkpoint loading restores state
    3. Best model tracking works
    4. Cleanup mechanism works
    """
    print("Testing Checkpoint Manager implementation...")

    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Create checkpoint manager
        mgr = CheckpointManager(temp_dir, save_total_limit=2)
        assert mgr.save_dir.exists(), "Save directory not created"
        print("✓ Test 1 passed: Checkpoint manager created")

        # Create dummy model
        model = torch.nn.Linear(10, 10)

        # Test 2: Save checkpoint
        path = mgr.save(model, step=100, loss=1.0)
        assert os.path.exists(path), "Checkpoint file not created"
        print("✓ Test 2 passed: Checkpoint saved")

        # Test 3: Load checkpoint
        model2 = torch.nn.Linear(10, 10)
        metadata = mgr.load(path, model2)

        assert metadata["step"] == 100, "Step not loaded correctly"
        assert torch.equal(model2.weight, model.weight), "Model weights not loaded"
        print("✓ Test 3 passed: Checkpoint loaded")

        # Test 4: Best model tracking
        mgr.save(model, step=200, loss=0.5)  # Better
        mgr.save(model, step=300, loss=0.3)  # Even better

        assert mgr.best_metric == 0.3, "Best metric not tracked"
        assert (mgr.save_dir / "best.pt").exists(), "Best checkpoint not saved"
        print("✓ Test 4 passed: Best model tracking works")

        # Test 5: Cleanup
        for i in range(5):
            mgr.save(model, step=400 + i, loss=1.0 + i * 0.1)

        checkpoints = list(mgr.save_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) <= 2, f"Cleanup failed: {len(checkpoints)} checkpoints"
        print("✓ Test 5 passed: Cleanup mechanism works")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_checkpoint_manager()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Checkpoint Manager Demonstration")
    print("=" * 70)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        # Create manager
        mgr = CheckpointManager(temp_dir, save_total_limit=3)
        print(f"\nCheckpoint Manager created:")
        print(f"  Save directory: {temp_dir}")
        print(f"  Max checkpoints: 3")

        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())

        print(f"\nSaving checkpoints:")

        # Save several checkpoints
        losses = [2.5, 2.0, 1.8, 1.5, 1.3, 1.2]
        for step, loss in enumerate(losses, start=100):
            path = mgr.save(
                model, optimizer,
                step=step,
                loss=loss,
                metrics={"val_loss": loss, "accuracy": 0.7 + loss * 0.1}
            )
            print(f"  Step {step}: loss={loss:.2f} → {os.path.basename(path)}")

        # Show info
        info = mgr.get_checkpoint_info()
        print(f"\nCheckpoint info:")
        print(f"  Total checkpoints: {info['total_checkpoints']}")
        print(f"  Best metric: {info['best_metric']:.2f}")

        # Load best
        print(f"\nLoading best checkpoint:")
        metadata = mgr.load_best(model)
        print(f"  Step: {metadata['step']}")
        print(f"  Loss: {metadata['loss']:.2f}")

        print("\n" + "=" * 70)
        print("Educational notes:")
        print("- Checkpoints save complete training state")
        print("- Best model tracked automatically")
        print("- Old checkpoints cleaned up (saves disk space)")
        print("- Can resume training from any checkpoint")
        print("- Critical for long training runs")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir)
