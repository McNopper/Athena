"""
Optimizer and Learning Rate Scheduler

This module implements optimizer setup and learning rate scheduling for LLM training.

Educational Notes:
- Optimizer: Algorithm for updating model parameters (AdamW is standard)
- Learning Rate Schedule: Adjusts LR during training (warmup + cosine decay)
- AdamW: Adam with decoupled weight decay (better regularisation than standard Adam)
- Cosine Annealing: Smooth LR decay following a half-cosine curve
- Warmup: Gradually increase LR at start to avoid instability with a fresh model

Why AdamW?
- Adam: Per-parameter adaptive learning rates (faster convergence than SGD)
- Standard Adam: Weight decay is folded into the gradient update — this is not
  true L2 regularisation
- AdamW: Weight decay applied directly to parameters, decoupled from the gradient
  update — more principled regularisation, standard for LLMs

Why Learning Rate Scheduling?
- High LR early: Fast progress through the loss landscape
- Warmup: Prevents gradient instability when model weights are random
- Cosine decay: Smooth convergence; avoids abrupt LR drops

Training Flow:
    Warmup (steps 0-N) → Peak LR → Cosine decay (steps N-end) → Min LR

References:
    Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019) — https://arxiv.org/abs/1711.05101
    Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015) — https://arxiv.org/abs/1412.6980
    Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016) — https://arxiv.org/abs/1608.03983
"""

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def get_optimizer(
    model_params,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamW:
    """
    Create AdamW optimizer for training.

    Educational notes:
    - AdamW is standard for transformer training
    - Decoupled weight decay (better than L2 regularization)
    - Adaptive learning rates (different per parameter)

    AdamW Hyperparameters:
    - learning_rate: Step size for updates
    - weight_decay: L2 regularization strength
    - beta1, beta2: Momentum parameters (0.9, 0.999 are standard)
    - epsilon: Small constant for numerical stability

    Args:
        model_params: Model parameters (model.parameters())
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        beta1: First momentum parameter
        beta2: Second momentum parameter
        epsilon: Numerical stability constant

    Returns:
        AdamW optimizer

    Example:
        >>> optimizer = get_optimizer(model.parameters(), learning_rate=1e-3)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """
    optimizer = AdamW(
        model_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=epsilon,
    )

    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 1e-5,
) -> LambdaLR:
    """
    Create cosine learning rate schedule with warmup.

    Educational notes:
    - Warmup: Gradually increase LR from 0 to peak (prevents instability)
    - Cosine decay: Smoothly decrease LR following cosine curve
    - Min LR: Don't decay to 0 (keep small LR for stability)

    Schedule Formula:
        For step < warmup_steps:
            lr = step / warmup_steps * peak_lr  # Linear warmup

        For step >= warmup_steps:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π * progress))

    Why Cosine?
    - Smooth decay (better than step decay)
    - Proven effective for transformers
    - Gradual reduction (not too aggressive)

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate (after decay)

    Returns:
        Learning rate scheduler

    Example:
        >>> optimizer = get_optimizer(model.parameters())
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, warmup_steps=1000, max_steps=50000
        ... )
        >>> for step in range(max_steps):
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """
    # Educational note: LambdaLR multiplies each param group's *base* LR by the
    # value we return, so min_lr (an absolute LR) must be converted to a ratio of
    # the base LR to act as a real floor (otherwise the effective floor would be
    # min_lr * base_lr, i.e. far too small).
    base_lr = max((group["lr"] for group in optimizer.param_groups), default=1.0)
    min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        """
        Compute learning rate multiplier for current step.

        Educational note: Returns value in [min_lr_ratio, 1] that multiplies peak LR
        """
        if current_step < warmup_steps:
            # Educational note: Linear warmup
            # +1 so the very first optimizer step already uses a non-zero LR
            # (LambdaLR evaluates the lambda at step 0 on construction, which
            # would otherwise make the first update a no-op with LR = 0).
            return float(current_step + 1) / float(max(1, warmup_steps))
        else:
            # Educational note: Cosine decay
            # Progress from 0 to 1 over training
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))

            # Cosine curve: starts at 1, ends at 0
            # Formula: 0.5 * (1 + cos(π * progress))
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))

            # Scale to [min_lr_ratio, 1.0] so the absolute floor equals min_lr
            # When progress=0: cosine=1, result=1 (peak LR)
            # When progress=1: cosine=0, result=min_lr_ratio (i.e. min_lr absolute)
            return max(min_lr_ratio, cosine_decay.item())

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float = 0.0,
) -> LambdaLR:
    """
    Create linear learning rate schedule with warmup.

    Educational notes:
    - Alternative to cosine decay
    - Simpler: Linear decrease from peak to min
    - Sometimes works better for certain tasks

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler
    """
    # See get_cosine_schedule_with_warmup: convert absolute min_lr to a ratio of
    # the base LR so LambdaLR's multiplier yields a true absolute floor.
    base_lr = max((group["lr"] for group in optimizer.param_groups), default=1.0)
    min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # +1 so the first optimizer step uses a non-zero LR (see cosine).
            return float(current_step + 1) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(min_lr_ratio, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


class OptimizerScheduler:
    """
    Combined optimizer and scheduler manager.

    Educational notes:
    - Wraps optimizer and scheduler together
    - Handles step() calls for both
    - Provides convenient interface for training loop

    Args:
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler

    Example:
        >>> opt_sched = OptimizerScheduler(optimizer, scheduler)
        >>> loss.backward()
        >>> opt_sched.step()  # Updates both optimizer and scheduler
        >>> opt_sched.zero_grad()
    """

    def __init__(self, optimizer: Optimizer, scheduler: Optional[LambdaLR] = None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def step(self):
        """Update both optimizer and scheduler."""
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self):
        """Zero optimizer gradients."""
        self.optimizer.zero_grad()

    @property
    def learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def test_optimizer():
    """
    Test optimizer and scheduler implementation.

    Educational note: Tests verify:
    1. Optimizer creation works
    2. Scheduler creates correct schedule
    3. Learning rate follows expected curve
    """
    print("Testing Optimizer implementation...")

    # Test 1: Optimizer creation
    model_params = [torch.randn(10, requires_grad=True)]
    optimizer = get_optimizer(model_params, learning_rate=1e-3)

    assert optimizer is not None, "Optimizer creation failed"
    print("✓ Test 1 passed: Optimizer created")

    # Test 2: Scheduler creation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=10,
        max_steps=100,
    )

    assert scheduler is not None, "Scheduler creation failed"
    print("✓ Test 2 passed: Scheduler created")

    # Test 3: Learning rate schedule
    lrs = []
    for step in range(100):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Check warmup
    assert lrs[0] < lrs[9], "Warmup failed (LR should increase)"
    print("✓ Test 3 passed: Warmup works")

    # Check decay
    assert lrs[50] < lrs[20], "Decay failed (LR should decrease)"
    assert lrs[-1] < lrs[50], "Decay failed (LR should continue decreasing)"
    print("✓ Test 4 passed: Decay works")

    # Test 4: Combined manager
    opt_sched = OptimizerScheduler(optimizer, scheduler)
    assert opt_sched.learning_rate == lrs[-1], "Learning rate mismatch"
    print("✓ Test 5 passed: OptimizerScheduler works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_optimizer()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Optimizer and Scheduler Demonstration")
    print("=" * 70)

    # Create dummy model parameters
    model_params = [torch.randn(100, 100, requires_grad=True)]

    # Create optimizer
    optimizer = get_optimizer(model_params, learning_rate=1e-3, weight_decay=0.1)
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")

    # Create scheduler
    warmup_steps = 100
    max_steps = 1000

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=1e-5,
    )

    print(f"\nScheduler: Cosine with Warmup")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Max steps: {max_steps}")

    # Simulate training and track LR
    lrs = []
    steps_to_show = [0, 50, 100, 250, 500, 750, 999]

    print(f"\nLearning rate at key steps:")
    for step in range(max_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

        if step in steps_to_show:
            print(f"  Step {step:4d}: LR = {lrs[-1]:.6f}")

    # Plot-like summary
    print(f"\nSchedule summary:")
    print(f"  Min LR (warmup):   {min(lrs[:warmup_steps]):.6f}")
    print(f"  Peak LR (after warmup): {lrs[warmup_steps]:.6f}")
    print(f"  Final LR:          {lrs[-1]:.6f}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- AdamW: Standard optimizer for transformer training")
    print("- Weight decay: L2 regularization (prevents overfitting)")
    print("- Warmup: Gradually increase LR (training stability)")
    print("- Cosine decay: Smooth LR reduction (ensures convergence)")
    print("- Learning rate schedule is critical for training success")
    print("=" * 70)
