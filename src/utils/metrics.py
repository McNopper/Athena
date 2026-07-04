"""
Training Metrics

This module implements metrics computation for language model training.

Educational Notes:
- Loss: Cross-entropy loss (measures prediction error)
- Perplexity: Exponential of loss (intuitive measure of model uncertainty)
- Accuracy: Token-level accuracy (exact match predictions)
- Gradient Norm: Magnitude of gradients (checks for exploding gradients)

Key Metrics:
- Loss: Lower is better (measures prediction error)
- Perplexity: Lower is better (exp(loss), measures uncertainty)
- Accuracy: Higher is better (percentage of correct predictions)
- Gradient Norm: Should stay reasonable (check for instability)

Metric Interpretation:
- Loss = 2.0 → Perplexity = exp(2.0) ≈ 7.4
- Model is uncertain among ~7 tokens on average
- Lower perplexity = more confident predictions
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.

    Educational notes:
    - Measures difference between predictions and targets
    - Cross-entropy: -sum(target * log(predictions))
    - Lower loss = better predictions

    Loss Formula:
        Loss = -log(P(target_token))

    Args:
        logits: Model output logits
               Shape: (batch_size, seq_len, vocab_size)
        targets: Target token IDs
                 Shape: (batch_size, seq_len)
        ignore_index: Token ID to ignore in loss (e.g., padding)

    Returns:
        Scalar loss tensor

    Example:
        >>> logits = torch.randn(32, 128, 32000)
        >>> targets = torch.randint(0, 32000, (32, 128))
        >>> loss = compute_loss(logits, targets)
        >>> print(f"Loss: {loss.item():.4f}")
    """
    # Reshape for computation
    # logits: (batch, seq, vocab) → (batch * seq, vocab)
    # targets: (batch, seq) → (batch * seq,)
    batch_size, seq_len, vocab_size = logits.shape

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction="mean",
    )

    return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from loss.

    Educational notes:
    - Perplexity = exp(loss)
    - Intuitive measure of model uncertainty
    - Lower is better

    Perplexity Interpretation:
    - Perplexity = 10: Model is uncertain among ~10 choices
    - Perplexity = 1: Perfect prediction (no uncertainty)
    - Perplexity = vocab_size: Random guessing

    Formula:
        Perplexity = exp(loss)

    Args:
        loss: Loss value (scalar tensor or float)

    Returns:
        Perplexity value

    Example:
        >>> loss = torch.tensor(2.0)
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl:.2f}")
        Perplexity: 7.39
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()

    # Perplexity = exp(loss). Guard against overflow for very large losses
    # (e.g. an untrained model) by capping at infinity.
    return math.exp(loss) if loss < 20 else float("inf")


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.

    Educational notes:
    - Percentage of correctly predicted tokens
    - Measures exact match accuracy
    - Not commonly used for LM (loss is better metric)

    Formula:
        Accuracy = correct_predictions / total_predictions

    Args:
        logits: Model output logits
        targets: Target token IDs
        ignore_index: Token ID to ignore

    Returns:
        Accuracy (0.0 to 1.0)

    Example:
        >>> accuracy = compute_accuracy(logits, targets)
        >>> print(f"Accuracy: {accuracy:.2%}")
    """
    # Get predictions (argmax)
    predictions = torch.argmax(logits, dim=-1)

    # Create mask (ignore padding)
    mask = targets != ignore_index

    # Count correct predictions
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute L2 norm of model gradients.

    Educational notes:
    - Measures magnitude of gradients
    - Too high: Exploding gradients (training instability)
    - Too low: Vanishing gradients (slow learning)
    - Should be reasonable (0.1 - 10.0 typically)

    Formula:
        GradNorm = sqrt(sum(grad^2))

    Args:
        model: PyTorch model

    Returns:
        Gradient norm value

    Example:
        >>> loss.backward()
        >>> grad_norm = compute_gradient_norm(model)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
    """
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm


def compute_throughput(
    num_tokens: int,
    elapsed_time: float,
) -> Dict[str, float]:
    """
    Compute training throughput metrics.

    Educational notes:
    - Measures training speed
    - Tokens/second: Most common metric
    - Samples/second: Also useful
    - Steps/second: Training iteration speed

    Args:
        num_tokens: Number of tokens processed
        elapsed_time: Time elapsed in seconds

    Returns:
        Dictionary with throughput metrics

    Example:
        >>> throughput = compute_throughput(10000, 5.0)
        >>> print(f"Tokens/sec: {throughput['tokens_per_sec']}")
    """
    return {
        "tokens_per_sec": num_tokens / elapsed_time if elapsed_time > 0 else 0,
        "steps_per_sec": 1.0 / elapsed_time if elapsed_time > 0 else 0,
    }


class MetricsTracker:
    """
    Track and compute multiple metrics during training.

    Educational notes:
    - Aggregates metrics over batches
    - Computes averages and statistics
    - Provides convenient interface

    Args:
        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.update(loss=2.5, num_tokens=1000)
            >>> metrics = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0
        self.start_time = None

    def update(self, loss: float, num_tokens: int):
        """
        Update metrics with new batch.

        Args:
            loss: Batch loss
            num_tokens: Number of tokens in batch
        """
        self.total_loss += loss * num_tokens
        self.total_tokens += num_tokens
        self.num_batches += 1

        if self.start_time is None:
            import time
            self.start_time = time.time()

    def compute(self) -> Dict[str, Any]:
        """
        Compute aggregated metrics.

        Returns:
            Dictionary with computed metrics
        """
        if self.total_tokens == 0:
            return {"loss": 0.0, "perplexity": float("inf")}

        avg_loss = self.total_loss / self.total_tokens
        perplexity = compute_perplexity(avg_loss)

        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_tokens": self.total_tokens,
            "num_batches": self.num_batches,
        }

        return metrics


def test_metrics():
    """
    Test metrics implementation.

    Educational note: Tests verify:
    1. Loss computation is correct
    2. Perplexity calculation is correct
    3. Accuracy computation works
    4. Gradient norm is computed correctly
    """
    print("Testing Metrics implementation...")

    # Test 1: Loss computation
    logits = torch.randn(4, 10, 100)
    targets = torch.randint(0, 100, (4, 10))

    loss = compute_loss(logits, targets)
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Test 1 passed: Loss computation works")

    # Test 2: Perplexity
    loss_tensor = torch.tensor(2.0)
    ppl = compute_perplexity(loss_tensor)

    expected_ppl = 2.718 ** 2.0  # exp(2.0)
    assert abs(ppl - expected_ppl) < 0.1, "Perplexity incorrect"
    print("✓ Test 2 passed: Perplexity computation works")

    # Test 3: Accuracy
    # Create logits where predictions match targets
    logits_match = torch.zeros(4, 10, 100)
    targets_match = torch.randint(0, 100, (4, 10))

    for i in range(4):
        for j in range(10):
            logits_match[i, j, targets_match[i, j]] = 10.0  # High value for target

    acc = compute_accuracy(logits_match, targets_match)
    assert acc == 1.0, f"Accuracy should be 1.0, got {acc}"
    print("✓ Test 3 passed: Accuracy computation works")

    # Test 4: Gradient norm
    # Educational note: backprop a loss that actually depends on the model so
    # its parameters receive gradients.
    model = torch.nn.Linear(10, 10)
    model_out = model(torch.randn(4, 10))
    model_loss = model_out.pow(2).mean()
    model_loss.backward()

    grad_norm = compute_gradient_norm(model)
    assert grad_norm > 0, "Gradient norm should be positive"
    print("✓ Test 4 passed: Gradient norm computation works")

    # Test 5: Metrics tracker
    tracker = MetricsTracker()
    tracker.update(loss=2.5, num_tokens=100)
    tracker.update(loss=2.3, num_tokens=100)

    metrics = tracker.compute()
    assert metrics["loss"] > 0, "Tracked loss incorrect"
    assert metrics["num_tokens"] == 200, "Tracked tokens incorrect"
    print("✓ Test 5 passed: Metrics tracker works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_metrics()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Metrics Demonstration")
    print("=" * 70)

    # Create dummy data
    batch_size = 4
    seq_len = 128
    vocab_size = 32000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Targets: {targets.shape}")

    # Compute metrics
    loss = compute_loss(logits, targets)
    ppl = compute_perplexity(loss)
    acc = compute_accuracy(logits, targets)

    print(f"\nMetrics:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Accuracy: {acc:.2%}")

    # Interpret perplexity
    print(f"\nPerplexity interpretation:")
    print(f"  Model is uncertain among ~{int(ppl)} tokens on average")
    print(f"  Lower perplexity = more confident predictions")

    # Throughput
    num_tokens = batch_size * seq_len
    elapsed_time = 2.5

    throughput = compute_throughput(num_tokens, elapsed_time)

    print(f"\nThroughput (simulated):")
    print(f"  Tokens: {num_tokens}")
    print(f"  Time: {elapsed_time}s")
    print(f"  Tokens/sec: {throughput['tokens_per_sec']:.0f}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Loss: Cross-entropy (lower is better)")
    print("- Perplexity: exp(loss) (intuitive uncertainty measure)")
    print("- Accuracy: Exact match percentage")
    print("- Gradient norm: Check for exploding gradients")
    print("- Throughput: Training speed metrics")
    print("=" * 70)
