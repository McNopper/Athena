"""
Layer Normalization

This module implements RMS Layer Normalization (RMSNorm), a simplified variant
of Layer Normalization used in modern LLMs like LLaMA.

Educational Notes:
- Layer Normalization stabilizes training by normalizing activations
- RMSNorm is simpler than standard LayerNorm (no mean centering)
- RMSNorm: output = x / sqrt(mean(x^2) + eps) * gamma
- Standard LayerNorm: output = (x - mean) / sqrt(var) * gamma + beta
- RMSNorm has fewer parameters (no beta) and is computationally simpler
- Despite being simpler, RMSNorm often performs as well as LayerNorm

Why RMSNorm over LayerNorm?
1. Simpler: No mean centering, no beta parameter
2. Faster: Less computation, better for large models
3. Effective: Works just as well in practice for transformers

References:
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) — https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Educational notes:
    - Normalizes activations using RMS (Root Mean Square)
    - Simplifies standard LayerNorm by removing mean centering
    - Only has a learnable gain (gamma) parameter, no bias (beta)
    - Formula: output = x * gamma / sqrt(mean(x^2) + epsilon)

    Standard LayerNorm vs RMSNorm:
    - LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    - RMSNorm: x / sqrt(mean(x^2) + eps) * gamma

    Reference: "Root Mean Square Layer Normalization" (Zhang et al., 2019)
    Used in: LLaMA, LLaMA 2, and many modern LLMs

    Args:
        dim: Dimension of the input tensor (last dimension is normalized)
        eps: Epsilon for numerical stability (prevents division by zero)
        device: Device to place parameters on

    Example:
        >>> rms_norm = RMSNorm(dim=512)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq, dim)
        >>> normalized = rms_norm(x)
        >>> print(normalized.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(self, dim: int, eps: float = 1e-6, device: Optional[torch.device] = None):
        super().__init__()

        self.dim = dim
        self.eps = eps

        # Learnable gain parameter (gamma)
        # Educational note: Initialize to 1.0 so RMSNorm is identity at start
        self.gamma = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS Normalization to the input tensor.

        Educational notes:
        - Normalization is applied over the last dimension (feature dimension)
        - For a batch of sequences: (batch, seq_len, dim) -> normalize over dim
        - The gamma parameter scales the normalized values

        Steps:
        1. Square the input: x^2
        2. Compute mean of squared values: mean(x^2)
        3. Add epsilon for stability: mean(x^2) + eps
        4. Take square root: sqrt(...)
        5. Divide input by RMS: x / RMS
        6. Scale by gamma: * gamma

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape as input

        Example:
            >>> rms_norm = RMSNorm(dim=4)
            >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            >>> normalized = rms_norm(x)
            >>> # Values will be normalized to unit RMS
        """
        # Educational: Compute RMS (Root Mean Square)
        # Formula: RMS = sqrt(mean(x^2) + epsilon)
        #
        # Step 1: Square the input
        squared = torch.square(x.float())

        # Step 2: Compute mean over the last dimension (feature dimension)
        # keepdim=True preserves the shape for broadcasting
        mean_square = torch.mean(squared, dim=-1, keepdim=True)

        # Step 3: Add epsilon for numerical stability
        # This prevents division by zero if all inputs are zero
        rms = torch.sqrt(mean_square + self.eps)

        # Step 4: Normalize and scale
        # Divide by RMS, then multiply by learnable gamma
        output = x.to(torch.float32) / rms * self.gamma

        return output.type_as(x)  # Return same dtype as input

    def extra_repr(self) -> str:
        """
        Extra representation for printing the module.

        Educational note: This is what shows up when you print(model)
        """
        return f"dim={self.dim}, eps={self.eps}"


def test_rms_norm():
    """
    Test RMSNorm implementation.

    Educational note: Always test normalization layers to ensure:
    1. Output shape matches input shape
    2. Normalization actually changes the distribution
    3. gradients flow correctly through the layer
    """
    print("Testing RMSNorm implementation...")

    # Test 1: Basic functionality
    dim = 128
    batch_size = 32
    seq_len = 64

    rms_norm = RMSNorm(dim=dim)
    x = torch.randn(batch_size, seq_len, dim)

    # Forward pass
    output = rms_norm(x)

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✓ Test 1 passed: Shape preserved")

    # Test 2: Check normalization effect
    # Compute RMS of output
    output_rms = torch.sqrt(torch.mean(output**2, dim=-1))
    # RMS should be close to 1.0 (since gamma is initialized to 1.0)
    assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-4), "RMS not normalized to ~1"
    print("✓ Test 2 passed: RMS normalized to ~1")

    # Test 3: Check gradient flow
    loss = output.sum()
    loss.backward()

    # Check that gamma has gradients
    assert rms_norm.gamma.grad is not None, "Gamma has no gradients"
    assert not torch.isnan(rms_norm.gamma.grad).any(), "Gradient contains NaN"
    print("✓ Test 3 passed: Gradients flow correctly")

    # Test 4: Learnable parameter
    # Change gamma and verify it affects output
    original_gamma = rms_norm.gamma.data.clone()
    rms_norm.gamma.data.fill_(2.0)  # Multiply output by 2

    output_new = rms_norm(x)
    # Output should be doubled (approximately, since normalization happens first)
    assert torch.allclose(output_new, output * 2.0, atol=1e-4), "Gamma scaling not working"
    print("✓ Test 4 passed: Gamma parameter works correctly")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_rms_norm()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("RMSNorm Demonstration")
    print("=" * 70)

    # Create RMSNorm layer
    dim = 512
    rms_norm = RMSNorm(dim=dim)

    print(f"\nRMSNorm layer:")
    print(rms_norm)

    # Create sample input
    batch_size = 4
    seq_len = 128
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"Input RMS (mean over batch and seq): {torch.sqrt(torch.mean(x**2)):.4f}")

    # Apply RMSNorm
    output = rms_norm(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output RMS (mean over batch and seq): {torch.sqrt(torch.mean(output**2)):.4f}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- RMSNorm normalizes activations to unit RMS")
    print("- Gamma parameter scales the normalized values")
    print("- Simpler than LayerNorm: no mean centering, no beta")
    print("- Works well in practice for transformer models")
    print("=" * 70)
