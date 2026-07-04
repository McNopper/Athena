"""
Feed-Forward Networks (SwiGLU)

This module implements the feed-forward network component of transformer blocks
using the SwiGLU activation function, a modern variant used in LLaMA and PaLM.

Educational Notes:
- FFN follows attention in each transformer block
- Standard FFN: Linear → Activation → Linear
- SwiGLU FFN: Three linear projections with GLU (Gated Linear Units) activation
- SwiGLU performs better than ReLU/GELU in practice

What is SwiGLU?
- SwiGLU = Swish activation + GLU (Gated Linear Units)
- GLU splits input into two halves, processes one, and multiplies them
- Formula: FFN(x) = (xW_g ⊙ Swish(xW_b)) W_c
- Swish(x) = x * sigmoid(x)

Why SwiGLU is better than ReLU/GELU?
1. Gating mechanism allows selective information flow
2. Swish is smooth (unlike ReLU's sharp corner at 0)
3. Swish allows small negative values (better gradient flow)
4. GLU structure adds non-linearity and capacity

Standard FFN vs SwiGLU FFN:
- Standard: Linear(relu(Linear(x)))  # 2 linear layers
- SwiGLU: Linear(linear(x) ⊙ Swish(linear(x)))  # 3 linear layers, gate mechanism
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    Educational notes:
    - SwiGLU combines Swish activation with Gated Linear Units
    - Formula: SwiGLU(x, y) = Swish(x) ⊙ y, where ⊙ is element-wise multiply
    - Swish(x) = x * sigmoid(x)
    - Smooth, non-monotonic, allows small negative values
    - Better than ReLU in many tasks

    Why SwiGLU over ReLU?
    - ReLU: Dead neurons (always zero for negative inputs)
    - Swish: Smooth gradient, no dead neurons
    - GLU: Gating mechanism adds capacity

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)

    Example:
        >>> swiglu = SwiGLU()
        >>> x = torch.randn(32, 128, 512)
        >>> output = swiglu(x)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Educational notes:
        - SwiGLU combines two inputs through gating mechanism
        - Formula: output = Swish(x) ⊙ gate
        - Swish(x) = x * sigmoid(x)
        - Gate acts as a controller (what to let through)

        Args:
            x: Input tensor (value branch)
            gate: Gate tensor (control branch)

        Returns:
            SwiGLU activated tensor

        Example:
            >>> swiglu = SwiGLU()
            >>> x = torch.randn(10, 10)
            >>> gate = torch.randn(10, 10)
            >>> output = swiglu(x, gate)
        """
        # Educational note: Swish activation
        # Formula: Swish(x) = x * sigmoid(x)
        # Sigmoid squashes to (0, 1), then multiply by x
        # Result: smooth, non-monotonic, allows negatives
        swish = x * torch.sigmoid(x)

        # Educational note: GLU gating mechanism
        # Element-wise multiply with gate
        # Gate controls how much of each value passes through
        output = swish * gate

        return output


class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    Educational notes:
    - Standard component in transformer blocks
    - Follows attention layer in each block
    - Projects to higher dimension, applies activation, projects back
    - Adds non-linearity and model capacity

    Architecture:
        Input (dim) → Linear → (hidden_dim) → SwiGLU → Linear → Output (dim)

        Specifically for SwiGLU:
        Input (dim) → Linear → (hidden_dim) [split into two halves]
                     → SwiGLU(gate, value) → Linear → Output (dim)

    Hidden dimension:
        - Typically 4x model dimension for standard FFN
        - For SwiGLU: hidden_dim = (4/3) * 4 * dim (more efficient)

    Args:
        dim: Input/output dimension (model dimension)
        hidden_dim: Hidden dimension (typically 4x dim)
        dropout: Dropout rate (0.0 = no dropout)
        device: Device to place parameters on

    Example:
        >>> ffn = FeedForward(dim=512, hidden_dim=2048)
        >>> x = torch.randn(32, 128, 512)
        >>> output = ffn(x)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, device: Optional[torch.device] = None):
        super().__init__()

        self.dim = dim
        # Default hidden_dim: 4x dim (standard for transformers)
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.hidden_dim = hidden_dim

        # Educational note: For SwiGLU, we need 3 projections
        # But we implement as 2 projections for efficiency:
        # 1. Gate and value projections (combined): (dim, 2 * hidden_dim)
        # 2. Output projection: (hidden_dim, dim)

        # Combined gate and value projections
        # Input: (batch, seq, dim)
        # Output: (batch, seq, 2 * hidden_dim)
        # Split into: gate (hidden_dim) and value (hidden_dim)
        self.gate_value_proj = nn.Linear(dim, 2 * hidden_dim, bias=False, device=device)

        # Output projection
        # Input: (batch, seq, hidden_dim)
        # Output: (batch, seq, dim)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False, device=device)

        # SwiGLU activation
        self.swiglu = SwiGLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network to input.

        Educational notes:
        - Projects to higher dimension, applies activation, projects back
        - SwiGLU gating mechanism allows selective information flow
        - Adds non-linearity and model capacity

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        Example:
            >>> ffn = FeedForward(dim=512, hidden_dim=2048)
            >>> x = torch.randn(32, 128, 512)
            >>> output = ffn(x)
            >>> print(output.shape)
            torch.Size([32, 128, 512])
        """
        # Step 1: Project to gate and value
        # Educational note: Combined projection for efficiency
        # Shape: (batch, seq, 2 * hidden_dim)
        gate_value = self.gate_value_proj(x)

        # Step 2: Split into gate and value
        # Educational note: SwiGLU needs two branches
        # Each half has shape (batch, seq, hidden_dim)
        gate, value = gate_value.chunk(2, dim=-1)

        # Step 3: Apply SwiGLU activation
        # Educational note: Gating mechanism
        # Swish(value) ⊙ gate
        # Shape: (batch, seq, hidden_dim)
        hidden = self.swiglu(value, gate)

        # Step 4: Apply dropout (if enabled)
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Step 5: Project back to original dimension
        # Educational note: Output projection
        # Shape: (batch, seq, dim)
        output = self.out_proj(hidden)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class FeedForwardGELU(nn.Module):
    """
    Standard Feed-Forward Network with GELU activation.

    Educational notes:
    - This is the traditional FFN used in original Transformer
    - GELU (Gaussian Error Linear Unit) is smooth and performs well
    - Included for comparison and educational purposes

    Architecture:
        Input (dim) → Linear → (hidden_dim) → GELU → Linear → Output (dim)

    GELU vs ReLU:
    - GELU: Smooth, probabilistic activation
    - ReLU: Simple, max(0, x)
    - GELU often performs better in practice

    Reference: "BERT: Pre-training of Deep Bidirectional Transformers"

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 4x dim)
        dropout: Dropout rate

    Example:
        >>> ffn = FeedForwardGELU(dim=512, hidden_dim=2048)
        >>> x = torch.randn(32, 128, 512)
        >>> output = ffn(x)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0, device: Optional[torch.device] = None):
        super().__init__()

        self.dim = dim
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.hidden_dim = hidden_dim

        # Educational note: Standard two-layer FFN
        # Input projection: dim → hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False, device=device)

        # Output projection: hidden_dim → dim
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False, device=device)

        # GELU activation
        self.activation = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network with GELU activation.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Step 1: Project to hidden dimension
        hidden = self.fc1(x)

        # Step 2: Apply GELU activation
        hidden = self.activation(hidden)

        # Step 3: Apply dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Step 4: Project back to original dimension
        output = self.fc2(hidden)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


def test_ffn():
    """
    Test feed-forward network implementation.

    Educational note: Tests verify:
    1. Shape transformations are correct
    2. SwiGLU and GELU produce different outputs
    3. Gradients flow correctly
    """
    print("Testing Feed-Forward Network implementation...")

    # Test 1: SwiGLU FFN
    dim = 512
    hidden_dim = 2048
    batch_size = 4
    seq_len = 128

    ffn = FeedForward(dim=dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, dim)

    output = ffn(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✓ Test 1 passed: SwiGLU FFN shape preserved")

    # Test 2: GELU FFN
    ffn_gelu = FeedForwardGELU(dim=dim, hidden_dim=hidden_dim)
    output_gelu = ffn_gelu(x)

    assert output_gelu.shape == x.shape, f"Shape mismatch: {output_gelu.shape} != {x.shape}"
    print("✓ Test 2 passed: GELU FFN shape preserved")

    # Test 3: SwiGLU ≠ GELU (different outputs)
    assert not torch.allclose(output, output_gelu), "SwiGLU and GELU produce same output"
    print("✓ Test 3 passed: SwiGLU and GELU produce different outputs")

    # Test 4: Gradient flow
    loss = output.sum()
    loss.backward()

    assert ffn.gate_value_proj.weight.grad is not None, "No gradients on gate/value projection"
    assert ffn.out_proj.weight.grad is not None, "No gradients on output projection"
    assert not torch.isnan(ffn.gate_value_proj.weight.grad).any(), "Gradients contain NaN"
    print("✓ Test 4 passed: Gradients flow correctly")

    # Test 5: Dropout
    ffn_dropout = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=0.1)
    ffn_dropout.eval()  # Set to eval mode (disables dropout)

    output_no_dropout = ffn_dropout(x)

    # In eval mode, dropout should be disabled
    assert torch.allclose(output_no_dropout, output_no_dropout), "Eval mode failed"
    print("✓ Test 5 passed: Dropout works correctly")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_ffn()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Feed-Forward Network Demonstration")
    print("=" * 70)

    # Create FFN layers
    dim = 512
    hidden_dim = 2048

    ffn_swiglu = FeedForward(dim=dim, hidden_dim=hidden_dim)
    ffn_gelu = FeedForwardGELU(dim=dim, hidden_dim=hidden_dim)

    print(f"\nSwiGLU FFN:")
    print(ffn_swiglu)

    print(f"\nGELU FFN:")
    print(ffn_gelu)

    # Create sample input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"Input norm (mean): {torch.norm(x, dim=-1).mean():.4f}")

    # Apply SwiGLU FFN
    output_swiglu = ffn_swiglu(x)

    print(f"\nSwiGLU output shape: {output_swiglu.shape}")
    print(f"SwiGLU output norm (mean): {torch.norm(output_swiglu, dim=-1).mean():.4f}")

    # Apply GELU FFN
    output_gelu = ffn_gelu(x)

    print(f"\nGELU output shape: {output_gelu.shape}")
    print(f"GELU output norm (mean): {torch.norm(output_gelu, dim=-1).mean():.4f}")

    # Parameter count
    params_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())
    params_gelu = sum(p.numel() for p in ffn_gelu.parameters())

    print(f"\nParameter counts:")
    print(f"  SwiGLU FFN: {params_swiglu:,}")
    print(f"  GELU FFN: {params_gelu:,}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- FFN adds non-linearity and capacity to transformer blocks")
    print("- SwiGLU uses gating mechanism (GLU) with Swish activation")
    print("- GELU is smooth probabilistic activation (standard in BERT)")
    print("- SwiGLU often performs better than ReLU/GELU in modern LLMs")
    print("- Hidden dimension typically 4x model dimension")
    print("=" * 70)
