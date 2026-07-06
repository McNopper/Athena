"""
Rotary Position Embeddings (RoPE)

This module implements Rotary Position Embeddings, a modern approach to encoding
position information in transformer models.

Educational Notes:
- Traditional approaches: Add positional embeddings to token embeddings
- RoPE approach: Rotate token embeddings based on position
- Key insight: Rotating Q and K by position-dependent angles encodes relative
  distance directly into the attention dot-product — no extra parameters
- RoPE is used in LLaMA, LLaMA 2, PaLM, and many modern LLMs

Why RoPE is better than absolute position embeddings?
1. Relative position encoding: Naturally captures "token A is N positions from token B"
2. No learned parameters: Position encoding is deterministic
3. Better extrapolation: Handles longer sequences than seen during training
4. Multi-dimensional: Encodes position in the embedding space, not added to it

How RoPE works:
1. Represent token embeddings as complex numbers
2. Rotate embeddings by angle proportional to position
3. Rotation angle depends on dimension (different frequencies for different dims)
4. Apply rotation to Query and Key matrices before attention

Mathematical intuition:
- Rotation in complex plane: (x + yi) * (cos θ + i sin θ)
- This preserves vector magnitude (norm) while encoding angle (position)
- Different dimensions rotate at different frequencies (like musical notes)

References:
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for transformer models.

    Educational notes:
    - RoPE rotates token embeddings based on their absolute position
    - Rotation angle depends on both position and dimension index
    - Different dimensions use different frequencies (like FFT basis)
    - Applied to Query and Key matrices before attention computation

    Key formula:
        For position m and dimension i (where 2i and 2i+1 are pairs):
        θ(m, i) = m / (base ^ (2i / dim))

        where base is typically 10,000

        This creates different rotation frequencies for different dimensions

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Used in: LLaMA, LLaMA 2, PaLM, and many modern LLMs

    Args:
        dim: Dimension of the embeddings (must be even)
        max_sequence_length: Maximum sequence length for pre-computed frequencies
        base: Base frequency (10,000 is standard)
        device: Device to place pre-computed frequencies on

    Example:
        >>> rope = RotaryPositionEmbedding(dim=512, max_sequence_length=2048)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq, dim)
        >>> rotated = rope(x, positions=0)
        >>> print(rotated.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(self, dim: int, max_sequence_length: int = 2048, base: float = 10000.0, device: Optional[torch.device] = None):
        super().__init__()

        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_sequence_length = max_sequence_length
        self.base = base

        # Pre-compute inverse frequencies
        # Educational note: These are the rotation frequencies for each dimension pair
        inv_freq = self._compute_inv_freq(dim, base)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute position-specific rotation matrices
        # Educational note: Cache these for efficiency
        self._build_cache(max_sequence_length)

    def _compute_inv_freq(self, dim: int, base: float) -> torch.Tensor:
        """
        Compute inverse frequencies for RoPE.

        Educational notes:
        - Formula: θ_i = 1 / (base ^ (2i / dim))
        - Creates geometric sequence of frequencies
        - Each dimension pair rotates at different frequency
        - Lower dimensions rotate slower (capture long-range relationships)
        - Higher dimensions rotate faster (capture short-term relationships)

        Analogy: Like musical notes - different frequencies create harmony

        Args:
            dim: Embedding dimension
            base: Base frequency

        Returns:
            Tensor of shape (dim // 2,) containing inverse frequencies
        """
        # Dimension indices for pairs: [0, 2, 4, 6, ..., dim-2]
        # We have dim/2 pairs of dimensions
        dims = torch.arange(0, dim, 2, dtype=torch.float32)  # [0, 2, 4, ..., dim-2]

        # Compute inverse frequencies
        # Formula: 1 / (base ^ (i / (dim / 2)))
        # Simplified: base ^ (-2i / dim)
        inv_freq = 1.0 / (base ** (dims / dim))  # Shape: (dim // 2,)

        return inv_freq

    def _build_cache(self, seq_len: int):
        """
        Pre-compute rotation matrices for all positions up to seq_len.

        Educational notes:
        - Cosine and sine values for each position and dimension
        - Shape: (seq_len, dim // 2)
        - Cached for efficiency during training/inference
        """
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        # Compute rotation frequencies for each position
        # Outer product: positions × inv_freq
        # Result shape: (seq_len, dim // 2)
        # Educational note: freqs[m, i] = m * theta_i is the rotation angle for
        # dimension pair i at position m. We keep one angle per pair (dim // 2
        # angles), which is exactly what a 2D rotation of the pair (2i, 2i+1)
        # needs. Cosine and sine are cached for efficiency.
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)

        cos_cached = torch.cos(freqs)  # (seq_len, dim // 2)
        sin_cached = torch.sin(freqs)  # (seq_len, dim // 2)

        # Register as buffers (persistent=False = don't save in checkpoint)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.

        Educational notes:
        - Input shape: (batch_size, seq_len, dim) or (batch_size, num_heads, seq_len, head_dim)
        - Rotates each dimension pair (2i, 2i+1) by angle m * theta_i
        - Uses pre-computed cosine and sine values for efficiency
        - Being a true 2D rotation, it preserves the norm of every pair

        Rotation formula (applied to each dimension pair):
            For dimensions (2i, 2i+1) at position m:
            x'_2i   = x_2i * cos(m * theta_i) - x_2i+1 * sin(m * theta_i)
            x'_2i+1 = x_2i * sin(m * theta_i) + x_2i+1 * cos(m * theta_i)

        Args:
            x: Input tensor of shape (..., seq_len, dim)
            positions: Optional absolute positions. Can be:
                       - None: use [0, 1, ..., seq_len-1]
                       - an int: use [start, start+1, ..., start+seq_len-1]
                         (useful for KV-cached incremental decoding)
                       - a 1D LongTensor of length seq_len with explicit positions

        Returns:
            Rotated tensor of same shape as input
        """
        seq_len = x.shape[-2]

        # Resolve absolute positions for each element in the sequence dimension.
        if positions is None:
            pos = torch.arange(seq_len, device=x.device)
        elif isinstance(positions, int):
            pos = torch.arange(positions, positions + seq_len, device=x.device)
        else:
            pos = positions.to(device=x.device).long()

        # Extend the cache if we are asked for a position beyond what we cached.
        max_pos = int(pos.max().item()) + 1 if pos.numel() > 0 else seq_len
        if max_pos > self.max_sequence_length:
            self.max_sequence_length = max_pos
            self._build_cache(max_pos)

        # Gather cos/sin for the requested positions.
        # Shape: (seq_len, dim // 2)
        cos = self.cos_cached[pos]
        sin = self.sin_cached[pos]

        # Split into even and odd dimensions (the two halves of each pair).
        # Shapes: (..., seq_len, dim // 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply the 2D rotation to every pair. cos/sin broadcast over the
        # leading (batch/head) dimensions because they align on (seq_len, dim//2).
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # Interleave the rotated even/odd dimensions back to the original layout.
        output = torch.stack((x_even_rotated, x_odd_rotated), dim=-1)
        output = output.flatten(-2)

        return output.type_as(x)

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"dim={self.dim}, max_seq_len={self.max_sequence_length}, base={self.base}"


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: RotaryPositionEmbedding,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Query and Key matrices.

    Educational notes:
    - RoPE is applied to both Query and Key before attention computation
    - This makes attention scores position-aware
    - The same position encoding is applied to both Q and K
    - Result: attention scores depend on relative positions, not absolute

    Why apply to both Q and K?
    - Attention score = Q * K^T
    - For positions m and n the rotation leaves a dependence on (m - n),
      i.e. the relative position, which is exactly what we want.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        rope: RotaryPositionEmbedding module
        start_pos: Absolute position of the first token in q/k. This is > 0
                   during KV-cached incremental decoding, where q/k only hold
                   the newly generated tokens.

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Apply RoPE to both Query and Key at their absolute positions.
    q_rotated = rope(q, positions=start_pos)
    k_rotated = rope(k, positions=start_pos)

    return q_rotated, k_rotated


def test_rope():
    """
    Test Rotary Position Embedding implementation.

    Educational note: RoPE should:
    1. Preserve tensor shape
    2. Change values based on position
    3. Preserve relative distances between tokens
    """
    print("Testing RoPE implementation...")

    # Test 1: Basic functionality
    dim = 512
    seq_len = 128
    batch_size = 4

    rope = RotaryPositionEmbedding(dim=dim, max_sequence_length=seq_len)
    x = torch.randn(batch_size, seq_len, dim)

    # Apply RoPE
    output = rope(x)

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✓ Test 1 passed: Shape preserved")

    # Test 2: Position-dependent encoding
    # RoPE should modify values based on position
    output_pos0 = rope(x[:, 0:1, :], positions=torch.tensor([0]))
    output_pos10 = rope(x[:, 10:11, :], positions=torch.tensor([10]))

    # Different positions should give different rotations
    assert not torch.allclose(output_pos0, output_pos10), "Position encoding not working"
    print("✓ Test 2 passed: Position-dependent encoding works")

    # Test 3: Magnitude preservation
    # Rotation should preserve vector magnitudes
    original_norm = torch.norm(x, dim=-1)
    rotated_norm = torch.norm(output, dim=-1)

    # Norms should be very close (rotation preserves magnitude)
    assert torch.allclose(original_norm, rotated_norm, atol=1e-5), "Rotation changed magnitude"
    print("✓ Test 3 passed: Magnitude preserved")

    # Test 4: Q and K application
    # Test the common use case: applying RoPE to Query and Key
    num_heads = 8
    head_dim = dim // num_heads

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    rope_qk = RotaryPositionEmbedding(dim=head_dim, max_sequence_length=seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, rope_qk)

    assert q_rot.shape == q.shape, "Query shape changed"
    assert k_rot.shape == k.shape, "Key shape changed"
    print("✓ Test 4 passed: Q and K rotation works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_rope()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Rotary Position Embedding Demonstration")
    print("=" * 70)

    # Create RoPE module
    dim = 512
    max_seq_len = 2048
    rope = RotaryPositionEmbedding(dim=dim, max_sequence_length=max_seq_len)

    print(f"\nRoPE module:")
    print(rope)

    # Create sample input
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"Input norm (mean): {torch.norm(x, dim=-1).mean():.4f}")

    # Apply RoPE
    output = rope(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output norm (mean): {torch.norm(output, dim=-1).mean():.4f}")
    print(f"Norm preserved: {torch.allclose(torch.norm(x, dim=-1), torch.norm(output, dim=-1), atol=1e-5)}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- RoPE rotates embeddings based on position")
    print("- Different dimensions rotate at different frequencies")
    print("- Rotation preserves vector magnitude (norm)")
    print("- Applied to Query and Key before attention computation")
    print("- Naturally encodes relative positions, not absolute")
    print("- No learned parameters (unlike traditional position embeddings)")
    print("=" * 70)
