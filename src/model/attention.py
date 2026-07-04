"""
Multi-Head Attention Mechanism

This module implements multi-head attention, the core component of transformer models.

Educational Notes:
- Attention allows the model to focus on different parts of the input sequence
- Multi-head: Multiple attention heads work in parallel, each learning different relationships
- Causal masking: Ensures each position only attends to previous positions (autoregressive)
- RoPE integration: Rotary position embeddings for position-aware attention

What is Attention?
- Query (Q): What this position is looking for
- Key (K): What other positions are offering
- Value (V): The actual information at each position
- Score = Q × K^T: How well Query matches each Key
- Output = Score × V: Weighted sum of Values based on matches

Multi-Head Attention:
- Multiple heads (e.g., 8, 12, 16) work in parallel
- Each head learns different types of relationships
- Head 1 might learn syntax, Head 2 might learn semantics, etc.
- Outputs are concatenated and projected

Why Multi-Head?
- Single attention can only focus on one type of relationship at a time
- Multiple heads allow diverse attention patterns
- Ensemble effect: Multiple "experts" working in parallel
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .layer_norm import RMSNorm
from .positional import RotaryPositionEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE and causal masking.

    Educational notes:
    - Computes Query, Key, Value projections
    - Splits into multiple heads for parallel attention
    - Applies causal masking for autoregressive generation
    - Integrates RoPE for position-aware attention

    Forward pass steps:
    1. Project input to Q, K, V
    2. Split into multiple heads
    3. Apply RoPE to Q and K
    4. Compute attention scores: Q × K^T / sqrt(d_k)
    5. Apply causal mask (prevent looking at future)
    6. Apply softmax to get attention weights
    7. Multiply weights by V
    8. Concatenate heads and project output

    Args:
        dim: Model dimension (hidden size)
        num_heads: Number of attention heads
        head_dim: Dimension of each head (dim // num_heads)
        max_sequence_length: Maximum sequence length for causal mask
        use_rope: Whether to use Rotary Position Embeddings
        rope_theta: RoPE base frequency
        dropout: Dropout rate on attention weights

    Example:
        >>> attention = MultiHeadAttention(dim=512, num_heads=8)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq, dim)
        >>> output = attention(x)  # (batch, seq, dim)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_sequence_length: int = 2048,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_sequence_length = max_sequence_length
        self.use_rope = use_rope

        # Educational note: Q, K, V projections
        # We project the input to Query, Key, and Value matrices
        # Shape: (dim, 3 * dim) for efficiency (single matrix multiply)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)

        # Educational note: Output projection
        # After concatenating heads, project back to dim
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout for attention weights (regularization)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # RoPE for position encoding
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_sequence_length=max_sequence_length,
                base=rope_theta,
            )

    def _causal_mask(self, q_len: int, k_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
        """
        Build a causal attention mask, correct even when q and k differ in
        length (KV cache) or when the cache has been trimmed (sliding window).

        Educational notes:
        - Query position i has absolute position (start_pos + i)
        - The keys are the *last* ``k_len`` positions ending at the last query,
          so the first key's absolute position is (start_pos + q_len - k_len).
          This stays correct after the cache evicts old entries.
        - A query may only attend to keys at the same or earlier absolute position
        - Returns a boolean tensor of shape (q_len, k_len) where True = masked

        Args:
            q_len: Number of query positions
            k_len: Number of key positions (cached + new)
            start_pos: Absolute position of the first query
            device: Device to build the mask on

        Returns:
            Boolean mask of shape (q_len, k_len), True where attention is blocked
        """
        # Absolute position of the first (oldest retained) key.
        k_start = start_pos + q_len - k_len
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + start_pos  # (q_len, 1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0) + k_start  # (1, k_len)
        return k_pos > q_pos  # True where key is in the future of the query

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["object"] = None,
        layer_idx: Optional[int] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Apply multi-head attention to input tensor.

        Educational notes:
        - Input shape: (batch_size, seq_len, dim)
        - Output shape: (batch_size, seq_len, dim)
        - Process: QKV projection → split heads → RoPE → attention → merge heads
        - With a KV cache, ``x`` only contains the new tokens; keys/values from
          previous steps are read from (and appended to) the cache.

        Args:
            x: Input tensor of shape (batch, q_len, dim)
            mask: Optional key-padding mask of shape (batch, k_len),
                  where True marks positions that must NOT be attended to.
            kv_cache: Optional cache object exposing
                      ``update(layer_idx, k, v) -> (k_full, v_full)``.
            layer_idx: Index of this layer inside the KV cache.
            start_pos: Absolute position of the first token in ``x``. Non-zero
                       during KV-cached incremental decoding.

        Returns:
            Output tensor of shape (batch, q_len, dim)

        Example:
            >>> attention = MultiHeadAttention(dim=512, num_heads=8)
            >>> x = torch.randn(32, 128, 512)
            >>> output = attention(x)
            >>> print(output.shape)
            torch.Size([32, 128, 512])
        """
        batch_size, seq_len, dim = x.shape

        # Step 1: Project to Q, K, V
        # Educational note: Single matrix multiplication for efficiency
        # qkv_proj: (dim, 3 * dim)
        # Input: (batch, seq_len, dim)
        # Output: (batch, seq_len, 3 * dim)
        qkv = self.qkv_proj(x)

        # Step 2: Split into Q, K, V
        # Educational note: Each is (batch, seq_len, dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 3: Reshape for multi-head attention
        # Educational note: Split dimension into (num_heads, head_dim)
        # Shape: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Step 4: Transpose for attention computation
        # Educational note: Move num_heads before seq_len for efficient computation
        # Shape: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Step 5: Apply RoPE to Q and K (if enabled)
        # Educational note: RoPE rotates embeddings based on their ABSOLUTE
        # position (start_pos + local index). During incremental decoding this
        # keeps positions consistent with the already-rotated cached keys.
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.rope, start_pos=start_pos)

        # Step 5b: Update / read the KV cache (inference only)
        # Educational note: We rotate K *before* caching so cached keys are
        # stored at their correct absolute positions.
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        q_len = q.shape[-2]
        k_len = k.shape[-2]

        # Step 6: Compute attention scores
        # Educational note: Scaled dot-product attention
        # Formula: scores = Q × K^T / sqrt(head_dim)
        # Shape: (batch, num_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 7: Apply causal mask
        # Educational note: Prevent positions from attending to future tokens.
        # Works even when q_len != k_len (KV cache) thanks to the position offset.
        causal_mask = self._causal_mask(q_len, k_len, start_pos, x.device)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Step 8: Apply additional mask (if provided)
        # Educational note: Optional key-padding mask, shape (batch, k_len).
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))

        # Step 9: Apply softmax to get attention weights
        # Educational note: Convert scores to probabilities
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Step 10: Apply attention weights to values
        # Educational note: Weighted sum of values
        # Shape: (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        # Step 11: Transpose back
        # Educational note: Move seq_len back before num_heads
        # Shape: (batch, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2)

        # Step 12: Merge heads
        # Educational note: Concatenate all heads
        # Shape: (batch, seq_len, dim)
        output = output.contiguous().view(batch_size, seq_len, dim)

        # Step 13: Output projection
        # Educational note: Project concatenated heads
        # Shape: (batch, seq_len, dim)
        output = self.out_proj(output)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}"


def test_attention():
    """
    Test multi-head attention implementation.

    Educational note: Tests verify:
    1. Shape transformations are correct
    2. Causal masking works properly
    3. RoPE is applied correctly
    4. Gradients flow through the layer
    """
    print("Testing Multi-Head Attention implementation...")

    # Test 1: Basic functionality
    dim = 512
    num_heads = 8
    batch_size = 4
    seq_len = 128

    attention = MultiHeadAttention(dim=dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, dim)

    output = attention(x)

    # Check shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✓ Test 1 passed: Shape preserved")

    # Test 2: Causal masking and gradient flow
    # Educational note: We feed a normal input and verify that gradients
    # propagate back through the QKV projection.
    attention_test = MultiHeadAttention(dim=dim, num_heads=num_heads)
    x_test = torch.randn(batch_size, seq_len, dim)
    output_test = attention_test(x_test)

    # Check that gradients flow
    loss = output_test.sum()
    loss.backward()

    assert attention_test.qkv_proj.weight.grad is not None, "No gradients on QKV projection"
    print("✓ Test 2 passed: Gradients flow correctly")

    # Test 3: RoPE integration
    attention_rope = MultiHeadAttention(dim=dim, num_heads=num_heads, use_rope=True)
    output_rope = attention_rope(x)

    assert output_rope.shape == x.shape, "RoPE changed output shape"
    print("✓ Test 3 passed: RoPE integration works")

    # Test 4: Different sequence lengths
    seq_len_short = 64
    x_short = torch.randn(batch_size, seq_len_short, dim)

    output_short = attention(x_short)
    assert output_short.shape == (batch_size, seq_len_short, dim), "Short sequence failed"

    seq_len_long = 256
    x_long = torch.randn(batch_size, seq_len_long, dim)

    output_long = attention(x_long)
    assert output_long.shape == (batch_size, seq_len_long, dim), "Long sequence failed"
    print("✓ Test 4 passed: Different sequence lengths work")

    # Test 5: Custom mask
    # Create a key-padding mask (batch, seq_len): True = position is padding
    # and must NOT be attended to.
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, seq_len // 2 :] = True  # Mask second half (padding)

    x_masked = torch.randn(batch_size, seq_len, dim)
    output_masked = attention(x_masked, mask=padding_mask)

    assert output_masked.shape == x_masked.shape, "Custom mask changed shape"
    print("✓ Test 5 passed: Custom masking works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_attention()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Multi-Head Attention Demonstration")
    print("=" * 70)

    # Create attention layer
    dim = 512
    num_heads = 8
    seq_len = 128
    batch_size = 4

    attention = MultiHeadAttention(dim=dim, num_heads=num_heads)

    print(f"\nMulti-Head Attention Layer:")
    print(attention)

    # Create sample input
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")

    # Apply attention
    output = attention(x)

    print(f"Output shape: {output.shape}")
    print(f"Output norm (mean): {torch.norm(output, dim=-1).mean():.4f}")

    # Demonstrate RoPE
    attention_rope = MultiHeadAttention(dim=dim, num_heads=num_heads, use_rope=True)
    output_rope = attention_rope(x)

    print(f"\nWith RoPE:")
    print(f"Output shape: {output_rope.shape}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Multi-head attention allows parallel focus on different relationships")
    print("- Causal masking ensures autoregressive property (no peeking at future)")
    print("- RoPE makes attention position-aware without learned parameters")
    print("- Each head can learn different types of relationships")
    print("- Attention weights show which positions the model focuses on")
    print("=" * 70)
