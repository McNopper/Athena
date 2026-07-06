"""
Transformer Block

This module implements a complete transformer block, combining attention and
feed-forward networks with residual connections and normalization.

Educational Notes:
- Transformer block is the repeating unit in transformer models
- Architecture: Attention → Norm → FFN → Norm (with residual connections)
- Pre-normalization: Norm before attention/FFN (modern approach, used in LLaMA)
- Post-normalization: Norm after attention/FFN (original Transformer)
- Residual connections: Add input back to output (helps gradient flow)

Block Architecture (Pre-Norm - Modern):
    Input
      ↓
    RMSNorm ← Pre-normalization
      ↓
    Multi-Head Attention
      ↓
    Add (Residual Connection)
      ↓
    RMSNorm ← Pre-normalization
      ↓
    Feed-Forward Network (SwiGLU)
      ↓
    Add (Residual Connection)
      ↓
    Output

Why Pre-Norm?
- More stable training gradients (no vanishing/exploding at depth)
- Allows deeper networks without careful initialisation tuning
- Used in modern LLMs (LLaMA, GPT-3, etc.)

Residual Connections:
- Formula: output = layer(input) + input
- Helps gradients flow through deep networks
- Allows layers to learn modifications rather than full transformations
- Critical for training very deep models (100+ layers)

References:
    Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
    Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) — https://arxiv.org/abs/2302.13971
"""

import torch
import torch.nn as nn
from typing import Optional

from .layer_norm import RMSNorm
from .attention import MultiHeadAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """
    Complete transformer block with attention and feed-forward layers.

    Educational notes:
    - Combines multi-head attention and feed-forward network
    - Uses pre-normalization (modern approach)
    - Residual connections for gradient flow
    - This is the repeating unit stacked to form deep transformers

    Architecture:
        x → RMSNorm → Attention → Add → RMSNorm → FFN → Add → output

    Each component:
    - RMSNorm: Stabilizes training (normalizes activations)
    - Attention: Allows focusing on different parts of sequence
    - FFN: Adds non-linearity and capacity
    - Residual: Helps gradient flow in deep networks

    Args:
        dim: Model dimension (hidden size)
        num_heads: Number of attention heads
        hidden_dim: Feed-forward hidden dimension
        max_sequence_length: Maximum sequence length for attention
        use_rope: Whether to use Rotary Position Embeddings
        rope_theta: RoPE base frequency
        dropout: Dropout rate

    Example:
        >>> block = TransformerBlock(dim=512, num_heads=8)
        >>> x = torch.randn(32, 128, 512)  # (batch, seq, dim)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        max_sequence_length: int = 2048,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Educational note: Pre-normalization before attention
        # Modern approach (LLaMA, GPT-3, etc.)
        self.attention_norm = RMSNorm(dim)

        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            use_rope=use_rope,
            rope_theta=rope_theta,
            dropout=dropout,
        )

        # Educational note: Pre-normalization before FFN
        self.ffn_norm = RMSNorm(dim)

        # Feed-forward network with SwiGLU
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["object"] = None,
        layer_idx: Optional[int] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Apply transformer block to input.

        Educational notes:
        - Process: Norm → Attention → Residual → Norm → FFN → Residual
        - Residual connections: output = layer(input) + input
        - Pre-normalization: Norm before layer (modern approach)

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional key-padding mask
            kv_cache: Optional KV cache for incremental decoding
            layer_idx: Index of this block inside the KV cache
            start_pos: Absolute position of the first token in ``x``

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        Example:
            >>> block = TransformerBlock(dim=512, num_heads=8)
            >>> x = torch.randn(32, 128, 512)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([32, 128, 512])
        """
        # Educational note: Save input for residual connection
        residual = x

        # Step 1: Pre-normalization before attention
        # Educational note: Modern approach - normalize before layer
        # This helps training stability
        normalized = self.attention_norm(x)

        # Step 2: Apply multi-head attention
        # Educational note: Compute attention with causal masking
        attn_output = self.attention(
            normalized,
            mask=mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
        )

        # Step 3: Residual connection
        # Educational note: Add input back to attention output
        # Formula: x = x + Attention(Norm(x))
        # This helps gradients flow through deep networks
        x = residual + attn_output

        # Step 4: Save for next residual connection
        residual = x

        # Step 5: Pre-normalization before FFN
        normalized = self.ffn_norm(x)

        # Step 6: Apply feed-forward network
        # Educational note: Add non-linearity and capacity
        ffn_output = self.ffn(normalized)

        # Step 7: Residual connection
        # Educational note: Add input back to FFN output
        # Formula: x = x + FFN(Norm(x))
        x = residual + ffn_output

        return x

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"dim={self.dim}, num_heads={self.num_heads}"


class TransformerStack(nn.Module):
    """
    Stack of multiple transformer blocks.

    Educational notes:
    - Stacks multiple transformer blocks sequentially
    - Deep models (12-100+ layers) by stacking blocks
    - Each block learns increasingly complex patterns
    - Lower layers: simple patterns (syntax, local structure)
    - Higher layers: complex patterns (semantics, long-range dependencies)

    Args:
        num_layers: Number of transformer blocks to stack
        dim: Model dimension
        num_heads: Number of attention heads
        hidden_dim: Feed-forward hidden dimension
        max_sequence_length: Maximum sequence length
        use_rope: Whether to use RoPE
        dropout: Dropout rate

    Example:
        >>> stack = TransformerStack(num_layers=12, dim=512, num_heads=8)
        >>> x = torch.randn(32, 128, 512)
        >>> output = stack(x)
        >>> print(output.shape)
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        max_sequence_length: int = 2048,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dim = dim

        # Educational note: Create stack of transformer blocks
        # nn.ModuleList is preferred over Python list for PyTorch models
        # It ensures parameters are properly registered
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    max_sequence_length=max_sequence_length,
                    use_rope=use_rope,
                    rope_theta=rope_theta,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["object"] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Apply stack of transformer blocks to input.

        Educational notes:
        - Sequentially applies each block
        - Output of block i is input to block i+1
        - Deep stack allows learning hierarchical representations

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional key-padding mask
            kv_cache: Optional KV cache for incremental decoding
            start_pos: Absolute position of the first token in ``x``

        Returns:
            Output tensor of shape (batch, seq_len, dim)

        Example:
            >>> stack = TransformerStack(num_layers=12, dim=512, num_heads=8)
            >>> x = torch.randn(32, 128, 512)
            >>> output = stack(x)
            >>> print(output.shape)
            torch.Size([32, 128, 512])
        """
        # Educational note: Pass input through each block sequentially.
        # Each block owns one slot (layer_idx) in the KV cache.
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, kv_cache=kv_cache, layer_idx=i, start_pos=start_pos)

        return x

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"num_layers={self.num_layers}, dim={self.dim}"


def test_transformer():
    """
    Test transformer block implementation.

    Educational note: Tests verify:
    1. Shape transformations are correct
    2. Residual connections work
    3. Stack of blocks works correctly
    4. Gradients flow through deep stacks
    """
    print("Testing Transformer Block implementation...")

    # Test 1: Single block
    dim = 512
    num_heads = 8
    batch_size = 4
    seq_len = 128

    block = TransformerBlock(dim=dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, dim)

    output = block(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✓ Test 1 passed: Single block shape preserved")

    # Test 2: Residual connections
    # Educational note: Residual connections should allow gradient flow
    loss = output.sum()
    loss.backward()

    # Check that all layers have gradients
    assert block.attention.qkv_proj.weight.grad is not None, "No gradients on attention"
    assert block.ffn.gate_value_proj.weight.grad is not None, "No gradients on FFN"
    assert not torch.isnan(block.attention.qkv_proj.weight.grad).any(), "Gradients contain NaN"
    print("✓ Test 2 passed: Gradients flow correctly")

    # Test 3: Stack of blocks
    num_layers = 6
    stack = TransformerStack(num_layers=num_layers, dim=dim, num_heads=num_heads)

    output_stack = stack(x)

    assert output_stack.shape == x.shape, f"Stack shape mismatch: {output_stack.shape} != {x.shape}"
    print("✓ Test 3 passed: Stack of blocks works")

    # Test 4: Deep stack gradient flow
    loss = output_stack.sum()
    loss.backward()

    # Check that all blocks in stack have gradients
    for i, blk in enumerate(stack.blocks):
        assert blk.attention.qkv_proj.weight.grad is not None, f"Block {i} has no attention gradients"
        assert blk.ffn.gate_value_proj.weight.grad is not None, f"Block {i} has no FFN gradients"
    print("✓ Test 4 passed: Gradients flow through deep stack")

    # Test 5: Custom mask (key-padding mask, shape (batch, seq_len))
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, seq_len // 2 :] = True

    x_masked = torch.randn(batch_size, seq_len, dim)

    output_masked = block(x_masked, mask=padding_mask)

    assert output_masked.shape == x_masked.shape, "Custom mask changed shape"
    print("✓ Test 5 passed: Custom masking works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_transformer()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Transformer Block Demonstration")
    print("=" * 70)

    # Create transformer block
    dim = 512
    num_heads = 8
    seq_len = 128
    batch_size = 4

    block = TransformerBlock(dim=dim, num_heads=num_heads)

    print(f"\nSingle Transformer Block:")
    print(block)

    # Create sample input
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"Input norm (mean): {torch.norm(x, dim=-1).mean():.4f}")

    # Apply single block
    output = block(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output norm (mean): {torch.norm(output, dim=-1).mean():.4f}")

    # Demonstrate stack of blocks
    print("\n" + "=" * 70)
    print("Stack of Transformer Blocks")
    print("=" * 70)

    num_layers = 6
    stack = TransformerStack(num_layers=num_layers, dim=dim, num_heads=num_heads)

    print(f"\nTransformer Stack ({num_layers} layers):")
    print(stack)

    # Apply stack
    output_stack = stack(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output_stack.shape}")
    print(f"Output norm (mean): {torch.norm(output_stack, dim=-1).mean():.4f}")

    # Parameter count
    params_block = sum(p.numel() for p in block.parameters())
    params_stack = sum(p.numel() for p in stack.parameters())

    print(f"\nParameter counts:")
    print(f"  Single block: {params_block:,}")
    print(f"  Stack ({num_layers} blocks): {params_stack:,}")
    print(f"  Ratio: {params_stack / params_block:.1f}x")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Transformer block = Attention + FFN with residual connections")
    print("- Pre-normalization: Normalize before layer (modern approach)")
    print("- Residual connections: Add input to output (helps gradient flow)")
    print("- Stacking blocks creates deep models with hierarchical learning")
    print("- Lower layers learn simple patterns, higher layers learn complex")
    print("- LLaMA uses 32-80 layers, GPT-3 uses 96 layers")
    print("=" * 70)
