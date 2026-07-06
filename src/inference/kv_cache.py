"""
KV Cache for Efficient Generation

This module implements Key-Value caching for efficient autoregressive generation.

Educational Notes:
- KV Cache: Stores keys and values from previous positions so they do not
  need to be recomputed on every token step
- Reduces generation complexity from O(N²) to O(N)
- Essential for practical inference — the speedup is ~50× for 100-token sequences

Without KV Cache:
    Generation of N tokens:
    - Forward pass 1: Process 1 token → Output token 1
    - Forward pass 2: Process 2 tokens (1 + 1 new) → Output token 2
    - Forward pass 3: Process 3 tokens (2 + 1 new) → Output token 3
    ...
    Complexity: O(N²) — quadratic in sequence length

With KV Cache:
    Generation of N tokens:
    - Forward pass 1: Process 1 token, cache K,V → Output token 1
    - Forward pass 2: Process 1 token (new only), reuse cached K,V → Output token 2
    - Forward pass 3: Process 1 token (new only), reuse cached K,V → Output token 3
    ...
    Complexity: O(N) — linear in sequence length

How KV Cache Works:
    Each attention layer has:
    - Key cache: Stored keys from all previous positions
    - Value cache: Stored values from all previous positions

    For new token:
    - Compute new K,V for current position only
    - Concatenate with cached K,V
    - Compute attention using full K,V (cached + new)

References:
    Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
        (Section 3.2: the decoder self-attention is the origin of the K,V reuse pattern)
"""

import torch
from typing import Optional, Dict


class KVCache:
    """
    Key-Value Cache for efficient autoregressive generation.

    Educational notes:
    - Stores keys and values from previous positions
    - Avoids recomputing attention for past tokens
    - Dramatically speeds up generation
    - Used during inference only (not training)

    Cache Structure:
        - One cache per layer
        - Each cache stores (K, V) for all positions
        - Shape: (batch, num_heads, seq_len, head_dim)

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_sequence_length: Maximum cache size
        dtype: Data type for cache

    Example:
        >>> cache = KVCache(batch_size=1, num_heads=8, head_dim=64)
        >>> k, v = cache.get(0)  # Get cache for layer 0
        >>> cache.update(0, k_new, v_new)  # Add new keys/values
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_sequence_length: int = 2048,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device = device

        # Educational note: Dictionary to store cache for each layer
        # Key: layer index
        # Value: tuple of (key_cache, value_cache)
        self.caches: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached keys and values for a layer.

        Educational notes:
        - Returns (key_cache, value_cache) for layer
        - If cache doesn't exist, creates empty cache
        - Shape: (batch, num_heads, seq_len, head_dim)

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (key_cache, value_cache)
            Shape: (batch, num_heads, seq_len, head_dim)

        Example:
            >>> k_cache, v_cache = cache.get(0)
            >>> print(f"Cache shape: {k_cache.shape}")
        """
        if layer_idx not in self.caches:
            # Educational note: Create empty cache
            # Shape: (batch, num_heads, 0, head_dim) - empty for now
            k_cache = torch.zeros(
                (self.batch_size, self.num_heads, 0, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v_cache = torch.zeros(
                (self.batch_size, self.num_heads, 0, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.caches[layer_idx] = (k_cache, v_cache)

        return self.caches[layer_idx]

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new keys/values for a layer and return the full cached tensors.

        Educational notes:
        - Concatenates the new K,V onto whatever is already cached for the layer
        - Returns the complete (cached + new) K,V so attention can be computed
        - Called once per layer, per generation step

        Args:
            layer_idx: Layer index
            new_k: New keys, shape (batch, num_heads, new_len, head_dim)
            new_v: New values, shape (batch, num_heads, new_len, head_dim)

        Returns:
            Tuple (k_full, v_full) of shape (batch, num_heads, total_len, head_dim)

        Example:
            >>> k_new = torch.randn(1, 8, 1, 64)
            >>> v_new = torch.randn(1, 8, 1, 64)
            >>> k_full, v_full = cache.update(0, k_new, v_new)
        """
        if layer_idx in self.caches:
            # Educational note: Concatenate new K,V with cached K,V along seq dim
            # Old cache: (batch, num_heads, old_len, head_dim)
            # New K,V:   (batch, num_heads, new_len, head_dim)
            k_cache, v_cache = self.caches[layer_idx]
            k_cache = torch.cat([k_cache, new_k], dim=2)
            v_cache = torch.cat([v_cache, new_v], dim=2)
        else:
            k_cache, v_cache = new_k, new_v

        # Educational note: Trim to max length (keep most recent) if we overflow.
        if k_cache.shape[2] > self.max_sequence_length:
            k_cache = k_cache[:, :, -self.max_sequence_length:, :]
            v_cache = v_cache[:, :, -self.max_sequence_length:, :]

        self.caches[layer_idx] = (k_cache, v_cache)

        return k_cache, v_cache

    def reset(self):
        """
        Reset all caches.

        Educational note:
        - Clears all cached keys and values
        - Call when starting new generation
        """
        self.caches.clear()

    @property
    def current_length(self) -> int:
        """
        Current cached sequence length (measured on the first layer).

        Educational note: All layers advance together, so the length of any
        cached layer is representative of the whole cache.
        """
        if 0 in self.caches:
            return self.caches[0][0].shape[2]
        # Fall back to any populated layer (e.g. when only later layers are used).
        for k_cache, _ in self.caches.values():
            return k_cache.shape[2]
        return 0

    @property
    def seq_len(self) -> int:
        """Return current sequence length."""
        return self.current_length


def test_kv_cache():
    """
    Test KV cache implementation.

    Educational note: Tests verify:
    1. Cache creation works
    2. Cache updates correctly
    3. Cache reset works
    4. Multiple layers work independently
    """
    print("Testing KV Cache implementation...")

    # Test 1: Create cache
    cache = KVCache(
        batch_size=1,
        num_heads=8,
        head_dim=64,
        max_sequence_length=100,
    )

    assert cache.current_length == 0, "Initial length should be 0"
    print("✓ Test 1 passed: Cache created")

    # Test 2: Update cache
    k_new = torch.randn(1, 8, 1, 64)
    v_new = torch.randn(1, 8, 1, 64)

    cache.update(0, k_new, v_new)

    assert cache.current_length == 1, "Length should be 1"
    print("✓ Test 2 passed: Cache updated")

    # Test 3: Multiple updates
    # Educational note: Start from a clean cache so the count is unambiguous.
    cache.reset()
    for i in range(10):
        k_new = torch.randn(1, 8, 1, 64)
        v_new = torch.randn(1, 8, 1, 64)
        cache.update(0, k_new, v_new)

    assert cache.current_length == 10, "Length should be 10"

    k_cache, v_cache = cache.get(0)
    assert k_cache.shape[2] == 10, "Cache sequence length incorrect"
    print("✓ Test 3 passed: Multiple updates work")

    # Test 4: Reset
    cache.reset()
    assert cache.current_length == 0, "Length should be 0 after reset"
    print("✓ Test 4 passed: Reset works")

    # Test 5: Multiple layers
    cache = KVCache(
        batch_size=2,
        num_heads=8,
        head_dim=64,
        max_sequence_length=100,
    )

    for layer in range(6):
        k_new = torch.randn(2, 8, 1, 64)
        v_new = torch.randn(2, 8, 1, 64)
        cache.update(layer, k_new, v_new)

    for layer in range(6):
        k_cache, v_cache = cache.get(layer)
        assert k_cache.shape[2] == 1, f"Layer {layer} cache incorrect"

    print("✓ Test 5 passed: Multiple layers work")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_kv_cache()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("KV Cache Demonstration")
    print("=" * 70)

    # Create cache
    cache = KVCache(
        batch_size=1,
        num_heads=8,
        head_dim=64,
        max_sequence_length=100,
    )

    print(f"\nKV Cache created:")
    print(f"  Batch size: {cache.batch_size}")
    print(f"  Num heads: {cache.num_heads}")
    print(f"  Head dim: {cache.head_dim}")
    print(f"  Max sequence length: {cache.max_sequence_length}")
    print(f"  Current length: {cache.current_length}")

    # Simulate generation
    print(f"\nSimulating token generation:")

    for i in range(5):
        # Compute new K,V (in real model, this comes from forward pass)
        k_new = torch.randn(1, 8, 1, 64)
        v_new = torch.randn(1, 8, 1, 64)

        # Update cache
        cache.update(0, k_new, v_new)

        k_cache, v_cache = cache.get(0)

        print(f"  Token {i+1}: Updated cache, sequence length = {cache.current_length}")

    # Show cache shape
    k_cache, v_cache = cache.get(0)
    print(f"\nCache shape: {k_cache.shape}")
    print(f"  (batch, num_heads, seq_len, head_dim)")

    # Reset
    print(f"\nResetting cache...")
    cache.reset()
    print(f"Current length after reset: {cache.current_length}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- KV cache stores keys/values from previous positions")
    print("- Avoids recomputing attention for past tokens")
    print("- Dramatically speeds up generation (O(N²) → O(N))")
    print("- Each layer has its own cache")
    print("- Essential for practical inference")
    print("=" * 70)
