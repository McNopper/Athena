"""
Token Embeddings

This module implements token embeddings for language models.

Educational Notes:
- Token embeddings convert discrete token IDs into continuous vectors
- Each token in vocabulary gets a learnable vector representation
- Embeddings capture semantic relationships between tokens
- During training, these vectors are learned to capture meaning

Key concept: Tied Embeddings
- Input embeddings: Token → Vector (used at the model input)
- Output embeddings: Vector → Logits (used at the model output)
- Tied embeddings: Use the same weight matrix for both (saves parameters,
  improves generalisation by ensuring token representations are consistent)
- Untied embeddings: Separate matrices (more parameters, potentially better
  for tasks where input and output token semantics differ)
- Modern LLMs typically tie embeddings for efficiency

Why embeddings matter:
- They form the foundation of how the model understands language
- Similar words end up with similar embeddings (e.g., "cat" ≈ "dog")
- The model learns these representations from data during training

References:
    Press & Wolf, "Using the Output Embedding to Improve Language Models" (2016) — https://arxiv.org/abs/1608.05859
"""

import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for language models.

    Educational notes:
    - Converts token IDs to dense vectors
    - Learnable parameters: one vector per vocabulary token
    - Shape: (vocab_size, dim) where dim is embedding dimension
    - During training, these vectors are learned to capture semantic meaning

    How it works:
    1. Each token ID (integer) is used as an index into the embedding matrix
    2. The corresponding row is retrieved as the embedding vector
    3. This vector represents the semantic meaning of the token

    Example:
        Token "cat" (ID: 123) -> embedding vector [0.2, -0.5, 0.8, ...]
        Token "dog" (ID: 456) -> embedding vector [0.3, -0.4, 0.7, ...]
        These vectors will be similar because "cat" and "dog" are semantically related

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        dim: Embedding dimension (size of each embedding vector)
        device: Device to place embeddings on

    Example:
        >>> embeddings = TokenEmbedding(vocab_size=32000, dim=512)
        >>> token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (batch, seq)
        >>> embedded = embeddings(token_ids)
        >>> print(embedded.shape)
        torch.Size([2, 3, 512])
    """

    def __init__(self, vocab_size: int, dim: int, device: Optional[torch.device] = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        # Educational note: nn.Embedding is a lookup table
        # Shape: (vocab_size, dim)
        # Each row is the embedding for one token
        self.token_embedding = nn.Embedding(vocab_size, dim, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up token embeddings.

        Educational notes:
        - Input: Token IDs (integers)
        - Output: Embedding vectors (continuous)
        - Operation: Simple lookup/index operation
        - No computation, just retrieving pre-learned vectors

        Args:
            token_ids: Tensor of token IDs
                      Shape: (batch_size, seq_len) or (seq_len,)

        Returns:
            Embedding vectors
            Shape: (batch_size, seq_len, dim) or (seq_len, dim)

        Example:
            >>> embeddings = TokenEmbedding(vocab_size=1000, dim=128)
            >>> token_ids = torch.tensor([1, 2, 3])
            >>> embedded = embeddings(token_ids)
            >>> print(embedded.shape)
            torch.Size([3, 128])
        """
        # Educational note: nn.Embedding does a lookup operation
        # token_ids contains indices into the embedding matrix
        embeddings = self.token_embedding(token_ids)

        return embeddings

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"vocab_size={self.vocab_size}, dim={self.dim}"


class OutputProjection(nn.Module):
    """
    Output projection layer for language models.

    Educational notes:
    - Converts model's hidden states to logits over vocabulary
    - Projects from (dim,) to (vocab_size,)
    - Logits are scores for each possible next token
    - During training, used with cross-entropy loss
    - During inference, used to predict next token (softmax + sampling)

    Tied vs Untied Embeddings:
    - Tied: Use same weights as input embeddings (saves parameters)
    - Untied: Separate learned weights (more parameters)

    Tied embeddings are common in modern LLMs because:
    1. Reduces parameter count significantly (vocab_size * dim)
    2. Often improves generalization (forces symmetric input/output)
    3. Works well in practice

    Args:
        vocab_size: Size of the vocabulary
        dim: Model dimension (hidden size)
        tied_embeddings: If True, tie weights with input embeddings
                        If False, use separate projection layer
        input_embeddings: If tied_embeddings is True, provide input embeddings
                         to share weights with

    Example:
        >>> input_emb = TokenEmbedding(vocab_size=32000, dim=512)
        >>> output_proj = OutputProjection(vocab_size=32000, dim=512,
        ...                                tied_embeddings=True,
        ...                                input_embeddings=input_emb.token_embedding)
        >>> hidden = torch.randn(32, 128, 512)  # (batch, seq, dim)
        >>> logits = output_proj(hidden)
        >>> print(logits.shape)
        torch.Size([32, 128, 32000])
    """

    def __init__(self, vocab_size: int, dim: int, tied_embeddings: bool = False, input_embeddings: Optional[nn.Embedding] = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.tied_embeddings = tied_embeddings

        if tied_embeddings:
            # Educational note: Use same weights as input embeddings
            # This reduces parameters and often improves generalization
            assert input_embeddings is not None, "input_embeddings must be provided if tied_embeddings=True"
            assert input_embeddings.weight.shape == (vocab_size, dim), "Embedding dimensions must match"

            # Reference to input embedding weights (not a copy)
            self.weight = input_embeddings.weight
        else:
            # Educational note: Separate learned projection matrix
            # Shape: (dim, vocab_size)
            # Projects from hidden dimension to vocabulary size
            self.projection = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Educational notes:
        - Input: Model's hidden states
        - Output: Logits (scores) for each vocabulary token
        - During training: Used with cross-entropy loss
        - During inference: Used for token prediction (softmax + sampling)

        Args:
            hidden_states: Model's output hidden states
                           Shape: (batch_size, seq_len, dim)

        Returns:
            Logits over vocabulary
            Shape: (batch_size, seq_len, vocab_size)

        Example:
            >>> output_proj = OutputProjection(vocab_size=32000, dim=512)
            >>> hidden = torch.randn(32, 128, 512)
            >>> logits = output_proj(hidden)
            >>> print(logits.shape)
            torch.Size([32, 128, 32000])
        """
        if self.tied_embeddings:
            # Educational note: When embeddings are tied, we use the transpose
            # of input embeddings as the projection matrix
            # Input embeddings: (vocab_size, dim)
            # Output projection: (dim, vocab_size) = transpose(input_embeddings)
            #
            # Matrix operation: hidden_states @ embedding_weights.T
            # This projects from hidden space back to vocabulary space

            # For efficiency, we use embedding's transpose operation
            # hidden_states: (batch, seq, dim)
            # weight: (vocab_size, dim)
            # We want: (batch, seq, vocab_size) = (batch, seq, dim) @ (dim, vocab_size)

            logits = torch.nn.functional.linear(hidden_states, self.weight)
        else:
            # Use separate projection layer
            logits = self.projection(hidden_states)

        return logits

    def extra_repr(self) -> str:
        """Extra representation for printing the module."""
        return f"vocab_size={self.vocab_size}, dim={self.dim}, tied={self.tied_embeddings}"


def test_embeddings():
    """
    Test embedding and projection layers.

    Educational note: Tests verify:
    1. Shape transformations are correct
    2. Tied embeddings actually share weights
    3. Gradients flow correctly
    """
    print("Testing embedding implementation...")

    # Test 1: Token embedding
    vocab_size = 1000
    dim = 128
    batch_size = 4
    seq_len = 32

    token_emb = TokenEmbedding(vocab_size, dim)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    embedded = token_emb(token_ids)
    assert embedded.shape == (batch_size, seq_len, dim), f"Wrong shape: {embedded.shape}"
    print("✓ Test 1 passed: Token embedding works")

    # Test 2: Output projection (untied)
    output_proj = OutputProjection(vocab_size, dim, tied_embeddings=False)
    hidden = torch.randn(batch_size, seq_len, dim)

    logits = output_proj(hidden)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong shape: {logits.shape}"
    print("✓ Test 2 passed: Output projection works (untied)")

    # Test 3: Tied embeddings
    input_emb = TokenEmbedding(vocab_size, dim)
    output_proj_tied = OutputProjection(vocab_size, dim, tied_embeddings=True, input_embeddings=input_emb.token_embedding)

    # Verify weights are actually shared
    assert torch.allclose(input_emb.token_embedding.weight, output_proj_tied.weight), "Weights not tied"
    print("✓ Test 3 passed: Tied embeddings share weights")

    # Test 4: Gradient flow through tied embeddings
    logits_tied = output_proj_tied(embedded)
    loss = logits_tied.sum()
    loss.backward()

    # Check that shared weights have gradients
    assert input_emb.token_embedding.weight.grad is not None, "No gradients on tied embeddings"
    assert not torch.isnan(input_emb.token_embedding.weight.grad).any(), "Gradients contain NaN"
    print("✓ Test 4 passed: Gradients flow through tied embeddings")

    # Test 5: Parameter count comparison
    # Educational note: A tied projection *shares* its weight with the input
    # embedding, so the saving shows up when you count unique parameters across
    # the embedding + projection pair (PyTorch de-duplicates shared tensors).
    def count_unique_params(*modules):
        seen = set()
        total = 0
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
        return total

    untied_total = count_unique_params(token_emb, output_proj)
    tied_total = count_unique_params(input_emb, output_proj_tied)

    # Tied should have fewer unique parameters (projection reuses the embedding)
    assert tied_total < untied_total, "Tied embeddings should save parameters"
    print(f"✓ Test 5 passed: Tied embeddings save parameters ({untied_total:,} -> {tied_total:,})")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_embeddings()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Token Embedding Demonstration")
    print("=" * 70)

    # Create embedding layers
    vocab_size = 1000
    dim = 256

    token_emb = TokenEmbedding(vocab_size, dim)
    output_proj_untied = OutputProjection(vocab_size, dim, tied_embeddings=False)
    output_proj_tied = OutputProjection(vocab_size, dim, tied_embeddings=True, input_embeddings=token_emb.token_embedding)

    print(f"\nToken Embedding Layer:")
    print(token_emb)

    print(f"\nOutput Projection (Untied):")
    print(output_proj_untied)

    print(f"\nOutput Projection (Tied):")
    print(output_proj_tied)

    # Create sample input
    batch_size = 2
    seq_len = 16
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput token IDs shape: {token_ids.shape}")
    print(f"Sample token IDs: {token_ids[0, :10].tolist()}")

    # Embed tokens
    embedded = token_emb(token_ids)
    print(f"\nEmbedded shape: {embedded.shape}")
    print(f"Embedding norm (mean): {torch.norm(embedded, dim=-1).mean():.4f}")

    # Project back to vocabulary (untied)
    logits_untied = output_proj_untied(embedded)
    print(f"\nOutput logits (untied) shape: {logits_untied.shape}")

    # Project back to vocabulary (tied)
    logits_tied = output_proj_tied(embedded)
    print(f"Output logits (tied) shape: {logits_tied.shape}")

    # Parameter count comparison
    params_emb = sum(p.numel() for p in token_emb.parameters())
    params_untied = sum(p.numel() for p in output_proj_untied.parameters())
    params_tied = sum(p.numel() for p in output_proj_tied.parameters())

    print(f"\nParameter counts:")
    print(f"  Token embeddings: {params_emb:,}")
    print(f"  Output projection (untied): {params_untied:,}")
    print(f"  Output projection (tied): {params_tied:,}")
    print(f"  Total (untied): {params_emb + params_untied:,}")
    print(f"  Total (tied): {params_emb + params_tied:,}")
    print(f"  Parameters saved: {params_untied - params_tied:,}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Token embeddings convert discrete IDs to continuous vectors")
    print("- Output projection converts hidden states back to vocabulary logits")
    print("- Tied embeddings share input/output matrices (saves parameters)")
    print("- Learned representations capture semantic meaning of tokens")
    print("- Similar tokens end up with similar embeddings")
    print("=" * 70)
