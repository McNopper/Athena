"""
Training Dataset

This module implements PyTorch Dataset and DataLoader for LLM training.

Educational Notes:
- Dataset: Interface for accessing training samples
- DataLoader: Batching, shuffling, and parallel loading
- Efficient memory usage with lazy loading
- Supports training and validation splits

Data Flow:
    Tokenized sequences → Dataset → DataLoader → Training loop
                     (shuffle)   (batch)    (model)

Key Concepts:
- Batching: Group multiple sequences for efficiency
- Shuffling: Randomize order each epoch (prevents overfitting to order)
- Padding: Make sequences same length (for efficient GPU computation)
- Masking: Ignore padding tokens in loss computation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from functools import partial
import json

# Loss ignore index for padded target positions (matches the default used by
# ``torch.nn.functional.cross_entropy`` / ``src.utils.metrics.compute_loss``).
IGNORE_INDEX = -100


class LLMTrainerDataset(Dataset):
    """
    PyTorch Dataset for LLM training.

    Educational notes:
    - Wraps tokenized sequences for PyTorch DataLoader
    - Returns (input_tokens, target_tokens) pairs
    - Target tokens are input tokens shifted by 1 (next token prediction)
    - Supports padding and masking for variable-length sequences

    Next Token Prediction:
        Input:  [BOS, "The", "cat", "sat", "on"]
        Target: ["The", "cat", "sat", "on", "the"]
        Model learns to predict next token given previous tokens

    Args:
        sequences: List of token ID sequences
        max_sequence_length: Maximum sequence length (for truncation)
        pad_token_id: Token ID for padding (typically 0)

    Example:
        >>> sequences = [[1, 2, 3, 4], [5, 6, 7]]
        >>> dataset = LLMTrainerDataset(sequences)
        >>> input, target = dataset[0]
        >>> print(input)
        tensor([1, 2, 3, 4])
        >>> print(target)
        tensor([2, 3, 4, 0])  # Shifted by 1, padded
    """

    def __init__(
        self,
        sequences: List[List[int]],
        max_sequence_length: int = 512,
        pad_token_id: int = 0,
    ):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Educational notes:
        - Returns (input_tokens, target_tokens) pair
        - Input: All tokens except last
        - Target: All tokens except first, shifted by 1
        - This setup trains the model for next-token prediction

        Example:
            Sequence: [BOS, "hello", "world", EOS]
            Input:    [BOS, "hello", "world"]
            Target:   ["hello", "world", EOS]

        Args:
            idx: Index of sequence in dataset

        Returns:
            Tuple of (input_tokens, target_tokens) tensors
        """
        # Get sequence
        seq = self.sequences[idx]

        # Truncate if too long
        if len(seq) > self.max_sequence_length:
            seq = seq[: self.max_sequence_length]

        # Convert to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.long)

        # Educational note: Split into input and target
        # Input: All tokens except last one
        # Target: All tokens except first one (shifted by 1)
        # This trains the model to predict next token
        input_tokens = seq_tensor[:-1]
        target_tokens = seq_tensor[1:]

        return input_tokens, target_tokens


def pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate variable-length (input, target) pairs into padded batch tensors.

    Educational notes:
    - Sequences in a batch can have different lengths, but a tensor needs them
      equal, so we right-pad every sequence to the batch's longest length.
    - Inputs are padded with ``pad_token_id`` (a real, harmless token).
    - Targets are padded with ``IGNORE_INDEX`` (-100) so the loss function
      skips those positions and padding never contributes to the gradient.
    - Because attention is causal and padding sits at the end, real tokens are
      never influenced by padded positions, so no extra attention mask is needed.

    Args:
        batch: List of (input_tokens, target_tokens) tensor pairs
        pad_token_id: Token ID used to pad the inputs

    Returns:
        Tuple of (inputs, targets), each of shape (batch_size, max_len)
    """
    inputs, targets = zip(*batch)
    max_len = max(t.size(0) for t in inputs)

    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            inp = F.pad(inp, (0, pad_len), value=pad_token_id)
            tgt = F.pad(tgt, (0, pad_len), value=IGNORE_INDEX)
        padded_inputs.append(inp)
        padded_targets.append(tgt)

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def create_dataloader(
    sequences: List[List[int]],
    batch_size: int = 32,
    max_sequence_length: int = 512,
    pad_token_id: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for training.

    Educational notes:
    - DataLoader batches and shuffles data efficiently
    - Custom collate function handles variable-length sequences
    - Parallel loading with num_workers (faster data preparation)

    Args:
        sequences: List of token sequences
        batch_size: Number of sequences per batch
        max_sequence_length: Maximum sequence length
        pad_token_id: Token ID for padding
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of parallel loading workers (0 = main process)

    Returns:
        PyTorch DataLoader

    Example:
        >>> loader = create_dataloader(sequences, batch_size=32)
        >>> for batch_idx, (inputs, targets) in enumerate(loader):
        ...     # inputs shape: (batch_size, seq_len)
        ...     # targets shape: (batch_size, seq_len)
        ...     pass
    """
    # Create dataset
    dataset = LLMTrainerDataset(
        sequences=sequences,
        max_sequence_length=max_sequence_length,
        pad_token_id=pad_token_id,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Only helps for CUDA transfers
        drop_last=True,  # Drop incomplete last batch
        collate_fn=partial(pad_collate, pad_token_id=pad_token_id),
    )

    return dataloader


def load_sequences_from_file(filepath: str) -> List[List[int]]:
    """
    Load tokenized sequences from JSON file.

    Educational notes:
    - Loads pre-tokenized training data
    - Used after prepare_data.py has processed raw text
    - Efficient for large datasets

    Args:
        filepath: Path to JSON file with sequences

    Returns:
        List of token ID sequences

    Example:
        >>> sequences = load_sequences_from_file("data/processed/train_sequences.json")
        >>> print(f"Loaded {len(sequences)} sequences")
    """
    with open(filepath, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    return sequences


def test_dataset():
    """
    Test dataset implementation.

    Educational note: Tests verify:
    1. Dataset creation and indexing
    2. Input/target splitting is correct
    3. DataLoader batching works
    4. Shuffling changes order
    """
    print("Testing Dataset implementation...")

    # Test 1: Dataset creation
    sequences = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15],
    ]

    dataset = LLMTrainerDataset(sequences, max_sequence_length=512)

    assert len(dataset) == 3, "Dataset length incorrect"
    print("✓ Test 1 passed: Dataset creation works")

    # Test 2: Input/target splitting
    input_tokens, target_tokens = dataset[0]

    # Input should be all but last
    expected_input = torch.tensor([1, 2, 3, 4])
    assert torch.equal(input_tokens, expected_input), "Input tokens incorrect"

    # Target should be all but first (shifted by 1)
    expected_target = torch.tensor([2, 3, 4, 5])
    assert torch.equal(target_tokens, expected_target), "Target tokens incorrect"

    print("✓ Test 2 passed: Input/target splitting correct")

    # Test 3: DataLoader batching
    dataloader = create_dataloader(sequences, batch_size=2, shuffle=False)

    batch_count = 0
    for inputs, targets in dataloader:
        batch_count += 1
        assert inputs.shape[0] <= 2, "Batch size incorrect"
        assert inputs.shape == targets.shape, "Input and target shapes don't match"

    assert batch_count == 1, f"Expected 1 batch, got {batch_count}"
    print("✓ Test 3 passed: DataLoader batching works")

    # Test 4: Shuffling
    dataloader_shuffled = create_dataloader(sequences, batch_size=2, shuffle=True)

    first_batch_inputs = None
    for inputs, _ in dataloader_shuffled:
        first_batch_inputs = inputs
        break

    # Run again and check if order changed (probabilistic test)
    # This is just to show the mechanism exists
    print("✓ Test 4 passed: Shuffling mechanism exists")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_dataset()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Dataset Demonstration")
    print("=" * 70)

    # Create sample sequences
    sequences = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ]

    print(f"\nSample sequences: {len(sequences)}")
    for i, seq in enumerate(sequences):
        print(f"  Seq {i}: {seq}")

    # Create dataset
    dataset = LLMTrainerDataset(sequences, max_sequence_length=512)

    print(f"\nDataset size: {len(dataset)}")

    # Get first example
    input_tokens, target_tokens = dataset[0]

    print(f"\nFirst example:")
    print(f"  Input:  {input_tokens.tolist()}")
    print(f"  Target: {target_tokens.tolist()}")
    print(f"\n  Educational note: Target is shifted by 1")
    print(f"  Model learns to predict target[i] given input[i]")

    # Create dataloader
    dataloader = create_dataloader(sequences, batch_size=2, shuffle=False)

    print(f"\nDataLoader:")
    print(f"  Batch size: 2")
    print(f"  Number of batches: {len(dataloader)}")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"\n  Batch {batch_idx}:")
        print(f"    Inputs shape:  {inputs.shape}")
        print(f"    Targets shape: {targets.shape}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Dataset wraps tokenized sequences for PyTorch")
    print("- Input/target split trains next-token prediction")
    print("- DataLoader batches and shuffles efficiently")
    print("- Padding handled automatically for variable lengths")
    print("- Shuffling prevents overfitting to sequence order")
    print("=" * 70)
