#!/usr/bin/env python3
"""
Data Preparation Script

This script prepares training data by:
1. Loading raw text files
2. Training or loading a tokenizer
3. Tokenizing the text
4. Saving processed data for training

Educational Notes:
- Converts raw text into token IDs
- Splits into train/validation sets
- Saves in efficient format for training
- Handles large datasets with memory-efficient processing

Usage:
    python scripts/prepare_data.py --data-dir data/raw/ --output-dir data/processed/
"""

import argparse
import os
import sys
from typing import List
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.vocabulary import Vocabulary


def load_text_files(data_dir: str, pattern: str = "*.txt") -> List[str]:
    """
    Load all text files from directory.

    Educational notes:
    - Recursively finds all .txt files
    - Reads each file as a separate document
    - Returns list of text strings

    Args:
        data_dir: Directory containing text files
        pattern: File pattern to match

    Returns:
        List of text strings
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Find all text files
    text_files = list(data_path.rglob(pattern))

    if not text_files:
        raise ValueError(f"No text files found in {data_dir} with pattern {pattern}")

    print(f"Found {len(text_files)} text files")

    # Load all files
    documents = []

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():  # Skip empty files
                    documents.append(text)
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")

    print(f"Loaded {len(documents)} documents")

    return documents


def train_tokenizer(documents: List[str], vocab_size: int, output_dir: str) -> BPETokenizer:
    """
    Train BPE tokenizer on documents.

    Educational notes:
    - Learns vocabulary from training data
    - Saves tokenizer for reuse
    - Vocabulary size affects model size and capability

    Args:
        documents: List of text documents
        vocab_size: Target vocabulary size
        output_dir: Directory to save tokenizer

    Returns:
        Trained BPETokenizer
    """
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(documents, verbose=True)

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    print(f"Saved tokenizer to {tokenizer_path}")

    return tokenizer


def tokenize_documents(
    documents: List[str],
    tokenizer: BPETokenizer,
    max_sequence_length: int = 512,
    stride: int = 256,
) -> List[List[int]]:
    """
    Tokenize documents into sequences.

    Educational notes:
    - Tokenizes each document
    - Splits into fixed-length sequences
    - Uses stride to create overlapping sequences (more data)
    - Filters sequences that are too short

    Args:
        documents: List of text documents
        tokenizer: Trained tokenizer
        max_sequence_length: Maximum sequence length
        stride: Step size for splitting (creates overlap)

    Returns:
        List of token ID sequences
    """
    print(f"\nTokenizing {len(documents)} documents...")

    all_sequences = []

    # Reserve two positions for the BOS/EOS tokens so the final sequence length
    # never exceeds max_sequence_length (otherwise the dataset would later
    # truncate and silently drop the EOS token).
    minimum_length = 10
    content_length = max(1, max_sequence_length - 2)
    # Never require a window longer than a full content chunk (keeps very small
    # max_sequence_length configurations from dropping every window).
    window_min = min(minimum_length, content_length)
    bos_id = tokenizer.vocab.bos_token_id
    eos_id = tokenizer.vocab.eos_token_id

    for i, doc in enumerate(documents):
        if i % 100 == 0:
            print(f"  Processing document {i+1}/{len(documents)}...")

        # Tokenize document
        token_ids = tokenizer.encode(doc, add_special_tokens=False)

        # Skip if too short
        if len(token_ids) < minimum_length:
            continue

        # Split into fixed-length, optionally overlapping windows.
        # Educational note: stride < content_length creates overlap, which
        # increases training data and helps the model learn continuity.
        # max(1, ...) guarantees at least one window for documents shorter
        # than content_length (which the plain range would skip entirely).
        emitted_end = 0
        for start_idx in range(0, max(1, len(token_ids) - content_length + 1), stride):
            sequence = token_ids[start_idx:start_idx + content_length]
            if len(sequence) < window_min:
                continue
            all_sequences.append([bos_id] + sequence + [eos_id])
            emitted_end = start_idx + len(sequence)

        # Emit a final partial window for any leftover tail the strided loop
        # did not already cover (also covers short single-window documents).
        if emitted_end < len(token_ids):
            remaining = token_ids[-content_length:]
            if len(remaining) >= window_min:
                all_sequences.append([bos_id] + remaining + [eos_id])

    print(f"Created {len(all_sequences)} sequences")

    return all_sequences


def save_sequences(sequences: List[List[int]], output_path: str):
    """
    Save sequences to JSON file.

    Educational notes:
    - JSON format for easy inspection
    - Could use more efficient format (binary, pickle) for large datasets
    - Simple format good for educational purposes

    Args:
        sequences: List of token ID sequences
        output_path: Path to save
    """
    print(f"\nSaving {len(sequences)} sequences to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f)

    # Print statistics
    seq_lengths = [len(seq) for seq in sequences]

    print(f"Sequence statistics:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Min length: {min(seq_lengths)}")
    print(f"  Max length: {max(seq_lengths)}")
    print(f"  Mean length: {sum(seq_lengths) / len(seq_lengths):.1f}")


def split_train_val(sequences: List[List[int]], val_split: float = 0.1):
    """
    Split sequences into train and validation sets.

    Educational notes:
    - Validation set for monitoring overfitting
    - Typical split: 90% train, 10% validation
    - Random shuffle before splitting

    Args:
        sequences: List of token sequences
        val_split: Fraction for validation

    Returns:
        Tuple of (train_sequences, val_sequences)
    """
    import random

    # Shuffle sequences
    shuffled = sequences.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * (1 - val_split))
    train_seqs = shuffled[:split_idx]
    val_seqs = shuffled[split_idx:]

    print(f"\nSplit into {len(train_seqs)} train and {len(val_seqs)} validation sequences")

    return train_seqs, val_seqs


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare training data for LLM")

    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw text files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocabulary size")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sequence splitting")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Load existing tokenizer")
    parser.add_argument("--test-run", action="store_true", help="Run on small subset for testing")

    args = parser.parse_args()

    print("=" * 70)
    print("Data Preparation Pipeline")
    print("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load documents
    print("\nStep 1: Loading documents...")
    documents = load_text_files(args.data_dir)

    if args.test_run:
        print("\nTest run: Using only 100 documents")
        documents = documents[:100]

    print(f"Loaded {len(documents)} documents")

    # Step 2: Train or load tokenizer
    print("\nStep 2: Preparing tokenizer...")

    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        print(f"Loading existing tokenizer from {args.tokenizer_path}")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = train_tokenizer(documents, args.vocab_size, args.output_dir)

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Step 3: Tokenize documents
    print("\nStep 3: Tokenizing documents...")
    sequences = tokenize_documents(documents, tokenizer, args.max_seq_length, args.stride)

    if not sequences:
        print("Error: No sequences created")
        return

    # Step 4: Split train/validation
    print("\nStep 4: Splitting train/validation...")
    train_sequences, val_sequences = split_train_val(sequences, args.val_split)

    # Step 5: Save processed data
    print("\nStep 5: Saving processed data...")

    train_path = os.path.join(args.output_dir, "train_sequences.json")
    val_path = os.path.join(args.output_dir, "val_sequences.json")

    save_sequences(train_sequences, train_path)
    save_sequences(val_sequences, val_path)

    # Save metadata
    metadata = {
        "vocab_size": len(tokenizer),
        "max_sequence_length": args.max_seq_length,
        "num_train_sequences": len(train_sequences),
        "num_val_sequences": len(val_sequences),
        "tokenizer_path": os.path.join(args.output_dir, "tokenizer.json"),
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")

    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Train sequences: {train_path}")
    print(f"  Val sequences: {val_path}")
    print(f"  Tokenizer: {metadata['tokenizer_path']}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
