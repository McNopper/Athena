"""
Vocabulary Management

This module handles vocabulary management for the language model tokenizer.

Educational Notes:
- Vocabulary is the set of unique tokens the model can understand
- Special tokens: PAD (padding), UNK (unknown), BOS (start of sequence), EOS (end)
- Vocabulary size affects model size and capability
- Larger vocabulary = more parameters but richer representations

Vocabulary Design:
- Typical size: 32K (GPT-2) to 100K+ (modern LLMs)
- Trade-off: Larger vocab means larger embedding matrix
- Special tokens serve specific purposes in training/inference

Special Tokens:
- PAD: Padding token (for batching sequences of different lengths)
- UNK: Unknown token (for characters/words not in vocabulary)
- BOS: Beginning of sequence (marks start of generated text)
- EOS: End of sequence (marks end of generated text)
- MASK: Mask token (for masked language modeling, optional)
"""

from typing import List, Dict, Optional, Set
from collections import Counter


class Vocabulary:
    """
    Vocabulary class for managing token-to-ID mappings.

    Educational notes:
    - Maps between tokens (strings) and token IDs (integers)
    - Special tokens serve specific purposes
    - Can be built from a corpus or loaded from a file
    - Used by tokenizer to encode/decode text

    Token ID Assignment:
    - Special tokens get lowest IDs (0-3 typically)
    - Regular tokens follow (4, 5, 6, ...)
    - Most frequent tokens get lower IDs (common practice)

    Args:
        special_tokens: List of special tokens
        vocab_size: Maximum vocabulary size

    Example:
        >>> vocab = Vocabulary()
        >>> vocab.build_from_corpus(["hello world", "hello there"])
        >>> print(vocab.token_to_id["hello"])
        4
    """

    # Special token definitions
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    MASK_TOKEN = "<MASK>"

    def __init__(self, special_tokens: Optional[List[str]] = None, vocab_size: Optional[int] = None):
        """
        Initialize vocabulary.

        Args:
            special_tokens: List of special tokens (default: PAD, UNK, BOS, EOS)
            vocab_size: Maximum vocabulary size
        """
        # Default special tokens
        if special_tokens is None:
            special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        # Token to ID mapping
        self.token_to_id: Dict[str, int] = {}

        # ID to token mapping
        self.id_to_token: Dict[int, str] = {}

        # Token frequencies
        self.token_counts: Counter = Counter()

        # Initialize special tokens
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """
        Initialize special tokens with lowest IDs.

        Educational note:
        - Special tokens get IDs 0, 1, 2, 3, etc.
        - This reserves them for specific purposes
        - Regular tokens start after special tokens
        """
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def num_special_tokens(self) -> int:
        """Number of special tokens in vocabulary."""
        return len(self.special_tokens)

    @property
    def pad_token_id(self) -> int:
        """ID of the padding token."""
        return self.token_to_id.get(self.PAD_TOKEN, 0)

    @property
    def unk_token_id(self) -> int:
        """ID of the unknown token."""
        return self.token_to_id.get(self.UNK_TOKEN, 1)

    @property
    def bos_token_id(self) -> int:
        """ID of the beginning-of-sequence token."""
        return self.token_to_id.get(self.BOS_TOKEN, 2)

    @property
    def eos_token_id(self) -> int:
        """ID of the end-of-sequence token."""
        return self.token_to_id.get(self.EOS_TOKEN, 3)

    def add_token(self, token: str) -> int:
        """
        Add a token to vocabulary if not already present.

        Educational notes:
        - If token already exists, return its current ID
        - If token is new, assign next available ID
        - If vocab_size is set and reached, don't add new tokens

        Args:
            token: Token string to add

        Returns:
            Token ID

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.add_token("hello")
            4
            >>> vocab.add_token("hello")  # Already exists
            4
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        # Check if we've reached vocab size limit
        if self.vocab_size is not None and len(self.token_to_id) >= self.vocab_size:
            return self.unk_token_id

        # Assign next available ID
        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token

        return new_id

    def build_from_corpus(self, corpus: List[str], min_frequency: int = 2):
        """
        Build vocabulary from a corpus of text.

        Educational notes:
        - Counts token frequencies across corpus
        - Adds tokens that appear at least min_frequency times
        - Most frequent tokens get lowest IDs (after special tokens)
        - Common practice for better model learning

        Args:
            corpus: List of text strings
            min_frequency: Minimum frequency for a token to be included

        Example:
            >>> vocab = Vocabulary()
            >>> corpus = ["hello world", "hello there", "world"]
            >>> vocab.build_from_corpus(corpus)
            >>> print(len(vocab))
            8  # 4 special + 4 regular tokens
        """
        # Educational note: Count token frequencies
        self.token_counts.clear()

        for text in corpus:
            # Simple whitespace tokenization (for illustration)
            tokens = text.split()
            self.token_counts.update(tokens)

        # Educational note: Sort by frequency (most frequent first)
        # Then add tokens in order of frequency
        for token, count in self.token_counts.most_common():
            if count >= min_frequency:
                self.add_token(token)

            # Stop if we've reached vocab size
            if self.vocab_size is not None and len(self.token_to_id) >= self.vocab_size:
                break

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to list of token IDs.

        Educational notes:
        - Splits text into tokens (whitespace for now)
        - Looks up each token's ID
        - Unknown tokens become UNK token ID
        - Optionally adds BOS/EOS tokens

        Args:
            text: Input text string
            add_special_tokens: If True, add BOS at start, EOS at end

        Returns:
            List of token IDs

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.build_from_corpus(["hello world"])
            >>> ids = vocab.encode("hello world", add_special_tokens=True)
            >>> print(ids)
            [2, 4, 5, 3]  # BOS, hello, world, EOS
        """
        # Simple whitespace tokenization
        tokens = text.split()

        # Look up token IDs
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode list of token IDs to text.

        Educational notes:
        - Looks up each token ID's string
        - Special tokens can be skipped or included
        - Reconstructs text by joining with spaces

        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, don't include special tokens in output

        Returns:
            Decoded text string

        Example:
            >>> vocab = Vocabulary()
            >>> vocab.build_from_corpus(["hello world"])
            >>> text = vocab.decode([2, 4, 5, 3], skip_special_tokens=True)
            >>> print(text)
            "hello world"
        """
        tokens = []

        for token_id in token_ids:
            # Get token string
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join tokens with spaces
        return " ".join(tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)

    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return f"Vocabulary(size={len(self)}, special_tokens={len(self.special_tokens)})"

    def save(self, filepath: str):
        """
        Save vocabulary to file.

        Args:
            filepath: Path to save vocabulary
        """
        import json

        data = {
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "token_counts": dict(self.token_counts),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        Load vocabulary from file.

        Args:
            filepath: Path to load vocabulary from

        Returns:
            Vocabulary object
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(special_tokens=data["special_tokens"], vocab_size=data["vocab_size"])

        # Restore token mappings
        vocab.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}

        # Restore token counts
        if "token_counts" in data:
            vocab.token_counts = Counter(data["token_counts"])

        return vocab


def test_vocabulary():
    """
    Test vocabulary implementation.

    Educational note: Tests verify:
    1. Special tokens are properly initialized
    2. Token addition works correctly
    3. Encoding/encoding work bidirectionally
    4. Vocabulary can be saved/loaded
    """
    print("Testing Vocabulary implementation...")

    # Test 1: Special tokens
    vocab = Vocabulary()
    assert vocab.PAD_TOKEN in vocab.token_to_id, "PAD token not in vocabulary"
    assert vocab.UNK_TOKEN in vocab.token_to_id, "UNK token not in vocabulary"
    assert vocab.BOS_TOKEN in vocab.token_to_id, "BOS token not in vocabulary"
    assert vocab.EOS_TOKEN in vocab.token_to_id, "EOS token not in vocabulary"
    print("✓ Test 1 passed: Special tokens initialized")

    # Test 2: Add tokens
    hello_id = vocab.add_token("hello")
    world_id = vocab.add_token("world")

    assert hello_id == vocab.num_special_tokens, "Hello ID incorrect"
    assert world_id == vocab.num_special_tokens + 1, "World ID incorrect"
    assert vocab.add_token("hello") == hello_id, "Duplicate token added again"
    print("✓ Test 2 passed: Token addition works")

    # Test 3: Build from corpus
    vocab2 = Vocabulary()
    corpus = ["hello world", "hello there", "world is great"]
    vocab2.build_from_corpus(corpus, min_frequency=1)

    assert "hello" in vocab2.token_to_id, "Hello not in vocabulary"
    assert "world" in vocab2.token_to_id, "World not in vocabulary"
    assert "there" in vocab2.token_to_id, "There not in vocabulary"
    print("✓ Test 3 passed: Building from corpus works")

    # Test 4: Encode/Decode
    vocab3 = Vocabulary()
    vocab3.build_from_corpus(["hello world"], min_frequency=1)

    # Encode
    encoded = vocab3.encode("hello world", add_special_tokens=True)
    assert encoded[0] == vocab3.bos_token_id, "BOS token not added"
    assert encoded[-1] == vocab3.eos_token_id, "EOS token not added"
    print("✓ Test 4a passed: Encoding works")

    # Decode
    decoded = vocab3.decode(encoded, skip_special_tokens=True)
    assert decoded == "hello world", f"Decoding failed: '{decoded}'"
    print("✓ Test 4b passed: Decoding works")

    # Test 5: Unknown tokens
    unknown_text = "hello unknown_word"
    encoded_unknown = vocab3.encode(unknown_text)
    unk_count = sum(1 for id in encoded_unknown if id == vocab3.unk_token_id)
    assert unk_count == 1, "Unknown token not handled correctly"
    print("✓ Test 5 passed: Unknown tokens handled")

    # Test 6: Save/Load
    import tempfile
    import os

    temp_file = tempfile.mktemp(suffix=".json")
    vocab3.save(temp_file)

    loaded_vocab = Vocabulary.load(temp_file)
    assert len(loaded_vocab) == len(vocab3), "Loaded vocab size mismatch"
    assert loaded_vocab.token_to_id == vocab3.token_to_id, "Loaded vocab mismatch"

    os.remove(temp_file)
    print("✓ Test 6 passed: Save/Load works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_vocabulary()

    # Educational: Demonstrate usage
    print("\n" + "=" * 70)
    print("Vocabulary Demonstration")
    print("=" * 70)

    # Create vocabulary
    vocab = Vocabulary(vocab_size=1000)

    print(f"\nInitial vocabulary:")
    print(f"  Size: {len(vocab)}")
    print(f"  Special tokens: {vocab.special_tokens}")
    print(f"  PAD ID: {vocab.pad_token_id}")
    print(f"  UNK ID: {vocab.unk_token_id}")
    print(f"  BOS ID: {vocab.bos_token_id}")
    print(f"  EOS ID: {vocab.eos_token_id}")

    # Build from corpus
    corpus = [
        "hello world",
        "hello there",
        "world is great",
        "great things happen",
        "things to learn",
    ]

    vocab.build_from_corpus(corpus, min_frequency=1)

    print(f"\nAfter building from corpus:")
    print(f"  Size: {len(vocab)}")
    print(f"  Regular tokens: {len(vocab) - vocab.num_special_tokens}")

    # Encode/Decode
    text = "hello great world"
    encoded = vocab.encode(text, add_special_tokens=True)
    decoded = vocab.decode(encoded, skip_special_tokens=True)

    print(f"\nEncode/Decode:")
    print(f"  Original: '{text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")

    # Unknown token
    unknown_text = "hello unknown_token"
    encoded_unknown = vocab.encode(unknown_text)
    decoded_unknown = vocab.decode(encoded_unknown, skip_special_tokens=True)

    print(f"\nUnknown token:")
    print(f"  Original: '{unknown_text}'")
    print(f"  Encoded: {encoded_unknown}")
    print(f"  Decoded: '{decoded_unknown}'")

    # Token frequencies
    print(f"\nToken frequencies (top 5):")
    for token, count in vocab.token_counts.most_common(5):
        print(f"  '{token}': {count}")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Vocabulary maps tokens to IDs for model processing")
    print("- Special tokens serve specific purposes (PAD, UNK, BOS, EOS)")
    print("- Most frequent tokens get lowest IDs (better for learning)")
    print("- Unknown tokens map to UNK token ID")
    print("- Vocabulary size affects model size and capability")
    print("- Modern LLMs use 32K-100K+ vocabularies")
    print("=" * 70)
