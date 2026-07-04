"""
Byte-Pair Encoding (BPE) Tokenizer

This module implements a BPE tokenizer for subword tokenization.

Educational Notes:
- BPE is a subword tokenization algorithm
- Balances character-level and word-level tokenization
- Compresses common character sequences into single tokens
- Handles unknown words by breaking them into subwords
- Used in GPT-2, GPT-3, RoBERTa, and many modern LLMs

What is BPE?
- Iteratively merges the most frequent pair of characters/subwords
- Starts with characters as initial vocabulary
- Builds vocabulary by merging frequent pairs
- Results in subword tokens (e.g., "ing", "tion", "ness")

Why Subword Tokenization?
- Character-level: Too many tokens, no semantic meaning
- Word-level: Too large vocabulary, can't handle unknown words
- Subword: Best of both - manageable size, handles unknown words

BPE Algorithm:
1. Start with character vocabulary
2. Count all character pairs in corpus
3. Merge most frequent pair into new token
4. Repeat until desired vocabulary size

Example:
    Start: "h", "e", "l", "l", "o"
    Step 1: Merge "l" + "l" → "ll"
    Step 2: Merge "e" + "ll" → "ell"
    Step 3: Merge "h" + "ell" → "hell"
    Step 4: Merge "hell" + "o" → "hello"
    Result: "hello" is now a single token
"""

from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import json
import re

from .vocabulary import Vocabulary


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.

    Educational notes:
    - Learns subword vocabulary from training corpus
    - Encodes text by greedy longest-match first algorithm
    - Can handle any text by breaking into subwords
    - Vocabulary size is fixed after training

    Training Process:
        1. Start with character vocabulary
        2. Count all symbol pairs in corpus
        3. Merge most frequent pair
        4. Add merged pair to vocabulary
        5. Repeat until vocabulary size reached

    Encoding Process:
        1. Split text into characters
        2. Greedily merge most frequent pairs (from learned merges)
        3. Convert to token IDs

    Args:
        vocab_size: Target vocabulary size (including special tokens)
        special_tokens: List of special tokens

    Example:
        >>> tokenizer = BPETokenizer(vocab_size=1000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> ids = tokenizer.encode("hello world")
        >>> print(ids)
        [2, 45, 67, 3]  # BOS, hello, world, EOS
    """

    # Marker appended to the end of every word so the tokenizer can tell word
    # boundaries apart and reconstruct spaces when decoding.
    WORD_END = "</w>"

    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] = None):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens
        """
        self.vocab_size = vocab_size

        # Initialize vocabulary
        self.vocab = Vocabulary(special_tokens=special_tokens, vocab_size=vocab_size)

        # Learned merge rules in the order they were learned.
        # Example: [('h', 'e'), ('he', 'l'), ('hel', 'lo</w>')]
        self.merges: List[Tuple[str, str]] = []

        # pair -> rank (lower rank = learned earlier = higher priority).
        # Rebuilt from self.merges after training or loading.
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    def _word_to_symbols(self, word: str) -> List[str]:
        """
        Split a word into its initial symbol sequence.

        Educational note:
        - Each character becomes a symbol, plus a trailing word-end marker.
        - The marker (</w>) lets us reconstruct spaces during decoding and
          keeps merges from crossing word boundaries.

        Example: "cat" -> ["c", "a", "t", "</w>"]
        """
        return list(word) + [self.WORD_END]

    def _get_pair_counts(self, splits: Dict[str, List[str]], word_freqs: Counter) -> Counter:
        """
        Count adjacent symbol pairs across the corpus, weighted by word frequency.

        Educational note:
        - Operates on the *symbol lists*, not their string representation, so a
          pair is always two neighbouring subwords (never a stray separator).
        - Frequent pairs are the candidates for the next merge.

        Args:
            splits: Mapping of word -> current list of symbols
            word_freqs: Frequency of each word in the corpus

        Returns:
            Counter mapping (symbol_a, symbol_b) -> total frequency
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge_symbols(symbols: List[str], pair: Tuple[str, str]) -> List[str]:
        """
        Merge every occurrence of ``pair`` in ``symbols`` into a single symbol.

        Example: merge ("l", "l") in ["h","e","l","l","o"] -> ["h","e","ll","o"]
        """
        merged_symbol = pair[0] + pair[1]
        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                result.append(merged_symbol)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return result

    def train(self, corpus: List[str], verbose: bool = True):
        """
        Train BPE tokenizer on corpus.

        Educational notes:
        - Builds vocabulary by iteratively merging the most frequent pair
        - Starts with a character-level vocabulary
        - Learns an ordered list of merge rules

        Steps:
        1. Preprocess corpus (lowercase, split into words)
        2. Build initial vocabulary (characters + word-end marker)
        3. Iteratively merge the most frequent adjacent pair
        4. Stop when the target vocabulary size is reached

        Args:
            corpus: List of text strings
            verbose: Print progress

        Example:
            >>> tokenizer = BPETokenizer(vocab_size=1000)
            >>> tokenizer.train(["hello world", "hello there"])
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(corpus)} documents...")

        # Step 1: Preprocess corpus into word frequencies.
        word_freqs: Counter = Counter()
        for text in corpus:
            for word in text.lower().split():
                word_freqs[word] += 1

        if verbose:
            print(f"  Found {len(word_freqs)} unique words")

        # Step 2: Represent each word as a list of symbols (chars + </w>) and
        # seed the vocabulary with the base alphabet.
        splits: Dict[str, List[str]] = {word: self._word_to_symbols(word) for word in word_freqs}

        alphabet = set()
        for word in word_freqs:
            alphabet.update(word)
        for char in sorted(alphabet):
            self.vocab.add_token(char)
        self.vocab.add_token(self.WORD_END)

        if verbose:
            print(f"  Initial character vocabulary: {len(self.vocab)} tokens")

        # Step 3: Iteratively merge the most frequent pair.
        # Educational note: len(self.vocab) already counts the special tokens, so
        # we grow until the whole vocabulary (specials + base chars + merges)
        # reaches the requested size. add_token() also enforces this cap.
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_counts(splits, word_freqs)
            if not pair_freqs:
                break  # No more pairs to merge

            # Most frequent pair (ties broken deterministically by symbol order).
            best_pair, best_freq = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))

            merged_symbol = best_pair[0] + best_pair[1]
            new_token_id = self.vocab.add_token(merged_symbol)
            if new_token_id == self.vocab.unk_token_id:
                # Vocabulary is full.
                break

            self.merges.append(best_pair)

            # Apply the merge to every word.
            for word in splits:
                splits[word] = self._merge_symbols(splits[word], best_pair)

            if verbose and len(self.merges) % 1000 == 0:
                shown = merged_symbol.replace(self.WORD_END, " ")
                print(f"  Progress: {len(self.vocab)}/{self.vocab_size} tokens, "
                      f"last merge: '{shown}' (freq: {best_freq})")

        # Cache merge priorities for fast, correct encoding.
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if verbose:
            print(f"  Final vocabulary size: {len(self.vocab)} tokens")
            print(f"  Learned {len(self.merges)} merge rules")

    def _bpe_encode_word(self, word: str) -> List[str]:
        """
        Apply the learned merges to a single word, respecting merge priority.

        Educational note:
        - At each step we pick the adjacent pair with the *lowest rank* (the
          earliest-learned merge), matching how BPE is meant to be applied.
        - This is more correct than merging the left-most learnable pair.
        """
        symbols = self._word_to_symbols(word)

        while len(symbols) > 1:
            # Find the highest-priority (lowest-rank) adjacent pair.
            best_rank = None
            best_index = None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break  # No more applicable merges

            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]

        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs using BPE.

        Educational notes:
        - Applies learned merge rules in priority order
        - Converts text to subwords, then subwords to token IDs
        - Unknown symbols map to the UNK token

        Args:
            text: Input text string
            add_special_tokens: Add BOS/EOS tokens

        Returns:
            List of token IDs

        Example:
            >>> tokenizer = BPETokenizer(vocab_size=1000)
            >>> tokenizer.train(["hello world"])
            >>> ids = tokenizer.encode("hello world")
        """
        token_ids = []

        for word in text.lower().split():
            for symbol in self._bpe_encode_word(word):
                token_ids.append(self.vocab.token_to_id.get(symbol, self.vocab.unk_token_id))

        if add_special_tokens:
            token_ids = [self.vocab.bos_token_id] + token_ids + [self.vocab.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text.

        Educational notes:
        - Converts token IDs back to subword strings and concatenates them
        - The word-end marker (</w>) is turned back into a space

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text string

        Example:
            >>> tokenizer = BPETokenizer(vocab_size=1000)
            >>> tokenizer.train(["hello world"])
            >>> text = tokenizer.decode(tokenizer.encode("hello world"))
        """
        tokens = []

        for token_id in token_ids:
            token = self.vocab.id_to_token.get(token_id, self.vocab.UNK_TOKEN)

            if skip_special_tokens and token in self.vocab.special_tokens:
                continue

            tokens.append(token)

        # Concatenate subwords, then turn word-end markers back into spaces.
        text = "".join(tokens).replace(self.WORD_END, " ")

        return text.strip()

    def save(self, filepath: str):
        """
        Save tokenizer to file.

        Args:
            filepath: Path to save
        """
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": {
                "token_to_id": self.vocab.token_to_id,
                "special_tokens": self.vocab.special_tokens,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        Load tokenizer from file.

        Args:
            filepath: Path to load from

        Returns:
            BPETokenizer object
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])

        # Restore vocabulary
        tokenizer.vocab = Vocabulary(special_tokens=data["vocab"]["special_tokens"], vocab_size=data["vocab_size"])
        tokenizer.vocab.token_to_id = {k: int(v) for k, v in data["vocab"]["token_to_id"].items()}
        tokenizer.vocab.id_to_token = {v: k for k, v in tokenizer.vocab.token_to_id.items()}

        # Restore merges
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """String representation."""
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


def test_bpe_tokenizer():
    """
    Test BPE tokenizer implementation.

    Educational note: Tests verify:
    1. Training works correctly
    2. Encoding/decoding work bidirectionally
    3. Special tokens are handled
    4. Save/load works
    """
    print("Testing BPE Tokenizer implementation...")

    # Test 1: Training
    tokenizer = BPETokenizer(vocab_size=100)
    corpus = ["hello world", "hello there", "world is great", "great things"]

    tokenizer.train(corpus, verbose=False)

    assert len(tokenizer.vocab) > 0, "Vocabulary is empty"
    assert len(tokenizer.merges) > 0, "No merges learned"
    print("✓ Test 1 passed: Training works")

    # Test 2: Encoding
    text = "hello world"
    encoded = tokenizer.encode(text)

    assert len(encoded) > 0, "Encoding produced empty list"
    assert all(isinstance(id, int) for id in encoded), "Non-integer token IDs"
    print("✓ Test 2 passed: Encoding works")

    # Test 3: Encoding with special tokens
    encoded_special = tokenizer.encode(text, add_special_tokens=True)

    assert encoded_special[0] == tokenizer.vocab.bos_token_id, "BOS token not added"
    assert encoded_special[-1] == tokenizer.vocab.eos_token_id, "EOS token not added"
    assert len(encoded_special) == len(encoded) + 2, "Special tokens not added correctly"
    print("✓ Test 3 passed: Special tokens work")

    # Test 4: Decoding
    decoded = tokenizer.decode(encoded_special, skip_special_tokens=True)

    # Check that we get back something similar (might not be exact due to preprocessing)
    assert len(decoded) > 0, "Decoding produced empty string"
    print("✓ Test 4 passed: Decoding works")

    # Test 5: Round-trip
    # Encode and decode should approximately preserve text
    text2 = "hello great world"
    encoded2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(encoded2)

    # Check that key words are preserved
    assert "hello" in decoded2.lower() or "hel" in decoded2.lower(), "Round-trip failed"
    print("✓ Test 5 passed: Round-trip works")

    # Test 6: Save/Load
    import tempfile
    import os

    temp_file = tempfile.mktemp(suffix=".json")
    tokenizer.save(temp_file)

    loaded_tokenizer = BPETokenizer.load(temp_file)

    assert len(loaded_tokenizer.vocab) == len(tokenizer.vocab), "Loaded vocab size mismatch"
    assert loaded_tokenizer.merges == tokenizer.merges, "Loaded merges mismatch"

    os.remove(temp_file)
    print("✓ Test 6 passed: Save/Load works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    # Educational: Run tests
    test_bpe_tokenizer()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("BPE Tokenizer Demonstration")
    print("=" * 70)

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=200)

    corpus = [
        "hello world",
        "hello there",
        "world is great",
        "great things happen",
        "things to learn",
        "learning is wonderful",
        "wonderful world",
    ]

    print(f"\nTraining on {len(corpus)} example sentences...")

    tokenizer.train(corpus, verbose=True)

    # Test encoding/decoding
    print("\n" + "=" * 70)
    print("Encoding/Decoding Examples")
    print("=" * 70)

    test_texts = ["hello world", "great things", "unknown word here"]

    for text in test_texts:
        # Encode
        encoded = tokenizer.encode(text, add_special_tokens=True)

        # Decode
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        print(f"\nOriginal:  '{text}'")
        print(f"Encoded:   {encoded[:10]}{'...' if len(encoded) > 10 else ''} ({len(encoded)} tokens)")
        print(f"Decoded:   '{decoded}'")

    # Show some learned merges
    print("\n" + "=" * 70)
    print("Learned Merge Rules (first 10)")
    print("=" * 70)

    for i, (pair1, pair2) in enumerate(tokenizer.merges[:10]):
        merged = pair1 + pair2
        print(f"  {i+1}. '{pair1.replace('_', ' ')}' + '{pair2.replace('_', ' ')}' → '{merged.replace('_', ' ')}'")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- BPE learns subword vocabulary from training corpus")
    print("- Iteratively merges most frequent character pairs")
    print("- Balances character-level and word-level tokenization")
    print("- Can handle any word by breaking into subwords")
    print("- Vocabulary size is fixed after training")
    print("- Used in GPT-2, GPT-3, RoBERTa, and modern LLMs")
    print("=" * 70)
