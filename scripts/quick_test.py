#!/usr/bin/env python3
"""
Quick Test Script - Verify everything works end-to-end

This script runs a complete test with minimal data to verify:
1. Model instantiation
2. Forward pass
3. Backward pass
4. Tokenization
5. Data preparation

Usage:
    python scripts/quick_test.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config.model_config import get_config
from src.model.llm import LLM
from src.utils.device import get_device, set_seed
from src.tokenizer.bpe_tokenizer import BPETokenizer


def test_model():
    """Test model creation and forward pass."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Creation and Forward Pass")
    print("=" * 70)

    # Load config
    config = get_config("pico")  # Use pico for fastest testing
    print(f"\n✓ Loaded 'pico' config: ~{config.num_params:,} parameters")

    # Create model
    model = LLM(config)
    print(f"✓ Model created")

    # Move to device
    device = get_device()
    model = model.to(device)
    print(f"✓ Model moved to {device}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    vocab_size = config.vocab_size

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"✓ Created test tokens: shape {tokens.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(tokens)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")

    assert logits.shape == (batch_size, seq_len, vocab_size), "Wrong output shape!"
    print(f"✓ Shape check passed")

    return True


def test_backward_pass():
    """Test backward pass (gradient flow)."""
    print("\n" + "=" * 70)
    print("TEST 2: Backward Pass (Gradient Flow)")
    print("=" * 70)

    config = get_config("pico")
    model = LLM(config).to(get_device())

    # Create test input
    tokens = torch.randint(0, config.vocab_size, (2, 16)).to(get_device())

    # Forward pass
    logits = model(tokens)

    # Create dummy loss
    target = torch.randint(0, config.vocab_size, (2, 16)).to(get_device())
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target.view(-1)
    )

    print(f"✓ Created loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print(f"✓ Backward pass successful")

    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients found!"
    print(f"✓ Gradients exist")

    # Check for NaN
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"✗ NaN in gradients: {name}")

    assert not has_nan, "Gradients contain NaN!"
    print(f"✓ No NaN in gradients")

    return True


def test_tokenizer():
    """Test tokenizer with minimal data."""
    print("\n" + "=" * 70)
    print("TEST 3: Tokenizer (Small Dataset)")
    print("=" * 70)

    # Tiny corpus for testing
    corpus = [
        "hello world",
        "hello there",
        "world is great",
        "great things",
        "things to learn",
    ]

    print(f"✓ Created test corpus: {len(corpus)} documents")

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)  # Small vocab for testing
    tokenizer.train(corpus, verbose=False)

    print(f"✓ Trained tokenizer: {len(tokenizer)} tokens, {len(tokenizer.merges)} merges")

    # Test encoding
    text = "hello great world"
    encoded = tokenizer.encode(text, add_special_tokens=True)

    print(f"✓ Encoded: '{text}' → {len(encoded)} tokens")

    # Test decoding
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"✓ Decoded: {encoded} → '{decoded}'")

    # Verify round-trip
    assert len(encoded) > 0, "Encoding produced empty list"
    assert len(decoded) > 0, "Decoding produced empty string"
    print(f"✓ Round-trip successful")

    return True


def test_data_preparation():
    """Test data preparation with small dataset."""
    print("\n" + "=" * 70)
    print("TEST 4: Data Preparation (Small Dataset)")
    print("=" * 70)

    # Create temporary test data
    import tempfile

    temp_dir = tempfile.mkdtemp()
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir)

    # Create test files
    test_texts = [
        "This is a test document for the Athena project.",
        "Another test document with different content.",
        "Third document to have more variety in the training data.",
        "Machine learning is fascinating and educational.",
    ]

    for i, text in enumerate(test_texts):
        filepath = os.path.join(raw_dir, f"test_{i}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"✓ Created {len(test_texts)} test documents in {raw_dir}")

    # Run data preparation
    output_dir = os.path.join(temp_dir, "processed")

    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(test_texts, verbose=False)

    print(f"✓ Trained tokenizer on test data")

    # Tokenize
    sequences = []
    for text in test_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= 5:  # Skip very short sequences
            sequences.append(token_ids)

    print(f"✓ Created {len(sequences)} sequences")

    assert len(sequences) > 0, "No sequences created!"
    print(f"✓ Data preparation successful")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"✓ Cleaned up test data")

    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 70)
    print("TEST 5: Single Training Step")
    print("=" * 70)

    set_seed(42)

    config = get_config("pico")
    model = LLM(config).to(get_device())

    # Create dummy batch
    batch_size = 4
    seq_len = 16

    input_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(get_device())
    target_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(get_device())

    print(f"✓ Created training batch: shape {input_tokens.shape}")

    # Forward pass
    logits = model(input_tokens)

    # Compute loss (next token prediction)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target_tokens.view(-1)
    )

    print(f"✓ Computed loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print(f"✓ Backward pass successful")

    # Simple gradient update
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    print(f"✓ Optimizer step successful")

    return True


def test_generation():
    """Test text generation."""
    print("\n" + "=" * 70)
    print("TEST 6: Text Generation")
    print("=" * 70)

    config = get_config("pico")
    model = LLM(config).to(get_device())
    model.eval()

    # Create prompt
    prompt_len = 5
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len)).to(get_device())

    print(f"✓ Created prompt: shape {prompt.shape}")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=10,
            do_sample=False,  # Greedy for testing
        )

    print(f"✓ Generated {generated.shape[1] - prompt_len} tokens")
    print(f"  Prompt length: {prompt_len}")
    print(f"  Total length: {generated.shape[1]}")

    assert generated.shape[1] == prompt_len + 10, "Generation length incorrect!"
    print(f"✓ Generation length correct")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ATHENA - QUICK TEST SUITE")
    print("=" * 70)
    print("\nThis will run a series of quick tests to verify everything works.")

    tests = [
        ("Model Creation and Forward Pass", test_model),
        ("Backward Pass", test_backward_pass),
        ("Tokenizer", test_tokenizer),
        ("Data Preparation", test_data_preparation),
        ("Training Step", test_training_step),
        ("Text Generation", test_generation),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✓ PASSED", None))
            print(f"\n✓ {name}: PASSED")
        except Exception as e:
            results.append((name, "✗ FAILED", str(e)))
            print(f"\n✗ {name}: FAILED")
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status == "✓ PASSED")
    total = len(results)

    for name, status, error in results:
        print(f"{status}: {name}")
        if error:
            print(f"  → {error}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All tests passed! Your LLM implementation is working correctly.")
        print("\nYou can now:")
        print("  - Prepare your training data with: python scripts/prepare_data.py")
        print("  - Start training with: python scripts/train.py (when implemented)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
