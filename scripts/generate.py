#!/usr/bin/env python3
"""
Text Generation Script

This script generates text from a trained language model.

Usage:
    python scripts/generate.py --checkpoint data/checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json --prompt "Once upon a time"

Educational Notes:
- Loads a trained model (architecture is read from the checkpoint when present)
- Loads the BPE tokenizer used during data preparation
- Encodes the prompt, generates tokens autoregressively, then decodes to text
- Supports greedy decoding and temperature / top-k / top-p sampling

Examples:
    # Single generation
    python scripts/generate.py --checkpoint data/checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json --prompt "Once upon a time"

    # Interactive mode
    python scripts/generate.py --checkpoint data/checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json --interactive

    # Batch generation from a file (one prompt per line)
    python scripts/generate.py --checkpoint data/checkpoints/best.pt \
        --tokenizer data/processed/tokenizer.json --prompts-file prompts.txt
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config.model_config import get_config
from src.model.llm import LLM
from src.utils.device import get_device
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.inference.interactive import InteractiveCLI


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text from a trained language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="small",
                        choices=["pico", "nano", "micro", "tiny", "small", "medium", "large"],
                        help="Model configuration (fallback if the checkpoint has none)")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer.json (defaults to data/processed/tokenizer.json)")

    # Input
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="File containing prompts (one per line)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--no-sample", action="store_true",
                        help="Use greedy decoding instead of sampling")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for generated text")

    # Misc
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "cuda", "xpu", "mps", "directml", "cpu"],
                        help="Compute backend / GPU vendor to use")

    return parser.parse_args()


def load_tokenizer(args) -> BPETokenizer:
    """Locate and load the BPE tokenizer."""
    path = args.tokenizer
    if path is None:
        # Sensible defaults: next to the checkpoint, then data/processed/.
        candidates = [
            Path(args.checkpoint).parent / "tokenizer.json",
            Path("data/processed/tokenizer.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "Tokenizer not found. Pass --tokenizer <path/to/tokenizer.json> "
            "(created by scripts/prepare_data.py)."
        )

    print(f"Loading tokenizer from {path}")
    return BPETokenizer.load(path)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, config_name: str = "small") -> LLM:
    """
    Load a model from a checkpoint.

    Educational notes:
    - Prefers the architecture stored in the checkpoint so weights always match.
    - Falls back to a named preset config only if the checkpoint has none.
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_config" in checkpoint and checkpoint["model_config"] is not None:
        config = checkpoint["model_config"]
        print("Using model configuration stored in checkpoint")
    else:
        config = get_config(config_name)
        print(f"Checkpoint has no config; using preset '{config_name}'")

    model = LLM(config)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model weights")

    model = model.to(device)
    model.eval()
    return model


def generate_text(model: LLM, tokenizer: BPETokenizer, prompt: str, args, device: torch.device) -> str:
    """Encode a prompt, generate, and decode back to text."""
    # Prepend BOS only. EOS marks end-of-sequence in the training data, so
    # appending it to the prompt would tell the model the document is already
    # over and make it ignore the prompt instead of continuing it.
    prompt_ids = [tokenizer.vocab.bos_token_id] + tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def main():
    """Main generation function."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("Text Generation Script")
    print("=" * 70)

    device = get_device(backend=args.backend)

    model = load_model_from_checkpoint(args.checkpoint, device, args.config)
    tokenizer = load_tokenizer(args)
    print("Model and tokenizer loaded successfully")

    # Interactive mode
    if args.interactive:
        cli = InteractiveCLI(model, tokenizer=tokenizer)
        cli.max_tokens = args.max_tokens
        cli.temperature = args.temperature
        cli.top_k = args.top_k
        cli.top_p = args.top_p
        cli.do_sample = not args.no_sample
        cli.run()
        return 0

    outputs = []

    # Single prompt
    if args.prompt:
        print("\n" + "-" * 70)
        print(f"Prompt: {args.prompt}")
        print("-" * 70)
        text = generate_text(model, tokenizer, args.prompt, args, device)
        print(text)
        outputs.append(text)

    # Batch from file
    elif args.prompts_file:
        if not os.path.exists(args.prompts_file):
            print(f"Error: Prompts file not found: {args.prompts_file}")
            return 1

        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        for i, prompt in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}] Prompt: {prompt}")
            text = generate_text(model, tokenizer, prompt, args, device)
            print(text)
            outputs.append(text)

    else:
        print("\nError: Please provide --prompt, --prompts-file, or --interactive")
        return 1

    # Optional output file
    if args.output and outputs:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("\n\n".join(outputs))
        print(f"\nSaved generated text to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
