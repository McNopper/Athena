"""
Text Generator

This module implements text generation with various sampling strategies.

Educational Notes:
- Generates text autoregressively (token by token)
- Supports multiple sampling strategies
- Uses KV cache for efficient O(N) generation (vs. O(N²) without cache)
- Controls output diversity with temperature, top-k, and top-p

Sampling Strategies:
1. Greedy: Always pick the most likely token (deterministic, often repetitive)
2. Top-k: Sample from the top k most likely tokens only
3. Top-p (Nucleus): Sample from the smallest token set whose cumulative
   probability ≥ p — more adaptive than fixed top-k
4. Temperature: Scale logits before softmax to control sharpness of distribution

Temperature:
- Low (< 1.0): More deterministic, focused output
- High (> 1.0): More random, diverse output
- 1.0: No scaling (standard distribution)

Top-k Sampling:
- Only consider the top k tokens
- Prevents sampling from very unlikely tokens
- k=1: Same as greedy
- k=vocab_size: No filtering

Top-p (Nucleus) Sampling:
- Consider tokens with cumulative probability ≥ p
- More adaptive than top-k: the set can be large for uncertain distributions
  and small for confident ones
- p=0.9: Common choice
- p=1.0: No filtering

Generation Flow:
    Start with prompt tokens
    Loop until max tokens or EOS:
        1. Forward pass (using KV cache for efficiency)
        2. Apply sampling strategy (temperature, top-k, top-p)
        3. Sample next token
        4. Append to sequence
        5. Update KV cache
    Return generated sequence

References:
    Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019) — https://arxiv.org/abs/1904.09751
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict

from ..model.llm import LLM
from .kv_cache import KVCache


class TextGenerator:
    """
    Text generator with sampling strategies.

    Educational notes:
    - Generates text autoregressively
    - Supports multiple sampling strategies
    - Uses KV cache for efficiency
    - Can generate multiple sequences in parallel

    Args:
        model: Language model
        max_sequence_length: Maximum generation length
        use_kv_cache: Whether to use KV cache (recommended)

    Example:
        >>> generator = TextGenerator(model)
        >>> output = generator.generate("Once upon a time", max_tokens=50)
        >>> print(output)
    """

    def __init__(
        self,
        model: LLM,
        max_sequence_length: int = 512,
        use_kv_cache: bool = True,
    ):
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.use_kv_cache = use_kv_cache

        # Model config
        self.config = model.config
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from prompt.

        Educational notes:
        - Generates text autoregressively
        - Supports various sampling strategies
        - Uses KV cache for efficiency

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if set)
            top_p: Top-p (nucleus) sampling (if set)
            do_sample: If False, use greedy decoding
            eos_token_id: End-of-sequence token (stops generation)
            num_return_sequences: Number of samples to return per prompt
                (prompts are repeated along the batch dimension)

        Returns:
            List of generated token-id sequences, length
            num_prompts * num_return_sequences (a prompt's samples are contiguous)

        Example:
            >>> generator = TextGenerator(model)
            >>> outputs = generator.generate(
            ...     "Once upon a time",
            ...     max_tokens=50,
            ...     temperature=0.8
            ... )
        """
        self.model.eval()

        # Robustness: temperature <= 0 means greedy (avoid divide-by-zero);
        # top_k <= 0 disables top-k filtering.
        if temperature is None or temperature <= 0:
            do_sample = False
            temperature = 1.0
        if top_k is not None and top_k <= 0:
            top_k = None
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")

        # Educational note: In practice, you'd use a tokenizer
        # For now, assume prompt is already tokenized
        if isinstance(prompt, str):
            # This would need a tokenizer - placeholder for now
            raise ValueError("Please provide tokenized prompt as tensor")

        prompt_tokens = prompt  # Already tokenized

        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        # Produce num_return_sequences samples per prompt by repeating each
        # prompt along the batch dimension (interleaved so a prompt's samples
        # stay contiguous). With do_sample=True each copy is sampled
        # independently; with greedy decoding the copies are identical.
        if num_return_sequences > 1:
            prompt_tokens = prompt_tokens.repeat_interleave(num_return_sequences, dim=0)

        batch_size = prompt_tokens.shape[0]
        prompt_len = prompt_tokens.shape[1]

        # Check max length
        if prompt_len + max_tokens > self.max_sequence_length:
            max_tokens = self.max_sequence_length - prompt_len

        # Initialize KV cache (one shared object holds every layer's K/V).
        kv_cache = None
        if self.use_kv_cache:
            kv_cache = KVCache(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                head_dim=self.config.dim // self.config.num_heads,
                max_sequence_length=self.max_sequence_length,
                dtype=next(self.model.parameters()).dtype,
                device=self.device,
            )

        # Store all generated tokens
        generated_tokens = prompt_tokens.clone()

        # Generate tokens one by one
        for step in range(max_tokens):
            # Educational note: With a KV cache, after the first step we only
            # feed the single new token and tell the model its absolute
            # position. Without a cache we re-feed the whole sequence each step.
            if kv_cache is not None and step > 0:
                input_tokens = generated_tokens[:, -1:]
                start_pos = generated_tokens.shape[1] - 1
            else:
                input_tokens = generated_tokens
                start_pos = 0

            # Forward pass
            logits = self.model(input_tokens, kv_cache=kv_cache, start_pos=start_pos)

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                next_token_logits = self._apply_top_k(next_token_logits, top_k)

            # Apply top-p filtering
            if top_p is not None:
                next_token_logits = self._apply_top_p(next_token_logits, top_p)

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

            # Append to generated tokens
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        # Decode (placeholder - would need tokenizer in practice)
        # For now, return token IDs
        return [generated_tokens[i].cpu().tolist() for i in range(generated_tokens.shape[0])]

    def _apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k filtering to logits.

        Educational notes:
        - Keep only top k logits
        - Set others to -inf (so they have 0 probability)
        - Prevents model from sampling unlikely tokens

        Args:
            logits: Logits tensor (batch, vocab_size)
            k: Number of top tokens to keep

        Returns:
            Filtered logits

        Example:
            >>> filtered = _apply_top_k(logits, k=50)
        """
        k = min(k, logits.shape[-1])
        v, _ = torch.topk(logits, k)
        # Return a new tensor (avoid mutating the caller's logits in place).
        return logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.

        Educational notes:
        - Keep tokens with cumulative probability >= p
        - More adaptive than top-k
        - Commonly used with p=0.9

        Args:
            logits: Logits tensor (batch, vocab_size)
            p: Cumulative probability threshold

        Returns:
            Filtered logits

        Example:
            >>> filtered = _apply_top_p(logits, p=0.9)
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits


def test_generator():
    """
    Test text generator implementation.

    Educational note: Tests verify:
    1. Generator creation works
    2. Greedy generation works
    3. Sampling works
    """
    print("Testing Text Generator implementation...")

    from ..model.llm import LLM
    from ..config.model_config import get_config

    # Test 1: Create generator
    config = get_config("pico")
    model = LLM(config)

    device = get_device()
    model = model.to(device)

    generator = TextGenerator(model)
    print("✓ Test 1 passed: Generator created")

    # Test 2: Generate with greedy decoding
    prompt = torch.randint(0, config.vocab_size, (1, 10)).to(device)

    outputs = generator.generate(
        prompt,
        max_tokens=5,
        do_sample=False,
    )

    assert len(outputs) == 1, "Should return 1 sequence"
    assert len(outputs[0]) == 15, "Should have 10 + 5 tokens"
    print("✓ Test 2 passed: Greedy generation works")

    # Test 3: Generate with sampling
    outputs = generator.generate(
        prompt,
        max_tokens=5,
        temperature=0.8,
        do_sample=True,
    )

    assert len(outputs) == 1, "Should return 1 sequence"
    print("✓ Test 3 passed: Sampling generation works")

    print("\nAll tests passed! ✓")


def get_device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    from ..config.model_config import get_config
    from ..model.llm import LLM

    # Educational: Run tests
    test_generator()

    # Educational: Demonstration
    print("\n" + "=" * 70)
    print("Text Generator Demonstration")
    print("=" * 70)

    # Create model and generator
    config = get_config("pico")
    model = LLM(config)

    device = get_device()
    model = model.to(device)

    generator = TextGenerator(model)

    print(f"\nGenerator created:")
    print(f"  Model: {config}")
    print(f"  Max sequence length: {generator.max_sequence_length}")

    # Create prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10)).to(device)

    print(f"\nGenerating with different strategies:")

    # Greedy
    output_greedy = generator.generate(prompt, max_tokens=5, do_sample=False)
    print(f"  Greedy: Generated {len(output_greedy[0])} tokens")

    # Sampling
    output_temp = generator.generate(prompt, max_tokens=5, temperature=0.7, do_sample=True)
    print(f"  Temperature (0.7): Generated {len(output_temp[0])} tokens")

    # Top-k
    output_topk = generator.generate(prompt, max_tokens=5, top_k=50, do_sample=True)
    print(f"  Top-k (50): Generated {len(output_topk[0])} tokens")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Generator produces text token by token")
    print("- Greedy: Always pick most likely token")
    print("- Sampling: Random pick based on probabilities")
    print("- Temperature: Control randomness")
    print("- Top-k: Sample from top k tokens")
    print("- Top-p: Sample from tokens with cumulative prob p")
    print("- KV cache: Dramatically speeds up generation")
    print("=" * 70)
