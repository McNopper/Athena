"""
Interactive Generation CLI

This module provides an interactive command-line interface for text generation.

Educational Notes:
- Interactive prompt-based generation
- User can enter prompts and see generated text
- Supports various generation parameters
- Useful for experimentation and exploration

Features:
- Continuous prompting loop
- Adjustable generation parameters
- Clear display of generated text
- Commands for special operations

Commands:
- /quit: Exit the interactive mode
- /params: Show or set generation parameters
- /reset: Clear conversation history
- /help: Show help message
"""

import torch
from typing import Optional

from ..model.llm import LLM
from .generator import TextGenerator


class InteractiveCLI:
    """
    Interactive command-line interface for text generation.

    Educational notes:
    - Provides REPL-like interface
    - User enters prompts, sees generated text
    - Can adjust parameters on the fly
    - Useful for experimentation

    Args:
        model: Language model
        tokenizer: Tokenizer for encoding/decoding (optional)

    Example:
        >>> cli = InteractiveCLI(model, tokenizer)
        >>> cli.run()
    """

    def __init__(self, model: LLM, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerator(model)

        # Generation parameters
        self.max_tokens = 100
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        self.do_sample = True

        # Conversation history
        self.history = []

    def run(self):
        """
        Start interactive CLI.

        Educational notes:
        - Runs continuous prompt loop
        - Accepts user input
        - Generates and displays text
        - Supports special commands

        Example:
            >>> cli.run()
            Enter prompt (or /help for commands): Hello, how are you?
            Generated: Hello! I'm doing well, thank you for asking...
        """
        print("\n" + "=" * 70)
        print("Interactive Text Generation")
        print("=" * 70)
        print("\nEnter prompts to generate text.")
        print("Special commands: /help, /params, /reset, /quit")
        print("-" * 70)

        while True:
            try:
                # Get user input
                user_input = input("\nPrompt: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Encode prompt (if tokenizer available)
                if self.tokenizer:
                    prompt_tokens = torch.tensor(
                        [self.tokenizer.encode(user_input, add_special_tokens=True)]
                    )
                else:
                    # Placeholder: Assume user provides token IDs
                    print("Note: No tokenizer provided. Please provide token IDs.")
                    continue

                # Move to device
                device = next(self.model.parameters()).device
                prompt_tokens = prompt_tokens.to(device)

                print(f"\nGenerating...")

                # Generate
                outputs = self.generator.generate(
                    prompt_tokens,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )

                # Decode output
                output_tokens = outputs[0]

                if self.tokenizer:
                    generated_text = self.tokenizer.decode(
                        output_tokens,
                        skip_special_tokens=True
                    )
                else:
                    # Placeholder: Show token IDs
                    generated_text = str(output_tokens)

                # Display output
                print(f"\nGenerated text:")
                print("-" * 70)
                print(generated_text)
                print("-" * 70)

                # Add to history
                self.history.append((user_input, generated_text))

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

    def _handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Educational notes:
        - /help: Show help
        - /params: Show/set parameters
        - /reset: Clear history
        - /quit: Exit CLI

        Args:
            command: Command string

        Returns:
            True to continue, False to exit

        Example:
            >>> should_continue = cli._handle_command("/help")
        """
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/quit":
            print("\nExiting...")
            return False

        elif cmd == "/help":
            self._show_help()
            return True

        elif cmd == "/params":
            if len(parts) == 1:
                self._show_params()
            else:
                self._set_params(parts[1:])
            return True

        elif cmd == "/reset":
            self.history.clear()
            print("Conversation history cleared.")
            return True

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")
            return True

    def _show_help(self):
        """Show help message."""
        print("\n" + "=" * 70)
        print("Available Commands")
        print("=" * 70)
        print("/help          - Show this help message")
        print("/params        - Show current generation parameters")
        print("/params <key> <value> - Set a parameter")
        print("                Available: max_tokens, temperature, top_k, top_p")
        print("                Example: /params temperature 0.7")
        print("/reset         - Clear conversation history")
        print("/quit          - Exit the program")
        print("=" * 70)

    def _show_params(self):
        """Show current parameters."""
        print("\n" + "=" * 70)
        print("Current Generation Parameters")
        print("=" * 70)
        print(f"max_tokens:   {self.max_tokens}")
        print(f"temperature:   {self.temperature}")
        print(f"top_k:         {self.top_k}")
        print(f"top_p:         {self.top_p}")
        print(f"do_sample:     {self.do_sample}")
        print("=" * 70)

    def _set_params(self, args: list):
        """
        Set generation parameters.

        Args:
            args: List of [key, value, ...]
        """
        if len(args) < 2:
            print("Usage: /params <key> <value>")
            return

        key = args[0]
        value = args[1]

        try:
            if key == "max_tokens":
                self.max_tokens = int(value)
                print(f"Set max_tokens = {self.max_tokens}")

            elif key == "temperature":
                self.temperature = float(value)
                print(f"Set temperature = {self.temperature}")

            elif key == "top_k":
                self.top_k = int(value)
                print(f"Set top_k = {self.top_k}")

            elif key == "top_p":
                self.top_p = float(value)
                print(f"Set top_p = {self.top_p}")

            else:
                print(f"Unknown parameter: {key}")
                print("Available: max_tokens, temperature, top_k, top_p")

        except ValueError as e:
            print(f"Invalid value: {e}")


def demo_interactive():
    """
    Demonstrate interactive CLI (without actually running).

    Educational note: Shows what the interactive CLI looks like.
    """
    print("\n" + "=" * 70)
    print("Interactive CLI Demonstration (Simulated)")
    print("=" * 70)

    print("\nInteractive Text Generation")
    print("Enter prompts to generate text.")
    print("Special commands: /help, /params, /reset, /quit")
    print("-" * 70)

    print("\nPrompt: Once upon a time")
    print("\nGenerating...")
    print("\nGenerated text:")
    print("-" * 70)
    print("Once upon a time, there was a small village nestled in the mountains...")
    print("-" * 70)

    print("\nPrompt: /params")
    print("\nCurrent Generation Parameters")
    print("-" * 70)
    print("max_tokens:   100")
    print("temperature:   0.8")
    print("top_k:         50")
    print("top_p:         0.9")
    print("-" * 70)

    print("\nPrompt: /params temperature 1.0")
    print("\nSet temperature = 1.0")

    print("\nPrompt: The meaning of life is")
    print("\nGenerated text:")
    print("-" * 70)
    print("The meaning of life is to find happiness and purpose in our actions...")
    print("-" * 70)

    print("\nPrompt: /quit")
    print("\nExiting...")

    print("\n" + "=" * 70)
    print("Educational notes:")
    print("- Interactive CLI for real-time generation")
    print("- Adjust parameters on the fly")
    print("- Experiment with different prompts")
    print("- Great for exploration and testing")
    print("=" * 70)


if __name__ == "__main__":
    demo_interactive()
