"""Command-line entry point for Adventure Story Weaver.

Usage examples:
    # Check that everything is wired up
    python -m app.cli hello

    # Generate a story continuation
    python -m app.cli "You stand at the mouth of a dark cave" --temperature 0.8

    # Start interactive storytelling mode
    python -m app.cli --interactive

    # Train tokenizer or model
    python -m app.cli train-tokenizer
    python -m app.cli train --max_steps 500
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.generate import generate, load_checkpoint, validate_generation_params


def build_parser() -> argparse.ArgumentParser:
    """Constructs the unified CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="adventure-story-weaver",
        description="Adventure Story Weaver: GPT language model for text-adventure story generation.",
    )
    # Positional prompt or subcommand
    parser.add_argument(
        "prompt_or_cmd",
        nargs="?",
        default=None,
        help="Story prompt string, or subcommand ('hello', 'train', 'train-tokenizer')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive story session in your terminal",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling threshold (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Path to trained model checkpoint (default: checkpoints/model.pt)",
    )

    # Optional training flags
    parser.add_argument("--corpus", default="data/corpus.txt", help="Path to text corpus")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--output", default="artifacts/tokenizer.json", help="Output path for tokenizer")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--max_steps", type=int, default=200, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Training learning rate")

    return parser


def run_interactive(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    """Runs a loop for interactive story generation until 'quit' or 'exit'."""
    print("=" * 60)
    print("      Adventure Story Weaver - Interactive Story Mode")
    print("=" * 60)
    print("Enter a story prompt to generate a continuation.")
    print("Type 'quit' or 'exit' (or Ctrl+C) to end the session.\n")

    while True:
        try:
            prompt = input("Story Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode. Farewell, adventurer!")
            break

        if prompt.lower() in ("quit", "exit"):
            print("Farewell, adventurer!")
            break

        if not prompt:
            continue

        try:
            story = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
            print("\n--- Continuation ---")
            print(story)
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"Error generating continuation: {e}\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Hello command (smoke test / sanity check)
    if args.prompt_or_cmd == "hello":
        print("Adventure Story Weaver is ready. Open README.md for your milestones.")
        return

    # 2. Tokenizer training subcommand
    if args.prompt_or_cmd == "train-tokenizer":
        from scripts.train_tokenizer import train_and_save_tokenizer
        _, stats = train_and_save_tokenizer(
            corpus_path=Path(args.corpus),
            output_path=Path(args.output),
            vocab_size=args.vocab_size,
        )
        print(f"Trained tokenizer with vocab size {stats['vocab_size']}, compression ratio {stats['compression_ratio']:.2f} chars/token.")
        return

    # 3. Model training subcommand
    if args.prompt_or_cmd == "train":
        from app.train import train
        train(
            corpus_path=args.corpus,
            tokenizer_path=args.tokenizer,
            checkpoint_path=args.checkpoint,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed or 1337,
        )
        return

    # 4. Parameter validation for generation
    try:
        validate_generation_params(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    except ValueError as err:
        print(f"Error: Invalid argument: {err}", file=sys.stderr)
        sys.exit(1)

    # 5. Checkpoint loading
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(
            f"Error: Model checkpoint not found at '{args.checkpoint}'.\n"
            f"Please train the model first by running:\n"
            f"    python -m app.cli train --max_steps 500",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model, tokenizer = load_checkpoint(ckpt_path)
    except Exception as err:
        print(f"Error loading checkpoint '{args.checkpoint}': {err}", file=sys.stderr)
        sys.exit(1)

    # 6. Interactive mode
    if args.interactive:
        run_interactive(model, tokenizer, args)
        return

    # 7. Positional prompt generation
    prompt = args.prompt_or_cmd
    if not prompt:
        print("Error: Please provide a story prompt or use --interactive.", file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        sys.exit(1)

    continuation = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(continuation)


if __name__ == "__main__":
    main()
