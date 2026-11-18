"""Script to train the BPE tokenizer on the adventure corpus and save to artifacts.

Usage:
    python scripts/train_tokenizer.py --vocab-size 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tokenizer import BPETokenizer


def train_and_save_tokenizer(
    corpus_path: Path = Path("data/corpus.txt"),
    output_path: Path = Path("artifacts/tokenizer.json"),
    vocab_size: int = 1000,
    max_chars: int = 250_000,
) -> tuple[BPETokenizer, dict[str, float | int | str]]:
    """Trains a BPE tokenizer on corpus_path and saves it to output_path."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file '{corpus_path}' not found.\n"
            f"Please run 'python scripts/build_corpus.py --download-sample' first."
        )

    text = corpus_path.read_text(encoding="utf-8")
    print(f"Loaded corpus from '{corpus_path}' ({len(text):,} characters).")
    print(f"Training BPE tokenizer (vocab_size={vocab_size}, training sample={min(len(text), max_chars):,} chars)...")

    tokenizer = BPETokenizer()
    tokenizer.train(
        text=text,
        vocab_size=vocab_size,
        max_chars=max_chars,
        special_tokens=["<|endoftext|>"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved successfully to '{output_path}'.")

    # Compute compression statistics on a test sample
    sample_text = text[:20_000]
    encoded_ids = tokenizer.encode(sample_text)
    num_chars = len(sample_text)
    num_tokens = len(encoded_ids)
    compression_ratio = num_chars / max(num_tokens, 1)

    # Example sentence demonstration
    example_sentence = (
        "The brave adventurer drew a glowing sword as the dragon descended upon the ancient ruins."
    )
    example_ids = tokenizer.encode(example_sentence)
    example_decoded = tokenizer.decode(example_ids)

    stats = {
        "vocab_size": tokenizer.vocab_size,
        "sample_chars": num_chars,
        "sample_tokens": num_tokens,
        "compression_ratio": compression_ratio,
        "example_sentence": example_sentence,
        "example_ids": example_ids,
        "example_decoded": example_decoded,
        "round_trip_success": example_decoded == example_sentence,
    }

    return tokenizer, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on story corpus.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus.txt"),
        help="Path to corpus text file (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tokenizer.json"),
        help="Output path for tokenizer JSON (default: artifacts/tokenizer.json)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Target vocabulary size (default: 1000)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=250_000,
        help="Maximum characters to sample for fast training (default: 250,000)",
    )
    args = parser.parse_args(argv)

    try:
        _, stats = train_and_save_tokenizer(
            corpus_path=args.corpus,
            output_path=args.output,
            vocab_size=args.vocab_size,
            max_chars=args.max_chars,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Tokenizer Training Summary ---")
    print(f"Vocab size         : {stats['vocab_size']}")
    print(f"Compression ratio  : {stats['compression_ratio']:.2f} characters per token")
    print(f"Sample compression : {stats['sample_chars']:,} chars -> {stats['sample_tokens']:,} tokens")
    print("\nExample Encode / Decode:")
    print(f"Original : {stats['example_sentence']}")
    print(f"Encoded  : {stats['example_ids']}")
    print(f"Decoded  : {stats['example_decoded']}")
    print(f"Roundtrip: {'MATCH (OK)' if stats['round_trip_success'] else 'MISMATCH (FAIL)'}")


if __name__ == "__main__":
    main()
