"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adventure-story-weaver", description="Game developers build a text adventure story segment generator, where a custom GPT, trained on existing adventure game narratives, creates dynamic story continuations.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    
    # Milestone 2: Train BPE Tokenizer
    tok_parser = sub.add_parser("train-tokenizer", help="train BPE tokenizer on corpus and save to artifacts/tokenizer.json")
    tok_parser.add_argument("--corpus", default="data/corpus.txt", help="path to corpus text")
    tok_parser.add_argument("--output", default="artifacts/tokenizer.json", help="path to save tokenizer JSON")
    tok_parser.add_argument("--vocab-size", type=int, default=1000, help="target vocabulary size")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Adventure Story Weaver is ready. Open README.md for your milestones.")
    elif args.command == "train-tokenizer":
        from pathlib import Path
        from scripts.train_tokenizer import train_and_save_tokenizer
        _, stats = train_and_save_tokenizer(
            corpus_path=Path(args.corpus),
            output_path=Path(args.output),
            vocab_size=args.vocab_size,
        )
        print(f"Trained tokenizer with vocab size {stats['vocab_size']}, compression ratio {stats['compression_ratio']:.2f} chars/token.")


if __name__ == "__main__":
    main()
