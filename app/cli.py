"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adventure-story-weaver", description="Game developers build a text adventure story segment generator, where a custom GPT, trained on existing adventure game narratives, creates dynamic story continuations.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Adventure Story Weaver is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
