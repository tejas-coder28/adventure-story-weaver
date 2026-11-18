"""Corpus preparation script for Adventure Story Weaver.

Processes raw Project Gutenberg plain-text files:
- Strips Gutenberg legal headers and footers (handling marker variations)
- Normalizes line endings to LF (\n)
- Collapses 3+ blank lines into 2
- Removes chapter-table-of-contents clutter safely when detected
- Merges cleaned stories into data/corpus.txt separated by a blank line
- Prints summary statistics (books processed, total characters, size in MB, preview)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Any

import httpx

# Markers commonly used in Project Gutenberg plain text files
START_MARKERS = [
    re.compile(r"\*\*\*\s*START OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^*]*\*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\*\s*START OF [^*]+\*\*\*", re.IGNORECASE),
    re.compile(r"\*END\*THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS\*", re.IGNORECASE),
]

END_MARKERS = [
    re.compile(r"\*\*\*\s*END OF (?:THE|THIS)?\s*PROJECT GUTENBERG[^*]*\*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\*\s*END OF [^*]+\*\*\*", re.IGNORECASE),
    re.compile(r"End of (?:the )?Project Gutenberg['’]?s?", re.IGNORECASE),
]

# Sample classic public domain adventure & fantasy novels
SAMPLE_BOOKS: dict[str, str] = {
    "treasure_island.txt": "https://raw.githubusercontent.com/GITenberg/Treasure-Island_120/master/120.txt",
    "the_time_machine.txt": "https://raw.githubusercontent.com/GITenberg/The-Time-Machine_35/master/35.txt",
    "the_lost_world.txt": "https://raw.githubusercontent.com/GITenberg/The-Lost-World_139/master/139.txt",
    "alice_in_wonderland.txt": "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt",
    "a_princess_of_mars.txt": "https://raw.githubusercontent.com/GITenberg/A-Princess-of-Mars_62/master/62.txt",
}


def strip_gutenberg_header_footer(text: str) -> str:
    """Strips Project Gutenberg license headers and footers from raw text."""
    # Locate header end
    start_pos = 0
    for marker in START_MARKERS:
        match = marker.search(text)
        if match and match.end() > start_pos:
            start_pos = match.end()

    body = text[start_pos:]

    # Locate footer start: take earliest matching footer marker
    earliest_end = len(body)
    for marker in END_MARKERS:
        match = marker.search(body)
        if match and match.start() < earliest_end:
            earliest_end = match.start()

    body = body[:earliest_end]

    # Additional safety: filter any residual lines mentioning Project Gutenberg
    lines = [
        line for line in body.split("\n")
        if "project gutenberg" not in line.lower()
    ]
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    """Normalizes line endings and collapses 3+ blank lines into 2."""
    # Normalize Windows (\r\n) and legacy Mac (\r) line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ blank lines into 2.
    # 1 blank line = 2 newlines (\n\n)
    # 2 blank lines = 3 newlines (\n\n\n)
    # 3+ blank lines = 4+ newlines (\n{4,}) -> collapsed into 3 newlines (\n\n\n)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def remove_toc_clutter(text: str) -> str:
    """Safely removes self-contained table-of-contents sections if clearly delimited."""
    # Look for a standalone CONTENTS block before Chapter 1 / Chapter I / Part 1
    toc_pattern = re.compile(
        r"(?:\n\s*CONTENTS\s*\n|\n\s*TABLE OF CONTENTS\s*\n)(.*?)(?=\n\s*(?:CHAPTER|PART|BOOK)\s+(?:I|1|ONE)\b)",
        re.IGNORECASE | re.DOTALL,
    )
    # Only remove if the TOC block is reasonably short (< 5000 chars) to prevent accidental text loss
    match = toc_pattern.search(text)
    if match and len(match.group(1)) < 5000:
        text = text[:match.start()] + "\n\n" + text[match.end():]
    return text


def clean_gutenberg_text(text: str) -> str:
    """Performs the full cleaning pipeline on a raw Gutenberg book string."""
    body = strip_gutenberg_header_footer(text)
    body = remove_toc_clutter(body)
    return normalize_text(body)


def download_sample_books(raw_dir: Path) -> list[Path]:
    """Downloads curated public-domain adventure stories into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    print(f"Downloading {len(SAMPLE_BOOKS)} classic adventure stories into '{raw_dir}'...")
    for filename, url in SAMPLE_BOOKS.items():
        dest = raw_dir / filename
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"  Fetching {filename}...")
            r = httpx.get(url, timeout=30.0, follow_redirects=True)
            r.raise_for_status()
            dest.write_text(r.text, encoding="utf-8")
        else:
            print(f"  Already exists: {filename}")
        downloaded.append(dest)
    print("Done downloading sample books.")
    return downloaded


def build_corpus(
    raw_dir: Path = Path("data/raw"),
    output_file: Path = Path("data/corpus.txt"),
) -> dict[str, Any]:
    """Reads all .txt files in raw_dir, cleans them, and writes the joined corpus."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"The directory '{raw_dir}' does not exist.\n"
            f"Please create '{raw_dir}' and place your Project Gutenberg .txt files in it,\n"
            f"or run: python scripts/build_corpus.py --download-sample"
        )

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in '{raw_dir}'.\n"
            f"Please add Project Gutenberg plain-text files to '{raw_dir}',\n"
            f"or run: python scripts/build_corpus.py --download-sample"
        )

    cleaned_books: list[str] = []
    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        cleaned = clean_gutenberg_text(content)
        if cleaned:
            cleaned_books.append(cleaned)

    # Join all books, separated by a blank line
    corpus_text = "\n\n".join(cleaned_books).strip() + "\n"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(corpus_text, encoding="utf-8")

    total_chars = len(corpus_text)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    preview = corpus_text[:300]

    return {
        "num_books": len(cleaned_books),
        "total_chars": total_chars,
        "size_mb": size_mb,
        "preview": preview,
        "output_path": output_file,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble clean text corpus from Project Gutenberg stories.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory containing raw .txt files (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus.txt"),
        help="Path to output corpus text file (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Automatically download sample public domain adventure stories if raw directory is empty",
    )
    args = parser.parse_args(argv)

    if args.download_sample or (not args.raw_dir.exists()) or (not list(args.raw_dir.glob("*.txt"))):
        if args.download_sample or not args.raw_dir.exists() or not list(args.raw_dir.glob("*.txt")):
            if not args.download_sample:
                print(f"Note: '{args.raw_dir}' is empty or missing.")
                print(f"Run with --download-sample to download standard public domain adventure stories,")
                print(f"or place your own .txt files into '{args.raw_dir}'.")
                sys.exit(1)
            download_sample_books(args.raw_dir)

    try:
        stats = build_corpus(raw_dir=args.raw_dir, output_file=args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Corpus Build Summary ---")
    print(f"Books processed : {stats['num_books']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Corpus file size: {stats['size_mb']:.2f} MB")
    print("\nFirst 300 characters of corpus:")
    print("-" * 50)
    print(stats["preview"])
    print("-" * 50)


if __name__ == "__main__":
    main()
