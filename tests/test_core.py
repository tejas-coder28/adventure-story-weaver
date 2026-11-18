"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
def test_milestone_1():
    from pathlib import Path
    from app.core import (
        milestone_1,
        strip_gutenberg_header_footer,
        normalize_text,
        clean_gutenberg_text,
    )

    # 1. Unit tests for cleaner functions
    sample_raw = (
        "Project Gutenberg preamble\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURE ***\n\n"
        "Chapter 1\n\n\n\n\nOnce upon a time in a distant kingdom.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURE ***\n"
        "License text and donation details"
    )
    cleaned = clean_gutenberg_text(sample_raw)
    assert "preamble" not in cleaned
    assert "License text" not in cleaned
    assert "START OF" not in cleaned
    assert "END OF" not in cleaned
    assert "Chapter 1" in cleaned
    assert "\r" not in cleaned
    # Ensure 4+ newlines collapsed to 3 newlines (2 blank lines)
    assert "\n\n\n\n" not in cleaned
    assert "\n\n\nOnce upon a time" in cleaned

    # 2. Check generated corpus file
    corpus_path = Path("data/corpus.txt")
    assert corpus_path.exists(), "data/corpus.txt does not exist. Run scripts/build_corpus.py first."
    
    text = milestone_1("data/corpus.txt")
    assert len(text) > 0, "Corpus text is empty"
    assert "project gutenberg" not in text.lower(), "Corpus contains Project Gutenberg license text"
    
    # Check corpus size (expected between 1 MB and 5 MB)
    size_mb = corpus_path.stat().st_size / (1024 * 1024)
    assert 0.5 <= size_mb <= 5.0, f"Corpus size {size_mb:.2f} MB is outside expected range"


# Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a text generation feature that creates coherent story continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

