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
def test_milestone_2():
    import torch
    from app.core import milestone_2, BPETokenizer, GPTEmbedding

    # 1. Milestone 2 integration: train/load tokenizer and instantiate embedding module
    tokenizer, embedding = milestone_2(vocab_size=500, n_embd=48, block_size=64)
    assert isinstance(tokenizer, BPETokenizer)
    assert isinstance(embedding, GPTEmbedding)
    assert tokenizer.vocab_size >= 256
    assert embedding.n_embd == 48
    assert embedding.block_size == 64

    # 2. Test encoding and embedding forward pass
    text = "The knight walked into the dark forest."
    token_ids = tokenizer.encode(text)
    assert len(token_ids) > 0
    assert tokenizer.decode(token_ids) == text

    # Forward through embedding module with shape (B, T)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)  # (1, T)
    output = embedding(input_tensor)
    assert output.shape == (1, len(token_ids), 48)
    assert output.dtype == torch.float32


def test_tokenizer_roundtrip_all_inputs():
    from app.tokenizer import BPETokenizer

    tok = BPETokenizer()
    sample_corpus = (
        "In a land of myth and magic, heroes embark on daring quests.\n"
        "Castles stand tall against fiery dragons and ancient sorcerers."
    )
    tok.train(sample_corpus, vocab_size=320, special_tokens=["<|endoftext|>"])

    cases = [
        "",  # Empty string
        "a",  # Single character
        "A simple sentence for the adventure game.",
        "Multiline text\nwith spaces   and\ttabs\r\nand line breaks.",
        "Accented characters: café, naïve, façade, résumé, señor, Über.",
        "Emojis: ⚔️ 🛡️ 🧙‍♂️ 🐉 🏰 💎 👑",
        "Punctuation: !@#$%^&*()_+-=[]{}|;':,.<>/?`~",
        "Special token: Quest completed!<|endoftext|>",
    ]

    for text in cases:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text, f"Roundtrip failed for {text!r}: got {decoded!r}"


def test_tokenizer_save_and_load(tmp_path):
    from app.tokenizer import BPETokenizer

    tok1 = BPETokenizer()
    tok1.train("Dragons and dungeons and heroic adventures.", vocab_size=300, special_tokens=["<|endoftext|>"])

    save_path = tmp_path / "tokenizer.json"
    tok1.save(save_path)
    assert save_path.exists()

    tok2 = BPETokenizer.load(save_path)
    assert tok2.vocab_size == tok1.vocab_size
    assert tok2.merges == tok1.merges

    text = "Heroic dragons in dungeons.<|endoftext|>"
    assert tok1.encode(text) == tok2.encode(text)
    assert tok1.decode(tok1.encode(text)) == tok2.decode(tok2.encode(text))


def test_gpt_embedding_shapes_and_errors():
    import pytest
    import torch
    from app.model import GPTEmbedding

    vocab_size = 500
    n_embd = 64
    block_size = 32

    embedding = GPTEmbedding(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size)

    # Valid forward pass with batch size 3 and sequence length 16
    x = torch.randint(0, vocab_size, (3, 16))
    out = embedding(x)
    assert out.shape == (3, 16, n_embd)

    # Valid sequence length exactly equal to block_size
    x_max = torch.randint(0, vocab_size, (2, block_size))
    out_max = embedding(x_max)
    assert out_max.shape == (2, block_size, n_embd)

    # Sequence length exceeding block_size should raise ValueError
    x_overflow = torch.randint(0, vocab_size, (1, block_size + 1))
    with pytest.raises(ValueError, match="exceeds"):
        embedding(x_overflow)

    # Verify gradients flow through both token and position embeddings
    loss = out.sum()
    loss.backward()
    assert embedding.tok_emb.weight.grad is not None
    assert embedding.pos_emb.weight.grad is not None


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

