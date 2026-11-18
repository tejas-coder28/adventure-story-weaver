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
def test_milestone_3():
    import torch
    from app.core import milestone_3, GPTConfig, GPT

    config = GPTConfig(
        vocab_size=300,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = milestone_3(config)
    assert isinstance(model, GPT)
    assert model.get_num_params() > 0

    # 1. Forward pass without targets: returns (logits, None)
    B, T = 2, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None

    # 2. Forward pass with targets: returns (logits, finite scalar loss)
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar tensor
    assert torch.isfinite(loss).item()

    # 3. Verify backward pass computes gradients
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"


def test_causal_self_attention_properties():
    import pytest
    import torch
    from app.model import CausalSelfAttention, MLP, TransformerBlock

    # Verify assertion on n_embd % n_head != 0
    with pytest.raises(AssertionError, match="divisible"):
        CausalSelfAttention(n_embd=65, n_head=4)

    # Attention forward pass shape
    attn = CausalSelfAttention(n_embd=48, n_head=4, block_size=32, dropout=0.0)
    x = torch.randn(3, 12, 48)
    out = attn(x)
    assert out.shape == (3, 12, 48)

    # MLP forward pass shape
    mlp = MLP(n_embd=48, dropout=0.0)
    mlp_out = mlp(x)
    assert mlp_out.shape == (3, 12, 48)

    # TransformerBlock forward pass shape
    block = TransformerBlock(n_embd=48, n_head=4, block_size=32, dropout=0.0)
    block_out = block(x)
    assert block_out.shape == (3, 12, 48)


def test_gpt_causality():
    """Proves that changing a future token does not change the logits at earlier positions."""
    import torch
    from app.model import GPT, GPTConfig

    config = GPTConfig(
        vocab_size=200,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()  # Disable dropout to verify deterministic causality

    # Two sequences identical up to position 2; token at position 3 differs
    seq1 = torch.tensor([[15, 42, 88, 10]], dtype=torch.long)
    seq2 = torch.tensor([[15, 42, 88, 99]], dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Earlier positions (0, 1, 2) MUST produce exactly identical logits
    max_diff_earlier = (logits1[:, :3, :] - logits2[:, :3, :]).abs().max().item()
    assert max_diff_earlier < 1e-5, f"Causality violated: earlier logits changed by {max_diff_earlier}"

    # Modified position (3) MUST produce different logits
    max_diff_modified = (logits1[:, 3, :] - logits2[:, 3, :]).abs().max().item()
    assert max_diff_modified > 1e-4, "Modified position did not affect logits"


# Milestone 4: Set up the training loop and train the GPT model on the narrative data.
def test_milestone_4(tmp_path):
    import torch
    from app.core import milestone_4

    ckpt_file = tmp_path / "model.pt"
    result = milestone_4(
        checkpoint_path=str(ckpt_file),
        max_steps=4,
        batch_size=2,
    )
    assert result["model"] is not None
    assert result["best_val_loss"] > 0
    assert ckpt_file.exists(), "Checkpoint file was not created"

    # Verify checkpoint contents
    ckpt = torch.load(ckpt_file, weights_only=False)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert "tokenizer_path" in ckpt
    assert "val_loss" in ckpt


def test_overfit_tiny_batch_loss_decreases():
    """Verifies optimization loop by proving loss strictly decreases on a tiny fixed batch."""
    import torch
    from app.model import GPT, GPTConfig

    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Fixed synthetic batch
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        _, loss = model(x, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.5, (
        f"Loss did not decrease sufficiently: started at {initial_loss:.4f}, ended at {final_loss:.4f}"
    )


def test_get_batch_shifted_targets():
    """Verifies that get_batch returns x and y where y is x shifted by exactly 1 token."""
    import torch
    from app.train import get_batch

    # Sequential token stream 0, 1, 2, ..., 99
    train_data = torch.arange(100, dtype=torch.long)
    val_data = torch.arange(100, 150, dtype=torch.long)

    x, y = get_batch("train", train_data, val_data, batch_size=4, block_size=16, device=torch.device("cpu"))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)

    # For each row, y[0:T-1] should match x[1:T]
    for b in range(4):
        assert torch.equal(x[b, 1:], y[b, :-1]), "Target y is not x shifted by +1"


# Milestone 5: Implement a text generation feature that creates coherent story continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

