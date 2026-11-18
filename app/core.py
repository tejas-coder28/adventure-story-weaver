"""Your code. One stub per milestone; build them in order and keep each one small."""
from pathlib import Path
from scripts.build_corpus import (
    strip_gutenberg_header_footer,
    normalize_text,
    clean_gutenberg_text,
    build_corpus,
)


from app.tokenizer import BPETokenizer, Tokenizer
from app.model import (
    GPTConfig,
    GPTEmbedding,
    GPTEmbeddings,
    TransformerEmbedding,
    EmbeddingLayer,
    StoryEmbedding,
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    GPT,
)
from app.train import train, get_batch, estimate_loss, load_dataset
from app.generate import generate, load_checkpoint, validate_generation_params


# --- Milestone 1: Assemble a small corpus of text from public domain adventure stories or fantasy novels.
# Hint: Curating your raw text corpus.
def milestone_1(corpus_path: str = "data/corpus.txt") -> str:
    """Milestone 1: Read and return the curated adventure stories corpus.

    Ensures the corpus exists, is non-empty, and does not contain Project Gutenberg license headers.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at '{path}'. Run python scripts/build_corpus.py to generate it."
        )
    return path.read_text(encoding="utf-8")


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the story text.
# Hint: Byte-pair tokenization and embedding layers.
def milestone_2(
    corpus_path: str = "data/corpus.txt",
    vocab_size: int = 1000,
    n_embd: int = 64,
    block_size: int = 128,
) -> tuple[BPETokenizer, GPTEmbedding]:
    """Milestone 2: Train a byte-pair tokenizer and generate embeddings from the story text.

    Returns:
        tuple of (tokenizer, embedding_module)
    """
    tok_path = Path("artifacts/tokenizer.json")
    if tok_path.exists():
        tokenizer = BPETokenizer.load(tok_path)
    else:
        path = Path(corpus_path)
        tokenizer = BPETokenizer()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            tokenizer.train(
                text,
                vocab_size=vocab_size,
                max_chars=20_000,
                special_tokens=["<|endoftext|>"],
            )
        else:
            tokenizer.train(
                "An adventurous journey across enchanted lands and perilous dungeons.",
                vocab_size=min(vocab_size, 300),
                special_tokens=["<|endoftext|>"],
            )

    embedding = GPTEmbedding(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
    )
    return tokenizer, embedding


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal self-attention and transformer block construction.
def milestone_3(config: GPTConfig | None = None) -> GPT:
    """Milestone 3: Build and return the initialized GPT model with causal self-attention.

    Returns:
        An initialized GPT model ready for training or evaluation.
    """
    if config is None:
        config = GPTConfig(
            vocab_size=1000,
            block_size=128,
            n_layer=4,
            n_head=4,
            n_embd=64,
            dropout=0.1,
        )
    return GPT(config)


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
# Hint: Training loop and gradient descent.
def milestone_4(
    corpus_path: str = "data/corpus.txt",
    tokenizer_path: str = "artifacts/tokenizer.json",
    checkpoint_path: str = "checkpoints/model.pt",
    max_steps: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
) -> dict:
    """Milestone 4: Run training loop and train the GPT model on narrative data.

    Returns:
        Dictionary containing trained model, config, history, and checkpoint path.
    """
    return train(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        checkpoint_path=checkpoint_path,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_interval=max_steps,
        eval_iters=2,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
# Hint: Text generation with sampling strategies.
def milestone_5(
    prompt: str = "The adventurer stepped into the dark cavern,",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    seed: int | None = 42,
    checkpoint_path: str = "checkpoints/model.pt",
) -> str:
    """Milestone 5: Generate a story continuation using trained GPT checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        model, tokenizer = load_checkpoint(ckpt_file)
    else:
        tokenizer, _ = milestone_2()
        config = GPTConfig(vocab_size=tokenizer.vocab_size)
        model = GPT(config)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# Hint: Building an interface for text generation.
def milestone_6(
    prompt: str = "You stand at the mouth of a dark cave",
    max_tokens: int = 50,
    temperature: float = 0.8,
    checkpoint: str = "checkpoints/model.pt",
) -> str:
    """Milestone 6: CLI story continuation interface."""
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        model, tokenizer = load_checkpoint(ckpt_path)
    else:
        tokenizer, _ = milestone_2()
        config = GPTConfig(vocab_size=tokenizer.vocab_size)
        model = GPT(config)

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
