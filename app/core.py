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
    GPTEmbedding,
    GPTEmbeddings,
    TransformerEmbedding,
    EmbeddingLayer,
    StoryEmbedding,
)


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
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Set up the training loop and train the GPT model on the narrative data.
# Hint: Training loop and gradient descent.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a text generation feature that creates coherent story continuations.
# Hint: Text generation with sampling strategies.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Design a CLI where users can input a story prompt and receive generated segments.
# Hint: Building an interface for text generation.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
