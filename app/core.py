"""Your code. One stub per milestone; build them in order and keep each one small."""
from pathlib import Path
from scripts.build_corpus import (
    strip_gutenberg_header_footer,
    normalize_text,
    clean_gutenberg_text,
    build_corpus,
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
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


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
