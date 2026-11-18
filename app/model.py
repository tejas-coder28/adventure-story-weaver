"""Transformer architecture modules built from scratch in PyTorch for Adventure Story Weaver.

Provides:
- GPTEmbedding: combines learned token embeddings and learned positional embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    """Combines token identity embeddings and learned positional embeddings.

    For an input tensor of token IDs with shape (B, T):
    - Maps each token ID to an embedding vector of dimension n_embd.
    - Generates position indices [0, 1, ..., T-1] and maps them to positional embedding vectors.
    - Adds token and position embeddings element-wise: output = tok_emb + pos_emb.
    - Applies optional dropout.
    - Raises ValueError if sequence length T exceeds block_size.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the embedding module.

        Args:
            vocab_size: Size of the tokenizer vocabulary.
            n_embd: Dimensionality of the embedding space.
            block_size: Maximum allowable context length (sequence length).
            dropout: Dropout probability applied to the combined embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Token embedding lookup table: maps each token id to an n_embd vector
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # Learned positional embedding lookup table: maps position [0..block_size-1] to n_embd vector
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # Dropout layer for regularization
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding module.

        Args:
            idx: Tensor of token IDs with shape (B, T) where B is batch size and T is sequence length.

        Returns:
            Tensor of shape (B, T, n_embd) representing the combined embeddings.

        Raises:
            ValueError: If sequence length T exceeds block_size.
        """
        if idx.dim() != 2:
            raise ValueError(f"Expected 2D input tensor of shape (B, T), got shape {idx.shape}")

        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(
                f"Sequence length T={t} exceeds maximum block_size={self.block_size}"
            )

        # Positions tensor: [0, 1, 2, ..., t-1] on the same device as the input
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        # Token embeddings: shape (B, T, n_embd)
        tok_embeddings = self.tok_emb(idx)

        # Positional embeddings: shape (T, n_embd), broadcasted to (B, T, n_embd)
        pos_embeddings = self.pos_emb(pos)

        # Element-wise addition and dropout
        x = self.drop(tok_embeddings + pos_embeddings)
        return x


# Aliases for architectural clarity and test compatibility
GPTEmbeddings = GPTEmbedding
TransformerEmbedding = GPTEmbedding
EmbeddingLayer = GPTEmbedding
StoryEmbedding = GPTEmbedding
