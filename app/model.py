"""Transformer architecture modules built from scratch in PyTorch for Adventure Story Weaver.

Provides:
- GPTConfig: Configuration dataclass with small, laptop-friendly defaults.
- GPTEmbedding: Combines learned token embeddings and learned positional embeddings.
- CausalSelfAttention: Multi-head causal self-attention mechanism with explicit Q, K, V projections and causal mask.
- MLP: Position-wise feed-forward network (Linear -> GELU -> Linear -> Dropout).
- TransformerBlock: Pre-LayerNorm transformer decoder block with residual connections.
- GPT: Full autoregressive language model with weight tying and cross-entropy loss computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters for the custom GPT model with laptop-friendly defaults."""

    vocab_size: int = 1000
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1


class GPTEmbedding(nn.Module):
    """Combines token identity embeddings and learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.dim() != 2:
            raise ValueError(f"Expected 2D input tensor of shape (B, T), got shape {idx.shape}")

        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(
                f"Sequence length T={t} exceeds maximum block_size={self.block_size}"
            )

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_embeddings = self.tok_emb(idx)
        pos_embeddings = self.pos_emb(pos)
        return self.drop(tok_embeddings + pos_embeddings)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention built strictly from scratch.

    Computes:
        scores = (Q @ K.T) / sqrt(head_dim)
        scores = scores.masked_fill(causal_mask == 0, -inf)
        attention_weights = softmax(scores)
        output = attention_weights @ V
    """

    def __init__(
        self,
        config: GPTConfig | None = None,
        n_embd: int | None = None,
        n_head: int | None = None,
        block_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Allow instantiation from either a GPTConfig or direct arguments
        if config is not None:
            self.n_embd = config.n_embd
            self.n_head = config.n_head
            self.block_size = config.block_size
            self.dropout = config.dropout
        else:
            self.n_embd = n_embd or 64
            self.n_head = n_head or 4
            self.block_size = block_size
            self.dropout = dropout

        assert (
            self.n_embd % self.n_head == 0
        ), f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        self.head_dim = self.n_embd // self.n_head

        # Explicit Query, Key, Value projections
        self.q_proj = nn.Linear(self.n_embd, self.n_embd)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd)

        # Output projection
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Causal mask registered buffer: lower-triangular matrix ensuring tokens only attend to past
        causal_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(
            1, 1, self.block_size, self.block_size
        )
        self.register_buffer("bias", causal_mask)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        B, T, C = x.shape

        # 1. Linear projections and reshape to (B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Scaled dot-product: QK^T / sqrt(head_dim) -> (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Apply causal mask (-inf for future positions so softmax sets their weight to 0)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 4. Softmax over the last dimension (keys)
        att = F.softmax(scores, dim=-1)
        att = self.attn_dropout(att)

        # 5. Weighted combination with values V: (B, n_head, T, head_dim)
        y = att @ v

        # 6. Reassemble heads: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Output projection and residual dropout
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise Feed-Forward Network: Linear -> GELU -> Linear -> Dropout."""

    def __init__(
        self,
        config: GPTConfig | None = None,
        n_embd: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dim = config.n_embd if config is not None else (n_embd or 64)
        drop_p = config.dropout if config is not None else dropout

        # Expanding projection: 4 * n_embd
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        # Contracting projection: back to n_embd
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Pre-LayerNorm architecture.

    x = x + attn(ln1(x))
    x = x + mlp(ln2(x))
    """

    def __init__(
        self,
        config: GPTConfig | None = None,
        n_embd: int | None = None,
        n_head: int | None = None,
        block_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if config is not None:
            self.n_embd = config.n_embd
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config=config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config=config)
        else:
            dim = n_embd or 64
            self.n_embd = dim
            self.ln_1 = nn.LayerNorm(dim)
            self.attn = CausalSelfAttention(
                n_embd=dim, n_head=n_head or 4, block_size=block_size, dropout=dropout
            )
            self.ln_2 = nn.LayerNorm(dim)
            self.mlp = MLP(n_embd=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm attention with residual skip connection
        x = x + self.attn(self.ln_1(x))
        # Pre-LayerNorm MLP with residual skip connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Full Autoregressive Generative Pretrained Transformer for Adventure Story Weaver."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.embeddings = GPTEmbedding(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            block_size=config.block_size,
            dropout=config.dropout,
        )

        # N Transformer Blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.n_layer)]
        )

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Final Language Modeling Head: projects n_embd -> vocab_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between token embedding and LM head
        self.lm_head.weight = self.embeddings.tok_emb.weight

        # Initialize network weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using standard small Gaussian distribution (mean=0, std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns the total number of parameters in the model.

        Args:
            non_embedding: If True, excludes learned positional embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.pos_emb.weight.numel()
        return n_params

    def count_parameters(self) -> int:
        """Convenience alias for total parameter count."""
        return self.get_num_params(non_embedding=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for training and inference.

        Args:
            idx: Tensor of token IDs with shape (B, T).
            targets: Optional ground-truth target token IDs with shape (B, T).

        Returns:
            logits: Predicted next-token logits with shape (B, T, vocab_size).
            loss: Cross-entropy scalar loss tensor if targets are provided, else None.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds maximum block_size={self.config.block_size}"
            )

        # 1. Embedding lookup (token + position)
        x = self.embeddings(idx)  # (B, T, n_embd)

        # 2. Sequential transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final layer normalization
        x = self.ln_f(x)

        # 4. Compute unnormalized logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        # 5. Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten predictions and targets to compute cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss


# Aliases for architectural clarity and test compatibility
GPTEmbeddings = GPTEmbedding
TransformerEmbedding = GPTEmbedding
EmbeddingLayer = GPTEmbedding
StoryEmbedding = GPTEmbedding
