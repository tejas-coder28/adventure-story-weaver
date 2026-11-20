"""Training loop and data pipeline for Adventure Story Weaver GPT model.

Features:
- Encodes full corpus into token tensors with 90/10 train/val split.
- get_batch(split) samples random windows of block_size with target shifted by +1.
- Optimizer: AdamW with weight decay.
- Learning rate schedule: Linear warmup followed by cosine decay down to min_lr.
- Gradient clipping (max_norm=1.0) to stabilize attention gradients.
- Periodic evaluation on train and val splits with model.eval() and torch.no_grad().
- Best checkpoint saving (model state, config, tokenizer path, step, val_loss) to checkpoints/model.pt.
- Automatic device selection (cuda, mps, cpu).
- CLI arguments for max_steps, batch_size, learning_rate, seed, and network dimensions.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tokenizer import BPETokenizer
from app.model import GPT, GPTConfig


def get_device() -> torch.device:
    """Selects available compute accelerator (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(
    corpus_path: str | Path = "data/corpus.txt",
    tokenizer_path: str | Path = "artifacts/tokenizer.json",
    train_split_ratio: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, BPETokenizer]:
    """Loads corpus text, encodes with tokenizer, and splits into train and validation tensors."""
    corpus_p = Path(corpus_path)
    tok_p = Path(tokenizer_path)

    if not corpus_p.exists():
        raise FileNotFoundError(
            f"Corpus not found at '{corpus_p}'. Run scripts/build_corpus.py first."
        )

    if tok_p.exists():
        tokenizer = BPETokenizer.load(tok_p)
    else:
        # If tokenizer file missing, train a fallback tokenizer
        tokenizer = BPETokenizer()
        text = corpus_p.read_text(encoding="utf-8")
        tokenizer.train(text, vocab_size=1000, special_tokens=["<|endoftext|>"])

    text = corpus_p.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    data_tensor = torch.tensor(token_ids, dtype=torch.long)

    # 90% train / 10% validation split
    split_idx = int(len(data_tensor) * train_split_ratio)
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]

    return train_data, val_data, tokenizer


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples random context windows of length block_size, with y shifted by +1."""
    data = train_data if split == "train" else val_data
    if len(data) <= block_size:
        raise ValueError(f"Dataset length ({len(data)}) must be greater than block_size ({block_size})")

    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Calculates learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        # Linear warmup phase
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        # After max_steps, clamp to minimum learning rate
        return min_lr
    # Cosine decay phase
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    eval_iters: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimates average cross-entropy loss over multiple batches for both splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    corpus_path: str | Path = "data/corpus.txt",
    tokenizer_path: str | Path = "artifacts/tokenizer.json",
    checkpoint_path: str | Path = "checkpoints/model.pt",
    max_steps: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-4,
    warmup_steps: int = 20,
    weight_decay: float = 0.01,
    eval_interval: int = 50,
    eval_iters: int = 10,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.1,
    seed: int = 1337,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Executes the training loop and saves the best model checkpoint."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    # Load and encode data
    train_data, val_data, tokenizer = load_dataset(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        train_split_ratio=0.9,
    )
    print(f"Dataset loaded: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # Instantiate model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config).to(device)
    print(f"Model instantiated with {model.get_num_params():,} parameters.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.99),
    )

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    print(f"Starting training run: {max_steps} steps, batch_size={batch_size}, lr={learning_rate}...")

    for step in range(max_steps):
        # Determine learning rate for this step
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate periodically
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(
                model=model,
                eval_iters=eval_iters,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )
            print(
                f"step {step:4d}/{max_steps} | lr {lr:.6f} | "
                f"train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )
            history.append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
            })

            # Checkpoint saving for best validation loss
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt_data = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "tokenizer_path": str(tokenizer_path),
                    "val_loss": best_val_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt_data, checkpoint_path)
                print(f"Saved new best checkpoint to '{checkpoint_path}' (val_loss: {best_val_loss:.4f})")

        # Forward pass on training batch
        model.train()
        x, y = get_batch("train", train_data, val_data, batch_size, block_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return {
        "model": model,
        "config": config,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "best_val_loss": best_val_loss,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train GPT model on story corpus.")
    parser.add_argument("--corpus", default="data/corpus.txt", help="Path to text corpus")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt", help="Path to output checkpoint")
    parser.add_argument("--max_steps", type=int, default=200, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--block_size", type=int, default=128, help="Context sequence length")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    args = parser.parse_args(argv)

    train(
        corpus_path=args.corpus,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )


if __name__ == "__main__":
    main()
