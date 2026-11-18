"""Text generation and sampling engine for Adventure Story Weaver.

Features:
- generate(): Autoregressive generation with temperature, top-k, and top-p (nucleus) sampling.
- Deterministic reproducibility using torch.Generator with seed.
- Greedy decoding fallback when temperature == 0 or top_k == 1.
- Safe context window cropping to model block_size.
- load_checkpoint(): Reconstructs model and tokenizer from saved checkpoint dictionary.
- Comprehensive parameter validation.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tokenizer import BPETokenizer
from app.model import GPT, GPTConfig


def validate_generation_params(
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> None:
    """Validates generation hyperparameters and raises informative ValueErrors."""
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be between 0 and 1 (0 < top_p <= 1.0), got {top_p}")


def load_checkpoint(
    checkpoint_path: str | Path = "checkpoints/model.pt",
    device: torch.device | None = None,
) -> tuple[GPT, BPETokenizer]:
    """Loads model weights, configuration, and tokenizer from checkpoint file."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{ckpt_path}' not found. Run training first."
        )

    if device is None:
        from app.train import get_device
        device = get_device()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct tokenizer
    tok_path = Path(checkpoint.get("tokenizer_path", "artifacts/tokenizer.json"))
    if not tok_path.exists():
        tok_path = Path("artifacts/tokenizer.json")

    tokenizer = BPETokenizer.load(tok_path)
    return model, tokenizer


def generate_ids(
    model: nn.Module,
    tokenizer: BPETokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> list[int]:
    """Generates autoregressive token IDs given a prompt string.

    Args:
        model: GPT model.
        tokenizer: BPETokenizer for encoding prompt.
        prompt: Initial text string.
        max_new_tokens: Number of subsequent tokens to generate.
        temperature: Sampling randomness (0.0 = greedy argmax, > 0 = random sampling).
        top_k: Truncate sampling distribution to top K most probable tokens.
        top_p: Nucleus sampling cumulative probability threshold.
        seed: Random seed for reproducible output via torch.Generator.
        device: Device on which model is running.

    Returns:
        List of integer token IDs (prompt tokens + newly generated tokens).
    """
    validate_generation_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Set up generator for reproducible sampling
    generator: torch.Generator | None = None
    if seed is not None:
        gen_device = "cpu" if device.type == "mps" else device.type
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(seed)

    # Encode prompt (handle empty prompt gracefully)
    if prompt:
        prompt_ids = tokenizer.encode(prompt)
    else:
        # If prompt is empty, start from endoftext token or 0
        special_id = tokenizer.special_tokens.get("<|endoftext|>")
        prompt_ids = [special_id] if special_id is not None else [0]

    if max_new_tokens == 0:
        return prompt_ids if prompt else []

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Retrieve context block_size
    block_size = getattr(model, "config", None).block_size if hasattr(model, "config") else 128

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 1. Crop context to last block_size tokens to prevent overflow
                idx_crop = idx[:, -block_size:]

                # 2. Forward pass to get logits for next token
                logits, _ = model(idx_crop)

                # 3. Focus only on logits at the last position: (B, vocab_size)
                logits = logits[:, -1, :]

                # 4. Greedy decoding if temperature is 0 or top_k == 1
                if temperature == 0.0 or top_k == 1:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # Apply temperature scaling
                    scaled_logits = logits / temperature

                    # Optional top-k filtering
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits = scaled_logits.clone()
                        scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")

                    # Optional top-p (nucleus) filtering
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Mask tokens exceeding cumulative probability
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift right to keep the first token above threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        scaled_logits = scaled_logits.clone()
                        scaled_logits[indices_to_remove] = float("-inf")

                    # Convert to probability distribution
                    probs = F.softmax(scaled_logits, dim=-1)

                    # Sample next token
                    if generator is not None:
                        idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
                    else:
                        idx_next = torch.multinomial(probs, num_samples=1)

                # Append sampled token to sequence
                idx = torch.cat((idx, idx_next), dim=1)
    finally:
        model.train(was_training)

    all_tokens = idx[0].tolist()

    # If prompt was empty, drop the synthetic start token if present
    if not prompt and prompt_ids:
        if all_tokens and all_tokens[0] == prompt_ids[0]:
            all_tokens = all_tokens[1:]

    return all_tokens


def generate(
    model: nn.Module,
    tokenizer: BPETokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    return_ids: bool = False,
) -> str | list[int]:
    """Generates autoregressive story continuations given a prompt string.

    Args:
        model: GPT model.
        tokenizer: BPETokenizer for encoding prompt and decoding generated tokens.
        prompt: Initial text string.
        max_new_tokens: Number of subsequent tokens to generate.
        temperature: Sampling randomness (0.0 = greedy argmax, > 0 = random sampling).
        top_k: Truncate sampling distribution to top K most probable tokens.
        top_p: Nucleus sampling cumulative probability threshold.
        seed: Random seed for reproducible output via torch.Generator.
        device: Device on which model is running.
        return_ids: If True, returns list of integer token IDs instead of decoded text.

    Returns:
        The prompt string combined with the generated continuation text, or token IDs.
    """
    token_ids = generate_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        device=device,
    )
    if return_ids:
        return token_ids
    return tokenizer.decode(token_ids)
