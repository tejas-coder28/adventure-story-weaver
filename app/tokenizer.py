"""Byte-level Byte-Pair Encoding (BPE) Tokenizer built from scratch.

Features:
- Starts from 256 individual byte values (0-255).
- Iteratively merges the most frequent adjacent pair until vocab_size is reached.
- Fast inverted-index training over word chunks.
- Exact UTF-8 round-trip guarantee: decode(encode(text)) == text for all inputs,
  including empty strings, emojis, and accented characters.
- JSON serialization: save() and load() methods preserving merge order.
- Special token support (e.g. <|endoftext|>).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple, Any
from collections import Counter, defaultdict


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer."""

    def __init__(self) -> None:
        # Base vocabulary of 256 single bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Mapping from (token_id_1, token_id_2) -> merged_token_id
        self.merges: dict[tuple[int, int], int] = {}
        # Special tokens: string -> token_id
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including bytes, merges, and special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def train(
        self,
        text: str,
        vocab_size: int = 1000,
        max_chars: int | None = 200_000,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Trains the BPE tokenizer on raw text until reaching vocab_size.

        Args:
            text: Raw training corpus text.
            vocab_size: Target vocabulary size (>= 256).
            max_chars: Optional limit on characters sampled for training speed.
            special_tokens: List of special token strings to register (e.g. ['<|endoftext|>']).
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]

        # Reset vocabulary and merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        # Pre-tokenize text into word chunks (leading space + non-space or pure space)
        # This keeps words intact and prevents cross-word runaway merges
        words = re.findall(r"\s*\S+|\s+", text)
        word_counts = Counter(words)

        # Convert words to tuples of byte integers
        vocab_words: dict[tuple[int, ...], int] = {
            tuple(w.encode("utf-8")): count for w, count in word_counts.items()
        }

        # Track pair frequencies and inverted index (pair -> set of words containing pair)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

        for word, count in vocab_words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(word)

        # Merge iteratively until target vocab_size is reached
        num_merges = vocab_size - 256
        current_id = 256

        while len(self.merges) < num_merges and pair_counts:
            # Find the most frequent adjacent pair
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
            if pair_counts[best_pair] < 2:
                # No more frequent pairs to merge
                break

            # Register new token and merge rule
            self.merges[best_pair] = current_id
            self.vocab[current_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            del pair_counts[best_pair]

            # Update words containing best_pair
            words_to_update = list(pair_to_words.pop(best_pair, set()))
            p0, p1 = best_pair

            for old_word in words_to_update:
                if old_word not in vocab_words:
                    continue
                freq = vocab_words.pop(old_word)

                # Remove pair counts from old_word
                for pair in zip(old_word, old_word[1:]):
                    if pair in pair_counts:
                        pair_counts[pair] -= freq
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(old_word)

                # Merge occurrences of best_pair in old_word
                new_word: list[int] = []
                j = 0
                nw = len(old_word)
                while j < nw:
                    if j < nw - 1 and old_word[j] == p0 and old_word[j + 1] == p1:
                        new_word.append(current_id)
                        j += 2
                    else:
                        new_word.append(old_word[j])
                        j += 1
                merged_word = tuple(new_word)
                vocab_words[merged_word] = freq

                # Add pair counts from new_word
                for pair in zip(merged_word, merged_word[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(merged_word)

            current_id += 1

        # Register special tokens if provided
        if special_tokens:
            for st in special_tokens:
                self.add_special_token(st)

    def add_special_token(self, token: str) -> int:
        """Registers a special token string and assigns it the next available ID."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        token_id = self.vocab_size
        self.special_tokens[token] = token_id
        self.inverse_special_tokens[token_id] = token
        return token_id

    def _encode_bytes(self, byte_seq: bytes) -> list[int]:
        """Encodes a single sequence of bytes using learned merges."""
        tokens = list(byte_seq)
        while len(tokens) >= 2:
            # Find the pair present in merges with the lowest rank (earliest merge)
            min_rank = float("inf")
            best_pair = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.merges.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Replace occurrences of best_pair with its merged token ID
            new_tokens: list[int] = []
            i = 0
            n = len(tokens)
            p0, p1 = best_pair
            merged_id = self.merges[best_pair]
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encodes a string into a list of token IDs.

        Guarantees round-trip fidelity when decoded.
        """
        if not text:
            return []

        if allowed_special is None:
            allowed_special = set(self.special_tokens.keys())

        # If special tokens are present and allowed, split text by special tokens
        if allowed_special:
            escaped_tokens = [re.escape(t) for t in allowed_special if t in self.special_tokens]
            if escaped_tokens:
                pattern = f"({'|'.join(escaped_tokens)})"
                parts = re.split(pattern, text)
                tokens: list[int] = []
                for part in parts:
                    if not part:
                        continue
                    if part in self.special_tokens:
                        tokens.append(self.special_tokens[part])
                    else:
                        tokens.extend(self._encode_text_segment(part))
                return tokens

        return self._encode_text_segment(text)

    def _encode_text_segment(self, text: str) -> list[int]:
        """Encodes text by splitting into word chunks and merging each chunk."""
        chunks = re.findall(r"\s*\S+|\s+", text)
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(self._encode_bytes(chunk.encode("utf-8")))
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Guarantees decode(encode(text)) == text for all inputs.
        """
        if not ids:
            return ""

        parts: list[str] = []
        byte_buffer: list[bytes] = []

        for idx in ids:
            if idx in self.inverse_special_tokens:
                if byte_buffer:
                    parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))
                    byte_buffer = []
                parts.append(self.inverse_special_tokens[idx])
            elif idx in self.vocab:
                byte_buffer.append(self.vocab[idx])
            else:
                # Unknown token fallback
                if byte_buffer:
                    parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))
                    byte_buffer = []
                parts.append(f"<unk_{idx}>")

        if byte_buffer:
            parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))

        return "".join(parts)

    def save(self, path: str | Path) -> None:
        """Saves the tokenizer vocabulary, merges, and special tokens to a JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges.keys()],
            "special_tokens": self.special_tokens,
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Loads a tokenizer from a saved JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))
        tokenizer = cls()

        # Reconstruct base vocabulary and merges
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.merges = {}

        for idx, pair in enumerate(data.get("merges", []), start=256):
            pair_tuple = (int(pair[0]), int(pair[1]))
            tokenizer.merges[pair_tuple] = idx
            tokenizer.vocab[idx] = (
                tokenizer.vocab[pair_tuple[0]] + tokenizer.vocab[pair_tuple[1]]
            )

        tokenizer.special_tokens = {
            k: int(v) for k, v in data.get("special_tokens", {}).items()
        }
        tokenizer.inverse_special_tokens = {
            v: k for k, v in tokenizer.special_tokens.items()
        }

        return tokenizer


# Friendly alias
Tokenizer = BPETokenizer
