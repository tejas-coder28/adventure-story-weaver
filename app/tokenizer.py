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
from collections import Counter, defaultdict


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer."""

    def __init__(self) -> None:
        # Base vocabulary of 256 single bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Merges stored in chronological order: (p0, p1) -> new_id
        self.merges: dict[tuple[int, int], int] = {}
        # Special tokens: token_str -> token_id
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size including base bytes, merges, and special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def register_special_tokens(self, tokens: list[str]) -> None:
        """Registers custom special tokens (e.g. '<|endoftext|>') with IDs beyond the merges."""
        for token_str in tokens:
            if token_str not in self.special_tokens:
                new_id = self.vocab_size
                self.special_tokens[token_str] = new_id
                self.inverse_special_tokens[new_id] = token_str

    def train(
        self,
        text: str,
        vocab_size: int = 1000,
        max_chars: int = 250_000,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Trains the BPE tokenizer on input text up to the specified vocab_size.

        Args:
            text: Raw training corpus string.
            vocab_size: Total vocabulary size to reach (>= 256).
            max_chars: Maximum characters sampled for fast training.
            special_tokens: List of special token strings to register (e.g. ['<|endoftext|>']).
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

        num_special = len(special_tokens) if special_tokens else 0
        target_merges = max(0, vocab_size - 256 - num_special)

        # Reset vocabulary and merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        if target_merges == 0:
            if special_tokens:
                self.register_special_tokens(special_tokens)
            return

        # Sample text if corpus is large to keep training fast
        sample_text = text[:max_chars] if len(text) > max_chars else text
        words = re.findall(r"\S+|\s+", sample_text)
        word_counts = Counter(words)

        # Convert words to tuples of byte integers
        vocab_words: dict[tuple[int, ...], int] = {
            tuple(w.encode("utf-8")): count for w, count in word_counts.items()
        }

        # Track pair frequencies and inverted index (pair -> set of words containing pair)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

        for word, count in vocab_words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(word)

        # Iteratively merge most frequent adjacent pairs
        next_id = 256
        for _ in range(target_merges):
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if pair_counts[best_pair] <= 0:
                break

            # Register merge
            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            next_id += 1

            # Update words containing best_pair
            words_to_update = list(pair_to_words.pop(best_pair, set()))
            del pair_counts[best_pair]

            for word in words_to_update:
                count = vocab_words.pop(word)

                # Decrement old pair counts in this word
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    if pair != best_pair:
                        pair_counts[pair] -= count
                        if pair_counts[pair] <= 0 and pair in pair_counts:
                            del pair_counts[pair]
                        if word in pair_to_words[pair]:
                            pair_to_words[pair].remove(word)

                # Merge occurrences of best_pair in the word
                new_word: list[int] = []
                i = 0
                w_len = len(word)
                while i < w_len:
                    if i < w_len - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                merged_word = tuple(new_word)
                vocab_words[merged_word] = vocab_words.get(merged_word, 0) + count

                # Increment new pair counts
                for j in range(len(merged_word) - 1):
                    pair = (merged_word[j], merged_word[j + 1])
                    pair_counts[pair] += count
                    pair_to_words[pair].add(merged_word)

        if special_tokens:
            self.register_special_tokens(special_tokens)

    def _get_stats(self, ids: list[int]) -> Counter[tuple[int, int]]:
        """Counts frequencies of adjacent pairs in an ID list."""
        return Counter(zip(ids, ids[1:]))

    def _merge_pair(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replaces all occurrences of pair with new_id in ids."""
        res: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                res.append(new_id)
                i += 2
            else:
                res.append(ids[i])
                i += 1
        return res

    def _encode_bytes(self, byte_seq: bytes) -> list[int]:
        """Encodes raw bytes by applying merges in priority order."""
        tokens = list(byte_seq)
        if len(tokens) < 2:
            return tokens

        while len(tokens) >= 2:
            # Find candidate pairs present in our merges
            pairs = zip(tokens, tokens[1:])
            # Pick the merge with the smallest index (earliest learned = highest priority)
            mergeable = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not mergeable:
                break

            _, best_pair = min(mergeable, key=lambda x: x[0])
            new_id = self.merges[best_pair]

            new_tokens: list[int] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encodes a string into a list of token IDs.

        Guarantees lossless round-trip for any Unicode string.
        Handles special tokens when allowed.
        """
        if not text:
            return []

        if allowed_special is None:
            allowed_special = set(self.special_tokens.keys())

        if not self.special_tokens or not any(s in text for s in allowed_special):
            return self._encode_text_segment(text)

        # Build regex to split on allowed special tokens
        pattern = "(" + "|".join(re.escape(s) for s in allowed_special if s in self.special_tokens) + ")"
        parts = re.split(pattern, text)
        tokens: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens and part in allowed_special:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_text_segment(part))
        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        """Encodes a regular text segment by splitting on words/spaces."""
        words = re.findall(r"\S+|\s+", text)
        tokens: list[int] = []
        for word in words:
            raw_bytes = word.encode("utf-8")
            tokens.extend(self._encode_bytes(raw_bytes))
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Guarantees decode(encode(text)) == text for all inputs.
        """
        if not ids:
            return ""

        parts: list[str] = []
        byte_buffer: list[bytes] = []

        def flush_buffer():
            if byte_buffer:
                all_bytes = b"".join(byte_buffer)
                parts.append(all_bytes.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            if token_id in self.inverse_special_tokens:
                flush_buffer()
                parts.append(self.inverse_special_tokens[token_id])
            elif token_id in self.vocab:
                byte_buffer.append(self.vocab[token_id])
            else:
                flush_buffer()
                parts.append("")

        flush_buffer()
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        """Saves vocabulary, merges, and special tokens to JSON."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "merges": [list(pair) for pair in self.merges.keys()],
            "special_tokens": self.special_tokens,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Loads vocabulary, merges, and special tokens from JSON."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.merges = {}

        next_id = 256
        for pair in data.get("merges", []):
            pair_tuple = (int(pair[0]), int(pair[1]))
            tokenizer.merges[pair_tuple] = next_id
            tokenizer.vocab[next_id] = (
                tokenizer.vocab[pair_tuple[0]] + tokenizer.vocab[pair_tuple[1]]
            )
            next_id += 1

        for token_str, token_id in data.get("special_tokens", {}).items():
            tokenizer.special_tokens[token_str] = int(token_id)
            tokenizer.inverse_special_tokens[int(token_id)] = token_str

        return tokenizer


# Friendly alias
Tokenizer = BPETokenizer
