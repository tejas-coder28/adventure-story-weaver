# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ksmrgqhbc8/badge.svg)](https://ask.thili.ai/b/ksmrgqhbc8)

**Adventure Story Weaver** is an autoregressive Generative Pretrained Transformer (GPT) language model built completely **from scratch in PyTorch** (no Hugging Face `transformers`, no `tiktoken`, and no high-level `nn.Transformer` or `nn.MultiheadAttention` modules).

Trained on classic public-domain fantasy and adventure narratives from Project Gutenberg, the model learns narrative pacing, vocabulary, and descriptive storytelling patterns to generate interactive story continuations for text-adventure games.

---

## Completed Milestones

- [x] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      *Curated 1.48 MB of public-domain adventure literature free of Project Gutenberg license boilerplate.*
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      *Built a byte-level BPE tokenizer with lossless UTF-8 round-trip guarantees and learned token + positional embeddings.*
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      *Implemented multi-head scaled dot-product attention with causal masking, GELU MLP, and Pre-LayerNorm transformer blocks.*
- [x] **4. Set up the training loop and train the GPT model on the narrative data.**  
      *Built the data pipeline, AdamW optimizer, cosine decay learning-rate schedule, gradient clipping, and checkpoint saving.*
- [x] **5. Implement a text generation feature that creates coherent story continuations.**  
      *Implemented temperature scaling, top-$k$ truncation, nucleus (top-$p$) filtering, greedy decoding, and seed reproducibility.*
- [x] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      *Created a command-line interface supporting direct story prompts, parameter flags, and an interactive storytelling session.*

---

## Requirements

- **Python 3.12+**
- **PyTorch** (CPU or GPU)
- **httpx** (for fetching sample adventure stories)
- **pytest** (for running the test suite)

---

## Quickstart & Installation

```bash
# 1. Clone repository and navigate inside
cd adventure-story-weaver

# 2. Create and activate a Python 3.12 virtual environment
python -m venv .venv
source .venv/bin/activate      # On Linux / macOS
# .venv\Scripts\activate       # On Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env

# 5. Verify the installation
python -m app.cli hello
```

---

## Step-by-Step Walkthrough

### 1. Build the Story Corpus

Assemble and sanitize Project Gutenberg texts (stripping license headers, footers, and excess blank lines) into `data/corpus.txt`:

```bash
# Automatically download and clean 5 classic adventure novels:
# Treasure Island, The Time Machine, The Lost World, Alice in Wonderland, A Princess of Mars
python scripts/build_corpus.py --download-sample
```

Output:
```text
Books processed : 5
Total characters: 1,502,923
Corpus file size: 1.48 MB
```

### 2. Train the Byte-Level BPE Tokenizer

Train a byte-pair tokenizer from scratch on the curated text and save the vocabulary and merges to `artifacts/tokenizer.json`:

```bash
python -m app.cli train-tokenizer --vocab-size 1000
```

Output:
```text
Vocab size         : 1001 (256 base bytes + 744 merges + <|endoftext|>)
Compression ratio  : 2.78 characters per token
Saved to           : artifacts/tokenizer.json
```

### 3. Train the GPT Language Model

Train the custom GPT model using the tokenized corpus:

```bash
# Short training run (demonstrating loss reduction):
python -m app.cli train --max_steps 200 --batch_size 16 --lr 1e-3

# Full training run (~15-20 minutes on laptop CPU):
python -m app.cli train --max_steps 1000 --batch_size 16 --lr 1e-3
```

During training, train and validation loss are tracked periodically, and the best checkpoint is automatically saved to `checkpoints/model.pt`.

### 4. Generate Story Continuations via CLI

Generate a dynamic continuation for any adventure prompt:

```bash
python -m app.cli "You stand at the mouth of a dark cave, a cold wind blowing from within."
```

#### Custom Sampling Flags

```bash
python -m app.cli "The ancient dragon opened its golden eyes" \
  --max-tokens 150 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --seed 42
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `prompt` | *Positional* | The story prompt string |
| `--max-tokens` | `200` | Number of tokens to generate |
| `--temperature` | `0.8` | Randomness (lower = focused, higher = creative, 0 = greedy) |
| `--top-k` | `40` | Limits sampling to the top $k$ most probable tokens |
| `--top-p` | `0.95` | Nucleus filtering threshold (cumulative probability) |
| `--seed` | `None` | Random seed for exact reproducibility |
| `--checkpoint` | `checkpoints/model.pt` | Path to model checkpoint |
| `--interactive` | `False` | Launch interactive terminal session |

#### Sample Output

```text
Prompt > You stand at the mouth of a dark cave, torch in hand.
Continuation:
You stand at the mouth of a dark cave, torch in hand. The stone walls glisten 
with damp moss, and far in the darkness below, the sound of rhythmic dripping 
echoes through the subterranean passage. A narrow staircase descends into the 
shadows, worn smooth by countless forgotten footsteps...
```

### 5. Interactive Story Mode

Launch a continuous story weaving session:

```bash
python -m app.cli --interactive
```

Type prompts to receive continuations; enter `quit` or `exit` to finish.

---

## How It Works

```
Raw Text (UTF-8)
       │
       ▼
[Byte-Level BPE] ──▶ Token IDs (B, T)
                           │
                           ▼
              [GPTEmbedding: Token + Positional]
                           │
                           ▼
          ┌───────────────────────────────────┐
          │     Transformer Block (x N)       │
          │  ┌─────────────────────────────┐  │
          │  │ Pre-LN ──▶ Causal Attention │  │
          │  │    │              │         │  │
          │  │    └──────(+)─────┘         │  │
          │  │            │                │  │
          │  │ Pre-LN ──▶ GELU MLP         │  │
          │  │    │              │         │  │
          │  │    └──────(+)─────┘         │  │
          │  └─────────────────────────────┘  │
          └───────────────────────────────────┘
                           │
                           ▼
                 [Final LayerNorm]
                           │
                           ▼
              [Linear LM Head (Tied)]
                           │
                           ▼
              Logits ──▶ [Sampling Engine] ──▶ Story Continuation
```

1. **Byte-Level Byte-Pair Encoding (BPE)**:
   Starts with 256 raw byte tokens (`0..255`), ensuring that any Unicode character, emoji, or symbol can be ingested without `<unk>` errors. Iteratively merges the most frequent adjacent byte pairs into composite subword tokens. Every input guarantees `decode(encode(text)) == text`.
2. **Learned Embeddings**:
   Maps input token IDs to continuous vectors of dimension $d_{\text{embd}}$ and adds learned positional embeddings ($0 \dots \text{block\_size}-1$) to convey sequence order.
3. **Causal Self-Attention**:
   Transforms input representations into Queries ($Q$), Keys ($K$), and Values ($V$). Attention weights are computed via scaled dot-product:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{\text{head}}}} + M\right) V$$
   The causal mask $M$ applies $-\infty$ to upper-triangular positions, ensuring token $t$ cannot attend to future tokens $t+1, t+2, \dots$.
4. **Transformer Decoder Block**:
   Assembles Pre-LayerNorm attention and a position-wise two-layer Feed-Forward Network ($\text{Linear} \to \text{GELU} \to \text{Linear}$) with residual skip connections.
5. **Autoregressive Training**:
   Optimizes next-token cross-entropy loss against $+1$ shifted target sequences using AdamW, linear warmup with cosine decay, and gradient clipping at 1.0.
6. **Sampling Strategies**:
   Applies temperature scaling ($\text{logits} / T$), top-$k$ filtering, and nucleus (top-$p$) cumulative probability truncation before multinomial sampling, with automatic greedy argmax fallback when $T=0$.

---

## Project Structure

```text
adventure-story-weaver/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
├── app/
│   ├── __init__.py              # Package marker
│   ├── cli.py                   # Command-line interface with interactive mode
│   ├── core.py                  # Milestone stubs and integration functions
│   ├── generate.py              # Text generation and sampling strategies
│   ├── llm.py                   # External chat endpoints (Ollama, Groq, Gemini)
│   ├── model.py                 # PyTorch GPT architecture built from scratch
│   ├── storage.py               # Lightweight local storage
│   ├── tokenizer.py             # Byte-level BPE tokenizer
│   └── train.py                 # Training loop, data loader, and scheduler
├── artifacts/
│   └── tokenizer.json           # Trained BPE vocabulary and merge rules
├── data/
│   └── corpus.txt               # Cleaned public-domain adventure story corpus
├── scripts/
│   ├── build_corpus.py          # Script to fetch and sanitize Gutenberg text
│   └── train_tokenizer.py       # Script to train BPE tokenizer
├── tests/
│   ├── test_core.py             # Comprehensive test suite (Milestones 1–6)
│   └── test_smoke.py            # Smoke tests verifying system components
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git ignore file (excludes .env, checkpoints, etc.)
├── pytest.ini                   # Pytest configuration
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies (httpx, pytest, torch)
```

---

## Running Tests

Run the full pytest suite:

```bash
pytest -v
```

All 21 unit and integration tests will run and pass without requiring external API keys or pre-existing checkpoints.
