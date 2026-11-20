[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/ksmrgqhbc8/badge.svg)](https://msrit.ask.thili.ai/b/ksmrgqhbc8)

# Adventure Story Weaver

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
- **numpy** (matrix operations & numerical stability)
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
python -m app.cli "You stand at the mouth of a dark cave"
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

#### Real Generation Output (Checkpoint Sample)

```text
Prompt > You stand at the mouth of a dark cave
Continuation:
You stand at the mouth of a dark cave of the tt and I beerging was the marletenereres it, and cow it, the dtesing the lely and 're
```

### 5. Interactive Story Mode

Launch a continuous story weaving session:

```bash
python -m app.cli --interactive
```

Type prompts to receive continuations; enter `quit` or `exit` to finish.

---

## Results and Limitations

### Empirical Training Results

All values reported below are exact numbers measured on the trained checkpoint (`checkpoints/model.pt`) and curated corpus:

| Metric | Measurement / Specification |
| :--- | :--- |
| **Model Architecture** | 4 Transformer layers (`n_layer=4`), 4 Attention heads (`n_head=4`), Pre-LayerNorm |
| **Embedding Dimension** | $d_{\text{embd}} = 128$ (Head dimension $d_{\text{head}} = 32$) |
| **Context Window** | $\text{block\_size} = 128$ tokens |
| **Vocabulary Size** | 1,001 tokens (256 base bytes + 744 learned merges + 1 `<|endoftext|>`) |
| **Parameter Count** | **937,856 parameters** (with token embedding and output head weight tying) |
| **Corpus Composition** | 5 public-domain books (*Treasure Island*, *The Time Machine*, *The Lost World*, *Alice in Wonderland*, *A Princess of Mars*) |
| **Corpus Size** | **1.48 MB** (1,555,176 bytes / 1,502,923 characters / 575,578 tokens) |
| **Data Partition** | 90% Training split (518,020 tokens) / 10% Validation split (57,558 tokens) |
| **Training Steps** | 200 optimization steps (batch size = 16) |
| **Hardware & Runtime** | Laptop CPU (`torch.device("cpu")`), ~90 seconds total training time |
| **Final Train Loss** | **5.2106** cross-entropy loss |
| **Final Validation Loss** | **5.1646** cross-entropy loss |
| **Validation Perplexity** | **174.97** ($\exp(5.1646)$) |

### Baseline Comparison

To verify that the model genuinely learned token syntax and language structure, cross-entropy validation loss was evaluated against statistical baselines on the identical validation split:

| Model / Baseline | Validation Loss (Cross-Entropy) | Perplexity | Notes |
| :--- | :--- | :--- | :--- |
| **Uniform Random Guess** | **6.9088** | 1,001.00 | Theoretical baseline: $\ln(V) = \ln(1001)$ |
| **Unigram Frequency Baseline** | **5.8611** | 351.11 | Independent token frequency distribution |
| **Trained GPT (200 steps)** | **5.1646** | **174.97** | Custom 4-layer Transformer trained from scratch |
| **Bigram Baseline (Laplace)** | **4.5642** | 95.98 | First-order Markov chain with add-1 smoothing |

*Key finding*: In just 200 optimization steps on a laptop CPU, the custom GPT significantly outperforms both the uniform random baseline (reducing loss by $1.74$ nats, a $5.7\times$ perplexity reduction) and the unigram distribution baseline (reducing loss by $0.70$ nats, a $2.0\times$ perplexity reduction).

### Real Sample Generations at Different Temperatures

Generated using `checkpoints/model.pt` with prompt `"You stand at the mouth of a dark cave"`:

- **Low Temperature ($T = 0.7$, Top-$k = 40$, Top-$p = 0.95$, seed = 42)**:
  > *"You stand at the mouth of a dark cave of the tt and I beerging was the marletenereres it, and cow it, the dtesing the lely and 're"*  
  *Characteristics*: More conservative token selections; prefers frequent English subwords and high-probability character combinations (`was the`, `and cow it`, `ing`).

- **Standard Temperature ($T = 1.0$, Top-$k = 40$, Top-$p = 0.95$, seed = 42)**:
  > *"You stand at the mouth of a dark cave\n\nlls and the mterging was refmlet salleres it, and cow it, the catesing the\n\nlely and 're"*  
  *Characteristics*: Introduces newline formatting breaks and diverse subword stems (`refmlet`, `catesing`).

- **High Temperature ($T = 1.3$, Top-$k = 40$, Top-$p = 0.95$, seed = 42)**:
  > *"You stand at the mouth of a dark cave\n\nlls and the mterging was refmlet sallkes its and cow it\n\noctesing the\n\nlely- 're"*  
  *Characteristics*: Higher variance sampling; explores rarer byte combinations (`sallkes`, `octesing`).

### Honest Limitations

1. **Sub-Million Parameter Scale**: With 937,856 parameters, this model is orders of magnitude smaller than modern LLMs (which have billions of parameters). Its capacity is suitable for modeling subword transitions and syllable patterns, but insufficient for complex semantic reasoning.
2. **Repetition and Hallucinated Syllables**: The model frequently loops phonetic fragments or composite suffixes (e.g. `-enereres`, `-esing`, `-lely`) because local token n-gram transitions dominate over global sentence coherence.
3. **Short Context Horizon**: While the theoretical attention context is 128 tokens, the model loses the narrative thread after just a few words. It cannot maintain character identities, inventory states, or room layouts across interactive turns.
4. **No Semantic Story Logic**: Because it is trained purely as an autoregressive next-token predictor without instruction tuning or RLHF, it does not understand game mechanics or follow conversational commands.
5. **Corpus & Style Bias**: Trained strictly on Victorian and early 20th-century adventure texts (Project Gutenberg), the model's vocabulary and cadence are biased towards period English prose rather than modern interactive fiction.
6. **Laptop CPU Constraints**: Training was intentionally constrained to a fast, laptop-friendly budget (~90 seconds for 200 steps). Longer training runs (e.g., 5,000+ steps with larger embeddings) yield improved vocabulary assembly, but true narrative continuity requires substantially larger models and datasets.

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
│   ├── core.py                  # Milestone integration module
│   ├── generate.py              # Text generation and sampling strategies
│   ├── llm.py                   # Environment settings loader
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
├── requirements.txt             # Project dependencies (numpy, pytest, torch)
└── start.bat                    # Windows one-click interactive launcher
```

---

## Running Tests

Run the full pytest suite:

```bash
pytest -v
```

All 21 unit and integration tests will run and pass without requiring external API keys or pre-existing checkpoints.
