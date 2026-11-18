# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ksmrgqhbc8/badge.svg)](https://ask.thili.ai/b/ksmrgqhbc8)

Ever wanted to dynamically generate engaging story segments for your text adventures? This project guides you through building a custom GPT from scratch. By training on a corpus of adventure game narratives, your GPT will learn story structures and common phrases, enabling it to create coherent and exciting continuations for your game worlds.

## What you are building

**Goal.** You will build a text adventure story segment generator, where a custom GPT, trained on existing adventure game narratives, creates dynamic story continuations.

**Done when.** You will have successfully applied byte-pair tokenization and embeddings to prepare raw text, implemented causal self-attention, assembled a transformer block, and trained your GPT model from scratch to generate coherent text adventure story continuations.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Curating your raw text corpus.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: Byte-pair tokenization and embedding layers.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal self-attention and transformer block construction.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Training loop and gradient descent.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Text generation with sampling strategies.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: Building an interface for text generation.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to generate multiple alternative continuations for a given prompt.
- Implement a feature to save and load ongoing story sessions.
- Add an option to specify the desired length of the generated story segment.
