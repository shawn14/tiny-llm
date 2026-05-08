# tiny-llm

Small, readable, end-to-end implementation of a modern transformer language model — covering pretraining, sparse-attention variants, and supervised fine-tuning. Same recipe Llama-3 / Mistral / Qwen use, scaled down to run on a MacBook in hours instead of months on thousands of GPUs.

The goal is **pedagogical**: every script is self-contained, under a few hundred lines, and uses the same architectural recipe as frontier models. Read the code, run it, watch it learn.

## What's here

```
tiny-llm/
├── train.py                    # Original char-level kid (Shakespeare, 825K params)
├── train_sparse.py             # Naive top-K sparse attention variant
├── train_modern.py             # Modern dense (RoPE, RMSNorm, SwiGLU, GQA, bf16)
├── train_modern_sparse.py      # Modern + NSA-style hybrid sparse
├── train_sft.py                # Supervised fine-tuning on Alpaca
├── prepare_data.py             # Download FineWeb-Edu corpus
├── export_for_web.py           # Export kid weights → JSON for browser
├── export_sparse_for_web.py    # Export sparse variants → JSON
├── export_dense_for_web.py     # Export non-default-context dense → JSON
└── show_model.py               # Inspect a saved checkpoint
```

## The five training stages of a frontier model

This repo walks through the first two stages explicitly, with the architecture wired up for all five:

```
1. Pretraining          ✓  train_modern.py     ("completes text")
2. Supervised FT (SFT)  ✓  train_sft.py         ("follows instructions")
3. Preference learning  -  (DPO, not yet)       ("prefers helpful responses")
4. Reasoning training   -  (CoT, not yet)       ("thinks step by step")
5. Tool use             -  (not yet)            ("uses external tools")
```

## Modern architecture used in `train_modern.py`

Every component matches Llama-3 / Mistral / Qwen:

- **RoPE** rotary position embeddings (replaces learned `pos_emb`)
- **RMSNorm** (replaces LayerNorm)
- **SwiGLU** MLP (replaces ReLU MLP)
- **GQA** Grouped-Query Attention (8 query heads, 4 KV heads)
- **bfloat16** mixed precision
- **AdamW** with `betas=(0.9, 0.95)`, `wd=0.1`
- **Cosine LR** schedule with linear warmup
- **Tied embeddings** (input embedding shared with output projection)
- **Gradient clipping** at 1.0
- `bias=False` on linear layers

About 25M parameters. Trains in ~6-10 hours on an M-series Mac with 16 GB unified memory.

## Quick start

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch datasets

# 2. Train the original char-level kid on Shakespeare (~5-10 min)
python train.py

# 3. Or: full modern pipeline
python prepare_data.py                # download 50 MB of FineWeb-Edu
python train_modern.py                # ~6-10 hours pretraining
python train_sft.py                   # ~1-2 hours SFT
```

## Sparse-attention experiments

`train_sparse.py` replaces standard attention with **top-K sparse attention**: each query attends only to its K most-relevant past positions. Configurable via `--k`. Works at the level of individual attention heads.

Findings from running this at multiple scales:

| Run | Train | Val | Notes |
|---|---|---|---|
| Dense T=128 | 1.513 | **1.694** | baseline |
| Sparse K=4 T=128 | 1.478 | **1.646** | wins by 0.05 |
| Sparse K=16 T=128 | 1.447 | 1.652 | wins by 0.04 |
| Dense T=512 (avg of 2 seeds) | 1.529 | 1.697 | longer context didn't help |
| Sparse K=4 T=512 | 1.689 | 1.920 | underfits badly |
| Sparse K=16 T=512 | 1.596 | 1.806 | still loses to dense |
| Dense T=1024 | 1.583 | 1.731 | dense degrading |
| Sparse K=32 T=1024 | 1.829 | **1.994** | loses by 0.26 |

**Key finding:** naive top-K sparse beats dense at small T (regularization effect when overfitting is the bottleneck) but loses badly at long T. Production systems (NSA, Longformer, BigBird) use **hybrid sparsity** (top-K + sliding window + global tokens) instead. `train_modern_sparse.py` implements the NSA-style hybrid.

## How small is "small"?

| | tiny-llm modern | Llama-3-8B |
|---|---|---|
| Parameters | 25M | 8B (320×) |
| Training tokens | 10M | 15T (1,500,000×) |
| Training compute | hours on M4 Mac | ~1.3M H100-hours (~$30M) |
| Tokenizer | char-level (95 chars) | BPE (~128K subword tokens) |
| Context length | 1024 | 128K |
| Post-training | optional SFT | SFT + RLHF + Constitutional AI |
| Final val loss | ~1.34 nats/char (~1.93 BPC) | ~0.76 BPC |

Same architecture. Same training algorithm. Different scale.

## What this repo doesn't have

To get to "actually useful chatbot," you'd need:
- A real (BPE) tokenizer
- ~100×–1000× more data
- A GPU cluster (or more patience)
- DPO / preference learning
- Reasoning training (CoT)
- Safety training
- Multi-modal training (if you want vision)

This is the floor. The recipe here, scaled up, is roughly what frontier labs ship.

## Companion site

Live, interactive playground for the trained models and a chapter-by-chapter walkthrough of how they work:

→ https://peek-three-beta.vercel.app

Includes:
- `/playground` — type a prompt, the original kid generates char-by-char in your browser
- `/sparse` — sparse-attention lab with multiple trained variants
- `/atlas` — the entire forward pass on one page

## License

MIT. Use it, fork it, scale it.

## Credits

- Original tinyshakespeare dataset: Andrej Karpathy ([char-rnn](https://github.com/karpathy/char-rnn))
- `train.py` structure inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
- Modern architecture patterns from Llama-3, Mistral, Qwen, DeepSeek-V3
- FineWeb-Edu corpus by Hugging Face's FineWeb team
- Sparse attention concepts from Native Sparse Attention (DeepSeek, 2025), Longformer, BigBird
