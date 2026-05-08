"""
Tiny LLM — Modern Edition (sparse).

Identical architecture to train_modern.py — RoPE, RMSNorm, SwiGLU, GQA, bf16,
AdamW, cosine LR — with NSA-style hybrid sparse attention replacing dense
attention.

The hybrid sparse pattern (inspired by DeepSeek's Native Sparse Attention,
Feb 2025) combines three masks unioned together:

  1. Top-K (K=64): the K positions with highest q·k score per query
                    (content-based, the "smart" part)
  2. Sliding window (W=128): the W most-recent positions for every query
                              (guaranteed locality, prevents catastrophic
                               failure when top-K misses obvious neighbors)
  3. Global tokens (G=4): the first G tokens, always attended to from
                           anywhere (anchor / "attention sink" tokens)

A position attends to a key iff (top-K) OR (within W) OR (j < G).

This is much more robust than pure top-K, which we saw can underfit at long
context when K is too small relative to T.

Run:
    python prepare_data.py
    python train_modern_sparse.py
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- CLI --------
parser = argparse.ArgumentParser()
parser.add_argument("--block-size", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--steps",      type=int, default=8000)
parser.add_argument("--warmup",     type=int, default=800)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--corpus",     type=str, default="corpus.txt")
parser.add_argument("--top-k",      type=int, default=64,  help="content-based attention budget per query")
parser.add_argument("--window",     type=int, default=128, help="sliding-window attention budget per query")
parser.add_argument("--global-tokens", type=int, default=4, help="number of always-attended-to tokens at start")
args = parser.parse_args()

# -------- Hyperparameters --------
N_EMBD     = 384
N_LAYER    = 8
N_HEAD     = 8
N_KV_HEAD  = 4
HEAD_DIM   = N_EMBD // N_HEAD
MLP_HIDDEN = 1024
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size
LR         = args.lr
WD         = 0.1
WARMUP     = args.warmup
MAX_STEPS  = args.steps

TOP_K   = args.top_k
WINDOW  = args.window
GLOBAL  = args.global_tokens

CKPT_NAME    = f"kid_modern_sparse_t{BLOCK_SIZE}_k{TOP_K}.pt"
SAMPLES_NAME = f"samples_modern_sparse_t{BLOCK_SIZE}_k{TOP_K}.txt"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_BF16 = (DEVICE in ("cuda", "mps")) and hasattr(torch, "bfloat16")
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

# -------- Data --------
if not os.path.exists(args.corpus):
    raise SystemExit(f"Corpus not found: {args.corpus}\nRun `python prepare_data.py` first.")

print(f"Loading corpus: {args.corpus}")
text = open(args.corpus).read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"  Vocab: {VOCAB} chars")
print(f"  Train: {len(train_data):,} tokens")
print(f"  Val:   {len(val_data):,} tokens")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# -------- Modern building blocks (same as train_modern.py) --------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    T = x.size(-2)
    cos_t = cos[:T].to(x.device, dtype=x.dtype)
    sin_t = sin[:T].to(x.device, dtype=x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos_t - x2 * sin_t
    rot2 = x1 * sin_t + x2 * cos_t
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def precompute_window_mask(T, W, G):
    """Static part of the sparse mask: causal AND (within W OR j < G).
    Doesn't depend on content; can be precomputed once."""
    j = torch.arange(T)[None, :]      # (1, T)
    i = torch.arange(T)[:, None]      # (T, 1)
    causal      = j <= i
    in_window   = (i - j) < W
    is_global   = j < G
    static_mask = causal & (in_window | is_global)   # (T, T)
    return static_mask


class GQAttentionSparse(nn.Module):
    """Grouped Query Attention with NSA-style hybrid sparse mask.

    Per query, attend to the union of:
      (1) the W most-recent positions (sliding window)
      (2) the first G positions (global anchors)
      (3) the K positions with highest score (content-based top-K)

    All within the causal (j <= i) constraint.
    """
    def __init__(self):
        super().__init__()
        assert N_HEAD % N_KV_HEAD == 0
        self.q_proj = nn.Linear(N_EMBD, N_HEAD    * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(N_EMBD, N_KV_HEAD * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(N_EMBD, N_KV_HEAD * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEAD * HEAD_DIM, N_EMBD,    bias=False)

        causal_mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool))
        static_mask = precompute_window_mask(BLOCK_SIZE, WINDOW, GLOBAL)  # window + global
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer("static_mask", static_mask)   # window OR global, ANDed with causal

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, N_HEAD,    HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        repeats = N_HEAD // N_KV_HEAD
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Raw scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        # Causal mask (j > i → -inf)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        # ── Top-K content-based selection ──
        # For each (B, head, query) row of scores, find the K-th largest valid
        # value as a threshold, then mask everything below it to -inf — UNLESS
        # the position is in our static (window OR global) mask.
        if 0 < TOP_K < T:
            # Find K-th largest score per row
            # NOTE: at very early positions (i < TOP_K), the row has fewer than
            # TOP_K valid (non-inf) entries, so topk includes -inf entries.
            # That's fine — the threshold becomes -inf and nothing gets masked
            # by the topK rule.
            k_for_row = min(TOP_K, T)
            topk_vals, _ = scores.topk(k_for_row, dim=-1)
            kth = topk_vals[..., -1:]                      # (B, H, T, 1)
            keep_topk = scores >= kth

            # Final mask = (top-K) OR (sliding window) OR (global)
            static = self.static_mask[:T, :T]              # (T, T)
            keep = keep_topk | static                      # broadcast static across (B, H)
            scores = scores.masked_fill(~keep, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, N_HEAD * HEAD_DIM)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(N_EMBD)
        self.attn  = GQAttentionSparse()
        self.norm2 = RMSNorm(N_EMBD)
        self.mlp   = SwiGLU(N_EMBD, MLP_HIDDEN)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp (self.norm2(x))
        return x


class ModernKidSparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.blocks  = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.norm_f  = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(HEAD_DIM, BLOCK_SIZE)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def cosine_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR * 0.1 + LR * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train():
    model = ModernKidSparse().to(DEVICE)
    if USE_BF16:
        model = model.to(DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModern kid (sparse): {n_params/1e6:.2f}M params")
    print(f"  Architecture: N_EMBD={N_EMBD}  L={N_LAYER}  H={N_HEAD}  KV={N_KV_HEAD}")
    print(f"  Sparsity: top-K={TOP_K} + window={WINDOW} + global={GLOBAL}")
    print(f"  Training: T={BLOCK_SIZE}  batch={BATCH_SIZE}  lr={LR}  steps={MAX_STEPS}")
    print(f"  Device: {DEVICE}  Dtype: {DTYPE}\n")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WD,
    )

    seed = "Once upon a time" if "Once" in text[:5000] else text[:32]
    seed_ids = torch.tensor([encode(seed)], dtype=torch.long, device=DEVICE)
    log = open(SAMPLES_NAME, "w")

    checkpoints = sorted(set([0, 100, 250, 500, 1000, 2000, 4000, 6000, MAX_STEPS]))

    for step in range(MAX_STEPS + 1):
        lr_now = cosine_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr_now

        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch("val")
                _, val_loss = model(xb, yb)
                xb, yb = get_batch("train")
                _, train_loss = model(xb, yb)
                sample = decode(model.generate(seed_ids, 200)[0].tolist())
            header = (
                f"=== step {step:>5} | train {train_loss.item():.3f} "
                f"| val {val_loss.item():.3f} | lr {lr_now:.2e} "
                f"| modern sparse K={TOP_K} W={WINDOW} ==="
            )
            print(f"\n{header}\n{sample}\n{'=' * 70}")
            log.write(f"\n{header}\n{sample}\n{'=' * 70}\n")
            log.flush()
            model.train()

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log.close()
    torch.save(model.state_dict(), CKPT_NAME)
    size_mb = os.path.getsize(CKPT_NAME) / 1e6
    print(f"\nModern kid (sparse) saved to {CKPT_NAME} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train()
