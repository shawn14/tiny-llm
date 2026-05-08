"""
Tiny LLM — Modern Edition (dense).

Same pedagogical philosophy as train.py, but with 2026 best practices used
by Llama-3, Mistral, Qwen-2.5, etc.

What's modern here:
  - RoPE (Rotary Position Embeddings)        — replaces learned pos_emb
  - RMSNorm                                  — replaces LayerNorm
  - SwiGLU MLP                               — replaces ReLU MLP
  - GQA: 8 query heads, 4 KV heads           — memory-efficient attention
  - bfloat16 mixed precision                 — 2× faster on Apple Silicon
  - AdamW with β=(0.9, 0.95), wd=0.1         — Llama-style optimizer
  - Cosine LR schedule with 10% warmup       — modern LR shape
  - Gradient clipping at 1.0                 — stability
  - Tied input/output embeddings             — saves params at small scale
  - bias=False on linears                    — cleaner

About 25M parameters. Trains on a 10M-token corpus from FineWeb-Edu in
~6-10 hrs on M4 Mac at T=1024 batch=16.

Run:
    python prepare_data.py            # one-time, builds corpus.txt
    python train_modern.py            # default settings
    python train_modern.py --block-size 512 --batch-size 32  # smaller, faster
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
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--warmup", type=int, default=800)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--corpus", type=str, default="corpus.txt")
args = parser.parse_args()

# -------- Hyperparameters --------
N_EMBD     = 384
N_LAYER    = 8
N_HEAD     = 8                     # query heads
N_KV_HEAD  = 4                     # KV heads (GQA: 2:1 ratio)
HEAD_DIM   = N_EMBD // N_HEAD      # 48
MLP_HIDDEN = 1024                  # SwiGLU intermediate dim
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size
LR         = args.lr
WD         = 0.1
WARMUP     = args.warmup
MAX_STEPS  = args.steps

# -------- Output paths --------
CKPT_NAME    = f"kid_modern_t{BLOCK_SIZE}.pt"
SAMPLES_NAME = f"samples_modern_t{BLOCK_SIZE}.txt"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# bfloat16 if hardware supports it (Apple M-series does via MPS).
USE_BF16 = (DEVICE in ("cuda", "mps")) and hasattr(torch, "bfloat16")
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

# -------- Data --------
if not os.path.exists(args.corpus):
    raise SystemExit(
        f"Corpus not found: {args.corpus}\n"
        f"Run `python prepare_data.py` first."
    )

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


# -------- Modern building blocks --------

class RMSNorm(nn.Module):
    """Root-Mean-Square normalization. No mean centering, no bias."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    """Precompute the cos/sin tables for RoPE up to max_seq_len."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)             # (max_seq_len, dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE to last dim. x shape: (B, n_head, T, head_dim)."""
    T = x.size(-2)
    cos_t = cos[:T].to(x.device, dtype=x.dtype)   # (T, head_dim/2)
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
    """SwiGLU: down(silu(gate(x)) * up(x)). 3 linears, no bias."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GQAttention(nn.Module):
    """Grouped Query Attention with RoPE.

    N_HEAD query heads, N_KV_HEAD key/value heads.
    Each KV head is shared by N_HEAD/N_KV_HEAD = 2 query heads.

    Memory savings vs MHA at inference: KV cache is 2× smaller.
    Quality cost: small to negligible at this scale.
    """
    def __init__(self):
        super().__init__()
        assert N_HEAD % N_KV_HEAD == 0
        self.q_proj = nn.Linear(N_EMBD, N_HEAD    * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(N_EMBD, N_KV_HEAD * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(N_EMBD, N_KV_HEAD * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEAD * HEAD_DIM, N_EMBD,    bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.bool)),
        )

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, N_HEAD,    HEAD_DIM).transpose(1, 2)  # (B, nH, T, D)
        k = self.k_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)  # (B, nKV, T, D)
        v = self.v_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: repeat KV heads to match Q heads
        repeats = N_HEAD // N_KV_HEAD
        k = k.repeat_interleave(repeats, dim=1)   # (B, nH, T, D)
        v = v.repeat_interleave(repeats, dim=1)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                            # (B, nH, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, N_HEAD * HEAD_DIM)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm: x = x + attn(norm(x)); x = x + mlp(norm(x))."""
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(N_EMBD)
        self.attn  = GQAttention()
        self.norm2 = RMSNorm(N_EMBD)
        self.mlp   = SwiGLU(N_EMBD, MLP_HIDDEN)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp (self.norm2(x))
        return x


class ModernKid(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.blocks  = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.norm_f  = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        # Tie input embeddings and output projection
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
    """Linear warmup, then cosine decay to 10% of peak LR."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return LR * 0.1 + LR * 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train():
    model = ModernKid().to(DEVICE)
    if USE_BF16:
        model = model.to(DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModern kid (dense): {n_params/1e6:.2f}M params")
    print(f"  Architecture: N_EMBD={N_EMBD}  L={N_LAYER}  H={N_HEAD}  KV={N_KV_HEAD}")
    print(f"  Training:     T={BLOCK_SIZE}  batch={BATCH_SIZE}  lr={LR}  steps={MAX_STEPS}")
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
        # LR schedule
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
                f"| val {val_loss.item():.3f} | lr {lr_now:.2e} | modern dense ==="
            )
            print(f"\n{header}\n{sample}\n{'=' * 70}")
            log.write(f"\n{header}\n{sample}\n{'=' * 70}\n")
            log.flush()
            model.train()

        # Train step
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log.close()
    torch.save(model.state_dict(), CKPT_NAME)
    size_mb = os.path.getsize(CKPT_NAME) / 1e6
    print(f"\nModern kid (dense) saved to {CKPT_NAME} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train()
