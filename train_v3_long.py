"""
Tiny LLM — v3-long (architecture identical to v3, run further).

v3 hit val 1.625 at 5K steps but val loss was still descending and the
cosine schedule had cooled LR to 3e-5 by the end — model was crawling.
This run doubles step budget to 10K and raises the cosine floor to 5e-5
so the late-stage steps still actually learn.

Architecture is unchanged from v3 (RoPE + RMSNorm + GELU, all Tier-1
training discipline).  Only the training schedule changed:

  MAX_STEPS    5000  →  10000
  MIN_LR       3e-5  →  5e-5
  CHECKPOINTS  ...add 6K/7K/8K/9K/10K so the slider keeps making sense

Expected: val ~1.55-1.60 (v1: 1.741, v2: 1.718, v3: 1.625).

Run: python train_v3_long.py
"""
import math
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters ---
BLOCK_SIZE   = 128
BATCH_SIZE   = 32
N_EMBD       = 128
N_LAYER      = 4
N_HEAD       = 4
PEAK_LR      = 3e-4
MIN_LR       = 5e-5      # raised from 3e-5 so end-of-training still learns
WARMUP_STEPS = 100
MAX_STEPS    = 10000     # doubled from 5000 — v3's val was still descending
DROPOUT      = 0.2
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
ROPE_BASE    = 10000.0

CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000,
               3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
SEED_PROMPT = "ROMEO:"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Data: tinyshakespeare ---
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    print("Downloading tinyshakespeare...")
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt").read()

chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
#  T2#3  Rotary positional embeddings (RoPE)
#
#  Position is no longer a learned lookup added to the input embedding.
#  Instead, every (Q, K) vector is *rotated* by an angle proportional to
#  its position in the sequence, with different rotation rates per
#  dimension pair.  Two positions m and n produce a dot product that
#  depends only on (m − n), so attention naturally encodes relative
#  position with zero learned parameters.
#
#  Convention used here is the GPT-NeoX/Llama "split-half" form:
#    given x = [x1 | x2] with x1, x2 ∈ R^(head_size/2)
#    rotated = [x1·cos − x2·sin  |  x1·sin + x2·cos]
#  This is mathematically equivalent to the "interleaved-pairs" form
#  in the original RoFormer paper and easier to express in PyTorch.
# ─────────────────────────────────────────────────────────────────────
def precompute_rope(head_size, max_seq, base=ROPE_BASE):
    """Returns cos, sin of shape (max_seq, head_size // 2)."""
    half = head_size // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)            # (max_seq, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE to x of shape (B, T, head_size). cos, sin: (T, head_size//2)."""
    x1, x2 = x.chunk(2, dim=-1)                 # each (B, T, half)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


# ─────────────────────────────────────────────────────────────────────
#  Attention head — same shape as v1/v2, but Q and K get rotated.
# ─────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        cos, sin = precompute_rope(head_size, BLOCK_SIZE)
        self.register_buffer("rope_cos", cos)   # T2#3
        self.register_buffer("rope_sin", sin)   # T2#3
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # T2#3  rotate Q and K (not V) by their absolute position
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(head_size) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),                           # T2#1  was nn.ReLU()
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )
        self.ln1 = nn.RMSNorm(N_EMBD)            # T2#2  was nn.LayerNorm
        self.ln2 = nn.RMSNorm(N_EMBD)            # T2#2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        # T2#3  No more pos_emb — RoPE handles position inside attention.
        #       Saves 16,384 params (BLOCK_SIZE × N_EMBD).
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.RMSNorm(N_EMBD)        # T2#2
        self.lm_head = nn.Linear(N_EMBD, VOCAB)
        self.lm_head.weight = self.tok_emb.weight   # tied
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)                    # T2#3  no + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
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


def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def make_optimizer(model):
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay,   "weight_decay": WEIGHT_DECAY},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def estimate_val_loss(model, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch("val")
        _, l = model(xb, yb)
        losses.append(l.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    model = TinyLLM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.3f}M params (RoPE+RMSNorm+GELU, tied) | device={DEVICE} | vocab={VOCAB}")
    print(f"Watching the prompt {SEED_PROMPT!r} get smarter at each checkpoint.\n")
    opt = make_optimizer(model)

    seed_ids = torch.tensor([encode(SEED_PROMPT)], dtype=torch.long, device=DEVICE)
    log = open("samples_v3_long.txt", "w")
    best_val = float("inf")
    best_step = 0

    for step in range(MAX_STEPS + 1):
        lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr

        if step in CHECKPOINTS:
            val_loss = estimate_val_loss(model)
            with torch.no_grad():
                xb, yb = get_batch("train")
                _, train_loss = model(xb, yb)
            model.eval()
            sample = decode(model.generate(seed_ids, 160)[0].tolist())
            model.train()

            gap = val_loss - train_loss.item()
            header = (f"=== step {step:>5} | lr {lr:.2e} | "
                      f"train {train_loss.item():.3f} | val {val_loss:.3f} | gap {gap:+.3f} ===")
            print(f"\n{header}\n{sample}\n{'=' * 60}")
            log.write(f"\n{header}\n{sample}\n{'=' * 60}\n")
            log.flush()

            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                torch.save(model.state_dict(), "kid_best_v3_long.pt")
                print(f"   ↳ new best val {best_val:.3f} at step {step} → kid_best_v3_long.pt")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    log.close()
    torch.save(model.state_dict(), "kid_v3_long.pt")
    print()
    print(f"Final model:  kid_v3_long.pt        ({os.path.getsize('kid_v3_long.pt')/1e6:.1f} MB) — last step")
    print(f"Best model:   kid_best_v3_long.pt   ({os.path.getsize('kid_best_v3_long.pt')/1e6:.1f} MB) — step {best_step}")
    print(f"Best val loss: {best_val:.3f}   (v1: 1.741, v2: 1.718, v3@5k: 1.625)")
    print(f"Sample log:   samples_v3_long.txt")


if __name__ == "__main__":
    train()
