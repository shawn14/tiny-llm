"""
Tiny LLM — v2 (regularized).

Same architecture as train.py. Same data. Same prompt. Same step budget.
The only thing that changes is the *training discipline*, and that's enough
to close the train/val gap visible in v1's run (train 1.476 / val 1.741
at step 5,000 — overfit by 0.27).

What changed and why:

  T1#1  Dropout 0.2 on attn output and MLP output     regularize the small model
  T1#2  Cosine LR schedule with linear warmup         stable early, slow finish
  T1#3  Tied embeddings (lm_head.weight = tok_emb)    saves 8,320 params, regularizes
  T1#4  AdamW param groups (decay 2D weights only)    don't decay biases / LN gains
  T1#5  Gradient clip 1.0                             insurance against loss spikes
  T1#6  Save best-val checkpoint, not last            v1's val loss bottoms early
  T1#7  Finer early checkpoints                       most drama lives before step 200

Expected: val 1.74 → ~1.55-1.62.

Run: python train_v2.py
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
PEAK_LR      = 3e-4    # T1#2  peak after warmup
MIN_LR       = 3e-5    # T1#2  cosine floor (10% of peak)
WARMUP_STEPS = 100     # T1#2
MAX_STEPS    = 5000
DROPOUT      = 0.2     # T1#1
WEIGHT_DECAY = 0.1     # T1#4
GRAD_CLIP    = 1.0     # T1#5

# T1#7  finer early checkpoints — uppercase clustering, ':' patterns, line breaks
# all emerge before step 200.  These are nearly free and make the slider far better.
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
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
#  Attention head — unchanged from v1.
# ─────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
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
        self.dropout = nn.Dropout(DROPOUT)   # T1#1

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))   # T1#1  attn-output dropout


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),             # T1#1  MLP-output dropout
        )
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB)
        # T1#3  Tie input and output embeddings.  Same (VOCAB × N_EMBD) matrix,
        #       two roles: looking up the meaning of a token going IN, scoring
        #       every token going OUT.  Saves 8,320 params (= VOCAB × N_EMBD)
        #       and is a known regularizer for char-level models.  The lm_head
        #       bias (65 numbers) is kept so the math story stays intact.
        self.lm_head.weight = self.tok_emb.weight

        # Init: nanoGPT-style N(0, 0.02) for all Linear/Embedding weights, zero biases.
        # This matters because torch's default nn.Embedding init is N(0, 1) — once we
        # tie that into lm_head, logits explode and step-0 loss becomes ~80 instead of
        # the expected ~log(VOCAB) = 4.17.  Apply explicit init so tying is safe.
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
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
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


# T1#2  Linear warmup → cosine decay.
def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# T1#4  Two AdamW param groups.  Decay 2D matmul weights; leave biases and
#       LayerNorm gains/biases alone.  Both ends of the spectrum hurt val loss
#       if you get them wrong.
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
    """Average over a few batches so the val number isn't fooled by one lucky chunk.
    v1's val numbers are noisy because they come from a single 32×128 batch."""
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
    n_params = sum(p.numel() for p in model.parameters())   # parameters() dedupes ties
    print(f"Model: {n_params/1e6:.3f}M params (tied) | device={DEVICE} | vocab={VOCAB}")
    print(f"Watching the prompt {SEED_PROMPT!r} get smarter at each checkpoint.\n")
    opt = make_optimizer(model)

    seed_ids = torch.tensor([encode(SEED_PROMPT)], dtype=torch.long, device=DEVICE)
    log = open("samples_v2.txt", "w")
    best_val = float("inf")     # T1#6
    best_step = 0

    for step in range(MAX_STEPS + 1):
        lr = get_lr(step)        # T1#2
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

            # T1#6  best-val checkpointing
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                torch.save(model.state_dict(), "kid_best.pt")
                print(f"   ↳ new best val {best_val:.3f} at step {step} → kid_best.pt")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # T1#5
        opt.step()

    log.close()
    torch.save(model.state_dict(), "kid_v2.pt")
    print()
    print(f"Final model: kid_v2.pt    ({os.path.getsize('kid_v2.pt')/1e6:.1f} MB) — last step")
    print(f"Best model:  kid_best.pt  ({os.path.getsize('kid_best.pt')/1e6:.1f} MB) — step {best_step}")
    print(f"Best val loss: {best_val:.3f}   (v1 was 1.741)")
    print(f"Sample log:  samples_v2.txt")


if __name__ == "__main__":
    train()
