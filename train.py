"""
Tiny LLM — see it learn.

A character-level transformer language model in one file. Trains on
Shakespeare and prints a sample every 200 steps so you can literally
watch it go from random noise to coherent-ish text.

The ONE piece you write yourself: Head.forward (the attention mechanism).
That's the heart of an LLM. Everything else is scaffolding.

Run: python train.py                       # default T=128 → kid.pt
     python train.py --block-size 512      # T=512        → kid_t512.pt
"""
import argparse
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CLI: lets us train at different context lengths without overwriting kid.pt ---
parser = argparse.ArgumentParser()
parser.add_argument("--block-size", type=int, default=128, help="context length in chars (default 128)")
parser.add_argument("--batch-size", type=int, default=32, help="sequences per batch (default 32). Reduce for long T to fit RAM.")
args = parser.parse_args()

# --- Hyperparameters: BIGGER KID + MULTI-HEAD EDITION ---
# Multi-head attention: instead of ONE pair of glasses per re-read, the kid
# wears 4 pairs in parallel, each looking for different patterns. Same total
# compute, more diverse "lenses." Should drop loss another ~0.05-0.10.
BLOCK_SIZE = args.block_size      # context length in chars
BATCH_SIZE = args.batch_size

# Output filenames: keep legacy names when T=128 + batch=32 so existing
# exports/tests don't break; suffix when either differs.
def _suffix() -> str:
    parts = []
    if BLOCK_SIZE != 128:
        parts.append(f"t{BLOCK_SIZE}")
    if BATCH_SIZE != 32:
        parts.append(f"b{BATCH_SIZE}")
    return ("_" + "_".join(parts)) if parts else ""

_S = _suffix()
CKPT_NAME    = f"kid{_S}.pt"
SAMPLES_NAME = f"samples{_S}.txt"
N_EMBD     = 128      # embedding dimension
N_LAYER    = 4        # number of transformer blocks
N_HEAD     = 4        # NEW — number of attention heads per block ("pairs of glasses")
LR         = 3e-4
MAX_STEPS  = 5000

# More checkpoints because more steps; still log-spaced so we catch early drama.
CHECKPOINTS = [0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
SEED_PROMPT = "ROMEO:"   # same prompt every checkpoint — watch this exact line get smarter

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Data: tinyshakespeare (~1MB) ---
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
    """Sample BATCH_SIZE random chunks of length BLOCK_SIZE.
    x is the input, y is x shifted by one (the next-char target)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
#  THE ONE THING YOU WRITE
# ─────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    """Single self-attention head — the core of the transformer.

    Input:  x of shape (B, T, C)   batch, time (= chars in context), channels (= N_EMBD)
    Output: same shape (B, T, C)   each position is a weighted blend of past positions
    """
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # Lower-triangular mask: position t can only attend to positions 0..t
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)                                              # (B, T, head_size)
        k = self.key(x)                                                # (B, T, head_size)
        v = self.value(x)                                              # (B, T, head_size)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)     # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                            # rows sum to 1
        return weights @ v                                             # (B, T, head_size)


# ─────────────────────────────────────────────────────────────────────
#  Everything below is plumbing — already filled in.
# ─────────────────────────────────────────────────────────────────────
class MultiHead(nn.Module):
    """N_HEAD attention heads running in parallel — different 'lenses' on the same input.
    Each head sees N_EMBD // N_HEAD dimensions; their outputs are concatenated back
    to N_EMBD and projected through one linear layer to mix what the heads found."""
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(head_size) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, N_EMBD)
        return self.proj(out)


class Block(nn.Module):
    """One transformer block: multi-head attention + feed-forward, with residuals + LayerNorm."""
    def __init__(self):
        super().__init__()
        self.attn = MultiHead()
        self.ffwd = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ffwd(self.ln2(x))   # feed-forward with residual
        return x


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)        # what each char means
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # where it is in the sequence
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB)           # back to vocab logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, VOCAB)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                     # only the last position matters
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def train():
    model = TinyLLM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params | device={DEVICE} | vocab={VOCAB}")
    print(f"Watching the prompt {SEED_PROMPT!r} get smarter at each checkpoint.\n")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    seed_ids = torch.tensor([encode(SEED_PROMPT)], dtype=torch.long, device=DEVICE)
    log = open(SAMPLES_NAME, "w")

    for step in range(MAX_STEPS + 1):
        if step in CHECKPOINTS:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch("val")
                _, val_loss = model(xb, yb)
                xb, yb = get_batch("train")
                _, train_loss = model(xb, yb)
            sample = decode(model.generate(seed_ids, 160)[0].tolist())
            header = f"=== step {step:>5} | train {train_loss.item():.3f} | val {val_loss.item():.3f} ==="
            print(f"\n{header}\n{sample}\n{'=' * 60}")
            log.write(f"\n{header}\n{sample}\n{'=' * 60}\n")
            log.flush()
            model.train()

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    log.close()
    torch.save(model.state_dict(), CKPT_NAME)
    size_mb = os.path.getsize(CKPT_NAME) / 1e6
    print(f"\nFull progression saved to {SAMPLES_NAME} — scroll it to watch dumb → smart.")
    print(f"Trained model saved to {CKPT_NAME} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train()
