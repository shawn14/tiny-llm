"""
Tiny LLM — sparse-attention edition.

Same architecture as train.py, same Shakespeare data, same training loop —
the ONE difference is that Head.forward does top-K masking before softmax.
At T=128 with K=8, each position only attends to its 8 most-relevant past
positions instead of all of them. This is the SubQ / Native-Sparse-Attention
family: content-dependent sparse attention, learned end-to-end.

Run: python train_sparse.py            # default K=8 → kid_sparse.pt
     python train_sparse.py --k 4      # K=4      → kid_sparse_k4.pt
     python train_sparse.py --k 16     # K=16     → kid_sparse_k16.pt
"""
import argparse
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CLI ---
parser = argparse.ArgumentParser(description="Train the sparse kid")
parser.add_argument("--k", type=int, default=8, help="top-K positions per query (default 8)")
parser.add_argument("--block-size", type=int, default=128, help="context length in chars (default 128)")
parser.add_argument("--batch-size", type=int, default=32, help="sequences per batch (default 32). Reduce for long T to fit RAM.")
args = parser.parse_args()
TOP_K = args.k

# --- Hyperparameters: same as train.py + one new knob ---
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size
N_EMBD     = 128
N_LAYER    = 4
N_HEAD     = 4
LR         = 3e-4
MAX_STEPS  = 5000

# Output filenames. T=128 + K=8 + batch=32 keeps the legacy filename.
# Otherwise suffix with _t{T} / _k{K} / _b{B} as needed so we never overwrite.
def _suffix() -> str:
    parts = []
    if BLOCK_SIZE != 128:
        parts.append(f"t{BLOCK_SIZE}")
    if TOP_K != 8:
        parts.append(f"k{TOP_K}")
    if BATCH_SIZE != 32:
        parts.append(f"b{BATCH_SIZE}")
    return ("_" + "_".join(parts)) if parts else ""

CKPT_NAME    = f"kid_sparse{_suffix()}.pt"
SAMPLES_NAME = f"samples_sparse{_suffix()}.txt"

CHECKPOINTS = [0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
SEED_PROMPT = "ROMEO:"

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
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
#  THE ONE THING THAT'S DIFFERENT
# ─────────────────────────────────────────────────────────────────────
class SparseHead(nn.Module):
    """Single self-attention head with top-K sparse masking.

    For each query, after computing q·k scores, we keep only the K largest
    scores and mask everything else to -inf. The softmax then puts ~all
    probability mass on those K positions.

    The selection is content-dependent: which K positions get picked depends
    on the actual q·k similarities, not a fixed pattern. This is the SubQ /
    NSA family. End-to-end differentiable because top-K just routes
    gradients through the kept positions.
    """
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

        # ── top-K sparse mask ──
        # For each (B, query_pos) row of scores, keep the TOP_K largest
        # values, set the rest to -inf. After softmax, attention is sparse.
        if T > TOP_K:
            topk_vals, _ = scores.topk(TOP_K, dim=-1)
            kth = topk_vals[..., -1:]                          # (B, T, 1)
            scores = scores.masked_fill(scores < kth, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ v


# ─────────────────────────────────────────────────────────────────────
#  Everything below is plumbing — identical to train.py.
# ─────────────────────────────────────────────────────────────────────
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([SparseHead(head_size) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class Block(nn.Module):
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
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyLLMSparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB)

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


def train():
    model = TinyLLMSparse().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sparse model: {n_params/1e6:.2f}M params | TOP_K={TOP_K} | device={DEVICE}")
    print(f"Watching {SEED_PROMPT!r} get smarter at each checkpoint.\n")
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
            header = f"=== step {step:>5} | train {train_loss.item():.3f} | val {val_loss.item():.3f} | sparse K={TOP_K} ==="
            print(f"\n{header}\n{sample}\n{'=' * 70}")
            log.write(f"\n{header}\n{sample}\n{'=' * 70}\n")
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
    print(f"\nSparse-trained model saved to {CKPT_NAME} ({size_mb:.1f} MB)")
    print(f"Per-position attention is now O(K)=O({TOP_K}) instead of O(T)=O({BLOCK_SIZE}).")


if __name__ == "__main__":
    train()
