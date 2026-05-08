"""
Stage 2: Supervised Fine-Tuning (SFT).

Take the pretrained kid (kid_modern_t1024.pt) and continue training on
instruction-response pairs from Alpaca. After SFT the kid should respond
to instructions instead of just continuing text.

What's NEW vs pretraining (the things you didn't have to think about before):
  1. Loss masking — we only train on response tokens. The instruction tokens
     are masked to -100 so cross_entropy ignores them. (We don't want the
     model learning to write the user's question.)
  2. Lower LR — start at 2e-5, ~15× lower than pretraining. The model is
     already trained; we're nudging it.
  3. Linear LR decay — no warmup needed; SFT is shorter and the model is
     already in a good region of weight space.
  4. No weight decay — typical for SFT to avoid pulling weights toward zero.
  5. Chat template — Alpaca-style:
        ### Instruction:
        {instruction}
        ### Response:
        {output}
  6. Pre-tokenize entire dataset — Alpaca is small enough to fit in memory.
     Different from pretraining where we sampled chunks from a giant corpus.

Run:
    pip install datasets         # if not already
    python train_sft.py          # uses kid_modern_t1024.pt as base
"""
import argparse
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- CLI --------
parser = argparse.ArgumentParser()
parser.add_argument("--base",       type=str,   default="kid_modern_t1024.pt",
                    help="pretrained checkpoint to start from")
parser.add_argument("--corpus",     type=str,   default="corpus.txt",
                    help="text corpus that defines the vocab (must match pretraining)")
parser.add_argument("--steps",      type=int,   default=2000)
parser.add_argument("--lr",         type=float, default=2e-5)
parser.add_argument("--batch-size", type=int,   default=8)
parser.add_argument("--block-size", type=int,   default=512,
                    help="context length for SFT (shorter than pretraining is fine)")
parser.add_argument("--max-examples", type=int, default=None,
                    help="limit dataset size for quick tests")
args = parser.parse_args()

# -------- Architecture (MUST match train_modern.py) --------
N_EMBD     = 384
N_LAYER    = 8
N_HEAD     = 8
N_KV_HEAD  = 4
HEAD_DIM   = N_EMBD // N_HEAD
MLP_HIDDEN = 1024
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size
LR         = args.lr
WD         = 0.0       # no weight decay during SFT
MAX_STEPS  = args.steps

CKPT_NAME    = "kid_sft.pt"
SAMPLES_NAME = "samples_sft.txt"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_BF16 = (DEVICE in ("cuda", "mps")) and hasattr(torch, "bfloat16")
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

# -------- Vocab (same as pretraining; built from corpus.txt) --------
if not os.path.exists(args.corpus):
    sys.exit(f"Corpus not found: {args.corpus}. Need same one used for pretraining.")

print(f"Loading vocab from {args.corpus}")
text = open(args.corpus).read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
print(f"  Vocab: {VOCAB} chars")

def encode(s: str) -> list[int]:
    """Encode a string, dropping any chars not in vocab."""
    return [stoi[c] for c in s if c in stoi]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids if i in itos)


# -------- Model classes (duplicated from train_modern.py for self-containment) --------

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


class GQAttention(nn.Module):
    def __init__(self):
        super().__init__()
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
        q = self.q_proj(x).view(B, T, N_HEAD,    HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV_HEAD, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        repeats = N_HEAD // N_KV_HEAD
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, N_HEAD * HEAD_DIM)
        return self.o_proj(out)


class Block(nn.Module):
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
        # ignore_index=-100 → masked instruction tokens skip the loss
        loss = F.cross_entropy(
            logits.view(-1, VOCAB),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, stop_seq=None):
        for _ in range(max_new):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_seq:
                tail = decode(idx[0, -len(stop_seq):].tolist())
                if tail.endswith(stop_seq):
                    break
        return idx


# -------- Load pretrained weights --------

if not os.path.exists(args.base):
    sys.exit(
        f"Pretrained checkpoint not found: {args.base}\n"
        f"Run `python train_modern.py` first to produce it."
    )

print(f"\nLoading pretrained model: {args.base}")
model = ModernKid().to(DEVICE)
state = torch.load(args.base, map_location=DEVICE)

# Drop block-size-dependent buffers (cos, sin, attention masks).
# These are deterministic and the new model rebuilt them at the SFT block
# size in __init__, so we just skip them when loading. Only learned weights
# need to come from the pretrained checkpoint.
for k in list(state.keys()):
    if k in ("cos", "sin") or k.endswith(".attn.mask"):
        del state[k]
missing, unexpected = model.load_state_dict(state, strict=False)
# Sanity: missing should only be the buffers we deliberately skipped
print(f"  Loaded weights. (Skipped buffers: {len(missing)} expected, 0 unexpected)")
assert len(unexpected) == 0, f"Unexpected keys in checkpoint: {unexpected}"

if USE_BF16:
    model = model.to(DTYPE)

n_params = sum(p.numel() for p in model.parameters())
print(f"  {n_params/1e6:.2f}M params loaded")
print(f"  Device: {DEVICE}  Dtype: {DTYPE}")


# -------- Load + format Alpaca dataset --------

print("\nLoading Alpaca dataset (~52K instruction-response pairs)...")
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Need `datasets`: pip install datasets")

ds = load_dataset("tatsu-lab/alpaca", split="train")
print(f"  Loaded {len(ds)} examples")


def format_example(ex) -> tuple[str, str]:
    """Build the (prompt, response) pair using Alpaca chat template."""
    instruction = ex["instruction"]
    inp = ex.get("input", "") or ""
    output = ex["output"]
    if inp.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    response = output + "\n"   # newline acts as our soft EOS
    return prompt, response


def tokenize_with_mask(prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Return (input_ids, target_ids) of length BLOCK_SIZE.

    Targets are shifted by one (standard LM training: predict next token)
    AND masked: positions whose target is a prompt token get -100, which
    cross_entropy(ignore_index=-100) skips. So we only train on the model
    predicting the response tokens, not the prompt.
    """
    prompt_ids   = encode(prompt)
    response_ids = encode(response)
    full = prompt_ids + response_ids

    # Truncate to BLOCK_SIZE+1 (we'll shift by 1)
    if len(full) > BLOCK_SIZE + 1:
        full = full[:BLOCK_SIZE + 1]

    input_ids  = full[:-1]
    target_ids = full[1:]

    # Mask all positions whose target is in the prompt.
    # If prompt has P tokens, the prompt occupies positions 0..P-1 of `full`.
    # In `target_ids` (= full[1:]), prompt tokens are at positions 0..P-2.
    P = min(len(prompt_ids), len(input_ids) + 1)
    masked = [-100] * (P - 1) + target_ids[P - 1:]

    # Right-pad to BLOCK_SIZE
    pad_len = BLOCK_SIZE - len(input_ids)
    if pad_len > 0:
        input_ids  = input_ids  + [0] * pad_len
        masked     = masked     + [-100] * pad_len

    return input_ids[:BLOCK_SIZE], masked[:BLOCK_SIZE]


print("\nTokenizing dataset (this is also where we mask instruction tokens)...")
all_inputs:  list[list[int]] = []
all_targets: list[list[int]] = []
skipped = 0
limit = args.max_examples if args.max_examples else len(ds)
for i, ex in enumerate(ds):
    if i >= limit:
        break
    if not ex["output"].strip():
        skipped += 1
        continue
    prompt, response = format_example(ex)
    ids, tgt = tokenize_with_mask(prompt, response)
    # Skip examples whose entire target is masked (response truncated away)
    if all(t == -100 for t in tgt):
        skipped += 1
        continue
    all_inputs.append(ids)
    all_targets.append(tgt)

print(f"  Tokenized {len(all_inputs)} examples (skipped {skipped})")

inputs_tensor  = torch.tensor(all_inputs,  dtype=torch.long)
targets_tensor = torch.tensor(all_targets, dtype=torch.long)


def get_batch():
    ix = torch.randint(0, len(inputs_tensor), (BATCH_SIZE,))
    return inputs_tensor[ix].to(DEVICE), targets_tensor[ix].to(DEVICE)


# -------- LR schedule (linear decay, no warmup) --------

def linear_lr(step):
    return LR * (1.0 - step / MAX_STEPS) + LR * 0.1 * (step / MAX_STEPS)


# -------- Test prompts to watch the kid learn to follow instructions --------

TEST_PROMPTS = [
    "What is the capital of France?",
    "Write a one-sentence summary of photosynthesis.",
    "List three primary colors.",
    "Explain why the sky is blue.",
]

def render_test_prompt(p: str) -> str:
    return f"### Instruction:\n{p}\n\n### Response:\n"


# -------- Train --------

def train():
    print(f"\nSFT training:")
    print(f"  Steps:    {MAX_STEPS}")
    print(f"  LR:       {LR} (linear decay to 10%)")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  Block:    {BLOCK_SIZE}\n")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WD,
    )

    log = open(SAMPLES_NAME, "w")
    checkpoints = sorted(set([0, 50, 200, 500, 1000, MAX_STEPS]))

    for step in range(MAX_STEPS + 1):
        # LR
        lr_now = linear_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # Periodic eval: generate responses to test prompts
        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch()
                _, val_loss = model(xb, yb)
                samples = []
                for tp in TEST_PROMPTS:
                    prompt = render_test_prompt(tp)
                    seed_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
                    out = model.generate(seed_ids, 100, stop_seq="### ")
                    response = decode(out[0, len(seed_ids[0]):].tolist())
                    samples.append(f"Q: {tp}\nA: {response.strip()}")
                sample_block = "\n\n".join(samples)
            header = (
                f"=== step {step:>5} | val {val_loss.item():.3f} "
                f"| lr {lr_now:.2e} | sft ==="
            )
            print(f"\n{header}\n{sample_block}\n{'=' * 70}")
            log.write(f"\n{header}\n{sample_block}\n{'=' * 70}\n")
            log.flush()
            model.train()

        # Train step
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log.close()
    torch.save(model.state_dict(), CKPT_NAME)
    size_mb = os.path.getsize(CKPT_NAME) / 1e6
    print(f"\nSFT'd kid saved to {CKPT_NAME} ({size_mb:.1f} MB)")
    print(f"Sample log written to {SAMPLES_NAME}")


if __name__ == "__main__":
    train()
