"""
Peek inside your trained kid — see the actual numbers and behavior.

Loads kid.pt (saved by train.py at end of training). If kid.pt doesn't exist,
quick-trains a smaller kid (~90 sec) so you can see the inspection right away.

Run: python show_model.py
"""
import os
import time
import torch
import torch.nn.functional as F
from train import TinyLLM, encode, decode, VOCAB, DEVICE, get_batch

CKPT = "kid.pt"


def quick_train():
    """1500-step training so you have something to inspect even if you haven't run train.py."""
    model = TinyLLM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"No saved model found. Quick-training a kid (1500 steps)...")
    t0 = time.time()
    for step in range(1500):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"  step {step:4d} | loss {loss.item():.3f} | {time.time() - t0:5.1f}s")
    torch.save(model.state_dict(), CKPT)
    print(f"  done in {time.time() - t0:.1f}s. Saved to {CKPT}.\n")
    return model


# ── 1. Load (or quick-train) the model ────────────────────────────────
if os.path.exists(CKPT):
    model = TinyLLM().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f"Loaded kid from {CKPT}.")
else:
    model = quick_train()
model.eval()


# ── 2. Top-level summary ──────────────────────────────────────────────
size_mb = os.path.getsize(CKPT) / 1e6
n_params = sum(p.numel() for p in model.parameters())
print("\n" + "=" * 64)
print(" YOUR KID — top-level facts")
print("=" * 64)
print(f"File on disk:  {CKPT}  ({size_mb:.2f} MB)")
print(f"Total params:  {n_params:,} learned numbers")
print(f"Vocab:         {VOCAB} characters")
print(f"Device:        {DEVICE}")


# ── 3. Where the numbers live (parameter breakdown) ────────────────────
print("\n" + "=" * 64)
print(" WHERE THE NUMBERS LIVE — every layer, every weight")
print("=" * 64)
total = 0
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    shape_str = " × ".join(map(str, p.shape))
    print(f"  {name:48s} ({shape_str:>17}) {n:>9,}")
print(f"  {'TOTAL':48s} {'':>19} {total:>9,}")


# ── 4. A peek at one learned weight ───────────────────────────────────
print("\n" + "=" * 64)
print(" WHAT ONE LEARNED WEIGHT LOOKS LIKE")
print("=" * 64)
for letter in ["R", "e", " "]:
    lid = encode(letter)[0]
    vec = model.tok_emb.weight[lid]
    label = repr(letter).replace(" ", "·")
    print(f"\nThe kid's 'meaning' for letter {label} (first 8 of {vec.numel()} dims):")
    nums = [f"{x:+.3f}" for x in vec[:8].tolist()]
    print(f"  [ {', '.join(nums)}, ... ]")
print("\n(These started as random numbers. Training nudged each one until")
print(" similar letters ended up with similar vectors.)")


# ── 5. What the kid predicts for "ROMEO:" ─────────────────────────────
print("\n" + "=" * 64)
print(" WHAT THE KID PREDICTS COMES NEXT AFTER 'ROMEO:'")
print("=" * 64)
prompt = "ROMEO:"
ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    logits, _ = model(ids)
probs = F.softmax(logits[0, -1], dim=-1)
top_p, top_i = torch.topk(probs, 12)
print(f"\nTop 12 most likely next characters (probability + bar):\n")
for p, i in zip(top_p, top_i):
    char = decode([i.item()]).replace("\n", "\\n").replace(" ", "·").replace("\t", "\\t")
    bar = "█" * int(p.item() * 60)
    print(f"  {char!r:>6s}  {p.item() * 100:5.2f}%  {bar}")


# ── 6. One attention head's actual viewing pattern ────────────────────
print("\n" + "=" * 64)
print(" ONE ATTENTION HEAD'S VIEW")
print("=" * 64)
inspect_prompt = "ROMEO: To be"
ids = torch.tensor([encode(inspect_prompt)], dtype=torch.long, device=DEVICE)
B, T = ids.shape
with torch.no_grad():
    pos = torch.arange(T, device=DEVICE)
    x = model.tok_emb(ids) + model.pos_emb(pos)
    x_n = model.blocks[0].ln1(x)
    head = model.blocks[0].attn.heads[0]   # block 0, head 0
    q = head.query(x_n)
    k = head.key(x_n)
    scores = q @ k.transpose(-2, -1) / (head.head_size ** 0.5)
    scores = scores.masked_fill(head.mask[:T, :T] == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)

print(f"\nProcessing prompt: {inspect_prompt!r}")
print("Block 1, Head 1 — attention weights as percentages.")
print("Each row = current position. Columns = positions it looked at.")
print("Numbers are % of attention. Empty cells = ~0% (or future = masked).\n")

display_chars = [c.replace(" ", "·").replace("\n", "↵") for c in inspect_prompt]
header = "          " + "".join(f"{c:>5}" for c in display_chars)
print(header)
for i, c in enumerate(display_chars):
    row = weights[0, i].tolist()
    cells = "".join(f"{w * 100:5.0f}" if w > 0.005 else "    ." for w in row)
    print(f"  ({i:2d}) {c}  {cells}")

print("\n(Notice the lower-triangular shape: positions can only look BACKWARD.")
print(" That's the causal mask. Where the kid concentrates attention tells you")
print(" what kind of pattern this head learned to detect.)")


print("\n" + "=" * 64)
print(" That's your kid. Every number above is one of the 825,000 dials.")
print("=" * 64)
