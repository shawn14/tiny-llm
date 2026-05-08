"""
Export the trained kid's weights, intermediates, and training trace to JSON
for the peek.school web tutorial to consume.

Run: python export_for_web.py

Outputs (to ~/projects/peek/public/data/):
  vocab.json          — list of 65 chars in order
  tok_emb.json        — 65 × 128 token embedding matrix
  pos_emb.json        — 128 × 128 position embedding matrix
  lm_head.json        — { weight: 65 × 128, bias: 65 } output projection
  hidden_state.json   — { prompt: "ROMEO", state: 128 numbers }
  predictions.json    — { prompt, logits[65], probs[65], top: [{char, prob}] }
  attention.json      — attention weights for "ROMEO: To be" across all heads/layers
  training.json       — parsed samples.txt: per-checkpoint step, losses, sample
  data_sample.json    — first ~1500 chars of input.txt + a worked (input, target) pair
  meta.json           — model architecture summary
"""
import os
import re
import json
import torch
import torch.nn.functional as F
from train import TinyLLM, encode, decode, VOCAB, DEVICE, BLOCK_SIZE

OUT = os.path.expanduser("~/projects/peek/public/data")
os.makedirs(OUT, exist_ok=True)

CKPT = "kid.pt"
if not os.path.exists(CKPT):
    raise SystemExit("Run train.py first to create kid.pt")

model = TinyLLM().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()


def round_mat(m, d=4):
    return [[round(v, d) for v in row] for row in m]


def round_vec(v, d=4):
    return [round(x, d) for x in v]


# 1. vocab
chars = sorted(set(open("input.txt").read()))
with open(f"{OUT}/vocab.json", "w") as f:
    json.dump(chars, f)
print(f"  vocab.json          {len(chars)} chars")

# 2. token embeddings
tok_emb = model.tok_emb.weight.detach().cpu().tolist()
with open(f"{OUT}/tok_emb.json", "w") as f:
    json.dump(round_mat(tok_emb), f)
print(f"  tok_emb.json        {len(tok_emb)} × {len(tok_emb[0])}")

# 3. position embeddings (128 × 128)
pos_emb = model.pos_emb.weight.detach().cpu().tolist()
with open(f"{OUT}/pos_emb.json", "w") as f:
    json.dump(round_mat(pos_emb), f)
print(f"  pos_emb.json        {len(pos_emb)} × {len(pos_emb[0])}")

# 4. lm_head weight + bias
lm_w = model.lm_head.weight.detach().cpu().tolist()
lm_b = model.lm_head.bias.detach().cpu().tolist()
with open(f"{OUT}/lm_head.json", "w") as f:
    json.dump({"weight": round_mat(lm_w), "bias": round_vec(lm_b)}, f)
print(f"  lm_head.json        weight {len(lm_w)} × {len(lm_w[0])}, bias {len(lm_b)}")

# 5. hidden state for "ROMEO" — the 128-dim vector right before lm_head
prompt = "ROMEO"
ids = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    pos = torch.arange(ids.shape[1], device=DEVICE)
    x = model.tok_emb(ids) + model.pos_emb(pos)
    for blk in model.blocks:
        x = blk(x)
    x = model.ln_f(x)
    hidden = x[0, -1].detach().cpu().tolist()
with open(f"{OUT}/hidden_state.json", "w") as f:
    json.dump({"prompt": prompt, "state": round_vec(hidden)}, f)
print(f"  hidden_state.json   prompt={prompt!r}, state of {len(hidden)} numbers")

# 6. precomputed predictions for "ROMEO"
with torch.no_grad():
    logits, _ = model(ids)
    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=-1)

logits_list = round_vec(last_logits.tolist())
probs_list = [round(v, 6) for v in probs.tolist()]

top_p, top_i = torch.topk(probs, 12)
top = []
for p, i in zip(top_p, top_i):
    top.append({
        "char": chars[i.item()],
        "id": i.item(),
        "logit": round(last_logits[i.item()].item(), 4),
        "prob": round(p.item(), 6),
    })

with open(f"{OUT}/predictions.json", "w") as f:
    json.dump({
        "prompt": prompt,
        "logits": logits_list,
        "probs": probs_list,
        "top": top,
    }, f)
print(f"  predictions.json    top: {[(t['char'], round(t['prob']*100, 1)) for t in top[:5]]}")

# 7. attention weights for "ROMEO: To be" — every head, every layer
attn_prompt = "ROMEO: To be"
ids = torch.tensor([encode(attn_prompt)], dtype=torch.long, device=DEVICE)
B, T = ids.shape

with torch.no_grad():
    pos = torch.arange(T, device=DEVICE)
    x = model.tok_emb(ids) + model.pos_emb(pos)

    layers = []
    for blk in model.blocks:
        x_n = blk.ln1(x)
        heads = []
        for head in blk.attn.heads:
            q = head.query(x_n)
            k = head.key(x_n)
            scores = q @ k.transpose(-2, -1) / (head.head_size ** 0.5)
            scores = scores.masked_fill(head.mask[:T, :T] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            heads.append([round_vec(row, 4) for row in weights[0].tolist()])
        layers.append(heads)
        # advance x through the full block so deeper layers see the right input
        x = x + blk.attn(blk.ln1(x))
        x = x + blk.ffwd(blk.ln2(x))

with open(f"{OUT}/attention.json", "w") as f:
    json.dump({
        "prompt": attn_prompt,
        "tokens": list(attn_prompt),
        "layers": layers,
    }, f)
print(f"  attention.json      prompt={attn_prompt!r}, "
      f"{len(layers)} layers × {len(layers[0])} heads × {T}×{T}")

# 8. training trace — parse samples.txt into structured per-checkpoint records
samples_path = "samples.txt"
checkpoints = []
if os.path.exists(samples_path):
    raw = open(samples_path).read()
    # blocks look like: "=== step  X | train Y | val Z ===\n<sample>\n===...===\n"
    pattern = re.compile(
        r"=== step\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) ===\n(.*?)\n=+",
        re.DOTALL,
    )
    for match in pattern.finditer(raw):
        step, train_loss, val_loss, sample = match.groups()
        checkpoints.append({
            "step": int(step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "sample": sample,
        })
with open(f"{OUT}/training.json", "w") as f:
    json.dump({"checkpoints": checkpoints, "seed_prompt": "ROMEO:"}, f)
print(f"  training.json       {len(checkpoints)} checkpoints "
      f"(loss {checkpoints[0]['train_loss']:.2f} → {checkpoints[-1]['train_loss']:.2f})"
      if checkpoints else "  training.json       (no samples.txt found)")

# 9. raw data snippet + a worked (input, target) example
text = open("input.txt").read()
snippet = text[:1500]
example_input = text[:32]
example_target = text[1:33]
with open(f"{OUT}/data_sample.json", "w") as f:
    json.dump({
        "snippet": snippet,
        "total_chars": len(text),
        "example": {
            "input": example_input,
            "target": example_target,
            "input_ids": encode(example_input),
            "target_ids": encode(example_target),
        },
    }, f)
print(f"  data_sample.json    {len(text):,} chars total, snippet of {len(snippet)}")

# 10. metadata about the model — augmented with parameter inventory
param_inventory = []
for name, p in model.named_parameters():
    param_inventory.append({
        "name": name,
        "shape": list(p.shape),
        "n": p.numel(),
    })

with open(f"{OUT}/meta.json", "w") as f:
    json.dump({
        "n_params": sum(p.numel() for p in model.parameters()),
        "vocab_size": VOCAB,
        "n_embd": model.tok_emb.weight.shape[1],
        "n_layer": len(model.blocks),
        "n_head": len(model.blocks[0].attn.heads),
        "block_size": model.pos_emb.weight.shape[0],
        "device_trained": str(DEVICE),
        "params": param_inventory,
    }, f)
# 11. all transformer block weights + final LayerNorm — for browser inference
final_ln = {
    "gain": round_vec(model.ln_f.weight.detach().cpu().tolist()),
    "bias": round_vec(model.ln_f.bias.detach().cpu().tolist()),
}

blocks_out = []
for blk in model.blocks:
    heads_out = []
    for head in blk.attn.heads:
        # PyTorch nn.Linear stores weight as [out_features, in_features].
        # head.query/key/value are Linear(N_EMBD, head_size, bias=False),
        # so weight shape is [head_size, N_EMBD] = [32, 128].
        heads_out.append({
            "q": round_mat(head.query.weight.detach().cpu().tolist()),
            "k": round_mat(head.key.weight.detach().cpu().tolist()),
            "v": round_mat(head.value.weight.detach().cpu().tolist()),
        })
    # blk.attn.proj is Linear(N_EMBD, N_EMBD) — has bias by default.
    # blk.ffwd is Sequential(Linear(128,512), ReLU, Linear(512,128)) — both have bias.
    blocks_out.append({
        "ln1": {
            "gain": round_vec(blk.ln1.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.ln1.bias.detach().cpu().tolist()),
        },
        "heads": heads_out,
        "proj": {
            "weight": round_mat(blk.attn.proj.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.attn.proj.bias.detach().cpu().tolist()),
        },
        "ln2": {
            "gain": round_vec(blk.ln2.weight.detach().cpu().tolist()),
            "bias": round_vec(blk.ln2.bias.detach().cpu().tolist()),
        },
        "mlp": {
            "fc1": round_mat(blk.ffwd[0].weight.detach().cpu().tolist()),
            "b1":  round_vec(blk.ffwd[0].bias.detach().cpu().tolist()),
            "fc2": round_mat(blk.ffwd[2].weight.detach().cpu().tolist()),
            "b2":  round_vec(blk.ffwd[2].bias.detach().cpu().tolist()),
        },
    })

with open(f"{OUT}/blocks.json", "w") as f:
    json.dump({"final_ln": final_ln, "blocks": blocks_out}, f)

n_blocks = len(blocks_out)
n_heads  = len(blocks_out[0]["heads"])
print(f"  blocks.json         {n_blocks} blocks × {n_heads} heads + final_ln")

print(f"\nDone. Files in {OUT}/")
