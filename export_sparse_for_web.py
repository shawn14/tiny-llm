"""
Export a SPARSE-trained kid's weights to peek/public/data/sparse{...}/ for
the /sparse browser page.

Run: python export_sparse_for_web.py                            # default → /data/sparse/
     python export_sparse_for_web.py --k 4                       # K=4 → /data/sparse_k4/
     python export_sparse_for_web.py --k 16                      # K=16 → /data/sparse_k16/
     python export_sparse_for_web.py --block-size 512 --k 4     # T=512 K=4 → /data/sparse_t512_k4/
"""
import argparse
import os
import json
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=8)
parser.add_argument("--block-size", type=int, default=128)
args = parser.parse_args()
TOP_K = args.k
T = args.block_size

# Pass through to train_sparse so its module-level CLI parses the same way.
sys.argv = ["train_sparse.py", "--k", str(TOP_K), "--block-size", str(T)]

from train_sparse import TinyLLMSparse, VOCAB, DEVICE, BLOCK_SIZE

# Suffix matching the train_sparse output names.
def _suffix() -> str:
    parts = []
    if T != 128:
        parts.append(f"t{T}")
    if TOP_K != 8:
        parts.append(f"k{TOP_K}")
    return ("_" + "_".join(parts)) if parts else ""

SUFFIX = _suffix()
CKPT = f"kid_sparse{SUFFIX}.pt"
OUT = os.path.expanduser(f"~/projects/peek/public/data/sparse{SUFFIX}")
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(CKPT):
    raise SystemExit(
        f"Run `python train_sparse.py --block-size {T} --k {TOP_K}` first to create {CKPT}"
    )

model = TinyLLMSparse().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()


def round_mat(m, d=4):
    return [[round(v, d) for v in row] for row in m]


def round_vec(v, d=4):
    return [round(x, d) for x in v]


# 1. vocab — copy of the main one (same training data)
chars = sorted(set(open("input.txt").read()))
with open(f"{OUT}/vocab.json", "w") as f:
    json.dump(chars, f)
print(f"  vocab.json          {len(chars)} chars")

# 2. token embeddings
tok_emb = model.tok_emb.weight.detach().cpu().tolist()
with open(f"{OUT}/tok_emb.json", "w") as f:
    json.dump(round_mat(tok_emb), f)
print(f"  tok_emb.json        {len(tok_emb)} × {len(tok_emb[0])}")

# 3. position embeddings
pos_emb = model.pos_emb.weight.detach().cpu().tolist()
with open(f"{OUT}/pos_emb.json", "w") as f:
    json.dump(round_mat(pos_emb), f)
print(f"  pos_emb.json        {len(pos_emb)} × {len(pos_emb[0])}")

# 4. lm_head
lm_w = model.lm_head.weight.detach().cpu().tolist()
lm_b = model.lm_head.bias.detach().cpu().tolist()
with open(f"{OUT}/lm_head.json", "w") as f:
    json.dump({"weight": round_mat(lm_w), "bias": round_vec(lm_b)}, f)
print(f"  lm_head.json        weight {len(lm_w)} × {len(lm_w[0])}, bias {len(lm_b)}")

# 5. transformer blocks + final LayerNorm — same schema as blocks.json from the dense kid
final_ln = {
    "gain": round_vec(model.ln_f.weight.detach().cpu().tolist()),
    "bias": round_vec(model.ln_f.bias.detach().cpu().tolist()),
}

blocks_out = []
for blk in model.blocks:
    heads_out = []
    for head in blk.attn.heads:
        heads_out.append({
            "q": round_mat(head.query.weight.detach().cpu().tolist()),
            "k": round_mat(head.key.weight.detach().cpu().tolist()),
            "v": round_mat(head.value.weight.detach().cpu().tolist()),
        })
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
print(f"  blocks.json         {len(blocks_out)} blocks × {len(blocks_out[0]['heads'])} heads + final_ln")

# 6. metadata
param_inventory = []
for name, p in model.named_parameters():
    param_inventory.append({
        "name": name,
        "shape": list(p.shape),
        "n": p.numel(),
    })

# TRAINED_TOP_K is just TOP_K from this script's argv — train_sparse may have
# loaded a different value into its own globals depending on import timing.
TRAINED_TOP_K = TOP_K

with open(f"{OUT}/meta.json", "w") as f:
    json.dump({
        "n_params": sum(p.numel() for p in model.parameters()),
        "vocab_size": VOCAB,
        "n_embd": model.tok_emb.weight.shape[1],
        "n_layer": len(model.blocks),
        "n_head": len(model.blocks[0].attn.heads),
        "block_size": model.pos_emb.weight.shape[0],
        "device_trained": str(DEVICE),
        "trained_top_k": TRAINED_TOP_K,
        "params": param_inventory,
    }, f)
print(f"\nDone. Sparse files in {OUT}/  (trained with K={TRAINED_TOP_K})")
