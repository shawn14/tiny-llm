# Improving the kid — a journey log

A real, honest record of taking the standard Karpathy nanoGPT-style char-level
Shakespeare model and trying to make it better. Same prompt every time
(`ROMEO:`), same architecture family, same 65-character vocabulary. What
changes is the training discipline, the architecture lineage, and the data.

This document is meant to be read top-to-bottom. Each version is a single
small idea (or set of ideas) and the numbers + samples that came out of it.

## v1 — vanilla nanoGPT (the starting point)

`train.py`. ~825K params. 4 transformer blocks, 4 heads, 128-dim embeddings.
LayerNorm. ReLU MLP. Learned positional embeddings. AdamW at constant 3e-4.
5,000 steps, no regularization, no LR schedule, no gradient clipping, no
dropout, no weight tying. Save the last-step model.

**Result.** Train loss `1.476`, val loss `1.741`. Train/val gap `+0.265`.
That gap is the bug — the model is starting to memorize Tiny Shakespeare,
not learning the patterns. (Tiny Shakespeare is ~1.1MB; at batch 32 ×
block 128 × 5,000 steps, the kid sees ~20× the dataset. Of course it
overfits.) The val loss number on its own looks fine. The *gap* is the
diagnostic.

## v2 — Tier 1: training discipline

`train_v2.py`. Same architecture, seven changes to how training is run:

| Tag | Change | Why |
|---|---|---|
| `T1#1` | Dropout 0.2 on attention output and MLP output | regularize an overfit small model |
| `T1#2` | Cosine LR schedule, linear warmup 100 steps, peak 3e-4 → floor 3e-5 | stable early, slow finish |
| `T1#3` | Tie `lm_head.weight = tok_emb.weight` | regularizer + saves 8,320 params |
| `T1#4` | AdamW param groups — decay only on 2D matmul weights | don't decay biases or LayerNorm gains |
| `T1#5` | `clip_grad_norm_(..., 1.0)` | insurance against rare loss spikes |
| `T1#6` | Track best-val checkpoint, save when val improves | ship the right model, not the last one |
| `T1#7` | Finer early checkpoints `[0, 10, 25, 50, 100, ...]` | first-200-steps drama is where the most-teachable transitions happen |

**Bug along the way.** First v2 run produced step-0 loss of **82** instead of
the expected `log(65) ≈ 4.17`. Cause: tying embeddings means
`lm_head.weight = tok_emb.weight`, but PyTorch's default `nn.Embedding` init
is `N(0, 1)` — way larger than `nn.Linear`'s default. After tying, logits
exploded. The fix is the standard nanoGPT init pass: `N(0, 0.02)` on every
Linear/Embedding weight, zero biases. With that added, step-0 loss came
back to `4.193`. This is the *single most common bug* introduced when
adding weight tying to nanoGPT. Documenting it because it's instructive.

**Result.** Train `1.623`, val `1.718`, gap `+0.095`.

```
ROMEO:
To remolo Richmond ence shemble god me been thee the my
all be not heart 'Till be Romes.
```

The val loss barely moved (−0.023), but the **gap closed by 64%** (0.265 →
0.095). Train loss went *up* — which is the signature of regularization
working. The model is no longer memorizing; it's generalizing. Sample text
quality jumped accordingly: Richmond is real, "be not heart" parses,
"What rest" parses. Loss numbers don't capture that.

## v3 — Tier 2: modernize the architecture

`train_v3.py`. Keep all of v2's discipline. Swap three architecture pieces
to bring the kid from "2017 nanoGPT" to "2023 Llama":

| Tag | Change | Why |
|---|---|---|
| `T2#1` | ReLU → GELU in MLP | smoother activation; GPT-2 onward |
| `T2#2` | LayerNorm → RMSNorm | simpler, fewer params, equal/better quality |
| `T2#3` | Learned `pos_emb` → RoPE | zero-param positions via Q/K rotation |

RoPE is the interesting one. Instead of adding a learned 128-dim "where am
I" vector to the input embedding (16,384 weights of position lookup), every
(Q, K) pair gets *rotated* by an angle proportional to its position in the
sequence — different rotation rates per dimension pair. Two positions m
and n produce a dot-product that depends only on (m − n), so attention
naturally encodes relative position with **zero learned parameters**.

**Param count went DOWN** (799,041 vs v2's 816,577), because dropping
`pos_emb` saved 16,384 weights and LayerNorm→RMSNorm saved another ~1,150.

**Result.** Train `1.539`, val `1.625`, gap `+0.086`.

```
ROMEO:
But you tell the woe!

SICINIUS:
How north to go.

KING EDWARD IV:
But stands now?

CORIOLANUS:
Then of Volsceta,
```

Real character names from real plays. Proper scene structure. The qualitative
jump is bigger than the loss number suggests. Tier-2 architecture (−0.093)
beat Tier-1 discipline (−0.023) by ~4×. RoPE is doing most of that work.

## v3-long — same architecture, more patience

`train_v3_long.py`. v3's val loss was still descending at step 5000 and
its cosine schedule had cooled LR to 3e-5 — the model was crawling at the
end. Same architecture, two constants changed:

- `MAX_STEPS`: 5,000 → 10,000
- `MIN_LR`: 3e-5 → 5e-5

**Result.** Best val `1.551` at step **9,000** (val turned back up to
`1.570` by step 10,000 — first time `T1#6`'s "save best-val checkpoint"
actually mattered).

```
ROMEO:
Their oer-casters?

POLIXENES:
How now, tell him? What, Lord Starful close
And I foot so draw Lord Hastings, and my lord blood:
```

`POLIXENES` (Winter's Tale), `Lord Hastings` (Richard III). "How now, tell
him?" is a perfectly valid Shakespearean question. The kid is firmly inside
the world of the plays now.

## Scoreboard so far

| | Steps | Params | Val loss | Gap | Best step |
|---|---|---|---|---|---|
| v1 vanilla | 5,000 | 824,897 | 1.741 | +0.265 | last |
| v2 +regularization | 5,000 | 816,577 | 1.718 | +0.095 | last |
| v3 +Llama arch | 5,000 | 799,041 | 1.625 | +0.086 | last |
| **v3-long** same arch, longer | 10,000 | 799,041 | **1.551** | +0.218 | **9,000** |

Cumulative: **−10.9% val loss with 3.1% fewer parameters.** The kid 2024
edition is a smaller, more disciplined, architecturally-modern model that
trains better than its 2017 ancestor.

But by step 9,000 the train/val gap has widened back to `+0.218`. We're
hitting the natural ceiling of what 800K params can extract from 1.1MB of
training text. To go further we need *capacity* (bigger model) and *more
text* (so the bigger model has new patterns to find instead of memorizing
old ones). That's v4.

## Detour: a full disk

Real data point. Halfway through this work the Mac's data volume was
`135 MiB free` of `228 GiB` and the CLI's per-command output buffer
couldn't write — anything I tried to run failed with `ENOSPC`. Cleanup
priorities, ranked by safety: `~/Library/Developer/Xcode/DerivedData`
(rebuilds), Codex/Playwright/SwiftPM caches (regenerate), old Electron
auto-updater leftovers. Recovered ~8.5 GiB without touching anything user
data could plausibly live in. **Do not** touch `~/Library/Developer/Xcode/Archives`
(App Store crash symbolication) or `Wispr Flow` (audio recordings).

## v4 — bigger kid, more data (in progress)

Two changes layered together:

**Bigger architecture.** `N_EMBD` 128 → 192, `N_LAYER` 4 → 6, `N_HEAD` 4 → 6.
~2.7M params (3.4× v3-long). Same RoPE / RMSNorm / GELU / Tier-1 discipline.

**More data.** `prep_corpus.py` downloads:

- Tiny Shakespeare (anchor — defines the canonical 65-char vocab)
- Complete Works of Shakespeare (Project Gutenberg #100)
- Plays of Christopher Marlowe (Project Gutenberg #779)

…strips Gutenberg headers/footers, filters every character to the anchor
vocab so the existing tokenizer/decoder still works without changes. Actual
result of the prep step:

```
Anchor vocab: 65 chars from input.txt (1.12M chars)
complete_shakespeare.txt       raw  5.38M → in-vocab  5.31M  ( 1.0% dropped)
marlowe_plays.txt              raw  0.14M → in-vocab  0.12M  ( 2.2% dropped)
Wrote input_v4.txt: 6.54M chars  (5.9× tiny shakespeare)
Vocab preserved: True
```

Only 1% of Gutenberg's text drops (curly quotes, em-dashes — meaning
preserved, vocabulary intact). Marlowe's contribution is small (2% of
the corpus) but stylistically diverse — same era, different ear.

The params/char ratio drops from `0.72` (v3-long) to `0.41` (v4) — the
new kid is *less* over-parameterized relative to its data, which is
exactly the direction we want for the overfit story.

**Result.** Train `1.264`, **val `1.456`** at step **15,000**. Val drifted
back up to `1.460` by step 20,000 — third instance of `T1#6` paying
for itself.

```
ROMEO:
And her, sir?

DUMAINE.
Madam, you that Duke Words.

ANTONIO:
How now! yea?

GLOUCESTER:
It came to Biondello?

Lord:
Come; come, good lord, but yet a good wa
```

The qualitative leap is bigger than any prior step. Multiple
character names from across the corpus appear in *the same scene* —
DUMAINE (*Love's Labour's Lost*), ANTONIO (many plays), GLOUCESTER
(*Lear/Henry VI/Richard III*), and Biondello (*Taming of the Shrew*),
referenced by name *inside the dialogue*. Short complete utterances
like "How now! yea?" and "It came to Biondello?" parse as valid
Shakespearean exchanges. The kid is now recalling phrases it has read,
not just reinventing morphology character-by-character.

Worth noting: the train/val gap (+0.192) is still wide despite 5.9×
more text. The bigger model has the capacity to memorize even Complete
Shakespeare. To extract more juice from here would mean either more
dropout (0.3), more text (Webster, Jonson, Middleton — the rest of
the Elizabethan dramatists), or a v5 that's bigger still. Diminishing
returns are real but not exhausted.

## Final scoreboard

| | Steps | Params | Val loss | Train loss | Gap | Best step |
|---|---|---|---|---|---|---|
| v1 vanilla | 5,000 | 824,897 | 1.741 | 1.476 | +0.265 | last |
| v2 +regularization | 5,000 | 816,577 | 1.718 | 1.623 | +0.095 | last |
| v3 +Llama arch | 5,000 | 799,041 | 1.625 | 1.539 | +0.086 | last |
| v3-long same arch, longer | 10,000 | 799,041 | 1.551 | 1.333 | +0.218 | 9,000 |
| **v4 bigger + more data** | 20,000 | 2,676,161 | **1.456** | 1.264 | +0.192 | **15,000** |

Cumulative: **val loss 1.741 → 1.456 (−16.4%)** with the same 65-char
vocabulary, same one-file training script, model still small enough
to fit in a browser tab.

Per-step impact:

- v1 → v2: −0.023 val (Tier-1 training discipline). Closed train/val gap by 64%. Made every later improvement count by removing the noise of an overfit baseline.
- v2 → v3: −0.093 val (Tier-2 architecture). RoPE + RMSNorm + GELU. **The single biggest single-change-set jump**, with FEWER parameters.
- v3 → v3-long: −0.074 val (patience). Same code, more steps, slower cooldown. T1#6 paid for itself.
- v3-long → v4: −0.095 val (scale). Bigger model + 5.9× more text. Largest absolute jump in the arc.

## What I'd do next

This is a good place to stop training and start polishing. Three productive
follow-ups:

1. **Re-export weights for the browser.** `export_for_web.py` currently
   handles the v1 architecture. Extending it to RoPE buffers + tied
   embeddings would let the playground load any of v2/v3/v3-long/v4 — a
   true side-by-side demo nobody else's char-LLM tutorial offers.
2. **Write the new chapters.** `/regularization` (v1→v2 story),
   rewrite `/position` around RoPE (v2→v3 story), `/evolution` as the
   meta-chapter that unifies the arc.
3. **One ablation.** Drop just RoPE into v2 (leave LayerNorm + ReLU)
   to confirm RoPE is doing most of v3's heavy lifting. Maybe ~0.06
   of v3's 0.09 gain. Worth knowing for the chapter prose.

