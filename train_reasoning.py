"""
train_reasoning.py — GRPO reasoning training on GSM8K (Mac-friendly)

Trains a base model with LoRA via Group Relative Policy Optimization (GRPO),
the algorithm DeepSeek used to bootstrap R1's reasoning behavior. Works for
both dense models (Qwen, Llama) and MoE models (OLMoE, Mixtral) — the script
auto-detects MoE from the model name and adapts LoRA targets + adds the
router auxiliary loss accordingly.

Recipe (R1-Zero, simplified for a single-GPU Mac):
  1. For each math problem, sample N completions at temperature 0.7
  2. Score each: 1.0 if extracted answer matches ground truth, else 0.0
  3. Compute group-relative advantages (reward standardized within group)
  4. Policy gradient update on chosen tokens, with KL penalty against base
  5. (MoE only) add small router-load-balancing auxiliary loss

Why this works without human labels: math problems have verifiable answers.
The reward function is just a regex-then-compare. No human evaluators required.

Model options:

  DENSE (recommended for 16 GB Mac):
    Qwen/Qwen2.5-1.5B          ~3.0 GB bf16   ✓ fits with room to spare
    meta-llama/Llama-3.2-1B    ~2.0 GB bf16   ✓ fits, gated on HF

  SPARSE / MoE (combines sparse + reasoning, "the frontier recipe"):
    allenai/OLMoE-1B-7B-0924   ~14 GB bf16    ✗ won't fit 16 GB Mac in bf16
                                                → use 4-bit (cloud) or rent GPU
    mistralai/Mixtral-8x7B     ~94 GB bf16    ✗ requires multi-GPU setup

Memory budget on a 16 GB Mac (bf16, MPS), Qwen-2.5-1.5B:
  Base model:        ~3.0 GB
  LoRA adapters:     ~30 MB
  Optimizer state:   ~90 MB
  KV cache:          ~1-2 GB during generation
  Activations:       ~1-2 GB during backward
  Total peak:        ~6-8 GB → fits

Usage:
  pip install torch transformers peft datasets accelerate
  python train_reasoning.py --steps 10                                  # smoke
  python train_reasoning.py --steps 1000                                # full
  python train_reasoning.py --model allenai/OLMoE-1B-7B-0924 --steps 1000   # MoE
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ───────────────────────────────────────────── Config ──

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    dataset_name: str = "openai/gsm8k"

    # Training
    num_steps: int = 1000
    n_rollouts: int = 4
    max_new_tokens: int = 512
    learning_rate: float = 1e-5
    kl_beta: float = 0.05
    temperature: float = 0.7
    top_p: float = 0.95

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: str = "auto"  # "auto" | "attention" | "mlp" | "all"

    # MoE (no-op for dense models)
    moe_aux_loss_weight: float = 0.001

    # Logging / checkpointing
    log_every: int = 5
    save_every: int = 25  # mac steps are slow; more frequent saves = less lost work on shutdown
    save_dir: str = "checkpoints/reasoning"
    resume: bool = False  # resume from latest step_N/ in save_dir

    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


# ────────────────────────────────────── Architecture helpers ──

# Dense MLP projections (Llama/Qwen-style SwiGLU MLP)
DENSE_MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def is_moe_model(model_name: str) -> bool:
    """Heuristic: detect MoE architectures by name. Conservative — false negatives are fine."""
    name = model_name.lower()
    return any(tag in name for tag in ["moe", "mixtral", "deepseek-v"])


def resolve_lora_targets(model_name: str, mode: str) -> list[str]:
    """
    Pick which modules to attach LoRA adapters to.

    Why this matters for MoE: each MoE layer has N expert MLPs (e.g. OLMoE has 64).
    Targeting MLP projections on a MoE means LoRA adapts every expert separately,
    which can be 50-100× more LoRA parameters than the dense equivalent. Default
    for MoE is attention-only — experts are already specialized; attention is
    where reasoning behavior lives.
    """
    if mode == "auto":
        return ATTENTION_TARGETS if is_moe_model(model_name) else ATTENTION_TARGETS + DENSE_MLP_TARGETS
    if mode == "attention":
        return ATTENTION_TARGETS
    if mode == "mlp":
        return DENSE_MLP_TARGETS
    if mode == "all":
        return ATTENTION_TARGETS + DENSE_MLP_TARGETS
    raise ValueError(f"unknown lora_targets mode: {mode!r}")


def forward_with_aux(model, input_ids):
    """
    Forward pass that opportunistically captures MoE router auxiliary loss.

    For MoE models, we ask for `output_router_logits=True` so the model returns
    its load-balancing aux_loss alongside logits. For dense models, that kwarg
    raises TypeError — we fall back to a plain forward.
    """
    try:
        out = model(input_ids, output_router_logits=True)
    except TypeError:
        out = model(input_ids)
    aux = getattr(out, "aux_loss", None)
    return out.logits, aux


# ──────────────────────────────────────── Prompt format ──

PROMPT_TEMPLATE = """Solve the following math problem step by step. Show your reasoning, then give the final numeric answer after "####".

Problem: {question}

Solution:"""


def make_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def extract_answer(text: str) -> str | None:
    """
    Pull the final numeric answer from a generation. GSM8K canonically uses
    '#### <number>' as the answer marker — we follow that convention.

    Examples that should match:
      'After working through this... #### 42'        → '42'
      'The answer is: #### -3.5'                     → '-3.5'
      '#### 1,200 dollars'                           → '1200'

    Returns None if no answer marker is found.
    """
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


# ───────────────────────────────────────── Reward function ──

def reward_fn(completion: str, ground_truth: str) -> float:
    """
    TODO (you write this): turn a model completion + ground truth into a scalar reward.

    `completion` is the model-generated text AFTER the prompt. Example:
        "First, find x. Setting up: 3x + 5 = 20. Subtract 5: 3x = 15.
         Divide by 3: x = 5. #### 5"

    `ground_truth` is the canonical answer string for the problem (e.g. "5", "42", "-3.5").

    DESIGN CHOICES (each shapes the policy you train — none are obvious):

    1. CORRECTNESS SIGNAL — binary or graduated?
       Binary (1.0 / 0.0): clean training signal, what R1 used. RECOMMENDED.
       Graduated: partial credit for being close. For math, "close" rarely makes sense
                  (3.99 is wrong if the answer is 4).

    2. FORMAT REWARD — should we reward following the "####" template?
       Without it: the model outputs "the answer is 5" → extract_answer returns None
                   → reward 0.0 → bad signal even when it got the answer right.
       With it (e.g. +0.1 for ANY output containing "####"): encourages format
                   compliance early in training, before reasoning quality matures.
       RECOMMENDED: small format bonus (~0.1).

    3. NUMERICAL COMPARISON — strict string or float?
       Strict: "5" == "5" but "5" != "5.0".
       Float: convert both, compare with tolerance. Robust to formatting noise.
       RECOMMENDED: float comparison with small tolerance (e.g. 1e-4).

    4. LENGTH PENALTY?
       The model can learn to emit massive reasoning chains for the same reward.
       A small penalty for length > N tokens prevents runaway. But too much penalty
       discourages thinking, which is the WHOLE POINT of this exercise.
       RECOMMENDED: skip for now. Add only if you observe runaway in training logs.

    Return: a float, typically in [0.0, 1.1] if you use a format bonus.
    Higher is better.
    """
    predicted = extract_answer(completion)

    # Format bonus: rewards using the "####" answer marker even before
    # correctness emerges. Prevents the all-zero-reward collapse early in training.
    format_reward = 0.1 if "####" in completion else 0.0

    # Correctness: float compare with tolerance handles "5" vs "5.0" etc.
    # Falls back to string equality for non-numeric answers.
    correctness_reward = 0.0
    if predicted is not None and ground_truth is not None:
        try:
            if abs(float(predicted) - float(ground_truth)) < 1e-4:
                correctness_reward = 1.0
        except ValueError:
            if predicted.strip() == ground_truth.strip():
                correctness_reward = 1.0

    return correctness_reward + format_reward


# ───────────────────────────────────────── Generation ──

@torch.no_grad()
def generate_completions(model, tokenizer, prompt: str, cfg: Config):
    """Generate N completions for a prompt. Returns list of (text, full_token_ids)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cfg.device)
    out = []
    for _ in range(cfg.n_rollouts):
        gen = model.generate(
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_ids = gen[0]
        completion_ids = full_ids[prompt_ids.shape[1]:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        out.append((text, full_ids))
    return out, prompt_ids.shape[1]


# ───────────────────────────────────────── GRPO step ──

def grpo_step(model, tokenizer, prompt: str, ground_truth: str, cfg: Config):
    """
    One GRPO update step. Returns (loss_tensor_or_None, rewards, completion_lengths).

    Algorithm:
      1. Generate N completions for the prompt
      2. Score each with reward_fn → rewards[i]
      3. Standardize within group → advantages[i] = (r - mean) / std
      4. For each completion, compute:
           pg_loss = -advantages[i] * mean(log π(a_t|s_t))   over completion tokens
           kl     = mean(log π_policy - log π_reference)     over completion tokens
         total_loss += pg_loss + kl_beta * kl
    """
    # 1. Rollouts
    candidates, prompt_len = generate_completions(model, tokenizer, prompt, cfg)

    # 2. Score
    rewards = torch.tensor(
        [reward_fn(text, ground_truth) for text, _ in candidates],
        device=cfg.device, dtype=torch.float32,
    )
    completion_lengths = [int(full_ids.shape[0] - prompt_len) for _, full_ids in candidates]

    # No learning signal if all rewards are equal — skip update
    if rewards.std().item() < 1e-6:
        return None, rewards.tolist(), completion_lengths

    # 3. Group-relative advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # 4. Policy gradient + KL penalty (+ MoE aux loss when present)
    total_loss = torch.zeros((), device=cfg.device, dtype=torch.bfloat16)
    aux_loss_sum = torch.zeros((), device=cfg.device, dtype=torch.bfloat16)
    aux_loss_count = 0

    for i, (_, full_ids) in enumerate(candidates):
        full_ids_batch = full_ids.unsqueeze(0).to(cfg.device)
        completion_token_ids = full_ids[prompt_len:].to(cfg.device)

        # Policy forward (LoRA active, gradients on; aux_loss returned for MoE)
        policy_logits_full, policy_aux = forward_with_aux(model, full_ids_batch)
        policy_logits = policy_logits_full[0]
        # Logits at positions [prompt_len-1 ... -1] predict tokens [prompt_len ... last]
        completion_logits = policy_logits[prompt_len - 1 : -1]
        policy_logp = F.log_softmax(completion_logits, dim=-1)
        chosen_logp_policy = policy_logp.gather(
            1, completion_token_ids.unsqueeze(1)
        ).squeeze(1)

        # Reference log-probs (LoRA disabled, no gradients)
        with model.disable_adapter():
            with torch.no_grad():
                ref_logits_full, _ = forward_with_aux(model, full_ids_batch)
                ref_completion_logits = ref_logits_full[0][prompt_len - 1 : -1]
                ref_logp = F.log_softmax(ref_completion_logits, dim=-1)
                chosen_logp_ref = ref_logp.gather(
                    1, completion_token_ids.unsqueeze(1)
                ).squeeze(1)

        kl = (chosen_logp_policy - chosen_logp_ref).mean()
        pg_loss = -(advantages[i] * chosen_logp_policy.mean())
        total_loss = total_loss + pg_loss + cfg.kl_beta * kl

        if policy_aux is not None:
            aux_loss_sum = aux_loss_sum + policy_aux
            aux_loss_count += 1

    total_loss = total_loss / cfg.n_rollouts
    if aux_loss_count > 0 and cfg.moe_aux_loss_weight > 0:
        total_loss = total_loss + cfg.moe_aux_loss_weight * (aux_loss_sum / aux_loss_count)

    return total_loss, rewards.tolist(), completion_lengths


# ─────────────────────────────────────── Checkpoint helpers ──

def find_latest_checkpoint(save_dir: Path) -> int | None:
    """
    Return the highest step number N for which `save_dir/step_N/` exists,
    or None if no such directory is present. The integer in the name is the
    source of truth — sorting by string or by mtime is unreliable.
    """
    if not save_dir.exists():
        return None
    pattern = re.compile(r"^step_(\d+)$")
    steps = [int(m.group(1)) for d in save_dir.iterdir()
             if d.is_dir() and (m := pattern.match(d.name))]
    return max(steps, default=None)


# ───────────────────────────────────────── Training loop ──

def train(cfg: Config):
    print(f"[setup] device = {cfg.device}")
    print(f"[setup] loading tokenizer + model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in base.parameters())
    base_gb = total_params * 2 / 1e9  # bf16 = 2 bytes/param
    moe = is_moe_model(cfg.model_name)
    print(f"[setup] {total_params/1e9:.2f}B params, ~{base_gb:.1f} GB in bf16, "
          f"arch={'MoE' if moe else 'dense'}")
    if base_gb > 8:
        print(f"        ⚠️  large base model — peak Mac memory will be ~{base_gb*1.5:.0f} GB during training")
        print(f"        if this exceeds 16 GB, consider 4-bit quantization or cloud GPU")

    target_modules = resolve_lora_targets(cfg.model_name, cfg.lora_targets)
    print(f"[setup] LoRA targets: {target_modules}")
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    print(f"[setup] loading dataset: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name, "main", split="train")
    print(f"        {len(ds)} training examples")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "log.jsonl"

    # Resume from latest step_N/ if asked. Reloads LoRA adapter + optimizer state
    # so momentum survives — without it, the first few resumed steps spike while
    # AdamW re-warms from zero.
    start_step = 0
    if cfg.resume:
        latest = find_latest_checkpoint(save_dir)
        if latest is not None:
            ckpt = save_dir / f"step_{latest}"
            print(f"[resume] loading LoRA + optimizer from step {latest}")
            model = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.learning_rate,
            )
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=cfg.device))
            start_step = latest + 1
        else:
            print(f"[resume] no checkpoint in {save_dir}, starting fresh")

    log_file = log_path.open("a")

    print(f"[train] {cfg.num_steps} steps × {cfg.n_rollouts} rollouts (start at {start_step})")
    print(f"        save_dir = {save_dir}")
    t_start = time.time()
    rolling_reward = []

    for step in range(start_step, cfg.num_steps):
        ex = ds[step % len(ds)]
        question = ex["question"]
        ground_truth = extract_answer(ex["answer"])
        if ground_truth is None:
            continue
        prompt = make_prompt(question)

        loss, rewards, lengths = grpo_step(model, tokenizer, prompt, ground_truth, cfg)

        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            loss_val = float(loss.item())
        else:
            loss_val = None

        avg_reward = sum(rewards) / len(rewards)
        avg_length = sum(lengths) / len(lengths)
        rolling_reward.append(avg_reward)
        if len(rolling_reward) > 50:
            rolling_reward.pop(0)

        record = {
            "step": step,
            "loss": loss_val,
            "rewards": rewards,
            "lengths": lengths,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "rolling_reward_50": sum(rolling_reward) / len(rolling_reward),
            "elapsed_sec": time.time() - t_start,
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / max(step + 1, 1)
            eta_min = sec_per_step * (cfg.num_steps - step) / 60
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "  ---  "
            print(
                f"step {step:4d} | loss {loss_str} | "
                f"r {avg_reward:.2f} (roll {sum(rolling_reward)/len(rolling_reward):.2f}) | "
                f"len {avg_length:5.0f} | "
                f"{sec_per_step:5.1f}s/step | ETA {eta_min:.0f}m"
            )

        if step > 0 and step % cfg.save_every == 0:
            ckpt = save_dir / f"step_{step}"
            model.save_pretrained(ckpt)
            torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
            print(f"  → saved {ckpt}")

    model.save_pretrained(save_dir / "final")
    log_file.close()
    print(f"\n[done] total {(time.time() - t_start)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--save-dir", type=str, default="checkpoints/reasoning")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                    help="HF model id. Examples: Qwen/Qwen2.5-1.5B (dense), "
                         "allenai/OLMoE-1B-7B-0924 (MoE)")
    ap.add_argument("--lora-targets", type=str, default="auto",
                    choices=["auto", "attention", "mlp", "all"],
                    help="auto: attention-only for MoE, attention+MLP for dense")
    ap.add_argument("--moe-aux-weight", type=float, default=0.001,
                    help="weight on MoE router auxiliary loss (no-op for dense)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest step_N/ in --save-dir")
    args = ap.parse_args()

    cfg = Config(
        model_name=args.model,
        num_steps=args.steps,
        n_rollouts=args.n_rollouts,
        learning_rate=args.lr,
        kl_beta=args.kl_beta,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
        lora_targets=args.lora_targets,
        moe_aux_loss_weight=args.moe_aux_weight,
        resume=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
