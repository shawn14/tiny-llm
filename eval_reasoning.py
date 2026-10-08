"""
eval_reasoning.py — measure GSM8K accuracy of a reasoning model

Three evaluation modes, each demonstrating a different inference-time strategy:

  --mode greedy           Take argmax token at each step. Deterministic, single shot.
                          The "honest baseline" — what the model actually thinks
                          its best answer is.

  --mode sample           Temperature sampling, single completion. Same compute as
                          greedy but stochastic. Usually similar to greedy on math.

  --mode self-consistent  Sample N completions, majority vote on the final answer.
                          The simplest test-time compute scaling trick. Costs N×
                          inference but typically lifts accuracy 5-15%.

Typical workflow (before/after training comparison):

  # 1. Baseline — how good is the unmodified base model?
  python eval_reasoning.py --n-problems 200 --output evals/baseline.json

  # 2. After GRPO training — load the LoRA adapter
  python eval_reasoning.py \\
      --adapter checkpoints/reasoning/final \\
      --n-problems 200 \\
      --output evals/after_grpo.json

  # 3. Test-time scaling — does sampling 8x and voting help?
  python eval_reasoning.py \\
      --adapter checkpoints/reasoning/final \\
      --mode self-consistent --n-samples 8 \\
      --n-problems 200 \\
      --output evals/after_grpo_sc8.json

The improvement from (1)→(2) measures the value of training. The improvement
from (2)→(3) measures the value of test-time compute. Both should be positive.
If (2) ≤ (1), training failed. If (3) ≤ (2), reward function is over-fit to
greedy outputs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_reasoning import extract_answer, make_prompt


def normalize_for_compare(s: str | None):
    """Convert a string answer to comparable form: float if possible, else lowercased string."""
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except (TypeError, ValueError):
        return s.strip().lower()


def is_correct(predicted: str | None, ground_truth: str | None, tol: float = 1e-4) -> bool:
    p = normalize_for_compare(predicted)
    g = normalize_for_compare(ground_truth)
    if p is None or g is None:
        return False
    if isinstance(p, float) and isinstance(g, float):
        return abs(p - g) < tol
    return p == g


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(prompt_ids, **kwargs)
    completion_ids = out[0][prompt_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def majority_vote_answer(completions: list[str]) -> tuple[str | None, list[str | None]]:
    """Extract answers from N completions and return the majority. Returns (winner, all_extracted)."""
    extracted = [extract_answer(c) for c in completions]
    valid = [e for e in extracted if e is not None]
    if not valid:
        return None, extracted
    # Bucket by normalized form (so "5" and "5.0" merge), but report the original string
    buckets: dict = {}
    for e in valid:
        key = str(normalize_for_compare(e))
        buckets.setdefault(key, []).append(e)
    winner_key = max(buckets, key=lambda k: len(buckets[k]))
    return buckets[winner_key][0], extracted


def evaluate(args):
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[setup] device={device}")

    print(f"[setup] loading base: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)

    if args.adapter:
        print(f"[setup] loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    print(f"[setup] loading GSM8K test split")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_problems and args.n_problems < len(ds):
        ds = ds.select(range(args.n_problems))
    print(f"        evaluating {len(ds)} problems in mode={args.mode}")

    results = []
    correct = 0
    t_start = time.time()

    for i, ex in enumerate(ds):
        question = ex["question"]
        ground_truth = extract_answer(ex["answer"])
        prompt = make_prompt(question)

        completions: list[str] = []
        if args.mode == "greedy":
            completions = [generate_one(model, tokenizer, prompt, args.max_new_tokens, 0.0, device)]
            predicted = extract_answer(completions[0])
            extracted_all = [predicted]
        elif args.mode == "sample":
            completions = [generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, device)]
            predicted = extract_answer(completions[0])
            extracted_all = [predicted]
        elif args.mode == "self-consistent":
            completions = [
                generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, device)
                for _ in range(args.n_samples)
            ]
            predicted, extracted_all = majority_vote_answer(completions)
        else:
            raise ValueError(args.mode)

        ok = is_correct(predicted, ground_truth)
        if ok:
            correct += 1

        results.append({
            "i": i,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "extracted_all": extracted_all,
            "correct": ok,
            "completions": completions,
            "avg_completion_chars": sum(len(c) for c in completions) / len(completions),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t_start
            acc = correct / (i + 1)
            sec_per = elapsed / (i + 1)
            eta = sec_per * (len(ds) - i - 1) / 60
            print(f"  [{i+1:3d}/{len(ds)}] acc {acc:5.1%} | "
                  f"{sec_per:.1f}s/problem | ETA {eta:5.1f}m")

    accuracy = correct / len(ds) if len(ds) > 0 else 0.0
    avg_length = sum(r["avg_completion_chars"] for r in results) / max(len(results), 1)

    print()
    print("=== Results ===")
    print(f"  model        : {args.model}")
    print(f"  adapter      : {args.adapter or '(none — base model)'}")
    print(f"  mode         : {args.mode}"
          + (f" (n_samples={args.n_samples})" if args.mode == "self-consistent" else ""))
    print(f"  n_problems   : {len(ds)}")
    print(f"  accuracy     : {accuracy:.1%}  ({correct}/{len(ds)})")
    print(f"  avg length   : {avg_length:.0f} chars")
    print(f"  total time   : {(time.time() - t_start)/60:.1f} min")

    # Quick qualitative sample: 2 correct + 2 incorrect
    correct_examples = [r for r in results if r["correct"]][:2]
    wrong_examples = [r for r in results if not r["correct"]][:2]
    if correct_examples or wrong_examples:
        print()
        print("=== Sample outputs ===")
        for label, exs in [("✓ CORRECT", correct_examples), ("✗ WRONG", wrong_examples)]:
            for r in exs:
                print(f"\n  [{label}] gt={r['ground_truth']} pred={r['predicted']}")
                print(f"    Q: {r['question'][:120]}{'...' if len(r['question']) > 120 else ''}")
                completion = r["completions"][0]
                preview = completion[-200:] if len(completion) > 200 else completion
                preview = preview.replace("\n", "\n       ")
                print(f"    A (last 200 chars): ...{preview}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "summary": {
                "model": args.model,
                "adapter": args.adapter,
                "mode": args.mode,
                "n_problems": len(ds),
                "n_samples": args.n_samples if args.mode == "self-consistent" else 1,
                "accuracy": accuracy,
                "correct": correct,
                "avg_length_chars": avg_length,
                "total_time_sec": time.time() - t_start,
            },
            "results": results,
        }, indent=2))
        print(f"\n  saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter", type=str, default=None,
                    help="path to a saved LoRA adapter (e.g. checkpoints/reasoning/final)")
    ap.add_argument("--mode", type=str, default="greedy",
                    choices=["greedy", "sample", "self-consistent"])
    ap.add_argument("--n-problems", type=int, default=100,
                    help="how many GSM8K test problems to use (full test = 1319)")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="for self-consistent mode: number of samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output", type=str, default=None,
                    help="optional path to save full results JSON")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
