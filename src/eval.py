import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from utils import (
    extract_code,
    safe_execute_code,
    get_gsm8k_final_answer,
    parse_and_compare
)
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DEFAULT_ADAPTER = "dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k"
DEFAULT_BASE = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_SYSTEM = "You are a helpful assistant. Solve the math problem by writing Python code only."


def build_prompt(tokenizer, system: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=False)


@dataclass
class OneResult:
    idx: int
    gt: Optional[int]
    has_code: bool
    exec_ok: bool
    pred_val: Optional[float]
    correct: bool
    exec_msg: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_counts(out_dir: str, counts: Dict[str, int]) -> str:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.title("GSM8K Test Evaluation Counts")
    plt.ylabel("Count")
    plt.xticks(rotation=20)

    path = os.path.join(out_dir, "eval_counts.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_accuracy_curve(out_dir: str, correctness: List[int]) -> str:
    xs = list(range(1, len(correctness) + 1))
    run = []
    c = 0
    for i, v in enumerate(correctness, start=1):
        c += v
        run.append(c / i)

    plt.figure()
    plt.plot(xs, run)
    plt.title("Running Accuracy over GSM8K Test Samples")
    plt.xlabel("Samples")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    path = os.path.join(out_dir, "accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    ap.add_argument("--n", type=int, default=100,
                    help="number of test samples")
    ap.add_argument("--start", type=int, default=0,
                    help="start index in GSM8K test")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--images_dir", type=str, default="images")
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.images_dir)

    print("Loading GSM8K test split...")
    ds = load_dataset("gsm8k", "main", split="test")

    end = min(args.start + args.n, len(ds))
    print(f"Eval range: [{args.start}, {end}) total={end-args.start}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading base model:", args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

    print("Loading adapter:", args.adapter)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    counts = {"correct": 0, "wrong": 0, "no_code": 0, "exec_fail": 0}
    correctness_curve: List[int] = []
    error_examples: List[Dict[str, Any]] = []

    t0 = time.time()
    for idx in range(args.start, end):
        item = ds[idx]
        q = item["question"]
        gt = get_gsm8k_final_answer(item["answer"])

        prompt = build_prompt(tokenizer, DEFAULT_SYSTEM, q)
        raw = generate(model, tokenizer, prompt,
                       max_new_tokens=args.max_new_tokens)

        code = extract_code(raw)
        if not code:
            counts["no_code"] += 1
            correctness_curve.append(0)
            if len(error_examples) < 10:
                error_examples.append(
                    {"idx": idx, "type": "no_code", "question": q, "raw": raw[:1500], "gt": gt})
            continue

        exec_res = safe_execute_code(code)

        if exec_res["error"]:
            counts["exec_fail"] += 1
            correctness_curve.append(0)
            if len(error_examples) < 10:
                error_examples.append({"idx": idx, "type": "exec_fail", "question": q,
                                       "code": code[:1500], "exec_msg": exec_res["error"], "gt": gt})
            continue

        corr = parse_and_compare(exec_res["logs"], gt)

        if corr:
            counts["correct"] += 1
            correctness_curve.append(1)
        else:
            counts["wrong"] += 1
            correctness_curve.append(0)
            if len(error_examples) < 10:
                error_examples.append({"idx": idx, "type": "wrong", "question": q,
                                       "code": code[:1500], "stdout": exec_res["logs"], "gt": gt})

        done = idx - args.start + 1
        if done % 5 == 0:
            acc = counts["correct"] / done
            print(f"[{done}/{end-args.start}] correct={counts['correct']} acc={acc:.2%} "
                  f"no_code={counts['no_code']} exec_fail={counts['exec_fail']} wrong={counts['wrong']}")
    dt = time.time() - t0
    total = end - args.start
    acc = counts["correct"] / total if total else 0.0

    print("\n==== SUMMARY ====")
    print("Total:", total)
    print("Correct:", counts["correct"])
    print("Wrong:", counts["wrong"])
    print("No code:", counts["no_code"])
    print("Exec fail:", counts["exec_fail"])
    print(f"Accuracy: {acc:.2%}")
    print(f"Runtime: {dt:.1f}s")

    p1 = plot_counts(args.images_dir, counts)
    p2 = plot_accuracy_curve(args.images_dir, correctness_curve)
    print("\nSaved plots:")
    print("-", p1)
    print("-", p2)

    report = {
        "base": args.base,
        "adapter": args.adapter,
        "range": {"start": args.start, "end": end, "n": total},
        "counts": counts,
        "accuracy": acc,
        "runtime_sec": dt,
        "plots": {"counts": p1, "accuracy_curve": p2},
        "errors_sample": error_examples,
    }

    if args.save_json:
        report_path = os.path.join(args.images_dir, "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Saved report:", report_path)


if __name__ == "__main__":
    main()
