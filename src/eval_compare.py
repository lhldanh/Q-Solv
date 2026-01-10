import os
import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from eval import (
    DEFAULT_BASE, DEFAULT_ADAPTER, DEFAULT_SYSTEM,
    build_prompt, generate, extract_code, safe_exec_inline,
    parse_exec_output, is_correct, get_gsm8k_final, ensure_dir
)


def run_eval_on_model(model, tokenizer, dataset, args, name="Model"):
    print(f"\n>>> Running Evaluation for: {name}")
    results = []
    counts = {"correct": 0, "wrong": 0, "no_code": 0, "exec_fail": 0}

    for i in range(args.start, args.start + args.n):
        item = dataset[i]
        q = item["question"]
        gt = get_gsm8k_final(item["answer"])

        prompt = build_prompt(tokenizer, DEFAULT_SYSTEM, q)
        raw = generate(model, tokenizer, prompt,
                       max_new_tokens=args.max_new_tokens)

        code = extract_code(raw)
        status = "wrong"

        if not code:
            status = "no_code"
        else:
            ok, out = safe_exec_inline(code)
            if not ok:
                status = "exec_fail"
            else:
                pred_val = parse_exec_output(out)
                if is_correct(pred_val, gt):
                    status = "correct"

        counts[status] += 1
        results.append(1 if status == "correct" else 0)

        done = i - args.start + 1
        if done % 10 == 0:
            acc = counts["correct"] / done
            print(f"  [{name}] {done}/{args.n} | Acc: {acc:.2%} | Wrong: {counts['wrong']} | No-Code: {counts['no_code']} | Exec-Fail: {counts['exec_fail']}")

    return {
        "accuracy": counts["correct"] / args.n,
        "counts": counts,
        "curve": results
    }


def plot_comparison(out_dir, base_data, adapter_data):
    labels = ['Correct', 'Wrong', 'Exec Fail', 'No Code']
    colors = ['green', 'red', 'orange', 'gray']

    # --- 1. Vẽ Pie Charts (So sánh cấu trúc lỗi nội bộ) ---
    fig_pie, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    base_vals = [base_data['counts'][k.lower().replace(' ', '_')]
                 for k in labels]
    ax1.pie(base_vals, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=colors)
    ax1.set_title(
        f"Base Model Structure\n(Total Acc: {base_data['accuracy']:.1%})")

    adapter_vals = [adapter_data['counts'][k.lower().replace(' ', '_')]
                    for k in labels]
    ax2.pie(adapter_vals, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=colors)
    ax2.set_title(
        f"Adapter Model Structure\n(Total Acc: {adapter_data['accuracy']:.1%})")

    plt.tight_layout()
    fig_pie.savefig(os.path.join(out_dir, "comparison_pie.png"), dpi=200)
    plt.close()

    # --- 2. Vẽ Bar Chart (So sánh trực tiếp giữa 2 model) ---
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    width = 0.35

    # Vẽ cột cho Base
    plt.bar([i - width/2 for i in x], base_vals,
            width, label='Base Model', color='#a1c9f4')
    # Vẽ cột cho Adapter
    plt.bar([i + width/2 for i in x], adapter_vals, width,
            label='Adapter (Finetuned)', color='#ffb482')

    plt.xlabel('Category')
    plt.ylabel('Number of Samples')
    plt.title('Direct Comparison: Base vs Adapter')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Thêm số liệu cụ thể trên đầu mỗi cột
    for i in x:
        plt.text(i - width/2, base_vals[i] +
                 0.5, str(base_vals[i]), ha='center')
        plt.text(i + width/2, adapter_vals[i] +
                 0.5, str(adapter_vals[i]), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_bar.png"), dpi=200)
    plt.close()

    print(f"Đã lưu biểu đồ Pie: {os.path.join(out_dir, 'comparison_pie.png')}")
    print(f"Đã lưu biểu đồ Bar: {os.path.join(out_dir, 'comparison_bar.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--images_dir", type=str, default="images")
    ap.add_argument("--save_json", action="store_true",
                    help="Save results to JSON")
    args = ap.parse_args()

    ensure_dir(args.images_dir)
    ds = load_dataset("gsm8k", "main", split="test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto", torch_dtype=dtype)

    # Eval Base
    base_res = run_eval_on_model(model, tokenizer, ds, args, name="Base Model")

    # Eval Adapter
    model = PeftModel.from_pretrained(model, args.adapter)
    adapter_res = run_eval_on_model(
        model, tokenizer, ds, args, name="Finetuned (Adapter)")

    # Vẽ biểu đồ Pie đối chiếu
    plot_comparison(args.images_dir, base_res, adapter_res)

    if args.save_json:
        report_path = os.path.join(args.images_dir, "compare_report.json")
        report = {
            "base_metrics": {k: v for k, v in base_res.items() if k != 'curve'},
            "adapter_metrics": {k: v for k, v in adapter_res.items() if k != 'curve'},
            "improvement": adapter_res["accuracy"] - base_res["accuracy"]
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Report JSON đã lưu tại: {report_path}")

    print(
        f"\nCOMPLETED: Base {base_res['accuracy']:.2%} vs Adapter {adapter_res['accuracy']:.2%}")


if __name__ == "__main__":
    main()
