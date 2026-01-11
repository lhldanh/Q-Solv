import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import extract_code, safe_execute_code

DEFAULT_BASE = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_ADAPTER = "dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k"
DEFAULT_SYSTEM = "You are a helpful assistant. Solve the math problem by writing Python code only."


def build_prompt(tokenizer, system: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=0.95 if temperature > 0 else None,
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--base", type=str, default=DEFAULT_BASE)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--question", type=str, default="")
    args = parser.parse_args()

    question = args.question.strip() or (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability(0)[
                               0] >= 8) else torch.float16
    if device == "cpu":
        dtype = torch.float32

    print(f"--- Loading Model ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    prompt = build_prompt(tokenizer, DEFAULT_SYSTEM, question)
    raw_output = generate(model, tokenizer, prompt,
                          args.max_new_tokens, args.temperature)

    code = extract_code(raw_output)

    if not code:
        print("===== RAW OUTPUT =====")
        print(raw_output)
        print("\n[LỖI] Không tìm thấy block code Python.")
        return

    print("===== EXTRACTED CODE =====")
    print(code)

    result = safe_execute_code(code)

    print("\n===== EXECUTION RESULT =====")
    if result["error"]:
        print(f"❌ Error:\n{result['error']}")
    else:
        print(f"✅ Output:\n{result['logs']}")


if __name__ == "__main__":
    main()
