import re
import math
import argparse
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# DEFAULT_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"  # nếu windows bị lỗi load bnb 4bit thì dùng bản này
DEFAULT_BASE = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_ADAPTER = "dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k"
DEFAULT_SYSTEM = "You are a helpful assistant. Solve the math problem by writing Python code only."


def extract_code(md: str) -> Optional[str]:
    m = re.search(r"```python\s*(.*?)\s*```", md, re.S)
    if not m:
        return None
    return m.group(1).strip()


def safe_exec(code: str) -> Tuple[bool, str]:
    banned = ["import ", "__import__",
              "open(", "exec(", "eval(", "os.", "subprocess", "sys.", "shutil", "pathlib"]
    low = code.lower()
    if any(b in low for b in banned):
        return False, "Blocked: unsafe code patterns"

    import io
    import contextlib

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "print": print,
        "round": round,
    }
    glb: Dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    loc: Dict[str, Any] = {}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, glb, loc)
        return True, buf.getvalue().strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


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
    parser.add_argument("--max_new_tokens", type=int, default=256)
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

    print("Base:", args.base)
    print("Adapter:", args.adapter)
    print("Device:", device)
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    prompt = build_prompt(tokenizer, DEFAULT_SYSTEM, question)
    text = generate(model, tokenizer, prompt,
                    args.max_new_tokens, args.temperature)

    print("===== RAW OUTPUT =====")
    print(text)
    print()

    code = extract_code(text)
    if not code:
        print("No ```python``` code block found.")
        return

    print("===== EXTRACTED CODE =====")
    print(code)
    print()

    ok, out = safe_exec(code)
    print("===== EXEC RESULT =====")
    print("ok =", ok)
    print(out)


if __name__ == "__main__":
    main()
