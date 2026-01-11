import json
import re
import os
import math
import time
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from utils import (
    extract_code,
    safe_execute_code,
    get_gsm8k_final_answer,
    parse_and_compare
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
dataset = load_dataset("gsm8k", "main", split="train[:2000]")

SYSTEM = "You are a helpful assistant. Solve the math problem by writing Python code only."


def generate_python_solution(question: str) -> str:
    prompt = f"""
You are an expert Python programmer with a strong background in mathematics.
Solve the following problem by writing a Python script.

MANDATORY REQUIREMENTS:
1. Enclose the code within a markdown block: ```python ... ```
2. Define variables clearly.
3. The last line of the code must be `print(final_result)`.
4. Output ONLY the code block. No explanation.
5. No comments in the code.

Problem: {question}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0,
    )
    return resp.choices[0].message.content


out_path = "gsm8k_python.jsonl"

kept = 0
total = 0

with open(out_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        total += 1
        q = item["question"]
        gt = get_gsm8k_final_answer(item["answer"])

        for attempt in range(3):
            try:
                md = generate_python_solution(q)
                code = extract_code(md)

                exec_res = safe_execute_code(code)

                ok = parse_and_compare(exec_res["logs"], gt)
                if ok:
                    entry = {
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": f"```python\n{code}\n```"},
                        ],
                        "original_answer": item["answer"],
                        "exec_output": exec_res["logs"],
                        "gt_final": gt,
                        "is_correct": True,
                    }
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")
                    kept += 1

                break  # thành công (dù ok hay không)
            except Exception as e:
                if attempt == 2:
                    print(f"[{i}] failed: {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))

        if i % 10 == 0:
            print(f"Processed {i} | kept(correct)={kept}/{total}")

print("Done.", "kept:", kept, "total:", total)
