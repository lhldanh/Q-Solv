from src.utils import extract_code, safe_execute_code
from llama_cpp import Llama
import sys
import io
import contextlib
import traceback


class QSolvLogicCPU:
    def __init__(self):
        self.llm = None

    def load_model(self, repo_id="dainlieu/qsolv-qwen2.5-coder-7b-gguf",
                   filename="qwen2.5-coder-7b-instruct.Q4_K_M.gguf"):
        """Tải và nạp model GGUF từ Hugging Face hoặc Cache"""
        print(f"--- Đang khởi tạo Llama-CPP trên CPU ---")
        print(f"--- Repo: {repo_id} ---")

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=2048,
            n_threads=None,
            verbose=False
        )
        print("--- Model GGUF đã sẵn sàng trên CPU ---")

    def generate_code(self, user_question: str):
        if not self.llm:
            raise Exception("Model CPU chưa được nạp!")

        # Format prompt
        prompt = f"<|im_start|>system\nYou are a helpful assistant. Solve the math problem by writing Python code only.<|im_end|>\n<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n"

        response = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|im_end|>", "```\n"],
            temperature=0.1
        )

        full_response = response["choices"][0]["text"].strip()

        return extract_code(full_response)

    def execute_code(self, code: str):
        return safe_execute_code(code)
