from src.utils import extract_code, safe_execute_code
from unsloth import FastLanguageModel
import torch
import sys
import io
import contextlib
import traceback


class QSolvLogic:
    def __init__(self):
        # Khởi tạo rỗng, sẽ nạp model sau qua hàm load_model
        self.model = None
        self.tokenizer = None

    def load_model(self, model_id="dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k"):
        """Nạp model vào GPU"""
        print(f"--- Đang nạp model: {model_id} ---")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("--- Model đã sẵn sàng! ---")

    def generate_code(self, user_question: str):
        if not self.model:
            raise Exception(
                "Model chưa được nạp. Vui lòng gọi load_model() trước.")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem by writing Python code only."},
            {"role": "user", "content": user_question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.1,
        )

        full_response = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)[0]

        return extract_code(full_response)

    def execute_code(self, code: str):
        return safe_execute_code(code)
