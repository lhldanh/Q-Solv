import re
import io
import math
import contextlib
import traceback
from typing import Optional, Tuple, Dict, Any

# --- NHÓM 1: TRÍCH XUẤT MÃ ---


def extract_code(md_text: str) -> Optional[str]:
    """Tách đoạn mã nằm trong block ```python ... ```"""
    if not md_text:
        return None
    m = re.search(r"```python\s*(.*?)\s*```", md_text, re.S)
    if m:
        return m.group(1).strip()
    # Trường hợp không có chữ 'python' sau dấu ```
    m = re.search(r"```\s*(.*?)\s*```", md_text, re.S)
    if m:
        return m.group(1).strip()
    return md_text.strip()

# --- NHÓM 2: THỰC THI MÃ AN TOÀN ---


def safe_execute_code(code: str) -> Dict[str, Any]:
    """
    Thực thi mã Python trong môi trường Sandbox hạn chế.
    Kết hợp cơ chế whitelist builtins và chặn import nguy hiểm.
    """
    output_buffer = io.StringIO()
    error = None

    # Danh sách các hàm builtin cho phép (giúp giải toán)
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "int": int, "float": float, "str": str,
        "print": print, "round": round, "list": list, "dict": dict,
        "pow": pow, "enumerate": enumerate, "zip": zip
    }

    # Hàm chặn import các module hệ thống
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        forbidden = ["os", "sys", "shutil", "subprocess",
                     "requests", "socket", "pathlib"]
        if name in forbidden:
            raise ImportError(
                f"Quyền truy cập vào module '{name}' bị chặn vì lý do bảo mật.")
        return __import__(name, globals, locals, fromlist, level)

    # Thiết lập Global scope an toàn
    safe_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "__import__": restricted_import
    }

    try:
        # Làm sạch định dạng dòng cho Windows
        cleaned_code = code.replace('\r\n', '\n')

        # Chặn các từ khóa import trực tiếp trong chuỗi code nếu cần nghiêm ngặt hơn
        if "import os" in cleaned_code or "sys." in cleaned_code:
            raise ImportError("Phát hiện hành vi import trái phép.")

        with contextlib.redirect_stdout(output_buffer):
            exec_scope = {}
            exec(cleaned_code, safe_globals, exec_scope)
    except Exception:
        error = traceback.format_exc()

    return {
        "logs": output_buffer.getvalue().strip(),
        "error": error
    }

# --- NHÓM 3: XỬ LÝ GSM8K ---


def get_gsm8k_final_answer(answer_text: str) -> Optional[int]:
    """Lấy con số đáp án sau ký tự ####"""
    if "####" not in answer_text:
        return None
    try:
        # Loại bỏ dấu phẩy trong số lớn (ví dụ: 1,000 -> 1000)
        s = answer_text.split("####")[-1].strip().replace(",", "")
        return int(float(s))
    except:
        return None


def parse_and_compare(stdout: str, ground_truth: Any) -> bool:
    """So sánh kết quả thực thi với đáp án gốc"""
    if not stdout:
        return False

    # Tìm tất cả số trong stdout
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", stdout.replace(",", ""))
    if not nums:
        return False

    try:
        pred_val = float(nums[-1])
        if math.isnan(pred_val) or math.isinf(pred_val):
            return False
        return int(round(pred_val)) == int(ground_truth)
    except:
        return False
