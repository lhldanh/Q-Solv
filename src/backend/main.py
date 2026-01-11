import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
if torch.cuda.is_available():
    from src.backend.logic import QSolvLogic as Logic
else:
    from src.backend.logic_cpu import QSolvLogicCPU as Logic

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "="*50)
print("Backend of Q-SOLV", f"running on device: {device}")
print("="*50)

# 1. Khởi tạo object
solver = Logic()

# 2. Gọi nạp model (Quan trọng)
try:
    solver.load_model()
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    os._exit(1)

app = FastAPI(title="Q-Solv Backend API", version="1.0.0")


class QuestionRequest(BaseModel):
    question: str


class CodeRequest(BaseModel):
    code: str


@app.get("/")
async def health_check():
    return {"status": "online", "device": device}


@app.post("/generate")
async def generate_endpoint(request: QuestionRequest):
    try:
        code = solver.generate_code(request.question)
        return {"status": "success", "generated_code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_endpoint(request: CodeRequest):
    try:
        result = solver.execute_code(request.code)
        return {"status": "success", "execution_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Check API at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
