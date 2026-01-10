# Q-Solv 🔧

**Compact repo for training, fine-tuning, and evaluating Qwen-based models with LoRA adapters and Unsloth trainers.**

## Quick overview ✨

- **What:** Tools and scripts to train, evaluate, compare, and run inference for Qwen models (LoRA adapters included).
- **Primary use-cases:** Fine-tuning, evaluation on GSM8K-style datasets, comparison of evaluation reports, and quick inference.
- **Languages / Frameworks:** Python, PyTorch ecosystem (see `requirements.txt`).

---

## Repo structure 📁

- `configs/` — training configs (e.g., `train_qwen25_coder_7b_lora.yaml`)
- `data/` — example datasets (e.g., `gsm8k_python.jsonl`)
- `models/` — checkpoints & final model adapters (e.g., `qwen25_coder_7b_lora/`)
- `src/` — main scripts:
  - `data_gen.py` — dataset generation utilities
  - `train_unsloth.py` — training/finetuning entrypoint
  - `infer.py` — inference / demo usage
  - `eval.py` — evaluation script
  - `eval_compare.py` — compare evaluation reports
- `unsloth_compiled_cache/` — compiled trainer modules (Unsloth trainers)
- `images/` — generated evaluation/compare reports
- `requirements.txt`, `LICENSE`, `README.md`

---

## Quickstart (terminal) ⚡

1. Create and activate a virtual environment:
    ```powershell
     python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Train (example):
   ```powershell
   python src/train_unsloth.py
   ```
4. Evaluate:
   ```powershell
   python src/eval.py
   ```
5. Inference:
   ```powershell
   python src/infer.py --question "Mike has two candies, Anna give him 1. How many candies does he has?"
   ```
6. Compare evaluation reports:
   ```powershell
   python src/eval_compare.py
   ```

Tip: Run `python src/<script>.py --help` for available flags and options.

---

## Models & Checkpoints 🧠

- **Finetuned Adapter**: [dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k](https://huggingface.co/dainlieu/qsolv-qwen2.5-coder-7b-lora-gsm8k)
- **Base Model**: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
---

## Contributing & development 🤝

- Open issues or PRs for bugs, features, or docs.
- Follow existing code style and add tests for new functionality where feasible.

---

## License 📜

- See the `LICENSE` file in the repository root.

---

If you'd like, I can add a short examples section showing sample config flags or a `docker` / `launcher` script for quick reproducible runs. Let me know which you'd prefer.