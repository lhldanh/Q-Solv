import yaml
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"


class CompletionOnlyDataCollator:
    """
    Mask labels so loss is computed only on assistant completion.
    Expects dataset items to have 'text' (chat-templated full conversation).
    """

    def __init__(self, tokenizer, response_template: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = tokenizer(
            response_template, add_special_tokens=False)["input_ids"]
        if not self.marker_ids:
            raise ValueError(
                "response_template tokenized to empty ids. Check marker string.")

    @staticmethod
    def _find_sublist(lst, sub):
        n, m = len(lst), len(sub)
        for i in range(n - m + 1):
            if lst[i:i + m] == sub:
                return i
        return -1

    def __call__(self, features):
        # Case A: Trainer gửi text (chưa tokenize)
        if "text" in features[0]:
            texts = [f["text"] for f in features]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        # Case B: Trainer đã tokenize -> chỉ có input_ids (và có thể attention_mask)
        elif "input_ids" in features[0]:
            # tokenizer.pad sẽ tạo attention_mask nếu chưa có
            enc = self.tokenizer.pad(
                features,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            keys = [list(f.keys()) for f in features[:3]]
            raise KeyError(
                f"Batch missing both 'text' and 'input_ids'. First keys: {keys}")

        input_ids = enc["input_ids"]
        attention_mask = enc.get(
            "attention_mask", (input_ids != self.tokenizer.pad_token_id).long())

        labels = input_ids.clone()

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = self._find_sublist(ids, self.marker_ids)

            if start == -1:
                # không tìm thấy marker assistant -> bỏ sample
                labels[i, :] = -100
            else:
                cut = start + len(self.marker_ids)
                labels[i, :cut] = -100

            # mask padding
            labels[i, attention_mask[i] == 0] = -100

        enc["labels"] = labels
        return enc


def load_cfg(path=r"configs\train_qwen25_coder_7b_lora.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg, tokenizer):
    ds = load_dataset("json", data_files={
                      "train": cfg["data"]["path"]}, split="train")

    def formatting(examples):
        texts = []
        for messages in examples[cfg["data"]["dataset_field"]]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    # 🔥 Quan trọng: xoá toàn bộ cột cũ, chỉ giữ text để collator/trainer luôn có "text"
    original_cols = ds.column_names
    ds = ds.map(
        formatting,
        batched=True,
        num_proc=cfg["data"]["num_proc"],
        remove_columns=original_cols,
    )

    # double-safe: chỉ giữ cột text
    ds = ds.select_columns(["text"])

    print("DATASET COLUMNS:", ds.column_names)
    sample_text = ds[0]["text"]
    print("==== PREVIEW (first 600 chars) ====")
    print(sample_text[:600])
    print("==== END PREVIEW ====")

    return ds, sample_text


def pick_response_template(cfg, sample_text):
    for cand in cfg["masking"]["response_template_candidates"]:
        if cand in sample_text:
            print("Using response_template:", repr(cand))
            return cand
    raise ValueError(
        "Không tìm thấy marker assistant trong sample_text.\n"
        "Hãy copy đoạn preview phần bắt đầu assistant và thêm vào YAML masking.response_template_candidates"
    )


def main():
    cfg = load_cfg()

    # 1) Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["data"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # 2) Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing=cfg["lora"]["gradient_checkpointing"],
        random_state=cfg["project"]["seed"],
    )

    # 3) Dataset
    dataset, sample_text = build_dataset(cfg, tokenizer)

    # 4) Masking collator
    response_template = pick_response_template(cfg, sample_text)
    data_collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        max_length=cfg["data"]["max_seq_length"],
    )

    # 5) Train args
    tcfg = cfg["trainer"]
    args = TrainingArguments(
        output_dir=cfg["project"]["output_dir"],
        seed=cfg["project"]["seed"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["data"]["max_seq_length"],
        # Windows ổn định hơn; nếu máy bạn ok thì đổi lại cfg["data"]["num_proc"]
        dataset_num_proc=1,
        data_collator=data_collator,
        args=args,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)


if __name__ == "__main__":
    main()
