"""SFT QLoRA (Unsloth). Example building and the masked loss are pure and tested on CPU;
train() needs the GPU env (unsloth, trl, bitsandbytes) and has NOT been run yet.
"""

import json
from pathlib import Path

import torch

from lipogram.generate import build_messages

IGNORE = -100


def build_example(tok, source: str, target: str, fewshot: bool, max_len: int) -> dict | None:
    """input_ids = exact inference prompt + target + end-of-turn; loss only on the answer.
    Returns None if the example does not fit in max_len (dropping beats truncating the answer)."""
    prompt = tok.apply_chat_template(build_messages(source, fewshot), add_generation_prompt=True,
                                     enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]
    answer = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt) + len(answer) > max_len:
        return None
    return {"input_ids": prompt + answer, "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [IGNORE] * len(prompt) + answer}


def load_examples(path: str | Path, tok, fewshot: bool, max_len: int) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [build_example(tok, r["source"], r["target"], fewshot, max_len) for r in rows]
    return [e for e in examples if e is not None], sum(e is None for e in examples)


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, forbidden: torch.Tensor | None) -> torch.Tensor:
    """Causal LM cross-entropy; with `forbidden`, the softmax is renormalized over allowed tokens only
    (train/inference alignment ablation). Valid because lipogram targets only use allowed tokens."""
    logits = logits[:, :-1].float()
    if forbidden is not None:
        logits = logits.masked_fill(forbidden.to(logits.device), float("-inf"))
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                             ignore_index=IGNORE)


def train(cfg: dict) -> None:  # pragma: no cover - GPU only
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from lipogram.token_mask import load_or_build_mask

    model, tok = FastLanguageModel.from_pretrained(cfg["model"], max_seq_length=cfg["max_seq_length"],
                                                   load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    splits = {}
    for split in ("train", "val"):
        examples, dropped = load_examples(Path(cfg["data_dir"]) / f"{split}.jsonl", tok, cfg["fewshot"], cfg["max_seq_length"])
        print(f"{split}: {len(examples)} examples ({dropped} dropped: longer than max_seq_length)")
        splits[split] = Dataset.from_list(examples)
    forbidden = load_or_build_mask(tok, model.config.vocab_size) if cfg.get("masked_loss") else None

    class Trainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            if forbidden is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = masked_lm_loss(out.logits, inputs["labels"], forbidden)
            return (loss, out) if return_outputs else loss

    class PeakVram(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and torch.cuda.is_available():
                logs["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    trainer = Trainer(
        model=model, processing_class=tok, train_dataset=splits["train"], eval_dataset=splits["val"],
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
        callbacks=[PeakVram()],
        args=SFTConfig(
            output_dir=cfg["output_dir"], per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"], warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type="cosine", num_train_epochs=cfg["epochs"], optim="adamw_8bit", bf16=True,
            eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"],
            load_best_model_at_end=True, metric_for_best_model="eval_loss", save_total_limit=2,
            logging_steps=10, seed=cfg["seed"], report_to="none", dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False))
    trainer.train()
    model.save_pretrained(Path(cfg["output_dir"]) / "adapter")
    tok.save_pretrained(Path(cfg["output_dir"]) / "adapter")
