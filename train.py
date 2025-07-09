import argparse, os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # fully offline

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

from src import config, utils

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def wrap_inst(prompt: str, response: str = "") -> str:
    """Apply Mistral‑Instruct chat template."""
    return f"<s>[INST] {prompt} [/INST] {response}</s>"


# -------------------------------------------------------------------------
# data functions
# -------------------------------------------------------------------------

def prepare_data():
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    utils.build_dataset()


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_datasets(tokenizer):
    data_files = {"train": str(config.TRAIN_FILE), "validation": str(config.VALID_FILE)}
    ds = load_dataset("json", data_files=data_files)

    def tokenize_fn(record):
        full_text = wrap_inst(record["prompt"], record["response"])
        toks = tokenizer(
            full_text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
        )
        prompt_only = wrap_inst(record["prompt"])  # w/out response
        prompt_len  = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
        labels = toks["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len  # mask loss on prompt
        toks["labels"] = labels
        return toks

    token_ds = ds.map(tokenize_fn, batched=False, remove_columns=["prompt", "response"])
    return token_ds


# -------------------------------------------------------------------------
# training loop
# -------------------------------------------------------------------------

def train():
    tokenizer = load_tokenizer()

    # ---- device & quantisation settings ----------------------------------
    has_cuda = torch.cuda.is_available()
    bf16_ok  = has_cuda and torch.cuda.is_bf16_supported()

    if has_cuda:  # QLoRA 4‑bit
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16" if bf16_ok else "float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs = dict(quantization_config=bnb_cfg, device_map="auto")
    else:  # CPU fallback
        load_kwargs = dict(device_map={"": "cpu"}, torch_dtype=torch.float16)

    base_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_PATH, **load_kwargs)

    # ---- PEFT LoRA --------------------------------------------------------
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ---- dataset & collator ---------------------------------------------
    token_ds = make_datasets(tokenizer)

    # use default_data_collator so our custom labels stay intact
    data_collator = default_data_collator

    # ---- training arguments ---------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        warmup_ratio=0.05,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        logging_steps=50,
        fp16=(has_cuda and not bf16_ok),
        bf16=bf16_ok,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_ds["train"],
        eval_dataset=token_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ---- save adapter ----------------------------------------------------
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"LoRA adapter saved to {config.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true", help="Generate JSONL dataset")
    parser.add_argument("--train",        action="store_true", help="Run fine‑tuning")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
    if args.train:
        train()
    if not (args.prepare_data or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()
