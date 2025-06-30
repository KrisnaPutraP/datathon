import argparse, os
from pathlib import Path

# ensure we use offline mode (optional env flags)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"      # no telemetry
aos = os.environ

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from src import config, utils


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
        full_text = record["prompt"] + record["response"]
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
        )
        # Mask loss on prompt tokens by setting label -100
        prompt_len = len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"])
        labels = tokens["input_ids"].copy()
        labels[:prompt_len] = [-100]*prompt_len
        tokens["labels"] = labels
        return tokens

    ds_token = ds.map(tokenize_fn, batched=False, remove_columns=["prompt","response"])
    return ds_token


def train():
    tokenizer = load_tokenizer()

    # ---- 4‑bit quantization config ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16" if tokenizer.backend_kwargs.get("torch_dtype_str") == "bfloat16" else "float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    # ---- apply LoRA ----
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Dataset ----
    token_ds = make_datasets(tokenizer)

    # ---- Data collator ----
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        bf16=True,  # use fp16 if your GPU doesn’t support bfloat16
        report_to="none",  # disable wandb etc.
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
    # save final LoRA adapter & tokenizer
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"LoRA adapter saved to {config.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true",
                        help="Generate jsonl dataset from CSVs")
    parser.add_argument("--train", action="store_true", help="Run fine‑tuning")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
    if args.train:
        train()
    if not (args.prepare_data or args.train):
        parser.print_help()

if __name__ == "__main__":
    main()