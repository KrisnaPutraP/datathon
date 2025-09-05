import argparse
import os

import numpy as np
import torch
from bert_score import score as bert_score
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src import config, utils

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
tokenizer = None


def wrap_inst(prompt: str, response: str = "") -> str:
    return (
        f"<s>[INST] {prompt}\n\n"
        f"Jawablah dengan format COPY1:, COPY2:, COPY3:, BUNDLE:, TIME:. [/INST] "
        f"{response}</s>"
    )


def prepare_data():
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    utils.build_full_dataset()


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_datasets(tok):
    data_files = {"train": str(config.TRAIN_FILE), "validation": str(config.VALID_FILE)}
    ds = load_dataset("json", data_files=data_files)

    def tokenize_fn(record):
        full_text = wrap_inst(record["prompt"], record["response"])
        toks = tok(
            full_text,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
        )
        prompt_len = len(
            tok(wrap_inst(record["prompt"]), add_special_tokens=False)["input_ids"]
        )
        labels = toks["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len
        toks["labels"] = labels
        return toks

    return ds.map(tokenize_fn, batched=False, remove_columns=["prompt", "response"])


def _first_line(texts):
    return [t.splitlines()[0] if t else "" for t in texts]


def _field(texts, prefix):
    out = []
    for t in texts:
        for line in t.splitlines():
            if line.startswith(prefix):
                out.append(line[len(prefix) :].strip())
                break
        else:
            out.append("∅")
    return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Evaluate copywriting quality menggunakan BERT score
    _, _, f1 = bert_score(_first_line(pred_text), _first_line(label_text), lang="id")
    
    # Evaluate accuracy untuk setiap field
    acc_copy1 = np.mean(
        np.array(_field(pred_text, "COPY1:")) == np.array(_field(label_text, "COPY1:"))
    )
    acc_copy2 = np.mean(
        np.array(_field(pred_text, "COPY2:")) == np.array(_field(label_text, "COPY2:"))
    )
    acc_copy3 = np.mean(
        np.array(_field(pred_text, "COPY3:")) == np.array(_field(label_text, "COPY3:"))
    )
    acc_time = np.mean(
        np.array(_field(pred_text, "TIME:")) == np.array(_field(label_text, "TIME:"))
    )
    acc_bundle = np.mean(
        np.array(_field(pred_text, "BUNDLE:")) == np.array(_field(label_text, "BUNDLE:"))
    )
    
    return {
        "bert_f1": f1.mean().item(),
        "acc_copy1": acc_copy1,
        "acc_copy2": acc_copy2,
        "acc_copy3": acc_copy3,
        "acc_time": acc_time,
        "acc_bundle": acc_bundle,
    }


def train():
    global tokenizer
    tokenizer = load_tokenizer()

    has_cuda = torch.cuda.is_available()
    bf16_ok = has_cuda and torch.cuda.is_bf16_supported()

    if has_cuda:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16" if bf16_ok else "float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",  
        )
        load_kwargs = dict(
            quantization_config=bnb_cfg, 
            device_map="auto",
            torch_dtype="auto"
        )
    else:
        load_kwargs = dict(device_map={"": "cpu"}, torch_dtype=torch.float16)

    base_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_PATH, **load_kwargs)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    token_ds = make_datasets(tokenizer)
    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        warmup_ratio=0.05,
        logging_steps=50,
        fp16=has_cuda and not bf16_ok,
        bf16=bf16_ok,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_ds["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
    if args.train:
        train()
    if not (args.prepare_data or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()