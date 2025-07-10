import sys
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset
from bert_score import score as bert_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

import train                     
from src import config           

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    config.OUTPUT_DIR,
    device_map="auto",
    offload_folder="/content/offload",  
    quantization_config=bnb_cfg,
)
tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)

ds = load_dataset("json", data_files={"validation": str(config.VALID_FILE)})
valid = ds["validation"]

pred_texts, label_texts = [], []

for record in valid:
    prompt = train.wrap_inst(record["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    pred_texts.append(pred)
    label_texts.append(train.wrap_inst(record["prompt"], record["response"]))

P_copy = train._first_line(pred_texts)
T_copy = train._first_line(label_texts)
_, _, f1 = bert_score(P_copy, T_copy, lang="id")

acc_host = np.mean(np.array(train._field(pred_texts, "HOST:")) == np.array(train._field(label_texts, "HOST:")))
acc_time = np.mean(np.array(train._field(pred_texts, "TIME:")) == np.array(train._field(label_texts, "TIME:")))
acc_bundle = np.mean(np.array(train._field(pred_texts, "BUNDLE:")) == np.array(train._field(label_texts, "BUNDLE:")))

print("\n=== Evaluation Results ===")
print(f"BERTScore F1 (COPY:)      : {f1.mean().item():.4f}")
print(f"Accuracy HOST             : {acc_host:.4f}")
print(f"Accuracy TIME             : {acc_time:.4f}")
print(f"Accuracy BUNDLE           : {acc_bundle:.4f}")