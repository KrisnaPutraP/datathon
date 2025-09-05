import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from datasets import load_dataset
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
import train 

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",  # Gunakan fp4 (lebih baik dari nf4)
)

tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)

lora_model = AutoModelForCausalLM.from_pretrained(
    config.OUTPUT_DIR,
    device_map="auto",
    offload_folder="/content/offload/lora",
    quantization_config=bnb_cfg,
)

base_model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_PATH,
    device_map="auto",
    offload_folder="/content/offload/base",
    quantization_config=bnb_cfg,
)

ds = load_dataset("json", data_files={"validation": str(config.VALID_FILE)})
examples = ds["validation"].select(range(min(3, len(ds["validation"]))))  

gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

def clean_response(text):
    text = re.sub(r"<s>|\[/?INST\]|\</s\>", "", text)
    return text.strip()

print("\n=== Inference Comparison ===\n")
for i, record in enumerate(examples):
    print(f"--- Prompt #{i+1} ---")
    print("PROMPT:")
    print(record["prompt"])
    print()

    prompt = train.wrap_inst(record["prompt"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        out_lora = lora_model.generate(input_ids, generation_config=gen_cfg)
        out_base = base_model.generate(input_ids, generation_config=gen_cfg)

    pred_lora = clean_response(tokenizer.decode(out_lora[0], skip_special_tokens=True))
    pred_base = clean_response(tokenizer.decode(out_base[0], skip_special_tokens=True))

    print("🔷 Fine-tuned Output:")
    print("-" * 40)
    for line in pred_lora.splitlines():
        if line.strip():
            print("   ", line.strip())
    print("-" * 40)

    print("🔸 Base Model Output:")
    print("-" * 40)
    for line in pred_base.splitlines():
        if line.strip():
            print("   ", line.strip())
    print("-" * 40)
    print()