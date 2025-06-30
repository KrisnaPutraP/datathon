import json, math, random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from . import config

random.seed(42)

# ----- build prompt/response pairs -------------------------------------------------
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Kamu adalah host live TikTok Shop yang berbahasa santai. "
    "Berdasarkan detail produk di bawah, buat 1 kalimat promosi singkat, hype, "
    "dan persuasif agar penonton segera beli.\n\n"
    "### Input:\n"
    "Nama Produk   : {name}\n"
    "Harga (diskon): Rp{price:,}\n"
    "Stok Tersisa  : {stock}\n"
    "Penonton Live : {viewers}\n"
    "Event         : {event}\n\n"
    "### Response:\n"
)


def build_dataset():
    """Read our CSVs and produce copy_train.jsonl + copy_valid.jsonl"""
    p = config.DATA_DIR
    products      = pd.read_csv(p / "products.csv")
    segments      = pd.read_csv(p / "speech_segments.csv.gz")
    # pick only rows with promo_flag==1 so target kalimat memang promosi
    promo_segs    = segments[segments["promo_flag"] == 1]

    rows = []
    for _, seg in tqdm(promo_segs.iterrows(), total=len(promo_segs)):
        # find any referenced product name (use first id if list)
        try:
            prod_ids = eval(seg["product_refs"])
            prod_id  = prod_ids[0] if prod_ids else None
        except Exception:
            prod_id = None
        if prod_id is None:
            continue
        prod = products.loc[products["product_id"] == prod_id].iloc[0]
        prompt = PROMPT_TEMPLATE.format(
            name    = prod["name"],
            price   = int(prod["price"]),
            stock   = int(prod["stock_qty"]),
            viewers = random.randint(200,2000),
            event   = random.choice(["flash sale", "bonus ongkir", "diskon kilat"]),
        )
        response = seg["raw_text"].strip()
        rows.append({"prompt": prompt, "response": response})

    # shuffle & split
    random.shuffle(rows)
    split_idx = int(len(rows)*(1-config.VALID_SPLIT))
    train_rows = rows[:split_idx]
    valid_rows = rows[split_idx:]

    # save jsonl
    def dump(path: Path, data):
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump(config.TRAIN_FILE, train_rows)
    dump(config.VALID_FILE, valid_rows)
    print(f"Saved {len(train_rows)} train & {len(valid_rows)} valid samples")