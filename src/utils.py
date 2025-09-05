import json
import random
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config

random.seed(42)

PROMPT_TEMPLATE = (
    "Kamu adalah host live TikTok Shop yang berbahasa santai. "
    "Berdasarkan detail 3 produk di bawah, buat copywriting yang menarik untuk masing-masing produk, "
    "rekomendasi bundling 2 produk dengan diskon, dan tentukan jam live yang optimal.\n\n"
    "Contoh jawaban:\n"
    "COPY1: Kaos oversize keren banget, cocok buat hangout santai!\n"
    "COPY2: Kemeja flannel premium, bikin tampilan makin stylish!\n"
    "COPY3: Celana jeans berkualitas, wajib punya buat koleksi!\n"
    "BUNDLE: Kaos Oversize + Kemeja Flannel (diskon 20%)\n"
    "TIME: 19:00-21:00 WIB\n\n"
    "PRODUK 1:\n"
    "Nama: {name1}\n"
    "Harga: Rp{price1:,}\n"
    "Terjual hari ini: {sold1} pcs\n\n"
    "PRODUK 2:\n"
    "Nama: {name2}\n"
    "Harga: Rp{price2:,}\n"
    "Terjual hari ini: {sold2} pcs\n\n"
    "PRODUK 3:\n"
    "Nama: {name3}\n"
    "Harga: Rp{price3:,}\n"
    "Terjual hari ini: {sold3} pcs\n"
)


def _best_host_for_category(session_df, sp_df, hosts_df, products_df):
    """Mapping kategori → host dengan total viewers tertinggi."""
    join = (
        sp_df.merge(session_df[["session_id", "host_id", "avg_viewers"]], on="session_id")
        .merge(products_df[["product_id", "category"]], on="product_id")
    )
    g = join.groupby(["category", "host_id"]).agg({"avg_viewers": "sum"}).reset_index()
    idx = g.groupby("category")["avg_viewers"].idxmax()
    best = g.loc[idx].merge(hosts_df[["host_id", "name"]], on="host_id")[["category", "name"]]
    return dict(zip(best["category"], best["name"]))


def _best_timeslot_for_category(session_df, sp_df, products_df):
    """Mapping kategori → jam mulai (0-23) dengan median viewers tertinggi."""
    session_df = session_df.copy()
    session_df["hour"] = pd.to_datetime(session_df["start_ts"]).dt.hour
    join = (
        sp_df.merge(session_df[["session_id", "hour", "avg_viewers"]], on="session_id")
        .merge(products_df[["product_id", "category"]], on="product_id")
    )
    g = join.groupby(["category", "hour"]).agg({"avg_viewers": "median"}).reset_index()
    idx = g.groupby("category")["avg_viewers"].idxmax()
    best = g.loc[idx][["category", "hour"]]
    return dict(zip(best["category"], best["hour"]))


def _top_cooccurring_pairs(sp_df, products_df):
    """Mapping product_id → nama produk yang paling sering muncul bersama."""
    pair_counter = Counter()
    for _, grp in sp_df.groupby("session_id"):
        prods = list(grp["product_id"])
        for p1, p2 in itertools.combinations(sorted(prods), 2):
            pair_counter[(p1, p2)] += 1
    best_pair = {}
    for (p1, p2), cnt in pair_counter.items():
        for a, b in [(p1, p2), (p2, p1)]:
            if (a not in best_pair) or (cnt > best_pair[a][1]):
                best_pair[a] = (b, cnt)
    id2name = dict(zip(products_df.product_id, products_df.name))
    return {k: id2name[v[0]] for k, v in best_pair.items()}


def build_full_dataset():
    """Buat JSONL prompt-response dengan 3 produk, 3 copywriting, bundling, dan jam live."""
    p = config.DATA_DIR
    products = pd.read_csv(p / "products.csv")
    sessions = pd.read_csv(p / "live_sessions.csv")
    session_prods = pd.read_csv(p / "session_products.csv")
    hosts = pd.read_csv(p / "hosts.csv")
    segments = pd.read_csv(p / "speech_segments.csv.gz")
    orders = pd.read_csv(p / "orders.csv")

    # Data preprocessing untuk sales per hari
    orders['order_date'] = pd.to_datetime(orders['order_ts']).dt.date
    daily_sales = orders.groupby(['order_date']).size().reset_index(name='daily_orders')
    
    # Mapping best timeslots
    best_hour = _best_timeslot_for_category(sessions, session_prods, products)
    
    # Get product combinations yang sering muncul bersama
    best_bundle = _top_cooccurring_pairs(session_prods, products)

    rows = []
    promo_segs = segments[segments["promo_flag"] == 1]
    
    # Group produk berdasarkan kategori untuk membuat kombinasi yang masuk akal
    category_products = products.groupby('category')['product_id'].apply(list).to_dict()
    
    for _ in tqdm(range(min(500, len(promo_segs))), desc="Building dataset"):
        # Pilih 3 produk dari kategori yang sama atau related
        categories = list(category_products.keys())
        main_category = random.choice(categories)
        
        # Ambil 3 produk dari kategori utama atau campuran
        if len(category_products[main_category]) >= 3:
            selected_products = random.sample(category_products[main_category], 3)
        else:
            # Jika tidak cukup, ambil dari kategori lain juga
            all_products = []
            for cat in categories:
                all_products.extend(category_products[cat])
            selected_products = random.sample(all_products, 3)
        
        # Get product details
        prod1 = products[products.product_id == selected_products[0]].iloc[0]
        prod2 = products[products.product_id == selected_products[1]].iloc[0]
        prod3 = products[products.product_id == selected_products[2]].iloc[0]
        
        # Generate sales data (jumlah terjual hari ini)
        sold1 = random.randint(5, 50)
        sold2 = random.randint(5, 50)
        sold3 = random.randint(5, 50)
        
        # Create prompt
        prompt = PROMPT_TEMPLATE.format(
            name1=prod1["name"],
            price1=int(prod1["price"]),
            sold1=sold1,
            name2=prod2["name"],
            price2=int(prod2["price"]),
            sold2=sold2,
            name3=prod3["name"],
            price3=int(prod3["price"]),
            sold3=sold3,
        )

        # Generate copywriting untuk masing-masing produk
        sample_seg = promo_segs.sample(3)
        copy1 = sample_seg.iloc[0]['raw_text'].strip()
        copy2 = sample_seg.iloc[1]['raw_text'].strip()
        copy3 = sample_seg.iloc[2]['raw_text'].strip()
        
        # Pilih 2 produk untuk bundling (yang paling mahal + random)
        prices = [(prod1["name"], int(prod1["price"])), 
                 (prod2["name"], int(prod2["price"])), 
                 (prod3["name"], int(prod3["price"]))]
        prices.sort(key=lambda x: x[1], reverse=True)
        
        # Bundle 2 produk termahal dengan diskon
        bundle_products = prices[:2]
        discount = random.choice([15, 20, 25])
        bundle_text = f"{bundle_products[0][0]} + {bundle_products[1][0]} (diskon {discount}%)"
        
        # Generate jam live berdasarkan kategori
        jam = best_hour.get(main_category, random.randint(18, 21))
        jam_str = f"{jam:02d}:00-{(jam + 2) % 24:02d}:00 WIB"

        response = (
            f"COPY1: {copy1}\n"
            f"COPY2: {copy2}\n"
            f"COPY3: {copy3}\n"
            f"BUNDLE: {bundle_text}\n"
            f"TIME: {jam_str}"
        )

        rows.append({"prompt": prompt, "response": response})

    random.shuffle(rows)
    split = int(len(rows) * (1 - config.VALID_SPLIT))
    train_rows, valid_rows = rows[:split], rows[split:]

    def dump(path: Path, data):
        with path.open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(config.TRAIN_FILE, train_rows)
    dump(config.VALID_FILE, valid_rows)