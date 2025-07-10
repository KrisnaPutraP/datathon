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
    "Berdasarkan detail produk di bawah, buat 1 kalimat promosi singkat, hype, "
    "dan persuasif agar penonton segera beli.\n\n"
    "Contoh jawaban:\n"
    "COPY: Sandal kece anti licin wajib punya buat gaya santai kamu!\n"
    "HOST: Rini\n"
    "TIME: 18:00-20:00 WIB\n"
    "BUNDLE: Sandal Jepit Stylish Anti Licin + Topi Keren (diskon 15%)\n\n"
    "Nama Produk   : {name}\n"
    "Harga (diskon): Rp{price:,}\n"
    "Stok Tersisa  : {stock}\n"
    "Penonton Live : {viewers}\n"
    "Event         : {event}\n"
)


def _best_host_for_category(session_df, sp_df, hosts_df, products_df):
    """Mapping kategori → host dengan total viewers tertinggi."""
    join = (
        sp_df.merge(session_df[["session_id", "host_id", "unique_viewers"]], on="session_id")
        .merge(products_df[["product_id", "category"]], on="product_id")
    )
    g = join.groupby(["category", "host_id"]).agg({"unique_viewers": "sum"}).reset_index()
    idx = g.groupby("category")["unique_viewers"].idxmax()
    best = g.loc[idx].merge(hosts_df[["host_id", "name"]], on="host_id")[["category", "name"]]
    return dict(zip(best["category"], best["name"]))


def _best_timeslot_for_category(session_df, sp_df, products_df):
    """Mapping kategori → jam mulai (0-23) dengan median viewers tertinggi."""
    session_df = session_df.copy()
    session_df["hour"] = pd.to_datetime(session_df["start_ts"]).dt.hour
    join = (
        sp_df.merge(session_df[["session_id", "hour", "unique_viewers"]], on="session_id")
        .merge(products_df[["product_id", "category"]], on="product_id")
    )
    g = join.groupby(["category", "hour"]).agg({"unique_viewers": "median"}).reset_index()
    idx = g.groupby("category")["unique_viewers"].idxmax()
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
    """Buat JSONL prompt-response dengan COPY, HOST, TIME, BUNDLE."""
    p = config.DATA_DIR
    products = pd.read_csv(p / "products.csv")
    sessions = pd.read_csv(p / "live_sessions.csv")
    session_prods = pd.read_csv(p / "session_products.csv")
    hosts = pd.read_csv(p / "hosts.csv")
    segments = pd.read_csv(p / "speech_segments.csv.gz")

    best_host = _best_host_for_category(sessions, session_prods, hosts, products)
    best_hour = _best_timeslot_for_category(sessions, session_prods, products)
    best_bundle = _top_cooccurring_pairs(session_prods, products)

    rows = []
    promo_segs = segments[segments["promo_flag"] == 1]
    for _, seg in tqdm(promo_segs.iterrows(), total=len(promo_segs)):
        try:
            prod_ids = eval(seg["product_refs"], {})
            prod_id = prod_ids[0] if prod_ids else None
        except Exception:
            continue
        if prod_id is None or prod_id not in best_bundle:
            continue
        prod = products.loc[products.product_id == prod_id].iloc[0]

        prompt = PROMPT_TEMPLATE.format(
            name=prod["name"],
            price=int(prod["price"]),
            stock=int(prod["stock_qty"]),
            viewers=random.randint(200, 2000),
            event=random.choice(["flash sale", "bonus ongkir", "diskon kilat"]),
        )

        cat = prod["category"]
        host_name = best_host.get(cat, random.choice(hosts["name"]))
        jam = best_hour.get(cat, 19)
        jam_str = f"{jam:02d}:00-{(jam + 2) % 24:02d}:00"
        bundling = best_bundle[prod_id]
        disc = random.choice([10, 15, 20])

        response = (
            f"COPY: {seg['raw_text'].strip()}\n"
            f"HOST: {host_name}\n"
            f"TIME: {jam_str} WIB\n"
            f"BUNDLE: {prod['name']} + {bundling} (diskon {disc}%)"
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