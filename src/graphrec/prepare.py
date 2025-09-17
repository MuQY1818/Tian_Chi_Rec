import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def to_end_of_day(d: datetime) -> datetime:
    return d + timedelta(hours=23, minutes=59, seconds=59)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_mappings(users: np.ndarray, items: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    uniq_u = np.unique(users.astype(np.int64))
    uniq_i = np.unique(items.astype(np.int64))
    user2id = {int(u): int(i) for i, u in enumerate(uniq_u)}
    item2id = {int(it): int(i) for i, it in enumerate(uniq_i)}
    return user2id, item2id


def main():
    ap = argparse.ArgumentParser(description="Prepare graph data for LightGCN/Graph models")
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--out_dir", type=str, default="data/graph")
    ap.add_argument("--train_cutoff", type=str, default="2014-12-17")
    ap.add_argument("--val_date", type=str, default="2014-12-18")
    ap.add_argument("--min_interactions", type=int, default=2)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ensure_dir(args.out_dir)

    # Load raw user interactions (use small dataset for testing)
    user_file = os.path.join(args.data_dir, "small_dataset.txt")
    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
    df = pd.read_csv(user_file, sep="\t", names=columns)
    if 0 < args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.seed)

    # Basic cleaning
    df = df[df["behavior_type"].isin([1, 2, 3, 4])].copy()
    df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)

    # Deduplicate per hour interaction to reduce redundancy
    df = df.drop_duplicates(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)

    # Time split
    train_cut = to_end_of_day(parse_time(args.train_cutoff))
    val_day = parse_time(args.val_date)

    # Map ids
    user2id, item2id = build_mappings(df["user_id"].values, df["item_id"].values)
    df["uid"] = df["user_id"].map(user2id)
    df["iid"] = df["item_id"].map(item2id)

    # Filter users by min_interactions (pre split)
    cnt = df.groupby("uid").size()
    keep_users = set(cnt[cnt >= max(args.min_interactions, 1)].index)
    df = df[df["uid"].isin(keep_users)].reset_index(drop=True)

    # Train interactions (<= cutoff)
    train_df = df[df["datetime"] <= train_cut].copy()

    # Validation positives: use latest interaction on val_date per user
    val_date_mask = df["datetime"].dt.date == val_day.date()
    val_day_df = df[val_date_mask].copy()
    # pick last interaction in the day per user as positive
    pos_df = val_day_df.sort_values(["uid", "datetime"]).groupby("uid").tail(1)

    # Build train user->items mapping
    user_pos_train = (
        train_df.groupby("uid")["iid"].apply(lambda x: sorted(set(map(int, x)))).to_dict()
    )

    # Edge index (undirected)
    src = []
    dst = []
    num_users = len(user2id)
    for u, items in user_pos_train.items():
        for i in items:
            ui = int(u)
            ii = num_users + int(i)
            src.extend([ui, ii])
            dst.extend([ii, ui])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Save artifacts
    out_inter = os.path.join(args.out_dir, "interactions.parquet")
    df_out = df[["user_id", "item_id", "behavior_type", "datetime", "uid", "iid"]].copy()
    df_out["ts"] = (df_out["datetime"].astype("int64") // 10**9).astype(np.int64)
    df_out.drop(columns=["datetime"], inplace=True)
    df_out.to_parquet(out_inter, index=False)

    with open(os.path.join(args.out_dir, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_users": int(num_users),
                "num_items": int(len(item2id)),
                "user2id": {str(k): int(v) for k, v in user2id.items()},
                "item2id": {str(k): int(v) for k, v in item2id.items()},
            },
            f,
        )

    with open(os.path.join(args.out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_cutoff": args.train_cutoff,
                "val_date": args.val_date,
                "min_interactions": int(args.min_interactions),
            },
            f,
        )

    torch.save(
        {"edge_index": edge_index, "num_users": int(num_users), "num_items": int(len(item2id))},
        os.path.join(args.out_dir, "train_edges.pt"),
    )

    pos_out = pos_df[["uid", "iid", "datetime"]].copy()
    pos_out.rename(columns={"iid": "iid_pos"}, inplace=True)
    pos_out["ts"] = (pos_out["datetime"].astype("int64") // 10**9).astype(np.int64)
    pos_out.drop(columns=["datetime"], inplace=True)
    pos_out.to_parquet(os.path.join(args.out_dir, "val_pairs.parquet"), index=False)

    # Save seen mapping as json (for simplicity)
    with open(os.path.join(args.out_dir, "seen_train.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in user_pos_train.items()}, f)

    # Summary
    print(
        f"Prepared graph: users={num_users}, items={len(item2id)}, train_edges={edge_index.size(1)//2}, val_users={pos_out['uid'].nunique()}"
    )


if __name__ == "__main__":
    main()

