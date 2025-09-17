#!/usr/bin/env python3

import json
import pandas as pd
from src.graphrec.data import GraphDataset

def check_item_overlap():
    print("=== 检查训练集和验证集商品重叠情况 ===")

    dataset = GraphDataset(data_dir="dataset", sample_ratio=1.0, min_interactions=1, seed=42)
    dataset.load_from_prepared("data/graph")

    # 收集所有训练商品
    train_items = set()
    for items in dataset.user_pos_train.values():
        train_items.update(items)

    # 收集所有验证商品
    val_items = set()
    for items in dataset.user_pos_val.values():
        val_items.update(items)

    print(f"训练集商品数: {len(train_items)}")
    print(f"验证集商品数: {len(val_items)}")
    print(f"重叠商品数: {len(train_items.intersection(val_items))}")
    print(f"重叠比例: {len(train_items.intersection(val_items))/len(val_items)*100:.2f}%")

    # 统计有多少验证用户的商品在训练中见过
    valid_eval_users = 0
    eval_users = dataset.get_eval_users()

    for u in eval_users:
        val_items_user = dataset.user_pos_val[u]
        if train_items.intersection(val_items_user):
            valid_eval_users += 1

    print(f"\n有效验证用户数: {valid_eval_users}/{len(eval_users)}")
    print(f"有效比例: {valid_eval_users/len(eval_users)*100:.2f}%")

if __name__ == "__main__":
    check_item_overlap()