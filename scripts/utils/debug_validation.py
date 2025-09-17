#!/usr/bin/env python3

import json
import pandas as pd
import torch
from src.graphrec.data import GraphDataset

def debug_validation():
    print("=== 调试验证集评估问题 ===")

    # 加载数据集
    dataset = GraphDataset(data_dir="dataset", sample_ratio=1.0, min_interactions=1, seed=42)
    dataset.load_from_prepared("data/graph")

    print(f"数据集规模:")
    print(f"- 用户数: {dataset.num_users}")
    print(f"- 商品数: {dataset.num_items}")
    print(f"- 训练边数: {dataset.edge_index.size(1)//2}")
    print(f"- 验证用户数: {len(dataset.get_eval_users())}")

    # 检查验证数据
    eval_users = dataset.get_eval_users()[:5]  # 取前5个用户
    print(f"\n前5个验证用户: {eval_users}")

    for u in eval_users:
        pos_items = dataset.user_pos_val[u]
        seen_items = dataset.get_user_seen_items(u)
        print(f"\n用户 {u}:")
        print(f"- 验证正样本: {pos_items}")
        print(f"- 训练已见商品数: {len(seen_items)}")
        print(f"- 验证商品是否在训练中见过: {pos_items.intersection(seen_items)}")

        # 检查商品ID范围
        if pos_items:
            pos_item = next(iter(pos_items))
            print(f"- 验证商品ID: {pos_item}, 商品总数: {dataset.num_items}")
            print(f"- 商品ID是否超范围: {pos_item >= dataset.num_items}")

if __name__ == "__main__":
    debug_validation()