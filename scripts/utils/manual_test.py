#!/usr/bin/env python3

import torch
import json
from src.graphrec.data import GraphDataset
from src.graphrec.model import LightGCN

def manual_validation_test():
    print("=== 手动验证测试 ===")

    # 加载数据
    dataset = GraphDataset(data_dir="dataset", sample_ratio=1.0, min_interactions=1, seed=42)
    dataset.load_from_prepared("data/graph")

    # 简单模型
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        edge_index=dataset.edge_index,
        embed_dim=32,
        n_layers=2
    )

    # 获取embedding (随机初始化)
    user_emb, item_emb = model()

    # 找一个有效的验证用户
    train_items = set()
    for items in dataset.user_pos_train.values():
        train_items.update(items)

    valid_user = None
    for u in dataset.get_eval_users():
        pos_item = next(iter(dataset.user_pos_val[u]))
        if pos_item in train_items:
            valid_user = u
            valid_pos_item = pos_item
            break

    if valid_user is None:
        print("没找到有效验证用户！")
        return

    print(f"测试用户: {valid_user}")
    print(f"验证商品: {valid_pos_item}")

    # 计算推荐分数
    u_vec = user_emb[valid_user]
    scores = (item_emb @ u_vec)

    # 过滤已见商品
    seen = dataset.get_user_seen_items(valid_user)
    if len(seen) > 0:
        scores[list(seen)] = -1e9

    # Top-10推荐
    topk = torch.topk(scores, k=10).indices.tolist()
    print(f"Top-10推荐: {topk}")
    print(f"验证商品是否在Top-10中: {valid_pos_item in topk}")

    # 检查验证商品的分数
    pos_score = scores[valid_pos_item].item()
    print(f"验证商品分数: {pos_score:.4f}")
    print(f"最高分数: {scores.max().item():.4f}")
    print(f"最低分数: {scores.min().item():.4f}")

if __name__ == "__main__":
    manual_validation_test()