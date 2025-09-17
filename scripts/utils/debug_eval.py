#!/usr/bin/env python3

import sys
import torch
sys.path.append('src/graphrec')
from data import GraphDataset
from model import LightGCN

def debug_evaluation():
    # 加载数据
    dataset = GraphDataset()
    dataset.load_from_prepared('data/graph')

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        edge_index=dataset.edge_index.to(device),
        embed_dim=32,
        n_layers=2
    ).to(device)

    # 获取嵌入
    with torch.no_grad():
        user_emb, item_emb = model()

    # 检查前3个验证用户
    eval_users = dataset.get_eval_users()[:3]
    print(f"调试前3个用户的推荐...")

    for u in eval_users:
        pos_item = next(iter(dataset.user_pos_val[u]))
        seen = dataset.get_user_seen_items(u)

        u_vec = user_emb[u]
        scores = (item_emb @ u_vec).cpu()

        print(f"\n用户 {u}:")
        print(f"  验证商品: {pos_item}")
        print(f"  已见商品数: {len(seen)}")
        print(f"  验证商品得分: {scores[pos_item]:.4f}")

        # 屏蔽已见商品
        if len(seen) > 0:
            scores[list(seen)] = -1e9

        # Top-10推荐
        topk = torch.topk(scores, k=10).indices.tolist()
        top_scores = [scores[i].item() for i in topk]

        print(f"  Top-10推荐: {topk}")
        print(f"  Top-10得分: {[f'{s:.4f}' for s in top_scores]}")
        print(f"  命中?: {pos_item in topk}")

if __name__ == "__main__":
    debug_evaluation()