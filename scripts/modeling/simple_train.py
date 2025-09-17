#!/usr/bin/env python3

import torch
import time
from torch.optim import Adam
from src.graphrec.data import GraphDataset
from src.graphrec.model import LightGCN

def simple_train():
    print("=== 简化训练 ===")

    # 加载数据
    print("加载数据...")
    dataset = GraphDataset(data_dir="dataset", sample_ratio=1.0, min_interactions=1, seed=42)
    dataset.load_from_prepared("data/graph")
    print(f"数据规模: {dataset.num_users:,}用户, {dataset.num_items:,}商品, {dataset.edge_index.size(1)//2:,}边")

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        edge_index=dataset.edge_index.to(device),
        embed_dim=32,
        n_layers=2
    ).to(device)

    optim = Adam(model.parameters(), lr=0.001)

    # 简化训练循环
    epochs = 5
    batch_size = 2048
    steps_per_epoch = max(1, dataset.num_users // batch_size)

    print(f"开始训练: {epochs} epochs, {steps_per_epoch} steps/epoch")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        print(f"Epoch {epoch}/{epochs} - 训练中...")

        for step in range(steps_per_epoch):
            # 采样批量数据
            u, pi, ni = dataset.sample_batch(batch_size)
            if u.numel() == 0:
                continue

            u = u.to(device)
            pi = pi.to(device)
            ni = ni.to(device)

            # 前向传播
            user_emb, item_emb = model()
            u_e = user_emb[u]
            pi_e = item_emb[pi]
            ni_e = item_emb[ni]

            # 计算损失
            loss = LightGCN.bpr_loss(u_e, pi_e, ni_e, l2_reg=1e-4)

            # 反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

            # 每50步输出一次进度
            if step % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"  Step {step:3d}/{steps_per_epoch} - Loss: {avg_loss:.4f}")

        # epoch结束
        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch} 完成 - 用时: {epoch_time:.1f}s, 平均Loss: {avg_loss:.4f}")
        print("-" * 50)

    print("训练完成！")

if __name__ == "__main__":
    simple_train()