#!/usr/bin/env python3

import torch
from src.graphrec.data import GraphDataset
from src.graphrec.model import LightGCN

def debug_training():
    print("=== 调试训练过程 ===")

    print("1. 开始加载数据集...")
    try:
        dataset = GraphDataset(data_dir="dataset", sample_ratio=1.0, min_interactions=1, seed=42)
        print("   ✅ GraphDataset初始化完成")

        print("2. 从预处理文件加载图数据...")
        dataset.load_from_prepared("data/graph")
        print(f"   ✅ 图数据加载完成: {dataset.num_users:,}用户, {dataset.num_items:,}商品, {dataset.edge_index.size(1)//2:,}边")

        print("3. 创建LightGCN模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device}")

        model = LightGCN(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            edge_index=dataset.edge_index.to(device),
            embed_dim=32,
            n_layers=2
        ).to(device)
        print("   ✅ 模型创建完成")

        print("4. 测试前向传播...")
        with torch.no_grad():
            user_emb, item_emb = model()
        print(f"   ✅ 前向传播成功: user_emb={user_emb.shape}, item_emb={item_emb.shape}")

        print("5. 测试批量采样...")
        u, pi, ni = dataset.sample_batch(1024)
        print(f"   ✅ 批量采样成功: u={u.shape}, pi={pi.shape}, ni={ni.shape}")

        print("\n🎉 所有组件都工作正常！问题可能在训练循环的某个地方。")

    except Exception as e:
        print(f"❌ 错误发生在: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training()