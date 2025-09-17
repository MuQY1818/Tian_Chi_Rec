#!/usr/bin/env python3
"""
快速统计用户数量
"""

import pandas as pd
from tqdm import tqdm
import os

def count_unique_users():
    """快速统计用户总数"""
    print("📊 快速统计用户数量...")

    files = [
        "dataset/tianchi_fresh_comp_train_user_online_partA.txt",
        "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
    ]

    all_users = set()
    total_rows = 0

    for file_path in files:
        if os.path.exists(file_path):
            print(f"\n📂 处理文件: {file_path}")

            # 计算文件行数
            print("  📏 计算文件大小...")
            file_rows = sum(1 for _ in open(file_path, 'r'))
            print(f"  📊 文件行数: {file_rows:,}")

            # 分块读取只读user_id列
            chunk_size = 2000000  # 200万行一批
            chunk_reader = pd.read_csv(
                file_path,
                sep="\t",
                names=["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"],
                usecols=["user_id"],  # 只读user_id列
                chunksize=chunk_size
            )

            file_users = set()
            processed = 0

            for chunk_num, chunk in enumerate(chunk_reader):
                processed += len(chunk)
                progress = (processed / file_rows) * 100

                # 清理数据
                chunk = chunk.dropna(subset=["user_id"])
                chunk_users = set(chunk["user_id"].astype(int))

                file_users.update(chunk_users)
                all_users.update(chunk_users)

                print(f"  🔄 批次 {chunk_num + 1}: 进度 {progress:.1f}% | 文件用户数: {len(file_users):,} | 总用户数: {len(all_users):,}")

            total_rows += processed
            print(f"  ✅ 文件完成: {len(file_users):,} 个用户")

        else:
            print(f"❌ 文件不存在: {file_path}")

    print(f"\n📈 最终统计:")
    print(f"  📝 总数据行数: {total_rows:,}")
    print(f"  👥 唯一用户数: {len(all_users):,}")
    print(f"  📊 平均每用户行数: {total_rows / len(all_users):.1f}")

    return len(all_users), total_rows

if __name__ == "__main__":
    user_count, total_rows = count_unique_users()