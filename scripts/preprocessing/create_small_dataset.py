#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_small_no_coldstart_dataset():
    """创建一个小的、无冷启动商品的数据集用于测试"""

    print("=== 创建小数据集（无冷启动商品）===")

    # 读取原始数据的一个子集
    print("1. 读取原始数据...")
    raw_file = "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

    # 只读取前100万行作为基础
    df = pd.read_csv(raw_file, sep="\t", names=columns, nrows=1000000)
    print(f"原始数据: {len(df)} 行")

    # 清理数据
    df = df[df["behavior_type"].isin([1, 2, 3, 4])].copy()
    df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)

    # 时间过滤，只要12月的数据
    df = df[df["datetime"] >= "2014-12-01"].copy()
    print(f"12月数据: {len(df)} 行")

    # 选择活跃用户和商品（至少有5次交互）
    print("2. 筛选活跃用户和商品...")
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    active_users = user_counts[user_counts >= 5].index[:5000]  # 选择前5000活跃用户
    active_items = item_counts[item_counts >= 5].index[:2000]  # 选择前2000活跃商品

    # 过滤到活跃用户和商品
    df = df[df["user_id"].isin(active_users) & df["item_id"].isin(active_items)].copy()
    print(f"活跃用户商品数据: {len(df)} 行, {df['user_id'].nunique()} 用户, {df['item_id'].nunique()} 商品")

    # 确保时间切分后的数据完整性
    print("3. 时间切分...")
    train_cutoff = pd.to_datetime("2014-12-16 23:59:59")
    val_date = pd.to_datetime("2014-12-17")

    train_df = df[df["datetime"] <= train_cutoff].copy()
    val_df = df[df["datetime"].dt.date == val_date.date()].copy()

    # 确保验证集中的商品都在训练集中出现过
    train_items = set(train_df["item_id"].unique())
    val_df = val_df[val_df["item_id"].isin(train_items)].copy()

    # 确保验证集中的用户都在训练集中出现过
    train_users = set(train_df["user_id"].unique())
    val_df = val_df[val_df["user_id"].isin(train_users)].copy()

    print(f"训练数据: {len(train_df)} 行, {train_df['user_id'].nunique()} 用户, {train_df['item_id'].nunique()} 商品")
    print(f"验证数据: {len(val_df)} 行, {val_df['user_id'].nunique()} 用户, {val_df['item_id'].nunique()} 商品")

    # 验证无冷启动
    val_items = set(val_df["item_id"].unique())
    cold_items = val_items - train_items
    print(f"冷启动商品数: {len(cold_items)} (应该为0)")

    if len(cold_items) > 0:
        print("❌ 仍有冷启动商品，继续过滤...")
        val_df = val_df[~val_df["item_id"].isin(cold_items)]
        print(f"过滤后验证数据: {len(val_df)} 行")

    # 合并并保存
    print("4. 保存小数据集...")
    small_df = pd.concat([train_df, val_df], ignore_index=True)
    small_df = small_df.sort_values(["user_id", "datetime"]).reset_index(drop=True)

    # 保存到新文件
    output_file = "dataset/small_dataset.txt"
    small_df.to_csv(output_file, sep="\t", header=False, index=False,
                   columns=["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"])

    print(f"✅ 小数据集已保存: {output_file}")
    print(f"📊 最终统计:")
    print(f"   - 总行数: {len(small_df):,}")
    print(f"   - 用户数: {small_df['user_id'].nunique():,}")
    print(f"   - 商品数: {small_df['item_id'].nunique():,}")
    print(f"   - 训练行数: {len(train_df):,}")
    print(f"   - 验证行数: {len(val_df):,}")
    print(f"   - 验证用户数: {val_df['user_id'].nunique():,}")

    return output_file

if __name__ == "__main__":
    create_small_no_coldstart_dataset()