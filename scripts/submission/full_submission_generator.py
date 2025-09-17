#!/usr/bin/env python3
"""
全量数据submission生成器
只使用最高效的算法：流行度推荐
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from tqdm import tqdm

sys.path.append('src')

def load_full_data():
    """加载全量数据 - 时间窗口优化+分块处理"""
    print("📂 加载全量数据（时间窗口优化）...")

    files = [
        "dataset/tianchi_fresh_comp_train_user_online_partA.txt",
        "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
    ]

    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

    # 时间窗口：重点关注最近7天 (2014-12-11 到 2014-12-17)
    target_dates = ["2014-12-11", "2014-12-12", "2014-12-13", "2014-12-14", "2014-12-15", "2014-12-16", "2014-12-17"]
    print(f"  🎯 时间窗口: {target_dates[0]} 到 {target_dates[-1]}")

    # 分块大小：每次处理100万行
    chunk_size = 1000000
    all_chunks = []

    for file_path in tqdm(files, desc="📁 加载文件"):
        print(f"  正在分块加载 {file_path}")

        # 分块读取
        chunk_reader = pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size)

        file_chunks = []
        for chunk_num, chunk in enumerate(chunk_reader):
            # 立即进行时间过滤以减少内存
            chunk = chunk[chunk["behavior_type"].isin([1, 2, 3, 4])]
            chunk = chunk.dropna(subset=["user_id", "item_id", "time"])

            # 时间过滤：只保留目标日期
            chunk = chunk[chunk["time"].str.startswith(tuple(target_dates))]

            if len(chunk) > 0:
                file_chunks.append(chunk)
                print(f"    处理块 {chunk_num + 1}: {len(chunk):,} 行 (时间过滤后)")

        # 合并当前文件的所有块
        if file_chunks:
            file_df = pd.concat(file_chunks, ignore_index=True)
            all_chunks.append(file_df)
            print(f"  ✓ {file_path}: {len(file_df):,} 行")

    print("🔄 合并所有数据...")
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"✅ 时间窗口内数据量: {len(df):,} 行")

    return df

def preprocess_data(df):
    """基础数据预处理 - 加入地理和类别特征"""
    print("🧹 数据预处理...")

    print("  - 解析时间...")
    df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")

    print("  - 清理缺失值...")
    df = df.dropna(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)

    print("  - 转换数据类型...")
    df["user_id"] = df["user_id"].astype(np.int32)
    df["item_id"] = df["item_id"].astype(np.int32)
    df["item_category"] = df["item_category"].astype("category")

    print("  - 处理地理位置特征...")
    # 处理地理位置hash
    df["user_geohash"] = df["user_geohash"].fillna("unknown")
    df["geo_region"] = df["user_geohash"].astype("category")

    print("  - 添加时间权重...")
    # 基于日期的时间衰减权重
    current_date = pd.to_datetime("2014-12-18")
    df["days_ago"] = (current_date - df["datetime"]).dt.days
    df["time_weight"] = np.exp(-0.2 * df["days_ago"])  # 时间衰减因子

    print("  - 添加行为权重...")
    # 增强行为权重考虑地理因素
    behavior_weights = {1: 1.0, 2: 2.5, 3: 4.0, 4: 6.0}  # 提高高价值行为权重
    df["behavior_weight"] = df["behavior_type"].map(behavior_weights)
    df["final_weight"] = df["behavior_weight"] * df["time_weight"]

    print(f"✅ 预处理完成: {len(df):,} 行")
    print(f"  📍 地理位置数: {df['geo_region'].nunique()}")
    print(f"  🏷️  商品类别数: {df['item_category'].nunique()}")

    return df

def build_popularity_model(df):
    """构建增强流行度模型 - 融合地理和类别特征"""
    print("📊 构建增强流行度模型...")

    print("  - 基础商品流行度...")
    # 使用预处理中计算的final_weight
    base_popularity = df.groupby("item_id")["final_weight"].sum()

    print("  - 地理加权流行度...")
    # 计算地理位置的购买力权重
    geo_purchase_power = df[df["behavior_type"] == 4].groupby("geo_region").size()
    geo_weights = (geo_purchase_power / geo_purchase_power.max()).fillna(0.1)  # 最小权重0.1

    # 为每个商品计算地理加权分数
    geo_weighted_scores = {}
    for item_id in base_popularity.index:
        item_data = df[df["item_id"] == item_id]
        geo_score = 0
        for _, row in item_data.iterrows():
            geo_weight = geo_weights.get(row["geo_region"], 0.1)
            geo_score += row["final_weight"] * geo_weight
        geo_weighted_scores[item_id] = geo_score

    print("  - 类别热度加权...")
    # 计算类别热度
    category_popularity = df.groupby("item_category")["final_weight"].sum()
    category_weights = (category_popularity / category_popularity.max()).fillna(0.1)

    # 商品类别映射
    item_category_map = df.groupby("item_id")["item_category"].first()

    print("  - 融合多维度分数...")
    final_scores = {}
    for item_id in base_popularity.index:
        base_score = base_popularity[item_id]
        geo_score = geo_weighted_scores.get(item_id, base_score * 0.1)
        category = item_category_map.get(item_id)
        category_weight = category_weights.get(category, 0.1) if category else 0.1

        # 加权融合：基础0.5 + 地理0.3 + 类别0.2
        final_score = (0.5 * base_score +
                      0.3 * geo_score +
                      0.2 * base_score * category_weight)
        final_scores[item_id] = final_score

    print("  - 排序商品...")
    popular_items = pd.Series(final_scores).sort_values(ascending=False)

    print(f"✅ 增强热门商品数: {len(popular_items):,}")
    print(f"  📍 考虑 {df['geo_region'].nunique()} 个地理区域")
    print(f"  🏷️  考虑 {df['item_category'].nunique()} 个商品类别")

    return popular_items

def get_user_history(df):
    """获取用户历史行为 - 使用groupby优化"""
    print("📚 构建用户历史...")

    print("  - 聚合用户行为数据...")
    # 使用groupby更高效
    user_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

    print(f"✅ 用户数: {len(user_items):,}")
    return user_items

def generate_recommendations(user_items, popular_items, top_n=5):
    """为用户生成推荐"""
    print(f"🎯 为 {len(user_items):,} 个用户生成推荐...")

    # 获取热门商品列表
    print("  - 选择候选商品...")
    top_popular = popular_items.head(200).index.tolist()  # 取前200热门作为候选

    print("  - 生成用户推荐...")
    recommendations = {}

    for user_id, seen_items in tqdm(user_items.items(), desc="🤖 生成推荐"):
        # 过滤用户已见商品
        candidates = [item for item in top_popular if item not in seen_items]

        # 取前N个推荐
        user_recs = candidates[:top_n]

        recommendations[user_id] = user_recs

    print(f"✅ 推荐生成完成")
    return recommendations

def save_submission(recommendations, filename="enhanced_submission.txt"):
    """保存提交文件"""
    print(f"💾 保存提交文件: {filename}")

    total_lines = 0
    print("  - 写入文件...")

    with open(filename, "w", encoding="utf-8") as f:
        for user_id, items in tqdm(recommendations.items(), desc="💽 写入推荐"):
            for item_id in items:
                f.write(f"{user_id}\t{item_id}\n")
                total_lines += 1

    print(f"✅ 提交文件已生成")
    print(f"📊 用户数: {len(recommendations):,}")
    print(f"📝 推荐对数: {total_lines:,}")
    return filename

def main():
    start_time = time.time()

    print("=== 全量数据快速Submission生成器 ===")
    print("🚀 使用高效流行度算法")

    try:
        # 1. 加载数据
        df = load_full_data()

        # 2. 数据预处理
        train_df = preprocess_data(df)

        # 3. 构建流行度模型
        popular_items = build_popularity_model(train_df)

        # 4. 获取用户历史
        user_items = get_user_history(train_df)

        # 5. 生成推荐
        recommendations = generate_recommendations(user_items, popular_items, top_n=5)

        # 6. 保存提交文件
        submission_file = save_submission(recommendations)

        # 统计信息
        total_time = time.time() - start_time
        print(f"\n🎉 处理完成!")
        print(f"⏱️  总用时: {total_time:.1f}秒")
        print(f"🎯 可以提交 {submission_file} 到比赛平台")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()