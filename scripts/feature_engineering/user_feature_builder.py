#!/usr/bin/env python3
"""
用户中心特征工程器
构建以用户为key的特征数据集
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import time
from tqdm import tqdm

sys.path.append('src')


class UserFeatureBuilder:
    """用户特征构建器"""

    def __init__(self, time_window_days=7):
        self.time_window_days = time_window_days
        self.target_date = "2014-12-18"
        self.features = {}

    def load_data(self, sample_frac=0.1):
        """加载数据 - 使用采样避免内存问题"""
        print(f"📂 加载数据 (采样率: {sample_frac})")

        files = [
            "dataset/tianchi_fresh_comp_train_user_online_partA.txt",
            "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
        ]

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        # 时间窗口过滤
        target_dates = []
        base_date = pd.to_datetime("2014-12-18")
        for i in range(self.time_window_days):
            date = base_date - pd.Timedelta(days=i+1)
            target_dates.append(date.strftime("%Y-%m-%d"))

        print(f"  🎯 时间窗口: {target_dates[-1]} 到 {target_dates[0]}")

        dfs = []
        for file_path in tqdm(files, desc="📁 加载文件"):
            print(f"  正在加载 {file_path}")

            # 分块读取并采样
            chunk_reader = pd.read_csv(file_path, sep="\t", names=columns, chunksize=1000000)

            file_chunks = []
            for chunk in chunk_reader:
                # 时间过滤
                chunk = chunk[chunk["time"].str.startswith(tuple(target_dates))]

                if len(chunk) > 0 and sample_frac < 1.0:
                    chunk = chunk.sample(frac=sample_frac, random_state=42)

                if len(chunk) > 0:
                    file_chunks.append(chunk)

            if file_chunks:
                file_df = pd.concat(file_chunks, ignore_index=True)
                dfs.append(file_df)
                print(f"  ✓ {file_path}: {len(file_df):,} 行")

        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ 总数据量: {len(df):,} 行")

        # 预处理
        df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
        df = df.dropna(subset=["user_id", "item_id", "datetime"])
        df["user_id"] = df["user_id"].astype(np.int32)
        df["item_id"] = df["item_id"].astype(np.int32)

        return df

    def build_basic_features(self, df):
        """构建基础用户特征"""
        print("🔧 构建基础用户特征...")

        user_features = {}

        for user_id in tqdm(df["user_id"].unique(), desc="📊 基础特征"):
            user_data = df[df["user_id"] == user_id]

            # 基础统计特征
            features = {
                # 活跃度特征
                "total_actions": len(user_data),
                "unique_items": user_data["item_id"].nunique(),
                "unique_categories": user_data["item_category"].nunique(),
                "active_days": user_data["datetime"].dt.date.nunique(),

                # 行为模式特征
                "browse_count": len(user_data[user_data["behavior_type"] == 1]),
                "collect_count": len(user_data[user_data["behavior_type"] == 2]),
                "cart_count": len(user_data[user_data["behavior_type"] == 3]),
                "purchase_count": len(user_data[user_data["behavior_type"] == 4]),

                # 转化率特征
                "collect_rate": len(user_data[user_data["behavior_type"] == 2]) / max(len(user_data[user_data["behavior_type"] == 1]), 1),
                "cart_rate": len(user_data[user_data["behavior_type"] == 3]) / max(len(user_data[user_data["behavior_type"] == 1]), 1),
                "purchase_rate": len(user_data[user_data["behavior_type"] == 4]) / max(len(user_data[user_data["behavior_type"] == 1]), 1),

                # 时间特征
                "avg_hour": user_data["datetime"].dt.hour.mean(),
                "weekend_rate": (user_data["datetime"].dt.weekday >= 5).mean(),
                "last_action_days_ago": (pd.to_datetime(self.target_date) - user_data["datetime"].max()).days,
            }

            user_features[user_id] = features

        self.features.update(user_features)
        return user_features

    def build_geo_features(self, df):
        """构建地理位置特征"""
        print("🌍 构建地理位置特征...")

        # 地理位置购买力分析
        geo_purchase_power = df[df["behavior_type"] == 4].groupby("user_geohash").size()
        geo_power_rank = geo_purchase_power.rank(pct=True)

        for user_id in tqdm(df["user_id"].unique(), desc="📍 地理特征"):
            user_data = df[df["user_id"] == user_id]

            # 地理特征
            geo_features = {
                "unique_locations": user_data["user_geohash"].nunique(),
                "primary_location": user_data["user_geohash"].mode().iloc[0] if len(user_data["user_geohash"].mode()) > 0 else "unknown",
            }

            # 主要地理位置的购买力
            primary_geo = geo_features["primary_location"]
            geo_features["location_purchase_power"] = geo_power_rank.get(primary_geo, 0.1)

            self.features[user_id].update(geo_features)

    def build_category_preferences(self, df):
        """构建类别偏好特征"""
        print("🏷️ 构建类别偏好特征...")

        # 全局类别热度
        global_category_popularity = df.groupby("item_category").size()

        for user_id in tqdm(df["user_id"].unique(), desc="🎯 偏好特征"):
            user_data = df[df["user_id"] == user_id]

            # 类别偏好特征
            user_categories = user_data["item_category"].value_counts()

            category_features = {
                "top_category": user_categories.index[0] if len(user_categories) > 0 else -1,
                "category_concentration": (user_categories.iloc[0] / len(user_data)) if len(user_categories) > 0 else 0,
                "category_diversity": len(user_categories),
            }

            # 偏好热门程度
            if len(user_categories) > 0:
                top_cat_popularity = global_category_popularity.get(user_categories.index[0], 1)
                category_features["prefers_popular_categories"] = top_cat_popularity / global_category_popularity.max()
            else:
                category_features["prefers_popular_categories"] = 0

            self.features[user_id].update(category_features)

    def build_temporal_features(self, df):
        """构建时间序列特征"""
        print("⏰ 构建时间序列特征...")

        for user_id in tqdm(df["user_id"].unique(), desc="📈 时序特征"):
            user_data = df[df["user_id"] == user_id].sort_values("datetime")

            # 时间序列特征
            temporal_features = {
                "action_frequency": len(user_data) / max(user_data["datetime"].dt.date.nunique(), 1),
                "morning_rate": (user_data["datetime"].dt.hour < 12).mean(),
                "evening_rate": (user_data["datetime"].dt.hour >= 18).mean(),
            }

            # 行为间隔分析
            if len(user_data) > 1:
                time_diffs = user_data["datetime"].diff().dt.total_seconds() / 3600  # 小时
                temporal_features.update({
                    "avg_action_interval_hours": time_diffs.mean(),
                    "action_regularity": 1 / (time_diffs.std() + 1),  # 规律性
                })
            else:
                temporal_features.update({
                    "avg_action_interval_hours": 24,
                    "action_regularity": 0,
                })

            self.features[user_id].update(temporal_features)

    def build_target_labels(self, df):
        """构建目标标签 - 预测19日是否购买"""
        print("🎯 构建目标标签...")

        # 加载19日数据 (如果有的话，这里先假设没有)
        # 实际中我们需要预测，所以这里构建伪标签用于验证

        for user_id in self.features.keys():
            user_data = df[df["user_id"] == user_id]

            # 基于历史行为预测购买概率 (伪标签)
            purchase_count = len(user_data[user_data["behavior_type"] == 4])
            recent_activity = len(user_data[user_data["datetime"] >= (pd.to_datetime(self.target_date) - pd.Timedelta(days=2))])

            # 简单的启发式标签
            will_purchase = (purchase_count > 0 and recent_activity > 2) or (purchase_count > 2)

            self.features[user_id]["target_will_purchase"] = int(will_purchase)

    def export_features(self, filename="user_features.csv"):
        """导出特征数据集"""
        print(f"💾 导出特征数据集: {filename}")

        # 转换为DataFrame
        feature_df = pd.DataFrame.from_dict(self.features, orient="index")
        feature_df.index.name = "user_id"
        feature_df = feature_df.reset_index()

        # 保存
        feature_df.to_csv(filename, index=False)

        print(f"✅ 特征数据集已保存")
        print(f"  👥 用户数: {len(feature_df):,}")
        print(f"  🔧 特征数: {len(feature_df.columns)-1}")
        print(f"  🎯 目标比例: {feature_df['target_will_purchase'].mean():.3f}")

        return feature_df


def main():
    """主函数"""
    print("=== 用户特征工程器 ===")

    builder = UserFeatureBuilder(time_window_days=7)

    # 1. 加载数据 (使用采样)
    df = builder.load_data(sample_frac=0.05)  # 5%采样开始测试

    # 2. 构建各类特征
    builder.build_basic_features(df)
    builder.build_geo_features(df)
    builder.build_category_preferences(df)
    builder.build_temporal_features(df)
    builder.build_target_labels(df)

    # 3. 导出特征
    feature_df = builder.export_features("user_features.csv")

    print("\n🎉 用户特征工程完成!")
    print("下一步可以使用这个特征数据集训练机器学习模型")


if __name__ == "__main__":
    main()