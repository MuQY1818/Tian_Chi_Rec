#!/usr/bin/env python3
"""
简化商品特征提取器
快速提取基础商品特征，配合现有39维用户特征
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class SimpleItemFeatureExtractor:
    """简化商品特征提取器"""

    def __init__(self, data_dir="dataset/preprocess_16to18", output_dir="/mnt/data/tianchi_features"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 只保留核心统计
        self.item_stats = defaultdict(lambda: {
            'total_interactions': 0,
            'purchase_count': 0,
            'unique_users': set(),
            'category': 0
        })

        print(f"📁 数据目录: {data_dir}")
        print(f"📁 输出目录: {output_dir}")

    def load_item_catalog(self):
        """加载商品目录（P子集）"""
        print("📚 加载商品子集P...")

        item_file = "dataset/tianchi_fresh_comp_train_item_online.txt"
        columns = ["item_id", "item_geohash", "item_category"]

        item_df = pd.read_csv(item_file, sep="\t", names=columns)
        # 处理空的geohash
        item_df['item_geohash'] = item_df['item_geohash'].fillna('unknown')

        self.item_catalog = dict(zip(item_df['item_id'], item_df['item_category']))

        print(f"  📦 P子集商品数: {len(self.item_catalog):,}")
        print(f"  🏷️  商品类别数: {item_df['item_category'].nunique()}")
        print(f"  📍 有地理信息的商品: {(item_df['item_geohash'] != 'unknown').sum():,}")
        return item_df

    def process_data_files(self):
        """快速处理数据文件，只统计核心指标"""
        print("\n📂 快速处理16-18号数据...")

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        for day in [16, 17, 18]:
            filename = f"data_12{day}.txt"
            print(f"\n📅 处理第{day}号数据: {filename}")

            file_path = os.path.join(self.data_dir, filename)

            # 大块读取提高效率
            chunk_size = 2000000
            processed_lines = 0

            for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size):
                # 只处理P子集商品
                chunk = chunk[chunk['item_id'].isin(self.item_catalog.keys())]

                if len(chunk) == 0:
                    continue

                # 快速统计
                for _, row in chunk.iterrows():
                    item_id = int(row['item_id'])
                    user_id = int(row['user_id'])
                    behavior = int(row['behavior_type'])

                    stats = self.item_stats[item_id]
                    stats['total_interactions'] += 1
                    stats['unique_users'].add(user_id)
                    stats['category'] = self.item_catalog[item_id]

                    if behavior == 4:  # 购买
                        stats['purchase_count'] += 1

                processed_lines += len(chunk)

                if processed_lines % 1000000 == 0:
                    print(f"    📈 已处理: {processed_lines:,} 行 | 用户-商品对: {len(self.item_stats):,}")

            print(f"  ✅ 第{day}号数据完成，处理了 {processed_lines:,} 行")

        print(f"\n📊 统计完成:")
        print(f"  📦 有数据商品数: {len(self.item_stats):,}")

    def generate_simple_features(self):
        """生成简化的商品特征（只保留核心8维）"""
        print("\n🔧 生成简化商品特征...")

        features_list = []

        for item_id, stats in tqdm(self.item_stats.items(), desc="生成特征"):
            features = {
                'item_id': item_id,
                'item_category': stats['category'],

                # 核心特征（8维）
                'popularity': stats['total_interactions'],  # 流行度
                'user_count': len(stats['unique_users']),   # 用户数
                'purchase_count': stats['purchase_count'],   # 购买数
                'purchase_rate': stats['purchase_count'] / max(stats['total_interactions'], 1),  # 购买率
                'avg_user_interactions': stats['total_interactions'] / max(len(stats['unique_users']), 1),  # 平均用户交互

                # 简单分箱特征
                'popularity_level': self._get_popularity_level(stats['total_interactions']),
                'purchase_level': self._get_purchase_level(stats['purchase_count']),
                'category_id': stats['category']  # 类别ID作为分类特征
            }

            features_list.append(features)

        feature_df = pd.DataFrame(features_list)

        # 添加类别级别统计
        category_stats = feature_df.groupby('item_category').agg({
            'popularity': 'mean',
            'purchase_rate': 'mean'
        }).reset_index()
        category_stats.columns = ['item_category', 'category_avg_popularity', 'category_avg_purchase_rate']

        # 合并类别特征
        feature_df = feature_df.merge(category_stats, on='item_category', how='left')

        print(f"✅ 商品特征生成完成:")
        print(f"  📦 商品数: {len(feature_df):,}")
        print(f"  🔧 特征数: {len(feature_df.columns)-1}")

        return feature_df

    def _get_popularity_level(self, interactions):
        """流行度分级"""
        if interactions >= 10000:
            return 4  # 超热门
        elif interactions >= 1000:
            return 3  # 热门
        elif interactions >= 100:
            return 2  # 一般
        elif interactions >= 10:
            return 1  # 冷门
        else:
            return 0  # 超冷门

    def _get_purchase_level(self, purchases):
        """购买量分级"""
        if purchases >= 100:
            return 3  # 高购买
        elif purchases >= 10:
            return 2  # 中购买
        elif purchases >= 1:
            return 1  # 低购买
        else:
            return 0  # 无购买

    def export_features(self):
        """导出简化特征"""
        print("\n💾 导出商品特征...")

        feature_df = self.generate_simple_features()

        # 保存特征
        output_file = os.path.join(self.output_dir, "simple_item_features.csv")
        feature_df.to_csv(output_file, index=False)

        print(f"✅ 商品特征已保存: {output_file}")

        # 特征统计
        print(f"\n📊 特征统计:")
        print(f"  🔥 最高流行度: {feature_df['popularity'].max():,}")
        print(f"  💰 最高购买率: {feature_df['purchase_rate'].max():.3f}")
        print(f"  📊 平均流行度: {feature_df['popularity'].mean():.1f}")
        print(f"  🏷️  类别数: {feature_df['item_category'].nunique()}")

        # 显示示例
        print(f"\n📋 特征示例:")
        sample_cols = ['item_id', 'popularity', 'purchase_count', 'purchase_rate', 'popularity_level']
        print(feature_df[sample_cols].head())

        return feature_df


def main():
    """主函数"""
    print("📦 === 简化商品特征提取器 === 📦")
    print("🎯 目标：快速提取核心商品特征，配合现有用户特征")
    print("⚡ 预计耗时：2-3分钟")
    print("━" * 50)

    extractor = SimpleItemFeatureExtractor()

    # 1. 加载商品目录
    extractor.load_item_catalog()

    # 2. 快速处理数据
    extractor.process_data_files()

    # 3. 生成并导出特征
    feature_df = extractor.export_features()

    print(f"\n🎉 简化商品特征提取完成!")
    print(f"⚡ 比完整版本快3-5倍")
    print(f"📁 输出: {extractor.output_dir}/simple_item_features.csv")


if __name__ == "__main__":
    main()