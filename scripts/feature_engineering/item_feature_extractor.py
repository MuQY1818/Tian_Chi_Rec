#!/usr/bin/env python3
"""
商品特征提取器
从16-18号数据中提取商品特征
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class ItemFeatureExtractor:
    """商品特征提取器"""

    def __init__(self, data_dir="dataset/preprocess_16to18", output_dir="/mnt/data/tianchi_features"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 数据文件
        self.data_files = [
            "data_1216.txt",
            "data_1217.txt",
            "data_1218.txt"
        ]

        # 商品统计
        self.item_stats = defaultdict(lambda: {
            # 基础统计
            'total_interactions': 0,
            'browse_count': 0,
            'collect_count': 0,
            'cart_count': 0,
            'purchase_count': 0,
            'unique_users': set(),

            # 时间分布
            'day_interactions': defaultdict(int),  # 按日期统计
            'hour_interactions': defaultdict(int), # 按小时统计

            # 地理分布
            'geo_interactions': defaultdict(int),

            # 时间序列
            'interaction_times': [],
        })

    def load_item_catalog(self):
        """加载商品目录(P子集)"""
        print("📚 加载商品子集P...")

        item_file = "dataset/tianchi_fresh_comp_train_item_online.txt"
        columns = ["item_id", "item_geohash", "item_category"]

        item_df = pd.read_csv(item_file, sep="\t", names=columns)
        # 处理空的geohash
        item_df['item_geohash'] = item_df['item_geohash'].fillna('unknown')

        print(f"  📦 P子集商品数: {len(item_df):,}")

        # 转换为字典便于查找
        self.item_catalog = dict(zip(item_df['item_id'], item_df['item_category']))
        print(f"  🏷️  商品类别数: {item_df['item_category'].nunique()}")

        return item_df

    def process_data_files(self):
        """处理16-18号数据文件"""
        print("\n📂 处理预处理数据文件...")

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        for day, filename in enumerate(self.data_files, 16):
            print(f"\n📅 处理第{day}号数据: {filename}")

            file_path = os.path.join(self.data_dir, filename)

            # 分块读取大文件
            chunk_size = 1000000
            chunk_count = 0

            for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size):
                chunk_count += 1

                print(f"  🔄 处理块 {chunk_count} (行数: {len(chunk):,})")

                # 只处理P子集商品
                chunk = chunk[chunk['item_id'].isin(self.item_catalog.keys())]

                if len(chunk) == 0:
                    continue

                # 解析时间
                chunk['datetime'] = pd.to_datetime(chunk['time'], format="%Y-%m-%d %H", errors="coerce")
                chunk = chunk.dropna(subset=['datetime'])

                # 添加时间特征
                chunk['date'] = chunk['datetime'].dt.date
                chunk['hour'] = chunk['datetime'].dt.hour

                # 更新商品统计
                for _, row in tqdm(chunk.iterrows(),
                                 total=len(chunk),
                                 desc=f"    更新统计",
                                 leave=False,
                                 miniters=len(chunk)//10):
                    self._update_item_stats(row, day)

            print(f"  ✅ 第{day}号数据处理完成")

        print(f"\n📊 统计汇总:")
        print(f"  📦 处理商品数: {len(self.item_stats):,}")
        total_interactions = sum(stats['total_interactions'] for stats in self.item_stats.values())
        print(f"  📈 总交互数: {total_interactions:,}")

    def _update_item_stats(self, row, day):
        """更新单个商品的统计信息"""
        item_id = int(row['item_id'])
        user_id = int(row['user_id'])
        behavior = int(row['behavior_type'])
        geo = row['user_geohash'] if pd.notna(row['user_geohash']) else "unknown"
        hour = int(row['hour'])
        date = row['date']

        stats = self.item_stats[item_id]

        # 基础统计
        stats['total_interactions'] += 1
        stats['unique_users'].add(user_id)

        if behavior == 1:
            stats['browse_count'] += 1
        elif behavior == 2:
            stats['collect_count'] += 1
        elif behavior == 3:
            stats['cart_count'] += 1
        elif behavior == 4:
            stats['purchase_count'] += 1

        # 时间分布
        stats['day_interactions'][day] += 1
        stats['hour_interactions'][hour] += 1

        # 地理分布
        stats['geo_interactions'][geo] += 1

        # 时间序列
        stats['interaction_times'].append(row['datetime'])

    def generate_item_features(self):
        """生成商品特征"""
        print("\n🔧 生成商品特征...")

        features_list = []

        for item_id, stats in tqdm(self.item_stats.items(), desc="生成特征"):
            features = {
                'item_id': item_id,
                'item_category': self.item_catalog.get(item_id, -1),

                # 基础流行度特征
                'total_interactions': stats['total_interactions'],
                'unique_users_count': len(stats['unique_users']),
                'browse_count': stats['browse_count'],
                'collect_count': stats['collect_count'],
                'cart_count': stats['cart_count'],
                'purchase_count': stats['purchase_count'],

                # 转化率特征
                'collect_rate': stats['collect_count'] / max(stats['browse_count'], 1),
                'cart_rate': stats['cart_count'] / max(stats['browse_count'], 1),
                'purchase_rate': stats['purchase_count'] / max(stats['browse_count'], 1),
                'buy_conversion': stats['purchase_count'] / max(stats['total_interactions'], 1),

                # 用户多样性
                'user_interaction_avg': stats['total_interactions'] / max(len(stats['unique_users']), 1),

                # 时间趋势特征
                'day16_interactions': stats['day_interactions'].get(16, 0),
                'day17_interactions': stats['day_interactions'].get(17, 0),
                'day18_interactions': stats['day_interactions'].get(18, 0),

                # 时间模式
                'morning_rate': sum(stats['hour_interactions'][h] for h in range(6, 12)) / max(stats['total_interactions'], 1),
                'afternoon_rate': sum(stats['hour_interactions'][h] for h in range(12, 18)) / max(stats['total_interactions'], 1),
                'evening_rate': sum(stats['hour_interactions'][h] for h in range(18, 24)) / max(stats['total_interactions'], 1),
                'night_rate': sum(stats['hour_interactions'][h] for h in range(0, 6)) / max(stats['total_interactions'], 1),

                # 地理特征
                'unique_geo_count': len(stats['geo_interactions']),
                'geo_concentration': max(stats['geo_interactions'].values()) / max(stats['total_interactions'], 1) if stats['geo_interactions'] else 0,
            }

            # 计算趋势特征
            day_counts = [stats['day_interactions'].get(d, 0) for d in [16, 17, 18]]
            features['trend_slope'] = self._calculate_trend(day_counts)
            features['trend_volatility'] = np.std(day_counts) if len(day_counts) > 1 else 0

            # 活跃小时数
            features['active_hours_count'] = sum(1 for count in stats['hour_interactions'].values() if count > 0)

            features_list.append(features)

        return pd.DataFrame(features_list)

    def _calculate_trend(self, day_counts):
        """计算趋势斜率"""
        if len(day_counts) < 2:
            return 0

        x = np.arange(len(day_counts))
        y = np.array(day_counts)

        # 简单线性回归斜率
        if np.sum((x - np.mean(x))**2) == 0:
            return 0

        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
        return slope

    def calculate_category_features(self, item_df):
        """计算类别级别特征"""
        print("🏷️ 计算类别特征...")

        category_stats = defaultdict(lambda: {
            'total_interactions': 0,
            'total_purchases': 0,
            'item_count': 0
        })

        # 聚合类别统计
        for item_id, stats in self.item_stats.items():
            category = self.item_catalog.get(item_id, -1)
            if category != -1:
                category_stats[category]['total_interactions'] += stats['total_interactions']
                category_stats[category]['total_purchases'] += stats['purchase_count']
                category_stats[category]['item_count'] += 1

        # 为每个商品添加类别特征
        for features in item_df:
            category = features['item_category']
            if category in category_stats:
                cat_stats = category_stats[category]
                features['category_popularity'] = cat_stats['total_interactions']
                features['category_purchase_rate'] = cat_stats['total_purchases'] / max(cat_stats['total_interactions'], 1)
                features['category_competition'] = cat_stats['item_count']  # 类别内商品竞争度
            else:
                features['category_popularity'] = 0
                features['category_purchase_rate'] = 0
                features['category_competition'] = 1

        return item_df

    def export_features(self):
        """导出商品特征"""
        print("\n💾 导出商品特征...")

        # 生成特征
        item_features_df = self.generate_item_features()

        # 添加类别特征
        features_list = item_features_df.to_dict('records')
        features_list = self.calculate_category_features(features_list)
        item_features_df = pd.DataFrame(features_list)

        # 保存特征
        output_file = os.path.join(self.output_dir, "item_features.csv")
        item_features_df.to_csv(output_file, index=False)

        print(f"✅ 商品特征已保存: {output_file}")
        print(f"  📦 商品数: {len(item_features_df):,}")
        print(f"  🔧 特征数: {len(item_features_df.columns)-1}")

        # 显示特征示例
        print(f"\n📋 特征示例:")
        sample_features = ['item_id', 'total_interactions', 'purchase_count', 'purchase_rate', 'trend_slope']
        print(item_features_df[sample_features].head())

        # 特征统计
        print(f"\n📊 特征统计:")
        print(f"  🔥 最热门商品交互数: {item_features_df['total_interactions'].max():,}")
        print(f"  💰 最高购买率: {item_features_df['purchase_rate'].max():.3f}")
        print(f"  📈 平均趋势斜率: {item_features_df['trend_slope'].mean():.2f}")

        return item_features_df


def main():
    """主函数"""
    print("=== 商品特征提取器 ===")
    print("🎯 目标：从16-18号数据提取商品特征")

    extractor = ItemFeatureExtractor()

    # 1. 加载商品目录
    item_catalog_df = extractor.load_item_catalog()

    # 2. 处理数据文件
    extractor.process_data_files()

    # 3. 生成并导出特征
    item_features_df = extractor.export_features()

    print(f"\n🎉 商品特征提取完成!")
    print(f"📁 输出目录: {extractor.output_dir}")
    print(f"下一步：生成用户-商品交互训练样本")


if __name__ == "__main__":
    main()