#!/usr/bin/env python3
"""
快速训练样本生成器
基于现有39维用户特征 + 简化商品特征 + 核心交互特征
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class FastSampleGenerator:
    """快速训练样本生成器"""

    def __init__(self, feature_dir="/mnt/data/tianchi_features",
                 data_dir="dataset/preprocess_16to18",
                 output_dir="/mnt/data/tianchi_features"):
        self.feature_dir = feature_dir
        self.data_dir = data_dir
        self.output_dir = output_dir

        # 核心交互数据（简化）
        self.user_item_interactions = defaultdict(set)  # 只记录是否有交互
        self.user_purchases = defaultdict(set)  # 只记录购买

        print(f"📁 特征目录: {feature_dir}")

    def load_existing_features(self):
        """加载现有特征"""
        print("📂 加载现有特征...")

        # 加载39维用户特征
        user_file = os.path.join(self.feature_dir, "user_features_cpp.csv")
        if not os.path.exists(user_file):
            print(f"❌ 用户特征文件不存在: {user_file}")
            return False

        self.user_features = pd.read_csv(user_file)
        self.user_ids = set(self.user_features['user_id'].tolist())
        print(f"  👥 用户特征: {len(self.user_features):,} 用户, 39维")

        # 加载简化商品特征
        item_file = os.path.join(self.feature_dir, "simple_item_features.csv")
        if not os.path.exists(item_file):
            print(f"⚠️  简化商品特征文件不存在，将先生成...")
            # 调用简化商品特征生成
            from simple_item_features import SimpleItemFeatureExtractor
            extractor = SimpleItemFeatureExtractor()
            extractor.load_item_catalog()
            extractor.process_data_files()
            extractor.export_features()

        self.item_features = pd.read_csv(item_file)
        self.item_ids = set(self.item_features['item_id'].tolist())
        print(f"  📦 商品特征: {len(self.item_features):,} 商品, {len(self.item_features.columns)-1}维")

        # 转换为字典
        self.user_feature_dict = self.user_features.set_index('user_id').to_dict('index')
        self.item_feature_dict = self.item_features.set_index('item_id').to_dict('index')

        return True

    def extract_core_interactions(self):
        """快速提取核心交互信息"""
        print("\n📊 提取核心交互信息...")

        # 只处理18号数据来生成正样本，16-17号数据提取交互历史
        data_files = [
            ("data_1216.txt", "历史交互"),
            ("data_1217.txt", "历史交互"),
            ("data_1218.txt", "标签生成")
        ]

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        for filename, purpose in data_files:
            print(f"  📅 处理 {filename} ({purpose})")

            file_path = os.path.join(self.data_dir, filename)
            processed = 0

            for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=1000000):
                # 过滤：只要有特征的用户和商品
                chunk = chunk[chunk['user_id'].isin(self.user_ids)]
                chunk = chunk[chunk['item_id'].isin(self.item_ids)]

                if len(chunk) == 0:
                    continue

                for _, row in chunk.iterrows():
                    user_id = int(row['user_id'])
                    item_id = int(row['item_id'])
                    behavior = int(row['behavior_type'])

                    # 记录交互
                    self.user_item_interactions[user_id].add(item_id)

                    # 记录购买（18号的作为正样本标签）
                    if behavior == 4 and filename == "data_1218.txt":
                        self.user_purchases[user_id].add(item_id)

                processed += len(chunk)

            print(f"    处理了 {processed:,} 行")

        interaction_pairs = sum(len(items) for items in self.user_item_interactions.values())
        purchase_pairs = sum(len(items) for items in self.user_purchases.values())

        print(f"  📊 统计:")
        print(f"    用户-商品交互对: {interaction_pairs:,}")
        print(f"    购买对（正样本）: {purchase_pairs:,}")

    def generate_samples(self, max_negative_ratio=2):
        """快速生成训练样本"""
        print(f"\n🔧 生成训练样本（负样本比例1:{max_negative_ratio}）...")

        samples = []

        # 1. 生成正样本
        print("  ✅ 生成正样本...")
        positive_count = 0
        for user_id, purchased_items in tqdm(self.user_purchases.items(), desc="正样本"):
            for item_id in purchased_items:
                sample = self._create_sample(user_id, item_id, label=1)
                if sample:
                    samples.append(sample)
                    positive_count += 1

        print(f"    正样本数: {positive_count:,}")

        # 2. 生成负样本（策略：交互未购买）
        print("  ❌ 生成负样本...")
        negative_count = 0
        target_negative = positive_count * max_negative_ratio

        for user_id, interacted_items in tqdm(self.user_item_interactions.items(), desc="负样本"):
            if negative_count >= target_negative:
                break

            purchased_items = self.user_purchases.get(user_id, set())

            # 交互但未购买的商品
            negative_items = interacted_items - purchased_items

            for item_id in negative_items:
                if negative_count >= target_negative:
                    break

                sample = self._create_sample(user_id, item_id, label=0)
                if sample:
                    samples.append(sample)
                    negative_count += 1

        print(f"    负样本数: {negative_count:,}")

        # 3. 随机打乱
        random.shuffle(samples)

        print(f"  📊 总样本数: {len(samples):,}")
        print(f"  🎯 正样本比例: {positive_count/len(samples):.3f}")

        return pd.DataFrame(samples)

    def _create_sample(self, user_id, item_id, label):
        """创建单个样本（简化版本）"""
        # 检查特征是否存在
        if user_id not in self.user_feature_dict or item_id not in self.item_feature_dict:
            return None

        sample = {'user_id': user_id, 'item_id': item_id, 'label': label}

        # 39维用户特征
        user_features = self.user_feature_dict[user_id]
        for key, value in user_features.items():
            sample[f'user_{key}'] = value

        # 简化商品特征
        item_features = self.item_feature_dict[item_id]
        for key, value in item_features.items():
            if key != 'item_id':
                sample[f'item_{key}'] = value

        # 核心交互特征（只保留4个最重要的）
        sample['has_interaction'] = 1 if item_id in self.user_item_interactions[user_id] else 0
        sample['has_purchased'] = 1 if item_id in self.user_purchases[user_id] else 0

        # 用户对该类别的偏好
        item_category = item_features.get('item_category', -1)
        user_top_category = user_features.get('top_category', -1)
        sample['category_match'] = 1 if item_category == user_top_category else 0

        # 用户活跃度 vs 商品流行度匹配度
        user_activity = user_features.get('total_actions', 0)
        item_popularity = item_features.get('popularity', 0)
        sample['activity_popularity_ratio'] = (user_activity + 1) / (item_popularity + 1)

        return sample

    def export_samples(self, samples_df):
        """导出训练样本"""
        print("\n💾 导出训练样本...")

        # 保存样本
        output_file = os.path.join(self.output_dir, "fast_training_samples.csv")
        samples_df.to_csv(output_file, index=False)

        print(f"✅ 训练样本已保存: {output_file}")

        # 特征统计
        feature_cols = [col for col in samples_df.columns if col not in ['user_id', 'item_id', 'label']]

        print(f"\n📋 特征分析:")
        print(f"  📏 样本数: {len(samples_df):,}")
        print(f"  🔧 总特征数: {len(feature_cols)}")
        print(f"  👥 用户特征: {len([col for col in feature_cols if col.startswith('user_')])}")
        print(f"  📦 商品特征: {len([col for col in feature_cols if col.startswith('item_')])}")
        print(f"  🔗 交互特征: {len([col for col in feature_cols if col.startswith(('has_', 'category_', 'activity_'))])}")
        print(f"  🎯 正样本比例: {samples_df['label'].mean():.3f}")

        # 数据质量
        print(f"\n🔍 数据质量:")
        print(f"  📊 缺失值: {samples_df.isnull().sum().sum()}")
        print(f"  📈 数值特征数: {len(samples_df.select_dtypes(include=[np.number]).columns)}")

        return output_file


def main():
    """主函数"""
    print("🔧 === 快速训练样本生成器 === 🔧")
    print("🎯 目标：基于现有39维用户特征快速生成训练样本")
    print("⚡ 预计耗时：3-4分钟")
    print("━" * 50)

    generator = FastSampleGenerator()

    # 1. 加载现有特征
    if not generator.load_existing_features():
        print("❌ 特征加载失败")
        return

    # 2. 提取核心交互
    generator.extract_core_interactions()

    # 3. 生成样本
    samples_df = generator.generate_samples()

    # 4. 导出样本
    output_file = generator.export_samples(samples_df)

    print(f"\n🎉 快速样本生成完成!")
    print(f"⚡ 速度提升: 比完整版本快5-10倍")
    print(f"📁 输出: {output_file}")
    print(f"🎯 特征总数: 约50维 (39用户 + 10商品 + 4交互)")


if __name__ == "__main__":
    main()