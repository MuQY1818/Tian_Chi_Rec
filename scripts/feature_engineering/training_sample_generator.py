#!/usr/bin/env python3
"""
训练样本生成器
构建用户-商品交互预测的训练样本
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


class TrainingSampleGenerator:
    """训练样本生成器"""

    def __init__(self, data_dir="dataset/preprocess_16to18",
                 feature_dir="/mnt/data/tianchi_features",
                 output_dir="/mnt/data/tianchi_features"):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.user_features = None
        self.item_features = None
        self.user_item_interactions = defaultdict(lambda: defaultdict(list))

        print(f"📁 数据目录: {data_dir}")
        print(f"📁 特征目录: {feature_dir}")
        print(f"📁 输出目录: {output_dir}")

    def load_features(self):
        """加载用户和商品特征"""
        print("📂 加载特征数据...")

        # 加载用户特征
        user_feature_file = os.path.join(self.feature_dir, "user_features_cpp.csv")
        if os.path.exists(user_feature_file):
            self.user_features = pd.read_csv(user_feature_file)
            print(f"  👥 用户特征: {len(self.user_features):,} 用户, {len(self.user_features.columns)-1} 维特征")
        else:
            print(f"  ❌ 未找到用户特征文件: {user_feature_file}")
            return False

        # 加载商品特征
        item_feature_file = os.path.join(self.feature_dir, "item_features.csv")
        if os.path.exists(item_feature_file):
            self.item_features = pd.read_csv(item_feature_file)
            print(f"  📦 商品特征: {len(self.item_features):,} 商品, {len(self.item_features.columns)-1} 维特征")
        else:
            print(f"  ❌ 未找到商品特征文件: {item_feature_file}")
            return False

        # 转换为字典便于查找
        self.user_feature_dict = self.user_features.set_index('user_id').to_dict('index')
        self.item_feature_dict = self.item_features.set_index('item_id').to_dict('index')

        return True

    def load_item_catalog(self):
        """加载商品目录(P子集)"""
        print("📚 加载商品子集P...")

        item_file = "dataset/tianchi_fresh_comp_train_item_online.txt"
        columns = ["item_id", "item_geohash", "item_category"]

        item_df = pd.read_csv(item_file, sep="\t", names=columns)
        # 处理空的geohash
        item_df['item_geohash'] = item_df['item_geohash'].fillna('unknown')

        self.p_items = set(item_df['item_id'].tolist())
        print(f"  📦 P子集商品数: {len(self.p_items):,}")
        print(f"  🏷️  商品类别数: {item_df['item_category'].nunique()}")

        return item_df

    def extract_user_item_interactions(self):
        """提取用户-商品交互历史"""
        print("\n📊 提取用户-商品交互历史...")

        data_files = [
            ("data_1216.txt", 16),
            ("data_1217.txt", 17),
            ("data_1218.txt", 18)
        ]

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        for filename, day in data_files:
            print(f"\n📅 处理第{day}号数据: {filename}")

            file_path = os.path.join(self.data_dir, filename)
            chunk_size = 1000000
            chunk_count = 0

            for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size):
                chunk_count += 1
                print(f"  🔄 处理块 {chunk_count} (行数: {len(chunk):,})")

                # 只处理P子集商品和有特征的用户
                chunk = chunk[chunk['item_id'].isin(self.p_items)]
                chunk = chunk[chunk['user_id'].isin(self.user_feature_dict.keys())]

                if len(chunk) == 0:
                    continue

                # 解析时间
                chunk['datetime'] = pd.to_datetime(chunk['time'], format="%Y-%m-%d %H", errors="coerce")
                chunk = chunk.dropna(subset=['datetime'])

                # 提取交互信息
                for _, row in tqdm(chunk.iterrows(),
                                 total=len(chunk),
                                 desc=f"    提取交互",
                                 leave=False,
                                 miniters=len(chunk)//10):

                    user_id = int(row['user_id'])
                    item_id = int(row['item_id'])
                    behavior = int(row['behavior_type'])
                    timestamp = row['datetime']

                    # 记录交互
                    self.user_item_interactions[user_id][item_id].append({
                        'behavior': behavior,
                        'timestamp': timestamp,
                        'day': day
                    })

            print(f"  ✅ 第{day}号数据处理完成")

        print(f"\n📊 交互统计:")
        total_users = len(self.user_item_interactions)
        total_pairs = sum(len(items) for items in self.user_item_interactions.values())
        print(f"  👥 有交互用户数: {total_users:,}")
        print(f"  🔗 用户-商品对数: {total_pairs:,}")

    def generate_interaction_features(self, user_id, item_id):
        """生成用户-商品交互特征"""
        interactions = self.user_item_interactions[user_id].get(item_id, [])

        # 基础交互特征
        features = {
            'has_interaction': 1 if interactions else 0,
            'total_interactions': len(interactions),
            'browse_count': sum(1 for i in interactions if i['behavior'] == 1),
            'collect_count': sum(1 for i in interactions if i['behavior'] == 2),
            'cart_count': sum(1 for i in interactions if i['behavior'] == 3),
            'purchase_count': sum(1 for i in interactions if i['behavior'] == 4),
        }

        if interactions:
            # 时间特征
            timestamps = [i['timestamp'] for i in interactions]
            features['first_interaction_days_ago'] = (pd.Timestamp('2014-12-19') - min(timestamps)).days
            features['last_interaction_days_ago'] = (pd.Timestamp('2014-12-19') - max(timestamps)).days

            # 最近交互
            recent_interactions = [i for i in interactions if i['day'] >= 17]  # 最近2天
            features['recent_interactions'] = len(recent_interactions)
            features['recent_purchase_count'] = sum(1 for i in recent_interactions if i['behavior'] == 4)

            # 行为进展 (是否有购买倾向)
            features['max_behavior_type'] = max(i['behavior'] for i in interactions)
            features['behavior_progression'] = features['max_behavior_type'] / 4.0

            # 交互频率
            if len(set(i['day'] for i in interactions)) > 1:
                features['interaction_frequency'] = len(interactions) / len(set(i['day'] for i in interactions))
            else:
                features['interaction_frequency'] = len(interactions)

        else:
            # 无交互的默认值
            features.update({
                'first_interaction_days_ago': 999,
                'last_interaction_days_ago': 999,
                'recent_interactions': 0,
                'recent_purchase_count': 0,
                'max_behavior_type': 0,
                'behavior_progression': 0,
                'interaction_frequency': 0,
            })

        return features

    def generate_positive_samples(self):
        """生成正样本：18号实际购买的用户-商品对"""
        print("\n✅ 生成正样本...")

        positive_samples = []

        # 从18号数据中找购买行为
        file_path = os.path.join(self.data_dir, "data_1218.txt")
        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

        chunk_size = 1000000
        total_positive = 0

        for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size):
            # 只要购买行为
            purchase_chunk = chunk[chunk['behavior_type'] == 4]

            # 只处理P子集商品和有特征的用户
            purchase_chunk = purchase_chunk[purchase_chunk['item_id'].isin(self.p_items)]
            purchase_chunk = purchase_chunk[purchase_chunk['user_id'].isin(self.user_feature_dict.keys())]

            if len(purchase_chunk) == 0:
                continue

            for _, row in tqdm(purchase_chunk.iterrows(),
                             desc="生成正样本",
                             leave=False):
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])

                # 检查是否有特征
                if user_id in self.user_feature_dict and item_id in self.item_feature_dict:
                    positive_samples.append((user_id, item_id, 1))  # 标签为1
                    total_positive += 1

        print(f"  ✅ 正样本数量: {total_positive:,}")
        return positive_samples

    def generate_negative_samples(self, positive_samples, ratio=3):
        """生成负样本：用户交互但未购买的商品 + 随机负样本"""
        print(f"\n❌ 生成负样本 (正负比例 1:{ratio})...")

        negative_samples = []
        positive_set = set((user_id, item_id) for user_id, item_id, _ in positive_samples)
        target_negative = len(positive_samples) * ratio

        # 策略1: 用户交互但未购买的商品
        print("  📊 策略1: 交互未购买")
        interaction_negatives = 0

        for user_id, item_interactions in tqdm(self.user_item_interactions.items(),
                                              desc="  扫描交互"):
            if user_id not in self.user_feature_dict:
                continue

            for item_id, interactions in item_interactions.items():
                if (user_id, item_id) in positive_set:
                    continue

                if item_id not in self.item_feature_dict:
                    continue

                # 检查是否有购买行为
                has_purchase = any(i['behavior'] == 4 for i in interactions)
                if not has_purchase:
                    negative_samples.append((user_id, item_id, 0))
                    interaction_negatives += 1

                    if len(negative_samples) >= target_negative:
                        break

            if len(negative_samples) >= target_negative:
                break

        print(f"    交互负样本: {interaction_negatives:,}")

        # 策略2: 随机负样本补充
        if len(negative_samples) < target_negative:
            print("  🎲 策略2: 随机负样本")

            users_with_features = list(self.user_feature_dict.keys())
            items_with_features = list(self.item_feature_dict.keys())

            remaining = target_negative - len(negative_samples)
            attempts = 0
            max_attempts = remaining * 10  # 防止无限循环

            for _ in tqdm(range(remaining), desc="  生成随机负样本"):
                if attempts > max_attempts:
                    break

                user_id = random.choice(users_with_features)
                item_id = random.choice(items_with_features)
                attempts += 1

                if (user_id, item_id) not in positive_set and (user_id, item_id, 0) not in negative_samples:
                    negative_samples.append((user_id, item_id, 0))

        print(f"  ❌ 负样本总数: {len(negative_samples):,}")
        return negative_samples[:target_negative]

    def build_training_samples(self):
        """构建完整的训练样本"""
        print("\n🔧 构建训练样本...")

        # 生成正负样本
        positive_samples = self.generate_positive_samples()
        negative_samples = self.generate_negative_samples(positive_samples)

        # 合并样本
        all_samples = positive_samples + negative_samples
        print(f"\n📊 样本统计:")
        print(f"  ✅ 正样本: {len(positive_samples):,}")
        print(f"  ❌ 负样本: {len(negative_samples):,}")
        print(f"  📝 总样本: {len(all_samples):,}")

        # 随机打乱
        random.shuffle(all_samples)

        # 构建特征矩阵
        print(f"\n🏗️  构建特征矩阵...")
        feature_data = []

        for user_id, item_id, label in tqdm(all_samples, desc="构建特征"):
            # 基础ID
            sample = {
                'user_id': user_id,
                'item_id': item_id,
                'label': label
            }

            # 用户特征 (39维)
            user_features = self.user_feature_dict.get(user_id, {})
            for key, value in user_features.items():
                sample[f'user_{key}'] = value

            # 商品特征
            item_features = self.item_feature_dict.get(item_id, {})
            for key, value in item_features.items():
                if key != 'item_id':  # 避免重复
                    sample[f'item_{key}'] = value

            # 交互特征
            interaction_features = self.generate_interaction_features(user_id, item_id)
            for key, value in interaction_features.items():
                sample[f'interaction_{key}'] = value

            feature_data.append(sample)

        # 转换为DataFrame
        feature_df = pd.DataFrame(feature_data)

        print(f"✅ 特征矩阵构建完成:")
        print(f"  📏 样本数: {len(feature_df):,}")
        print(f"  🔧 特征数: {len(feature_df.columns)-3}")  # 减去user_id, item_id, label
        print(f"  🎯 正样本比例: {feature_df['label'].mean():.3f}")

        return feature_df

    def export_training_data(self, feature_df):
        """导出训练数据"""
        print("\n💾 导出训练数据...")

        # 保存完整数据
        train_file = os.path.join(self.output_dir, "training_samples.csv")
        feature_df.to_csv(train_file, index=False)

        print(f"✅ 训练数据已保存: {train_file}")

        # 特征统计
        feature_cols = [col for col in feature_df.columns if col not in ['user_id', 'item_id', 'label']]
        print(f"\n📋 特征组分析:")
        print(f"  👥 用户特征数: {len([col for col in feature_cols if col.startswith('user_')])}")
        print(f"  📦 商品特征数: {len([col for col in feature_cols if col.startswith('item_')])}")
        print(f"  🔗 交互特征数: {len([col for col in feature_cols if col.startswith('interaction_')])}")

        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        print(f"  📊 缺失值: {feature_df.isnull().sum().sum()}")
        print(f"  📈 数值列数: {len(feature_df.select_dtypes(include=[np.number]).columns)}")

        # 保存特征名列表
        feature_names_file = os.path.join(self.output_dir, "feature_names.txt")
        with open(feature_names_file, 'w') as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        print(f"  📝 特征名已保存: {feature_names_file}")

        return train_file


def main():
    """主函数"""
    print("=== 训练样本生成器 ===")
    print("🎯 目标：生成用户-商品交互预测训练样本")

    generator = TrainingSampleGenerator()

    # 1. 加载特征
    if not generator.load_features():
        print("❌ 特征加载失败")
        return

    # 2. 加载商品目录
    generator.load_item_catalog()

    # 3. 提取交互历史
    generator.extract_user_item_interactions()

    # 4. 构建训练样本
    feature_df = generator.build_training_samples()

    # 5. 导出数据
    train_file = generator.export_training_data(feature_df)

    print(f"\n🎉 训练样本生成完成!")
    print(f"📁 输出文件: {train_file}")
    print(f"下一步：训练LightGBM模型")


if __name__ == "__main__":
    main()