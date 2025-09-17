#!/usr/bin/env python3
"""
分批用户特征提取器
处理全量数据，为每个用户生成统计特征
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
import os
from tqdm import tqdm
import gc

sys.path.append('src')


class BatchUserFeatureExtractor:
    """分批用户特征提取器"""

    def __init__(self, output_dir="/mnt/data/tianchi_features"):
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")

        self.user_stats = defaultdict(lambda: {
            # 基础统计
            'total_actions': 0,
            'browse_count': 0,
            'collect_count': 0,
            'cart_count': 0,
            'purchase_count': 0,

            # 商品相关
            'unique_items': set(),
            'unique_categories': set(),
            'item_interactions': defaultdict(int),
            'category_preferences': defaultdict(int),

            # 时间模式
            'hour_activity': defaultdict(int),
            'day_activity': defaultdict(int),
            'first_action_time': None,
            'last_action_time': None,

            # 地理位置
            'geo_locations': defaultdict(int),
            'primary_geo': None,

            # 行为序列
            'behavior_sequence': [],
            'recent_behaviors': [],  # 最近3天
        })

    def process_data_batch(self, file_path, batch_size=1000000):
        """分批处理单个数据文件"""
        print(f"📂 处理文件: {file_path}")

        # 先计算文件总行数
        print("  📏 计算文件大小...")
        total_lines = sum(1 for _ in open(file_path, 'r'))
        estimated_chunks = (total_lines // batch_size) + 1
        print(f"  📊 文件总行数: {total_lines:,}")
        print(f"  🔢 预计批次数: {estimated_chunks}")

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
        chunk_reader = pd.read_csv(file_path, sep="\t", names=columns, chunksize=batch_size)

        processed_rows = 0
        valid_rows = 0

        for chunk_num, chunk in enumerate(chunk_reader):
            processed_rows += len(chunk)
            progress_pct = (processed_rows / total_lines) * 100

            print(f"\n  🔄 批次 {chunk_num + 1}/{estimated_chunks}")
            print(f"     📥 原始行数: {len(chunk):,}")
            print(f"     📈 总进度: {progress_pct:.1f}% ({processed_rows:,}/{total_lines:,})")

            # 数据清洗
            print(f"     🧹 数据清洗中...")
            chunk = chunk[chunk["behavior_type"].isin([1, 2, 3, 4])]
            chunk = chunk.dropna(subset=["user_id", "item_id", "time"])
            print(f"     ✓ 清洗后: {len(chunk):,} 行")

            # 解析时间
            print(f"     🕐 解析时间中...")
            chunk["datetime"] = pd.to_datetime(chunk["time"], format="%Y-%m-%d %H", errors="coerce")
            chunk = chunk.dropna(subset=["datetime"])
            print(f"     ✓ 时间解析后: {len(chunk):,} 行")

            # 添加时间特征
            chunk["hour"] = chunk["datetime"].dt.hour
            chunk["date"] = chunk["datetime"].dt.date
            chunk["weekday"] = chunk["datetime"].dt.weekday

            # 处理每一行数据
            print(f"     👥 更新用户统计中...")
            users_in_chunk = set()

            for idx, row in tqdm(chunk.iterrows(),
                               total=len(chunk),
                               desc=f"     处理行数",
                               leave=False,
                               miniters=len(chunk)//10):
                self._update_user_stats(row)
                users_in_chunk.add(int(row["user_id"]))

            valid_rows += len(chunk)

            print(f"     ✅ 批次完成")
            print(f"     👤 新增用户数: {len(users_in_chunk)}")
            print(f"     📊 累计用户数: {len(self.user_stats)}")
            print(f"     📝 累计有效行数: {valid_rows:,}")

            # 每5个批次保存一次检查点
            if (chunk_num + 1) % 5 == 0:
                checkpoint_name = os.path.join(self.output_dir, f"checkpoint_batch_{chunk_num + 1}.pkl")
                print(f"     💾 保存检查点: {checkpoint_name}")
                self.save_checkpoint(checkpoint_name)
                print(f"     ✅ 检查点已保存")

            # 内存清理
            del chunk
            gc.collect()

        print(f"\n  🎉 文件处理完成!")
        print(f"     📊 最终统计:")
        print(f"     - 处理行数: {processed_rows:,}")
        print(f"     - 有效行数: {valid_rows:,}")
        print(f"     - 用户数量: {len(self.user_stats):,}")
        print(f"     - 数据利用率: {(valid_rows/processed_rows)*100:.1f}%")

    def _update_user_stats(self, row):
        """更新单个用户的统计信息"""
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])
        behavior = int(row["behavior_type"])
        category = row["item_category"]
        geo = row["user_geohash"] if pd.notna(row["user_geohash"]) else "unknown"
        datetime_obj = row["datetime"]
        hour = int(row["hour"])
        date = row["date"]

        stats = self.user_stats[user_id]

        # 基础统计
        stats['total_actions'] += 1
        if behavior == 1:
            stats['browse_count'] += 1
        elif behavior == 2:
            stats['collect_count'] += 1
        elif behavior == 3:
            stats['cart_count'] += 1
        elif behavior == 4:
            stats['purchase_count'] += 1

        # 商品和类别
        stats['unique_items'].add(item_id)
        if pd.notna(category):
            stats['unique_categories'].add(category)
            stats['category_preferences'][category] += 1
        stats['item_interactions'][item_id] += 1

        # 时间模式
        stats['hour_activity'][hour] += 1
        stats['day_activity'][date] += 1

        if stats['first_action_time'] is None or datetime_obj < stats['first_action_time']:
            stats['first_action_time'] = datetime_obj
        if stats['last_action_time'] is None or datetime_obj > stats['last_action_time']:
            stats['last_action_time'] = datetime_obj

        # 地理位置
        stats['geo_locations'][geo] += 1
        if stats['primary_geo'] is None or stats['geo_locations'][geo] > stats['geo_locations'].get(stats['primary_geo'], 0):
            stats['primary_geo'] = geo

        # 行为序列 (保留最近1000个)
        stats['behavior_sequence'].append((datetime_obj, behavior, item_id))
        if len(stats['behavior_sequence']) > 1000:
            stats['behavior_sequence'] = stats['behavior_sequence'][-1000:]

        # 最近行为 (最近3天)
        recent_threshold = pd.to_datetime("2014-12-18") - pd.Timedelta(days=3)
        if datetime_obj >= recent_threshold:
            stats['recent_behaviors'].append((datetime_obj, behavior, item_id))

    def generate_user_features(self):
        """从原始统计生成机器学习特征"""
        print("🔧 生成用户特征...")

        features_list = []

        for user_id, stats in tqdm(self.user_stats.items(), desc="生成特征"):
            features = {
                'user_id': user_id,

                # 基础活跃度特征
                'total_actions': stats['total_actions'],
                'browse_count': stats['browse_count'],
                'collect_count': stats['collect_count'],
                'cart_count': stats['cart_count'],
                'purchase_count': stats['purchase_count'],

                # 转化率特征
                'collect_rate': stats['collect_count'] / max(stats['browse_count'], 1),
                'cart_rate': stats['cart_count'] / max(stats['browse_count'], 1),
                'purchase_rate': stats['purchase_count'] / max(stats['browse_count'], 1),
                'purchase_conversion': stats['purchase_count'] / max(stats['total_actions'], 1),

                # 商品和类别多样性
                'unique_items_count': len(stats['unique_items']),
                'unique_categories_count': len(stats['unique_categories']),
                'avg_interactions_per_item': stats['total_actions'] / max(len(stats['unique_items']), 1),

                # 时间模式特征
                'active_days': len(stats['day_activity']),
                'avg_daily_actions': stats['total_actions'] / max(len(stats['day_activity']), 1),
                'active_hours_count': len(stats['hour_activity']),

                # 时间偏好
                'morning_activity': sum(stats['hour_activity'][h] for h in range(6, 12)) / max(stats['total_actions'], 1),
                'afternoon_activity': sum(stats['hour_activity'][h] for h in range(12, 18)) / max(stats['total_actions'], 1),
                'evening_activity': sum(stats['hour_activity'][h] for h in range(18, 24)) / max(stats['total_actions'], 1),
                'night_activity': sum(stats['hour_activity'][h] for h in range(0, 6)) / max(stats['total_actions'], 1),

                # 地理特征
                'unique_geo_count': len(stats['geo_locations']),
                'geo_concentration': max(stats['geo_locations'].values()) / max(stats['total_actions'], 1) if stats['geo_locations'] else 0,

                # 最近活跃度
                'recent_actions_count': len(stats['recent_behaviors']),
                'recent_purchase_count': sum(1 for _, behavior, _ in stats['recent_behaviors'] if behavior == 4),
                'days_since_last_action': 0,  # 会在后面计算
                'days_since_first_action': 0,  # 会在后面计算
            }

            # 计算时间间隔
            if stats['last_action_time']:
                features['days_since_last_action'] = (pd.to_datetime("2014-12-18") - stats['last_action_time']).days
            if stats['first_action_time']:
                features['days_since_first_action'] = (pd.to_datetime("2014-12-18") - stats['first_action_time']).days

            # 最喜欢的类别
            if stats['category_preferences']:
                top_category = max(stats['category_preferences'], key=stats['category_preferences'].get)
                features['top_category'] = top_category
                features['top_category_ratio'] = stats['category_preferences'][top_category] / max(stats['total_actions'], 1)
            else:
                features['top_category'] = -1
                features['top_category_ratio'] = 0

            # 行为规律性
            if len(stats['hour_activity']) > 1:
                hour_counts = np.array(list(stats['hour_activity'].values()))
                features['activity_regularity'] = 1 - (hour_counts.std() / max(hour_counts.mean(), 1))
            else:
                features['activity_regularity'] = 0

            # 购买预测特征 (基于最近行为)
            features['likely_to_purchase'] = self._predict_purchase_likelihood(stats)

            features_list.append(features)

        return pd.DataFrame(features_list)

    def _predict_purchase_likelihood(self, stats):
        """预测购买可能性 (启发式)"""
        score = 0

        # 历史购买行为权重
        if stats['purchase_count'] > 0:
            score += 0.4
        if stats['purchase_count'] > 2:
            score += 0.2

        # 最近活跃度
        if len(stats['recent_behaviors']) > 5:
            score += 0.2

        # 最近购买行为
        recent_purchases = sum(1 for _, behavior, _ in stats['recent_behaviors'] if behavior == 4)
        if recent_purchases > 0:
            score += 0.3

        # 高转化行为
        if stats['cart_count'] > 0:
            score += 0.1
        if stats['collect_count'] > 0:
            score += 0.05

        return min(score, 1.0)

    def save_checkpoint(self, filename=None):
        """保存检查点"""
        if filename is None:
            filename = os.path.join(self.output_dir, "user_stats_checkpoint.pkl")

        print(f"💾 保存检查点: {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.user_stats), f)

    def load_checkpoint(self, filename=None):
        """加载最新的检查点"""
        # 在输出目录中寻找所有可能的检查点文件
        checkpoint_files = []

        # 默认检查点
        if filename is None:
            filename = os.path.join(self.output_dir, "user_stats_checkpoint.pkl")
        if os.path.exists(filename):
            checkpoint_files.append(filename)

        # 批次检查点
        import glob
        batch_checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint_batch_*.pkl"))
        checkpoint_files.extend(batch_checkpoints)

        # 文件检查点
        file_checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint_after_file_*.pkl"))
        checkpoint_files.extend(file_checkpoints)

        if checkpoint_files:
            # 选择最新的检查点
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"📂 找到 {len(checkpoint_files)} 个检查点文件")
            print(f"📂 加载最新检查点: {latest_checkpoint}")

            with open(latest_checkpoint, 'rb') as f:
                loaded_stats = pickle.load(f)
                for user_id, stats in loaded_stats.items():
                    self.user_stats[user_id] = stats

            print(f"✅ 检查点加载完成")
            print(f"  👥 恢复用户数: {len(self.user_stats):,}")
            total_actions = sum(stats['total_actions'] for stats in self.user_stats.values())
            print(f"  📈 恢复交互数: {total_actions:,}")

            return True

        print("📂 未找到检查点文件，从头开始处理")
        return False


def main():
    """主函数"""
    print("=== 分批用户特征提取器 ===")
    print("🎯 目标：处理全量11.65亿行数据，生成用户特征")

    extractor = BatchUserFeatureExtractor()

    # 尝试加载检查点
    if not extractor.load_checkpoint():
        print("\n🚀 开始全量数据处理...")

        # 处理数据文件
        files = [
            "dataset/tianchi_fresh_comp_train_user_online_partA.txt",
            "dataset/tianchi_fresh_comp_train_user_online_partB.txt"
        ]

        print(f"📁 待处理文件数: {len(files)}")

        for file_idx, file_path in enumerate(files, 1):
            print(f"\n{'='*60}")
            print(f"🗂️  文件 {file_idx}/{len(files)}: {file_path}")
            print(f"{'='*60}")

            if os.path.exists(file_path):
                # 显示文件大小
                file_size = os.path.getsize(file_path) / (1024**3)  # GB
                print(f"📏 文件大小: {file_size:.1f} GB")

                extractor.process_data_batch(file_path)

                # 每处理完一个文件保存检查点
                print(f"\n💾 保存进度检查点...")
                checkpoint_path = os.path.join(extractor.output_dir, f"checkpoint_after_file_{file_idx}.pkl")
                extractor.save_checkpoint(checkpoint_path)

                print(f"\n📊 当前进度汇总:")
                print(f"  ✅ 已完成文件: {file_idx}/{len(files)}")
                print(f"  👥 累计用户数: {len(extractor.user_stats):,}")
                total_actions = sum(stats['total_actions'] for stats in extractor.user_stats.values())
                print(f"  📈 累计交互数: {total_actions:,}")
                print(f"  💾 内存占用: {len(extractor.user_stats) * 0.001:.1f} MB (估算)")

            else:
                print(f"❌ 文件不存在: {file_path}")

        print(f"\n🎉 全部文件处理完成!")

    else:
        print("📂 从检查点恢复数据...")

    print(f"\n📊 最终统计信息:")
    print(f"  👥 总用户数: {len(extractor.user_stats):,}")
    total_actions = sum(stats['total_actions'] for stats in extractor.user_stats.values())
    print(f"  📈 总交互数: {total_actions:,}")
    avg_actions = total_actions / len(extractor.user_stats) if extractor.user_stats else 0
    print(f"  📊 平均每用户交互数: {avg_actions:.1f}")

    # 生成特征
    print(f"\n🔧 开始生成机器学习特征...")
    features_df = extractor.generate_user_features()

    # 保存特征
    print(f"\n💾 保存特征文件...")
    feature_file_path = os.path.join(extractor.output_dir, "user_features_full.csv")
    features_df.to_csv(feature_file_path, index=False)

    print(f"\n✅ 特征提取完成!")
    print(f"  📁 输出目录: {extractor.output_dir}")
    print(f"  📄 特征文件: {feature_file_path}")
    print(f"  👥 用户数: {len(features_df):,}")
    print(f"  🔧 特征数: {len(features_df.columns)-1}")
    print(f"  🎯 预测购买用户比例: {features_df['likely_to_purchase'].mean():.3f}")

    # 显示特征示例
    print(f"\n📋 特征示例 (前5个用户):")
    print(features_df[['user_id', 'total_actions', 'purchase_count', 'unique_items_count', 'likely_to_purchase']].head())

    print(f"\n🎊 特征工程完成！下一步可以用这些特征训练机器学习模型")
    print(f"📁 所有文件都保存在: {extractor.output_dir}")


if __name__ == "__main__":
    main()