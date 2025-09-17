#!/usr/bin/env python3
"""
推荐生成器
使用训练好的模型生成最终推荐列表
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class RecommendationGenerator:
    """推荐生成器"""

    def __init__(self, feature_dir="/mnt/data/tianchi_features",
                 model_dir="/mnt/data/tianchi_features",
                 output_dir="/mnt/data/tianchi_features"):
        self.feature_dir = feature_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = None
        self.feature_names = None
        self.user_features = None
        self.item_features = None

        print(f"📁 特征目录: {feature_dir}")
        print(f"📁 模型目录: {model_dir}")
        print(f"📁 输出目录: {output_dir}")

    def load_model(self):
        """加载训练好的模型"""
        print("📂 加载训练模型...")

        model_file = os.path.join(self.model_dir, "lightgbm_model.pkl")
        feature_file = os.path.join(self.model_dir, "feature_names.pkl")
        metrics_file = os.path.join(self.model_dir, "model_metrics.pkl")

        if not all(os.path.exists(f) for f in [model_file, feature_file]):
            print(f"❌ 模型文件缺失")
            return False

        self.model = joblib.load(model_file)
        self.feature_names = joblib.load(feature_file)

        if os.path.exists(metrics_file):
            self.metrics = joblib.load(metrics_file)
            print(f"✅ 模型加载成功 (AUC: {self.metrics['auc']:.4f})")
        else:
            print(f"✅ 模型加载成功")

        return True

    def load_features(self):
        """加载用户和商品特征"""
        print("📂 加载特征数据...")

        # 加载用户特征
        user_feature_file = os.path.join(self.feature_dir, "user_features_cpp.csv")
        if os.path.exists(user_feature_file):
            self.user_features = pd.read_csv(user_feature_file)
            print(f"  👥 用户特征: {len(self.user_features):,} 用户")
        else:
            print(f"  ❌ 用户特征文件不存在")
            return False

        # 加载商品特征
        item_feature_file = os.path.join(self.feature_dir, "item_features.csv")
        if os.path.exists(item_feature_file):
            self.item_features = pd.read_csv(item_feature_file)
            print(f"  📦 商品特征: {len(self.item_features):,} 商品")
        else:
            print(f"  ❌ 商品特征文件不存在")
            return False

        # 转换为字典便于查找
        self.user_feature_dict = self.user_features.set_index('user_id').to_dict('index')
        self.item_feature_dict = self.item_features.set_index('item_id').to_dict('index')

        return True

    def load_historical_interactions(self):
        """加载历史交互数据（用于生成交互特征）"""
        print("📊 加载历史交互数据...")

        data_dir = "dataset/preprocess_16to18"
        data_files = [
            ("data_1216.txt", 16),
            ("data_1217.txt", 17),
            ("data_1218.txt", 18)
        ]

        columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
        self.user_item_interactions = defaultdict(lambda: defaultdict(list))

        for filename, day in data_files:
            print(f"  📅 处理第{day}号数据...")

            file_path = os.path.join(data_dir, filename)
            chunk_size = 1000000

            for chunk in pd.read_csv(file_path, sep="\t", names=columns, chunksize=chunk_size):
                # 只处理有特征的用户和商品
                chunk = chunk[chunk['user_id'].isin(self.user_feature_dict.keys())]
                chunk = chunk[chunk['item_id'].isin(self.item_feature_dict.keys())]

                if len(chunk) == 0:
                    continue

                # 解析时间
                chunk['datetime'] = pd.to_datetime(chunk['time'], format="%Y-%m-%d %H", errors="coerce")
                chunk = chunk.dropna(subset=['datetime'])

                # 提取交互信息
                for _, row in chunk.iterrows():
                    user_id = int(row['user_id'])
                    item_id = int(row['item_id'])
                    behavior = int(row['behavior_type'])
                    timestamp = row['datetime']

                    self.user_item_interactions[user_id][item_id].append({
                        'behavior': behavior,
                        'timestamp': timestamp,
                        'day': day
                    })

        total_users = len(self.user_item_interactions)
        total_pairs = sum(len(items) for items in self.user_item_interactions.values())
        print(f"  ✅ 交互数据加载完成: {total_users:,} 用户, {total_pairs:,} 用户-商品对")

    def generate_interaction_features(self, user_id, item_id):
        """生成用户-商品交互特征（与训练时保持一致）"""
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

            # 行为进展
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

    def generate_candidate_items(self, user_id, top_k=100):
        """为用户生成候选商品"""
        # 策略1: 用户历史交互商品
        historical_items = set()
        if user_id in self.user_item_interactions:
            historical_items = set(self.user_item_interactions[user_id].keys())

        # 策略2: 热门商品
        popular_items = self.item_features.nlargest(50, 'total_interactions')['item_id'].tolist()

        # 策略3: 用户类别偏好商品
        user_feature = self.user_feature_dict.get(user_id, {})
        preferred_category = user_feature.get('top_category', -1)

        category_items = []
        if preferred_category != -1:
            category_items = self.item_features[
                self.item_features['item_category'] == preferred_category
            ].nlargest(30, 'total_interactions')['item_id'].tolist()

        # 合并候选集
        candidates = list(historical_items) + popular_items + category_items
        candidates = list(set(candidates))  # 去重

        # 确保有足够候选商品
        if len(candidates) < top_k:
            # 补充随机商品
            all_items = self.item_features['item_id'].tolist()
            additional_items = [item for item in all_items if item not in candidates]
            candidates.extend(additional_items[:top_k - len(candidates)])

        return candidates[:top_k]

    def predict_user_item_batch(self, user_id, item_ids, batch_size=1000):
        """批量预测用户对商品的购买概率"""
        results = []

        # 分批处理避免内存问题
        for i in range(0, len(item_ids), batch_size):
            batch_items = item_ids[i:i + batch_size]
            batch_features = []

            for item_id in batch_items:
                # 构建特征向量
                sample = {}

                # 用户特征
                user_features = self.user_feature_dict.get(user_id, {})
                for key, value in user_features.items():
                    sample[f'user_{key}'] = value

                # 商品特征
                item_features = self.item_feature_dict.get(item_id, {})
                for key, value in item_features.items():
                    if key != 'item_id':
                        sample[f'item_{key}'] = value

                # 交互特征
                interaction_features = self.generate_interaction_features(user_id, item_id)
                for key, value in interaction_features.items():
                    sample[f'interaction_{key}'] = value

                batch_features.append(sample)

            # 转换为DataFrame并确保特征顺序
            batch_df = pd.DataFrame(batch_features)

            # 确保所有特征都存在，缺失的用0填充
            for feature in self.feature_names:
                if feature not in batch_df.columns:
                    batch_df[feature] = 0

            # 按训练时的特征顺序排列
            batch_df = batch_df[self.feature_names]

            # 预测
            probabilities = self.model.predict(batch_df, num_iteration=self.model.best_iteration)

            # 保存结果
            for j, prob in enumerate(probabilities):
                results.append({
                    'item_id': batch_items[j],
                    'probability': prob
                })

        return results

    def generate_recommendations_for_user(self, user_id, top_k=50):
        """为单个用户生成推荐"""
        # 生成候选商品
        candidates = self.generate_candidate_items(user_id, top_k=100)

        # 预测购买概率
        predictions = self.predict_user_item_batch(user_id, candidates)

        # 排序并返回Top-K
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        return predictions[:top_k]

    def generate_all_recommendations(self, top_k=50):
        """为所有用户生成推荐"""
        print(f"\n🎯 生成推荐 (每用户Top-{top_k})...")

        all_recommendations = []
        users = list(self.user_feature_dict.keys())

        print(f"  👥 待处理用户数: {len(users):,}")

        for user_id in tqdm(users, desc="生成推荐"):
            try:
                user_recs = self.generate_recommendations_for_user(user_id, top_k)

                for rec in user_recs:
                    all_recommendations.append({
                        'user_id': user_id,
                        'item_id': rec['item_id'],
                        'probability': rec['probability']
                    })

            except Exception as e:
                print(f"  ⚠️  用户 {user_id} 推荐生成失败: {e}")
                continue

        print(f"✅ 推荐生成完成:")
        print(f"  📊 总推荐数: {len(all_recommendations):,}")
        print(f"  👥 用户数: {len(set(r['user_id'] for r in all_recommendations)):,}")

        return all_recommendations

    def format_submission(self, recommendations):
        """格式化为提交格式"""
        print("\n📝 格式化提交文件...")

        # 转换为DataFrame
        rec_df = pd.DataFrame(recommendations)

        # 按概率排序
        rec_df = rec_df.sort_values(['user_id', 'probability'], ascending=[True, False])

        # 生成最终提交格式 (user_id, item_id)
        submission = rec_df[['user_id', 'item_id']].copy()

        print(f"✅ 提交格式化完成:")
        print(f"  📊 推荐记录数: {len(submission):,}")
        print(f"  👥 用户数: {submission['user_id'].nunique():,}")
        print(f"  📦 商品数: {submission['item_id'].nunique():,}")

        return submission

    def export_recommendations(self, recommendations):
        """导出推荐结果"""
        print("\n💾 导出推荐结果...")

        # 导出详细推荐（包含概率）
        detailed_file = os.path.join(self.output_dir, "detailed_recommendations.csv")
        pd.DataFrame(recommendations).to_csv(detailed_file, index=False)
        print(f"  📊 详细推荐已保存: {detailed_file}")

        # 导出提交格式（tab分隔，无表头）
        submission = self.format_submission(recommendations)
        submission_file = os.path.join(self.output_dir, "final_submission.txt")
        submission.to_csv(submission_file, index=False, header=False, sep='\t')
        print(f"  📝 提交文件已保存: {submission_file}")

        # 统计信息
        stats = {
            'total_recommendations': len(recommendations),
            'unique_users': len(set(r['user_id'] for r in recommendations)),
            'unique_items': len(set(r['item_id'] for r in recommendations)),
            'avg_probability': np.mean([r['probability'] for r in recommendations]),
            'min_probability': min(r['probability'] for r in recommendations),
            'max_probability': max(r['probability'] for r in recommendations)
        }

        stats_file = os.path.join(self.output_dir, "recommendation_stats.txt")
        with open(stats_file, 'w') as f:
            f.write("=== 推荐统计信息 ===\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print(f"  📊 统计信息已保存: {stats_file}")

        print(f"✅ 推荐导出完成!")
        return submission_file

    def analyze_recommendations(self, recommendations):
        """分析推荐结果"""
        print("\n🔍 推荐结果分析...")

        rec_df = pd.DataFrame(recommendations)

        # 概率分布分析
        print(f"📊 概率分布:")
        print(f"  最高概率: {rec_df['probability'].max():.4f}")
        print(f"  最低概率: {rec_df['probability'].min():.4f}")
        print(f"  平均概率: {rec_df['probability'].mean():.4f}")
        print(f"  概率标准差: {rec_df['probability'].std():.4f}")

        # 用户推荐数分析
        user_rec_counts = rec_df.groupby('user_id').size()
        print(f"\n👥 用户推荐数分析:")
        print(f"  平均每用户推荐数: {user_rec_counts.mean():.1f}")
        print(f"  最多推荐数: {user_rec_counts.max()}")
        print(f"  最少推荐数: {user_rec_counts.min()}")

        # 商品推荐次数分析
        item_rec_counts = rec_df.groupby('item_id').size()
        print(f"\n📦 商品推荐次数分析:")
        print(f"  被推荐商品数: {len(item_rec_counts):,}")
        print(f"  平均被推荐次数: {item_rec_counts.mean():.1f}")
        print(f"  最多被推荐次数: {item_rec_counts.max()}")

        # Top推荐商品
        top_items = item_rec_counts.nlargest(10)
        print(f"\n🔥 最热门推荐商品:")
        for item_id, count in top_items.items():
            print(f"  商品 {item_id}: 被推荐 {count} 次")


def main():
    """主函数"""
    print("=== 推荐生成器 ===")
    print("🎯 目标：生成最终推荐列表")

    generator = RecommendationGenerator()

    # 1. 加载模型
    if not generator.load_model():
        print("❌ 模型加载失败")
        return

    # 2. 加载特征
    if not generator.load_features():
        print("❌ 特征加载失败")
        return

    # 3. 加载历史交互
    generator.load_historical_interactions()

    # 4. 生成推荐
    recommendations = generator.generate_all_recommendations(top_k=50)

    # 5. 分析推荐结果
    generator.analyze_recommendations(recommendations)

    # 6. 导出推荐
    submission_file = generator.export_recommendations(recommendations)

    print(f"\n🎉 推荐生成完成!")
    print(f"📁 提交文件: {submission_file}")
    print(f"🏆 可用于比赛提交!")


if __name__ == "__main__":
    main()