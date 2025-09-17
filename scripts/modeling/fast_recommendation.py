#!/usr/bin/env python3
"""
快速推荐生成器
基于现有39维用户特征的快速推荐
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class FastRecommendationGenerator:
    """快速推荐生成器"""

    def __init__(self, feature_dir="/mnt/data/tianchi_features"):
        self.feature_dir = feature_dir

        self.model = None
        self.config = None
        self.user_features = None
        self.item_features = None

        print(f"📁 特征目录: {feature_dir}")

    def load_model_and_features(self):
        """加载模型和特征"""
        print("📂 加载模型和特征...")

        # 加载快速模型
        model_file = os.path.join(self.feature_dir, "fast_lightgbm_model.pkl")
        config_file = os.path.join(self.feature_dir, "fast_model_config.pkl")

        if not os.path.exists(model_file):
            print(f"❌ 模型文件不存在: {model_file}")
            return False

        self.model = joblib.load(model_file)
        self.config = joblib.load(config_file)
        print(f"  🤖 模型加载成功 (AUC: {self.config['metrics']['auc']:.4f})")

        # 加载用户特征
        user_file = os.path.join(self.feature_dir, "user_features_cpp.csv")
        self.user_features = pd.read_csv(user_file)
        self.user_feature_dict = self.user_features.set_index('user_id').to_dict('index')
        print(f"  👥 用户特征: {len(self.user_features):,} 用户")

        # 加载商品特征
        item_file = os.path.join(self.feature_dir, "simple_item_features.csv")
        if not os.path.exists(item_file):
            print(f"❌ 商品特征文件不存在，请先运行商品特征提取")
            return False

        self.item_features = pd.read_csv(item_file)
        self.item_feature_dict = self.item_features.set_index('item_id').to_dict('index')
        print(f"  📦 商品特征: {len(self.item_features):,} 商品")

        return True

    def generate_candidates_for_user(self, user_id, top_k=50):
        """为用户生成候选商品（快速策略）"""
        user_info = self.user_feature_dict.get(user_id)
        if not user_info:
            return []

        candidates = []

        # 策略1: 用户偏好类别的热门商品
        user_top_category = user_info.get('top_category', -1)
        if user_top_category != -1:
            category_items = self.item_features[
                self.item_features['item_category'] == user_top_category
            ].nlargest(20, 'popularity')['item_id'].tolist()
            candidates.extend(category_items)

        # 策略2: 全局热门商品
        popular_items = self.item_features.nlargest(30, 'popularity')['item_id'].tolist()
        candidates.extend(popular_items)

        # 策略3: 高购买率商品
        high_conversion_items = self.item_features[
            self.item_features['purchase_count'] >= 5
        ].nlargest(20, 'purchase_rate')['item_id'].tolist()
        candidates.extend(high_conversion_items)

        # 去重并限制数量
        candidates = list(set(candidates))[:top_k]

        return candidates

    def predict_for_user(self, user_id, candidate_items):
        """为用户预测候选商品的购买概率"""
        user_info = self.user_feature_dict.get(user_id)
        if not user_info:
            return []

        # 构建特征矩阵
        features_list = []
        valid_items = []

        for item_id in candidate_items:
            item_info = self.item_feature_dict.get(item_id)
            if not item_info:
                continue

            # 构建特征向量（与训练时保持一致）
            sample = {}

            # 用户特征
            for key, value in user_info.items():
                sample[f'user_{key}'] = value

            # 商品特征
            for key, value in item_info.items():
                if key != 'item_id':
                    sample[f'item_{key}'] = value

            # 交互特征（简化版本 - 新用户默认无交互）
            sample['has_interaction'] = 0  # 预测场景下假设无历史交互
            sample['has_purchased'] = 0

            # 类别匹配
            item_category = item_info.get('item_category', -1)
            user_top_category = user_info.get('top_category', -1)
            sample['category_match'] = 1 if item_category == user_top_category else 0

            # 活跃度匹配
            user_activity = user_info.get('total_actions', 0)
            item_popularity = item_info.get('popularity', 0)
            sample['activity_popularity_ratio'] = (user_activity + 1) / (item_popularity + 1)

            features_list.append(sample)
            valid_items.append(item_id)

        if not features_list:
            return []

        # 转换为DataFrame并预测
        features_df = pd.DataFrame(features_list)

        # 确保特征顺序与训练时一致
        feature_names = self.config['feature_names']
        for feature in feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0

        features_df = features_df[feature_names]

        # 预测概率
        probabilities = self.model.predict(features_df, num_iteration=self.model.best_iteration)

        # 组合结果
        results = []
        for item_id, prob in zip(valid_items, probabilities):
            results.append({
                'item_id': item_id,
                'probability': prob
            })

        # 按概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results

    def generate_all_recommendations(self, max_users=None, rec_per_user=30):
        """为所有用户生成推荐"""
        print(f"\n🎯 生成推荐 (每用户{rec_per_user}个)...")

        user_ids = list(self.user_feature_dict.keys())
        if max_users:
            user_ids = user_ids[:max_users]
            print(f"  👥 处理用户数: {len(user_ids):,} (限制)")
        else:
            print(f"  👥 处理用户数: {len(user_ids):,}")

        all_recommendations = []

        for user_id in tqdm(user_ids, desc="生成推荐"):
            try:
                # 生成候选
                candidates = self.generate_candidates_for_user(user_id, top_k=60)

                # 预测概率
                predictions = self.predict_for_user(user_id, candidates)

                # 取Top-K
                for pred in predictions[:rec_per_user]:
                    all_recommendations.append({
                        'user_id': user_id,
                        'item_id': pred['item_id'],
                        'probability': pred['probability']
                    })

            except Exception as e:
                print(f"  ⚠️  用户 {user_id} 推荐失败: {e}")
                continue

        print(f"✅ 推荐生成完成:")
        print(f"  📊 总推荐数: {len(all_recommendations):,}")

        return all_recommendations

    def export_recommendations(self, recommendations, filename_suffix="fast"):
        """导出推荐结果"""
        print("\n💾 导出推荐结果...")

        # 详细推荐
        detailed_file = os.path.join(self.feature_dir, f"recommendations_{filename_suffix}.csv")
        pd.DataFrame(recommendations).to_csv(detailed_file, index=False)
        print(f"  📊 详细推荐: {detailed_file}")

        # 提交格式（tab分隔，无表头）
        submission_df = pd.DataFrame(recommendations)[['user_id', 'item_id']]
        submission_file = os.path.join(self.feature_dir, f"submission_{filename_suffix}.txt")
        submission_df.to_csv(submission_file, index=False, header=False, sep='\t')
        print(f"  📝 提交文件: {submission_file}")

        # 统计分析
        rec_df = pd.DataFrame(recommendations)
        stats = {
            'total_recommendations': len(recommendations),
            'unique_users': rec_df['user_id'].nunique(),
            'unique_items': rec_df['item_id'].nunique(),
            'avg_probability': rec_df['probability'].mean(),
            'min_probability': rec_df['probability'].min(),
            'max_probability': rec_df['probability'].max()
        }

        print(f"\n📊 推荐统计:")
        for key, value in stats.items():
            if 'probability' in key:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")

        return submission_file

    def quick_recommend_sample(self, sample_users=10):
        """快速推荐示例（用于测试）"""
        print(f"\n🔍 快速推荐示例 ({sample_users}个用户)...")

        user_ids = list(self.user_feature_dict.keys())[:sample_users]

        for user_id in user_ids:
            candidates = self.generate_candidates_for_user(user_id, top_k=10)
            predictions = self.predict_for_user(user_id, candidates)

            print(f"\n👤 用户 {user_id}:")
            user_info = self.user_feature_dict[user_id]
            print(f"  活跃度: {user_info.get('total_actions', 0)}")
            print(f"  偏好类别: {user_info.get('top_category', -1)}")
            print(f"  Top-3推荐:")

            for i, pred in enumerate(predictions[:3], 1):
                item_info = self.item_feature_dict.get(pred['item_id'], {})
                print(f"    {i}. 商品{pred['item_id']} (概率:{pred['probability']:.3f}, "
                      f"类别:{item_info.get('item_category', '?')}, "
                      f"流行度:{item_info.get('popularity', 0)})")


def main():
    """主函数"""
    print("🎯 === 快速推荐生成器 === 🎯")
    print("🎯 目标：基于现有模型快速生成推荐")
    print("⚡ 预计耗时：3-5分钟")
    print("━" * 50)

    generator = FastRecommendationGenerator()

    # 1. 加载模型和特征
    if not generator.load_model_and_features():
        print("❌ 加载失败")
        return

    # 2. 快速示例（可选）
    generator.quick_recommend_sample(sample_users=5)

    # 3. 生成全量推荐
    recommendations = generator.generate_all_recommendations(
        max_users=None,  # None = 全部用户
        rec_per_user=30
    )

    # 4. 导出结果
    submission_file = generator.export_recommendations(recommendations)

    print(f"\n🎉 快速推荐完成!")
    print(f"📁 提交文件: {submission_file}")
    print(f"⚡ 速度: 比完整版本快5-10倍")
    print(f"🏆 可用于比赛提交!")


if __name__ == "__main__":
    main()