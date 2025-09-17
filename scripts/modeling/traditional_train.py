#!/usr/bin/env python3

import sys
import os
import time
from typing import Dict, List

# 添加路径
sys.path.append('src')

from traditional.data_processor import TraditionalDataProcessor
from traditional.itemcf import ItemCF
from traditional.popularity import TimeBasedPopularity, TrendingItems
from traditional.matrix_factorization import ALSMatrixFactorization
from traditional.ensemble import EnsembleRecommender, ColdStartHandler


def evaluate_recommendations(val_user_items: Dict[int, List[int]],
                           recommendations: Dict[int, List[int]],
                           k: int = 5) -> Dict[str, float]:
    """评估推荐结果"""
    hits = 0
    total_users = 0
    total_recall = 0.0

    for user_id, true_items in val_user_items.items():
        if user_id in recommendations: 
            pred_items = recommendations[user_id][:k]
            pred_set = set(pred_items)
            true_set = set(true_items)

            hit_items = len(pred_set & true_set)
            hits += min(hit_items, 1)  # HR@K: 至少命中1个
            total_recall += hit_items / len(true_set) if true_set else 0

            total_users += 1

    hr_k = hits / total_users if total_users > 0 else 0
    recall_k = total_recall / total_users if total_users > 0 else 0

    return {
        f'HR@{k}': hr_k,
        f'Recall@{k}': recall_k,
        'Coverage': len(set().union(*recommendations.values())) if recommendations else 0
    }


def main():
    print("=== 传统推荐算法训练和测试 ===")

    # 1. 数据加载和预处理
    print("\n1. 数据预处理...")
    processor = TraditionalDataProcessor(data_dir="dataset")

    # 检查是否使用全量数据
    import sys
    use_full_data = "--full" in sys.argv
    sample_frac = 0.1 if "--sample" in sys.argv else 1.0  # 默认10%采样或全量

    if use_full_data:
        print("🚀 使用全量数据集训练")
        df = processor.load_data(sample_frac=sample_frac, use_full_data=True)
    else:
        print("🧪 使用小数据集测试")
        df = processor.load_data(sample_frac=1.0)  # 小数据集

    # 时间划分
    if use_full_data:
        # 全量数据使用正确的时间划分
        train_df, val_df = processor.split_by_time(df, train_end="2014-12-17", val_date="2014-12-18")
    else:
        # 小数据集调整日期
        train_df, val_df = processor.split_by_time(df, train_end="2014-12-16", val_date="2014-12-17")

    # 构建交互矩阵
    train_matrix = processor.build_interaction_matrix(train_df)
    time_weighted_matrix = processor.get_time_weighted_matrix(train_df)

    print(f"训练矩阵维度: {train_matrix.shape}")
    print(f"交互数量: {train_matrix.nnz}")

    # 获取训练和验证用户-商品映射
    train_user_items = processor.get_user_items(train_df)
    val_user_items = processor.get_user_items(val_df, behavior_types=[4])  # 只考虑购买行为

    print(f"训练用户数: {len(train_user_items)}")
    print(f"验证用户数: {len(val_user_items)}")

    # 2. 训练各个算法
    print("\n2. 训练推荐算法...")

    # ItemCF
    print("\n2.1 训练ItemCF...")
    itemcf = ItemCF(k=20, similarity_metric="cosine")
    itemcf.fit(train_matrix)

    # 流行度模型
    print("\n2.2 训练流行度模型...")
    current_date = "2014-12-18" if use_full_data else "2014-12-17"
    popularity = TimeBasedPopularity(time_window_days=7)
    popularity.fit(train_df, current_date=current_date)

    # 趋势商品
    print("\n2.3 训练趋势模型...")
    trending = TrendingItems(short_window=3, long_window=14)
    trending.fit(train_df, current_date=current_date)

    # ALS矩阵分解
    print("\n2.4 训练ALS...")
    als = ALSMatrixFactorization(factors=32, iterations=10)
    try:
        als.fit(time_weighted_matrix)
        als_available = True
    except Exception as e:
        print(f"ALS训练失败: {e}")
        als_available = False

    # 3. 单独评估各算法
    print("\n3. 单独评估各算法...")

    # 过滤验证用户（只保留训练中出现的用户）
    common_users = set(train_user_items.keys()) & set(val_user_items.keys())
    filtered_val_user_items = {u: val_user_items[u] for u in common_users}

    print(f"有效验证用户数: {len(filtered_val_user_items)}")

    results = {}

    # 评估ItemCF
    print("\n3.1 评估ItemCF...")
    try:
        itemcf_recs = itemcf.recommend_for_users(
            train_user_items, top_n=5
        )
        # 转换为简单格式
        itemcf_simple = {u: [item for item, score in recs]
                        for u, recs in itemcf_recs.items() if u in filtered_val_user_items}
        results['ItemCF'] = evaluate_recommendations(filtered_val_user_items, itemcf_simple)
    except Exception as e:
        print(f"ItemCF评估失败: {e}")
        results['ItemCF'] = {'HR@5': 0, 'Recall@5': 0, 'Coverage': 0}

    # 评估流行度
    print("\n3.2 评估流行度模型...")
    try:
        pop_recs = popularity.recommend_for_users(
            train_user_items, top_n=5
        )
        pop_simple = {u: [item for item, score in recs]
                     for u, recs in pop_recs.items() if u in filtered_val_user_items}
        results['Popularity'] = evaluate_recommendations(filtered_val_user_items, pop_simple)
    except Exception as e:
        print(f"流行度评估失败: {e}")
        results['Popularity'] = {'HR@5': 0, 'Recall@5': 0, 'Coverage': 0}

    # 评估ALS
    if als_available:
        print("\n3.3 评估ALS...")
        try:
            als_recs = als.recommend_for_users(
                train_user_items, top_n=5
            )
            als_simple = {u: [item for item, score in recs]
                         for u, recs in als_recs.items() if u in filtered_val_user_items}
            results['ALS'] = evaluate_recommendations(filtered_val_user_items, als_simple)
        except Exception as e:
            print(f"ALS评估失败: {e}")
            results['ALS'] = {'HR@5': 0, 'Recall@5': 0, 'Coverage': 0}
    else:
        results['ALS'] = {'HR@5': 0, 'Recall@5': 0, 'Coverage': 0}

    # 4. 融合推荐
    print("\n4. 融合推荐...")
    ensemble = EnsembleRecommender(
        weights={'itemcf': 0.5, 'popularity': 0.3, 'als': 0.2},
        fusion_method="weighted_score"
    )

    ensemble.add_model('itemcf', itemcf, 0.5)
    ensemble.add_model('popularity', popularity, 0.3)
    if als_available:
        ensemble.add_model('als', als, 0.2)

    try:
        ensemble_recs = ensemble.recommend_for_users(
            train_user_items, top_n=5
        )
        ensemble_simple = {u: [item for item, score in recs]
                          for u, recs in ensemble_recs.items() if u in filtered_val_user_items}
        results['Ensemble'] = evaluate_recommendations(filtered_val_user_items, ensemble_simple)
    except Exception as e:
        print(f"融合评估失败: {e}")
        results['Ensemble'] = {'HR@5': 0, 'Recall@5': 0, 'Coverage': 0}

    # 5. 显示结果
    print("\n5. 评估结果:")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'HR@5':<8} {'Recall@5':<10} {'Coverage':<10}")
    print("-" * 60)

    for alg_name, metrics in results.items():
        print(f"{alg_name:<12} {metrics['HR@5']:<8.4f} {metrics['Recall@5']:<10.4f} {metrics['Coverage']:<10}")

    # 6. 生成最终提交文件
    print("\n6. 生成提交文件...")

    # 选择最佳算法（基于HR@5）
    best_algorithm = max(results.keys(), key=lambda x: results[x]['HR@5'])
    print(f"最佳算法: {best_algorithm} (HR@5: {results[best_algorithm]['HR@5']:.4f})")

    # 为所有用户生成推荐
    if best_algorithm == 'ItemCF':
        final_recs = itemcf.recommend_for_users(train_user_items, top_n=5)
    elif best_algorithm == 'Popularity':
        final_recs = popularity.recommend_for_users(train_user_items, top_n=5)
    elif best_algorithm == 'ALS' and als_available:
        final_recs = als.recommend_for_users(train_user_items, top_n=5)
    else:
        final_recs = ensemble.recommend_for_users(train_user_items, top_n=5)

    # 生成提交文件
    submission_file = "full_traditional_submission.txt" if use_full_data else "traditional_submission.txt"

    with open(submission_file, "w", encoding="utf-8") as f:
        for user_id, recs in final_recs.items():
            original_user_id = processor.id2user[user_id]
            for item_id, _ in recs:
                original_item_id = processor.id2item[item_id]
                f.write(f"{original_user_id}\t{original_item_id}\n")

    print(f"✅ 提交文件已生成: {submission_file}")
    print(f"📊 推荐用户数: {len(final_recs)}")
    print(f"📝 推荐商品对数: {sum(len(recs) for recs in final_recs.values())}")
    print(f"🎯 使用算法: {best_algorithm}")

    if use_full_data:
        print("\n🎉 全量数据训练完成！")
        print(f"🎯 可以提交 {submission_file} 到比赛平台")


if __name__ == "__main__":
    main()