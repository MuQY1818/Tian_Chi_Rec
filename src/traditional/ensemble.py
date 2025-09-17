import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import time


class EnsembleRecommender:
    """多算法融合推荐器"""

    def __init__(self, weights: Dict[str, float] = None, fusion_method: str = "weighted_score"):
        """
        Args:
            weights: 各算法的权重字典，如 {'itemcf': 0.4, 'popularity': 0.3, 'als': 0.3}
            fusion_method: 融合方法 ('weighted_score', 'rank_fusion', 'vote')
        """
        self.weights = weights or {}
        self.fusion_method = fusion_method
        self.models = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """添加推荐模型"""
        self.models[name] = model
        if name not in self.weights:
            self.weights[name] = weight
        print(f"添加模型: {name} (权重: {self.weights[name]})")

    def weighted_score_fusion(self, all_recommendations: Dict[str, Dict[int, List[Tuple[int, float]]]],
                            top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """加权分数融合"""
        user_scores = defaultdict(lambda: defaultdict(float))

        # 收集所有用户ID
        all_users = set()
        for model_recs in all_recommendations.values():
            all_users.update(model_recs.keys())

        # 为每个用户融合推荐
        for user_id in all_users:
            for model_name, model_recs in all_recommendations.items():
                if user_id in model_recs:
                    weight = self.weights.get(model_name, 1.0)
                    for item_id, score in model_recs[user_id]:
                        user_scores[user_id][item_id] += weight * score

        # 生成最终推荐
        final_recommendations = {}
        for user_id, item_scores in user_scores.items():
            # 按分数排序
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            final_recommendations[user_id] = sorted_items[:top_n]

        return final_recommendations

    def rank_fusion(self, all_recommendations: Dict[str, Dict[int, List[Tuple[int, float]]]],
                   top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """基于排名的融合（类似RRF）"""
        user_rank_scores = defaultdict(lambda: defaultdict(float))

        # 收集所有用户ID
        all_users = set()
        for model_recs in all_recommendations.values():
            all_users.update(model_recs.keys())

        # 为每个用户计算排名分数
        for user_id in all_users:
            for model_name, model_recs in all_recommendations.items():
                if user_id in model_recs:
                    weight = self.weights.get(model_name, 1.0)
                    for rank, (item_id, _) in enumerate(model_recs[user_id], 1):
                        # RRF分数: 1/(k+rank)，这里k=60是常用值
                        rrf_score = 1.0 / (60 + rank)
                        user_rank_scores[user_id][item_id] += weight * rrf_score

        # 生成最终推荐
        final_recommendations = {}
        for user_id, item_scores in user_rank_scores.items():
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            final_recommendations[user_id] = sorted_items[:top_n]

        return final_recommendations

    def vote_fusion(self, all_recommendations: Dict[str, Dict[int, List[Tuple[int, float]]]],
                   top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """投票融合"""
        user_votes = defaultdict(lambda: defaultdict(float))

        # 收集所有用户ID
        all_users = set()
        for model_recs in all_recommendations.values():
            all_users.update(model_recs.keys())

        # 为每个用户收集投票
        for user_id in all_users:
            for model_name, model_recs in all_recommendations.items():
                if user_id in model_recs:
                    weight = self.weights.get(model_name, 1.0)
                    for item_id, _ in model_recs[user_id]:
                        user_votes[user_id][item_id] += weight

        # 生成最终推荐
        final_recommendations = {}
        for user_id, item_votes in user_votes.items():
            sorted_items = sorted(item_votes.items(), key=lambda x: x[1], reverse=True)
            final_recommendations[user_id] = sorted_items[:top_n]

        return final_recommendations

    def recommend_for_users(self, user_items_dict: Dict[int, List[int]],
                           excluded_items_dict: Dict[int, List[int]] = None,
                           top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """生成融合推荐"""
        print(f"生成融合推荐 (方法: {self.fusion_method})")
        start_time = time.time()

        # 收集所有模型的推荐结果
        all_recommendations = {}

        for model_name, model in self.models.items():
            print(f"获取 {model_name} 的推荐...")
            try:
                if hasattr(model, 'recommend_for_users'):
                    model_recs = model.recommend_for_users(
                        user_items_dict, excluded_items_dict, top_n * 2  # 获取更多候选
                    )
                    all_recommendations[model_name] = model_recs
                else:
                    print(f"模型 {model_name} 不支持批量推荐")
            except Exception as e:
                print(f"模型 {model_name} 推荐失败: {e}")

        # 选择融合方法
        if self.fusion_method == "weighted_score":
            final_recs = self.weighted_score_fusion(all_recommendations, top_n)
        elif self.fusion_method == "rank_fusion":
            final_recs = self.rank_fusion(all_recommendations, top_n)
        elif self.fusion_method == "vote":
            final_recs = self.vote_fusion(all_recommendations, top_n)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

        fusion_time = time.time() - start_time
        print(f"融合推荐完成，耗时: {fusion_time:.2f}s")

        return final_recs

    def diversity_rerank(self, recommendations: Dict[int, List[Tuple[int, float]]],
                        item_features: Dict[int, str] = None,
                        diversity_weight: float = 0.1) -> Dict[int, List[Tuple[int, float]]]:
        """多样性重排"""
        if item_features is None:
            return recommendations

        print("进行多样性重排...")
        reranked_recs = {}

        for user_id, user_recs in recommendations.items():
            if len(user_recs) <= 1:
                reranked_recs[user_id] = user_recs
                continue

            # 多样性重排算法
            final_list = []
            remaining_items = user_recs.copy()
            selected_features = set()

            while remaining_items and len(final_list) < len(user_recs):
                best_item = None
                best_score = -float('inf')

                for item_id, score in remaining_items:
                    # 原始分数
                    total_score = score

                    # 多样性惩罚
                    if item_id in item_features:
                        feature = item_features[item_id]
                        if feature in selected_features:
                            total_score -= diversity_weight * score

                    if total_score > best_score:
                        best_score = total_score
                        best_item = (item_id, score)

                if best_item:
                    final_list.append(best_item)
                    remaining_items.remove(best_item)

                    # 更新已选择的特征
                    if best_item[0] in item_features:
                        selected_features.add(item_features[best_item[0]])

            reranked_recs[user_id] = final_list

        return reranked_recs


class ColdStartHandler:
    """冷启动处理器"""

    def __init__(self, fallback_strategy: str = "popularity"):
        """
        Args:
            fallback_strategy: 冷启动回退策略 ('popularity', 'random', 'category_popular')
        """
        self.fallback_strategy = fallback_strategy
        self.popular_items = []
        self.category_popular = {}

    def set_popular_items(self, popular_items: List[Tuple[int, float]]):
        """设置热门商品列表"""
        self.popular_items = popular_items

    def set_category_popular(self, category_popular: Dict[str, List[Tuple[int, float]]]):
        """设置各类别热门商品"""
        self.category_popular = category_popular

    def handle_cold_users(self, user_items_dict: Dict[int, List[int]],
                         recommendations: Dict[int, List[Tuple[int, float]]],
                         top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """处理冷启动用户"""
        handled_recs = recommendations.copy()

        for user_id, user_items in user_items_dict.items():
            # 如果用户没有推荐或推荐数量不足
            current_recs = handled_recs.get(user_id, [])

            if len(current_recs) < top_n:
                needed = top_n - len(current_recs)
                existing_items = set([item for item, _ in current_recs] + user_items)

                if self.fallback_strategy == "popularity":
                    # 使用热门商品填充
                    fallback_items = [(item, score) for item, score in self.popular_items
                                    if item not in existing_items][:needed]

                elif self.fallback_strategy == "random":
                    # 随机推荐（简化实现）
                    all_items = set(range(1000))  # 假设有1000个商品
                    available_items = list(all_items - existing_items)
                    if available_items:
                        random_items = np.random.choice(available_items,
                                                      min(needed, len(available_items)),
                                                      replace=False)
                        fallback_items = [(int(item), 1.0) for item in random_items]
                    else:
                        fallback_items = []

                else:
                    fallback_items = []

                # 合并推荐
                handled_recs[user_id] = current_recs + fallback_items

        return handled_recs