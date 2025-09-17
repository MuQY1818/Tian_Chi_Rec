import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import time


class ItemCF:
    """物品协同过滤推荐算法"""

    def __init__(self, k: int = 20, similarity_metric: str = "cosine"):
        """
        Args:
            k: 相似商品的数量
            similarity_metric: 相似度计算方法 ('cosine', 'jaccard')
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.item_similarity_matrix = None
        self.user_item_matrix = None

    def fit(self, user_item_matrix: csr_matrix):
        """训练ItemCF模型"""
        print(f"训练ItemCF模型 (相似度: {self.similarity_metric}, Top-{self.k})")
        start_time = time.time()

        self.user_item_matrix = user_item_matrix

        # 转置获得item-user矩阵
        item_user_matrix = user_item_matrix.T

        # 计算商品相似度矩阵
        if self.similarity_metric == "cosine":
            # 使用余弦相似度
            self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        elif self.similarity_metric == "jaccard":
            # 使用Jaccard相似度
            self.item_similarity_matrix = self._jaccard_similarity(item_user_matrix)
        else:
            raise ValueError(f"不支持的相似度计算方法: {self.similarity_metric}")

        # 将自相似度设为0，避免推荐自己
        np.fill_diagonal(self.item_similarity_matrix, 0)

        # 只保留Top-K相似商品，减少内存占用
        self._keep_topk_similarities()

        training_time = time.time() - start_time
        print(f"ItemCF训练完成，耗时: {training_time:.2f}s")

    def _jaccard_similarity(self, item_user_matrix: csr_matrix) -> np.ndarray:
        """计算Jaccard相似度"""
        print("计算Jaccard相似度...")

        # 将矩阵二值化
        binary_matrix = (item_user_matrix > 0).astype(int)

        # 计算交集和并集
        intersection = binary_matrix.dot(binary_matrix.T).toarray()

        # 计算每个商品的用户数
        item_user_counts = np.array(binary_matrix.sum(axis=1)).flatten()

        # 计算并集
        union = item_user_counts[:, np.newaxis] + item_user_counts[np.newaxis, :] - intersection

        # 避免除零
        union[union == 0] = 1

        # 计算Jaccard相似度
        jaccard_sim = intersection / union

        return jaccard_sim

    def _keep_topk_similarities(self):
        """只保留每个商品的Top-K相似商品"""
        print(f"保留Top-{self.k}相似商品...")

        num_items = self.item_similarity_matrix.shape[0]

        for i in range(num_items):
            # 获取第i个商品的相似度
            similarities = self.item_similarity_matrix[i]

            # 找到Top-K相似商品的索引
            if len(similarities) > self.k:
                topk_indices = np.argpartition(similarities, -self.k)[-self.k:]

                # 将非Top-K的相似度设为0
                mask = np.ones(len(similarities), dtype=bool)
                mask[topk_indices] = False
                similarities[mask] = 0

    def predict_for_user(self, user_id: int, user_items: List[int],
                        excluded_items: List[int] = None, top_n: int = 5) -> List[Tuple[int, float]]:
        """为单个用户生成推荐"""
        if excluded_items is None:
            excluded_items = []

        # 获取用户历史交互商品的权重
        user_weights = self.user_item_matrix[user_id].toarray().flatten()

        # 计算推荐分数
        scores = np.zeros(self.item_similarity_matrix.shape[0])

        for item_id in user_items:
            if user_weights[item_id] > 0:
                # 累加相似商品的分数
                scores += self.item_similarity_matrix[item_id] * user_weights[item_id]

        # 排除已交互和指定排除的商品
        excluded_set = set(user_items + excluded_items)
        for item_id in excluded_set:
            scores[item_id] = -np.inf

        # 获取Top-N推荐
        top_items = np.argpartition(scores, -top_n)[-top_n:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]

        # 返回商品ID和分数
        recommendations = [(item_id, scores[item_id]) for item_id in top_items if scores[item_id] > 0]

        return recommendations

    def recommend_for_users(self, user_items_dict: Dict[int, List[int]],
                           excluded_items_dict: Dict[int, List[int]] = None,
                           top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """为多个用户生成推荐"""
        print(f"为 {len(user_items_dict)} 个用户生成ItemCF推荐...")

        if excluded_items_dict is None:
            excluded_items_dict = {}

        recommendations = {}

        for user_id, user_items in user_items_dict.items():
            excluded = excluded_items_dict.get(user_id, [])
            recommendations[user_id] = self.predict_for_user(
                user_id, user_items, excluded, top_n
            )

        return recommendations

    def get_similar_items(self, item_id: int, top_k: int = None) -> List[Tuple[int, float]]:
        """获取与指定商品最相似的商品"""
        if top_k is None:
            top_k = self.k

        if self.item_similarity_matrix is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        similarities = self.item_similarity_matrix[item_id]

        # 获取Top-K相似商品
        top_items = np.argpartition(similarities, -top_k)[-top_k:]
        top_items = top_items[np.argsort(similarities[top_items])[::-1]]

        return [(item_id, similarities[item_id]) for item_id in top_items if similarities[item_id] > 0]

    def save_model(self, filepath: str):
        """保存模型"""
        np.save(filepath, {
            'item_similarity_matrix': self.item_similarity_matrix,
            'k': self.k,
            'similarity_metric': self.similarity_metric
        })

    def load_model(self, filepath: str):
        """加载模型"""
        data = np.load(filepath, allow_pickle=True).item()
        self.item_similarity_matrix = data['item_similarity_matrix']
        self.k = data['k']
        self.similarity_metric = data['similarity_metric']