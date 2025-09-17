import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
import time

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False
    print("Warning: implicit库未安装，将使用简化的SVD实现")


class ALSMatrixFactorization:
    """使用ALS算法的矩阵分解推荐"""

    def __init__(self, factors: int = 64, regularization: float = 0.1,
                 iterations: int = 20, alpha: float = 40.0):
        """
        Args:
            factors: 隐因子数量
            regularization: 正则化参数
            iterations: 迭代次数
            alpha: 置信度参数
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model = None
        self.user_item_matrix = None

    def fit(self, user_item_matrix: csr_matrix):
        """训练ALS模型"""
        print(f"训练ALS模型 (factors={self.factors}, iterations={self.iterations})")
        start_time = time.time()

        self.user_item_matrix = user_item_matrix

        if HAS_IMPLICIT:
            # 使用implicit库
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                alpha=self.alpha,
                random_state=42
            )

            # implicit库期望item-user矩阵
            item_user_matrix = user_item_matrix.T.tocsr()

            # 转换为置信度矩阵
            confidence_matrix = (item_user_matrix * self.alpha).astype(np.float32)

            # 训练模型
            self.model.fit(confidence_matrix)

        else:
            # 简化实现：使用SVD
            print("使用简化的SVD实现...")
            self.model = self._simple_svd(user_item_matrix)

        training_time = time.time() - start_time
        print(f"ALS训练完成，耗时: {training_time:.2f}s")

    def _simple_svd(self, matrix: csr_matrix):
        """简化的SVD实现（当implicit库不可用时）"""
        # 转换为密集矩阵（仅适用于小数据）
        if matrix.shape[0] * matrix.shape[1] > 1000000:
            print("警告: 矩阵太大，使用随机采样")
            # 随机采样
            sample_users = np.random.choice(matrix.shape[0], min(1000, matrix.shape[0]), replace=False)
            matrix = matrix[sample_users]

        dense_matrix = matrix.toarray()

        # SVD分解
        U, s, Vt = np.linalg.svd(dense_matrix, full_matrices=False)

        # 保留前k个因子
        k = min(self.factors, len(s))
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]

        return {
            'U': U_k,
            's': s_k,
            'Vt': Vt_k,
            'user_factors': U_k * np.sqrt(s_k),
            'item_factors': (Vt_k * np.sqrt(s_k[:, np.newaxis])).T
        }

    def recommend_for_users(self, user_items_dict: Dict[int, List[int]],
                           excluded_items_dict: Dict[int, List[int]] = None,
                           top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """为用户生成推荐"""
        print(f"为 {len(user_items_dict)} 个用户生成ALS推荐...")

        if excluded_items_dict is None:
            excluded_items_dict = {}

        recommendations = {}

        if HAS_IMPLICIT:
            for user_id, user_items in user_items_dict.items():
                excluded = excluded_items_dict.get(user_id, [])

                # 使用implicit库推荐
                try:
                    # 获取推荐
                    recs = self.model.recommend(
                        user_id,
                        self.user_item_matrix[user_id],
                        N=top_n + len(excluded),
                        filter_already_liked_items=True
                    )

                    # 过滤排除的商品
                    excluded_set = set(excluded)
                    filtered_recs = [(item_id, score) for item_id, score in recs
                                   if item_id not in excluded_set]

                    recommendations[user_id] = filtered_recs[:top_n]

                except Exception as e:
                    print(f"用户 {user_id} 推荐失败: {e}")
                    recommendations[user_id] = []

        else:
            # 使用简化实现
            for user_id, user_items in user_items_dict.items():
                if user_id < len(self.model['user_factors']):
                    excluded = set(user_items + excluded_items_dict.get(user_id, []))

                    # 计算用户对所有商品的分数
                    user_vector = self.model['user_factors'][user_id]
                    scores = np.dot(self.model['item_factors'], user_vector)

                    # 排除已交互的商品
                    for item_id in excluded:
                        if item_id < len(scores):
                            scores[item_id] = -np.inf

                    # 获取Top-N
                    top_items = np.argpartition(scores, -top_n)[-top_n:]
                    top_items = top_items[np.argsort(scores[top_items])[::-1]]

                    recommendations[user_id] = [(int(item_id), float(scores[item_id]))
                                              for item_id in top_items if scores[item_id] > -np.inf]
                else:
                    recommendations[user_id] = []

        return recommendations

    def get_item_embeddings(self) -> np.ndarray:
        """获取商品嵌入向量"""
        if HAS_IMPLICIT:
            return self.model.item_factors
        else:
            return self.model['item_factors']

    def get_user_embeddings(self) -> np.ndarray:
        """获取用户嵌入向量"""
        if HAS_IMPLICIT:
            return self.model.user_factors
        else:
            return self.model['user_factors']

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取相似商品"""
        if HAS_IMPLICIT:
            try:
                similar = self.model.similar_items(item_id, N=top_k)
                return [(int(item), float(score)) for item, score in similar]
            except:
                return []
        else:
            # 使用商品嵌入向量计算相似度
            if item_id >= len(self.model['item_factors']):
                return []

            item_vector = self.model['item_factors'][item_id]
            similarities = np.dot(self.model['item_factors'], item_vector)

            # 排除自己
            similarities[item_id] = -np.inf

            # 获取Top-K
            top_items = np.argpartition(similarities, -top_k)[-top_k:]
            top_items = top_items[np.argsort(similarities[top_items])[::-1]]

            return [(int(item_id), float(similarities[item_id]))
                   for item_id in top_items if similarities[item_id] > -np.inf]


class SimpleNMF:
    """简化的非负矩阵分解实现"""

    def __init__(self, n_components: int = 64, max_iter: int = 200):
        self.n_components = n_components
        self.max_iter = max_iter
        self.W = None  # 用户因子矩阵
        self.H = None  # 商品因子矩阵

    def fit(self, user_item_matrix: csr_matrix):
        """训练NMF模型"""
        print(f"训练NMF模型 (components={self.n_components})")

        # 转换为密集矩阵（仅适用于小数据）
        if user_item_matrix.nnz > 100000:
            print("数据量过大，跳过NMF训练")
            return

        X = user_item_matrix.toarray()
        m, n = X.shape

        # 随机初始化
        self.W = np.random.rand(m, self.n_components)
        self.H = np.random.rand(self.n_components, n)

        # NMF迭代
        for iteration in range(self.max_iter):
            # 更新H
            self.H = self.H * (self.W.T @ X) / (self.W.T @ self.W @ self.H + 1e-8)

            # 更新W
            self.W = self.W * (X @ self.H.T) / (self.W @ self.H @ self.H.T + 1e-8)

            if iteration % 50 == 0:
                # 计算重构误差
                reconstruction = self.W @ self.H
                error = np.linalg.norm(X - reconstruction, 'fro')
                print(f"NMF Iteration {iteration}, Error: {error:.4f}")

        print("NMF训练完成")

    def predict(self, user_id: int, item_id: int) -> float:
        """预测用户对商品的评分"""
        if self.W is None or self.H is None:
            return 0.0

        return float(np.dot(self.W[user_id], self.H[:, item_id]))

    def recommend_for_user(self, user_id: int, excluded_items: List[int] = None, top_n: int = 5):
        """为用户推荐商品"""
        if self.W is None or self.H is None:
            return []

        if excluded_items is None:
            excluded_items = []

        # 计算用户对所有商品的预测评分
        scores = self.W[user_id] @ self.H

        # 排除已交互的商品
        for item_id in excluded_items:
            if item_id < len(scores):
                scores[item_id] = -np.inf

        # 获取Top-N
        top_items = np.argpartition(scores, -top_n)[-top_n:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]

        return [(int(item_id), float(scores[item_id]))
               for item_id in top_items if scores[item_id] > -np.inf]