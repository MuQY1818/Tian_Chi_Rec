import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class TimeBasedPopularity:
    """基于时间的流行度推荐算法"""

    def __init__(self, time_window_days: int = 7, decay_factor: float = 0.1):
        """
        Args:
            time_window_days: 时间窗口天数
            decay_factor: 时间衰减因子
        """
        self.time_window_days = time_window_days
        self.decay_factor = decay_factor
        self.item_scores = {}
        self.category_scores = {}
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}

    def fit(self, df: pd.DataFrame, current_date: str = "2014-12-18"):
        """训练流行度模型"""
        print(f"训练时间流行度模型 (时间窗口: {self.time_window_days}天)")

        current_dt = pd.to_datetime(current_date)
        start_date = current_dt - timedelta(days=self.time_window_days)

        # 筛选时间窗口内的数据
        window_data = df[df["datetime"] >= start_date].copy()

        print(f"时间窗口数据: {len(window_data)} 行")

        # 计算时间衰减权重
        window_data["time_diff"] = (current_dt - window_data["datetime"]).dt.total_seconds() / (24 * 3600)
        window_data["time_weight"] = np.exp(-self.decay_factor * window_data["time_diff"])

        # 结合行为权重和时间权重
        window_data["final_weight"] = (window_data["behavior_type"].map(self.behavior_weights) *
                                     window_data["time_weight"])

        # 计算商品流行度分数
        self.item_scores = window_data.groupby("iid")["final_weight"].sum().to_dict()

        # 计算类别流行度分数
        if "item_category" in window_data.columns:
            self.category_scores = window_data.groupby("item_category")["final_weight"].sum().to_dict()

        print(f"计算了 {len(self.item_scores)} 个商品的流行度")

    def get_popular_items(self, top_n: int = 100, excluded_items: List[int] = None) -> List[Tuple[int, float]]:
        """获取最流行的商品"""
        if excluded_items is None:
            excluded_items = set()
        else:
            excluded_items = set(excluded_items)

        # 过滤排除的商品
        filtered_scores = {item: score for item, score in self.item_scores.items()
                          if item not in excluded_items}

        # 排序并返回Top-N
        sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

    def recommend_for_users(self, user_items_dict: Dict[int, List[int]],
                           excluded_items_dict: Dict[int, List[int]] = None,
                           top_n: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """为用户推荐流行商品"""
        print(f"为 {len(user_items_dict)} 个用户生成流行度推荐...")

        if excluded_items_dict is None:
            excluded_items_dict = {}

        recommendations = {}

        # 获取全局热门商品
        global_popular = self.get_popular_items(top_n * 3)  # 获取更多候选

        for user_id, user_items in user_items_dict.items():
            excluded = set(user_items + excluded_items_dict.get(user_id, []))

            # 过滤用户已交互的商品
            user_recommendations = [(item, score) for item, score in global_popular
                                  if item not in excluded][:top_n]

            recommendations[user_id] = user_recommendations

        return recommendations

    def get_category_popular_items(self, category: str, top_n: int = 20,
                                 excluded_items: List[int] = None) -> List[Tuple[int, float]]:
        """获取特定类别的流行商品"""
        if excluded_items is None:
            excluded_items = set()
        else:
            excluded_items = set(excluded_items)

        # 这里简化处理，实际中需要维护category到item的映射
        # 目前返回全局流行商品作为近似
        return self.get_popular_items(top_n, excluded_items)


class TrendingItems:
    """趋势商品推荐"""

    def __init__(self, short_window: int = 3, long_window: int = 14):
        """
        Args:
            short_window: 短期时间窗口（天）
            long_window: 长期时间窗口（天）
        """
        self.short_window = short_window
        self.long_window = long_window
        self.trending_scores = {}

    def fit(self, df: pd.DataFrame, current_date: str = "2014-12-18"):
        """计算趋势分数"""
        print(f"计算趋势商品 (短期: {self.short_window}天, 长期: {self.long_window}天)")

        current_dt = pd.to_datetime(current_date)

        # 短期数据
        short_start = current_dt - timedelta(days=self.short_window)
        short_data = df[df["datetime"] >= short_start]
        short_counts = short_data.groupby("iid").size()

        # 长期数据
        long_start = current_dt - timedelta(days=self.long_window)
        long_data = df[df["datetime"] >= long_start]
        long_counts = long_data.groupby("iid").size()

        # 计算趋势分数 (短期频率 / 长期平均频率)
        for item_id in short_counts.index:
            short_freq = short_counts[item_id]
            long_freq = long_counts.get(item_id, 1)

            # 标准化到每天
            short_daily = short_freq / self.short_window
            long_daily = long_freq / self.long_window

            # 趋势分数
            trend_score = short_daily / long_daily if long_daily > 0 else 0
            self.trending_scores[item_id] = trend_score * short_freq  # 乘以频率作为权重

        print(f"计算了 {len(self.trending_scores)} 个商品的趋势分数")

    def get_trending_items(self, top_n: int = 20, excluded_items: List[int] = None) -> List[Tuple[int, float]]:
        """获取趋势商品"""
        if excluded_items is None:
            excluded_items = set()
        else:
            excluded_items = set(excluded_items)

        # 过滤排除的商品
        filtered_scores = {item: score for item, score in self.trending_scores.items()
                          if item not in excluded_items}

        # 排序并返回Top-N
        sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]