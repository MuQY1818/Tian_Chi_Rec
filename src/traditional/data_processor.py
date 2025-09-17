import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta


class TraditionalDataProcessor:
    """传统推荐算法的数据预处理器"""

    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.user2id: Dict[int, int] = {}
        self.item2id: Dict[int, int] = {}
        self.id2user: List[int] = []
        self.id2item: List[int] = []
        self.num_users: int = 0
        self.num_items: int = 0

        # 行为权重
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}

    def load_data(self, sample_frac: float = 1.0, use_full_data: bool = False) -> pd.DataFrame:
        """加载用户行为数据"""
        print("加载数据...")

        if use_full_data:
            print("使用全量数据集")
            # 加载partA和partB数据
            files = [
                os.path.join(self.data_dir, "tianchi_fresh_comp_train_user_online_partA.txt"),
                os.path.join(self.data_dir, "tianchi_fresh_comp_train_user_online_partB.txt")
            ]

            dfs = []
            for file_path in files:
                if os.path.exists(file_path):
                    print(f"加载 {file_path}")
                    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
                    df_part = pd.read_csv(file_path, sep="\t", names=columns)
                    dfs.append(df_part)
                    print(f"  {file_path}: {len(df_part)} 行")

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                print(f"合并后总数据: {len(df)} 行")
            else:
                raise FileNotFoundError("未找到数据文件")
        else:
            # 优先使用小数据集
            small_data_path = os.path.join(self.data_dir, "small_dataset.txt")
            if os.path.exists(small_data_path):
                print("使用小数据集进行测试")
                file_path = small_data_path
            else:
                # 如果没有小数据集，使用partA
                file_path = os.path.join(self.data_dir, "tianchi_fresh_comp_train_user_online_partA.txt")

            columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
            df = pd.read_csv(file_path, sep="\t", names=columns)

        if 0 < sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"采样后数据: {len(df)} 行")

        print(f"原始数据: {len(df)} 行")
        return self._preprocess_data(df)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和预处理"""
        print("数据预处理...")

        # 基本清理
        df = df[df["behavior_type"].isin([1, 2, 3, 4])].copy()
        df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H", errors="coerce")
        df = df.dropna(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)
        df["user_id"] = df["user_id"].astype(np.int64)
        df["item_id"] = df["item_id"].astype(np.int64)

        # 去重
        df = df.drop_duplicates(subset=["user_id", "item_id", "datetime"]).reset_index(drop=True)

        # 过滤活跃用户和商品
        user_counts = df.groupby("user_id").size()
        item_counts = df.groupby("item_id").size()

        active_users = user_counts[user_counts >= 2].index
        active_items = item_counts[item_counts >= 2].index

        df = df[df["user_id"].isin(active_users) & df["item_id"].isin(active_items)].reset_index(drop=True)

        # 构建ID映射
        self._build_mappings(df)

        # 添加映射后的ID
        df["uid"] = df["user_id"].map(self.user2id)
        df["iid"] = df["item_id"].map(self.item2id)

        # 添加权重
        df["weight"] = df["behavior_type"].map(self.behavior_weights)

        print(f"预处理后: {len(df)} 行, {self.num_users} 用户, {self.num_items} 商品")
        return df

    def _build_mappings(self, df: pd.DataFrame):
        """构建用户和商品ID映射"""
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user2id = {user: idx for idx, user in enumerate(unique_users)}
        self.item2id = {item: idx for idx, item in enumerate(unique_items)}
        self.id2user = unique_users
        self.id2item = unique_items

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

    def build_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """构建用户-商品交互矩阵"""
        print("构建交互矩阵...")

        # 聚合每个用户-商品对的权重
        interactions = df.groupby(["uid", "iid"])["weight"].sum().reset_index()

        # 构建稀疏矩阵
        rows = interactions["uid"].values
        cols = interactions["iid"].values
        data = interactions["weight"].values

        matrix = coo_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
        return matrix.tocsr()

    def split_by_time(self, df: pd.DataFrame, train_end: str = "2014-12-17",
                      val_date: str = "2014-12-18") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按时间划分训练和验证集"""
        print("时间划分...")

        train_end_dt = pd.to_datetime(train_end + " 23:59:59")
        val_date_dt = pd.to_datetime(val_date)

        train_df = df[df["datetime"] <= train_end_dt].copy()
        val_df = df[df["datetime"].dt.date == val_date_dt.date()].copy()

        print(f"训练集: {len(train_df)} 行")
        print(f"验证集: {len(val_df)} 行")

        return train_df, val_df

    def get_user_items(self, df: pd.DataFrame, behavior_types: List[int] = None) -> Dict[int, List[int]]:
        """获取用户交互的商品列表"""
        if behavior_types is not None:
            df = df[df["behavior_type"].isin(behavior_types)]

        user_items = df.groupby("uid")["iid"].apply(list).to_dict()
        return user_items

    def get_time_weighted_matrix(self, df: pd.DataFrame, decay_factor: float = 0.1) -> csr_matrix:
        """构建带时间衰减的交互矩阵"""
        print("构建时间衰减权重矩阵...")

        # 计算时间衰减权重
        max_time = df["datetime"].max()
        df = df.copy()
        df["time_diff"] = (max_time - df["datetime"]).dt.days
        df["time_weight"] = np.exp(-decay_factor * df["time_diff"])
        df["final_weight"] = df["weight"] * df["time_weight"]

        # 聚合权重
        interactions = df.groupby(["uid", "iid"])["final_weight"].sum().reset_index()

        rows = interactions["uid"].values
        cols = interactions["iid"].values
        data = interactions["final_weight"].values

        matrix = coo_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))
        return matrix.tocsr()