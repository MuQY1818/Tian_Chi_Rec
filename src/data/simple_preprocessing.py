"""
天池移动电商推荐算法 - 简化版数据预处理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import warnings
warnings.filterwarnings('ignore')

class SimpleDataPreprocessor:
    """简化版数据预处理类"""
    
    def __init__(self):
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}
        
    def load_sample_data(self, data_path, sample_size=100000):
        """加载小样本数据"""
        print(f"加载样本数据，大小: {sample_size}")
        
        # 加载用户数据
        columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        user_data = pd.read_csv(
            f"{data_path}/tianchi_fresh_comp_train_user_online_partA.txt", 
            sep='\t', 
            names=columns, 
            nrows=sample_size
        )
        
        # 加载商品数据
        item_columns = ['item_id', 'item_geohash', 'item_category']
        item_data = pd.read_csv(
            f"{data_path}/tianchi_fresh_comp_train_item_online.txt", 
            sep='\t', 
            names=item_columns
        )
        
        print(f"用户数据形状: {user_data.shape}")
        print(f"商品数据形状: {item_data.shape}")
        
        return user_data, item_data
    
    def basic_preprocessing(self, user_data, item_data):
        """基础预处理"""
        print("开始基础预处理...")
        
        # 时间处理
        user_data['datetime'] = pd.to_datetime(user_data['time'], format='%Y-%m-%d %H')
        user_data['timestamp'] = user_data['datetime'].astype(np.int64) // 10**9
        user_data['hour'] = user_data['datetime'].dt.hour
        user_data['day_of_week'] = user_data['datetime'].dt.dayofweek
        user_data['is_weekend'] = (user_data['day_of_week'] >= 5).astype(int)
        
        # 行为权重
        user_data['behavior_weight'] = user_data['behavior_type'].map(self.behavior_weights)
        
        # 地理位置处理
        user_data['user_geohash_filled'] = user_data['user_geohash'].fillna('unknown')
        item_data['item_geohash_filled'] = item_data['item_geohash'].fillna('unknown')
        
        print("基础预处理完成")
        return user_data, item_data
    
    def create_simple_features(self, user_data, item_data):
        """创建简单特征"""
        print("创建简单特征...")
        
        # 用户行为统计
        user_stats = user_data.groupby('user_id').agg({
            'behavior_type': 'count',
            'behavior_weight': 'sum',
            'item_id': 'nunique'
        }).rename(columns={
            'behavior_type': 'user_behavior_count',
            'behavior_weight': 'user_weighted_behavior',
            'item_id': 'user_unique_items'
        })
        
        # 商品统计
        item_stats = user_data.groupby('item_id').agg({
            'behavior_type': 'count',
            'behavior_weight': 'sum',
            'user_id': 'nunique'
        }).rename(columns={
            'behavior_type': 'item_behavior_count',
            'behavior_weight': 'item_weighted_behavior',
            'user_id': 'item_unique_users'
        })
        
        # 用户-商品交互特征
        user_item_features = user_data.groupby(['user_id', 'item_id']).agg({
            'behavior_type': 'max',
            'behavior_weight': 'max',
            'datetime': ['min', 'max']
        }).reset_index()
        
        user_item_features.columns = ['user_id', 'item_id', 'max_behavior_type', 'max_behavior_weight', 'first_interaction', 'last_interaction']
        
        print("特征创建完成")
        return user_stats, item_stats, user_item_features
    
    def create_train_test_split(self, user_data, test_date='2014-12-15'):
        """创建训练测试集"""
        print("创建训练测试集...")
        
        test_date = pd.to_datetime(test_date)
        
        # 训练集：早于测试日期的数据
        train_data = user_data[user_data['datetime'] < test_date].copy()
        
        # 测试集：测试日期之后的数据
        test_data = user_data[user_data['datetime'] >= test_date].copy()
        
        # 创建标签（购买行为）
        train_labels = train_data[train_data['behavior_type'] == 4][['user_id', 'item_id']].drop_duplicates()
        test_labels = test_data[test_data['behavior_type'] == 4][['user_id', 'item_id']].drop_duplicates()
        
        train_labels['label'] = 1
        test_labels['label'] = 1
        
        print(f"训练数据: {train_data.shape}")
        print(f"测试数据: {test_data.shape}")
        print(f"训练标签: {train_labels.shape}")
        print(f"测试标签: {test_labels.shape}")
        
        return train_data, test_data, train_labels, test_labels
    
    def run_simple_preprocessing(self, data_path, sample_size=100000):
        """运行简化版预处理"""
        print("开始简化版数据预处理...")
        
        # 加载数据
        user_data, item_data = self.load_sample_data(data_path, sample_size)
        
        # 基础预处理
        user_data, item_data = self.basic_preprocessing(user_data, item_data)
        
        # 创建特征
        user_stats, item_stats, user_item_features = self.create_simple_features(user_data, item_data)
        
        # 划分数据集
        train_data, test_data, train_labels, test_labels = self.create_train_test_split(user_data)
        
        # 保存结果
        user_data.to_pickle('user_data_processed.pkl')
        item_data.to_pickle('item_data_processed.pkl')
        user_stats.to_pickle('user_stats.pkl')
        item_stats.to_pickle('item_stats.pkl')
        user_item_features.to_pickle('user_item_features.pkl')
        train_labels.to_pickle('train_labels.pkl')
        test_labels.to_pickle('test_labels.pkl')
        
        print("简化版预处理完成！")
        
        return {
            'user_data': user_data,
            'item_data': item_data,
            'user_stats': user_stats,
            'item_stats': item_stats,
            'user_item_features': user_item_features,
            'train_labels': train_labels,
            'test_labels': test_labels
        }


if __name__ == "__main__":
    # 运行简化版预处理
    preprocessor = SimpleDataPreprocessor()
    results = preprocessor.run_simple_preprocessing('dataset', sample_size=500000)