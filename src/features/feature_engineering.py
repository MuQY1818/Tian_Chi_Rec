"""
天池移动电商推荐算法 - 特征工程模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}
        self.scalers = {}
        self.encoders = {}
        
    def load_processed_data(self):
        """加载预处理后的数据"""
        print("加载预处理后的数据...")
        
        user_data = pd.read_pickle('user_data_processed.pkl')
        item_data = pd.read_pickle('item_data_processed.pkl')
        user_stats = pd.read_pickle('user_stats.pkl')
        item_stats = pd.read_pickle('item_stats.pkl')
        user_item_features = pd.read_pickle('user_item_features.pkl')
        train_labels = pd.read_pickle('train_labels.pkl')
        test_labels = pd.read_pickle('test_labels.pkl')
        
        print("数据加载完成")
        return user_data, item_data, user_stats, item_stats, user_item_features, train_labels, test_labels
    
    def create_user_behavior_features(self, user_data, user_stats):
        """创建用户行为特征"""
        print("创建用户行为特征...")
        
        # 按行为类型统计
        behavior_counts = user_data.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
        behavior_counts.columns = [f'user_behavior_{col}_count' for col in behavior_counts.columns]
        
        # 按行为类型加权统计
        behavior_weights = user_data.groupby(['user_id', 'behavior_type'])['behavior_weight'].sum().unstack(fill_value=0)
        behavior_weights.columns = [f'user_behavior_{col}_weight' for col in behavior_weights.columns]
        
        # 用户活跃度特征
        user_activity = user_data.groupby('user_id').agg({
            'datetime': ['count', 'nunique'],
            'timestamp': ['min', 'max'],
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
        })
        
        user_activity.columns = ['user_total_behaviors', 'user_active_days', 
                               'user_first_behavior', 'user_last_behavior', 'user_preferred_hour']
        
        # 用户时间间隔特征
        user_intervals = user_data.sort_values(['user_id', 'datetime']).groupby('user_id')['datetime'].diff().dt.total_seconds()
        user_interval_features = user_intervals.groupby(user_data['user_id']).agg(['mean', 'std', 'min', 'max'])
        user_interval_features.columns = ['user_interval_mean', 'user_interval_std', 
                                       'user_interval_min', 'user_interval_max']
        
        print("用户行为特征创建完成")
        return behavior_counts, behavior_weights, user_activity, user_interval_features
    
    def create_time_features(self, user_data):
        """创建时间特征"""
        print("创建时间特征...")
        
        # 时间衰减权重（指数衰减）
        predict_date = datetime(2014, 12, 19)
        user_data['days_to_predict'] = (predict_date - user_data['datetime']).dt.days
        user_data['time_decay_weight'] = np.exp(-0.1 * user_data['days_to_predict'])
        
        # 时间窗口特征
        time_windows = [3, 7, 14]  # 3天、7天、14天
        
        time_window_features = []
        for window in time_windows:
            window_end = predict_date - timedelta(days=1)
            window_start = window_end - timedelta(days=window)
            
            window_mask = (user_data['datetime'] >= window_start) & (user_data['datetime'] <= window_end)
            window_data = user_data[window_mask]
            
            # 窗口内行为统计
            window_stats = window_data.groupby('user_id').agg({
                'behavior_type': 'count',
                'behavior_weight': 'sum',
                'item_id': 'nunique'
            }).rename(columns={
                'behavior_type': f'user_last_{window}days_behavior_count',
                'behavior_weight': f'user_last_{window}days_weight_sum',
                'item_id': f'user_last_{window}days_unique_items'
            })
            
            time_window_features.append(window_stats)
        
        # 合并时间窗口特征
        time_window_df = pd.concat(time_window_features, axis=1).fillna(0)
        
        # 周期性特征
        periodic_features = user_data.groupby('user_id').agg({
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
            'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
            'is_weekend': 'mean'
        }).rename(columns={
            'hour': 'user_preferred_hour',
            'day_of_week': 'user_preferred_day',
            'is_weekend': 'user_weekend_ratio'
        })
        
        print("时间特征创建完成")
        return time_window_df, periodic_features
    
    def create_item_features(self, user_data, item_data, item_stats):
        """创建商品特征"""
        print("创建商品特征...")
        
        # 商品热度特征
        item_popularity = item_stats.copy()
        item_popularity['item_conversion_rate'] = item_popularity['item_behavior_count'] / item_popularity['item_unique_users']
        
        # 商品类别特征
        item_category_stats = user_data.groupby(['item_id', 'item_category']).agg({
            'behavior_type': 'count',
            'behavior_weight': 'sum',
            'user_id': 'nunique'
        }).reset_index()
        
        item_category_pivot = item_category_stats.pivot_table(
            index='item_id', 
            columns='item_category', 
            values='behavior_type', 
            fill_value=0
        )
        
        # 商品时间特征
        item_time_features = user_data.groupby('item_id').agg({
            'datetime': ['min', 'max'],
            'timestamp': lambda x: x.max() - x.min()
        })
        
        item_time_features.columns = ['item_first_interaction', 'item_last_interaction', 'item_lifespan']
        
        print("商品特征创建完成")
        return item_popularity, item_category_pivot, item_time_features
    
    def create_user_item_interaction_features(self, user_data, user_item_features):
        """创建用户-商品交互特征"""
        print("创建用户-商品交互特征...")
        
        # 用户-商品交互强度
        interaction_strength = user_item_features.copy()
        
        # 用户-商品时间间隔
        interaction_strength['interaction_duration'] = (
            interaction_strength['last_interaction'] - interaction_strength['first_interaction']
        ).dt.total_seconds()
        
        # 用户-商品最近交互时间
        predict_date = datetime(2014, 12, 19)
        interaction_strength['days_since_last_interaction'] = (
            predict_date - interaction_strength['last_interaction']
        ).dt.days
        
        # 用户-商品交互频率
        interaction_counts = user_data.groupby(['user_id', 'item_id']).size().reset_index(name='interaction_count')
        interaction_strength = interaction_strength.merge(interaction_counts, on=['user_id', 'item_id'], how='left')
        
        print("用户-商品交互特征创建完成")
        return interaction_strength
    
    def create_collaborative_filtering_features(self, user_data, top_k=10):
        """创建协同过滤特征"""
        print("创建协同过滤特征...")
        
        # 用户-物品交互矩阵
        user_item_matrix = user_data.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='behavior_weight', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # 物品相似度（余弦相似度）
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 为减少计算量，只对热门物品计算相似度
        popular_items = user_data['item_id'].value_counts().head(top_k).index
        item_similarity_matrix = cosine_similarity(user_item_matrix[popular_items].T)
        
        # 创建物品相似度DataFrame
        item_similarity_df = pd.DataFrame(
            item_similarity_matrix,
            index=popular_items,
            columns=popular_items
        )
        
        # 为每个用户推荐相似物品
        cf_features = []
        
        for user_id in user_item_matrix.index[:100]:  # 限制用户数量以减少计算量
            user_items = user_item_matrix.loc[user_id]
            user_positive_items = user_items[user_items > 0].index
            
            if len(user_positive_items) > 0:
                # 计算相似物品得分
                similar_scores = {}
                for item in user_positive_items:
                    if item in item_similarity_df.index:
                        similar_items = item_similarity_df[item].sort_values(ascending=False)[1:6]  # Top-5相似物品
                        for similar_item, score in similar_items.items():
                            if similar_item not in user_positive_items:
                                similar_scores[similar_item] = similar_scores.get(similar_item, 0) + score
                
                # 创建CF特征
                for item, score in similar_scores.items():
                    cf_features.append({
                        'user_id': user_id,
                        'item_id': item,
                        'cf_score': score
                    })
        
        cf_df = pd.DataFrame(cf_features)
        
        print("协同过滤特征创建完成")
        return cf_df
    
    def merge_all_features(self, user_data, item_data, train_labels, test_labels):
        """合并所有特征"""
        print("合并所有特征...")
        
        # 加载基础特征
        user_stats = pd.read_pickle('user_stats.pkl')
        item_stats = pd.read_pickle('item_stats.pkl')
        user_item_features = pd.read_pickle('user_item_features.pkl')
        
        # 创建各类特征
        behavior_counts, behavior_weights, user_activity, user_interval_features = self.create_user_behavior_features(user_data, user_stats)
        time_window_df, periodic_features = self.create_time_features(user_data)
        item_popularity, item_category_pivot, item_time_features = self.create_item_features(user_data, item_data, item_stats)
        interaction_strength = self.create_user_item_interaction_features(user_data, user_item_features)
        
        # 合并用户特征
        user_features = pd.concat([
            user_stats,
            behavior_counts,
            behavior_weights,
            user_activity,
            user_interval_features,
            time_window_df,
            periodic_features
        ], axis=1).fillna(0)
        
        # 合并商品特征
        item_features = pd.concat([
            item_stats,
            item_popularity,
            item_time_features
        ], axis=1).fillna(0)
        
        # 为训练集创建特征矩阵
        print("创建训练集特征矩阵...")
        train_features = train_labels.copy()
        
        # 合并用户特征
        train_features = train_features.merge(user_features, left_on='user_id', right_index=True, how='left')
        
        # 合并商品特征
        train_features = train_features.merge(item_features, left_on='item_id', right_index=True, how='left')
        
        # 合并交互特征
        train_features = train_features.merge(interaction_strength, on=['user_id', 'item_id'], how='left')
        
        # 为测试集创建特征矩阵
        print("创建测试集特征矩阵...")
        test_features = test_labels.copy()
        
        # 合并用户特征
        test_features = test_features.merge(user_features, left_on='user_id', right_index=True, how='left')
        
        # 合并商品特征
        test_features = test_features.merge(item_features, left_on='item_id', right_index=True, how='left')
        
        # 合并交互特征
        test_features = test_features.merge(interaction_strength, on=['user_id', 'item_id'], how='left')
        
        # 填充缺失值
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)
        
        print(f"训练特征矩阵形状: {train_features.shape}")
        print(f"测试特征矩阵形状: {test_features.shape}")
        
        return train_features, test_features, user_features, item_features
    
    def run_feature_engineering(self):
        """运行完整特征工程"""
        print("开始特征工程...")
        
        # 加载数据
        user_data, item_data, user_stats, item_stats, user_item_features, train_labels, test_labels = self.load_processed_data()
        
        # 合并所有特征
        train_features, test_features, user_features, item_features = self.merge_all_features(
            user_data, item_data, train_labels, test_labels
        )
        
        # 保存特征
        train_features.to_pickle('train_features.pkl')
        test_features.to_pickle('test_features.pkl')
        user_features.to_pickle('user_features_final.pkl')
        item_features.to_pickle('item_features_final.pkl')
        
        print("特征工程完成！")
        
        return train_features, test_features, user_features, item_features


if __name__ == "__main__":
    # 运行特征工程
    feature_engineer = FeatureEngineer()
    train_features, test_features, user_features, item_features = feature_engineer.run_feature_engineering()