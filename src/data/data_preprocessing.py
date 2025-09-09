"""
天池移动电商推荐算法 - 数据预处理模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}  # 行为权重
        self.label_encoders = {}
        
    def load_data(self, file_path, sample_ratio=0.1):
        """加载数据并采样"""
        print(f"正在加载数据: {file_path}")
        print(f"采样比例: {sample_ratio}")
        
        # 定义列名
        if 'user' in file_path:
            columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        else:
            columns = ['item_id', 'item_geohash', 'item_category']
        
        # 计算采样行数
        if 'user' in file_path:
            total_rows = 600000000  # 约6亿行
            sample_rows = int(total_rows * sample_ratio)
        else:
            total_rows = 6781009    # 约678万行
            sample_rows = total_rows  # 商品数据全部加载
            
        # 读取数据
        df = pd.read_csv(file_path, sep='\t', names=columns, nrows=sample_rows)
        
        # 内存优化
        df = self.reduce_memory(df)
        
        print(f"数据加载完成，形状: {df.shape}")
        return df
    
    def reduce_memory(self, df):
        """内存优化"""
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # 保留时间列不被转换为category
                if col != 'time' and df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif col_type == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
                
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"内存优化: {start_mem:.2f}MB -> {end_mem:.2f}MB")
        
        return df
    
    def preprocess_time_features(self, df):
        """预处理时间特征"""
        print("预处理时间特征...")
        
        # 时间格式转换
        df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
        df['timestamp'] = df['datetime'].astype(np.int64) // 10**9
        
        # 提取时间特征
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 计算到预测日期的天数（预测日期为2014-12-19）
        predict_date = datetime(2014, 12, 19)
        df['days_to_predict'] = (predict_date - df['datetime']).dt.days
        
        print("时间特征预处理完成")
        return df
    
    def preprocess_behavior_features(self, df):
        """预处理行为特征"""
        print("预处理行为特征...")
        
        # 添加行为权重
        df['behavior_weight'] = df['behavior_type'].map(self.behavior_weights)
        
        # 行为类型编码
        df['behavior_type_encoded'] = df['behavior_type'] - 1  # 0-3
        
        print("行为特征预处理完成")
        return df
    
    def preprocess_geohash_features(self, df):
        """预处理地理位置特征"""
        print("预处理地理位置特征...")
        
        # 处理用户地理位置 - 先转换为字符串再填充
        if 'user_geohash' in df.columns:
            df['user_geohash_filled'] = df['user_geohash'].astype(str).fillna('unknown')
        
        # 处理商品地理位置
        if 'item_geohash' in df.columns:
            df['item_geohash_filled'] = df['item_geohash'].astype(str).fillna('unknown')
        
        print("地理位置特征预处理完成")
        return df
    
    def encode_categorical_features(self, df):
        """编码分类特征"""
        print("编码分类特征...")
        
        # 用户ID编码
        if 'user_id' not in self.label_encoders:
            self.label_encoders['user_id'] = LabelEncoder()
        
        # 商品ID编码
        if 'item_id' not in self.label_encoders:
            self.label_encoders['item_id'] = LabelEncoder()
        
        # 商品类别编码
        if 'item_category' not in self.label_encoders:
            self.label_encoders['item_category'] = LabelEncoder()
        
        # 拟合编码器
        if 'user_id' in df.columns:
            self.label_encoders['user_id'].fit(df['user_id'].astype(str))
            df['user_id_encoded'] = self.label_encoders['user_id'].transform(df['user_id'].astype(str))
        
        if 'item_id' in df.columns:
            self.label_encoders['item_id'].fit(df['item_id'].astype(str))
            df['item_id_encoded'] = self.label_encoders['item_id'].transform(df['item_id'].astype(str))
        
        if 'item_category' in df.columns:
            self.label_encoders['item_category'].fit(df['item_category'].astype(str))
            df['item_category_encoded'] = self.label_encoders['item_category'].transform(df['item_category'].astype(str))
        
        print("分类特征编码完成")
        return df
    
    def split_train_validation(self, df):
        """时间序列划分训练集和验证集"""
        print("划分训练集和验证集...")
        
        # 训练集：2014-11-18 到 2014-12-15 的行为，以2014-12-16的购买作为标签
        # 验证集：2014-11-21 到 2014-12-18 的行为，以2014-12-19的购买作为标签
        
        train_behavior_end = datetime(2014, 12, 15)
        validation_behavior_end = datetime(2014, 12, 18)
        
        train_behavior_mask = df['datetime'] <= train_behavior_end
        validation_behavior_mask = df['datetime'] <= validation_behavior_end
        
        train_behavior = df[train_behavior_mask].copy()
        validation_behavior = df[validation_behavior_mask].copy()
        
        # 提取购买行为作为标签
        train_labels = df[(df['datetime'].dt.date == datetime(2014, 12, 16).date()) & 
                        (df['behavior_type'] == 4)][['user_id', 'item_id']].copy()
        
        validation_labels = df[(df['datetime'].dt.date == datetime(2014, 12, 19).date()) & 
                              (df['behavior_type'] == 4)][['user_id', 'item_id']].copy()
        
        train_labels['label'] = 1
        validation_labels['label'] = 1
        
        print(f"训练集行为数据: {train_behavior.shape}")
        print(f"验证集行为数据: {validation_behavior.shape}")
        print(f"训练集标签: {train_labels.shape}")
        print(f"验证集标签: {validation_labels.shape}")
        
        return train_behavior, validation_behavior, train_labels, validation_labels
    
    def create_negative_samples(self, behavior_data, positive_labels, negative_ratio=5):
        """创建负样本"""
        print("创建负样本...")
        
        # 获取所有用户和商品
        all_users = set(behavior_data['user_id'].unique())
        all_items = set(behavior_data['item_id'].unique())
        
        # 获取正样本
        positive_pairs = set(zip(positive_labels['user_id'], positive_labels['item_id']))
        
        # 创建负样本
        negative_samples = []
        users_in_positive = positive_labels['user_id'].unique()
        
        for user_id in users_in_positive:
            # 获取用户未交互的商品
            user_items = set(behavior_data[behavior_data['user_id'] == user_id]['item_id'].unique())
            user_positive_items = set(positive_labels[positive_labels['user_id'] == user_id]['item_id'])
            user_negative_items = list(all_items - user_items - user_positive_items)
            
            # 随机选择负样本
            n_positive = len(user_positive_items)
            n_negative = min(n_positive * negative_ratio, len(user_negative_items))
            
            if n_negative > 0:
                selected_items = np.random.choice(user_negative_items, n_negative, replace=False)
                for item_id in selected_items:
                    negative_samples.append({'user_id': user_id, 'item_id': item_id, 'label': 0})
        
        negative_df = pd.DataFrame(negative_samples)
        
        # 合并正负样本
        positive_df = positive_labels.copy()
        all_samples = pd.concat([positive_df, negative_df], ignore_index=True)
        
        print(f"正样本数量: {len(positive_df)}")
        print(f"负样本数量: {len(negative_df)}")
        if len(positive_df) > 0:
            print(f"正负样本比例: 1:{len(negative_df)/len(positive_df):.1f}")
        
        return all_samples
    
    def run_preprocessing(self, data_path, sample_ratio=0.1):
        """运行完整的数据预处理"""
        print("开始数据预处理...")
        
        # 加载数据
        user_data = self.load_data(
            f"{data_path}/tianchi_fresh_comp_train_user_online_partA.txt",
            sample_ratio=sample_ratio
        )
        item_data = self.load_data(
            f"{data_path}/tianchi_fresh_comp_train_item_online.txt"
        )
        
        # 时间特征预处理
        user_data = self.preprocess_time_features(user_data)
        
        # 行为特征预处理
        user_data = self.preprocess_behavior_features(user_data)
        
        # 地理位置特征预处理
        user_data = self.preprocess_geohash_features(user_data)
        
        # 分类特征编码
        user_data = self.encode_categorical_features(user_data)
        
        # 划分训练集和验证集
        train_behavior, validation_behavior, train_labels, validation_labels = self.split_train_validation(user_data)
        
        # 创建负样本
        train_samples = self.create_negative_samples(train_behavior, train_labels)
        validation_samples = self.create_negative_samples(validation_behavior, validation_labels)
        
        # 保存预处理结果
        train_behavior.to_pickle('train_behavior.pkl')
        validation_behavior.to_pickle('validation_behavior.pkl')
        train_samples.to_pickle('train_samples.pkl')
        validation_samples.to_pickle('validation_samples.pkl')
        item_data.to_pickle('item_data.pkl')
        
        # 保存编码器
        import pickle
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("数据预处理完成！")
        
        return train_behavior, validation_behavior, train_samples, validation_samples, item_data


if __name__ == "__main__":
    # 运行数据预处理
    preprocessor = DataPreprocessor()
    train_behavior, validation_behavior, train_samples, validation_samples, item_data = preprocessor.run_preprocessing(
        'dataset', sample_ratio=0.01  # 使用1%数据进行快速测试
    )