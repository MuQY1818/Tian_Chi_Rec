"""
大规模数据处理模块 - 使用Dask处理50GB数据
"""

import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.distributed import Client
import os
import gc
from datetime import datetime
import logging
from tqdm import tqdm

class LargeScalePreprocessor:
    """大规模数据预处理器"""
    
    def __init__(self, data_path="dataset"):
        self.data_path = data_path
        self.client = None
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/large_scale_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start_dask_client(self):
        """启动Dask客户端"""
        try:
            self.client = Client(n_workers=4, memory_limit='16GB')
            self.logger.info(f"Dask客户端启动: {self.client}")
            return True
        except Exception as e:
            self.logger.error(f"Dask客户端启动失败: {e}")
            return False
    
    def load_full_data(self):
        """加载全量数据"""
        self.logger.info("开始加载全量数据...")
        
        # 定义列名
        user_columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 
                       'item_category', 'time']
        item_columns = ['item_id', 'item_geohash', 'item_category']
        
        try:
            # 加载用户行为数据 - 使用Dask
            user_data_a = dd.read_csv(
                os.path.join(self.data_path, 'tianchi_fresh_comp_train_user_online_partA.txt'),
                sep='\t',
                names=user_columns,
                dtype={'user_geohash': 'object', 'item_category': 'object'},
                blocksize='256MB'  # 分块大小
            )
            
            user_data_b = dd.read_csv(
                os.path.join(self.data_path, 'tianchi_fresh_comp_train_user_online_partB.txt'),
                sep='\t',
                names=user_columns,
                dtype={'user_geohash': 'object', 'item_category': 'object'},
                blocksize='256MB'
            )
            
            # 合并用户数据
            user_data = dd.concat([user_data_a, user_data_b], ignore_index=True)
            
            # 加载商品数据
            item_data = dd.read_csv(
                os.path.join(self.data_path, 'tianchi_fresh_comp_train_item_online.txt'),
                sep='\t',
                names=item_columns,
                dtype={'item_geohash': 'object', 'item_category': 'object'},
                blocksize='256MB'
            )
            
            self.logger.info(f"用户数据加载完成: {len(user_data):,} 行")
            self.logger.info(f"商品数据加载完成: {len(item_data):,} 行")
            
            return user_data, item_data
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return None, None
    
    def clean_data(self, user_data, item_data):
        """数据清洗"""
        self.logger.info("开始数据清洗...")
        
        try:
            # 处理时间列
            user_data['time'] = dd.to_datetime(user_data['time'], format='%Y-%m-%d %H')
            
            # 填充缺失值
            user_data['user_geohash'] = user_data['user_geohash'].fillna('unknown')
            item_data['item_geohash'] = item_data['item_geohash'].fillna('unknown')
            
            # 过滤异常数据
            user_data = user_data[user_data['behavior_type'].between(1, 4)]
            
            # 添加时间相关列
            user_data['hour'] = user_data['time'].dt.hour
            user_data['day'] = user_data['time'].dt.day
            user_data['weekday'] = user_data['time'].dt.weekday
            
            self.logger.info("数据清洗完成")
            return user_data, item_data
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {e}")
            return None, None
    
    def create_time_series_features(self, user_data):
        """创建时间序列特征"""
        self.logger.info("创建时间序列特征...")
        
        try:
            # 按用户分组，创建行为序列
            def create_sequences(group):
                group = group.sort_values('time')
                return {
                    'user_id': group['user_id'].iloc[0],
                    'item_sequence': group['item_id'].tolist(),
                    'behavior_sequence': group['behavior_type'].tolist(),
                    'time_sequence': group['time'].tolist(),
                    'category_sequence': group['item_category'].tolist()
                }
            
            # 使用Dask的groupby_apply
            user_sequences = user_data.groupby('user_id').apply(
                create_sequences, 
                meta={'user_id': 'int64', 'item_sequence': 'object', 
                      'behavior_sequence': 'object', 'time_sequence': 'object',
                      'category_sequence': 'object'}
            )
            
            self.logger.info(f"时间序列特征创建完成: {len(user_sequences):,} 用户")
            return user_sequences
            
        except Exception as e:
            self.logger.error(f"时间序列特征创建失败: {e}")
            return None
    
    def create_aggregate_features(self, user_data, item_data):
        """创建聚合特征"""
        self.logger.info("创建聚合特征...")
        
        try:
            # 用户行为统计
            user_stats = user_data.groupby('user_id').agg({
                'behavior_type': ['count', 'nunique'],
                'item_id': 'nunique',
                'item_category': 'nunique'
            }).compute()
            
            # 商品统计
            item_stats = user_data.groupby('item_id').agg({
                'user_id': 'count',
                'behavior_type': 'mean'
            }).compute()
            
            # 类别统计
            category_stats = user_data.groupby('item_category').agg({
                'user_id': 'count',
                'item_id': 'nunique'
            }).compute()
            
            self.logger.info("聚合特征创建完成")
            return user_stats, item_stats, category_stats
            
        except Exception as e:
            self.logger.error(f"聚合特征创建失败: {e}")
            return None, None, None
    
    def split_train_test(self, user_data):
        """时间序列数据集划分"""
        self.logger.info("划分训练测试集...")
        
        try:
            # 按时间划分
            predict_date = '2014-12-19'
            train_end_date = '2014-12-15'
            
            # 训练集
            train_data = user_data[user_data['time'] < train_end_date]
            
            # 验证集
            val_data = user_data[
                (user_data['time'] >= train_end_date) & 
                (user_data['time'] < predict_date)
            ]
            
            # 测试集（预测目标）
            test_users = user_data[user_data['time'] < predict_date]['user_id'].unique()
            
            self.logger.info(f"训练集: {len(train_data):,} 行")
            self.logger.info(f"验证集: {len(val_data):,} 行")
            self.logger.info(f"测试用户数: {len(test_users):,}")
            
            return train_data, val_data, test_users
            
        except Exception as e:
            self.logger.error(f"数据集划分失败: {e}")
            return None, None, None
    
    def save_processed_data(self, data_dict, output_dir="data/processed"):
        """保存处理后的数据"""
        self.logger.info("保存处理后的数据...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for name, data in data_dict.items():
                if isinstance(data, dd.DataFrame):
                    # 保存为parquet格式（支持Dask）
                    data.to_parquet(os.path.join(output_dir, f"{name}.parquet"))
                elif isinstance(data, pd.DataFrame):
                    # 保存为pickle格式
                    data.to_pickle(os.path.join(output_dir, f"{name}.pkl"))
                else:
                    # 保存为其他格式
                    import pickle
                    with open(os.path.join(output_dir, f"{name}.pkl"), 'wb') as f:
                        pickle.dump(data, f)
                
                self.logger.info(f"数据已保存: {name}")
                
        except Exception as e:
            self.logger.error(f"数据保存失败: {e}")
    
    def run_full_pipeline(self):
        """运行完整的数据处理流水线"""
        self.logger.info("开始大规模数据处理流水线...")
        
        # 启动Dask客户端
        if not self.start_dask_client():
            return False
        
        # 1. 加载数据
        user_data, item_data = self.load_full_data()
        if user_data is None or item_data is None:
            return False
        
        # 2. 数据清洗
        user_data, item_data = self.clean_data(user_data, item_data)
        if user_data is None or item_data is None:
            return False
        
        # 3. 创建时间序列特征
        user_sequences = self.create_time_series_features(user_data)
        
        # 4. 创建聚合特征
        user_stats, item_stats, category_stats = self.create_aggregate_features(
            user_data, item_data
        )
        
        # 5. 数据集划分
        train_data, val_data, test_users = self.split_train_test(user_data)
        
        # 6. 保存处理后的数据
        processed_data = {
            'user_sequences': user_sequences,
            'user_stats': user_stats,
            'item_stats': item_stats,
            'category_stats': category_stats,
            'train_data': train_data,
            'val_data': val_data,
            'test_users': test_users,
            'item_data': item_data
        }
        
        self.save_processed_data(processed_data)
        
        # 关闭Dask客户端
        if self.client:
            self.client.close()
        
        self.logger.info("大规模数据处理流水线完成！")
        return True


def main():
    """主函数"""
    processor = LargeScalePreprocessor()
    success = processor.run_full_pipeline()
    
    if success:
        print("大规模数据处理成功完成！")
    else:
        print("数据处理失败，请检查日志。")


if __name__ == "__main__":
    main()