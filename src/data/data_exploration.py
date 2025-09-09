"""
天池移动电商推荐算法 - 数据探索与预处理模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import gc
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataExplorer:
    """数据探索类"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.behavior_weights = {1: 1, 2: 2, 3: 3, 4: 4}  # 行为权重
        
    def load_data_sample(self, file_path, nrows=1000000):
        """加载数据样本"""
        print(f"正在加载数据: {file_path}")
        
        # 定义列名
        if 'user' in file_path:
            columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']
        else:
            columns = ['item_id', 'item_geohash', 'item_category']
            
        # 读取数据
        df = pd.read_csv(file_path, sep='\t', names=columns, nrows=nrows)
        
        # 内存优化
        df = self.reduce_memory(df)
        
        print(f"数据加载完成，形状: {df.shape}")
        return df
    
    def reduce_memory(self, df):
        """内存优化"""
        start_mem = df.memory_usage().sum() / 1024**2
        print(f"初始内存使用: {start_mem:.2f} MB")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type == 'object':
                # 对象类型转换为category，但保留时间列
                if col != 'time' and df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif col_type == 'int64':
                # 整数类型降级
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type == 'float64':
                # 浮点类型降级
                df[col] = pd.to_numeric(df[col], downcast='float')
                
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"优化后内存使用: {end_mem:.2f} MB")
        print(f"内存节省: {(start_mem - end_mem) / start_mem * 100:.1f}%")
        
        return df
    
    def analyze_behavior_distribution(self, df):
        """分析行为类型分布"""
        print("\n=== 行为类型分布分析 ===")
        
        behavior_dist = df['behavior_type'].value_counts().sort_index()
        behavior_names = {1: '浏览', 2: '收藏', 3: '加购物车', 4: '购买'}
        
        print("行为类型统计:")
        for behavior_type, count in behavior_dist.items():
            behavior_name = behavior_names.get(behavior_type, f'类型{behavior_type}')
            percentage = count / len(df) * 100
            print(f"{behavior_name}: {count:,} ({percentage:.2f}%)")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        behavior_dist.index = [behavior_names.get(x, f'Type{x}') for x in behavior_dist.index]
        behavior_dist.plot(kind='bar')
        plt.title('User Behavior Type Distribution')
        plt.xlabel('Behavior Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('behavior_distribution.png')
        plt.close()
        
        return behavior_dist
    
    def analyze_time_distribution(self, df):
        """分析时间分布"""
        print("\n=== 时间分布分析 ===")
        
        # 时间格式转换
        df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
        
        # 时间范围
        min_time = df['datetime'].min()
        max_time = df['datetime'].max()
        print(f"时间范围: {min_time} 到 {max_time}")
        
        # 按日期统计
        df['date'] = df['datetime'].dt.date
        daily_stats = df.groupby('date').size()
        
        print(f"平均每日行为数: {daily_stats.mean():.0f}")
        print(f"最大每日行为数: {daily_stats.max():.0f}")
        print(f"最小每日行为数: {daily_stats.min():.0f}")
        
        # 按小时统计
        df['hour'] = df['datetime'].dt.hour
        hourly_stats = df.groupby('hour').size()
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        daily_stats.plot()
        plt.title('Daily Behavior Count Trend')
        plt.xlabel('Date')
        plt.ylabel('Behavior Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        hourly_stats.plot(kind='bar')
        plt.title('Hourly Behavior Distribution')
        plt.xlabel('Hour')
        plt.ylabel('Behavior Count')
        
        plt.tight_layout()
        plt.savefig('time_distribution.png')
        plt.close()
        
        return daily_stats, hourly_stats
    
    def analyze_user_item_stats(self, df):
        """分析用户和商品统计"""
        print("\n=== 用户和商品统计 ===")
        
        # 用户统计
        user_stats = df['user_id'].value_counts()
        print(f"总用户数: {len(user_stats):,}")
        print(f"平均每用户行为数: {user_stats.mean():.2f}")
        print(f"最多行为用户: {user_stats.max():,}")
        print(f"最少行为用户: {user_stats.min():,}")
        
        # 商品统计
        item_stats = df['item_id'].value_counts()
        print(f"总商品数: {len(item_stats):,}")
        print(f"平均每商品行为数: {item_stats.mean():.2f}")
        print(f"最多行为商品: {item_stats.max():,}")
        print(f"最少行为商品: {item_stats.min():,}")
        
        # 类别统计
        category_stats = df['item_category'].value_counts()
        print(f"总类别数: {len(category_stats):,}")
        print(f"平均每类别商品数: {category_stats.mean():.2f}")
        
        return user_stats, item_stats, category_stats
    
    def analyze_missing_values(self, df):
        """分析缺失值"""
        print("\n=== 缺失值分析 ===")
        
        missing_stats = df.isnull().sum()
        missing_percentage = (missing_stats / len(df)) * 100
        
        for col in df.columns:
            missing_count = missing_stats[col]
            missing_pct = missing_percentage[col]
            if missing_count > 0:
                print(f"{col}: {missing_count:,} ({missing_pct:.2f}%)")
            else:
                print(f"{col}: 无缺失值")
        
        return missing_stats
    
    def run_exploration(self, sample_size=1000000):
        """运行完整的数据探索"""
        print("开始数据探索分析...")
        
        # 加载数据
        user_data = self.load_data_sample(
            f"{self.data_path}/tianchi_fresh_comp_train_user_online_partA.txt", 
            nrows=sample_size
        )
        item_data = self.load_data_sample(
            f"{self.data_path}/tianchi_fresh_comp_train_item_online.txt"
        )
        
        # 行为分布分析
        behavior_dist = self.analyze_behavior_distribution(user_data)
        
        # 时间分布分析
        daily_stats, hourly_stats = self.analyze_time_distribution(user_data)
        
        # 用户商品统计
        user_stats, item_stats, category_stats = self.analyze_user_item_stats(user_data)
        
        # 缺失值分析
        missing_stats = self.analyze_missing_values(user_data)
        
        # 保存探索结果
        exploration_results = {
            'behavior_distribution': behavior_dist.to_dict(),
            'daily_stats': {str(k): v for k, v in daily_stats.to_dict().items()},
            'hourly_stats': hourly_stats.to_dict(),
            'user_count': len(user_stats),
            'item_count': len(item_stats),
            'category_count': len(category_stats),
            'missing_values': missing_stats.to_dict()
        }
        
        import json
        with open('exploration_results.json', 'w', encoding='utf-8') as f:
            json.dump(exploration_results, f, ensure_ascii=False, indent=2)
        
        print("\n数据探索完成！结果已保存到 exploration_results.json")
        
        return user_data, item_data, exploration_results


if __name__ == "__main__":
    # 运行数据探索
    explorer = DataExplorer('dataset')
    user_data, item_data, results = explorer.run_exploration(sample_size=1000000)