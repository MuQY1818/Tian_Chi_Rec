#!/usr/bin/env python3
"""
LightGBM模型训练器
用于用户-商品交互预测
"""

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')


class LightGBMTrainer:
    """LightGBM训练器"""

    def __init__(self, feature_dir="/mnt/data/tianchi_features",
                 model_dir="/mnt/data/tianchi_features"):
        self.feature_dir = feature_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model = None
        self.feature_names = None

        print(f"📁 特征目录: {feature_dir}")
        print(f"📁 模型目录: {model_dir}")

    def load_training_data(self):
        """加载训练数据"""
        print("📂 加载训练数据...")

        train_file = os.path.join(self.feature_dir, "training_samples.csv")
        if not os.path.exists(train_file):
            print(f"❌ 训练文件不存在: {train_file}")
            return None

        # 分块加载大文件
        print("  📊 分析文件大小...")
        df_sample = pd.read_csv(train_file, nrows=1000)
        total_lines = sum(1 for _ in open(train_file)) - 1  # 减去头部
        print(f"  📏 总样本数: {total_lines:,}")

        if total_lines > 5000000:  # 超过500万行分块加载
            print("  🔄 分块加载大文件...")
            chunks = []
            chunk_size = 1000000

            for chunk in tqdm(pd.read_csv(train_file, chunksize=chunk_size),
                            total=(total_lines // chunk_size) + 1,
                            desc="加载数据"):
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(train_file)

        print(f"✅ 训练数据加载完成:")
        print(f"  📏 样本数: {len(df):,}")
        print(f"  🔧 特征数: {len(df.columns)-3}")
        print(f"  🎯 正样本比例: {df['label'].mean():.3f}")

        return df

    def prepare_features(self, df):
        """准备特征数据"""
        print("\n🔧 准备特征数据...")

        # 分离特征和标签
        feature_cols = [col for col in df.columns if col not in ['user_id', 'item_id', 'label']]
        X = df[feature_cols]
        y = df['label']

        # 保存特征名
        self.feature_names = feature_cols

        print(f"  🔧 特征维度: {X.shape}")
        print(f"  🎯 标签分布: {y.value_counts().to_dict()}")

        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            print(f"  ⚠️  有缺失值的列: {len(missing_cols)}")
            # 简单填充缺失值
            X = X.fillna(0)
            print(f"  🔧 已用0填充缺失值")
        else:
            print(f"  ✅ 无缺失值")

        # 检查无穷值
        inf_cols = X.columns[np.isinf(X).any()].tolist()
        if inf_cols:
            print(f"  ⚠️  有无穷值的列: {len(inf_cols)}")
            X = X.replace([np.inf, -np.inf], 0)
            print(f"  🔧 已用0替换无穷值")

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """划分训练集和验证集"""
        print(f"\n📊 划分数据集 (验证集比例: {test_size})...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"  📈 训练集: {len(X_train):,} 样本, 正样本率: {y_train.mean():.3f}")
        print(f"  📊 验证集: {len(X_val):,} 样本, 正样本率: {y_val.mean():.3f}")

        return X_train, X_val, y_train, y_val

    def train_model(self, X_train, y_train, X_val, y_val):
        """训练LightGBM模型"""
        print("\n🚀 开始训练LightGBM模型...")

        # LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        print(f"📋 模型参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # 准备数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)

        # 训练模型
        print(f"\n🔄 开始训练...")
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        print(f"✅ 模型训练完成!")
        print(f"  🌳 最佳轮数: {self.model.best_iteration}")

        return self.model

    def evaluate_model(self, X_val, y_val):
        """评估模型"""
        print("\n📊 模型评估...")

        # 预测
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"  📈 AUC: {auc:.4f}")

        # 找最佳阈值
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]

        print(f"  🎯 最佳阈值: {best_threshold:.4f}")
        print(f"  📊 最佳F1: {best_f1:.4f}")

        # 使用最佳阈值重新预测
        y_pred_best = (y_pred_proba > best_threshold).astype(int)
        precision_best = np.sum((y_pred_best == 1) & (y_val == 1)) / np.sum(y_pred_best == 1)
        recall_best = np.sum((y_pred_best == 1) & (y_val == 1)) / np.sum(y_val == 1)

        print(f"  ✅ 最佳阈值下:")
        print(f"    Precision: {precision_best:.4f}")
        print(f"    Recall: {recall_best:.4f}")

        return {
            'auc': auc,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'precision': precision_best,
            'recall': recall_best
        }

    def analyze_feature_importance(self, top_n=20):
        """分析特征重要性"""
        print(f"\n🔍 特征重要性分析 (Top {top_n})...")

        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"📊 Top {top_n} 重要特征:")
        for i, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        # 按特征类型分组分析
        user_features = feature_importance[feature_importance['feature'].str.startswith('user_')]
        item_features = feature_importance[feature_importance['feature'].str.startswith('item_')]
        interaction_features = feature_importance[feature_importance['feature'].str.startswith('interaction_')]

        print(f"\n📋 特征类型重要性:")
        print(f"  👥 用户特征平均重要性: {user_features['importance'].mean():.2f}")
        print(f"  📦 商品特征平均重要性: {item_features['importance'].mean():.2f}")
        print(f"  🔗 交互特征平均重要性: {interaction_features['importance'].mean():.2f}")

        return feature_importance

    def save_model(self, metrics):
        """保存模型和相关文件"""
        print("\n💾 保存模型...")

        # 保存模型
        model_file = os.path.join(self.model_dir, "lightgbm_model.pkl")
        joblib.dump(self.model, model_file)
        print(f"  📦 模型已保存: {model_file}")

        # 保存特征名
        feature_file = os.path.join(self.model_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, feature_file)
        print(f"  📝 特征名已保存: {feature_file}")

        # 保存评估指标
        metrics_file = os.path.join(self.model_dir, "model_metrics.pkl")
        joblib.dump(metrics, metrics_file)
        print(f"  📊 指标已保存: {metrics_file}")

        # 保存模型信息
        model_info = {
            'model_type': 'LightGBM',
            'feature_count': len(self.feature_names),
            'best_iteration': self.model.best_iteration,
            'auc': metrics['auc'],
            'best_f1': metrics['best_f1'],
            'best_threshold': metrics['best_threshold']
        }

        info_file = os.path.join(self.model_dir, "model_info.txt")
        with open(info_file, 'w') as f:
            f.write("=== LightGBM模型信息 ===\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        print(f"  📋 模型信息已保存: {info_file}")

        print(f"✅ 模型保存完成!")

    def load_model(self):
        """加载已训练的模型"""
        print("📂 加载已训练模型...")

        model_file = os.path.join(self.model_dir, "lightgbm_model.pkl")
        feature_file = os.path.join(self.model_dir, "feature_names.pkl")

        if os.path.exists(model_file) and os.path.exists(feature_file):
            self.model = joblib.load(model_file)
            self.feature_names = joblib.load(feature_file)
            print(f"✅ 模型加载成功")
            return True
        else:
            print(f"❌ 模型文件不存在")
            return False

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练或加载")

        return self.model.predict(X, num_iteration=self.model.best_iteration)


def main():
    """主函数"""
    print("=== LightGBM模型训练器 ===")
    print("🎯 目标：训练用户-商品交互预测模型")

    trainer = LightGBMTrainer()

    # 1. 加载训练数据
    df = trainer.load_training_data()
    if df is None:
        return

    # 2. 准备特征
    X, y = trainer.prepare_features(df)

    # 3. 划分数据
    X_train, X_val, y_train, y_val = trainer.split_data(X, y)

    # 4. 训练模型
    model = trainer.train_model(X_train, y_train, X_val, y_val)

    # 5. 评估模型
    metrics = trainer.evaluate_model(X_val, y_val)

    # 6. 特征重要性分析
    feature_importance = trainer.analyze_feature_importance()

    # 7. 保存模型
    trainer.save_model(metrics)

    print(f"\n🎉 模型训练完成!")
    print(f"📈 最终AUC: {metrics['auc']:.4f}")
    print(f"📊 最佳F1: {metrics['best_f1']:.4f}")
    print(f"📁 模型已保存到: {trainer.model_dir}")
    print(f"下一步：生成最终推荐")


if __name__ == "__main__":
    main()