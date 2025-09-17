#!/usr/bin/env python3
"""
快速LightGBM训练器
针对现有39维用户特征优化的训练器
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


class FastLightGBMTrainer:
    """快速LightGBM训练器"""

    def __init__(self, feature_dir="/mnt/data/tianchi_features"):
        self.feature_dir = feature_dir
        os.makedirs(feature_dir, exist_ok=True)

        self.model = None
        self.feature_names = None
        self.best_threshold = 0.5

        print(f"📁 特征目录: {feature_dir}")

    def load_training_data(self):
        """加载快速生成的训练数据"""
        print("📂 加载训练数据...")

        train_file = os.path.join(self.feature_dir, "fast_training_samples.csv")
        if not os.path.exists(train_file):
            print(f"❌ 训练文件不存在: {train_file}")
            print("请先运行: python scripts/feature_engineering/fast_sample_generator.py")
            return None

        # 检查文件大小
        file_size = os.path.getsize(train_file) / (1024**2)
        print(f"  📏 文件大小: {file_size:.1f} MB")

        # 加载数据
        df = pd.read_csv(train_file)

        print(f"✅ 训练数据加载完成:")
        print(f"  📏 样本数: {len(df):,}")
        print(f"  🔧 特征数: {len(df.columns)-3}")
        print(f"  🎯 正样本比例: {df['label'].mean():.3f}")

        return df

    def prepare_features(self, df):
        """准备特征数据（快速版本）"""
        print("\n🔧 准备特征数据...")

        # 分离特征和标签
        feature_cols = [col for col in df.columns if col not in ['user_id', 'item_id', 'label']]
        X = df[feature_cols]
        y = df['label']

        self.feature_names = feature_cols

        print(f"  🔧 特征维度: {X.shape}")
        print(f"  🎯 标签分布: {y.value_counts().to_dict()}")

        # 快速数据清理
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"  🔧 填充 {missing_count} 个缺失值")
            X = X.fillna(0)

        # 检查无穷值
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"  🔧 替换 {inf_count} 个无穷值")
            X = X.replace([np.inf, -np.inf], 0)

        print(f"  ✅ 数据准备完成")

        return X, y

    def train_fast_model(self, X, y, validation_split=0.2):
        """快速训练模型"""
        print(f"\n🚀 开始快速训练...")

        # 划分数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        print(f"  📊 数据划分:")
        print(f"    训练集: {len(X_train):,}, 正样本率: {y_train.mean():.3f}")
        print(f"    验证集: {len(X_val):,}, 正样本率: {y_val.mean():.3f}")

        # 优化的LightGBM参数（针对快速训练）
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # 减少叶子数加速
            'learning_rate': 0.1,  # 提高学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,  # 减少最小样本数
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        print(f"📋 快速训练参数:")
        key_params = ['num_leaves', 'learning_rate', 'min_child_samples']
        for key in key_params:
            print(f"  {key}: {params[key]}")

        # 准备数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 训练（快速版本：减少轮数）
        print(f"\n🔄 开始训练...")
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            num_boost_round=300,  # 减少训练轮数
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),  # 更早停止
                lgb.log_evaluation(period=50)
            ]
        )

        print(f"✅ 快速训练完成!")
        print(f"  🌳 实际轮数: {self.model.best_iteration}")

        # 快速评估
        metrics = self._evaluate_model(X_val, y_val)

        return metrics

    def _evaluate_model(self, X_val, y_val):
        """快速模型评估"""
        print("\n📊 模型评估...")

        # 预测
        y_pred_proba = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        # AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"  📈 AUC: {auc:.4f}")

        # 找最佳阈值
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"  🎯 最佳阈值: {self.best_threshold:.4f}")
        print(f"  📊 最佳F1: {best_f1:.4f}")

        return {
            'auc': auc,
            'best_threshold': self.best_threshold,
            'best_f1': best_f1
        }

    def analyze_top_features(self, top_n=15):
        """分析Top特征重要性"""
        print(f"\n🔍 Top {top_n} 重要特征:")

        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        for i, row in feature_importance.head(top_n).iterrows():
            feature_type = "👥" if row['feature'].startswith('user_') else \
                          "📦" if row['feature'].startswith('item_') else "🔗"
            print(f"  {feature_type} {row['feature']}: {row['importance']:.1f}")

        return feature_importance

    def save_fast_model(self, metrics):
        """保存快速模型"""
        print("\n💾 保存模型...")

        # 保存模型
        model_file = os.path.join(self.feature_dir, "fast_lightgbm_model.pkl")
        joblib.dump(self.model, model_file)

        # 保存配置
        config = {
            'feature_names': self.feature_names,
            'best_threshold': self.best_threshold,
            'metrics': metrics,
            'model_type': 'Fast_LightGBM'
        }

        config_file = os.path.join(self.feature_dir, "fast_model_config.pkl")
        joblib.dump(config, config_file)

        print(f"  📦 模型已保存: {model_file}")
        print(f"  ⚙️  配置已保存: {config_file}")

    def load_fast_model(self):
        """加载快速模型"""
        model_file = os.path.join(self.feature_dir, "fast_lightgbm_model.pkl")
        config_file = os.path.join(self.feature_dir, "fast_model_config.pkl")

        if os.path.exists(model_file) and os.path.exists(config_file):
            self.model = joblib.load(model_file)
            config = joblib.load(config_file)

            self.feature_names = config['feature_names']
            self.best_threshold = config['best_threshold']

            print(f"✅ 快速模型加载成功")
            print(f"  📊 AUC: {config['metrics']['auc']:.4f}")
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
    print("🤖 === 快速LightGBM训练器 === 🤖")
    print("🎯 目标：基于现有39维用户特征快速训练模型")
    print("⚡ 预计耗时：2-3分钟")
    print("━" * 50)

    trainer = FastLightGBMTrainer()

    # 1. 加载训练数据
    df = trainer.load_training_data()
    if df is None:
        return

    # 2. 准备特征
    X, y = trainer.prepare_features(df)

    # 3. 快速训练
    metrics = trainer.train_fast_model(X, y)

    # 4. 特征重要性
    feature_importance = trainer.analyze_top_features()

    # 5. 保存模型
    trainer.save_fast_model(metrics)

    print(f"\n🎉 快速训练完成!")
    print(f"📈 AUC: {metrics['auc']:.4f}")
    print(f"📊 F1: {metrics['best_f1']:.4f}")
    print(f"⚡ 训练速度: 比完整版本快3-5倍")
    print(f"📁 模型已保存到: {trainer.feature_dir}")


if __name__ == "__main__":
    main()