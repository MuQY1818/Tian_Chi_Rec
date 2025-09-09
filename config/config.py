# 数据处理配置
DATA_CONFIG = {
    # 数据路径
    "data_path": "dataset",
    "raw_data_path": "data/raw",
    "processed_data_path": "data/processed",
    "models_path": "data/models",
    
    # 数据采样配置
    "sample_ratio": 1.0,  # 采样比例 - 1.0表示全量数据
    "sample_size": None,  # 采样大小 - None表示不限制
    
    # 行为权重配置
    "behavior_weights": {
        1: 1,  # 浏览
        2: 2,  # 收藏
        3: 3,  # 加购物车
        4: 4   # 购买
    },
    
    # 时间配置
    "predict_date": "2014-12-19",
    "train_end_date": "2014-12-15",
    "validation_end_date": "2014-12-18",
    
    # 时间衰减参数
    "time_decay_lambda": 0.1,
    
    # 时间窗口配置
    "time_windows": [3, 7, 14],  # 天数
    
    # 负采样配置
    "negative_ratio": 3,  # 负样本与正样本比例
}

# 特征工程配置
FEATURE_CONFIG = {
    # 用户行为特征
    "user_behavior_features": {
        "enabled": True,
        "include_weighted": True,
        "include_temporal": True
    },
    
    # 时间特征
    "time_features": {
        "enabled": True,
        "include_decay": True,
        "include_windows": True,
        "include_periodic": True
    },
    
    # 商品特征
    "item_features": {
        "enabled": True,
        "include_popularity": True,
        "include_category": True,
        "include_temporal": True
    },
    
    # 交互特征
    "interaction_features": {
        "enabled": True,
        "include_strength": True,
        "include_temporal": True
    },
    
    # 协同过滤特征
    "cf_features": {
        "enabled": False,  # 计算量较大，默认关闭
        "top_k": 10,
        "similarity_threshold": 0.1
    }
}

# 模型配置
MODEL_CONFIG = {
    # 逻辑回归配置
    "logistic_regression": {
        "enabled": True,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
        "C": 1.0
    },
    
    # 随机森林配置
    "random_forest": {
        "enabled": True,
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    },
    
    # XGBoost配置
    "xgboost": {
        "enabled": True,
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    },
    
    # GRU4Rec配置
    "gru4rec": {
        "enabled": True,
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "max_seq_len": 50,
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 50,
        "negative_sampling": True,
        "attention": True
    },
    
    # DeepFM配置
    "deepfm": {
        "enabled": True,
        "embedding_dim": 64,
        "hidden_units": [128, 64, 32],
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 50
    },
    
    # ItemCF配置
    "itemcf": {
        "enabled": True,
        "top_k": 10,
        "similarity_metric": "cosine"
    }
}

# 训练配置
TRAINING_CONFIG = {
    # 数据集划分
    "test_size": 0.2,
    "random_state": 42,
    
    # 交叉验证
    "cv_folds": 5,
    "scoring": "f1",
    
    # 超参数调优
    "hyperparameter_tuning": {
        "enabled": False,  # 计算量较大，默认关闭
        "cv": 3,
        "n_jobs": -1,
        "verbose": 1
    },
    
    # 早停配置
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "monitor": "val_f1"
    }
}

# 推荐配置
RECOMMENDATION_CONFIG = {
    # 推荐数量
    "top_n": 5,
    
    # 推荐策略
    "strategies": {
        "popularity": {
            "enabled": True,
            "top_k": 20
        },
        "user_history": {
            "enabled": True,
            "lookback_days": 30
        },
        "category_based": {
            "enabled": True,
            "top_categories": 10
        }
    },
    
    # 过滤规则
    "filters": {
        "remove_purchased": True,
        "remove_interacted": True,
        "category_diversity": True,
        "max_same_category": 2
    },
    
    # 输出格式
    "output_format": {
        "columns": ["user_id", "item_id"],
        "separator": "\t",
        "header": False
    }
}

# 评估配置
EVALUATION_CONFIG = {
    # 评估指标
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "ndcg@5"
    ],
    
    # 评估阈值
    "thresholds": [0.3, 0.5, 0.7],
    
    # 报告配置
    "report": {
        "save_plots": True,
        "save_metrics": True,
        "save_predictions": True
    }
}

# 系统配置
SYSTEM_CONFIG = {
    # 日志配置
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/recommendation_system.log"
    },
    
    # 内存配置
    "memory": {
        "optimize_dtypes": True,
        "chunk_size": 100000,
        "garbage_collection": True
    },
    
    # 并行配置
    "parallel": {
        "n_jobs": -1,
        "backend": "multiprocessing"
    },
    
    # 缓存配置
    "cache": {
        "enabled": True,
        "directory": "cache",
        "max_size": "10GB"
    },
    
    # 大规模数据处理配置
    "large_scale": {
        "enabled": True,
        "dask_workers": 4,
        "dask_memory_limit": "16GB",
        "block_size": "256MB",
        "chunk_size": 100000,
        "use_gpu": True,
        "gpu_memory_limit": "8GB"
    }
}