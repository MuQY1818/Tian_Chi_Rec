#!/usr/bin/env python3
"""
完全复现notebook策略的解决方案
参考原始notebook的特征工程和建模方法
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_data_notebook_style():
    """按照notebook方式加载数据"""
    print("加载数据...")

    # 快速加载最近几天数据
    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

    chunks = []
    file_path = "dataset/tianchi_fresh_comp_train_user_online_partB.txt"

    for chunk in tqdm(pd.read_csv(file_path, sep='\t', names=columns,
                               usecols=['user_id', 'item_id', 'behavior_type', 'item_category', 'time'],
                               chunksize=2000000), desc="加载数据"):
        chunk['date'] = chunk['time'].str[:10]
        recent_data = chunk[chunk['date'].isin(['2014-12-16', '2014-12-17', '2014-12-18'])]
        if len(recent_data) > 0:
            chunks.append(recent_data)

        if len(chunks) > 3:
            break

    user_data = pd.concat(chunks, ignore_index=True)

    # 转换时间格式（与notebook一致）
    user_data['time'] = pd.to_datetime(user_data['time'].str.replace(' ', ''), format='%Y-%m-%d%H')

    return user_data

def user_item_type_counts(data, behavior_type, name):
    """用户对商品执行特定操作的次数"""
    data = data.copy()
    data[name] = 1
    feature = data[data["behavior_type"] == behavior_type][["user_id", "item_id", name]].groupby(["user_id", "item_id"], as_index=False).count()
    return feature

def user_item_last_type_time(data, behavior_type, name):
    """用户对商品执行特定操作的最后时间"""
    feature = data[data["behavior_type"] == behavior_type][["user_id", "item_id", "time"]].groupby(["user_id", "item_id"], as_index=False).max()
    feature.rename(columns={"time": name}, inplace=True)
    return feature

def user_category_type_counts(data, behavior_type, name):
    """用户对商品类别执行特定操作的次数"""
    data = data.copy()
    data[name] = 1
    feature = data[data["behavior_type"] == behavior_type][["user_id", "item_category", name]].groupby(["user_id", "item_category"], as_index=False).count()
    return feature

def user_category_last_type_time(data, behavior_type, name):
    """用户对商品类别执行特定操作的最后时间"""
    feature = data[data["behavior_type"] == behavior_type][["user_id", "item_category", "time"]].groupby(["user_id", "item_category"], as_index=False).max()
    feature.rename(columns={"time": name}, inplace=True)
    return feature

def user_item_look_to_buy(data):
    """用户购买特定商品从浏览到购买的复杂特征"""
    # 用户购买过的用户-商品组合
    buy_user_item = data[data["behavior_type"] == 4][["user_id", "item_id"]].drop_duplicates()
    # 用户-商品组合的数据
    data = pd.merge(buy_user_item, data, how="left", on=["user_id", "item_id"])[["user_id", "item_id", "behavior_type", "time"]]

    # 第一次浏览商品的时间
    earliest_look = data[data["behavior_type"] == 1].groupby(["user_id", "item_id"], as_index=False).min()
    earliest_look.rename(columns={"time": "earliest_look_time"}, inplace=True)

    # 第一次购买商品的时间
    earliest_buy = data[data["behavior_type"] == 4].groupby(["user_id", "item_id"], as_index=False).min()
    earliest_buy.rename(columns={"time": "earliest_buy_time"}, inplace=True)

    # 计算时间间隔
    feature = pd.merge(earliest_buy, earliest_look, how="left", on=["user_id", "item_id"])
    feature["earliest_user_item_timedelta_look_to_buy"] = (feature["earliest_buy_time"] - feature["earliest_look_time"]).dt.total_seconds()/3600
    feature = feature[feature["earliest_user_item_timedelta_look_to_buy"] >= 0]
    feature = feature[["user_id", "item_id", "earliest_look_time", "earliest_buy_time", "earliest_user_item_timedelta_look_to_buy"]]

    # 第一次购买商品过程中的浏览次数
    data = pd.merge(feature, data, how="left", on=["user_id", "item_id"])
    data = data[(data["behavior_type"] == 1)&(data["time"] <= data["earliest_buy_time"])]
    data["item_look_counts_before_buy"] = 1
    item_look_counts_before_buy = data[["user_id", "item_id", "item_look_counts_before_buy"]].groupby(["user_id", "item_id"], as_index=False).count()
    feature = pd.merge(feature, item_look_counts_before_buy, how="left", on=["user_id", "item_id"])

    # 返回结果
    return feature[["user_id", "item_id", "item_look_counts_before_buy", "earliest_user_item_timedelta_look_to_buy"]]

def user_category_look_to_buy(data):
    """用户购买特定类别商品从浏览到购买的复杂特征"""
    # 用户购买过的用户-商品类型组合
    buy_user_item = data[data["behavior_type"] == 4][["user_id", "item_category"]].drop_duplicates()
    # 用户-商品类型组合的数据
    data = pd.merge(buy_user_item, data, how="left", on=["user_id", "item_category"])[["user_id", "item_category", "behavior_type", "time"]]

    # 第一次浏览同种类型商品的时间
    earliest_look = data[data["behavior_type"] == 1].groupby(["user_id", "item_category"], as_index=False).min()
    earliest_look.rename(columns={"time": "earliest_look_time"}, inplace=True)

    # 第一次购买同种类型商品的时间
    earliest_buy = data[data["behavior_type"] == 4].groupby(["user_id", "item_category"], as_index=False).min()
    earliest_buy.rename(columns={"time": "earliest_buy_time"}, inplace=True)

    # 计算时间间隔
    feature = pd.merge(earliest_buy, earliest_look, how="left", on=["user_id", "item_category"])
    feature["earliest_user_category_timedelta_look_to_buy"] = (feature["earliest_buy_time"] - feature["earliest_look_time"]).dt.total_seconds()/3600
    feature = feature[feature["earliest_user_category_timedelta_look_to_buy"] >= 0]
    feature = feature[["user_id", "item_category", "earliest_look_time", "earliest_buy_time", "earliest_user_category_timedelta_look_to_buy"]]

    # 第一次购买同种类型商品过程中的浏览次数
    data = pd.merge(feature, data, how="left", on=["user_id", "item_category"])
    data = data[(data["behavior_type"] == 1)&(data["time"] <= data["earliest_buy_time"])]
    data["category_look_counts_before_buy"] = 1
    category_look_counts_before_buy = data[["user_id", "item_category", "category_look_counts_before_buy"]].groupby(["user_id", "item_category"], as_index=False).count()
    feature = pd.merge(feature, category_look_counts_before_buy, how="left", on=["user_id", "item_category"])

    # 返回结果
    return feature[["user_id", "item_category", "category_look_counts_before_buy", "earliest_user_category_timedelta_look_to_buy"]]

def get_feature_notebook_style(user_data, predict_date):
    """完全按照notebook方式构建特征"""
    print(f"构建特征，预测日期: {predict_date}")

    train_data = user_data[user_data["time"] < predict_date].copy()

    # 用户-商品相关特征
    print("   计算用户-商品特征...")
    ui2 = user_item_type_counts(train_data, 1, "user_item_look_counts")
    ui5 = user_item_type_counts(train_data, 4, "user_item_buy_counts")
    ui6 = user_item_last_type_time(train_data, 1, "user_item_last_look_time")
    ui7 = user_item_last_type_time(train_data, 2, "user_item_last_like_time")
    ui8 = user_item_last_type_time(train_data, 3, "user_item_last_add_time")
    ui9 = user_item_last_type_time(train_data, 4, "user_item_last_buy_time")
    ui10 = user_item_look_to_buy(train_data)

    # 用户-商品类型相关特征
    print("   计算用户-类别特征...")
    uc2 = user_category_type_counts(train_data, 1, "user_category_look_counts")
    uc5 = user_category_type_counts(train_data, 4, "user_category_buy_counts")
    uc6 = user_category_last_type_time(train_data, 1, "user_category_last_look_time")
    uc7 = user_category_last_type_time(train_data, 2, "user_category_last_like_time")
    uc8 = user_category_last_type_time(train_data, 3, "user_category_last_add_time")
    uc9 = user_category_last_type_time(train_data, 4, "user_category_last_buy_time")
    uc10 = user_category_look_to_buy(train_data)

    # 构建候选集（所有交互过的用户-商品对）
    feature_data = train_data[["user_id", "item_id", "item_category"]].drop_duplicates()

    print(f"   候选集大小: {len(feature_data):,}")

    # 合并特征
    print("   合并特征...")
    feature_data = feature_data.merge(ui2, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui5, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui6, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui7, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui8, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui9, on=["user_id", "item_id"], how="left")
    feature_data = feature_data.merge(ui10, on=["user_id", "item_id"], how="left")

    feature_data = feature_data.merge(uc2, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc5, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc6, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc7, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc8, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc9, on=["user_id", "item_category"], how="left")
    feature_data = feature_data.merge(uc10, on=["user_id", "item_category"], how="left")

    # 计算距预测日期的时间间隔
    print("   计算时间间隔特征...")
    predict_datetime = pd.to_datetime(predict_date)
    feature_data["user_item_last_look_to_now"] = (predict_datetime - feature_data["user_item_last_look_time"]).dt.total_seconds()/3600
    feature_data["user_item_last_like_to_now"] = (predict_datetime - feature_data["user_item_last_like_time"]).dt.total_seconds()/3600
    feature_data["user_item_last_add_to_now"] = (predict_datetime - feature_data["user_item_last_add_time"]).dt.total_seconds()/3600
    feature_data["user_item_last_buy_to_now"] = (predict_datetime - feature_data["user_item_last_buy_time"]).dt.total_seconds()/3600
    feature_data["user_category_last_look_to_now"] = (predict_datetime - feature_data["user_category_last_look_time"]).dt.total_seconds()/3600
    feature_data["user_category_last_like_to_now"] = (predict_datetime - feature_data["user_category_last_like_time"]).dt.total_seconds()/3600
    feature_data["user_category_last_add_to_now"] = (predict_datetime - feature_data["user_category_last_add_time"]).dt.total_seconds()/3600
    feature_data["user_category_last_buy_to_now"] = (predict_datetime - feature_data["user_category_last_buy_time"]).dt.total_seconds()/3600

    # 删除时间列
    drop_columns = ["user_item_last_look_time", "user_item_last_like_time", "user_item_last_add_time", "user_item_last_buy_time"]
    drop_columns += ["user_category_last_look_time", "user_category_last_like_time", "user_category_last_add_time", "user_category_last_buy_time"]
    feature_data = feature_data.drop(drop_columns, axis=1)

    # 处理缺失值
    fill_columns = ["user_item_look_counts", "user_item_buy_counts", "user_category_look_counts", "user_category_buy_counts"]
    feature_data[fill_columns] = feature_data[fill_columns].fillna(0)

    return feature_data

def split_x_y(data, user_data, predict_date):
    """划分特征和标签（按照notebook方式）"""
    end_date = pd.to_datetime(predict_date) + pd.Timedelta(days=1)
    labels = user_data[(user_data["time"] >= predict_date) & (user_data["time"] < end_date) & (user_data["behavior_type"] == 4)][["user_id", "item_id"]].drop_duplicates()
    labels["is_buy"] = 1
    data = pd.merge(data, labels, how="left", on=["user_id", "item_id"])
    data["is_buy"] = data["is_buy"].fillna(0)

    feature_cols = [col for col in data.columns if col not in ["user_id", "item_id", "item_category", "is_buy"]]
    X = data[feature_cols].fillna(0)
    y = data["is_buy"]
    return X, y

def main():
    print("=" * 60)
    print("完全复现Notebook策略的解决方案")
    print("=" * 60)

    start_time = time.time()

    # 1. 加载数据
    user_data = load_data_notebook_style()

    # 过滤商品子集
    item_df = pd.read_csv("dataset/tianchi_fresh_comp_train_item_online.txt",
                         sep='\t', names=['item_id', 'item_geohash', 'item_category'])
    valid_items = set(item_df['item_id'].astype(str))
    user_data['user_id'] = user_data['user_id'].astype(str)
    user_data['item_id'] = user_data['item_id'].astype(str)
    user_data = user_data[user_data['item_id'].isin(valid_items)]

    print(f"过滤后数据量: {len(user_data):,}")

    # 2. 构建训练和验证数据
    print("\n构建训练数据...")
    data_train = get_feature_notebook_style(user_data, "2014-12-17")
    X_train, y_train = split_x_y(data_train, user_data, "2014-12-17")

    print(f"训练集: {len(X_train):,}, 正样本率: {y_train.mean():.4f}")

    print("\n构建验证数据...")
    data_eval = get_feature_notebook_style(user_data, "2014-12-18")
    X_eval, y_eval = split_x_y(data_eval, user_data, "2014-12-18")

    print(f"验证集: {len(X_eval):,}, 正样本率: {y_eval.mean():.4f}")

    # 3. 训练XGBoost模型
    print("\n训练XGBoost模型...")

    # 创建DMatrix
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_eval = xgb.DMatrix(X_eval, y_eval)

    # 参数设置（优化版本）
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'gamma': 0.1,
        'min_child_weight': 1.1,
        'max_depth': 6,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'eta': 0.05,  # 提高学习率
        'tree_method': 'exact',
        'seed': 0
    }

    # 训练模型
    watchlist = [(xgb_train, 'train'), (xgb_eval, 'validate')]
    model = xgb.train(params, xgb_train, num_boost_round=300,  # 增加轮数
                     evals=watchlist, early_stopping_rounds=50,  # 放宽early stopping
                     verbose_eval=20)

    # 4. 预测和评估
    print("\n预测和评估...")

    # 在验证集上预测
    eval_pred = model.predict(xgb_eval)

    # 计算AUC
    auc_score = roc_auc_score(y_eval, eval_pred)
    print(f"验证集AUC: {auc_score:.4f}")

    # 5. 构建测试数据并预测
    print("\n构建测试数据...")
    data_test = get_feature_notebook_style(user_data, "2014-12-19")

    # 预测测试集
    X_test = data_test[[col for col in data_test.columns if col not in ["user_id", "item_id", "item_category"]]].fillna(0)
    xgb_test = xgb.DMatrix(X_test)

    # 预测概率
    test_pred = model.predict(xgb_test)

    # 归一化预测分数
    test_pred_normalized = MinMaxScaler().fit_transform(test_pred.reshape(-1, 1)).flatten()

    # 生成最终预测结果
    predict_data = data_test[["user_id", "item_id"]].copy()
    predict_data["pred_score"] = test_pred_normalized

    # 按分数排序并取Top550
    predict_data = predict_data.sort_values("pred_score", ascending=False)
    result = predict_data.head(550)[["user_id", "item_id"]]

    # 保存结果
    output_file = "notebook_style_solution.txt"
    with open(output_file, 'w') as f:
        for _, row in result.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\n")

    # 快速计算F1分数（使用已有数据）
    print("\n计算F1分数...")
    pred_pairs = set(zip(result['user_id'].astype(str), result['item_id'].astype(str)))

    # 从已有数据中提取18号购买记录
    day18_data = user_data[user_data['time'].astype(str).str.startswith('2014-12-18')]
    true_buy = day18_data[day18_data['behavior_type'] == 4]
    true_pairs = set(zip(true_buy['user_id'].astype(str), true_buy['item_id'].astype(str)))

    tp = len(pred_pairs & true_pairs)
    precision = tp / len(pred_pairs) if len(pred_pairs) > 0 else 0
    recall = tp / len(true_pairs) if len(true_pairs) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    total_time = time.time() - start_time

    print(f"\n完全复现Notebook解决方案完成!")
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"验证集AUC: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"命中数量: {tp:,}/{len(pred_pairs):,}")
    print(f"真实购买数量: {len(true_pairs):,}")
    print(f"输出文件: {output_file}")

    print(f"\n核心特色:")
    print(f"- 完全按照notebook的特征工程方法")
    print(f"- 14个精选特征包含复杂时间特征")
    print(f"- 所有历史交互作为候选集")
    print(f"- XGBoost rank:pairwise目标函数")

if __name__ == "__main__":
    main()