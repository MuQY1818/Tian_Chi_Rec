#!/usr/bin/env python3
"""
候选集生成脚本（流式/分片版）

目标：从用户行为数据 D 中，针对商品子集 P，构造训练与预测候选集。

改进点：
- 不再一次性加载 Part A/B 的全量数据，改为读取 dataset/daily_data 下的日分片文件
- 多路召回：最近 7 天任意交互 + 最近 14 天强信号（收藏/加购） + 标签日（2014-12-18）购买全覆盖
- 生成训练候选（用于 2014-12-18 标签监督）与预测候选（用于 2014-12-19 提交）

输出：train_candidates.csv, pred_candidates.csv
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from tqdm import tqdm


def _check_conda_env():
    """确保运行在 kaggle_env 环境，避免误用 base 环境。"""
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "kaggle_env":
        raise SystemExit(
            f"当前环境为 '{env}'，请激活 conda 环境 'kaggle_env' 后再运行。"
        )


def load_item_subset():
    """加载商品子集P（仅所需列，类型压缩）。"""
    print("加载商品子集P...")
    item_df = pd.read_csv(
        "dataset/tianchi_fresh_comp_train_item_online.txt",
        sep="\t",
        header=None,
        names=["item_id", "item_geohash", "item_category"],
        usecols=[0, 2],
        dtype={"item_id": np.int64, "item_category": np.int64},
    )
    print(f"商品子集P大小: {len(item_df)}")
    return set(item_df["item_id"]), item_df


def _iter_daily_files(daily_dir: str):
    """遍历日分片文件，返回 (path, 日期date对象)。文件名格式：data_YYYYMMDD.txt"""
    for fname in sorted(os.listdir(daily_dir)):
        if not fname.startswith("data_") or not fname.endswith(".txt"):
            continue
        ymd = fname[5:13]
        try:
            d = datetime.strptime(ymd, "%Y%m%d").date()
        except Exception:
            continue
        yield os.path.join(daily_dir, fname), d


def _read_daily(path: str):
    """读取单日日志（无时间列），字段：user_id, item_id, behavior_type, user_geohash, item_category"""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "behavior_type", "user_geohash", "item_category"],
        dtype={
            "user_id": np.int64,
            "item_id": np.int64,
            "behavior_type": np.uint8,
            "user_geohash": "object",
            "item_category": np.int64,
        },
    )
    return df


def _in_range(d: date, start: date, end: date) -> bool:
    return (d >= start) and (d <= end)


def generate_training_and_prediction_candidates(
    daily_dir: str,
    target_items: set,
    max_per_user: int = 200,
    enable_hot_backfill: bool = False,
):
    """生成训练与预测候选集（流式）。

    策略（训练/预测各自独立）：
    - 最近7天任意交互召回
    - 最近14天强信号（收藏/加购）召回
    - 标签日（2014-12-18）购买全覆盖（训练）
    - 可选：热门回填（默认关闭，避免爆炸）
    """

    label_date = date(2014, 12, 18)
    pred_date = date(2014, 12, 19)

    # 时间窗口
    train_recent_win = (label_date - timedelta(days=7), label_date - timedelta(days=1))  # 12-11~12-17
    train_strong_win = (label_date - timedelta(days=14), label_date - timedelta(days=1))  # 12-04~12-17
    pred_recent_win = (pred_date - timedelta(days=7), pred_date - timedelta(days=1))  # 12-12~12-18
    pred_strong_win = (pred_date - timedelta(days=14), pred_date - timedelta(days=1))  # 12-05~12-18

    # 容器
    train_recent_pairs = set()
    train_strong_pairs = set()
    train_label_pairs = set()
    pred_recent_pairs = set()
    pred_strong_pairs = set()

    # 可选的热门统计
    hot_counter_7d_train = defaultdict(int)
    hot_counter_7d_pred = defaultdict(int)

    for path, d in tqdm(list(_iter_daily_files(daily_dir)), desc="扫描日分片"):
        df = _read_daily(path)

        # 仅保留商品子集 P
        df = df[df["item_id"].isin(target_items)]
        if df.empty:
            continue

        # 训练窗口召回
        if _in_range(d, *train_recent_win):
            train_recent_pairs.update(zip(df["user_id"], df["item_id"]))
        if _in_range(d, *train_strong_win):
            strong_mask = df["behavior_type"].isin([2, 3])
            if strong_mask.any():
                tmp = df.loc[strong_mask, ["user_id", "item_id"]]
                train_strong_pairs.update(zip(tmp["user_id"], tmp["item_id"]))

        # 标签日正样本
        if d == label_date:
            buy_mask = df["behavior_type"] == 4
            if buy_mask.any():
                tmp = df.loc[buy_mask, ["user_id", "item_id"]]
                train_label_pairs.update(zip(tmp["user_id"], tmp["item_id"]))

        # 预测窗口召回
        if _in_range(d, *pred_recent_win):
            pred_recent_pairs.update(zip(df["user_id"], df["item_id"]))
        if _in_range(d, *pred_strong_win):
            strong_mask = df["behavior_type"].isin([2, 3])
            if strong_mask.any():
                tmp = df.loc[strong_mask, ["user_id", "item_id"]]
                pred_strong_pairs.update(zip(tmp["user_id"], tmp["item_id"]))

        # 热度统计（可选）
        if enable_hot_backfill:
            if _in_range(d, *train_recent_win):
                # 全局热度（7天）
                vc = df["item_id"].value_counts()
                for iid, c in vc.items():
                    hot_counter_7d_train[iid] += int(c)
            if _in_range(d, *pred_recent_win):
                vc = df["item_id"].value_counts()
                for iid, c in vc.items():
                    hot_counter_7d_pred[iid] += int(c)

    # 组装训练候选，保证正样本优先纳入
    train_candidates = defaultdict(set)

    def _add_pairs(target: dict, pairs: set, cap: int):
        for u, i in pairs:
            s = target[u]
            if len(s) < cap:
                s.add(i)

    # 先加正样本，再强信号，再最近交互
    _add_pairs(train_candidates, train_label_pairs, max_per_user)
    _add_pairs(train_candidates, train_strong_pairs, max_per_user)
    _add_pairs(train_candidates, train_recent_pairs, max_per_user)

    # 训练候选 DataFrame + 标签
    train_rows = []
    label_pairs = train_label_pairs
    for u, items in train_candidates.items():
        for i in items:
            train_rows.append((u, i, 1 if (u, i) in label_pairs else 0))
    train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id", "label"])

    # 组装预测候选：强信号优先，再最近交互
    pred_candidates = defaultdict(set)
    _add_pairs(pred_candidates, pred_strong_pairs, max_per_user)
    _add_pairs(pred_candidates, pred_recent_pairs, max_per_user)

    # 可选：热门回填（关闭默认避免爆炸）
    if enable_hot_backfill and len(pred_candidates) > 0:
        hot_top = [k for k, _ in sorted(hot_counter_7d_pred.items(), key=lambda x: -x[1])[:50]]
        for u, items in pred_candidates.items():
            if len(items) >= max_per_user:
                continue
            for iid in hot_top:
                if len(items) >= max_per_user:
                    break
                if iid not in items:
                    items.add(iid)

    pred_rows = []
    for u, items in pred_candidates.items():
        for i in items:
            pred_rows.append((u, i))
    pred_df = pd.DataFrame(pred_rows, columns=["user_id", "item_id"])

    # 日志
    pos = int(train_df["label"].sum())
    neg = len(train_df) - pos
    ratio = (neg / pos) if pos > 0 else np.inf
    print(f"训练候选: {len(train_df)}，正样本: {pos}，负样本: {neg}，正负比 1:{ratio:.2f}")
    print(f"预测候选: {len(pred_df)}")

    return train_df, pred_df


def save_candidates(train_samples, pred_candidates):
    """保存候选集"""
    print("保存候选集...")

    # 保存训练样本
    train_samples.to_csv("train_candidates.csv", index=False)
    print(f"训练样本保存完成: {len(train_samples)} 条")

    # 保存预测候选集
    pred_candidates.to_csv("pred_candidates.csv", index=False)
    print(f"预测候选集保存完成: {len(pred_candidates)} 条")


def main():
    print("=" * 50)
    print("候选集生成 - 天池推荐算法竞赛")
    print("=" * 50)

    _check_conda_env()

    # 1. 加载商品子集P
    target_items, _ = load_item_subset()

    # 2. 生成候选（日分片流式）
    daily_dir = "dataset/daily_data"
    train_samples, pred_candidates = generate_training_and_prediction_candidates(
        daily_dir, target_items, max_per_user=200, enable_hot_backfill=False
    )

    # 3. 保存结果
    save_candidates(train_samples, pred_candidates)

    print("\n候选集生成完成！")
    print(f"训练样本: train_candidates.csv ({len(train_samples)} 条)")
    print(f"预测候选集: pred_candidates.csv ({len(pred_candidates)} 条)")


if __name__ == "__main__":
    main()
