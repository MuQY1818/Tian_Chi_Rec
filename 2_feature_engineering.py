#!/usr/bin/env python3
"""
特征工程脚本（流式/分片+向量化）
基于候选集构建用户、商品、交互与用户-品类特征，避免一次性载入全量数据。
"""

import os
import time
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict
from tqdm import tqdm


def _check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "kaggle_env":
        raise SystemExit(
            f"当前环境为 '{env}'，请激活 conda 环境 'kaggle_env' 后再运行。"
        )


def _load_config_from_ini(path: str = "config/config.ini"):
    if not os.path.exists(path):
        return
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "training" in cfg:
        sec = cfg["training"]
        if "label_date" in sec:
            os.environ.setdefault("LABEL_DATE", sec["label_date"]) 


def _iter_daily_files(daily_dir: str):
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


def _load_item_subset():
    item_df = pd.read_csv(
        "dataset/tianchi_fresh_comp_train_item_online.txt",
        sep="\t",
        header=None,
        names=["item_id", "item_geohash", "item_category"],
        usecols=[0, 2],
        dtype={"item_id": np.int64, "item_category": np.int64},
    )
    return item_df


def _build_features_for(candidates: pd.DataFrame, reference_date: date) -> pd.DataFrame:
    daily_dir = "dataset/daily_data"
    item_df = _load_item_subset()

    # 候选集合（限制聚合范围）
    cand_users = set(candidates["user_id"].unique().tolist())
    cand_items = set(candidates["item_id"].unique().tolist())
    target_items = set(item_df["item_id"].unique().tolist())

    # 需要的 (user, category) 组合（仅候选对的类别）
    cand_with_cat = candidates.merge(item_df, on="item_id", how="left")
    needed_uc = cand_with_cat[["user_id", "item_category"]].drop_duplicates()

    # 用户特征容器
    user_total = defaultdict(int)
    user_bt = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
    user_active_days = defaultdict(int)
    user_last_day = {}
    user_recent7 = defaultdict(int)
    user_recent3 = defaultdict(int)
    user_unique_cats = defaultdict(set)

    # 商品特征容器（仅候选商品 ∩ P）
    item_total = defaultdict(int)
    item_views = defaultdict(int)
    item_purchases = defaultdict(int)
    item_unique_users = defaultdict(set)
    item_last_day = {}
    item_recent7_views = defaultdict(int)
    item_recent3_views = defaultdict(int)

    # 交互特征容器（仅候选对）
    ui_total = defaultdict(int)
    ui_bt = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
    ui_first_day = {}
    ui_last_day = {}

    # 用户-品类（仅需要的组合）
    uc_actions = defaultdict(int)
    uc_items = defaultdict(set)

    recent7_start = reference_date - timedelta(days=7)
    recent3_start = reference_date - timedelta(days=3)

    files = list(_iter_daily_files(daily_dir))
    print(
        f"准备聚合: 候选用户 {len(cand_users)}, 候选商品 {len(cand_items)}, 参考日 {reference_date}, 分片数 {len(files)}"
    )
    pbar = tqdm(files, desc="聚合日分片", total=len(files), dynamic_ncols=True)

    total_rows = 0
    total_kept = 0
    total_pairs = 0

    for path, d in pbar:
        t0 = time.perf_counter()
        if d >= reference_date:
            continue
        df = _read_daily(path)

        rows_orig = len(df)

        # 过滤到 P 且候选用户
        df = df[df["item_id"].isin(target_items)]
        if df.empty:
            pbar.set_postfix(day=str(d), rows=rows_orig, kept=0, pairs=0, sec=f"{time.perf_counter()-t0:.1f}")
            continue
        df = df[df["user_id"].isin(cand_users)]
        if df.empty:
            pbar.set_postfix(day=str(d), rows=rows_orig, kept=0, pairs=0, sec=f"{time.perf_counter()-t0:.1f}")
            continue

        kept_after_filters = len(df)

        # 用户层聚合
        # 当日活跃用户（用于 active_days）
        active_users = df["user_id"].drop_duplicates().values.tolist()
        for u in active_users:
            user_active_days[u] += 1
            # 更新最后活跃日
            prev = user_last_day.get(u)
            user_last_day[u] = d if (prev is None or d > prev) else prev

        # 行为类型按用户计数（一次 pivot 避免多次 groupby）
        bt_user = (
            df.pivot_table(index="user_id", columns="behavior_type", aggfunc="size", fill_value=0)
            .astype(np.int64)
        )
        # 总行为数
        user_total_update = bt_user.sum(axis=1)
        for u, c in user_total_update.items():
            user_total[int(u)] += int(c)
        for bt in [1, 2, 3, 4]:
            if bt in bt_user.columns:
                for u, c in bt_user[bt].items():
                    user_bt[bt][int(u)] += int(c)

        # 近 7/3 天用户行为
        if d >= recent7_start:
            for u, c in user_total_update.items():
                user_recent7[int(u)] += int(c)
        if d >= recent3_start:
            for u, c in user_total_update.items():
                user_recent3[int(u)] += int(c)

        # 用户唯一类别
        # 每个用户当日出现的类别集合（减少集合膨胀开销）
        tmp = df[["user_id", "item_category"]].drop_duplicates()
        for u, cat in tmp.itertuples(index=False):
            user_unique_cats[int(u)].add(int(cat))

        # 商品层（仅候选商品 ∩ P）
        df_i = df[df["item_id"].isin(cand_items)]
        if not df_i.empty:
            # 基础计数
            vc_item = df_i["item_id"].value_counts()
            for i, c in vc_item.items():
                item_total[int(i)] += int(c)
                # 更新最后出现日
                prev = item_last_day.get(int(i))
                item_last_day[int(i)] = d if (prev is None or d > prev) else prev
            # 视图与购买
            views = df_i[df_i["behavior_type"] == 1]["item_id"].value_counts()
            buys = df_i[df_i["behavior_type"] == 4]["item_id"].value_counts()
            for i, c in views.items():
                item_views[int(i)] += int(c)
            for i, c in buys.items():
                item_purchases[int(i)] += int(c)
            # 近7/3天视图
            if d >= recent7_start:
                for i, c in views.items():
                    item_recent7_views[int(i)] += int(c)
            if d >= recent3_start:
                for i, c in views.items():
                    item_recent3_views[int(i)] += int(c)
            # 唯一用户集合（精确，限定候选 item）
            tmp_u = df_i[["item_id", "user_id"]].drop_duplicates()
            for i, u in tmp_u.itertuples(index=False):
                item_unique_users[int(i)].add(int(u))

        # 交互层（仅候选对）
        df_ui = df[df["item_id"].isin(cand_items)]
        day_pairs = 0
        if not df_ui.empty:
            # 总次数
            vc_ui = df_ui.groupby(["user_id", "item_id"]).size()
            day_pairs = int(vc_ui.shape[0])
            for (u, i), c in vc_ui.items():
                key = (int(u), int(i))
                ui_total[key] += int(c)
                # first/last day
                first = ui_first_day.get(key)
                ui_first_day[key] = d if (first is None or d < first) else first
                last = ui_last_day.get(key)
                ui_last_day[key] = d if (last is None or d > last) else last

            # 行为类型细分
            bt_ui = (
                df_ui.pivot_table(
                    index=["user_id", "item_id"], columns="behavior_type", aggfunc="size", fill_value=0
                ).astype(np.int64)
            )
            for bt in [1, 2, 3, 4]:
                if bt in bt_ui.columns:
                    for (u, i), c in bt_ui[bt].items():
                        ui_bt[bt][(int(u), int(i))] += int(c)

        # 进度统计与展示
        total_rows += rows_orig
        total_kept += kept_after_filters
        total_pairs += day_pairs
        pbar.set_postfix(
            day=str(d), rows=rows_orig, kept=kept_after_filters, pairs=day_pairs, sec=f"{time.perf_counter()-t0:.1f}"
        )

        # 用户-品类（仅需要的 UC 组合）
        # 半连接过滤：
        df_uc = df.merge(needed_uc, on=["user_id", "item_category"], how="inner")
        if not df_uc.empty:
            # 动作次数
            vc_uc = df_uc.groupby(["user_id", "item_category"]).size()
            for (u, cat), c in vc_uc.items():
                uc_actions[(int(u), int(cat))] += int(c)
            # item 去重计数（使用集合累计，最后取大小）
            tmp_uc_items = df_uc[["user_id", "item_category", "item_id"]].drop_duplicates()
            for u, cat, i in tmp_uc_items.itertuples(index=False):
                uc_items[(int(u), int(cat))].add(int(i))

    t_build_start = time.perf_counter()

    # 组装用户特征表
    t0 = time.perf_counter()
    users = list(cand_users)
    uf = pd.DataFrame({"user_id": users})
    uf["total_actions"] = uf["user_id"].map(lambda u: user_total.get(int(u), 0))
    for bt in [1, 2, 3, 4]:
        uf[f"behavior_{bt}_count"] = uf["user_id"].map(lambda u, b=bt: user_bt[b].get(int(u), 0))
    uf["purchase_rate"] = uf.apply(
        lambda r: (r["behavior_4_count"] / r["total_actions"]) if r["total_actions"] > 0 else 0, axis=1
    )
    uf["active_days"] = uf["user_id"].map(lambda u: user_active_days.get(int(u), 0))
    uf["avg_actions_per_day"] = uf.apply(
        lambda r: (r["total_actions"] / r["active_days"]) if r["active_days"] > 0 else 0, axis=1
    )
    uf["recent_7d_actions"] = uf["user_id"].map(lambda u: user_recent7.get(int(u), 0))
    uf["recent_3d_actions"] = uf["user_id"].map(lambda u: user_recent3.get(int(u), 0))
    uf["days_since_last_action"] = uf["user_id"].map(
        lambda u: (reference_date - user_last_day[int(u)]).days if int(u) in user_last_day else 999
    )
    uf["unique_categories"] = uf["user_id"].map(lambda u: len(user_unique_cats.get(int(u), set())))
    print(f"用户特征组装完成: {uf.shape}, 用时 {time.perf_counter()-t0:.1f}s")

    # 组装商品特征表
    t0 = time.perf_counter()
    items = list(cand_items)
    itf = pd.DataFrame({"item_id": items})
    itf["total_actions"] = itf["item_id"].map(lambda i: item_total.get(int(i), 0))
    itf["total_purchases"] = itf["item_id"].map(lambda i: item_purchases.get(int(i), 0))
    itf["unique_users"] = itf["item_id"].map(lambda i: len(item_unique_users.get(int(i), set())))
    itf["views"] = itf["item_id"].map(lambda i: item_views.get(int(i), 0))
    itf["conversion_rate"] = itf.apply(
        lambda r: (r["total_purchases"] / r["views"]) if r["views"] > 0 else 0, axis=1
    )
    itf["avg_actions_per_user"] = itf.apply(
        lambda r: (r["total_actions"] / r["unique_users"]) if r["unique_users"] > 0 else 0, axis=1
    )
    itf["recent_7d_views"] = itf["item_id"].map(lambda i: item_recent7_views.get(int(i), 0))
    itf["recent_3d_views"] = itf["item_id"].map(lambda i: item_recent3_views.get(int(i), 0))
    itf["days_since_last_view"] = itf["item_id"].map(
        lambda i: (reference_date - item_last_day[int(i)]).days if int(i) in item_last_day else 999
    )
    # 合并类别
    itf = itf.merge(_load_item_subset(), on="item_id", how="left")
    print(f"商品特征组装完成: {itf.shape}, 用时 {time.perf_counter()-t0:.1f}s")

    # 组装交互特征表
    t0 = time.perf_counter()
    ui_keys = candidates[["user_id", "item_id"]].drop_duplicates().copy()
    ui_keys["_ui_key"] = list(zip(ui_keys["user_id"].astype(int), ui_keys["item_id"].astype(int)))
    # 字典映射（向量化）
    ui_keys["ui_total_actions"] = ui_keys["_ui_key"].map(ui_total).fillna(0).astype(np.int64)
    for bt in [1, 2, 3, 4]:
        col = f"ui_{['','view','collect','cart','purchase'][bt]}_count"
        ui_keys[col] = ui_keys["_ui_key"].map(ui_bt[bt]).fillna(0).astype(np.int64)
    # 最近一次与首次
    last_series = ui_keys["_ui_key"].map(ui_last_day)
    first_series = ui_keys["_ui_key"].map(ui_first_day)
    # 天数差
    ui_keys["ui_days_since_last_action"] = last_series.map(
        lambda d: (reference_date - d).days if pd.notnull(d) else 999
    ).astype(np.int64)
    span_days = []
    for f, l in zip(first_series.tolist(), last_series.tolist()):
        if pd.notnull(f) and pd.notnull(l):
            span_days.append((l - f).days + 1)
        else:
            span_days.append(0)
    ui_keys["ui_action_span_days"] = np.array(span_days, dtype=np.int64)
    ui_keys["ui_avg_actions_per_day"] = ui_keys.apply(
        lambda r: (r.ui_total_actions / r.ui_action_span_days) if r.ui_action_span_days > 0 else 0, axis=1
    )
    ui_keys = ui_keys.drop(columns=["_ui_key"])  # 清理临时列
    print(f"交互特征组装完成: {ui_keys.shape}, 用时 {time.perf_counter()-t0:.1f}s")

    # 用户-品类偏好（按候选对所属类别）
    t0 = time.perf_counter()
    cand_uc = cand_with_cat[["user_id", "item_category"]].drop_duplicates().copy()
    cand_uc["_uc_key"] = list(zip(cand_uc["user_id"].astype(int), cand_uc["item_category"].astype(int)))
    cand_uc["user_category_actions"] = cand_uc["_uc_key"].map(uc_actions).fillna(0).astype(np.int64)
    cand_uc["user_category_items"] = cand_uc["_uc_key"].map(lambda k: len(uc_items.get(k, set())))
    cand_uc = cand_uc.drop(columns=["_uc_key"])  # 清理临时列
    print(f"用户-品类特征组装完成: {cand_uc.shape}, 用时 {time.perf_counter()-t0:.1f}s")

    # 合并特征
    t0 = time.perf_counter()
    features = candidates.copy()
    features = features.merge(uf, on="user_id", how="left")
    print(f"合并用户特征完成: {features.shape}")
    features = features.merge(itf, on="item_id", how="left")
    print(f"合并商品特征完成: {features.shape}")
    features = features.merge(ui_keys, on=["user_id", "item_id"], how="left")
    print(f"合并交互特征完成: {features.shape}")
    features = features.merge(cand_uc, on=["user_id", "item_category"], how="left")
    print(f"合并用户-品类特征完成: {features.shape}; 合并总用时 {time.perf_counter()-t0:.1f}s")

    # 缺失填充
    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].fillna(0)

    print(f"最终特征维度: {features.shape[1]}，总组装用时 {time.perf_counter()-t_build_start:.1f}s")
    return features


def build_features(candidates: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    print(f"构建{'训练' if is_training else '预测'}特征...")
    # 允许通过配置覆盖标签日
    label_env = os.environ.get("LABEL_DATE")
    if label_env:
        ld = datetime.strptime(label_env, "%Y%m%d").date()
    else:
        ld = date(2014, 12, 18)
    ref = ld if is_training else (ld + timedelta(days=1))
    return _build_features_for(candidates, ref)


def main():
    print("=" * 50)
    print("特征工程 - 天池推荐算法竞赛")
    print("=" * 50)

    _check_conda_env()
    _load_config_from_ini()

    # 1. 构建训练特征
    print("\n构建训练特征...")
    train_candidates = pd.read_csv("train_candidates.csv")
    train_features = build_features(train_candidates, is_training=True)
    train_features.to_csv("train_features.csv", index=False)
    print(f"训练特征保存: {len(train_features)} 条")

    # 2. 构建预测特征
    print("\n构建预测特征...")
    pred_candidates = pd.read_csv("pred_candidates.csv")
    pred_features = build_features(pred_candidates, is_training=False)
    pred_features.to_csv("pred_features.csv", index=False)
    print(f"预测特征保存: {len(pred_features)} 条")

    print("\n特征工程完成！")


if __name__ == "__main__":
    main()
