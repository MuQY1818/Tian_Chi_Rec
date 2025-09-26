#!/usr/bin/env python3
"""
LightGBM 训练与预测（5 折 CV + 特征筛选）
基于特征数据训练模型，生成最终推荐结果。
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import configparser

warnings.filterwarnings("ignore")


def _check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "kaggle_env":
        raise SystemExit(
            f"当前环境为 '{env}'，请激活 conda 环境 'kaggle_env' 后再运行。"
        )


def _load_config_from_ini(path: str = "config/config.ini"):
    """从 INI 加载配置，写入环境变量，便于沿用现有读取逻辑。
    - 仅在环境变量没有显式设置时才覆盖（setdefault）。
    - 支持的 section: training/strategy/eval/viz。
    """
    if not os.path.exists(path):
        return
    cfg = configparser.ConfigParser()
    cfg.read(path)

    def set_env(mapping, section):
        if section not in cfg:
            return
        for key, env_key in mapping.items():
            if key in cfg[section]:
                val = cfg[section][key]
                os.environ.setdefault(env_key, str(val))

    set_env({
        "downsample_neg": "DOWNSAMPLE_NEG",
        "neg_pos_ratio": "NEG_POS_RATIO",
        "neg_max_gap_days": "NEG_MAX_GAP_DAYS",
        "use_group_kfold": "USE_GROUP_KFOLD",
        "fast_params": "FAST_PARAMS",
        "n_splits": "N_SPLITS",
        "early_stopping": "EARLY_STOPPING",
        "log_eval_every": "LOG_EVAL_EVERY",
    }, "training")

    set_env({
        "force_strategy": "FORCE_STRATEGY",
        "strong_only": "STRONG_ONLY",
        "min_prob": "MIN_PROB",
        "topn": "TOPN",
        "max_submission": "MAX_SUBMISSION",
        "exclude_day_purchases": "EXCLUDE_DAY_PURCHASES",
        "topk_k": "TOPK_K",
        "topk_m": "TOPK_M",
        "cart_days_max": "CART_DAYS_MAX",
        "threshold": "THRESHOLD",
    }, "strategy")

    set_env({
        "eval_strategy": "EVAL_STRATEGY",
        "eval_sample_user_frac": "EVAL_SAMPLE_USER_FRAC",
        "val_full": "VAL_FULL",
        "val_batch": "VAL_BATCH",
    }, "eval")

    set_env({
        "feature_viz": "FEATURE_VIZ",
        "viz_only": "VIZ_ONLY",
        "viz_sample_per_class": "VIZ_SAMPLE_PER_CLASS",
        "viz_methods": "VIZ_METHODS",
        "viz_out": "VIZ_OUT",
        "tsne_learning_rate": "TSNE_LEARNING_RATE",
        "tsne_n_iter": "TSNE_N_ITER",
    }, "viz")


def load_features():
    print("加载特征数据...")
    train_features = pd.read_csv("train_features.csv")
    pred_features = pd.read_csv("pred_features.csv")
    print(f"训练数据: {len(train_features)} 条")
    print(f"预测数据: {len(pred_features)} 条")
    return train_features, pred_features


def prepare_training_data(train_features):
    print("准备训练数据...")
    feature_columns = [c for c in train_features.columns if c not in ["user_id", "item_id", "label"]]
    X = train_features[feature_columns]
    y = train_features["label"].astype(int)
    print(f"特征数量: {len(feature_columns)}")
    print(f"正样本比例: {y.mean():.4f}")
    return X, y, feature_columns


def downsample_negatives(
    df: pd.DataFrame,
    neg_pos_ratio: int = 20,
    by_user: bool = False,
    random_state: int = 42,
    neg_max_gap_days: int = None,
) -> pd.DataFrame:
    """对训练特征做负样本下采样，加速训练。保留全部正样本，负样本按比例采样。
    - by_user=False：全局采样，简单高效
    - by_user=True：按用户采样，控制每个用户的负样本数（更均匀，稍慢）
    """
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    if neg_max_gap_days is not None and "ui_days_since_last_action" in neg.columns:
        before = len(neg)
        neg = neg[neg["ui_days_since_last_action"] <= int(neg_max_gap_days)]
        print(f"负采样过滤: ui_days_since_last_action <= {neg_max_gap_days}: {before}->{len(neg)}")
    if len(pos) == 0 or len(neg) == 0:
        return df
    target_neg = min(len(neg), int(len(pos) * neg_pos_ratio))
    if not by_user:
        neg_s = neg.sample(n=target_neg, random_state=random_state)
    else:
        rng = np.random.RandomState(random_state)
        parts = []
        # 每用户的负样本采样数量按其占比近似分配
        user_counts = neg["user_id"].value_counts()
        for uid, cnt in user_counts.items():
            take = int(np.round(cnt / len(neg) * target_neg))
            if take <= 0:
                continue
            g = neg[neg["user_id"] == uid]
            parts.append(g.sample(n=min(take, len(g)), random_state=rng.randint(1e9)))
        if parts:
            neg_s = pd.concat(parts, axis=0, ignore_index=True)
            if len(neg_s) > target_neg:
                neg_s = neg_s.sample(n=target_neg, random_state=random_state)
        else:
            neg_s = neg.sample(n=target_neg, random_state=random_state)
    ds = pd.concat([pos, neg_s], axis=0, ignore_index=True).sample(frac=1.0, random_state=random_state)
    print(f"负采样: pos={len(pos)}, neg={len(neg)} -> neg_sample={len(neg_s)}, 合计={len(ds)}，比例 1:{len(neg_s)/len(pos):.2f}")
    return ds


def kfold_train_predict(X, y, pred_features, feature_columns, n_splits=5, random_state=42, groups=None, early_stopping_rounds=100, fast_params=False, class_weight="balanced"):
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )

    if fast_params:
        params.update({
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "n_estimators": 1500,
        })

    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
        print("使用 GroupKFold 按 user_id 分组切分")
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
        print("使用 StratifiedKFold 分层切分")
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(pred_features))
    fold_importances = []
    folds_auc = []

    log_every = int(os.environ.get("LOG_EVAL_EVERY", "20"))

    models = []

    for fold, (trn_idx, val_idx) in enumerate(split_iter, 1):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_trn,
            y_trn,
            eval_set=[(X_trn, y_trn), (X_val, y_val)],
            eval_names=["train", "val"],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(log_every)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        folds_auc.append(auc)
        print(f"Fold {fold} AUC: {auc:.4f}, best_iter: {model.best_iteration_}")

        # 测试集预测累计平均
        test_pred += model.predict_proba(pred_features[feature_columns])[:, 1] / n_splits
        models.append(model)

        # 特征重要性
        imp = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.booster_.feature_importance(importance_type="gain"),
            "fold": fold,
        })
        fold_importances.append(imp)

    oof_auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {oof_auc:.4f}; Folds AUC mean: {np.mean(folds_auc):.4f}")

    fi = pd.concat(fold_importances, axis=0)
    fi_group = fi.groupby("feature")["importance"].sum().sort_values(ascending=False).reset_index()

    return test_pred, oof, fi_group, models


def drop_zero_importance_features(fi_group, feature_columns):
    zero_feats = set(fi_group[fi_group["importance"] <= 0]["feature"].tolist())
    kept = [f for f in feature_columns if f not in zero_feats]
    dropped = [f for f in feature_columns if f in zero_feats]
    return kept, dropped


def select_final_recommendations(results, top_k_per_user=5):
    print("选择最终推荐...")
    # 按用户分组 top-K
    # 为效率，先按 user, prob 排序后 groupby().head(K)
    results = results.sort_values(["user_id", "pred_prob"], ascending=[True, False])
    recs = results.groupby("user_id").head(top_k_per_user)[["user_id", "item_id"]]
    print(f"最终推荐数量: {len(recs)}")
    print(f"覆盖用户数: {recs['user_id'].nunique()}")
    return recs


def save_submission(recommendations):
    print("保存提交文件...")
    # 去重保护再写出
    rec = recommendations.drop_duplicates()
    with open("submission.txt", "w") as f:
        for _, row in rec.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\n")
    print("提交文件已保存: submission.txt")
    print(f"提交条数: {len(rec)}")


def _predict_full_in_chunks(models: list, df: pd.DataFrame, feature_columns: list, batch: int = 1000000) -> np.ndarray:
    """用多折模型对 df 做分块预测，并按折平均。"""
    n = len(df)
    out = np.zeros(n, dtype=np.float32)
    idx = np.arange(n)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        Xb = df.iloc[start:end][feature_columns]
        pb = np.zeros(len(Xb), dtype=np.float32)
        for m in models:
            pb += m.predict_proba(Xb)[:, 1] / max(1, len(models))
        out[start:end] = pb
    return out


def build_strong_mask(df: pd.DataFrame) -> pd.Series:
    """强信号规则掩码，用于二阶段重排。
    规则：购物车>0 或 收藏>0 或 近3天活跃>0 或 最近间隔<=3天 或 历史购买>0。
    """
    cols_needed = [
        "ui_cart_count",
        "ui_collect_count",
        "recent_3d_actions",
        "ui_days_since_last_action",
        "ui_purchase_count",
    ]
    for c in cols_needed:
        if c not in df.columns:
            # 不存在该列则视为0/无
            df[c] = 0
    mask = (
        (df["ui_cart_count"].astype(int) > 0)
        | (df["ui_collect_count"].astype(int) > 0)
        | (df["recent_3d_actions"].astype(int) > 0)
        | (df["ui_days_since_last_action"].astype(int) <= 3)
        | (df["ui_purchase_count"].astype(int) > 0)
    )
    return mask


def select_final_recommendations_two_stage(results: pd.DataFrame, strong_pairs: pd.DataFrame, top_k_per_user: int, strong_first: int):
    """二阶段重排：
    1) 每用户优先从强信号集合里取 strong_first 个（按 pred_prob 降序）
    2) 再从非强信号集合按 pred_prob 降序补齐到 top_k_per_user
    """
    df = results.copy()
    # 与强信号对进行连接，构造 is_strong 标记（与 results 行对齐，避免长度不匹配）
    if strong_pairs is not None and len(strong_pairs) > 0:
        sp = strong_pairs.drop_duplicates().assign(is_strong=True)
        df = df.merge(sp, on=["user_id", "item_id"], how="left")
        df["is_strong"] = df["is_strong"].fillna(False).astype(bool)
    else:
        df["is_strong"] = False

    # 先按 user, is_strong, pred_prob 排序
    df = df.sort_values(["user_id", "is_strong", "pred_prob"], ascending=[True, False, False])
    # 每用户内去重保护
    df = df.drop_duplicates(subset=["user_id", "item_id"])

    recs = []
    for uid, g in df.groupby("user_id", sort=False):
        g_str = g[g["is_strong"]]
        g_weak = g[~g["is_strong"]]

        k1 = min(strong_first, len(g_str), top_k_per_user)
        part1 = g_str.head(k1)
        k2 = top_k_per_user - len(part1)
        if k2 > 0:
            part2 = g_weak.head(k2)
            sel = pd.concat([part1, part2], axis=0)
        else:
            sel = part1
        if not sel.empty:
            recs.append(sel[["user_id", "item_id"]])

    if len(recs) == 0:
        return pd.DataFrame(columns=["user_id", "item_id"])
    return pd.concat(recs, axis=0, ignore_index=True)


def select_final_recommendations_by_threshold(results: pd.DataFrame, threshold: float):
    """阈值法：全局概率阈值筛选。"""
    df = results[results["pred_prob"] >= threshold][["user_id", "item_id"]]
    return df.reset_index(drop=True)


def cap_submission_size(recommendations: pd.DataFrame, results_with_prob: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """全局提交数量上限：按 pred_prob 降序截断到 max_rows。"""
    if max_rows is None or max_rows <= 0:
        return recommendations
    n = len(recommendations)
    if n <= max_rows:
        return recommendations
    tmp = recommendations.merge(
        results_with_prob[["user_id", "item_id", "pred_prob"]], on=["user_id", "item_id"], how="left"
    )
    tmp["pred_prob"] = tmp["pred_prob"].fillna(0)
    tmp = tmp.sort_values("pred_prob", ascending=False).head(max_rows)
    print(f"应用全局上限: {n} -> {len(tmp)} (按概率降序截断)")
    return tmp[["user_id", "item_id"]].reset_index(drop=True)


def apply_pre_filters(results: pd.DataFrame, pred_features: pd.DataFrame, strong_mask: pd.Series) -> pd.DataFrame:
    """根据环境变量对候选结果进行预过滤：
    - STRONG_ONLY=1 只保留强信号样本
    - MIN_PROB=0.x 按概率阈值过滤
    """
    df = results
    if os.environ.get("STRONG_ONLY", "0") == "1":
        # 与 results 对齐：基于 (user_id,item_id) 内连接强信号子集
        strong_df = pred_features[["user_id", "item_id"]].copy()
        strong_df["is_strong"] = strong_mask.astype(bool).values
        strong_df = strong_df[strong_df["is_strong"]][["user_id", "item_id"]].drop_duplicates()
        before = len(df)
        df = df.merge(strong_df, on=["user_id", "item_id"], how="inner")
        print(f"预过滤: STRONG_ONLY 生效，{before}->{len(df)}")
    min_prob = os.environ.get("MIN_PROB")
    if min_prob is not None and len(min_prob) > 0:
        try:
            t = float(min_prob)
            before = len(df)
            df = df[df["pred_prob"] >= t]
            print(f"预过滤: MIN_PROB={t} 生效，{before}->{len(df)}")
        except Exception:
            pass
    return df


def select_global_topn(results: pd.DataFrame, topn: int) -> pd.DataFrame:
    """全局 Top-N：不分用户，按概率降序取前 topn。"""
    if topn <= 0:
        return results.copy()[["user_id", "item_id"]]
    df = results.sort_values("pred_prob", ascending=False).head(topn)
    print(f"全局 Top-N 选择: topn={topn}, 实际输出 {len(df)}")
    return df[["user_id", "item_id"]].reset_index(drop=True)


def select_cart_only(pred_features: pd.DataFrame, results: pd.DataFrame, max_rows: int = 100000) -> pd.DataFrame:
    """近似“仅购物车”策略：优先选择 12-18 最近一天内加购过的样本。
    由于特征是到参考日前一日的累计，这里近似用：ui_cart_count>0 且 ui_days_since_last_action<=1。
    若数量不足，则放宽到 <=2、<=3。
    最后按概率降序，截断到 max_rows。
    """
    days_max = int(os.environ.get("CART_DAYS_MAX", "3"))  # 允许的最大天数（默认放宽至3天）

    def _pick(days):
        # 在 pred_features 上选出满足条件的 (user_id,item_id)，再与 results 按键连接，保证长度匹配
        sub = pred_features[["user_id", "item_id", "ui_cart_count", "ui_days_since_last_action"]].copy()
        sub["ui_cart_count"] = sub["ui_cart_count"].fillna(0).astype(int)
        sub["ui_days_since_last_action"] = sub["ui_days_since_last_action"].fillna(999).astype(int)
        sub = sub[(sub["ui_cart_count"] > 0) & (sub["ui_days_since_last_action"] <= days)][["user_id", "item_id"]]
        df = results.merge(sub.drop_duplicates(), on=["user_id", "item_id"], how="inner")
        print(f"cart_only: days<= {days} 命中 {len(df)}")
        return df

    pool = _pick(1)
    # 仅在需要补量且 days_max>1 时逐步放宽
    for d in [2, 3]:
        if d > days_max:
            break
        if max_rows and len(pool) >= max_rows:
            break
        pool = pd.concat([pool, _pick(d)], axis=0).drop_duplicates(subset=["user_id", "item_id"])  # 放宽

    pool = pool.sort_values("pred_prob", ascending=False)
    if max_rows and len(pool) > max_rows:
        pool = pool.head(max_rows)
        print(f"cart_only: 截断到 {max_rows}")
    print(f"cart_only: 最终输出 {len(pool)}")
    return pool[["user_id", "item_id"]].reset_index(drop=True)


def exclude_purchases_on_day(recommendations: pd.DataFrame, day: str = "20141218") -> pd.DataFrame:
    """从提交中排除给定日期已经购买过的 (user_id,item_id)。默认排除 2014-12-18。
    day 形如 'YYYYMMDD'。若文件缺失则直接返回原结果。
    """
    daily_path = f"dataset/daily_data/data_{day}.txt"
    if not os.path.exists(daily_path):
        print(f"未找到 {daily_path}，跳过当日已购排除。")
        return recommendations
    try:
        df = pd.read_csv(
            daily_path,
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
        df_buy = df[df["behavior_type"] == 4][["user_id", "item_id"]].drop_duplicates()
        before = len(recommendations)
        rec = recommendations.merge(df_buy, on=["user_id", "item_id"], how="left", indicator=True)
        rec = rec[rec["_merge"] == "left_only"][ ["user_id", "item_id"] ]
        removed = before - len(rec)
        print(f"排除 {day} 当日已购: 移除 {removed} 条")
        return rec.reset_index(drop=True)
    except Exception as e:
        print(f"排除当日已购时出错: {e}. 跳过该步骤。")
        return recommendations


def _f1_from_counts(tp: int, pred_cnt: int, true_pos: int):
    if pred_cnt == 0 or true_pos == 0:
        return 0.0, 0.0, 0.0
    precision = tp / pred_cnt
    recall = tp / true_pos
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _eval_two_stage_on_sample(sample_df: pd.DataFrame, k: int, m: int):
    tp = 0
    pred_cnt = 0
    true_pos = int(sample_df["label"].sum())

    # 预排序：减少每组排序代价
    sample_df = sample_df.sort_values(["user_id", "pred_prob"], ascending=[True, False])
    for uid, g in sample_df.groupby("user_id", sort=False):
        gs = g[g["is_strong"]]
        gw = g[~g["is_strong"]]
        k1 = min(m, len(gs), k)
        sel1 = gs.head(k1)
        k2 = k - k1
        if k2 > 0:
            sel2 = gw.head(k2)
            sel = pd.concat([sel1, sel2], axis=0)
        else:
            sel = sel1
        pred_cnt += len(sel)
        if len(sel) > 0:
            tp += int(sel["label"].sum())

    return _f1_from_counts(tp, pred_cnt, true_pos)


def _eval_threshold_on_sample(sample_df: pd.DataFrame, thresh: float):
    s = sample_df["pred_prob"]
    y = sample_df["label"].astype(int)
    mask = s >= thresh
    pred_cnt = int(mask.sum())
    tp = int((y & mask.astype(int)).sum())
    true_pos = int(y.sum())
    return _f1_from_counts(tp, pred_cnt, true_pos)


def _eval_cart_only_on_sample(sample_df: pd.DataFrame, days_max: int = 1, min_prob: float = 0.0, max_submission: int = 0):
    """在训练样本上评估 cart_only 策略（基于 OOF 概率）。
    sample_df 需包含列：user_id,item_id,label,pred_prob,ui_cart_count,ui_days_since_last_action
    """
    df = sample_df.copy()
    df["ui_cart_count"] = df["ui_cart_count"].fillna(0).astype(int)
    df["ui_days_since_last_action"] = df["ui_days_since_last_action"].fillna(999).astype(int)
    sel = df[(df["ui_cart_count"] > 0) & (df["ui_days_since_last_action"] <= int(days_max))]
    if min_prob is not None and min_prob > 0:
        sel = sel[sel["pred_prob"] >= float(min_prob)]
    # 全局截断
    if max_submission and len(sel) > max_submission:
        sel = sel.sort_values("pred_prob", ascending=False).head(max_submission)
    tp = int(sel["label"].sum())
    pred_cnt = len(sel)
    true_pos = int(df["label"].sum())
    return _f1_from_counts(tp, pred_cnt, true_pos), pred_cnt


def evaluate_forced_strategy(train_df: pd.DataFrame, oof: np.ndarray, strategy: dict, sample_user_frac: float = 0.02,
                             min_prob: float = 0.0, max_submission: int = 0, strong_only: bool = False,
                             cart_days_max: int = 1):
    """在训练日(12-18)的 OOF 上离线评估当前强制策略，避免盲目提交。
    仅抽样部分用户以控制耗时。
    """
    rng = np.random.RandomState(42)
    df = train_df.copy().reset_index(drop=True)
    df["pred_prob"] = oof
    users = df["user_id"].drop_duplicates().values
    n_sample = min(len(users), max(20000, int(len(users) * sample_user_frac)))
    sample_users = set(rng.choice(users, size=n_sample, replace=False))
    sdf = df[df["user_id"].isin(sample_users)].copy()

    if strong_only:
        sdf["is_strong"] = build_strong_mask(sdf)
        sdf = sdf[sdf["is_strong"]].copy()

    name = strategy.get("name")
    f1 = p = r = 0.0
    selected = 0
    if name == "topk_two_stage":
        k, m = int(strategy["k"]), int(strategy["m"])
        # 应用 per-user Top-K，先强信号后弱信号；再按 min_prob 和全局截断
        sdf = sdf.sort_values(["user_id", "pred_prob"], ascending=[True, False])
        sdf["is_strong"] = build_strong_mask(sdf)
        tp = 0
        pred_cnt = 0
        true_pos = int(sdf["label"].sum())
        rec = []
        for uid, g in sdf.groupby("user_id", sort=False):
            gs = g[g["is_strong"]]
            gw = g[~g["is_strong"]]
            k1 = min(m, len(gs), k)
            sel1 = gs.head(k1)
            k2 = k - k1
            if k2 > 0:
                sel2 = gw.head(k2)
                sel = pd.concat([sel1, sel2], axis=0)
            else:
                sel = sel1
            if min_prob > 0:
                sel = sel[sel["pred_prob"] >= min_prob]
            if len(sel) > 0:
                rec.append(sel)
        if rec:
            rec = pd.concat(rec, axis=0)
            if max_submission and len(rec) > max_submission:
                rec = rec.sort_values("pred_prob", ascending=False).head(max_submission)
            pred_cnt = len(rec)
            tp = int(rec["label"].sum())
        f1, p, r = _f1_from_counts(tp, pred_cnt, true_pos)
        selected = pred_cnt
    elif name == "threshold":
        t = float(strategy.get("threshold", 0.5))
        f1, p, r = _eval_threshold_on_sample(sdf, t)
        selected = int((sdf["pred_prob"] >= t).sum())
    elif name == "global_topn":
        topn = int(os.environ.get("TOPN", os.environ.get("MAX_SUBMISSION", "100000")))
        sdf = sdf.sort_values("pred_prob", ascending=False)
        sel = sdf.head(topn)
        tp = int(sel["label"].sum())
        pred_cnt = len(sel)
        true_pos = int(sdf["label"].sum())
        f1, p, r = _f1_from_counts(tp, pred_cnt, true_pos)
        selected = pred_cnt
    elif name == "cart_only":
        (f1, p, r), pred_cnt = _eval_cart_only_on_sample(sdf, days_max=cart_days_max, min_prob=min_prob, max_submission=max_submission)
        selected = pred_cnt
    else:
        print(f"未知策略 {name}，跳过离线评估。")
        return
    print(f"离线评估(12-18, 抽样{len(sample_users)}用户): F1={f1:.4f}, P={p:.4f}, R={r:.4f}, 选中={selected}")


def offline_tune_and_choose_strategy(train_features: pd.DataFrame, oof: np.ndarray, sample_user_frac: float = 0.02, random_state: int = 42):
    """在训练集的 OOF 上离线调 F1：
    - Top-K 二阶段（K ∈ {3,5,8,10,12,15}, m ∈ {0,1,2,3,5}）
    - 全局阈值（在高分区间扫描）
    选择 F1 更优的策略，并返回策略字典。
    为了效率，仅抽样部分用户进行调参。
    """
    rng = np.random.RandomState(random_state)
    # 仅保留需要的列，避免内存峰值
    cols = [
        "user_id",
        "item_id",
        "label",
        "ui_cart_count",
        "ui_collect_count",
        "recent_3d_actions",
        "ui_days_since_last_action",
        "ui_purchase_count",
    ]
    missing = [c for c in cols if c not in train_features.columns]
    if missing:
        # 对缺失列补0，保证兼容
        for c in missing:
            if c not in train_features.columns:
                train_features[c] = 0
    df = train_features[cols].copy()
    df["pred_prob"] = oof

    # 按用户抽样
    users = df["user_id"].drop_duplicates().values
    n_sample = min(len(users), max(20000, int(len(users) * sample_user_frac)))
    sample_users = set(rng.choice(users, size=n_sample, replace=False))
    sdf = df[df["user_id"].isin(sample_users)].copy()

    # 构建强信号掩码
    sdf["is_strong"] = build_strong_mask(sdf).astype(bool)

    print(
        f"离线调参样本: users={len(sample_users)}, rows={len(sdf)}, pos={int(sdf['label'].sum())}, pos_ratio={sdf['label'].mean():.4f}"
    )

    if len(sdf) == 0:
        print("调参抽样为空，使用默认策略 Top-K(K=5,m=1)")
        return {"name": "topk_two_stage", "k": 5, "m": 1, "fallback": None}

    # 1) Top-K 二阶段网格
    K_list = [3, 5, 8, 10, 12, 15]
    M_list = [0, 1, 2, 3, 5]
    best_topk = {"f1": -1, "k": None, "m": None, "precision": 0, "recall": 0}
    for k in K_list:
        for m in M_list:
            if m > k:
                continue
            f1, p, r = _eval_two_stage_on_sample(sdf, k, m)
            if f1 > best_topk["f1"]:
                best_topk.update({"f1": f1, "k": k, "m": m, "precision": p, "recall": r})
    print(
        f"Top-K调参最优: K={best_topk['k']}, m={best_topk['m']}, F1={best_topk['f1']:.4f}, P={best_topk['precision']:.4f}, R={best_topk['recall']:.4f}"
    )

    # 2) 阈值扫描（在高分区间）
    scores = sdf["pred_prob"].values
    q_low = float(np.quantile(scores, 0.90))
    q_high = float(np.quantile(scores, 0.999))
    grid = np.linspace(q_low, q_high, num=25, endpoint=True)
    best_thr = {"f1": -1, "t": None, "precision": 0, "recall": 0, "q_level": None, "pred_cnt": 0}
    for t in grid:
        f1, p, r = _eval_threshold_on_sample(sdf, t)
        if f1 > best_thr["f1"]:
            q_level = float((scores >= t).mean())
            pred_cnt = int((sdf["pred_prob"] >= t).sum())
            best_thr.update({"f1": f1, "t": float(t), "precision": p, "recall": r, "q_level": q_level, "pred_cnt": pred_cnt})
    print(
        f"阈值调参最优: t={best_thr['t']:.6f}, F1={best_thr['f1']:.4f}, P={best_thr['precision']:.4f}, R={best_thr['recall']:.4f}"
    )

    # 3) 策略选择
    if best_topk["f1"] >= best_thr["f1"]:
        print("选择策略: Top-K 二阶段重排")
        return {
            "name": "topk_two_stage",
            "k": int(best_topk["k"]),
            "m": int(best_topk["m"]),
            "fallback": None,
        }
    else:
        print("选择策略: 概率阈值")
        return {
            "name": "threshold",
            "threshold": float(best_thr["t"]),
            "q_level": float(best_thr["q_level"]),
            "fallback": {"k": int(best_topk["k"]), "m": int(best_topk["m"])},
        }


def _adjust_threshold_by_quantile(pred_scores: np.ndarray, q_level: float, default_t: float) -> float:
    """将离线阈值按分位映射到预测集，避免分布偏移导致全空。
    q_level 为离线分布中“≥t”的占比；在预测集上取相同占比对应的阈值。
    """
    try:
        if q_level is None:
            return default_t
        q = max(1e-6, min(1 - 1e-6, 1 - q_level))
        return float(np.quantile(pred_scores, q))
    except Exception:
        return default_t


def _tsne_fit_transform(X: np.ndarray, perplexity: int, n_iter_env: int | None = None) -> np.ndarray:
    """兼容不同 sklearn 版本的 t-SNE 调用，自动探测可用参数并 fit_transform。"""
    from inspect import signature
    sig = signature(TSNE.__init__)
    kwargs = {
        "n_components": 2,
        "init": "pca",
        "perplexity": perplexity,
        "random_state": 42,
        "verbose": 1,
    }
    # learning_rate 可能不存在或不支持 'auto'，做兼容
    if "learning_rate" in sig.parameters:
        lr_str = os.environ.get("TSNE_LEARNING_RATE", "auto")
        try:
            lr_val = float(lr_str)
        except Exception:
            lr_val = lr_str  # 允许 'auto'
        kwargs["learning_rate"] = lr_val
    # n_iter 在某些版本不存在
    if "n_iter" in sig.parameters:
        niter = int(os.environ.get("TSNE_N_ITER", str(n_iter_env if n_iter_env is not None else 1000)))
        kwargs["n_iter"] = niter
    tsne = TSNE(**kwargs)
    return tsne.fit_transform(X)


def main():
    print("=" * 50)
    print("LightGBM训练和预测 - 天池推荐算法竞赛")
    print("=" * 50)

    _check_conda_env()

    # 0. 优先从 config/config.ini 加载配置，减少命令行环境变量书写
    _load_config_from_ini()

    # 1. 加载特征数据
    train_features, pred_features = load_features()

    # 可选：仅做特征可视化
    FEATURE_VIZ = os.environ.get("FEATURE_VIZ", "0") == "1"
    VIZ_ONLY = os.environ.get("VIZ_ONLY", "0") == "1"
    if FEATURE_VIZ or VIZ_ONLY:
        sample_per_class = int(os.environ.get("VIZ_SAMPLE_PER_CLASS", "10000"))
        methods = os.environ.get("VIZ_METHODS", "pca,tsne").split(",")
        out_dir = os.environ.get("VIZ_OUT", "viz")

        # 选取特征列
        feat_cols = [c for c in train_features.columns if c not in ["user_id", "item_id", "label"]]
        # 采样同量正负样本
        pos = train_features[train_features["label"] == 1]
        neg = train_features[train_features["label"] == 0]
        n_pos = min(sample_per_class, len(pos))
        n_neg = min(sample_per_class, len(neg))
        pos_s = pos.sample(n=n_pos, random_state=42)
        neg_s = neg.sample(n=n_neg, random_state=42)
        df_s = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=42)
        X = df_s[feat_cols].values
        y = df_s["label"].values

        # 标准化
        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        os.makedirs(out_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            # PCA 可视化
            if any(m.strip().lower() == "pca" for m in methods):
                p = PCA(n_components=2, random_state=42)
                emb = p.fit_transform(Xs)
                plt.figure(figsize=(7, 6))
                plt.scatter(emb[y==0,0], emb[y==0,1], s=4, c="#a6cee3", label="label=0", alpha=0.5)
                plt.scatter(emb[y==1,0], emb[y==1,1], s=6, c="#e31a1c", label="label=1", alpha=0.7)
                plt.legend(loc="best")
                plt.title(f"PCA-2D (n_pos={n_pos}, n_neg={n_neg})")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "feature_viz_pca.png"), dpi=150)
                plt.close()
                pd.DataFrame({"x": emb[:,0], "y": emb[:,1], "label": y}).to_csv(os.path.join(out_dir, "feature_viz_pca.csv"), index=False)

            # t-SNE 可视化
            if any(m.strip().lower() == "tsne" for m in methods):
                perpl = max(5, min(50, int((n_pos+n_neg)/200)))
                emb2 = _tsne_fit_transform(Xs, perplexity=perpl)
                plt.figure(figsize=(7, 6))
                plt.scatter(emb2[y==0,0], emb2[y==0,1], s=4, c="#a6cee3", label="label=0", alpha=0.5)
                plt.scatter(emb2[y==1,0], emb2[y==1,1], s=6, c="#e31a1c", label="label=1", alpha=0.7)
                plt.legend(loc="best")
                plt.title(f"t-SNE-2D (perplexity={perpl}, n_pos={n_pos}, n_neg={n_neg})")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "feature_viz_tsne.png"), dpi=150)
                plt.close()
                pd.DataFrame({"x": emb2[:,0], "y": emb2[:,1], "label": y}).to_csv(os.path.join(out_dir, "feature_viz_tsne.csv"), index=False)

            print(f"特征可视化已保存至目录: {out_dir}")
        except Exception as e:
            # 如果 matplotlib 不可用，则只输出 CSV 嵌入
            print(f"绘图失败或未安装 matplotlib，改为仅导出嵌入CSV: {e}")
            p = PCA(n_components=2, random_state=42)
            emb = p.fit_transform(Xs)
            pd.DataFrame({"x": emb[:,0], "y": emb[:,1], "label": y}).to_csv(os.path.join(out_dir, "feature_viz_pca.csv"), index=False)
            emb2 = _tsne_fit_transform(Xs, perplexity=30)
            pd.DataFrame({"x": emb2[:,0], "y": emb2[:,1], "label": y}).to_csv(os.path.join(out_dir, "feature_viz_tsne.csv"), index=False)

        if VIZ_ONLY:
            return

    # 2. 准备训练数据
    X, y, feat_cols = prepare_training_data(train_features)
    # 可选：剔除指定特征
    EXCLUDE_FEATURES = os.environ.get("EXCLUDE_FEATURES", "").strip()
    DROP_TIME_GAP_FEATURES = os.environ.get("DROP_TIME_GAP_FEATURES", "0") == "1"
    exclude_cols = set()
    if DROP_TIME_GAP_FEATURES:
        exclude_cols.update({
            "days_since_last_action",
            "ui_days_since_last_action",
            "recent_3d_actions",
            "recent_7d_actions",
            "recent_7d_views",
            "recent_3d_views",
        })
    if EXCLUDE_FEATURES:
        exclude_cols.update([c.strip() for c in EXCLUDE_FEATURES.split(",") if c.strip()])
    if exclude_cols:
        keep = [c for c in feat_cols if c not in exclude_cols]
        removed = sorted(list(set(feat_cols) - set(keep)))
        print(f"剔除特征列: {removed}")
        feat_cols = keep

    # 3. 首轮 5 折训练与预测
    # 训练提速/更真实验证的可配置项（环境变量便于快速切换）
    USE_GROUP_KFOLD = os.environ.get("USE_GROUP_KFOLD", "1") == "1"
    N_SPLITS = int(os.environ.get("N_SPLITS", "3"))
    EARLY_STOPPING = int(os.environ.get("EARLY_STOPPING", "50"))
    DOWNSAMPLE_NEG = os.environ.get("DOWNSAMPLE_NEG", "1") == "1"
    NEG_POS_RATIO = int(os.environ.get("NEG_POS_RATIO", "20"))
    FAST_PARAMS = os.environ.get("FAST_PARAMS", "1") == "1"

    # 可选：负样本下采样（大幅加速）
    train_df_for_train = train_features
    if DOWNSAMPLE_NEG:
        NEG_MAX_GAP_DAYS = os.environ.get("NEG_MAX_GAP_DAYS")
        neg_gap = int(NEG_MAX_GAP_DAYS) if (NEG_MAX_GAP_DAYS and NEG_MAX_GAP_DAYS.isdigit()) else None
        train_df_for_train = downsample_negatives(
            train_features,
            neg_pos_ratio=NEG_POS_RATIO,
            by_user=False,
            neg_max_gap_days=neg_gap,
        )

    # 同步使用 feat_cols 作为训练特征列，保证与预测阶段一致
    X2 = train_df_for_train[feat_cols]
    y2 = train_df_for_train["label"].astype(int)

    groups = train_df_for_train["user_id"].values if USE_GROUP_KFOLD else None

    # 当使用负采样时，class_weight 改为 None，避免重复加权导致偏置
    class_weight = None if DOWNSAMPLE_NEG else "balanced"

    test_pred, oof, fi_group, models = kfold_train_predict(
        X2,
        y2,
        pred_features,
        feat_cols,
        n_splits=N_SPLITS,
        random_state=42,
        groups=groups,
        early_stopping_rounds=EARLY_STOPPING,
        fast_params=FAST_PARAMS,
        class_weight=class_weight,
    )

    # 4. 简易特征筛选：剔除 0 重要性特征后再跑一轮（若确有0重要性特征）
    kept, dropped = drop_zero_importance_features(fi_group, feat_cols)
    if len(dropped) > 0 and len(kept) >= max(10, int(0.7 * len(feat_cols))):
        print(f"剔除 {len(dropped)} 个0重要性特征，重新训练...")
        test_pred, oof, fi_group, models = kfold_train_predict(
            X2[kept], y2, pred_features, kept,
            n_splits=N_SPLITS, random_state=42, groups=groups,
            early_stopping_rounds=EARLY_STOPPING, fast_params=FAST_PARAMS, class_weight=class_weight
        )
        feat_cols = kept
    else:
        print("未发现需要剔除的特征或剩余特征过少，跳过筛选重训。")

    # 5. 基于 OOF 做离线 F1 调参（Top-K 与 阈值 二选一），并二阶段重排
    print("开始离线F1调参（基于OOF抽样）...")
    strategy = offline_tune_and_choose_strategy(
        train_df_for_train,  # 与 OOF 对齐
        oof,
        sample_user_frac=0.02,
        random_state=42,
    )

    # 可选：强制策略（通过环境变量覆盖离线选择）
    force = os.environ.get("FORCE_STRATEGY", "").strip().lower()
    if force in {"topk", "top-k", "two_stage", "2stage"}:
        k = int(os.environ.get("TOPK_K", "5"))
        m = int(os.environ.get("TOPK_M", "2"))
        strategy = {"name": "topk_two_stage", "k": k, "m": m, "fallback": None}
        print(f"强制策略: Top-K 二阶段重排, K={k}, m={m}")
    elif force in {"threshold", "thr"}:
        t = float(os.environ.get("THRESHOLD", "0.5"))
        strategy = {"name": "threshold", "threshold": t, "q_level": None, "fallback": None}
        print(f"强制策略: 概率阈值, threshold={t}")
    elif force in {"global_topn", "topn"}:
        strategy = {"name": "global_topn"}
        print("强制策略: 全局 Top-N")
    elif force in {"cart", "cart_only"}:
        strategy = {"name": "cart_only"}
        print("强制策略: cart_only")

    # 5.1 若开启评估，则在训练日(12-18) OOF 上对当前策略做离线评估
    if os.environ.get("EVAL_STRATEGY", "1") == "1":
        strong_only = os.environ.get("STRONG_ONLY", "0") == "1"
        min_prob = float(os.environ.get("MIN_PROB", "0") or 0)
        max_sub = int(os.environ.get("MAX_SUBMISSION", "0") or 0)
        cart_days_max = int(os.environ.get("CART_DAYS_MAX", "1") or 1)
        eval_frac = float(os.environ.get("EVAL_SAMPLE_USER_FRAC", "0.02"))
        evaluate_forced_strategy(
            train_df_for_train,
            oof,
            strategy,
            sample_user_frac=eval_frac,
            min_prob=min_prob,
            max_submission=max_sub,
            strong_only=strong_only,
            cart_days_max=cart_days_max,
        )

    # 6. 在预测集上使用选择的策略生成最终推荐
    results = pred_features[["user_id", "item_id"]].copy()
    results["pred_prob"] = test_pred
    # 去重保护：同一 user-item 取最大概率
    results = results.groupby(["user_id", "item_id"], as_index=False)["pred_prob"].max()
    print(f"预测完成，候选对数量: {len(results)} (去重后)")

    # 计算强信号掩码（与离线一致）
    pred_strong_mask = build_strong_mask(pred_features)

    # 选择前进行预过滤（可选）
    results = apply_pre_filters(results, pred_features, pred_strong_mask)

    if strategy["name"] == "topk_two_stage":
        k, m = strategy["k"], strategy["m"]
        print(f"使用二阶段重排策略: K={k}, m={m}")
        strong_pairs_df = pred_features.loc[build_strong_mask(pred_features).astype(bool), ["user_id", "item_id"]]
        recommendations = select_final_recommendations_two_stage(results, strong_pairs_df, top_k_per_user=k, strong_first=m)
    elif strategy["name"] == "threshold":
        t_offline = strategy["threshold"]
        q_level = strategy.get("q_level")
        # 先基于预测集分布做分位对齐
        t_adj = _adjust_threshold_by_quantile(results["pred_prob"].values, q_level=q_level, default_t=t_offline)
        q_str = f"{q_level:.6f}" if q_level is not None else "nan"
        print(f"使用阈值策略: offline_t={t_offline:.6f}, q_level={q_str}, adjusted_t={t_adj:.6f}")
        recommendations = select_final_recommendations_by_threshold(results, threshold=t_adj)
        # 安全兜底：若为空或过少，回退二阶段 Top-K
        if len(recommendations) == 0 or len(recommendations) < 1000:
            fb = strategy.get("fallback") or {"k": 5, "m": 1}
            print(f"阈值结果过少（{len(recommendations)}），回退二阶段重排: K={fb['k']}, m={fb['m']}")
            strong_pairs_df = pred_features.loc[build_strong_mask(pred_features).astype(bool), ["user_id", "item_id"]]
            recommendations = select_final_recommendations_two_stage(results, strong_pairs_df, top_k_per_user=int(fb['k']), strong_first=int(fb['m']))
    elif strategy["name"] == "global_topn":
        topn = int(os.environ.get("TOPN", os.environ.get("MAX_SUBMISSION", "100000")))
        recommendations = select_global_topn(results, topn=topn)
        print(f"使用全局 Top-N 策略: topn={topn}")
    elif strategy["name"] == "cart_only":
        max_rows = int(os.environ.get("MAX_SUBMISSION", "100000"))
        recommendations = select_cart_only(pred_features, results, max_rows=max_rows)
        print(f"使用 cart_only 策略: max={max_rows}")
    else:
        # 默认回到二阶段
        k = int(os.environ.get("TOPK_K", "5"))
        m = int(os.environ.get("TOPK_M", "2"))
        print(f"未知策略，回退二阶段: K={k}, m={m}")
        recommendations = select_final_recommendations_two_stage(results, pred_strong_mask, top_k_per_user=k, strong_first=m)

    # 可选：先排除 2014-12-18 当日已购（可通过环境变量覆盖或关闭）
    exclude_day = os.environ.get("EXCLUDE_DAY_PURCHASES", "20141218")
    if exclude_day and exclude_day != "0":
        recommendations = exclude_purchases_on_day(recommendations, day=exclude_day)

    # 可选：全局提交数量上限（按概率降序截断）
    MAX_SUB = int(os.environ.get("MAX_SUBMISSION", "0"))
    if MAX_SUB > 0:
        recommendations = cap_submission_size(recommendations, results, MAX_SUB)

    # 7. 保存提交文件
    save_submission(recommendations)

    print("\n训练和预测完成！ 提交文件: submission.txt")

    # 8. 可选：在 12-18 全量候选上做“完整离线验证”（可能较慢）
    if os.environ.get("VAL_FULL", "0") == "1":
        print("开始 12-18 全量离线验证（VAL_FULL=1）...")
        tf = train_features.copy()
        # 使用与当前模型一致的特征列
        vf_cols = [c for c in tf.columns if c not in ["user_id", "item_id", "label"]]
        # 预测概率（分块平均各折）
        val_probs = _predict_full_in_chunks(models, tf, vf_cols, batch=int(os.environ.get("VAL_BATCH", "1000000")))
        res_v = tf[["user_id", "item_id", "label"]].copy()
        res_v["pred_prob"] = val_probs
        # 去重聚合
        res_v = res_v.groupby(["user_id", "item_id"], as_index=False).agg({"pred_prob":"max", "label":"max"})

        # 预过滤
        strong_pairs_val = tf.loc[build_strong_mask(tf).astype(bool), ["user_id", "item_id"]]
        res_tmp = apply_pre_filters(res_v[["user_id","item_id","pred_prob"]].copy(), tf, build_strong_mask(tf))

        # 应用策略（不排除 12-18 已购）
        if strategy["name"] == "topk_two_stage":
            rec_v = select_final_recommendations_two_stage(res_tmp, strong_pairs_val, top_k_per_user=int(strategy['k']), strong_first=int(strategy['m']))
        elif strategy["name"] == "threshold":
            t = float(strategy.get("threshold", 0.5))
            rec_v = select_final_recommendations_by_threshold(res_tmp, threshold=t)
        elif strategy["name"] == "global_topn":
            topn = int(os.environ.get("TOPN", os.environ.get("MAX_SUBMISSION", "100000")))
            rec_v = select_global_topn(res_tmp, topn=topn)
        elif strategy["name"] == "cart_only":
            # 用 cart_only 评估时，使用 tf 作为特征源
            rec_v = select_cart_only(tf, res_tmp, max_rows=int(os.environ.get("MAX_SUBMISSION", "100000")))
        else:
            rec_v = select_final_recommendations_two_stage(res_tmp, strong_pairs_val, top_k_per_user=5, strong_first=2)

        # 计算 F1/P/R
        lab_map = res_v.set_index(["user_id","item_id"])['label']
        pred_cnt = len(rec_v)
        tp = int(lab_map.reindex(pd.MultiIndex.from_frame(rec_v), fill_value=0).sum())
        true_pos = int(res_v['label'].sum())
        f1, precision, recall = _f1_from_counts(tp, pred_cnt, true_pos)
        print(f"VAL_FULL on 12-18: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, 选中={pred_cnt}, 正例总数={true_pos}")


if __name__ == "__main__":
    main()
