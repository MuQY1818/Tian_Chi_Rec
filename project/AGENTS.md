# AGENTS 变更记录

日期: 2025-09-26
作者: muqy1818 项目协作代理（自动记录）

本次重大修改聚焦于“避免一次性加载全量数据、提升候选召回与特征工程效率”，并补充 K 折训练流程。

## 1. 候选集生成（1_candidate_generation.py）
- 改为读取 `dataset/daily_data/` 日分片，避免直接加载 Part A/B。
- 多路召回：
  - 最近 7 天任意交互（训练: 12-11~12-17；预测: 12-12~12-18）。
  - 最近 14 天强信号（收藏/加购）（训练: 12-04~12-17；预测: 12-05~12-18）。
  - 标签日（2014-12-18）购买对全覆盖为正样本。
- 每用户候选上限 `max_per_user=200`（可调），优先级：正样本 > 强信号 > 最近交互。
- 预留热门回填逻辑（默认关闭以防规模爆炸）。
- 新增运行环境校验：必须在 `kaggle_env` conda 环境下执行。

输出仍为：`train_candidates.csv`，`pred_candidates.csv`。

## 2. 特征工程（2_feature_engineering.py）
- 流式/分片 + 向量化实现，按候选用户/候选商品与 P 子集限制聚合范围。
- 用户特征：总行为、行为型计数、活跃天数、近 7/3 天活跃、最近一次行为间隔、唯一类别数等。
- 商品特征（仅候选商品 ∩ P）：总行为、浏览/购买、唯一用户数、近 7/3 天浏览、最近一次浏览间隔、转化率、均次等。
- 交互特征（仅候选对）：总次数、各行为型次数、首次/末次间隔、均次/天。
- 用户-品类偏好：按候选对所属品类统计 user-category 的行为次数与唯一商品数。
- 新增运行环境校验：必须在 `kaggle_env` conda 环境下执行。
- 增强进度反馈：主循环采用 `tqdm` 动态展示，每日分片显示 `day/rows/kept/pairs/sec` 指标，便于掌控进度与耗时。
- 聚合完成后的阶段性进度：分别在“用户特征组装、商品特征组装、交互特征组装、用户-品类特征组装、逐步合并”阶段打印维度与耗时，便于定位耗时点。

输出仍为：`train_features.csv`，`pred_features.csv`。

## 3. 训练与预测（3_lgbm_training.py）
- 使用 `StratifiedKFold(n_splits=5)`，折内早停，输出 OOF AUC 与折均 AUC。
- 汇总折内 `gain` 重要性，剔除 0 重要性特征后可选重训一轮（阈值安全限制）。
- 预测对按 `pred_prob` 排序后每用户 Top-K（默认 5）生成提交。
- 新增运行环境校验：必须在 `kaggle_env` conda 环境下执行。
- 新增类别不平衡处理：`class_weight='balanced'`，自动按类频率加权训练。
- 新增离线 F1 调参：基于 OOF 对样本用户抽样（默认 2%）同时搜索两类策略并择优：
  - Top-K 二阶段重排（K∈{3,5,8,10,12,15}；强信号优先 m∈{0,1,2,3,5}），强信号定义：加购/收藏/近3天活跃/最近间隔≤3天/历史购买>0。
  - 概率阈值策略（在高分区间扫描 25 个阈值）。
  线上预测阶段自动采用离线最优策略生成提交。
- 阈值策略鲁棒性增强：
  - 将离线最优阈值按“分位占比”映射到预测集分布（quantile 对齐），降低分布偏移导致的全空结果风险。
  - 若阈值筛选结果为空或过少（<1000），自动回退到离线最优 Top-K 二阶段重排（兜底）。
- 新增训练加速与更真实验证选项（通过环境变量控制）：
  - `DOWNSAMPLE_NEG=1`、`NEG_POS_RATIO=20`：启用负样本下采样，大幅减少训练集规模（默认开启）。
  - `USE_GROUP_KFOLD=1`：按 `user_id` 做 GroupKFold，避免同一用户跨折的信息泄露（默认开启）。
  - `N_SPLITS=3`、`EARLY_STOPPING=50`、`FAST_PARAMS=1`：减少折数、提升学习率与叶子约束，缩短迭代时间（默认快速设置）。
  - `LOG_EVAL_EVERY=20`：训练日志评估打印间隔（轮数），默认 20，可按需调大或调小。
- 训练难度与消融：
  - `NEG_MAX_GAP_DAYS=3`：负样本仅从 `ui_days_since_last_action<=3` 的集合采样，提升难度，降低“易分”带来的虚高 AUC。
  - `DROP_TIME_GAP_FEATURES=1` 或 `EXCLUDE_FEATURES=col1,col2`：在训练与预测中排除时间/近期窗口等强信号列，做消融验证是否存在泄露。
- 离线策略评估（避免盲目提交）：
  - `EVAL_STRATEGY=1`（默认开启）：基于训练日(12-18)的 OOF 概率，对当前策略（含强制策略）在抽样用户上计算 F1/Precision/Recall 与选中条数；
  - 抽样比例：`EVAL_SAMPLE_USER_FRAC=0.02`；评估时尊重 `STRONG_ONLY`、`MIN_PROB`、`MAX_SUBMISSION`、`CART_DAYS_MAX` 等参数。

## 4. 配置管理
- 新增 `config/config.ini`，用于集中配置训练/策略/评估/可视化参数，减少环境变量书写；脚本启动时自动加载：
  - [training] downsample_neg, neg_pos_ratio, neg_max_gap_days, use_group_kfold, fast_params, n_splits, early_stopping, log_eval_every
  - [strategy] force_strategy, strong_only, min_prob, topn, max_submission, exclude_day_purchases, topk_k, topk_m, cart_days_max, threshold
  - [eval] eval_strategy, eval_sample_user_frac, val_full, val_batch
  - [viz] feature_viz, viz_only, viz_sample_per_class, viz_methods, viz_out, tsne_learning_rate, tsne_n_iter
- 加载规则：仅在对应环境变量未设置时，才使用 INI 中的值（setdefault）。

### 2025-09-27 推荐系统评估指标完善
- 目的：完善推荐系统评估体系，认识到高AUC在推荐场景中是正常的。
- 关键认知更新：
  - **高AUC不是数据泄露**：在推荐系统中，0.98+的AUC是合理的，因为模型利用了强信号（加购、收藏等）
  - **时间分布差异是有价值信号**：正负样本在时间间隔上的差异反映真实用户行为模式，不是泄露
- 新增完整推荐系统评估指标（3_lgbm_training.py）：
  - `calculate_recommendation_metrics()`：计算Precision@K、Recall@K、F1@K、Hit Ratio@K、NDCG@K
  - `print_recommendation_metrics()`：格式化输出推荐指标表格
  - 集成到训练流程：每折验证显示P@5/R@5/F1@5，OOF显示完整指标表格（K∈{1,3,5,10}）
  - 解决"只看AUC无法评估推荐效果"的问题，提供更全面的性能视角
- 保持原有配置：
  - [strategy] cart_only策略，`max_submission=50000`
  - [training] 负采样1:1，GroupKFold，early stopping
  - 所有时间特征保留，不再认为是数据泄露

### 2025-09-26 策略更新（高精起步 A 案）
- 目的：线上 F1 偏低，改为"cart_only 高精方案"。
- [strategy]
  - `force_strategy=cart_only`
  - `cart_days_max=1`
  - `strong_only=1`
  - `min_prob=0.09`
  - `max_submission=50000`
  - `exclude_day_purchases=0`
- [training] 保持：`downsample_neg=1`, `neg_pos_ratio=1`, `neg_max_gap_days=3`, `use_group_kfold=1`, `n_splits=5`, `early_stopping=200`。
- [eval] 保持：`eval_strategy=1`, `val_full=1`。

### 2025-09-26 方案A（根因修复）规划与部分落地
- 目的：修复“训练候选≠预测候选”导致的分布偏差与时间间隔捷径。
- 候选对齐（已加入 1_candidate_generation）：
  - `training.align_recall`（ENV: ALIGN_RECALL，默认 1）：训练候选与预测一致的召回窗口；
  - `training.pos_filter_in_recall`（ENV: POS_FILTER_IN_RECALL，默认 1）：仅保留召回窗口内可召回的正例；
  - 支持 `LABEL_DATE` 从 `config.ini` 读取，自动设置预测日为 +1 天。
- 负样本分桶采样（已加入 3_lgbm_training）：
  - `BUCKETED_NEG=1`、`RECENCY_BOUNDS="1,3,7"`：按 ui_days_since_last_action 分桶，桶内按 `NEG_POS_RATIO` 采样；
  - 可叠加 `NEG_MAX_GAP_DAYS` 先做近邻过滤再分桶采样。
- 时间验证（计划）：
  - 后续将加入 `LABEL_DATE` 打通 1/2/3 脚本一键前移到 12-17 做时间验证。
- Bugfix：OOF 与离线调参数据对齐。离线调参改为基于实际训练用的样本（负采样后）执行，避免长度不匹配错误；阈值日志打印兼容 q_level=None。
- 新增策略强制开关（环境变量）：
  - `FORCE_STRATEGY=topk` 并可配 `TOPK_K=5`、`TOPK_M=2` 强制使用二阶段 Top-K；
  - 或 `FORCE_STRATEGY=threshold` 配 `THRESHOLD=0.5` 强制使用固定阈值；
  - `FORCE_STRATEGY=global_topn`：全局按概率降序取前 `TOPN`（或 `MAX_SUBMISSION`）。
  - `FORCE_STRATEGY=cart_only`：仅购物车近一天（不足放宽至2、3天）优先，按概率降序；配合 `MAX_SUBMISSION`。
  - 未设置时默认采用离线调参择优策略。
- 新增提交数量上限：`MAX_SUBMISSION=100000` 按预测概率全局降序截断提交条数，默认不限制。
- 预过滤开关：`STRONG_ONLY=1` 仅保留强信号样本；`MIN_PROB=0.05` 先按概率阈值过滤，再做选择（Top-K/阈值/TopN）。
- cart_only 策略新增参数：`CART_DAYS_MAX`（默认 3）。设置为 1 可严格仅使用“最近1天加购”，避免放宽到第2/3天导致 precision 下滑。
- 提交去重保护：
  - 预测阶段对 `(user_id,item_id)` 聚合取最大概率；
  - 二阶段选择、最终写出前均执行去重，避免重复提交行。
- 提交前排除当日已购：
  - 新增 `EXCLUDE_DAY_PURCHASES`（默认 `20141218`），在预测阶段读取 `dataset/daily_data/data_YYYYMMDD.txt`，移除该日已购买的 `(user_id,item_id)` 对；设为 `0` 可关闭。
- Bugfix：对齐掩码与结果长度
  - 预过滤 `STRONG_ONLY` 与 `cart_only` 的筛选条件改为基于 `(user_id,item_id)` 内连接，不再用布尔掩码直接切 `results`，避免长度不匹配错误。
  - 二阶段 Top‑K 的强信号标记改为与强信号对 DataFrame 连接，避免与 `results` 行数不一致导致报错。

输出：`submission.txt`。

## 使用说明（快速）
1. 确保当前在 `kaggle_env` 环境：`echo $CONDA_DEFAULT_ENV` 应为 `kaggle_env`。
2. 生成候选：`python 1_candidate_generation.py`
3. 特征工程：`python 2_feature_engineering.py`
4. 训练与预测：`python 3_lgbm_training.py`

备注：所有脚本均默认读取 `dataset/daily_data/` 分片，并以 `dataset/tianchi_fresh_comp_train_item_online.txt` 为子集 P。
- 特征可视化（基于训练特征的采样）：
  - 环境变量：`FEATURE_VIZ=1` 开启；`VIZ_ONLY=1` 仅做可视化不训练；
  - 采样大小：`VIZ_SAMPLE_PER_CLASS=10000`（每类采样数，默认 1w/类）；
  - 方法选择：`VIZ_METHODS=pca,tsne`；输出目录：`VIZ_OUT=viz`；
  - 产物：`viz/feature_viz_pca.png/.csv`、`viz/feature_viz_tsne.png/.csv`；若 matplotlib 不可用，仅输出 CSV。
  - 兼容性：t‑SNE 参数根据本机 sklearn 版本自动适配，不支持的参数会自动忽略。可通过 `TSNE_LEARNING_RATE`、`TSNE_N_ITER` 覆盖默认。
