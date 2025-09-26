# 天池推荐算法竞赛

基于用户行为数据预测用户对商品子集的购买行为，使用三阶段流水线：候选集生成 → 特征工程 → LightGBM训练预测。

## 项目结构

```
Tian_Chi_Rec/
├── 1_candidate_generation.py    # 候选集生成
├── 2_feature_engineering.py     # 特征工程
├── 3_lgbm_training.py           # LightGBM训练与预测
├── config/
│   └── config.ini               # 配置文件
├── dataset/                     # 数据集目录（需手动创建）
│   ├── daily_data/              # 日分片数据
│   └── tianchi_fresh_comp_train_item_online.txt
├── project/
│   └── CLAUDE.md               # 项目修改记录
└── README.md
```

## 环境配置

1. 创建conda环境：
```bash
conda create -n kaggle_env python=3.9
conda activate kaggle_env
```

2. 安装依赖：
```bash
pip install pandas numpy lightgbm scikit-learn tqdm configparser
```

## 数据准备

1. 创建数据集目录：
```bash
mkdir -p dataset/daily_data
```

2. 将原始数据按日切分到 `dataset/daily_data/` 目录：
   - 格式：`data_YYYYMMDD.txt`
   - 每个文件包含当日的用户行为数据

3. 将商品子集文件放到：
   - `dataset/tianchi_fresh_comp_train_item_online.txt`

## 使用方法

### 完整流水线运行

激活环境后按顺序运行三个脚本：

```bash
conda activate kaggle_env

# 1. 生成候选集
python 1_candidate_generation.py

# 2. 特征工程
python 2_feature_engineering.py

# 3. 模型训练预测
python 3_lgbm_training.py
```

### 配置文件

通过 `config/config.ini` 调整参数：

```ini
[training]
downsample_neg = 1              # 负样本下采样
neg_pos_ratio = 1               # 负正样本比例
use_group_kfold = 1             # 使用GroupKFold
n_splits = 5                    # 交叉验证折数
early_stopping = 200            # 早停轮数

[strategy]
force_strategy = cart_only      # 强制推荐策略
strong_only = 1                 # 仅使用强信号候选
min_prob = 0.09                # 最小概率阈值
max_submission = 60000          # 最大提交数量

[eval]
eval_strategy = 1               # 启用离线评估
val_full = 1                   # 全量验证
```

### 环境变量配置

也可通过环境变量覆盖配置：

```bash
# 数据泄露修复
export DROP_TIME_GAP_FEATURES=1  # 排除时间间隔特征
export ALIGN_RECALL=1            # 对齐召回策略

# 训练加速
export DOWNSAMPLE_NEG=1          # 负样本下采样
export NEG_POS_RATIO=10          # 下采样比例
export FAST_PARAMS=1             # 快速参数
```

## 输出文件

- `train_candidates.csv` - 训练候选集
- `pred_candidates.csv` - 预测候选集
- `train_features.csv` - 训练特征
- `pred_features.csv` - 预测特征
- `submission.txt` - 最终提交文件

## 算法流程

### 1. 候选集生成 (`1_candidate_generation.py`)

**目标**：从全量行为数据D中筛选可能的用户-商品推荐对

**策略**：
- 训练候选：最近7天交互 + 最近14天强信号（收藏/加购）+ 标签日购买
- 预测候选：最近7天交互 + 最近14天强信号
- 可选正样本召回过滤（ALIGN_RECALL=1）

### 2. 特征工程 (`2_feature_engineering.py`)

**特征类型**：
- **用户特征**：总行为数、各类型行为统计、活跃度、购买率等
- **商品特征**：总行为数、独立用户数、转化率、流行度等
- **交互特征**：用户-商品历史交互统计、时间间隔等
- **用户-品类特征**：用户对特定品类的偏好

**时间窗口**：
- 训练：使用 < 2014-12-18 的数据构建特征
- 预测：使用 < 2014-12-19 的数据构建特征

### 3. 模型训练预测 (`3_lgbm_training.py`)

**训练**：
- 5折交叉验证（GroupKFold按用户分组）
- LightGBM二分类模型
- 可选负样本下采样加速训练

**推荐策略**：
- `cart_only`：优先推荐最近加购的商品
- `topk_two_stage`：二阶段重排（强信号优先）
- `threshold`：概率阈值筛选
- `global_topn`：全局Top-N

## 数据泄露修复

针对发现的候选集构造导致的数据泄露问题，已加入以下修复：

```bash
# 启用修复选项
export ALIGN_RECALL=1            # 对齐训练预测召回策略
export POS_FILTER_IN_RECALL=1    # 正样本仅保留召回内的
export DROP_TIME_GAP_FEATURES=1  # 排除强时间依赖特征
```

## 注意事项

1. **环境要求**：必须在kaggle_env环境中运行
2. **内存需求**：建议16GB+内存，数据量较大
3. **时间成本**：完整流水线约需1-2小时
4. **数据格式**：确保数据文件格式正确，字段对齐

## 问题排查

- **AUC过高但F1很低**：检查是否存在数据泄露，启用修复选项
- **内存不足**：降低负样本采样比例或启用分块处理
- **训练过慢**：启用FAST_PARAMS=1或增加early_stopping
- **推荐效果差**：调整推荐策略或阈值参数