# 天池推荐算法竞赛

## 任务描述

预测用户在2014年12月19日对商品子集P的购买行为。

## 数据文件

- `dataset/tianchi_fresh_comp_train_item_online.txt`: 商品子集P
- `dataset/tianchi_fresh_comp_train_user_online_partA.txt`: 用户行为数据A
- `dataset/tianchi_fresh_comp_train_user_online_partB.txt`: 用户行为数据B

## 算法流程

### 1. 候选集生成 (`1_candidate_generation.py`)
- 从全量用户行为数据中筛选商品子集P相关的交互
- 生成训练样本：2014-12-18的购买行为作为正样本标签
- 生成预测候选集：历史有交互的用户-商品对

### 2. 特征工程 (`2_feature_engineering.py`)
- **用户特征**: 活跃度、购买率、时间模式、类别偏好
- **商品特征**: 流行度、转化率、最近趋势
- **交互特征**: 用户对商品的历史行为统计

### 3. 模型训练 (`3_lgbm_training.py`)
- 使用LightGBM训练二分类模型
- 预测用户-商品购买概率
- 为每个用户选择top-5商品生成最终推荐

## 运行方法

```bash
# 1. 生成候选集
python 1_candidate_generation.py

# 2. 特征工程
python 2_feature_engineering.py

# 3. 训练和预测
python 3_lgbm_training.py
```

## 输出文件

- `submission.txt`: 最终提交文件，格式为 user_id\titem_id
- 中间文件: `train_candidates.csv`, `pred_candidates.csv`, `train_features.csv`, `pred_features.csv`

## 环境要求

- Python 3.7+
- pandas, numpy, lightgbm, scikit-learn, tqdm