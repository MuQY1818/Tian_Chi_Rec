# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个天池移动电商推荐竞赛项目，主要任务是基于用户行为数据构建个性化推荐模型，预测用户对特定商品子集的购买行为。

## 数据结构

### 核心数据文件
- `dataset/tianchi_fresh_comp_train_item_online.txt`: 商品子集P数据，包含商品属性
- `dataset/tianchi_fresh_comp_train_user_online_partA.txt`: 用户行为数据A部分
- `dataset/tianchi_fresh_comp_train_user_online_partB.txt`: 用户行为数据B部分

### 数据字段说明
**用户行为数据**:
- `user_id`: 用户标识（脱敏）
- `item_id`: 商品标识（脱敏）
- `behavior_type`: 行为类型（1-浏览，2-收藏，3-加购物车，4-购买）
- `user_geohash`: 用户位置空间标识（可为空）
- `item_category`: 商品分类标识（脱敏）
- `time`: 行为时间（精确到小时）

**商品子集数据**:
- `item_id`: 商品标识
- `item_geohash`: 商品位置空间标识（可为空）
- `item_category`: 商品分类标识

## 核心技术架构

### 推荐算法流程
1. **数据预处理**: 缺失值处理、行为编码、时间格式转换
2. **特征工程**: 
   - 用户行为特征（加权行为统计）
   - 时间特征（时间衰减、窗口特征）
   - 商品特征（热度、类别、位置）
   - 用户-商品交互特征
3. **模型训练**: 传统模型（LR、FM、ItemCF）+ 深度学习模型（GRU4Rec、DeepFM）
4. **模型融合**: 加权融合或Stacking集成
5. **结果生成**: Top-K推荐与过滤

### 关键算法选择
- **传统模型**: 逻辑回归、因子分解机、ItemCF协同过滤
- **深度学习模型**: GRU4Rec、DeepFM、MAGRU（带注意力机制）
- **融合策略**: 加权融合（LR+DeepFM）或Stacking（LR+FM+GRU4Rec+XGBoost）

## 开发环境设置

### Conda环境
项目使用专门的conda环境运行：
```bash
conda activate kaggle_env
```

### 依赖要求
基于方案文档，主要需要的Python库包括：
- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: 机器学习模型
- `tensorflow`/`pytorch`: 深度学习模型
- `xgboost`: 梯度提升树

### 数据处理规范
- 内存优化：使用`reduce_mem`函数将数值型特征转换为更小数据类型
- 时间处理：将时间字符串转换为Unix时间戳并提取衍生特征
- 商品子集过滤：确保所有推荐商品属于商品子集P

## 重要开发注意事项

### 特征工程原则
- 行为权重：浏览(1)、收藏(2)、加购物车(3)、购买(4)
- 时间衰减：使用指数衰减函数处理行为时效性
- 冷启动处理：基于人口统计学特征、热门商品兜底、兴趣探索

### 模型训练策略
- 数据集划分：时间序列划分（训练集：11.18-12.15，验证集：11.21-12.18）
- 负采样：困难负采样，正负样本比例1:5
- 损失函数：加权交叉熵（正样本权重5，负样本权重1）
- 优化器：Adam（初始学习率0.001）

### 评估指标
- 主要指标：F1值（最终评测标准）
- 辅助指标：AUC、NDCG@5、覆盖率

### 输出格式要求
- 列：user_id和item_id，tab分隔
- 排序：user_id升序，同一用户的item_id按预测概率降序
- 数量：每个用户最多推荐5个商品
- 文件格式：txt文件，ASCII文本编码

### 输出格式修复记录
- **问题**: 原始输出为CSV格式，不符合比赛要求
- **解决方案**: 修改`simple_baseline_models.py`中的`format_submission`方法
- **实现**: 生成txt文件，使用tab分隔，按user_id升序排列
- **验证**: 每个用户推荐5个商品，格式正确

## 代码规范

### Google Style规范
- 遵循Google Python Style Guide
- 函数和变量使用snake_case命名
- 类使用CamelCase命名
- 常量使用UPPER_CASE命名

### 注释要求
- 中文注释，简洁明了
- 复杂算法需要详细注释说明
- 关键参数需要说明取值依据

## 常用命令

### 数据处理
```bash
# 查看数据文件大小
ls -lh dataset/

# 查看数据前几行
head -n 5 dataset/tianchi_fresh_comp_train_user_online_partA.txt
```

### Python开发
```bash
# 启动Python环境
python

# 或使用jupyter notebook
jupyter notebook
```

## 性能优化建议

### 大数据处理
- 使用分块处理大型数据文件
- 实现增量特征计算
- 考虑使用Dask处理超大规模数据

### 模型优化
- ItemCF相似度计算优化
- 深度学习模型批量训练
- 模型参数早停机制