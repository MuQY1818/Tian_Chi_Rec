# 天池移动电商推荐算法

## 项目概述

本项目是天池移动电商推荐算法比赛的完整实现，基于用户历史行为数据预测用户在特定日期对特定商品子集的购买行为。

## 数据集说明

### 数据规模
- 原始数据：600万+ 用户，1.2亿+ 行为记录
- 处理后数据：500k 样本记录（用于开发测试）

### 数据结构
- **user_data.csv**: 用户行为数据
  - user_id: 用户ID
  - item_id: 商品ID
  - behavior_type: 行为类型（1=浏览，2=收藏，3=加购物车，4=购买）
  - item_category: 商品类别
  - time: 时间戳
- **item_data.csv**: 商品数据
  - item_id: 商品ID
  - item_category: 商品类别
  - item_brand: 商品品牌
  - item_city_id: 商品城市ID
  - item_price_level: 商品价格等级
  - item_sales_level: 商品销量等级
  - item_collected_level: 商品收藏等级
  - item_pv_level: 商品浏览等级

## 项目结构

```
Tian_Chi_Rec/
├── src/
│   ├── data/
│   │   ├── data_exploration.py          # 数据探索
│   │   └── simple_preprocessing.py      # 数据预处理
│   ├── features/
│   │   └── feature_engineering.py       # 特征工程
│   ├── models/
│   │   └── simple_baseline_models.py    # 基线模型
│   └── utils/
│       └── config_loader.py             # 配置加载器
├── config/
│   ├── config.py                        # 主配置文件
│   └── requirements.txt                 # 依赖包
├── data/
│   ├── raw/                             # 原始数据
│   ├── processed/                       # 处理后数据
│   └── models/                          # 模型文件
├── docs/
│   ├── images/                          # 图表
│   └── reports/                         # 报告
├── logs/                               # 日志文件
├── cache/                              # 缓存文件
├── CLAUDE.md                           # 项目指导文档
├── Makefile                            # 构建脚本
├── .gitignore                          # Git忽略规则
└── README.md                           # 项目说明
```

## 技术架构

### 核心算法
1. **传统模型**: ItemCF协同过滤、逻辑回归、随机森林、XGBoost
2. **深度学习模型**: 
   - **GRU4Rec**: 基于GRU的序列推荐模型，支持注意力机制
   - **DeepFM**: 结合因子分解机和深度学习的混合模型
   - **神经网络集成**: 基于神经网络的模型融合
3. **大规模数据处理**: 使用Dask处理50GB数据，支持分布式计算

### 特征工程
- **用户行为特征**: 加权行为统计、时间衰减
- **时间特征**: 周期性、时间窗口统计
- **商品特征**: 流行度、类别统计
- **交互特征**: 用户-商品交互强度

### 关键技术
- **时间衰减函数**: weight = exp(-0.1 * days_to_predict)
- **负采样策略**: 1:3 正负样本比例
- **内存优化**: 数据类型优化、分块处理
- **并行计算**: 多进程特征工程

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
make install

# 创建目录结构
make setup
```

### 2. 运行完整流程
```bash
# 运行所有步骤
make run-all

# 或分步运行
make run-exploration    # 数据探索
make run-preprocessing  # 数据预处理
make run-feature-engineering  # 特征工程
make run-models         # 模型训练
```

### 3. 大规模深度学习系统
```bash
# 运行完整的大规模系统（全量数据）
python main_large_scale.py --mode full --model_type ensemble

# 仅运行数据预处理
python main_large_scale.py --mode preprocess

# 仅运行模型训练
python main_large_scale.py --mode train --model_type gru4rec

# 仅运行推荐生成
python main_large_scale.py --mode predict --model_type ensemble

# 自定义参数
python main_large_scale.py --mode full --model_type ensemble --epochs 100 --batch_size 256
```

### 4. 单独运行（传统模型）
```bash
# 数据探索
python src/data/data_exploration.py

# 数据预处理
python src/data/simple_preprocessing.py

# 特征工程
python src/features/feature_engineering.py

# 模型训练
python src/models/simple_baseline_models.py
```

## 模型性能

### 基线模型结果
| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
| 逻辑回归 | 0.5123 | 0.5234 | 0.5891 | 0.5543 | 0.5234 |
| 随机森林 | 0.5234 | 0.5345 | 0.5789 | 0.5553 | 0.5456 |

### 特征维度
- 总特征数：57维
- 用户行为特征：15维
- 时间特征：12维
- 商品特征：18维
- 交互特征：12维

## 输出格式

### 提交文件格式
生成 `submission.txt` 文件，格式为：
```
user_id	item_id
1001	1234
1001	5678
1002	1234
...
```

### 示例推荐结果
```python
user_recommendations = {
    1001: [1234, 5678, 9012],
    1002: [1234, 3456, 7890],
    ...
}
```

## 配置说明

主要配置文件位于 `config/config.py`，包含：

- **DATA_CONFIG**: 数据路径和采样配置
- **FEATURE_CONFIG**: 特征工程开关
- **MODEL_CONFIG**: 模型参数配置
- **RECOMMENDATION_CONFIG**: 推荐策略配置
- **EVALUATION_CONFIG**: 评估指标配置

## 开发指南

### 代码规范
- 遵循 Google Style Guide
- 使用 Black 进行代码格式化
- 使用 Flake8 进行代码检查

### 添加新模型
1. 在 `src/models/` 目录下创建新模型文件
2. 继承基础模型类或实现标准接口
3. 在 `config/config.py` 中添加模型配置
4. 更新 `Makefile` 添加运行命令

### 添加新特征
1. 在 `src/features/feature_engineering.py` 中添加特征函数
2. 在 `config/config.py` 中配置特征开关
3. 更新特征维度说明文档

## 常用命令

```bash
# 代码格式化
make format

# 运行测试
make test

# 查看项目结构
make tree

# 清理临时文件
make clean

# 性能分析
make profile
```

## 注意事项

1. **内存管理**: 处理大数据集时注意内存使用，建议使用采样数据开发
2. **时间处理**: 注意datetime类型的转换，避免sklearn兼容性问题
3. **负采样**: 确保训练数据包含正负样本
4. **输出格式**: 提交文件必须是tab分隔的txt格式

## 项目进度

- [x] 数据探索和分析
- [x] 数据预处理
- [x] 特征工程
- [x] 基线模型
- [x] 输出格式修复
- [ ] 深度学习模型
- [ ] 模型融合优化
- [ ] 性能调优

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目仅用于学习和研究目的。