# 天池推荐算法项目

## 🎯 项目概述
天池新鲜电商推荐算法比赛 - 基于用户行为数据预测用户购买行为

## 📁 项目结构

```
├── src/                          # 核心源码模块
│   ├── traditional/              # 传统推荐算法
│   └── graphrec/                # 图神经网络推荐
├── scripts/                     # 执行脚本
│   ├── preprocessing/           # 数据预处理
│   ├── feature_engineering/     # 特征工程
│   ├── modeling/               # 模型训练
│   ├── submission/             # 结果提交
│   └── utils/                  # 工具脚本
├── cpp_tools/                  # C++高性能工具
├── dataset/                    # 数据集
├── outputs/                    # 输出结果
├── docs/                      # 项目文档
└── config/                    # 配置文件
```

## 🚀 快速开始

### 1. 数据预处理
```bash
# 创建小数据集用于测试
python scripts/preprocessing/create_small_dataset.py

# 统计用户数量
python scripts/preprocessing/quick_user_count.py
```

### 2. 特征工程
```bash
# Python版本 (丰富特征)
python scripts/feature_engineering/user_feature_builder.py

# C++版本 (高性能)
cd cpp_tools
./build_cpp.sh
./fast_user_extractor
```

### 3. 模型训练
```bash
# 传统推荐算法
python scripts/modeling/traditional_train.py

# 简化训练版本
python scripts/modeling/simple_train.py
```

### 4. 生成提交
```bash
# 快速提交生成
python scripts/submission/full_submission_generator.py
```

## 📊 算法对比

| 算法类型 | 实现位置 | 性能 | 特点 |
|----------|----------|------|------|
| ItemCF | `src/traditional/itemcf.py` | 中等 | 协同过滤 |
| 流行度 | `src/traditional/popularity.py` | 快 | 简单有效 |
| 矩阵分解 | `src/traditional/matrix_factorization.py` | 慢 | 效果好 |
| 融合模型 | `src/traditional/ensemble.py` | 中等 | 综合最优 |

## 🛠️ 开发工具

- **C++高性能处理器**: `cpp_tools/fast_user_extractor.cpp`
- **特征对比分析**: `docs/design/feature_comparison.md`
- **系统设计文档**: `docs/design/cpp_preprocessor_design.md`

## 📈 实验结果

- 用户数量: ~100万
- 数据规模: 11.65亿行交互数据
- 最佳算法: 传统融合模型
- 处理速度: C++版本提升24倍

## 🔧 配置说明

项目配置文件位于 `config/` 目录，包含数据路径、模型参数等设置。

## 📝 更新日志

详细更新记录请查看 `CLAUDE.md` 文件。
