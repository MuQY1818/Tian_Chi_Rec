#!/usr/bin/env python3
"""
仓库整理脚本
"""
import os
import shutil
from pathlib import Path

def reorganize_repository():
    """重新组织仓库结构"""
    print("🧹 开始整理仓库结构...")

    # 创建新的目录结构
    directories = [
        "scripts/preprocessing",
        "scripts/feature_engineering",
        "scripts/modeling",
        "scripts/submission",
        "scripts/utils",
        "cpp_tools",
        "docs/design",
        "experiments",
        "archive",
        "outputs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 创建目录: {directory}")

    # 文件重新组织映射
    file_moves = {
        # C++ 工具
        "fast_user_extractor.cpp": "cpp_tools/",
        "fast_user_extractor": "cpp_tools/",
        "build_cpp.sh": "cpp_tools/",

        # 预处理脚本
        "create_small_dataset.py": "scripts/preprocessing/",
        "quick_user_count.py": "scripts/preprocessing/",

        # 特征工程
        "batch_user_feature_extractor.py": "scripts/feature_engineering/",
        "user_feature_builder.py": "scripts/feature_engineering/",

        # 建模脚本
        "traditional_train.py": "scripts/modeling/",
        "simple_train.py": "scripts/modeling/",

        # 提交脚本
        "full_submission_generator.py": "scripts/submission/",

        # 工具脚本
        "check_overlap.py": "scripts/utils/",
        "time_estimator.py": "scripts/utils/",
        "manual_test.py": "scripts/utils/",

        # 调试脚本
        "debug_eval.py": "scripts/utils/",
        "debug_training.py": "scripts/utils/",
        "debug_validation.py": "scripts/utils/",

        # 设计文档
        "cpp_preprocessor_design.md": "docs/design/",
        "feature_comparison.md": "docs/design/",

        # 输出文件
        "submission.txt": "outputs/",
        "traditional_submission.txt": "outputs/",
    }

    # 移动文件
    for src_file, dest_dir in file_moves.items():
        if os.path.exists(src_file):
            dest_path = os.path.join(dest_dir, src_file)
            shutil.move(src_file, dest_path)
            print(f"  📄 移动: {src_file} -> {dest_path}")

    # 更新主要文档
    update_main_readme()
    print("📝 更新了 README.md")

    print("\n✅ 仓库整理完成!")
    print("\n📊 新的目录结构:")
    show_new_structure()

def update_main_readme():
    """更新主README文件"""
    readme_content = """# 天池推荐算法项目

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
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def show_new_structure():
    """显示新的目录结构"""
    def print_tree(directory, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return

        path = Path(directory)
        if not path.exists():
            return

        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            if item.name.startswith('.'):
                continue

            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")

            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)

    print_tree(".", max_depth=3)

if __name__ == "__main__":
    reorganize_repository()