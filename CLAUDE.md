# Repository Guidelines

## 重要修改记录

### 2024-09-15: 修正时间划分策略
- **问题**: 原设计将2014-12-19作为验证集，但实际上这是预测目标日期，没有真实标签
- **修正**:
  - 训练集: ≤ 2014-12-17
  - 验证集: 2014-12-18 (用于模型调优)
  - 预测目标: 2014-12-19 (最终提交)
- **影响文件**: `src/graphrec/prepare.py`, `config/config.py`, `README.md`

### 2024-09-15: 小数据集训练验证成功
- **问题**: 大规模数据集训练挂起，需要创建小数据集测试
- **解决方案**:
  - 创建`create_small_dataset.py`筛选663用户、1904商品的小数据集
  - 修复`prepare.py`验证样本选择逻辑（不再优先购买行为）
  - 验证用户从2个增加到111个，解决数据稀疏问题
- **训练结果**: 50轮训练后loss从0.6062降到0.0093（降幅85%），HR@10=0.0090
- **最佳参数**: epochs=50, lr=0.01, embed_dim=128, layers=4, batch_size=2048
- **影响文件**: `create_small_dataset.py`, `src/graphrec/prepare.py`, `src/graphrec/train.py`

### 2024-09-15: 传统推荐算法实现完成
- **背景**: 深度学习数据量过大训练困难，转向传统推荐算法
- **实现算法**:
  - ItemCF（物品协同过滤）: 余弦相似度，Top-20相似商品
  - 时间流行度模型: 7天时间窗口，时间衰减权重
  - ALS矩阵分解: 简化SVD实现（32因子）
  - 结果融合策略: 加权分数融合，权重比例5:3:2
- **测试结果**: 558用户，1567商品，所有算法正常运行
- **优势**: 训练速度快（总耗时<1秒），计算效率高，适合大规模数据
- **影响文件**: `src/traditional/`, `traditional_train.py`

### 2024-09-16: 机器学习推荐系统完整实现
- **背景**: 基于39维用户特征和16-18号预处理数据（1.1亿行），实现完整的ML推荐流程
- **核心架构**: 用户-商品交互预测模型（方案二）
- **实现模块**:
  - 商品特征提取器: 从16-18号数据提取商品流行度、转化率、时间趋势等25维特征
  - 训练样本生成器: 构建用户-商品交互特征，正负样本比例1:3，包含39维用户+25维商品+12维交互特征
  - LightGBM训练器: 二分类模型预测购买概率，支持特征重要性分析和模型评估
  - 推荐生成器: 候选召回+概率排序，生成Top-K推荐列表
- **特征体系**:
  - 用户特征（39维）: 活跃度、时间模式、地理偏好、商品类别偏好等
  - 商品特征（25维）: 流行度、转化率、时间趋势、类别特征等
  - 交互特征（12维）: 历史交互、时间衰减、行为进展、交互频率等
- **技术亮点**:
  - 支持大规模数据处理（1.1亿行交互数据）
  - 多策略候选生成（历史交互+热门商品+类别偏好）
  - 完整的模型评估体系（AUC、F1-Score、特征重要性）
  - 端到端流水线，从特征工程到最终提交
- **运行方式**: `python run.py ml-pipeline`
- **影响文件**: `scripts/feature_engineering/`, `scripts/modeling/`, `scripts/run_ml_pipeline.py`

### 2024-09-17: XGBoost模型优化重大突破
- **背景**: 初始notebook复现AUC仅0.69，训练参数不合理导致收敛不充分
- **关键问题**: 学习率过低(0.01)，训练轮数不足(100)，early stopping过于激进(20)
- **优化措施**:
  - 学习率提升5倍: 0.01 → 0.05
  - 训练轮数增加3倍: 100 → 300
  - 放宽early stopping: 20 → 50轮
- **性能提升**:
  - 验证集AUC: 0.6929 → 0.7432 (提升7.3%)
  - 训练轮数: 27轮 → 66轮 (更充分训练)
  - 执行时间: 保持~13秒高效
- **特征工程**: 14维特征包含复杂时间特征，全量历史交互候选集(30,687对)
- **影响文件**: `notebook_style_solution.py`

## Project Structure & Modules
- `src/data`: Exploration and preprocessing (`data_exploration.py`, `simple_preprocessing.py`).
- `src/features`: Feature engineering (`feature_engineering.py`).
- `src/models`: Baselines and ensembling (`simple_baseline_models.py`, others).
- `src/utils`: Helpers and config loaders.
- `config`: Project config (`config.py`) and dependencies (`requirements.txt`).
- `data`: `raw/`, `processed/`, `models/` outputs. Large artifacts are git‑ignored.
- Root helpers: `Makefile`, `main_large_scale.py`, `README.md`.

## Build, Test, Run
- `make install`: Install Python deps from `config/requirements.txt`.
- `make run-all`: Run exploration → preprocessing → features → models.
- Stepwise: `make run-exploration`, `make run-preprocessing`, `make run-feature-engineering`, `make run-models`.
- Large pipeline: `python main_large_scale.py --mode full --model_type ensemble` (see `README.md` for modes).
- Utilities: `make format` (Black + Flake8), `make test` (pytest), `make tree`, `make clean`.

## Coding Style & Naming
- Python, 4‑space indentation, UTF‑8, Unix newlines.
- Files/modules: `snake_case.py`; classes: `CamelCase`; functions/vars: `snake_case`.
- Docstrings: triple quotes with concise summaries; prefer type hints.
- Lint/format: Black + Flake8 via `make format` before pushing.

## Testing Guidelines
- Framework: `pytest` (run with `make test`).
- Location: create `tests/` with files named `test_*.py` mirroring `src/` structure.
- Focus: data transforms, feature outputs, and model utilities. Use small, deterministic fixtures.
- Artifacts: do not read large pickles in unit tests; mock or sample.

## Commit & PR Guidelines
- Style: Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`). Example in history: `feat: 初始化天池推荐算法项目`.
- Commits: small, scoped, imperative mood; include affected paths if helpful.
- PRs: clear description, rationale, and scope; link issues; include run command(s), logs/metrics tables (e.g., accuracy/AUC), and any data sampling notes. Screenshots for docs/plots.

## Security & Configuration
- Configure paths and switches in `config/config.py`; do not hardcode secrets or absolute paths.
- Data and large artifacts are ignored by `.gitignore` (e.g., `dataset/`, `*.pkl`, `*.csv`). Do not commit them.
- For deep models, document GPU/CPU assumptions and parameters used.

## Examples
- Run full pipeline: `make run-all`
- Format and lint before PR: `make format`
- Large‑scale run: `python main_large_scale.py --mode preprocess` then `--mode train`/`--mode predict`

