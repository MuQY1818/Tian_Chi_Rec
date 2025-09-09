# Makefile for Recommendation System

.PHONY: help install run-exploration run-preprocessing run-feature-engineering run-models run-all clean

# 默认目标
help:
	@echo "可用的命令："
	@echo "  install        - 安装依赖"
	@echo "  run-exploration - 运行数据探索"
	@echo "  run-preprocessing - 运行数据预处理"
	@echo "  run-feature-engineering - 运行特征工程"
	@echo "  run-models      - 运行模型训练"
	@echo "  run-all         - 运行完整流程"
	@echo "  clean           - 清理临时文件"

# 安装依赖
install:
	pip install -r config/requirements.txt

# 运行数据探索
run-exploration:
	python src/data/data_exploration.py

# 运行数据预处理
run-preprocessing:
	python src/data/simple_preprocessing.py

# 运行特征工程
run-feature-engineering:
	python src/features/feature_engineering.py

# 运行模型训练
run-models:
	python src/models/simple_baseline_models.py

# 运行完整流程
run-all: run-exploration run-preprocessing run-feature-engineering run-models

# 清理临时文件
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -rf logs/
	rm -rf cache/
	rm -f *.png
	rm -f exploration_results.json
	rm -f submission.*
	rm -f baseline_model_results.csv
	rm -f sample_recommendations.csv
	rm -f simple_baseline_results.csv

# 创建目录结构
setup:
	mkdir -p src/{data,models,features,utils} data/{raw,processed,models} docs/{images,reports} config logs cache

# 查看项目结构
tree:
	@echo "项目结构："
	@tree -I '__pycache__|*.pyc|logs|cache' -L 3

# 运行测试
test:
	pytest tests/ -v

# 代码格式化
format:
	black src/ config/
	flake8 src/ config/

# 性能分析
profile:
	python -m memory_profiler src/models/simple_baseline_models.py