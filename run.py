#!/usr/bin/env python3
"""
天池推荐算法项目主控制脚本
"""
import sys
import subprocess
import os
from pathlib import Path

def show_help():
    """显示帮助信息"""
    print("""
🎯 天池推荐算法项目控制台

📋 可用命令:

数据预处理:
  run.py prep-small         # 创建小数据集
  run.py count-users        # 统计用户数量

特征工程:
  run.py features-python    # Python版本特征提取
  run.py features-cpp       # C++版本特征提取 (推荐)
  run.py build-cpp          # 编译C++工具

机器学习流程:
  run.py fast-ml            # 快速机器学习流程 (推荐)
  run.py ml-pipeline        # 完整机器学习流程
  run.py item-features      # 商品特征提取
  run.py training-samples   # 训练样本生成
  run.py lightgbm-train     # LightGBM模型训练
  run.py recommendation     # 推荐生成

传统模型:
  run.py train-traditional  # 传统推荐算法训练
  run.py train-simple       # 简化训练

提交生成:
  run.py submission-fast    # 快速提交生成

项目管理:
  run.py clean              # 清理临时文件
  run.py status             # 显示项目状态

示例:
  run.py features-cpp       # 运行C++特征提取 (15分钟)
  run.py train-traditional  # 训练传统模型
""")

def run_command(cmd, description):
    """运行命令"""
    print(f"🚀 {description}")
    print(f"📝 命令: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode == 0:
        print(f"✅ {description} 完成")
    else:
        print(f"❌ {description} 失败")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1]

    # 确保在正确目录
    os.chdir(Path(__file__).parent)

    if command == "prep-small":
        run_command("python scripts/preprocessing/create_small_dataset.py",
                   "创建小数据集")

    elif command == "count-users":
        run_command("python scripts/preprocessing/quick_user_count.py",
                   "统计用户数量")

    elif command == "features-python":
        run_command("python scripts/feature_engineering/user_feature_builder.py",
                   "Python版本特征提取")

    elif command == "features-cpp":
        # 先检查是否需要编译
        cpp_exe = Path("cpp_tools/fast_user_extractor")
        if not cpp_exe.exists():
            print("📦 C++程序不存在，开始编译...")
            run_command("cd cpp_tools && ./build_cpp.sh", "编译C++工具")

        run_command("cd cpp_tools && ./fast_user_extractor",
                   "C++版本特征提取")

    elif command == "build-cpp":
        run_command("cd cpp_tools && ./build_cpp.sh", "编译C++工具")

    elif command == "train-traditional":
        run_command("python scripts/modeling/traditional_train.py",
                   "传统推荐算法训练")

    elif command == "train-simple":
        run_command("python scripts/modeling/simple_train.py",
                   "简化模型训练")

    elif command == "fast-ml":
        run_command("python scripts/fast_ml_pipeline.py",
                   "快速机器学习流程")

    elif command == "ml-pipeline":
        run_command("python scripts/run_ml_pipeline.py",
                   "完整机器学习流程")

    elif command == "item-features":
        run_command("python scripts/feature_engineering/item_feature_extractor.py",
                   "商品特征提取")

    elif command == "training-samples":
        run_command("python scripts/feature_engineering/training_sample_generator.py",
                   "训练样本生成")

    elif command == "lightgbm-train":
        run_command("python scripts/modeling/lightgbm_trainer.py",
                   "LightGBM模型训练")

    elif command == "recommendation":
        run_command("python scripts/modeling/recommendation_generator.py",
                   "推荐生成")

    elif command == "submission-fast":
        run_command("python scripts/submission/full_submission_generator.py",
                   "快速提交生成")

    elif command == "clean":
        print("🧹 清理临时文件...")
        # 清理Python缓存
        run_command("find . -name '__pycache__' -type d -exec rm -rf {} +",
                   "清理Python缓存")
        # 清理checkpoint文件
        run_command("find . -name '*.pkl' -delete", "清理checkpoint文件")
        print("✅ 清理完成")

    elif command == "status":
        show_project_status()

    else:
        print(f"❌ 未知命令: {command}")
        show_help()

def show_project_status():
    """显示项目状态"""
    print("📊 项目状态检查")
    print("=" * 50)

    # 检查数据集
    dataset_files = [
        "dataset/tianchi_fresh_comp_train_user_online_partA.txt",
        "dataset/tianchi_fresh_comp_train_user_online_partB.txt",
        "dataset/tianchi_fresh_comp_train_item_online.txt"
    ]

    print("\n📁 数据集状态:")
    for file in dataset_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**3)
            print(f"  ✅ {os.path.basename(file)}: {size:.1f}GB")
        else:
            print(f"  ❌ {os.path.basename(file)}: 缺失")

    # 检查C++工具
    print("\n🛠️  C++工具状态:")
    cpp_files = [
        "cpp_tools/fast_user_extractor.cpp",
        "cpp_tools/fast_user_extractor",
        "cpp_tools/build_cpp.sh"
    ]

    for file in cpp_files:
        if os.path.exists(file):
            print(f"  ✅ {os.path.basename(file)}: 存在")
        else:
            print(f"  ❌ {os.path.basename(file)}: 缺失")

    # 检查输出文件
    print("\n📤 输出文件状态:")
    output_files = [
        "outputs/submission.txt",
        "/mnt/data/tianchi_features/user_features_cpp.csv"
    ]

    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**2)
            print(f"  ✅ {os.path.basename(file)}: {size:.1f}MB")
        else:
            print(f"  ❌ {os.path.basename(file)}: 不存在")

    # 推荐下一步操作
    print("\n💡 推荐操作:")
    if not os.path.exists("/mnt/data/tianchi_features/user_features_cpp.csv"):
        print("  1. 运行: python run.py features-cpp  (生成用户特征)")
    if not os.path.exists("outputs/submission.txt"):
        print("  2. 运行: python run.py train-traditional  (训练模型)")
    print("  3. 运行: python run.py submission-fast  (生成提交文件)")

if __name__ == "__main__":
    main()