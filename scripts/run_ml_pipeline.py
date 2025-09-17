#!/usr/bin/env python3
"""
机器学习推荐系统完整流程
"""

import sys
import os
import time
import subprocess

# 添加路径
sys.path.append('src')
sys.path.append('scripts/feature_engineering')
sys.path.append('scripts/modeling')

def run_step(step_name, script_path, description):
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"🚀 步骤: {step_name}")
    print(f"📝 描述: {description}")
    print(f"📁 脚本: {script_path}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        if script_path.endswith('.py'):
            result = subprocess.run([sys.executable, script_path],
                                  capture_output=False,
                                  text=True,
                                  check=True)
        else:
            # 如果是模块，直接导入运行
            if script_path == "item_feature_extractor":
                from item_feature_extractor import main
                main()
            elif script_path == "training_sample_generator":
                from training_sample_generator import main
                main()
            elif script_path == "lightgbm_trainer":
                from lightgbm_trainer import main
                main()
            elif script_path == "recommendation_generator":
                from recommendation_generator import main
                main()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ {step_name} 完成!")
        print(f"⏱️  耗时: {duration:.1f} 秒")
        return True

    except Exception as e:
        print(f"\n❌ {step_name} 失败!")
        print(f"💥 错误: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("🔍 检查运行环境...")

    # 检查Python包
    required_packages = ['pandas', 'numpy', 'lightgbm', 'scikit-learn', 'tqdm', 'joblib']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 缺少必要包: {missing_packages}")
        print(f"请运行: pip install {' '.join(missing_packages)}")
        return False

    # 检查数据文件
    required_files = [
        "dataset/preprocess_16to18/data_1216.txt",
        "dataset/preprocess_16to18/data_1217.txt",
        "dataset/preprocess_16to18/data_1218.txt",
        "dataset/tianchi_fresh_comp_train_item_online.txt"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ 缺少数据文件: {missing_files}")
        return False

    # 检查用户特征文件
    user_feature_file = "/mnt/data/tianchi_features/user_features_cpp.csv"
    if not os.path.exists(user_feature_file):
        print(f"❌ 缺少用户特征文件: {user_feature_file}")
        print(f"请先运行 C++ 特征提取器")
        return False

    print("✅ 环境检查通过!")
    return True

def main():
    """主函数"""
    print("=== 机器学习推荐系统完整流程 ===")
    print("🎯 目标：从特征工程到最终推荐的完整流程")

    # 1. 环境检查
    if not check_dependencies():
        print("❌ 环境检查失败，流程终止")
        return

    # 2. 定义执行步骤
    steps = [
        {
            'name': '商品特征提取',
            'script': 'item_feature_extractor',
            'description': '从16-18号数据提取商品特征（流行度、转化率等）'
        },
        {
            'name': '训练样本生成',
            'script': 'training_sample_generator',
            'description': '构建用户-商品交互预测训练样本（正负样本）'
        },
        {
            'name': 'LightGBM模型训练',
            'script': 'lightgbm_trainer',
            'description': '训练用户-商品购买概率预测模型'
        },
        {
            'name': '推荐生成',
            'script': 'recommendation_generator',
            'description': '生成最终推荐列表和提交文件'
        }
    ]

    # 3. 执行流程
    start_time = time.time()
    success_count = 0

    for i, step in enumerate(steps, 1):
        step_success = run_step(
            f"{i}. {step['name']}",
            step['script'],
            step['description']
        )

        if step_success:
            success_count += 1
        else:
            print(f"\n💥 流程在第{i}步失败，终止执行")
            break

    # 4. 流程总结
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"📊 流程执行总结")
    print(f"{'='*60}")
    print(f"✅ 成功步骤: {success_count}/{len(steps)}")
    print(f"⏱️  总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")

    if success_count == len(steps):
        print(f"\n🎉 完整流程执行成功!")
        print(f"📁 输出目录: /mnt/data/tianchi_features/")
        print(f"📝 提交文件: /mnt/data/tianchi_features/final_submission.csv")
        print(f"🏆 可用于比赛提交!")
    else:
        print(f"\n❌ 流程未完全成功，请检查错误信息")

    # 5. 文件检查
    print(f"\n📁 输出文件检查:")
    output_files = [
        "/mnt/data/tianchi_features/item_features.csv",
        "/mnt/data/tianchi_features/training_samples.csv",
        "/mnt/data/tianchi_features/lightgbm_model.pkl",
        "/mnt/data/tianchi_features/final_submission.csv"
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"  ✅ {os.path.basename(file_path)}: {file_size:.1f} MB")
        else:
            print(f"  ❌ {os.path.basename(file_path)}: 不存在")

if __name__ == "__main__":
    main()