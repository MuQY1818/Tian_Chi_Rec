#!/usr/bin/env python3
"""
快速机器学习流水线
基于现有39维用户特征的快速推荐系统
"""

import sys
import os
import time
import subprocess

# 添加路径
sys.path.append('src')
sys.path.append('scripts/feature_engineering')
sys.path.append('scripts/modeling')

def run_step(step_name, module_name, description):
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"🚀 步骤: {step_name}")
    print(f"📝 描述: {description}")
    print(f"{'='*60}")
    print(f"⏰ 开始时间: {time.strftime('%H:%M:%S')}")

    start_time = time.time()

    try:
        print(f"🔄 正在执行 {module_name}...")

        if module_name == "simple_item_features":
            from simple_item_features import main
            main()
        elif module_name == "fast_sample_generator":
            from fast_sample_generator import main
            main()
        elif module_name == "fast_lightgbm_trainer":
            from fast_lightgbm_trainer import main
            main()
        elif module_name == "ultra_fast_recommendation":
            from ultra_fast_recommendation import main
            main()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'🎉'*20}")
        print(f"✅ {step_name} 成功完成!")
        print(f"⏱️  耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"⏰ 完成时间: {time.strftime('%H:%M:%S')}")
        print(f"{'🎉'*20}\n")
        return True

    except Exception as e:
        print(f"\n{'❌'*20}")
        print(f"❌ {step_name} 执行失败!")
        print(f"💥 错误类型: {type(e).__name__}")
        print(f"💥 错误信息: {str(e)}")
        print(f"{'❌'*20}\n")
        import traceback
        traceback.print_exc()
        return False

def check_prerequisites():
    """检查前置条件"""
    print("🔍 检查前置条件...")

    # 检查用户特征文件
    user_feature_file = "/mnt/data/tianchi_features/user_features_cpp.csv"
    if not os.path.exists(user_feature_file):
        print(f"❌ 缺少关键文件: {user_feature_file}")
        print(f"请先运行: python run.py features-cpp")
        return False

    file_size = os.path.getsize(user_feature_file) / (1024**2)
    print(f"  ✅ 用户特征文件: {file_size:.1f} MB")

    # 检查数据文件
    data_files = [
        "dataset/preprocess_16to18/data_1216.txt",
        "dataset/preprocess_16to18/data_1217.txt",
        "dataset/preprocess_16to18/data_1218.txt"
    ]

    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少数据文件: {file_path}")
            return False

    print(f"  ✅ 16-18号数据文件齐全")

    # 检查Python包
    try:
        import lightgbm
        import pandas
        import numpy
        import sklearn
        print(f"  ✅ Python依赖包完整")
    except ImportError as e:
        print(f"❌ 缺少Python包: {e}")
        return False

    print("✅ 前置条件检查通过!")
    return True

def main():
    """主函数"""
    print(f"{'='*70}")
    print("🚀 快速机器学习推荐流水线 🚀")
    print("🎯 目标: 基于现有39维用户特征的快速推荐")
    print("⚡ 特点: 速度优化，充分利用已有特征")
    print(f"{'='*70}")
    print(f"⏰ 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # 1. 检查前置条件
    print("\n📋 第0步: 环境检查")
    print("━" * 50)
    if not check_prerequisites():
        print("❌ 前置条件不满足，流程终止")
        return

    # 2. 定义执行步骤
    steps = [
        {
            'name': '简化商品特征提取',
            'module': 'simple_item_features',
            'description': '快速提取核心商品特征（流行度、购买率等）'
        },
        {
            'name': '快速样本生成',
            'module': 'fast_sample_generator',
            'description': '基于39维用户特征快速构建训练样本'
        },
        {
            'name': '快速模型训练',
            'module': 'fast_lightgbm_trainer',
            'description': '优化参数的LightGBM快速训练'
        },
        {
            'name': '超快速推荐生成',
            'module': 'ultra_fast_recommendation',
            'description': '基于规则的超快速推荐生成(每用户1-3个)'
        }
    ]

    # 3. 执行流程
    print(f"\n🎯 开始执行 {len(steps)} 步流程")
    print(f"预计总耗时: 5-8分钟（含超快速推荐）")
    print("━" * 70)

    start_time = time.time()
    success_count = 0

    for i, step in enumerate(steps, 1):
        # 显示整体进度
        progress_bar = "█" * i + "░" * (len(steps) - i)
        progress_pct = (i / len(steps)) * 100
        print(f"\n📊 总体进度: [{progress_bar}] {progress_pct:.0f}% ({i}/{len(steps)})")

        step_success = run_step(
            f"第{i}步: {step['name']}",
            step['module'],
            step['description']
        )

        if step_success:
            success_count += 1
            elapsed = time.time() - start_time
            remaining_steps = len(steps) - i
            if i > 1:
                avg_time = elapsed / i
                estimated_remaining = avg_time * remaining_steps
                print(f"🕐 已耗时: {elapsed/60:.1f}分钟, 预计剩余: {estimated_remaining/60:.1f}分钟")
        else:
            print(f"\n💥 流程在第{i}步失败，终止执行")
            break

    # 4. 流程总结
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'🏁'*20}")
    print(f"📊 快速流程执行总结")
    print(f"{'━'*70}")
    print(f"✅ 成功步骤: {success_count}/{len(steps)}")
    print(f"⏱️  总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
    print(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if success_count == len(steps):
        print(f"\n{'🎉'*25}")
        print(f"🎊 快速流程执行成功! 🎊")
        print(f"{'🎉'*25}")
        print(f"📁 输出目录: /mnt/data/tianchi_features/")
        print(f"📝 提交文件: /mnt/data/tianchi_features/ultra_fast_submission.txt")
        print(f"⚡ 速度优势: 比完整流程快10-20倍")
        print(f"🏆 可用于比赛提交!")
        print(f"{'🎉'*25}")

        # 性能对比
        print(f"\n📈 性能对比:")
        print(f"  完整流程: ~60-90分钟")
        print(f"  超快速流程: ~5-8分钟")
        print(f"  推荐数量: 每用户1-3个（精准推荐）")
        print(f"  特征使用: 充分利用现有39维用户特征")
        print(f"  推荐策略: 规则优先+模型辅助")

    else:
        print(f"\n❌ 流程未完全成功，请检查错误信息")

    # 5. 输出文件检查
    print(f"\n📁 输出文件检查:")
    output_files = [
        ("/mnt/data/tianchi_features/simple_item_features.csv", "简化商品特征"),
        ("/mnt/data/tianchi_features/fast_training_samples.csv", "快速训练样本"),
        ("/mnt/data/tianchi_features/fast_lightgbm_model.pkl", "快速训练模型"),
        ("/mnt/data/tianchi_features/ultra_fast_submission.txt", "超快速提交文件")
    ]

    for file_path, description in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"  ✅ {description}: {file_size:.1f} MB")
        else:
            print(f"  ❌ {description}: 不存在")

    print(f"\n💡 使用建议:")
    print(f"  1. 快速流程适合快速迭代和测试")
    print(f"  2. 如需更高精度可运行完整流程")
    print(f"  3. 两种方案可以ensemble融合")

if __name__ == "__main__":
    main()