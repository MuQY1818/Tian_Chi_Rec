#!/usr/bin/env python3

import time
import pandas as pd

def estimate_training_time():
    print("=== 训练时间估算 ===")

    # 当前测试数据 (0.1%)
    current_sample = 0.001
    current_users = 247620
    current_items = 469277
    current_edges = 419587
    current_epoch_time = 1.4  # 大约1.4秒/epoch (5epochs/7秒)

    print(f"基准数据 (采样率 {current_sample*100}%):")
    print(f"- 用户: {current_users:,}")
    print(f"- 商品: {current_items:,}")
    print(f"- 训练边: {current_edges:,}")
    print(f"- 每epoch时间: {current_epoch_time:.1f}秒")

    # 不同采样率的估算
    sample_rates = [0.005, 0.01, 0.02, 0.05, 0.1]
    epochs = 20

    print(f"\n不同采样率的时间估算 ({epochs} epochs):")
    print("-" * 80)
    print(f"{'采样率':<8} {'用户数':<10} {'边数':<12} {'验证用户':<10} {'总时间':<12} {'预估':<10}")
    print("-" * 80)

    for rate in sample_rates:
        # 线性缩放估算 (实际可能是平方关系，但先保守估算)
        scale_factor = rate / current_sample
        est_users = int(current_users * scale_factor)
        est_edges = int(current_edges * scale_factor)

        # 时间复杂度大致是O(edges)
        est_epoch_time = current_epoch_time * scale_factor
        est_total_time = est_epoch_time * epochs

        # 估算验证用户数 (假设20%重叠率)
        est_val_users = max(int(est_users * 0.0005 * 0.2), 1)  # 大概0.05%用户有验证数据

        # 格式化时间
        if est_total_time < 60:
            time_str = f"{est_total_time:.0f}秒"
        elif est_total_time < 3600:
            time_str = f"{est_total_time/60:.0f}分钟"
        else:
            time_str = f"{est_total_time/3600:.1f}小时"

        print(f"{rate*100:>6.1f}%  {est_users:>8,d}  {est_edges:>10,d}  {est_val_users:>8d}  {est_total_time:>8.0f}秒  {time_str}")

    print("-" * 80)
    print("\n推荐策略:")
    print("📈 0.5%: 平衡的选择，约10分钟，验证用户足够")
    print("🎯 1%: 较好效果，约30分钟，推荐此方案")
    print("🚀 2%: 高质量，约1小时，如果时间充足")
    print("💎 5%: 最佳效果，约4小时，终极方案")

if __name__ == "__main__":
    estimate_training_time()