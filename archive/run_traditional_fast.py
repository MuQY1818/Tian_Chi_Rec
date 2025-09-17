#!/usr/bin/env python3
"""
快速传统推荐算法
使用预处理好的16-18号数据，直接生成推荐
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import time
from tqdm import tqdm

def load_preprocessed_data():
    """加载预处理的16-18号数据"""
    print("=" * 60)
    print("📂 步骤1: 加载预处理数据")
    print("=" * 60)

    data_dir = "dataset/preprocess_16to18"
    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

    all_data = []
    total_rows = 0

    for i, day in enumerate([16, 17, 18], 1):
        filename = f"data_12{day}.txt"
        file_path = os.path.join(data_dir, filename)

        print(f"\n📅 [{i}/3] 加载 {filename}...")

        # 显示文件大小
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        print(f"   📏 文件大小: {file_size:.1f} MB")

        start_time = time.time()
        df = pd.read_csv(file_path, sep='\t', names=columns)
        load_time = time.time() - start_time

        df['day'] = day
        all_data.append(df)
        total_rows += len(df)

        print(f"   📊 行数: {len(df):,}")
        print(f"   ⏱️  加载时间: {load_time:.2f}秒")

        # 进度条
        progress = "█" * i + "░" * (3 - i)
        print(f"   📈 进度: [{progress}] {i*100//3}%")

    print(f"\n✅ 数据加载完成")
    print(f"   📊 总数据量: {total_rows:,} 行")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"   💾 合并后大小: {len(combined_df):,} 行")

    return combined_df

def load_item_subset():
    """加载商品子集P"""
    print("\n" + "=" * 60)
    print("🛍️  步骤2: 加载商品子集P")
    print("=" * 60)

    item_file = "dataset/tianchi_fresh_comp_train_item_online.txt"
    columns = ["item_id", "item_geohash", "item_category"]

    print(f"📁 加载文件: {item_file}")

    # 检查文件大小
    if os.path.exists(item_file):
        file_size = os.path.getsize(item_file) / (1024**2)  # MB
        print(f"📏 文件大小: {file_size:.1f} MB")
    else:
        print(f"❌ 文件不存在: {item_file}")
        return set()

    start_time = time.time()
    item_df = pd.read_csv(item_file, sep='\t', names=columns)
    load_time = time.time() - start_time

    valid_items = set(item_df['item_id'].tolist())

    print(f"📊 商品数量: {len(valid_items):,}")
    print(f"⏱️  加载时间: {load_time:.2f}秒")
    print(f"🏷️  类别数量: {item_df['item_category'].nunique()}")
    print(f"📍 有地理信息: {(item_df['item_geohash'].notna()).sum():,}")

    return valid_items

class FastTraditionalRecommender:
    """快速传统推荐器"""

    def __init__(self):
        self.user_item_matrix = defaultdict(lambda: defaultdict(int))
        self.user_purchases = defaultdict(set)
        self.item_popularity = defaultdict(int)
        self.item_purchase_count = defaultdict(int)
        self.user_categories = defaultdict(lambda: defaultdict(int))

    def fit(self, df, valid_items):
        """训练模型"""
        print("\n" + "=" * 60)
        print("🤖 步骤3: 训练传统推荐模型")
        print("=" * 60)

        start_time = time.time()

        # 只保留商品子集P中的数据
        print("📊 过滤数据到商品子集P...")
        filter_start = time.time()
        df = df[df['item_id'].isin(valid_items)].copy()
        filter_time = time.time() - filter_start
        print(f"   📏 过滤后数据量: {len(df):,}")
        print(f"   ⏱️  过滤时间: {filter_time:.2f}秒")

        # 使用16-17号数据训练，18号数据作为验证
        print("\n🗓️  划分训练和验证集...")
        train_df = df[df['day'].isin([16, 17])].copy()
        val_df = df[df['day'] == 18].copy()

        print(f"   📊 训练数据量 (16-17号): {len(train_df):,}")
        print(f"   📊 验证数据量 (18号): {len(val_df):,}")

        # 构建用户-商品交互矩阵
        print("\n🔧 构建交互矩阵...")
        matrix_start = time.time()

        processed_count = 0
        # 更频繁的进度更新：每1%或每10万行更新一次
        update_interval = min(100000, max(1, len(train_df) // 100))
        print(f"📊 矩阵构建进度更新间隔: 每{update_interval:,}行")

        for idx, row in train_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            behavior = int(row['behavior_type'])
            category = int(row['item_category'])

            # 记录交互（所有行为）
            self.user_item_matrix[user_id][item_id] += 1

            # 记录购买行为
            if behavior == 4:
                self.user_purchases[user_id].add(item_id)
                self.item_purchase_count[item_id] += 1

            # 商品流行度
            self.item_popularity[item_id] += 1

            # 用户类别偏好
            self.user_categories[user_id][category] += 1

            processed_count += 1

            # 显示进度 - 更频繁更新
            if processed_count % update_interval == 0 or processed_count == len(train_df) or processed_count == 1:
                progress_pct = (processed_count / len(train_df)) * 100
                progress_bars = int(progress_pct / 5)  # 20个进度条
                progress_str = "█" * progress_bars + "░" * (20 - progress_bars)
                elapsed = time.time() - matrix_start
                eta = (elapsed / processed_count) * (len(train_df) - processed_count) if processed_count > 0 else 0
                speed = processed_count / elapsed if elapsed > 0 else 0

                # 实时更新显示
                if processed_count > 1:
                    print("\033[4A", end="")  # 向上移动4行

                print(f"   📈 进度: [{progress_str}] {progress_pct:.1f}%")
                print(f"   📊 已处理: {processed_count:,}/{len(train_df):,} 行")
                print(f"   🚀 速度: {speed:.0f} 行/秒")
                print(f"   ⏱️  已用时: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")

                import sys
                sys.stdout.flush()

        matrix_time = time.time() - matrix_start
        total_time = time.time() - start_time

        print(f"\n✅ 模型训练完成!")
        print(f"   👥 用户数: {len(self.user_item_matrix):,}")
        print(f"   🛍️  商品数: {len(self.item_popularity):,}")
        print(f"   🔗 交互总数: {sum(sum(items.values()) for items in self.user_item_matrix.values()):,}")
        print(f"   💰 购买总数: {sum(len(items) for items in self.user_purchases.values()):,}")
        print(f"   ⏱️  矩阵构建时间: {matrix_time:.1f}秒")
        print(f"   ⏱️  总训练时间: {total_time:.1f}秒")

        return val_df

    def get_user_category_preference(self, user_id):
        """获取用户最偏好的类别"""
        if user_id in self.user_categories:
            return max(self.user_categories[user_id].items(), key=lambda x: x[1])[0]
        return None

    def popularity_recommend(self, user_id, top_k=5):
        """基于流行度推荐"""
        # 获取用户已交互的商品
        interacted_items = set(self.user_item_matrix[user_id].keys())

        # 获取用户偏好类别
        preferred_category = self.get_user_category_preference(user_id)

        # 候选商品：热门且未交互的
        candidates = []
        for item_id, popularity in self.item_popularity.items():
            if item_id not in interacted_items and popularity >= 5:  # 至少5次交互
                score = popularity

                # 如果商品属于用户偏好类别，加权
                if preferred_category is not None:
                    # 这里需要商品类别信息，简化处理
                    score *= 1.2  # 简单加权

                candidates.append((item_id, score))

        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def itemcf_recommend(self, user_id, top_k=5):
        """基于物品协同过滤推荐"""
        user_items = self.user_item_matrix[user_id]
        if not user_items:
            return []

        # 计算候选商品分数
        candidate_scores = defaultdict(float)

        for item_id, rating in user_items.items():
            # 寻找相似商品（简化版：共现频率）
            for other_user_id, other_items in self.user_item_matrix.items():
                if other_user_id != user_id and item_id in other_items:
                    # 推荐该用户的其他商品
                    for other_item_id, other_rating in other_items.items():
                        if other_item_id not in user_items:  # 未交互过的商品
                            similarity = min(rating, other_rating) / max(rating, other_rating)
                            candidate_scores[other_item_id] += similarity

        # 按分数排序
        candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def hybrid_recommend(self, user_id, top_k=5):
        """混合推荐策略"""
        # 获取流行度推荐
        pop_recs = self.popularity_recommend(user_id, top_k * 2)

        # 获取协同过滤推荐
        cf_recs = self.itemcf_recommend(user_id, top_k * 2)

        # 简单融合：流行度0.6 + 协同过滤0.4
        final_scores = defaultdict(float)

        for item_id, score in pop_recs:
            final_scores[item_id] += 0.6 * score

        for item_id, score in cf_recs:
            final_scores[item_id] += 0.4 * score

        # 排序并返回
        final_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return final_candidates[:top_k]

    def recommend_for_all_users(self, top_k=3):
        """为所有用户生成推荐"""
        print("\n" + "=" * 60)
        print(f"🎯 步骤4: 生成推荐 (每用户top-{top_k})")
        print("=" * 60)

        start_time = time.time()
        recommendations = {}
        total_users = len(self.user_item_matrix)
        processed_users = 0

        print(f"👥 待处理用户数: {total_users:,}")

        # 更频繁的进度更新：至少每1000个用户或每1%更新一次
        update_interval = min(1000, max(1, total_users // 100))
        print(f"📊 进度更新间隔: 每{update_interval:,}个用户")

        for user_id in self.user_item_matrix.keys():
            recs = self.hybrid_recommend(user_id, top_k)
            if recs:
                recommendations[user_id] = [item_id for item_id, score in recs]

            processed_users += 1

            # 显示进度 - 更频繁更新
            if processed_users % update_interval == 0 or processed_users == total_users or processed_users == 1:
                progress_pct = (processed_users / total_users) * 100
                progress_bars = int(progress_pct / 5)  # 20个进度条
                progress_str = "█" * progress_bars + "░" * (20 - progress_bars)

                elapsed = time.time() - start_time
                eta = (elapsed / processed_users) * (total_users - processed_users) if processed_users > 0 else 0
                speed = processed_users / elapsed if elapsed > 0 else 0

                # 清屏并重新打印进度（实现滚动效果）
                if processed_users > 1:
                    print("\033[5A", end="")  # 向上移动5行
                    print("\033[K", end="")   # 清除当前行

                print(f"   📈 进度: [{progress_str}] {progress_pct:.1f}%")
                print(f"   👥 已处理: {processed_users:,}/{total_users:,} 用户")
                print(f"   🚀 速度: {speed:.1f} 用户/秒")
                print(f"   ⏱️  已用时: {elapsed:.1f}秒, 预计剩余: {eta:.1f}秒")

                # 显示最近处理的用户ID
                print(f"   🆔 当前用户: {user_id}")

                import sys
                sys.stdout.flush()  # 立即输出

        total_time = time.time() - start_time
        avg_recs = sum(len(recs) for recs in recommendations.values()) / len(recommendations) if recommendations else 0

        print(f"✅ 推荐生成完成!")
        print(f"   👥 成功推荐用户数: {len(recommendations):,}")
        print(f"   📊 平均每用户推荐数: {avg_recs:.1f}")
        print(f"   ⏱️  总耗时: {total_time:.1f}秒")
        print(f"   🚀 平均速度: {len(recommendations)/total_time:.1f} 用户/秒")

        return recommendations

def evaluate_simple(recommendations, val_df):
    """简单评估"""
    print("\n" + "=" * 60)
    print("📊 步骤5: 评估推荐效果")
    print("=" * 60)

    eval_start = time.time()

    # 构建验证集用户购买记录
    print("📋 构建验证集购买记录...")
    val_purchases = defaultdict(set)
    purchase_count = 0

    for _, row in val_df.iterrows():
        if int(row['behavior_type']) == 4:  # 购买行为
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            val_purchases[user_id].add(item_id)
            purchase_count += 1

    print(f"   💰 验证集购买用户数: {len(val_purchases):,}")
    print(f"   🛒 验证集购买记录数: {purchase_count:,}")

    # 计算命中率
    print("\n🎯 计算推荐命中率...")
    hits = 0
    total_users = 0
    total_recommendations = 0
    covered_items = set()

    for user_id, rec_items in recommendations.items():
        if user_id in val_purchases:
            purchased_items = val_purchases[user_id]
            rec_set = set(rec_items)

            if rec_set & purchased_items:  # 有交集
                hits += 1

            total_users += 1

        total_recommendations += len(rec_items)
        covered_items.update(rec_items)

    hit_rate = hits / total_users if total_users > 0 else 0
    eval_time = time.time() - eval_start

    print(f"\n✅ 评估完成!")
    print(f"   🎯 命中率 (HR): {hit_rate:.4f} ({hits}/{total_users})")
    print(f"   📊 总推荐数: {total_recommendations:,}")
    print(f"   🛍️  覆盖商品数: {len(covered_items):,}")
    print(f"   👥 推荐用户数: {len(recommendations):,}")
    print(f"   📈 平均每用户推荐: {total_recommendations/len(recommendations):.1f}")
    print(f"   ⏱️  评估耗时: {eval_time:.2f}秒")

    return hit_rate

def export_submission(recommendations, filename="traditional_fast_submission.txt"):
    """导出提交文件"""
    print("\n" + "=" * 60)
    print("💾 步骤6: 导出提交文件")
    print("=" * 60)

    export_start = time.time()
    print(f"📁 输出文件: {filename}")

    total_recs = 0

    with open(filename, 'w') as f:
        for user_id, item_list in recommendations.items():
            for item_id in item_list:
                f.write(f"{user_id}\t{item_id}\n")
                total_recs += 1

    export_time = time.time() - export_start

    # 检查文件大小
    file_size = os.path.getsize(filename) / (1024**2)  # MB

    print(f"\n✅ 文件导出完成!")
    print(f"   📊 总推荐数: {total_recs:,}")
    print(f"   👥 用户数: {len(recommendations):,}")
    print(f"   📈 平均每用户推荐数: {total_recs/len(recommendations):.1f}")
    print(f"   📏 文件大小: {file_size:.2f} MB")
    print(f"   ⏱️  导出耗时: {export_time:.2f}秒")
    print(f"   📁 文件路径: {os.path.abspath(filename)}")

    return filename

def main():
    """主函数"""
    print("=" * 70)
    print("🚀 快速传统推荐算法")
    print("🎯 基于16-17号训练，18号验证")
    print("⚡ 混合策略: 流行度(0.6) + 协同过滤(0.4)")
    print("=" * 70)
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start_time = time.time()

    try:
        # 1. 加载数据
        df = load_preprocessed_data()
        valid_items = load_item_subset()

        # 2. 训练模型
        recommender = FastTraditionalRecommender()
        val_df = recommender.fit(df, valid_items)

        # 3. 生成推荐
        recommendations = recommender.recommend_for_all_users(top_k=3)

        # 4. 评估
        hit_rate = evaluate_simple(recommendations, val_df)

        # 5. 导出
        filename = export_submission(recommendations)

        # 总结
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        print("\n" + "🎉" * 25)
        print("🎊 传统推荐算法运行成功! 🎊")
        print("🎉" * 25)
        print(f"⏱️  总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
        print(f"🎯 命中率: {hit_rate:.4f}")
        print(f"📊 推荐策略: 流行度(0.6) + 协同过滤(0.4)")
        print(f"👥 每用户推荐数: 3个")
        print(f"📁 提交文件: {filename}")
        print(f"⏰ 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n📈 性能统计:")
        print(f"   数据处理速度: 约{len(df)/total_duration:.0f} 行/秒")
        print(f"   推荐生成速度: 约{len(recommendations)/total_duration:.0f} 用户/秒")

        print(f"\n💡 使用建议:")
        print(f"   1. 可以直接提交 {filename} 到比赛平台")
        print(f"   2. 传统算法速度快，适合快速迭代")
        print(f"   3. 可以与机器学习方法进行ensemble")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()