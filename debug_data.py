#!/usr/bin/env python3
"""
调试数据问题
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# 快速加载最近几天数据
def load_recent_data():
    columns = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]

    print("加载最近数据...")
    file_path = "dataset/tianchi_fresh_comp_train_user_online_partB.txt"

    chunks = []
    for chunk in pd.read_csv(file_path, sep='\t', names=columns,
                           usecols=['user_id', 'item_id', 'behavior_type', 'time'],
                           chunksize=2000000):
        chunk['date'] = chunk['time'].str[:10]
        recent_data = chunk[chunk['date'].isin(['2014-12-16', '2014-12-17', '2014-12-18'])]
        if len(recent_data) > 0:
            chunks.append(recent_data)

        if len(chunks) > 3:  # 限制数据量
            break

    return pd.concat(chunks, ignore_index=True)

def main():
    data = load_recent_data()

    print(f"数据量: {len(data):,}")
    print(f"日期分布:")
    print(data['date'].value_counts().sort_index())

    print(f"\n行为类型分布:")
    print(data['behavior_type'].value_counts().sort_index())

    # 分析16号购物车 -> 17号购买
    day16_cart = data[(data['date'] == '2014-12-16') & (data['behavior_type'] == 3)]
    day17_buy = data[(data['date'] == '2014-12-17') & (data['behavior_type'] == 4)]

    cart_pairs = set(zip(day16_cart['user_id'].astype(str), day16_cart['item_id'].astype(str)))
    buy_pairs = set(zip(day17_buy['user_id'].astype(str), day17_buy['item_id'].astype(str)))

    print(f"\n16号购物车用户-商品对: {len(cart_pairs):,}")
    print(f"17号购买用户-商品对: {len(buy_pairs):,}")

    overlap = cart_pairs & buy_pairs
    print(f"交集（16号购物车->17号购买）: {len(overlap):,}")
    print(f"转化率: {len(overlap)/len(cart_pairs)*100:.2f}%")

    # 分析17号购物车 -> 18号购买
    day17_cart = data[(data['date'] == '2014-12-17') & (data['behavior_type'] == 3)]
    day18_buy = data[(data['date'] == '2014-12-18') & (data['behavior_type'] == 4)]

    cart_pairs_17 = set(zip(day17_cart['user_id'].astype(str), day17_cart['item_id'].astype(str)))
    buy_pairs_18 = set(zip(day18_buy['user_id'].astype(str), day18_buy['item_id'].astype(str)))

    print(f"\n17号购物车用户-商品对: {len(cart_pairs_17):,}")
    print(f"18号购买用户-商品对: {len(buy_pairs_18):,}")

    overlap_17_18 = cart_pairs_17 & buy_pairs_18
    print(f"交集（17号购物车->18号购买）: {len(overlap_17_18):,}")
    print(f"转化率: {len(overlap_17_18)/len(cart_pairs_17)*100:.2f}%")

    # 分析18号购物车 -> 19号购买？
    print(f"\n18号数据:")
    day18_data = data[data['date'] == '2014-12-18']
    print(f"18号总数据: {len(day18_data):,}")
    print(f"18号行为分布:")
    print(day18_data['behavior_type'].value_counts().sort_index())

if __name__ == "__main__":
    main()