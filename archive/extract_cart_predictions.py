#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从data_1218.txt中提取"只加购物车但没购买"的商品作为12月19日的购买预测结果
并与白名单文件进行过滤，符合阿里巴巴移动电商推荐大赛提交格式要求
"""

import pandas as pd
from collections import defaultdict

def extract_cart_predictions():
    """
    从data_1218.txt中提取"只加购物车但没购买"的用户-商品对作为预测结果
    """
    input_file = r'c:\Users\ASUS\Desktop\tianchi\preprocess\data_1218.txt'
    whitelist_file = r'c:\Users\ASUS\Desktop\tianchi\tianchi_fresh_comp_train_item_online.txt'
    output_file = r'c:\Users\ASUS\Desktop\tianchi\cart_only_predictions_1219.txt'
    
    print("开始读取data_1218.txt文件...")
    
    # 用于存储不同行为的用户-商品对
    cart_items = set()  # 购物车行为
    purchase_items = set()  # 购买行为
    
    # 分块读取大文件
    chunk_size = 100000
    total_lines = 0
    cart_lines = 0
    purchase_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                chunk.append(line.strip())
                total_lines += 1
                
                if len(chunk) >= chunk_size:
                    # 处理当前块
                    chunk_cart_count, chunk_purchase_count = process_chunk(chunk, cart_items, purchase_items)
                    cart_lines += chunk_cart_count
                    purchase_lines += chunk_purchase_count
                    chunk = []
                    
                    if total_lines % 1000000 == 0:
                        print(f"已处理 {total_lines:,} 行数据，找到 {cart_lines:,} 个购物车行为，{purchase_lines:,} 个购买行为")
            
            # 处理最后一个不完整的chunk
            if chunk:
                chunk_cart_count, chunk_purchase_count = process_chunk(chunk, cart_items, purchase_items)
                cart_lines += chunk_cart_count
                purchase_lines += chunk_purchase_count
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    print(f"\n数据处理完成:")
    print(f"总行数: {total_lines:,}")
    print(f"购物车行为数: {cart_lines:,}")
    print(f"购买行为数: {purchase_lines:,}")
    
    # 计算"只加购物车但没购买"的用户-商品对
    cart_only_items = cart_items - purchase_items
    print(f"只加购物车但没购买的用户-商品对数量: {len(cart_only_items):,}")
    
    # 读取白名单文件
    print("\n读取白名单文件...")
    try:
        whitelist_items = set()
        with open(whitelist_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    item_id = parts[0]
                    whitelist_items.add(item_id)
        
        print(f"白名单商品数量: {len(whitelist_items):,}")
        
        # 过滤：只保留在白名单中的商品
        filtered_predictions = set()
        for user_id, item_id in cart_only_items:
            if item_id in whitelist_items:
                filtered_predictions.add((user_id, item_id))
        
        print(f"过滤后的预测结果数量: {len(filtered_predictions):,}")
        
    except Exception as e:
        print(f"读取白名单文件时出错: {e}")
        return
    
    # 写入预测结果文件
    print(f"\n正在写入预测结果到 {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for user_id, item_id in sorted(filtered_predictions):
                f.write(f"{user_id}\t{item_id}\n")
        
        print(f"预测结果已保存到: {output_file}")
        print(f"预测的用户-商品对数量: {len(filtered_predictions):,}")
        
    except Exception as e:
        print(f"写入文件时出错: {e}")

def process_chunk(chunk, cart_items, purchase_items):
    """
    处理数据块，分别收集购物车行为（behavior_type=3）和购买行为（behavior_type=4）
    """
    cart_count = 0
    purchase_count = 0
    
    for line in chunk:
        if not line:
            continue
            
        try:
            parts = line.split('\t')
            if len(parts) >= 6:
                user_id = parts[0]
                item_id = parts[1]
                behavior_type = parts[2]
                
                if behavior_type == '3':  # 购物车行为
                    cart_items.add((user_id, item_id))
                    cart_count += 1
                elif behavior_type == '4':  # 购买行为
                    purchase_items.add((user_id, item_id))
                    purchase_count += 1
                    
        except Exception as e:
            print(f"处理行时出错: {line[:100]}... 错误: {e}")
            continue
    
    return cart_count, purchase_count

def analyze_predictions():
    """
    分析预测结果文件
    """
    output_file = r'c:\Users\ASUS\Desktop\tianchi\cart_only_predictions_1219.txt'
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n=== 预测结果分析 ===")
        print(f"预测文件: {output_file}")
        print(f"总预测数量: {len(lines):,}")
        
        # 统计用户数和商品数
        users = set()
        items = set()
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                users.add(parts[0])
                items.add(parts[1])
        
        print(f"涉及用户数: {len(users):,}")
        print(f"涉及商品数: {len(items):,}")
        print(f"平均每用户预测商品数: {len(lines)/len(users):.2f}")
        
        # 显示前10行
        print("\n前10行预测结果:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1}: {line.strip()}")
            
    except Exception as e:
        print(f"分析预测结果时出错: {e}")

if __name__ == "__main__":
    print("=== 阿里巴巴移动电商推荐大赛 - 购物车预测提取 ===")
    print("正在从data_1218.txt提取'只加购物车但没购买'的商品作为12月19日预测结果...\n")
    
    extract_cart_predictions()
    
    print("\n" + "="*50)
    analyze_predictions()