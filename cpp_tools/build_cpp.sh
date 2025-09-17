#!/bin/bash

echo "🔨 编译高性能用户特征提取器"

# 创建输出目录
mkdir -p /mnt/data/tianchi_features

# 编译C++程序
echo "📦 编译中..."
g++ -O3 -march=native -std=c++17 \
    -pthread \
    -Wall -Wextra \
    -o fast_user_extractor \
    fast_user_extractor.cpp

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "📊 程序信息:"
    ls -lh fast_user_extractor

    echo ""
    echo "🚀 使用方法:"
    echo "  ./fast_user_extractor"
    echo ""
    echo "📁 输出文件将保存到: /mnt/data/tianchi_features/user_features_cpp.csv"
else
    echo "❌ 编译失败"
    exit 1
fi