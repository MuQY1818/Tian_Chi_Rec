#!/bin/bash

echo "🔧 编译C++快速传统推荐算法..."

# 编译优化选项
g++ -std=c++17 -O3 -march=native -mtune=native -fopenmp \
    -Wall -Wextra \
    cpp_tools/fast_traditional_recommender.cpp \
    -o cpp_traditional_recommender

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo ""
    echo "🚀 开始运行..."
    echo ""

    # 运行程序
    time ./cpp_traditional_recommender

    echo ""
    echo "📊 检查输出文件..."
    if [ -f "cpp_traditional_submission.txt" ]; then
        file_size=$(stat -f%z cpp_traditional_submission.txt 2>/dev/null || stat -c%s cpp_traditional_submission.txt)
        line_count=$(wc -l < cpp_traditional_submission.txt)

        echo "   📁 文件: cpp_traditional_submission.txt"
        echo "   📏 大小: $((file_size / 1024)) KB"
        echo "   📊 行数: $line_count"
        echo "   ✅ 可直接提交到比赛平台"
    else
        echo "   ❌ 输出文件未生成"
    fi
else
    echo "❌ 编译失败!"
    exit 1
fi