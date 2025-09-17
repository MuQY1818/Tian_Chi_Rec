#!/bin/bash

echo "🔧 编译优化版C++传统推荐算法..."

# 清理之前的程序
pkill -f cpp_traditional_recommender 2>/dev/null

# 编译优化版本
g++ -std=c++17 -O3 -march=native -mtune=native \
    -Wall -Wextra \
    cpp_tools/optimized_traditional_recommender.cpp \
    -o optimized_cpp_recommender

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo ""
    echo "🚀 开始运行优化版..."
    echo ""

    # 运行程序
    time ./optimized_cpp_recommender

    echo ""
    echo "📊 检查输出文件..."
    if [ -f "optimized_cpp_submission.txt" ]; then
        file_size=$(stat -f%z optimized_cpp_submission.txt 2>/dev/null || stat -c%s optimized_cpp_submission.txt)
        line_count=$(wc -l < optimized_cpp_submission.txt)

        echo "   📁 文件: optimized_cpp_submission.txt"
        echo "   📏 大小: $((file_size / 1024)) KB"
        echo "   📊 行数: $line_count"
        echo ""
        echo "🎯 优化要点:"
        echo "   - ItemCF限制计算量：每用户最多20个商品，每商品最多100个相似用户"
        echo "   - 详细进度显示：实时显示处理速度和剩余时间"
        echo "   - 流行度权重70%，协同过滤权重30%"
        echo "   - 文件大小预计算，精确进度条"
        echo ""
        echo "✅ 可直接提交到比赛平台"
    else
        echo "   ❌ 输出文件未生成"
    fi
else
    echo "❌ 编译失败!"
    exit 1
fi