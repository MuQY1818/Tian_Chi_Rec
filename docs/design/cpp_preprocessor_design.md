# C++高性能用户特征提取器设计方案

## 🎯 设计目标
- 处理11.65亿行数据，提取100万用户特征
- 速度提升10-50倍（相比Python）
- 内存占用控制在合理范围
- 输出标准CSV供Python ML模型使用

## 🏗️ 架构设计

### 核心组件
```cpp
class HighPerformanceUserExtractor {
private:
    // 用户统计映射 (user_id -> UserStats)
    std::unordered_map<uint32_t, UserStats> user_stats_;

    // 内存池管理
    MemoryPool memory_pool_;

    // 多线程处理器
    ThreadPool thread_pool_;

public:
    void ProcessFile(const std::string& filename);
    void GenerateFeatures(const std::string& output_csv);
};

struct UserStats {
    uint32_t total_actions = 0;
    uint32_t browse_count = 0;
    uint32_t collect_count = 0;
    uint32_t cart_count = 0;
    uint32_t purchase_count = 0;

    // 高效集合存储
    std::unordered_set<uint32_t> unique_items;
    std::unordered_set<uint16_t> unique_categories;

    // 时间统计
    uint64_t first_action_time = UINT64_MAX;
    uint64_t last_action_time = 0;
    std::array<uint32_t, 24> hour_activity{};  // 按小时统计

    // 地理位置
    std::unordered_map<std::string, uint32_t> geo_activity;

    // 行为序列 (环形缓冲区)
    CircularBuffer<BehaviorRecord, 1000> recent_behaviors;
};
```

## ⚡ 性能优化策略

### 1. 内存优化
- **预分配哈希表**：初始容量100万用户
- **内存池管理**：减少动态分配开销
- **数据紧凑存储**：使用uint32_t而非int64_t
- **字符串优化**：geo hash使用intern机制

### 2. I/O优化
- **mmap文件映射**：避免系统调用开销
- **批量处理**：每次处理10MB数据块
- **零拷贝解析**：直接在内存中解析

### 3. 并发优化
- **分区并行**：按用户ID哈希分区
- **无锁数据结构**：每线程独立用户集合
- **最终合并**：单线程快速合并结果

### 4. 算法优化
- **布隆过滤器**：快速判断用户是否存在
- **时间戳缓存**：避免重复解析时间
- **增量特征计算**：边读边计算特征

## 📊 预期特征输出

### 基础统计特征 (20维)
- 行为计数：browse, collect, cart, purchase
- 转化率：collect_rate, cart_rate, purchase_rate
- 活跃度：total_actions, unique_items, unique_categories, active_days

### 时间模式特征 (12维)
- 时段偏好：morning_rate, afternoon_rate, evening_rate, night_rate
- 活跃模式：weekend_rate, avg_action_interval, activity_regularity
- 时间跨度：days_since_first, days_since_last, action_span

### 地理特征 (8维)
- 地理多样性：unique_geo_count, geo_concentration
- 位置稳定性：primary_geo_ratio, geo_entropy

### 行为序列特征 (15维)
- 序列模式：avg_session_length, session_count
- 决策特征：browse_to_purchase_time, repeat_purchase_rate
- 最近行为：recent_activity_score, recent_purchase_trend

**总计约55维特征**

## 🚀 实施方案

### Phase 1: 核心功能 (2小时)
```cpp
// 基础框架和文件处理
class FastFileParser {
    void ParseLine(const char* line, UserBehavior& behavior);
    void ProcessChunk(const char* start, size_t size);
};
```

### Phase 2: 特征工程 (3小时)
```cpp
// 特征计算器
class FeatureCalculator {
    void CalculateBasicFeatures(const UserStats& stats, UserFeatures& features);
    void CalculateTimeFeatures(const UserStats& stats, UserFeatures& features);
    void CalculateGeoFeatures(const UserStats& stats, UserFeatures& features);
};
```

### Phase 3: 输出优化 (1小时)
```cpp
// CSV输出器
class CSVWriter {
    void WriteHeader(const std::vector<std::string>& columns);
    void WriteUserFeatures(uint32_t user_id, const UserFeatures& features);
};
```

## 💾 编译和使用

### 编译命令
```bash
g++ -O3 -march=native -std=c++17 \
    -fopenmp -pthread \
    -o fast_extractor \
    fast_user_extractor.cpp
```

### 运行命令
```bash
# 单线程版本
./fast_extractor --input-dir dataset/ --output user_features.csv

# 多线程版本
./fast_extractor --input-dir dataset/ --output user_features.csv --threads 8

# 内存限制版本
./fast_extractor --input-dir dataset/ --output user_features.csv --max-memory 8GB
```

## 📈 预期性能提升

| 指标 | Python版本 | C++版本 | 提升倍数 |
|------|------------|---------|----------|
| 处理速度 | ~6小时 | ~20分钟 | 18x |
| 内存占用 | ~8GB | ~2GB | 4x |
| CPU使用率 | 单核 | 多核 | 8x |
| 特征质量 | 基础 | 丰富 | 更全面 |

## 🎯 下一步行动

1. **立即开始**: 编写C++核心处理框架
2. **并行开发**: 同时设计特征工程算法
3. **增量测试**: 先用小数据集验证正确性
4. **性能调优**: 基于profiling结果优化瓶颈

这个C++方案可以将处理时间从几小时缩短到20分钟！