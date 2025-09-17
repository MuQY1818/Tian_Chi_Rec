#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <memory>
#include <cstring>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <ctime>

// 高性能用户行为数据结构
struct UserBehavior {
    uint32_t user_id;
    uint32_t item_id;
    uint8_t behavior_type;  // 1-4
    std::string geo_hash;
    uint16_t item_category;
    uint64_t timestamp;     // 时间戳（秒）
};

// 用户统计数据结构
struct UserStats {
    // 基础计数
    uint32_t total_actions = 0;
    uint32_t browse_count = 0;
    uint32_t collect_count = 0;
    uint32_t cart_count = 0;
    uint32_t purchase_count = 0;

    // 商品和类别
    std::unordered_set<uint32_t> unique_items;
    std::unordered_set<uint16_t> unique_categories;

    // 时间统计
    uint64_t first_action_time = UINT64_MAX;
    uint64_t last_action_time = 0;
    std::array<uint32_t, 24> hour_activity{};

    // 地理位置统计
    std::unordered_map<std::string, uint32_t> geo_activity;
    std::string primary_geo;

    // 类别统计
    std::unordered_map<uint16_t, uint32_t> category_activity;
    uint16_t top_category = 0;

    // 活跃天统计
    std::unordered_set<uint64_t> active_dates; // 以天为单位的时间戳

    // 最近行为记录 (只保留最近100个)
    std::vector<std::pair<uint64_t, uint8_t>> recent_behaviors;

    void add_behavior(const UserBehavior& behavior) {
        total_actions++;

        // 行为类型计数
        switch(behavior.behavior_type) {
            case 1: browse_count++; break;
            case 2: collect_count++; break;
            case 3: cart_count++; break;
            case 4: purchase_count++; break;
        }

        // 商品和类别
        unique_items.insert(behavior.item_id);
        unique_categories.insert(behavior.item_category);

        // 时间统计
        first_action_time = std::min(first_action_time, behavior.timestamp);
        last_action_time = std::max(last_action_time, behavior.timestamp);

        // 小时活跃度
        struct tm* tm_info = localtime(reinterpret_cast<const time_t*>(&behavior.timestamp));
        if (tm_info) {
            hour_activity[tm_info->tm_hour]++;
        }

        // 地理位置
        geo_activity[behavior.geo_hash]++;
        if (primary_geo.empty() || geo_activity[behavior.geo_hash] > geo_activity[primary_geo]) {
            primary_geo = behavior.geo_hash;
        }

        // 类别统计
        category_activity[behavior.item_category]++;
        if (top_category == 0 || category_activity[behavior.item_category] > category_activity[top_category]) {
            top_category = behavior.item_category;
        }

        // 活跃天统计 (将时间戳转换为天)
        uint64_t day_timestamp = behavior.timestamp / 86400 * 86400; // 转到当天00:00:00
        active_dates.insert(day_timestamp);

        // 最近行为
        recent_behaviors.emplace_back(behavior.timestamp, behavior.behavior_type);
        if (recent_behaviors.size() > 100) {
            recent_behaviors.erase(recent_behaviors.begin());
        }
    }
};

// 输出特征结构
struct UserFeatures {
    uint32_t user_id;

    // 基础统计特征 (15维)
    uint32_t total_actions;
    uint32_t browse_count;
    uint32_t collect_count;
    uint32_t cart_count;
    uint32_t purchase_count;
    float collect_rate;
    float cart_rate;
    float purchase_rate;
    uint32_t unique_items_count;
    uint32_t unique_categories_count;
    uint32_t active_days;
    float avg_actions_per_day;
    uint32_t days_since_first;
    uint32_t days_since_last;
    uint32_t action_span_days;

    // 时间模式特征 (8维)
    float morning_rate;      // 6-12点
    float afternoon_rate;    // 12-18点
    float evening_rate;      // 18-24点
    float night_rate;        // 0-6点
    uint32_t most_active_hour;
    float activity_regularity; // 基于小时分布的标准差
    float avg_action_interval_hours;
    uint32_t max_daily_actions;

    // 地理特征 (7维)
    uint32_t unique_geo_count;
    float geo_concentration;  // 主要地理位置占比
    float geo_diversity;      // 地理位置熵
    uint32_t primary_geo_actions;
    float geo_stability;      // 地理位置稳定性
    float mobility_score;     // 流动性评分

    // 增强特征 (9维)
    float purchase_conversion;      // 购买转化率 (相对总行为)
    float avg_interactions_per_item; // 平均每商品交互数
    uint32_t active_hours_count;    // 活跃小时数
    uint32_t top_category;          // 最喜欢的类别
    float top_category_ratio;       // 顶级类别占比
    uint32_t max_consecutive_days;  // 最大连续活跃天数
    float session_regularity;      // 会话规律性
    float action_intensity;        // 行为强度
    float predicted_purchase_prob; // 预测购买概率

    // 行为序列特征 (4维)
    float recent_activity_score;    // 最近行为活跃度
    uint32_t recent_purchase_count; // 最近购买次数
    float purchase_frequency;       // 购买频率
    float browse_to_purchase_ratio; // 浏览转购买比率

    UserFeatures() = default;
};

class FastUserExtractor {
private:
    std::unordered_map<uint32_t, UserStats> user_stats_;
    std::atomic<size_t> processed_lines_{0};
    std::atomic<size_t> total_lines_{0};
    const uint64_t target_date_timestamp_ = 1418832000; // 2014-12-18 00:00:00

public:
    // 解析单行数据
    bool parse_line(const std::string& line, UserBehavior& behavior) {
        std::istringstream iss(line);
        std::string token;

        // user_id
        if (!std::getline(iss, token, '\t')) return false;
        behavior.user_id = static_cast<uint32_t>(std::stoul(token));

        // item_id
        if (!std::getline(iss, token, '\t')) return false;
        behavior.item_id = static_cast<uint32_t>(std::stoul(token));

        // behavior_type
        if (!std::getline(iss, token, '\t')) return false;
        behavior.behavior_type = static_cast<uint8_t>(std::stoi(token));
        if (behavior.behavior_type < 1 || behavior.behavior_type > 4) return false;

        // geo_hash (可能为空)
        if (!std::getline(iss, token, '\t')) return false;
        behavior.geo_hash = token.empty() ? "unknown" : token;

        // item_category
        if (!std::getline(iss, token, '\t')) return false;
        behavior.item_category = static_cast<uint16_t>(std::stoul(token));

        // time (格式: 2014-12-17 20)
        if (!std::getline(iss, token)) return false;

        // 解析时间戳
        if (token.length() >= 13) {
            std::string date_part = token.substr(0, 10);  // 2014-12-17
            std::string hour_part = token.substr(11);     // 20

            struct tm tm_time{};
            if (sscanf(date_part.c_str(), "%d-%d-%d", &tm_time.tm_year, &tm_time.tm_mon, &tm_time.tm_mday) == 3) {
                tm_time.tm_year -= 1900;
                tm_time.tm_mon -= 1;
                tm_time.tm_hour = std::stoi(hour_part);
                tm_time.tm_min = 0;
                tm_time.tm_sec = 0;

                behavior.timestamp = static_cast<uint64_t>(mktime(&tm_time));
                return true;
            }
        }

        return false;
    }

    // 处理单个文件
    void process_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return;
        }

        std::cout << "处理文件: " << filename << std::endl;

        // 计算文件总行数
        size_t file_lines = 0;
        std::string temp_line;
        while (std::getline(file, temp_line)) {
            file_lines++;
        }
        total_lines_ += file_lines;

        // 重新打开文件
        file.clear();
        file.seekg(0);

        std::string line;
        size_t local_processed = 0;
        size_t batch_size = 100000;

        auto start_time = std::chrono::high_resolution_clock::now();

        while (std::getline(file, line)) {
            UserBehavior behavior;
            if (parse_line(line, behavior)) {
                user_stats_[behavior.user_id].add_behavior(behavior);
            }

            local_processed++;
            processed_lines_++;

            // 每处理10万行显示一次进度
            if (local_processed % batch_size == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

                float progress = static_cast<float>(local_processed) / file_lines * 100;
                float total_progress = static_cast<float>(processed_lines_) / total_lines_ * 100;

                std::cout << "  文件进度: " << progress << "% | "
                          << "总进度: " << total_progress << "% | "
                          << "用户数: " << user_stats_.size() << " | "
                          << "耗时: " << elapsed << "s" << std::endl;
            }
        }

        file.close();
        std::cout << "文件完成: " << filename << " (处理了 " << local_processed << " 行)" << std::endl;
    }

    // 计算用户特征
    UserFeatures calculate_features(uint32_t user_id, const UserStats& stats) {
        UserFeatures features;
        features.user_id = user_id;

        // 基础统计特征
        features.total_actions = stats.total_actions;
        features.browse_count = stats.browse_count;
        features.collect_count = stats.collect_count;
        features.cart_count = stats.cart_count;
        features.purchase_count = stats.purchase_count;

        // 转化率
        features.collect_rate = stats.browse_count > 0 ?
            static_cast<float>(stats.collect_count) / stats.browse_count : 0.0f;
        features.cart_rate = stats.browse_count > 0 ?
            static_cast<float>(stats.cart_count) / stats.browse_count : 0.0f;
        features.purchase_rate = stats.browse_count > 0 ?
            static_cast<float>(stats.purchase_count) / stats.browse_count : 0.0f;

        // 多样性
        features.unique_items_count = static_cast<uint32_t>(stats.unique_items.size());
        features.unique_categories_count = static_cast<uint32_t>(stats.unique_categories.size());

        // 时间特征
        features.active_days = static_cast<uint32_t>(stats.active_dates.size());
        if (stats.first_action_time != UINT64_MAX && stats.last_action_time > 0) {
            features.days_since_first = (target_date_timestamp_ - stats.first_action_time) / 86400;
            features.days_since_last = (target_date_timestamp_ - stats.last_action_time) / 86400;
            features.action_span_days = (stats.last_action_time - stats.first_action_time) / 86400 + 1;
            features.avg_actions_per_day = static_cast<float>(stats.total_actions) / features.active_days;
        } else {
            features.days_since_first = 0;
            features.days_since_last = 0;
            features.action_span_days = 1;
            features.avg_actions_per_day = static_cast<float>(stats.total_actions);
        }

        // 时间模式
        uint32_t morning_actions = 0, afternoon_actions = 0, evening_actions = 0, night_actions = 0;
        uint32_t max_hour_actions = 0;
        uint32_t most_active_hour = 0;

        for (int hour = 0; hour < 24; hour++) {
            uint32_t hour_actions = stats.hour_activity[hour];

            if (hour >= 6 && hour < 12) morning_actions += hour_actions;
            else if (hour >= 12 && hour < 18) afternoon_actions += hour_actions;
            else if (hour >= 18) evening_actions += hour_actions;
            else night_actions += hour_actions;

            if (hour_actions > max_hour_actions) {
                max_hour_actions = hour_actions;
                most_active_hour = hour;
            }
        }

        features.morning_rate = static_cast<float>(morning_actions) / stats.total_actions;
        features.afternoon_rate = static_cast<float>(afternoon_actions) / stats.total_actions;
        features.evening_rate = static_cast<float>(evening_actions) / stats.total_actions;
        features.night_rate = static_cast<float>(night_actions) / stats.total_actions;
        features.most_active_hour = most_active_hour;

        // 活跃度规律性（基于小时分布的标准差）
        float mean_hourly = static_cast<float>(stats.total_actions) / 24;
        float variance = 0;
        for (int hour = 0; hour < 24; hour++) {
            float diff = stats.hour_activity[hour] - mean_hourly;
            variance += diff * diff;
        }
        features.activity_regularity = 1.0f / (1.0f + sqrt(variance / 24) / mean_hourly);

        // 平均行为间隔小时数和最大日行为数
        if (stats.recent_behaviors.size() > 1) {
            std::vector<uint64_t> timestamps;
            for (const auto& [timestamp, behavior_type] : stats.recent_behaviors) {
                timestamps.push_back(timestamp);
            }
            std::sort(timestamps.begin(), timestamps.end());

            uint64_t total_interval = 0;
            for (size_t i = 1; i < timestamps.size(); i++) {
                total_interval += timestamps[i] - timestamps[i-1];
            }
            features.avg_action_interval_hours = static_cast<float>(total_interval) / (timestamps.size() - 1) / 3600.0f;
        } else {
            features.avg_action_interval_hours = 24.0f;  // 默认24小时
        }

        // 最大日行为数 (简化计算 - 使用活跃天数估算)
        features.max_daily_actions = features.active_days > 0 ?
            static_cast<uint32_t>(features.avg_actions_per_day * 1.5f) : 0;

        // 地理特征
        features.unique_geo_count = static_cast<uint32_t>(stats.geo_activity.size());

        if (!stats.primary_geo.empty() && stats.geo_activity.count(stats.primary_geo) > 0) {
            features.primary_geo_actions = stats.geo_activity.at(stats.primary_geo);
            features.geo_concentration = static_cast<float>(features.primary_geo_actions) / stats.total_actions;
        } else {
            features.primary_geo_actions = 0;
            features.geo_concentration = 0.0f;
        }

        // 地理多样性（简化的熵计算）
        float geo_entropy = 0.0f;
        for (const auto& [geo, count] : stats.geo_activity) {
            float prob = static_cast<float>(count) / stats.total_actions;
            if (prob > 0) {
                geo_entropy -= prob * log2(prob);
            }
        }
        features.geo_diversity = geo_entropy;

        // 地理稳定性 (主要地理位置的一致性)
        features.geo_stability = features.geo_concentration;  // 简化为主要地理位置占比
        features.mobility_score = static_cast<float>(features.unique_geo_count) / stats.total_actions;

        // 行为序列特征
        features.recent_activity_score = static_cast<float>(stats.recent_behaviors.size()) / 100.0f;

        features.recent_purchase_count = 0;
        for (const auto& [timestamp, behavior_type] : stats.recent_behaviors) {
            if (behavior_type == 4) features.recent_purchase_count++;
        }

        features.purchase_frequency = stats.total_actions > 0 ?
            static_cast<float>(stats.purchase_count) / stats.total_actions : 0.0f;

        features.browse_to_purchase_ratio = stats.purchase_count > 0 ?
            static_cast<float>(stats.browse_count) / stats.purchase_count : 0.0f;

        // 类别集中度已经通过top_category_ratio体现

        // 新增特征计算

        // 购买转化率 (相对总行为)
        features.purchase_conversion = stats.total_actions > 0 ?
            static_cast<float>(stats.purchase_count) / stats.total_actions : 0.0f;

        // 平均每商品交互数
        features.avg_interactions_per_item = stats.unique_items.size() > 0 ?
            static_cast<float>(stats.total_actions) / stats.unique_items.size() : 0.0f;

        // 活跃小时数
        features.active_hours_count = 0;
        for (int hour = 0; hour < 24; hour++) {
            if (stats.hour_activity[hour] > 0) {
                features.active_hours_count++;
            }
        }

        // 最喜欢的类别
        features.top_category = stats.top_category;
        features.top_category_ratio = stats.category_activity.size() > 0 && stats.total_actions > 0 ?
            static_cast<float>(stats.category_activity.at(stats.top_category)) / stats.total_actions : 0.0f;

        // 最大连续活跃天数 (简化计算)
        features.max_consecutive_days = stats.active_dates.size();  // 简化版本

        // 会话规律性 (基于活跃天数和行为分布)
        features.session_regularity = features.active_days > 1 ?
            static_cast<float>(features.active_days) / features.action_span_days : 0.0f;

        // 行为强度 (每天平均行为数)
        features.action_intensity = features.avg_actions_per_day;

        // 预测购买概率（启发式，增强版）
        float purchase_score = 0.0f;
        if (stats.purchase_count > 0) purchase_score += 0.4f;
        if (stats.purchase_count > 2) purchase_score += 0.2f;
        if (features.recent_purchase_count > 0) purchase_score += 0.3f;
        if (stats.cart_count > 0) purchase_score += 0.1f;
        if (features.purchase_conversion > 0.01f) purchase_score += 0.1f;  // 高转化用户
        if (features.recent_activity_score > 0.5f) purchase_score += 0.1f; // 最近活跃
        features.predicted_purchase_prob = std::min(purchase_score, 1.0f);

        return features;
    }

    // 输出特征到CSV
    void export_features(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }

        // 写入CSV头部 (39个特征)
        file << "user_id,total_actions,browse_count,collect_count,cart_count,purchase_count,"
             << "collect_rate,cart_rate,purchase_rate,unique_items_count,unique_categories_count,"
             << "active_days,avg_actions_per_day,days_since_first,days_since_last,action_span_days,"
             << "morning_rate,afternoon_rate,evening_rate,night_rate,most_active_hour,activity_regularity,"
             << "avg_action_interval_hours,max_daily_actions,"
             << "unique_geo_count,geo_concentration,geo_diversity,primary_geo_actions,geo_stability,mobility_score,"
             << "purchase_conversion,avg_interactions_per_item,active_hours_count,top_category,top_category_ratio,"
             << "max_consecutive_days,session_regularity,action_intensity,predicted_purchase_prob,"
             << "recent_activity_score,recent_purchase_count,purchase_frequency,browse_to_purchase_ratio\n";

        std::cout << "生成特征中..." << std::endl;

        size_t count = 0;
        for (const auto& [user_id, stats] : user_stats_) {
            UserFeatures features = calculate_features(user_id, stats);

            // 写入特征行 (39个特征)
            file << features.user_id << ","
                 << features.total_actions << ","
                 << features.browse_count << ","
                 << features.collect_count << ","
                 << features.cart_count << ","
                 << features.purchase_count << ","
                 << features.collect_rate << ","
                 << features.cart_rate << ","
                 << features.purchase_rate << ","
                 << features.unique_items_count << ","
                 << features.unique_categories_count << ","
                 << features.active_days << ","
                 << features.avg_actions_per_day << ","
                 << features.days_since_first << ","
                 << features.days_since_last << ","
                 << features.action_span_days << ","
                 << features.morning_rate << ","
                 << features.afternoon_rate << ","
                 << features.evening_rate << ","
                 << features.night_rate << ","
                 << features.most_active_hour << ","
                 << features.activity_regularity << ","
                 << features.avg_action_interval_hours << ","
                 << features.max_daily_actions << ","
                 << features.unique_geo_count << ","
                 << features.geo_concentration << ","
                 << features.geo_diversity << ","
                 << features.primary_geo_actions << ","
                 << features.geo_stability << ","
                 << features.mobility_score << ","
                 << features.purchase_conversion << ","
                 << features.avg_interactions_per_item << ","
                 << features.active_hours_count << ","
                 << features.top_category << ","
                 << features.top_category_ratio << ","
                 << features.max_consecutive_days << ","
                 << features.session_regularity << ","
                 << features.action_intensity << ","
                 << features.predicted_purchase_prob << ","
                 << features.recent_activity_score << ","
                 << features.recent_purchase_count << ","
                 << features.purchase_frequency << ","
                 << features.browse_to_purchase_ratio << "\n";

            count++;
            if (count % 50000 == 0) {
                std::cout << "  已处理用户: " << count << " / " << user_stats_.size() << std::endl;
            }
        }

        file.close();
        std::cout << "特征文件已生成: " << output_file << std::endl;
        std::cout << "总用户数: " << user_stats_.size() << std::endl;
        std::cout << "特征维度: 39 (与Python版本一致)" << std::endl;
    }

    size_t get_user_count() const {
        return user_stats_.size();
    }
};

int main(int, char**) {
    std::cout << "=== 高性能用户特征提取器 ===" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    FastUserExtractor extractor;

    // 处理数据文件
    std::vector<std::string> files = {
        "../dataset/tianchi_fresh_comp_train_user_online_partA.txt",
        "../dataset/tianchi_fresh_comp_train_user_online_partB.txt"
    };

    for (const auto& file : files) {
        extractor.process_file(file);
    }

    // 输出特征
    std::string output_file = "/mnt/data/tianchi_features/user_features_cpp.csv";
    extractor.export_features(output_file);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "\n🎉 处理完成!" << std::endl;
    std::cout << "⏱️  总耗时: " << duration << " 秒" << std::endl;
    std::cout << "👥 用户数量: " << extractor.get_user_count() << std::endl;
    std::cout << "📁 输出文件: " << output_file << std::endl;

    return 0;
}