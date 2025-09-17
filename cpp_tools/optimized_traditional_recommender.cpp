#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cmath>
#include <random>

using namespace std;
using namespace chrono;

struct UserStats {
    unordered_map<uint32_t, uint32_t> item_interactions;
    unordered_set<uint32_t> purchased_items;
    unordered_map<uint16_t, uint32_t> category_counts;
    uint16_t preferred_category = 0;
    uint32_t total_interactions = 0;
};

struct ItemStats {
    uint32_t popularity = 0;
    uint32_t purchase_count = 0;
    unordered_set<uint32_t> users;  // 交互过的用户集合
    uint16_t category = 0;
    double purchase_rate = 0.0;
};

class OptimizedTraditionalRecommender {
private:
    unordered_map<uint32_t, UserStats> user_data;
    unordered_map<uint32_t, ItemStats> item_data;
    unordered_set<uint32_t> valid_items;

    uint64_t total_interactions = 0;
    uint64_t total_purchases = 0;

    void print_progress_bar(const string& task, int current, int total,
                           auto start_time, const string& extra_info = "") {
        double progress = static_cast<double>(current) / total * 100;
        int bars = static_cast<int>(progress / 5);  // 20个进度条
        string progress_str = string(bars, '#') + string(20 - bars, '-');

        auto now = high_resolution_clock::now();
        auto elapsed = duration_cast<milliseconds>(now - start_time);
        double speed = current * 1000.0 / (elapsed.count() + 1);
        double eta = (total - current) / (speed / 1000.0);

        cout << "\r   " << task << ": [" << progress_str << "] "
             << fixed << setprecision(1) << progress << "% "
             << "(" << current << "/" << total << ") "
             << "速度: " << setprecision(0) << speed << "/秒 "
             << "剩余: " << setprecision(1) << eta << "秒 "
             << extra_info;
        cout.flush();
    }

public:
    void load_item_subset() {
        cout << "\n" << string(60, '=') << "\n";
        cout << "🛍️  步骤1: 加载商品子集P\n";
        cout << string(60, '=') << "\n";

        auto start = high_resolution_clock::now();
        ifstream file("dataset/tianchi_fresh_comp_train_item_online.txt");
        if (!file.is_open()) {
            cerr << "❌ 无法打开商品文件\n";
            return;
        }

        string line;
        uint32_t count = 0;

        cout << "📁 正在加载商品目录...\n";
        while (getline(file, line)) {
            istringstream iss(line);
            string item_id_str, geohash, category_str;

            if (getline(iss, item_id_str, '\t') &&
                getline(iss, geohash, '\t') &&
                getline(iss, category_str)) {

                uint32_t item_id = stoul(item_id_str);
                uint16_t category = static_cast<uint16_t>(stoul(category_str));

                valid_items.insert(item_id);
                item_data[item_id].category = category;
                count++;

                // 每100万显示一次进度
                if (count % 1000000 == 0) {
                    cout << "\r   📊 已加载: " << count << " 个商品" << flush;
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "\n✅ 商品加载完成!\n";
        cout << "   📊 商品总数: " << count << "\n";
        cout << "   ⏱️  加载时间: " << duration.count() << "ms\n";
    }

    void process_data_file(const string& filename, int day) {
        cout << "\n📅 处理 " << filename << " (第" << day << "天)\n";

        ifstream file("dataset/preprocess_16to18/" + filename);
        if (!file.is_open()) {
            cerr << "❌ 无法打开文件: " << filename << "\n";
            return;
        }

        // 先计算文件总行数
        cout << "📏 计算文件大小...\n";
        uint64_t total_lines = 0;
        string temp_line;
        while (getline(file, temp_line)) {
            total_lines++;
        }
        file.clear();
        file.seekg(0, ios::beg);

        cout << "   📊 文件总行数: " << total_lines << "\n";

        auto start = high_resolution_clock::now();
        string line;
        uint64_t processed = 0;
        uint64_t filtered = 0;
        uint64_t valid_interactions = 0;
        uint64_t purchases = 0;

        const uint64_t update_interval = max(1UL, total_lines / 100);  // 每1%更新

        while (getline(file, line)) {
            istringstream iss(line);
            string user_str, item_str, behavior_str, geohash, category_str, time_str;

            if (getline(iss, user_str, '\t') &&
                getline(iss, item_str, '\t') &&
                getline(iss, behavior_str, '\t') &&
                getline(iss, geohash, '\t') &&
                getline(iss, category_str, '\t') &&
                getline(iss, time_str)) {

                uint32_t user_id = stoul(user_str);
                uint32_t item_id = stoul(item_str);
                uint8_t behavior = static_cast<uint8_t>(stoul(behavior_str));
                uint16_t category = static_cast<uint16_t>(stoul(category_str));

                processed++;

                // 只处理商品子集P中的数据
                if (valid_items.find(item_id) == valid_items.end()) {
                    continue;
                }

                filtered++;

                // 更新用户统计
                auto& user_stats = user_data[user_id];
                user_stats.item_interactions[item_id]++;
                user_stats.category_counts[category]++;
                user_stats.total_interactions++;

                // 更新商品统计
                auto& item_stats = item_data[item_id];
                item_stats.popularity++;
                item_stats.users.insert(user_id);
                item_stats.category = category;

                valid_interactions++;

                if (behavior == 4) {  // 购买行为
                    if (day != 18) {  // 16-17号作为训练
                        user_stats.purchased_items.insert(item_id);
                        item_stats.purchase_count++;
                        total_purchases++;
                        purchases++;
                    }
                }

                total_interactions++;

                // 显示详细进度
                if (processed % update_interval == 0 || processed == total_lines) {
                    string extra = "有效:" + to_string(filtered) + " 购买:" + to_string(purchases);
                    print_progress_bar("处理数据", processed, total_lines, start, extra);
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "\n✅ " << filename << " 处理完成!\n";
        cout << "   📊 总行数: " << processed << ", 有效: " << filtered << ", 购买: " << purchases << "\n";
        cout << "   ⏱️  耗时: " << duration.count() << "ms\n";
        cout << "   🚀 平均速度: " << fixed << setprecision(0)
             << processed * 1000.0 / duration.count() << " 行/秒\n";
    }

    void train_model() {
        cout << "\n" << string(60, '=') << "\n";
        cout << "🤖 步骤2: 训练传统推荐模型\n";
        cout << string(60, '=') << "\n";

        // 处理16-17号数据作为训练
        process_data_file("data_1216.txt", 16);
        process_data_file("data_1217.txt", 17);

        cout << "\n🔧 后处理计算...\n";

        // 计算用户偏好类别
        cout << "   📊 计算用户偏好类别...\n";
        uint32_t processed_users = 0;
        auto start = high_resolution_clock::now();

        for (auto& [user_id, stats] : user_data) {
            if (!stats.category_counts.empty()) {
                auto max_it = max_element(stats.category_counts.begin(),
                                        stats.category_counts.end(),
                                        [](const auto& a, const auto& b) {
                                            return a.second < b.second;
                                        });
                stats.preferred_category = max_it->first;
            }
            processed_users++;

            if (processed_users % 10000 == 0) {
                print_progress_bar("用户偏好", processed_users, user_data.size(), start);
            }
        }

        cout << "\n   📊 计算商品购买率...\n";
        uint32_t processed_items = 0;
        start = high_resolution_clock::now();

        for (auto& [item_id, stats] : item_data) {
            if (stats.popularity > 0) {
                stats.purchase_rate = static_cast<double>(stats.purchase_count) / stats.popularity;
            }
            processed_items++;

            if (processed_items % 100000 == 0) {
                print_progress_bar("商品特征", processed_items, item_data.size(), start);
            }
        }

        cout << "\n✅ 模型训练完成!\n";
        cout << "   👥 用户数: " << user_data.size() << "\n";
        cout << "   🛍️  商品数: " << item_data.size() << "\n";
        cout << "   🔗 总交互数: " << total_interactions << "\n";
        cout << "   💰 总购买数: " << total_purchases << "\n";
    }

    vector<pair<uint32_t, double>> optimized_itemcf_recommend(uint32_t user_id, int top_k) {
        unordered_map<uint32_t, double> candidate_scores;
        const auto& user_stats = user_data[user_id];
        unordered_set<uint32_t> user_items;

        for (const auto& [item_id, count] : user_stats.item_interactions) {
            user_items.insert(item_id);
        }

        // 优化的ItemCF：只考虑用户交互过的前20个最热门商品
        vector<pair<uint32_t, uint32_t>> user_item_pairs;
        for (const auto& [item_id, count] : user_stats.item_interactions) {
            user_item_pairs.emplace_back(item_id, count);
        }

        // 按交互次数排序，只取前20个
        sort(user_item_pairs.begin(), user_item_pairs.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });

        int limit = min(20, static_cast<int>(user_item_pairs.size()));

        for (int i = 0; i < limit; i++) {
            uint32_t item_id = user_item_pairs[i].first;
            uint32_t interaction_count = user_item_pairs[i].second;

            // 找到也交互过这个商品的用户（限制数量）
            const auto& item_users = item_data[item_id].users;
            int user_count = 0;

            for (uint32_t other_user_id : item_users) {
                if (other_user_id != user_id && user_count < 100) {  // 最多考虑100个相似用户
                    const auto& other_stats = user_data[other_user_id];

                    // 推荐其他用户的热门商品（最多5个）
                    vector<pair<uint32_t, uint32_t>> other_items;
                    for (const auto& [other_item_id, other_count] : other_stats.item_interactions) {
                        if (user_items.find(other_item_id) == user_items.end()) {
                            other_items.emplace_back(other_item_id, other_count);
                        }
                    }

                    sort(other_items.begin(), other_items.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });

                    int item_limit = min(5, static_cast<int>(other_items.size()));
                    for (int j = 0; j < item_limit; j++) {
                        uint32_t other_item_id = other_items[j].first;
                        uint32_t other_count = other_items[j].second;

                        double similarity = static_cast<double>(min(interaction_count, other_count)) /
                                          max(interaction_count, other_count);
                        candidate_scores[other_item_id] += similarity;
                    }
                    user_count++;
                }
            }
        }

        // 转换并排序
        vector<pair<uint32_t, double>> candidates;
        for (const auto& [item_id, score] : candidate_scores) {
            candidates.emplace_back(item_id, score);
        }

        partial_sort(candidates.begin(),
                    candidates.begin() + min(top_k, static_cast<int>(candidates.size())),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

        if (candidates.size() > top_k) {
            candidates.resize(top_k);
        }

        return candidates;
    }

    vector<pair<uint32_t, double>> popularity_recommend(uint32_t user_id, int top_k) {
        vector<pair<uint32_t, double>> candidates;

        const auto& user_stats = user_data[user_id];
        unordered_set<uint32_t> interacted_items;

        for (const auto& [item_id, count] : user_stats.item_interactions) {
            interacted_items.insert(item_id);
        }

        for (const auto& [item_id, stats] : item_data) {
            if (interacted_items.find(item_id) == interacted_items.end() &&
                stats.popularity >= 5) {

                double score = static_cast<double>(stats.popularity);

                // 用户偏好类别加权
                if (stats.category == user_stats.preferred_category) {
                    score *= 1.3;
                }

                // 购买率加权
                score *= (1.0 + stats.purchase_rate);

                candidates.emplace_back(item_id, score);
            }
        }

        partial_sort(candidates.begin(),
                    candidates.begin() + min(top_k, static_cast<int>(candidates.size())),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

        if (candidates.size() > top_k) {
            candidates.resize(top_k);
        }

        return candidates;
    }

    vector<uint32_t> hybrid_recommend(uint32_t user_id, int top_k) {
        auto pop_recs = popularity_recommend(user_id, top_k * 2);
        auto cf_recs = optimized_itemcf_recommend(user_id, top_k * 2);

        unordered_map<uint32_t, double> final_scores;

        for (const auto& [item_id, score] : pop_recs) {
            final_scores[item_id] += 0.7 * score;  // 流行度权重70%
        }

        for (const auto& [item_id, score] : cf_recs) {
            final_scores[item_id] += 0.3 * score;  // 协同过滤权重30%
        }

        vector<pair<uint32_t, double>> candidates;
        for (const auto& [item_id, score] : final_scores) {
            candidates.emplace_back(item_id, score);
        }

        partial_sort(candidates.begin(),
                    candidates.begin() + min(top_k, static_cast<int>(candidates.size())),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

        vector<uint32_t> result;
        for (int i = 0; i < min(top_k, static_cast<int>(candidates.size())); i++) {
            result.push_back(candidates[i].first);
        }

        return result;
    }

    void generate_recommendations(int top_k = 3) {
        cout << "\n" << string(60, '=') << "\n";
        cout << "🎯 步骤3: 生成推荐 (每用户top-" << top_k << ")\n";
        cout << string(60, '=') << "\n";

        auto start = high_resolution_clock::now();
        uint32_t total_users = user_data.size();
        uint32_t processed_users = 0;

        cout << "👥 开始为 " << total_users << " 个用户生成推荐...\n";

        ofstream output("optimized_cpp_submission.txt");
        if (!output.is_open()) {
            cerr << "❌ 无法创建输出文件\n";
            return;
        }

        uint64_t total_recs = 0;
        const uint32_t update_interval = max(1U, total_users / 100);

        for (const auto& [user_id, stats] : user_data) {
            if (stats.total_interactions > 0) {
                auto recommendations = hybrid_recommend(user_id, top_k);

                for (uint32_t item_id : recommendations) {
                    output << user_id << "\t" << item_id << "\n";
                    total_recs++;
                }
            }

            processed_users++;

            if (processed_users % update_interval == 0 || processed_users == total_users) {
                string extra = "推荐数:" + to_string(total_recs);
                print_progress_bar("生成推荐", processed_users, total_users, start, extra);
            }
        }

        output.close();

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "\n✅ 推荐生成完成!\n";
        cout << "   👥 处理用户数: " << processed_users << "\n";
        cout << "   📊 总推荐数: " << total_recs << "\n";
        cout << "   📈 平均每用户推荐: " << fixed << setprecision(1)
             << static_cast<double>(total_recs) / processed_users << "\n";
        cout << "   ⏱️  总耗时: " << duration.count() << "ms ("
             << fixed << setprecision(2) << duration.count() / 1000.0 << "秒)\n";
        cout << "   🚀 平均速度: " << fixed << setprecision(1)
             << processed_users * 1000.0 / duration.count() << " 用户/秒\n";
        cout << "   📁 输出文件: optimized_cpp_submission.txt\n";
    }
};

int main() {
    cout << string(70, '=') << "\n";
    cout << "🚀 优化版C++传统推荐算法\n";
    cout << "⚡ 高性能 + 详细进度显示\n";
    cout << "🎯 流行度(70%) + 优化协同过滤(30%)\n";
    cout << string(70, '=') << "\n";

    auto total_start = high_resolution_clock::now();

    OptimizedTraditionalRecommender recommender;

    try {
        recommender.load_item_subset();
        recommender.train_model();
        recommender.generate_recommendations(3);

        auto total_end = high_resolution_clock::now();
        auto total_duration = duration_cast<milliseconds>(total_end - total_start);

        cout << "\n" << string(20, '=') << " 完成 " << string(20, '=') << "\n";
        cout << "🎊 优化版推荐算法运行成功! 🎊\n";
        cout << "⏱️  总耗时: " << total_duration.count() << "ms ("
             << fixed << setprecision(2) << total_duration.count() / 1000.0 << "秒)\n";
        cout << "📁 提交文件: optimized_cpp_submission.txt\n";
        cout << "⚡ 优化点: 限制ItemCF计算量，加强进度可视化\n";

    } catch (const exception& e) {
        cerr << "❌ 运行错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}