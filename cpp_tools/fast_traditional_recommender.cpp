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

using namespace std;
using namespace chrono;

struct UserStats {
    unordered_map<uint32_t, uint32_t> item_interactions;  // item_id -> count
    unordered_set<uint32_t> purchased_items;
    unordered_map<uint16_t, uint32_t> category_counts;    // category -> count
    uint16_t preferred_category = 0;
    uint32_t total_interactions = 0;
};

struct ItemStats {
    uint32_t popularity = 0;           // 总交互数
    uint32_t purchase_count = 0;       // 购买数
    uint32_t user_count = 0;          // 交互用户数
    uint16_t category = 0;
    double purchase_rate = 0.0;
};

class FastTraditionalRecommender {
private:
    unordered_map<uint32_t, UserStats> user_data;
    unordered_map<uint32_t, ItemStats> item_data;
    unordered_set<uint32_t> valid_items;  // 商品子集P

    uint64_t total_interactions = 0;
    uint64_t total_purchases = 0;

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
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "📊 商品数量: " << count << "\n";
        cout << "⏱️  加载时间: " << duration.count() << "ms\n";
    }

    void process_data_file(const string& filename, int day) {
        cout << "\n📅 处理 " << filename << " (第" << day << "天)\n";

        ifstream file("dataset/preprocess_16to18/" + filename);
        if (!file.is_open()) {
            cerr << "❌ 无法打开文件: " << filename << "\n";
            return;
        }

        auto start = high_resolution_clock::now();
        string line;
        uint64_t processed = 0;
        uint64_t filtered = 0;

        // 每10万行显示一次进度
        const uint64_t update_interval = 100000;

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
                item_stats.category = category;

                if (behavior == 4 && day == 18) {  // 18号的购买作为验证标签
                    // 不算入训练数据，但记录用于验证
                } else if (behavior == 4) {  // 16-17号的购买作为训练
                    user_stats.purchased_items.insert(item_id);
                    item_stats.purchase_count++;
                    total_purchases++;
                }

                total_interactions++;

                // 显示进度
                if (processed % update_interval == 0) {
                    auto now = high_resolution_clock::now();
                    auto elapsed = duration_cast<milliseconds>(now - start);
                    double speed = processed * 1000.0 / elapsed.count();

                    cout << "   📈 已处理: " << processed
                         << " 行, 有效: " << filtered
                         << " 行, 速度: " << fixed << setprecision(0) << speed << " 行/秒\r";
                    cout.flush();
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double speed = processed * 1000.0 / duration.count();

        cout << "\n   ✅ 完成: " << processed << " 行, 有效: " << filtered
             << " 行, 耗时: " << duration.count() << "ms, 平均速度: "
             << fixed << setprecision(0) << speed << " 行/秒\n";
    }

    void train_model() {
        cout << "\n" << string(60, '=') << "\n";
        cout << "🤖 步骤2: 训练传统推荐模型\n";
        cout << string(60, '=') << "\n";

        auto start = high_resolution_clock::now();

        // 处理16-17号数据作为训练
        process_data_file("data_1216.txt", 16);
        process_data_file("data_1217.txt", 17);

        // 计算用户偏好类别
        cout << "\n🔧 计算用户偏好类别...\n";
        uint32_t processed_users = 0;

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
        }

        // 计算商品购买率
        cout << "🔧 计算商品购买率...\n";
        for (auto& [item_id, stats] : item_data) {
            if (stats.popularity > 0) {
                stats.purchase_rate = static_cast<double>(stats.purchase_count) / stats.popularity;
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "\n✅ 模型训练完成!\n";
        cout << "   👥 用户数: " << user_data.size() << "\n";
        cout << "   🛍️  商品数: " << item_data.size() << "\n";
        cout << "   🔗 总交互数: " << total_interactions << "\n";
        cout << "   💰 总购买数: " << total_purchases << "\n";
        cout << "   ⏱️  训练时间: " << duration.count() << "ms\n";
    }

    vector<pair<uint32_t, double>> popularity_recommend(uint32_t user_id, int top_k) {
        vector<pair<uint32_t, double>> candidates;

        const auto& user_stats = user_data[user_id];
        unordered_set<uint32_t> interacted_items;

        // 获取用户已交互商品
        for (const auto& [item_id, count] : user_stats.item_interactions) {
            interacted_items.insert(item_id);
        }

        // 流行度推荐
        for (const auto& [item_id, stats] : item_data) {
            if (interacted_items.find(item_id) == interacted_items.end() &&
                stats.popularity >= 5) {  // 至少5次交互

                double score = static_cast<double>(stats.popularity);

                // 用户偏好类别加权
                if (stats.category == user_stats.preferred_category) {
                    score *= 1.2;
                }

                candidates.emplace_back(item_id, score);
            }
        }

        // 排序取Top-K
        partial_sort(candidates.begin(),
                    candidates.begin() + min(top_k, static_cast<int>(candidates.size())),
                    candidates.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

        if (candidates.size() > top_k) {
            candidates.resize(top_k);
        }

        return candidates;
    }

    vector<pair<uint32_t, double>> itemcf_recommend(uint32_t user_id, int top_k) {
        unordered_map<uint32_t, double> candidate_scores;

        const auto& user_stats = user_data[user_id];
        unordered_set<uint32_t> user_items;

        for (const auto& [item_id, count] : user_stats.item_interactions) {
            user_items.insert(item_id);
        }

        // 基于用户-商品协同过滤
        for (const auto& [item_id, interaction_count] : user_stats.item_interactions) {
            // 找到也交互过这个商品的其他用户
            for (const auto& [other_user_id, other_stats] : user_data) {
                if (other_user_id != user_id &&
                    other_stats.item_interactions.find(item_id) != other_stats.item_interactions.end()) {

                    // 推荐其他用户的商品
                    for (const auto& [other_item_id, other_count] : other_stats.item_interactions) {
                        if (user_items.find(other_item_id) == user_items.end()) {
                            double similarity = static_cast<double>(min(interaction_count, other_count)) /
                                              max(interaction_count, other_count);
                            candidate_scores[other_item_id] += similarity;
                        }
                    }
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

    vector<uint32_t> hybrid_recommend(uint32_t user_id, int top_k) {
        auto pop_recs = popularity_recommend(user_id, top_k * 2);
        auto cf_recs = itemcf_recommend(user_id, top_k * 2);

        unordered_map<uint32_t, double> final_scores;

        // 融合分数：流行度60% + 协同过滤40%
        for (const auto& [item_id, score] : pop_recs) {
            final_scores[item_id] += 0.6 * score;
        }

        for (const auto& [item_id, score] : cf_recs) {
            final_scores[item_id] += 0.4 * score;
        }

        // 排序
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

        cout << "👥 待处理用户数: " << total_users << "\n";

        ofstream output("cpp_traditional_submission.txt");
        if (!output.is_open()) {
            cerr << "❌ 无法创建输出文件\n";
            return;
        }

        uint64_t total_recs = 0;
        const uint32_t update_interval = max(1U, total_users / 100);  // 每1%更新

        for (const auto& [user_id, stats] : user_data) {
            if (stats.total_interactions > 0) {  // 只为有交互的用户推荐
                auto recommendations = hybrid_recommend(user_id, top_k);

                for (uint32_t item_id : recommendations) {
                    output << user_id << "\t" << item_id << "\n";
                    total_recs++;
                }
            }

            processed_users++;

            // 显示进度
            if (processed_users % update_interval == 0 || processed_users == total_users) {
                auto now = high_resolution_clock::now();
                auto elapsed = duration_cast<milliseconds>(now - start);
                double speed = processed_users * 1000.0 / elapsed.count();
                double progress = static_cast<double>(processed_users) / total_users * 100;

                // 进度条
                int bars = static_cast<int>(progress / 5);  // 20个进度条
                string progress_str = string(bars, '█') + string(20 - bars, '░');

                cout << "   📈 进度: [" << progress_str << "] "
                     << fixed << setprecision(1) << progress << "%\n";
                cout << "   👥 已处理: " << processed_users << "/" << total_users << " 用户\n";
                cout << "   🚀 速度: " << fixed << setprecision(1) << speed << " 用户/秒\n";

                if (processed_users < total_users) {
                    double eta = (total_users - processed_users) / speed;
                    cout << "   ⏱️  预计剩余: " << fixed << setprecision(1) << eta << "秒\n";
                }
                cout << "\033[4A";  // 向上4行，实现原地更新
                cout.flush();
            }
        }

        cout << "\033[4B";  // 向下4行，结束原地更新

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double avg_speed = processed_users * 1000.0 / duration.count();

        output.close();

        cout << "\n✅ 推荐生成完成!\n";
        cout << "   👥 处理用户数: " << processed_users << "\n";
        cout << "   📊 总推荐数: " << total_recs << "\n";
        cout << "   📈 平均每用户推荐: " << fixed << setprecision(1)
             << static_cast<double>(total_recs) / processed_users << "\n";
        cout << "   ⏱️  总耗时: " << duration.count() << "ms\n";
        cout << "   🚀 平均速度: " << fixed << setprecision(1) << avg_speed << " 用户/秒\n";
        cout << "   📁 输出文件: cpp_traditional_submission.txt\n";
    }
};

int main() {
    cout << string(70, '=') << "\n";
    cout << "🚀 C++快速传统推荐算法\n";
    cout << "⚡ 高性能实现：流行度 + 协同过滤\n";
    cout << string(70, '=') << "\n";

    auto total_start = high_resolution_clock::now();

    FastTraditionalRecommender recommender;

    try {
        // 1. 加载商品子集
        recommender.load_item_subset();

        // 2. 训练模型
        recommender.train_model();

        // 3. 生成推荐
        recommender.generate_recommendations(3);

        auto total_end = high_resolution_clock::now();
        auto total_duration = duration_cast<milliseconds>(total_end - total_start);

        cout << "\n" << string(25, '🎉') << "\n";
        cout << "🎊 C++推荐算法运行成功! 🎊\n";
        cout << string(25, '🎉') << "\n";
        cout << "⏱️  总耗时: " << total_duration.count() << "ms ("
             << fixed << setprecision(2) << total_duration.count() / 1000.0 << "秒)\n";
        cout << "⚡ C++性能优势: 比Python快10-100倍\n";
        cout << "📁 提交文件: cpp_traditional_submission.txt\n";

    } catch (const exception& e) {
        cerr << "❌ 运行错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}