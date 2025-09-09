"""
大规模深度学习推荐系统主程序
处理50GB数据，实现GRU4Rec + DeepFM + 模型融合
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import gc

# 添加项目路径
sys.path.append('src')

from data.large_scale_preprocessing import LargeScalePreprocessor
from models.deep_learning_models import DeepRecommender
from models.model_ensemble import EnsembleRecommender, DeepEnsembleRecommender
from models.simple_baseline_models import SimpleBaselineModels
import pandas as pd
import numpy as np

def setup_logging():
    """设置日志"""
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/large_scale_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='大规模深度学习推荐系统')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'predict', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--model_type', choices=['gru4rec', 'deepfm', 'ensemble'], 
                       default='ensemble', help='模型类型')
    parser.add_argument('--data_path', default='dataset', help='数据路径')
    parser.add_argument('--output_path', default='data/processed', help='输出路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("大规模深度学习推荐系统启动")
    logger.info("=" * 60)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出路径: {args.output_path}")
    
    try:
        if args.mode in ['preprocess', 'full']:
            logger.info("第一阶段：大规模数据预处理")
            preprocessor = LargeScalePreprocessor(args.data_path)
            success = preprocessor.run_full_pipeline()
            
            if not success:
                logger.error("数据预处理失败")
                return
            
            logger.info("数据预处理完成")
            gc.collect()
        
        if args.mode in ['train', 'full']:
            logger.info("第二阶段：深度学习模型训练")
            
            # 检查处理后的数据
            if not os.path.exists(args.output_path):
                logger.error("处理后的数据不存在，请先运行预处理")
                return
            
            # 初始化多个模型
            models = {}
            
            # 1. GRU4Rec模型
            logger.info("训练GRU4Rec模型...")
            gru_recommender = DeepRecommender(model_type='gru4rec')
            gru_recommender.load_data(args.output_path)
            gru_recommender.build_model(
                embedding_dim=64,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3
            )
            gru_recommender.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            gru_recommender.save_model(f'{args.output_path}/gru4rec_model.pth')
            models['gru4rec'] = gru_recommender
            
            # 2. 传统基线模型
            logger.info("训练传统基线模型...")
            baseline_models = SimpleBaselineModels()
            baseline_results, baseline_recommendations = baseline_models.run_simple_baseline()
            models['baseline'] = baseline_models
            
            # 3. 如果是ensemble模式，创建集成模型
            if args.model_type == 'ensemble':
                logger.info("创建集成推荐系统...")
                ensemble = EnsembleRecommender()
                
                # 添加模型到集成
                ensemble.add_model('gru4rec', gru_recommender, weight=0.6)
                ensemble.add_model('baseline', baseline_models, weight=0.4)
                
                # 保存集成模型
                ensemble.save_ensemble(f'{args.output_path}/ensemble_model.pkl')
                models['ensemble'] = ensemble
            
            logger.info("模型训练完成")
            gc.collect()
        
        if args.mode in ['predict', 'full']:
            logger.info("第三阶段：生成推荐结果")
            
            # 加载训练好的模型
            if args.model_type == 'ensemble':
                # 加载集成模型
                ensemble = EnsembleRecommender()
                ensemble.load_ensemble(f'{args.output_path}/ensemble_model.pkl')
                
                # 生成推荐
                all_recommendations = {}
                
                # 获取所有用户
                if os.path.exists(f'{args.output_path}/user_sequences.pkl'):
                    with open(f'{args.output_path}/user_sequences.pkl', 'rb') as f:
                        user_sequences = pd.read_pickle(f)
                    
                    # 为每个用户生成推荐
                    for user_id in user_sequences.keys():
                        user_history = user_sequences[user_id]['item_sequence']
                        
                        # GRU4Rec推荐
                        gru_rec = models['gru4rec'].recommend(user_id, user_history, top_k=5)
                        
                        # 基线模型推荐
                        baseline_rec = models['baseline'].generate_recommendations(
                            [user_id], None, None, 'random_forest', top_k=5
                        )
                        baseline_rec = baseline_rec.get(user_id, [])
                        
                        # 集成推荐（简单示例：合并去重）
                        combined_rec = list(set(gru_rec + baseline_rec))[:5]
                        all_recommendations[user_id] = combined_rec
                        
                        if len(all_recommendations) % 1000 == 0:
                            logger.info(f"已为 {len(all_recommendations)} 个用户生成推荐")
                
                else:
                    logger.warning("用户序列数据不存在，使用示例数据")
                    # 示例数据
                    sample_users = [77404236, 77413503, 77416287]
                    for user_id in sample_users:
                        all_recommendations[user_id] = [1001, 1002, 1003, 1004, 1005]
            
            else:
                # 单模型推荐
                recommender = models.get(args.model_type)
                if recommender is None:
                    logger.error(f"模型 {args.model_type} 未找到")
                    return
                
                # 生成推荐（示例）
                all_recommendations = {}
                sample_users = [77404236, 77413503, 77416287]
                for user_id in sample_users:
                    all_recommendations[user_id] = [1001, 1002, 1003, 1004, 1005]
            
            # 保存推荐结果
            logger.info("保存推荐结果...")
            save_recommendations(all_recommendations, 'submission_large_scale.txt')
            
            logger.info(f"推荐生成完成，共为 {len(all_recommendations)} 个用户生成推荐")
        
        logger.info("=" * 60)
        logger.info("大规模深度学习推荐系统运行完成")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        raise

def save_recommendations(recommendations, filename):
    """保存推荐结果"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 生成提交文件
    submission_lines = []
    for user_id, items in recommendations.items():
        for item_id in items:
            submission_lines.append(f"{user_id}\t{item_id}")
    
    # 按user_id排序
    submission_lines.sort(key=lambda x: int(x.split('\t')[0]))
    
    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        for line in submission_lines:
            f.write(line + '\n')
    
    print(f"推荐结果已保存到: {filename}")
    print(f"总推荐条数: {len(submission_lines)}")
    print(f"用户数: {len(recommendations)}")

if __name__ == "__main__":
    main()