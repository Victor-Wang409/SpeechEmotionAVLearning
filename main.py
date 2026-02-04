"""
主程序模块
负责整个程序的入口和执行流程
"""

import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EmotionDataset
from trainer_executor import TrainerExecutor
from util import split_iemocap, split_msppodcast

def main():
    """
    主函数，负责整个程序的执行
    """
    parser = argparse.ArgumentParser(description='Training VAD prediction model')
    parser.add_argument('--emotion2vec_dir', type=str, required=True, help='Directory containing emo2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True, help='Directory containing hubert features')
    parser.add_argument('--wav2vec_dir', type=str, default=None, help='Directory containing wav2vec features')
    parser.add_argument('--data2vec_dir', type=str, default=None, help='Directory containing data2vec features')
    parser.add_argument('--num_emotions', type=int, default=4, help='Numbers of emotion')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.01)

    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of epochs for learning rate warmup')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate') 
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['cosine', 'linear', 'step'], help='Type of learning rate scheduler')
    parser.add_argument('--lr_decay_step', type=int, default=15, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Decay rate for StepLR scheduler')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, 'training.log'))
        ]
    )

    logging.info(f"Actual batch size: {args.batch_size}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 加载数据集
    dataset = EmotionDataset(
        args.emotion2vec_dir, 
        args.hubert_dir, 
        args.csv_path,
        wav2vec_dir=args.wav2vec_dir,
        data2vec_dir=args.data2vec_dir
    )
    
    # 基于说话人进行5折交叉验证
    folds = split_iemocap(dataset.df)
    fold_results = []
    
    # 对每个fold进行训练
    for fold in range(len(folds)):
        logging.info(f"\n{'='*50}\nFold {fold+1}/{len(folds)}\n{'='*50}")
        
        # 创建当前fold的保存目录
        fold_dir = os.path.join(args.save_dir, f'fold{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # 获取当前fold的数据
        fold_data = folds[fold]
        
        # 训练模型并获取测试结果
        fold_results.append(
            TrainerExecutor.train_model(
                args=args,
                fold=fold,
                fold_dir=fold_dir,
                dataset=dataset,
                train_idx=fold_data['train_idx'],
                eval_idx=fold_data['eval_idx'],
                test_idx=fold_data['test_idx'],
                device=device
            )
        )
    
    # 提取 Accuracy (res[0]) 和 F1 (res[1])
    # 注意：res[2] 是 trainer_executor 返回的占位符 0.0，这里忽略
    acc_list = [res[0] for res in fold_results]
    f1_list = [res[1] for res in fold_results]
    
    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    
    avg_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    final_results = (
        f"Final Cross-Validation Results\n"
        f"Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n"
        f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n"
    )
    
    logging.info(f"\n{'='*50}\n{final_results}\n{'='*50}")
    
    # 保存最终结果
    with open(os.path.join(args.save_dir, 'final_results.txt'), 'w') as f:
        f.write(final_results)

if __name__ == '__main__':
    main()