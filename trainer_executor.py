"""
训练执行模块
负责整个训练流程的执行 (已修改为分类任务)
"""

import os
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from lr_scheduler import LRSchedulerFactory
from data_processor import DataProcessor
from loss_functions import LossFactory
from model import VADConfig, VADModelWithGating
from trainer import TrainingManager

class TrainerExecutor:
    """
    训练执行器，负责整个训练流程的执行
    """
    @staticmethod
    def train_model(args, fold, fold_dir, dataset, train_idx, eval_idx, test_idx, device):
        """
        训练单个fold的模型
        """
        # 创建数据加载器
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            collate_fn=DataProcessor.collate_fn
        )
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(eval_idx),
            collate_fn=DataProcessor.collate_fn
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_idx),
            collate_fn=DataProcessor.collate_fn
        )
        
        # 配置模型
        config = VADConfig(
            emotion2vec_dim=1024,
            hubert_dim=1024,
            hidden_dim=1024,
            num_hidden_layers=4,
            num_groups=8,
            wav2vec_dim=0,
            data2vec_dim=0,
            num_emotions=4 # 确保类别数正确
        )

        # 创建模型
        model = VADModelWithGating(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        num_training_steps = len(train_loader) * args.epochs
        
        # 学习率调度器
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, args, num_training_steps)
        
        # [修改] 使用分类损失函数
        cls_criterion = torch.nn.CrossEntropyLoss()
        contrast_criterion = LossFactory.SupervisedContrastiveLoss(temperature=0.1)
        
        best_val_acc = 0.0 # [修改] 跟踪最佳准确率
        best_model = None
        
        # [修改] 创建性能跟踪文件 (CSV头改为分类指标)
        metrics_file = os.path.join(fold_dir, 'metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_acc,val_f1\n')
        
        for epoch in range(args.epochs):
            # 训练一个epoch
            metrics = TrainingManager.train_one_epoch(
                model, 
                optimizer, 
                cls_criterion,
                contrast_criterion, 
                train_loader, 
                device
            )
            train_loss = metrics['loss']
            train_acc = metrics['acc']
            
            # 更新学习率
            if args.lr_scheduler == 'step':
                scheduler.step()
            else:
                # 对于 cosine 等需要每步更新的 scheduler，通常在 train_one_epoch 内部处理
                # 这里简单处理为每个 epoch 更新
                pass 
                    
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1} | LR: {current_lr:.2e}")
            
            # 验证
            val_acc, val_f1 = TrainingManager.validate_and_test(model, eval_loader, device)
            
            # 保存每个epoch的模型
            epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir, safe_serialization=False)
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, 'optimizer.pt'))
            
            # [修改] 保存训练指标 (写入分类指标)
            with open(metrics_file, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_acc:.4f},{val_f1:.4f}\n')
            
            # [修改] 日志打印分类指标
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

            # 测试
            test_acc, test_f1 = TrainingManager.validate_and_test(model, test_loader, device)
            logging.info(f"Test: Acc={test_acc:.4f}, F1={test_f1:.4f}")
            
            # [修改] 更新最佳模型 (基于 Accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict()
                
                best_model_dir = os.path.join(fold_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir, safe_serialization=False)
                logging.info(f"Saved new best model with val_acc={val_acc:.4f}")
            
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc, # [修改]
                'config': config.to_dict()
            }
            torch.save(checkpoint, os.path.join(fold_dir, 'checkpoint.pt'))
        
        # 加载最佳模型进行最终测试
        if best_model is not None:
            model.load_state_dict(best_model)
        
        test_acc, test_f1 = TrainingManager.validate_and_test(model, test_loader, device)
        
        # [修改] 保存最终测试结果
        with open(os.path.join(fold_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Results:\nAccuracy: {test_acc:.4f}\nF1 Score: {test_f1:.4f}")
        
        # 返回分类指标
        return test_acc, test_f1, 0.0 # 返回三个值以保持与 main.py 可能的接口兼容，或者修改 main.py