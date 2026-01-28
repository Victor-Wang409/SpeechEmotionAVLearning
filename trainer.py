"""
训练和评估模块
负责模型的训练和评估 (已修改为分类任务)
"""

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class TrainingManager:
    """
    训练管理器，封装训练和评估相关的函数
    """
    @staticmethod
    def train_one_epoch(model, optimizer, criterion, contrast_criterion, train_loader, device):
        """
        训练一个epoch (分类模式)
        
        参数:
            model: 模型
            optimizer: 优化器
            criterion: 主损失函数 (CrossEntropyLoss)
            contrast_criterion: 对比损失函数
            train_loader: 训练数据加载器
            device: 设备
            
        返回:
            训练指标字典
        """
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_acc = 0.0
        total_weights = {}  # 存储各特征的权重
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(total=len(train_loader), desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 准备特征字典
            features = {
                "emotion2vec": batch["emotion2vec_features"].to(device),
                "hubert": batch["hubert_features"].to(device)
            }
            
            # 添加其他特征（如果存在）
            if "wav2vec_features" in batch:
                features["wav2vec"] = batch["wav2vec_features"].to(device)
            if "data2vec_features" in batch:
                features["data2vec"] = batch["data2vec_features"].to(device)
                
            padding_mask = batch["padding_mask"].to(device)
            
            # 获取分类标签 (one-hot -> index)
            emotion_labels = batch["emotion_labels"].to(device)
            targets = torch.argmax(emotion_labels, dim=1)
            
            # 前向传播
            logits, feature_weights, contrast_features, current_temp = model(
                features,
                padding_mask
            )
            
            # 计算损失
            cls_loss = criterion(logits, targets)
            
            # 对比损失
            contrast_loss = contrast_criterion(contrast_features, targets)
            
            # 总损失 (对比损失权重可调)
            loss = cls_loss + 0.1 * contrast_loss
            
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache() # 清理显存
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = (preds == targets).float().mean().item()
            total_acc += acc
            
            total_loss += loss.item()
            total_batches += 1
            
            # 记录门控权重
            for feat_type, weight in feature_weights.items():
                if feat_type not in total_weights:
                    total_weights[feat_type] = 0.0
                total_weights[feat_type] += weight.mean().item()
            
            # 更新进度条
            avg_weights = {f"{k}_w": f"{total_weights[k] / total_batches:.3f}" for k in total_weights}
            
            postfix_info = {
                'loss': f'{(total_loss / total_batches):.4f}',
                'acc': f'{(total_acc / total_batches):.4f}',
                'temp': f'{current_temp.item():.4f}'
            }
            postfix_info.update(avg_weights)
            
            progress_bar.set_postfix(postfix_info)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # 返回平均指标
        result = {
            'loss': total_loss / total_batches,
            'acc': total_acc / total_batches
        }
    
        # 添加各特征的平均权重
        for feat_type in total_weights:
            result[f'{feat_type}_weight'] = total_weights[feat_type] / total_batches
        
        return result

    @staticmethod
    def validate_and_test(model, data_loader, device):
        """
        验证和测试模型 (分类模式)
        
        参数:
            model: 模型
            data_loader: 数据加载器
            device: 设备
            
        返回:
            验证/测试指标 (Accuracy, F1)
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                # 准备特征字典
                features = {
                    "emotion2vec": batch["emotion2vec_features"].to(device),
                    "hubert": batch["hubert_features"].to(device)
                }
                
                # 添加其他特征（如果存在）
                if "wav2vec_features" in batch:
                    features["wav2vec"] = batch["wav2vec_features"].to(device)
                if "data2vec_features" in batch:
                    features["data2vec"] = batch["data2vec_features"].to(device)
                    
                padding_mask = batch["padding_mask"].to(device)
                
                # 获取真实标签
                targets = torch.argmax(batch["emotion_labels"].to(device), dim=1)
                
                # 前向传播
                logits, _, _, _ = model(features, padding_mask)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算分类指标
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return val_acc, val_f1