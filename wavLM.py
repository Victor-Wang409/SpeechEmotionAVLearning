from transformers import AutoProcessor, WavLMModel, Wav2Vec2FeatureExtractor, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
import random
import os
import wandb
import argparse
import audmetric
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader
from datasets import concatenate_datasets

device = torch.device("cuda:0")

# 全局初始化特征提取器，供多进程 collate_fn 调用
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')

def my_collate(batch):
    audios, emotions = [], []
    for data in batch:
        au, emo = data['audio'], data['emotion']
        audios.append(au['array'])
        emotions.append(emo)
    
    # Batch 内部动态 Padding 规避 OOM。
    # 注意：为了兼容多进程 (num_workers > 0)，这里去除了 .to(device)，统一留到训练循环中转移到显卡
    processed = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    return {"audio": processed, "emotion": emotions}

# class EmotionClassifier(nn.Module):
#     def __init__(self, layer_num, emb_dim, num_labels, hidden_dim=100):
#         super().__init__()
#         self.layer_num = layer_num
#         self.emb_dim = emb_dim

#         self.weights = nn.Parameter(torch.randn(layer_num))
#         self.proj = nn.Linear(emb_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, num_labels)
#         nn.init.xavier_uniform_(self.proj.weight)
#         nn.init.xavier_uniform_(self.out.weight)
    
#     def forward(self, feature, feature_lens):
#         stacked_feature = torch.stack(feature, dim=0)
#         _, *origin_shape = stacked_feature.shape
#         stacked_feature = stacked_feature.view(self.layer_num, -1)
#         norm_weights = F.softmax(self.weights, dim=-1)
#         weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
#         weighted_feature = weighted_feature.view(*origin_shape)

#         agg_vec_list = []
#         for i in range(len(weighted_feature)):
#             agg_vec = torch.mean(weighted_feature[i][:feature_lens[i]], dim=0)
#             agg_vec_list.append(agg_vec)

#         avg_emb = torch.stack(agg_vec_list)

#         final_emb = self.proj(avg_emb)
#         pred = self.out(final_emb)
#         return pred, final_emb

class EmotionClassifier(nn.Module):
    def __init__(self, layer_num, emb_dim, num_labels, hidden_dim=100):
        super().__init__()
        self.layer_num = layer_num
        self.emb_dim = emb_dim

        # 1. 动态门控网络 (Dynamic Gating Network)
        # 负责根据每层的特征，输出一个对应的门控打分
        self.gate = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # 2. 最终的分类投影网络
        self.proj = nn.Linear(emb_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
        # 为 gate 网络的最后一层进行较小的初始化，使初始阶段各层权重趋于均等
        nn.init.normal_(self.gate[2].weight, std=0.01)
    
    def forward(self, feature, feature_lens):
        # feature: 包含 layer_num 个张量，每个形状为 (batch_size, seq_len, emb_dim)
        stacked_feature = torch.stack(feature, dim=0) 
        layer_num, batch_size, seq_len, emb_dim = stacked_feature.shape

        # 【优化】先在时间维度做 Average Pooling (极大降低后续门控计算的显存消耗)
        pooled_layers = []
        for b in range(batch_size):
            valid_len = feature_lens[b]
            # 取出该样本的所有层特征: (layer_num, valid_len, emb_dim)
            valid_feat = stacked_feature[:, b, :valid_len, :]
            # 在时间维度取平均: (layer_num, emb_dim)
            avg_feat = torch.mean(valid_feat, dim=1)
            pooled_layers.append(avg_feat)

        # 堆叠后形状: (batch_size, layer_num, emb_dim)
        pooled_features = torch.stack(pooled_layers, dim=0)

        # 【核心】计算动态门控权重
        # 对每层特征打分: (batch_size, layer_num, 1)
        gate_scores = self.gate(pooled_features)
        
        # 在 layer 维度进行 Softmax，得到样本级别的动态归一化权重
        dynamic_weights = F.softmax(gate_scores, dim=1)

        # 使用动态权重进行加权求和
        # (batch_size, layer_num, emb_dim) * (batch_size, layer_num, 1) -> sum over dim=1
        final_emb = (pooled_features * dynamic_weights).sum(dim=1)

        # 分类器输出
        proj_emb = F.relu(self.proj(final_emb)) # 加入 ReLU 增加非线性表达
        pred = self.out(proj_emb)
        
        return pred, final_emb, dynamic_weights

class Trainer():
    def __init__(self, config):
        self.config = config
        device = config.device

        # 回归读取原始的 .pickle 文件，保留原汁原味的 HuggingFace Dataset 结构和 V, A, D 标签
        with open(config.data, "rb") as f:
            dataset = pickle.load(f)
        
        train_data, val_data, test_data = dataset['train'], dataset['val'], dataset['test']

        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.num_labels = config.num_labels
        
        # 开启多进程加速和锁页内存
        self.train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate, num_workers=4, pin_memory=True)
        self.val_dataloader = DataLoader(val_data, batch_size=config.batch_size, collate_fn=my_collate, num_workers=4, pin_memory=True)
        self.test_dataloader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=my_collate, num_workers=4, pin_memory=True)

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        wavlm_param = self.get_upstream_param()
        self.downsample_rate = wavlm_param['downsample_rate']

        self.clf = EmotionClassifier(layer_num=wavlm_param['layer_num'], emb_dim=wavlm_param['emb_dim'], num_labels=config.num_labels)
        if self.config.load_path != '':
            self.load_model()
        self.clf.to(device)
        self.opt = torch.optim.Adam(self.clf.parameters(), lr=config.lr, weight_decay=config.reg_lr)
        self.best_accuracy = 0.0

        self.loss = nn.CrossEntropyLoss()
        self.write = config.write
        
        if config.use_wandb:
            wandb_save_path = "/dump"
            wandb.init(project=config.wandb_name, config=config, dir=wandb_save_path)
    
    def get_upstream_param(self):
        paired_wavs = torch.randn(16000).reshape(1, 16000).to(self.wavlm.device)
        with torch.no_grad():
            outputs = self.wavlm(paired_wavs, output_hidden_states=True)
        downsample_rate = round(max(len(wav) for wav in paired_wavs) / outputs.extract_features.size(1))
        layer_num = len(outputs.hidden_states)
        emb_dim = outputs.last_hidden_state.size(2)
        return {'downsample_rate': downsample_rate, "layer_num": layer_num, "emb_dim": emb_dim}
    
    def get_feature_seq_length(self, wav_attention_mask):
        actual_wav_length = wav_attention_mask.sum(dim=1).cpu().numpy()
        feature_lens = [round(wav_length / self.downsample_rate) for wav_length in actual_wav_length]
        return feature_lens

    def train_pass(self, epoch, is_training=True):
        config = self.config
        opt = self.opt
        if is_training:
            dataloader = self.train_dataloader
            status = 'TRAIN'
        else:
            dataloader = self.val_dataloader
            status = 'EVAL'

        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} | {status}", leave=False)
        for i, batch in enumerate(pbar):
            opt.zero_grad()

            input_dict = batch['audio']
            input_values = input_dict['input_values'].to(device)
            attention_mask = input_dict['attention_mask'].to(device)

            outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
            
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(attention_mask)

            pred_logits, final_emb, _ = self.clf(hiddens, feature_length)
            label = torch.tensor(batch['emotion'], device=device)
            
            loss = self.loss(pred_logits, label) 

            if is_training:
                loss.backward()
                opt.step()
            
            running_loss += loss.item()
            running_corrects += sum(pred_logits.argmax(1).cpu().numpy() == label.cpu().numpy())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)

        print('Epoch: {:d} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, status, epoch_loss, epoch_acc))
                
        if self.config.use_wandb:
            wandb.log({f"{status} Loss": epoch_loss, f"{status} accuracy": epoch_acc, "epoch": epoch})
        
        if epoch % 10 == 0:
            self.save_model(epoch, f"epoch_{epoch}")
        self.save_model(epoch)
        if not is_training and epoch_acc > self.best_accuracy:
            self.save_model(epoch, "best")
            self.best_accuracy = epoch_acc
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_pass(epoch)
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.train_pass(epoch, is_training=False)

    def eval(self):
        dataloader = self.val_dataloader
        status = 'EVAL'

        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        predictions = []
        gts = []
        for i, batch in enumerate(dataloader):
            input_dict = batch['audio']
            input_values = input_dict['input_values'].to(device)
            attention_mask = input_dict['attention_mask'].to(device)

            outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
            
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(attention_mask)
            pred_logits, final_emb, _ = self.clf(hiddens, feature_length)
            pred = F.softmax(pred, dim=1)

            label = torch.tensor(batch['emotion'], device=device)

            predictions.append(pred.argmax(1).cpu().numpy())
            gts.append(label.cpu().numpy())

        predictions = np.concatenate(predictions)
        gts = np.concatenate(gts)

        acc = balanced_accuracy_score(gts, predictions)
        war = recall_score(gts, predictions, average='weighted')
        uar = audmetric.unweighted_average_recall(gts, predictions)

        print(f"accuracy: {acc}, weighted recall: {war}, Unweighted Recall: {uar}")
            
    def inference(self):
        # 恢复 HuggingFace Dataset 的拼接方式
        dataset = concatenate_datasets([self.train_data, self.val_data, self.test_data])
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=my_collate, num_workers=4, pin_memory=True)

        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()
        self.clf.eval()

        embeddings = []
        predictions = []
        all_weights = [] # 【新增】：用于收集所有样本的动态权重

        for batch in tqdm(dataloader):
            input_dict = batch['audio']
            input_values = input_dict['input_values'].to(device)
            attention_mask = input_dict['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
                hiddens = outputs.hidden_states
                feature_length = self.get_feature_seq_length(attention_mask)
                pred, final_emb, batch_weights = self.clf(hiddens, feature_length)
                pred = F.softmax(pred, dim=1)
                
            predictions.append(pred.argmax(1).cpu().numpy())
            embeddings += [final_emb.detach().cpu().numpy()]
            # batch_weights 形状是 (batch_size, layer_num, 1)，去掉最后一维并转为 numpy
            all_weights.append(batch_weights.squeeze(-1).detach().cpu().numpy())
        
        predictions = np.concatenate(predictions)
        embeddings = np.concatenate(embeddings)
        all_weights = np.concatenate(all_weights, axis=0) # 【新增】：拼接成 (总样本数, 25) 的矩阵

        status = ["train"] * len(self.train_data) + ["val"] * len(self.val_data) + ["test"] * len(self.test_data)
        status = np.array(status)
        
        # 将完整包含 V, A, D 的 dataset 对象传入
        save_iemocap_partial(self.config.save_path, embeddings, status, predictions, dataset, all_weights)
    
    def save_model(self, epoch, type='last_epoch'):
        if self.write:
            save_name = "model_{}.pth".format(type)
            torch.save({'model_state_dict': self.clf.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'epoch': epoch,}, 
                        os.path.join(self.config.save_path, save_name))
    
    def load_model(self):
        checkpoint = torch.load(self.config.load_path, weights_only=False)
        self.clf.load_state_dict(checkpoint['model_state_dict'])

# 格式原封不动：依然直接使用 dataset['V'] 等语法，确保 AVLearner 读取完全一致
# 【修改点】：增加 dynamic_weights 参数，并设置默认值为 None 防止报错
def save_iemocap_partial(dump_path, embeddings, status, pred, dataset, dynamic_weights=None):
    save_data = {
        "embeddings": embeddings, 
        "emotion": dataset['emotion'], 
        "pred_emotion": pred,
        "status": status
    }
    # 保存权重矩阵
    if dynamic_weights is not None:
        save_data["dynamic_weights"] = dynamic_weights
        
    if 'V' in dataset[0]:
        save_data["V"] = [item['V'] for item in dataset]
        save_data["A"] = [item['A'] for item in dataset]
        save_data["D"] = [item['D'] for item in dataset]
        
    with open(os.path.join(dump_path, "embeddings.pickle"), "wb") as f:
        pickle.dump(save_data, f)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

    parser = argparse.ArgumentParser(description='Train MLP for multiclass classification')

    parser.add_argument('--name', type=str, default='tmp', help="folder name to store the model file")
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for optimizer')
    parser.add_argument('--reg-lr', type=float, default=1e-6, help='learning rate for optimizer')
    parser.add_argument('--num-epochs', type=int, default=21, help='number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')

    # 注意：请确保运行脚本里的 --data 参数指向最初的 .pickle 原始数据集
    parser.add_argument('--data', type=str, default='/data/audio_dataset.pickle', help='path to training data')
    parser.add_argument('--num-labels', type=int, default=4, help='number of categories in data')
    parser.add_argument('--save-path', type=str, default='/dump/', help='path to save trained model')
    parser.add_argument('--load-path', type=str, default='', help='path to load pretrained model') 

    parser.add_argument('--device', type=str, default='cuda:0', help='running device')

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', type=str, default='tmp', help='wandb name')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--mode', type=str, default='eval', help='running mode: train/eval/inference')

    args = parser.parse_args() 
    args.save_path = os.path.join(args.save_path, args.name)

    if not os.path.exists(args.save_path):
        print(f"save path {args.save_path} not exist, creating...")
        os.mkdir(args.save_path) 

    trainer = Trainer(args)
   
    if args.mode == 'train':
        trainer.train()  
    elif args.mode == 'eval':
        trainer.eval()
    else:
        trainer.inference()