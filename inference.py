import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入项目模块
try:
    from model import VADConfig, VADModelWithGating
    from dataset import EmotionDataset
    from data_processor import DataProcessor
except ImportError as e:
    print(f"导入错误: 请确保 model.py, dataset.py 和 data_processor.py 在当前目录下。\n详细信息: {e}")
    exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="Inference Script for VAD Model")
    
    # 1. 模型路径
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the model checkpoint (e.g., checkpoint.pt)')
    
    # 2. 数据路径
    parser.add_argument('--emotion2vec_dir', type=str, required=True, help='Directory containing emotion2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True, help='Directory containing hubert features')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--wav2vec_dir', type=str, default=None, help='Directory containing wav2vec features (optional)')
    parser.add_argument('--data2vec_dir', type=str, default=None, help='Directory containing data2vec features (optional)')
    
    # 3. 推理设置
    parser.add_argument('--output_path', type=str, default='embeddings.pickle', help='Path to save the output pickle file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # [新增] 手动覆盖参数 (如果 checkpoint 中没有 config)
    parser.add_argument('--num_groups', type=int, default=16, help='Number of groups for gating (default: 16)')
    parser.add_argument('--num_emotions', type=int, default=4, help='Number of emotion classes (default: 4)')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. 加载 Checkpoint 并解析配置
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    
    # 先加载 checkpoint 字典
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    else:
        # 如果是目录，假设是 huggingface 格式，稍后处理
        checkpoint = None

    # 初始化配置
    config = None
    
    # 尝试从 checkpoint 中读取训练时的配置 (这是最稳健的方法)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("Found configuration in checkpoint, loading...")
        saved_config = checkpoint['config']
        # 将字典转换为 Config 对象
        config = VADConfig(**saved_config)
    else:
        print("No configuration found in checkpoint. Using command line args/defaults.")
        # 如果 checkpoint 中没有 config，则使用默认值或命令行参数
        # 注意：这里默认值改为了匹配您报错信息的 16 和 4
        config = VADConfig(
            emotion2vec_dim=1024,
            hubert_dim=1024,
            hidden_dim=1024,
            intermediate_dim=1024,
            num_hidden_layers=4,
            num_groups=args.num_groups,       # 使用参数: 16
            num_emotions=args.num_emotions,   # 使用参数: 4
            wav2vec_dim=0 if args.wav2vec_dir is None else 1024,
            data2vec_dim=0 if args.data2vec_dir is None else 1024,
        )

    print(f"Model Configuration: num_groups={config.num_groups}, num_emotions={config.num_emotions}")

    # -------------------------------------------------------------------------
    # 2. 初始化模型并加载权重
    # -------------------------------------------------------------------------
    model = VADModelWithGating(config)

    # 加载权重
    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 加载权重 (strict=True 确保完全匹配)
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully.")
        except RuntimeError as e:
            print(f"Weight loading failed: {e}")
            print("Attempting with strict=False...")
            model.load_state_dict(state_dict, strict=False)
    
    elif os.path.isdir(args.checkpoint_path):
        # HuggingFace 文件夹格式加载
        model = VADModelWithGating.from_pretrained(args.checkpoint_path)

    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 3. 准备数据
    # -------------------------------------------------------------------------
    print("Loading dataset...")
    dataset = EmotionDataset(
        emotion2vec_dir=args.emotion2vec_dir,
        hubert_dir=args.hubert_dir,
        csv_path=args.csv_path,
        wav2vec_dir=args.wav2vec_dir,
        data2vec_dir=args.data2vec_dir
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataProcessor.collate_fn,
        num_workers=4
    )

    # -------------------------------------------------------------------------
    # 4. 执行推理
    # -------------------------------------------------------------------------
    results = {
        "embeddings": [],
        "pred_emotion": [],
        "emotion": [],
        "V": [], "A": [], "D": [],
        "id": [],
    }
    
    print(f"Starting inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 1. 准备输入特征
            input_features = {}
            if 'emotion2vec_features' in batch:
                input_features['emotion2vec'] = batch['emotion2vec_features'].to(device)
            if 'hubert_features' in batch:
                input_features['hubert'] = batch['hubert_features'].to(device)
            if 'wav2vec_features' in batch:
                input_features['wav2vec'] = batch['wav2vec_features'].to(device)
            if 'data2vec_features' in batch:
                input_features['data2vec'] = batch['data2vec_features'].to(device)
                
            padding_mask = batch['padding_mask'].to(device)

            # 2. 前向传播
            logits, gate_weights, pooled_features, _ = model(input_features, padding_mask=padding_mask)

            # 3. 处理输出
            preds = logits.argmax(dim=1).cpu().numpy()
            embeddings = pooled_features.cpu().numpy()
            
            # 4. 获取 Ground Truth
            gt_emotions = batch['emotion_labels'].argmax(dim=1).numpy()
            vad_values = batch['labels'].numpy()

            # 5. 收集数据
            results["embeddings"].append(embeddings)
            results["pred_emotion"].append(preds)
            results["emotion"].append(gt_emotions)
            results["V"].append(vad_values[:, 0])
            results["A"].append(vad_values[:, 1])
            results["D"].append(vad_values[:, 2])
            results["id"].extend(batch["id"])

    # -------------------------------------------------------------------------
    # 5. 整合与保存
    # -------------------------------------------------------------------------
    print("Concatenating results...")
    
    final_data = {
        "embeddings": np.concatenate(results["embeddings"], axis=0),
        "pred_emotion": np.concatenate(results["pred_emotion"], axis=0),
        "emotion": np.concatenate(results["emotion"], axis=0),
        "V": np.concatenate(results["V"], axis=0),
        "A": np.concatenate(results["A"], axis=0),
        "D": np.concatenate(results["D"], axis=0),
        "id": results["id"]
    }

    # 获取 Status
    try:
        df = pd.read_csv(args.csv_path)
        if 'Set' in df.columns and len(df) == len(final_data["id"]):
            final_data["status"] = df['Set'].values
        else:
            final_data["status"] = np.array(['inference'] * len(final_data["id"]))
    except Exception as e:
        final_data["status"] = np.array(['inference'] * len(final_data["id"]))

    print(f"Saving embeddings to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(final_data, f)
        
    print("Done!")

if __name__ == "__main__":
    main()