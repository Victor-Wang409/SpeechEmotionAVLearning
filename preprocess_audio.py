import pickle
import torch
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import os

def preprocess_and_save_dynamic(input_pickle_path, output_pt_path):
    print("正在加载预训练的 Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')

    print(f"正在读取原始数据集: {input_pickle_path}")
    with open(input_pickle_path, "rb") as f:
        dataset = pickle.load(f)

    processed_dataset = {}

    for split in ['train', 'val', 'test']:
        if split not in dataset:
            continue
            
        print(f"\n开始逐条提取 {split} 集特征 (不进行 Padding)...")
        data_list = dataset[split]
        
        # 存储处理好的字典列表，而不是一个巨大的固定 Tensor
        split_data = []

        for item in tqdm(data_list, desc=f"处理 {split}"):
            audio_array = item['audio']['array']
            emotion = item['emotion']

            # 针对单条音频提取特征 (自动完成归一化等操作)
            processed = feature_extractor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # 取出 1D 张量，形状从 (1, L) 压缩为 (L,)
            input_val_1d = processed['input_values'].squeeze(0)

            split_data.append({
                "input_values": input_val_1d,
                "emotion": emotion
            })

        processed_dataset[split] = split_data
        print(f"{split} 集处理完毕! 共 {len(split_data)} 条变长音频。")

    print(f"\n正在保存变长数据集至: {output_pt_path}")
    torch.save(processed_dataset, output_pt_path)
    print("全部动态特征预处理完成！")

if __name__ == "__main__":
    INPUT_PICKLE = './data/audio_partial5_train_dataset.pickle'
    OUTPUT_PT = './data/processed_dynamic_dataset.pt'
    preprocess_and_save_dynamic(INPUT_PICKLE, OUTPUT_PT)