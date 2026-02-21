import sys, os
import matplotlib.pyplot as plt
sys.path.append('module/umap')
import umap
import numpy as np
import pandas as pd
import pickle

# --- Emotional Anchors Definitions ---

def ld_mead():
    # ['A', 'C', 'D', 'F', 'H', 'N', 'S', 'U']
    ld = np.array([[-0.43,0.67],[-0.8,0.2],[-0.6,0.35],[-0.64,0.6],[0.76,0.48],[0,0],[-0.63,-0.27],[0, 0.6]] )
    return np.array(ld)

def ld_iemocap():
    """
    对应 dataset.py (4类):
    0: ANG (Angry)
    1: HAP (Happy)
    2: NEU (Neutral)
    3: SAD (Sad)
    """
    # 坐标: [Valence, Arousal, Dominance]
    ld = np.array([
        [-0.51,  0.59, 0.25], # 0: ang
        [ 0.81,  0.51, 0.46], # 1: hap
        [ 0.00,  0.00, 0.00], # 2: neu
        [-0.63, -0.27, -0.33]  # 3: sad
    ])
    return np.array(ld)

def ld_iemocap_partial5():
    # ["ANG",'dis',"HAP","NEU","SAD"]
    ld = np.array([
        [-0.51,  0.59,  0.25],
        [-0.60,  0.35,  0.11],
        [ 0.81,  0.51,  0.46],
        [ 0.00,  0.00,  0.00],
        [-0.63, -0.27, -0.33]
    ])
    return np.array(ld)

def ld_emodb():
    # ['angry', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    return np.array([
        [-0.43,0.67],
        [-0.65,-0.62],
        [-0.6,0.35],
        [-0.64,0.6],
        [0.76,0.48],
        [0,0],
        [-0.63,-0.27]])

# --- Utils ---

def load_data(name):
    # 默认加载路径
    path = "./dump/inference/tmp/embeddings.pickle"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find embeddings.pickle in current directory")

    print(f"Loading data from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    # 自动检测类别数或根据 name 指定
    unique_labels = len(np.unique(data['emotion']))
    
    if name == "iemocap_partial5":
        init_global = ld_iemocap_partial5()
    elif unique_labels == 4:
        print(f"Detected 4 classes. Using ld_iemocap (Ang, Hap, Neu, Sad).")
        init_global = ld_iemocap()
    elif unique_labels == 5:
        print(f"Detected 5 classes. Using ld_iemocap_partial5.")
        init_global = ld_iemocap_partial5()
    else:
        # 默认回退到 4 类 (因为您明确要求了)
        print(f"Warning: Detected {unique_labels} classes, but defaulting to ld_iemocap (4 classes) as requested.")
        init_global = ld_iemocap()

    embeddings, label = data['embeddings'], data['emotion']
    return {"embedding":embeddings, "label": label, "init_global": init_global, "data":data}

# --- Anchored Dimensionality Reduction ---

class AVLearner:
    def __init__(self,
                n_neighbors=20,
                n_epochs=150,
                negative_sample_rate=10,
                min_dist=0.1,
                learning_rate=0.01,
                target_weight=0.1,
                repulsion_strength=0.1,
                n_components=3  # [修改] 默认为 3 维
                ) -> None:
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_epochs=n_epochs,
            n_components=n_components, # [修改] 设置输出维度
            negative_sample_rate=negative_sample_rate,
            min_dist=min_dist,
            learning_rate=learning_rate,
            target_metric="categorical",
            target_metric_kwds = {},
            target_weight=target_weight,
            repulsion_strength=repulsion_strength,
            spread=1.0
        )
    
    def fit(self, embedding, labels, anchor_mappings):
        init = self.reducer.set_custom_intialization(embedding, labels, anchor_mappings)
        self.reducer.fit(embedding, labels)
    
    def transform(self, embedding):
        return self.reducer.transform(embedding)

    def fit_transform(self, embedding, labels, anchor_mappings):
        # anchor_mappings 现在是 (4, 3) 的矩阵
        init = self.reducer.set_custom_intialization(embedding, labels, anchor_mappings)
        return self.reducer.fit_transform(embedding, labels)

# --- Logic: No Split (Unified Processing) ---

def train_inference(data):
    """
    不再区分 Train/Test，对所有数据进行统一拟合与转换
    """
    init_global = data['init_global']
    raw_data = data['data']
    
    # 提取所有数据
    embedding = np.array(raw_data['embeddings'])
    label = np.array(raw_data['emotion']) # 使用 Ground Truth Label 进行锚点对齐
    
    # 安全检查：确保 Label 没有超出锚点范围
    # 如果只有4个锚点，label 必须是 0,1,2,3
    if label.max() >= len(init_global):
        print(f"Error: Max label index ({label.max()}) exceeds anchor count ({len(init_global)}).")
        print("Please check if dataset.py and embeddings.pickle are consistent with 4 classes.")
        sys.exit(1)
    
    print(f"Processing {len(embedding)} samples with AVLearner (No Split)...")

    reducer = AVLearner(n_components=3)
    # 对所有数据进行 fit_transform
    final_coords = reducer.fit_transform(embedding, label, init_global)
    
    return final_coords

def calc_ccc(x, y):
    """
    计算 Concordance Correlation Coefficient (CCC)
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    covariance = np.mean((x - x_mean) * (y - y_mean))
    
    x_var = np.var(x)
    y_var = np.var(y)
    
    # 防止分母为0
    denominator = x_var + y_var + (x_mean - y_mean)**2
    if denominator == 0:
        return 0.0
        
    ccc = (2 * covariance) / denominator
    return ccc


if __name__ == "__main__":
    # 1. 加载数据
    data = load_data('iemocap_partial5')
    
    # 2. 运行 AVLearner (3D)
    final_coords = train_inference(data) # shape: (N, 3)

    # 3. 评估
    raw_data = data['data']
    
    # 确保 V、A、D 存在
    if 'V' in raw_data and 'A' in raw_data and 'D' in raw_data:
        gt_val = np.array(raw_data['V'])
        gt_aro = np.array(raw_data['A'])
        gt_dom = np.array(raw_data['D']) # [新增] 获取真实 Dominance

        # 提取预测值
        pred_val = final_coords[:, 0]
        pred_aro = final_coords[:, 1]
        pred_dom = final_coords[:, 2] # [新增] 第3维是 Dominance

        if len(pred_val) == len(gt_val):
            ccc_v = calc_ccc(pred_val, gt_val)
            ccc_a = calc_ccc(pred_aro, gt_aro)
            ccc_d = calc_ccc(pred_dom, gt_dom) # [新增] 计算 Dominance CCC

            print("\n" + "="*40)
            print(" >>> 最终评估结果 (3D VAD) <<<")
            print("-" * 40)
            print(f" Valence   CCC: {ccc_v:.4f}")
            print(f" Arousal   CCC: {ccc_a:.4f}")
            print(f" Dominance CCC: {ccc_d:.4f}")
            print("="*40 + "\n")
            
            # 保存结果包含 Dominance
            np.savez("av_results_3d.npz", 
                     pred=final_coords, 
                     gt_v=gt_val, gt_a=gt_aro, gt_d=gt_dom, 
                     labels=raw_data['emotion'])
            print("Results saved to av_results_3d.npz")
    else:
        print("Warning: 'V', 'A', or 'D' labels not found. Skipping evaluation.")