import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_dynamic_weights_heatmap(pickle_path, output_dir):
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "dynamic_weights" not in data:
        raise ValueError("Error: 'dynamic_weights' not found in the pickle file.")

    weights = data['dynamic_weights'] # 形状: (样本数, layer_num)
    emotions = data['emotion']        # 形状: (样本数,)
    
    # 情感标签映射字典 (请根据你 IEMOCAP 预处理的实际 mapping 调整)
    # 假设 0: Ang, 1: Hap, 2: Neu, 3: Sad
    emo_mapping = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
    
    # 1. 计算每种情感的平均层级权重
    layer_num = weights.shape[1]
    avg_weights_per_emo = {}
    
    for emo_idx, emo_name in emo_mapping.items():
        # 找出属于当前情感的所有样本的索引
        idx = np.where(np.array(emotions) == emo_idx)[0]
        if len(idx) > 0:
            # 提取这些样本的权重，并在样本维度(axis=0)上取平均
            emo_weights = weights[idx, :]
            avg_weights_per_emo[emo_name] = np.mean(emo_weights, axis=0)

    # 2. 转换为 Pandas DataFrame 方便 Seaborn 绘图
    # 行：WavLM Layers (0 到 24), 列：Emotions
    df = pd.DataFrame(avg_weights_per_emo)
    df.index = [f"Layer {i}" for i in range(layer_num)]

    # 为了让第 0 层在最下方，深层在最上方，我们将 DataFrame 翻转一下
    df = df.iloc[::-1]

    # 3. 设置高规格的学术绘图风格
    plt.figure(figsize=(10, 8), dpi=300) # 300 DPI 保证高清
    sns.set_theme(style="white")         # 纯白底色，去除网格干扰

    # 绘制热力图
    # cmap="YlOrRd" 呈现由浅黄到深红的渐变，非常适合表现注意力权重
    ax = sns.heatmap(df, 
                     cmap="YlOrRd", 
                     annot=False,            # 如果想在帖子上显示具体数值，可以改为 True
                     linewidths=0.5,         # 增加格子间的微小分割线，显得更清爽
                     cbar_kws={'label': 'Average Attention Weight'})

    # 4. 优化字体和标签细节
    plt.title('Dynamic Feature Fusion Weights across WavLM Layers', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Emotion Category', fontsize=14, fontweight='bold')
    plt.ylabel('WavLM Transformer Layers', fontsize=14, fontweight='bold')
    
    # 调整坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10, rotation=0)

    plt.tight_layout()

    # 5. 保存为高清 PDF (矢量图，放大不失真，期刊首选) 和 PNG
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf_path = os.path.join(output_dir, "layer_weights_heatmap.pdf")
    png_path = os.path.join(output_dir, "layer_weights_heatmap.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    
    print(f"Heatmaps successfully saved to {pdf_path} and {png_path}")
    plt.close()

if __name__ == "__main__":
    # 请将这里的路径替换为你实际生成 embeddings.pickle 的路径
    PICKLE_FILE = "./dump/tmp/embeddings.pickle"
    OUTPUT_FOLDER = "./plots"
    
    plot_dynamic_weights_heatmap(PICKLE_FILE, OUTPUT_FOLDER)