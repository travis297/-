import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    base_dir = r"d:\Case-二手车价格预测"
    train_path = os.path.join(base_dir, "used_car_train_20200313.csv")
    test_path = os.path.join(base_dir, "used_car_testB_20200421.csv")
    output_dir = os.path.join(base_dir, "EDA_Figures")
    
    print("正在加载数据...")
    train_df = pd.read_csv(train_path, sep=' ')
    test_df = pd.read_csv(test_path, sep=' ')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    print("\n训练集缺失值:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
    
    print("\n测试集缺失值:")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0])
    
    print("\n价格统计信息:")
    print(train_df['price'].describe())
    
    # 1. 价格分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df['price'], kde=True)
    plt.title('价格分布 (原始)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(train_df['price']), kde=True, color='green')
    plt.title('价格分布 (Log变换后)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '价格分布分析.png'))
    plt.close()
    
    # 2. 类别特征与价格的关系
    cat_features = ['bodyType', 'fuelType', 'gearbox']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, col in enumerate(cat_features):
        sns.boxplot(x=col, y='price', data=train_df, ax=axes[i])
        axes[i].set_title(f'{col} vs Price')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '类别特征与价格关系.png'))
    plt.close()
    
    # 3. 数值特征与价格的关系 (散点图)
    num_features = ['power', 'kilometer']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, col in enumerate(num_features):
        # 采样以加速绘图并避免重叠
        sample = train_df.sample(n=10000, random_state=42)
        axes[i].scatter(sample[col], sample['price'], alpha=0.5, s=5)
        axes[i].set_title(f'{col} vs Price (Sampled)')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Price')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '数值特征与价格散点图.png'))
    plt.close()
    
    # 4. 训练集与测试集分布对比 (检查协变量偏移)
    # 我们检查几个关键特征
    features_to_check = ['power', 'kilometer', 'model', 'brand']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_check):
        sns.kdeplot(train_df[col], label='Train', ax=axes[i], fill=True, alpha=0.3)
        sns.kdeplot(test_df[col], label='Test', ax=axes[i], fill=True, alpha=0.3)
        axes[i].set_title(f'{col} Distribution Comparison')
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '训练集与测试集分布对比.png'))
    plt.close()
    
    print("\nEDA 图表已生成至 EDA_Figures 文件夹。")

if __name__ == "__main__":
    main()
