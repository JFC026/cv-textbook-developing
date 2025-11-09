# data/improved_preprocess.py
import numpy as np

def preprocess_data(x, y, normalize=True, standardize=False):
    """
    数据预处理
    """
    if normalize:
        x = x.astype(np.float32) / 255.0
    
    if standardize:
        # 数据标准化：减去均值，除以标准差
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean) / (std + 1e-8)
    
    # 确保数据形状正确
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    
    return x, y

def data_augmentation(x_batch, y_batch):
    """简单数据增强"""
    augmented_x = []
    augmented_y = []
    
    for x, y in zip(x_batch, y_batch):
        # 原始样本
        augmented_x.append(x)
        augmented_y.append(y)
        
        # 水平翻转（对图像数据）
        if len(x) == 3072:  # CIFAR-10形状
            x_reshaped = x.reshape(3, 32, 32).transpose(1, 2, 0)
            x_flipped = np.fliplr(x_reshaped).transpose(2, 0, 1).reshape(3072)
            augmented_x.append(x_flipped)
            augmented_y.append(y)
    
    return np.array(augmented_x), np.array(augmented_y)