import numpy as np
import gzip
import struct
from typing import Tuple

def load_idx_images(file_path: str) -> np.ndarray:
    """
    加载IDX格式的图像数据
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        images: 归一化后的图像数据，形状为(N, 784)
    """
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        images = images.reshape(num_images, -1).astype(np.float32) / 255.0
        return images

def load_idx_labels(file_path: str) -> np.ndarray:
    """
    加载IDX格式的标签数据
    
    Args:
        file_path: 标签文件路径
        
    Returns:
        labels: 标签数组
    """
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从本地文件加载Fashion-MNIST数据集
    
    Returns:
        X_train, y_train: 训练数据和one-hot编码标签
        X_val, y_val: 验证数据和one-hot编码标签  
        X_test, y_test: 测试数据和one-hot编码标签
    """
    train_images_path = 'train-images-idx3-ubyte.gz'
    train_labels_path = 'train-labels-idx1-ubyte.gz'
    test_images_path = 't10k-images-idx3-ubyte.gz'
    test_labels_path = 't10k-labels-idx1-ubyte.gz'

    X_train = load_idx_images(train_images_path)
    y_train = load_idx_labels(train_labels_path)
    X_test = load_idx_images(test_images_path)
    y_test = load_idx_labels(test_labels_path)

    def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        N = y.shape[0]
        y_onehot = np.zeros((N, num_classes))
        y_onehot[np.arange(N), y] = 1
        return y_onehot

    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)

    X_val, y_val = X_train[:10000], y_train_onehot[:10000]
    X_train, y_train = X_train[10000:], y_train_onehot[10000:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test_onehot