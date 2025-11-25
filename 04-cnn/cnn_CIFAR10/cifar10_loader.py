"""
CIFAR-10数据加载器
自动下载并加载CIFAR-10数据集
"""

import numpy as np
import pickle
import os
from pathlib import Path
import urllib.request
import tarfile


def download_cifar10(data_dir='data'):
    """
    下载CIFAR-10数据集
    
    参数:
        data_dir: 数据保存目录
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    filepath = data_path / filename
    extract_path = data_path / 'cifar-10-batches-py'
    
    # 检查是否已经下载和解压
    if extract_path.exists():
        print("CIFAR-10数据集已存在")
        return
    
    # 下载
    if not filepath.exists():
        print(f"开始下载CIFAR-10数据集 (大小: ~170MB)...")
        print("这可能需要几分钟...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("下载完成!")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载CIFAR-10数据集:")
            print(f"  URL: {url}")
            print(f"  保存到: {filepath}")
            raise
    
    # 解压
    print("正在解压...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(data_path)
    print("解压完成!")


def load_cifar10_batch(file):
    """
    加载CIFAR-10的一个batch文件
    
    参数:
        file: batch文件路径
    
    返回:
        images: 图像数据 (N, 3, 32, 32)
        labels: 标签 (N,)
    """
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # 获取图像和标签
    images = batch[b'data']
    labels = batch[b'labels']
    
    # 重塑图像: (N, 3072) -> (N, 3, 32, 32)
    # CIFAR-10格式: [R通道1024个像素, G通道1024个像素, B通道1024个像素]
    images = images.reshape(-1, 3, 32, 32)
    labels = np.array(labels)
    
    return images, labels


def load_cifar10_data(data_dir='data', download=True):
    """
    加载CIFAR-10数据集
    
    参数:
        data_dir: 数据目录
        download: 是否自动下载数据
    
    返回:
        X_train: 训练图像 (50000, 3, 32, 32)
        y_train: 训练标签 (50000,)
        X_test: 测试图像 (10000, 3, 32, 32)
        y_test: 测试标签 (10000,)
    """
    cifar_dir = Path(data_dir) / 'cifar-10-batches-py'
    
    # 检查数据是否存在，不存在则下载
    if not cifar_dir.exists() and download:
        download_cifar10(data_dir)
    
    # 加载训练集 (5个batch)
    print("加载CIFAR-10训练集...")
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch_file = cifar_dir / f'data_batch_{i}'
        images, labels = load_cifar10_batch(batch_file)
        X_train_list.append(images)
        y_train_list.append(labels)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # 加载测试集
    print("加载CIFAR-10测试集...")
    test_file = cifar_dir / 'test_batch'
    X_test, y_test = load_cifar10_batch(test_file)
    
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test


def get_cifar10_class_names():
    """获取CIFAR-10类别名称"""
    return [
        'airplane',    # 0: 飞机
        'automobile',  # 1: 汽车
        'bird',        # 2: 鸟
        'cat',         # 3: 猫
        'deer',        # 4: 鹿
        'dog',         # 5: 狗
        'frog',        # 6: 青蛙
        'horse',       # 7: 马
        'ship',        # 8: 船
        'truck'        # 9: 卡车
    ]


def normalize_cifar10(X_train, X_test):
    """
    CIFAR-10数据归一化
    
    方法1: 简单归一化到[0, 1]
    方法2: 标准化（减均值除标准差）
    
    这里使用方法2，效果更好
    """
    # 转换为float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # 计算训练集的均值和标准差（每个通道分别计算）
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    
    # 标准化
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    
    return X_train, X_test, mean, std


def simple_normalize_cifar10(X_train, X_test):
    """
    简单归一化到[0, 1]
    """
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return X_train, X_test


if __name__ == '__main__':
    # 测试数据加载
    print("="*60)
    print("测试CIFAR-10数据加载")
    print("="*60)
    
    X_train, y_train, X_test, y_test = load_cifar10_data()
    
    print("\n数据统计:")
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"图像形状: {X_train[0].shape} (C, H, W)")
    print(f"像素范围: [{X_train.min()}, {X_train.max()}]")
    print(f"标签范围: {y_train.min()} - {y_train.max()}")
    
    # 类别分布
    class_names = get_cifar10_class_names()
    print("\n类别分布:")
    for i in range(10):
        count = np.sum(y_train == i)
        print(f"  {i}: {class_names[i]:<12} - {count} 个样本")
    
    # 可视化几个样本
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        # CIFAR-10: (C, H, W) -> (H, W, C)
        img = X_train[i].transpose(1, 2, 0)
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'{class_names[y_train[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("\n样本图像已保存到 cifar10_samples.png")
    plt.close()
    
    # 测试归一化
    print("\n测试归一化...")
    X_train_norm, X_test_norm, mean, std = normalize_cifar10(X_train, X_test)
    print(f"归一化后范围: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    print(f"均值: {mean.flatten()}")
    print(f"标准差: {std.flatten()}")