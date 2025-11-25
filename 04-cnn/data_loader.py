"""
MNIST数据加载器
自动下载并加载MNIST数据集
"""

import numpy as np
import gzip
import os
from pathlib import Path
import urllib.request


def download_mnist(data_dir='data'):
    """
    下载MNIST数据集

    参数:
        data_dir: 数据保存目录
    """
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    print("下载MNIST数据集...")
    for filename in files:
        filepath = data_path / filename
        if not filepath.exists():
            print(f"正在下载 {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                print(f"{filename} 下载完成")
            except Exception as e:
                print(f"下载 {filename} 失败: {e}")
                print("请手动下载MNIST数据集并放置到 data/ 目录")
                raise
        else:
            print(f"{filename} 已存在")


def load_mnist_images(filepath):
    """
    加载MNIST图像文件

    参数:
        filepath: 图像文件路径

    返回:
        images: 图像数组 (N, 28, 28)
    """
    with gzip.open(filepath, 'rb') as f:
        # 读取magic number和维度信息
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # 读取图像数据
        buf = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return images


def load_mnist_labels(filepath):
    """
    加载MNIST标签文件

    参数:
        filepath: 标签文件路径

    返回:
        labels: 标签数组 (N,)
    """
    with gzip.open(filepath, 'rb') as f:
        # 读取magic number和数量信息
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # 读取标签数据
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return labels


def load_mnist_data(data_dir='data', download=True):
    """
    加载MNIST数据集

    参数:
        data_dir: 数据目录
        download: 是否自动下载数据

    返回:
        X_train: 训练图像 (60000, 28, 28)
        y_train: 训练标签 (60000,)
        X_test: 测试图像 (10000, 28, 28)
        y_test: 测试标签 (10000,)
    """
    data_path = Path(data_dir)

    # 检查数据是否存在,不存在则下载
    train_images_file = data_path / 'train-images-idx3-ubyte.gz'
    if not train_images_file.exists() and download:
        download_mnist(data_dir)

    # 加载训练集
    print("加载训练集...")
    X_train = load_mnist_images(data_path / 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_path / 'train-labels-idx1-ubyte.gz')

    # 加载测试集
    print("加载测试集...")
    X_test = load_mnist_images(data_path / 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_path / 't10k-labels-idx1-ubyte.gz')

    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # 测试数据加载
    X_train, y_train, X_test, y_test = load_mnist_data()

    print("\n数据统计:")
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"图像尺寸: {X_train[0].shape}")
    print(f"标签范围: {y_train.min()} - {y_train.max()}")
    print(f"类别分布:")
    for i in range(10):
        count = np.sum(y_train == i)
        print(f"  数字 {i}: {count} 个样本")

    # 显示几个样本
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(f'Label: {y_train[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("\n样本图像已保存到 mnist_samples.png")
    plt.close()