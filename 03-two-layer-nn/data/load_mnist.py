import numpy as np
import os
import gzip
import struct

def load_mnist_local(data_dir=None):
    """
    从本地.gz文件加载MNIST数据集
    """
    if data_dir is None:
        # .gz文件放在data/fashion-mnist目录下
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'fashion-mnist')
    
    print("从本地文件加载MNIST数据集...")
    
    # 检查文件是否存在
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz', 
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # 验证文件存在
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {filepath}")
            print("请确保以下文件存在:")
            for f in files.values():
                print(f"  - {f}")
            raise FileNotFoundError(f"找不到MNIST数据文件: {filename}")
    
    # 加载训练图像
    with gzip.open(os.path.join(data_dir, files['train_images']), 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        x_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    # 加载训练标签
    with gzip.open(os.path.join(data_dir, files['train_labels']), 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_train = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 加载测试图像  
    with gzip.open(os.path.join(data_dir, files['test_images']), 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        x_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    
    # 加载测试标签
    with gzip.open(os.path.join(data_dir, files['test_labels']), 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        y_test = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 转换为float32并归一化到[0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    print("✓ MNIST数据集加载成功!")
    print(f"训练集: {x_train.shape} ({x_train.shape[0]}张图像)")
    print(f"测试集: {x_test.shape} ({x_test.shape[0]}张图像)")
    print(f"像素范围: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"标签范围: {np.unique(y_train)}")
    
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    """
    主加载函数 - 使用本地文件
    """
    return load_mnist_local()

def get_mnist_batches(x, y, batch_size, shuffle=True):
    """
    将MNIST数据分成批次
    """
    num_samples = x.shape[0]
    
    if shuffle:
        indices = np.random.permutation(num_samples)
        x = x[indices]
        y = y[indices]
    
    batches = []
    for i in range(0, num_samples, batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        batches.append((x_batch, y_batch))
    
    return batches

# 测试函数
if __name__ == "__main__":
    try:
        data = load_mnist()
        print("MNIST数据加载测试成功!")
    except Exception as e:
        print(f"加载失败: {e}")