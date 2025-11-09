# data/load_data.py
import numpy as np
import pickle
import os
from urllib.request import urlretrieve
import tarfile


def download_cifar10(data_dir, force_download=False):
    """下载并解压CIFAR-10数据集"""
    if force_download and os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    
    print("下载CIFAR-10数据集...")
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    try:
        # 下载文件
        print("正在下载...")
        urlretrieve(url, tar_path)
        print("下载完成，正在解压...")
        
        # 解压文件
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(data_dir))
        
        # 移动文件到正确位置
        extracted_dir = os.path.join(os.path.dirname(data_dir), 'cifar-10-batches-py')
        if os.path.exists(extracted_dir) and extracted_dir != data_dir:
            # 如果解压到了不同目录，移动文件
            import shutil
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            shutil.move(extracted_dir, data_dir)
        
        # 清理压缩文件
        if os.path.exists(tar_path):
            os.remove(tar_path)
        
        print("数据集准备完成！")
        
    except Exception as e:
        print(f"下载失败: {e}")


def load_cifar10(data_dir='./03-two-layer-nn/data/cifar-10-batches-py'):
    """
    加载CIFAR-10数据集
    """
    # 如果数据目录不存在，则下载数据
    if not os.path.exists(data_dir):
        download_cifar10(data_dir)
    
    # 检查数据文件是否存在
    batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
    if not all(os.path.exists(f) for f in batch_files):
        print("数据文件不完整，重新下载...")
        download_cifar10(data_dir, force_download=True)
    
    # 加载训练数据
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        print(f"加载文件: {batch_file}")
        
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            # CIFAR-10的数据键名是bytes类型
            x_train.append(batch[b'data'])
            y_train.append(batch[b'labels'])
    
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)
    
    # 加载测试数据
    test_file = os.path.join(data_dir, 'test_batch')
    print(f"加载测试文件: {test_file}")
    
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        x_test = batch[b'data']
        y_test = np.array(batch[b'labels'])
    
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)




def preprocess_data(x, y, normalize=True, flatten=True):
    """
    数据预处理
    - 归一化像素值到[0, 1]
    - 调整数据形状
    """
    if normalize:
        x = x.astype(np.float32) / 255.0
    
    if flatten:
        # 如果数据是图像格式(32,32,3)，展平它
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        # 如果已经是展平的，确保形状正确
        elif x.shape[1] == 3072:
            # 已经是正确的形状
            pass
        else:
            # 尝试自动检测并调整形状
            x = x.reshape(x.shape[0], -1)
    
    return x, y

def get_data_batches(x, y, batch_size, shuffle=True):
    """将数据划分为批次"""
    num_samples = x.shape[0]
    
    if shuffle:
        indices = np.random.permutation(num_samples)
    else:
        indices = np.arange(num_samples)
    
    batches = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        
        batches.append((x_batch, y_batch))
    
    return batches


# 测试函数
if __name__ == "__main__":
    print("\n加载数据...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")