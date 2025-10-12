import numpy as np
import os
from sklearn.datasets import fetch_openml

def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    """
    使用NumPy实现训练集和测试集的划分
    
    参数:
    X: 特征数据
    y: 标签数据
    test_size: 测试集比例，默认0.2
    random_state: 随机种子
    
    返回:
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # 生成随机索引
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def load_and_save_mnist(data_dir='../data'):
    """
    下载 MNIST 数据集，进行预处理，并将其保存为 K-NN 模型所需的 .npy 文件。
    """
    print("-> 步骤 1: 正在下载 MNIST 数据集 (可能需要一些时间)...")
    
    # 修正：将 parser='auto' 更改为 parser='liac-arff'，以解决 pandas 依赖问题。
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X_data = mnist.data.astype(np.float32)
        y_data = mnist.target.astype(np.int8)
    except Exception as e:
        print(f"!!! 下载 MNIST 失败，请检查网络连接或依赖库: {e}")
        return

    print("-> 步骤 2: 数据预处理和归一化...")
    
    # 归一化：将像素值从 [0, 255] 缩放到 [0, 1]
    X_data /= 255.0
    
    print(f"   - 原始数据形状: {X_data.shape}, 标签形状: {y_data.shape}")

    print("-> 步骤 3: 使用NumPy划分训练集和测试集...")
    
    # 使用自定义的NumPy实现划分训练集和测试集
    # 为了与原始MNIST标准划分保持一致，我们使用约85.7%的训练集比例 (60000/70000 ≈ 0.857)
    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X_data, y_data, test_size=10000/70000, random_state=42
    )
    
    print(f"   - 训练集形状: {X_train.shape}")
    print(f"   - 测试集形状: {X_test.shape}")

    # 确保 'data' 目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"   - 创建了目录: '{data_dir}'")

    print("-> 步骤 4: 保存数据文件...")
    
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    print("\n--- 任务完成 ---")
    print(f"🎉 真实 MNIST 数据已成功保存到 '{data_dir}/' 目录中。")
    print(f"   - 训练集 X_train 形状: {X_train.shape}")
    print(f"   - 训练集 y_train 形状: {y_train.shape}")
    print(f"   - 测试集 X_test 形状: {X_test.shape}")
    print(f"   - 测试集 y_test 形状: {y_test.shape}")
    print(f"   - 训练集样本数: {len(y_train)}")
    print(f"   - 测试集样本数: {len(y_test)}")

if __name__ == '__main__':
    load_and_save_mnist()