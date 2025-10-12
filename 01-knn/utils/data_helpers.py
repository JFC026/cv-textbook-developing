import numpy as np
import os

def load_data(data_dir='../data'):
    """
    加载图像数据。
    """
    print("-> 正在加载原始数据...")
    
    X_train_path = os.path.join(data_dir, 'X_train.npy')
    y_train_path = os.path.join(data_dir, 'y_train.npy')
    X_test_path = os.path.join(data_dir, 'X_test.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')
    
    try:
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
    except FileNotFoundError as e:
        print(f"!!! 致命错误：未找到数据文件。请确保以下文件存在于 '{data_dir}' 目录中：")
        print(f"    - {X_train_path}\n    - {y_train_path}")
        raise e
        
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    """
    对图像数据进行展平（Flatten）和预处理（Normalization）。
    """
    print("-> 正在进行数据展平和预处理...")

    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(num_train, -1)
    X_test_flat = X_test.reshape(num_test, -1)
    
    mean_image = np.mean(X_train_flat, axis=0)
    X_train_processed = X_train_flat - mean_image
    X_test_processed = X_test_flat - mean_image
    
    print(f"    - 原始数据形状（训练）: {X_train.shape} -> 展平后: {X_train_processed.shape}")
    print(f"    - 原始数据形状（测试）: {X_test.shape} -> 展平后: {X_test_processed.shape}")
    print("-> 预处理完成。")
    
    return X_train_processed, y_train, X_test_processed, y_test