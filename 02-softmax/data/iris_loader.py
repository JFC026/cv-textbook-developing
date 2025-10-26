import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

def load_iris_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载鸢尾花数据集并进行预处理
    
    Returns:
        X_train, y_train: 标准化后的训练数据和one-hot编码标签
        X_val, y_val: 标准化后的验证数据和one-hot编码标签
    """
    df = pd.read_csv('iris.csv')
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y = df['species'].map(label_map).values.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val