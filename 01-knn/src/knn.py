"""
knn_classifier.py

基于纯 NumPy 的 K-最近邻 (K-NN) 示例实现与示范流程。

功能包括:
- KNearestNeighbor 类（训练=存储数据、无循环距离计算、分批预测、K 折交叉验证）
- 本地数据加载函数 (从 .npy)
- 数据预处理函数（展平 + 去均值）
- 主脚本流程示例：加载 -> 预处理 -> 交叉验证选择 K -> 使用最佳 K 在测试集上评估

"""

import numpy as np
import os

np.random.seed(42)


class KNearestNeighbor(object):
    """
    K-NN 分类器：使用纯 Python 和 Numpy 实现，专注于高性能矩阵运算。
    """

    def __init__(self):
        """初始化分类器。"""
        self.X_train = None
        self.y_train = None
        self.best_k = None  # 存储最佳K值

    def train(self, X, y):
        """K-NN 分类器仅存储数据。"""
        self.X_train = X.astype(np.float64)
        self.y_train = y

    def predict(self, X, k=1, batch_size=1000):
        """
        使用分批处理对测试数据 X 进行预测，避免内存溢出。
        """
        X_test = X.astype(np.float64)
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        # 分批处理
        num_batches = (num_test + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_test)

            print(f"   处理批次 {i + 1}/{num_batches} ({end_idx - start_idx} 个样本)...")

            X_batch = X_test[start_idx:end_idx]

            # 1. 计算距离矩阵 (batch_size x N)
            dists = self.compute_distances_no_loops(X_batch)

            # 2. 根据距离进行 K-NN 投票预测
            y_pred[start_idx:end_idx] = self.predict_labels(dists, k=k)

        return y_pred

    def compute_distances_no_loops(self, X):
        """
        【核心实现】使用纯 Numpy 矩阵运算，实现欧氏距离 (L2) 计算。
        避免任何 Python 循环。
        """
        # --- 欧氏距离平方的矩阵化公式: ||x - y||^2 = ||x||^2 - 2*x*y.T + ||y||^2 ---

        # 1. 计算测试集的平方和 (M x 1 矩阵)
        X_test_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]

        # 2. 计算训练集的平方和 (1 x N 矩阵)
        X_train_sq = np.sum(self.X_train ** 2, axis=1)[np.newaxis, :]

        # 3. 计算交叉项 (M x N 矩阵)
        cross_term = X @ self.X_train.T

        # 4. 组合所有项
        dists_sq = X_test_sq + X_train_sq - 2 * cross_term

        # 5. 最后取平方根，并确保没有负值（防止浮点精度误差）
        dists = np.sqrt(np.maximum(dists_sq, 0))

        return dists

    def predict_labels(self, dists, k=1):
        """
        根据距离矩阵，找到 K 个最近邻并进行多数投票。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            # 1. 获取距离排序后的 K 个最近邻索引
            closest_indices = np.argsort(dists[i, :])
            k_nearest_labels = self.y_train[closest_indices[:k]]

            # 2. 进行多数投票：统计出现次数最多的标签
            counts = np.bincount(k_nearest_labels)
            y_pred[i] = np.argmax(counts)

        return y_pred

    def k_fold_split(self, n_samples, n_folds=5, random_state=42):
        """
        使用NumPy实现K折交叉验证的数据划分
        """
        if random_state is not None:
            np.random.seed(random_state)

        # 生成随机索引
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds

        for fold in range(n_folds):
            # 验证集的起始和结束位置
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

            # 验证集索引
            val_indices = indices[val_start:val_end]

            # 训练集索引（除了验证集以外的所有样本）
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            yield train_indices, val_indices

    def cross_validate(self, X_train, y_train, k_values=[1, 3, 5, 7, 9], n_folds=5):
        """
        使用交叉验证选择最佳的K值。
        """
        print(f"-> 开始 {n_folds} 折交叉验证，测试K值: {k_values}")

        k_scores = {k: [] for k in k_values}

        for fold, (train_idx, val_idx) in enumerate(self.k_fold_split(len(X_train), n_folds), 1):
            print(f"   第 {fold}/{n_folds} 折...")

            # 分割数据
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # 训练当前折的模型
            self.train(X_fold_train, y_fold_train)

            # 计算距离矩阵（只计算一次，供所有K值使用）
            dists = self.compute_distances_no_loops(X_fold_val)

            # 测试所有K值
            for k in k_values:
                y_pred = self.predict_labels(dists, k=k)
                accuracy = np.mean(y_pred == y_fold_val)
                k_scores[k].append(accuracy)

        # 计算每个K值的平均准确率
        k_mean_scores = {k: np.mean(scores) for k, scores in k_scores.items()}

        # 选择最佳K值
        best_k = max(k_mean_scores.items(), key=lambda x: x[1])[0]
        self.best_k = best_k

        print("\n--- 交叉验证结果 ---")
        for k in k_values:
            mean_acc = k_mean_scores[k] * 100
            std_acc = np.std(k_scores[k]) * 100
            marker = " ← 最佳" if k == best_k else ""
            print(f"K={k}: {mean_acc:.2f}% ± {std_acc:.2f}%{marker}")

        return best_k, k_mean_scores


# =========================================================================
# 图像数据加载与预处理模块
# =========================================================================

def load_data(data_dir='data'):
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


if __name__ == '__main__':

    # ==================================================
    # 步骤 1: 加载和预处理数据
    # ==================================================
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(data_dir='data')
    X_train, y_train, X_test, y_test = preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

    # ==================================================
    # 步骤 2: 使用交叉验证选择最佳K值
    # ==================================================
    classifier = KNearestNeighbor()

    # 为了节省时间，使用部分数据进行交叉验证
    cv_sample_size = 10000  # 用于交叉验证的样本数
    if len(X_train) > cv_sample_size:
        indices = np.random.choice(len(X_train), cv_sample_size, replace=False)
        X_train_cv = X_train[indices]
        y_train_cv = y_train[indices]
    else:
        X_train_cv = X_train
        y_train_cv = y_train

    print(f"\n-> 使用 {len(X_train_cv)} 个样本进行交叉验证...")

    # 定义要测试的K值范围
    k_values = [1, 3, 5, 7, 9, 11]
    best_k, k_scores = classifier.cross_validate(X_train_cv, y_train_cv, k_values=k_values, n_folds=3)

    # ==================================================
    # 步骤 3: 使用最佳K值在完整训练集上训练并测试
    # ==================================================
    print(f"\n-> 使用最佳 K={best_k} 在完整训练集上训练...")
    classifier.train(X_train, y_train)

    # 在测试集上评估
    test_subset_size = 10000  # 现在可以处理全部测试集
    X_test_subset = X_test[:test_subset_size]
    y_test_subset = y_test[:test_subset_size]

    print(f"-> 在测试集上评估 (K={best_k}, {test_subset_size} 个样本)...")
    print("   注意：使用分批处理避免内存溢出...")

    # 使用分批处理
    y_pred = classifier.predict(X_test_subset, k=best_k, batch_size=1000)

    # ==================================================
    # 步骤 4: 评估结果
    # ==================================================
    accuracy = np.mean(y_pred == y_test_subset) * 100

    print("\n--- 最终测试结果 ---")
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试子集形状: {X_test_subset.shape}")
    print(f"通过交叉验证选择的最佳 K 值: {best_k}")
    print(f"测试集分类准确率: {accuracy:.2f}%")

    # 显示每个类别的准确率
    print("\n--- 各类别准确率 ---")
    for i in range(10):
        mask = y_test_subset == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test_subset[mask]) * 100
            print(f"类别 {i}: {class_acc:.2f}% ({np.sum(mask)} 个样本)")
