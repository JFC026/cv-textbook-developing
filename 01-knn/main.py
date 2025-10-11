"""
K-NN 核心算法实现
"""
import numpy as np
from sklearn.model_selection import KFold


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

    def predict(self, X, k=1):
        """
        使用最高效的矩阵化方法对测试数据 X 进行预测。
        """
        X_test = X.astype(np.float64)
        
        # 1. 计算距离矩阵 (M x N)
        dists = self.compute_distances_no_loops(X_test)

        # 2. 根据距离进行 K-NN 投票预测
        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        【核心实现】使用纯 Numpy 矩阵运算，实现欧氏距离 (L2) 计算。
        避免任何 Python 循环。
        """
        # --- 欧氏距离平方的矩阵化公式: ||x - y||^2 = ||x||^2 - 2*x*y.T + ||y||^2 ---
        
        # 1. 计算测试集的平方和 (M x 1 矩阵)
        X_test_sq = np.sum(X**2, axis=1)[:, np.newaxis] 

        # 2. 计算训练集的平方和 (1 x N 矩阵)
        X_train_sq = np.sum(self.X_train**2, axis=1)[np.newaxis, :]

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

    def cross_validate(self, X_train, y_train, k_values, n_folds=5, random_state=42):
        """
        使用交叉验证选择最佳的K值。
        """
        print(f"-> 开始 {n_folds} 折交叉验证，测试K值: {k_values}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        k_scores = {k: [] for k in k_values}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
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
