import numpy as np

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
            
            print(f"   处理批次 {i+1}/{num_batches} ({end_idx-start_idx} 个样本)...")
            
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
    def save_model(self, filepath):
        """保存模型到文件"""
        model_data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'best_k': self.best_k
        }
        np.savez(filepath, **model_data)

    @classmethod
    def load_model(cls, filepath):
        """从文件加载模型"""
        model_data = np.load(filepath)
        classifier = cls()
        classifier.X_train = model_data['X_train']
        classifier.y_train = model_data['y_train']
        classifier.best_k = model_data['best_k']
        return classifier