import numpy as np

class SoftmaxClassifier:
    """Softmax分类器实现，用于多类别分类问题"""
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        初始化Softmax分类器
        
        Args:
            input_dim: 输入特征维度
            num_classes: 类别数量
        """
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(input_dim, num_classes))
        self.b = np.zeros(shape=(1, num_classes))

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        计算softmax概率
        
        Args:
            scores: 原始得分矩阵，形状为(N, C)，N为样本数，C为类别数
            
        Returns:
            probs: softmax概率矩阵，每行和为1
        """
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X: np.ndarray, y: np.ndarray = None, reg: float = 0.0) -> tuple:
        """
        前向传播计算预测概率和损失
        """
        N = X.shape[0]
        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)

        if y is None:
            return probs, 0.0

        cross_entropy = -np.sum(y * np.log(probs + 1e-10)) / N
        reg_loss = 0.5 * reg * np.sum(self.W ** 2)
        total_loss = cross_entropy + reg_loss
        return probs, total_loss

    def backward(self, X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> tuple:
        """
        反向传播计算梯度
        """
        N = X.shape[0]
        probs, _ = self.forward(X)
        dscores = (probs - y) / N
        dW = np.dot(X.T, dscores) + reg * self.W
        db = np.sum(dscores, axis=0, keepdims=True)
        return dW, db

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, 
              y_val: np.ndarray, epochs: int = 100, batch_size: int = 32, 
              lr: float = 0.01, reg: float = 0.001) -> dict:
        """
        训练模型
        """
        N_train = X_train.shape[0]
        num_batches = max(1, N_train // batch_size)
        
        loss_history = []
        acc_history = []

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(N_train)
            X_shuffled = X_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]

            total_loss = 0.0
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, N_train)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                _, batch_loss = self.forward(X_batch, y_batch, reg)
                total_loss += batch_loss

                dW, db = self.backward(X_batch, y_batch, reg)
                self.W -= lr * dW
                self.b -= lr * db

            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_loss = total_loss / num_batches
                val_acc = self.evaluate(X_val, y_val)
                loss_history.append(avg_loss)
                acc_history.append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        return {'loss_history': loss_history, 'acc_history': acc_history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型准确率"""
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)