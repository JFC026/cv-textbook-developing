import numpy as np

class CrossEntropyLoss:
    """交叉熵损失函数"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, scores, labels):
        """
        计算交叉熵损失
        scores: 模型输出 (batch_size, num_classes)
        labels: 真实标签 (batch_size,) 或 one-hot编码
        """
        batch_size = scores.shape[0]
        
        # 如果labels是整数标签，转换为one-hot编码
        if labels.ndim == 1:
            num_classes = scores.shape[1]
            labels_one_hot = np.eye(num_classes)[labels]
        else:
            labels_one_hot = labels
        
        # 计算softmax概率（数值稳定版本）
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 计算交叉熵损失: L = -sum(y_true * log(y_pred))
        losses = -np.sum(labels_one_hot * np.log(probs + 1e-8), axis=1)
        loss = np.mean(losses)
        
        self.cache = (probs, labels_one_hot)
        return loss
    
    def backward(self):
        """交叉熵损失的反向传播"""
        probs, labels_one_hot = self.cache
        
        # 简化梯度: dL/dscores = probs - labels
        dscores = probs - labels_one_hot
        dscores /= probs.shape[0]  # 平均梯度
        
        return dscores