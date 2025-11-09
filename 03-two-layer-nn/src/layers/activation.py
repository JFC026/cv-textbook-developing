import numpy as np

class ReLU:
    """ReLU激活函数: max(0, x)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """前向传播: y = max(0, x)"""
        out = np.maximum(0, x)
        self.cache = x.copy()
        return out
    
    def backward(self, dout):
        """反向传播"""
        x = self.cache
        # ReLU的导数: 1 if x > 0 else 0
        dx = dout * (x > 0)
        return dx

class Softmax:
    """Softmax激活函数"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """
        Softmax前向传播
        softmax(x_i) = exp(x_i) / sum(exp(x))
        """
        # 数值稳定性: 减去最大值防止指数爆炸
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        self.cache = probs.copy()
        return probs
    
    def backward(self, dout):
        """
        Softmax反向传播
        通常与交叉熵损失结合，有简化形式
        这里实现完整的雅可比矩阵计算
        """
        probs = self.cache
        batch_size, num_classes = probs.shape
        
        # 计算Softmax的雅可比矩阵
        dx = np.zeros_like(probs)
        
        for i in range(batch_size):
            p = probs[i]
            # 创建雅可比矩阵: J_ij = p_i * (δ_ij - p_j)
            jacobian = np.diag(p) - np.outer(p, p)
            dx[i] = np.dot(dout[i], jacobian)
        
        return dx
    
