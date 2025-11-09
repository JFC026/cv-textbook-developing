import numpy as np

class SGD:
    """随机梯度下降优化器"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        参数更新: W = W - learning_rate * dW
        params: 参数字典 {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        grads: 梯度字典 {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        """
        updated_params = {}
        
        for key in params:
            updated_params[key] = params[key] - self.learning_rate * grads[key]
        
        return updated_params
    

class AdamOptimizer:
    """Adam优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
    
    def update(self, params, grads):
        self.t += 1
        
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        updated_params = {}
        
        for key in params:
            # 更新一阶矩和二阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            updated_params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return updated_params
