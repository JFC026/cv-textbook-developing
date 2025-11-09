import numpy as np

class LinearLayer:
    """
    全连接层 - 使用Xavier初始化
    """
    
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        使用Xavier/Glorot初始化
        """
        if activation == 'relu':
            # He初始化，适合ReLU
            scale = np.sqrt(2.0 / input_dim)
        else:
            # Xavier初始化，适合tanh/sigmoid
            scale = np.sqrt(1.0 / input_dim)
        
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
        self.cache = None
    
    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.cache = x.copy()
        return out
    
    def backward(self, dout):
        x = self.cache
        batch_size = x.shape[0]
        
        dx = np.dot(dout, self.W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        
        dW /= batch_size
        db /= batch_size
        
        return dx, dW, db