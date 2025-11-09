import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from layers.linear import LinearLayer
from layers.activation import ReLU, Softmax
from utils.loss import CrossEntropyLoss

class TwoLayerFullyConnectedNet:
    """改进的两层网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 使用更好的初始化尺度
        # 第一层：使用He初始化
        w1_scale = np.sqrt(2.0 / input_size)
        # 第二层：使用Xavier初始化  
        w2_scale = np.sqrt(1.0 / hidden_size)
        
        self.fc1 = LinearLayer(input_size, hidden_size, w1_scale)
        self.relu = ReLU()
        self.fc2 = LinearLayer(hidden_size, output_size, w2_scale)
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        self.params = {
            'W1': self.fc1.W,
            'b1': self.fc1.b,
            'W2': self.fc2.W,
            'b2': self.fc2.b
        }
    
    def forward(self, x):
        h1 = self.fc1.forward(x)
        a1 = self.relu.forward(h1)
        scores = self.fc2.forward(a1)
        return scores
    
    def backward(self, dscores):
        da1, dW2, db2 = self.fc2.backward(dscores)
        dh1 = self.relu.backward(da1)
        dx, dW1, db1 = self.fc1.backward(dh1)
        
        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }
        return grads
    
    def train_step(self, x_batch, y_batch, optimizer):
        scores = self.forward(x_batch)
        loss = self.loss_fn.forward(scores, y_batch)
        dscores = self.loss_fn.backward()
        grads = self.backward(dscores)
        
        # 梯度裁剪，防止梯度爆炸
        grads = self._clip_gradients(grads, max_norm=5.0)
        
        self.params = optimizer.update(self.params, grads)
        self._update_layer_params()
        
        return loss
    
    def _clip_gradients(self, grads, max_norm=5.0):
        """梯度裁剪"""
        total_norm = 0
        for grad in grads.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for key in grads:
                grads[key] *= clip_coef
                
        return grads
    
    def _update_layer_params(self):
        self.fc1.W = self.params['W1']
        self.fc1.b = self.params['b1']
        self.fc2.W = self.params['W2']
        self.fc2.b = self.params['b2']
    
    def predict(self, x):
        scores = self.forward(x)
        probs = self.softmax.forward(scores)
        return np.argmax(probs, axis=1)
    
    def get_accuracy(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == y)