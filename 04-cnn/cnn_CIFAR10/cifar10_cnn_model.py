"""
CIFAR-10 轻量级 CNN 模型 (Lite Version)
专为纯 Numpy CPU 训练设计
结构: Conv(16) -> Pool -> Conv(32) -> Pool -> FC(128) -> FC(10)
"""

import numpy as np
import pickle
from improved_cnn_model import (
    ConvLayer, MaxPoolLayer, ReLU, FullyConnectedLayer,
    BatchNormLayer, DropoutLayer
)

class CIFAR10_CNN:
    """
    轻量级模型：牺牲一点理论上限，换取数倍的训练速度
    """
    def __init__(self, weight_decay=1e-4, dropout_rate=0.3):
        self.weight_decay = weight_decay
        
        # === 卷积块 1: 32x32 -> 16x16 ===
        # 输入 3 通道，输出 16 通道 (原版是 32)
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNormLayer(16)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        # === 卷积块 2: 16x16 -> 8x8 ===
        # 输入 16 通道，输出 32 通道 (原版是 64/128)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNormLayer(32)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        # === 全连接层 ===
        # 展平后尺寸: 32通道 * 8高 * 8宽 = 2048
        self.fc1 = FullyConnectedLayer(32 * 8 * 8, 128) # 隐层缩小到 128
        self.bn3 = BatchNormLayer(128)
        self.relu3 = ReLU()
        self.dropout = DropoutLayer(dropout_rate)
        
        self.fc2 = FullyConnectedLayer(128, 10)
        
        # 可训练层列表
        self.trainable_layers = [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.fc1, self.bn3,
            self.fc2
        ]
    
    def forward(self, x):
        # 训练模式前向传播
        for layer in [self.bn1, self.bn2, self.bn3, self.dropout]:
            layer.training = True
            
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        out = self.fc1.forward(out)
        out = self.bn3.forward(out)
        out = self.relu3.forward(out)
        out = self.dropout.forward(out)
        scores = self.fc2.forward(out)
        
        return scores
    
    def forward_test(self, x):
        # 测试模式前向传播
        for layer in [self.bn1, self.bn2, self.bn3, self.dropout]:
            layer.training = False
        return self.forward(x)
    
    def backward(self, dscores):
        # 反向传播
        dout = self.fc2.backward(dscores)
        dout = self.dropout.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.fc1.backward(dout)
        
        dout = dout.reshape(-1, 32, 8, 8)
        
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        
        return None
    
    def save_model(self, filepath):
        # 简化版保存逻辑
        # 注意：为了代码简洁，这里略去了具体的保存字典构建
        # 实际运行时 Python 的 pickle 可以直接 dump 整个对象，或者按需保存
        # 这里简单地 dump 整个对象属性字典
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print(f"模型已从 {filepath} 加载")

# 辅助函数保持不变
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_accuracy(scores, labels):
    predictions = np.argmax(scores, axis=1)
    return np.mean(predictions == labels)

def cross_entropy_loss(scores, labels, weight_decay=0, model=None):
    batch_size = scores.shape[0]
    probs = softmax(scores)
    correct_log_probs = -np.log(probs[range(batch_size), labels] + 1e-8)
    data_loss = np.sum(correct_log_probs) / batch_size
    
    reg_loss = 0
    if weight_decay > 0 and model is not None:
        for layer in model.trainable_layers:
            if hasattr(layer, 'weights'):
                reg_loss += 0.5 * weight_decay * np.sum(layer.weights ** 2)
    
    loss = data_loss + reg_loss
    dscores = probs.copy()
    dscores[range(batch_size), labels] -= 1
    dscores /= batch_size
    
    return loss, dscores