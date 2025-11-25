"""
改进版卷积神经网络 (ImprovedCNN) - 纯Numpy实现
集成BatchNorm、Dropout、Adam优化器
用于MNIST手写数字分类
"""

import numpy as np
import pickle
from pathlib import Path


# ==================== 基础层（保持不变） ====================

class ConvLayer:
    """卷积层实现"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                       np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros(out_channels)
        
        # 梯度缓存
        self.dweights = None
        self.dbiases = None
        self.cache = None
        
    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        out_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                              (self.padding, self.padding)), mode='constant')
        out = np.zeros((batch_size, self.out_channels, out_H, out_W))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(x_slice * self.weights[k], axis=(1, 2, 3)) + self.biases[k]
        
        self.cache = (x, x_padded)
        return out
    
    def backward(self, dout):
        x, x_padded = self.cache
        batch_size, in_channels, H, W = x.shape
        _, out_channels, out_H, out_W = dout.shape
        
        dx_padded = np.zeros_like(x_padded)
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(out_channels):
                    self.dweights[k] += np.sum(x_slice * dout[:, k, i, j][:, None, None, None], axis=0)
                    self.dbiases[k] += np.sum(dout[:, k, i, j])
                    dx_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * dout[:, k, i, j][:, None, None, None]
        
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx


class MaxPoolLayer:
    """最大池化层"""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, x):
        batch_size, channels, H, W = x.shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_H, out_W))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        
        self.cache = x
        return out
    
    def backward(self, dout):
        x = self.cache
        batch_size, channels, H, W = x.shape
        _, _, out_H, out_W = dout.shape
        dx = np.zeros_like(x)
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                
                for b in range(batch_size):
                    for c in range(channels):
                        slice_2d = x_slice[b, c]
                        max_val = np.max(slice_2d)
                        mask = (slice_2d == max_val)
                        dx[b, c, h_start:h_end, w_start:w_end] += mask * dout[b, c, i, j]
        
        return dx


class ReLU:
    """ReLU激活函数"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        x = self.cache
        dx = dout * (x > 0)
        return dx


class FullyConnectedLayer:
    """全连接层"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # He初始化
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
        
        # 梯度缓存
        self.dweights = None
        self.dbiases = None
        self.cache = None
    
    def forward(self, x):
        self.cache = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, dout):
        x = self.cache
        
        self.dweights = np.dot(x.T, dout)
        self.dbiases = np.sum(dout, axis=0)
        dx = np.dot(dout, self.weights.T)
        
        return dx


# ==================== 新增：BatchNorm层 ====================

class BatchNormLayer:
    """批量归一化层"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # 移动平均（用于测试）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # 梯度缓存
        self.dgamma = None
        self.dbeta = None
        self.cache = None
        
        # 训练/测试模式
        self.training = True
    
    def forward(self, x):
        """
        x: (N, C, H, W) 或 (N, D)
        """
        if len(x.shape) == 4:
            # 卷积层输出: (N, C, H, W)
            N, C, H, W = x.shape
            x_reshape = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
            out_reshape = self._forward_impl(x_reshape)
            out = out_reshape.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            # 全连接层输出: (N, D)
            out = self._forward_impl(x)
        
        return out
    
    def _forward_impl(self, x):
        """实际的BN计算"""
        if self.training:
            # 训练模式：使用批次统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # 归一化
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            out = self.gamma * x_norm + self.beta
            
            # 缓存用于反向传播
            self.cache = (x, x_norm, mean, var)
        else:
            # 测试模式：使用移动平均
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        
        return out
    
    def backward(self, dout):
        """
        dout: (N, C, H, W) 或 (N, D)
        """
        if len(dout.shape) == 4:
            N, C, H, W = dout.shape
            dout_reshape = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            dx_reshape = self._backward_impl(dout_reshape)
            dx = dx_reshape.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            dx = self._backward_impl(dout)
        
        return dx
    
    def _backward_impl(self, dout):
        """实际的BN反向传播"""
        x, x_norm, mean, var = self.cache
        N = x.shape[0]
        
        # 梯度计算
        self.dgamma = np.sum(dout * x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * np.power(var + self.eps, -1.5), axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        
        dx = dx_norm / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        return dx


# ==================== 新增：Dropout层 ====================

class DropoutLayer:
    """Dropout层"""
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training:
            # 训练模式：随机丢弃
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        else:
            # 测试模式：不丢弃
            return x
    
    def backward(self, dout):
        return dout * self.mask / (1 - self.dropout_rate)


# ==================== 新增：Adam优化器 ====================

class AdamOptimizer:
    """Adam优化器"""
    
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # 初始化动量
        self.m = {}
        self.v = {}
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                self.m[f'w_{i}'] = np.zeros_like(layer.weights)
                self.v[f'w_{i}'] = np.zeros_like(layer.weights)
                self.m[f'b_{i}'] = np.zeros_like(layer.biases)
                self.v[f'b_{i}'] = np.zeros_like(layer.biases)
            if hasattr(layer, 'gamma'):
                self.m[f'gamma_{i}'] = np.zeros_like(layer.gamma)
                self.v[f'gamma_{i}'] = np.zeros_like(layer.gamma)
                self.m[f'beta_{i}'] = np.zeros_like(layer.beta)
                self.v[f'beta_{i}'] = np.zeros_like(layer.beta)
    
    def update(self, layers, grads):
        self.t += 1
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'dweights') and layer.dweights is not None:
                # 更新权重
                self.m[f'w_{i}'] = self.beta1 * self.m[f'w_{i}'] + (1 - self.beta1) * layer.dweights
                self.v[f'w_{i}'] = self.beta2 * self.v[f'w_{i}'] + (1 - self.beta2) * (layer.dweights ** 2)
                
                m_hat = self.m[f'w_{i}'] / (1 - self.beta1 ** self.t)
                v_hat = self.v[f'w_{i}'] / (1 - self.beta2 ** self.t)
                
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
                
                # 更新偏置
                self.m[f'b_{i}'] = self.beta1 * self.m[f'b_{i}'] + (1 - self.beta1) * layer.dbiases
                self.v[f'b_{i}'] = self.beta2 * self.v[f'b_{i}'] + (1 - self.beta2) * (layer.dbiases ** 2)
                
                m_hat = self.m[f'b_{i}'] / (1 - self.beta1 ** self.t)
                v_hat = self.v[f'b_{i}'] / (1 - self.beta2 ** self.t)
                
                layer.biases -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            
            if hasattr(layer, 'dgamma') and layer.dgamma is not None:
                # 更新BatchNorm参数
                self.m[f'gamma_{i}'] = self.beta1 * self.m[f'gamma_{i}'] + (1 - self.beta1) * layer.dgamma
                self.v[f'gamma_{i}'] = self.beta2 * self.v[f'gamma_{i}'] + (1 - self.beta2) * (layer.dgamma ** 2)
                
                m_hat = self.m[f'gamma_{i}'] / (1 - self.beta1 ** self.t)
                v_hat = self.v[f'gamma_{i}'] / (1 - self.beta2 ** self.t)
                
                layer.gamma -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
                
                self.m[f'beta_{i}'] = self.beta1 * self.m[f'beta_{i}'] + (1 - self.beta1) * layer.dbeta
                self.v[f'beta_{i}'] = self.beta2 * self.v[f'beta_{i}'] + (1 - self.beta2) * (layer.dbeta ** 2)
                
                m_hat = self.m[f'beta_{i}'] / (1 - self.beta1 ** self.t)
                v_hat = self.v[f'beta_{i}'] / (1 - self.beta2 ** self.t)
                
                layer.beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


# ==================== CNN模型 ====================

class ImprovedCNN:
    """
    CNN模型
    集成BatchNorm、Dropout、Adam优化器
    
    网络结构:
    Conv(1->16) -> BN -> ReLU -> MaxPool -> 
    Conv(16->32) -> BN -> ReLU -> MaxPool -> 
    FC(1568->256) -> BN -> ReLU -> Dropout -> 
    FC(256->10)
    """
    
    def __init__(self, weight_decay=1e-4, dropout_rate=0.2):
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        
        # 第一个卷积块
        self.conv1 = ConvLayer(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNormLayer(16)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNormLayer(32)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        # 全连接层
        self.fc1 = FullyConnectedLayer(32 * 7 * 7, 256)
        self.bn3 = BatchNormLayer(256)
        self.relu3 = ReLU()
        self.dropout = DropoutLayer(dropout_rate)
        self.fc2 = FullyConnectedLayer(256, 10)
        
        # 可训练层（用于优化器）
        self.trainable_layers = [
            self.conv1, self.bn1, 
            self.conv2, self.bn2,
            self.fc1, self.bn3, 
            self.fc2
        ]
    
    def set_training_mode(self, training=True):
        """设置训练/测试模式"""
        for layer in [self.bn1, self.bn2, self.bn3, self.dropout]:
            layer.training = training
    
    def forward(self, x):
        """前向传播（训练模式）"""
        self.set_training_mode(True)
        
        # 第一个卷积块
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        # 第二个卷积块
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        
        # 展平
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        # 全连接层
        out = self.fc1.forward(out)
        out = self.bn3.forward(out)
        out = self.relu3.forward(out)
        out = self.dropout.forward(out)
        scores = self.fc2.forward(out)
        
        return scores
    
    def forward_test(self, x):
        """前向传播（测试模式）"""
        self.set_training_mode(False)
        return self.forward(x)
    
    def backward(self, dscores):
        """反向传播"""
        # 全连接层反向
        dout = self.fc2.backward(dscores)
        dout = self.dropout.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.fc1.backward(dout)
        
        # 重塑为卷积层输出形状
        dout = dout.reshape(-1, 32, 7, 7)
        
        # 第二个卷积块反向
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        
        # 第一个卷积块反向
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        
        return None  # 梯度已存储在各层中
    
    def save_model(self, filepath):
        """保存模型"""
        model_params = {
            'conv1_weights': self.conv1.weights,
            'conv1_biases': self.conv1.biases,
            'bn1_gamma': self.bn1.gamma,
            'bn1_beta': self.bn1.beta,
            'bn1_running_mean': self.bn1.running_mean,
            'bn1_running_var': self.bn1.running_var,
            
            'conv2_weights': self.conv2.weights,
            'conv2_biases': self.conv2.biases,
            'bn2_gamma': self.bn2.gamma,
            'bn2_beta': self.bn2.beta,
            'bn2_running_mean': self.bn2.running_mean,
            'bn2_running_var': self.bn2.running_var,
            
            'fc1_weights': self.fc1.weights,
            'fc1_biases': self.fc1.biases,
            'bn3_gamma': self.bn3.gamma,
            'bn3_beta': self.bn3.beta,
            'bn3_running_mean': self.bn3.running_mean,
            'bn3_running_var': self.bn3.running_var,
            
            'fc2_weights': self.fc2.weights,
            'fc2_biases': self.fc2.biases,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"改进版模型已保存到 {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_params = pickle.load(f)
        
        self.conv1.weights = model_params['conv1_weights']
        self.conv1.biases = model_params['conv1_biases']
        self.bn1.gamma = model_params['bn1_gamma']
        self.bn1.beta = model_params['bn1_beta']
        self.bn1.running_mean = model_params['bn1_running_mean']
        self.bn1.running_var = model_params['bn1_running_var']
        
        self.conv2.weights = model_params['conv2_weights']
        self.conv2.biases = model_params['conv2_biases']
        self.bn2.gamma = model_params['bn2_gamma']
        self.bn2.beta = model_params['bn2_beta']
        self.bn2.running_mean = model_params['bn2_running_mean']
        self.bn2.running_var = model_params['bn2_running_var']
        
        self.fc1.weights = model_params['fc1_weights']
        self.fc1.biases = model_params['fc1_biases']
        self.bn3.gamma = model_params['bn3_gamma']
        self.bn3.beta = model_params['bn3_beta']
        self.bn3.running_mean = model_params['bn3_running_mean']
        self.bn3.running_var = model_params['bn3_running_var']
        
        self.fc2.weights = model_params['fc2_weights']
        self.fc2.biases = model_params['fc2_biases']
        
        print(f"改进版模型已从 {filepath} 加载")


# ==================== 损失函数和评估 ====================

def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(scores, labels, weight_decay=0, model=None):
    """
    交叉熵损失 + L2正则化
    """
    batch_size = scores.shape[0]
    probs = softmax(scores)
    
    # 交叉熵损失
    correct_log_probs = -np.log(probs[range(batch_size), labels] + 1e-8)
    data_loss = np.sum(correct_log_probs) / batch_size
    
    # L2正则化
    reg_loss = 0
    if weight_decay > 0 and model is not None:
        for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
            if hasattr(layer, 'weights'):
                reg_loss += 0.5 * weight_decay * np.sum(layer.weights ** 2)
    
    loss = data_loss + reg_loss
    
    # 梯度
    dscores = probs.copy()
    dscores[range(batch_size), labels] -= 1
    dscores /= batch_size
    
    return loss, dscores


def compute_accuracy(scores, labels):
    """计算准确率"""
    predictions = np.argmax(scores, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy