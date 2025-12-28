"""
rnn.py - 基础RNN模型实现（支持批处理）
"""

import numpy as np
import pickle

class SimpleRNN:
    """
    简单的字符级RNN实现（支持批处理）
    """
    
    def __init__(self, vocab_size, hidden_size=128, seq_length=25, 
                 batch_size=32, learning_rate=0.001):
        """
        初始化RNN参数
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        scale = np.sqrt(2.0 / (vocab_size + hidden_size))
        self.Wxh = np.random.randn(hidden_size, vocab_size) * scale
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        
        self.vWxh = np.zeros_like(self.Wxh)
        self.vWhh = np.zeros_like(self.Whh)
        self.vWhy = np.zeros_like(self.Why)
        self.vbh = np.zeros_like(self.bh)
        self.vby = np.zeros_like(self.by)
    
    def forward(self, inputs_batch, h_prev):
        """
        前向传播计算
        """
        batch_size = inputs_batch.shape[0]
        xs, hs, ys, ps = [], [], [], []
        
        h = h_prev.copy()
        
        for t in range(self.seq_length):
            x_t = inputs_batch[:, t]
            
            x_onehot = np.zeros((self.vocab_size, batch_size))
            for b in range(batch_size):
                x_onehot[x_t[b], b] = 1
            xs.append(x_onehot)
            
            h = np.tanh(np.dot(self.Wxh, x_onehot) + np.dot(self.Whh, h) + self.bh)
            hs.append(h)
            
            y = np.dot(self.Why, h) + self.by
            ys.append(y)
            
            p = self.batch_softmax(y)
            ps.append(p)
        
        return xs, hs, ys, ps, h
    
    def backward(self, xs, hs, ps, targets_batch):
        """
        反向传播计算
        """
        batch_size = targets_batch.shape[0]
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        for t in reversed(range(self.seq_length)):
            target_t = targets_batch[:, t]
            
            dy = ps[t].copy()
            for b in range(batch_size):
                dy[target_t[b], b] -= 1
            
            dWhy += np.dot(dy, hs[t].T) / batch_size
            dby += np.sum(dy, axis=1, keepdims=True) / batch_size
            
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            dbh += np.sum(dh_raw, axis=1, keepdims=True) / batch_size
            dWxh += np.dot(dh_raw, xs[t].T) / batch_size
            
            if t > 0:
                dWhh += np.dot(dh_raw, hs[t-1].T) / batch_size
            
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        for dparam in grads:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def batch_softmax(self, x):
        """
        批处理Softmax函数
        """
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def compute_loss(self, ps, targets_batch):
        """
        计算交叉熵损失
        """
        batch_size = targets_batch.shape[0]
        total_loss = 0
        
        for t in range(self.seq_length):
            for b in range(batch_size):
                target_idx = targets_batch[b, t]
                prob = ps[t][target_idx, b]
                total_loss += -np.log(prob + 1e-8)
        
        return total_loss / (batch_size * self.seq_length)
    
    def softmax(self, x):
        """
        单样本Softmax
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def sample(self, h, seed_ix, n, temperature=1.0):
        """
        生成文本样本
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            y = y / temperature
            p = self.softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        
        return ixes
    
    def train_step(self, inputs_batch, targets_batch, h_prev):
        """
        单步训练
        """
        xs, hs, ys, ps, h_last = self.forward(inputs_batch, h_prev)
        loss = self.compute_loss(ps, targets_batch)
        dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets_batch)
        
        self.t += 1
        
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        m_params = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        v_params = [self.vWxh, self.vWhh, self.vWhy, self.vbh, self.vby]
        
        for i, (param, grad, m, v) in enumerate(zip(params, grads, m_params, v_params)):
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param[:] -= update
        
        return loss, h_last
    
    def save_model(self, filepath):
        """
        保存模型
        """
        model_data = {
            'Wxh': self.Wxh,
            'Whh': self.Whh,
            'Why': self.Why,
            'bh': self.bh,
            'by': self.by,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'mWxh': self.mWxh,
            'mWhh': self.mWhh,
            'mWhy': self.mWhy,
            'mbh': self.mbh,
            'mby': self.mby,
            'vWxh': self.vWxh,
            'vWhh': self.vWhh,
            'vWhy': self.vWhy,
            'vbh': self.vbh,
            'vby': self.vby,
            't': self.t
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        加载模型
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.Wxh = model_data['Wxh']
        self.Whh = model_data['Whh']
        self.Why = model_data['Why']
        self.bh = model_data['bh']
        self.by = model_data['by']
        self.vocab_size = model_data['vocab_size']
        self.hidden_size = model_data['hidden_size']
        self.seq_length = model_data['seq_length']
        self.batch_size = model_data.get('batch_size', 32)
        self.learning_rate = model_data.get('learning_rate', 0.001)
        
        if 'mWxh' in model_data:
            self.mWxh = model_data['mWxh']
            self.mWhh = model_data['mWhh']
            self.mWhy = model_data['mWhy']
            self.mbh = model_data['mbh']
            self.mby = model_data['mby']
            self.vWxh = model_data['vWxh']
            self.vWhh = model_data['vWhh']
            self.vWhy = model_data['vWhy']
            self.vbh = model_data['vbh']
            self.vby = model_data['vby']
            self.t = model_data.get('t', 0)


def create_batches(data, seq_length, batch_size):
    """
    创建训练批次
    """
    n_batches = len(data) // (seq_length * batch_size)
    data = data[:n_batches * seq_length * batch_size]
    data = np.array(data).reshape(batch_size, -1)
    
    for i in range(0, data.shape[1] - seq_length, seq_length):
        inputs = data[:, i:i+seq_length]
        targets = data[:, i+1:i+seq_length+1]
        yield inputs, targets


if __name__ == '__main__':
    vocab_size = 50
    hidden_size = 32
    seq_length = 10
    batch_size = 8
    
    rnn = SimpleRNN(vocab_size, hidden_size, seq_length, batch_size)
    
    np.random.seed(42)
    inputs_batch = np.random.randint(0, vocab_size, (batch_size, seq_length))
    targets_batch = np.random.randint(0, vocab_size, (batch_size, seq_length))
    h_prev = np.zeros((hidden_size, batch_size))
    
    xs, hs, ys, ps, h_last = rnn.forward(inputs_batch, h_prev)
    
    loss, h_last = rnn.train_step(inputs_batch, targets_batch, h_prev)
    
    h_sample = np.zeros((hidden_size, 1))
    sample_ixes = rnn.sample(h_sample, 0, 10)