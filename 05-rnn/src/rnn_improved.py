"""
rnn_improved.py - RNN改进版本
"""

import numpy as np
import pickle
import math

class ImprovedRNN:
    """
    改进版RNN实现
    """
    
    def __init__(self, vocab_size, hidden_size=128, seq_length=25, 
                 batch_size=32, learning_rate=0.001, dropout_rate=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
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
    
    def forward(self, inputs_batch, h_prev, training=True):
        batch_size = inputs_batch.shape[0]
        xs, hs, ys, ps = [], [], [], []
        
        h = h_prev.copy()
        
        for t in range(self.seq_length):
            x_t = inputs_batch[:, t]
            
            x_onehot = np.zeros((self.vocab_size, batch_size))
            for b in range(batch_size):
                x_onehot[x_t[b], b] = 1
            xs.append(x_onehot)
            
            h_pre_activation = np.dot(self.Wxh, x_onehot) + np.dot(self.Whh, h) + self.bh
            h = np.tanh(h_pre_activation)
            
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*h.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                h = h * mask
            
            hs.append(h)
            
            y = np.dot(self.Why, h) + self.by
            ys.append(y)
            
            p = self.batch_softmax(y)
            ps.append(p)
        
        return xs, hs, ys, ps, h
    
    def backward(self, xs, hs, ps, targets_batch):
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
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def compute_loss(self, ps, targets_batch):
        batch_size = targets_batch.shape[0]
        total_loss = 0
        
        for t in range(self.seq_length):
            for b in range(batch_size):
                target_idx = targets_batch[b, t]
                prob = ps[t][target_idx, b]
                total_loss += -np.log(prob + 1e-8)
        
        return total_loss / (batch_size * self.seq_length)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def sample(self, h, seed_ix, n, temperature=1.0, top_k=None):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            y = y / temperature
            
            if top_k is not None and top_k > 0:
                p = self._top_k_sampling(y, top_k)
            else:
                p = self.softmax(y)
            
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        
        return ixes
    
    def _top_k_sampling(self, logits, k):
        probs = self.softmax(logits).ravel()
        top_k_indices = np.argsort(probs)[-k:]
        new_probs = np.zeros_like(probs)
        new_probs[top_k_indices] = probs[top_k_indices]
        new_probs = new_probs / np.sum(new_probs)
        return new_probs.reshape(-1, 1)
    
    def update_learning_rate(self, epoch, total_epochs, warmup_epochs=5):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        if epoch < warmup_epochs:
            self.learning_rate = self.initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            self.learning_rate = self.initial_lr * cosine_decay
    
    def train_step(self, inputs_batch, targets_batch, h_prev):
        xs, hs, ys, ps, h_last = self.forward(inputs_batch, h_prev, training=True)
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
            'initial_lr': self.initial_lr,
            'dropout_rate': self.dropout_rate,
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
        self.initial_lr = model_data.get('initial_lr', self.learning_rate)
        self.dropout_rate = model_data.get('dropout_rate', 0.1)
        
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