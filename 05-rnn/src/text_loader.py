"""
text_loader.py

文本数据加载和预处理

功能包括:
- 文本文件读取
- 字符到索引的映射
- 训练数据生成
- 数据集划分
"""

import numpy as np
import os
import urllib.request

def download_sample_text(data_dir='../data', url=None):
    """
    下载示例文本数据
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'input.txt')
    
    if os.path.exists(filepath):
        print(f"文本文件已存在: {filepath}")
        return filepath
    
    print(f"正在下载文本数据...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"下载完成: {filepath}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载文本文件并放置到 data/ 目录")
        raise
    
    return filepath

def load_text_data(filepath):
    """
    加载文本数据
    """
    print(f"正在加载文本数据: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"文本加载完成")
    print(f"文本长度: {len(text)} 字符")
    print(f"词汇表大小: {vocab_size} 个不同字符")
    if len(chars) > 50:
        print(f"字符集: {''.join(chars[:50])}...")
    else:
        print(f"字符集: {''.join(chars)}")
    
    return text, chars, char_to_ix, ix_to_char

def prepare_sequences(text, char_to_ix, seq_length=25):
    """
    准备训练序列
    """
    print(f"准备训练序列（序列长度={seq_length}）...")
    
    inputs = []
    targets = []
    
    for i in range(0, len(text) - seq_length):
        input_seq = [char_to_ix[ch] for ch in text[i:i+seq_length]]
        target_seq = [char_to_ix[ch] for ch in text[i+1:i+seq_length+1]]
        inputs.append(input_seq)
        targets.append(target_seq)
    
    print(f"生成了 {len(inputs)} 个训练序列")
    
    return inputs, targets

def split_data(inputs, targets, train_split=0.9, random_seed=42):
    """
    划分训练集和验证集
    """
    np.random.seed(random_seed)
    
    n_samples = len(inputs)
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_inputs = [inputs[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    val_inputs = [inputs[i] for i in val_indices]
    val_targets = [targets[i] for i in val_indices]
    
    print(f"数据集划分完成")
    print(f"训练集: {len(train_inputs)} 个序列")
    print(f"验证集: {len(val_inputs)} 个序列")
    
    return train_inputs, train_targets, val_inputs, val_targets

def create_batches(inputs, targets, batch_size=32):
    """
    创建批次数据
    """
    n_samples = len(inputs)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        
        batches.append((batch_inputs, batch_targets))
    
    return batches

if __name__ == '__main__':
    print("="*60)
    print("文本数据加载器测试")
    print("="*60)
    
    filepath = download_sample_text()
    text, chars, char_to_ix, ix_to_char = load_text_data(filepath)
    inputs, targets = prepare_sequences(text, char_to_ix, seq_length=25)
    
    train_inputs, train_targets, val_inputs, val_targets = split_data(
        inputs, targets, train_split=0.9
    )
    
    batches = create_batches(train_inputs, train_targets, batch_size=32)
    
    print(f"测试完成")
    print(f"批次数量: {len(batches)}")
    print(f"第一个批次大小: {len(batches[0][0])}")