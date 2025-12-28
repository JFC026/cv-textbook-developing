"""
data_helpers.py

数据处理辅助函数

功能包括:
- 文本清洗
- 数据增强
- 批次生成器
- 序列填充
"""

import numpy as np
import re

def clean_text(text, lowercase=True, remove_special=False):
    """
    清洗文本数据
    
    Args:
        text: 原始文本
        lowercase: 是否转换为小写
        remove_special: 是否移除特殊字符
        
    Returns:
        cleaned_text: 清洗后的文本
    """
    if lowercase:
        text = text.lower()
    
    if remove_special:
        # 只保留字母、数字、空格和基本标点
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', '', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def create_vocabulary(text, min_freq=1):
    """
    创建词汇表（过滤低频字符）
    
    Args:
        text: 文本数据
        min_freq: 最小频率阈值
        
    Returns:
        vocab: 词汇表（字符列表）
        char_to_ix: 字符到索引映射
        ix_to_char: 索引到字符映射
    """
    # 统计字符频率
    char_freq = {}
    for ch in text:
        char_freq[ch] = char_freq.get(ch, 0) + 1
    
    # 过滤低频字符
    vocab = [ch for ch, freq in char_freq.items() if freq >= min_freq]
    vocab = sorted(vocab)
    
    # 创建映射
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    print(f"词汇表大小: {len(vocab)} (过滤前: {len(char_freq)})")
    
    return vocab, char_to_ix, ix_to_char

def batch_generator(inputs, targets, batch_size=32, shuffle=True):
    """
    批次生成器
    
    Args:
        inputs: 输入序列列表
        targets: 目标序列列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Yields:
        (batch_inputs, batch_targets): 批次数据
    """
    n_samples = len(inputs)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_inputs = [inputs[i] for i in batch_indices]
        batch_targets = [targets[i] for i in batch_indices]
        
        yield batch_inputs, batch_targets

def pad_sequences(sequences, max_length=None, padding_value=0):
    """
    填充序列到相同长度
    
    Args:
        sequences: 序列列表
        max_length: 最大长度（None则使用最长序列长度）
        padding_value: 填充值
        
    Returns:
        padded_sequences: 填充后的序列数组
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), max_length), padding_value, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    
    return padded

def split_text_by_sentences(text, max_length=100):
    """
    按句子分割文本
    
    Args:
        text: 原始文本
        max_length: 最大句子长度
        
    Returns:
        sentences: 句子列表
    """
    # 按句号、问号、感叹号分割
    sentences = re.split(r'[.!?]+', text)
    
    # 过滤空句子和过长句子
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) <= max_length]
    
    return sentences

def augment_text(text, char_to_ix, ix_to_char, noise_level=0.05):
    """
    文本数据增强（添加噪声）
    
    Args:
        text: 原始文本
        char_to_ix: 字符到索引映射
        ix_to_char: 索引到字符映射
        noise_level: 噪声比例
        
    Returns:
        augmented_text: 增强后的文本
    """
    chars = list(text)
    n_noise = int(len(chars) * noise_level)
    
    # 随机选择位置添加噪声
    noise_positions = np.random.choice(len(chars), n_noise, replace=False)
    
    for pos in noise_positions:
        # 随机替换为其他字符
        random_char = ix_to_char[np.random.randint(0, len(char_to_ix))]
        chars[pos] = random_char
    
    return ''.join(chars)

def calculate_text_statistics(text):
    """
    计算文本统计信息
    
    Args:
        text: 文本数据
        
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'total_chars': len(text),
        'unique_chars': len(set(text)),
        'avg_word_length': 0,
        'n_words': 0,
        'n_lines': text.count('\n') + 1
    }
    
    # 计算单词统计
    words = text.split()
    if words:
        stats['n_words'] = len(words)
        stats['avg_word_length'] = sum(len(w) for w in words) / len(words)
    
    return stats

def encode_text(text, char_to_ix, unknown_char='<UNK>'):
    """
    将文本编码为索引序列
    
    Args:
        text: 文本
        char_to_ix: 字符到索引映射
        unknown_char: 未知字符标记
        
    Returns:
        encoded: 索引序列
    """
    unk_idx = char_to_ix.get(unknown_char, 0)
    encoded = [char_to_ix.get(ch, unk_idx) for ch in text]
    return encoded

def decode_text(indices, ix_to_char):
    """
    将索引序列解码为文本
    
    Args:
        indices: 索引序列
        ix_to_char: 索引到字符映射
        
    Returns:
        text: 解码后的文本
    """
    text = ''.join(ix_to_char.get(ix, '?') for ix in indices)
    return text

if __name__ == '__main__':
    # 测试数据处理函数
    print("数据处理工具测试")
    
    # 测试文本清洗
    test_text = "  Hello,   World!  This is a TEST.  "
    cleaned = clean_text(test_text, lowercase=True)
    print(f"原始文本: '{test_text}'")
    print(f"清洗后: '{cleaned}'")
    
    # 测试词汇表创建
    sample_text = "hello world hello python"
    vocab, char_to_ix, ix_to_char = create_vocabulary(sample_text)
    print(f"\n词汇表: {vocab}")
    
    # 测试编码解码
    encoded = encode_text("hello", char_to_ix)
    decoded = decode_text(encoded, ix_to_char)
    print(f"\n编码: {encoded}")
    print(f"解码: {decoded}")
    
    # 测试统计
    stats = calculate_text_statistics(sample_text)
    print(f"\n文本统计: {stats}")
    
    print("\n✅ 测试完成")