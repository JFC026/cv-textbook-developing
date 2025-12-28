#!/usr/bin/env python3
"""
模型诊断工具
用于检查RNN模型的训练状态和潜在问题
"""

import numpy as np
import sys
import os
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from src.rnn_original import SimpleRNN
from src.text_loader import load_text_data

def diagnose_model():
    print("="*70)
    print("RNN模型状态检查")
    print("="*70)
    
    # 加载模型
    print("\n[1] 加载模型...")
    try:
        model_paths = [
            'results/rnn_model_original.pkl',
            'results/rnn_model.pkl'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("   错误：未找到模型文件，请先运行训练脚本")
            print("   尝试查找以下路径:")
            for path in model_paths:
                print(f"     - {path}")
            return
        
        rnn = SimpleRNN(vocab_size=1, hidden_size=128)
        rnn.load_model(model_path)
        print(f"   成功加载模型: {model_path}")
        print(f"   学习率: {rnn.learning_rate}")
        print(f"   隐藏层维度: {rnn.hidden_size}")
        print(f"   词汇表大小: {rnn.vocab_size}")
    except Exception as e:
        print(f"   模型加载失败: {e}")
        return
    
    # 检查权重参数
    print("\n[2] 权重参数统计...")
    print(f"   Wxh均值: {rnn.Wxh.mean():.6f}, 标准差: {rnn.Wxh.std():.6f}, 最大绝对值: {np.abs(rnn.Wxh).max():.6f}")
    print(f"   Whh均值: {rnn.Whh.mean():.6f}, 标准差: {rnn.Whh.std():.6f}, 最大绝对值: {np.abs(rnn.Whh).max():.6f}")
    print(f"   Why均值: {rnn.Why.mean():.6f}, 标准差: {rnn.Why.std():.6f}, 最大绝对值: {np.abs(rnn.Why).max():.6f}")
    
    if np.abs(rnn.Wxh).max() > 10:
        print("   注意：Wxh权重值偏大，可能存在梯度爆炸风险")
    if rnn.Wxh.std() < 0.001:
        print("   注意：Wxh权重变化较小，模型可能未充分学习")
    
    # 加载测试数据
    print("\n[3] 加载测试数据...")
    try:
        data_paths = [
            'data/input.txt',
            '../data/input.txt',
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data', 'input.txt')
        ]
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("   错误：未找到数据文件，请确认data/input.txt存在")
            return
        
        text, chars, char_to_ix, ix_to_char = load_text_data(data_path)
        test_text = text[:1000]
        print(f"   数据加载完成: {data_path}")
        print(f"   文本长度: {len(text)} 字符")
        print(f"   词汇表大小: {len(chars)} 字符")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        return
    
    # 测试前向传播
    print("\n[4] 前向传播测试...")
    h = np.zeros((rnn.hidden_size, 1))
    inputs = [char_to_ix[ch] for ch in test_text[:25]]
    targets = [char_to_ix[ch] for ch in test_text[1:26]]
    
    inputs_batch = np.array([inputs])
    targets_batch = np.array([targets])
    h_prev = np.zeros((rnn.hidden_size, 1))
    
    xs, hs, ys, ps, _ = rnn.forward(inputs_batch, h_prev)
    loss = rnn.compute_loss(ps, targets_batch)
    
    print(f"   序列长度: {len(inputs)}")
    print(f"   损失值: {loss:.4f}")
    print(f"   平均字符损失: {loss/len(inputs):.4f}")
    
    random_loss = -np.log(1.0 / rnn.vocab_size)
    print(f"   随机预测损失: {random_loss:.4f}")
    
    if loss > random_loss * len(inputs) * 1.5:
        print("   问题：损失值高于随机预测，模型未有效学习")
    elif loss > random_loss * len(inputs):
        print("   注意：损失值接近随机预测，学习效果有限")
    else:
        print("   正常：损失值低于随机预测，模型在学习")
    
    # 检查概率分布
    print("\n[5] 输出概率分布检查...")
    sample_probs = ps[0].flatten()
    print(f"   最大概率值: {sample_probs.max():.6f}")
    print(f"   最小概率值: {sample_probs.min():.6f}")
    print(f"   概率分布熵: {-np.sum(sample_probs * np.log(sample_probs + 1e-8)):.4f}")
    print(f"   理论最大熵: {np.log(rnn.vocab_size):.4f}")
    
    if sample_probs.max() < 0.1:
        print("   注意：所有输出概率均较低，模型确定性不足")
    
    # 测试梯度
    print("\n[6] 梯度测试...")
    dWxh, dWhh, dWhy, dbh, dby = rnn.backward(xs, hs, ps, targets_batch)
    
    print(f"   dWxh均值: {np.abs(dWxh).mean():.6f}, 最大值: {np.abs(dWxh).max():.6f}")
    print(f"   dWhh均值: {np.abs(dWhh).mean():.6f}, 最大值: {np.abs(dWhh).max():.6f}")
    print(f"   dWhy均值: {np.abs(dWhy).mean():.6f}, 最大值: {np.abs(dWhy).max():.6f}")
    
    if np.abs(dWxh).max() > 5:
        print("   注意：梯度值较大，已进行裁剪处理")
    if np.abs(dWxh).mean() < 1e-6:
        print("   问题：梯度值过小，学习率可能设置偏低")
    
    # 生成文本样本
    print("\n[7] 文本生成测试...")
    sample_ix = rnn.sample(h, 0, 100)
    sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
    print(f"   生成文本样本:")
    print(f"   {sample_text[:100]}")
    
    # 诊断建议
    print("\n" + "="*70)
    print("诊断结果与建议")
    print("="*70)
    
    if loss > random_loss * len(inputs) * 1.5:
        print("\n主要问题：模型未有效学习")
        print("\n可能原因:")
        print("  1. 学习率参数设置不当")
        print("  2. 权重初始化问题")
        print("  3. 梯度计算存在错误")
        print("\n建议措施:")
        print("  1. 重新初始化模型参数")
        print("  2. 检查训练脚本中的学习率设置")
        print("  3. 尝试调整学习率至0.5或1.0")
    elif loss > random_loss * len(inputs):
        print("\n注意：模型学习效果一般")
        print("\n改进建议:")
        print("  1. 增加训练迭代次数至200-500轮")
        print("  2. 检查学习率是否合适")
        print("  3. 考虑减小模型复杂度以加速训练")
    else:
        print("\n状态正常：模型在学习中")
        print("\n优化建议:")
        print("  1. 继续训练以降低损失值")
        print("  2. 调整超参数提升性能")
        print("  3. 可尝试LSTM等改进结构")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    diagnose_model()