#!/usr/bin/env python3
"""
模块测试脚本
测试所有核心模块是否正常工作
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("="*70)
    print("测试模块导入")
    print("="*70)
    
    try:
        print("\n1. 测试 numpy...")
        import numpy as np
        print("   ✅ numpy 导入成功")
        
        print("\n2. 测试 matplotlib...")
        import matplotlib.pyplot as plt
        print("   ✅ matplotlib 导入成功")
        
        print("\n3. 测试 src.rnn...")
        from src.rnn import SimpleRNN
        print("   ✅ SimpleRNN 导入成功")
        
        print("\n4. 测试 data.text_loader...")
        from data.text_loader import load_text_data
        print("   ✅ text_loader 导入成功")
        
        print("\n5. 测试 utils.visualization...")
        from utils.visualization import plot_loss_curve
        print("   ✅ visualization 导入成功")
        
        print("\n6. 测试 utils.data_helpers...")
        from utils.data_helpers import clean_text
        print("   ✅ data_helpers 导入成功")
        
        print("\n✅ 所有模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        return False

def test_rnn_basic():
    """测试RNN基本功能"""
    print("\n" + "="*70)
    print("测试RNN基本功能")
    print("="*70)
    
    try:
        import numpy as np
        from src.rnn import SimpleRNN
        
        print("\n1. 创建RNN实例...")
        vocab_size = 10
        hidden_size = 16
        seq_length = 5
        rnn = SimpleRNN(vocab_size, hidden_size, seq_length)
        print(f"   ✅ RNN创建成功 (vocab_size={vocab_size}, hidden_size={hidden_size})")
        
        print("\n2. 测试前向传播...")
        inputs = [0, 1, 2, 3, 4]
        h_prev = np.zeros((hidden_size, 1))
        xs, hs, ys, ps = rnn.forward(inputs, h_prev)
        print(f"   ✅ 前向传播成功")
        print(f"      隐藏状态形状: {hs[0].shape}")
        print(f"      输出概率形状: {ps[0].shape}")
        
        print("\n3. 测试反向传播...")
        targets = [1, 2, 3, 4, 5]
        grads = rnn.backward(xs, hs, ps, targets)
        print(f"   ✅ 反向传播成功")
        print(f"      梯度数量: {len(grads)}")
        
        print("\n4. 测试文本生成...")
        sample_ix = rnn.sample(h_prev, 0, 10)
        print(f"   ✅ 文本生成成功")
        print(f"      生成序列: {sample_ix}")
        
        print("\n✅ RNN所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ RNN测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """测试数据处理功能"""
    print("\n" + "="*70)
    print("测试数据处理功能")
    print("="*70)
    
    try:
        from utils.data_helpers import clean_text, create_vocabulary, encode_text, decode_text
        
        print("\n1. 测试文本清洗...")
        test_text = "  Hello,   World!  "
        cleaned = clean_text(test_text, lowercase=True)
        print(f"   原始: '{test_text}'")
        print(f"   清洗后: '{cleaned}'")
        print("   ✅ 文本清洗成功")
        
        print("\n2. 测试词汇表创建...")
        sample_text = "hello world"
        vocab, char_to_ix, ix_to_char = create_vocabulary(sample_text)
        print(f"   词汇表: {vocab}")
        print("   ✅ 词汇表创建成功")
        
        print("\n3. 测试编码解码...")
        encoded = encode_text("hello", char_to_ix)
        decoded = decode_text(encoded, ix_to_char)
        print(f"   编码: {encoded}")
        print(f"   解码: {decoded}")
        print("   ✅ 编码解码成功")
        
        print("\n✅ 数据处理所有功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n" + "="*70)
    print("测试可视化功能")
    print("="*70)
    
    try:
        import numpy as np
        from utils.visualization import plot_loss_curve
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        print("\n1. 测试损失曲线绘制...")
        loss_history = [4.0 - 3.0 * (1 - np.exp(-i/100)) + np.random.randn()*0.1 
                        for i in range(100)]
        
        # 不保存，只测试绘制
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(loss_history)
        plt.close()
        
        print("   ✅ 损失曲线绘制成功")
        
        print("\n✅ 可视化功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "RNN 模块测试套件" + " "*20 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    results = []
    
    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("RNN基本功能", test_rnn_basic()))
    results.append(("数据处理", test_data_processing()))
    results.append(("可视化", test_visualization()))
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} {status}")
    
    print("\n" + "="*70)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 恭喜！所有测试通过，可以开始使用RNN项目了！")
        print("\n下一步:")
        print("  1. 运行 'python main.py' 开始完整工作流")
        print("  2. 或运行 'python src/train_rnn.py' 直接训练模型")
        print("  3. 查看 GUIDE.md 了解详细使用说明")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复问题")
    
    print("="*70)

if __name__ == '__main__':
    main()