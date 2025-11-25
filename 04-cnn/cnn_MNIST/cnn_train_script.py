"""
改进版CNN训练脚本
集成BatchNorm、Dropout、Adam优化器
在MNIST数据集上训练
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from improved_cnn_model import ImprovedCNN, cross_entropy_loss, compute_accuracy, AdamOptimizer
from data_loader import load_mnist_data

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def train_cnn(epochs=10, batch_size=64, learning_rate=0.001, weight_decay=1e-4, dropout_rate=0.2, data_dir='data'):
    """
    训练改进版CNN模型
    
    参数:
        epochs: 训练轮数 (建议10-15)
        batch_size: 批次大小 (建议64)
        learning_rate: 学习率 (Adam推荐0.001)
        weight_decay: L2正则化系数
        dropout_rate: Dropout丢弃率
        data_dir: 数据目录
    """
    print("=" * 70)
    print("改进版CNN - MNIST手写数字分类")
    print("集成: BatchNorm + Dropout + Adam优化器")
    print("=" * 70)
    
    # 加载数据
    print("\n加载MNIST数据集...")
    X_train, y_train, X_test, y_test = load_mnist_data(data_dir)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 数据预处理
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    # 使用部分数据加快训练
    n_train = 10000
    n_test = 2000
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    print(f"使用 {n_train} 个训练样本, {n_test} 个测试样本")
    
    # 初始化改进版模型
    print(f"\n初始化改进版CNN模型...")
    print(f"  - 卷积通道: 1->16->32")
    print(f"  - BatchNorm: 启用")
    print(f"  - Dropout率: {dropout_rate}")
    print(f"  - L2正则化: {weight_decay}")
    model = ImprovedCNN(weight_decay=weight_decay, dropout_rate=dropout_rate)
    
    # 初始化Adam优化器
    print(f"  - 优化器: Adam (lr={learning_rate})")
    optimizer = AdamOptimizer(model.trainable_layers, learning_rate=learning_rate)
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 训练循环
    num_batches = len(X_train) // batch_size
    print(f"\n开始训练: {epochs} 个epoch, 每个epoch {num_batches} 个batch")
    print("-" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # 打乱数据
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # 小批次训练
        for batch_idx in range(num_batches):
            # 获取批次数据
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # 前向传播（训练模式）
            scores = model.forward(X_batch)
            
            # 计算损失（含L2正则化）
            loss, dscores = cross_entropy_loss(scores, y_batch, weight_decay=weight_decay, model=model)
            
            # 反向传播（收集梯度）
            model.backward(dscores)
            
            # Adam更新参数
            optimizer.update(model.trainable_layers, None)
            
            # 计算准确率
            acc = compute_accuracy(scores, y_batch)
            
            epoch_loss += loss
            epoch_acc += acc
            
            # 每20个batch打印一次
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Batch [{batch_idx+1}/{num_batches}], "
                      f"Loss: {loss:.4f}, "
                      f"Acc: {acc*100:.2f}%")
        
        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # 在测试集上评估（使用测试模式）
        print(f"\n在测试集上评估...")
        test_scores = model.forward_test(X_test)
        test_acc = compute_accuracy(test_scores, y_test)
        
        # 记录指标
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            model.save_model(models_dir / 'best_improved_cnn.pkl')
            print(f"  ✓ 保存最佳模型 (测试准确率: {test_acc*100:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} 完成 (耗时: {epoch_time:.2f}秒)")
        print(f"训练损失: {avg_loss:.4f}, 训练准确率: {avg_acc*100:.2f}%")
        print(f"测试准确率: {test_acc*100:.2f}%")
        print("-" * 70)
    
    # 保存最终模型
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'improved_cnn_final.pkl'
    model.save_model(model_path)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, test_accs)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳测试准确率: {best_test_acc*100:.2f}%")
    print(f"最终测试准确率: {test_accs[-1]*100:.2f}%")
    print(f"模型已保存到: {model_path}")
    print("=" * 70)
    
    return model, train_losses, train_accs, test_accs


def plot_training_curves(train_losses, train_accs, test_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss (with L2 Regularization)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-', marker='o', label='Training Accuracy')
    ax2.plot(epochs, [acc * 100 for acc in test_accs], 'r-', marker='s', label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 标注最佳测试准确率
    best_idx = np.argmax(test_accs)
    best_acc = test_accs[best_idx] * 100
    ax2.plot(best_idx + 1, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.2f}%')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'improved_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到 results/improved_training_curves.png")
    plt.close()


if __name__ == '__main__':
    # 训练改进版模型
    model, losses, train_accs, test_accs = train_cnn(
        epochs=10,              # 训练轮数
        batch_size=64,          # 批次大小
        learning_rate=0.001,    # Adam学习率
        weight_decay=1e-4,      # L2正则化
        dropout_rate=0.2,       # Dropout率
        data_dir='data'
    )