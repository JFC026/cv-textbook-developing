"""
CIFAR-10 CNN训练脚本
集成BatchNorm、Dropout、Adam优化器
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') 
# =========================================
import matplotlib.pyplot as plt
from pathlib import Path
import time
from cifar10_cnn_model import CIFAR10_CNN, cross_entropy_loss, compute_accuracy
from cifar10_loader import load_cifar10_data, normalize_cifar10, get_cifar10_class_names
from improved_cnn_model import AdamOptimizer

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def train_cifar10_cnn(epochs=50, batch_size=128, learning_rate=0.001, 
                       weight_decay=1e-3, dropout_rate=0.3, data_dir='data'):
    print("=" * 70)
    print("CIFAR-10 CNN训练 (高精度模式)")
    print("集成: BatchNorm + Dropout + Adam + Data Augmentation")
    print("=" * 70)
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    
    # 数据归一化
    print("数据归一化...")
    X_train, X_test, mean, std = normalize_cifar10(X_train, X_test)
    
    # === 【修复2】数据量配置 (冲击 70% 必须用全量数据) ===
    n_train = 50000   # 留 1000 张做验证/丢弃，用 8000 张训练
    n_test = 10000     # 测试集保持 1000
    # =================================================
    
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    print(f"使用 {n_train} 个训练样本, {n_test} 个测试样本")
    
    # 初始化模型
    print(f"\n初始化CIFAR-10 CNN模型...")
    model = CIFAR10_CNN(weight_decay=weight_decay, dropout_rate=dropout_rate)
    
    # 初始化Adam优化器
    optimizer = AdamOptimizer(model.trainable_layers, learning_rate=learning_rate)
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    
    # 训练循环
    num_batches = len(X_train) // batch_size
    print(f"\n开始训练: {epochs} 个epoch, 每个epoch {num_batches} 个batch")
    print("-" * 70)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # === 【修复3】学习率衰减 (确保只在长跑时触发) ===
        # 在进度 40% 和 80% 时衰减
        if epoch == int(epochs * 0.4) or epoch == int(epochs * 0.8):
            optimizer.learning_rate *= 0.1
            print(f"\n[优化] 学习率已降低为: {optimizer.learning_rate}")
        # ============================================

        # 打乱数据
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # 小批次训练
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # === 数据增强 (随机水平翻转) ===
            if np.random.rand() > 0.5:
                X_batch = X_batch[:, :, :, ::-1]
            # ============================

            # 前向传播
            scores = model.forward(X_batch)
            
            # 计算损失
            loss, dscores = cross_entropy_loss(scores, y_batch, 
                                              weight_decay=weight_decay, 
                                              model=model)
            
            # 反向传播
            model.backward(dscores)
            optimizer.update(model.trainable_layers, None)
            
            # 计算准确率
            acc = compute_accuracy(scores, y_batch)
            epoch_loss += loss
            epoch_acc += acc
            
            # 打印进度 (每 50 个 batch 打印一次，避免刷屏)
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], "
                      f"Loss: {loss:.4f}, Acc: {acc*100:.2f}%")
        
        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # 在测试集上评估
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
            model.save_model(models_dir / 'best_cifar10_cnn.pkl')
            print(f" ✓ 保存最佳模型 (测试准确率: {test_acc*100:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} 耗时: {epoch_time:.2f}秒 | "
              f"Train Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        print("-" * 70)
    
    # 保存最终模型
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model.save_model(models_dir / 'cifar10_cnn_final.pkl')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, test_accs)
    
    # 评估每个类别
    evaluate_per_class(model, X_test, y_test)
    
    return model, train_losses, train_accs, test_accs


def plot_training_curves(train_losses, train_accs, test_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('CIFAR-10 Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-', marker='o', 
             label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in test_accs], 'r-', marker='s', 
             label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('CIFAR-10 Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 标注最佳准确率
    best_idx = np.argmax(test_accs)
    best_acc = test_accs[best_idx] * 100
    ax2.plot(best_idx + 1, best_acc, 'r*', markersize=15, 
             label=f'Best: {best_acc:.2f}%')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'cifar10_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到 results/cifar10_training_curves.png")
    plt.close()


def evaluate_per_class(model, X_test, y_test):
    """评估每个类别的性能"""
    print("\n各类别性能评估:")
    print("-" * 70)
    
    class_names = get_cifar10_class_names()
    scores = model.forward_test(X_test)
    predictions = np.argmax(scores, axis=1)
    
    print(f"{'类别':<15} {'样本数':<10} {'正确数':<10} {'准确率':<10}")
    print("-" * 70)
    
    for i in range(10):
        mask = y_test == i
        num_samples = np.sum(mask)
        if num_samples > 0:
            num_correct = np.sum(predictions[mask] == y_test[mask])
            class_acc = num_correct / num_samples
            print(f"{class_names[i]:<15} {num_samples:<10} {num_correct:<10} {class_acc*100:>6.2f}%")
    
    print("-" * 70)


if __name__ == '__main__':
    # 冲击 70%+ 的最终配置
    # 预计耗时：8-10 小时 (建议睡觉前运行)
    model, losses, train_accs, test_accs = train_cifar10_cnn(
        epochs=50,              
        batch_size=128,          # 64 是 CPU 矩阵运算的甜点区
        learning_rate=0.001,    # 初始学习率
        weight_decay=1e-3,      # 较强的正则化，防止在 50 轮中过拟合
        dropout_rate=0.3,       # 保持 Dropout
        data_dir='data'
    )