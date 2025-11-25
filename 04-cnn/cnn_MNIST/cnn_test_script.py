"""
改进版CNN模型测试脚本
加载训练好的模型并在测试集上评估
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from improved_cnn_model import ImprovedCNN, compute_accuracy
from data_loader import load_mnist_data

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_predictions(model, X_test, y_test, num_samples=25):
    """
    可视化模型预测结果
    """
    # 随机选择样本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # 预测（测试模式）
    scores = model.forward_test(X_samples)
    predictions = np.argmax(scores, axis=1)
    
    # 可视化
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        # 显示图像
        ax.imshow(X_samples[i, 0], cmap='gray')
        
        # 设置标题
        true_label = y_samples[i]
        pred_label = predictions[i]
        color = 'green' if true_label == pred_label else 'red'
        
        # 获取预测概率
        prob = np.exp(scores[i] - np.max(scores[i]))
        prob = prob / np.sum(prob)
        confidence = prob[pred_label] * 100
        
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'improved_predictions.png', dpi=150, bbox_inches='tight')
    print(f"预测结果可视化已保存到 results/improved_predictions.png")
    plt.close()


def compute_confusion_matrix(predictions, labels, num_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(labels, predictions):
        cm[true_label, pred_label] += 1
    return cm


def plot_confusion_matrix(cm, class_names=None):
    """绘制混淆矩阵"""
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix - Improved CNN')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", color=color, fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'improved_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到 results/improved_confusion_matrix.png")
    plt.close()


def analyze_errors(model, X_test, y_test, num_samples=20):
    """分析错误分类的样本"""
    # 预测
    scores = model.forward_test(X_test)
    predictions = np.argmax(scores, axis=1)
    
    # 找出错误分类的样本
    errors_mask = predictions != y_test
    error_indices = np.where(errors_mask)[0]
    
    if len(error_indices) == 0:
        print("没有错误分类的样本！")
        return
    
    print(f"\n错误分类样本数: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
    
    # 随机选择错误样本
    num_samples = min(num_samples, len(error_indices))
    selected_errors = np.random.choice(error_indices, num_samples, replace=False)
    
    # 可视化错误样本
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < len(selected_errors):
            idx = selected_errors[i]
            
            # 显示图像
            ax.imshow(X_test[idx, 0], cmap='gray')
            
            # 获取预测概率
            prob = np.exp(scores[idx] - np.max(scores[idx]))
            prob = prob / np.sum(prob)
            
            true_label = y_test[idx]
            pred_label = predictions[idx]
            confidence = prob[pred_label] * 100
            
            ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%', 
                        color='red', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Error Analysis - Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'improved_error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"错误分析已保存到 results/improved_error_analysis.png")
    plt.close()


def evaluate_model(model_path='models/best_improved_cnn.pkl', data_dir='data'):
    """
    评估训练好的改进版模型
    """
    print("=" * 70)
    print("改进版CNN模型评估")
    print("=" * 70)
    
    # 加载数据
    print("\n加载MNIST测试数据...")
    _, _, X_test, y_test = load_mnist_data(data_dir)
    
    # 数据预处理
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    # 使用部分测试数据
    n_test = 2000
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    print(f"使用 {n_test} 个测试样本")
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = ImprovedCNN()
    
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先运行训练脚本: python train.py")
        return
    
    model.load_model(model_path)
    
    # 在测试集上评估（使用测试模式）
    print("\n在测试集上进行预测...")
    scores = model.forward_test(X_test)
    predictions = np.argmax(scores, axis=1)
    
    # 计算总体准确率
    test_acc = compute_accuracy(scores, y_test)
    print(f"\n总体测试准确率: {test_acc*100:.2f}%")
    
    # 计算每个类别的准确率
    print("\n各类别性能:")
    print("-" * 70)
    print(f"{'数字':<8} {'样本数':<10} {'正确数':<10} {'准确率':<10}")
    print("-" * 70)
    
    for i in range(10):
        mask = y_test == i
        num_samples = np.sum(mask)
        if num_samples > 0:
            num_correct = np.sum(predictions[mask] == y_test[mask])
            class_acc = num_correct / num_samples
            print(f"{i:<8} {num_samples:<10} {num_correct:<10} {class_acc*100:>6.2f}%")
    
    print("-" * 70)
    
    # 可视化预测结果
    print("\n生成预测结果可视化...")
    visualize_predictions(model, X_test, y_test, num_samples=25)
    
    # 计算并绘制混淆矩阵
    print("\n计算混淆矩阵...")
    cm = compute_confusion_matrix(predictions, y_test, num_classes=10)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
    
    # 错误分析
    print("\n进行错误分析...")
    analyze_errors(model, X_test, y_test, num_samples=20)
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)


if __name__ == '__main__':
    evaluate_model()