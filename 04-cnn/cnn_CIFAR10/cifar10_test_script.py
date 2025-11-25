"""
CIFAR-10 模型测试与可视化脚本
适配: 并行训练生成的模型 (cifar10_cnn_parallel.pkl)
功能: 生成混淆矩阵、预测结果图、错误样本分析
"""

import numpy as np
# === 关键修复: 强制后台绘图，防止 Qt 报错 ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# =========================================
from pathlib import Path
from cifar10_cnn_model import CIFAR10_CNN, compute_accuracy
from cifar10_loader import load_cifar10_data, normalize_cifar10, get_cifar10_class_names

# 配置中文字体 (如果有的话，没有则回退到英文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def visualize_predictions(X_test, y_test, scores, class_names, num_samples=25):
    """可视化预测结果"""
    # 随机选择样本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    scores_samples = scores[indices]
    
    predictions = np.argmax(scores_samples, axis=1)
    
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    
    for i, ax in enumerate(axes.flat):
        # 还原图像: (C, H, W) -> (H, W, C)
        img = X_samples[i].transpose(1, 2, 0)
        # 反归一化到 [0, 1] 用于显示
        img = (img - img.min()) / (img.max() - img.min())
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        true_label = class_names[y_samples[i]]
        pred_label = class_names[predictions[i]]
        color = 'green' if y_samples[i] == predictions[i] else 'red'
        
        # 计算置信度
        prob = np.exp(scores_samples[i] - np.max(scores_samples[i]))
        prob = prob / np.sum(prob)
        confidence = prob[predictions[i]] * 100
        
        ax.set_title(f'T: {true_label}\nP: {pred_label}\n{confidence:.1f}%', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = Path('results') / 'cifar10_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测结果图已保存到: {save_path}")
    plt.close()

def plot_confusion_matrix(cm, class_names):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='CIFAR-10 Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", color=color, fontsize=9)
    
    plt.tight_layout()
    save_path = Path('results') / 'cifar10_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵图已保存到: {save_path}")
    plt.close()

def analyze_errors(X_test, y_test, predictions, scores, class_names, num_samples=20):
    """可视化错误样本"""
    errors_mask = predictions != y_test
    error_indices = np.where(errors_mask)[0]
    
    if len(error_indices) == 0:
        print("没有错误分类的样本！")
        return
    
    print(f"\n错误分类样本数: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
    
    num_samples = min(num_samples, len(error_indices))
    selected_errors = np.random.choice(error_indices, num_samples, replace=False)
    
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 11))
    
    for i, ax in enumerate(axes.flat):
        if i < len(selected_errors):
            idx = selected_errors[i]
            img = X_test[idx].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            
            prob = np.exp(scores[idx] - np.max(scores[idx]))
            prob = prob / np.sum(prob)
            
            true_label = class_names[y_test[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = prob[predictions[idx]] * 100
            
            ax.set_title(f'T: {true_label}\nP: {pred_label}\n{confidence:.1f}%', 
                        color='red', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('CIFAR-10 Error Analysis (Misclassified)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = Path('results') / 'cifar10_error_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"错误分析图已保存到: {save_path}")
    plt.close()

def analyze_difficult_pairs(cm, class_names):
    """打印最容易混淆的类别对"""
    print("\n[分析] 最容易混淆的类别对 (Top 10):")
    print("-" * 50)
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j:
                confusions.append((cm[i, j], i, j))
    
    confusions.sort(reverse=True)
    print(f"{'真实类别':<15} {'预测类别':<15} {'混淆次数':<10}")
    print("-" * 50)
    for count, true_idx, pred_idx in confusions[:10]:
        print(f"{class_names[true_idx]:<15} {class_names[pred_idx]:<15} {count:<10}")
    print("-" * 50)

def evaluate_cifar10_model(data_dir='data'):
    """主评估函数"""
    print("=" * 70)
    print("CIFAR-10 模型评估与可视化")
    print("=" * 70)
    
    # 1. 自动查找模型文件
    models_dir = Path('models')
    # 优先找并行训练的模型，如果找不到则找普通模型
    if (models_dir / 'cifar10_cnn_parallel.pkl').exists():
        model_path = models_dir / 'cifar10_cnn_parallel.pkl'
    elif (models_dir / 'best_cifar10_cnn.pkl').exists():
        model_path = models_dir / 'best_cifar10_cnn.pkl'
    else:
        print("错误: 未找到模型文件！请等待训练完成。")
        return

    print(f"\n加载模型: {model_path}")
    
    # 2. 加载数据
    print("加载测试数据...")
    _, _, X_test, y_test = load_cifar10_data(data_dir, download=False)
    
    # 归一化 (注意：这里为了简单，使用测试集自身的统计量，更严谨应该用训练集统计量)
    # 但在可视化分析中，这通常足够了
    X_test = X_test.astype(np.float32)
    mean = np.mean(X_test, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_test, axis=(0, 2, 3), keepdims=True)
    X_test_norm = (X_test - mean) / (std + 1e-7)
    
    # 使用全部测试集 (或者你可以切片 [:1000] 来加快速度)
    # 既然只是前向传播，10000张在CPU上也很快
    X_test_final = X_test_norm
    y_test_final = y_test
    print(f"评估样本数: {len(X_test_final)}")
    
    # 3. 初始化并加载模型
    model = CIFAR10_CNN()
    model.load_model(model_path)
    
    # 4. 预测 (只做一次前向传播)
    print("正在进行预测...")
    scores = model.forward_test(X_test_final)
    predictions = np.argmax(scores, axis=1)
    
    # 5. 计算准确率
    test_acc = compute_accuracy(scores, y_test_final)
    print(f"\n>>> 最终测试准确率: {test_acc*100:.2f}% <<<")
    
    # 6. 生成可视化图表
    class_names = get_cifar10_class_names()
    Path('results').mkdir(exist_ok=True)
    
    print("\n正在生成可视化图表...")
    visualize_predictions(X_test_final, y_test_final, scores, class_names)
    
    cm = np.zeros((10, 10), dtype=np.int32)
    for t, p in zip(y_test_final, predictions):
        cm[t, p] += 1
        
    plot_confusion_matrix(cm, class_names)
    analyze_difficult_pairs(cm, class_names)
    analyze_errors(X_test_final, y_test_final, predictions, scores, class_names)
    
    print("\n所有结果已保存到 results/ 文件夹！")

if __name__ == '__main__':
    evaluate_cifar10_model()