import matplotlib.pyplot as plt
import numpy as np

def visualize_fashion_samples(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray = None, num: int = 5) -> None:
    """
    可视化Fashion-MNIST样本图像及其标签
    
    Args:
        X: 图像数据
        y_true: 真实标签
        y_pred: 预测标签（可选）
        num: 显示样本数量
    """
    plt.figure(figsize=(10, 4))
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        true_label = class_names[np.argmax(y_true[i])]
        if y_pred is not None:
            pred_label = class_names[y_pred[i]]
            plt.title(f"True: {true_label}\nPred: {pred_label}")
        else:
            plt.title(f"True: {true_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(loss_history: list, acc_history: list) -> None:
    """
    绘制训练过程中的损失和准确率曲线
    
    Args:
        loss_history: 损失历史记录
        acc_history: 准确率历史记录
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_history)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(acc_history)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()