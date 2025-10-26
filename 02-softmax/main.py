#!/usr/bin/env python3
"""
Softmax分类器主程序
支持Fashion-MNIST和鸢尾花数据集
"""

import argparse
from data.fashion_mnist_loader import load_fashion_mnist
from data.iris_loader import load_iris_data
from models.softmax_classifier import SoftmaxClassifier
from utils.visualization import visualize_fashion_samples, plot_training_history
from config.fashion_mnist_config import FASHION_MNIST_CONFIG
from config.iris_config import IRIS_CONFIG

def train_fashion_mnist():
    """训练Fashion-MNIST分类器"""
    print("正在加载Fashion-MNIST数据集...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()
    
    print(f"训练集：{X_train.shape}，验证集：{X_val.shape}，测试集：{X_test.shape}")
    print(f"类别数：{y_train.shape[1]}（10类服饰）")
    
    visualize_fashion_samples(X_train, y_train, num=5)
    
    model = SoftmaxClassifier(
        input_dim=FASHION_MNIST_CONFIG['input_dim'], 
        num_classes=FASHION_MNIST_CONFIG['num_classes']
    )
    
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=FASHION_MNIST_CONFIG['epochs'],
        batch_size=FASHION_MNIST_CONFIG['batch_size'],
        lr=FASHION_MNIST_CONFIG['learning_rate'],
        reg=FASHION_MNIST_CONFIG['regularization']
    )
    
    test_acc = model.evaluate(X_test, y_test)
    print(f"\n测试集最终准确率：{test_acc:.4f}")
    
    y_pred = model.predict(X_test[:5])
    visualize_fashion_samples(X_test[:5], y_test[:5], y_pred, num=5)
    
    return model, history

def train_iris():
    """训练鸢尾花分类器"""
    print("正在加载鸢尾花数据集...")
    X_train, y_train, X_val, y_val = load_iris_data()
    
    print(f"训练集：{X_train.shape}，验证集：{X_val.shape}")
    print(f"类别数：{y_train.shape[1]}（3类鸢尾花）")
    
    model = SoftmaxClassifier(
        input_dim=IRIS_CONFIG['input_dim'], 
        num_classes=IRIS_CONFIG['num_classes']
    )
    
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=IRIS_CONFIG['epochs'],
        batch_size=IRIS_CONFIG['batch_size'],
        lr=IRIS_CONFIG['learning_rate'],
        reg=IRIS_CONFIG['regularization']
    )
    
    final_acc = model.evaluate(X_val, y_val)
    print(f"\n最终验证集准确率：{final_acc:.4f}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Softmax分类器训练')
    parser.add_argument('--dataset', type=str, choices=['fashion_mnist', 'iris', 'both'], 
                       default='both', help='选择要训练的数据集')
    
    args = parser.parse_args()
    
    if args.dataset in ['fashion_mnist', 'both']:
        print("=" * 50)
        print("训练Fashion-MNIST分类器")
        print("=" * 50)
        train_fashion_mnist()
    
    if args.dataset in ['iris', 'both']:
        print("=" * 50)
        print("训练鸢尾花分类器")
        print("=" * 50)
        train_iris()

if __name__ == "__main__":
    main()