import numpy as np
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knn import KNearestNeighbor
from utils.data_helpers import load_data, preprocess_data
from utils.metrics import print_evaluation_metrics
from utils.visualization import plot_confusion_matrix, plot_class_accuracy

def load_trained_model():
    """
    加载之前训练好的模型和配置
    """
    print("-> Loading previously trained model...")
    
    # 加载交叉验证结果获取最佳K值
    try:
        with open('../results/cross_validation_results.json', 'r') as f:
            cv_results = json.load(f)
        best_k = cv_results['best_k']
        print(f"-> Loaded best K value: {best_k}")
    except FileNotFoundError:
        print("!!! Cross-validation results not found. Please run train_knn.py first.")
        return None, None, None, None
    
    # 加载数据
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(data_dir='../data')
    X_train, y_train, X_test, y_test = preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    # 创建并训练模型（使用完整训练集）
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    classifier.best_k = best_k
    
    return classifier, X_test, y_test, best_k

def evaluate_model(classifier, X_test, y_test, best_k, test_subset_size=10000):
    """
    Evaluate model performance
    """
    print("\n" + "=" * 60)
    print("       K-NN Classifier Evaluation Script")
    print("=" * 60)
    
    print(f"-> Evaluating on test set (K={best_k}, {test_subset_size} samples)...")
    print("   Note: Using batch processing to avoid memory overflow...")
    
    # 使用分批处理
    X_test_subset = X_test[:test_subset_size]
    y_test_subset = y_test[:test_subset_size]
    
    y_pred = classifier.predict(X_test_subset, k=best_k, batch_size=1000)
    
    # ==================================================
    # 评估结果
    # ==================================================
    print_evaluation_metrics(y_test_subset, y_pred)
    
    # 可视化结果
    plot_confusion_matrix(y_test_subset, y_pred, save_path='../results/confusion_matrix.png')
    plot_class_accuracy(y_test_subset, y_pred, save_path='../results/class_accuracy.png')
    
    # 保存测试结果
    accuracy = np.mean(y_pred == y_test_subset) * 100
    with open('../results/test_results.txt', 'w') as f:
        f.write("K-NN Classifier Test Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Best K value: {best_k}\n")
        f.write(f"Test samples: {test_subset_size}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Training data shape: {classifier.X_train.shape}\n")
        f.write(f"Test data shape: {X_test_subset.shape}\n")
    
    print(f"\n✅ Evaluation completed! Results saved to results/ directory")
    return accuracy

def main():
    """
    Main evaluation function
    """
    # 尝试加载之前训练好的模型
    classifier, X_test, y_test, best_k = load_trained_model()
    
    if classifier is None:
        print("!!! No trained model found. Please run train_knn.py first.")
        return
    
    # 评估模型
    accuracy = evaluate_model(classifier, X_test, y_test, best_k)
    
    print(f"\n🎯 Final Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()