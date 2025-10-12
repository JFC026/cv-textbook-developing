import numpy as np
import os
import sys
import json

# 添加父目录到路径，这样可以从src的同级目录导入utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knn import KNearestNeighbor
from utils.data_helpers import load_data, preprocess_data
from utils.visualization import plot_accuracy_vs_k

def main():
    """
    主训练函数：加载数据、交叉验证、训练模型并保存结果
    """
    print("=" * 60)
    print("        K-NN 分类器训练脚本")
    print("=" * 60)
    
    # ==================================================
    # 步骤 1: 加载和预处理数据
    # ==================================================
    print("\n[步骤 1] 加载和预处理数据...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(data_dir='../data')
    X_train, y_train, X_test, y_test = preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    # ==================================================
    # 步骤 2: 使用交叉验证选择最佳K值
    # ==================================================
    print("\n[步骤 2] 使用交叉验证选择最佳K值...")
    classifier = KNearestNeighbor()
    
    # 为了节省时间，使用部分数据进行交叉验证
    cv_sample_size = 10000  # 用于交叉验证的样本数
    if len(X_train) > cv_sample_size:
        indices = np.random.choice(len(X_train), cv_sample_size, replace=False)
        X_train_cv = X_train[indices]
        y_train_cv = y_train[indices]
    else:
        X_train_cv = X_train
        y_train_cv = y_train
    
    print(f"-> 使用 {len(X_train_cv)} 个样本进行交叉验证...")
    
    # 定义要测试的K值范围
    k_values = [1, 3, 5, 7, 9, 11]
    best_k, k_scores = classifier.cross_validate(X_train_cv, y_train_cv, k_values=k_values, n_folds=3)
    
    # 可视化K值选择结果
    plot_accuracy_vs_k(k_scores, save_path='../results/accuracy_vs_k.png')
    
    # 保存交叉验证结果
    cv_results = {
        'best_k': best_k,
        'k_scores': {k: float(score) for k, score in k_scores.items()}
    }
    
    with open('../results/cross_validation_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # ==================================================
    # 步骤 3: 使用最佳K值在完整训练集上训练
    # ==================================================
    print(f"\n[步骤 3] 使用最佳 K={best_k} 在完整训练集上训练...")
    classifier.train(X_train, y_train)
    
    print("✅ 训练完成！")
    # 保存训练好的模型
    classifier.save_model('../results/knn_model.npz')
    print("✅ Model saved to ../results/knn_model.npz")
    return classifier, X_test, y_test, best_k

if __name__ == '__main__':
    classifier, X_test, y_test, best_k = main()
