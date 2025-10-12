import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn import KNearestNeighbor
from utils.data_helpers import load_data, preprocess_data
from utils.visualization import plot_nearest_neighbors

def main():
    """
    演示脚本：展示K-NN的工作原理和结果
    """
    print("=" * 60)
    print("        K-NN 分类器演示脚本")
    print("=" * 60)
    
    # ==================================================
    # 步骤 1: 加载数据
    # ==================================================
    print("\n[步骤 1] 加载数据...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data(data_dir='../data')
    X_train, y_train, X_test, y_test = preprocess_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    # ==================================================
    # 步骤 2: 训练模型（使用预计算的最佳K值或默认值）
    # ==================================================
    print("\n[步骤 2] 训练模型...")
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    
    # 尝试加载之前交叉验证得到的最佳K值
    try:
        import json
        with open('../results/cross_validation_results.json', 'r') as f:
            cv_results = json.load(f)
        best_k = cv_results['best_k']
        print(f"-> 使用之前计算的最佳K值: {best_k}")
    except:
        best_k = 3
        print(f"-> 使用默认K值: {best_k}")
    
    # ==================================================
    # 步骤 3: 可视化最近邻
    # ==================================================
    print("\n[步骤 3] 可视化测试样本及其最近邻...")
    
    # 使用少量测试样本进行演示
    demo_size = 1000
    X_demo = X_test[:demo_size]
    y_demo = y_test[:demo_size]
    
    print(f"-> 对 {demo_size} 个测试样本进行预测...")
    y_pred = classifier.predict(X_demo, k=best_k, batch_size=500)
    
    accuracy = np.mean(y_pred == y_demo) * 100
    print(f"-> 演示样本准确率: {accuracy:.2f}%")
    
    # 可视化最近邻
    plot_nearest_neighbors(
        classifier, 
        X_demo, 
        y_demo, 
        n_examples=5, 
        save_path='../results/nearest_neighbors.png'
    )
    
    # ==================================================
    # 步骤 4: 展示一些预测结果
    # ==================================================
    print("\n[步骤 4] 预测结果示例:")
    print("-" * 40)
    
    # 随机选择10个样本展示预测结果
    indices = np.random.choice(len(X_demo), 10, replace=False)
    correct_count = 0
    
    for i, idx in enumerate(indices):
        actual = y_demo[idx]
        predicted = y_pred[idx]
        status = "✓" if actual == predicted else "✗"
        
        if actual == predicted:
            correct_count += 1
            
        print(f"样本 {i+1:2d}: 真实={actual}, 预测={predicted} {status}")
    
    print(f"\n示例准确率: {correct_count}/10 ({correct_count*10}%)")
    print("\n✅ 演示完成！")

if __name__ == '__main__':
    main()
