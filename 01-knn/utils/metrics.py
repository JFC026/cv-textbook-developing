import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def calculate_precision_recall_f1(y_true, y_pred):
    """计算精确率、召回率和F1分数"""
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return precision, recall, f1

def detailed_classification_report(y_true, y_pred):
    """生成详细的分类报告"""
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

def print_evaluation_metrics(y_true, y_pred):
    """打印评估指标"""
    accuracy = calculate_accuracy(y_true, y_pred) * 100
    precision, recall, f1 = calculate_precision_recall_f1(y_true, y_pred)
    
    print("\n--- 详细评估指标 ---")
    print(f"准确率: {accuracy:.2f}%")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 各类别准确率
    print("\n--- 各类别准确率 ---")
    for i in range(10):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_true[mask]) * 100
            print(f"类别 {i}: {class_acc:.2f}% ({np.sum(mask)} 个样本)")
