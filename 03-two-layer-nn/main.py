import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.load_cifar import load_cifar10
from data.preprocess import preprocess_data, data_augmentation
from src.models.two_layer_net import TwoLayerFullyConnectedNet
from src.utils.optimizer import AdamOptimizer
from data.load_mnist import load_mnist, get_mnist_batches

# ==================== 配置定义区域 ====================

# CIFAR-10 训练配置
CIFAR10_CONFIG = {
    'input_size': 3072,        # 32x32x3=3072
    'hidden_size': 512,
    'output_size': 10,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'batch_size': 128,
    'data_augmentation': True,  # 是否使用数据增强
    'val_samples': 1000,       # 验证时使用的样本数
}

# MNIST 训练配置
MNIST_CONFIG = {
    'input_size': 784,         # 28x28=784
    'hidden_size': 256,
    'output_size': 10,
    'learning_rate': 0.001,
    'num_epochs':20,
    'batch_size': 128,
    'data_augmentation': False, # MNIST通常不需要数据增强
    'train_val_samples': 5000,  # 训练验证样本数
    'test_val_samples': 2000,   # 测试验证样本数
}

# 数据集配置映射
DATASET_CONFIGS = {
    'cifar10': CIFAR10_CONFIG,
    'mnist': MNIST_CONFIG
}

# ==================== 通用训练函数 ====================

def train_model(dataset_name='cifar10'):
    """通用训练函数"""
    
    # 获取配置
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"=== {dataset_name.upper()} 训练 ===")
    print(f"配置: 隐藏层{config['hidden_size']}, 学习率{config['learning_rate']}")
    print(f"      轮次{config['num_epochs']}, 批大小{config['batch_size']}")
    
    # 加载数据
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        x_train, y_train = preprocess_data(x_train, y_train, standardize=True)
        x_test, y_test = preprocess_data(x_test, y_test, standardize=True)
        from data.load_cifar import get_data_batches
        get_batches_fn = get_data_batches
    else:  # mnist
        (x_train, y_train), (x_test, y_test) = load_mnist()
        get_batches_fn = get_mnist_batches
    
    print(f"数据形状: 训练集{x_train.shape}, 测试集{x_test.shape}")
    
    # 初始化模型
    model = TwoLayerFullyConnectedNet(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size']
    )
    
    optimizer = AdamOptimizer(learning_rate=config['learning_rate'])
    
    # 训练循环
    best_test_acc = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(config['num_epochs']):
        batches = get_batches_fn(x_train, y_train, config['batch_size'])
        epoch_loss = 0
        num_batches = len(batches)
        
        for i, (x_batch, y_batch) in enumerate(batches):
            # 数据增强（仅CIFAR-10）
            if dataset_name == 'cifar10' and config['data_augmentation'] and i % 2 == 0:
                x_batch, y_batch = data_augmentation(x_batch, y_batch)
            
            loss = model.train_step(x_batch, y_batch, optimizer)
            epoch_loss += loss
            
            # 进度打印
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss:.4f}")
        
        # 计算本epoch的准确率
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 验证准确率
        if dataset_name == 'cifar10':
            val_samples = config['val_samples']
            train_acc = model.get_accuracy(x_train[:val_samples], y_train[:val_samples])
            test_acc = model.get_accuracy(x_test[:val_samples], y_test[:val_samples])
        else:  # mnist
            train_val_samples = config['train_val_samples']
            test_val_samples = config['test_val_samples']
            train_acc = model.get_accuracy(x_train[:train_val_samples], y_train[:train_val_samples])
            test_acc = model.get_accuracy(x_test[:test_val_samples], y_test[:test_val_samples])
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # 更新最佳准确率
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} 完成:")
        print(f"  Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        print(f"  最佳测试准确率: {best_test_acc:.4f}")
        print("-" * 60)
    
    # 最终评估
    final_train_acc = model.get_accuracy(x_train, y_train)
    final_test_acc = model.get_accuracy(x_test, y_test)
    
    print(f"训练完成!")
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    
    return {
        'model': model,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

# ==================== 单独的训练函数（保持兼容性） ====================

def train_cifar10():
    """训练CIFAR-10"""
    return train_model('cifar10')

def train_mnist():
    """训练MNIST"""
    return train_model('mnist')

def compare_datasets():
    """比较两个数据集的表现"""
    print("开始比较CIFAR-10和MNIST数据集...")
    
    # 训练CIFAR-10
    cifar_result = train_model('cifar10')
    
    print("\n" + "="*80 + "\n")
    
    # 训练MNIST
    mnist_result = train_model('mnist')
    
    print("\n" + "="*80)
    print("=== 数据集比较结果 ===")
    print(f"CIFAR-10 最终测试准确率: {cifar_result['final_test_acc']:.4f}")
    print(f"MNIST    最终测试准确率: {mnist_result['final_test_acc']:.4f}")
    print(f"准确率差异: {abs(cifar_result['final_test_acc'] - mnist_result['final_test_acc']):.4f}")
    
    return {
        'cifar10': cifar_result,
        'mnist': mnist_result
    }

# ==================== 主程序 ====================

if __name__ == "__main__":
    print("两层全连接网络")
    print("选择训练模式:")
    print("1.训练CIFAR-10")
    print("2.训练MNIST")
    print("3.比较两个数据集")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "2":
        train_model('mnist')
    elif choice == "3":
        compare_datasets()
    else:
        train_model('cifar10')