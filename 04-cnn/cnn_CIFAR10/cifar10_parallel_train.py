"""
CIFAR-10 并行训练脚本 (Data Parallelism)
原理：利用 Multiprocessing 实现多核并行计算梯度，模拟分布式训练
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # 后台绘图
import matplotlib.pyplot as plt
from pathlib import Path
import time
import multiprocessing as mp
from copy import deepcopy

# 导入你的底层库
from cifar10_cnn_model import CIFAR10_CNN, cross_entropy_loss, compute_accuracy
from cifar10_loader import load_cifar10_data, normalize_cifar10, get_cifar10_class_names
from improved_cnn_model import AdamOptimizer

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# =================================================================
# 核心辅助函数：处理梯度的提取与合并
# =================================================================

def get_grads_from_model(model):
    """
    从模型中提取所有梯度的扁平化列表
    用于从子进程传输梯度回主进程
    """
    grads = {}
    for i, layer in enumerate(model.trainable_layers):
        if hasattr(layer, 'dweights') and layer.dweights is not None:
            grads[f'dw_{i}'] = layer.dweights
        if hasattr(layer, 'dbiases') and layer.dbiases is not None:
            grads[f'db_{i}'] = layer.dbiases
        if hasattr(layer, 'dgamma') and layer.dgamma is not None:
            grads[f'dgamma_{i}'] = layer.dgamma
        if hasattr(layer, 'dbeta') and layer.dbeta is not None:
            grads[f'dbeta_{i}'] = layer.dbeta
    return grads

def accumulate_grads(main_model, grad_list):
    """
    将多个子进程的梯度累加到主模型中
    """
    # 清空主模型梯度
    for layer in main_model.trainable_layers:
        if hasattr(layer, 'dweights'): layer.dweights = np.zeros_like(layer.weights)
        if hasattr(layer, 'dbiases'): layer.dbiases = np.zeros_like(layer.biases)
        if hasattr(layer, 'dgamma'): layer.dgamma = np.zeros_like(layer.gamma)
        if hasattr(layer, 'dbeta'): layer.dbeta = np.zeros_like(layer.beta)

    # 累加梯度
    num_workers = len(grad_list)
    for grads in grad_list:
        for i, layer in enumerate(main_model.trainable_layers):
            if f'dw_{i}' in grads:
                layer.dweights += grads[f'dw_{i}']
            if f'db_{i}' in grads:
                layer.dbiases += grads[f'db_{i}']
            if f'dgamma_{i}' in grads:
                layer.dgamma += grads[f'dgamma_{i}']
            if f'dbeta_{i}' in grads:
                layer.dbeta += grads[f'dbeta_{i}']
    
    # 取平均 (梯度是 Sum，需要除以 Batch 总数，
    # 但原代码 backward 是求 Mean 还是 Sum? 
    # 原代码 cross_entropy_loss 中是 / batch_size。
    # 子进程算的是 sub_batch 的 mean。
    # Global mean = sum(sub_means * sub_size) / total_size
    # 这里简化处理：假设各子批次大小相等，直接把梯度累加后除以 worker 数即可)
    for layer in main_model.trainable_layers:
        if hasattr(layer, 'dweights'): layer.dweights /= num_workers
        if hasattr(layer, 'dbiases'): layer.dbiases /= num_workers
        if hasattr(layer, 'dgamma'): layer.dgamma /= num_workers
        if hasattr(layer, 'dbeta'): layer.dbeta /= num_workers

# =================================================================
# 子进程工作函数
# =================================================================

def worker_train_step(model, X_sub, y_sub, weight_decay):
    """
    子进程执行的任务：
    1. 接收模型副本和部分数据
    2. 前向传播 + 反向传播
    3. 返回梯度和 Loss
    """
    # 前向传播
    scores = model.forward(X_sub)
    
    # 计算损失
    loss, dscores = cross_entropy_loss(scores, y_sub, 
                                      weight_decay=weight_decay, 
                                      model=model)
    
    # 反向传播 (计算梯度)
    model.backward(dscores)
    
    # 计算准确率
    acc = compute_accuracy(scores, y_sub)
    
    # 提取梯度
    grads = get_grads_from_model(model)
    
    return grads, loss, acc

# =================================================================
# 主训练流程
# =================================================================

def train_parallel(epochs=10, batch_size=128, learning_rate=0.001, 
                   weight_decay=1e-3, dropout_rate=0.3, data_dir='data', num_workers=4):
    
    print("=" * 70)
    print(f"CIFAR-10 并行训练 (Data Parallelism)")
    print(f"并行核心数: {num_workers}")
    print("=" * 70)

    # 1. 加载数据
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
    X_train, X_test, mean, std = normalize_cifar10(X_train, X_test)
    
    # 配置数据量
    n_train = 20000  # 示例数据量
    n_test = 1000
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    print(f"训练样本: {n_train}, 测试样本: {n_test}")

    # 2. 初始化主模型
    main_model = CIFAR10_CNN(weight_decay=weight_decay, dropout_rate=dropout_rate)
    optimizer = AdamOptimizer(main_model.trainable_layers, learning_rate=learning_rate)
    
    train_losses = []
    train_accs = []
    
    # 3. 启动进程池
    # Windows 下创建 Pool 比较耗时，建议在循环外创建
    print(f"正在初始化进程池 ({num_workers} 核心)...")
    
    num_batches = len(X_train) // batch_size
    
    # 训练循环
    with mp.Pool(processes=num_workers) as pool:
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            # 打乱数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                # 获取当前 Batch 数据
                X_batch_total = X_train_shuffled[start_idx:end_idx]
                y_batch_total = y_train_shuffled[start_idx:end_idx]
                
                # === 步骤 A: 将 Batch 切分为子 Batch ===
                # np.array_split 自动处理切分不均的情况
                X_splits = np.array_split(X_batch_total, num_workers)
                y_splits = np.array_split(y_batch_total, num_workers)
                
                # 准备任务参数
                # 注意：必须把 main_model 传进去，这是最耗时的步骤（序列化模型）
                # 在工业界通常使用共享内存，这里为了简洁使用传参
                tasks = []
                for i in range(num_workers):
                    # 数据增强 (可选，在主进程做完再分发也可以，这里在分发前没做，可以在 worker 里做)
                    if len(X_splits[i]) > 0:
                        # 简单的随机翻转
                        if np.random.rand() > 0.5:
                            X_splits[i] = X_splits[i][:, :, :, ::-1]
                        
                        tasks.append((main_model, X_splits[i], y_splits[i], weight_decay))
                
                # === 步骤 B: 并行执行 (Map) ===
                # starmap 会阻塞直到所有子进程完成
                results = pool.starmap(worker_train_step, tasks)
                
                # === 步骤 C: 聚合结果 (Reduce) ===
                grad_list = [res[0] for res in results]
                losses = [res[1] for res in results]
                accs = [res[2] for res in results]
                
                # 累加梯度到主模型
                accumulate_grads(main_model, grad_list)
                
                # === 步骤 D: 更新权重 ===
                optimizer.update(main_model.trainable_layers, None)
                
                # 记录指标
                batch_loss = np.mean(losses)
                batch_acc = np.mean(accs)
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"\rEpoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{num_batches}] "
                          f"Loss: {batch_loss:.4f} Acc: {batch_acc*100:.2f}%", end="")
            
            # Epoch 结束
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            train_losses.append(avg_loss)
            train_accs.append(avg_acc)
            
            # 测试 (单进程测试即可，因为不需要反向传播，很快)
            test_scores = main_model.forward_test(X_test)
            test_acc = compute_accuracy(test_scores, y_test)
            
            print(f"\nEpoch {epoch+1} 耗时: {time.time()-epoch_start:.2f}s | "
                  f"Train Acc: {avg_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            
            # 学习率衰减
            if epoch == 10:
                optimizer.learning_rate *= 0.1

    print("\n并行训练完成！")
    
    # 保存模型
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    main_model.save_model(models_dir / 'cifar10_cnn_parallel.pkl')
    
    return main_model

if __name__ == '__main__':
    # Windows 下必须加这行
    mp.freeze_support()
    
    # 建议核心数：设为你 CPU 核心数的一半或全核
    # 例如 4 核或 8 核
    num_cores = min(4, mp.cpu_count())
    
    train_parallel(
        epochs=15,
        batch_size=128,  # 并行时 batch 可以设大一点
        learning_rate=0.001,
        weight_decay=1e-3,
        num_workers=num_cores
    )