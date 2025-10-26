"""Fashion-MNIST数据集训练配置参数"""

FASHION_MNIST_CONFIG = {
    'epochs': 30,
    'batch_size': 128,
    'learning_rate': 0.01,
    'regularization': 0.001,
    'validation_size': 10000,
    'input_dim': 28 * 28,
    'num_classes': 10
}