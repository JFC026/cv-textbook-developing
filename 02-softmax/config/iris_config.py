"""鸢尾花数据集训练配置参数"""

IRIS_CONFIG = {
    'epochs': 500,
    'batch_size': 16,
    'learning_rate': 0.1,
    'regularization': 0.001,
    'test_size': 0.2,
    'random_state': 42,
    'input_dim': 4,
    'num_classes': 3
}