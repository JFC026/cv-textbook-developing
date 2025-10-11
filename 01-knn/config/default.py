"""
配置文件 - 存储所有可配置参数
"""

# 数据配置
DATA_DIR = 'data'
CV_SAMPLE_SIZE = 10000  # 交叉验证样本数
TEST_SUBSET_SIZE = 1000  # 测试子集大小

# 模型配置
K_VALUES = [1, 3, 5, 7, 9, 11]  # 要测试的K值
N_FOLDS = 3  # 交叉验证折数

# 随机种子
RANDOM_STATE = 42
