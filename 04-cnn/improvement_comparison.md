# 小作业4: 简化版卷积神经网络 (CNN)

## 项目简介

本项目使用纯Python和Numpy从零开始实现一个简化版的卷积神经网络(CNN)，用于MNIST手写数字分类任务。这是《用纯Python手搓经典计算机视觉算法》开源教材项目的第四部分。

## 模型架构

本项目实现的CNN网络结构如下：

```
输入 (Input)
  ↓ (N, 1, 28, 28)  # N=批次大小，1=单通道（MNIST），28×28=图像尺寸
┌─────────────────────────────────────────────────────────────────┐
│ 第一卷积块（特征提取 + 降维）                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │ Conv1       │  │ BatchNorm1  │  │ ReLU        │  │ MaxPool1    │
│  │ 1→16通道    │  │ 标准化      │  │ 非线性激活  │  │ 2×2池化     │
│  │ 3×3卷积核   │  │ (可学习γ/β) │  │             │  │ 步幅=2      │
│  │ 步幅=1, 填充=1│  │             │  │             │  │ 降维→14×14  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│        ↓              ↓              ↓              ↓
│  (N,16,28,28)    (N,16,28,28)    (N,16,28,28)    (N,16,14,14)
└─────────────────────────────────────────────────────────────────┘
  ↓ (N, 16, 14, 14)
┌─────────────────────────────────────────────────────────────────┐
│ 第二卷积块（深层特征提取 + 降维）                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │ Conv2       │  │ BatchNorm2  │  │ ReLU        │  │ MaxPool2    │
│  │ 16→32通道   │  │ 标准化      │  │ 非线性激活  │  │ 2×2池化     │
│  │ 3×3卷积核   │  │ (可学习γ/β) │  │             │  │ 步幅=2      │
│  │ 步幅=1, 填充=1│  │             │  │             │  │ 降维→7×7    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│        ↓              ↓              ↓              ↓
│  (N,32,14,14)    (N,32,14,14)    (N,32,14,14)    (N,32,7,7)
└─────────────────────────────────────────────────────────────────┘
  ↓ (N, 32, 7, 7) → 展平 (Flatten) → (N, 32×7×7=1568)  # 特征向量化
┌─────────────────────────────────────────────────────────────────┐
│ 全连接块（分类 + 正则化）                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │ FC1         │  │ BatchNorm3  │  │ ReLU        │  │ Dropout     │
│  │ 1568→256维  │  │ 标准化      │  │ 非线性激活  │  │ 随机失活    │
│  │ 全连接层    │  │ (可学习γ/β) │  │             │  │ 训练模式：  │
│  │             │  │             │  │             │  │ 丢弃率=0.2  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│        ↓              ↓              ↓              ↓
│  (N,256)        (N,256)        (N,256)        (N,256)  # 测试模式无变化
│  ┌─────────────┐
│  │ FC2         │  # 最终分类层
│  │ 256→10维    │  # 10=MNIST类别数（0-9）
│  │ 全连接层    │
│  └─────────────┘
│        ↓
│  (N, 10)  # 输出10类的预测分数（未经过Softmax）
└─────────────────────────────────────────────────────────────────┘
  ↓ (N, 10)
输出 (Scores) → 后续通过Softmax转换为类别概率，计算交叉熵损失

```

## 核心组件实现

### 1. 卷积层 (ConvLayer)
- **功能**: 提取图像的局部特征
- **关键实现**:
  - 使用滑动窗口进行卷积计算
  - 支持padding和stride参数
  - He初始化权重
  - 实现前向传播和反向传播

### 2. 最大池化层 (MaxPoolLayer)
- **功能**: 降低特征图维度，增强特征不变性
- **关键实现**:
  - 选择池化窗口内的最大值
  - 反向传播时只对最大值位置传递梯度

### 3. ReLU激活函数
- **功能**: 引入非线性，加速训练
- **公式**: `f(x) = max(0, x)`

### 4. 全连接层 (FullyConnectedLayer)
- **功能**: 学习特征的高级组合
- **关键实现**:
  - 矩阵乘法: `y = Wx + b`
  - He初始化权重

### 5. Softmax分类器
- **功能**: 将输出转换为概率分布
- **损失函数**: 交叉熵损失

## 数据集

**MNIST手写数字数据集**
- **训练集**: 60,000张28×28灰度图像
- **测试集**: 10,000张28×28灰度图像
- **类别**: 0-9共10个数字



# CNN模型改进说明



## 🚀 核心改进点

### 1. Batch Normalization (批量归一化)

**作用**:
- 加速训练收敛
- 允许使用更大的学习率
- 减少对参数初始化的依赖
- 提供轻微的正则化效果

**实现位置**:
```python
Conv -> BatchNorm -> ReLU -> Pool
FC -> BatchNorm -> ReLU
```

**数学原理**:
```
# 训练时
μ = mean(x)           # 批次均值
σ² = var(x)           # 批次方差
x̂ = (x - μ) / √(σ² + ε)  # 归一化
y = γ * x̂ + β         # 缩放和平移

# 测试时使用移动平均
x̂ = (x - running_mean) / √(running_var + ε)
y = γ * x̂ + β
```

**关键代码**:
```python
class BatchNormLayer:
    def forward(self, x):
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            out = self.gamma * x_norm + self.beta
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out
```

### 2. Dropout (随机失活)

**作用**:
- 防止过拟合
- 相当于集成多个子网络
- 提升模型泛化能力

**实现位置**:
```python
FC1 -> ReLU -> Dropout(0.2) -> FC2
```

**数学原理**:
```
# 训练时
mask = random(0,1) > dropout_rate
out = x * mask / (1 - dropout_rate)  # Inverted Dropout

# 测试时
out = x  # 不应用dropout
```

**关键代码**:
```python
class DropoutLayer:
    def forward(self, x):
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x
```

### 3. Adam优化器

**优势对比SGD**:
- 自适应学习率
- 结合动量和RMSprop的优点
- 更快收敛
- 对超参数不敏感

**数学原理**:
```
# 一阶动量（梯度的指数移动平均）
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

# 二阶动量（梯度平方的指数移动平均）
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

# 偏差修正
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# 参数更新
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**关键代码**:
```python
class AdamOptimizer:
    def update(self, layers, grads):
        self.t += 1
        for i, layer in enumerate(layers):
            # 更新一阶动量
            self.m[f'w_{i}'] = self.beta1 * self.m[f'w_{i}'] + (1 - self.beta1) * layer.dweights
            # 更新二阶动量
            self.v[f'w_{i}'] = self.beta2 * self.v[f'w_{i}'] + (1 - self.beta2) * (layer.dweights ** 2)
            # 偏差修正
            m_hat = self.m[f'w_{i}'] / (1 - self.beta1 ** self.t)
            v_hat = self.v[f'w_{i}'] / (1 - self.beta2 ** self.t)
            # 更新参数
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
```

### 4. L2正则化

**作用**:
- 防止权重过大
- 提升泛化能力
- 平滑决策边界

**实现方式**:
```python
# 损失函数
loss = cross_entropy_loss + 0.5 * weight_decay * Σ(w²)

# 梯度中自动包含正则化项
dw = dw_ce + weight_decay * w
```

### 5. 网络结构优化



**改进版**:
```
Conv(1->16) -> BN -> ReLU -> Pool -> 
Conv(16->32) -> BN -> ReLU -> Pool -> 
FC(1568->256) -> BN -> ReLU -> Dropout -> 
FC(256->10)
```

**改进点**:
- 增加通道数: 8/16 → 16/32 (更强的特征提取能力)
- 增加全连接层维度: 128 → 256 (更强的表达能力)
- 每层后添加BatchNorm (训练稳定)
- 添加Dropout (防止过拟合)

## 📈 训练策略改进

### 超参数调整

| 参数 | 基础版 | 改进版 | 原因 |
|------|--------|--------|------|
| 学习率 | 0.01 | 0.001 | Adam适合更小学习率 |
| Batch Size | 32 | 64 | BatchNorm需要足够样本 |
| Epochs | 5 | 10-15 | 更充分训练 |
| 优化器 | SGD | Adam | 更快收敛 |


