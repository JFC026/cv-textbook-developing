# Softmax分类器及交叉熵损失

## 引言

欢迎来到《纯Python"手搓"图像分类器》中的softmax分类器讲解。在本章，我们将从概率分布的朴素理解出发，层层剖析Softmax分类器的核心逻辑。它不仅是深度学习中多类别分类的基石，更是帮我们打通"原始分数"到"概率决策"的关键桥梁。Softmax分类器是一种"生成概率分布"的工具，它通过简洁的数学变换将模型输出的无序数值转化为可解释的类别概率，这种确定性的映射关系使其成为图像分类、文本识别等任务中最经典的输出层方案，帮助我们建立对"置信度量化"与"损失优化"的核心认知。

在机器学习与深度学习的分类任务中，模型的核心目标是将输入数据映射到离散的类别标签。例如，图像识别中判断一张图片是"猫""狗"还是"汽车"，文本分类中确定一段文字的情感是"积极""消极"还是"中性"。这类任务的关键挑战在于：如何将模型输出的连续数值（通常称为"logits"或"分数"）转化为具有明确概率意义的类别分布，同时设计合理的损失函数衡量预测结果与真实标签的差异，以指导模型参数优化。

Softmax分类器正是为解决这一问题而生的经典工具。它通过Softmax函数将原始分数转换为归一化的概率分布，再结合交叉熵损失函数量化预测误差，构成了端到端的分类框架。这一组合不仅具备良好的数学可解释性，还能高效地与反向传播算法结合，成为深度学习中多类别分类任务的基础组件，广泛应用于图像、文本、语音等领域。

## 初步介绍

什么是Softmax分类器？

简单来说，Softmax分类器是逻辑回归（Logistic Regression）在多类别分类问题上的自然推广。

逻辑回归：用于二分类，它将一个输入向量映射到一个介于0和1之间的概率值，表示属于正类的概率。

Softmax分类器：用于多分类（假设有K个类别），它将一个输入向量映射到一个包含K个元素的概率分布。每个元素的值介于0和1之间，且所有元素之和为1。

它的**核心思想**是：将模型输出的原始分数（Logits）转化为所有类别的概率分布，从而我们可以选择概率最大的类别作为预测结果。

## 算法原理

#### 线性变换

对于输入数据x（一个特征向量），我们为每个类别i计算一个原始得分 $z_i$ 

先通过 $z_i = W_i x + b_i$ 计算每个类别的"原始分数"（logits）。其中，x是输入特征，W是权重矩阵，b是偏置向量。 $z_i$ 即第i类的"原始分数"（logits），其值可正可负，不直接具备概率意义，但能反映输入x与第i类的"匹配程度"——数值越高，说明从线性变换的角度看，x越可能属于第i类。

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/LinearLayer.png" alt="图1.1 线性层的计算流程" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1.1 线性层的计算流程</div>
</div>

这步其实就是普通的线性层（Linear Layer）。

#### Sigmoid函数和Softmax函数

Sigmoid函数可以将输入的一个实数映射到0-1区间上。从图中我们可以看到自变量X的取值范围为正无穷到负无穷的一切实数，Y的取值范围是从0到1的。那么任意的一个X1我们都可以得到一个在[0,1]上的Y1，也就是我们可以把所有的值都压缩到0到1这个区间内，结合之前的得分函数，一个输入对于每一个类别的得分X，我们都可以把这个得分映射到[0,1]区间内，也就是把我们的得分数值转成了相应的概率值。

Sigmoid的计算公式如下：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/SigGraph.png" alt="图1.2 Sigmoid函数图" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1.2 Sigmoid函数图</div>
</div>

Softmax函数又称归一化指数函数，是基于sigmoid二分类函数在多分类任务上的推广；在多分类网络中，常用Softmax作为最后一层进行分类。分子 $e^{z_i}$ 是对第i类的原始分数取自然指数，确保数值为正，且所有类别的概率之和为1。

Softmax的计算公式如下：

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}} \quad \text{for } i = 1,2,\ldots,n $$

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/SMLayer.png" alt="图1.3 Softmax层计算流程 width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1.3 Softmax层计算流程</div>
</div>

这步其实就是Softmax层（Softmax Layer）。

softmax可以对线性层的输出做规范化校准：保证输出为非负且总和为1。因为如果直接将未规范化的输出看作概率，会存在2点问题：

1. 线性层的输出并没有限制每个神经元输出数字的总和为1；
2. 根据输入的不同，线性层的输出可能为负值。

#### Softmax分类器与交叉熵损失

简单的说，softmax函数会将输出结果缩小到0到1的一个值，并且所有值相加为1，使用softmax函数对前面线性分类求得分，所求得的得分值即为该类别的概率。因为我们需要他们之间的差异更大，这样我们就可以很轻松的分辨出来这个输入到底是什么。这个时候我们就引入交叉熵损失函数，来衡量分类的好坏。
<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/SMAlg.png" alt="图2.1 交叉熵损失函数流程" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2.1 交叉熵损失函数流程</div>
</div>
交叉熵损失衡量分类模型的性能，其输出是介于0和1之间的概率值。交叉熵损失随着预测概率与实际标签的偏离而增加。因此，当实际观察标签为1时预测的概率模型不好，并导致高损失值。cross-entropy一般再softmax函数求得结果后再用。

对于一个包含K个类别的分类任务，设样本的真实标签为向量 $t = [t_1,t_2,\ldots,t_K]$ （其中$t_i$表示样本属于第i类，其余 $t_j=0$ ），模型通过Softmax函数输出的预测概率分布为 $p = [p_1,p_2,\ldots,p_K]$ （ $p_i$ 为样本属于第i类的概率），且 $\sum_{i=1}^{K}p_i = 1$ ，则交叉熵损失公式为：

$$L_{CE} = -\sum_{i=1}^{n}t_i\log(p_i) = -\log(p_{\text{true}})$$

因为softmax求出结果在(0,1)之间，所以cross-entropy结果为负值，加负号使得损失为正。直观体现了预测偏差与损失的正相关，值越小，预测结果越精准，值越大反之。

接下来，我们举一个例子来阐述分类器最终损失值的计算逻辑：

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/SMex.png" alt="图2.2 Softmax分类器计算损失值流程" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2.2 Softmax分类器计算损失值流程</div>
</div>

计算步骤：

第一步：对应于一个输入计算出它属于每种类别的得分数值。
第二步：利用sigmoid函数把所有的得分值映射成一个概率值。
第三步：对最终正确分类所占的概率求一个log值再取负号

整个流程清晰呈现了从"原始得分→概率分布→损失计算"的完整逻辑，帮助理解多分类任务中Softmax和交叉熵损失的协同作用。此处损失为0.89，量化了模型预测与真实标签的差异。
<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/SMGeneral.jpg" alt="图2.3 Softmax总流程图 width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2.3 Softmax总流程图</div>
</div>

## 项目简介

## 代码实现
# 实验一：在Fashion-MNIST数据集上测试
## 1. 数据集介绍
Fashion-MNIST 是一个用于图像分类任务的基准数据集，旨在替代传统 MNIST 手写数字数据集，更贴近真实世界的计算机视觉应用场景。Fashion-MNIST 由 60,000 个示例的训练集和 10,000 个示例的测试集组成。每个示例都是 28x28 的灰度图像，与 10 个类的标签相关联。每个训练和测试示例都分配给以下标签之一：
| 标签 | 描述     |
| ---- | -------- |
| 0    | T恤/上衣 |
| 1    | 裤子     |
| 2    | 套衫     |
| 3    | 连衣裙   |
| 4    | 外套     |
| 5    | 凉鞋     |
| 6    | 衬衫     |
| 7    | 运动鞋   |
| 8    | 袋       |
| 9    | 踝靴     |

数据集外观展示

![](https://ai-studio-static-online.cdn.bcebos.com/c3e436607367476794bb3bfaf4ac3da17ad5870fcf514bbe8ff0fe57da99212c)



![](https://ai-studio-static-online.cdn.bcebos.com/02b6e7eb3c9c46ff95986e9ffd69e9731000120f30454f67beadbd1d40080067)

这个数据集获取很简单，GitHub上就有，下载到本地后就可以在项目中运行了，具体处理步骤不再赘述。由于Github文件大小的限制，data文件夹这里不再提供具体数据，可以去aistudio的链接查看。

## 2.结合算法数学逻辑书写代码

有了对数据集的基本了解就可以开始手撕了，为了阅读方便起见这里先给出一些数学公式的符号说明：
 $N$  — 当前批次样本数（batch size），训练集中总样本数记为  $N_{\text{train}}$ 。
 $D$ — 特征维度（图像展平成向量时 $D = 28 \times 28 = 784$ ）。
 $C$ — 类别数（Fashion-MNIST $C = 10$ ）。
矩阵/向量形状约定：
  $X \in \mathbb{R}^{N \times D}$ （每一个样本）
  $W \in \mathbb{R}^{D \times C}$ （权重）
  $b \in \mathbb{R}^{1 \times C}$ （偏置，广播到 $N \times C$ ）
  $S = XW + 1b \in \mathbb{R}^{N \times C}$ （logits / scores）（后面提到的 $s_{ij}$ 是里面的元素）
  $P \in \mathbb{R}^{N \times C}$ （softmax 概率，每行和为 1）
  $Y \in \{0,1\}^{N \times C}$ （one-hot 标签）
 
 先来看看代码主体部分的数学逻辑，我们先对权重矩阵以及偏置初始化一下：
```python
class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes):
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(input_dim, num_classes))
        self.b = np.zeros(shape=(1, num_classes))

```
这里选择一些较小的正态随机数比较常用，也可以避免数值溢出。

```python
    def softmax(self, scores):
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

```
这个函数就定义了softmax的概率计算方式即：  
 $p_{i,j} = \frac{e^{s_{i,j}}}{\sum_{k=1}^{C} e^{s_{i,k}}}$ 

考虑到数值的稳定性不至于太大，我们可以做一个这样的变换：先减去行最大值 $m_i = \max_k s_{i,k}$ ，因为 $\frac{e^{s_{i,j}}}{\sum_{k} e^{s_{i,k}}} = \frac{e^{s_{i,j} - m_i}}{\sum_{k} e^{s_{i,k} - m_i}}$

最终返回的 $\text{probs}$ 即矩阵 $P$ $(N, C)$ ，每行和为 1。

```python
    def forward(self, X, y=None, reg=0.0):
        N = X.shape[0]
        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)

        if y is None:
            return probs, 0.0

        cross_entropy = -np.sum(y * np.log(probs + 1e-10)) / N
        reg_loss = 0.5 * reg * np.sum(self.W **2)
        total_loss = cross_entropy + reg_loss
        return probs, total_loss
    
    
    def backward(self, X, y, reg=0.0):
        N = X.shape[0]
        probs, _ = self.forward(X)
        dscores = (probs - y) / N
        dW = np.dot(X.T, dscores) + reg * self.W
        db = np.sum(dscores, axis=0, keepdims=True)
        return dW, db

    
```
这两个函数设计了前向传播和反向传播过程，是极为重要的两个过程，它的步骤是这样的：首先为了计算效率和内存考虑，选取一个batch的值作为一次前向加反向传播中同时处理的样本数，值通常为2的幂次如128.然后调用上面说过的线性映射方式，计算对于第 $i$ 个样本，第 $j$ 类的分数：
$s_{i,j} = \sum_{k=1}^{D} x_{i,k} W_{k,j} + b_j$

然后 $softamx$ 把刚才的logits转概率：

 $p_{i,j} = \frac{e^{s_{i,j} - m_i}}{\sum_{k=1}^{C} e^{s_{i,k} - m_i}}, \quad m_i = \max_k s_{i,k}$ 
 
然后计算交叉熵的损失（每个batch平均）：
 $L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} Y_{ij} \log p_{ij}$ 

以及一个L2正则项：  
$L_{\text{reg}} = \frac{\lambda}{2} \sum_{a,b} W_{ab}^2$

（为什么这里加了一个正则项呢，因为这里的特征太多了并且一个批次数据较少，是为了防止过拟合！）

总损失为他们两的和。
从这点上来看前向传播就是先把原始的像素通过线性映射到每一类的“分数”，然后softmax把分数归一化为概率，交叉熵就是把概率和真实标签比对并且量化误差。

然后是反向传播，我们要求 $\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial b}$ 。因为损失表达式里 $W$ 与 $S$ 的关系是 $S = XW + 1b$ ，所以按链式法则：


对单个样本 $i$ 的交叉熵：
$$L^{(i)} = -\sum_{k=1}^{C} Y_{i,k} \log p_{i,k}, \quad p_{i,k} = \frac{e^{s_{i,k}}}{\sum_{m} e^{s_{i,m}}}$$

对 $s_{i,j}$ 求导（步骤）：
1. $\log p_{i,k} = s_{i,k} - \log \sum_{m} e^{s_{i,m}}$
2. $\frac{\partial \log p_{i,k}}{\partial s_{i,j}} = \delta_{jk} - \frac{e^{s_{i,j}}}{\sum_{m} e^{s_{i,m}}} = \delta_{jk} - p_{i,j}$
3. 因此
$$\frac{\partial L^{(i)}}{\partial s_{i,j}} = -\sum_{k} Y_{i,k} (\delta_{jk} - p_{i,j}) = -Y_{i,j} + p_{i,j} \sum_{k} Y_{i,k}$$

对于 one-hot 标签， $\sum_{k} Y_{i,k} = 1$  ，得到：

$$\frac{\partial L^{(i)}}{\partial s_{i,j}} = p_{i,j} - Y_{i,j}$$



批量（平均）则是把每个样本除以 $N$ ：
$$D = \frac{1}{N}(P - Y) \quad \text{}$$
这里 $D$ 的 shape 为 $N \times C$ 。
于是得到了损失对$S$的偏导。同理我们求出其他的偏导：

 从 $\partial L/\partial S$ 到 $\partial L/\partial W$ 与 $\partial L/\partial b$
因为 $S = XW + 1b$ ，对 $W$ 的导数（矩阵形式）：
$$\frac{\partial L}{\partial W} = X^{\top} \frac{\partial L}{\partial S} + \lambda W = X^{\top} D + \lambda W$$



对偏置 $b$ ：
$$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial s_{i,\cdot}} = \sum_{i=1}^{N} D_{i,\cdot}$$

即把 $D$ 按行求和。

更新规则（梯度下降）：
$$W \leftarrow W - \eta \nabla_W L, \quad b \leftarrow b - \eta \nabla_b L$$

所以从整个反向传播的过程来看，反向传播实际上提供了一个系统化可以复用的方式来计算这些导数避免繁琐重复的工作，他也解决了一个问题：如何把最终损失分配到各层以及各个权重上去？其实这些导数的值已经给出了答案，它告诉我们如果我微调某个权重，会怎么样去影响我们的损失。真正意义是解决 **“谁对错误负责、应该怎样调整”** 的问题。


有了这些个步骤，我们就可以开始写代码训练了：

```python
        def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=128, lr=0.01, reg=0.001):
        N_train = X_train.shape[0]
        num_batches = max(1, N_train // batch_size)

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(N_train)
            X_shuffled = X_train[shuffle_idx]
            y_shuffled = y_train[shuffle_idx]

            total_loss = 0.0
            for batch in range(num_batches):
                start = batch * batch_size
                end = min(start + batch_size, N_train)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                _, batch_loss = self.forward(X_batch, y_batch, reg)
                total_loss += batch_loss

                dW, db = self.backward(X_batch, y_batch, reg)
                self.W -= lr * dW
                self.b -= lr * db

            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / num_batches
                val_acc = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        def predict(self, X):
            probs, _ = self.forward(X)
            return np.argmax(probs, axis=1)

        def evaluate(self, X, y):
            y_pred = self.predict(X)
            y_true = np.argmax(y, axis=1)
            return np.mean(y_pred == y_true)

```

简单来说代码就是对训练样本随机重排后，分批次输入到我们的计算网络中，不断调用之前写的函数，累计每一个batch的损失。并且在若干次参数更新后打印一次验证准确率和平均损失值。最后返回每行概率最大索引，输出预测为正确的比例。
## 3.代码运行结果
在Fashion-MNIST数据集上运行代码得到:

训练集：(50000, 784)，验证集：(10000, 784)，测试集：(10000, 784)  
类别数：10（10类服饰）  
Epoch 5/30 | Loss: 0.6426 | Val Acc: 0.8003  
Epoch 10/30 | Loss: 0.5687 | Val Acc: 0.8215  
Epoch 15/30 | Loss: 0.5369 | Val Acc: 0.8296  
Epoch 20/30 | Loss: 0.5187 | Val Acc: 0.8343  
Epoch 25/30 | Loss: 0.5065 | Val Acc: 0.8377  
Epoch 30/30 | Loss: 0.4978 | Val Acc: 0.8410

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA94AAAD1CAYAAACm7i1DAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAABLWklEQVR4nO3dd5RUVdb38V8DHegANHQgSY42CIoCIlEQBlBBQdI4AmJEZZxxDOCzxOxgTgOKjyIiKmIeARUcwDQqCoioICBBcm4yDc19/+Dteiz67kNXwaW74ftZi8XqfWvXPVV9z617uqr2jvE8zxMAAAAAAAhEicIeAAAAAAAAJzMW3gAAAAAABIiFNwAAAAAAAWLhDQAAAABAgFh4AwAAAAAQIBbeAAAAAAAEiIU3AAAAAAABYuENAAAAAECAWHgDAAAAABAgFt5F1KxZsxQTE6O33nrLebuXX35ZMTExWrFixYkZGFDEDRo0SDVq1AiLxcTE6O677y6U8QAAABQngwYNUnJycoFuyzVWwRWJhXdMTEyB/s2aNauwh+q0fft2JSQkKCYmRr/88kthDycqeQv+P/4rX768WrZsqYkTJwa+/9dee01PPvlk4Ps5mRT3+fPHMZYoUUKVK1dW586di+x4UbwU1/kxaNCgAo170KBBhT1UFDPMCSB6xXX+5Nm1a5dGjhypRo0aKSkpSRUqVFDTpk3117/+VWvXrg18/1OnTj2lF+mlCnsAkjRhwoSwn1955RVNnz49X7xhw4YnclgRmzx5smJiYlSxYkVNnDhR999/f2EPKWrDhg3TOeecI0nasmWLJk2apMsvv1zbt2/XDTfcENh+X3vtNS1cuFA333xzYPs42ZwM8+eCCy7QFVdcIc/ztHz5co0ePVrnn3++pkyZoq5duxb28FCMFdf5ce2116pTp06hn5cvX6677rpL11xzjdq0aROK165duzCGh2KMOQFEr7jOH0k6cOCA2rZtq0WLFmngwIG66aabtGvXLv3000967bXXdMkll6hy5coR3+/evXtVqlTBlpRTp07Vv/71r1N38e0VQTfccINXkKHt3r37BIym4Nq2betdeuml3t/+9jevZs2ax3RfM2fO9CR5kydPdt5u3LhxniRv+fLlx7S/o+13//79XpUqVbxWrVodl/1Yunfv7lWvXj3QfZzsitv8keTdcMMNYbEFCxZ4krzOnTtHfH8DBw7MdwxJ8kaOHHkMowzO3r17vdzc3MIeximjuM2PPHPmzPEkeePGjXPebteuXSdmQMcZ86DwMCeKJuZE8VCc5s+bb77pSfImTpyYb9vevXu97Ozs0M8DBw70kpKSjtu+8+ZhQZ+vk1WR+Kh5QbRv316NGjXS999/r7Zt2yoxMVEjRoyQZH+3oEaNGvk+crR9+3bdfPPNOu200xQfH686depo1KhROnToUNjt1q1bp0WLFunAgQMFGt+qVav0+eefq1+/furXr5+WL1+ur776ynwcP//8szp06KDExERVqVJFDz/88FH3sX//fl144YUqW7as733/0bRp09SmTRslJSUpJSVF3bt3108//VSgx+InLi5Oqamp+f6idfDgQd13332qXbu24uPjVaNGDY0YMUL79+/Pdx+jR49WVlaW4uPjVblyZd1www3avn17aHv79u01ZcoUrVy5MvRRnSO/q4voFPX5c6TGjRsrLS1Ny5cvl2TXMsj7akQ0H+maN2+eunbtqjJlyig5OVkdO3bU119/Hdr+3XffKSYmRuPHj8+X+/HHHysmJkYffvhhKLZmzRpdeeWVyszMVHx8vLKysvTSSy/5jveNN97Q//zP/6hKlSpKTEzUjh07Ih4/jp/iNj/y5M2L2bNna+jQocrIyFDVqlVD2492zrUeh3T4OWnfvn1Y7JlnnlFWVpYSExOVmpqqs88+W6+99lrYbZgHJwfmRPjjkJgTKLiiOn+WLVsmSTrvvPPybUtISFCZMmXyxdesWaOePXsqOTlZ6enp+sc//qHc3Nyw2xz5mO6++27FxMTo559/1oABA5SamqrWrVtr0KBB+te//hXKyft3KikSHzUvqC1btqhr167q16+fLr/8cmVmZkaUv2fPHrVr105r1qzRtddeq2rVqumrr77S8OHDtW7durDvFg8fPlzjx4/X8uXLC7T4e/3115WUlKQLL7xQpUuXVu3atTVx4kS1atUq3223bdumP/3pT7r00kvVp08fvfXWW7r99tvVuHFj82O1e/fuVY8ePfTdd99pxowZoY+B+5kwYYIGDhyoLl26aNSoUdqzZ4/GjBmj1q1ba968eQV6PDt37tTmzZslSVu3bg19BPzFF18Mu91VV12l8ePHq3fv3rrlllv0zTff6KGHHtIvv/yid999N3S7u+++W/fc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O3KMQejbt68eeOABbd68WSkpKfrggw/Uv3//0FzKm3OXX355vu+C5znjjDPCfuYdjaKnKM+PozmW48k1t/74WtWwYUMtXrxYH374oT766CO9/fbbGj16tO666y7dc889zIOTEHMiHHMCkSgO86d69eq68sordckll6hWrVr5WiFb65WC4njOr1gtvC2pqan5qlLm5ORo3bp1YbHatWtr165dzgv8SM2ePVurV6/Wvffem69n37Zt23TNNdfovffeC/toeSR69uypzp07a9CgQUpJSdGYMWOct8/rX5mRkXFcH6d0uIK5JO3atUvS4Ql76NAhLVmyJOyxb9iwQdu3bw8Vccj7f/HixWEf7c3JydHy5cvDxski/sQrzPlTUKmpqZKUb5zRvEuRnp6uxMRELV68ON+2RYsWqUSJEjrttNNCsb59++qee+7R22+/rczMTO3YsUP9+vULu7+UlBTl5uYWynODYBWH+XGkSM65fo9POjy3/pgrSUlJSerbt6/69u2rnJwcXXrppXrggQc0fPhw5sEphDnxf5gTiFRRnD+pqamqXbu2Fi5cGPi+TvXr/GL1HW9L7dq1w77rIEljx47N99ejPn366L///a/v9663b98eWlhKBS/Nn/cx81tvvVW9e/cO+3f11Verbt26mjhx4jE8OumKK67Q008/reeee063336787ZdunRRmTJl9OCDD/qOfdOmTVGPI691UpMmTSRJ3bp1k6SwyoqS9Pjjj0tS6F3xTp06KS4uTk8//XTYu/ovvviisrOzw949T0pKUnZ2dtRjROQKc/5EMkZJYePMzc01P77nUrJkSXXu3Fnvv/9+WHuyDRs26LXXXlPr1q3DvpLRsGFDNW7cWJMmTdKkSZNUqVIltW3bNuz+evXqpbffftv3RetY5hwKX3GYH0eK5Jxbu3Ztff3118rJyQnFPvzwQ/3+++9h97lly5awn+Pi4nT66afL8zwdOHCAeXAKYU4cxpxANApz/vzwww+hjkV/tHLlSv3888++X8E73vK+Ruj3x61TwUnxjvdVV12l6667Tr169dIFF1ygH374QR9//LHS0tLCbnfrrbfqgw8+0IUXXqhBgwapWbNm2r17t3788Ue99dZbWrFiRSinIKX59+/fr7ffflsXXHCBEhISfG9z8cUX66mnntLGjRuVkZER9WO88cYbtWPHDt15550qW7ZsqB/gkcqUKaMxY8boL3/5i8466yz169dP6enpWrVqlaZMmaLzzjtPzz777FH39/nnn2vfvn2SDrcT++CDDzR79mz169dPDRo0kHR4AT5w4ECNHTtW27dvV7t27fTtt99q/Pjx6tmzpzp06CDp8DuCw4cP1z333KM//elPuvjii7V48WKNHj1a55xzTtinAZo1a6ZJkybp73//u8455xwlJyfroosuivp5w9EV1vyJRFZWllq2bKnhw4dr69atKl++vN54442wF55I3H///Zo+fbpat26toUOHqlSpUnr++ee1f/9+Pfzww/lu37dvX911111KSEjQkCFD8n3l45///KdmzpypFi1a6Oqrr9bpp5+urVu3au7cuZoxY4a2bt0a1ThR+IrD/DhSJOfcq666Sm+99Zb+9Kc/qU+fPlq2bJleffXV0B+78nTu3FkVK1bUeeedp8zMTP3yyy969tln1b17d6WkpEhiHpwqmBOHMScQjcKcP9OnT9fIkSN18cUXq2XLlkpOTtZvv/2ml156Sfv37/ftL368NWvWTJI0bNgwdenSRSVLlgz7FOFJ70SXUS8IqzS/1bYhNzfXu/322720tDQvMTHR69Kli7d06VLflhA7d+70hg8f7tWpU8eLi4vz0tLSvFatWnmPPvqol5OTE7pdQUrzv/32254k78UXXzRvM2vWLE+S99RTTzkfx5EtkP7YTuyPbrvtNk+S9+yzz3qe526z1KVLF69s2bJeQkKCV7t2bW/QoEHed999Z471j/v947+4uDivQYMG3gMPPBD2HHme5x04cMC75557vJo1a3qxsbHeaaed5g0fPtzbt29fvvt+9tlnvQYNGnixsbFeZmamd/3113vbtm0Lu82uXbu8AQMGeOXKlQu1O0Bkisv8ySPJu+GGG456u2XLlnmdOnXy4uPjvczMTG/EiBHe9OnTo2on5nmeN3fuXK9Lly5ecnKyl5iY6HXo0MH76quvfPe9ZMmS0Hz44osvfG+zYcMG74YbbvBOO+00LzY21qtYsaLXsWNHb+zYsaHbWPMaJ05xmx95XK2T5syZ45tTkHOu53neY4895lWpUsWLj4/3zjvvPO+7777L1zrp+eef99q2betVqFDBi4+P92rXru3deuutXnZ2dth9MQ+KH+bEtny3Y06goIrT/Pntt9+8u+66y2vZsqWXkZHhlSpVyktPT/e6d+8eagH8x/tMSkrKdx95rcL+6MhrrLzbbNq0KV/+wYMHvZtuuslLT0/3YmJiTrnWYjGe51PRCwAAAAAAHBcnxXe8AQAAAAAoqlh4AwAAAAAQIBbeAAAAAAAEiIU3AAAAAAABYuENAAAAAECAWHgDAAAAABAgFt7/X40aNTRo0KDCHkaYu+++WzExMdq8ebPzdoMGDVKNGjVOzKAAQ1GbQ0eOZ9asWYqJidGsWbMKbUw4dRW1+QEUNuYEEJ0TNXdq1KihCy+88Ki34/qq4IrEwvvll19WTExM6F9CQoLq1aunG2+8URs2bCjs4UVk6tSpiomJUeXKlXXo0KHCHk5U8hb8ef9KlCihSpUq6cILL9TXX38d+P4ffPBBvffee4Hv52RS3OdQ3kk7719sbKxq1aqlK664Qr/99lthDw/FXHGeHzVq1Agbu/Xv5ZdfLuyhohhhTgDRKc5zJ8+KFSs0ePBg1a5dWwkJCapYsaLatm2rkSNHnpD9jx49+pSdn6UKewB/dO+996pmzZrat2+fvvjiC40ZM0ZTp07VwoULlZiYWNjDK5CJEyeqRo0aWrFihf7zn/+oU6dOhT2kqI0ZM0bJyck6dOiQfv/9d73wwgtq27atvv32WzVt2jSw/T744IPq3bu3evbsGdg+TlbFfQ4NGzZM55xzjg4cOKC5c+dq7NixmjJlin788UdVrly5sIeHYq44zo8nn3xSu3btCv08depUvf7663riiSeUlpYWirdq1aowhodijjkBRKc4zh1JWrp0qc455xyVLl1aV155pWrUqKF169Zp7ty5GjVqlO65556I77Nt27bau3ev4uLiCnT70aNHKy0t7ZT8xEuRWnh37dpVZ599tiTpqquuUoUKFfT444/r/fffV//+/X1zdu/eraSkpBM5TNPu3bv1/vvv66GHHtK4ceM0ceLEYr3w7t27d9iLWM+ePdWoUSNNnjw50IU3olfc51CbNm3Uu3dvSdLgwYNVr149DRs2TOPHj9fw4cMLeXTBKkq/h5NVcZwfR/4Bcv369Xr99dfVs2dP51eMCnvc0Squ4y6umBNFX3Ed98muOM4dSXriiSe0a9cuzZ8/X9WrVw/btnHjxqjus0SJEkpISDjq7fbs2VOk/yhxIhSJj5pbzj//fEnS8uXLJR3+LnNycrKWLVumbt26KSUlRX/+858lSYcOHdKTTz6prKwsJSQkKDMzU9dee622bdsWdp+e5+n+++9X1apVlZiYqA4dOuinn37y3f+yZcu0bNmyAo/33Xff1d69e3XZZZepX79+euedd7Rv3758t4uJidGNN96o9957T40aNVJ8fLyysrL00UcfHXUfK1euVJ06ddSoUSPnR1oK+nxEomLFipKkUqXC/16zceNGDRkyRJmZmUpISFCTJk00fvz4fPm7d+/WLbfcotNOO03x8fGqX7++Hn30UXmeF7pNTEyMdu/erfHjx4c+xnMq/kXseCluc6gg4/e7sMr7ekQ0Jk+erGbNmql06dJKS0vT5ZdfrjVr1oS2P/roo4qJidHKlSvz5Q4fPlxxcXFhz9E333yjP/3pTypbtqwSExPVrl07ffnll77j/fnnnzVgwAClpqaqdevWUY0f0Svu8yOPa9wFOe+uWLHC/GhuTEyM7r777tDPO3fu1M0336waNWooPj5eGRkZuuCCCzR37tywPOZB8cScOIw5gUgVl7mzbNkyVa1aNd+iW5IyMjJ8c7744gs1b95cCQkJqlWrll555ZWw7X7f8W7fvr0aNWqk77//Xm3btlViYqJGjBihGjVq6KefftLs2bND1/nt27c/6rhPFkXqHe8j5R1AFSpUCMUOHjyoLl26qHXr1nr00UdDfzm59tpr9fLLL2vw4MEaNmyYli9frmeffVbz5s3Tl19+qdjYWEnSXXfdpfvvv1/dunVTt27dNHfuXHXu3Fk5OTn59t+xY0dJh0/ABTFx4kR16NBBFStWVL9+/XTHHXfo3//+ty677LJ8t/3iiy/0zjvvaOjQoUpJSdHTTz+tXr16adWqVWGP98jn4/zzz1f58uU1ffr0sHejj1TQ58Nl69atkg6fINasWaP77rtPCQkJ6tOnT+g2e/fuVfv27bV06VLdeOONqlmzpiZPnqxBgwZp+/bt+utf/yrp8Mnj4osv1syZMzVkyBA1bdpUH3/8sW699VatWbNGTzzxhCRpwoQJuuqqq9S8eXNdc801kqTatWsfdazwV9zmUEHGfzzlPd5zzjlHDz30kDZs2KCnnnpKX375pebNm6dy5cqpT58+uu222/Tmm2/q1ltvDct/88031blzZ6WmpkqS/vOf/6hr165q1qyZRo4cqRIlSmjcuHE6//zz9fnnn6t58+Zh+Zdddpnq1q2rBx98MOyiDydGcZ8ff+Q37oKedyNx3XXX6a233tKNN96o008/XVu2bNEXX3yhX375RWeddZYk5kFxxpxgTiA6xWXuVK9eXTNmzNB//vOf0B8LXJYuXarevXtryJAhGjhwoF566SUNGjRIzZo1U1ZWljN3y5Yt6tq1q/r166fLL79cmZmZat++vW666SYlJyfrzjvvlCRlZmYedRwnDa8IGDdunCfJmzFjhrdp0ybv999/99544w2vQoUKXunSpb3Vq1d7nud5AwcO9CR5d9xxR1j+559/7knyJk6cGBb/6KOPwuIbN2704uLivO7du3uHDh0K3W7EiBGeJG/gwIFh+dWrV/eqV69eoMewYcMGr1SpUt4LL7wQirVq1crr0aNHvttK8uLi4rylS5eGYj/88IMnyXvmmWdCsZEjR3qSvE2bNnm//PKLV7lyZe+cc87xtm7dGnZ/AwcODBtnQZ8PS95+j/xXrlw576OPPgq77ZNPPulJ8l599dVQLCcnxzv33HO95ORkb8eOHZ7ned57773nSfLuv//+sPzevXt7MTExYc9FUlJSvt8F3Ir7HJo5c6YnyXvppZe8TZs2eWvXrvWmTJni1ahRw4uJifHmzJkTGr/f/eUds0fu+4/jydvHzJkzPc87fJxmZGR4jRo18vbu3Ru63YcffuhJ8u66665Q7Nxzz/WaNWsWdv/ffvutJ8l75ZVXPM/zvEOHDnl169b1unTpEvbc7Nmzx6tZs6Z3wQUX5Btv//79j/rc4NgV9/nxR4888ognyVu+fHkoZo27oOfd5cuXe5K8cePG5dufJG/kyJGhn8uWLevdcMMN5viYB8UDc4I5gegU97mzcOFCr3Tp0p4kr2nTpt5f//pX77333vN2796d77bVq1f3JHmfffZZKLZx40YvPj7eu+WWW0KxI6+vPM/z2rVr50nynnvuuXz3m5WV5bVr1+6oYz0ZFamPmnfq1Enp6ek67bTT1K9fPyUnJ+vdd99VlSpVwm53/fXXh/08efJklS1bVhdccIE2b94c+tesWTMlJydr5syZkqQZM2YoJydHN910U9jHUm+++Wbf8axYsaLAf3V94403VKJECfXq1SsU69+/v6ZNm+b78e5OnTqFvZN7xhlnqEyZMr4VnBcuXKh27dqpRo0amjFjRujdNUtBn4+jefvttzV9+nR98sknGjdunOrVq6devXrpq6++Ct1m6tSpqlixYtj3WWJjYzVs2DDt2rVLs2fPDt2uZMmSGjZsWNg+brnlFnmep2nTphVoTHArznNIkq688kqlp6ercuXK6t69e+hrB3nfozqevvvuO23cuFFDhw4N+25S9+7d1aBBA02ZMiUU69u3r77//vuwj3FNmjRJ8fHx6tGjhyRp/vz5WrJkiQYMGKAtW7aEnsPdu3erY8eO+uyzz/J1OrjuuuuO++OCrbjPj6M5ctxBnHfLlSunb775RmvXrvXdzjwoXpgThzEnEKniOneysrI0f/58XX755VqxYoWeeuop9ezZU5mZmXrhhRfy3f70009XmzZtQj+np6erfv36Beo4Ex8fr8GDBx/1dqeSIvVR83/961+qV6+eSpUqpczMTNWvX18lSoT/baBUqVKqWrVqWGzJkiXKzs42v5uQVywg7zuadevWDduenp5+1MXs0bz66qtq3ry5tmzZoi1btkiSzjzzTOXk5Gjy5Mmhj03nqVatWr77SE1N9V2kX3TRRcrMzNTHH3+s5OTko46loM/H0bRt2zbs4+y9e/dW3bp1ddNNN+n777+XdPg5rVu3br7fU8OGDUPb8/6vXLmyUlJSnLfDsSnOc0g6/LGqNm3aqGTJkkpLS1PDhg3z1RQ4XvIeS/369fNta9Cggb744ovQz5dddpn+/ve/a9KkSRoxYoQ8z9PkyZPVtWtXlSlTRtLh51CSBg4caO4zOzs77HmqWbPmcXksKJjiPj9c/MYdxHn34Ycf1sCBA3XaaaepWbNm6tatm6644grVqlVLEvOguGFOHMacQKSK89ypV6+eJkyYoNzcXP3888/68MMP9fDDD+uaa65RzZo1wwpDR7JeOVKVKlUKXOn8VFGkFt7Nmzc/6jtb8fHx+Q7sQ4cOKSMjQxMnTvTNSU9PP25j9LNkyRLNmTNHUv4JIh3+7veRC++SJUv63pfn812eXr16afz48Zo4caKuvfbao44nqOcjOTlZLVq00Pvvv18kKjMiv+I6h/I0btzY2QnAKqCWm5sb1JAkSZUrV1abNm305ptvasSIEfr666+1atUqjRo1KnSbvHcsHnnkEbPq/5F/OCtdunRgY0Z+xX1+uPiNu6AimVd9+vRRmzZt9O677+qTTz7RI488olGjRumdd95R165dmQfFDHPCH3MCR3MyzJ2SJUuqcePGaty4sc4991x16NAhX0emSNYrR+I4zq9ILbyjVbt2bc2YMUPnnXee85ecV8FvyZIlob9EStKmTZuOqdr3xIkTFRsbqwkTJuQ7QL/44gs9/fTTWrVqle9fjQrikUceUalSpUKF2AYMGOC8fUGfj2gcPHhQkrRr1y4lJSWpevXqWrBggQ4dOhR2clm0aJGk/3vO84o57Ny5M+wvzUfeTrJf8BCcwp5DBZWamqrt27fni0fzLkXeY1m8eHG+AiOLFy/OV/Gzb9++Gjp0qBYvXqxJkyYpMTFRF110UWh73ldHypQpU6zbCCK/4jI//MZTkPNu3rsnR84ta15VqlRJQ4cO1dChQ7Vx40adddZZeuCBB9S1a1fmwSmCORGOOYGCKqpzJ++PCOvWrTvu932kU/k6v0h9xztaffr0UW5uru6777582w4ePBg6cXbq1EmxsbF65plnwv5S8+STT/reb0FL80+cOFFt2rRR37591bt377B/eVWQX3/99cgf2P8XExOjsWPHqnfv3ho4cKA++OAD5+0L+nxEauvWrfrqq69UsWLF0EdkunXrpvXr12vSpElh+3jmmWeUnJysdu3ahW6Xm5urZ599Nuw+n3jiCcXExKhr166hWFJSUtRjRHQKew4VVO3atZWdna0FCxaEYuvWrdO7774b8X2dffbZysjI0HPPPaf9+/eH4tOmTdMvv/yi7t27h92+V69eKlmypF5//XVNnjxZF154YdinPpo1a6batWvr0Ucf1a5du/Ltb9OmTRGPEUVDcZkfRyroebdMmTJKS0vTZ599Fna70aNHh/2cm5ur7OzssFhGRoYqV64cmkPMg1MDc+Iw5gQiVdhz5/PPP9eBAwfyxadOnSrJ/+t3x9upfJ1/Urzj3a5dO1177bV66KGHNH/+fHXu3FmxsbFasmSJJk+erKeeekq9e/dWenq6/vGPf+ihhx7ShRdeqG7dumnevHmaNm2ab2uugpTm/+abb0KttPxUqVJFZ511liZOnKjbb7896sdYokQJvfrqq+rZs6f69OmjqVOnmm0ACvp8HM1bb72l5ORkeZ6ntWvX6sUXX9S2bdv03HPPhf5adc011+j555/XoEGD9P3336tGjRp666239OWXX+rJJ58M/UX5oosuUocOHXTnnXdqxYoVatKkiT755BO9//77uvnmm8MKzTVr1kwzZszQ448/rsqVK6tmzZpq0aJF1M8djq4w51Ak+vXrp9tvv12XXHKJhg0bpj179mjMmDGqV69evp6pRxMbG6tRo0Zp8ODBateunfr37x9qJ1ajRg397W9/C7t9RkaGOnTooMcff1w7d+5U3759w7aXKFFC//u//6uuXbsqKytLgwcPVpUqVbRmzRrNnDlTZcqU0b///e9jfg5w4hWX+XGkSM67V111lf75z3/qqquu0tlnn63PPvtMv/76a9j97dy5U1WrVlXv3r3VpEkTJScna8aMGZozZ44ee+wxScyDUwVz4jDmBCJV2HNn1KhR+v7773XppZfqjDPOkCTNnTtXr7zyisqXL28WbzuemjVrpjFjxuj+++9XnTp1lJGRUaDWZieFwimmHi6vNH9eyyDLwIEDvaSkJHP72LFjvWbNmnmlS5f2UlJSvMaNG3u33Xabt3bt2tBtcnNzvXvuucerVKmSV7p0aa99+/bewoUL87Ue8ryClea/6aabPEnesmXLzNvcfffdniTvhx9+8DzvcCsKv9YTR47hj+3E8uzZs8dr166dl5yc7H399dee59ktlgryfPjxayeWlJTknXvuud6bb76Z7/YbNmzwBg8e7KWlpXlxcXFe48aNfVtw7Ny50/vb3/7mVa5c2YuNjfXq1q3rPfLII2FtEjzP8xYtWuS1bds21O6A1mJHV5znkOf9XyuKyZMnH/W2n3zyideoUSMvLi7Oq1+/vvfqq69G1U4sz6RJk7wzzzzTi4+P98qXL+/9+c9/DrUDOdILL7zgSfJSUlLCWpD90bx587xLL73Uq1ChghcfH+9Vr17d69Onj/fpp5+GbuM3txGc4j4//shqnWSNu6Dn3T179nhDhgzxypYt66WkpHh9+vTxNm7cGNY6af/+/d6tt97qNWnSxEtJSfGSkpK8Jk2aeKNHj863X+ZB0cacYE4gOsV97nz55ZfeDTfc4DVq1MgrW7asFxsb61WrVs0bNGhQvrVM9erVve7du+e7j3bt2oW1A7PaiWVlZfmOYf369V737t29lJQUT9Ip1VosxvMK8O14AAAAAAAQlZPiO94AAAAAABRVLLwBAAAAAAgQC28AAAAAAALEwhsAAAAAgACx8AYAAAAAIEAsvAEAAAAACBALbwAAAAAAAlSqoDeMiYkJchxAoTiWNvbMCSklJcU33rx5czPn008/DWo4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ixYtzByrWvbTTz8d8f6jkZGRYW7r0qWLb3zHjh1BDafISktLO273FRsba26zqm+6qpq7qkhaZs+eHXHOxx9/7BuvVauWmbNlyxbfeLdu3cycmTNn+sZdldCtiueu58aqiuuq8ouipWzZsr5x63crRVfV/J133olsYEdhVdPNzc2N+L5cVXFxanF1Q7EqnruudazrE1dV5Xnz5vnGXVX+rU4frtdKaw65OhrgxKtZs6a5zXrddlXAT0pK8o27jq/y5cv7xl3Hl6s6vsW63nCd113zLz09PeIxWM+da85a54adO3dGvB/Xa691LFDVHAAAAACAkwwLbwAAAAAAAsTCGwAAAACAALHwBgAAAAAgQCy8AQAAAAAIEAtvAAAAAAACRDuxQmK1o5Dskvhnn322mdOwYUPf+IYNG8ycunXr+sbfffddM2fr1q2+caslhyStXLnSN16hQgUzp0yZMr7x1atXmzknq6pVq0acExMTE3HOnj17fOOuFldWOwrX/uvXr+8b/+c//2nm1K5d29xm+eWXX3zjDRo0MHOqV6/uGx86dKiZc+655/rGrbkiSTk5Ob5xV+s2FC1WO0RrHknutjOW119/PeKc/fv3m9us9jZW+z0XqxUMTj2u4956nbBaOrm4cubPnx/x/VnXLvv27TNzrPlFO7GipVq1auY26/cbTYtU136s61/rGkCy1weudYM1L1wttlzjtvbluj9rXrjaiVWqVMk37jqfWPPMNf/q1atnbjvReMcbAAAAAIAAsfAGAAAAACBALLwBAAAAAAgQC28AAAAAAALEwhsAAAAAgABR1TxgVoVEV2XApKQk3/hll11m5ljVBBMSEsyclJQU37irIrX1eFw5WVlZvvHff//dzNm2bZtv3FUd8WSVnp4ecY5VRdZVFdPa5qoi+8ADD/jGY2NjzZzOnTv7xps0aWLmNGrUyDduHcOSXb3cVT190qRJvvGmTZuaORbXc239flzPG4oWq6K3a75Ec/6aOXNmxDn//e9/zW1WFX7X8WqJphI6Tk7WOU2yqw27qvxb26KphL53715zW1xcnG989+7dZo51/ZabmxvZwBCoypUrm9us39WOHTvMnPj4eN+41YVHsueF67XAGpvrHG3NF2vMR7u/nTt3+sZTU1PNHKtSvKvrkfV8p6WlmTnbt2/3jbsq0kdzDRcU3vEGAAAAACBA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)  
测试集最终准确率：0.8275
# 实验二： 在鸢尾花数据集上测试
## 1.数据集介绍
### 鸢尾花数据集（Iris Dataset）介绍  

鸢尾花数据集是机器学习、统计学领域中最经典的入门级数据集之一，广泛用于分类算法的教学与验证，因简单直观、特征明确而成为初学者的首选案例。 
**来源**：由英国统计学家罗纳德·费希尔（Ronald Fisher）在1936年的论文《The use of multiple measurements in taxonomic problems》中首次提出，用于展示线性判别分析（LDA）的效果。数据组成如下：  
**样本数量**：共150个样本，无缺失值。  
**类别划分**：包含3种鸢尾花品种，每类50个样本，类别平衡：
  1. 山鸢尾（Iris setosa）  
  2. 变色鸢尾（Iris versicolor）  
  3. 维吉尼亚鸢尾（Iris virginica）  
  
**特征维度**：每个样本包含4个数值型特征（单位：厘米），描述花的形态特征：  
  1. 萼片长度（sepal length）  
  2. 萼片宽度（sepal width）  
  3. 花瓣长度（petal length）  
  4. 花瓣宽度（petal width）
  
 <div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/iris.png" alt="图2.3 Softmax总流程图 width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图2.3 Softmax总流程图</div>
</div>

  这个数据集简单易处理：特征均为连续数值，无缺失或异常值，无需复杂预处理。  
  
  ## 2.算法设计
  算法逻辑同上面的实验这里不再赘述，鸢尾花数据集是一个csv文件，无非是更改数据的输入处理方式，我们关心算法在此数据集上的表现。
  ## 3.代码运行
  训练集：(120, 4)，验证集：(30, 4)   
  类别数：3（3类鸢尾花）  
   Epoch 50/500 | Loss: 0.2113 | Val Acc: 0.9000   
   Epoch 100/500 | Loss: 0.1598 | Val Acc: 0.9333  
   Epoch 150/500 | Loss: 0.1375 | Val Acc: 0.9333   
   Epoch 200/500 | Loss: 0.1234 | Val Acc: 0.9333   
   Epoch 250/500 | Loss: 0.1051 | Val Acc: 0.9667  
   Epoch 300/500 | Loss: 0.1150 | Val Acc: 0.9667   
   Epoch 350/500 | Loss: 0.1122 | Val Acc: 0.9667  
   Epoch 400/500 | Loss: 0.1067 | Val Acc: 0.9667  
   Epoch 450/500 | Loss: 0.1059 | Val Acc: 0.9667  
   Epoch 500/500 | Loss: 0.1071 | Val Acc: 0.9667   
   最终验证集准确率：0.9667


## 小结

Fashion-MNIST作为图像数据集，其784维的像素特征空间存在高度冗余性和非线性关系。每个像素点与类别标签的关联是微弱且复杂的，线性模型难以捕捉图像中局部形状、边缘和纹理等关键判别特征。尽管Softmax分类器能够学习到一定的全局模式，但对于需要层次化特征提取的视觉任务，单一的线性变换层显然存在局限性。训练过程中损失值从0.64逐步下降至0.50，d表明模型确实在学习有效的特征组合，但收敛速度相对较慢，验证了线性模型在复杂视觉模式识别中的不足。

相比之下，鸢尾花数据集的特征工程更为理想。四个形态测量特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度）本身就是经过生物学家精心选择的判别性特征，这些特征与花品种之间存在较强的线性可分性。数据集仅包含150个样本，特征维度低，避免了维度灾难问题，使得Softmax分类器能够充分利用其数学特性，快速找到最优的线性决策边界。训练过程中仅需约100个epoch就达到93%以上的准确率，随后稳定在96.67%，体现了算法在适当地问题设置下的高效性。

尽管Softmax分类器在理论上有诸多优点，但其线性本质也带来了固有的局限性。线性决策边界无法处理复杂的非线性分类问题，这在Fashion-MNIST的表现中已经得到验证。当类别间存在复杂的交互特征或需要层次化表示时，单一的线性变换层显得力不从心。另外，对于特征尺度敏感的问题，虽然我们在预处理中进行了标准化，但在实际应用中，不同特征的重要性可能差异很大，线性模型难以自动学习这种重要性权重。此外，Softmax假设所有类别是互斥的，这在多标签分类场景中成为限制。

进一步的改进方向主要集中在深度学习的框架内。通过堆叠多个非线性变换层，可以构建深度神经网络，其中Softmax作为最终的输出层。卷积神经网络（CNN）能够自动学习图像的局部特征，循环神经网络（RNN）适合处理序列数据，而注意力机制则能动态调整不同特征的重要性。在这些架构中，Softmax保持了其作为概率输出层的核心地位，但特征提取的工作交给了前面的深层网络。

Softmax分类器及其变体在现实世界中有着广泛的应用。在自然语言处理领域，它常用于文本分类、情感分析和命名实体识别任务中，作为模型最后一层将隐藏状态映射到词汇表概率分布。在计算机视觉中，尽管现代架构使用更复杂的特征提取器，但Softmax仍然是多类别图像分类的标准输出层。
在推荐系统中，Softmax可以用于预测用户可能交互的下一个物品；在语音识别中，它将声学特征映射到音素或字符的概率分布......

近年来，基于Softmax的改进变体也不断涌现。如Margin-based Softmax通过引入间隔项增强类间分离性；Focal Loss针对类别不平衡问题调整损失权重；Knowledge Distillation中使用带温度的Softmax实现知识从复杂模型到简单模型的转移等等...虽然在复杂任务中需要与更强大的特征学习器结合使用，但其将原始分数转化为可解释的概率分布的核心思想将继续指导着分类算法的发展方向。