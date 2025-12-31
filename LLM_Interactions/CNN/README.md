
## 与LLM的协作过程展示(手搓CNN)

#### 了解算法底层原理（从原理到公式再到模块）
为了实现“手搓 CNN”，第一步是把 CNN 的核心组件（卷积/激活/池化/全连接/损失）理解清楚，并建立“从输入到输出”的完整链路：输入维度如何变化？每一层做了什么？反向传播要更新哪些参数？这些决定了我们能不能真正写出可训练的 CNN。

**提问：**
我们想请你指导我们用纯 Python / NumPy 手搓实现卷积神经网络（包含前向、反向传播、参数更新），并以 MNIST 或 CIFAR-10 为例给出完整思路；同时解释 CNN 的整体流程与各模块作用（必要时给出数学公式）。  
(目的是建立对 CNN 从原理到实现的完整认知，避免只会“调包”。)

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="文心一言_20251231 (4).png" alt="问题1" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">问题1</div>
</div>


#### 分析关键超参数与“尺寸变化”细节（stride / padding / kernel）
在开始写代码前，我们需要把 CNN 最容易写错的部分一次性搞清楚：卷积核大小、步幅 stride、填充 padding 如何影响输出尺寸？不同数据集（MNIST 单通道、CIFAR-10 三通道）设置有什么差异？这些细节会直接决定网络能否跑通、是否会 shape mismatch。

**提问：**
请分析卷积层的核心参数与意义：每个卷积核在做什么？stride / padding 的作用是什么？输出尺寸如何计算？并分别给出 MNIST 与 CIFAR-10 的典型设置建议（包含 stride、padding 的选择）。  
(目的是把“算尺寸 + 选参数”的规则一次性整理清楚，避免实现时反复踩坑。)

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="文心一言_20251231 (3).png" alt="问题2" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">问题2</div>
</div>


#### 请求可运行的代码框架（数据加载 + 基础模块 + 模型骨架）
理论清楚后进入工程实现：先要有一个能跑通的“最小闭环”，包括数据集加载、基础网络模块（Conv/BN/ReLU/Pooling）、以及一个用于 CIFAR-10 或 MNIST 的 baseline CNN。后续我们再逐步“拆包解构”成纯 NumPy 版本。

**提问：**
请给出一个清晰、可运行的 CNN 代码框架（可用 PyTorch 先搭好结构），包括：  
1）MNIST / CIFAR-10 的数据加载与预处理；  
2）卷积层的 stride / padding 设置建议；  
3）基础模块（如 BatchNorm、Pooling、GAP）的接入方式；  
4）一个完整的 CIFAR-10 CNN baseline 网络结构示例。  
(目的是先拿到可运行骨架，再逐步手搓替换成 NumPy 实现。)

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="文心一言_20251231 (2).png" alt="问题3" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">问题3</div>
</div>


#### 请求模型评估与训练调参建议（指标 + 可视化 + 数据增强）
模型能跑通并不代表“真的学会了”。我们需要科学评估：准确率怎么统计？混淆矩阵怎么看？错误样本怎么可视化？训练中如何做数据增强、正则化、学习率策略、损失函数选择，来提升最终效果并避免过拟合。

**提问：**
对于训练好的 CNN，请给出系统的评估与调参建议：  
- 如何计算 accuracy / class-wise accuracy；如何绘制混淆矩阵；如何可视化预测结果；  
- 训练阶段如何做 Data Augmentation、选择优化器与学习率策略；  
- 可选的损失改进（如 label smoothing / focal loss）适用场景；  
- 常见问题排查（过拟合/欠拟合/梯度不稳定）。  
(目的是把“训练—评估—改进”的闭环建立起来。)

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="文心一言_20251231 (1).png" alt="问题4" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">问题4</div>
</div>


**一些其他的提问：**

在“手搓 CNN”的过程中，我们还会遇到很多实现细节问题：例如训练循环怎么写更稳定、如何组织代码结构、如何定位维度错误、如何对比不同网络配置等。通过这些补充提问，我们能把 LLM 的建议和自己的调试实践结合起来，让实现过程更贴近真实工程流程。

eg. 让 LLM 给出更完整的训练脚本/工程化实现，并作为我们“拆包解构”的参考起点（之后再逐步替换为纯 NumPy 版本）。

<div align="center">
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="文心一言_20251231.png" alt="其他细节问题" width="600">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">其他细节问题</div>
</div>
