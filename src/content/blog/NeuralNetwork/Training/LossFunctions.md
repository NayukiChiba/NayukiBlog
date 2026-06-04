---
title: 损失函数完全指南：从交叉熵到均方误差
date: 2026-05-23
category: NeuralNetwork/Training
tags:
  - 基础
  - 深度学习
description: 深入理解损失函数的数学原理、参数细节和常见陷阱——从交叉熵到均方误差，涵盖分类与回归任务的完整损失函数指南。
image: https://img.yumeko.site/file/blog/LossFunctions.png
status: published
---

## 1. 损失函数概述

损失函数是一个"不满意度"的度量——模型预测得越好，损失值越低；预测得越差，损失值越高。训练的目标就是最小化这个损失值。

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(f_\theta(x), y)
$$

损失函数的选择直接影响模型的学习行为和最终性能。分类任务和回归任务需要不同类型的损失函数：分类任务关心概率分布之间的差异，回归任务关心连续值之间的距离。本文将系统介绍最常用的两类损失函数——交叉熵损失和均方误差损失。

## 2. 交叉熵损失

### 2.1 为什么叫"交叉熵"？

交叉熵来自信息论，衡量两个概率分布之间的差异：

$$
H(p, q) = -\sum_{i} p(i) \cdot \log q(i)
$$

其中：
- $p$ 是真实分布（one-hot 标签：正确类别概率为 1，其余为 0）
- $q$ 是模型预测的分布（Softmax 输出）

当 $p$ 是 one-hot 时，求和只剩正确类别那一项，就退化成了负对数似然：

$$
\mathcal{L} = -\log q(y)
$$

**直觉**：交叉熵越小，两个分布越接近。当模型对正确类别输出概率为 1 时，交叉熵为 0——这是理论上的最优值。对于回归任务，使用 [[NeuralNetwork/Training/LossFunctions|MSE]] 这类基于距离的损失函数。

### 2.2 内部两步拆解

PyTorch 的 `nn.CrossEntropyLoss()` 内部等价于 `LogSoftmax + NLLLoss`。

**第 1 步：Softmax —— 把 logits 转为概率分布**

将任意实数 logits 转为概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

关键性质：
- 输出范围 $(0, 1)$，所有类别概率之和为 1
- **保持排序**：如果 $z_a > z_b$，则 $\text{softmax}(z_a) > \text{softmax}(z_b)$
- 对整体加减常数不变（数值稳定性依赖于这个性质）

**第 2 步：负对数似然 —— 衡量预测概率与真实标签的差距**

提取正确类别的预测概率，取负对数：

$$
\text{NLL} = -\log(\text{softmax}(z_y))
$$

其中 $z_y$ 是正确类别对应的 logit。

### 2.3 数值例子

**3 分类示例**：某样本 logits 为 $[2.0, 1.0, 0.1]$，真实标签为类别 0：

```
Softmax 分母 = e^2.0 + e^1.0 + e^0.1 = 7.389 + 2.718 + 1.105 = 11.212

类别 0 概率 = 7.389 / 11.212 = 0.659
类别 1 概率 = 2.718 / 11.212 = 0.242
类别 2 概率 = 1.105 / 11.212 = 0.099

Loss = -log(0.659) = 0.417
```

如果模型对类别 0 的 logit 是 5.0（非常确信）：

```
Softmax: 类别 0 概率 ≈ 0.982
Loss = -log(0.982) = 0.018 （接近于 0）
```

如果模型对类别 0 的 logit 是 -1.0（预测错误方向）：

```
Softmax: 类别 0 概率 ≈ 0.090
Loss = -log(0.090) = 2.408 （较大的惩罚）
```

**10 分类示例（MNIST 风格）**：模型对一张 "3" 的图像输出 logits 为 $[0.1, 0.2, 0.05, 2.0, 0.1, 0.05, 0.1, 0.1, 0.2, 0.1]$。

Softmax 后，类别 3 的概率：

$$
p_3 = \frac{e^{2.0}}{e^{0.1} + e^{0.2} + e^{0.05} + e^{2.0} + ...} \approx 0.72
$$

Loss $= -\log(0.72) \approx 0.328$

如果模型对正确类别的概率是 0.9，Loss $= -\log(0.9) \approx 0.105$（很好）

如果模型对正确类别的概率是 0.01，Loss $= -\log(0.01) \approx 4.605$（很差）

**核心洞察**：损失函数对"自信地错误"惩罚极重——概率从 0.01 降到 0.001 时，loss 从 4.6 跳到 6.9。损失函数对"自信但错误"的惩罚远大于"不自信但正确"。

![CrossEntropyLoss.png](https://img.yumeko.site/file/articles/CNN/CrossEntropyLoss.png)

### 2.4 数值稳定性

PyTorch 的 CrossEntropyLoss 内部使用 log-sum-exp 技巧防止溢出：

$$
\log\left(\sum e^{z_j}\right) = a + \log\left(\sum e^{z_j - a}\right)
$$

其中 $a = \max(z_j)$。这避免了 $e^{100}$ 这样的大数直接参与计算。

**这就是为什么不要手动 Softmax 再传 CrossEntropyLoss**。如果你做了手动 Softmax，值域变成 $(0,1)$，再取 log 时可能遇到 $\log(0)$。

```python
# 正确
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # logits 是原始输出

# 错误
probs = F.softmax(logits, dim=1)
loss = criterion(probs, labels)  # 两次 Softmax，结果不对
```

### 2.5 参数详解

#### weight：类别权重

处理类别不平衡。给少数类更大的权重让模型更关注它们：

```python
# 假设类别 2 样本极少，给它 3 倍权重
classWeights = torch.tensor([1.0, 1.0, 3.0, 1.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=classWeights)
```

数学形式变为：

$$
\mathcal{L} = -w_y \cdot \log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)
$$

权重选择建议：
- 如果某个类别样本数是平均的 1/5，用 weight=5 补偿
- 具体数值通常通过验证集调优，不要死套公式

#### label_smoothing

Label Smoothing 是一种正则化技术，防止模型对预测过于自信（过于极端的概率分布）：

$$
y_i^{\text{smooth}} = (1 - \alpha) \cdot y_i^{\text{one-hot}} + \frac{\alpha}{K}
$$

$\alpha=0.1$ 时，正确类别标签从 1.0 变为 0.9，其余类别从 0 变为 $0.1/K$。

```python
# PyTorch 中直接使用 label_smoothing 参数
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**为什么有效？**
- 传统 one-hot 鼓励模型输出极度尖锐的概率分布（容易 [[NeuralNetwork/Troubleshooting/Overfitting|过拟合]]）
- Label Smoothing 让模型"谦虚"一点，迫使其保持对错误类别的少量概率质量
- 在 ImageNet 上，$\alpha=0.1$ 通常带来 0.2-0.3% 的 Top-1 提升

#### ignore_index

忽略特定标签（如分割任务中的背景类或填充类）：

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

被忽略的样本不参与 loss 计算和梯度回传。

#### reduction

控制输出形式：

| reduction | 输出 |
| --- | --- |
| `'mean'`（默认） | 所有样本 loss 的平均值 |
| `'sum'` | 所有样本 loss 求和 |
| `'none'` | 返回每个样本的 loss 张量 |

```python
criterion = nn.CrossEntropyLoss(reduction='none')
perSampleLoss = criterion(logits, labels)  # shape: (batch_size,)
```

### 2.6 PyTorch 基础用法

```python
import torch
import torch.nn as nn

# 标准用法：直接传入 logits，不需要手动 Softmax
criterion = nn.CrossEntropyLoss()

# logits: (batch_size, num_classes) — 原始输出
# labels: (batch_size,) — 类别索引，不是 one-hot
logits = torch.randn(4, 10)       # 4 个样本，10 分类
labels = torch.tensor([3, 7, 1, 0])

loss = criterion(logits, labels)
print(f"Loss: {loss:.4f}")
```

**重要**：不要手动对 logits 做 Softmax！CrossEntropyLoss 内部已经包含了 Softmax。如果传入已经 Softmax 过的值，结果会不正确（相当于做了两次 Softmax）。

### 2.7 CNN 项目中的使用

```python
from cnnlib.training.loss import createLoss

# 标准多分类
lossFn = createLoss("cross_entropy")

# 带 Label Smoothing
lossFn = createLoss("cross_entropy", label_smoothing=0.1)

# 带类别权重
classWeights = torch.tensor([1.0, 1.0, 2.0])
lossFn = createLoss("cross_entropy", weight=classWeights)
```

### 2.8 常见陷阱

**陷阱 1：传入 one-hot 标签**

CrossEntropyLoss 要求标签是**类别索引**（shape: `(N,)`），不是 one-hot 向量（shape: `(N, K)`）。

```python
# 正确
labels = torch.tensor([3, 7, 1])          # (3,)

# 错误
labels = torch.tensor([[0,0,0,1,0,...],   # (3, 10)
                       [0,0,0,0,0,0,0,1,...],
                       [0,1,0,0,0,...]])
```

**陷阱 2：对 logits 做了 Softmax**

重复强调——CrossEntropyLoss 内部已包含 Softmax。

**陷阱 3：忘记类别不平衡**

如果验证集上少数类的准确率明显差，大概率是类别不平衡问题——加上 `weight` 参数。

### 2.9 二分类：BCEWithLogitsLoss

对于二分类任务，推荐使用 `BCEWithLogitsLoss`（内置 Sigmoid）：

```python
criterion = nn.BCEWithLogitsLoss()

# logits: (batch_size,) — 每个样本一个值
# labels: (batch_size,) — 0 或 1
logits = torch.randn(32)
labels = torch.randint(0, 2, (32,)).float()
loss = criterion(logits, labels)
```

**为什么用 BCEWithLogitsLoss 而不是 BCELoss？** BCEWithLogitsLoss 将 Sigmoid 和 BCE 合并，利用 log-sum-exp 技巧保证数值稳定性。分开计算 Sigmoid + BCELoss 可能产生 $\log(0)$ 导致数值问题。

多标签分类（一个样本可以同时属于多个类别）同样使用 `BCEWithLogitsLoss`，此时每个类别独立做二分类。

## 3. 均方误差损失

### 3.1 公式与直觉

均方误差（Mean Squared Error）的定义：

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值。

**直觉**：MSE 衡量预测值与真实值的"平均平方距离"。平方放大了大误差——误差为 2 的样本对 loss 的贡献是误差为 1 的样本的 4 倍。

### 3.2 梯度行为

MSE 的梯度与误差大小成正比：

$$
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)
$$

这意味着：
- 误差越大，梯度越大 → 大误差样本驱动更多学习
- 误差接近 0 时，梯度也接近 0 → 收敛时自动"减速"
- 这种性质让 MSE 在误差服从高斯分布时是最优的（最大似然估计等价于最小化 MSE）

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()

# 回归任务示例
predictions = torch.tensor([2.5, 1.0, 3.0, 4.5])
targets = torch.tensor([3.0, 1.5, 2.5, 4.0])

loss = criterion(predictions, targets)
# = ((3.0-2.5)² + (1.5-1.0)² + (2.5-3.0)² + (4.0-4.5)²) / 4
# = (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
```

### 3.3 reduction 参数

```python
# 默认：所有样本平均
criterion = nn.MSELoss(reduction='mean')

# 求和（梯度会放大 batch_size 倍，需调整学习率）
criterion = nn.MSELoss(reduction='sum')

# 逐样本输出（用于样本加权或自定义聚合）
criterion = nn.MSELoss(reduction='none')
perSampleLoss = criterion(predictions, targets)  # shape: (batch_size,)
```

### 3.4 MSE vs MAE

| | MSE | MAE |
| --- | --- | --- |
| 公式 | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ |
| 梯度大小 | 正比于误差 | 恒为 $\pm 1$ |
| 对大误差 | 极度敏感（平方放大） | 线性敏感 |
| 收敛时 | 梯度渐小，自动精细 | 梯度恒定，可能震荡 |
| 最优性条件 | 误差服从高斯分布 | 误差服从拉普拉斯分布 |
| 异常值 | 被异常值主导 | 对异常值鲁棒 |

**选择判断**：

选 MSE 当：
- 异常值很少或已被清理
- 误差分布近似正态
- 需要收敛后期自动减速

选 MAE 当：
- 数据中有不易清理的异常值
- 不希望少数大误差样本主导训练
- 需要所有误差被平等对待

### 3.5 Huber Loss：MSE 与 MAE 的折中

Huber Loss 在误差小时用 MSE（平滑），误差大时用 MAE（鲁棒）：

$$
\text{Huber}(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\[6pt]
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

PyTorch 中为 Smooth L1 Loss（$\delta=1$ 时等同于 Huber）：

```python
criterion = nn.SmoothL1Loss(beta=1.0)

# beta 即 δ，控制 MSE/MAE 的切换点
# |误差| ≤ beta → MSE 行为
# |误差| > beta → MAE 行为
```

**三者对比**：

```
误差为 0.5 时:  MSE=0.25,  MAE=0.5,   Huber≈0.125
误差为 3.0 时:  MSE=9.0,   MAE=3.0,   Huber≈2.5
误差为 10.0 时: MSE=100,   MAE=10.0,  Huber≈9.5
```

Huber 对大误差的惩罚远小于 MSE，缓解了异常值问题；对小误差保持了平滑梯度。

### 3.6 使用时注意

**输入形状**：MSELoss 要求 predictions 和 targets 形状完全一致：

```python
# 正确：形状匹配
predictions = model(x)       # (batch_size, 1)
targets = y                  # (batch_size, 1)
loss = criterion(predictions, targets)

# 错误：形状不匹配（不会报错但结果不对）
predictions = model(x)       # (batch_size, 1)
targets = y.squeeze()        # (batch_size,)  ← 会被广播
```

**输出层**：回归任务最后一层不用激活函数（或仅用线性层），直接输出预测值。

**数值范围**：如果 targets 的值域很大（如房价预测中几十万），loss 数值会很大——这不一定是问题，但建议对 targets 做标准化（减均值除标准差）来让训练更稳定。

### 3.7 CNN 项目中的使用

```python
from cnnlib.training.loss import createLoss

lossFn = createLoss("mse")
```

CNN 分类任务以 [[NeuralNetwork/Training/LossFunctions|交叉熵]] 为主，MSE 用于回归类任务或作为对比实验的 baseline。

## 4. 总结与选择指南

### 4.1 损失函数选择速查

| 任务类型 | 损失函数 | 输出层 |
| --- | --- | --- |
| 多分类 | `CrossEntropyLoss` | 不需要 Softmax |
| 二分类 | `BCEWithLogitsLoss` | 不需要 Sigmoid |
| 多标签分类 | `BCEWithLogitsLoss` | 不需要 Sigmoid |
| 回归 | `MSELoss` 或 `L1Loss` | 线性输出 |
| 概率分布匹配 | `KLDivLoss` | LogSoftmax |

### 4.2 关键要点

**交叉熵损失**：
- 输入 logits，内部自动做 Softmax——永远不要手动 Softmax
- 标签是类别索引（整数），不是 one-hot 向量
- 类别不平衡时使用 `weight` 参数
- 使用 `label_smoothing` 防止过拟合，提升泛化能力

**均方误差损失**：
- 对大误差敏感（平方放大），适合异常值较少的场景
- 梯度与误差成正比，收敛后期自动减速
- 如果数据有异常值，考虑使用 MAE 或 Huber Loss（SmoothL1Loss）
- 回归任务输出层不加激活函数，建议对 targets 做标准化

### 4.3 损失函数之间的关系

交叉熵损失和均方误差损失本质上是同一框架下的不同特例：它们都可以从最大似然估计的角度推导出来。交叉熵对应于分类分布（Categorical Distribution）的负对数似然，MSE 对应于高斯分布的负对数似然。理解这一点有助于在新的任务中选择或设计合适的损失函数。
