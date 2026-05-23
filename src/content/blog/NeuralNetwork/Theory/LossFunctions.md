---
title: 损失函数详解：衡量预测与真实值的差距
date: 2026-05-07
category: NeuralNetwork/Theory
tags:
  - 基础
  - 深度学习
description: 从 Softmax 到 CrossEntropyLoss，用数值例子理解损失函数如何驱动模型学习。
image: https://img.yumeko.site/file/blog/LossFunctions.png
status: published
---

## 1. 损失函数是什么？

[[NeuralNetwork/Tips/LossFunctions/CrossEntropyLoss|交叉熵]] 损失函数是一个"不满意度"的度量——模型预测得越好，损失值越低；预测得越差，损失值越高。训练的目标就是最小化这个损失值。

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(f_\theta(x), y)
$$

## 2. CrossEntropyLoss 详解

CrossEntropyLoss 是多分类问题最标准的损失函数。在 PyTorch 中，它内部包含两步操作。

### 第 1 步：Softmax —— 把 logits 转为概率分布

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

- 每个输出值被转换为 $(0, 1)$ 之间的概率
- 所有类别的概率之和 $= 1$
- Softmax 保持相对大小关系：logit 越大，概率越大

### 第 2 步：负对数似然 —— 衡量预测概率与真实标签的差距

$$
\mathcal{L} = -\log(p_{y})
$$

其中 $p_y$ 是模型预测为正确类别 $y$ 的概率。

### 数值例子

模型对一张 "3" 的图像输出 logits 为 $[0.1, 0.2, 0.05, 2.0, 0.1, 0.05, 0.1, 0.1, 0.2, 0.1]$：

Softmax 后，类别 3 的概率：

$$
p_3 = \frac{e^{2.0}}{e^{0.1} + e^{0.2} + e^{0.05} + e^{2.0} + ...} \approx 0.72
$$

Loss $= -\log(0.72) \approx 0.328$

如果模型对正确类别的概率是 0.9，Loss $= -\log(0.9) \approx 0.105$（很好）

如果模型对正确类别的概率是 0.01，Loss $= -\log(0.01) \approx 4.605$（很差）

**直觉**：损失函数对"自信但错误"的惩罚远大于"不自信但正确"。

![CrossEntropyLoss.png](https://img.yumeko.site/file/articles/CNN/CrossEntropyLoss.png)

## 3. PyTorch 代码

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

## 4. 带权重的 CrossEntropyLoss

当类别不平衡时，可以给少数类更大的权重：

```python
# 假设类别 3 的样本很少，给它加倍权重
classWeights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=classWeights)
```

## 5. Label Smoothing

Label Smoothing 是一种正则化技术，防止模型对预测过于自信（过于极端的概率分布）：

$$
y_i^{\text{smooth}} = (1 - \alpha) \cdot y_i + \frac{\alpha}{K}
$$

其中 $\alpha$ 是平滑参数，$K$ 是类别数。

```python
# PyTorch 中直接使用 label_smoothing 参数
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**为什么有效？** 传统的 one-hot 标签要求模型输出对正确类别概率为 1、其他为 0。这鼓励模型输出极端的概率分布，容易过拟合。Label Smoothing 让标签"软"一点，鼓励模型在学习类别间关系的同时保持一定的"谦虚"。

## 6. MSELoss —— 回归任务的损失函数

对于回归任务（预测连续值），使用 [[NeuralNetwork/Tips/LossFunctions/MSELoss|均方误差]]：

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

```python
criterion = nn.MSELoss()

# 适用于：房价预测、温度预测、坐标回归等
predictions = model(images)      # (batch_size, 1)
loss = criterion(predictions, targets)  # targets 也是 (batch_size, 1)
```

**MSE vs MAE**：

| | MSE | MAE |
| --- | --- | --- |
| 对大误差的敏感度 | 高（平方放大） | 低（线性） |
| 梯度稳定性 | 误差大时梯度也大 | 梯度恒为 ±1 |
| 适用场景 | 异常值不多时 | 对异常值鲁棒 |

## 7. 二分类：BCEWithLogitsLoss

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

## 8. 损失函数选择速查

| 任务类型 | 损失函数 | 输出层 |
| --- | --- | --- |
| 多分类 | `CrossEntropyLoss` | 不需要 Softmax |
| 二分类 | `BCEWithLogitsLoss` | 不需要 Sigmoid |
| 多标签分类 | `BCEWithLogitsLoss` | 不需要 Sigmoid |
| 回归 | `MSELoss` 或 `L1Loss` | 线性输出 |
| 概率分布匹配 | `KLDivLoss` | LogSoftmax |
