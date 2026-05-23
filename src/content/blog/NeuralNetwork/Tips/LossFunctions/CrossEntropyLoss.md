---
title: CrossEntropyLoss：从信息论到代码实现
date: 2026-05-23
category: NeuralNetwork/Tips/LossFunctions
tags:
  - 损失函数
  - 交叉熵
  - Softmax
description: 深入理解交叉熵损失的数学原理、参数细节和常见陷阱——不只是 import torch.nn as nn。
image: https://img.yumeko.site/file/blog/CrossEntropyLoss.png
status: published
---

## 1. 为什么叫"交叉熵"？

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

**直觉**：交叉熵越小，两个分布越接近。当模型对正确类别输出概率为 1 时，交叉熵为 0——这是理论上的最优值。对于回归任务，使用 [[NeuralNetwork/Tips/LossFunctions/MSELoss|MSE]] 这类基于距离的损失函数。更多损失函数的分类和选择，参见 [[NeuralNetwork/Theory/LossFunctions|损失函数概述]]。

## 2. 内部两步拆解

PyTorch 的 `nn.CrossEntropyLoss()` 内部等价于 `LogSoftmax + NLLLoss`：

### 第 1 步：Softmax

将任意实数 logits 转为概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

关键性质：
- 输出范围 $(0, 1)$，所有类别概率之和为 1
- **保持排序**：如果 $z_a > z_b$，则 $\text{softmax}(z_a) > \text{softmax}(z_b)$
- 对整体加减常数不变（数值稳定性依赖于这个性质）

### 第 2 步：负对数似然

提取正确类别的预测概率，取负对数：

$$
\text{NLL} = -\log(\text{softmax}(z_y))
$$

### 完整计算示例

考虑一个 3 分类问题，某样本 logits 为 $[2.0, 1.0, 0.1]$，真实标签为类别 0：

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

**核心洞察**：损失函数对"自信地错误"惩罚极重——概率从 0.01 降到 0.001 时，loss 从 4.6 跳到 6.9。

## 3. 数值稳定性

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

## 4. 参数详解

### weight：类别权重

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

### label_smoothing

防止模型过度自信，提升泛化能力：

$$
y_i^{\text{smooth}} = (1 - \alpha) \cdot y_i^{\text{one-hot}} + \frac{\alpha}{K}
$$

$\alpha=0.1$ 时，正确类别标签从 1.0 变为 0.9，其余类别从 0 变为 $0.1/K$。

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**为什么有效？**
- 传统 one-hot 鼓励模型输出极度尖锐的概率分布（容易 [[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]]）
- Label Smoothing 让模型"谦虚"一点，迫使其保持对错误类别的少量概率质量
- 在 ImageNet 上，$\alpha=0.1$ 通常带来 0.2-0.3% 的 Top-1 提升

### ignore_index

忽略特定标签（如分割任务中的背景类或填充类）：

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

被忽略的样本不参与 loss 计算和梯度回传。

### reduction

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

## 5. CNN 项目中的使用

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

## 6. 常见陷阱

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
