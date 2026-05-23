---
title: MSELoss：回归任务的损失函数详解
date: 2026-05-23
category: NeuralNetwork/Tips/LossFunctions
tags:
  - 深度学习
  - 基础
description: 深入理解均方误差损失的数学性质、梯度行为、与 MAE 的选择逻辑，以及 Huber Loss 的折中方案。
image: https://img.yumeko.site/file/blog/MSELoss.png
status: published
---

## 1. 公式与直觉

均方误差（Mean Squared Error）的定义：

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值。

**直觉**：MSE 衡量预测值与真实值的"平均平方距离"。平方放大了大误差——误差为 2 的样本对 loss 的贡献是误差为 1 的样本的 4 倍。关于损失函数的分类体系，参见 [[NeuralNetwork/Theory/LossFunctions|损失函数概述]]。

## 2. 梯度行为

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

## 3. reduction 参数

```python
# 默认：所有样本平均
criterion = nn.MSELoss(reduction='mean')

# 求和（梯度会放大 batch_size 倍，需调整学习率）
criterion = nn.MSELoss(reduction='sum')

# 逐样本输出（用于样本加权或自定义聚合）
criterion = nn.MSELoss(reduction='none')
perSampleLoss = criterion(predictions, targets)  # shape: (batch_size,)
```

## 4. MSE vs MAE

| | MSE | MAE |
| --- | --- | --- |
| 公式 | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |
| 梯度大小 | 正比于误差 | 恒为 $\pm 1$ |
| 对大误差 | 极度敏感（平方放大） | 线性敏感 |
| 收敛时 | 梯度渐小，自动精细 | 梯度恒定，可能震荡 |
| 最优性条件 | 误差服从高斯分布 | 误差服从拉普拉斯分布 |
| 异常值 | 被异常值主导 | 对异常值鲁棒 |

### 选择判断

**选 MSE 当**：
- 异常值很少或已被清理
- 误差分布近似正态
- 需要收敛后期自动减速

**选 MAE 当**：
- 数据中有不易清理的异常值
- 不希望少数大误差样本主导训练
- 需要所有误差被平等对待

## 5. Huber Loss：MSE 与 MAE 的折中

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

### 三者对比

```
误差为 0.5 时:  MSE=0.25,  MAE=0.5,   Huber≈0.125
误差为 3.0 时:  MSE=9.0,   MAE=3.0,   Huber≈2.5
误差为 10.0 时: MSE=100,   MAE=10.0,  Huber≈9.5
```

Huber 对大误差的惩罚远小于 MSE，缓解了异常值问题；对小误差保持了平滑梯度。

## 6. 使用时注意

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

## 7. CNN 项目中的使用

```python
from cnnlib.training.loss import createLoss

lossFn = createLoss("mse")
```

CNN 分类任务以 [[NeuralNetwork/Tips/LossFunctions/CrossEntropyLoss|交叉熵]] 为主，MSE 用于回归类任务或作为对比实验的 baseline。
