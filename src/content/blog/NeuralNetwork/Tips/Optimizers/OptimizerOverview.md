---
title: 优化器概述：从梯度下降到 Adam
date: 2026-05-23
category: NeuralNetwork/Tips/Optimizers
tags:
  - 优化器
  - 基础
description: 理解优化器的作用、梯度下降的三种变体，以及 PyTorch 中 optimizer 的使用模式。
image: https://img.yumeko.site/file/blog/Optimizer.png
status: published
---

## 1. 优化器是什么？

优化器的任务：根据损失函数对模型参数的梯度，调整参数使损失越来越小。

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

其中 $\theta$ 是参数，$\eta$ 是学习率，$g_t = \nabla_\theta \mathcal{L}$ 是梯度。

## 2. 梯度下降的三种变体

| 变体 | 每次更新使用的数据 | 特点 |
| --- | --- | --- |
| BGD（批量梯度下降） | 全部训练数据 | 梯度最准，但计算代价太高 |
| SGD（随机梯度下降） | 单个样本 | 更新快但噪声大 |
| Mini-batch SGD | 一个 batch | 平衡速度和稳定性，实际使用最多 |

深度学习中的 "SGD" 通常指 Mini-batch SGD。

![TODO: 梯度下降3D示意图，损失曲面上的小球沿梯度方向逐步滚向极小值点]

## 3. PyTorch 使用模式

```python
# 1. 创建优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 2. 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()      # 清零梯度（PyTorch 默认累加）
        loss = criterion(model(x), y)
        loss.backward()            # 计算梯度
        optimizer.step()           # 更新参数
```

**为什么必须 `zero_grad()`？** PyTorch 默认累加梯度（`grad += new_grad`），这设计是为了支持 RNN 等需要梯度累积的场景。大多数情况下每次更新前需要清零。

## 4. 优化器分类

| 类型 | 代表 | 特点 |
| --- | --- | --- |
| 基础 SGD | [[NeuralNetwork/Tips/Optimizers/SGD|SGD]] | 所有参数共享一个学习率 |
| 带动量 | SGD+Momentum | 积累历史梯度方向，加速收敛 |
| 自适应学习率 | AdaGrad, RMSprop | 每个参数有自己的学习率 |
| 动量 + 自适应 | [[NeuralNetwork/Tips/Optimizers/Adam|Adam]], [[NeuralNetwork/Tips/Optimizers/AdamW|AdamW]] | 结合两者优势，现代默认选择 |

## 5. 选择速查

| 场景 | 推荐 |
| --- | --- |
| 快速实验/原型 | **Adam** (lr=0.001) |
| CNN 图像分类 | AdamW (lr=0.001, weight_decay=1e-4) |
| 追求最高精度 | SGD+Momentum (lr=0.01~0.1, momentum=0.9) |
| Transformer/大模型 | AdamW |
| RNN/LSTM | RMSprop 或 Adam |

## 6. CNN 项目支持的优化器

```python
from cnnlib.training.optimizer import createOptimizer

# Adam
optimizer = createOptimizer(model, 'adam', lr=0.001)

# AdamW
optimizer = createOptimizer(model, 'adamw', lr=0.001, weight_decay=1e-4)

# SGD
optimizer = createOptimizer(model, 'sgd', lr=0.01, momentum=0.9)

# RMSprop
optimizer = createOptimizer(model, 'rmsprop', lr=0.001)
```
