---
title: RMSprop：非平稳目标的利器
date: 2026-05-23
category: NeuralNetwork/Tips/Optimizers
tags:
  - 高级教程
  - 深度学习
description: 理解 RMSprop 的指数加权移动平均设计，及它在 RNN 和强化学习中的独特价值。
image: https://img.yumeko.site/file/blog/Optimizer.png
status: published
---
## 1. RMSprop 的设计动机

AdaGrad 给每个参数分配学习率 $\eta / \sqrt{G_t}$，其中 $G_t$ 是从训练开始以来所有梯度平方的累加和。问题是 $G_t$ 只增不减，学习率最终会降到 0。

RMSprop 的解决方案：用指数移动平均替代累加和，让"记忆"逐渐遗忘旧梯度：

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2
$$

## 2. 参数更新公式

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t
$$

其中 $v_t$ 是梯度平方的指数移动平均，$\beta$ 通常设为 $0.9$（类比动量）。

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.9,      # 平滑系数（类似 beta）
    eps=1e-8,
    momentum=0.0,   # 可选，加动量
    weight_decay=0.0
)
```

## 3. RMSprop 的适用场景

- **RNN / LSTM**：RMSprop 是 RNN 训练的经典选择（[[NeuralNetwork/Tips/Optimizers/Adam|Adam]] 出现之前）
- **强化学习**：非平稳目标（Q 值不断变化），RMSprop 的指数遗忘很合适
- **非平稳优化问题**：目标函数随时间改变的场景

## 4. RMSprop vs Adam

Adam 可以看作是 RMSprop + Momentum + Bias Correction 的组合。在 CNN 图像分类中，Adam/AdamW 通常优于 RMSprop。但在 RNN 和强化学习中，RMSprop（或带 momentum 的 RMSprop）仍有价值。不同优化器的分类与选择策略，参见 [[NeuralNetwork/Tips/Optimizers/OptimizerOverview|优化器概述]]。

## 5. 在 CNN 项目中使用

```python
from cnnlib.training.optimizer import createOptimizer

optimizer = createOptimizer(model, 'rmsprop', lr=0.001)
```
