---
title: AdamW：解耦权重衰减
date: 2026-05-23
category: NeuralNetwork/Tips/Optimizers
tags:
  - AdamW
  - 优化器
description: AdamW 修复了 Adam 中权重衰减实现错误，是目前图像分类和 Transformer 训练的最佳选择。
image: TODO
status: draft
---

## 1. Adam 的 Weight Decay 问题

在 [[NeuralNetwork/Tips/Optimizers/Adam|Adam]] 中，`weight_decay` 参数是这样应用的：

```python
# Adam 的做法（错误）
g_t = ∇L(w_t) + λ * w_t    # 在梯度中加入 L2 正则化
m_t = β1 * m_{t-1} + (1-β1) * g_t
# 后续自适应学习率会影响正则化项...

# 正确做法（AdamW）
g_t = ∇L(w_t)               # 梯度不含正则化
m_t = β1 * m_{t-1} + (1-β1) * g_t
w_t = w_{t-1} - η * (m̂_t / (√v̂_t + ε) + λ * w_{t-1})  # 权重衰减独立
```

**核心差异**：Adam 将 L2 正则化与自适应学习率耦合，导致正则化强度被学习率缩放。AdamW 将权重衰减解耦，直接对权重施加衰减。

## 2. AdamW 的优势

- **更强的正则化**：解耦后 weight_decay 效果更显著
- **泛化更好**：在 ImageNet 等大数据集上显著优于 Adam
- **更好的超参数解耦**：学习率和 weight_decay 可以独立调整

## 3. 使用

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01    # 比 Adam 中的 weight_decay 更有效
)
```

**weight_decay 典型值**：AdamW 中常用 `0.01 ~ 0.1`（比 Adam 中的典型值大）。

## 4. Adam vs AdamW

| | Adam | AdamW |
| --- | --- | --- |
| 权重衰减方式 | 与梯度耦合（L2 正则化） | 解耦，直接衰减权重 |
| weight_decay 效果 | 较弱 | **较强** |
| 泛化能力 | 一般 | **更好** |
| 当代使用 | 逐渐被 AdamW 取代 | **首选** |

**建议**：新项目直接使用 AdamW。如果从 Adam 迁移，原先的 weight_decay 可能需要调大（因为解耦后效果变强）。更多优化器的选择参考 [[NeuralNetwork/Tips/Optimizers/OptimizerOverview|优化器概述]]。

## 5. 完整配置示例

```python
# ImageNet 风格训练配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300
)
```
