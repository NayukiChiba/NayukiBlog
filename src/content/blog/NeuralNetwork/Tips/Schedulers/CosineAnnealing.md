---
title: CosineAnnealing：余弦退火详解
date: 2026-05-23
category: NeuralNetwork/Tips/Schedulers
tags:
  - 学习率
  - 调度器
description: 余弦退火以平滑曲线衰减学习率，是现代大模型训练的首选调度策略。
image: TODO
status: draft
---

## 1. 为什么用余弦曲线？

StepLR 在衰减点学习率突变，导致训练在这些点出现短暂的震荡。余弦退火提供了一条平滑的衰减曲线，让学习率连续地、优雅地降低：

$$
lr_t = lr_{\min} + \frac{1}{2}(lr_{\max} - lr_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)
$$

其中 $T_{\max}$ 是半周期长度。关于调度器的分类与选择，参见 [[NeuralNetwork/Tips/Schedulers/SchedulerOverview|调度器概述]]。

## 2. 基本使用

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,       # 半个余弦周期长度
    eta_min=1e-6     # 最小学习率（默认 0）
)
```

学习率从初始值按余弦曲线平滑降到 `eta_min`。

## 3. 带重启的余弦退火

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # 第一个周期长度
    T_mult=2,        # 后续周期倍增：10 → 20 → 40 → 80 ...
    eta_min=1e-6
)
```

每个周期结束，学习率跳回初始值。周期逐渐变长，给模型初期"多探索"的机会、后期"精细收敛"的时间。

**为什么需要重启？** 跳出局部最优——学习率突然变大给优化器一个"跳跃"的机会，可能跳出当前局部极小值。

## 4. 学习率曲线对比

![TODO: 三条学习率曲线对比图，StepLR阶梯状/Cosine平滑衰减/CosineWarmRestarts周期重启，X轴epoch Y轴学习率]

## 5. 为什么大模型训练常用？

现代 ViT、CLIP、GPT 等模型几乎全部使用 Cosine + [[NeuralNetwork/Tips/Schedulers/Warmup|Warmup]]：

- **平滑性**：大模型对学习率突变更敏感，余弦曲线的平滑性避免了震荡
- **后期精细**：余弦最后阶段的极小学习率让模型精细收敛
- **可预测**：学习率曲线完全由 `T_max` 确定，方便对比实验。关于学习率调参的完整指南，参见 [[NeuralNetwork/Tips/Techniques/LearningRateGuide|LR指南]]

## 6. 典型配置

```python
# 大模型标准配置
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                               weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-6
)
# 加上 Warmup：前 5 个 epoch 线性增长到 0.001

# 长时间训练配置（如 800 epoch）
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-6
)
# 周期：50 → 100 → 200 → 400
```
