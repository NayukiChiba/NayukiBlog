---
title: 学习率调度器概述
date: 2026-05-23
category: NeuralNetwork/Tips/Schedulers
tags:
  - 学习率
  - 调度器
description: 理解为什么要调整学习率，PyTorch 中调度器使用模式，以及如何选择最适合的调度策略。
image: https://img.yumeko.site/file/blog/SchedulerOverview.png
status: published
---

## 1. 为什么要调整学习率？

训练初期离最优解远 → 需要大步快走（大 LR）。训练后期接近最优解 → 需要小步微调（小 LR）。如果一直用大学习率，会在最优解附近来回震荡无法收敛。

![Comparision.png](https://img.yumeko.site/file/articles/SchedulerOverview/Comparision.png)
## 2. PyTorch 调度器使用模式

```python
# 创建调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    train(...)
    valLoss = validate(...)

    # 方式 1：按 epoch 调度（大多数调度器）
    scheduler.step()

    # 方式 2：按 metric 调度（ReduceLROnPlateau 专用）
    scheduler.step(valLoss)
```

**重要**：`scheduler.step()` 必须在 `optimizer.step()` 之后调用（每个 epoch 结束时），不是在每个 batch 之后。

## 3. 各调度器直观对比

| 调度器 | 曲线形态 | 核心参数 |
| --- | --- | --- |
| ReduceLROnPlateau | 阶梯下降（触发式） | factor=0.5, patience=3 |
| StepLR | 阶梯下降（固定间隔） | step_size=30, gamma=0.1 |
| CosineAnnealing | 平滑余弦下降 | T_max=50 |
| CosineWarmRestarts | 余弦周期 + 重启 | T_0=10, T_mult=2 |
| OneCycleLR | 先升后降 | max_lr=0.01, pct_start=0.3 |

## 4. 选择建议

| 场景 | 推荐 |
| --- | --- |
| epoch 数不确定，需要自动调整 | ReduceLROnPlateau |
| 已知训练总 epoch，简单可靠 | StepLR 或 MultiStepLR |
| 精细收敛，追求最优精度 | CosineAnnealing |
| 大模型，需要长时间训练 | CosineWarmRestarts |
| 快速训练，大 batch size | OneCycleLR |

## 5. 配合 Warmup

现代训练流程通常 [[NeuralNetwork/Tips/Schedulers/Warmup|Warmup]] + 调度器组合使用：

```python
# 自定义 Warmup + Cosine 调度
class WarmupCosineLR:
    def __init__(self, optimizer, warmupSteps, totalSteps, minLR=1e-6):
        self.optimizer = optimizer
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
        self.baseLRs = [g['lr'] for g in optimizer.param_groups]

    def step(self, step):
        if step < self.warmupSteps:
            # Warmup 阶段：线性增长
            scale = step / self.warmupSteps
        else:
            # Cosine 阶段：余弦衰减
            progress = (step - self.warmupSteps) / (self.totalSteps - self.warmupSteps)
            scale = self.minLR + 0.5 * (1 - self.minLR) * (1 + np.cos(np.pi * progress))

        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = self.baseLRs[i] * scale
```

## 6. CNN 项目支持的调度器

```python
from cnnlib.training.scheduler import createScheduler

# ReduceLROnPlateau
scheduler = createScheduler(optimizer, 'plateau', factor=0.5, patience=3)

# StepLR
scheduler = createScheduler(optimizer, 'step', step_size=10, gamma=0.1)

# CosineAnnealingLR
scheduler = createScheduler(optimizer, 'cosine', T_max=50)

# CosineAnnealingWarmRestarts
scheduler = createScheduler(optimizer, 'cosine_warm', T_0=10, T_mult=1)
```
