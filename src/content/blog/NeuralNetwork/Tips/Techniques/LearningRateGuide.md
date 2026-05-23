---
title: 学习率调度策略完全指南
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 高级教程
  - 深度学习
description: 学习率是最重要的超参数。详解 ReduceLROnPlateau、StepLR、CosineAnnealing、Warmup 及选择原则。
image: https://img.yumeko.site/file/articles/NNTrainingTips/SchedulingStrategy.png
status: published
---

## 1. 学习率是最重要的超参数

有句经验之谈：如果你只能调一个超参数，那就调学习率。

- 学习率太大 → 损失上下跳，无法收敛
- 学习率太小 → [[NeuralNetwork/Tips/Troubleshooting/SlowConvergence|收敛太慢]]，可能陷入局部最优
- 学习率刚好 → 快速稳定收敛

**为什么需要动态调整？** 训练初期离最优解远，需要大步快走（大 LR）。训练后期接近最优解，大步可能跨过头，需要小步微调（小 LR）。

## 2. [[NeuralNetwork/Tips/Schedulers/ReduceLROnPlateau|ReduceLROnPlateau]]：按需衰减

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控指标越低越好（val_loss）
    factor=0.5,        # 衰减因子：新 lr = 旧 lr × 0.5
    patience=3,        # 容忍 3 个 epoch 不改善
    min_lr=1e-6        # 学习率下限
)

# 每个 epoch 后调用，传入监控指标
scheduler.step(valLoss)
```

**优点**：自适应，不需要预先知道训练多少 epoch。
**缺点**：只在 loss 卡住时才降，可能反应不够快。
**使用场景**：epoch 数不确定时。

## 3. StepLR：阶梯衰减

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,      # 每 30 个 epoch
    gamma=0.1          # 衰减为原来的 0.1
)
```

学习率变化：Epoch 1-30: $lr=0.001$, Epoch 31-60: $lr=0.0001$, Epoch 61-90: $lr=0.00001$。

**优点**：可预测、易理解。**缺点**：阶梯式突变，不够平滑。

## 4. CosineAnnealingLR：余弦退火

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100           # 半个余弦周期的 epoch 数
)
```

学习率从初始 lr 按余弦曲线平滑衰减到 0。现代大模型训练（ViT、CLIP）普遍使用。

**优点**：平滑衰减，后期精细收敛。**缺点**：需要预设周期。

## 5. CosineAnnealingWarmRestarts：带重启的余弦退火

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # 第一个周期长度
    T_mult=2     # 后续周期倍增：10 → 20 → 40 → ...
)
```

每个周期结束学习率跳回初始值，给模型"逃离局部最优"的机会。周期逐渐变长，给最后阶段的精细收敛留出时间。

## 6. OneCycleLR：一周期策略

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(trainLoader),
    epochs=30,
    pct_start=0.3    # 前 30% 时间升温
)
```

学习率先升后降，将整个训练周期视为一个循环。常配合较大的 batch size 使用。

## 7. Warmup

**问题**：训练刚开始时，模型参数是随机的，梯度可能很大。直接用预设的大学习率可能导致模型在最初几步"跑偏"。

**方案**：前 N 个 step/epoch 用较小的学习率，线性增加到目标学习率。

```python
def warmupLR(optimizer, currentStep, warmupSteps, targetLR):
    if currentStep < warmupSteps:
        lr = targetLR * (currentStep + 1) / warmupSteps
        for paramGroup in optimizer.param_groups:
            paramGroup['lr'] = lr

# 训练循环中按 step 调用
for batchIdx, (images, labels) in enumerate(trainLoader):
    warmupLR(optimizer, batchIdx, warmupSteps=500, targetLR=0.001)
    # ... 正常训练
```

## 8. LR Finder：自动找最佳学习率

Leslie Smith 提出的方法：从一个很小的学习率开始，每个 batch 逐步增大，记录 Loss 变化。选择 Loss 下降最快的那段对应的学习率。

```python
# 简化的 LR Finder
def lrFinder(model, trainLoader, criterion, device):
    lrs = []
    losses = []
    lr = 1e-7
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for batch in trainLoader:
        lrs.append(lr)
        loss = trainOneBatch(model, batch, criterion, optimizer, device)
        losses.append(loss)

        lr *= 1.1  # 每 batch 增大 10%
        if lr > 1.0 or loss > 10 * min(losses):
            break
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Plot losses vs lrs, pick lr where loss drops fastest
    return lrs, losses
```

## 9. 策略对比和选择

| 策略 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| ReduceLROnPlateau | epoch 数不确定 | 自适应，简单 | 被动等待 |
| StepLR | 已知总 epoch | 可预测 | 阶梯式突变 |
| CosineAnnealing | 需要精细收敛 | 平滑，效果好 | 需要预设周期 |
| Cosine + Warmup | 大模型训练 | 最稳定 | 调参复杂 |
| OneCycleLR | 快速训练 | 收敛极快 | 对 batch size 敏感 |

## 10. CNN 项目的[[NeuralNetwork/Tips/Schedulers/SchedulerOverview|调度器]]实践

```python
# 从 CNN 项目 cnnlib/training/scheduler.py
def createScheduler(optimizer, name, **kwargs):
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                 patience=3, min_lr=1e-6)
    elif name == 'step':
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=50)
    elif name == 'cosine_warm':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
```



