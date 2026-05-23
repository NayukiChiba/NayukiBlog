---
title: ReduceLROnPlateau：自适应学习率衰减
date: 2026-05-23
category: NeuralNetwork/Tips/Schedulers
tags:
  - 高级教程
  - 深度学习
description: 当验证指标不再改善时自动降低学习率——最省心的调度策略。
image: https://img.yumeko.site/file/blog/ReduceLROnPlateau.png
status: published
---

## 1. 工作原理

```
if val_loss 连续 patience 个 epoch 没有显著改善:
    lr = lr × factor
```

这是最"懒人友好"的调度器——不需要预设总 epoch 数，调度器自己判断时机。关于调度器的分类和选择，参见 [[NeuralNetwork/Tips/Schedulers/SchedulerOverview|调度器概述]]。

## 2. 参数详解

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # 'min' for loss, 'max' for accuracy
    factor=0.5,         # 衰减因子
    patience=3,         # 容忍多少个 epoch 不改善
    threshold=1e-4,     # 改善至少多少才算"有效改善"
    min_lr=1e-6,        # 学习率下限
    verbose=True        # 打印衰减信息
)

scheduler.step(valLoss)  # 必须传入监控指标
```

## 3. 参数选择指南

**factor**：
- $0.5$（减半）：温和，常用
- $0.1$（减为 1/10）：激进，快速降低
- $0.3$：折中

**patience**：
- 太小（2-3）：容易因波动提前衰减
- 中等（5-10）：给模型时间跨过高原
- 太大：可能等太久

**mode**：
- `'min'`：监控 loss（越低越好）
- `'max'`：监控 accuracy（越高越好）

## 4. 训练曲线效果

```
学习率变化（factor=0.5, patience=3）：
Epoch  1- 9: lr = 0.001   (初始)
Epoch 10-19: lr = 0.0005  (第7-9 epoch 没改善 → 减半)
Epoch 20-24: lr = 0.00025 (第17-19 epoch 没改善 → 再减半)
...
```

![ReduceLROnPlateau.png](https://img.yumeko.site/file/articles/ReduceLROnPlateau/ReduceLROnPlateau.png)

## 5. 使用示例

```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                               factor=0.5, patience=3, min_lr=1e-6)

for epoch in range(epochs):
    trainLoss = train(...)
    valLoss = validate(...)

    scheduler.step(valLoss)  # ← 必须传入

    # 查看当前学习率
    currentLR = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: lr = {currentLR:.6f}")
```

## 6. 优缺点

**优点**：自适应，无需预设 epoch；简单可靠；适合实验探索阶段。

**缺点**：只在卡住时才降，不能主动规划学习率曲线；可能反应偏慢。

## 7. 与 Early Stopping 的区别

两者都监控验证指标，但目的不同：
- **ReduceLROnPlateau**：降学习率，继续训练
- **[[NeuralNetwork/Tips/Techniques/EarlyStoppingGuide|Early Stopping]]**：停止训练，防止过拟合

通常两者配合使用，ReduceLROnPlateau 的 patience 应该小于 Early Stopping 的 patience。
