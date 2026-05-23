---
title: StepLR：阶梯式学习率衰减
date: 2026-05-23
category: NeuralNetwork/Tips/Schedulers
tags:
  - 高级教程
  - 深度学习
description: 最简单的预设衰减策略——每 N 个 epoch 学习率乘以一个固定因子。
image: https://img.yumeko.site/file/blog/StepLR.png
status: published
---

## 1. 工作原理

每 `step_size` 个 epoch，学习率乘以 `gamma`：

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,      # 每 30 个 epoch
    gamma=0.1          # 衰减为原来的 0.1
)
```

学习率变化（初始 lr=0.1）：

$$
\text{Epoch 1-30: } lr = 0.1 \quad
\text{Epoch 31-60: } lr = 0.01 \quad
\text{Epoch 61-90: } lr = 0.001
$$

![StepLR.png](https://img.yumeko.site/file/articles/StepLR/StepLR.png)

## 2. MultiStepLR：自定义衰减点

如果希望更灵活地控制衰减时机：

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 80, 120],  # 在这些 epoch 衰减
    gamma=0.1
)
```

这种方式在经典论文中很常见（如 ResNet 在 epoch 30 和 60 各衰减一次）。

## 3. 参数选择

**step_size**：
- 太短：频繁衰减，训练后期学习率可能过早降到极小
- 太长：学习率保持高位太久，收敛不够精细
- 建议：总 epoch 的 1/3 到 1/5

**gamma**：
- 0.1：每次减为 1/10（经典选择）
- 0.5：每次减半（更温和）
- 0.3：折中

## 4. 何时使用

- 训练 epoch 数是预先确定的
- 希望学习率曲线可预测
- 经典论文复现（很多论文使用 MultiStepLR）
- 数据集和任务比较熟悉，知道大概在什么阶段需要降学习率。关于调度器的完整分类和选择策略，参见 [[NeuralNetwork/Tips/Schedulers/SchedulerOverview|调度器概述]]

## 5. 与其他调度器对比

| | StepLR | ReduceLROnPlateau | CosineAnnealing |
| --- | :---: | :---: | :---: |
| 衰减方式 | 固定时间点 | 触发式 | 连续平滑 |
| 可预测性 | 高 | 低 | 高 |
| 调参难度 | 需选时间点 | 调 patience | 需选周期 |
