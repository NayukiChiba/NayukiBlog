---
title: Warmup：平稳起步的艺术
date: 2026-05-23
category: NeuralNetwork/Tips/Schedulers
tags:
  - 学习率
  - Warmup
description: 为什么训练最初几步可能毁掉整个训练？理解 Warmup 的原理、类型和步数选择。
image: TODO
status: draft
---

## 1. 为什么需要 Warmup？

训练开始时，模型参数是随机初始化的。此时：
- 梯度可能很大且方向不稳定
- 如果直接用预设的大学习率，参数更新幅度过大
- 模型可能在最初几步就"跑偏"（损失面很差的方向）

Warmup 的解决方案：前 N 步用较小的学习率，逐渐增加到目标学习率，让模型先"站稳脚跟"。

![TODO: Warmup+CosineDecay完整学习率曲线，前N步线性上升(Warmup阶段)，之后余弦曲线衰减至eta_min]

## 2. Linear Warmup

最简单、最常用的 warmup 方式——学习率线性增长：

```python
def linearWarmup(optimizer, currentStep, warmupSteps, targetLR):
    if currentStep < warmupSteps:
        lr = targetLR * (currentStep + 1) / warmupSteps
        for g in optimizer.param_groups:
            g['lr'] = lr

# 训练循环中
for batchIdx, (images, labels) in enumerate(trainLoader):
    linearWarmup(optimizer, batchIdx, warmupSteps=500, targetLR=0.001)
    # ... 正常训练步骤
```

## 3. Cosine Warmup

学习率按余弦曲线从 0 上升到目标值：

```python
def cosineWarmup(optimizer, currentStep, warmupSteps, targetLR):
    if currentStep < warmupSteps:
        progress = currentStep / warmupSteps
        lr = targetLR * 0.5 * (1 - np.cos(np.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr
```

Cosine Warmup 比 Linear Warmup 更平滑（两端变化慢，中间快）。

## 4. Warmup 步数选择

| 数据集/模型规模 | 推荐 warmup 步数 |
| --- | :---: |
| 小型任务（MNIST, CIFAR-10） | 通常不需要，或 100-200 步 |
| 中型任务（ImageNet, ResNet-50） | 5 个 epoch 以内 |
| 大型任务（ViT, GPT） | 通常 2000-4000 步 |
| 超大 batch size 训练 | 必须 warmup，步数与 batch size 相关 |

经验法则：batch size 越大，warmup 越重要。使用 `batch_size=8192` 时，warmup 几乎是必须的。

## 5. Warmup + Decay 完整示例

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmupEpochs, totalEpochs, minLR=1e-6):
        self.optimizer = optimizer
        self.warmupEpochs = warmupEpochs
        self.totalEpochs = totalEpochs
        self.minLR = minLR
        self.baseLRs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmupEpochs:
            # Warmup: 线性增长
            scale = (epoch + 1) / self.warmupEpochs
        else:
            # Cosine decay
            progress = (epoch - self.warmupEpochs) / (self.totalEpochs - self.warmupEpochs)
            scale = self.minLR + 0.5 * (1 - self.minLR) * (1 + np.cos(np.pi * progress))

        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = self.baseLRs[i] * scale
```

## 6. Warmup 的"暗黑知识"

**不只在训练初期需要**：如果学习率调度使用了 [[NeuralNetwork/Tips/Schedulers/CosineAnnealing|CosineAnnealingWarmRestarts]]（周期性重启），每次学习率跳回高值时，也需要一个小规模的 warmup 来防止突然跳跃导致的震荡。

**Transformer 中的 warmup**：Transformer 对 warmup 特别敏感。原始 "Attention Is All You Need" 论文使用了 4000 步 warmup。不设 warmup 可能导致 Transformer 无法收敛。更多调度策略参见 [[NeuralNetwork/Tips/Schedulers/SchedulerOverview|调度器概述]]。
