---
title: 模型收敛太慢怎么办？
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 训练问题
description: Loss 在降但速度太慢？从优化器到数据归一化，加速模型收敛的完整方案。
image: https://img.yumeko.site/file/blog/SlowConvergence.png
status: published
---

## 现象描述

Loss 确实在下降（不同于[[NeuralNetwork/Tips/Troubleshooting/LossNotDown|Loss不降]]），但下降速度极慢：

```
Epoch  5: Train Loss 1.85
Epoch 10: Train Loss 1.72
Epoch 20: Train Loss 1.60
Epoch 50: Train Loss 1.42
```

50 个 epoch 才降了 0.43，明显太慢了。正常的 CNN 在 CIFAR-10 上应该在 10-20 个 epoch 内达到合理水平。

## 原因分析

1. **[[NeuralNetwork/Tips/Techniques/LearningRateGuide|学习率]]太小** —— 参数更新步幅太小
2. **[[NeuralNetwork/Tips/Optimizers/OptimizerOverview|优化器选择]]不当** —— 用 SGD 但没调好
3. **Batch Size 太大** —— 更新频率太低
4. **数据未归一化** —— 不同特征的尺度差异大
5. **模型初始化不当** —— 起点离最优解太远

## 解决方案

### 方案 1：增大学习率

```python
# 当前学习率太小
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# 先试默认值
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 如果还不够快，继续增大
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
```

### 方案 2：使用学习率 Warmup

如果初始学习率不能太大（怕发散），可以用 warmup 快速达到目标学习率：

```python
# CosineAnnealingWarmRestarts 自带 warmup 效果
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

### 方案 3：Adam 替代 SGD

如果当前使用 SGD，切换为 Adam 通常能显著加速收敛：

```python
# 从 SGD（需要精细调参）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 换为 Adam（开箱即用）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

| 优化器 | 收敛速度 | 需要调参 |
| --- | :---: | :---: |
| SGD | 慢 | 需要仔细调 |
| SGD+Momentum | 中等 | momentum 和 lr |
| Adam | **快** | 开箱即用 |
| AdamW | 快 | weight_decay |

### 方案 4：检查数据归一化

未归一化的数据会导致不同特征的学习速度不一致：

```python
# 确认 normalize 已应用
for images, _ in trainLoader:
    mean = images.mean().item()
    std = images.std().item()
    print(f"均值: {mean:.3f}, 标准差: {std:.3f}")
    # 理想：均值 ≈ 0, 标准差 ≈ 1

    if abs(mean) > 1 or abs(std - 1) > 0.5:
        print("⚠️ 数据可能未正确归一化")
    break
```

### 方案 5：减小 Batch Size（增加更新频率）

Batch Size 太大会减少每个 epoch 的更新次数：

```python
# 当前：128 样本/batch，60000 样本 → 每 epoch 只更新 468 次
trainLoader = DataLoader(dataset, batch_size=128, ...)

# 改进：64 样本/batch → 每 epoch 更新 937 次
trainLoader = DataLoader(dataset, batch_size=64, ...)
```

### 方案 6：使用 BatchNorm

BN 能显著加速收敛：

```python
# 在每个卷积后添加 BN
nn.Sequential(
    nn.Conv2d(inCh, outCh, 3, padding=1),
    nn.BatchNorm2d(outCh),    # ← 加速收敛
    nn.ReLU(),
)
```

### 方案 7：验证初始化

错误的初始化让模型起点离最优解更远：

```python
# 检查初始化质量
def checkInitialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            mean = param.data.mean().item()
            std = param.data.std().item()
            if abs(mean) > 0.1:
                print(f"⚠️ {name} 均值过大: {mean:.4f}")
            if std > 1 or std < 0.001:
                print(f"⚠️ {name} 标准差异常: {std:.6f}")
```

## 加速训练的命令行选项

```python
# DataLoader 优化
trainLoader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,         # 多进程加载数据
    pin_memory=True,       # GPU 训练时加速传输
    persistent_workers=True # 保持 worker 进程存活
)

# 使用混合精度训练（需要支持）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 收敛速度参考值

在 CIFAR-10 上使用 ResNet-18 + Adam(lr=0.001)，大致参考：

| Epoch | 预期 Train Acc |
| :---: | :---: |
| 1 | 30-45% |
| 5 | 65-80% |
| 10 | 80-90% |
| 20 | 90-95% |

如果在 epoch 5 时 Train Acc 仍低于 40%，说明明显偏慢。

## 解决优先级

```
1. 增大学习率                  → 最直接
2. 确认数据归一化              → 零成本
3. 添加 BatchNorm              → 效果显著
4. 换 Adam 优化器              → 开箱即用
5. 使用 Warmup + Cosine 调度   → 现代最佳实践
6. 减小 Batch Size             → 增加更新频率
7. 混合精度训练                → 加速但不改善收敛
```

![SlowConvergence.png](https://img.yumeko.site/file/articles/SlowConvergence/SlowConvergence.png)
