---
title: 梯度爆炸/消失怎么办？
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 训练问题
  - 梯度
description: Loss 突然 NaN（爆炸）或完全不下降（消失）？诊断并修复梯度异常。
image: TODO
status: draft
---

## 现象描述

### 梯度爆炸

```
Epoch 5: Train Loss 0.45
Epoch 6: Train Loss 0.52
Epoch 7: Train Loss NaN          ← 突然变成 NaN
```

或者更直观的：Loss 在某个 batch 突然从正常值跳到 `inf` 或 [[NeuralNetwork/Tips/Troubleshooting/NanLoss|NaN]]，属于[[NeuralNetwork/Tips/Troubleshooting/TrainingUnstable|训练不稳]]的典型现象。

### 梯度消失

```
Epoch 1: Train Loss 2.30, Acc 10%
Epoch 2: Train Loss 2.30, Acc 10%
Epoch 5: Train Loss 2.30, Acc 10%  ← 完全不变
```

Loss 几乎不变，模型的浅层梯度接近 0。

## 原因分析

### 梯度爆炸

反向传播时，梯度从输出层逐层传回。如果某些层的梯度 > 1，经过多层累积后梯度指数增长：

$$
\nabla_{W_1} \mathcal{L} \approx \prod_{l=2}^{L} \frac{\partial z_l}{\partial z_{l-1}} \cdot \nabla_{z_L} \mathcal{L}
$$

如果每层的雅可比行列式 > 1，这个连乘会爆炸。

### 梯度消失

同样原理，如果各层的梯度 < 1，连乘后梯度指数衰减到接近 0，浅层几乎收不到任何梯度信号。使用 Tanh 或 Sigmoid 激活函数时尤其容易发生（导数最大值也仅为 1 和 0.25）。

## 解决方案

### 梯度爆炸

**方案 1：梯度裁剪（首选）**

训练循环中添加梯度裁剪，将梯度范数限制在阈值内：

```python
# 在 loss.backward() 之后、optimizer.step() 之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

常见阈值：$1.0$ 或 $5.0$。这保留梯度的方向但限制了步长。

**方案 2：降低学习率**

```python
# 当前学习率太大
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 降低
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**方案 3：检查数据归一化**

```python
# 确认输入数据已经归一化
for images, _ in trainLoader:
    print(f"范围: [{images.min():.2f}, {images.max():.2f}]")
    # 应该在 [-2, 2] 左右
    if images.max() > 10:
        print("⚠️ 数据未归一化！")
    break
```

**方案 4：使用 BatchNorm**

BN 能有效控制每层的输出范围，防止梯度过大：

```python
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),    # ← BN 稳定梯度
    nn.ReLU(),
)
```

### 梯度消失

**方案 1：使用 ReLU 替代 Tanh/Sigmoid**

ReLU 的正数区域导数为常数 1，不会衰减：

```python
# 从 Tanh（导数 ≤1，饱和区 ≈0）
self.activation = nn.Tanh()

# 换为 ReLU（正数区导数恒为 1）
self.activation = nn.ReLU()
```

**方案 2：添加 BatchNorm**

BN 将每层输入调整到激活函数的活跃区，防止落入饱和区：

```python
# Conv → BN → ReLU（现代标准顺序）
nn.Sequential(
    nn.Conv2d(inCh, outCh, 3, padding=1),
    nn.BatchNorm2d(outCh),
    nn.ReLU(),
)
```

**方案 3：使用残差连接（ResNet）**

Shortcut 连接提供了梯度的"高速公路"：

```python
# ResNet 残差块
out = self.conv2(self.relu(self.conv1(x)))
out += x  # shortcut：梯度可以通过这条通道直达浅层
```

即使卷积层的梯度很小，shortcut 提供的常数 1 梯度也能保证信号到达浅层。

**方案 4：正确的[[NeuralNetwork/Tips/Techniques/WeightInitialization|权重初始化]]**

```python
# ReLU + Kaiming 初始化
nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')

# Tanh + Xavier 初始化
nn.init.xavier_uniform_(conv.weight)
```

**方案 5：减小网络深度**

如果是自定义的极深网络（比如 100+ 层）但没有残差连接：

```python
# 减少层数，或引入残差连接
```

## 梯度监控

在训练时监控梯度的范数，及早发现问题：

```python
def monitorGradients(model):
    """检查各层梯度的范数"""
    totalNorm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            paramNorm = param.grad.data.norm(2)
            totalNorm += paramNorm.item() ** 2

            if paramNorm > 100:
                print(f"⚠️ 梯度爆炸: {name} norm={paramNorm:.1f}")
            elif paramNorm < 1e-7:
                print(f"⚡ 梯度消失: {name} norm={paramNorm:.2e}")

    totalNorm = totalNorm ** 0.5
    return totalNorm

# 训练循环中使用
loss.backward()
gradNorm = monitorGradients(model)
print(f"总梯度范数: {gradNorm:.4f}")

# 自动裁减
if gradNorm > 10:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 快速诊断

| 症状 | 可能原因 | 首选方案 |
| --- | --- | --- |
| Loss → NaN | 梯度爆炸 | 梯度裁剪 + 降低 LR |
| Loss 震荡不定 | 梯度偏大 | 梯度裁剪 |
| Loss 完全不变 | 梯度消失 | 换 ReLU + Kaiming |
| 浅层梯度 ≈ 0 | 梯度消失 | BN + 残差连接 |
| 深层梯度 ≈ 0 | 梯度消失（深层） | 残差连接 |
