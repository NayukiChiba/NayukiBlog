---
title: Loss 变成 NaN 怎么办？
date: 2026-05-23
category: 神经网络/问题排查
tags:
  - 高级教程
description: 训练到一半 Loss 突然变成 NaN？排查 5 大常见原因及对应的修复方法。
image: https://img.yumeko.site/file/blog/cover/1780581814812.webp
status: published
---

## 现象描述

训练正常运行了几个 epoch，然后突然：

```
Epoch  8: Train Loss 0.324, Val Loss 0.412
Epoch  9: Train Loss 0.298, Val Loss 0.389
Epoch 10: Train Loss   NaN, Val Loss   NaN    <- 突然变成 NaN
```

之后所有 Loss 都是 NaN，模型参数也变成了 NaN——[[NeuralNetwork/Troubleshooting/TrainingUnstable|训练不稳]]的极端表现。

## 原因分析

NaN（Not a Number）在数值计算中由以下操作产生：
- $0/0$
- $\infty - \infty$
- $\infty / \infty$
- $\log(0)$
- $\sqrt{\text{负数}}$（浮点数）

在深度学习训练中，最常见的原因按频率排序：

## 原因一：梯度爆炸（最常见的[[NeuralNetwork/Troubleshooting/GradientExplodingVanishing|梯度问题]]）

**机制**：某次参数更新幅度过大，导致某个参数变成极值（如 +/-1e38），后续前向/反向传播产生 NaN。

**诊断**：

```python
# 在 optimizer.step() 之前检查梯度
totalNorm = 0.0
for p in model.parameters():
    if p.grad is not None:
        paramNorm = p.grad.data.norm(2)
        totalNorm += paramNorm.item() ** 2
        if torch.isnan(p.grad).any():
            print(f"NaN in gradient!")
        if paramNorm > 1000:
            print(f"Gradient explosion: norm={paramNorm:.1f}")
totalNorm = totalNorm ** 0.5
```

**解决**：

```python
# 梯度裁剪
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# 同时降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 从 0.001 降低
```

## 原因二：除零错误

**机制**：某个地方做了除法，分母为 0。

常见位置：
- BatchNorm 的 $\sqrt{\sigma^2 + \epsilon}$：如果某层的所有输出完全相同，方差为 0
- 自定义 Loss 中有除法
- 学习率调度器中做除法

**解决**：

```python
# BatchNorm 的 eps 太小
nn.BatchNorm2d(64, eps=1e-5)   # 默认
nn.BatchNorm2d(64, eps=1e-3)   # 增大 eps 避免除零

# 自定义除法加保护
x = numerator / (denominator + 1e-8)  # 而不是 / denominator
```

## 原因三：$\log(0)$ 或 $\log(\text{负数})$

**机制**：CrossEntropyLoss 内部计算 $-\log(p_y)$，如果 $p_y = 0$（理论上不会，但数值上可能），结果为 inf。

或者在自定义 Loss 中直接调用了 `torch.log()`：

```python
# 危险
loss = -torch.log(probability)

# 安全
loss = -torch.log(probability + 1e-8)
# 或使用 PyTorch 内置函数
loss = F.nll_loss(torch.log(probability + 1e-8), labels)
```

## 原因四：数据中存在 NaN 或 Inf

**诊断**：

```python
for batchIdx, (images, labels) in enumerate(trainLoader):
    if torch.isnan(images).any() or torch.isinf(images).any():
        print(f"⚠️ Batch {batchIdx} 包含 NaN/Inf!")
    if torch.isnan(labels).any() or torch.isinf(labels).any():
        print(f"⚠️ Batch {batchIdx} 标签包含 NaN/Inf!")
```

这在处理自定义数据集时尤其常见——预处理代码可能有 bug。

## 原因五：学习率过大

**机制**：学习率太大 -> 参数更新幅度过大 -> 权重变为 NaN -> 传播到所有输出。

```python
# 如果 lr > 0.1 且使用 Adam，容易出问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 降低
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## 定位 NaN 的精确位置

在训练循环中添加 NaN 检测：

```python
import torch

def detectNan(tensor, name=""):
    """检测并报告 NaN 的精确位置"""
    if torch.isnan(tensor).any():
        nanCount = torch.isnan(tensor).sum().item()
        totalCount = tensor.numel()
        print(f"❌ {name}: {nanCount}/{totalCount} 个值为 NaN")
        return True
    return False

# 训练循环中使用
loss = criterion(output, labels)
if detectNan(loss, "Loss"):
    # 打印详细信息进行调试
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  NaN in parameter: {name}")
    break  # 停止训练

loss.backward()
optimizer.step()
```

## 训练时自动保护

在生产训练代码中，建议添加自动保护：

```python
# 完整的安全训练步骤
def safeTrainingStep(model, images, labels, criterion, optimizer, gradClip=1.0):
    outputs = model(images)

    # 检测 NaN
    if torch.isnan(outputs).any():
        print("⚠️ 模型输出包含 NaN，跳过此次更新")
        return None

    loss = criterion(outputs, labels)

    if torch.isnan(loss):
        print("⚠️ Loss 为 NaN，跳过此次更新")
        return None

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradClip)

    # 检测梯度中的 NaN
    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print("⚠️ 梯度包含 NaN，跳过此次更新")
            optimizer.zero_grad()
            return None

    optimizer.step()
    return loss.item()
```

## 快速排查清单

```
1. 加了梯度裁剪吗？          -> torch.nn.utils.clip_grad_norm_()
2. 学习率合理吗？            -> 默认 0.001 (Adam), 0.01 (SGD)
3. 数据中有 NaN 吗？         -> 检查 DataLoader 输出
4. 模型有 BN 层且 eps 合理吗？ -> eps=1e-5 一般是安全的
5. 自定义 Loss 有除零保护吗？  -> 分母 + 1e-8
```
![NanLoss.png](https://img.yumeko.site/file/blog/articles/1780581591179.webp)
