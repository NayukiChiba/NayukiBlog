---
title: BatchNorm 实用指南
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 高级教程
  - 深度学习
description: BatchNorm 放哪里？训练和测试有何不同？小 batch 怎么处理？一份实用的 BN 避坑指南。
image: https://img.yumeko.site/file/articles/NNTrainingTips/BatchNorm.png
status: published
---

## 1. BN 应该放在哪里？

经过大量实验和论文论证，目前的主流共识是：

```
Conv2d → BatchNorm2d → ReLU     ← 推荐（现代标准）
Linear  → BatchNorm1d → ReLU    ← 推荐（现代标准）
```

**为什么 BN 在激活函数之前？**

[[NeuralNetwork/Theory/BatchNormalization|BN 的核心目的]]是让激活函数的输入保持稳定的分布。如果 BN 在 ReLU 之后，它归一化的是一个被"截断"过的分布（ReLU 把负值都变成了 0），效果不如归一化原始的全范围分布。

**历史演进**：
- 原始 BatchNorm 论文：$\text{Conv} \to \text{ReLU} \to \text{BN}$（在激活之后）
- 后来发现 $\text{Conv} \to \text{BN} \to \text{ReLU}$ 效果更好，成为现代标准

## 2. [[NeuralNetwork/Tips/Techniques/TrainEvalMode|训练/测试差异]]（最容易踩的坑）

```python
# 训练时
model.train()       # BN 使用当前 batch 的均值/方差
output = model(x)

# 测试/推理时
model.eval()        # BN 使用训练期间累积的移动平均
with torch.no_grad():
    output = model(x)
```

**为什么测试时不能用当前 batch 的统计量？**

测试时可能只有一个样本（`batch_size=1`），此时 batch 的均值和方差没有统计意义。

**错误示范**（常见 Bug）：

```python
# 错误：忘记切换到 eval 模式
model.train()
with torch.no_grad():
    testOutput = model(x)  # BN 使用了错误的统计量！
```

这会导致测试准确率明显偏低，且每次运行结果都不同。

## 3. 几个实用参数

**`eps`**：防止分母为零的小常数，默认 `1e-5`。一般不需要修改。

**`momentum`**：移动平均的更新速度。$\text{running\_mean} = \text{momentum} \times \text{running\_mean} + (1-\text{momentum}) \times \text{batch\_mean}$。

注意 PyTorch 的 momentum 定义和论文相反（PyTorch 的 0.1 ≈ 论文的 0.9）。

**`track_running_stats`**：是否跟踪全局统计量。默认 `True`。如果设为 `False`，测试时也使用 batch 统计量（通常只在调试时用）。

## 4. 小 batch size 问题与替代

当 `batch_size` 很小（比如 2 或 4）时，batch 统计量的噪声很大，BN 效果会下降。

| 归一化方法 | 归一化维度 | 适用场景 |
| --- | --- | --- |
| BatchNorm | 沿 batch 维度 | CNN, batch_size ≥ 16 |
| LayerNorm | 沿 feature 维度 | Transformer, RNN |
| InstanceNorm | 沿空间维度（单样本） | 风格迁移 |
| GroupNorm | 通道分组归一化 | batch_size 极小时 |

```python
# 小 batch 下替代 BN
# 从 nn.BatchNorm2d(128)
nn.GroupNorm(num_groups=32, num_channels=128)
```

## 5. BN 不适合的情况

- **Batch size = 1**（用 LayerNorm/InstanceNorm 替代）
- **RNN 等序列模型**（用 LayerNorm）
- **生成模型中对噪声敏感的任务**

## 6. 冻结 BN 进行微调

迁移学习微调时，通常冻结 BN 的统计量：

```python
# 方式一：设为 eval 模式（BN 使用预训练统计量，不更新）
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# 方式二：完全冻结（不更新统计量也不更新 affine 参数）
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
```

## 7. 删除 BN 的场景

在需要精确重现（如量化推理）或对 batch 依赖敏感的少数场景下，可以考虑完全移除 BN，用其他方式替代（如权重标准化）。



