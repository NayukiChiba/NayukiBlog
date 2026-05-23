---
title: Batch Normalization 详解：加速训练的利器
date: 2026-05-07
category: NeuralNetwork/Theory
tags:
  - 基础
  - BN
  - 正则化
description: 从 Internal Covariate Shift 到训练/测试行为差异，深入理解 BatchNorm 的原理与最佳实践。
image: TODO
status: draft
---

## 1. 问题：内部协变量偏移（Internal Covariate Shift）

在训练过程中，前面层的参数不断更新，导致后面层的输入分布也在不断变化。这种现象被称为"内部协变量偏移"。

**打个比方**：想象你在学习射箭，但每次射箭时靶子都在随机移动，你就很难学会。BatchNorm 就像把靶子固定在一个标准位置。

从数学角度看，如果某一层的输入分布一直在变（均值、方差不稳定），该层就需要不断适应新的分布，学习效率极低。

## 2. BatchNorm 做了什么？

对于一个小批量（mini-batch）中的某个特征维度，BatchNorm 执行以下四步：

**步骤 1**：计算小批量均值

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

**步骤 2**：计算小批量方差

$$
\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2
$$

**步骤 3**：归一化

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

（$\epsilon$ 是防止除零的小常数，默认 $10^{-5}$）

**步骤 4**：缩放和平移（可学习的参数）

$$
y_i = \gamma \cdot \hat{x}_i + \beta
$$

其中 $\gamma$（缩放因子）和 $\beta$（平移因子）是**可学习的参数**。这一步至关重要——如果归一化后分布不适合后续的激活函数，网络可以自己学出 $\gamma$ 和 $\beta$ 来恢复对表达有利的分布。

![TODO: BatchNorm计算流程示意图，输入mini-batch→减均值→除标准差→乘γ加β→输出，标注训练/测试差异]

## 3. BatchNorm 的好处

- **允许更大的学习率**：不用担心梯度爆炸/消失
- **加速收敛**：训练更快到达较低 loss
- **有轻微正则化效果**：小批量的统计噪声类似于噪声注入
- **减少对初始化的敏感性**：即使初始参数不太好，也能训练起来
- **允许使用 saturating 激活函数**：如 Sigmoid/Tanh，BN 保证它们的输入在活跃区间

## 4. 训练 vs 测试时的行为差异

这是最容易踩的坑之一。

| 模式 | 使用的均值/方差 | `running_mean` / `running_var` |
| --- | --- | :---: |
| `model.train()` | 当前 batch 的统计量 | **更新**（指数移动平均） |
| `model.eval()` | 训练期间累积的全局统计量 | **冻结**（不更新） |

**为什么测试时不能用当前 batch 的统计量？**

测试时可能只有一个样本（`batch_size=1`），此时 batch 的均值和方差没有统计意义。所以测试时使用训练过程中累积的全局统计量。

**全局统计量的更新方式**（指数移动平均）：

$$
\text{running\_mean} = \text{momentum} \times \text{running\_mean} + (1 - \text{momentum}) \times \text{batch\_mean}
$$

注意 PyTorch 的 `momentum` 定义和论文相反（PyTorch 的 0.1 ≈ 论文的 0.9）。

```python
# 正确做法
model.train()   # BN 使用 batch 统计量
output = model(train_x)

model.eval()    # BN 使用全局统计量
with torch.no_grad():
    output = model(test_x)

# 错误做法：忘记 eval()
model.train()
with torch.no_grad():
    output = model(test_x)  # BN 用了错误的统计量！
```

## 5. BN 的放置位置

经过大量实验和论文论证，目前的主流共识是：

```
Conv2d → BatchNorm2d → ReLU     ← 推荐（现代标准）
Linear  → BatchNorm1d → ReLU    ← 推荐（现代标准）
```

**为什么 BN 在激活函数之前？**

BN 的核心目的是让激活函数的输入保持稳定的分布。如果 BN 在 ReLU 之后，那它归一化的是一个被"截断"过的分布（ReLU 把负值都变成了 0），效果不如归一化原始的全范围分布。

**历史演进**：
- 原始 BatchNorm 论文：$\text{Conv} \to \text{ReLU} \to \text{BN}$（在激活之后）
- 后来发现 $\text{Conv} \to \text{BN} \to \text{ReLU}$ 效果更好，成为现代标准

## 6. 小 batch size 问题与替代方案

BatchNorm 的效果依赖于 batch size。当 `batch_size` 很小（比如 2 或 4）时，batch 统计量的噪声很大，BN 效果会明显下降。

| 归一化方法 | 归一化维度 | 适用场景 |
| --- | --- | --- |
| BatchNorm | 沿 batch 维度 | CNN，batch_size ≥ 16 |
| LayerNorm | 沿 feature 维度 | Transformer, RNN |
| InstanceNorm | 沿空间维度（单样本） | 风格迁移，图像生成 |
| GroupNorm | 通道分组归一化 | batch_size 极小时替代 BN |

小 batch 场景下，[[NeuralNetwork/Tips/Techniques/BatchNormGuide|BN实用指南]] 中提供了详细的替代方案和调参建议。

```python
# 小 batch 场景下的替代
nn.GroupNorm(num_groups=32, num_channels=128)  # 代替 BatchNorm2d(128)
```

## 7. PyTorch 代码示例

```python
import torch.nn as nn

# 构建一个带 BN 的卷积块
class ConvBlock(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.conv = nn.Conv2d(inCh, outCh, 3, padding=1)
        self.bn = nn.BatchNorm2d(outCh)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 验证 BN 参数
bn = nn.BatchNorm2d(64)
print(f"γ 形状: {bn.weight.shape}")  # (64,) — 可学习缩放
print(f"β 形状: {bn.bias.shape}")    # (64,) — 可学习平移
print(f"running_mean: {bn.running_mean.shape}")  # (64,) — 运行均值
```

## 8. 总结

- BN 的核心是让每一层的输入分布保持稳定
- BN 的 $\gamma$ 和 $\beta$ 是可学习的，网络可以恢复出最优分布
- **训练时用 batch 统计量，测试时用全局统计量**
- 现代标准位置：$\text{Conv} \to \text{BN} \to \text{ReLU}$
- 小 batch 场景用 GroupNorm 或 LayerNorm 替代

BatchNorm 与 [[NeuralNetwork/Theory/Dropout|Dropout]] 经常共用，两者在 train/eval 模式下的行为差异需特别注意。
