---
title: Batch Normalization：从原理到实践
date: 2026-05-09
category: NeuralNetwork/Training
tags:
  - 基础
  - 深度学习
  - 高级教程
description: 从 Internal Covariate Shift 到训练/测试行为差异，从放置策略到小 batch 处理与 Dropout 交互，一站式掌握 BatchNorm 的原理与最佳实践。
image: https://img.yumeko.site/file/blog/BatchNormalization.png
status: published
---

## 1. 问题：内部协变量偏移（Internal Covariate Shift）

在训练过程中，前面层的参数不断更新，导致后面层的输入分布也在不断变化。这种现象被称为"内部协变量偏移"。

**打个比方**：想象你在学习射箭，但每次射箭时靶子都在随机移动，你就很难学会。BatchNorm 就像把靶子固定在一个标准位置。

从数学角度看，如果某一层的输入分布一直在变（均值、方差不稳定），该层就需要不断适应新的分布，学习效率极低。

## 2. BatchNorm 的数学原理

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

![BatchNorm.png](https://img.yumeko.site/file/articles/BatchNormalization/BatchNorm.png)

### 2.1 BatchNorm 带来的好处

- **允许更大的学习率**：不用担心梯度爆炸/消失
- **加速收敛**：训练更快到达较低 loss
- **有轻微正则化效果**：小批量的统计噪声类似于噪声注入
- **减少对初始化的敏感性**：即使初始参数不太好，也能训练起来
- **允许使用 saturating 激活函数**：如 Sigmoid/Tanh，BN 保证它们的输入在活跃区间

## 3. BN 的放置位置：Conv-BN-ReLU vs Conv-ReLU-BN

经过大量实验和论文论证，目前的主流共识是：

```
Conv2d → BatchNorm2d → ReLU     ← 推荐（现代标准）
Linear  → BatchNorm1d → ReLU    ← 推荐（现代标准）
```

**为什么 BN 在激活函数之前？**

BN 的核心目的是让激活函数的输入保持稳定的分布。如果 BN 在 ReLU 之后，它归一化的是一个被"截断"过的分布（ReLU 把负值都变成了 0），效果不如归一化原始的全范围分布。

**历史演进**：
- 原始 BatchNorm 论文：$\text{Conv} \to \text{ReLU} \to \text{BN}$（在激活之后）
- 后来发现 $\text{Conv} \to \text{BN} \to \text{ReLU}$ 效果更好，成为现代标准

## 4. 训练与测试时的行为差异

这是最容易踩的坑之一。BN 在 `model.train()` 和 `model.eval()` 下的行为完全不同。

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

**正确做法**：

```python
# 训练时
model.train()       # BN 使用当前 batch 的均值/方差
output = model(trainX)

# 测试/推理时
model.eval()        # BN 使用训练期间累积的移动平均
with torch.no_grad():
    output = model(testX)
```

**错误示范**（常见 Bug）：

```python
# 错误：忘记切换到 eval 模式
model.train()
with torch.no_grad():
    testOutput = model(x)  # BN 使用了错误的统计量！
```

这会导致测试准确率明显偏低，且每次运行结果都不同。

### 4.1 几个实用参数

**`eps`**：防止分母为零的小常数，默认 `1e-5`。一般不需要修改。

**`momentum`**：移动平均的更新速度。如前所述，PyTorch 的 momentum 定义和论文相反（PyTorch 的 0.1 ≈ 论文的 0.9）。

**`track_running_stats`**：是否跟踪全局统计量。默认 `True`。如果设为 `False`，测试时也使用 batch 统计量（通常只在调试时用）。

## 5. 小 batch size 问题与替代方案

BatchNorm 的效果依赖于 batch size。当 `batch_size` 很小（比如 2 或 4）时，batch 统计量的噪声很大，BN 效果会明显下降。

| 归一化方法 | 归一化维度 | 适用场景 |
| --- | --- | --- |
| BatchNorm | 沿 batch 维度 | CNN，batch_size ≥ 16 |
| LayerNorm | 沿 feature 维度 | Transformer, RNN |
| InstanceNorm | 沿空间维度（单样本） | 风格迁移，图像生成 |
| GroupNorm | 通道分组归一化 | batch_size 极小时替代 BN |

小 batch 场景下的替代方案：

```python
# 从 nn.BatchNorm2d(128)
# 改为
nn.GroupNorm(num_groups=32, num_channels=128)
```

### 5.1 BN 不适合的情况

- **Batch size = 1**（用 LayerNorm/InstanceNorm 替代）
- **RNN 等序列模型**（用 LayerNorm）
- **生成模型中对噪声敏感的任务**

### 5.2 删除 BN 的场景

在需要精确重现（如量化推理）或对 batch 依赖敏感的少数场景下，可以考虑完全移除 BN，用其他方式替代（如权重标准化）。

## 6. BatchNorm 与 Dropout 的交互

BatchNorm 与 Dropout 经常在同一网络中共用，但两者在 train/eval 模式下的行为差异容易引发问题，需特别注意。

### 6.1 放置顺序

当 BN 和 Dropout 同时使用时，推荐顺序为：

```
Conv2d → BatchNorm2d → ReLU → Dropout
```

Dropout 放在激活函数之后，对激活值进行随机丢弃。BN 在激活函数之前做归一化，两者各司其职。

### 6.2 方差偏移（Variance Shift）问题

训练时 Dropout 会随机将部分神经元置零，这改变了后续层的输入方差。但在测试时 Dropout 被关闭（或缩放），导致训练和测试间出现"方差偏移"。BN 的 running mean/var 是在训练过程中累积的，可能也因此受到影响。

实践中，可以先单独调好 BN 的放置和参数，再加入 Dropout 并适当降低 Dropout 概率（如从 0.5 降至 0.2~0.3），观察验证集表现来微调。

### 6.3 train/eval 模式一致性

BN 和 Dropout 都会根据 `model.train()` / `model.eval()` 切换行为：

| 组件 | `model.train()` | `model.eval()` |
| --- | --- | --- |
| BatchNorm | 使用 batch 统计量，更新 running mean/var | 使用全局统计量 |
| Dropout | 随机丢弃神经元 | 关闭，不丢弃 |

因此，在验证/测试阶段**必须**调用 `model.eval()` 以同时切换 BN 和 Dropout 的行为，否则两者都会产生错误。

更多 Dropout 的细节可参考 [[NeuralNetwork/Training/Dropout|Dropout 详解]]。

## 7. PyTorch 代码示例

### 7.1 基础用法：构建带 BN 的卷积块

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

### 7.2 冻结 BN 进行微调

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

## 8. 总结

- BN 的核心是让每一层的输入分布保持稳定，解决 Internal Covariate Shift 问题
- BN 的四步算法：计算均值、计算方差、归一化、缩放平移（$\gamma$ 和 $\beta$ 可学习）
- **训练时用 batch 统计量，测试时用全局统计量**——`model.eval()` 不能忘
- 现代标准位置：$\text{Conv} \to \text{BN} \to \text{ReLU}$
- 小 batch 场景用 GroupNorm 或 LayerNorm 替代
- BN 与 Dropout 共用时，注意两者的 train/eval 行为一致性

