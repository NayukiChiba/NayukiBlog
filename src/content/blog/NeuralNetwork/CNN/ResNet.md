---
title: ResNet：残差学习的革命
date: 2026-05-23
category: 神经网络/CNN
tags:
  - 经典架构
description: ResNet 用残差连接解决了深层网络的退化问题，使得 152 层的网络训练成为可能。本文详解其核心思想和各类变体。
image: https://img.yumeko.site/file/blog/cover/1780581861848.webp
status: published
---

## 1. 退化问题：为什么更深不总是更好？

在 ResNet 提出之前（2015 年），一个直觉是：网络越深，表达能力越强，效果应该越好。但实验发现了一个反直觉的现象——**退化（degradation）**：

> 56 层的网络在训练集和测试集上的误差都比 20 层更高。

![Degeneration.png](https://img.yumeko.site/file/blog/articles/1780581172543.webp)

注意这不是过拟合（过拟合是训练误差低、测试误差高），而是**欠拟合**——更深的网络连训练集都学不好。

**原因**：深层网络的梯度在反向传播中反复乘以权重矩阵，加上非线性激活，导致梯度消失或信息衰减。深层网络实际上比浅层网络更难优化。

## 2. 残差学习的核心思想

ResNet 的解决方案优雅而简洁：**让网络学习"残差"而非直接学习目标映射。**

假设我们想让某一层学习映射 $\mathcal{H}(x)$。与其直接学习 $\mathcal{H}(x)$，不如学习残差：

$$
\mathcal{F}(x) = \mathcal{H}(x) - x
$$

然后通过 shortcut 把输入加回来：

$$
\text{Output} = \mathcal{F}(x) + x
$$

**直觉**：如果最优映射接近恒等变换（大部分信息直接传递，只需少量修改），那学习恒等映射附近的残差比从头学整个映射容易得多。在极端情况下，如果恒等变换就是最优的，只需把所有残差推为零。

## 3. 残差块（Residual Block）
![ResBlock.png](https://img.yumeko.site/file/blog/articles/1780581201285.webp)

数学表达：

$$
y = \text{ReLU}(\mathcal{F}(x, \{W_i\}) + x)
$$

其中 $\mathcal{F}(x, \{W_i\})$ 是两个卷积层学习的残差映射，$x$ 通过 shortcut 直接连接到加法节点。

**Shortcut 的两种形式**：

| 形式 | 条件 | 操作 |
| --- | --- | --- |
| Identity Shortcut | 输入输出维度相同 | $y = \mathcal{F}(x) + x$（直接相加） |
| Projection Shortcut | 输入输出维度不同（通道数变化、空间尺寸减半） | $y = \mathcal{F}(x) + W_s x$（$1 \times 1$ 卷积匹配维度） |

![ResNet34.png](https://img.yumeko.site/file/blog/articles/1780581196703.webp)

**维度匹配的 $1 \times 1$ 卷积 shortcut**：

```python
# 当 stride=2 且通道数加倍时，shortcut 需要调整维度
if stride != 1 or inCh != outCh:
    self.shortcut = nn.Conv2d(inCh, outCh, kernel_size=1, stride=stride)
```

## 4. Bottleneck 设计

对于深层 ResNet（50+ 层），使用 Bottleneck 设计替代基础块来减少计算量：

```
Bottleneck:    1×1(降维) → 3×3 → 1×1(升维)
Basic:                   3×3 → 3×3
```

参数量对比（输入输出通道均为 256）：

| 块类型 | 计算 | 参数量 |
| --- | --- | ---:|
| Basic | $2 \times 256^2 \times 3^2$ | ~1.2M |
| Bottleneck | $256 \times 64 \times 1^2 + 64 \times 64 \times 3^2 + 64 \times 256 \times 1^2$ | ~70K |

Bottleneck 将参数量降低了约 17 倍！这使得 152 层的 ResNet（参数量 60M）反而比 [[NeuralNetwork/CNN/VGG|VGG]]-16（138M）更轻量。

## 5. ResNet 家族

| 变体 | 层数 | 块结构 | 每 stage 块数 | 参数量 |
| --- | :---: | --- | :---: | ---: |
| ResNet-18 | 18 | Basic | [2,2,2,2] | 11.7M |
| ResNet-34 | 34 | Basic | [3,4,6,3] | 21.8M |
| ResNet-50 | 50 | Bottleneck | [3,4,6,3] | 25.6M |
| ResNet-101 | 101 | Bottleneck | [3,4,23,3] | 44.5M |
| ResNet-152 | 152 | Bottleneck | [3,8,36,3] | 60.2M |

**Stage 结构**（以 ResNet-50 为例）：

```
Input (3, 224, 224)
    ↓ Conv(7×7, s=2) + BN + ReLU + MaxPool(3×3, s=2)
(64, 56, 56)
    ↓ Stage 1: Bottleneck×3    通道 64→256, stride=1
(256, 56, 56)
    ↓ Stage 2: Bottleneck×4    通道 256→512, stride=2
(512, 28, 28)
    ↓ Stage 3: Bottleneck×6    通道 512→1024, stride=2
(1024, 14, 14)
    ↓ Stage 4: Bottleneck×3    通道 1024→2048, stride=2
(2048, 7, 7)
    ↓ GAP → FC(2048→1000)
Output
```
![ResNet50.png](https://img.yumeko.site/file/blog/articles/1780581196011.webp)
**关键设计**：每个 Stage 的第一个 Bottleneck 负责空间降维（stride=2）和通道提升，后面的 Bottleneck 在相同维度下重复。

## 6. 为什么 ResNet 能训这么深？

### 梯度高速公路

残差连接提供了梯度传播的"高速公路"：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)
$$

那个常数 $1$ 来自于 shortcut 的恒等映射——它确保即使在极端深层，梯度也不会衰减到 0。梯度至少以 $1$ 的倍率传播。

### 易于优化的损失曲面

实验发现，ResNet 的损失曲面比 VGG 等普通网络更平滑，优化器更容易找到好的最小值。残差连接的直观效果就是让损失曲面"变好"。与 [[NeuralNetwork/CNN/GoogLeNet|GoogLeNet]] 的 Inception 多分支不同，ResNet 通过简单的加法跳跃实现了更深层的训练。

## 7. PyTorch 实现

```python
class BasicBlock(nn.Module):
    """ResNet-18/34 的基础残差块"""
    expansion = 1

    def __init__(self, inCh, outCh, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inCh, outCh, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(outCh)
        self.conv2 = nn.Conv2d(outCh, outCh, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(outCh)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut：维度不匹配时用 1×1 卷积
        if stride != 1 or inCh != outCh:
            self.shortcut = nn.Conv2d(inCh, outCh, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity     # 残差连接
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet-50/101/152 的瓶颈残差块"""
    expansion = 4

    def __init__(self, inCh, outCh, stride=1):
        super().__init__()
        midCh = outCh
        self.conv1 = nn.Conv2d(inCh, midCh, 1)      # 降维
        self.bn1 = nn.BatchNorm2d(midCh)
        self.conv2 = nn.Conv2d(midCh, midCh, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(midCh)
        self.conv3 = nn.Conv2d(midCh, outCh * self.expansion, 1)  # 升维
        self.bn3 = nn.BatchNorm2d(outCh * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or inCh != outCh * self.expansion:
            self.shortcut = nn.Conv2d(
                inCh, outCh * self.expansion, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity  # 残差连接
        return self.relu(out)
```

## 8. BatchNorm 在 ResNet 中的位置

ResNet 在每个卷积后都用了 BN，顺序为：

$$
\text{Conv} \to \text{BN} \to \text{ReLU}
$$

这与现代标准一致。注意在残差块的最后，是**先加残差、后 ReLU**：`out = relu(out + identity)`。BN 在每个卷积后立即应用，但在加法之前。

## 9. 权重初始化

ResNet 使用 Kaiming 初始化（配合 ReLU），BN 的 $\gamma$ 初始化为 1、$\beta$ 初始化为 0。没有使用任何特殊的初始化技巧——残差连接本身已经极大地缓解了梯度问题。

## 10. ResNet 的影响

ResNet 是 CNN 历史上影响最深远的架构之一：

- 2015 年 ImageNet 全部赛道的冠军
- **残差连接成为几乎所有后续架构的基础组件**（DenseNet、ResNeXt、EfficientNet）
- 残差连接的思想被迁移到了 Transformer（残差连接 + LayerNorm），构成了现代大语言模型的基础块
- 今天，"带残差连接的模块化设计"已成为深度学习架构设计的默认范式

## 11. 后续变体

| 变体 | 年份 | 主要改进 |
| --- | --- | --- |
| ResNeXt | 2017 | 分组卷积 + cardinality 维度 |
| DenseNet | 2017 | 密集连接（每层连接所有前层） |
| SE-ResNet | 2018 | 添加通道注意力（Squeeze-and-Excitation） |
| ResNeSt | 2020 | 分组注意力 + Split-Attention |

ResNet 的残差连接思想已成为深度学习架构设计的默认范式，从 [[NeuralNetwork/CNN/GoogLeNet|GoogLeNet]] 的并行多分支到 ResNet 的残差跳跃，每一次架构演进都在解决不同的核心矛盾。
