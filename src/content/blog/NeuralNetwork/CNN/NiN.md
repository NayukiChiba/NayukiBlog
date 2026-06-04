---
title: NiN：用微型网络替代简单卷积
date: 2026-05-23
category: NeuralNetwork/CNN
tags:
  - 经典架构
description: Network in Network 提出了 mlpconv 和 Global Average Pooling 两项关键创新，深刻影响了后续的 GoogLeNet 和 ResNet。
image: https://img.yumeko.site/file/blog/NiN.png
status: published
---

## 1. 核心思想

NiN（Network in Network）由林敏等人在 2014 年 ICLR 上发表。虽然 NiN 本身的性能后来被超越，但它提出的两项创新深刻影响了整个 CNN 领域：

1. **mlpconv 层**：用微型多层感知机（MLP）替代简单的卷积操作
2. **全局平均池化（GAP）**：替代全连接层做分类

## 2. mlpconv：卷积中的"网络"

传统卷积层只能做**线性变换**——卷积核在输入上滑动，每次计算一个加权和。NiN 的 mlpconv 将这个简单的加权和替换为一个微型 MLP：

![NiN.png](https://img.yumeko.site/file/articles/NiN/NiN.png)

```
传统卷积：    Conv(k×k) → ReLU
mlpconv：    Conv(k×k) → ReLU → Conv(1×1) → ReLU → Conv(1×1) → ReLU
```

**$1 \times 1$ 卷积的作用**：
- 在通道维度上做全连接（每个空间位置独立）
- 增加非线性（每个 $1 \times 1$ 卷积后跟 ReLU）
- 不改变空间尺寸

直觉理解：传统卷积在每个空间位置只做一次"决策"（一个加权和 + ReLU），而 mlpconv 在每个位置构造了一个小型的多层网络来做"更复杂的决策"。

### 参数量分析

nin_block(192, kernel=5) 的参数量：

| 层 | 计算 | 参数量 |
| --- | --- | ---:|
| Conv(5×5): 192→192 | $192 \times 192 \times 25 + 192$ | 921,792 |
| Conv(1×1): 192→192 | $192 \times 192 \times 1 + 192$ | 37,056 |
| Conv(1×1): 192→192 | $192 \times 192 \times 1 + 192$ | 37,056 |

注意：$1 \times 1$ 卷积的参数量远小于 $5 \times 5$ 卷积。

## 3. 全局平均池化（GAP）：革命性的替代方案

在 NiN 之前，所有 CNN 都使用 [[NeuralNetwork/CNN/CNNCoreLayers|全连接层]] 做最终分类。NiN 首次提出用 GAP 完全替代 FC 层：
![GAP.png](https://img.yumeko.site/file/articles/NiN/GAP.png)

```python
# 传统分类器（如 AlexNet/VGG）
Flatten → FC(大维度) → FC(大维度) → FC(num_classes)   # 几千万参数

# NiN 的分类器
mlpconv → AdaptiveAvgPool2d(1) → Flatten              # 几乎零参数
```

**GAP 的三大优势**：

1. **零参数量**：GAP 没有可学习参数，直接消除了全连接层带来的巨量参数
2. **天然正则化**：没有参数就不会过拟合
3. **空间信息编码**：每个通道的平均值代表了该特征在空间上的整体响应——通道本身就对应了"某个类别在图像中出现的程度"

注意 NiN 的分类器仍然包含了一个 mlpconv（`nin_block(96→num_classes, k=3)`），再接 GAP。这个最后的 mlpconv 将倒数第二层的 96 个特征通道映射为 num_classes 个通道，每个通道对应一个类别——GAP 后直接得到每个类别的置信度。

## 4. 完整架构

| Stage      | 操作                                     | 输入形状            | 输出形状            |
| ---------- | -------------------------------------- | --------------- | --------------- |
| —          | Input                                  | —               | $(3, 32, 32)$   |
| Stage 1    | nin_block(3→192, k=5) + MaxPool(3×3)   | $(3, 32, 32)$   | $(192, 15, 15)$ |
| Stage 2    | nin_block(192→160, k=5) + MaxPool(3×3) | $(192, 15, 15)$ | $(160, 6, 6)$   |
| Stage 3    | nin_block(160→96, k=3) + MaxPool(3×3)  | $(160, 6, 6)$   | $(96, 2, 2)$    |
| Classifier | nin_block(96→num_classes, k=3) + GAP   | $(96, 2, 2)$    | (num_classes,)  |
![FullStage.png](https://img.yumeko.site/file/articles/NiN/FullStage.png)
## 5. PyTorch 实现

```python
class nin_block(nn.Module):
    """mlpconv 块：Conv → ReLU → 1×1 Conv → ReLU → 1×1 Conv → ReLU"""
    def __init__(self, inCh, outCh, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(inCh, outCh, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(outCh, outCh, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outCh, outCh, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stage1 = nn.Sequential(
            nin_block(3, 192, 5),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage2 = nn.Sequential(
            nin_block(192, 160, 5),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage3 = nn.Sequential(
            nin_block(160, 96, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nin_block(96, num_classes, 3),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x.flatten(1)
```

## 6. 对后续架构的影响

NiN 的两项创新被后来的架构广泛采纳：

| 创新                   | 采纳架构                                                    |
| -------------------- | ------------------------------------------------------- |
| $1 \times 1$ 卷积降维    | GoogLeNet, ResNet, DenseNet                     |
| $1 \times 1$ 卷积增加非线性 | GoogLeNet, ResNet, Bottleneck                   |
| GAP 替代 FC            | **几乎所有现代架构**（GoogLeNet, ResNet, DenseNet, MobileNet...） |

可以说不夸张地说：**如果用一个词概括 NiN 对 CNN 发展史的影响，那就是——GAP。** 完整的架构演进脉络见 [[NeuralNetwork/CNN/ArchitectureComparison|架构对比]]。

NiN 的 GAP 和 1x1 卷积直接启发了 [[NeuralNetwork/CNN/GoogLeNet|GoogLeNet]] 的 Inception 设计，并在几乎所有现代架构中得到延续。
