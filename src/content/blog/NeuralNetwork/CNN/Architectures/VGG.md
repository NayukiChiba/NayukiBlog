---
title: VGG：小卷积核的胜利
date: 2026-05-23
category: NeuralNetwork/CNN/Architectures
tags:
  - VGG
  - 经典架构
description: VGG 证明了"全部使用 3×3 小卷积核"的简洁设计哲学，本文详解 VGG-11/13/16/19 的结构和设计智慧。
image: https://img.yumeko.site/file/blog/VGG.png
status: draft
---

## 1. 设计哲学

VGG（Visual Geometry Group）由牛津大学的 Simonyan 和 Zisserman 在 2015 年 ICLR 上发表。其核心设计理念极其简洁：

> **全部使用 $3 \times 3$ 卷积核（stride=1, padding=1），通过堆叠深度来控制模型容量。**

这种极简设计带来了一个优雅的数学性质：

- 两个 $3 \times 3$ 卷积的感受野 = $5 \times 5$
- 三个 $3 \times 3$ 卷积的感受野 = $7 \times 7$

**为什么堆叠小卷积核优于一个大卷积核？** 以感受野 $5 \times 5$ 为例：

| | 一个 $5 \times 5$ 卷积 | 两个 $3 \times 3$ 卷积 |
| --- | :---: | :---: |
| 参数量 | $C \times C \times 25$ | $2 \times C \times C \times 9$ |
| 参数比 | $25/18 \approx 1.39\times$ | 更少 |
| 非线性层 | 1 个 ReLU | 2 个 ReLU |
| 表达能力 | 单次线性变换 | 更强的非线性 |

多一个 ReLU 意味着更强的非线性表达能力，而参数反而更少。

## 2. 架构概览

VGG 将网络分为 5 个 stage，每个 stage 由若干 $3 \times 3$ 卷积 + MaxPool 组成：

```
Input (3, 224, 224)
  → Stage 1: [3×3 Conv × N1] → MaxPool   通道: 64
  → Stage 2: [3×3 Conv × N2] → MaxPool   通道: 128
  → Stage 3: [3×3 Conv × N3] → MaxPool   通道: 256
  → Stage 4: [3×3 Conv × N4] → MaxPool   通道: 512
  → Stage 5: [3×3 Conv × N5] → MaxPool   通道: 512
  → FC(25088→4096) → ReLU → Dropout
  → FC(4096→4096) → ReLU → Dropout
  → FC(4096→1000)
```

**空间尺寸变化**：$224 \to 112 \to 56 \to 28 \to 14 \to 7$（每次 MaxPool 尺寸减半）

**通道数变化**：$3 \to 64 \to 128 \to 256 \to 512 \to 512$（每次池化后通道翻倍，保持"总信息量"相对稳定）

## 3. VGG 家族：四款变体

| 变体 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | 总卷积层 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG-11 | 1 | 1 | 2 | 2 | 2 | 8 |
| VGG-13 | 2 | 2 | 2 | 2 | 2 | 10 |
| VGG-16 | 2 | 2 | 3 | 3 | 3 | 13 |
| VGG-19 | 2 | 2 | 4 | 4 | 4 | 16 |

加上 3 个全连接层，总权重层数 = 卷积层数 + 3。

![TODO: VGG架构图，多个3x3卷积堆叠后接MaxPool的重复模式，展示VGG-11到VGG-19的深度递增]

## 4. 各变体参数量

| 变体 | 参数量 |
| --- | ---:|
| VGG-11 | ~133M |
| VGG-13 | ~133M |
| VGG-16 | ~138M |
| VGG-19 | ~144M |

注意 VGG-11 和 VGG-13 参数量几乎相同——因为参数主要集中在全连接层（约占 90%），而非卷积层。

## 5. PyTorch 实现

```python
# vgg_conv 块：Conv → BN → ReLU
def vgg_conv(inCh, outCh):
    return nn.Sequential(
        nn.Conv2d(inCh, outCh, 3, padding=1),
        nn.BatchNorm2d(outCh),
        nn.ReLU(inplace=True)
    )

# VGG 配置
VGG_CONFIGS = {
    "11": [1, 1, 2, 2, 2],
    "13": [2, 2, 2, 2, 2],
    "16": [2, 2, 3, 3, 3],
    "19": [2, 2, 4, 4, 4],
}
VGG_CHANNELS = [64, 128, 256, 512, 512]


class VGG(nn.Module):
    def __init__(self, config="16", num_classes=1000):
        super().__init__()
        layersPerStage = VGG_CONFIGS[config]

        # 构建 5 个 stage
        features = []
        inCh = 3
        for outCh, nConv in zip(VGG_CHANNELS, layersPerStage):
            for _ in range(nConv):
                features.append(vgg_conv(inCh, outCh))
                inCh = outCh
            features.append(nn.MaxPool2d(2, 2))

        self.features = nn.Sequential(*features)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
```

## 6. VGG 的优缺点

**优点**：
- 设计极其简洁、统一——全部 $3 \times 3$ 卷积 + $2 \times 2$ MaxPool
- 预训练模型迁移学习效果好（VGG-16 至今仍是很多任务的 baseline）
- 容易理解和实现

**缺点**：
- **参数巨大**：VGG-16 有 138M 参数，模型文件超过 500MB
- **全连接层臃肿**：3 层 FC 占总参数的 90%+
- **计算量大**：推理速度慢，显存占用高
- 没有像 ResNet 那样的 skip connection，网络加深后优化困难

## 7. 历史定位

VGG 在 2014 年 ImageNet 中获得定位任务冠军和分类任务亚军（冠军是 GoogLeNet）。它在设计上证明了"更深 + 更小的卷积核"这一方向是正确的。同年出现的 [[NeuralNetwork/CNN/Architectures/GoogLeNet|GoogLeNet]] 在效率上更胜一筹，而 2015 年的 [[NeuralNetwork/CNN/Architectures/ResNet|ResNet]] 用 skip connection 解决了深层网络难训练的问题，彻底超越了 VGG。各架构的全面对比见 [[NeuralNetwork/CNN/Architectures/ArchitectureComparison|架构对比]]。

但 VGG 因其简洁性至今仍被广泛用于教学和迁移学习。

VGG 的简洁设计至今仍是教学典范，但其臃肿的 FC 层促使后续架构探索更高效的分类器设计。
