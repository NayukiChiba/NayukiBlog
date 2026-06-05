---
title: GoogLeNet：多尺度特征融合的智慧
date: 2026-05-23
category: 神经网络/CNN
tags:
  - 经典架构
description: GoogLeNet 通过 Inception 模块实现多尺度特征并行提取，仅 7M 参数即获得 2014 年 ImageNet 冠军。本文详解其设计智慧。
image: https://img.yumeko.site/file/blog/cover/1780581760709.webp
status: published
---

## 1. 名字的由来

GoogLeNet 是来自 Google 的团队（Szegedy 等人）在 2014 年 CVPR 上提出的架构，在当年 ImageNet 分类任务中击败 VGG 获得冠军。其名字是对 LeNet 的致敬（"Le" → "GoogLe"），论文标题是《Going Deeper with Convolutions》。

GoogLeNet 的核心是 Inception 模块——其灵感来自 [[NeuralNetwork/CNN/NiN|NiN]] 的 1x1 卷积和微型网络思想。

## 2. 核心问题：如何同时捕获不同尺度的特征？

图像中的物体大小各异——一张照片中可能有占据画面 80% 的猫脸，也可能有占据 5% 的远处行人。用单一尺寸的卷积核（比如只有 $3 \times 3$ 或只有 $5 \times 5$）难以同时适应不同尺度的特征。

GoogLeNet 的答案：**每层同时使用多种卷积核，让网络自己学习该用哪个尺度。**

## 3. Inception 模块（v1）

一个 Inception 模块包含 4 条并行分支：

![Inception.png](https://img.yumeko.site/file/blog/articles/1780581182132.webp)

各分支的作用：

| 分支 | 设计意图 |
| --- | --- |
| $1 \times 1$ 卷积 | 提取最精细的局部特征 |
| $1 \times 1 \to 3 \times 3$ | 中等感受野的通用特征 |
| $1 \times 1 \to 5 \times 5$ | 大感受野的宏观特征 |
| MaxPool $\to 1 \times 1$ | 保留原始信号中最强的局部激活 |

所有分支的 $1 \times 1$ 卷积先做**降维**（减少通道数），将计算量控制住后再做昂贵的 $3 \times 3$ 和 $5 \times 5$ 卷积。没有这个降维步骤，Inception 模块的计算量将爆炸。

## 4. 完整架构

```
Input (3, 224, 224)
    ↓ Stem（预处理）
(192, 28, 28)
    ↓ Inception 3a → 3b → MaxPool
(480, 14, 14)
    ↓ Inception 4a → 4b → 4c → 4d → 4e → MaxPool
(832, 7, 7)
    ↓ Inception 5a → 5b
(1024, 7, 7)
    ↓ GAP → Dropout(0.4) → FC(1024→1000)
Output
```

**Stem 结构**（在 Inception 之前）：

| 层 | 操作 | 输出形状 |
| --- | --- | --- |
| Conv | 7×7, stride=2, BN+ReLU | $(64, 112, 112)$ |
| Pool | MaxPool 3×3, stride=2 | $(64, 56, 56)$ |
| Conv | 1×1, BN+ReLU | $(64, 56, 56)$ |
| Conv | 3×3, BN+ReLU | $(192, 56, 56)$ |
| Pool | MaxPool 3×3, stride=2 | $(192, 28, 28)$ |

Stem 用较大的卷积（$7 \times 7$）和池化快速降低空间尺寸，将 $224 \times 224$ 的图像压缩到 $28 \times 28$，然后交给 9 个 Inception 模块做精细的特征提取。

![GoogleNet.png](https://img.yumeko.site/file/blog/articles/1780581181713.webp)

## 5. 辅助分类器

GoogLeNet 在中间层添加了两个辅助分类器（位于 Inception 4a 和 4d 之后），在训练时以较小权重（0.3）参与损失计算：

```python
# 辅助分类器
AuxClassifier = nn.Sequential(
    nn.AvgPool2d(5, stride=3),
    nn.Conv2d(..., 128, 1),
    nn.Linear(..., 1024),
    nn.Linear(1024, num_classes)
)
```

**作用**：
- 向中间层注入梯度信号，缓解深层网络的梯度消失问题
- 提供额外的正则化

推理时辅助分类器被移除。

## 6. Inception 参数表

| 模块 | 输入通道 | $c_1$ | $c_2$ (降维/输出) | $c_3$ (降维/输出) | $c_4$ | 输出通道 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3a | 192 | 64 | 96/128 | 16/32 | 32 | 256 |
| 3b | 256 | 128 | 128/192 | 32/96 | 64 | 480 |
| 4a | 480 | 192 | 96/208 | 16/48 | 64 | 512 |
| 4b | 512 | 160 | 112/224 | 24/64 | 64 | 512 |
| 4c | 512 | 128 | 128/256 | 24/64 | 64 | 512 |
| 4d | 512 | 112 | 144/288 | 32/64 | 64 | 528 |
| 4e | 528 | 256 | 160/320 | 32/128 | 128 | 832 |
| 5a | 832 | 256 | 160/320 | 32/128 | 128 | 832 |
| 5b | 832 | 384 | 192/384 | 48/128 | 128 | 1024 |

其中 $c_1, c_2, c_3, c_4$ 分别对应 4 个分支的输出通道数。

## 7. 为什么 22 层仅 7M 参数？

GoogLeNet 的参数效率秘诀：

1. **$1 \times 1$ 降维**：在昂贵的 $3 \times 3$ 和 $5 \times 5$ 卷积前先用 $1 \times 1$ 卷积减少通道数
2. **GAP 替代 FC**：分类器几乎零参数
3. **无大 FC 层**：AlexNet 的 FC 层占了 96% 的参数

| 架构 | 层数 | 参数量 |
| --- | :---: | ---: |
| AlexNet | 8 | 62M |
| VGG-16 | 16 | 138M |
| **GoogLeNet** | **22** | **~7M** |

## 8. 后续 Inception 版本

| 版本 | 年份 | 主要改进 |
| --- | --- | --- |
| Inception v1 | 2014 | 原始 GoogLeNet，4 分支 |
| Inception v2 | 2015 | 添加 BN，$5 \times 5$ 替换为两个 $3 \times 3$ |
| Inception v3 | 2015 | 卷积核进一步因式分解（$7 \times 7 \to 1 \times 7 + 7 \times 1$） |
| Inception v4 | 2016 | 结合 ResNet 的 skip connection |

2015 年的 [[NeuralNetwork/CNN/ResNet|ResNet]] 用 skip connection 进一步突破了深度瓶颈，将层数推至 152+。完整的演进历史参见 [[NeuralNetwork/CNN/ArchitectureComparison|架构对比]]。
