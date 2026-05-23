---
title: 全连接层详解：从特征到分类
date: 2026-05-07
category: NeuralNetwork/CNN/Foundations
tags:
  - 深度学习
  - 基础
description: 理解全连接层在 CNN 中的角色、参数占比问题，以及现代架构中 GAP 和 1×1 卷积的替代方案。
image: https://img.yumeko.site/file/blog/FullyConnectedLayer.png
status: published
---

## 1. 为什么 CNN 需要全连接层？

卷积层提取的是**空间化的特征图**——特征以二维（或三维）的形式组织。但分类需要的是一个**全局判断**——"这张图整体上属于哪个类？"

全连接层的作用是把分布在各处的特征"汇总"起来，形成一个全局的判断。

## 2. Flatten：从三维到一维

在全连接层之前，需要把三维的特征图（通道 $\times$ 高 $\times$ 宽）拉平成一维向量。
![Flatten.png](https://img.yumeko.site/file/articles/FullyConnectedLayer/Flatten.png)

```
特征图：(Batch, 64, 7, 7)    →    Flatten    →    向量：(Batch, 3136)
                                                      ↑
                                               64 × 7 × 7 = 3136
```

Flatten 本身没有可学习参数，它只是改变了张量的形状。

## 3. 全连接层的计算

$$
y = Wx + b
$$

- $W$ 的形状是 $(\text{out\_features}, \text{in\_features})$
- $x$ 的形状是 $(\text{in\_features},)$
- $b$ 的形状是 $(\text{out\_features},)$

参数量：$\text{in\_features} \times \text{out\_features} + \text{out\_features}$

## 4. 参数占比问题

以一个典型的 CNN 为例（MNIST-CNN 的 FC 部分）：

```
Flatten (3136) → Linear(3136→128) → BN → ReLU → Dropout → Linear(128→10)
```

| 层 | 计算 | 参数量 | 占比 |
| --- | --- | ---:| ---:|
| FC1 | $3136 \times 128 + 128$ | 401,536 | **95.0%** |
| FC2 | $128 \times 10 + 10$ | 1,290 | 0.3% |
| 全部卷积层 | — | ~20,000 | 4.7% |

**全连接层承载了 95% 以上的参数！** 这是 CNN 的经典现象：卷积层通过权重共享大幅减少了参数，而全连接层每个输入-输出之间都有独立的连接，参数数量激增。

## 5. 全连接层的三大问题

1. **参数量过大**：如上表所示，FC 层占用了绝大多数参数，导致模型体积大、容易过拟合
2. **空间信息丢失**：Flatten 操作将空间结构完全破坏，无法保留特征的空间位置关系
3. **输入尺寸固定**：FC 层的权重矩阵是固定大小的，这意味着模型只能接受固定尺寸的输入

## 6. 替代方案一：Global Average Pooling

GAP 是 [[NeuralNetwork/CNN/Architectures/NiN|NiN]]（2014）提出的最经典的 FC 替代方案，利用 [[NeuralNetwork/CNN/Foundations/PoolingLayer|池化层]] 的全局平均替代全连接。对每个通道的特征图取全局平均，直接作为分类器的输入：

```python
# 传统 FC 方式
x = x.flatten(1)               # (B, 512, 7, 7) → (B, 25088)
x = nn.Linear(25088, 1024)(x)  # 参数：25088×1024 ≈ 25.7M

# GAP 方式
x = nn.AdaptiveAvgPool2d(1)(x) # (B, 512, 7, 7) → (B, 512, 1, 1)
x = x.flatten(1)               # (B, 512)
x = nn.Linear(512, 1024)(x)    # 参数：512×1024 ≈ 0.5M
```

参数量减少约 50 倍，且不再受输入尺寸限制。

![GAP.png](https://img.yumeko.site/file/articles/FullyConnectedLayer/GAP.png)

## 7. 替代方案二：1×1 卷积替代 FC


如果不想丢失空间信息，可以用 $1 \times 1$ 卷积实现与 FC 等价的功能：

```python
# 传统 FC
nn.Linear(512, 10)             # 输入必须是 (B, 512)

# 1×1 卷积等价（可接受任意空间尺寸）
nn.Conv2d(512, 10, kernel_size=1)  # 输入可以是 (B, 512, H, W)
```

$1 \times 1$ 卷积只在通道维度上做变换，不改变空间尺寸。它等价于对每个空间位置的通道向量做一次全连接操作。全卷积网络（FCN）正是利用了这一点，完全抛弃了 FC 层。
![1x1Conv.png](https://img.yumeko.site/file/articles/FullyConnectedLayer/1x1Conv.png)

## 8. 现代架构的 FC 使用趋势

| 架构 | FC 使用 | 策略 |
| --- | --- | --- |
| LeNet-5 (1998) | 3 层 FC | 传统方式 |
| AlexNet (2012) | 3 层 FC（含 4096 维巨层） | 大量 Dropout |
| VGG (2014) | 3 层 FC | 参数极多 |
| NiN (2014) | **无 FC** | 全部用 GAP + 1×1 卷积 |
| GoogLeNet (2014) | 1 层 FC（1024→1000） | GAP 后接单层 FC |
| ResNet (2015) | **仅 1 层 FC**（分类头） | GAP 后直接 FC |
| MobileNet (2017) | **仅 1 层 FC** | GAP 后直接 FC |

现代趋势很明确：**先用 GAP 将特征图压缩为每个通道一个值，再接一个线性层做分类**。参数量最小、灵活性最高、不易过拟合。

## 9. 代码对比

```python
# 传统方式：大 FC 层
class TraditionalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25088, 4096)  # 巨量参数
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

# 现代方式：GAP + 单层 FC
class ModernHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)     # 极少量参数
```

GAP 替代 FC 已成为现代架构的标准做法，详见 [[NeuralNetwork/CNN/Architectures/NiN|NiN]] 的完整设计。
