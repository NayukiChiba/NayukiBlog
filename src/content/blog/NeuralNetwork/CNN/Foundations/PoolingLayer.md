---
title: 池化层详解：缩小图像，保留特征
date: 2026-05-07
category: NeuralNetwork/CNN/Foundations
tags:
  - 基础
  - 池化
description: 从 Max Pooling 的手算例子到 Global Average Pooling，理解池化层的三种核心作用及选型原则。
image: https://img.yumeko.site/file/blog/PoolingLayer.png
status: published
---

## 1. 为什么需要池化？

- **降维**：减少特征图的空间尺寸，降低后续层的参数量和计算量
- **平移不变性**：小幅度的平移、旋转不会改变池化的输出（因为池化只看局部最大值/平均值）
- **扩大感受野**：经过池化后，同样 $3 \times 3$ 的 [[NeuralNetwork/CNN/Foundations/ConvolutionLayer|卷积层]] 卷积核能覆盖输入图像的更大区域

## 2. 最大池化（Max Pooling）

选取每个窗口中的最大值。

输入为 $4 \times 4$ 矩阵，池化窗口 $2 \times 2$，stride=2：

$$
\text{Input} = \begin{pmatrix}
1 & 3 & 2 & 4 \\
5 & 6 & 7 & 8 \\
4 & 2 & 1 & 0 \\
1 & 5 & 8 & 9
\end{pmatrix}
$$

将输入划分为 4 个不重叠的 $2 \times 2$ 窗口，取每个窗口的最大值：

$$
\underbrace{\begin{pmatrix} 1 & 3 \\ 5 & 6 \end{pmatrix}}_{\max=6\;(0,0)}
\quad
\underbrace{\begin{pmatrix} 2 & 4 \\ 7 & 8 \end{pmatrix}}_{\max=8\;(0,1)}
\quad
\underbrace{\begin{pmatrix} 4 & 2 \\ 1 & 5 \end{pmatrix}}_{\max=5\;(1,0)}
\quad
\underbrace{\begin{pmatrix} 1 & 0 \\ 8 & 9 \end{pmatrix}}_{\max=9\;(1,1)}
$$

$$
\text{Output} = \begin{pmatrix}
6 & 8 \\
5 & 9
\end{pmatrix}
$$

**"Max" 的含义**：只保留最强的激活信号，忽略较弱的响应。直观理解："这个区域有没有检测到目标特征？——有，而且强度是 X。"
## 3. 平均池化（Average Pooling）

选取每个窗口中所有值的平均值。使用相同输入：

$$
\underbrace{\begin{pmatrix} 1 & 3 \\ 5 & 6 \end{pmatrix}}_{\frac{1+3+5+6}{4}=3.75}
\quad
\underbrace{\begin{pmatrix} 2 & 4 \\ 7 & 8 \end{pmatrix}}_{\frac{2+4+7+8}{4}=5.25}
$$

$$
\underbrace{\begin{pmatrix} 4 & 2 \\ 1 & 5 \end{pmatrix}}_{\frac{4+2+1+5}{4}=3.00}
\quad
\underbrace{\begin{pmatrix} 1 & 0 \\ 8 & 9 \end{pmatrix}}_{\frac{1+0+8+9}{4}=4.50}
$$

$$
\text{Output} = \begin{pmatrix}
3.75 & 5.25 \\
3.00 & 4.50
\end{pmatrix}
$$

## 4. Max Pooling vs Average Pooling

| 特性 | Max Pooling | Average Pooling |
| --- | --- | --- |
| 保留的信息 | 最显著的特征 | 整体分布信息 |
| 对纹理的响应 | 强（提取边缘/角点） | 温和（平滑） |
| 梯度传导 | 只传给最大值位置 | 均匀传给所有位置 |
| 现代使用 | 更广泛 | 较少（被 stride=2 卷积替代） |
| 经典案例 | MNIST-CNN, VGG, ResNet | LeNet-5 |

![Pooling.png](https://img.yumeko.site/file/articles/CNN/Pooling.png)

## 5. Global Average Pooling（GAP）

![GAP.png](https://img.yumeko.site/file/articles/PoolingLayer/GAP.png)
GAP 是 [[NeuralNetwork/CNN/Architectures/NiN|NiN]]（Network in Network）在 2014 年提出的技术，对**整个特征图**取平均，每个通道输出一个值。

```
特征图：(Batch, 512, 7, 7)
    ↓ AdaptiveAvgPool2d(1)
输出：  (Batch, 512, 1, 1)
    ↓ Flatten
特征向量：(Batch, 512)
```

**GAP 的三大优势**：

1. **替代全连接层**：大幅减少参数量。假设最后一层特征图是 $512 \times 7 \times 7$，用 GAP 后直接接分类器，参数仅 $512 \times \text{num\_classes}$。如果用 FC，需要 $512 \times 7 \times 7 \times 1024 \approx 2500$ 万参数。
2. **天然的正则化**：没有可学习参数，不会过拟合。
3. **空间信息编码**：每个通道的平均值代表了该特征在空间上的整体响应强度。

**使用示例**：

```python
# 在特征提取之后，分类之前
self.gap = nn.AdaptiveAvgPool2d(1)  # 输出 (B, C, 1, 1)

# forward
x = self.features(x)   # (B, 512, 7, 7)
x = self.gap(x)        # (B, 512, 1, 1)
x = x.flatten(1)       # (B, 512)
x = self.classifier(x) # (B, num_classes)
```

现代架构中，GoogLeNet、ResNet、DenseNet 等都在分类层之前使用了 GAP。

## 6. Adaptive Pooling

PyTorch 提供了自适应池化，无需手动计算窗口大小：

```python
# 无论输入特征图是多大，都输出指定的空间尺寸
nn.AdaptiveMaxPool2d(1)     # 输出 1×1 → 等价于 GMP
nn.AdaptiveAvgPool2d(1)     # 输出 1×1 → 等价于 GAP
nn.AdaptiveAvgPool2d((4, 4)) # 输出 4×4
```

这对于处理变尺寸输入非常有用——不需要提前知道特征图的具体大小。
![AdaptivePooling.png](https://img.yumeko.site/file/articles/PoolingLayer/AdaptivePooling.png)

## 7. 现代趋势：用 Stride 卷积替代池化

近年来，越来越多的架构用 **stride=2 的卷积**替代池化层来做下采样。因为 stride 卷积有可学习的参数，能够自适应地学习最优的下采样方式。

```python
# 传统方式：Conv + Pool
nn.Conv2d(64, 128, 3, padding=1)
nn.MaxPool2d(2, 2)

# 现代方式：stride=2 的卷积
nn.Conv2d(64, 128, 3, stride=2, padding=1)
```

但池化层仍然广泛使用，尤其是在资源受限的场景下（无参数、计算量小）。
![StrideConv.png](https://img.yumeko.site/file/articles/PoolingLayer/StrideConv.png)

## 8. 输出尺寸公式

池化层的输出尺寸公式与卷积类似（但没有 padding 时更简单）：

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K}{S} \right\rfloor + 1
$$

对于最常用的 `kernel_size=2, stride=2`：输入尺寸每次减半，如 $28 \to 14 \to 7$。

关于 GAP 的开创性应用和完整架构，参见 [[NeuralNetwork/CNN/Architectures/NiN|NiN]]。
