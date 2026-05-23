---
title: 卷积层详解：CNN 的核心运算
date: 2026-05-07
category: NeuralNetwork/CNN/Foundations
tags:
  - 基础
  - 卷积
  - 数学
description: 从逐像素计算到多通道卷积，深入理解卷积层的每一个参数和每一步运算。
image: TODO
status: draft
---

## 1. 什么是卷积操作？

**直觉理解**：卷积就是用一个小的"模板"（卷积核/滤波器），在输入图像上滑动，每次计算模板覆盖区域和模板的"匹配程度"。

**数学定义（2D 离散卷积）**：

$$
\text{Output}[i, j] = \sum_{m}\sum_{n} \text{Input}[i+m, j+n] \times \text{Kernel}[m, n]
$$

其中 $(i,j)$ 是输出位置坐标，$(m,n)$ 是卷积核内的偏移坐标。

## 2. 手算一个具体例子

假设我们有一张 $5 \times 5$ 的灰度输入图像，以及一个 $3 \times 3$ 的卷积核（这是一个检测垂直边缘的卷积核）：

$$
\text{输入图像} = \begin{pmatrix}
1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 0
\end{pmatrix}
\qquad
\text{卷积核} = \begin{pmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{pmatrix}
$$

### 计算输出位置 $[0,0]$

卷积核覆盖输入图像的左上角 $3 \times 3$ 区域：

$$
\underbrace{\begin{pmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{pmatrix}}_{\text{输入区域}}
\;\odot\;
\underbrace{\begin{pmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{pmatrix}}_{\text{卷积核}}
\;=\;
\underbrace{\begin{pmatrix}
1 & 0 & -1 \\
0 & 0 & -1 \\
0 & 0 & -1
\end{pmatrix}}_{\text{逐元素乘积}}
$$

求和：$1 + 0 + (-1) + 0 + 0 + (-1) + 0 + 0 + (-1) = -2$，所以输出 $[0,0] = -2$。

### 计算输出位置 $[0,1]$

卷积核向右滑动一格：

$$
\underbrace{\begin{pmatrix}
1 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 1
\end{pmatrix}}_{\text{输入区域}}
\;\odot\;
\underbrace{\begin{pmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{pmatrix}}_{\text{卷积核}}
\;=\;
\underbrace{\begin{pmatrix}
1 & 0 & 0 \\
1 & 0 & -1 \\
0 & 0 & -1
\end{pmatrix}}_{\text{逐元素乘积}}
$$

求和：$1 + 0 + 0 + 1 + 0 + (-1) + 0 + 0 + (-1) = 0$，所以输出 $[0,1] = 0$。

### 完整输出

当卷积核滑过全部 9 个位置后：

$$
\text{Output} = \begin{pmatrix}
-2 & 0 & 1 \\
-2 & 0 & 1 \\
-2 & 0 & 1
\end{pmatrix}
$$

注意输出中，第三列的值（$1$）明显大于其他列，这说明在输入图像的第三列附近存在**垂直边缘**——从亮（$1$）到暗（$0$）的过渡。这个简单的 $3 \times 3$ 卷积核就是一个垂直边缘检测器！

![ConvolutionOperation.png](https://img.yumeko.site/file/articles/CNN/ConvolutionOperation.png)

![TODO: 3x3卷积核在5x5输入上滑动的逐步分解图，每次点乘求和产生输出特征图一个元素]

## 3. 多通道卷积

上面的例子是单通道（灰度图）的情况。实际中，输入往往有多个通道（比如 RGB 彩色图有 3 个通道）。

**规则**：卷积核的通道数必须等于输入通道数。

对于 RGB 输入（3 通道），一个 $3 \times 3$ 的卷积核实际上是 $3 \times 3 \times 3 = 27$ 个参数，每个通道对应一个 $3 \times 3$ 的矩阵。卷积操作是在三个通道上分别做逐元素相乘，然后把三个结果相加，得到输出特征图上的一个值。

$$
\text{Output}[i,j] = \sum_{c}\sum_{m}\sum_{n} \text{Input}[c, i+m, j+n] \times \text{Kernel}[c, m, n]
$$

其中 $c$ 遍历所有输入通道。

**一个卷积核 $\to$ 一张输出特征图**。如果有 $K$ 个卷积核，输出就有 $K$ 个通道。

## 4. 关键参数

| 参数 | 含义 | 常见取值 |
| --- | --- | --- |
| `kernel_size` | 卷积核大小 | 3, 5, 7 |
| `stride` | 滑动步长 | 1（最常用）, 2（用于下采样） |
| `padding` | 边缘填充 | 0（不填充）, 1（same padding for kernel=3） |
| `in_channels` | 输入通道数 | 1（灰度）, 3（RGB） |
| `out_channels` | 输出通道数（卷积核数量） | 32, 64, 128... |

卷积层参数量计算公式：

$$
\text{Params} = \text{in\_channels} \times \text{out\_channels} \times \text{kernel\_size}^2 + \text{out\_channels}
$$

## 5. Padding 详解

**问题**：如果不做填充，每次卷积后输出尺寸会缩小。

- $5 \times 5$ 输入，$3 \times 3$ 卷积核 $\to$ $3 \times 3$ 输出（缩小了）
- 如果一个网络有 10 层卷积，$28 \times 28$ 的图像很快就会被缩到 $0 \times 0$

**解决方法**：在输入图像边缘填充 0（zero-padding），让输出尺寸不变或可控。

**"same" padding 的计算公式**：

对于大小为 $K$ 的卷积核，要维持输入输出尺寸相同，需要 $P = (K-1)/2$。

- $K=3 \to \text{padding}=1$（四周各填充一行/列）
- $K=5 \to \text{padding}=2$
- $K=7 \to \text{padding}=3$

现代 CNN 普遍使用 `kernel_size=3, padding=1`（same 卷积），卷积前后空间尺寸保持不变。

## 6. 输出尺寸计算公式

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K + 2P}{S} \right\rfloor + 1
$$

**实例验证**（$28 \times 28$ 输入，$K=3$，$P=1$，$S=1$）：

$$
H_{\text{out}} = \left\lfloor\frac{28 - 3 + 2 \times 1}{1}\right\rfloor + 1 = 27 + 1 = 28
$$

尺寸不变。

**实例验证**（$32 \times 32$ 输入，$K=5$，$P=0$，$S=1$，LeNet-5 的 C1 层）：

$$
H_{\text{out}} = \left\lfloor\frac{32 - 5 + 0}{1}\right\rfloor + 1 = 27 + 1 = 28
$$

缩小了 4 个像素。

## 7. 感受野

感受野是指输出特征图上一个像素，对应原始输入图像上多大范围的像素。

- 第一层 $3 \times 3$ 卷积：感受野 $= 3 \times 3$
- 第二层 $3 \times 3$ 卷积（叠加在第一层上）：感受野 $= 5 \times 5$

因为第二层的一个像素看到的是第一层的 $3 \times 3$ 区域，而第一层的每个像素又看到输入的 $3 \times 3$，所以总感受野是 $5 \times 5$。

**公式**：

$$
RF_n = RF_{n-1} + (K_n - 1) \times \prod_{i=1}^{n-1} S_i
$$

感受野越大，神经元能"看到"的上下文越多。深层网络的神经元拥有很大的感受野，能够捕捉全局信息。这也是为什么用小卷积核堆叠（如 [[NeuralNetwork/CNN/Architectures/VGG|VGG]] 全部用 $3 \times 3$）可以替代大卷积核——两个 $3 \times 3$ 的感受野等于一个 $5 \times 5$，但参数更少、非线性更多。

![ExperienceWilderness.png](https://img.yumeko.site/file/articles/CNN/ExperienceWilderness.png)

## 8. PyTorch 代码示例

```python
import torch
import torch.nn as nn

# 一个典型的卷积块
conv = nn.Conv2d(
    in_channels=1,      # 灰度图
    out_channels=32,    # 32 个卷积核
    kernel_size=3,
    stride=1,
    padding=1           # same padding
)

# 输入：(batch=1, channels=1, height=28, width=28)
x = torch.randn(1, 1, 28, 28)
y = conv(x)
print(f"输入形状: {x.shape}")   # (1, 1, 28, 28)
print(f"输出形状: {y.shape}")   # (1, 32, 28, 28)

# 参数量验证
params = sum(p.numel() for p in conv.parameters())
print(f"参数量: {params}")      # 1×32×3×3 + 32 = 320
```

卷积是 CNN 特征提取的核心，下一阶段通常交给 [[NeuralNetwork/CNN/Foundations/PoolingLayer|池化层]] 进行下采样和降维。
