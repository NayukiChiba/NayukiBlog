---
title: CNN 核心组件详解：卷积层、池化层与全连接层
date: 2026-05-07
category: 神经网络/CNN
tags:
  - 深度学习
  - 基础
description: 深入解析 CNN 的三大核心组件：卷积层负责特征提取，池化层负责降维与平移不变性，全连接层负责汇总特征并输出分类结果。从手算示例到 PyTorch 代码，完整理解每个组件的工作原理与设计选择。
image: https://img.yumeko.site/file/blog/cover/1780581719035.webp
status: published
---

卷积神经网络（CNN）由三种核心层组件协同构成：**卷积层**负责从原始像素中提取空间特征，**池化层**对特征图进行下采样以降低计算量并增强平移不变性，**全连接层**则将分散的空间特征汇总为全局判断，最终输出分类或回归结果。

三者形成了一条清晰的流水线：卷积层逐层提取从边缘到语义的层次化特征 -> 池化层周期性地压缩空间尺寸、扩大感受野 -> 全连接层在网络的末端将所有特征整合，完成从图像到类别标签的映射。理解这三类层的计算原理与设计选择，是掌握 CNN 乃至现代深度学习视觉架构的基础。

下文将从卷积层开始，依次深入解析每个组件的数学定义、手算过程、关键参数和代码实现。

## 卷积层（特征提取）

### 1. 什么是卷积操作？

**直觉理解**：卷积就是用一个小的"模板"（卷积核/滤波器），在输入图像上滑动，每次计算模板覆盖区域和模板的"匹配程度"。

**数学定义（2D 离散卷积）**：

$$
\text{Output}[i, j] = \sum_{m}\sum_{n} \text{Input}[i+m, j+n] \times \text{Kernel}[m, n]
$$

其中 $(i,j)$ 是输出位置坐标，$(m,n)$ 是卷积核内的偏移坐标。

### 2. 手算一个具体例子

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

#### 计算输出位置 $[0,0]$

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

#### 计算输出位置 $[0,1]$

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

#### 完整输出

当卷积核滑过全部 9 个位置后：

$$
\text{Output} = \begin{pmatrix}
-2 & 0 & 1 \\
-2 & 0 & 1 \\
-2 & 0 & 1
\end{pmatrix}
$$

注意输出中，第三列的值（$1$）明显大于其他列，这说明在输入图像的第三列附近存在**垂直边缘**——从亮（$1$）到暗（$0$）的过渡。这个简单的 $3 \times 3$ 卷积核就是一个垂直边缘检测器！

![ConvolutionOperation.png](https://img.yumeko.site/file/blog/articles/1780581171172.webp)

### 3. 多通道卷积

上面的例子是单通道（灰度图）的情况。实际中，输入往往有多个通道（比如 RGB 彩色图有 3 个通道）。

**规则**：卷积核的通道数必须等于输入通道数。

对于 RGB 输入（3 通道），一个 $3 \times 3$ 的卷积核实际上是 $3 \times 3 \times 3 = 27$ 个参数，每个通道对应一个 $3 \times 3$ 的矩阵。卷积操作是在三个通道上分别做逐元素相乘，然后把三个结果相加，得到输出特征图上的一个值。

$$
\text{Output}[i,j] = \sum_{c}\sum_{m}\sum_{n} \text{Input}[c, i+m, j+n] \times \text{Kernel}[c, m, n]
$$

其中 $c$ 遍历所有输入通道。

**一个卷积核 $\to$ 一张输出特征图**。如果有 $K$ 个卷积核，输出就有 $K$ 个通道。

### 4. 关键参数

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

### 5. Padding 详解

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

### 6. 输出尺寸计算公式

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

### 7. 感受野

感受野是指输出特征图上一个像素，对应原始输入图像上多大范围的像素。

- 第一层 $3 \times 3$ 卷积：感受野 $= 3 \times 3$
- 第二层 $3 \times 3$ 卷积（叠加在第一层上）：感受野 $= 5 \times 5$

因为第二层的一个像素看到的是第一层的 $3 \times 3$ 区域，而第一层的每个像素又看到输入的 $3 \times 3$，所以总感受野是 $5 \times 5$。

**公式**：

$$
RF_n = RF_{n-1} + (K_n - 1) \times \prod_{i=1}^{n-1} S_i
$$

感受野越大，神经元能"看到"的上下文越多。深层网络的神经元拥有很大的感受野，能够捕捉全局信息。这也是为什么用小卷积核堆叠（如 [[NeuralNetwork/CNN/VGG|VGG]] 全部用 $3 \times 3$）可以替代大卷积核——两个 $3 \times 3$ 的感受野等于一个 $5 \times 5$，但参数更少、非线性更多。

![ExperienceWilderness.png](https://img.yumeko.site/file/blog/articles/1780581171123.webp)

### 8. PyTorch 代码示例

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
print(f"参数量: {params}")      # 1x32x3x3 + 32 = 320
```

卷积是 CNN 特征提取的核心，下一阶段通常交给池化层进行下采样和降维。

## 池化层（降维）

### 1. 为什么需要池化？

- **降维**：减少特征图的空间尺寸，降低后续层的参数量和计算量
- **平移不变性**：小幅度的平移、旋转不会改变池化的输出（因为池化只看局部最大值/平均值）
- **扩大感受野**：经过池化后，同样 $3 \times 3$ 的卷积层卷积核能覆盖输入图像的更大区域

### 2. 最大池化（Max Pooling）

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

### 3. 平均池化（Average Pooling）

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

### 4. Max Pooling vs Average Pooling

| 特性 | Max Pooling | Average Pooling |
| --- | --- | --- |
| 保留的信息 | 最显著的特征 | 整体分布信息 |
| 对纹理的响应 | 强（提取边缘/角点） | 温和（平滑） |
| 梯度传导 | 只传给最大值位置 | 均匀传给所有位置 |
| 现代使用 | 更广泛 | 较少（被 stride=2 卷积替代） |
| 经典案例 | MNIST-CNN, VGG, ResNet | LeNet-5 |

![Pooling.png](https://img.yumeko.site/file/blog/articles/1780581197734.webp)

### 5. Global Average Pooling（GAP）

![GAP.png](https://img.yumeko.site/file/blog/articles/1780736405337.png)

GAP 是 [[NeuralNetwork/CNN/NiN|NiN]]（Network in Network）在 2014 年提出的技术，对**整个特征图**取平均，每个通道输出一个值。

```
特征图：(Batch, 512, 7, 7)
    v AdaptiveAvgPool2d(1)
输出：  (Batch, 512, 1, 1)
    v Flatten
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

### 6. Adaptive Pooling

PyTorch 提供了自适应池化，无需手动计算窗口大小：

```python
# 无论输入特征图是多大，都输出指定的空间尺寸
nn.AdaptiveMaxPool2d(1)     # 输出 1x1 -> 等价于 GMP
nn.AdaptiveAvgPool2d(1)     # 输出 1x1 -> 等价于 GAP
nn.AdaptiveAvgPool2d((4, 4)) # 输出 4x4
```

这对于处理变尺寸输入非常有用——不需要提前知道特征图的具体大小。

![AdaptivePooling.png](https://img.yumeko.site/file/blog/articles/1780581132299.webp)

### 7. 现代趋势：用 Stride 卷积替代池化

近年来，越来越多的架构用 **stride=2 的卷积**替代池化层来做下采样。因为 stride 卷积有可学习的参数，能够自适应地学习最优的下采样方式。

```python
# 传统方式：Conv + Pool
nn.Conv2d(64, 128, 3, padding=1)
nn.MaxPool2d(2, 2)

# 现代方式：stride=2 的卷积
nn.Conv2d(64, 128, 3, stride=2, padding=1)
```

但池化层仍然广泛使用，尤其是在资源受限的场景下（无参数、计算量小）。

![StrideConv.png](https://img.yumeko.site/file/blog/articles/1780581198317.webp)

### 8. 输出尺寸公式

池化层的输出尺寸公式与卷积类似（但没有 padding 时更简单）：

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K}{S} \right\rfloor + 1
$$

对于最常用的 `kernel_size=2, stride=2`：输入尺寸每次减半，如 $28 \to 14 \to 7$。

关于 GAP 的开创性应用和完整架构，参见 [[NeuralNetwork/CNN/NiN|NiN]]。

## 全连接层（分类）

### 1. 为什么 CNN 需要全连接层？

卷积层提取的是**空间化的特征图**——特征以二维（或三维）的形式组织。但分类需要的是一个**全局判断**——"这张图整体上属于哪个类？"

全连接层的作用是把分布在各处的特征"汇总"起来，形成一个全局的判断。

### 2. Flatten：从三维到一维

在全连接层之前，需要把三维的特征图（通道 x 高 x 宽）拉平成一维向量。

![Flatten.png](https://img.yumeko.site/file/blog/articles/1780581172111.webp)

```
特征图：(Batch, 64, 7, 7)    ->    Flatten    ->    向量：(Batch, 3136)
                                                      ^
                                               64 x 7 x 7 = 3136
```

Flatten 本身没有可学习参数，它只是改变了张量的形状。

### 3. 全连接层的计算

$$
y = Wx + b
$$

- $W$ 的形状是 $(\text{out\_features}, \text{in\_features})$
- $x$ 的形状是 $(\text{in\_features},)$
- $b$ 的形状是 $(\text{out\_features},)$

参数量：$\text{in\_features} \times \text{out\_features} + \text{out\_features}$

### 4. 参数占比问题

以一个典型的 CNN 为例（MNIST-CNN 的 FC 部分）：

```
Flatten (3136) -> Linear(3136->128) -> BN -> ReLU -> Dropout -> Linear(128->10)
```

| 层 | 计算 | 参数量 | 占比 |
| --- | --- | ---:| ---:|
| FC1 | $3136 \times 128 + 128$ | 401,536 | **95.0%** |
| FC2 | $128 \times 10 + 10$ | 1,290 | 0.3% |
| 全部卷积层 | — | ~20,000 | 4.7% |

**全连接层承载了 95% 以上的参数！** 这是 CNN 的经典现象：卷积层通过权重共享大幅减少了参数，而全连接层每个输入-输出之间都有独立的连接，参数数量激增。

### 5. 全连接层的三大问题

1. **参数量过大**：如上表所示，FC 层占用了绝大多数参数，导致模型体积大、容易过拟合
2. **空间信息丢失**：Flatten 操作将空间结构完全破坏，无法保留特征的空间位置关系
3. **输入尺寸固定**：FC 层的权重矩阵是固定大小的，这意味着模型只能接受固定尺寸的输入

### 6. 替代方案一：Global Average Pooling

GAP 是 [[NeuralNetwork/CNN/NiN|NiN]]（2014）提出的最经典的 FC 替代方案，利用池化层的全局平均替代全连接。对每个通道的特征图取全局平均，直接作为分类器的输入：

```python
# 传统 FC 方式
x = x.flatten(1)               # (B, 512, 7, 7) -> (B, 25088)
x = nn.Linear(25088, 1024)(x)  # 参数：25088x1024 ~= 25.7M

# GAP 方式
x = nn.AdaptiveAvgPool2d(1)(x) # (B, 512, 7, 7) -> (B, 512, 1, 1)
x = x.flatten(1)               # (B, 512)
x = nn.Linear(512, 1024)(x)    # 参数：512x1024 ~= 0.5M
```

参数量减少约 50 倍，且不再受输入尺寸限制。

![GAP.png](https://img.yumeko.site/file/blog/articles/1780736405337.png)

### 7. 替代方案二：1x1 卷积替代 FC

如果不想丢失空间信息，可以用 $1 \times 1$ 卷积实现与 FC 等价的功能：

```python
# 传统 FC
nn.Linear(512, 10)             # 输入必须是 (B, 512)

# 1x1 卷积等价（可接受任意空间尺寸）
nn.Conv2d(512, 10, kernel_size=1)  # 输入可以是 (B, 512, H, W)
```

$1 \times 1$ 卷积只在通道维度上做变换，不改变空间尺寸。它等价于对每个空间位置的通道向量做一次全连接操作。全卷积网络（FCN）正是利用了这一点，完全抛弃了 FC 层。

![1x1Conv.png](https://img.yumeko.site/file/blog/articles/1780581107550.webp)

### 8. 现代架构的 FC 使用趋势

| 架构 | FC 使用 | 策略 |
| --- | --- | --- |
| LeNet-5 (1998) | 3 层 FC | 传统方式 |
| AlexNet (2012) | 3 层 FC（含 4096 维巨层） | 大量 Dropout |
| VGG (2014) | 3 层 FC | 参数极多 |
| NiN (2014) | **无 FC** | 全部用 GAP + 1x1 卷积 |
| GoogLeNet (2014) | 1 层 FC（1024->1000） | GAP 后接单层 FC |
| ResNet (2015) | **仅 1 层 FC**（分类头） | GAP 后直接 FC |
| MobileNet (2017) | **仅 1 层 FC** | GAP 后直接 FC |

现代趋势很明确：**先用 GAP 将特征图压缩为每个通道一个值，再接一个线性层做分类**。参数量最小、灵活性最高、不易过拟合。

### 9. 代码对比

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

GAP 替代 FC 已成为现代架构的标准做法，详见 [[NeuralNetwork/CNN/NiN|NiN]] 的完整设计。

## 总结

卷积层、池化层与全连接层共同构成了 CNN 的骨架，三者各司其职、层层递进：

- **卷积层**通过局部连接和权重共享，高效地从原始图像中提取边缘、纹理、语义等层次化特征。小卷积核堆叠（如 $3 \times 3$）是现代设计的标准选择，配合 padding 保持空间尺寸、stride 控制下采样，构成了特征提取的主干。
- **池化层**（尤其是最大池化）周期性地压缩特征图的空间维度，降低计算量的同时增强了模型对小平移的鲁棒性。Global Average Pooling 更是在现代架构中成为连接特征提取与分类的桥梁，以极小的代价实现从空间特征到分类向量的转换。
- **全连接层**在网络的末端将所有空间特征汇总为全局表示，完成最终的分类或回归。随着 GAP 和 $1 \times 1$ 卷积的普及，现代 CNN 中全连接层的规模已大幅缩减，通常仅保留单层分类头，在保证表达能力的同时大幅降低了参数量和过拟合风险。

从局部感知到全局决策，从底层纹理到高层语义，这三类层的协同工作赋予了 CNN 强大的视觉理解能力。理解它们的计算原理和设计选择，是深入学习和设计卷积神经网络不可或缺的基础。
