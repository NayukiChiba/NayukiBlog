---
title: Vanilla RNN 详解：前向传播与 BPTT
date: 2026-05-29
category: 神经网络/RNN
tags:
  - 深度学习
  - 基础
  - 优化
description: 从隐藏状态更新公式到通过时间的反向传播，全面理解 Vanilla RNN 的前向计算流程、参数矩阵的物理含义、手算推导、BPTT 梯度连乘机制、截断策略与梯度裁剪。
image: https://img.yumeko.site/file/blog/cover/1780581856894.webp
status: published
---

## 概述

Vanilla RNN（也称 Elman RNN、Simple RNN）是循环神经网络的原型。它的核心机制包含一体两面：**前向传播**沿时间轴逐步累积隐藏状态，**通过时间的反向传播（BPTT）**沿时间轴反向流动梯度、更新参数。前向传播决定了网络"看到了什么"，BPTT 决定了网络"学到了什么"——二者共享同一个计算图，理解其一就必须理解其二。

本文分为两大部分：
- **前向传播**：从 RNN Cell 的数学定义出发，追踪隐藏状态在时间维度的完整流动过程，配合手算例子和 PyTorch 代码，建立对 RNN 行为模式的直观理解。
- **BPTT 反向传播**：深入梯度沿时间轴连乘的数学原理，展示梯度消失/爆炸的结构性根源，讨论截断 BPTT 与梯度裁剪的实践策略，并对比 LSTM 中更稳定的梯度路径。

---

## 前向传播

### 1. RNN Cell 的数学定义

#### 1.1 两个核心公式

Vanilla RNN 在每个时间步 $t$ 执行以下计算：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

$$
y_t = W_{hy} \cdot h_t + b_y
$$

第一行是**状态更新**，第二行是**输出计算**。这两个公式就是 RNN 的全部。

#### 1.2 符号约定

| 符号 | 含义 | 形状（以 Char-RNN 为例） |
|------|------|------|
| $x_t$ | 第 $t$ 步的输入向量 | (65,) — 字符的 embedding |
| $h_{t-1}$ | 上一时刻的隐藏状态 | (256,) |
| $h_t$ | 当前时刻的隐藏状态 | (256,) |
| $y_t$ | 当前时刻的输出（logits） | (65,) — 每个字符的得分 |
| $W_{xh}$ | 输入到隐藏的权重 | (256, 65) |
| $W_{hh}$ | 隐藏到隐藏的权重 | (256, 256) |
| $W_{hy}$ | 隐藏到输出的权重 | (65, 256) |
| $b_h$ | 隐藏层偏置 | (256,) |
| $b_y$ | 输出层偏置 | (65,) |

---

### 2. 前向传播的完整流程

RNN 的前向传播不是"一次性"完成的——它在时间维度上**逐步展开**，同一个 RNN Cell 被重复调用 $T$ 次（$T$ 为序列长度）。理解这个逐步展开的过程，是理解 RNN 一切行为的基础。

#### 2.1 时间步展开：同一个 Cell，重复 T 次

RNN 的核心循环可以展开为在时间维度上的链式结构。设序列长度为 $T$，初始隐藏状态 $h_0$ 通常为零向量。对第 $t$ 步（$t = 1, 2, \dots, T$），执行完全相同的两个操作：

1. **状态更新**：将当前输入 $x_t$ 与上一时刻隐藏状态 $h_{t-1}$ 融合，产生新的隐藏状态 $h_t$
2. **输出计算**（可选）：从 $h_t$ 产生当前时刻的输出 $y_t$

关键点在于——第 $t$ 步操作完成后，$h_t$ 被"传递"给第 $t+1$ 步，成为第 $t+1$ 步计算 $h_{t+1}$ 时的 $h_t$ 输入。这就是"循环"的含义：当前步的输出（$h_t$）成为下一步的输入（作为 $h_{t-1}$ 的角色）。

#### 2.2 逐步执行流程

以处理一个长度为 $T=3$ 的序列为例，详细追踪每一步内部发生了什么。

**初始状态（$t=0$）：**

$h_0$ 为全零向量（或由调用者指定）。此时网络没有任何"历史记忆"。

**第 1 步（$t=1$）：**

- 输入：$x_1$（序列的第一个元素）+ $h_0$（全零向量）
- 计算 $W_{xh} \cdot x_1$：将输入 $x_1$ 从输入空间映射到隐藏空间。$W_{xh}$ 的每一行检测输入向量中的某种模式，匹配则激活值高。
- 计算 $W_{hh} \cdot h_0$：由于 $h_0 = 0$，此项为零向量，不贡献信息。
- 求和并加偏置：$\text{pre\_hidden} = W_{xh} \cdot x_1 + W_{hh} \cdot h_0 + b_h$。$b_h$ 提供基线偏置，使某些维度的激活偏向正值或负值。
- 非线性激活：$h_1 = \tanh(\text{pre\_hidden})$。$\tanh$ 将每个元素的值压缩到 $(-1, 1)$ 区间，防止数值在多次循环后发散。
- $h_1$ 现在是网络对"看到 $x_1$ 之后"的全部记忆。它被复制的两份：一份传给下一步作为 $h_{t-1}$，一份用于（可选的）当前步输出计算。
- 输出：$y_1 = W_{hy} \cdot h_1 + b_y$，将隐藏表示解码为输出空间的预测。

**第 2 步（$t=2$）：**

- 输入：$x_2$（序列的第二个元素）+ $h_1$（第 1 步产生的隐藏状态）
- 计算 $W_{xh} \cdot x_2$：对当前输入 $x_2$ 的特征提取。
- 计算 $W_{hh} \cdot h_1$：这是**循环连接发挥作用的时刻**。$h_1$ 中包含了 $x_1$ 的信息，$W_{hh}$ 将这个信息与当前输入融合。$W_{hh}$ 的每一行定义了如何从 $h_1$ 的 256 个维度中组合出当前需求的信号。
- 求和 + 偏置 + $\tanh$：$h_2 = \tanh(W_{xh} \cdot x_2 + W_{hh} \cdot h_1 + b_h)$。$h_2$ 现在同时包含了 $x_1$ 和 $x_2$ 的信息——它是序列前两个元素的压缩表示。
- $h_2$ 继续传给下一步。
- 输出：$y_2 = W_{hy} \cdot h_2 + b_y$，基于前两步的信息做预测。

**第 3 步（$t=3=T$）：**

- 输入：$x_3$ + $h_2$
- 计算：$h_3 = \tanh(W_{xh} \cdot x_3 + W_{hh} \cdot h_2 + b_h)$
- $h_3$ 包含了 $x_1$、$x_2$、$x_3$ 的全部信息——它是整个序列的压缩表示。
- 输出：$y_3 = W_{hy} \cdot h_3 + b_y$
- 由于这是最后一步，$h_3$ 不再传给任何后续步骤。如果需要序列级别的输出（如分类），通常取 $h_T$ 作为整个序列的表示。

#### 2.3 信息的累积与衰减

从上述流程可以观察到一个关键规律：$h_t$ 中的信息是**逐层叠加**的。

- $h_1$ 只包含 $x_1$ 的信息
- $h_2$ 包含 $x_1$（经过 $\tanh$ 压缩）和 $x_2$（直接输入）的混合
- $h_3$ 包含 $x_1$（经过两次 $\tanh$ 压缩）、$x_2$（经过一次压缩）和 $x_3$（直接输入）的混合

每经过一个时间步，旧信息就被 $\tanh$ 非线性再压缩一次。$\tanh'(x) = 1 - \tanh^2(x)$ 在 $|x| > 2$ 的区域远小于 1，因此早期信息的"信号强度"随着时间步增加而逐步衰减。这就是 Vanilla RNN 长期依赖能力不足的结构性根源——$x_1$ 的信息要到达 $h_{100}$，需要穿过 99 次 $\tanh$ 和 99 次 $W_{hh}$ 乘法。

#### 2.4 流程总结

用文字概括 RNN 一个时间步内的完整计算链：

> 当前输入 $x_t$ 经 $W_{xh}$ 编码，上一时刻隐藏状态 $h_{t-1}$ 经 $W_{hh}$ 变换，两者相加后加偏置 $b_h$，再通过 $\tanh$ 非线性，得到新的隐藏状态 $h_t$。$h_t$ 一路传给下一个时间步（成为新的 $h_{t-1}$），另一路经 $W_{hy}$ 解码为当前时刻的输出 $y_t$。

这个流程中的**每一个操作都是可微的**，因此整个计算图可以被自动微分穿透，梯度从最后的损失一路回传到第一个时间步——这就是 BPTT（Backpropagation Through Time）的基础。

---

### 3. 参数矩阵的物理含义

![RNNCell.png](https://img.yumeko.site/file/blog/articles/1780581435819.webp)

#### 3.1 $W_{xh}$：输入编码器

$W_{xh}$ 将当前输入 $x_t$ 映射到隐藏空间。

$$W_{xh} \cdot x_t + \text{(来自 } h_{t-1} \text{ 的信息)}$$

- 形状：(hidden_size, input_size)，每一行是一个"检测器"，检测输入的某种模式
- 如果 $W_{xh}$ 的某一行与 $x_t$ 高度匹配，该维度的激活值就高

#### 3.2 $W_{hh}$：状态转移矩阵

$W_{hh}$ 是 RNN **最核心**的矩阵——它决定了上一时刻的记忆如何影响当前时刻。

$$W_{hh} \cdot h_{t-1}$$

- 形状：(hidden_size, hidden_size)，方阵，定义了隐藏空间中的"动力学"
- 每一行决定了上一时刻的 256 维隐藏状态如何组合产生当前时刻该维度的输入
- 这个矩阵的特征值决定了信息在时间维度上能传播多远（特征值 < 1 会导致信息衰减）

#### 3.3 $W_{hy}$：输出解码器

$W_{hy}$ 将"内部记忆"解码为"外部输出"。

$$W_{hy} \cdot h_t$$

- 形状：(output_size, hidden_size)，将隐藏状态映射到输出空间
- 在 Char-RNN 中是 (65, 256)：65 个字符各有一个"打分器"，评估当前隐藏状态支持输出各字符的程度

#### 3.4 $b_h$ 和 $b_y$：偏置项

- $b_h$：允许神经元在没有有效输入时仍有基线激活水平
- $b_y$：允许某些输出类有更高的先验概率（如英语中 'e' 比 'z' 常见得多）

---

### 4. 完整手算例子

![RNNCalculate.png](https://img.yumeko.site/file/blog/articles/1780581429926.webp)

使用一个极端简化的设置来手算 RNN 的前向传播全过程：

- input_size = 2，hidden_size = 2，output_size = 3（只有三个字符 "H", "e", "l"）
- 初始隐藏状态 $h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

#### 4.1 权重矩阵

$$
W_{xh} = \begin{pmatrix}
0.5 & -0.3 \\
0.8 & 0.2
\end{pmatrix}
\qquad
W_{hh} = \begin{pmatrix}
0.1 & 0.4 \\
-0.2 & 0.6
\end{pmatrix}
\qquad
W_{hy} = \begin{pmatrix}
0.3 & -0.1 \\
0.5 & 0.2 \\
-0.4 & 0.7
\end{pmatrix}
$$

偏置全部初始化为零：$b_h = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$，$b_y = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$。

#### 4.2 时间步 1

输入 $x_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$（表示字符 "H"）

**输入编码**：

$$
W_{xh} \cdot x_1 = \begin{pmatrix}
0.5 & -0.3 \\
0.8 & 0.2
\end{pmatrix}
\cdot
\begin{pmatrix} 1 \\ 0 \end{pmatrix}
=
\begin{pmatrix} 0.5 \\ 0.8 \end{pmatrix}
$$

**状态转移**：

$$
W_{hh} \cdot h_0 = \begin{pmatrix}
0.1 & 0.4 \\
-0.2 & 0.6
\end{pmatrix}
\cdot
\begin{pmatrix} 0 \\ 0 \end{pmatrix}
=
\begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

**隐藏状态更新**：

$$
h_1 = \tanh\left( \begin{pmatrix} 0.5 \\ 0.8 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} \right)
= \tanh\left( \begin{pmatrix} 0.5 \\ 0.8 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.462 \\ 0.664 \end{pmatrix}
$$

**输出计算**：

$$
y_1 = \begin{pmatrix}
0.3 & -0.1 \\
0.5 & 0.2 \\
-0.4 & 0.7
\end{pmatrix}
\cdot
\begin{pmatrix} 0.462 \\ 0.664 \end{pmatrix}
=
\begin{pmatrix}
0.3 \times 0.462 + (-0.1) \times 0.664 \\
0.5 \times 0.462 + 0.2 \times 0.664 \\
(-0.4) \times 0.462 + 0.7 \times 0.664
\end{pmatrix}
=
\begin{pmatrix}
0.072 \\
0.364 \\
0.280
\end{pmatrix}
$$

Softmax 后：$\text{softmax}(y_1) \approx \begin{pmatrix} 0.31 \\ 0.41 \\ 0.28 \end{pmatrix}$。

模型以 41% 的概率预测下一个字符是 "e"。

#### 4.3 时间步 2

输入 $x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$（字符 "e"），$h_1 \approx \begin{pmatrix} 0.462 \\ 0.664 \end{pmatrix}$

**输入编码**：

$$
W_{xh} \cdot x_2 = \begin{pmatrix} -0.3 \\ 0.2 \end{pmatrix}
$$

**状态转移**：

$$
W_{hh} \cdot h_1 = \begin{pmatrix}
0.1 \times 0.462 + 0.4 \times 0.664 \\
-0.2 \times 0.462 + 0.6 \times 0.664
\end{pmatrix}
= \begin{pmatrix} 0.312 \\ 0.306 \end{pmatrix}
$$

**隐藏状态更新**：

$$
h_2 = \tanh\left( \begin{pmatrix} -0.3 \\ 0.2 \end{pmatrix} + \begin{pmatrix} 0.312 \\ 0.306 \end{pmatrix} \right)
= \tanh\left( \begin{pmatrix} 0.012 \\ 0.506 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.012 \\ 0.466 \end{pmatrix}
$$

$h_2$ 融合了 $x_2$（当前输入 "e"）和 $h_1$（包含之前 "H" 的信息）。

#### 4.4 时间步 3

输入 $x_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$（表示字符 "l"，这里用 [1,1] 表示），$h_2 \approx \begin{pmatrix} 0.012 \\ 0.466 \end{pmatrix}$

**状态转移**：

$$
W_{hh} \cdot h_2 = \begin{pmatrix}
0.1 \times 0.012 + 0.4 \times 0.466 \\
-0.2 \times 0.012 + 0.6 \times 0.466
\end{pmatrix}
= \begin{pmatrix} 0.188 \\ 0.277 \end{pmatrix}
$$

可以看到 $h_3$ 中来自 $h_2$ 的贡献（状态转移）已经和来自 $x_3$ 的贡献相当——模型在利用之前的上下文。

#### 4.5 关键观察

![InformationAttenuation.png](https://img.yumeko.site/file/blog/articles/1780581409002.webp)

手算揭示了 Vanilla RNN 的几个重要特性：

1. **信息是累加的**：每个时间步的 $h_t$ 都是 $x_t$ 和 $h_{t-1}$ 的**加和**（经过非线性），早期输入的信息不会突然消失
2. **但信号强度会逐步衰减**：每一步的 $\tanh$ 会将值压缩到 (-1, 1)，多次压缩后早期信息的"信号强度"逐步衰减
3. **$W_{hh}$ 是关键**：它的特征值决定了信息能传播多远。如果特征值普遍小于 1，信息会指数衰减——这就是梯度消失的结构性根源

---

### 5. 形状追踪表

以 Char-RNN 为例（batch_size=B, seq_len=100）：

| 组件 | 形状 | 说明 |
|------|------|------|
| 输入 $x$（batch） | (B, 100, 65) | 100 个字符，每个 65 维 embedding |
| 初始隐藏 $h_0$ | (num_layers, B, 256) | 全零初始化 |
| $W_{xh}$ | (256, 65) | 输入映射 |
| $W_{hh}$ | (256, 256) | 状态转移 |
| $W_{hy}$ | (65, 256) | 输出映射 |
| 中间结果（每步） | (B, 256) | 在循环中逐时间步处理 |
| 输出 $y$（全部步） | (B, 100, 65) | 每个时间步输出 65 个 logits |
| 最终隐藏 $h_{100}$ | (num_layers, B, 256) | 最后一个时间步的隐藏状态 |

---

### 6. RNN 的输出模式

PyTorch 的 `nn.RNN` forward 返回两个值：`output` 和 `h_n`。

#### 6.1 output：所有时间步的隐藏状态

```python
output, h_n = rnn(x, h_0)
# output.shape = (B, seq_len, hidden_size)  当 batch_first=True
```

`output` 包含**每个时间步**的隐藏状态——`output[:, t, :]` 就是 $h_t$。

#### 6.2 h_n：最后一个时间步的隐藏状态

```python
# h_n.shape = (num_layers, B, hidden_size)
```

`h_n` 就是 $h_T$（最后一个时间步）。对于单层 RNN，`h_n.squeeze(0) == output[:, -1, :]`。

#### 6.3 Many-to-Many vs Many-to-One

![TwoOutput.png](https://img.yumeko.site/file/blog/articles/1780581456047.webp)

```python
# Many-to-Many（序列标注）：使用 output
for t in range(seq_len):
    loss += criterion(fc(output[:, t, :]), labels[:, t])

# Many-to-One（序列分类）：使用 h_n
logits = fc(h_n.squeeze(0))
loss = criterion(logits, labels)
```

---

### 7. 梯度消失的数学根源——前向视角

回顾隐藏状态的更新公式：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

对 $h_{t-1}$ 求偏导：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}\big(\tanh'(W_{hh} \cdot h_{t-1} + ...)\big) \cdot W_{hh}
$$

两个因素共同作用：
- $\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$，在大多数区域远小于 1
- $W_{hh}$ 的范数如果小于 1，进一步压缩梯度

![GradientMultiplication.png](https://img.yumeko.site/file/blog/articles/1780581393466.webp)

在通过 $T$ 个时间步反向传播时：

$$
\frac{\partial h_T}{\partial h_1} \approx \prod_{t=2}^{T} \big( \text{diag}(\tanh'(\cdot)) \cdot W_{hh} \big)
$$

如果每一步的谱范数小于 1，这个连乘会随着 $T$ 增大而指数衰减至 0——这就是 Vanilla RNN 无法学习长距离依赖的根本原因。

#### 7.1 具体数值

当 $\tanh' \approx 0.5$，$||W_{hh}|| \approx 0.5$ 时，每一步的梯度衰减因子约 0.25。100 步后：

$$
||\frac{\partial h_{100}}{\partial h_1}|| \approx (0.25)^{100} \approx 6.2 \times 10^{-61}
$$

这意味着第 1 步的输入对第 100 步的损失**完全没有可用的梯度信号**。模型的有效信息窗口约为 20 步，超过此范围后梯度已衰减到无法用于参数更新。

> 以上是从前向传播的视角预览梯度消失问题。下一部分将通过 BPTT 的完整数学推导，深入分析梯度如何沿时间轴反向传播，以及 LSTM/GRU 如何在结构上解决这一问题。

---

### 8. PyTorch 代码实现

#### 8.1 基础用法

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=inputSize,     # x_t 的维度
            hidden_size=hiddenSize,   # h_t 的维度
            num_layers=1,             # RNN 层数
            batch_first=True,         # 输入 (B, L, input_size)
        )
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x, h0=None):
        # x: (B, L, input_size)
        B = x.size(0)
        if h0 is None:
            h0 = torch.zeros(1, B, self.rnn.hidden_size)

        output, hn = self.rnn(x, h0)
        # output: (B, L, hidden_size)
        # hn: (1, B, hidden_size)

        # Many-to-One：用最后一步做分类
        lastHidden = hn.squeeze(0)        # (B, hidden_size)
        logits = self.fc(lastHidden)      # (B, output_size)
        return logits
```

#### 8.2 batch_first=True vs False

| 参数 | 输入形状 | 输出形状（output） | 输出形状（hn） |
|------|----------|-------------------|----------------|
| `batch_first=True` | `(B, L, in)` | `(B, L, hidden)` | `(layers, B, hidden)` |
| `batch_first=False` | `(L, B, in)` | `(L, B, hidden)` | `(layers, B, hidden)` |

`batch_first=True` 更符合直觉，两个项目都使用此设置。

#### 8.3 多对多输出

```python
class SeqToSeqRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.rnn = nn.RNN(inputSize, hiddenSize, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        output, _ = self.rnn(x)           # (B, L, hidden)
        logits = self.fc(output)           # (B, L, outputSize)
        return logits
```

---

### 9. 前向传播小结

Vanilla RNN 是所有后续 RNN 变体的**原型**。它引入了循环连接和隐藏状态的基本概念，但因其梯度消失问题，在现代实践中几乎不被直接使用。

| 学习目标 | 推荐模型 |
|----------|----------|
| 理解 RNN 的原理 | **Vanilla RNN**（最简洁） |
| 实际工程项目 | LSTM 或 GRU |
| 快速实验 | GRU（参数少、收敛快） |

理解了前向传播，你就掌握了所有 RNN 变体共享的核心机制。LSTM 和 GRU 只是在这个基础上增加了门控——但它们解决的是同一个根本问题。下一部分将深入这个根本问题的数学本质。

---

## BPTT 反向传播

### 1. BPTT 的核心问题

#### 1.1 普通反向传播 vs BPTT

在普通 MLP 中，反向传播（Backpropagation）沿着网络层从输出向输入传播梯度。网络是一个有向无环图（DAG），梯度计算路径是唯一且确定的。

在 RNN 中，情况复杂得多——同一个参数（如 $W_{hh}$）在**每个时间步**都被使用。因此，$W_{hh}$ 的总梯度 = 它在所有时间步产生的梯度之和。

**"Through Time"的含义**：梯度不仅穿过网络层，还**穿过时间**——从最后一步反向流到第一步。

#### 1.2 展开计算图

RNN 的前向计算沿时间轴展开后，等价于一个**深度 = 序列长度**的"虚拟前馈网络"，且所有时间步的对应层共享参数：

![RNNBPTT.png](https://img.yumeko.site/file/blog/articles/1780581425079.webp)

所有从 $h_{t-1}$ 到 $h_t$ 的横向连接共享同一个 $W_{hh}$，所有从 $h_t$ 到 $y_t$ 的纵向连接共享同一个 $W_{hy}$，所有从 $x_t$ 到 $h_t$ 的纵向连接共享同一个 $W_{xh}$。

---

### 2. BPTT 的数学推导

#### 2.1 前向传播的展开形式

对于长度为 $T$ 的序列，前向传播完全展开后：

$$
h_1 = \tanh(W_{hh} \cdot h_0 + W_{xh} \cdot x_1 + b_h)
$$
$$
h_2 = \tanh(W_{hh} \cdot h_1 + W_{xh} \cdot x_2 + b_h)
$$
$$
\vdots
$$
$$
h_T = \tanh(W_{hh} \cdot h_{T-1} + W_{xh} \cdot x_T + b_h)
$$

每个时间步的输出：

$$
y_t = W_{hy} \cdot h_t + b_y
$$

总损失（各时间步损失之和或平均）：

$$
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
$$

#### 2.2 对 $W_{hh}$ 的梯度

![GradSum.png](https://img.yumeko.site/file/blog/articles/1780581401174.webp)

链式法则：$W_{hh}$ 在每个时间步都参与计算，所以总梯度等于它在每个时间步产生的梯度之和：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

下面集中推导**单个时间步 $t$** 的梯度 $\frac{\partial L_t}{\partial W_{hh}}$，总梯度只需把所有 $t$ 的贡献加起来。

##### 步骤 1：追踪依赖关系

以 $t=3$ 为例，写出 $L_3$ 与 $W_{hh}$ 之间的完整依赖链：

- $L_3$ 直接依赖于 $h_3$（损失函数以 $h_3$ 为输入）
- $h_3 = \tanh(W_{hh} \cdot h_2 + W_{xh} \cdot x_3 + b_h)$——$W_{hh}$ **直接**出现在 $h_3$ 的计算中（途径 A）
- 同时 $h_3$ 也依赖于 $h_2$，而 $h_2 = \tanh(W_{hh} \cdot h_1 + W_{xh} \cdot x_2 + b_h)$——$W_{hh}$ **间接**通过 $h_2$ 影响 $h_3$（途径 B）
- 同理，$h_2$ 又依赖于 $h_1$，$h_1$ 的计算同样包含 $W_{hh}$（途径 C）

推广到任意 $t$：$W_{hh}$ 总共出现了 $t$ 次——在 $h_1, h_2, \dots, h_t$ 的每一次计算中。梯度必须沿所有这 $t$ 条路径求和。

##### 步骤 2：写出全导数展开

根据多元链式法则（全导数公式），$\frac{\partial L_t}{\partial W_{hh}}$ 需要沿 $W_{hh}$ 出现的**所有位置**展开。$h_t$ 不仅直接依赖 $W_{hh}$，还通过 $h_{t-1}$ 间接依赖：

$$
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}
$$

但这个写法是不完整的——$\frac{\partial h_t}{\partial W_{hh}}$ 本身隐含了对 $h_{t-1}$ 的依赖。必须将 $h_t$ 的全微分展开：

$$
dh_t = \frac{\partial h_t}{\partial W_{hh}} dW_{hh} + \frac{\partial h_t}{\partial h_{t-1}} dh_{t-1}
$$

把 $dh_{t-1}$ 继续展开：

$$
dh_{t-1} = \frac{\partial h_{t-1}}{\partial W_{hh}} dW_{hh} + \frac{\partial h_{t-1}}{\partial h_{t-2}} dh_{t-2}
$$

一路展开到 $h_1$（$dh_1 = \frac{\partial h_1}{\partial W_{hh}} dW_{hh}$，因为 $h_1$ 之前没有更多依赖），带回 $L_t$ 得到：

$$
\begin{aligned}
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \Bigg(
&\frac{\partial h_t}{\partial W_{hh}} && \text{（$W_{hh}$ 在时刻 $t$ 的直接影响）} \\
+\ &\frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial W_{hh}} && \text{（经 $h_{t-1}$ 的间接影响）} \\
+\ &\frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial h_{t-2}} \cdot \frac{\partial h_{t-2}}{\partial W_{hh}} && \text{（经 $h_{t-2}$ 的间接影响）} \\
+\ &\cdots && \\
+\ &\frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial h_{t-2}} \cdot \cdots \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_{hh}} && \text{（经 $h_1$ 的间接影响）}
\Bigg)
\end{aligned}
$$

关键洞察：每一项的结构是 $\frac{\partial L_t}{\partial h_t}$ ×（沿时间轴的梯度连乘）×（某时刻 $W_{hh}$ 的直接梯度）。

##### 步骤 3：写成紧凑求和形式

将上述 $t$ 项合并为求和记号。令 $k$ 表示 $W_{hh}$ **直接**参与计算的时刻（$k=1,2,\dots,t$），从时刻 $k$ 到时刻 $t$ 需要经过 $t-k$ 步隐藏状态的传递：

$$
\boxed{\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}}
$$

分解每个因子的含义：

| 因子 | 含义 | 形状 |
|------|------|------|
| $\frac{\partial L_t}{\partial h_t}$ | 损失对当前隐藏状态的梯度 | $1 \times d_h$ |
| $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$ | 从时刻 $k$ 到时刻 $t$ 的**梯度连乘** | $d_h \times d_h$ |
| $\frac{\partial h_k}{\partial W_{hh}}$ | 时刻 $k$ 的隐藏状态对 $W_{hh}$ 的直接梯度 | $d_h \times (d_h \cdot d_h)$ |

##### 步骤 4：以 $t=3$ 为例逐项写出

设 $T=3$，$L = L_3$（只考虑最后一步的损失），将上述公式逐项展开：

**$k=1$（最远路径，经 $h_1$）：**
$$
\frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_{hh}}
$$
梯度流：$L_3 \to h_3 \to h_2 \to h_1 \to W_{hh}$，连乘两次 Jacobian。

**$k=2$（中间路径，经 $h_2$）：**
$$
\frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_{hh}}
$$
梯度流：$L_3 \to h_3 \to h_2 \to W_{hh}$，连乘一次 Jacobian。

**$k=3$（最短路径，经 $h_3$ 直接到 $W_{hh}$）：**
$$
\frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial W_{hh}}
$$
梯度流：$L_3 \to h_3 \to W_{hh}$，不经过时间轴连乘。

**三项相加**得到 $\frac{\partial L_3}{\partial W_{hh}}$。其中 $k=1$ 的路径经过 2 次 Jacobian 连乘，梯度衰减最严重；$k=3$ 不经过连乘，梯度最完整。这就是为什么**靠近损失的时间步贡献大，远离损失的时间步贡献小**。

> 注意：求和记号中 $\prod_{j=k+1}^{t}$ 在 $k=t$ 时为空积（没有因子），等于单位矩阵 $I$，对应的项退化为 $\frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}$。

#### 2.3 $\frac{\partial h_j}{\partial h_{j-1}}$ 的具体形式

对于 Vanilla RNN：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(\tanh'(W_{hh} \cdot h_{j-1} + W_{xh} \cdot x_j + b_h)\big) \cdot W_{hh}
$$

两个因素：
- $\text{diag}(\tanh'(\cdot))$：对角矩阵，对角线元素 $\in (0, 1]$，大部分远小于 1
- $W_{hh}$：训练过程中动态变化，范数可大可小

**连乘 $T$ 步后**：

- 如果每一步的谱范数 $< 1$：连乘趋近于 0，导致**梯度消失**
- 如果每一步的谱范数 $> 1$：连乘趋近于无穷大，导致**梯度爆炸**

#### 2.4 对 $W_{xh}$ 和 $W_{hy}$ 的梯度

$W_{hy}$ 的梯度相对简单——每次使用 $W_{hy}$ 是独立的（不通过隐藏状态传递），不涉及时间轴连乘：

$$
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T
$$

$W_{xh}$ 的梯度类似 $W_{hh}$，也需要沿时间轴反向传播，但连乘只从 $t$ 到 $k$：

$$
\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{xh}}
$$

#### 2.5 具体数值推导

以下用一个简化的小规模例子，完整计算 $\frac{\partial L_3}{\partial W_{hh}}$，展示 BPTT 中梯度连乘的实际衰减过程。

**设定：** $T = 3$，$d_h = 2$，$d_x = 2$。使用与前向传播手算例子相同的权重矩阵：

$$
W_{hh} = \begin{pmatrix}
0.1 & 0.4 \\
-0.2 & 0.6
\end{pmatrix}
$$

隐藏状态序列（取自前向手算结果）：

$$
h_1 = \begin{pmatrix} 0.462 \\ 0.664 \end{pmatrix}, \quad
h_2 = \begin{pmatrix} 0.012 \\ 0.466 \end{pmatrix}, \quad
h_3 = \begin{pmatrix} 0.188 \\ 0.277 \end{pmatrix}
$$

设损失仅来自第 3 步，且 $\frac{\partial L_3}{\partial h_3} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$（简化假设）。

**步骤 1：计算 $\frac{\partial h_3}{\partial h_2}$**

先求 $\tanh'$ 的对角矩阵。需要 $\tanh$ 激活之前的线性值 $a_3 = W_{hh} \cdot h_2 + W_{xh} \cdot x_3 + b_h$。设 $a_3 = \begin{pmatrix} 0.19 \\ 0.28 \end{pmatrix}$（与 $h_3 = \tanh(a_3)$ 的值一致），则：

$$
\tanh'(0.19) = 1 - \tanh^2(0.19) \approx 1 - 0.188^2 \approx 0.965
$$
$$
\tanh'(0.28) = 1 - \tanh^2(0.28) \approx 1 - 0.277^2 \approx 0.923
$$
$$
\text{diag}(\tanh'(a_3)) = \begin{pmatrix} 0.965 & 0 \\ 0 & 0.923 \end{pmatrix}
$$
$$
\frac{\partial h_3}{\partial h_2} = \text{diag}(\tanh'(a_3)) \cdot W_{hh}
= \begin{pmatrix} 0.965 & 0 \\ 0 & 0.923 \end{pmatrix} \cdot \begin{pmatrix} 0.1 & 0.4 \\ -0.2 & 0.6 \end{pmatrix}
= \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix}
$$

**步骤 2：计算 $\frac{\partial h_2}{\partial h_1}$**

同理，设 $a_2 = \begin{pmatrix} 0.012 \\ 0.506 \end{pmatrix}$：

$$
\tanh'(0.012) \approx 0.9999, \quad \tanh'(0.506) \approx 0.744
$$
$$
\text{diag}(\tanh'(a_2)) = \begin{pmatrix} 0.9999 & 0 \\ 0 & 0.744 \end{pmatrix}
$$
$$
\frac{\partial h_2}{\partial h_1} = \begin{pmatrix} 0.9999 & 0 \\ 0 & 0.744 \end{pmatrix} \cdot \begin{pmatrix} 0.1 & 0.4 \\ -0.2 & 0.6 \end{pmatrix}
= \begin{pmatrix} 0.100 & 0.400 \\ -0.149 & 0.446 \end{pmatrix}
$$

**步骤 3：计算连乘**

$$
\frac{\partial h_3}{\partial h_1} = \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}
= \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix} \cdot \begin{pmatrix} 0.100 & 0.400 \\ -0.149 & 0.446 \end{pmatrix}
= \begin{pmatrix} -0.048 & 0.211 \\ -0.101 & 0.173 \end{pmatrix}
$$

矩阵元素量级从第一步的约 $0.1-0.55$ 下降到约 $0.05-0.21$，两步连乘已衰减到原来的约一半。

**步骤 4：计算 $\frac{\partial h_k}{\partial W_{hh}}$**

对于 $k=1$（最远的时间步）：

$$
\frac{\partial h_1}{\partial W_{hh}} \text{ 是一个 } (2 \times 4) \text{ 的张量（对 } 2\times 2 \text{ 矩阵的导数）}
$$

取其中一个元素简化说明：对 $W_{hh}[0,0]$，$\frac{\partial h_1[0]}{\partial W_{hh}[0,0]} = \tanh'(a_1[0]) \cdot h_0[0] = 0$（因 $h_0 = 0$）。

对于 $k=2$：

$$
\frac{\partial h_2[0]}{\partial W_{hh}[0,0]} \approx 0.9999 \cdot h_1[0] = 0.9999 \times 0.462 \approx 0.462
$$

对于 $k=3$：

$$
\frac{\partial h_3[0]}{\partial W_{hh}[0,0]} \approx 0.965 \cdot h_2[0] = 0.965 \times 0.012 \approx 0.0116
$$

**步骤 5：汇总 $W_{hh}[0,0]$ 的梯度**

$$
\frac{\partial L_3}{\partial W_{hh}[0,0]} = \sum_{k=1}^{3} \frac{\partial L_3}{\partial h_3} \cdot \left( \prod_{j=k+1}^{3} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}[0,0]}
$$

- $k=1$：因 $h_0 = 0$，贡献为 $0$
- $k=2$：$[1, 1] \cdot \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix} \cdot \begin{pmatrix} 0.462 \\ \dots \end{pmatrix}$（量级约 $0.05-0.4$）
- $k=3$：$[1, 1] \cdot \begin{pmatrix} 0.0116 \\ \dots \end{pmatrix}$（量级约 $0.01$）

**关键观察：** 即使在这个仅有 3 步的例子中，$k=2$（回传2步）的贡献已经衰减到约 $0.05-0.4$ 的量级，而 $k=3$ 的贡献约为 $0.01$。当 $T=100$ 时，$k=1$ 的贡献经过 99 步连乘，每个元素约 $(0.5)^{99} \times 0.5 \approx 10^{-31}$ 量级——在 float32 精度下完全不可用。

**结论：** Vanilla RNN 中的梯度消失是结构性的。$\tanh$ 的导数被限制在 $(0, 1]$ 区间，$W_{hh}$ 的重复乘法进一步压缩梯度。即使 $W_{hh}$ 设计得再好，$\tanh$ 的非线性压缩效应无法被消除。

#### 2.6 公式汇总

为便于对照，将 BPTT 的核心公式集中列出。给定总损失 $L = \sum_{t=1}^{T} L_t$：

**对 $W_{hh}$ 的梯度：**

$$
\boxed{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}}
$$

其中每个时间步 $t$ 的贡献展开为对该参数在各时刻使用的偏导求和：

$$
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

**梯度连乘项（关键的衰减因子）：**

$$
\boxed{\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(\tanh'(W_{hh} \cdot h_{j-1} + W_{xh} \cdot x_j + b_h)\big) \cdot W_{hh}}
$$

**对 $W_{hy}$ 的梯度（无时间轴连乘，计算最简单）：**

$$
\boxed{\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T}
$$

**对 $W_{xh}$ 的梯度（结构与 $W_{hh}$ 相同）：**

$$
\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{xh}}
$$

**LSTM 中的对比：**

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t) \quad \text{vs} \quad \frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}
$$

核心差异：LSTM 的梯度路径是逐元素对角矩阵（无矩阵乘法、无非线性压缩），Vanilla RNN 的梯度路径包含 $W_{hh}$ 矩阵乘法和 $\tanh'$ 压缩。

---

### 3. Truncated BPTT

#### 3.1 基本思路

对于极长的序列（如一本书有几百万字符），完整 BPTT 在计算上不可行（需要存储整个序列的中间激活值，内存爆炸）。Truncated BPTT 的做法是：

1. 将长序列切分成固定长度 $k$ 的片段（如 100 步）
2. 在每个片段内做完整的 BPTT
3. 将前一个片段的最终隐藏状态传给下一个片段，但**截断**梯度流

具体而言：片段 1 的 $[x_1, \dots, x_{100}]$ 做完 BPTT 后更新参数，然后片段 2 的 $[x_{101}, \dots, x_{200}]$ 接收 $h_{100}$ 作为初始隐藏状态继续计算，但 $h_{100}$ 的计算图已被截断，损失对片段 2 的梯度不会反向传播回片段 1 的参数。

#### 3.2 PyTorch 实现

核心操作是 `detach()`——它创建一个与原张量共享数据但脱离计算图的新张量。梯度无法穿过 `detach()` 的边界，从而实现了跨片段的梯度截断。

```python
h = initHidden(batchSize)

for batch in dataloader:
    h = h.detach()
    output, h = model(batch, h)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

在每次迭代开始时调用 `h.detach()`，使得当前片段的 BPTT 只能反向传播到当前片段的第一时间步，无法继续往前追溯到上一个片段。

#### 3.3 Truncation Length 的选择

| 截断长度 | 优点 | 缺点 |
|----------|------|------|
| 短（20-50） | 内存小、计算快 | 无法学习超过长度的依赖 |
| 中（100-200） | **平衡点** | 字符级语言模型的常用设置 |
| 长（500+） | 更长依赖 | 内存大、梯度消失风险 |
| 完整 BPTT | 全部依赖 | 内存爆炸、序列长时不可行 |

#### 3.4 Truncated BPTT 的局限性

截断 BPTT 有一个根本限制：模型**无法学习跨片段的长距离依赖**。如果序列中的关键信息（如 "France"）在第 50 步出现，而需要它的位置（如填空 "French"）在第 200 步，且截断长度为 100，这两个位置将分属不同片段，梯度无法跨越。

这就是为什么尽管有 Truncated BPTT，LSTM/GRU 的门控机制仍然至关重要——在片段内部（如 100 步），门控确保梯度不会消失。

---

### 4. 梯度裁剪

#### 4.1 数学操作

梯度裁剪是应对梯度**爆炸**（而非消失）的标准方法：

$$
\text{if } ||g|| > \text{threshold}: \quad g = g \cdot \frac{\text{threshold}}{||g||}
$$

- $g$：所有参数的梯度拼接成的向量
- $||g||$：梯度的 L2 范数
- 不改变方向，只限制大小

#### 4.2 为什么裁剪有效？

梯度爆炸时，某一步的 $||g||$ 可能是 100、1000 甚至更大。如果不加限制，参数更新量 $\eta \cdot g$ 会巨大，导致参数跳到极端值区域，Loss 变成 NaN。

裁剪将梯度的范数限制在安全范围内（如 5.0 或 1.0），确保每步的更新量是可控的。

#### 4.3 裁剪不能解决梯度消失

这是一个常见的误解。梯度裁剪只限制梯度的**上限**，不能提升梯度的**下限**。如果梯度因为连乘趋近于 0（梯度消失），裁剪完全无能为力——0 放大多少倍还是 0。

| 问题 | 梯度裁剪有效？ | 真正解决方案 |
|------|:---:|------|
| 梯度爆炸 | ✅ **有效** | 梯度裁剪 |
| 梯度消失 | ❌ 无效 | LSTM/GRU 的门控机制 |

#### 4.4 阈值选择经验

阈值的选择取决于模型规模和任务难度：
- 较浅的模型（如单层 Vanilla RNN）梯度波动相对温和，阈值可设大一些（如 5.0）
- 较深或含 Attention 的模型梯度更容易爆炸，需要更保守的阈值（如 1.0）
- 实际使用中，1.0 到 5.0 是最常见的范围

#### 4.5 PyTorch 实现

`torch.nn.utils.clip_grad_norm_` 对整个模型的所有参数梯度做就地裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

该函数同时返回裁剪前的梯度总范数，可用于监控训练稳定性。如果返回的范数值持续增长或剧烈波动，说明梯度爆炸风险较高，应降低阈值或检查模型结构。

> 更多梯度裁剪实践参见 [[NeuralNetwork/Training/TrainingStability|梯度裁剪指南]]。

---

### 5. LSTM 中的 BPTT

#### 5.1 为什么 LSTM 的 BPTT 更稳定

在 LSTM 中，关键的梯度路径是细胞状态 $C_t$：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

对比 Vanilla RNN：

- Vanilla RNN：$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}$（矩阵乘法 + 非线性压缩）
- LSTM：$\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)$（逐元素乘法，无矩阵乘法，无非线性）

LSTM 的梯度路径**消除了 $W_{hh}$ 的矩阵乘法和 $\tanh$ 的非线性压缩**，只留下一个逐元素的遗忘门因子。当 $f_t[j] \approx 1$ 时，梯度可以几乎无损地穿过这一步。

#### 5.2 GRU 中的类似机制

GRU 的 $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ 提供了类似的机制：

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t)
$$

当 $z_t[j] \approx 0$ 时，梯度无损传递。

---

### 6. BPTT 小结

| 概念 | 要点 |
|------|------|
| BPTT | 梯度沿时间轴反向传播，$\partial L / \partial W_{hh} = \sum_t$（各时间步贡献之和） |
| 梯度连乘 | $\prod \frac{\partial h_j}{\partial h_{j-1}}$ 决定了长距离梯度的存亡 |
| Truncated BPTT | 每 $k$ 步截断梯度流，平衡计算量与长期依赖学习 |
| 梯度裁剪 | 限制梯度范数上限，只解决爆炸，不解决消失 |
| LSTM 的优势 | $C_t$ 更新路径是线性的逐元素操作，梯度可无损传递 |

---

## 综合总结

回望全文，Vanilla RNN 的前向传播与 BPTT 共同构成一个完整的循环计算图：

1. **前向传播**定义了信息在时间维度上如何累积——隐藏状态 $h_t$ 通过 $W_{hh}$ 连接过去与现在，$\tanh$ 提供非线性但也会压缩信号。
2. **BPTT** 定义了梯度如何沿这个计算图反向流动——通过 Jacobian 连乘 $\prod \partial h_j / \partial h_{j-1}$ 将损失信号传递回早期时间步，但 $\tanh'$ 与 $W_{hh}$ 的连乘导致梯度指数衰减。
3. **梯度消失**是 Vanilla RNN 的结构性缺陷——前向传播中信息的衰减在反向传播中表现为梯度的消失，二者是同一枚硬币的两面。
4. **LSTM/GRU** 通过门控机制在计算图中创造了"高速公路"——梯度可以沿细胞状态 $C_t$ 几乎无损地穿行，从根本上绕开了连乘衰减问题。

> 深入 LSTM 参见 [[NeuralNetwork/RNN/LSTM|LSTM 详解]]
> 深入 GRU 参见 [[NeuralNetwork/RNN/GRU|GRU 详解]]
> 回到主文档：[[NeuralNetwork/RNN/RNNOverview|RNN 详解主文档]]
> 梯度消失/爆炸的详细分析：[[NeuralNetwork/Troubleshooting/GradientExplodingVanishing|梯度消失与爆炸]]
