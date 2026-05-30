---
title: RNN 循环层详解：隐藏状态的数学本质
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
description: 从隐藏状态更新公式到手算例子，深入理解 Vanilla RNN 的每一步计算、参数矩阵的物理含义、输出模式、以及根本局限。
image: https://img.yumeko.site/file/blog/RecurrentLayer.webp
status: draft
---

## 1. RNN Cell 的数学定义

### 1.1 两个核心公式

Vanilla RNN（也叫 Elman RNN、Simple RNN）在每个时间步 $t$ 执行以下计算：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

$$
y_t = W_{hy} \cdot h_t + b_y
$$

第一行是**状态更新**，第二行是**输出计算**。这两个公式就是 RNN 的全部。

### 1.2 符号约定

| 符号 | 含义 | 形状（以 Char-RNN 为例） |
|------|------|------|
| $x_t$ | 第 $t$ 步的输入向量 | (65,) — 字符的 embedding |
| $h_{t-1}$ | 上一时刻的隐藏状态 | (256,) |
| $h_t$ | 当前时刻的隐藏状态 | (256,) |
| $y_t$ | 当前时刻的输出（logits） | (65,) — 每个字符的得分 |
| $W_{xh}$ | 输入→隐藏权重 | (256, 65) |
| $W_{hh}$ | 隐藏→隐藏权重 | (256, 256) |
| $W_{hy}$ | 隐藏→输出权重 | (65, 256) |
| $b_h$ | 隐藏层偏置 | (256,) |
| $b_y$ | 输出层偏置 | (65,) |

---

## 2. 参数矩阵的物理含义

![RNNCell.png](https://img.yumeko.site/file/articles/RecurrentLayer/RNNCell.webp)

### 2.1 $W_{xh}$：输入编码器

$W_{xh}$ 将当前输入 $x_t$ 映射到隐藏空间。

$$W_{xh} \cdot x_t + \text{(来自 } h_{t-1} \text{ 的信息)}$$

- 形状：(hidden_size, input_size)，每一行是一个"检测器"，检测输入的某种模式
- 如果 $W_{xh}$ 的某一行与 $x_t$ 高度匹配，该维度的激活值就高

### 2.2 $W_{hh}$：状态转移矩阵

$W_{hh}$ 是 RNN **最核心**的矩阵——它决定了上一时刻的记忆如何影响当前时刻。

$$W_{hh} \cdot h_{t-1}$$

- 形状：(hidden_size, hidden_size)，方阵，定义了隐藏空间中的"动力学"
- 每一行决定了上一时刻的 256 维隐藏状态如何组合产生当前时刻该维度的输入
- 这个矩阵的特征值决定了信息在时间维度上能传播多远（特征值 < 1 会导致信息衰减）

### 2.3 $W_{hy}$：输出解码器

$W_{hy}$ 将"内部记忆"解码为"外部输出"。

$$W_{hy} \cdot h_t$$

- 形状：(output_size, hidden_size)，将隐藏状态映射到输出空间
- 在 Char-RNN 中是 (65, 256)：65 个字符各有一个"打分器"，评估当前隐藏状态支持输出各字符的程度

### 2.4 $b_h$ 和 $b_y$：偏置项

- $b_h$：允许神经元在没有有效输入时仍有基线激活水平
- $b_y$：允许某些输出类有更高的先验概率（如英语中 'e' 比 'z' 常见得多）

---

## 3. 完整手算例子

![RNNCalculate.png](https://img.yumeko.site/file/articles/RecurrentLayer/RNNCalculate.webp)

使用一个极端简化的设置来手算 RNN 的前向传播全过程：

- input_size = 2，hidden_size = 2，output_size = 3（只有三个字符 "H", "e", "l"）
- 初始隐藏状态 $h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

### 3.1 权重矩阵

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

### 3.2 时间步 1

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

### 3.3 时间步 2

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

### 3.4 时间步 3

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

### 3.5 关键观察

![InformationAttenuation.png](https://img.yumeko.site/file/articles/RecurrentLayer/InformationAttenuation.webp)

手算揭示了 Vanilla RNN 的几个重要特性：

1. **信息是累加的**：每个时间步的 $h_t$ 都是 $x_t$ 和 $h_{t-1}$ 的**加和**（经过非线性），早期输入的信息不会突然消失
2. **但信号强度会逐步衰减**：每一步的 $\tanh$ 会将值压缩到 (-1, 1)，多次压缩后早期信息的"信号强度"逐步衰减
3. **$W_{hh}$ 是关键**：它的特征值决定了信息能传播多远。如果特征值普遍小于 1，信息会指数衰减——这就是梯度消失的结构性根源

---

## 4. 形状追踪表

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

## 5. RNN 的输出模式

PyTorch 的 `nn.RNN` forward 返回两个值：`output` 和 `h_n`。

### 5.1 output：所有时间步的隐藏状态

```python
output, h_n = rnn(x, h_0)
# output.shape = (B, seq_len, hidden_size)  当 batch_first=True
```

`output` 包含**每个时间步**的隐藏状态——`output[:, t, :]` 就是 $h_t$。

### 5.2 h_n：最后一个时间步的隐藏状态

```python
# h_n.shape = (num_layers, B, hidden_size)
```

`h_n` 就是 $h_T$（最后一个时间步）。对于单层 RNN，`h_n.squeeze(0) == output[:, -1, :]`。

### 5.3 Many-to-Many vs Many-to-One

![TwoOutput.png](https://img.yumeko.site/file/articles/RecurrentLayer/TwoOutput.webp)

```python
# Many-to-Many（序列标注）：使用 output
for t in range(seq_len):
    loss += criterion(fc(output[:, t, :]), labels[:, t])

# Many-to-One（序列分类）：使用 h_n
logits = fc(h_n.squeeze(0))
loss = criterion(logits, labels)
```

---

## 6. 梯度消失的数学根源

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

![GradientMultiplication.png](https://img.yumeko.site/file/articles/RecurrentLayer/GradientMultiplication.webp)

在通过 $T$ 个时间步反向传播时：

$$
\frac{\partial h_T}{\partial h_1} \approx \prod_{t=2}^{T} \big( \text{diag}(\tanh'(\cdot)) \cdot W_{hh} \big)
$$

如果每一步的谱范数小于 1，这个连乘会随着 $T$ 增大而指数衰减至 0——这就是 Vanilla RNN 无法学习长距离依赖的根本原因。

### 6.1 具体数值

当 $\tanh' \approx 0.5$，$||W_{hh}|| \approx 0.5$ 时，每一步的梯度衰减因子约 0.25。100 步后：

$$
||\frac{\partial h_{100}}{\partial h_1}|| \approx (0.25)^{100} \approx 6.2 \times 10^{-61}
$$

这意味着第 1 步的输入对第 100 步的损失**完全没有可用的梯度信号**。模型的有效信息窗口约为 20 步，超过此范围后梯度已衰减到无法用于参数更新。

---

## 7. PyTorch 代码实现

### 7.1 基础用法

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

### 7.2 batch_first=True vs False

| 参数 | 输入形状 | 输出形状（output） | 输出形状（hn） |
|------|----------|-------------------|----------------|
| `batch_first=True` | `(B, L, in)` | `(B, L, hidden)` | `(layers, B, hidden)` |
| `batch_first=False` | `(L, B, in)` | `(L, B, hidden)` | `(layers, B, hidden)` |

`batch_first=True` 更符合直觉，两个项目都使用此设置。

### 7.3 多对多输出

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

## 8. 总结：Vanilla RNN 的定位

Vanilla RNN 是所有后续 RNN 变体的**原型**。它引入了循环连接和隐藏状态的基本概念，但因其梯度消失问题，在现代实践中几乎不被直接使用。

| 学习目标 | 推荐模型 |
|----------|----------|
| 理解 RNN 的原理 | **Vanilla RNN**（最简洁） |
| 实际工程项目 | LSTM 或 GRU |
| 快速实验 | GRU（参数少、收敛快） |

理解了本节的内容，你就掌握了所有 RNN 变体共享的核心机制。LSTM 和 GRU 只是在这个基础上增加了门控——但它们解决的是同一个根本问题。

> 深入 LSTM 参见 [[NeuralNetwork/RNN/Foundations/LongShortTermMemory|LSTM 详解]]
> 深入 GRU 参见 [[NeuralNetwork/RNN/Foundations/GatedRecurrentUnit|GRU 详解]]
> 深入 BPTT 参见 [[RNNBPTT|BPTT 详解]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
