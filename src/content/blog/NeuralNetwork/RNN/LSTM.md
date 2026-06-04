---
title: LSTM 详解：从梯度消失危机到门控记忆网络
date: 2026-05-25
category: NeuralNetwork/RNN
tags:
  - 深度学习
  - 基础
  - 经典架构
description: 从 Hochreiter 1991 年对梯度消失的数学分析出发，完整讲解 LSTM 的诞生背景、CEC 核心洞察、三门控机制的逐公式推导、手算示例、参数追踪与 PyTorch 实现（含 nn.LSTM 和从零手写 LSTMCell），并回顾其沉寂与复兴的历史轨迹。
image: https://img.yumeko.site/file/blog/LongShortTermMemory.webp
status: published
---

## 概述

长短期记忆网络（LSTM, Long Short-Term Memory）是序列建模领域最重要的发明之一。1997 年，Hochreiter 和 Schmidhuber 通过引入**门控机制**和**细胞状态**，从根本上解决了 Vanilla RNN 的梯度消失问题，使神经网络首次能够可靠地学习百步以上的长距离依赖。

本文从历史背景出发，先回顾 Hochreiter 1991 年对梯度消失问题的数学分析，再引入 CEC（常量误差传送带）这一核心洞察，随后逐公式拆解现代 LSTM 的三门控机制，配合完整手算例子、参数量追踪、两种 PyTorch 实现方式（`nn.LSTM` 与从零手写 `LSTMCell`），最后回顾 LSTM 从沉寂到复兴的历史轨迹及其对深度学习的深远影响。

---

## 1. 诞生背景：梯度消失的理论分析

### 1.1 Hochreiter 的毕业论文（1991）

Sepp Hochreiter 在 1991 年的毕业论文（导师：Jurgen Schmidhuber）中对 RNN 的梯度消失问题做了严格的数学分析——这是该问题首次被系统性研究。

核心发现：在 Vanilla RNN 中，反向传播的误差信号随时间指数衰减：

$$
\left|\left| \frac{\partial L_T}{\partial h_1} \right|\right| \propto \prod_{t=2}^{T} \lambda_t
$$

其中 $\lambda_t$ 是 $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}$ 的特征值。当 $|\lambda_t| < 1$ 时（在 Vanilla RNN 中几乎总是如此），连乘导致指数衰减。这个分析为 LSTM 的设计提供了明确的方向：**创造一个 $\lambda_t \approx 1$ 的梯度路径**。

### 1.2 1990 年代中期的困境

到 1990 年代中期，RNN 研究陷入困境：理论分析已证明梯度消失的不可避免性，Elman RNN 的实践经验证实了 10-20 步的实用上限，但尚无有效的解决方案。

---

## 2. CEC：LSTM 的核心洞察

### 2.1 Vanilla RNN 的根本缺陷

回顾 Vanilla RNN 的隐藏状态更新：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

信息从 $h_{t-1}$ 到 $h_t$ 只有一条路径——经过矩阵乘法 $W_{hh}$ 和 $\tanh$ 非线性。这条路径在每个时间步都会**压缩和扭曲**信息，导致：
- 长距离的梯度指数衰减至 0（梯度消失）
- 早期输入的信息在经过多次 $\tanh$ 压缩后逐级衰减

### 2.2 常量误差传送带（CEC）

![CEC.png](https://img.yumeko.site/file/articles/LongShortTermMemory/CEC.webp)

Hochreiter 和 Schmidhuber 在 1997 年的突破性想法是：**如果信息可以通过一个没有非线性、没有矩阵乘法的线性单元传递，梯度就能无损流动**。他们把这个线性单元称为 **Constant Error Carousel（CEC，常量误差传送带）**。

CEC 的核心是纯加法更新：

$$
C_t = C_{t-1} + \text{(门控输入)}
$$

注意这里的更新方式是**加法**，而不是 Vanilla RNN 的矩阵乘法 + 非线性。对 $C_{t-1}$ 的梯度：

$$
\frac{\partial C_t}{\partial C_{t-1}} = I
$$

当没有任何新信息写入时，梯度以 **1.0 的保真度**穿过这一时间步——完全无损。

### 2.3 LSTM 的设计哲学

LSTM 的答案是：**引入一条几乎无损耗的信息传递路径**。在这条路径上，信息可以经过多个时间步后仍然保持可用的信号强度。

这条路径就是**细胞状态（Cell State）$C_t$**。它不是 $h_t$ 的替代品——$C_t$ 是内部长期存储，$h_t$ 是对外暴露的短期工作记忆。两者各司其职。

![RNNVSLSTM.png](https://img.yumeko.site/file/articles/LongShortTermMemory/RNNVSLSTM.webp)

---

## 3. LSTM 的演进：从两门到三门

### 3.1 原始 LSTM（1997 年版）：两个门

![LSTM.png](https://img.yumeko.site/file/articles/LSTMMilestone/LSTM.webp)

原始 LSTM 论文中的结构与现代 LSTM 有一个关键差异：**没有遗忘门**。只有两个门：

**输入门 $i_t$**——控制写入 CEC 的信息量：

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**输出门 $o_t$**——控制从 CEC 读取的信息量：

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**候选细胞状态**：

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**细胞状态更新**（纯加法，无法主动遗忘）：

$$
C_t = C_{t-1} + i_t \odot \tilde{C}_t
$$

**隐藏状态输出**（从 $C_t$ 经输出门过滤后对外暴露）：

$$
h_t = o_t \odot \tanh(C_t)
$$

没有遗忘门意味着 $C_t$ 只能增长——因为 $i_t \odot \tilde{C}_t$ 总是加到 $C_{t-1}$ 上。实践中通过输入门在大多数时间步接近 0 来缓解，但这不是对遗忘的主动控制。

### 3.2 遗忘门的加入（Gers, 2000）

2000 年，Felix Gers（Schmidhuber 的学生）在博士论文中提出给 LSTM 加上遗忘门。这个看似微小的改动使得 LSTM 能够主动选择遗忘旧信息，对于处理无限长流式数据至关重要。

**遗忘门 $f_t$**：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**加入遗忘门后的细胞状态更新**（现代 LSTM 的标准形式）：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

### 3.3 现代 LSTM 完整公式

以下是当前业界通用的 LSTM（含遗忘门）在一个时间步内的全部计算。

![LSTMCell.png](https://img.yumeko.site/file/articles/LongShortTermMemory/LSTMCell.webp)

**三个门控信号：**

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{遗忘门：决定丢弃 } C_{t-1} \text{ 中的哪些信息} \\[4pt]
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{输入门：决定在哪些位置写入新信息} \\[4pt]
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &\text{输出门：决定 } C_t \text{ 的哪些部分对外暴露}
\end{aligned}
$$

**候选值与状态更新：**

$$
\begin{aligned}
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad &\text{候选细胞状态：新信息的候选内容} \\[4pt]
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &\text{细胞状态更新：遗忘旧信息 + 写入新信息} \\[4pt]
h_t &= o_t \odot \tanh(C_t) \quad &\text{隐藏状态输出：从 } C_t \text{ 筛选后对外暴露}
\end{aligned}
$$

**各门控详解：**

**遗忘门 $f_t$**：
- $\sigma$ 是 sigmoid 函数：$\sigma(x) = \frac{1}{1+e^{-x}}$，输出范围 (0, 1)
- $f_t$ 的每个元素是一个 0~1 之间的标量
- $f_t[j] \approx 1$：几乎完全保留 $C_{t-1}[j]$
- $f_t[j] \approx 0$：几乎完全丢弃 $C_{t-1}[j]$

例如，在文本序列中出现话题转换时，遗忘门对旧话题相关的信息维度进行衰减，为新话题的信息写入腾出表示空间。

**输入门 $i_t$ 与候选 $\tilde{C}_t$**：
- $i_t$：决定在哪些位置写入新信息（Sigmoid，输出 0 到 1 的开关量）
- $\tilde{C}_t$：生成候选的新内容（Tanh，输出 -1 到 1 的信息值）
- $i_t \odot \tilde{C}_t$：只在输入门打开的位置写入候选值

例如，当序列中出现一个新实体"张三"时，输入门在相关维度激活，$\tilde{C}_t$ 将"张三"的表示信息写入细胞状态。

**输出门 $o_t$ 与隐藏状态 $h_t$**：
- $o_t$：决定细胞状态的哪些部分需要暴露给外部（下一层、下一个时间步、输出）
- $\tanh(C_t)$：将细胞状态压缩到 (-1, 1)，防止数值发散
- $h_t$：对外部可见的"工作记忆"

### 3.4 两条信息流的角色分工

- $C_t$（细胞状态）：内部长期记忆。更新路径 $C_{t-1} \to C_t$ 是逐元素的线性加和，梯度通过 $\text{diag}(f_t)$ 传递——无矩阵乘法，无非线性压缩。
- $h_t$（隐藏状态）：对外暴露的工作记忆。从 $C_t$ 经 $\tanh$ 压缩和 $o_t$ 筛选后产生，传给下一时间步的 Hidden Layer 和输出层。

$h_t$ 传给下一时间步的 $h_{t}$，而 $C_t$ 传给下一时间步的 $C_{t}$。两条信息流并行：$C_t$ 作为内部长期状态（通过线性加性路径更新），$h_t$ 作为对外暴露的输出状态（由输出门控制从 $C_t$ 中读取哪些信息）。

**参数总览：** LSTM 共有四套权重矩阵和偏置——遗忘门 $(W_f, b_f)$、输入门 $(W_i, b_i)$、输出门 $(W_o, b_o)$ 和候选值 $(W_C, b_C)$。每个 $W$ 的列数为 $d_h + d_x$（$h_{t-1}$ 和 $x_t$ 拼接后的维度），行数为 $d_h$。总参数量（不含偏置）为 $4 \times d_h \times (d_h + d_x)$。

### 3.5 遗忘门偏置的初始化策略

Gers 等人发现，遗忘门的偏置 $b_f$ 不应初始化为 0——这会让 $f_t$ 的初始值约 $\sigma(0) = 0.5$，导致训练早期就无差别地遗忘一半信息。更好的做法是将 $b_f$ 初始化为一个较大的正值（如 1.0），此时 $f_t \approx \sigma(1) \approx 0.73$，倾向于保留信息。PyTorch 的 `nn.LSTM` 默认已在内部做了类似优化。

---

## 4. 完整手算例子

![LSTMCalculate.png](https://img.yumeko.site/file/articles/LongShortTermMemory/LSTMCalculate.webp)

使用 input=2, hidden=2 的简化设置，追踪 LSTM 在两个时间步内的完整计算过程。

### 4.1 时间步 t=1

初始状态 $h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$，$C_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$。

输入 $x_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$。

由于 $h_0 = 0$ 且 $C_0 = 0$，第一步完全由当前输入驱动。

**遗忘门**：

简化权重：

$$
W_{fx} = \begin{pmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{pmatrix}
\qquad
W_{fh} = \begin{pmatrix} 0.2 & -0.1 \\ -0.3 & 0.4 \end{pmatrix}
\qquad
b_f = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

$$
W_{fx} \cdot x_1 = \begin{pmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{pmatrix} \cdot \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.3 \\ 0.5 \end{pmatrix}
$$

$$
f_1 = \sigma\left( \begin{pmatrix} 0.3 \\ 0.5 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix} \right)
= \sigma\left( \begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix}
$$

$f_1 = [0.60, 0.60]$ 意味着保留 60% 的 $C_0$——但 $C_0 = 0$，所以遗忘门的输出此时无实际影响。

**输入门**：

设 $W_i \cdot [h_0, x_1] + b_i = \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix}$：

$$
i_1 = \sigma\left( \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix} \right) \approx \begin{pmatrix} 0.69 \\ 0.57 \end{pmatrix}
$$

$i_1 = [0.69, 0.57]$ 意味着两个维度都相当程度地"打开"以写入新信息。

**候选细胞状态**：

设 $W_C \cdot [h_0, x_1] + b_C = \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix}$：

$$
\tilde{C}_1 = \tanh\left( \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix} \right) \approx \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
$$

$\tanh$ 将候选值压缩到 $(-1, 1)$ 区间。维度 1 的候选值（0.834）比维度 0（0.537）更大。

**细胞状态更新**：

$$
C_1 = f_1 \odot C_0 + i_1 \odot \tilde{C}_1
= \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix} \odot \begin{pmatrix} 0 \\ 0 \end{pmatrix}
+ \begin{pmatrix} 0.69 \\ 0.57 \end{pmatrix} \odot \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.69 \times 0.537 \\ 0.57 \times 0.834 \end{pmatrix}
= \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}
$$

由于 $C_0 = 0$，$C_1$ 完全由新输入的信息构成。$f_1$ 的遗忘效果在初始步不存在。

**输出门与隐藏状态**：

设 $W_o \cdot [h_0, x_1] + b_o = \begin{pmatrix} 0.7 \\ 1.1 \end{pmatrix}$：

$$
o_1 = \sigma\left( \begin{pmatrix} 0.7 \\ 1.1 \end{pmatrix} \right) \approx \begin{pmatrix} 0.67 \\ 0.75 \end{pmatrix}
$$

$$
h_1 = o_1 \odot \tanh(C_1) \approx \begin{pmatrix} 0.67 \\ 0.75 \end{pmatrix} \odot \begin{pmatrix} 0.354 \\ 0.444 \end{pmatrix}
= \begin{pmatrix} 0.237 \\ 0.333 \end{pmatrix}
$$

$h_1$ 是经过输出门筛选的 $C_1$ 的对外暴露版本。$h_1 \neq C_1$——输出门在两个维度上过滤掉了约 25%-33% 的信息。

### 4.2 时间步 t=2

$x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$，$h_1 = \begin{pmatrix} 0.237 \\ 0.333 \end{pmatrix}$，$C_1 = \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}$。

**遗忘门（首次真正发挥作用）**：

设 $f_2 \approx \begin{pmatrix} 0.45 \\ 0.55 \end{pmatrix}$。

维度 0 遗忘门 0.45——丢弃 55% 的 $C_1$ 旧信息。维度 1 遗忘门 0.55——保留略多。

**输入门与候选**：

$$
i_2 \approx \begin{pmatrix} 0.30 \\ 0.80 \end{pmatrix}, \quad \tilde{C}_2 \approx \begin{pmatrix} -0.40 \\ 0.60 \end{pmatrix}
$$

**细胞状态更新**（遗忘门 + 输入门协同）：

$$
C_2 = \begin{pmatrix} 0.45 \\ 0.55 \end{pmatrix} \odot \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}
+ \begin{pmatrix} 0.30 \\ 0.80 \end{pmatrix} \odot \begin{pmatrix} -0.40 \\ 0.60 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.45 \times 0.371 \\ 0.55 \times 0.475 \end{pmatrix}
+ \begin{pmatrix} 0.30 \times (-0.40) \\ 0.80 \times 0.60 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.167 \\ 0.261 \end{pmatrix} + \begin{pmatrix} -0.120 \\ 0.480 \end{pmatrix}
= \begin{pmatrix} 0.047 \\ 0.741 \end{pmatrix}
$$

两个维度呈现截然不同的信息动态：
- **维度 0**：$C_2[0] = 0.167 - 0.120 = 0.047$。大部分旧信息（$C_1[0] = 0.371$）被遗忘门（0.45）丢弃，新写入的候选值为负（$-0.40 \times 0.30 = -0.120$），两者相抵。维度 0 实际上"重置"了记忆。
- **维度 1**：$C_2[1] = 0.261 + 0.480 = 0.741$。旧信息保留了一半多（$0.475 \times 0.55 = 0.261$），大量新信息被写入（$0.60 \times 0.80 = 0.480$）。维度 1 在"累积"信息。

**输出门**：

设 $o_2 \approx \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix}$：

$$
h_2 = \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \tanh\left( \begin{pmatrix} 0.047 \\ 0.741 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \begin{pmatrix} 0.047 \\ 0.629 \end{pmatrix}
= \begin{pmatrix} 0.042 \\ 0.189 \end{pmatrix}
$$

维度 0 的 $C_2[0] = 0.047$ 很小，但输出门几乎完全打开（0.90），$h_2[0] = 0.042$。维度 1 的 $C_2[1] = 0.741$ 很大，但输出门只打开了 30%，$h_2[1] = 0.189$。输出门使得 $h_t$ 不等于 $C_t$——模型可以选择性地对外"保密"某些内部记忆。

---

## 5. LSTM 参数量和形状追踪

### 5.1 四套权重矩阵

LSTM 有 3 个门 + 1 个候选值生成，共四套参数。PyTorch 将它们打包为两个大矩阵：

| PyTorch 参数名 | 实际形状 | 包含内容 |
|------|------|------|
| `weight_ih` | (4 x hidden, input) | $W_{ix}, W_{fx}, W_{Cx}, W_{ox}$ 纵向拼接 |
| `weight_hh` | (4 x hidden, hidden) | $W_{ih}, W_{fh}, W_{Ch}, W_{oh}$ 纵向拼接 |
| `bias_ih` | (4 x hidden,) | $b_i, b_f, b_C, b_o$ 纵向拼接 |
| `bias_hh` | (4 x hidden,) | 另一组偏置（PyTorch 特有） |

### 5.2 参数量计算

以单个 LSTM 层（input=256, hidden=256）为例：

| 组件 | 计算 | 参数量 |
|------|------|--------|
| weight_ih | 4 x 256 x 256 | 262,144 |
| weight_hh | 4 x 256 x 256 | 262,144 |
| bias_ih | 4 x 256 | 1,024 |
| bias_hh | 4 x 256 | 1,024 |
| **单层合计** | | **526,336** |

### 5.3 双向 LSTM 的参数量

双向 LSTM 有两套独立的权重（正向+反向），但共享输入。单层双向的参数约是单向的 2 倍：约 1,052,672。

---

## 6. PyTorch 实现

### 6.1 使用 nn.LSTM

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=300,
    hidden_size=256,
    num_layers=2,
    dropout=0.5,         # 层间 dropout
    bidirectional=True,   # 双向
    batch_first=True,
)

# x: (batch_size, seq_len, input_size)
# 需要同时初始化 h_0 和 c_0
x = torch.randn(B, L, 300)  # 示例
h0 = torch.zeros(2 * 2, B, 256)  # 2层 x 2方向, B, 256
c0 = torch.zeros(2 * 2, B, 256)

output, (hn, cn) = lstm(x, (h0, c0))
# output: (B, L, 512)  —— 拼接了正反向
# hn: (4, B, 256)      —— 所有层的最终隐藏状态
# cn: (4, B, 256)      —— 所有层的最终细胞状态
```

### 6.2 输出形状详解

`nn.LSTM` 返回的 `output` 和 `(hn, cn)` 各有不同的含义：

```python
# x: (B, L, input_size)  —— B=batch, L=seq_len
output, (hn, cn) = lstm(x)

# output: (B, L, D * hidden_size)  其中 D=2（双向）或 1（单向）
#   包含所有时间步、所有层的最后一层输出
#   双向时，每步的输出是 [正向输出; 反向输出] 拼接

# hn: (D * num_layers, B, hidden_size)
#   每个层每个方向的最终隐藏状态
#   hn[-1] 是最后一层反向的最终状态
#   hn[-2] 是最后一层正向的最终状态（双向时）

# cn: (D * num_layers, B, hidden_size)
#   结构同 hn，存储细胞状态的最终值
```

`output` 和 `hn` 的关系：`output` 的最后一个时间步的前半部分（正向）等于 `hn[-2]`，后半部分（反向）等于 `hn[-1]`（双向时）。`output` 用于需要每个时间步输出的场景（如序列标注），`hn` 用于只需要最终表示的场景（如文本分类）。

### 6.3 逐时间步的循环（手动模式）

PyTorch 的 `nn.LSTM` 内部会自动循环处理整个序列。但理解手动循环有助于深入理解：

```python
# 手动逐时间步处理（教学代码，不是高效实现）
h = torch.zeros(1, B, 256)
c = torch.zeros(1, B, 256)
outputs = []

for t in range(seqLen):
    xt = x[:, t:t+1, :]            # (B, 1, input_size)
    _, (h, c) = lstm(xt, (h, c))   # 只处理一个时间步
    outputs.append(h)

output = torch.cat(outputs, dim=1)  # (B, seqLen, hidden_size)
```

### 6.4 从零实现 LSTMCell：公式到代码的精确映射

以下代码实现了 3.3 节中现代 LSTM 的 6 个公式。将 $W \cdot [h_{t-1}, x_t]$ 拆分为分别对 $x_t$ 和 $h_{t-1}$ 做线性变换，避免每次拼接。

```python
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """现代 LSTM 单步单元（含遗忘门）。

    六个公式:
      f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
      i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
      o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
      g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
      c_t = f_t * c_{t-1} + i_t * g_t
      h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 遗忘门 f_t —— 两套线性变换
        # W_if: (hidden_dim, input_dim) —— x_t 对遗忘门的贡献
        self.W_if = nn.Linear(input_dim, hidden_dim, bias=False)
        # W_hf: (hidden_dim, hidden_dim) —— h_{t-1} 对遗忘门的贡献
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 输入门 i_t —— 结构同遗忘门，独立参数
        self.W_ii = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 输出门 o_t
        self.W_io = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 候选细胞状态 g_t (= \tilde{C}_t)
        self.W_ig = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t:    当前输入, (batch, input_dim)
            h_prev: 上一时刻隐藏状态, (batch, hidden_dim)
            c_prev: 上一时刻细胞状态, (batch, hidden_dim)
        Returns:
            h_t: 新隐藏状态, (batch, hidden_dim)
            c_t: 新细胞状态, (batch, hidden_dim)
        """
        # 公式 1: 遗忘门
        # W_if(x_t):    (batch, input_dim)  -> (batch, hidden_dim)
        # W_hf(h_prev): (batch, hidden_dim) -> (batch, hidden_dim)
        f_t = torch.sigmoid(self.W_if(x_t) + self.W_hf(h_prev))

        # 公式 2: 输入门
        i_t = torch.sigmoid(self.W_ii(x_t) + self.W_hi(h_prev))

        # 公式 3: 输出门
        o_t = torch.sigmoid(self.W_io(x_t) + self.W_ho(h_prev))

        # 公式 4: 候选细胞状态
        g_t = torch.tanh(self.W_ig(x_t) + self.W_hg(h_prev))

        # 公式 5: 细胞状态更新
        # f_t * c_{t-1}: 遗忘门控制的旧信息保留
        # i_t * g_t:     输入门控制的新信息写入
        c_t = f_t * c_prev + i_t * g_t

        # 公式 6: 隐藏状态输出
        # o_t * tanh(c_t): 输出门控制的信息暴露
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
```

### 6.5 手动展开多步序列

```python
def lstm_forward(cell, input_sequence, h_0=None, c_0=None):
    """
    用 LSTMCell 手动展开整个序列。

    Args:
        cell:           LSTMCell 实例
        input_sequence: (batch, seq_len, input_dim)
        h_0, c_0:       初始状态, None 则用零向量
    Returns:
        outputs: (batch, seq_len, hidden_dim)
        h_final, c_final: (batch, hidden_dim)
    """
    batch_size, seq_len, _ = input_sequence.shape
    hidden_dim = cell.W_hf.out_features

    if h_0 is None:
        h = torch.zeros(batch_size, hidden_dim, device=input_sequence.device)
        c = torch.zeros(batch_size, hidden_dim, device=input_sequence.device)
    else:
        h, c = h_0, c_0

    outputs = []
    for t in range(seq_len):
        x_t = input_sequence[:, t, :]   # (batch, input_dim)
        h, c = cell(x_t, h, c)           # 一步 LSTM
        outputs.append(h)

    outputs = torch.stack(outputs, dim=1)
    return outputs, h, c
```

---

## 7. LSTM 的梯度传递分析

LSTM 解决梯度消失的核心在于细胞状态更新公式 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$。对 $C_{t-1}$ 求偏导：

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

这是一个**对角矩阵**——每个元素独立地由遗忘门控制，没有矩阵乘法（$W_{hh}$），没有非线性（$\tanh$）。对比 Vanilla RNN：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}
$$

LSTM 消除了两个梯度衰减源：$W_{hh}$ 的矩阵乘法（旋转和缩放）和 $\tanh'$ 的压缩（最大 1.0，通常远小于 1）。

如果 $f_t[j] = 0.95$，那么该维度的梯度在穿过这个时间步时只衰减 5%。即使经过 100 步：

$$
0.95^{100} \approx 0.006
$$

虽然衰减了不少，但仍然是有效的梯度信号（仍在 float32 的有效范围内）。对比 Vanilla RNN 在 100 步后梯度约为 $10^{-61}$，LSTM 的优势是天文数字级别的。

**梯度传递的核心优势总结：** $C_t$ 对 $C_{t-1}$ 的偏导是对角矩阵 $\text{diag}(f_t)$，梯度在时间轴上每步仅乘以遗忘门的对应元素。当 $f_t[j] \approx 1$ 时，梯度几乎无损地跨越该时间步。这与 Vanilla RNN 中梯度每步被矩阵乘法 $W_{hh}$ 反复压缩和旋转形成了根本区别。

---

## 8. 遗忘门偏置初始化代码

```python
def init_lstm_forget_gate_bias(lstm, value=1.0):
    """将 LSTM 遗忘门的偏置初始化为正值, 使训练初期倾向于保留信息。"""
    # PyTorch LSTM 的 bias_ih 和 bias_hh 排列: [i, f, g, o] 各 hidden_dim 个
    hidden_dim = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            # 遗忘门偏置在每段 hidden_dim 的第二个 quarter
            param.data[hidden_dim:2 * hidden_dim].fill_(value)
```

此策略使训练早期 $f_t \approx 0.73$，模型倾向于保留旧信息而非遗忘——这避免了在门控策略尚未学会时就将有用信号丢弃。

---

## 9. LSTM 的变体

### 9.1 Peephole Connections

标准 LSTM 中，门控只依赖于 $h_{t-1}$ 和 $x_t$。Peephole LSTM（Gers & Schmidhuber, 2000）还让门控"偷看"细胞状态：

$$
f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f)
$$
$$
i_t = \sigma(W_i \cdot [C_{t-1}, h_{t-1}, x_t] + b_i)
$$
$$
o_t = \sigma(W_o \cdot [C_t, h_{t-1}, x_t] + b_o)
$$

让门控直接感知细胞状态的值，可以做出更精细的控制决策。PyTorch 的 LSTM 不支持 peephole，需要自定义实现。

### 9.2 Coupled Forget-Input Gate

另一种变体将遗忘门和输入门耦合：$i_t = 1 - f_t$。这意味着"写入的新信息量 = 丢弃的旧信息量"，细胞状态的总信息量保持恒定。GRU 实际上就采用了这种思路（通过更新门 $z_t$）。

---

## 10. 沉寂与复兴（1997-2015）

![LSTMTimeline.png](https://img.yumeko.site/file/articles/LSTMMilestone/LSTMTimeline.webp)

LSTM 在 1997 年发表后并未立即改变世界，直到 2010 年代才迎来广泛采用。原因包括计算资源不足、数据量不足、优化技术不成熟以及 NLP 领域长期被统计方法主导。

| 年份 | 事件 | 意义 |
|------|------|------|
| 2011 | Graves 等人用 LSTM 赢得手写识别比赛 | 首次大规模 LSTM 应用 |
| 2013 | Graves 等人用 LSTM 实现语音识别突破 | 在 TIMIT 上达到当时最优 |
| 2014 | Sutskever 用 LSTM Seq2Seq 做机器翻译 | 接近当时最好的统计机器翻译系统 |
| 2015 | Google 用 LSTM 做语音识别，降低 49% 词错率 | 工业级应用的标志 |
| 2016 | Google 神经机器翻译系统（GNMT）上线 | LSTM Seq2Seq 进入数亿用户的产品 |

到 2016 年，LSTM 已成为序列学习的**事实标准**。

---

## 11. 历史定位与影响

即使在 Transformer 主导的 2020 年代，LSTM 的设计思想仍有深远影响：

1. **门控机制**成为神经网络设计的核心原语——门控是"软性条件分支"，允许网络基于数据动态决定信息流
2. **CEC 的设计思想**影响了后来的 Highway Networks 和 ResNet——跳跃连接本质上也是为梯度提供一条无损传递的路径
3. **$C_t$ / $h_t$ 分离**启发了 Neural Turing Machine 和 Memory Networks 等可微分外部记忆架构
4. **Attention 最初就是为了增强 LSTM Seq2Seq 而设计的**——可以说 LSTM 直接催生了 Transformer

Transformer 在长序列和并行化方面有明显优势，但 LSTM 在小数据、流式处理、推理效率方面仍有竞争力。2020 年代，Mamba、RWKV 等线性 RNN 重新发现了 LSTM 的核心价值——推理时比 Transformer 更高效的序列建模。

LSTM 的核心理念——"为梯度提供一条无损传递的路径"——超越了具体架构，成为深度学习设计的普适原则。

---

## 12. 总结

| 概念 | 公式/描述 | 作用 |
|------|------|------|
| 细胞状态 $C_t$ | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | 长期记忆，梯度可无损通过 |
| 隐藏状态 $h_t$ | $h_t = o_t \odot \tanh(C_t)$ | 短期工作记忆，对外暴露 |
| 遗忘门 $f_t$ | $\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | 控制丢弃多少旧记忆 |
| 输入门 $i_t$ | $\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | 控制写入多少新信息 |
| 输出门 $o_t$ | $\sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | 控制对外暴露多少记忆 |
| CEC | $C_{t-1} \to C_t$ 的线性路径 | 梯度通过逐元素独立控制的线性路径稳定传递 |

LSTM 设计的核心在于将状态存储和输出解耦——$C_t$ 作为受门控保护的内部状态（遗忘门和输入门控制其读写的规模和时机），$h_t$ 作为有选择性的输出暴露（输出门控制对外可见哪些内部信息）。这种解耦是 LSTM 能够学习长距离依赖的根本原因。

---

> 对比 GRU 参见 [[NeuralNetwork/RNN/GRU|GRU 详解]]
> 架构对比参见 [[NeuralNetwork/RNN/RNNComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNNOverview|RNN 详解主文档]]
