---
title: LSTM 详解：门控机制如何实现长短期记忆
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
description: 从遗忘门、输入门、输出门到细胞状态，逐公式拆解 LSTM 的每一个门控机制，配合完整手算例子、形状追踪和 PyTorch 代码。
image: https://img.yumeko.site/file/blog/LongShortTermMemory.webp
status: draft
---

## 1. 从 Vanilla RNN 到 LSTM：为什么需要门控？

### 1.1 Vanilla RNN 的根本缺陷

回顾 Vanilla RNN 的隐藏状态更新：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)
$$

信息从 $h_{t-1}$ 到 $h_t$ 只有一条路径——经过矩阵乘法 $W_{hh}$ 和 $\tanh$ 非线性。这条路径在每个时间步都会**压缩和扭曲**信息，导致：
- 长距离的梯度指数衰减至 0（梯度消失）
- 早期输入的信息在经过多次 $\tanh$ 压缩后逐级衰减

### 1.2 LSTM 的解决方案

LSTM（Long Short-Term Memory）的答案是：**引入一条几乎无损耗的信息传递路径**。在这条路径上，信息可以经过多个时间步后仍然保持可用的信号强度。

这条路径就是**细胞状态（Cell State）$C_t$**。它不是 $h_t$ 的替代品——$C_t$ 是内部长期存储，$h_t$ 是对外暴露的短期工作记忆。两者各司其职。

![RNNVSLSTM.png](https://img.yumeko.site/file/articles/LongShortTermMemory/RNNVSLSTM.webp)

### 1.3 CEC：LSTM 的核心洞察

**CEC（Constant Error Carousel，常量误差传送带**是 LSTM 最深刻的设计。它的意思是：如果我们可以让细胞状态的更新变成简单的**线性的**加法操作，梯度就能在不衰减的情况下沿时间轴流回任意远的位置。

在原始设计中（1997 年），CEC 就是一条没有非线性、没有矩阵乘法的纯线性路径。后来的遗忘门、输入门、输出门都是为了**控制**这条路径上信息的读写而添加的"阀门"。

![CEC.png](https://img.yumeko.site/file/articles/LongShortTermMemory/CEC.webp)



---

## 2. LSTM 的完整公式系统

![LSTMCell.png](https://img.yumeko.site/file/articles/LongShortTermMemory/LSTMCell.webp)

### 2.1 遗忘门（Forget Gate）

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- $\sigma$ 是 sigmoid 函数：$\sigma(x) = \frac{1}{1+e^{-x}}$，输出范围 (0, 1)
- $f_t$ 的每个元素是一个 0~1 之间的标量
- $f_t[j] \approx 1$：几乎完全保留 $C_{t-1}[j]$
- $f_t[j] \approx 0$：几乎完全丢弃 $C_{t-1}[j]$

例如，在文本序列中出现话题转换时，遗忘门对旧话题相关的信息维度进行衰减，为新话题的信息写入腾出表示空间。

> 历史注记：原始 LSTM（1997 年）并没有遗忘门。遗忘门是 Gers 等人在 2000 年添加的改进，已成为现代 LSTM 的标配。

### 2.2 输入门（Input Gate）

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- $i_t$：决定在哪些位置写入新信息（Sigmoid，输出 0 到 1 的开关量）
- $\tilde{C}_t$：生成候选的新内容（Tanh，输出 -1 到 1 的信息值）
- $i_t \odot \tilde{C}_t$：只在输入门打开的位置写入候选值

例如，当序列中出现一个新实体"张三"时，输入门在相关维度激活，$\tilde{C}_t$ 将"张三"的表示信息写入细胞状态。

### 2.3 细胞状态更新（LSTM 的核心）

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

这是**整个 LSTM 最关键的一行公式**。它由两项加和组成：

- $f_t \odot C_{t-1}$：按遗忘门的指示，选择性保留旧记忆
- $i_t \odot \tilde{C}_t$：按输入门的指示，选择性写入新信息

**为什么这能解决梯度消失？** 对 $C_{t-1}$ 求偏导：

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

这是一个对角矩阵！每个元素独立地由遗忘门控制。如果 $f_t[j] = 0.95$，那么该维度的梯度在穿过这个时间步时只衰减 5%。即使经过 100 步：

$$
0.95^{100} \approx 0.006
$$

虽然衰减了不少，但仍然是有效的梯度信号。对比 Vanilla RNN 在 100 步后梯度 = $10^{-61}$，LSTM 的优势是天文数字级别的。

### 2.4 输出门（Output Gate）

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

- $o_t$：决定细胞状态的哪些部分需要暴露给外部（下一层、下一个时间步、输出）
- $\tanh(C_t)$：将细胞状态压缩到 (-1, 1)，防止数值发散
- $h_t$：对外部可见的"工作记忆"

**注意**：$h_t$ 传给下一时间步的 $h_{t}$，而 $C_t$ 传给下一时间步的 $C_{t}$。两条信息流并行：$C_t$ 作为内部长期状态（通过线性加性路径更新），$h_t$ 作为对外暴露的输出状态（由输出门控制从 $C_t$ 中读取哪些信息）。

### 2.5 公式汇总

为便于对照，将 LSTM 的全部公式集中列出。给定当前输入 $x_t$、上一时刻隐藏状态 $h_{t-1}$ 和细胞状态 $C_{t-1}$，一个 LSTM 时间步的计算如下：

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
h_t &= o_t \odot \tanh(C_t) \quad &\text{隐藏状态输出：从 } C_t \text{ 中筛选后对外暴露}
\end{aligned}
$$

**参数总览：** LSTM 共有四套权重矩阵和偏置——遗忘门 $(W_f, b_f)$、输入门 $(W_i, b_i)$、输出门 $(W_o, b_o)$ 和候选值 $(W_C, b_C)$。每个 $W$ 的列数为 $d_h + d_x$（$h_{t-1}$ 和 $x_t$ 拼接后的维度），行数为 $d_h$。总参数量（不含偏置）为 $4 \times d_h \times (d_h + d_x)$。

**梯度传递的核心优势：** $C_t$ 对 $C_{t-1}$ 的偏导是对角矩阵 $\text{diag}(f_t)$，梯度在时间轴上每步仅乘以遗忘门的对应元素。当 $f_t[j] \approx 1$ 时，梯度几乎无损地跨越该时间步。这与 Vanilla RNN 中梯度每步被矩阵乘法 $W_{hh}$ 反复压缩和旋转形成了根本区别。

---

## 3. 完整手算例子

![LSTMCalculate.png](https://img.yumeko.site/file/articles/LongShortTermMemory/LSTMCalculate.webp)

使用与 Vanilla RNN 相同简化的设置：input=2, hidden=2, 仅展示遗忘门和细胞状态的完整计算过程。

### 3.1 遗忘门手算

**简化权重**：

$$
W_{fx} = \begin{pmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{pmatrix}
\qquad
W_{fh} = \begin{pmatrix} 0.2 & -0.1 \\ -0.3 & 0.4 \end{pmatrix}
\qquad
b_f = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

**时间步 1**：$h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, $x_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $C_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

$$
W_{fx} \cdot x_1 = \begin{pmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{pmatrix} \cdot \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.3 \\ 0.5 \end{pmatrix}
$$

$$
f_1 = \sigma\left( \begin{pmatrix} 0.3 \\ 0.5 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix} \right)
= \sigma\left( \begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix}
$$

### 3.2 输入门手算

**简化权重**（与遗忘门不同，但计算方式相同）：

设 $W_{ix} \cdot x_1 + W_{ih} \cdot h_0 + b_i = \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix}$：

$$
i_1 = \sigma\left( \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix} \right) \approx \begin{pmatrix} 0.69 \\ 0.57 \end{pmatrix}
$$

设候选值 $W_{Cx} \cdot x_1 + W_{Ch} \cdot h_0 + b_C = \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix}$：

$$
\tilde{C}_1 = \tanh\left( \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix} \right) \approx \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
$$

### 3.3 细胞状态更新

$$
C_1 = f_1 \odot C_0 + i_1 \odot \tilde{C}_1
= \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix} \odot \begin{pmatrix} 0 \\ 0 \end{pmatrix}
+ \begin{pmatrix} 0.69 \\ 0.57 \end{pmatrix} \odot \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0.69 \times 0.537 \\ 0.57 \times 0.834 \end{pmatrix}
= \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}
$$

注意：因为 $C_0 = 0$（初始状态），所以 $C_1$ 完全由新输入的信息构成。

**时间步 2**：$x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$，$C_1 = \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}$

设遗忘门 $f_2 \approx \begin{pmatrix} 0.45 \\ 0.55 \end{pmatrix}$，输入门 $i_2 \approx \begin{pmatrix} 0.30 \\ 0.80 \end{pmatrix}$，候选 $\tilde{C}_2 \approx \begin{pmatrix} -0.40 \\ 0.60 \end{pmatrix}$：

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

**关键观察**：
- $C_2[0] = 0.167 - 0.120 = 0.047$——大部分旧信息 ($C_1[0] = 0.371$) 被遗忘门（0.45）丢弃，新写入的信息也很少
- $C_2[1] = 0.261 + 0.480 = 0.741$——旧信息保留了一半多（0.55 × 0.475），加上大量新信息

两个维度展示了不同的信息动态：维度 0 "重置"了记忆，维度 1 "累积"了信息。

### 3.4 输出门手算

设 $o_2 = \sigma(\dots) \approx \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix}$：

$$
h_2 = o_2 \odot \tanh(C_2)
= \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \tanh\left( \begin{pmatrix} 0.047 \\ 0.741 \end{pmatrix} \right)
\approx \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \begin{pmatrix} 0.047 \\ 0.629 \end{pmatrix}
= \begin{pmatrix} 0.042 \\ 0.189 \end{pmatrix}
$$

输出门决定了哪些信息被"暴露"给外部：维度 0 的细胞状态很小（0.047）但几乎全部输出（0.90）；维度 1 的细胞状态很大（0.741）但只输出了 30%。

---

## 4. LSTM 参数量和形状追踪

### 4.1 四套权重矩阵

LSTM 有 3 个门 + 1 个候选值生成，共四套参数。PyTorch 将它们打包为两个大矩阵：

| PyTorch 参数名 | 实际形状 | 包含内容 |
|------|------|------|
| `weight_ih` | (4×hidden, input) | $W_{ix}, W_{fx}, W_{Cx}, W_{ox}$ 纵向拼接 |
| `weight_hh` | (4×hidden, hidden) | $W_{ih}, W_{fh}, W_{Ch}, W_{oh}$ 纵向拼接 |
| `bias_ih` | (4×hidden,) | $b_i, b_f, b_C, b_o$ 纵向拼接 |
| `bias_hh` | (4×hidden,) | 另一组偏置（PyTorch 特有） |

### 4.2 参数量计算

以单个 LSTM 层（input=256, hidden=256）为例：

| 组件 | 计算 | 参数量 |
|------|------|--------|
| weight_ih | 4 × 256 × 256 | 262,144 |
| weight_hh | 4 × 256 × 256 | 262,144 |
| bias_ih | 4 × 256 | 1,024 |
| bias_hh | 4 × 256 | 1,024 |
| **单层合计** | | **526,336** |

### 4.3 双向 LSTM 的参数量

双向 LSTM 有两套独立的权重（正向+反向），但共享输入。单层双向的参数约是单向的 2 倍：~1,052,672。

---

## 5. PyTorch 代码详解

### 5.1 基本用法

```python
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
B, L = x.size(0), x.size(1)
h0 = torch.zeros(2 * 2, B, 256)  # 2层×2方向, B, 256
c0 = torch.zeros(2 * 2, B, 256)

output, (hn, cn) = lstm(x, (h0, c0))
# output: (B, L, 512)  — 拼接了正反向
# hn: (4, B, 256)      — 所有层的最终隐藏状态
# cn: (4, B, 256)      — 所有层的最终细胞状态
```

### 5.2 输出形状详解

`nn.LSTM` 返回的 `output` 和 `(hn, cn)` 各有不同的含义：

```python
# x: (B, L, input_size)  — B=batch, L=seq_len
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

### 5.3 逐时间步的循环（手动模式）

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

---

## 6. LSTM 的变体

### 6.1 Peephole Connections

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

### 6.2 Coupled Forget-Input Gate

另一种变体将遗忘门和输入门耦合：$i_t = 1 - f_t$。这意味着"写入的新信息量 = 丢弃的旧信息量"，细胞状态的总信息量保持恒定。GRU 实际上就采用了这种思路（通过更新门 $z_t$）。

---

## 7. 总结

| 概念 | 公式/描述 | 作用 |
|------|------|------|
| 细胞状态 $C_t$ | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ | 长期记忆，梯度可无损通过 |
| 隐藏状态 $h_t$ | $h_t = o_t \odot \tanh(C_t)$ | 短期工作记忆，对外暴露 |
| 遗忘门 $f_t$ | $\sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | 控制丢弃多少旧记忆 |
| 输入门 $i_t$ | $\sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | 控制写入多少新信息 |
| 输出门 $o_t$ | $\sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | 控制对外暴露多少记忆 |
| CEC | $C_{t-1} \to C_t$ 的线性路径 | 梯度通过逐元素独立控制的线性路径稳定传递 |

LSTM 设计的核心在于将状态存储和输出解耦——$C_t$ 作为受门控保护的内部状态（遗忘门和输入门控制其读写的规模和时机），$h_t$ 作为有选择性的输出暴露（输出门控制对外可见哪些内部信息）。这种解耦是 LSTM 能够学习长距离依赖的根本原因。

> 对比 GRU 参见 [[NeuralNetwork/RNN/Foundations/GatedRecurrentUnit|GRU 详解]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
