---
title: LSTM 里程碑：1997 年改变序列学习的发明
date: 2026-05-24
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: Hochreiter 和 Schmidhuber 在 1997 年提出 LSTM，通过门控机制和细胞状态解决了 Vanilla RNN 的梯度消失问题。完整推导其数学原理、手算前向传播、从零 PyTorch 实现及梯度分析。
image: https://img.yumeko.site/file/blog/LSTMMilestone.webp
status: published
---

## 1. 诞生背景：梯度消失的理论分析

### 1.1 Hochreiter 的毕业论文 (1991)

Sepp Hochreiter 在 1991 年的毕业论文（导师：Jürgen Schmidhuber）中对 RNN 的梯度消失问题做了严格的数学分析——这是该问题首次被系统性研究。

核心发现：在 Vanilla RNN 中，反向传播的误差信号随时间指数衰减：

$$
\left|\left| \frac{\partial L_T}{\partial h_1} \right|\right| \propto \prod_{t=2}^{T} \lambda_t
$$

其中 $\lambda_t$ 是 $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}$ 的特征值。当 $|\lambda_t| < 1$ 时（在 Vanilla RNN 中几乎总是如此），连乘导致指数衰减。这个分析为 LSTM 的设计提供了明确的方向：**创造一个 $\lambda_t \approx 1$ 的梯度路径**。

![RNNVSLSTM.png](https://img.yumeko.site/file/articles/LongShortTermMemory/RNNVSLSTM.webp)

### 1.2 1990 年代中期的困境

到 1990 年代中期，RNN 研究陷入困境：理论分析已证明梯度消失的不可避免性，Elman RNN 的实践经验证实了 10-20 步的实用上限，但尚无有效的解决方案。

---

## 2. CEC：LSTM 的核心洞察与完整公式推导

### 2.1 常量误差传送带（CEC）

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

### 2.2 原始 LSTM（1997 年版）：两个门
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

### 2.3 遗忘门的加入（Gers, 2000）

2000 年，Felix Gers（Schmidhuber 的学生）在博士论文中提出给 LSTM 加上遗忘门。这个看似微小的改动使得 LSTM 能够主动选择遗忘旧信息，对于处理无限长流式数据至关重要。


**遗忘门 $f_t$**：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**加入遗忘门后的细胞状态更新**（现代 LSTM 的标准形式）：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

### 2.4 现代 LSTM 完整公式汇总

以下是当前业界通用的 LSTM（含遗忘门）在一个时间步内的全部计算：

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

**参数规模**：LSTM 有四套独立的权重矩阵——$(W_f, b_f)$、$(W_i, b_i)$、$(W_o, b_o)$、$(W_C, b_C)$。每个 $W$ 拼接 $h_{t-1}$ 和 $x_t$ 后做线性变换，列数为 $d_h + d_x$，行数为 $d_h$。总参数量为 $4 \times d_h \times (d_h + d_x) + 4 \times d_h$。

**两条信息流的角色分工**：

- $C_t$（细胞状态）：内部长期记忆。更新路径 $C_{t-1} \to C_t$ 是逐元素的线性加和，梯度通过 $\text{diag}(f_t)$ 传递——无矩阵乘法，无非线性压缩。
- $h_t$（隐藏状态）：对外暴露的工作记忆。从 $C_t$ 经 $\tanh$ 压缩和 $o_t$ 筛选后产生，传给下一时间步的 Hidden Layer 和输出层。

### 2.5 遗忘门偏置的初始化策略

Gers 等人发现，遗忘门的偏置 $b_f$ 不应初始化为 0——这会让 $f_t$ 的初始值约 $\sigma(0) = 0.5$，导致训练早期就无差别地遗忘一半信息。更好的做法是将 $b_f$ 初始化为一个较大的正值（如 1.0），此时 $f_t \approx \sigma(1) \approx 0.73$，倾向于保留信息。PyTorch 的 `nn.LSTM` 默认已在内部做了类似优化。

---

## 3. LSTM 前向传播的逐步流程

以 $d_h=2$、$d_x=2$、$T=2$ 为例，详细追踪 LSTM 在每个时间步的内部计算。

### 3.1 初始状态与权重

![LSTMCalculate.png](https://img.yumeko.site/file/articles/LongShortTermMemory/LSTMCalculate.webp)

初始状态 $h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$，$C_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$。

**时间步 $t=1$**：$x_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$。由于 $h_0 = 0$ 且 $C_0 = 0$，第一步完全由当前输入驱动。

**遗忘门**：

设 $W_f \cdot [h_0, x_1] + b_f = \begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix}$：

$$
f_1 = \sigma\left(\begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix}\right) \approx \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix}
$$

$f_1 = [0.60, 0.60]$ 意味着保留 60% 的 $C_0$——但 $C_0 = 0$，所以遗忘门的输出此时无实际影响。

**输入门**：

设 $W_i \cdot [h_0, x_1] + b_i = \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix}$：

$$
i_1 = \sigma\left(\begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix}\right) \approx \begin{pmatrix} 0.69 \\ 0.57 \end{pmatrix}
$$

$i_1 = [0.69, 0.57]$ 意味着两个维度都相当程度地"打开"以写入新信息。

**候选细胞状态**：

设 $W_C \cdot [h_0, x_1] + b_C = \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix}$：

$$
\tilde{C}_1 = \tanh\left(\begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix}\right) \approx \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
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
o_1 = \sigma\left(\begin{pmatrix} 0.7 \\ 1.1 \end{pmatrix}\right) \approx \begin{pmatrix} 0.67 \\ 0.75 \end{pmatrix}
$$

$$
h_1 = o_1 \odot \tanh(C_1) \approx \begin{pmatrix} 0.67 \\ 0.75 \end{pmatrix} \odot \begin{pmatrix} 0.354 \\ 0.444 \end{pmatrix}
= \begin{pmatrix} 0.237 \\ 0.333 \end{pmatrix}
$$

$h_1$ 是经过输出门筛选的 $C_1$ 的对外暴露版本。$h_1 \neq C_1$——输出门在两个维度上过滤掉了约 25%-33% 的信息。

**时间步 $t=2$**：$x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$，$h_1 = \begin{pmatrix} 0.237 \\ 0.333 \end{pmatrix}$，$C_1 = \begin{pmatrix} 0.371 \\ 0.475 \end{pmatrix}$。

**遗忘门**（首次真正发挥作用）：

设 $f_2$ 的计算结果：

$$
f_2 \approx \begin{pmatrix} 0.45 \\ 0.55 \end{pmatrix}
$$

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
h_2 = \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \tanh\left(\begin{pmatrix} 0.047 \\ 0.741 \end{pmatrix}\right)
\approx \begin{pmatrix} 0.90 \\ 0.30 \end{pmatrix} \odot \begin{pmatrix} 0.047 \\ 0.629 \end{pmatrix}
= \begin{pmatrix} 0.042 \\ 0.189 \end{pmatrix}
$$

维度 0 的 $C_2[0] = 0.047$ 很小，但输出门几乎完全打开（$0.90$），$h_2[0] = 0.042$。维度 1 的 $C_2[1] = 0.741$ 很大，但输出门只打开了 30%，$h_2[1] = 0.189$。输出门使得 $h_t$ 不等于 $C_t$——模型可以选择性地对外"保密"某些内部记忆。

---

## 4. 从零实现现代 LSTM

### 4.1 LSTM Cell：公式到代码的精确映射

以下代码实现了 2.4 节的 6 个公式。将 $W \cdot [h_{t-1}, x_t]$ 拆分为分别对 $x_t$ 和 $h_{t-1}$ 做线性变换，避免每次拼接。

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

### 4.2 手动展开多步序列

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

### 4.3 LSTM 的梯度传递分析

LSTM 解决梯度消失的核心在于细胞状态更新公式 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$。对 $C_{t-1}$ 求偏导：

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

这是一个**对角矩阵**——每个元素独立地由遗忘门控制，没有矩阵乘法（$W_{hh}$），没有非线性（$\tanh$）。对比 Vanilla RNN：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}
$$

LSTM 消除了两个梯度衰减源：$W_{hh}$ 的矩阵乘法（旋转和缩放）和 $\tanh'$ 的压缩（最大 1.0，通常远小于 1）。当 $f_t[j] \approx 1$ 时，该维度的梯度几乎无损通过。即使 $f_t[j] = 0.95$，100 步后 $0.95^{100} \approx 0.006$——仍在 float32 的有效范围内，而 Vanilla RNN 在相同条件下梯度约为 $10^{-61}$。

---

## 5. 遗忘门的初始化策略

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

## 6. 沉寂与复兴（1997-2015）

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

## 7. 历史定位与影响

即使在 Transformer 主导的 2020 年代，LSTM 的设计思想仍有深远影响：

1. **门控机制**成为神经网络设计的核心原语——门控是"软性条件分支"，允许网络基于数据动态决定信息流
2. **CEC 的设计思想**影响了后来的 Highway Networks 和 ResNet——跳跃连接本质上也是为梯度提供一条无损传递的路径
3. **$C_t$ / $h_t$ 分离**启发了 Neural Turing Machine 和 Memory Networks 等可微分外部记忆架构
4. **Attention 最初就是为了增强 LSTM Seq2Seq 而设计的**——可以说 LSTM 直接催生了 Transformer

Transformer 在长序列和并行化方面有明显优势，但 LSTM 在小数据、流式处理、推理效率方面仍有竞争力。2020 年代，Mamba、RWKV 等线性 RNN 重新发现了 LSTM 的核心价值——推理时比 Transformer 更高效的序列建模。

LSTM 的核心理念——"为梯度提供一条无损传递的路径"——超越了具体架构，成为深度学习设计的普适原则。

---

> LSTM 的详细数学解析：[[NeuralNetwork/RNN/Foundations/LongShortTermMemory|LSTM 详解]]
> 架构对比：[[NeuralNetwork/RNN/Architectures/ArchitectureComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
