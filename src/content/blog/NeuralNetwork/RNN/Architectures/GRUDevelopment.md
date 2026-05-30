---
title: GRU 的诞生：LSTM 的精简进化
date: 2026-05-23
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: 2014 年 Cho 等人提出 GRU 的背景、动机（为 Seq2Seq 机器翻译简化 LSTM），完整数学推导，从零 PyTorch 实现，门控机制对比分析及工程权衡。
image: https://img.yumeko.site/file/blog/GRUDevelopment.webp
status: published
---

## 1. 诞生的工程动机

### 1.1 Seq2Seq 的计算压力

2014 年，Cho 等人在开发 Seq2Seq（序列到序列的神经机器翻译模型）时面临一个实际问题：**Seq2Seq 需要两个 RNN**（一个编码器 + 一个解码器），参数和计算开销都是单个 RNN 的两倍。

如果使用 LSTM，编码器和解码器各有约 3.3 倍 Vanilla RNN 的参数量，总合约 6.6 倍。对于当时刚起步的神经机器翻译研究，这个开销相当大——训练一个 LSTM Seq2Seq 模型在当时的硬件上需要数周时间。

### 1.2 GRU 的设计目标

Cho 团队的目标非常明确——在保留门控核心能力的前提下大幅简化：

- **保留** LSTM 的门控机制（应对梯度消失的核心能力）
- **减少** 参数和计算量（让 Seq2Seq 训练更快、更省资源）
- **简化** 接口（去掉独立的细胞状态，减少实现复杂度）

结果就是 GRU——参数约为 LSTM 的 75%，但保留了门控应对梯度消失的核心能力。

---

## 2. 从 LSTM 到 GRU：三大简化策略的完整推导

### 2.1 回顾 LSTM 的完整公式

在理解 GRU 的简化之前，先回顾 LSTM 在一个时间步内的全部计算。给定 $x_t$、$h_{t-1}$、$C_{t-1}$：

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad &\text{遗忘门} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad &\text{输入门} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad &\text{候选细胞状态} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &\text{细胞状态更新} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad &\text{输出门} \\
h_t &= o_t \odot \tanh(C_t) \quad &\text{隐藏状态输出}
\end{aligned}
$$

LSTM 有 3 个门、2 个状态（$C_t$ 和 $h_t$）、4 套权重矩阵。GRU 的目标是将这一切压缩为更简洁的形式。

### 2.2 策略一：合并遗忘门和输入门为更新门 $z_t$

LSTM 的细胞状态更新公式为：

$$
C_t = \underbrace{f_t \odot C_{t-1}}_{\text{保留旧信息}} + \underbrace{i_t \odot \tilde{C}_t}_{\text{写入新信息}}
$$

这两个操作在数学上是互补的——旧信息被遗忘的空间恰好可供新信息写入。GRU 的洞察是：**用一个门同时控制两者的比例**。引入更新门 $z_t$：

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

状态更新变为线性插值：

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

这里 $(1 - z_t)$ 的功能等价于 LSTM 的遗忘门 $f_t$（控制保留多少旧信息），$z_t$ 等价于输入门 $i_t$（控制写入多少新信息）。两者自动互补：保留量 $+$ 写入量 $= 1$。

**代价**：这种耦合意味着 GRU 无法像 LSTM 那样在某个维度上"既大量保留旧信息又大量写入新信息"——但实践中这种灵活性很少被需要，因为旧状态的每个维度在同一时刻通常只需要执行一种操作（保留或覆盖）。

### 2.3 策略二：移除独立的细胞状态 $C_t$

LSTM 有两个并行状态——$C_t$（内部长期记忆）和 $h_t$（对外输出）。GRU 去掉了 $C_t$，让 $h_t$ 同时承担两者的角色。

在 LSTM 中，$C_t$ 的更新路径提供了梯度无损传递的能力：

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

GRU 的 $h_t$ 更新公式 $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ 同样提供了线性梯度路径：

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t)
$$

当 $z_t \approx 0$ 时，$\partial h_t / \partial h_{t-1} \approx I$（单位矩阵），梯度几乎无损通过。因此 GRU 不需要一个独立的 $C_t$——$h_t$ 本身的更新路径已经具备了梯度无损传递的性质。

同时，移除输出门意味着 $h_t$ 直接对外暴露，不再经过 $o_t \odot \tanh(C_t)$ 的筛选。GRU 放弃了 LSTM 中"控制对外暴露多少内部状态"的精细能力，换取了更简洁的结构。

### 2.4 策略三：添加重置门 $r_t$ 作为补偿

去掉了独立的 $C_t$ 和输出门后，GRU 损失了两种控制能力：(1) 独立于输出的内部记忆，(2) 对外输出的选择性过滤。为补偿这些损失，GRU 引入了 LSTM 中没有的新机制——**重置门 $r_t$**：

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

重置门作用在候选状态的计算中，控制**在生成候选值时参考多少过去状态**：

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

$r_t$ 实现了不同于 LSTM 的"遗忘"机制：

- LSTM：在细胞状态更新时遗忘——$f_t$ 控制 $C_{t-1}$ 的保留比例，影响最终状态 $C_t$ 的内容
- GRU：在候选值计算时忽略——$r_t$ 控制 $h_{t-1}$ 有多少参与候选 $\tilde{h}_t$ 的计算，影响候选本身的内容

当 $r_t[j] \approx 0$ 时，$h_{t-1}[j]$ 完全不参与 $\tilde{h}_t[j]$ 的计算——模型在生成候选时完全忽略该维度的历史信息，仅依赖当前输入 $x_t$。这对于处理句子边界（新句子开始时重置对上一句的记忆）或话题转换（忽略旧话题的语义）特别有用。

### 2.5 GRU 完整公式汇总

![GRU.png](https://img.yumeko.site/file/articles/GRUDevelopment/GRU.webp)

将上述三个策略整合，得到 GRU 在一个时间步内的完整计算：

**两个门控信号：**

$$
\begin{aligned}
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad &\text{重置门：控制候选值中过去信息的参与度} \\
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad &\text{更新门：控制新旧信息的混合比例}
\end{aligned}
$$

**候选隐藏状态与最终输出：**

$$
\begin{aligned}
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b) \quad &\text{候选状态：经重置门筛选后的新信息} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{最终状态：旧记忆与新候选的线性插值}
\end{aligned}
$$

**参数规模对比：**

| | LSTM | GRU | 比例 |
|------|:---:|:---:|:---:|
| 门控数量 | 3（$f_t, i_t, o_t$） | 2（$r_t, z_t$） | 67% |
| 状态数量 | 2（$C_t, h_t$） | 1（$h_t$） | 50% |
| 权重矩阵套数 | 4 | 3 | **75%** |
| 参数量（$d_h=256, d_x=256$） | 526,336 | 394,752 | **75%** |

---

## 3. GRU 前向传播的逐步流程

以序列长度 $T=3$、隐藏维度 $d_h=2$、输入维度 $d_x=2$ 为例，详细追踪 GRU 的逐步前向传播，展示每个门在每个时间步的具体作用。

### 3.1 初始状态与参数

初始隐藏状态 $h_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$。输入序列 $x_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$，$x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$，$x_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$。

**时间步 $t=1$（首步——重置门和更新门初次激活）：**

设 $W_r \cdot [h_0, x_1] + b_r = \begin{pmatrix} 0.3 \\ -0.5 \end{pmatrix}$，则：

$$
r_1 = \sigma\left(\begin{pmatrix} 0.3 \\ -0.5 \end{pmatrix}\right) \approx \begin{pmatrix} 0.57 \\ 0.38 \end{pmatrix}
$$

$r_1 = [0.57, 0.38]$ 意味着：维度 0 保留 57% 的 $h_0$ 信息参与候选计算，维度 1 仅保留 38%（更多重置）。

$$
z_1 = \sigma\left(W_z \cdot [h_0, x_1] + b_z\right) \approx \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix}
$$

$z_1 = [0.60, 0.60]$ 意味着：最终状态中 60% 来自新候选，40% 来自旧状态。

候选状态（$h_0$ 经重置门筛选后与 $x_1$ 拼接）：

$$
\tilde{h}_1 = \tanh\left(W \cdot [r_1 \odot h_0, x_1] + b\right)
$$

由于 $h_0 = 0$，$r_1 \odot h_0 = 0$，候选完全由 $x_1$ 驱动。设计算结果为 $\tilde{h}_1 \approx \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}$。

最终状态——线性插值：

$$
h_1 = (1 - 0.60) \odot \begin{pmatrix} 0 \\ 0 \end{pmatrix} + 0.60 \odot \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
= \begin{pmatrix} 0.322 \\ 0.500 \end{pmatrix}
$$

由于 $h_0 = 0$，$h_1$ 完全由新候选决定（被 $z_1 = 0.60$ 打了六折）。

**时间步 $t=2$（循环连接生效——重置门区分新旧信息）：**

$h_1 = [0.322, 0.500]$，$x_2 = [0, 1]$。重置门判断需要保留多少 $h_1$：

$$
r_2 \approx \begin{pmatrix} 0.45 \\ 0.65 \end{pmatrix}
$$

$$
r_2 \odot h_1 = \begin{pmatrix} 0.45 \times 0.322 \\ 0.65 \times 0.500 \end{pmatrix} = \begin{pmatrix} 0.145 \\ 0.325 \end{pmatrix}
$$

维度 0 被重置门大幅过滤（保留 45%），维度 1 保留较多（65%）。候选状态融合了过滤后的旧记忆和当前输入：

$$
\tilde{h}_2 = \tanh\left(W \cdot [r_2 \odot h_1, x_2] + b\right) \approx \begin{pmatrix} 0.200 \\ 0.450 \end{pmatrix}
$$

更新门决定新旧混合比例：

$$
z_2 \approx \begin{pmatrix} 0.40 \\ 0.80 \end{pmatrix}
$$

$$
h_2 = (1 - z_2) \odot h_1 + z_2 \odot \tilde{h}_2
= \begin{pmatrix} 0.60 \\ 0.20 \end{pmatrix} \odot \begin{pmatrix} 0.322 \\ 0.500 \end{pmatrix}
+ \begin{pmatrix} 0.40 \\ 0.80 \end{pmatrix} \odot \begin{pmatrix} 0.200 \\ 0.450 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.193 \\ 0.100 \end{pmatrix} + \begin{pmatrix} 0.080 \\ 0.360 \end{pmatrix}
= \begin{pmatrix} 0.273 \\ 0.460 \end{pmatrix}
$$

维度 0：$z_2[0] = 0.40$，保留 60% 旧状态（$0.193$），写入 40% 新候选（$0.080$）——偏保守。
维度 1：$z_2[1] = 0.80$，仅保留 20% 旧状态（$0.100$），大量写入新候选（$0.360$）——大幅更新。

GRU 的两个维度展现了不同的更新策略——这正是更新门逐元素独立控制的体现。

**时间步 $t=3$**：$h_2 = [0.273, 0.460]$，$x_3 = [1, 1]$。重置门和更新门继续按同样的机制运作。$h_3$ 将融合经过两个时间步压缩的 $x_1$、一个时间步压缩的 $x_2$ 和直接输入的 $x_3$。

### 3.2 与 LSTM 同输入对比

![LSTMGRUEncode.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/LSTMGRUEncode.webp)

使用相同的输入和初始化，LSTM 在 $t=2$ 时的细胞状态为 $C_2 = \begin{pmatrix} 0.047 \\ 0.741 \end{pmatrix}$，GRU 的 $h_2 = \begin{pmatrix} 0.273 \\ 0.460 \end{pmatrix}$。

| 维度 | LSTM $C_2$ | GRU $h_2$ | 差异原因 |
|------|:---:|:---:|------|
| 0 | 0.047 | 0.273 | GRU 保留更多旧信息（$1-z_2=0.60$，保留 60%） |
| 1 | 0.741 | 0.460 | LSTM 累积更多（遗忘门 0.55 + 输入门 0.80） |

GRU 的输出在两个维度之间更均衡，LSTM 的维度间差异更大——这反映了 LSTM 有更精细的独立门控能力（遗忘和输入分离），而 GRU 的耦合门（$z_t$ 和 $1-z_t$ 绑定）限制了极端非对称更新的自由度。

---

## 4. 从零实现 GRU

### 4.1 GRU Cell：公式到代码的精确映射

以下代码实现了 2.5 节的四个公式。每个 `nn.Linear` 对应公式中的一个权重矩阵。

```python
import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """GRU 单步单元。

    四个公式:
      r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
      z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
      n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
      h_t = (1 - z_t) * h_{t-1} + z_t * n_t

    注意：将 W·[h_{t-1}, x_t] 拆分为对 x_t 和 h_{t-1} 分别做线性变换，
    避免每次调用时拼接向量，提高计算效率。
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 重置门 r_t —— 两套线性变换
        # W_ir: (hidden_dim, input_dim) —— x_t 的输入变换
        self.W_ir = nn.Linear(input_dim, hidden_dim, bias=False)
        # W_hr: (hidden_dim, hidden_dim) —— h_{t-1} 的循环变换
        self.W_hr = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 更新门 z_t —— 结构同重置门，独立参数
        self.W_iz = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 候选隐藏状态 n_t (= \tilde{h}_t)
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t:    当前输入, (batch, input_dim)
            h_prev: 上一时刻隐藏状态, (batch, hidden_dim)
        Returns:
            h_t: 新的隐藏状态, (batch, hidden_dim)
        """
        # 公式 1: 重置门
        # r_t = sigmoid(W_ir(x_t) + W_hr(h_{t-1}))
        # W_ir(x_t):    (batch, input_dim)  -> (batch, hidden_dim)
        # W_hr(h_prev): (batch, hidden_dim) -> (batch, hidden_dim)
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev))

        # 公式 2: 更新门
        # z_t = sigmoid(W_iz(x_t) + W_hz(h_{t-1}))
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev))

        # 公式 3: 候选隐藏状态
        # n_t = tanh(W_in(x_t) + r_t * W_hn(h_{t-1}))
        # 注意 r_t 只乘在 h_{t-1} 的变换上，不乘在 x_t 上
        n_t = torch.tanh(self.W_in(x_t) + r_t * self.W_hn(h_prev))

        # 公式 4: 最终隐藏状态（线性插值）
        # h_t = (1 - z_t) * h_{t-1} + z_t * n_t
        h_t = (1 - z_t) * h_prev + z_t * n_t

        return h_t
```

**关于 bias 设计**：$x_t$ 的变换（`W_ir`, `W_iz`, `W_in`）设置了 `bias=False`，而 $h_{t-1}$ 的变换（`W_hr`, `W_hz`, `W_hn`）设置了 `bias=True`。每个门只需要一个偏置——如果 $x_t$ 的变换也带偏置，就会出现两个偏置相加的冗余。PyTorch 官方的 `nn.GRUCell` 使用相同策略。

### 4.2 手动展开多步序列

```python
def gru_forward(cell, input_sequence, h_0=None):
    """
    用 GRUCell 手动展开整个序列。

    Args:
        cell:           GRUCell 实例
        input_sequence: (batch, seq_len, input_dim)
        h_0:            初始隐藏状态, None 则用零向量
    Returns:
        outputs: (batch, seq_len, hidden_dim)
        h_final: (batch, hidden_dim)
    """
    batch_size, seq_len, _ = input_sequence.shape
    hidden_dim = cell.W_hr.out_features

    if h_0 is None:
        h = torch.zeros(batch_size, hidden_dim, device=input_sequence.device)
    else:
        h = h_0

    outputs = []
    for t in range(seq_len):
        x_t = input_sequence[:, t, :]  # (batch, input_dim)
        h = cell(x_t, h)               # 执行一步 GRU
        outputs.append(h)

    outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
    return outputs, h
```

### 4.3 GRU 的梯度传递路径

![GradinLSTM_GRU.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/GradinLSTM_GRU.webp)

GRU 的最终状态更新公式为 $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$。对 $h_{t-1}$ 求偏导（忽略 $\tilde{h}_t$ 对 $h_{t-1}$ 的间接依赖）：

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t)
$$

这是一个对角矩阵，每个元素独立由更新门控制：

- 当 $z_t[j] \approx 0$：$\partial h_t / \partial h_{t-1}[j,j] \approx 1$，梯度几乎无损通过该维度
- 当 $z_t[j] \approx 1$：$\partial h_t / \partial h_{t-1}[j,j] \approx 0$，梯度被完全阻断（该维度完全更新为候选值）
- 中间值：梯度以 $1 - z_t$ 的比例衰减

$T$ 步连乘后，第 $k$ 步的梯度到达第 $T$ 步的衰减因子约为 $\prod_{t=k+1}^{T} (1 - z_t[j])$。只要更新门在某维度上持续输出接近 0 的值，该维度的梯度可以跨越任意长度的时间步——这就是 GRU 缓解梯度消失的核心机制。

对比 LSTM 的 $\partial C_t / \partial C_{t-1} = \text{diag}(f_t)$——两者在梯度传递上的数学结构完全相同（逐元素对角矩阵），只是控制因子不同（GRU 用 $1 - z_t$，LSTM 用 $f_t$）。

---

## 5. 论文中的关键实验

### 5.1 英法翻译任务

GRU 原论文在 WMT'14 英法翻译任务上评估。使用 GRU Seq2Seq 的翻译质量（BLEU 分数）与 LSTM Seq2Seq **基本持平**。

| 模型 | 编码器 | 解码器 | BLEU |
|------|--------|--------|:----:|
| Seq2Seq (Cho 2014) | GRU | GRU | 与 LSTM 相当 |

这表明 GRU 的简化**没有损害模型能力**——至少在此任务上。

### 5.2 后续验证

GRU 论文发表后，社区进行了大量对比实验。总体结论：

- 在大多数 NLP 任务上，GRU 和 LSTM 的差异在统计噪声以内
- 在超长序列（大于 500 步）上，LSTM 的独立 CEC 略有优势——两个状态的分离提供了更稳定的长期记忆
- 在小数据集上，GRU 参数少更不容易过拟合
- 在模型集成中，LSTM 和 GRU 集成在一起效果最好——说明两者学到了不同的表示

---

## 6. 工程权衡总结

![LSTMVSGRU.png](https://img.yumeko.site/file/articles/GRUDevelopment/LSTMVSGRU.webp)

| 当你...... | 选 LSTM | 选 GRU |
|------------|:---:|:---:|
| 有超长序列（大于 500 步） | 是 | — |
| 需要精细的输出控制 | 是（有输出门） | — |
| 数据量很大（大于 100K） | 是 | — |
| 资源或时间受限 | — | 是 |
| 小数据集 | — | 是 |
| 快速实验或原型开发 | — | 是 |
| 不确定 | 均可 | 均可 |

在实践中，很多团队的策略是：**默认用 GRU 快速迭代，最终用 LSTM 或 Transformer 做精调**。

---

> GRU 的详细数学公式：[[NeuralNetwork/RNN/Foundations/GatedRecurrentUnit|GRU 详解]]
> 架构对比：[[NeuralNetwork/RNN/Architectures/ArchitectureComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
