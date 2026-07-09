---
title: 注意力机制详解：从信息瓶颈到 Bahdanau 注意力
date: 2026-05-26
category: 神经网络/RNN
tags:
  - 深度学习
  - 基础
  - 注意力
  - 经典架构
description: 从信息瓶颈问题出发，深入拆解 Bahdanau 加性注意力的评分计算、Softmax 归一化、加权求和的完整数学流程，详细展示 Attention 在 Seq2Seq 中的集成与从零 PyTorch 实现，并探讨注意力权重的可解释性价值。
image: https://img.yumeko.site/file/blog/cover/1780581693707.webp
status: published
---

## 1. 信息瓶颈问题（为什么需要注意力）

### 1.1 问题描述

在标准的 Seq2Seq 或序列分类 RNN 中，整个输入序列 $[x_1, x_2, \dots, x_T]$ 经过双向 RNN 处理后，所有时间步的信息最终被压缩到一个固定维度的向量 $h_{\text{final}} \in \mathbb{R}^{d_h}$（如 $d_h = 512$）中，然后送入分类器或解码器。

**无论输入有多长（10 词或 500 词），最终表示都是固定 $d_h$ 维。** 固定维度的向量无法完整编码长序列的全部信息，信息丢失是结构性的。这就是**信息瓶颈（Information Bottleneck）**。

![InformationBottleneck.png](https://img.yumeko.site/file/blog/articles/1780581414919.webp)

### 1.2 信息瓶颈的具体表现

以 Seq2Seq 机器翻译为例，"编码器的最后一个隐藏状态 -> 解码器"是唯一的信息传递通道：

- 短句（5-10 词）：$d_h$ 维向量足够编码
- 中句（20-30 词）：开始丢失细节
- 长句（50+ 词）：缺失关键信息，翻译质量急剧下降

2014 年 Bahdanau 等人的实验表明，标准 Seq2Seq 在超过 30 词的句子上性能明显退化。实际上，标准 Seq2Seq 对 30 词以上的句子几乎无法产生合理的翻译。

### 1.3 Attention 的核心洞察

与其把所有信息压缩到一个向量，不如让解码器/分类器直接访问编码器的**所有**输出 $\{h_1, h_2, \dots, h_T\}$，并通过可学习的注意力权重动态确定各位置的重要性。

> **与其把所有信息压缩到一个向量，为什么不保留编码器的所有隐藏状态，让解码器在每一步动态分配对不同位置的计算权重？**

这就是 Attention——一种**动态加权机制**，让模型能够为输入序列的每个位置分配一个重要性权重。它让解码器不再只依赖一个固定的上下文向量，而是动态地从编码器的所有输出中提取相关信息。

---

## 2. 注意力机制原理：Bahdanau 加性注意力

### 2.1 四步流程总览

给定编码器所有时刻的隐藏状态 $\{h_1, h_2, \dots, h_T\}$（每个 $h_i \in \mathbb{R}^{d_h}$），注意力机制通过以下四步计算上下文向量 $c$：

1. **评分（Scoring）**：对每个 $h_i$ 计算标量得分 $\text{score}_i$
2. **遮罩（Masking）**：将 PAD 位置的得分压制为负无穷
3. **Softmax 归一化**：将遮罩后的得分转化为概率分布 $\alpha_i$
4. **加权求和**：以 $\alpha_i$ 为权重对 $h_i$ 加权求和，得到上下文向量 $c$

![Attention.png](https://img.yumeko.site/file/blog/articles/1780581379082.webp)

下面逐步拆解。

### 2.2 第 1 步：评分（Scoring）——分类场景的自包含评分

在序列分类场景中（如情感分类），没有解码器的查询状态，每个 $h_i$ 通过一个**得分网络**独立计算"相关程度"：

![ScoreNetwork.png](https://img.yumeko.site/file/blog/articles/1780581448523.webp)

$$
\text{score}(h_i) = v^T \cdot \tanh(W \cdot h_i + b)
$$

- $W \in \mathbb{R}^{d_a \times d_h}$ — 将 $d_h$ 维隐藏状态映射到 $d_a$ 维注意力空间（如 $d_a = 256$，$d_h = 512$）
- $b \in \mathbb{R}^{d_a}$ — 偏置向量
- $v \in \mathbb{R}^{d_a}$ — 可学习的得分向量，将 $d_a$ 维中间表示压缩为一个标量
- $\text{score}(h_i) \in \mathbb{R}$ — 标量，正值表示"相关"，负值表示"不相关"

**为什么叫"加性"注意力？** 因为得分函数 $v^T \cdot \tanh(W \cdot h_i + b)$ 中没有 $h_i$ 和其他向量的乘法交互——得分完全由 $h_i$ 自身通过一个前馈网络产生（加性变换）。

### 2.3 第 1 步（续）：评分——Seq2Seq 场景的查询依赖评分

在 Seq2Seq 场景中，解码器在每一步 $t$ 需要关注输入序列的不同部分。此时引入解码器状态 $s_{t-1}$ 作为**查询向量（Query）**，将 $s_{t-1}$ 与每个编码器隐藏状态 $h_i^{enc}$ 拼接后计算得分：

![BahdanauAttention.png](https://img.yumeko.site/file/blog/articles/1780581380058.webp)

$$
e_{t,i} = \text{score}(s_{t-1}, h_i^{enc}) = v^T \cdot \tanh(W_a \cdot [s_{t-1}; h_i^{enc}] + b_a)
$$

其中 $[s_{t-1}; h_i^{enc}]$ 表示将两个向量沿最后一个维度拼接。$W_a \in \mathbb{R}^{d_a \times (d_s + d_h)}$ 将拼接向量映射到注意力空间，$v \in \mathbb{R}^{d_a}$ 将中间表示压缩为标量得分。

两种形式的本质相同——都是通过可学习的前馈网络（Linear -> Tanh -> Linear）产生标量得分。区别仅在于分类场景的得分网络"内建"了判断标准，而 Seq2Seq 场景将解码器状态作为外部查询信号。

### 2.4 第 2 步：遮罩（Masking）

PAD（填充）位置不包含有效信息，不应参与注意力计算。在 Softmax 之前，将 PAD 位置的得分强制设为 $-\infty$（实际实现中用一个很大的负数如 $-10^9$）：

$$
\text{score}_{\text{masked}}(h_i) = \begin{cases}
v^T \cdot \tanh(W \cdot h_i + b) & \text{if } i \text{ 不是 PAD} \\
-10^9 & \text{if } i \text{ 是 PAD}
\end{cases}
$$

因为 $\exp(-10^9) \approx 0$，在 Softmax 之后，PAD 位置的权重几乎严格为 0。

### 2.5 第 3 步：Softmax 归一化

将得分转化为概率分布（所有权重之和 = 1）：

$$
\alpha_i = \frac{e^{\text{score}_{\text{masked}}(h_i)}}{\sum_{j=1}^{T} e^{\text{score}_{\text{masked}}(h_j)}}
$$

在 Seq2Seq 场景中，$\alpha_{t,i}$ 表示解码器在生成第 $t$ 个词时对输入第 $i$ 个词的关注程度：

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
$$

- $\alpha_i \in (0, 1)$
- $\sum_{i=1}^{T} \alpha_i = 1$
- PAD 位置的 $\alpha_i \approx 0$

### 2.6 第 4 步：加权求和

用注意力权重对所有隐藏状态做加权平均：

$$
c = \sum_{i=1}^{T} \alpha_i \cdot h_i
$$

在 Seq2Seq 场景中，每个解码步 $t$ 都有一个**不同的**上下文向量：

$$
c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i^{enc}
$$

$c \in \mathbb{R}^{d_h}$ 与每个 $h_i$ 同维度（如 512），但它是一个**动态的**表示——不同于只使用最后一个隐藏状态的固定表示，$c$ 聚合了所有时间步的信息，并且重点关注得分高的位置。$c_t$ 则是从编码器输出中**动态筛选**的信息汇总——在生成 "dog" 时需要关注输入中的 "狗"，在生成 "running" 时需要关注输入中的 "跑"。

---

## 3. 注意力在 Seq2Seq 中的集成

### 3.1 Attention Seq2Seq 的结构

与标准 Seq2Seq 不同，Attention Seq2Seq 的编码器**保留所有时间步的隐藏状态**：

$$
\text{编码器输出} = \{h_1^{enc}, h_2^{enc}, h_3^{enc}, \dots, h_T^{enc}\}
$$

每个 $h_i^{enc}$ 都是第 $i$ 个输入词经过双向 RNN 后的表示，包含了该词的**上下文感知**编码。

![Seq2SeqVSAttention.png](https://img.yumeko.site/file/blog/articles/1780581452646.webp)

在解码器的第 $t$ 步，流程如下：

1. 用上一个解码器状态 $s_{t-1}$ 去"查询"编码器的所有隐藏状态
2. 计算 $s_{t-1}$ 与每个 $h_i^{enc}$ 的**匹配得分** $e_{t,i}$
3. 将得分归一化为**注意力权重** $\alpha_{t,i}$
4. 用 $\alpha_{t,i}$ 对所有 $h_i^{enc}$ 做加权求和，得到**当前步的上下文向量** $c_t$
5. 将 $c_t$ 和上一步的输出 $y_{t-1}$ 一起输入解码器 RNN

### 3.2 生成输出

将上下文向量与上一时刻的输出拼接后送入解码器 RNN：

$$
s_t = \text{RNN}_{\text{dec}}(s_{t-1}, [y_{t-1}; c_t])
$$

$$
p(y_t | y_{<t}, X) = \text{softmax}(W_o \cdot s_t + b_o)
$$

### 3.3 从零实现：公式到代码的精确映射

以下从零实现一个完整的 Bahdanau Attention Seq2Seq 模型，将第 2、3 节的每个公式逐行映射为 PyTorch 代码。

#### 3.3.1 编码器：双向 GRU 获取上下文表示

编码器逐词读取源语言序列，保留所有时间步的隐藏状态。双向 GRU 使每个位置的编码融合了前后文信息。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """双向 GRU 编码器，保留所有时间步的隐藏状态作为 Attention 的输入。"""

    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        # 源语言词嵌入: (vocab_size, embed_dim)
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # 嵌入层后的 dropout，防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 双向 GRU: 每步输出是正向和反向 hidden_dim 的拼接 = hidden_dim * 2
        # batch_first=True 使输入输出形状为 (batch, seq_len, feature)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )

    def forward(self, src):
        """
        Args:
            src: 源语言索引序列, (batch, src_len)
        Returns:
            outputs: 所有时间步的双向编码, (batch, src_len, hidden_dim * 2)
            hidden:  最后一层最终隐藏状态, (num_layers * 2, batch, hidden_dim)
        """
        # (batch, src_len) -> (batch, src_len, embed_dim)
        embedded = self.dropout(self.embedding(src))
        # (batch, src_len, embed_dim) -> outputs (batch, src_len, hidden_dim * 2)
        #                               hidden  (num_layers * 2, batch, hidden_dim)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
```

`outputs` 的每个位置 $i$ 对应 $h_i^{enc} \in \mathbb{R}^{2 \cdot d_h}$——这是 Attention 计算中 Key 和 Value 的来源。`hidden` 用于初始化解码器的初始状态 $s_0$（取编码器最后一层的正向最终状态）。

#### 3.3.2 Bahdanau Attention 模块

这是整个架构的核心，精确实现第 2.3、2.5、2.6 节的三个公式。输入解码器当前状态 $s_{t-1}$ 和编码器全部输出 $\{h_i^{enc}\}$，输出动态上下文向量 $c_t$ 和注意力权重 $\alpha_t$。

```python
class BahdanauAttention(nn.Module):
    """
    Bahdanau 加性注意力。

    计算流程（对应第 2 节的 Step 1-4）:
      1. 将 s_{t-1} 与每个 h_i^{enc} 拼接
      2. 经 W_a 线性变换 + tanh 非线性
      3. 经 v^T 压缩为标量得分 scores
      4. 对 scores 做 mask（PAD 位设为 -1e9）
      5. Softmax 归一化得到注意力权重 alpha_{t,i}
      6. 加权求和: c_t = Sigma alpha_{t,i} . h_i^{enc}
    """

    def __init__(self, hidden_dim, attn_dim):
        """
        Args:
            hidden_dim: 解码器单向隐藏维度 d_s
            attn_dim:   注意力中间表示维度 d_a
        """
        super().__init__()
        # 1 W_a: (d_s + 2*d_h, attn_dim)，将拼接向量映射到注意力空间
        # decoder 的 hidden_dim 拼接 encoder 的 hidden_dim * 2 (双向)
        self.W_a = nn.Linear(hidden_dim + hidden_dim * 2, attn_dim, bias=False)
        # 2 v^T: (attn_dim, 1)，将注意力表示压缩为标量得分
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        """
        Args:
            decoder_hidden:  s_{t-1}, 单向解码器状态, (batch, hidden_dim)
            encoder_outputs: {h_i^{enc}}, 编码器所有时间步输出, (batch, src_len, hidden_dim * 2)
            mask:            PAD mask, 1=有效 0=PAD, (batch, src_len)
        Returns:
            context:      c_t, 动态上下文向量, (batch, hidden_dim * 2)
            attn_weights: alpha_t, 注意力权重, (batch, src_len)
        """
        src_len = encoder_outputs.size(1)  # T

        # Step 1: 评分 —— 对应公式 e_{t,i} = v^T . tanh(W_a . [s_{t-1}; h_i^{enc}])

        # 将 decoder_hidden 沿 src_len 维度复制 T 份
        # (batch, hidden_dim) -> (batch, 1, hidden_dim) -> (batch, src_len, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 拼接 decoder 状态和 encoder 输出: [s_{t-1}; h_i^{enc}]
        # (batch, src_len, hidden_dim) + (batch, src_len, hidden_dim * 2)
        #   -> (batch, src_len, hidden_dim + hidden_dim * 2)
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=-1)

        # W_a . concat + tanh
        # (batch, src_len, hidden_dim + hidden_dim*2) -> (batch, src_len, attn_dim)
        energy = torch.tanh(self.W_a(concat))

        # v^T . energy -> 标量得分
        # (batch, src_len, attn_dim) -> (batch, src_len, 1) -> squeeze -> (batch, src_len)
        scores = self.v(energy).squeeze(-1)

        # Step 2: 遮罩 + Softmax —— 对应公式 alpha_{t,i} = exp(e_{t,i}) / Sigma exp(e_{t,j})

        # PAD 位置得分设为 -1e9，exp(-1e9) ~= 0
        scores = scores.masked_fill(mask == 0, -1e9)
        # (batch, src_len) -> (batch, src_len)，Sigma_i alpha_{t,i} = 1
        attn_weights = F.softmax(scores, dim=-1)

        # Step 3: 加权求和 —— 对应公式 c_t = Sigma alpha_{t,i} . h_i^{enc}

        # bmm: (batch, 1, src_len) x (batch, src_len, hidden_dim * 2)
        #    -> (batch, 1, hidden_dim * 2) -> squeeze -> (batch, hidden_dim * 2)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context, attn_weights
```

#### 3.3.3 解码器：自回归生成 + 注意力上下文

解码器在循环中的每一时间步 $t$，执行嵌入、注意力查询、拼接、GRU 前向、输出预测。Teacher Forcing 用于训练时加速收敛。

```python
class AttentionDecoder(nn.Module):
    """
    带 Bahdanau Attention 的自回归解码器。

    每步流程（对应第 3.1 节 Step 1-5）:
      1. 上一时刻输出 y_{t-1} 经 Embedding 得到稠密向量
      2. 拼接 [y_{t-1}; c_t]，其中 c_t = Attention(s_{t-1}, encoder_outputs)
      3. 单向 GRU 前向: s_t = GRU(s_{t-1}, [y_{t-1}; c_t])
      4. Linear 投影得到 p(y_t | y_{<t}, X)
    """

    def __init__(self, output_dim, embed_dim, hidden_dim,
                 attn_dim, num_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        # 目标语言词嵌入: (output_dim, embed_dim)
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Bahdanau 加性注意力模块
        self.attention = BahdanauAttention(hidden_dim, attn_dim)
        # 单向 GRU: 输入 = embed_dim(当前词嵌入) + hidden_dim*2(上下文向量)
        self.rnn = nn.GRU(
            embed_dim + hidden_dim * 2, hidden_dim,
            num_layers, dropout=dropout, batch_first=True
        )
        # 输出投影: hidden_dim -> output_dim (目标词表大小)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, encoder_outputs, hidden, mask,
                teacher_forcing_ratio=0.5):
        """
        Args:
            trg:  目标语言索引序列, (batch, trg_len)
            encoder_outputs: 编码器所有时间步输出, (batch, src_len, hidden_dim * 2)
            hidden: 解码器初始隐藏状态(来自编码器), (num_layers, batch, hidden_dim)
            mask:  PAD mask, (batch, src_len)
            teacher_forcing_ratio: Teacher Forcing 概率
        Returns:
            outputs: 所有时间步的预测, (batch, trg_len, output_dim)
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)

        # 第一个输入是 <SOS> token (index = trg[:, 0])
        # (batch,) —— 每个样本一个整数索引
        decoder_input = trg[:, 0]

        # 预分配输出张量，第 0 列保留为 <SOS> 的零填充
        # (batch, trg_len, output_dim)
        outputs = torch.zeros(
            batch_size, trg_len, self.output_dim, device=trg.device
        )

        # 逐时间步自回归生成: t = 1, 2, ..., trg_len-1
        for t in range(1, trg_len):
            # --- Step 1: 嵌入上一时刻的输出 y_{t-1} ---
            # (batch,) -> (batch, embed_dim) -> unsqueeze -> (batch, 1, embed_dim)
            embedded = self.dropout(self.embedding(decoder_input))
            embedded = embedded.unsqueeze(1)

            # --- Step 2: 获取当前解码器状态 s_{t-1} ---
            # hidden 形状 (num_layers, batch, hidden_dim)
            # hidden[-1] 取最后一层, (batch, hidden_dim)
            decoder_hidden = hidden[-1]

            # --- Step 3: 计算动态上下文向量 c_t ---
            # context:    (batch, hidden_dim * 2)
            # attn_weights: (batch, src_len) —— 当前步对各源语言位置的关注度
            context, attn_weights = self.attention(
                decoder_hidden, encoder_outputs, mask
            )

            # --- Step 4: 拼接 [y_{t-1}; c_t] 并送入 GRU ---
            # (batch, 1, embed_dim) + (batch, 1, hidden_dim * 2)
            #   -> (batch, 1, embed_dim + hidden_dim * 2)
            rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=-1)

            # s_t = GRU(s_{t-1}, [y_{t-1}; c_t])
            # rnn_output: (batch, 1, hidden_dim)
            # hidden:     (num_layers, batch, hidden_dim)
            rnn_output, hidden = self.rnn(rnn_input, hidden)

            # --- Step 5: 输出预测 ---
            # (batch, 1, hidden_dim) -> squeeze -> (batch, hidden_dim)
            # -> fc -> (batch, output_dim)
            prediction = self.fc(rnn_output.squeeze(1))
            outputs[:, t, :] = prediction

            # --- Step 6: 决定下一步的输入 (Teacher Forcing) ---
            # teacher_forcing_ratio=0 时完全自回归（推理模式）
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # 取当前预测中概率最大的词索引, (batch,)
            top1 = prediction.argmax(-1)
            # 训练: 以概率 teacher_forcing_ratio 使用真实目标词
            # 推理: 始终使用模型预测（teacher_forcing_ratio=0）
            decoder_input = trg[:, t] if teacher_force else top1

        return outputs
```

**流程对应对照表：**

| 公式 | 代码行 | 形状变化 |
|------|------|------|
| $y_{t-1}$ 的嵌入 | `self.embedding(decoder_input)` | `(B,)` 到 `(B, embed_dim)` |
| 获取 $s_{t-1}$ | `hidden[-1]` | `(num_layers, B, hidden_dim)` 到 `(B, hidden_dim)` |
| $c_t = \text{Attention}(s_{t-1}, \{h_i^{enc}\})$ | `self.attention(...)` | 详见 3.3.2 节 |
| $[y_{t-1}; c_t]$ | `torch.cat((embedded, context.unsqueeze(1)), dim=-1)` | `(B, 1, embed_dim + 2*hidden_dim)` |
| $s_t = \text{GRU}(s_{t-1}, [y_{t-1}; c_t])$ | `self.rnn(rnn_input, hidden)` | `(B, 1, hidden_dim)` |
| $p(y_t) = \text{softmax}(W_o . s_t + b_o)$ | `self.fc(rnn_output.squeeze(1))` | `(B, hidden_dim)` 到 `(B, output_dim)` |
| 下一步输入决策 | `decoder_input = trg[:, t] if teacher_force else top1` | `(B,)` |

#### 3.3.4 完整 Seq2Seq 模型

将编码器和解码器组合为端到端的 Attention Seq2Seq。整个前向流程依次为：源语言编码、目标语言逐词自回归解码。

```python
class AttentionSeq2Seq(nn.Module):
    """Bahdanau Attention Seq2Seq 完整模型。

    前向流程:
      src -> Encoder -> encoder_outputs, hidden
      trg -> AttentionDecoder(encoder_outputs, hidden, mask) -> outputs
    """

    def __init__(self, src_vocab_size, trg_vocab_size,
                 embed_dim, hidden_dim, attn_dim, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, embed_dim, hidden_dim, num_layers, dropout
        )
        self.decoder = AttentionDecoder(
            trg_vocab_size, embed_dim, hidden_dim,
            attn_dim, num_layers, dropout
        )

    def forward(self, src, trg, src_mask, teacher_forcing_ratio=0.5):
        """
        Args:
            src:      源语言索引, (batch, src_len)
            trg:      目标语言索引, (batch, trg_len)
            src_mask: 源语言 PAD mask, (batch, src_len)
            teacher_forcing_ratio: Teacher Forcing 概率
        Returns:
            outputs: 目标语言每步预测, (batch, trg_len, trg_vocab_size)
        """
        # Step 1: 编码 —— 源语言序列 -> 双向上下文表示
        # encoder_outputs: (batch, src_len, hidden_dim * 2)
        # hidden:          (num_layers * 2, batch, hidden_dim)
        encoder_outputs, hidden = self.encoder(src)

        # Step 2: 解码 —— 自回归生成，每步动态查询 Attention
        # outputs: (batch, trg_len, trg_vocab_size)
        outputs = self.decoder(
            trg, encoder_outputs, hidden, src_mask, teacher_forcing_ratio
        )
        return outputs
```

#### 3.3.5 维度变化追踪

以具体参数为例追踪一次前向传播中各张量的形状变化：

- `src_vocab_size = 10000`，`trg_vocab_size = 8000`
- `embed_dim = 256`，`hidden_dim = 512`，`attn_dim = 256`
- `batch = 32`，`src_len = 20`，`trg_len = 18`

**编码器阶段：**

- `src`: `(32, 20)`，整数索引
- `embedded`: `(32, 20, 256)`，词嵌入后
- `encoder_outputs`: `(32, 20, 1024)`，双向 GRU 输出（512 x 2）
- `hidden`: `(4, 32, 512)`，num_layers x 2 个方向的最终状态

**解码器第 $t$ 步的 Attention 内部：**

- `decoder_hidden`（$s_{t-1}$）：`(32, 512)`
- `concat`（$[s_{t-1}; h_i^{enc}]$）：`(32, 20, 1536)`，即 512 + 1024
- `energy`（$\tanh(W_a . \text{concat})$）：`(32, 20, 256)`
- `scores`（$v^T . \text{energy}$）：`(32, 20)`
- `attn_weights`（Softmax）：`(32, 20)`
- `context`（$\sum \alpha_i . h_i^{enc}$）：`(32, 1024)`

**解码器第 $t$ 步的 GRU 和输出：**

- `embedded`（$y_{t-1}$ 嵌入）：`(32, 1, 256)`
- `rnn_input`（$[y_{t-1}; c_t]$）：`(32, 1, 1280)`，即 256 + 1024
- `rnn_output`（$s_t$）：`(32, 1, 512)`
- `prediction`（$W_o . s_t + b_o$）：`(32, 8000)`

### 3.4 分类场景的 Attention 实现

第 2.2 节介绍了分类场景中不依赖查询向量的自包含评分。以下是在情感分类任务（EmotionClassification）中的具体实现。

#### 3.4.1 Attention 类结构

```python
"""
Attention 注意力机制模块

实现 Bahdanau 加性注意力（Additive Attention），
对 RNN 输出的所有时刻隐藏状态进行加权求和，生成上下文向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau 加性注意力层

    通过一个可学习的得分网络（Linear -> Tanh -> Linear）为每个时刻
    的隐藏状态打分，经过 softmax 归一化后对隐藏状态加权求和。
    PAD 位置的得分被设为极小值 (-1e9)，使其权重趋近于 0。

    Args:
        hidden_size: RNN 输出的隐藏状态维度
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, rnn_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            rnn_output: RNN 所有时刻的输出，形状为 (batch, seq_len, hidden_size)
            mask: 1=有效位置 0=PAD，形状为 (batch, seq_len)

        Returns:
            context_vector: 加权求和后的上下文向量，形状为 (batch, hidden_size)
        """
        # Step 1: 计算每个时刻的注意力得分, 形状 (batch, seq_len)
        scores = self.score_network(rnn_output).squeeze(-1)

        # Step 2: 将 PAD 位置的得分设为极小值
        scores = scores.masked_fill(mask == 0, -1e9)

        # Step 3: softmax 归一化得到注意力权重, 形状 (batch, seq_len)
        attention_weights = F.softmax(scores, dim=1)

        # Step 4: 加权求和, 形状 (batch, hidden_size)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), rnn_output
        ).squeeze(1)

        return context_vector
```

代码中的四步与数学流程严格对应：

- `self.score_network(rnn_output)` 实现了 $v^T \cdot \tanh(W \cdot h_i + b)$——`nn.Sequential` 将 `Linear -> Tanh -> Linear` 串联，等价于论文中的得分网络。
- `masked_fill(mask == 0, -1e9)` 实现了 PAD 位置的遮罩。
- `F.softmax(scores, dim=1)` 沿序列维度做 Softmax 归一化。
- `torch.bmm(attention_weights.unsqueeze(1), rnn_output)` 实现了加权求和 $\sum_i \alpha_i \cdot h_i$——利用批量矩阵乘法一次性完成所有样本的加权求和。

#### 3.4.2 得分网络的维度变化

以 EmotionClassification 的配置（$d_h = 512$，$d_a = 256$）为例。得分网络 `nn.Sequential` 内部包含三层：

- 第一层 `Linear(512, 256)`：$W \in \mathbb{R}^{256 \times 512}$，$b \in \mathbb{R}^{256}$，输出形状从 `(B, L, 512)` 变为 `(B, L, 256)`
- 第二层 `Tanh`：逐元素变换，形状不变 `(B, L, 256)`
- 第三层 `Linear(256, 1, bias=False)`：$v^T$ 的角色，将 256 维压缩为 1 个标量，输出 `(B, L, 1)`，squeeze 后得到 `(B, L)`

得分网络的参数量：

| 层 | 操作 | 参数量 |
|------|------|------:|
| 第一层 | Linear(512->256) | 512x256 + 256 = 131,328 |
| 第三层 | Linear(256->1, bias=False) | 256x1 = 256 |
| **合计** | | **131,584** |

约 13 万参数——相比 LSTM 层的数百万参数，Attention 的开销很小。

#### 3.4.3 Mask 的必要性

没有 Mask 的情况下，PAD 位置的隐藏状态虽然是零（因为 `padding_idx=0` 的嵌入是零向量），但经 RNN 后变为接近零的小值，Softmax 仍会为其分配非零权重——这会稀释有效位置的真实信息。

假设一个样本有 5 个有效词 + 123 个 PAD，若不加 Mask，PAD 位置的权重之和可能高达 $\frac{123}{128} \approx 0.96$，即 96% 的注意力给了无意义的填充位。加了 Mask 后，PAD 位置的 Softmax 输出严格趋近于 0，100% 的注意力集中在有效词上。

### 3.5 Bahdanau vs Luong Attention

| 维度 | Bahdanau（加性） | Luong（乘性） |
|------|------|------|
| 年份 | 2015 | 2015 |
| 得分函数 | $v^T \tanh(W \cdot h_i)$ 或 $v^T \tanh(W_a \cdot [s; h_i])$ | $h_i^T W \cdot h_{\text{query}}$ |
| 计算方式 | 前馈网络独立给每个 $h_i$ 打分（或与查询拼接后打分） | 每个 $h_i$ 与查询向量的点积 |
| 参数量 | 较多（额外的 $W$ 和 $v$） | 较少（只有 $W$，或直接点积无参数） |
| 是否需要 query | 分类场景不需要，Seq2Seq 场景需要 | 始终需要（解码器状态作为 query） |
| EmotionClassification | **使用此方式** | — |

分类场景选择 Bahdanau 自包含评分的原因：没有"解码器的当前查询状态"，使用加性注意力更自然。Seq2Seq 场景中 Bahdanau 通过拼接解码器状态和编码器状态来引入查询信息。

### 3.6 Attention 中的 QKV 范式

Bahdanau Attention 已经隐含了后来 Transformer 中通用的 QKV（Query-Key-Value）范式：

| Bahdanau Attention 概念 | 通用 QKV 概念 | 含义 |
|---|---|---|
| $s_{t-1}$（解码器状态） | Query | "我现在需要什么信息？" |
| $h_i^{enc}$（编码器状态） | Key | "我这段输入包含什么？" |
| $h_i^{enc}$（编码器状态） | Value | "我这段输入的实际内容是什么？" |
| $\alpha_{t,i}$ | Attention Weights | Query 与每个 Key 的匹配度 |
| $c_t$ | Weighted Sum of Values | 按匹配度筛选后的信息汇总 |

在原始的 Bahdanau Attention 中，Key 和 Value 是同一个 $h_i^{enc}$。Transformer 将 Key 和 Value 分开（通过不同的线性投影），提供了更大的灵活性。

---

## 4. 可解释性

### 4.1 情感分类注意力权重热力图

Attention 的一个重要特性是**可解释性**。权重 $\alpha$ 可以直接可视化为热力图，直观展示模型的关注分布。

对于评论"物流很快包装也很好"的注意力权重：

| 词 | 物流 | 很快 | 包装 | 也 | 很好 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 注意力权重 | 0.05 | **0.35** | 0.08 | 0.02 | **0.50** |

约 85% 的注意力权重集中在"很快"（0.35）和"很好"（0.50）两个正面词上——模型确实学会了关注情感相关词汇。

### 4.2 不同样本的注意力模式

训练好的 Attention 会展现出合理的模式：对于正面评论，注意力集中在正面情感词上（如"非常""好""满意"）；对于负面评论，注意力则集中在负面情感词上（如"坏了""太差"）。这种可解释性在实际应用中很有价值——你可以告诉用户"系统判断这条评论是正面/负面的原因是因为它重点关注了这些词"。

### 4.3 注意力对齐矩阵（翻译场景）

在 Seq2Seq 翻译场景中，注意力权重 $\alpha_{t,i}$ 可以可视化为**对齐矩阵**——行对应目标语言位置，列对应源语言位置。

对于 "I love you" 翻译为 "我爱你"：

$$
\begin{array}{c|cccc}
 & \text{I} & \text{love} & \text{you} & \text{〈EOS〉} \\ \hline
\text{我} & 0.80 & 0.10 & 0.05 & 0.05 \\
\text{爱} & 0.05 & 0.85 & 0.05 & 0.05 \\
\text{你} & 0.05 & 0.05 & 0.80 & 0.10
\end{array}
$$

对角线的强权重模式表明模型学到了词与词之间**单调对齐**——中英文词序大致相同时的合理模式。

对于语序不同的语言对（如日语翻译为英语），对齐矩阵会呈现非对角模式：

$$
\begin{array}{c|cccc}
 & \text{私} & \text{は} & \text{学生} & \text{です} \\ \hline
\text{I} & 0.90 & 0.05 & 0.05 & 0.00 \\
\text{am} & 0.05 & 0.80 & 0.05 & 0.10 \\
\text{a} & 0.05 & 0.05 & 0.85 & 0.05 \\
\text{student} & 0.00 & 0.05 & 0.10 & 0.85
\end{array}
$$

注意 "a" 和 "student" 在日语中来自同一个源词 "学生"——模型的注意力权重正确地将源语言的一个词分配给了目标语言的两个位置。这是 Attention 软对齐天然处理的"一对多"场景。

### 4.4 软对齐 vs 硬对齐

传统的统计机器翻译需要显式学习一个"词对齐"表（硬对齐——每个目标词严格对应一个源词）。Attention 实现的**软对齐**有根本性优势：

- **不需要人工标注对齐数据**：模型从平行语料中自动学习对齐关系
- **一对多、多对一自然处理**：一个目标词可以以不同权重关注多个源词，不需要离散的二选一决策
- **梯度可导**：Softmax 加权求和全程可微，对齐关系通过反向传播自动优化

---

## 5. 从 Bahdanau Attention 到 Transformer

### 5.1 对 Attention 的重新审视

Attention RNN 的成功引发了一个深层问题：Attention 本身已经提供了跨位置的直接信息通路，RNN 的循环连接是否仍然必要？

如果 Attention 能让解码器直接访问编码器的任意位置，那么 RNN 循环连接——它的主要作用是沿时间步传递信息——是否可以被纯粹的自注意力替代？

### 5.2 Transformer 的回答

2017 年，Vaswani 等人在《Attention Is All You Need》中给出了答案：**不需要 RNN**。

Transformer 完全抛弃了循环连接，纯粹依靠**自注意力**来建模序列内部的依赖关系。每个位置直接关注所有其他位置，计算复杂度 $O(T^2)$ 但梯度路径长度从 RNN 的 $O(T)$ 缩短为 $O(1)$。这一设计解决了 RNN 的梯度消失和训练并行化等固有难题。

### 5.3 知识迁移路径

理解 Bahdanau Attention 是理解 Transformer 的**必要前提**。以下概念一脉相承：

| Bahdanau Attention (2015) | 对应 Transformer (2017) |
|---|---|
| 解码器状态 Query $s_{t-1}$ | 每个位置的 Query 向量 $Q$ |
| 编码器状态 Key $h_i^{enc}$ | 每个位置的 Key 向量 $K$ |
| 编码器状态 Value $h_i^{enc}$ | 每个位置的 Value 向量 $V$ |
| 加性打分 $v^T \tanh(W[s;h])$ | 缩放点积打分 $\frac{QK^T}{\sqrt{d_k}}$ |
| Softmax + 加权求和 | Softmax + 加权求和（完全相同） |

### 5.4 Attention 的泛化意义

虽然 Attention 最初为 Seq2Seq 设计，但它的核心思想——**对所有位置做动态加权**——适用于任何需要"聚合序列信息"的场景：

- **序列分类**：加权所有时间步的隐藏状态，替代最后一步输出的简单取用
- **阅读理解**：找出与问题最相关的文档片段
- **图像描述**：生成每个词时关注图像的不同区域

---

> 回到主文档：[[NeuralNetwork/Overview/RNNOverview|RNN 详解主文档]]
