---
title: 注意力 RNN：打破信息瓶颈
date: 2026-05-26
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
  - 注意力
description: 2015 年 Bahdanau 等人将注意力机制引入 Seq2Seq，解决了上下文向量的信息瓶颈。逐公式拆解加性注意力的评分、归一化、加权求和过程，配合完整 PyTorch 从零实现。
image: https://img.yumeko.site/file/blog/AttentionRNN.webp
status: published
---

## 1. Seq2Seq 的结构性缺陷

### 1.1 回顾信息瓶颈

在标准 Seq2Seq 中，整个输入序列的信息被压缩到一个固定长度的上下文向量 $C = h_T^{enc}$。无论输入是 3 个词还是 100 个词，上下文向量都是 512 维（或 1024 维）。这导致了长句翻译质量的灾难性下降——Bahdanau 等人在 2015 年论文中的实验显示，标准 Seq2Seq 对 30 词以上的句子几乎无法产生合理的翻译。

### 1.2 Bahdanau 的关键洞察

> **与其把所有信息压缩到一个向量，为什么不保留编码器的所有隐藏状态，让解码器在每一步动态分配对不同位置的计算权重？**

这个想法就是**注意力机制**。它让解码器不再只依赖一个固定的上下文向量，而是动态地从编码器的所有输出中提取相关信息。

---

## 2. Attention Seq2Seq 的结构

![Seq2SeqVSAttention.png](https://img.yumeko.site/file/articles/AttentionRNN/Seq2SeqVSAttention.webp)

### 2.1 编码器：保留所有隐藏状态

与标准 Seq2Seq 不同，Attention Seq2Seq 的编码器**保留所有时间步的隐藏状态**：

$$
\text{编码器输出} = \{h_1^{enc}, h_2^{enc}, h_3^{enc}, \dots, h_T^{enc}\}
$$

每个 $h_i^{enc}$ 都是第 $i$ 个输入词经过双向 RNN 后的表示，包含了该词的**上下文感知**编码。

### 2.2 解码器：每一步动态计算上下文

在解码器的第 $t$ 步：

1. 用上一个解码器状态 $s_{t-1}$ 去"查询"编码器的所有隐藏状态
2. 计算 $s_{t-1}$ 与每个 $h_i^{enc}$ 的**匹配得分** $e_{t,i}$
3. 将得分归一化为**注意力权重** $\alpha_{t,i}$
4. 用 $\alpha_{t,i}$ 对所有 $h_i^{enc}$ 做加权求和，得到**当前步的上下文向量** $c_t$
5. 将 $c_t$ 和上一步的输出 $y_{t-1}$ 一起输入解码器 RNN

每个解码步都有一个**不同的**上下文向量——在生成 "dog" 时需要关注输入中的 "狗"，在生成 "running" 时需要关注输入中的 "跑"。

---

## 3. Bahdanau Attention 的数学拆解

![BahdanauAttention.png](https://img.yumeko.site/file/articles/AttentionRNN/BahdanauAttention.webp)

### 3.1 评分函数

对每个编码器位置 $i$，用解码器状态 $s_{t-1}$ 与编码器隐藏状态 $h_i^{enc}$ 计算匹配得分：

$$
e_{t,i} = \text{score}(s_{t-1}, h_i^{enc}) = v^T \cdot \tanh(W_a \cdot [s_{t-1}; h_i^{enc}] + b_a)
$$

其中 $[s_{t-1}; h_i^{enc}]$ 表示将两个向量沿最后一个维度拼接。$W_a \in \mathbb{R}^{d_a \times (d_s + d_h)}$ 将拼接向量映射到注意力空间，$v \in \mathbb{R}^{d_a}$ 将中间表示压缩为标量得分。

### 3.2 Softmax 归一化

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
$$

$\alpha_{t,i}$ 是解码器在生成第 $t$ 个词时对输入第 $i$ 个词的关注程度。$\sum_i \alpha_{t,i} = 1$。

### 3.3 加权求和得到上下文向量

$$
c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i^{enc}
$$

$c_t \in \mathbb{R}^{d_h}$ 是当前解码步的**动态上下文向量**——它不是固定的，而是根据当前解码需求从编码器输出中动态筛选的信息汇总。

### 3.4 生成输出

将上下文向量与上一时刻的输出拼接后送入解码器 RNN：

$$
s_t = \text{RNN}_{\text{dec}}(s_{t-1}, [y_{t-1}; c_t])
$$

$$
p(y_t | y_{<t}, X) = \text{softmax}(W_o \cdot s_t + b_o)
$$

---

## 4. 从零实现：公式到代码的精确映射

以下从零实现一个完整的 Bahdanau Attention Seq2Seq 模型，将第 3 节的每个公式逐行映射为 PyTorch 代码。

### 4.1 编码器：双向 GRU 获取上下文表示

编码器对应 2.1 节描述的流程——逐词读取源语言序列，保留所有时间步的隐藏状态。双向 GRU 使每个位置的编码融合了前后文信息。

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

### 4.2 Bahdanau Attention 模块

这是整个架构的核心，精确实现第 3.1-3.3 节的三个公式。输入解码器当前状态 $s_{t-1}$ 和编码器全部输出 $\{h_i^{enc}\}$，输出动态上下文向量 $c_t$ 和注意力权重 $\alpha_t$。

```python
class BahdanauAttention(nn.Module):
    """
    Bahdanau 加性注意力。

    计算流程（对应第 2.2 节的 Step 1-4）:
      1. 将 s_{t-1} 与每个 h_i^{enc} 拼接
      2. 经 W_a 线性变换 + tanh 非线性
      3. 经 v^T 压缩为标量得分 scores
      4. 对 scores 做 mask（PAD 位设为 -1e9）
      5. Softmax 归一化得到注意力权重 α_{t,i}
      6. 加权求和: c_t = Σ α_{t,i} · h_i^{enc}
    """

    def __init__(self, hidden_dim, attn_dim):
        """
        Args:
            hidden_dim: 解码器单向隐藏维度 d_s
            attn_dim:   注意力中间表示维度 d_a
        """
        super().__init__()
        # ① W_a: (d_s + 2*d_h, attn_dim)，将拼接向量映射到注意力空间
        # decoder 的 hidden_dim 拼接 encoder 的 hidden_dim * 2 (双向)
        self.W_a = nn.Linear(hidden_dim + hidden_dim * 2, attn_dim, bias=False)
        # ② v^T: (attn_dim, 1)，将注意力表示压缩为标量得分
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        """
        Args:
            decoder_hidden:  s_{t-1}, 单向解码器状态, (batch, hidden_dim)
            encoder_outputs: {h_i^{enc}}, 编码器所有时间步输出, (batch, src_len, hidden_dim * 2)
            mask:            PAD mask, 1=有效 0=PAD, (batch, src_len)
        Returns:
            context:      c_t, 动态上下文向量, (batch, hidden_dim * 2)
            attn_weights: α_t, 注意力权重, (batch, src_len)
        """
        src_len = encoder_outputs.size(1)  # T

        # Step 1: 评分 —— 对应公式 e_{t,i} = v^T · tanh(W_a · [s_{t-1}; h_i^{enc}])

        # 将 decoder_hidden 沿 src_len 维度复制 T 份
        # (batch, hidden_dim) -> (batch, 1, hidden_dim) -> (batch, src_len, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 拼接 decoder 状态和 encoder 输出: [s_{t-1}; h_i^{enc}]
        # (batch, src_len, hidden_dim) + (batch, src_len, hidden_dim * 2)
        #   -> (batch, src_len, hidden_dim + hidden_dim * 2)
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=-1)

        # W_a · concat + tanh
        # (batch, src_len, hidden_dim + hidden_dim*2) -> (batch, src_len, attn_dim)
        energy = torch.tanh(self.W_a(concat))

        # v^T · energy -> 标量得分
        # (batch, src_len, attn_dim) -> (batch, src_len, 1) -> squeeze -> (batch, src_len)
        scores = self.v(energy).squeeze(-1)

        # Step 2: 遮罩 + Softmax —— 对应公式 α_{t,i} = exp(e_{t,i}) / Σ exp(e_{t,j})

        # PAD 位置得分设为 -1e9，exp(-1e9) ≈ 0
        scores = scores.masked_fill(mask == 0, -1e9)
        # (batch, src_len) -> (batch, src_len)，Σ_i α_{t,i} = 1
        attn_weights = F.softmax(scores, dim=-1)

        # Step 3: 加权求和 —— 对应公式 c_t = Σ α_{t,i} · h_i^{enc}

        # bmm: (batch, 1, src_len) × (batch, src_len, hidden_dim * 2)
        #    -> (batch, 1, hidden_dim * 2) -> squeeze -> (batch, hidden_dim * 2)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context, attn_weights
```

### 4.3 解码器：自回归生成 + 注意力上下文

解码器对应 2.2 节的 Step 5 流程——在循环中的每一时间步 $t$，执行嵌入、注意力查询、拼接、GRU 前向、输出预测。Teacher Forcing 用于训练时加速收敛。

```python
class AttentionDecoder(nn.Module):
    """
    带 Bahdanau Attention 的自回归解码器。

    每步流程（对应第 2.2 节 Step 1-5）:
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
| $c_t = \text{Attention}(s_{t-1}, \{h_i^{enc}\})$ | `self.attention(...)` | 详见 4.2 节 |
| $[y_{t-1}; c_t]$ | `torch.cat((embedded, context.unsqueeze(1)), dim=-1)` | `(B, 1, embed_dim + 2*hidden_dim)` |
| $s_t = \text{GRU}(s_{t-1}, [y_{t-1}; c_t])$ | `self.rnn(rnn_input, hidden)` | `(B, 1, hidden_dim)` |
| $p(y_t) = \text{softmax}(W_o · s_t + b_o)$ | `self.fc(rnn_output.squeeze(1))` | `(B, hidden_dim)` 到 `(B, output_dim)` |
| 下一步输入决策 | `decoder_input = trg[:, t] if teacher_force else top1` | `(B,)` |

### 4.4 完整 Seq2Seq 模型

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

### 4.5 维度变化追踪

以具体参数为例追踪一次前向传播中各张量的形状变化：

- `src_vocab_size = 10000`，`trg_vocab_size = 8000`
- `embed_dim = 256`，`hidden_dim = 512`，`attn_dim = 256`
- `batch = 32`，`src_len = 20`，`trg_len = 18`

**编码器阶段：**

- `src`: `(32, 20)`，整数索引
- `embedded`: `(32, 20, 256)`，词嵌入后
- `encoder_outputs`: `(32, 20, 1024)`，双向 GRU 输出（512 × 2）
- `hidden`: `(4, 32, 512)`，num_layers × 2 个方向的最终状态

**解码器第 $t$ 步的 Attention 内部：**

- `decoder_hidden`（$s_{t-1}$）：`(32, 512)`
- `concat`（$[s_{t-1}; h_i^{enc}]$）：`(32, 20, 1536)`，即 512 + 1024
- `energy`（$\tanh(W_a · \text{concat})$）：`(32, 20, 256)`
- `scores`（$v^T · \text{energy}$）：`(32, 20)`
- `attn_weights`（Softmax）：`(32, 20)`
- `context`（$\sum \alpha_i · h_i^{enc}$）：`(32, 1024)`

**解码器第 $t$ 步的 GRU 和输出：**

- `embedded`（$y_{t-1}$ 嵌入）：`(32, 1, 256)`
- `rnn_input`（$[y_{t-1}; c_t]$）：`(32, 1, 1280)`，即 256 + 1024
- `rnn_output`（$s_t$）：`(32, 1, 512)`
- `prediction`（$W_o · s_t + b_o$）：`(32, 8000)`

---

## 5. Attention 中的 QKV 范式

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

## 6. 注意力对齐矩阵

注意力权重 $\alpha_{t,i}$ 可以可视化为**对齐矩阵**——行对应目标语言位置，列对应源语言位置。

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

### 6.1 软对齐 vs 硬对齐

传统的统计机器翻译需要显式学习一个"词对齐"表（硬对齐——每个目标词严格对应一个源词）。Attention 实现的**软对齐**有根本性优势：

- **不需要人工标注对齐数据**：模型从平行语料中自动学习对齐关系
- **一对多、多对一自然处理**：一个目标词可以以不同权重关注多个源词，不需要离散的二选一决策
- **梯度可导**：Softmax 加权求和全程可微，对齐关系通过反向传播自动优化

---

## 7. 从 Attention RNN 到 Transformer

### 7.1 对 Attention 的重新审视

Attention RNN 的成功引发了一个深层问题：Attention 本身已经提供了跨位置的直接信息通路，RNN 的循环连接是否仍然必要？

如果 Attention 能让解码器直接访问编码器的任意位置，那么 RNN 循环连接——它的主要作用是沿时间步传递信息——是否可以被纯粹的自注意力替代？

### 7.2 Transformer 的回答

2017 年，Vaswani 等人在《Attention Is All You Need》中给出了答案：**不需要 RNN**。

Transformer 完全抛弃了循环连接，纯粹依靠**自注意力**来建模序列内部的依赖关系。每个位置直接关注所有其他位置，计算复杂度 $O(T^2)$ 但梯度路径长度从 RNN 的 $O(T)$ 缩短为 $O(1)$。

### 7.3 知识迁移路径

理解 Bahdanau Attention 是理解 Transformer 的**必要前提**。以下概念一脉相承：

| Bahdanau Attention (2015) | 对应 Transformer (2017) |
|---|---|
| 解码器状态 Query $s_{t-1}$ | 每个位置的 Query 向量 $Q$ |
| 编码器状态 Key $h_i^{enc}$ | 每个位置的 Key 向量 $K$ |
| 编码器状态 Value $h_i^{enc}$ | 每个位置的 Value 向量 $V$ |
| 加性打分 $v^T \tanh(W[s;h])$ | 缩放点积打分 $\frac{QK^T}{\sqrt{d_k}}$ |
| Softmax + 加权求和 | Softmax + 加权求和（完全相同） |

---

> 注意力机制的详细数学：[[NeuralNetwork/RNN/Foundations/AttentionMechanism|注意力机制详解]]
> 回到主文档：[[NeuralNetwork/RNN/Foundations/RNNOverview|RNN 详解主文档]]
