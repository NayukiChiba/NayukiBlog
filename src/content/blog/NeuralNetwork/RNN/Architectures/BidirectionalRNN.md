---
title: 双向 RNN：同时看到过去和未来
date: 2026-05-22
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: Schuster 和 Paliwal 在 1997 年提出的双向 RNN 架构，理解它如何通过正反向两个 RNN 利用序列的双向上下文信息，完整的数学推导，从零 PyTorch 实现，以及为什么不能用于自回归文本生成。
image: https://img.yumeko.site/file/blog/BidirectionalRNN.webp
status: published
---

## 1. 动机：单向 RNN 的盲区

### 1.1 单向 RNN 的信息不对称

![UnidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/UnidirectionalRNN.webp)

标准（单向）RNN 在每个时间步 $t$ 只能看到 $x_1, x_2, \dots, x_t$——过去的信息。它看不到 $x_{t+1}$ 及之后的内容。这个性质称为**因果约束（Causal Constraint）**。

在文本理解任务中，这经常不够。考虑有歧义的中文片段：

> "乒乓球拍卖完了"

这句话至少有两种切分方式："乒乓/球拍/卖完了"或"乒乓球/拍卖/完了"。单向 RNN 从左到右读到"拍卖"时，还未看到后面的"完了"，无法利用后文来消歧。而人类在读到这句话时，视线会自然地前后扫描，利用完整上下文来确定正确切分。

### 1.2 形式化描述

单向 RNN 在位置 $t$ 的隐藏状态 $h_t$ 仅依赖于 $x_1, x_2, \dots, x_t$：

$$
h_t = f(x_t, h_{t-1}) = f(x_t, f(x_{t-1}, f(\dots, f(x_1, h_0)\dots)))
$$

$h_t$ 对 $x_{t+1}$ 及其之后的输入完全无感知。这意味着同一个词在不同上下文中的表示是**有偏的**——它只融合了左侧上下文，完全缺失了右侧上下文的信息。

---

## 2. 双向 RNN 的结构与数学

![BidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/BidirectionalRNN.webp)

### 2.1 核心思想：两个方向，独立运行

双向 RNN（BiRNN）使用**两个完全独立的 RNN**，分别沿正反两个方向读取序列。两者共享输入 $x_t$ 但使用各自的权重参数，在各自的时间方向独立运行，彼此之间没有直接的隐层交互。

正向 RNN 按 $t = 1, 2, \dots, T$ 顺序处理：

$$
\overrightarrow{h_t} = \text{RNN}_{\text{forward}}(x_t, \overrightarrow{h_{t-1}})
$$

反向 RNN 按 $t = T, T-1, \dots, 1$ 顺序处理——它从序列末尾开始，逐步向前扫描：

$$
\overleftarrow{h_t} = \text{RNN}_{\text{backward}}(x_t, \overleftarrow{h_{t+1}})
$$

注意反向 RNN 的索引：在时间步 $t$，它接收的是 $t+1$ 的隐藏状态——因为它的"过去"在序列的更后面。这意味着反向 RNN 在每个位置 $t$ 看到的"历史"实际上是 $x_T, x_{T-1}, \dots, x_{t+1}$，即位置 $t$ 之后的全部内容。

### 2.2 双向表示的构建：拼接策略

正反向独立运行后，在**每个时间步**将两个方向的隐藏状态拼接：

$$
h_t^{\text{bi}} = [\overrightarrow{h_t}; \overleftarrow{h_t}]
$$

其中 $[\cdot; \cdot]$ 表示沿特征维拼接。如果单向的 hidden_size 为 $d_h$，则：

$$
\overrightarrow{h_t} \in \mathbb{R}^{d_h}, \quad \overleftarrow{h_t} \in \mathbb{R}^{d_h}, \quad h_t^{\text{bi}} \in \mathbb{R}^{2d_h}
$$

**为什么是拼接而不是加和？** 如果采用加和 $\overrightarrow{h_t} + \overleftarrow{h_t}$，两个方向的信息被强制混合到一个维度较小的向量中，模型无法区分"这个维度的高激活值来自正向还是反向"。拼接保留了每个方向独立的特征空间，让后续层可以自由选择如何组合两个方向的信息。

### 2.3 各时间步的双向覆盖

以 $T=4$ 为例，追踪每个位置的正反向感知范围：

| 位置 $t$ | $h_t^{\text{bi}}$ 感知的输入范围 | 说明 |
|:---:|------|------|
| $t=1$ | $x_1$（正向）+ $x_4, x_3, x_2$（反向） | 能看到整个序列 |
| $t=2$ | $x_1, x_2$（正向）+ $x_4, x_3$（反向） | 能看到整个序列 |
| $t=3$ | $x_1, x_2, x_3$（正向）+ $x_4$（反向） | 能看到整个序列 |
| $t=4$ | $x_1, \dots, x_4$（正向）+ 无（反向） | 能看到整个序列 |

**关键结论**：$t=1$ 到 $t=4$ 的每个位置，双向拼接后都获取了整个序列的完整信息——只不过越靠前的位置，正向贡献少而反向贡献多；越靠后的位置则相反。

### 2.4 多层双向 RNN 的逐层展开

![MultiLayeredBidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/MultiLayeredBidirectionalRNN.webp)

在多层的双向 RNN 中，第 $l$ 层的输入是第 $l-1$ 层的双向输出。以两层为例：

**第 1 层**：接收原始输入 $x_t$，产出第一层双向表示。

正向第 1 层：
$$
\overrightarrow{h_t^{(1)}} = \tanh\left(W_{xh}^{\rightarrow(1)} \cdot x_t + W_{hh}^{\rightarrow(1)} \cdot \overrightarrow{h_{t-1}^{(1)}} + b_h^{\rightarrow(1)}\right)
$$

反向第 1 层：
$$
\overleftarrow{h_t^{(1)}} = \tanh\left(W_{xh}^{\leftarrow(1)} \cdot x_t + W_{hh}^{\leftarrow(1)} \cdot \overleftarrow{h_{t+1}^{(1)}} + b_h^{\leftarrow(1)}\right)
$$

拼接后传给第 2 层：
$$
h_t^{\text{bi}(1)} = \begin{pmatrix} \overrightarrow{h_t^{(1)}} \\ \overleftarrow{h_t^{(1)}} \end{pmatrix} \in \mathbb{R}^{2d_h}
$$

**第 2 层**：接收第 1 层的双向表示 $h_t^{\text{bi}(1)}$ 作为输入。此时输入维度已是 $2d_h$ 而不是原始的 $d_x$。

正向第 2 层：
$$
\overrightarrow{h_t^{(2)}} = \tanh\left(W_{xh}^{\rightarrow(2)} \cdot h_t^{\text{bi}(1)} + W_{hh}^{\rightarrow(2)} \cdot \overrightarrow{h_{t-1}^{(2)}} + b_h^{\rightarrow(2)}\right)
$$

反向第 2 层：
$$
\overleftarrow{h_t^{(2)}} = \tanh\left(W_{xh}^{\leftarrow(2)} \cdot h_t^{\text{bi}(1)} + W_{hh}^{\leftarrow(2)} \cdot \overleftarrow{h_{t+1}^{(2)}} + b_h^{\leftarrow(2)}\right)
$$

第 2 层拼接输出：
$$
h_t^{\text{bi}(2)} = \begin{pmatrix} \overrightarrow{h_t^{(2)}} \\ \overleftarrow{h_t^{(2)}} \end{pmatrix} \in \mathbb{R}^{2d_h}
$$

**维度约束**：第 2 层的 input_size 必须等于第 1 层的 $2d_h$（双向输出的拼接维度）。因此，如果第 1 层使用 hidden_size=256，第 2 层的 input_size 必须是 512。所有层的 hidden_size 通常保持一致（均为 $d_h$），但输入维度从第 2 层开始翻倍。

### 2.5 参数独立性

正向和反向 RNN 拥有**完全独立**的参数集。以单层为例：

- 正向参数：$W_{xh}^{\rightarrow} \in \mathbb{R}^{d_h \times d_x}$，$W_{hh}^{\rightarrow} \in \mathbb{R}^{d_h \times d_h}$，$b_h^{\rightarrow} \in \mathbb{R}^{d_h}$
- 反向参数：$W_{xh}^{\leftarrow} \in \mathbb{R}^{d_h \times d_x}$，$W_{hh}^{\leftarrow} \in \mathbb{R}^{d_h \times d_h}$，$b_h^{\leftarrow} \in \mathbb{R}^{d_h}$

两组参数合计 $2 \times (d_h \cdot d_x + d_h^2 + d_h)$。双向 RNN 的参数量约为单向的 $2$ 倍（精确倍数取决于输入维度和输出层）。

---

## 3. 从零实现双向 RNN

本节提供两种实现方式：PyTorch 内置的 `bidirectional=True`，以及手动实现两个 RNN 的正反向拼接，以展示其内部机制。

### 3.1 使用 PyTorch 内置参数

```python
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """双向 LSTM 序列分类器。

    流程:
      input (B, L) -> Embedding -> BiLSTM -> 取最后输出 -> Linear -> 分类
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, num_classes, dropout):
        super().__init__()
        # 词嵌入: (vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 双向 LSTM
        # bidirectional=True 后, 输出维度自动变为 hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        # 分类器: 输入维度 = hidden_dim * 2 (双向拼接)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) 整数索引序列
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 双向 LSTM 前向
        # output: (batch, seq_len, hidden_dim * 2)
        # hn:     (num_layers * 2, batch, hidden_dim)
        output, (hn, cn) = self.lstm(embedded)

        # 取最后一个时间步的双向表示做分类
        # (batch, seq_len, hidden_dim * 2) -> (batch, hidden_dim * 2)
        last_output = output[:, -1, :]

        # (batch, hidden_dim * 2) -> (batch, num_classes)
        logits = self.classifier(last_output)
        return logits
```

### 3.2 手动实现：两个独立 RNN + 拼接

以下代码展示了双向 RNN 的内部机制——创建两个独立的 RNN（正向和反向），分别前向传播，然后拼接。

```python
class ManualBiRNN(nn.Module):
    """手动实现的双向 RNN，展示内部机制。

    不使用 bidirectional=True，而是显式创建两个 RNN
    并分别沿正反向传播，最后拼接。
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        # 正向 RNN: 沿 t=1..T 顺序处理
        self.forward_rnn = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        # 反向 RNN: 沿 t=T..1 顺序处理（通过翻转输入实现）
        self.backward_rnn = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            bi_output: (batch, seq_len, hidden_dim * 2)
        """
        batch_size, seq_len, _ = x.shape

        # 正向: 正常顺序前向传播
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        forward_out, _ = self.forward_rnn(x)

        # 反向: 将输入沿时间轴翻转, 前向传播后再翻转回来
        # 翻转后的输入第1个位置是原始的 x_T, 最后是 x_1
        x_reversed = torch.flip(x, dims=[1])
        # (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        backward_out, _ = self.backward_rnn(x_reversed)
        # 翻转回来, 使位置 t 对应原始序列的位置 t
        backward_out = torch.flip(backward_out, dims=[1])

        # 拼接正反向输出
        # (batch, seq_len, hidden_dim) + (batch, seq_len, hidden_dim)
        #   -> (batch, seq_len, hidden_dim * 2)
        bi_output = torch.cat([forward_out, backward_out], dim=-1)
        return bi_output
```

**手动实现的关键操作**：`torch.flip(x, dims=[1])` 将序列沿时间轴翻转。翻转后，`backward_rnn` 第一个时间步接收原始的 $x_T$，第二个时间步接收 $x_{T-1}$，以此类推——这恰好是反向 RNN 所需的处理顺序，$\overleftarrow{h_t}$ 的依赖方向自然指向 $t+1$。前向完成后再次 `flip` 将输出恢复到与原始序列对齐的位置。

### 3.3 隐藏状态的维度

双向 RNN 的隐藏状态 `hn` 维度为 `(num_layers * 2, batch, hidden_dim)`。其中：

- `hn[0]`：第 1 层正向的最终状态
- `hn[1]`：第 1 层反向的最终状态
- `hn[2]`：第 2 层正向的最终状态
- `hn[3]`：第 2 层反向的最终状态

取最后一层双向最终状态的正确方式：

```python
# 最后一层正向最终状态: hn[-2]  (num_layers*2 的倒数第 2 个)
# 最后一层反向最终状态: hn[-1]  (num_layers*2 的倒数第 1 个)
last_forward = hn[-2, :, :]   # (batch, hidden_dim)
last_backward = hn[-1, :, :]  # (batch, hidden_dim)
# 拼接为完整双向表示
last_bi = torch.cat([last_forward, last_backward], dim=-1)  # (batch, hidden_dim * 2)
```

---

## 4. 什么时候不能用双向 RNN？

### 4.1 自回归生成的因果约束

在文本生成任务中，模型在预测 $y_t$ 时**不能看到** $x_{t+1}, x_{t+2}, \dots$——因为那些是未来的、还没有生成的内容。双向 RNN 的反向路径需要从序列末尾向前扫描，这意味着整个序列必须**完全可用**。对于自回归生成，这个前提不成立。

因此，**语言模型和文本生成的解码器必须是单向的**。但编码器（如在 Seq2Seq 机器翻译中）可以使用双向——因为编码时源语言句子已完整给出。

### 4.2 任务适用性

| 任务类型 | 双向可用？ | 原因 |
|----------|:---:|------|
| 文本分类 | 是 | 分类时整条文本已完整 |
| 序列标注 | 是 | 标注时整句话已完整 |
| 机器翻译（编码器） | 是 | 编码时源语言句子已完整 |
| 机器翻译（解码器） | 否 | 解码时目标语言逐词生成 |
| 语言模型 / 文本生成 | 否 | 预测下一个词时看不到未来 |
| 语音识别（离线） | 是 | 整段音频已录制 |
| 语音识别（流式/在线） | 否 | 只能看到已接收的音频片段 |

---

## 5. BiLSTM + Attention：NLP 分类的经典组合

在 Transformer 出现之前，BiLSTM + Attention 是 NLP 分类和标注任务的**事实标准**（2015-2018）。其完整流程为：

1. **嵌入层**：将离散的 token 索引映射为稠密向量 $e_t \in \mathbb{R}^{d_e}$
2. **BiLSTM 编码**：对嵌入序列双向编码，每步产出 $h_t^{\text{bi}} = [\overrightarrow{h_t}; \overleftarrow{h_t}] \in \mathbb{R}^{2d_h}$
3. **Attention 池化**：通过可学习参数对所有时间步的双向表示做加权求和，得到上下文向量 $c = \sum_t \alpha_t \cdot h_t^{\text{bi}}$
4. **分类器**：将上下文向量 $c$ 通过全连接层映射到类别空间

**为什么双向在这里至关重要？** Attention 的加权求和发生在双向编码之后。如果编码器是单向的，每个 $h_t$ 只包含左侧上下文；经过加权后，上下文向量 $c$ 也天然偏向序列的后半段（因为靠后的 $h_t$ 信息更多）。双向编码确保了每个位置的 $h_t^{\text{bi}}$ 都拥有**同等质量的前后文信息**，Attention 的权重分配才会准确反映各位置的语义重要性而非信息不对称性。

---

> 架构对比：[[NeuralNetwork/RNN/Architectures/ArchitectureComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
