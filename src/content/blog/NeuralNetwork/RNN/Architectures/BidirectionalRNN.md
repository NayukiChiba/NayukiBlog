---
title: 双向 RNN：同时看到过去和未来
date: 2026-05-29
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: Schuster 和 Paliwal 在 1997 年提出的双向 RNN 架构，理解它如何通过正反向两个 RNN 利用序列的双向上下文信息，以及为什么不能用于自回归文本生成。
image: https://img.yumeko.site/file/blog/BidirectionalRNN.webp
status: draft
---

## 1. 动机：单向 RNN 的盲区

### 1.1 单向 RNN 只能看到过去
![UnidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/UnidirectionalRNN.webp)

标准（单向）RNN 在每个时间步 $t$ 只能看到 $x_1, x_2, ..., x_t$——过去的信息。它看不到 $x_{t+1}$ 及之后的内容（未来）。

在文本理解任务中，这在很多时候是不够的。考虑中文分词任务：

> "乒乓球拍卖完了"

这句话有歧义——是"乒乓/球拍/卖完了"还是"乒乓球/拍卖/完了"？要正确分词，需要同时看到前后的词。单向 RNN 在读到"拍卖"时无法判断后面是"卖完了"还是结构助词。

### 1.2 生物学的启发

Schuster 和 Paliwal 在 1997 年的双向 RNN 论文中，借鉴了人类听觉皮层处理语音的机制——**大脑在理解一个音素时，同时参考它之前和之后的声音**。这不是简单的"从左到右"处理，而是一种同时考虑前后上下文的并行分析。

---

## 2. 双向 RNN 的结构

### 2.1 两个独立的 RNN
![BidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/BidirectionalRNN.webp)

双向 RNN（BiRNN）使用**两个完全独立的 RNN**，分别沿正反两个方向读取序列：

$$
\overrightarrow{h_t} = \text{RNN}_{\text{forward}}(x_t, \overrightarrow{h_{t-1}})
$$

$$
\overleftarrow{h_t} = \text{RNN}_{\text{backward}}(x_t, \overleftarrow{h_{t+1}})
$$

**关键观察**：
- 正向 RNN 按 $t = 1, 2, ..., T$ 顺序处理
- 反向 RNN 按 $t = T, T-1, ..., 1$ 顺序处理
- 两者**共享相同输入 $x_t$**，但使用**完全不同的权重参数**
- 两者之间**没有直接的交互**——它们在各自的时间方向独立运行

### 2.2 拼接而不是加和

每个时间步的双向表示是正反向隐藏状态的**拼接**：

$$
h_t^{\text{bi}} = [\overrightarrow{h_t}; \overleftarrow{h_t}]
$$

而不是加和。拼接保留了每个方向独立的"视角"。

### 2.3 输出维度翻倍

如果单向的 hidden_size = 256，双向后为 512：
$$
[\text{正向: } (B, 256); \text{反向: } (B, 256)] \rightarrow (B, 512)
$$
$$
\mathbf{h}^{\text{bi}} = [\overrightarrow{\mathbf{h}}; \overleftarrow{\mathbf{h}}] \quad \in \mathbb{R}^{512}
$$

这两个都可以说明正向和反向的256维向量拼接成512维的输出。

后续所有层（如 Attention、Linear）的输入维度都需要相应调整为 512。

### 2.4 多层双向 RNN

![MultiLayeredBidirectionalRNN.png](https://img.yumeko.site/file/articles/BidirectionalRNN/MultiLayeredBidirectionalRNN.webp)

在多层的双向 RNN 中，第 $l$ 层接收第 $l-1$ 层的双向输出作为输入：

$$\begin{array}{c}
x \longrightarrow \big[\text{正向RNN}_1 + \text{反向RNN}_1\big] \longrightarrow h_1^{\text{bi}} \; (512\text{维}) \\
\downarrow \\
\big[\text{正向RNN}_2 + \text{反向RNN}_2\big] \longrightarrow h_2^{\text{bi}} \; (512\text{维})
\end{array}$$

每层的 input_size = 上一层的 hidden_size × 2。

---

## 3. 双向 RNN 的数学

### 3.1 第 1 层

*正向：*

$$
\overrightarrow{h_t^{(1)}} = \tanh(W_{xh}^{\rightarrow} \cdot x_t + W_{hh}^{\rightarrow} \cdot \overrightarrow{h_{t-1}^{(1)}} + b_h^{\rightarrow})
$$

*反向：*

$$
\overleftarrow{h_t^{(1)}} = \tanh(W_{xh}^{\leftarrow} \cdot x_t + W_{hh}^{\leftarrow} \cdot \overleftarrow{h_{t+1}^{(1)}} + b_h^{\leftarrow})
$$

*拼接：*

$$
h_t^{\text{bi}(1)} = \begin{pmatrix} \overrightarrow{h_t^{(1)}} \\ \overleftarrow{h_t^{(1)}} \end{pmatrix}
$$

### 3.2 PyTorch 实现

```python
bilstm = nn.LSTM(
    input_size=300,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,   # ← 关键参数
    dropout=0.5,
    batch_first=True,
)

# 隐藏状态形状变化
# (num_layers * num_directions, batch, hidden_size)
# = (2 * 2, batch, 256) = (4, batch, 256)

h0 = torch.zeros(4, batch_size, 256)
c0 = torch.zeros(4, batch_size, 256)

output, (hn, cn) = bilstm(x, (h0, c0))
# output.shape = (batch, seq_len, 512)
# hn.shape = (4, batch, 256)

# 获取最后一层的最终双向隐藏状态
forward_last = hn[-2, :, :]   # (batch, 256) — 正向的最后一步
backward_last = hn[-1, :, :]  # (batch, 256) — 反向的最后一步
bidirectional_last = torch.cat([forward_last, backward_last], dim=-1)  # (batch, 512)
```

---

## 4. 什么时候不能用双向 RNN？

### 4.1 自回归生成的因果约束

在文本生成任务中，模型在预测 $y_t$ 时**不能看到** $x_{t+1}, x_{t+2}, ...$——因为那些是未来的、还没有生成的内容。

双向 RNN 的反向路径需要从序列末尾向前扫描，这意味着整个序列必须**完全可用**。对于自回归生成，这是不可能的。

### 4.2 任务适用性

| 任务类型 | 双向可用？ | 原因 |
|----------|:---:|------|
| 文本分类 | ✅ | 分类时整条评论已完整 |
| 序列标注 | ✅ | 标注时整句话已完整 |
| 机器翻译（编码器） | ✅ | 编码时源语言句子已完整 |
| 机器翻译（解码器） | ❌ | 解码时目标语言逐词生成 |
| 语言模型 / 文本生成 | ❌ | 预测下一个词时看不到未来 |
| 语音识别（离线） | ✅ | 整段音频已录制 |
| 语音识别（流式/在线） | ❌ | 只看到已接收的音频片段 |

### 4.3 两个项目中的选择

| 项目 | 双向？ | 原因 |
|------|:---:|------|
| Char-RNN | ❌ 否 | 文本生成是自回归的，必须用单向 |
| EmotionClassification | ✅ 是 | 分类时整条评论完全可用，双向显著提升效果 |

---

## 5. BiLSTM + Attention：NLP 分类的经典组合

在 Transformer 出现之前，BiLSTM + Attention 是 NLP 分类/标注任务的**事实标准**：

```
输入 → Embedding → BiLSTM → Attention Pooling → Classifier → 输出
```

- **BiLSTM**：提供丰富的双向上下文编码
- **Attention**：通过可学习参数给不同位置的编码分配权重
- 这个组合在情感分析、文本分类、问答系统等任务上统治了 2015-2018 年

EmotionClassification 项目采用的就是这个经典模式。

---

> 双向 RNN 在项目中的使用：[[Projects/EmotionClassification|EmotionClassification 项目]]
> 架构对比：[[NeuralNetwork/RNN/Architectures/ArchitectureComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
