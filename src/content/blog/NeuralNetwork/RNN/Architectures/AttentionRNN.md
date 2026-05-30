---
title: 注意力 RNN：打破信息瓶颈
date: 2026-05-29
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
  - 注意力
description: 2015 年 Bahdanau 等人将注意力机制引入 Seq2Seq，解决了上下文向量的信息瓶颈。理解这一创新如何提升长序列翻译质量并最终催生了 Transformer 架构。
image: https://img.yumeko.site/file/blog/AttentionRNN.webp
status: draft
---

## 1. Seq2Seq 的结构性缺陷

### 1.1 回顾信息瓶颈

在标准 Seq2Seq 中，整个输入序列的信息被压缩到一个固定长度的上下文向量 $C = h_T^{enc}$。无论输入是 3 个词还是 100 个词，上下文向量都是 512 维（或 1024 维）。

这导致了长句翻译质量的灾难性下降——Bahdanau 等人在 2015 年论文中的实验显示，标准 Seq2Seq 对 30 词以上的句子几乎无法产生合理的翻译。

### 1.2 Bahdanau 的关键洞察

2014-2015 年，Dzmitry Bahdanau（Yoshua Bengio 的学生）在蒙特利尔大学提出了一个直接的解决方案：

> **与其把所有信息压缩到一个向量，为什么不保留编码器的所有隐藏状态，让解码器在每一步动态分配对不同位置的计算权重？**

这个想法就是**注意力机制（Attention Mechanism）**。它让解码器不再只依赖一个固定的上下文向量，而是动态地从编码器的所有输出中提取相关信息。

---

## 2. Attention Seq2Seq 的结构



### 2.1 编码器：保留所有隐藏状态

与标准 Seq2Seq 不同，Attention Seq2Seq 的编码器**保留所有时间步的隐藏状态**：

$$
\text{编码器输出} = \{h_1^{enc}, h_2^{enc}, h_3^{enc}, ..., h_T^{enc}\}
$$

每个 $h_i^{enc}$ 都是第 $i$ 个输入词经过双向 RNN 后的表示，包含了该词的**上下文感知**编码（周围词的信息也融入了 $h_i^{enc}$）。

### 2.2 解码器：每一步动态计算上下文

在解码器的第 $t$ 步，不再是只接收一个固定的上下文向量，而是：

1. 用上一个解码器状态 $s_{t-1}$ 去"查询"编码器的所有隐藏状态
2. 计算 $s_{t-1}$ 与每个 $h_i^{enc}$ 的**匹配得分**
3. 将得分归一化为**注意力权重** $\alpha_{t,i}$
4. 用 $\alpha_{t,i}$ 对所有 $h_i^{enc}$ 做加权求和，得到**当前步的上下文向量** $c_t$
5. 将 $c_t$ 和上一步的输出一起输入解码器 RNN

**关键**：每个解码步都有一个**不同的**上下文向量 $c_t$——在生成"dog"时需要关注输入中的"狗"，在生成"running"时需要关注输入中的"跑"。

---

## 3. Attention 的数学（解码器视角）

### 3.1 第 t 个解码步

**输入**：
- 解码器上一步的状态：$s_{t-1}$
- 编码器的所有隐藏状态：$\{h_1^{enc}, h_2^{enc}, ..., h_T^{enc}\}$
- 上一个生成的词：$y_{t-1}$

**Step 1：评分**

对每个编码器位置 $i$：

$$
e_{t,i} = \text{score}(s_{t-1}, h_i^{enc}) = v^T \cdot \tanh(W_a \cdot [s_{t-1}; h_i^{enc}] + b_a)
$$

其中：
- $[s_{t-1}; h_i^{enc}]$：将解码器状态和编码器隐藏状态拼接
- $W_a, v, b_a$：得分网络的参数（所有解码步共享）

**Step 2：Softmax 归一化**

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
$$

$\alpha_{t,i}$ 是解码器在生成第 $t$ 个词时，对输入第 $i$ 个词的"关注程度"。

**Step 3：加权求和**

$$
c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i^{enc}
$$

$c_t$ 是当前解码步的**动态上下文向量**——它接收了编码器的全部输出，通过注意力权重进行加权筛选。

**Step 4：生成输出**

$$
s_t = \text{RNN}_{\text{dec}}(s_{t-1}, [y_{t-1}; c_t])
$$

$$
p(y_t | y_{<t}, X) = \text{softmax}(W_o \cdot s_t + b_o)
$$

### 3.2 Attention 中的"Query-Key-Value"范式

Bahdanau Attention 已经隐含了后来 Transformer 中通用的**QKV（Query-Key-Value）**范式：

![QKV.png](https://img.yumeko.site/file/articles/AttentionRNN/QKV.webp)

| Bahdanau Attention 概念 | 通用 QKV 概念 | 含义 |
|---|---|---|
| $s_{t-1}$（解码器状态） | Query | "我现在需要什么信息？" |
| $h_i^{enc}$（编码器状态） | Key | "我这段输入包含什么？" |
| $h_i^{enc}$（编码器状态） | Value | "我这段输入的实际内容是什么？" |
| $\alpha_{t,i}$ | Attention Weights | Query 与每个 Key 的匹配度 |
| $c_t$ | Weighted Sum of Values | 按匹配度筛选后的信息汇总 |

在原始的 Bahdanau Attention 中，Key 和 Value 是同一个 $h_i^{enc}$。Transformer 将 Key 和 Value 分开（通过不同的线性投影），提供了更大的灵活性。

---

## 4. 注意力对齐矩阵

### 4.1 可解释的翻译过程

注意力权重 $\alpha_{t,i}$ 可以可视化为**对齐矩阵**——行 = 目标语言位置，列 = 源语言位置，颜色深浅 = 注意力强度。

对于 "I love you" → "我爱你"：
$$

\begin{array}{c|cccc}
 & \text{I} & \text{love} & \text{you} & \text{〈EOS〉} \\ \hline
\text{我} & 0.80 & 0.10 & 0.05 & 0.05 \\
\text{爱} & 0.05 & 0.85 & 0.05 & 0.05 \\
\text{你} & 0.05 & 0.05 & 0.80 & 0.10
\end{array}
$$

对角线的强权重模式表明模型学到了词与词之间**单调对齐**——这是一个合理的翻译模式（中英文词序大致相同）。

对于语序不同的语言对（如日语→英语），对齐矩阵会呈现非对角模式：

$$\begin{array}{c|cccc}
 & \text{私} & \text{は} & \text{学生} & \text{です} \\ \hline
\text{I} & 0.90 & 0.05 & 0.05 & 0.00 \\
\text{am} & 0.05 & 0.80 & 0.05 & 0.10 \\
\text{a} & 0.05 & 0.05 & 0.85 & 0.05 \\
\text{student} & 0.00 & 0.05 & 0.10 & 0.85
\end{array}
$$

### 4.2 软对齐 vs 硬对齐

传统的统计机器翻译（SMT）需要显式学习一个"词对齐"表（硬对齐——每个目标词对应一个源词）。Attention 实现的是一种**软对齐**——每个目标词以不同权重关注所有源词。

这种软对齐有两个优势：
- **不需要人工标注对齐数据**：模型从翻译数据中自动学习
- **一对多、多对一自然处理**：一个目标词可以关注多个源词（用权重分布表示）

---

## 5. 从 Attention RNN 到 Transformer

### 5.1 对 Attention 的重新审视

Attention RNN 的成功引发了一个深层问题：**Attention 本身已经提供了跨位置的直接信息通路——RNN 的循环连接是否仍然必要？**

如果 Attention 能让解码器直接访问编码器的任意位置，那 RNN 的循环连接——它在 Seq2Seq 中的主要作用是将信息从一个时间步传到下一个——是否变得不那么关键了？

### 5.2 Transformer 的回答：不需要

2017 年，Vaswani 等人在《Attention Is All You Need》中直接回答了这个问题：**不需要 RNN**。

Transformer 完全抛弃了循环连接，纯粹依靠**自注意力（Self-Attention）**来建模序列内部的依赖关系。每个位置直接关注所有其他位置，计算复杂度 $O(T^2)$ 但梯度路径长度从 RNN 的 $O(T)$ 缩短为 $O(1)$。

### 5.3 知识迁移路径

理解 Bahdanau Attention 是理解 Transformer 的**必要前提**。以下概念一脉相承：

| Bahdanau Attention (2015) | → | Transformer Self-Attention (2017) |
|---|---|---|
| 解码器状态 Query $s_{t-1}$ | → | 每个位置的 Query 向量 $Q$ |
| 编码器状态 Key $h_i^{enc}$ | → | 每个位置的 Key 向量 $K$ |
| 编码器状态 Value $h_i^{enc}$ | → | 每个位置的 Value 向量 $V$ |
| 加性打分 $v^T \tanh(W[s;h])$ | → | 缩放点积打分 $\frac{QK^T}{\sqrt{d_k}}$ |
| Softmax + 加权求和 | → | Softmax + 加权求和（完全相同） |

---

## 6. EmotionClassification 中的 Attention

![Seq2SeqVSAttention.png](https://img.yumeko.site/file/articles/AttentionRNN/Seq2SeqVSAttention.webp)

EmotionClassification 项目使用的 Attention 是 Bahdanau Attention 的变体——但用于**序列分类**而非 Seq2Seq：

1. 编码器（BiLSTM）输出所有隐藏状态：
$$
\{\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T\}
$$

2. 注意力层对每个位置独立打分：
$$
\mathrm{Score}(\mathbf{h}_i) = \mathbf{v}^{\top} \, \tanh(\mathbf{W}\mathbf{h}_i + \mathbf{b})
$$

3. 经 softmax 归一化得到注意力权重：
$$
\alpha_i = \operatorname{softmax}(e_1, e_2, \dots, e_T)
$$

4. 加权求和得到上下文向量：
$$
\mathbf{c} = \sum_{i=1}^{T} \alpha_i \mathbf{h}_i
$$

5. 分类器通过线性变换和 Sigmoid 输出概率：
$$
p = \sigma\!\big(\mathrm{Linear}(\mathbf{c})\big)
$$
![BahdanauAttention.png](https://img.yumeko.site/file/articles/AttentionRNN/BahdanauAttention.webp)
与原始 Seq2Seq Attention 的关键区别：
- **没有解码器状态**作为 Query：改用一个可学习的 `v` 向量直接为每个 $h_i$ 打分
- **全局 Attention**：不是在每个解码步动态计算，而是一次性计算整个序列的加权表示
- **目标不同**：不是为了对齐翻译，而是为了找到对分类最重要的词

---

> 注意力机制的详细数学：[[NeuralNetwork/RNN/Foundations/AttentionMechanism|注意力机制详解]]
> EmotionClassification 的完整实现：[[Projects/EmotionClassification|EmotionClassification 项目]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
