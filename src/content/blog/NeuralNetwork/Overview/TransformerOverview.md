---
title: Transformer 架构总览
date: 2026-06-27
category: 神经网络/总览
tags:
  - Transformer
  - 深度学习
  - 架构
description: 从 RNN 的顺序计算问题出发，区分传统 Transformer 与现代 LLM Transformer，并解释 Encoder/Decoder 差异。
image: https://img.yumeko.site/file/blog/cover/1782558436087_TransformerOverview.webp
status: published
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/RNN/Attention|注意力机制详解]]，再阅读本文。

## 1. Transformer 想解决什么问题

RNN 按时间步递归计算：

$$
h_t = f(h_{t-1}, x_t)
$$

这种结构天然适合序列，但也带来两个限制：

| 限制 | 影响 |
|:--|:--|
| 难以并行 | 第 $t$ 步必须等待第 $t-1$ 步完成 |
| 长距离依赖困难 | 远处信息需要经过很多次状态传递 |

Transformer 的核心变化是：不再用一个隐藏状态逐步传递信息，而是让序列中所有 token 通过 Self-Attention 直接建立联系。

给定输入：

$$
X = [x_1, x_2, \dots, x_n]
$$

Self-Attention 会为每个 token 重新计算一个上下文表示：

$$
z_i = \sum_{j=1}^{n} \alpha_{ij} v_j
$$

其中 $\alpha_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的关注权重。

---

## 2. 传统 Transformer 与现代 Transformer

Transformer 可以分成两个层次理解：

| 层次 | 关注点 | 典型内容 |
|:--|:--|:--|
| 传统 Transformer | 模型结构本身 | Multi-Head Attention、FFN、残差、LayerNorm、位置编码 |
| 现代 Transformer | 大模型训练与推理改造 | RoPE、Pre-LN、RMSNorm、SwiGLU、GQA、KV Cache、FlashAttention、MoE |

这里的“传统”和“现代”不是两个完全不同的模型，而是同一条技术路线上的不同阶段。

传统 Transformer 更适合回答：一个 Transformer Block 如何计算？为什么多头注意力能让 token 交换信息？Encoder、Decoder、Decoder-only 有什么区别？

现代 Transformer 更适合回答：模型变大、序列变长、推理变慢之后，工程上如何继续训练和部署？例如 KV Cache 不是训练阶段的网络层，而是自回归推理时复用历史 Key/Value 的缓存机制。

> [!NOTE] 关键区分
> 手写一个教学版 Decoder-only Transformer，通常需要 Multi-Head Attention、FFN、残差、归一化、位置编码和 causal mask。  
> KV Cache、FlashAttention、GQA 这类内容属于现代推理或大模型工程优化，不是理解基础 Transformer 的前置条件。

---

## 3. Transformer 的三种常见结构

![ArchitectureComparison.png](https://img.yumeko.site/file/blog/articles/1782558536786_ArchitectureComparison.webp)

### 3.1 Encoder-only

Encoder-only 模型让所有位置互相可见，适合理解、分类、抽取类任务。

| 阶段 | 说明 |
|:--|:--|
| 输入序列 | token id 序列 |
| Embedding | Token Embedding + Position Encoding |
| Encoder 堆叠 | 多层 Encoder Block |
| 输出头 | 分类头、标注头或表示向量 |

代表模型：BERT、RoBERTa。

特点：

- 每个 token 可以看见左右两侧上下文。
- 不适合直接做自回归生成。
- 常用于文本分类、命名实体识别、句向量表示。

### 3.2 Encoder-Decoder

Encoder-Decoder 是原始 Transformer 的结构，适合序列到序列任务。

| 部分 | 作用 |
|:--|:--|
| Encoder | 将源序列编码为上下文表示 |
| Decoder | 在已生成目标 token 的条件下继续生成 |
| Cross-Attention | 让 Decoder 读取 Encoder 的输出 |

代表任务：机器翻译、摘要、文本改写。

Decoder 中通常包含三类子层：

| 子层 | 作用 |
|:--|:--|
| Masked Self-Attention | 目标端只能看已生成 token |
| Cross-Attention | 从 Encoder 输出中读取源序列信息 |
| FFN | 对每个位置做非线性变换 |

### 3.3 Decoder-only

Decoder-only 模型只保留自回归 Decoder，用于预测下一个 token。

| 阶段 | 说明 |
|:--|:--|
| 输入 token | 已有上下文 |
| Embedding | token 表示与位置表示 |
| Decoder 堆叠 | 多层带因果 mask 的 Block |
| LM Head | 输出下一个 token 的 logits |

训练目标是：

$$
P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

现代生成式语言模型大多采用这种结构。

---

## 4. 一个 Transformer Block 里有什么

典型 Transformer Block 包含四个关键部件：

| 部件 | 作用 |
|:--|:--|
| Self-Attention | 让 token 之间交换信息 |
| FFN | 对每个 token 的表示做非线性变换 |
| Residual | 保留原始信息，稳定梯度 |
| LayerNorm | 控制激活分布，稳定训练 |

Pre-LN 形式可以写成：

$$
x' = x + \operatorname{Attention}(\operatorname{LN}(x))
$$

$$
y = x' + \operatorname{FFN}(\operatorname{LN}(x'))
$$

这说明 Transformer Block 不是只有 Attention。Attention 负责“和谁交换信息”，FFN 负责“交换后如何加工表示”。

---

## 5. Embedding 与输出头

模型输入不是原始文本，而是 token id：

$$
[13, 104, 25, 8, \dots]
$$

Embedding 层将 token id 转为向量：

$$
E(x_i) \in \mathbb{R}^{d_{\text{model}}}
$$

因为 Attention 本身不感知顺序，还需要加入位置编码：

$$
h_i^{(0)} = E(x_i) + P_i
$$

最终输出头将隐藏状态映射回词表大小：

$$
\text{logits}_i = h_i W_{\text{vocab}} + b
$$

其中 `logits` 的形状通常是：

```text
(batch_size, seq_length, vocab_size)
```

---

## 6. Decoder-only 的完整数据流

以语言模型为例：

| 顺序 | 阶段 | 输出含义 |
|:--|:--|:--|
| 1 | `input_ids` | token id |
| 2 | token embedding | token 语义向量 |
| 3 | position encoding | 注入顺序信息 |
| 4 | Transformer Blocks | 上下文表示 |
| 5 | final LayerNorm | 稳定最终表示 |
| 6 | LM Head | 词表 logits |

训练时，输入和目标错开一位：

$$
\text{input} = [x_1, x_2, \dots, x_{T-1}]
$$

$$
\text{target} = [x_2, x_3, \dots, x_T]
$$

模型学的是“看到前文后，下一个 token 是什么”。

---

## 7. 常见误解

1. **Transformer 不等于 Attention。**  
   Attention 是最重要的子层，但 Transformer 还依赖 FFN、残差、归一化和位置编码。

2. **Decoder-only 不是原始 Transformer 的完整结构。**  
   原始 Transformer 是 Encoder-Decoder，Decoder-only 是后来在生成式语言模型中广泛使用的变体。

3. **Self-Attention 不会自动理解顺序。**  
   如果不加位置编码，模型只知道 token 集合，不知道 token 的先后。

4. **深层 Transformer 的稳定性来自多个设计共同作用。**  
   残差、LayerNorm、初始化、学习率 warmup、优化器都会影响训练稳定性。

5. **KV Cache 不等于 Multi-Head Attention。**  
   Multi-Head Attention 是模型结构；KV Cache 是生成推理时保存历史 $K,V$ 的缓存策略。一个模型可以有多头注意力，但不实现 KV Cache。

---

## 8. 总结

Transformer 的核心思想是用 Self-Attention 替代循环递归，让序列中任意两个位置可以直接交互。

从结构上看：

| 结构 | 适合任务 |
|:--|:--|
| Encoder-only | 理解、分类、抽取 |
| Encoder-Decoder | 翻译、摘要、序列到序列 |
| Decoder-only | 自回归生成 |

## 9. 阅读路线

| 顺序 | 层次 | 文章 | 重点 |
|:--|:--|:--|:--|
| 1 | 传统结构 | [[NeuralNetwork/Transformer/SelfAttentionMechanism\|Self-Attention 机制详解]] | $Q,K,V$、多头注意力、mask |
| 2 | 传统结构 | [[NeuralNetwork/Transformer/TransformerBlock\|Transformer Block 结构]] | Attention、FFN、残差、LayerNorm |
| 3 | 连接传统与现代 | [[NeuralNetwork/Transformer/PositionalEncoding\|Transformer 位置编码]] | Sinusoidal、Learned、RoPE、ALiBi |

---

> **相关文章**：
> - [[NeuralNetwork/RNN/Attention|注意力机制详解]]
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
> - [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]
