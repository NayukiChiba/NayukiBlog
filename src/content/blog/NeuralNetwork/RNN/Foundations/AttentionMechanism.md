---
title: 注意力机制详解：让 RNN 学会"看哪里"
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
  - 注意力
description: 从信息瓶颈问题出发，深入拆解 Bahdanau 加性注意力的评分计算、Softmax 归一化、加权求和和 Mask 处理的完整数学流程，并展示 Attention 权重的可解释性价值。
image: https://img.yumeko.site/file/blog/AttentionMechanism.webp
status: draft
---

## 1. RNN 的"信息瓶颈"问题

### 1.1 问题描述

在标准的 Seq2Seq 或序列分类 RNN 中，整个输入序列 $[x_1, x_2, \dots, x_T]$ 经过双向 RNN 处理后，所有时间步的信息最终被压缩到一个固定维度的向量 $h_{\text{final}} \in \mathbb{R}^{d_h}$（如 $d_h = 512$）中，然后送入分类器或解码器。

**无论输入有多长（10 词或 500 词），最终表示都是固定 $d_h$ 维。** 固定维度的向量无法完整编码长序列的全部信息，信息丢失是结构性的。这就是**信息瓶颈（Information Bottleneck）**。

![InformationBottleneck.png](https://img.yumeko.site/file/articles/AttentionMechanism/InformationBottleneck.webp)

### 1.2 信息瓶颈的具体表现

以 Seq2Seq 机器翻译为例，"编码器的最后一个隐藏状态 → 解码器"是唯一的信息传递通道：

- 短句（5-10 词）：$d_h$ 维向量足够编码
- 中句（20-30 词）：开始丢失细节
- 长句（50+ 词）：缺失关键信息，翻译质量急剧下降

2014 年 Bahdanau 等人的实验表明，标准 Seq2Seq 在超过 30 词的句子上性能明显退化。

### 1.3 Attention 的核心洞察

与其把所有信息压缩到一个向量，不如让解码器/分类器直接访问编码器的**所有**输出 $\{h_1, h_2, \dots, h_T\}$，并通过可学习的注意力权重动态确定各位置的重要性。

这就是 Attention——一种**动态加权机制**，让模型能够为输入序列的每个位置分配一个重要性权重。

---

## 2. Bahdanau 加性注意力：完整数学流程

### 2.1 四步流程总览

给定编码器所有时刻的隐藏状态 $\{h_1, h_2, \dots, h_T\}$（每个 $h_i \in \mathbb{R}^{d_h}$），注意力机制通过以下四步计算上下文向量 $c$：

1. **评分（Scoring）**：对每个 $h_i$ 计算标量得分 $\text{score}_i$
2. **遮罩（Masking）**：将 PAD 位置的得分压制为负无穷
3. **Softmax 归一化**：将遮罩后的得分转化为概率分布 $\alpha_i$
4. **加权求和**：以 $\alpha_i$ 为权重对 $h_i$ 加权求和，得到上下文向量 $c$

![Attention.png](https://img.yumeko.site/file/articles/AttentionMechanism/Attention.webp)

下面逐步拆解。

### 2.2 第 1 步：评分（Scoring）

对每个时间步 $i$ 的隐藏状态 $h_i$，用一个**得分网络**计算其"相关程度"：

![ScoreNetwork.png](https://img.yumeko.site/file/articles/AttentionMechanism/ScoreNetwork.webp)

$$
\text{score}(h_i) = v^T \cdot \tanh(W \cdot h_i + b)
$$

- $W \in \mathbb{R}^{d_a \times d_h}$ — 将 $d_h$ 维隐藏状态映射到 $d_a$ 维注意力空间（如 $d_a = 256$，$d_h = 512$）
- $b \in \mathbb{R}^{d_a}$ — 偏置向量
- $v \in \mathbb{R}^{d_a}$ — 可学习的得分向量，将 $d_a$ 维中间表示压缩为一个标量
- $\text{score}(h_i) \in \mathbb{R}$ — 标量，正值表示"相关"，负值表示"不相关"

**为什么叫"加性"注意力？** 因为得分函数 $v^T \cdot \tanh(W \cdot h_i + b)$ 中没有 $h_i$ 和其他向量的乘法交互——得分完全由 $h_i$ 自身通过一个前馈网络产生（加性变换）。

相比之下，"乘性"注意力（Luong）使用 $h_i^T \cdot W \cdot h_{\text{query}}$，涉及两个向量的点积。

### 2.3 第 2 步：遮罩（Masking）

PAD（填充）位置不包含有效信息，不应参与注意力计算。在 Softmax 之前，将 PAD 位置的得分强制设为 $-\infty$（实际实现中用一个很大的负数如 $-10^9$）：

$$
\text{score}_{\text{masked}}(h_i) = \begin{cases}
v^T \cdot \tanh(W \cdot h_i + b) & \text{if } i \text{ 不是 PAD} \\
-10^9 & \text{if } i \text{ 是 PAD}
\end{cases}
$$

因为 $\exp(-10^9) \approx 0$，在 Softmax 之后，PAD 位置的权重几乎严格为 0。

### 2.4 第 3 步：Softmax 归一化

将得分转化为概率分布（所有权重之和 = 1）：

$$
\alpha_i = \frac{e^{\text{score}_{\text{masked}}(h_i)}}{\sum_{j=1}^{T} e^{\text{score}_{\text{masked}}(h_j)}}
$$

- $\alpha_i \in (0, 1)$
- $\sum_{i=1}^{T} \alpha_i = 1$
- PAD 位置的 $\alpha_i \approx 0$

### 2.5 第 4 步：加权求和

用注意力权重对所有隐藏状态做加权平均：

$$
c = \sum_{i=1}^{T} \alpha_i \cdot h_i
$$

$c \in \mathbb{R}^{d_h}$ 与每个 $h_i$ 同维度（如 512），但它是一个**动态的**表示——不同于只使用最后一个隐藏状态的固定表示，$c$ 聚合了所有时间步的信息，并且重点关注得分高的位置。

---

## 3. EmotionClassification 的实现详解

### 3.1 Attention 类结构

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

- `self.score_network(rnn_output)` 实现了 $v^T \cdot \tanh(W \cdot h_i + b)$——`nn.Sequential` 将 `Linear → Tanh → Linear` 串联，等价于论文中的得分网络。
- `masked_fill(mask == 0, -1e9)` 实现了 PAD 位置的遮罩。
- `F.softmax(scores, dim=1)` 沿序列维度做 Softmax 归一化。
- `torch.bmm(attention_weights.unsqueeze(1), rnn_output)` 实现了加权求和 $\sum_i \alpha_i \cdot h_i$——利用批量矩阵乘法一次性完成所有样本的加权求和。

### 3.2 得分网络的维度变化

以 EmotionClassification 的配置（$d_h = 512$，$d_a = 256$）为例。得分网络 `nn.Sequential` 内部包含三层：

- 第一层 `Linear(512, 256)`：$W \in \mathbb{R}^{256 \times 512}$，$b \in \mathbb{R}^{256}$，输出形状从 `(B, L, 512)` 变为 `(B, L, 256)`
- 第二层 `Tanh`：逐元素变换，形状不变 `(B, L, 256)`
- 第三层 `Linear(256, 1, bias=False)`：$v^T$ 的角色，将 256 维压缩为 1 个标量，输出 `(B, L, 1)`，squeeze 后得到 `(B, L)`

得分网络的参数量：

| 层 | 操作 | 参数量 |
|------|------|------:|
| 第一层 | Linear(512→256) | 512×256 + 256 = 131,328 |
| 第三层 | Linear(256→1, bias=False) | 256×1 = 256 |
| **合计** | | **131,584** |

约 13 万参数——相比 LSTM 层的数百万参数，Attention 的开销很小。

### 3.3 Mask 的必要性

没有 Mask 的情况下，PAD 位置的隐藏状态虽然是零（因为 `padding_idx=0` 的嵌入是零向量），但经 RNN 后变为接近零的小值，Softmax 仍会为其分配非零权重——这会稀释有效位置的真实信息。

假设一个样本有 5 个有效词 + 123 个 PAD，若不加 Mask，PAD 位置的权重之和可能高达 $\frac{123}{128} \approx 0.96$，即 96% 的注意力给了无意义的填充位。加了 Mask 后，PAD 位置的 Softmax 输出严格趋近于 0，100% 的注意力集中在有效词上。

---

## 4. Attention 的可解释性

### 4.1 注意力权重热力图

Attention 的一个重要特性是**可解释性**。权重 $\alpha$ 可以直接可视化为热力图，直观展示模型的关注分布。

对于评论"物流很快包装也很好"的注意力权重：

| 词 | 物流 | 很快 | 包装 | 也 | 很好 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 注意力权重 | 0.05 | **0.35** | 0.08 | 0.02 | **0.50** |

约 85% 的注意力权重集中在"很快"（0.35）和"很好"（0.50）两个正面词上——模型确实学会了关注情感相关词汇。

### 4.2 不同样本的注意力模式

训练好的 Attention 会展现出合理的模式：对于正面评论，注意力集中在正面情感词上（如"非常""好""满意"）；对于负面评论，注意力则集中在负面情感词上（如"坏了""太差"）。这种可解释性在实际应用中很有价值——你可以告诉用户"系统判断这条评论是正面/负面的原因是因为它重点关注了这些词"。

---

## 5. Bahdanau vs Luong Attention

| 维度 | Bahdanau（加性） | Luong（乘性） |
|------|------|------|
| 年份 | 2015 | 2015 |
| 得分函数 | $v^T \tanh(W \cdot h_i)$ | $h_i^T W \cdot h_{\text{query}}$ |
| 计算方式 | 前馈网络独立给每个 $h_i$ 打分 | 每个 $h_i$ 与查询向量的点积 |
| 参数量 | 较多（额外的 $W$ 和 $v$） | 较少（只有 $W$，或直接点积无参数） |
| 是否需要 query | 否（内建于得分网络） | 是（需要解码器状态作为 query） |
| EmotionClassification | ✅ **使用此方式** | — |

选择 Bahdanau 的原因：EmotionClassification 是序列分类任务（非 Seq2Seq），没有"解码器的当前查询状态"，使用加性注意力更自然。实际上可以用一个可学习的查询向量或全局平均池化来适配 Luong，但加性注意力更直接。

---

## 6. Attention 的泛化意义

### 6.1 超越 Seq2Seq

虽然 Attention 最初为 Seq2Seq 设计，但它的核心思想——**对所有位置做动态加权**——适用于任何需要"聚合序列信息"的场景：

- **序列分类**：加权所有时间步的隐藏状态，替代最后一步输出的简单取用
- **阅读理解**：找出与问题最相关的文档片段
- **图像描述**：生成每个词时关注图像的不同区域

### 6.2 通向 Transformer

2017 年，Vaswani 等人提出了一个激进的想法：**完全放弃 RNN 的循环结构，纯粹依靠注意力机制**——这就是 Transformer。

Transformer 的 Self-Attention 允许序列中每个位置直接与所有其他位置交互，梯度路径长度从 RNN 的 $O(T)$ 缩短为 $O(1)$。这一设计解决了 RNN 的梯度消失和训练并行化等固有难题。

但理解 Bahdanau Attention 是理解 Self-Attention 和 Transformer 的前提——加权求和、Softmax 归一化、Masking 的概念一脉相承。

---

> 在 EmotionClassification 中的完整使用：[[Projects/EmotionClassification|EmotionClassification 项目]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
