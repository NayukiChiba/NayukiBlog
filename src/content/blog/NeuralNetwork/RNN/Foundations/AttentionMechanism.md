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

在标准的 Seq2Seq 或序列分类 RNN 中，整个输入序列的信息最终被压缩到一个固定维度的向量中：

```
输入序列: [x_1, x_2, x_3, ..., x_100]
            ↓    ↓    ↓         ↓
          [RNN 正向 + RNN 反向]
                    ↓
          最终表示 h_final: (512,)  ← 512维向量要承载100个词的全部信息
                    ↓
              [分类器/解码器]
```

**无论输入有多长（10 词 or 500 词），最终表示都是固定 512 维。** 固定维度的向量无法完整编码长序列的全部信息，信息丢失是结构性的。这就是**信息瓶颈（Information Bottleneck）**。

![InformationBottleneck.png](https://img.yumeko.site/file/articles/AttentionMechanism/InformationBottleneck.webp)

### 1.2 信息瓶颈的具体表现

以 Seq2Seq 机器翻译为例，"编码器的最后一个隐藏状态 → 解码器"是唯一的信息传递通道：

- 短句（5-10 词）：512 维向量足够编码
- 中句（20-30 词）：开始丢失细节
- 长句（50+ 词）：缺失关键信息，翻译质量急剧下降

2014 年 Bahdanau 等人的实验表明，标准 Seq2Seq 在超过 30 词的句子上性能明显退化。

### 1.3 Attention 的核心洞察

与其把所有信息压缩到一个向量，不如让解码器/分类器直接访问编码器的所有输出，并通过注意力权重动态确定各位置的重要性。

这就是 Attention——一种**动态加权机制**，让模型能够为输入序列的每个位置分配一个重要性权重。

---

## 2. Bahdanau 加性注意力：完整数学流程

### 2.1 四步流程总览


编码器所有隐藏状态: h_1, h_2, h_3, ..., h_T      (T个，每个hidden_dim维)

![Attention.png](https://img.yumeko.site/file/articles/AttentionMechanism/Attention.webp)

第1步: 评分(Scoring)  →  score_1, score_2, ..., score_T   (T个标量)
第2步: 遮罩(Masking)  →  将PAD位置的得分设为 -inf
第3步: Softmax归一化  →  α_1, α_2, ..., α_T              (权重，Σα=1)
第4步: 加权求和       →  c = Σ α_i · h_i                  (上下文向量)


下面逐步拆解。

### 2.2 第 1 步：评分（Scoring）

对每个时间步 $i$ 的隐藏状态 $h_i$，用一个**得分网络**计算其"相关程度"：

![ScoreNetwork.png](https://img.yumeko.site/file/articles/AttentionMechanism/ScoreNetwork.webp)
$$
\text{score}(h_i) = v^T \cdot \tanh(W \cdot h_i + b)
$$

- $W$：形状 (attn_dim, hidden_dim) — 如 (256, 512)，将隐藏状态映射到注意力空间
- $b$：偏置，形状 (attn_dim,)
- $v$：形状 (attn_dim,) — 可学习的得分向量，将 256 维的中间表示压缩为 1 个标量得分
- $\text{score}(h_i)$：标量，正值表示"相关"，负值表示"不相关"

**为什么叫"加性"注意力？** 因为 $v^T \cdot \tanh(W \cdot h_i + b)$ 中没有 $h_i$ 和任何其他向量的乘法交互——得分完全由 $h_i$ 自身通过一个前馈网络产生（加性变换）。

相比之下，"乘性"注意力（Luong）使用 $h_i^T \cdot W \cdot h_{\text{query}}$，涉及两个向量的点积。

### 2.3 第 2 步：遮罩（Masking）

PAD（填充）位置不包含有效信息，不应参与注意力计算。在 Softmax 之前，将 PAD 位置的得分强制设为 $-\infty$（实际实现中用一个很大的负数如 $-10^9$）：

$$
\text{score}_{\text{masked}}(h_i) = \begin{cases}
v^T \cdot \tanh(W \cdot h_i + b) & \text{if } i \text{ is NOT PAD} \\
-10^9 & \text{if } i \text{ is PAD}
\end{cases}
$$

因为 $e^{-10^9} \approx 0$，在 Softmax 之后，PAD 位置的权重几乎严格为 0。

### 2.4 第 3 步：Softmax 归一化

将得分转化为概率分布（所有权重之和 = 1）：

$$
\alpha_i = \frac{e^{\text{score}_{\text{masked}}(h_i)}}{\sum_{j=1}^{T} e^{\text{score}_{\text{masked}}(h_j)}}
$$

- $\alpha_i \in (0, 1)$
- $\sum_i \alpha_i = 1$
- PAD 位置的 $\alpha_i \approx 0$

### 2.5 第 4 步：加权求和

用注意力权重对所有隐藏状态做加权平均：

$$
c = \sum_{i=1}^{T} \alpha_i \cdot h_i
$$

$c$ 的形状与 $h_i$ 相同（如 512 维），但它是一个**动态的**表示——不同于只使用最后一个隐藏状态的固定表示，$c$ 聚合了所有时间步的信息，并且重点关注得分高的位置。

---

## 3. EmotionClassification 的实现详解

### 3.1 Attention 类结构

```python
class Attention(nn.Module):
    """Bahdanau 加性注意力"""
    def __init__(self, hiddenSize):
        super().__init__()
        # W: (hiddenSize, hiddenSize//2) — 将隐藏状态降维到注意力空间
        self.attn = nn.Linear(hiddenSize, hiddenSize // 2)
        # v^T: (hiddenSize//2, 1) — 产生标量得分
        self.v = nn.Linear(hiddenSize // 2, 1, bias=False)

    def forward(self, hidden, mask):
        """
        hidden: (batch, seq_len, hiddenSize) — 双向LSTM输出=512维
        mask: (batch, seq_len) — 1=有效, 0=PAD
        """
        # Step 1: 评分
        # attn(hidden): (B, L, 256) — Linear 对最后一维操作
        # v(attn(hidden)): (B, L, 1)
        scores = self.v(torch.tanh(self.attn(hidden)))  # (B, L, 1)
        scores = scores.squeeze(-1)                      # (B, L)

        # Step 2: Mask
        scores = scores.masked_fill(mask == 0, -1e9)

        # Step 3: Softmax
        attnWeights = torch.softmax(scores, dim=-1)      # (B, L)

        # Step 4: 加权求和
        # attnWeights.unsqueeze(1): (B, 1, L)
        # hidden: (B, L, 512)
        # bmm 结果: (B, 1, 512) → squeeze → (B, 512)
        context = torch.bmm(
            attnWeights.unsqueeze(1), hidden
        ).squeeze(1)                                     # (B, 512)

        return context, attnWeights
```

### 3.2 得分网络的维度变化

以 EmotionClassification 的配置（hiddenSize=512, attn_dim=256）：

```
hidden:    (B, L, 512)
    ↓ self.attn (Linear(512 → 256))
    (B, L, 256)
    ↓ tanh
    (B, L, 256)
    ↓ self.v (Linear(256 → 1, bias=False))
    (B, L, 1)  → squeeze → (B, L)  ← scores
```

得分网络的参数量计算：

| 层 | 操作 | 参数量 |
|------|------|------:|
| self.attn | Linear(512→256) | 512×256 + 256 = 131,328 |
| self.v | Linear(256→1) | 256×1 = 256 |
| **合计** | | **131,584** |

约 13 万参数——相比 LSTM 层的数百万参数，Attention 的开销很小。

### 3.3 Mask 的必要性

没有 Mask 的情况下，PAD 位置的隐藏状态虽然是零（因为 padding_idx=0 的嵌入是零向量，经 RNN 后变成小值），但仍然会参与加权求和，稀释有效位置的真实信息。

```python
# 错误做法（无Mask）
scores = self.v(torch.tanh(self.attn(hidden)))
attnWeights = torch.softmax(scores, dim=-1)
# 问题：PAD位置也有非零权重！
# 例如5个有效词 + 123个PAD → 约1/128的权重给了有效词，其余给了PAD

# 正确做法（有Mask）
scores = self.v(torch.tanh(self.attn(hidden)))
scores = scores.masked_fill(mask == 0, -1e9)  # PAD得分→-inf
attnWeights = torch.softmax(scores, dim=-1)
# 效果：PAD位置的权重≈0，100%的注意力集中在有效词上
```

---

## 4. Attention 的可解释性

### 4.1 注意力权重热力图

Attention 的一个重要特性是**可解释性**。权重 $\alpha$ 可以直接可视化为热力图，直观展示模型的关注分布。

对于评论"物流很快包装也很好"的注意力权重：

| 词 | 物流 | 很快 | 包装 | 也 | 很好 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 注意力权重 | 0.05 | **0.35** | 0.08 | 0.02 | **0.50** |

约 85% 的注意力权重集中在"很快"和"很好"两个正面词上。

### 4.2 不同样本的注意力模式

训练好的 Attention 会展现出合理的模式：

```
正面评论："质量非常好，很满意"
注意力:    [0.10, 0.50, 0.05, 0.35]
             ↑               ↑
          非常、好         很、满意  ← 集中在正面词

负面评论："用了两天就坏了，太差了"
注意力:    [0.05, 0.05, 0.45, 0.05, 0.40]
                            ↑           ↑
                          坏了        太差了  ← 集中在负面词
```

这种可解释性在实际应用中很有价值——你可以告诉用户"系统判断这条评论是正面/负面的原因是因为它关注了这些词"。

---

## 5. Bahdanau vs Luong Attention

| 维度 | Bahdanau（加性） | Luong（乘性） |
|------|------|------|
| 年份 | 2015 | 2015 |
| 得分函数 | $v^T \tanh(W \cdot h_i)$ | $h_i^T W \cdot h_{\text{query}}$ |
| 计算方式 | 前馈网络独立给每个 $h_i$ 打分 | 每个 $h_i$ 与查询向量的点积 |
| 参数量 | 较多（额外的 W 和 v） | 较少（只有 W，或直接点积无参数） |
| 是否需要 query | 否（内建于得分网络） | 是（需要解码器状态作为 query） |
| EmotionClassification | ✅ **使用此方式** | — |

选择 Bahdanau 的原因：EmotionClassification 是序列分类任务（非 Seq2Seq），没有"解码器的当前查询状态"，使用加性注意力更自然。（实际上，可以用一个可学习的查询向量或直接用全局平均池化来适配 Luong，但加性注意力更直接。）

---

## 6. Attention 的泛化意义

### 6.1 超越 Seq2Seq

虽然 Attention 最初为 Seq2Seq 设计，但它的核心思想——**对所有位置做动态加权**——适用于任何需要"聚合序列信息"的场景：

- 序列分类：加权所有时间步的隐藏状态
- 阅读理解：找出与问题最相关的文档片段
- 图像描述：生成每个词时关注图像的不同区域

### 6.2 通向 Transformer

2017 年，Vaswani 等人提出了一个激进的想法：**完全放弃 RNN 的循环结构，纯粹依靠注意力机制**——这就是 Transformer。

Transformer 的 Self-Attention 允许序列中每个位置直接与所有其他位置交互，梯度路径长度从 RNN 的 $O(T)$ 缩短为 $O(1)$。这一设计解决了 RNN 的梯度消失和训练并行化等固有难题。

但理解 Bahdanau Attention 是理解 Self-Attention 和 Transformer 的前提——加权求和、Softmax 归一化、Masking 的概念一脉相承。

---

> 在 EmotionClassification 中的完整使用：[[Projects/EmotionClassification|EmotionClassification 项目]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
