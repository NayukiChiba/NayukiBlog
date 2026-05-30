---
title: Seq2Seq：编码器-解码器架构
date: 2026-05-29
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: 2014 年 Sutskever 等人提出的 Sequence-to-Sequence 学习框架，理解编码器-解码器结构如何实现变长序列到变长序列的映射，以及 Teacher Forcing 训练策略。
image: https://img.yumeko.site/file/blog/Seq2Seq.webp
status: draft
---

## 1. 问题定义：变长序列到变长序列

### 1.1 标准分类的局限

许多重要的 AI 问题本质上是**序列到序列**的映射：

- 机器翻译："How are you?"(3 词) → "你好吗？"(3 字，长度可能不同)
- 文本摘要：500 词文章 → 50 词摘要
- 对话系统：用户问题 → 系统回复

这些任务的特点是：**输入和输出都是序列，且长度通常不同**。标准的多对一（分类）或同步多对多（序列标注）RNN 无法直接处理。

### 1.2 Seq2Seq 的核心创新

2014 年，Sutskever、Vinyals 和 Le（Google）以及 Cho 等人（Montreal）同时提出了 Seq2Seq 框架。核心思想是将问题分解为两个阶段：

1. **编码（Encode）**：将输入序列压缩为一个固定长度的**上下文向量（Context Vector）**
2. **解码（Decode）**：从这个上下文向量**自回归地**生成输出序列

---

## 2. 编码器（Encoder）

### 2.1 结构

编码器是一个标准的 RNN（通常是 LSTM），逐时间步读取输入序列：

![Seq2Seq.png](https://img.yumeko.site/file/articles/Seq2Seq/Seq2Seq.webp)

### 2.2 上下文向量的提取

编码完成后，整个输入序列的信息被压缩为**上下文向量 $C$**：

- 最简单的方式：$C = h_T$（最后一个隐藏状态）
- 对于 LSTM：$C = (h_T, c_T)$（隐藏状态 + 细胞状态）

上下文向量的维度是固定的（如 512 维或 1024 维），无论输入序列有多长。

这就是 Seq2Seq 的**信息瓶颈**——一个固定长度的向量必须编码变长输入的所有语义信息。

---

## 3. 解码器（Decoder）

### 3.1 结构

解码器是另一个独立的 RNN（通常也是 LSTM），以上下文向量为初始状态，自回归地生成输出序列：

```
上下文 C = h_T^{enc}

解码:  s_0=C → s_1 → s_2 → ... → s_S
        ↓      ↓      ↓           ↓
输出:  y_1    y_2    y_3         y_S (<EOS>)
```

- $s_0$ = 编码器的最终隐藏状态 $h_T^{enc}$
- $y_1$ = 基于 $s_0$ 和 `<SOS>` token 生成
- $y_t$ = 基于 $s_{t-1}$ 和 $y_{t-1}$ 生成
- 直到生成 `<EOS>` 或达到最大长度

### 3.2 训练 vs 推理

![TeacherForcing.png](https://img.yumeko.site/file/articles/Seq2Seq/TeacherForcing.webp)

**训练时（Teacher Forcing）**：

$$
s_t = \text{RNN}_{\text{dec}}(s_{t-1}, y_{t-1}^{\text{true}})
$$

使用**真实的**前一个目标词作为当前步的输入，而不是模型的预测。这避免了错误预测的累积。

**推理时（自回归）**：

$$
s_t = \text{RNN}_{\text{dec}}(s_{t-1}, \hat{y}_{t-1})
$$

使用模型**上一次的预测**作为当前步的输入。这是实际的生成模式。

**Teacher Forcing 的利弊**：
- 利：训练更快更稳定（真实目标提供正确引导）
- 弊：训练和推理存在"曝光偏差"（训练时看到完美输入，推理时看到不完美的预测）

---

## 4. 上下文向量的信息瓶颈

### 4.1 长句翻译的性能退化

Seq2Seq 被提出后，研究者很快发现一个问题：**翻译质量随输入长度急剧下降**。

根本原因：固定长度的上下文向量 $C$ 无法编码长句的全部信息。512 维的向量要编码 50+ 词的所有语法结构、语义内容和词语对应关系，超出其容量上限。

![BlueScore.png](https://img.yumeko.site/file/articles/Seq2Seq/BlueScore.webp)

**实验数据**（Bahdanau 2015）：BLEU 分数随句子长度增加而下降。
- 短句（<10 词）：BLEU 30+
- 中句（10-20 词）：BLEU ~25
- 长句（30+ 词）：BLEU <20

### 4.2 信息瓶颈的后果

- 长句靠前位置的信息在上下文向量中被后来的词覆盖
- 多义词语的消歧依赖于上下文——但上下文在瓶颈处被丢失
- 输出序列中的早期词可能缺少与输入序列中对应词的直接联系

这个瓶颈直接催生了 **Attention** 机制：不再只依赖一个最终的上下文向量，而是让解码器可以"回顾"编码器的所有时间步。

---

## 5. 解码策略

### 5.1 贪心解码（Greedy Decoding）

每个时间步选择概率最高的词：

```python
for t in range(maxLen):
    logits = decoder(context, prevToken)
    probs = softmax(logits)
    nextToken = argmax(probs)        # 选概率最高的
    output.append(nextToken)
    if nextToken == EOS:
        break
    prevToken = nextToken
```

优点：速度快。缺点：一旦选错无法回头，可能生成不连贯的序列。

### 5.2 Beam Search

![BeamSearch.png](https://img.yumeko.site/file/articles/Seq2Seq/BeamSearch.webp)

维护 $k$（beam size）条最优候选序列，每步扩展并保留前 $k$ 条：

```
Step 1: [A(0.6), B(0.3), C(0.1)]          → 保留 A, B
Step 2: A→AA(0.2), A→AB(0.5), B→BA(0.4), B→BB(0.1)
                                        → 保留 AB(0.6×0.5=0.30), BA(0.3×0.4=0.12)
...
```

- Beam size = 1：等价于贪心解码
- Beam size = 3-10：常见设置，在质量和速度间平衡
- Beam size = 100+：搜索更全面但计算成本高

---

## 6. Seq2Seq 的历史意义

Seq2Seq 是 NLP 历史上具有重要影响的架构创新之一。它的意义超越了机器翻译：

- **统一了多种任务**：翻译、摘要、对话、问答等都可以用同一框架
- **建立了编码-解码范式**：编码-解码的范式影响至今
- **激发了 Attention**：信息瓶颈问题直接催生了注意力机制
- **为预训练铺路**：Seq2Seq 证明了 RNN 可以从大量平行语料中学习

> "Sequence to Sequence Learning with Neural Networks" 这篇论文被引用了超过 24000 次（截至 2025 年），是深度学习历史上被引最多的论文之一。

---

> 克服信息瓶颈的方案：[[NeuralNetwork/RNN/Architectures/AttentionRNN|注意力 RNN]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
