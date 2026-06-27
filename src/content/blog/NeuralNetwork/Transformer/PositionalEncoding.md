---
title: Transformer 位置编码
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - 位置编码
  - RoPE
description: 解释为什么 Transformer 需要位置编码，并对比 Sinusoidal、Learned、RoPE 与 ALiBi。
image: https://img.yumeko.site/file/blog/PositionalEncoding.png
status: published
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]。

![图0: Transformer 位置编码 Banner](https://img.yumeko.site/file/blog/PositionalEncoding.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于 Transformer 位置编码技术博客封面。
> 设计概念：展示 token embedding 与 position encoding 相加后进入注意力层。
> 左侧是 token 序列，中间是正弦位置编码热力图和旋转向量示意，右侧是带位置信息的 attention。
> 配色：深蓝到暖金渐变，现代数据科学美学风格。
> 简洁无衬线标签，淡色网格背景，顶部留白供标题叠加。
> ```

## 1. 为什么需要位置编码

Self-Attention 本身对顺序不敏感。若不加入位置信息，下面两个序列在 Attention 看来只是同一组 token 的不同排列：

| 序列 | 语义 |
|:--|:--|
| 我 喜欢 你 | “我”是动作发出者 |
| 你 喜欢 我 | “你”是动作发出者 |

词相同，但语义不同。因此 Transformer 必须把“第几个位置”编码进 token 表示。

输入表示通常写成：

$$
\boxed{
h_i^{(0)} = E(x_i) + P_i
}
$$

其中 $E(x_i)$ 是 token embedding，$P_i$ 是位置编码。

---

## 2. Sinusoidal 位置编码

原始 Transformer 使用固定的正弦余弦位置编码：

$$
PE_{pos,2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{pos,2i+1} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

特点：

| 特性 | 说明 |
|:--|:--|
| 不需要训练 | 位置向量由公式直接生成 |
| 每个维度频率不同 | 低维变化慢，高维变化快 |
| 可生成任意长度 | 不受训练参数表长度限制 |

局限是：模型训练时没见过很长的位置，虽然可以生成编码，但未必能可靠外推。

---

## 3. Learned 位置编码

Learned Position Embedding 将每个位置看成一个可学习向量：

$$
P_i = \operatorname{Embedding}(i)
$$

优点：

- 实现简单。
- 可以让模型自己学习任务相关的位置模式。

缺点：

- 最大长度固定。
- 超过训练长度后很难外推。

如果训练时最大长度是 512，那么位置 513 没有学过对应向量，直接扩展会比较麻烦。

---

## 4. RoPE：旋转位置编码

![图1: RoPE 将位置信息写入旋转角度](https://img.yumeko.site/file/blog/PositionalEncoding/RopeRotation.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的科学示意图，解释 RoPE 如何把位置编码为二维子空间中的旋转角度。
> 左侧画两个二维向量 q 和 k，右侧画它们分别按位置 m、n 旋转后的 q_m、k_n，并标注相对角度 n-m。
> 关键标注使用 LaTeX 风格：R_m q、R_n k、(R_m q)^T(R_n k)。
> 白色背景，细坐标轴线，柔和蓝白配色，教科书插图风格。
> ```

RoPE 不直接把位置向量加到 token embedding 上，而是把位置以旋转方式作用到 Query 和 Key 上。

对第 $i$ 对维度：

$$
\begin{bmatrix}
q'_{2i} \\
q'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
q_{2i} \\
q_{2i+1}
\end{bmatrix}
$$

其中 $m$ 是当前位置。

RoPE 的关键性质是：位置 $m$ 的 Query 和位置 $n$ 的 Key 做点积后，会自然包含相对位置 $m-n$。

$$
(R_m q)^\top (R_n k) = q^\top R_{n-m} k
$$

这使得 RoPE 同时具有绝对位置注入和相对位置建模的特点。

---

## 5. RoPE 代码示例

```python
import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将最后一维按两两配对旋转。"""
    x_pair = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_pair[..., 0]
    x2 = x_pair[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    应用 RoPE。

    Args:
        x: 形状 (batch, heads, seqLen, headDim)
        cos: 形状可广播到 x
        sin: 形状可广播到 x
    """
    return x * cos + rotate_half(x) * sin
```

实际实现中通常会提前缓存 `cos` 和 `sin`，避免每次前向重复计算。

---

## 6. ALiBi：Attention Linear Bias

ALiBi 不修改 token embedding，也不旋转 Q/K，而是直接给注意力分数加一个与距离相关的偏置：

$$
S_{ij}' = S_{ij} + m_h \cdot (i - j)
$$

其中 $m_h$ 是每个注意力头的斜率。

特点：

- 不需要位置 embedding 表。
- 对长序列外推更友好。
- 实现上直接作用于 attention score。

它的思想很朴素：距离越远，默认惩罚越大；模型仍然可以在需要时关注远处 token。

---

## 7. 常见方案对比

| 方案 | 类型 | 是否可学习 | 外推能力 | 常见用途 |
|:--|:--|:--|:--|:--|
| Sinusoidal | 绝对位置 | 否 | 一般 | 原始 Transformer |
| Learned | 绝对位置 | 是 | 较差 | 早期 BERT/GPT |
| RoPE | 旋转位置 | 否 | 较好 | 现代 Decoder-only 模型 |
| ALiBi | 注意力偏置 | 否 | 较好 | 长上下文实验 |

---

## 8. 常见坑

1. **以为 Attention 自动知道顺序。**  
   Attention 只看 token 表示之间的相似度，本身没有位置概念。

2. **把 RoPE 当成普通加法位置编码。**  
   RoPE 作用在 Q/K 上，不是简单加到输入 embedding 上。

3. **只看最大长度，不看训练长度。**  
   位置编码能生成更长位置，不代表模型一定学会处理更长上下文。

4. **忽略位置编码与 KV Cache 的配合。**  
   自回归推理时，新 token 的位置必须延续历史长度，否则位置会错位。

---

## 9. 总结

位置编码解决的是 Transformer 的顺序感问题。

如果按直觉区分：

| 方案 | 一句话理解 |
|:--|:--|
| Sinusoidal | 用固定波形表示位置 |
| Learned | 让模型自己学每个位置向量 |
| RoPE | 通过旋转让注意力分数感知相对距离 |
| ALiBi | 给远距离注意力加线性惩罚 |

理解位置编码后，再继续看 Transformer Block 中的残差、归一化和 FFN，会更容易把完整结构串起来。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
> - [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]
