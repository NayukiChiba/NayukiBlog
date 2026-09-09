---
title: Transformer 位置编码
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - 位置编码
  - RoPE
description: 解释为什么 Transformer 需要位置编码，并对比 Sinusoidal、Learned、RoPE 与 ALiBi。
image: https://img.yumeko.site/file/blog/cover/1782558815778_PositionalEncoding.webp
status: published
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]。

## 1. 为什么需要位置编码

Self-Attention 本身对顺序不敏感。若不加入位置信息，下面两个序列在 Attention 看来只是同一组 token 的不同排列：

| 序列 | 语义 |
|:--|:--|
| 我 喜欢 你 | "我"是动作发出者 |
| 你 喜欢 我 | "你"是动作发出者 |

词相同，但语义不同。从数学上看，Attention 输出 $\operatorname{softmax}(QK^\top)V$ 对输入 token 的排列是等变的（Permutation Equivariant）：交换输入 token 的位置，输出只是跟着交换，模型本身感知不到"顺序"。因此 Transformer 必须把"第几个位置"编码进 token 表示。

输入表示通常写成：

$$
\boxed{
h_i^{(0)} = E(x_i) + P_i
}
$$

其中 $E(x_i)$ 是 token embedding，$P_i$ 是位置编码。

> [!NOTE] 三种注入位置
> 位置信息并非只能加在输入 embedding 上：
> 1. **输入侧**：Sinusoidal、Learned，把 $P_i$ 直接加到 $E(x_i)$ 上；
> 2. **Q/K 侧**：RoPE，把位置作为旋转作用在 Query 和 Key 上；
> 3. **分数侧**：ALiBi，直接在注意力分数上加偏置。

---

## 2. Sinusoidal 位置编码

### 2.1 公式

原始 Transformer 使用固定的正弦余弦位置编码：

$$
\boxed{
PE_{pos,2i} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{pos,2i+1} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
}
$$

其中 $pos$ 是 token 位置，$i = 0, 1, \dots, d/2-1$ 是维度对索引，$d$ 是模型维度。偶数列用 sin，奇数列用 cos，每对维度共用一个频率。

### 2.2 频率与波长

第 $i$ 对维度对应的角频率为：

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

因此波长（完成一次完整振荡所需的位置数）为：

$$
\lambda_i = 2\pi \cdot 10000^{2i/d}
$$

以 $d = 512$ 为例，波长构成几何级数：从 $\lambda_0 = 2\pi \approx 6.3$ 一直到约 $2\pi \cdot 10000 \approx 62832$ 个位置。短波长刻画相邻 token 的局部差异，长波长刻画长距离的全局位置。基数 10000 是经验选择的超参数，控制这个几何级数的间隔。

| 特性 | 说明 |
|:--|:--|
| 不需要训练 | 位置向量由公式直接生成 |
| 每个维度频率不同 | 低维频率高、变化快；高维频率低、变化慢 |
| 可生成任意长度 | 不受训练参数表长度限制 |

局限是：模型训练时没见过很长的位置，虽然可以生成编码，但未必能可靠外推。

### 2.3 数值实例

取 $d = 8$，四对维度的频率分别为 $10000^{0/8}, 10000^{2/8}, 10000^{4/8}, 10000^{6/8} = 1, 10, 100, 1000$。以位置 1 为例手工算几个分量（角度单位是弧度）：

$$
PE_{1,0} = \sin\left(\frac{1}{1}\right) = \sin 1 = 0.8415, \qquad
PE_{1,1} = \cos 1 = 0.5403
$$

$$
PE_{1,2} = \sin\left(\frac{1}{10}\right) = \sin 0.1 = 0.0998, \qquad
PE_{1,3} = \cos 0.1 = 0.9950
$$

$$
PE_{1,4} = \sin 0.01 = 0.0100, \qquad
PE_{1,6} = \sin 0.001 = 0.0010
$$

位置 0、1、2 的完整结果：

| $pos$ | 维度 0 $\sin(pos)$ | 维度 1 $\cos(pos)$ | 维度 2 $\sin(pos/10)$ | 维度 3 $\cos(pos/10)$ | 维度 4 $\sin(pos/100)$ | 维度 5 $\cos(pos/100)$ | 维度 6 $\sin(pos/1000)$ | 维度 7 $\cos(pos/1000)$ |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0998 | 0.9950 | 0.0100 | 1.0000 | 0.0010 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.1987 | 0.9801 | 0.0200 | 0.9998 | 0.0020 | 1.0000 |

观察：位置从 0 走到 2，维度 0-1 变化剧烈（0.84 → 0.91、0.54 → −0.42），维度 6-7 几乎不动（0.0010 → 0.0020）。低维对应高频短波长，高维对应低频长波长，这与 2.2 节的结论一致。

### 2.4 相对位置可由线性变换得到

选择 sin/cos 成对出现，是因为它们有一个关键性质：对固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的**线性变换**。对第 $i$ 对维度，由和角公式：

$$
PE_{pos+k,2i} = \sin\big(\omega_i (pos+k)\big) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)
$$

$$
PE_{pos+k,2i+1} = \cos\big(\omega_i (pos+k)\big) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)
$$

代入 $PE_{pos}$ 的分量，写成矩阵形式：

$$
\boxed{
\begin{bmatrix}
PE_{pos+k,2i} \\
PE_{pos+k,2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}
\begin{bmatrix}
PE_{pos,2i} \\
PE_{pos,2i+1}
\end{bmatrix}
}
$$

这正是一个旋转矩阵——位置平移 $k$ 对应每对维度上的一次旋转。Attention 的 Q/K 打分是线性运算，理论上模型可以从 $PE_{pos}$ 线性地推出 $PE_{pos+k}$ 的信息，从而学到"相对位置"。这正是原始论文选择 sin/cos 的动机。

> [!TIP] 记忆技巧
> 旋转矩阵是理解 RoPE 的核心道具。Sinusoidal 只是"顺便"具有旋转结构，RoPE 则把这个旋转显式用到了 Q/K 上。

### 2.5 NumPy 实现

```python
import numpy as np


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    生成 Sinusoidal 位置编码矩阵

    Args:
        seq_len: 序列长度
        d_model: 模型隐藏维度
    Returns:
        形状 (seq_len, d_model) 的位置编码矩阵
    """
    pos = np.arange(seq_len)[:, None]                  # (seq_len, 1)
    dim = np.arange(d_model)[None, :]                  # (1, d_model)
    # 维度 1 与 0 同频、3 与 2 同频，因此用 dim // 2 取频率索引
    angles = pos / np.power(10000, 2 * (dim // 2) / d_model)
    pe = np.empty((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe
```

---

## 3. Learned 位置编码

Learned Position Embedding 将每个位置看成一个可学习向量：

$$
P_i = \operatorname{Embedding}(i)
$$

实现上就是一张普通的 Embedding 表：

```python
import torch.nn as nn

pos_embedding = nn.Embedding(num_embeddings=512, embedding_dim=768)
```

优点：

- 实现简单。
- 可以让模型自己学习任务相关的位置模式。

缺点：

- 最大长度固定。
- 超过训练长度后很难外推。

如果训练时最大长度是 512，那么位置 513 没有学过对应向量，直接扩展会比较麻烦（要么截断，要么用插值补齐）。

---

## 4. RoPE：旋转位置编码

![RopeRotation.png](https://img.yumeko.site/file/blog/articles/1782559140604_RopeRotation.webp)

### 4.1 动机

加法位置编码把位置信号只注入到输入 embedding，经过多层变换后信号会逐渐衰减。RoPE 不把位置向量加到 token embedding 上，而是把位置以旋转方式作用到每一层的 Query 和 Key 上，让位置信号在每一层注意力中都直接生效。

### 4.2 二维旋转基础

对二维向量 $(x, y)$ 旋转角度 $\alpha$：

$$
\boxed{
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
}
$$

等价于在复平面上乘以 $e^{i\alpha}$：$x + iy \to e^{i\alpha}(x + iy)$。旋转不改变向量模长，只改变方向。旋转矩阵记作 $R(\alpha)$，满足 $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$（转置即取反角，两个旋转复合即角度相加）。

### 4.3 频率构造

RoPE 沿用 Sinusoidal 的频率表，第 $i$ 对维度的基频为：

$$
\theta_i = 10000^{-2i/d}
$$

位置 $m$ 的旋转角是 $m\theta_i$。整个 Q 向量按维度对做分块对角旋转：

$$
R_m = \operatorname{diag}\big(R(m\theta_0), R(m\theta_1), \dots, R(m\theta_{d/2-1})\big)
$$

即：

$$
q_m = R_m q, \qquad k_n = R_n k
$$

### 4.4 相对位置性质推导

位置 $m$ 的 Query 与位置 $n$ 的 Key 做点积：

$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

$$
\boxed{(R_m q)^\top (R_n k) = q^\top R_{n-m} k}
$$

点积结果只与相对位置 $n - m$ 有关，与绝对位置 $m$、$n$ 无关。这使得 RoPE 同时具有绝对位置注入（旋转角 $m\theta_i$ 本身编码绝对位置）和相对位置建模（分数依赖 $n-m$）的特点。

> [!TIP] 关键直觉
> 两个向量各自旋转后做点积，结果取决于**角度差**。RoPE 把"相对距离"变成了"相对角度"，点积中天然包含 $\cos(\theta_i(n-m))$ 这样的相对位置项。

### 4.5 数值实例

取 $d = 4$（两对维度），$\theta_0 = 1$，$\theta_1 = 10000^{-2/4} = 0.01$。设内容向量 $q = k = [1, 0, 1, 0]$，分别放在位置 $m = 1$ 和 $n = 2$。

对 $q$（位置 1），两对维度的旋转角分别是 $1 \times 1 = 1$ 和 $1 \times 0.01 = 0.01$：

$$
q_1 = [\cos 1,\ \sin 1,\ \cos 0.01,\ \sin 0.01] = [0.5403,\ 0.8415,\ 0.99995,\ 0.01000]
$$

对 $k$（位置 2），旋转角分别是 $2$ 和 $0.02$：

$$
k_2 = [\cos 2,\ \sin 2,\ \cos 0.02,\ \sin 0.02] = [-0.4161,\ 0.9093,\ 0.99980,\ 0.02000]
$$

点积逐对计算（用 $\cos a \cos b + \sin a \sin b = \cos(a-b)$）：

$$
q_1^\top k_2 = \cos(1 - 2) + \cos(0.01 - 0.02) = \cos 1 + \cos 0.01 = 0.5403 + 0.99995 \approx 1.5403
$$

再验证"只依赖相对位置"：同一对 $q$、$k$ 换到不同的绝对位置，点积不变：

| $m$ | $n$ | $n-m$ | $q_m^\top k_n$ | 计算 |
|:--|:--|:--|:--|:--|
| 1 | 1 | 0 | 2.0000 | $\cos 0 + \cos 0$ |
| 1 | 2 | 1 | 1.5403 | $\cos 1 + \cos 0.01$ |
| 1 | 3 | 2 | 0.5837 | $\cos 2 + \cos 0.02$ |
| 5 | 6 | 1 | 1.5403 | 与 (1, 2) 相同 |
| 100 | 101 | 1 | 1.5403 | 与 (1, 2) 相同 |

:::fold[一般情形的完整展开]

$$
q_m^\top k_n = \sum_{i=0}^{d/2-1} \Big[ \cos\big(\theta_i (n-m)\big) \big(q_{2i} k_{2i} + q_{2i+1} k_{2i+1}\big) + \sin\big(\theta_i (n-m)\big) \big(q_{2i} k_{2i+1} - q_{2i+1} k_{2i}\big) \Big]
$$

本例中 $q_{2i+1} = k_{2i+1} = 0$，sin 项全部消去，只剩 $\cos\theta_0(n-m) + \cos\theta_1(n-m)$。

:::

### 4.6 PyTorch 实现

```python
import torch


def precompute_rope_freqs(dim: int, seq_len: int, base: float = 10000.0) -> tuple:
    """
    预计算 RoPE 的 cos / sin 缓存

    Args:
        dim: 每个注意力头的维度
        seq_len: 最大序列长度
        base: 频率基数
    Returns:
        (cos, sin)，形状均为 (seq_len, dim)
    """
    pair_idx = torch.arange(0, dim, 2).float()      # 维度对索引 0, 2, 4, ...
    theta = base ** (-pair_idx / dim)               # 形状 (dim/2,)
    pos = torch.arange(seq_len).float()             # 形状 (seq_len,)
    angles = torch.outer(pos, theta)                # (seq_len, dim/2)
    # 同一对维度的两个分量共用同一角度
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


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
        x: 任意形状张量，沿最后一维按维度对旋转
        cos: 形状可广播到 x
        sin: 形状可广播到 x
    """
    return x * cos + rotate_half(x) * sin
```

实际实现中通常会提前缓存 `cos` 和 `sin`，避免每次前向重复计算；推理时按 KV Cache 的长度索引即可。

下面的代码用 4.5 节的数值验证相对位置性质：

```python
def verify_relative_position():
    """验证 RoPE 点积只依赖相对位置。"""
    dim = 4
    cos, sin = precompute_rope_freqs(dim, seq_len=1024)
    q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])   # (1, 1, 1, 4)
    k = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    for m, n in [(1, 2), (5, 6), (100, 101)]:
        q_m = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        k_n = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        score = (q_m * k_n).sum()
        print(f"m={m}, n={n}: q^T k = {score.item():.4f}")
```

三组 $(m, n)$ 的输出均为 `1.5403`，与 4.5 节的手工计算结果一致。

---

## 5. ALiBi：Attention Linear Bias

### 5.1 公式

ALiBi 不修改 token embedding，也不旋转 Q/K，而是直接给注意力分数加一个与距离相关的偏置：

$$
\boxed{
S'_{ij} = S_{ij} + m_h \cdot (j - i)
}
$$

其中 $i$ 是 Query 位置、$j$ 是 Key 位置，$m_h > 0$ 是第 $h$ 个注意力头的斜率。对因果注意力（$j \le i$），$j - i \le 0$：Key 离得越远，偏置越负，注意力分数被打压得越狠。对双向注意力，$j > i$ 一侧同样被惩罚，偏置对称。

斜率按几何级数分配给每个头：

$$
m_h = 2^{-8h/H}, \qquad h = 1, 2, \dots, H
$$

其中 $H$ 是总头数。

特点：

- 不需要位置 embedding 表。
- 对长序列外推更友好：外推只涉及线性偏置，没有训练时未见过的位置参数。
- 实现上直接作用于 attention score。

它的思想很朴素：距离越远，默认惩罚越大；模型仍然可以在需要时关注远处 token。不同头的斜率不同，相当于为每个头预设了不同的"关注半径"。

### 5.2 数值实例

取 $H = 8$，各头斜率：

| 头 $h$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| $m_h$ | 0.5 | 0.25 | 0.125 | 0.0625 | 0.03125 | 0.015625 | 0.0078125 | 0.00390625 |

对头 1（$m_1 = 0.5$），5 个 token 的偏置矩阵（行是 Query $i$，列是 Key $j$；实际使用中 $j > i$ 的上三角会被因果掩码覆盖，这里填 0 仅为展示）：

$$
B_1 =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
-0.5 & 0 & 0 & 0 & 0 \\
-1.0 & -0.5 & 0 & 0 & 0 \\
-1.5 & -1.0 & -0.5 & 0 & 0 \\
-2.0 & -1.5 & -1.0 & -0.5 & 0
\end{bmatrix}
$$

含义解读：位置 4 的 Query 看位置 2 的 Key（距离 2），偏置为 $-1.0$。若原注意力分数为 $S$，softmax 中的权重正比于 $e^{S - 1.0} \approx 0.368 \cdot e^{S}$，相当于打 37 折；距离 1 的 Key 打 $e^{-0.5} \approx 0.607$ 折。

头 8 的斜率只有 $1/256 \approx 0.0039$，距离 1 的惩罚为 $e^{-0.0039} \approx 0.996$，几乎无衰减——这个头可以放心看远处，各头分工不同。

### 5.3 NumPy 实现

```python
import numpy as np


def alibi_bias(num_heads: int, seq_len: int) -> np.ndarray:
    """
    生成 ALiBi 注意力偏置矩阵

    Args:
        num_heads: 注意力头数
        seq_len: 序列长度
    Returns:
        形状 (num_heads, seq_len, seq_len) 的偏置矩阵，
        bias[h, i, j] = m_h * (j - i)
    """
    # 各头斜率按几何级数衰减
    slopes = 2.0 ** (-8.0 * np.arange(1, num_heads + 1) / num_heads)
    # 距离矩阵：distance[i, j] = j - i
    idx = np.arange(seq_len)
    distance = idx[None, :] - idx[:, None]          # (seq_len, seq_len)
    bias = slopes[:, None, None] * distance[None, :, :]
    return bias
```

---

## 6. 常见方案对比

| 方案 | 类型 | 作用位置 | 是否可学习 | 外推能力 | 常见用途 |
|:--|:--|:--|:--|:--|:--|
| Sinusoidal | 绝对位置 | 输入 embedding | 否 | 一般 | 原始 Transformer |
| Learned | 绝对位置 | 输入 embedding | 是 | 较差 | 早期 BERT/GPT |
| RoPE | 旋转位置 | Q/K | 否 | 较好 | 现代 Decoder-only 模型 |
| ALiBi | 注意力偏置 | 注意力分数 | 否 | 较好 | 长上下文实验 |

---

## 7. 常见坑

1. **以为 Attention 自动知道顺序。**  
   Attention 只看 token 表示之间的相似度，本身没有位置概念。

2. **把 RoPE 当成普通加法位置编码。**  
   RoPE 作用在 Q/K 上，不是简单加到输入 embedding 上。

3. **只看最大长度，不看训练长度。**  
   位置编码能生成更长位置，不代表模型一定学会处理更长上下文。

4. **忽略位置编码与 KV Cache 的配合。**  
   自回归推理时，新 token 的位置必须延续历史长度，否则位置会错位。

5. **打乱 RoPE 的维度配对。**  
   RoPE 必须按相邻两维（$2i$ 与 $2i+1$）成对旋转，配对顺序错了会破坏相对位置性质。

---

## 8. 总结

位置编码解决的是 Transformer 的顺序感问题。

如果按直觉区分：

| 方案 | 一句话理解 |
|:--|:--|
| Sinusoidal | 用固定波形表示位置，位置平移等价于维度对上的旋转 |
| Learned | 让模型自己学每个位置向量 |
| RoPE | 通过旋转让注意力分数天然感知相对距离 |
| ALiBi | 给远距离注意力加线性惩罚 |

各方案的数学联系：Sinusoidal 与 RoPE 共用同一张频率表 $\theta_i = 10000^{-2i/d}$，前者把位置当作输入角度，后者把位置当作 Q/K 的旋转角；ALiBi 则绕开 embedding 与 Q/K，直接修正注意力分数。理解位置编码后，再继续看 Transformer Block 中的残差、归一化和 FFN，会更容易把完整结构串起来。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
> - [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]
> - [[NeuralNetwork/Overview/TransformerOverview|Transformer 全景概览]]
