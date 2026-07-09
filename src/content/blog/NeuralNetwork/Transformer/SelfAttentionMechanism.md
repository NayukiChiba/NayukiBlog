---
title: Self-Attention 机制详解
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - Self-Attention
  - 注意力机制
description: 从 Q、K、V 的含义出发，拆解缩放点积注意力、多头注意力、mask 与张量形状。
image: https://img.yumeko.site/file/blog/cover/1782558454107_SelfAttentionMechanism.webp
status: published
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Overview/TransformerOverview|Transformer 架构总览]]，理解 Transformer Block 的整体位置。

## 1. Self-Attention 在做什么

Self-Attention 的目标是：为序列中每个 token 重新计算一个上下文表示。

假设一句话有 $n$ 个 token：

$$
X = [x_1, x_2, \dots, x_n]
$$

第 $i$ 个 token 的新表示不是只由 $x_i$ 决定，而是由所有 token 加权求和得到：

$$
\boxed{
z_i = \sum_{j=1}^{n} \alpha_{ij} v_j
}
$$

其中 $\alpha_{ij}$ 是注意力权重，表示第 $i$ 个 token 应该从第 $j$ 个 token 读取多少信息。

---

## 2. Q、K、V 的含义

![QKVAttention.png](https://img.yumeko.site/file/blog/articles/1782558541565_QKVAttention.webp)

Self-Attention 会把输入 $X$ 投影成三组向量：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

可以这样理解：

| 向量 | 中文含义 | 直觉 |
|:--|:--|:--|
| Query | 查询 | 当前 token 想找什么 |
| Key | 索引 | 每个 token 提供什么匹配线索 |
| Value | 内容 | 匹配后真正读取的信息 |

注意力分数来自 Query 和 Key 的点积：

$$
S = QK^\top
$$

如果某个 Query 与某个 Key 方向接近，点积就大，说明当前 token 更应该关注那个位置。

---

## 3. 为什么要除以 $\sqrt{d_k}$

缩放点积注意力的公式是：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
}
$$

如果 $d_k$ 很大，点积 $QK^\top$ 的数值方差会变大，Softmax 容易进入饱和区。分数差异过大时，Softmax 会接近 one-hot，梯度变小，训练更不稳定。

除以 $\sqrt{d_k}$ 的作用是把注意力分数拉回更稳定的范围。

---

## 4. 张量形状

以批量输入为例：

| 张量 | 形状 |
|:--|:--|
| `x` | `(batch, seqLen, dModel)` |
| `q` | `(batch, seqLen, dModel)` |
| `k` | `(batch, seqLen, dModel)` |
| `v` | `(batch, seqLen, dModel)` |
| `scores` | `(batch, seqLen, seqLen)` |
| `attn` | `(batch, seqLen, seqLen)` |
| `output` | `(batch, seqLen, dModel)` |

单头注意力的计算过程：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / (dK ** 0.5)
attn = torch.softmax(scores, dim=-1)
output = attn @ v
```

Softmax 必须沿最后一维做，因为最后一维表示“当前 token 对所有 key 位置的分布”。

---

## 5. Mask 的作用

### 5.1 Padding Mask

批量训练时，不同样本长度不同，短句会用 `<PAD>` 补齐。Padding Mask 用来避免模型关注补齐位置。

| 位置类型 | 是否可见 |
|:--|:--|
| 真实 token | 可见 |
| `<PAD>` token | 不可见 |

### 5.2 Causal Mask

自回归生成中，位置 $t$ 不能看到未来 token：

$$
M_{i,j} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

加到注意力分数后，未来位置经过 Softmax 会变成接近 0 的权重。

---

## 6. Multi-Head Attention

多头注意力不是重复计算同一个注意力，而是把模型维度拆成多个子空间。

若：

$$
d_{\text{model}} = 512,\quad h = 8
$$

则每个头的维度是：

$$
d_{\text{head}} = 512 / 8 = 64
$$

多头注意力可以写成：

$$
\operatorname{MHA}(X)
= \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)W_O
$$

每个头独立计算：

$$
\operatorname{head}_i
= \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})
$$

直觉上，不同头可以关注不同关系：

| 头 | 可能关注 |
|:--|:--|
| Head 1 | 相邻词 |
| Head 2 | 主谓关系 |
| Head 3 | 长距离依赖 |
| Head 4 | 标点或边界 |

这不是硬编码规则，而是训练中自动学到的分工。

---

## 7. PyTorch 实现

```python
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头自注意力层。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch, seqLen, dModel)
            mask: 可选掩码，形状可广播到 (batch, numHeads, seqLen, seqLen)

        Returns:
            torch.Tensor: 输出张量，形状 (batch, seqLen, dModel)
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.d_head ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)
```

---

## 8. 常见坑

1. **Softmax 维度写错。**  
   应该沿 key 维度归一化，也就是最后一维。

2. **忘记缩放。**  
   不除以 $\sqrt{d_k}$ 会让大维度下的 Softmax 更容易饱和。

3. **Mask 语义混乱。**  
   有的实现用 `1` 表示可见，有的实现用 `True` 表示遮挡。写代码前必须统一约定。

4. **`view` 前忘记 `contiguous()`。**  
   转置后的张量内存不一定连续，直接 `view` 可能报错或行为不符合预期。

---

## 9. 总结

Self-Attention 的主线如下：

| 步骤 | 操作 | 输出 |
|:--|:--|:--|
| 1 | 线性投影 | $Q,K,V$ |
| 2 | 点积匹配 | $QK^\top$ |
| 3 | 缩放与 mask | 稳定且合法的注意力分数 |
| 4 | Softmax | 注意力权重 |
| 5 | 加权求和 | 上下文表示 |
| 6 | 多头拼接 | 最终输出 |

理解 Self-Attention 后，再看 [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]] 会更自然。

---

> **相关文章**：
> - [[NeuralNetwork/Overview/TransformerOverview|Transformer 架构总览]]
> - [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]
