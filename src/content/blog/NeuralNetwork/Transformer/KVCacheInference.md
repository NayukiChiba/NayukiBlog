---
title: KV Cache 推理加速
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - KV Cache
  - 推理优化
description: 解释自回归生成中的重复计算问题，拆解 KV Cache 如何缓存历史 Key/Value 并降低单步推理成本。
image: https://img.yumeko.site/file/blog/cover/1782558811921_KVCacheInference.webp
status: draft
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]] 与 [[NeuralNetwork/Transformer/PositionalEncoding|Transformer 位置编码]]。

## 1. 自回归推理为什么慢

自回归生成一次只生成一个 token：

| 步骤 | 输入上下文 | 输出 |
|:--|:--|:--|
| 1 | prompt | 第 1 个 token |
| 2 | prompt + 第 1 个 token | 第 2 个 token |
| 3 | prompt + 前 2 个 token | 第 3 个 token |

如果每一步都把完整上下文重新送进模型，就会重复计算大量历史 token。

第 $t$ 步生成时，历史 token 的 Key 和 Value 其实已经算过。重复计算它们没有必要。

---

## 2. KV Cache 的核心思想

Self-Attention 中：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

生成新 token 时：

- 新 token 的 Query 必须重新计算。
- 新 token 的 Key/Value 需要计算并追加到缓存。
- 历史 token 的 Key/Value 可以直接复用。

因此 KV Cache 缓存的是每一层、每个头的历史 Key 和 Value。

第 $t$ 步只输入新 token，计算 $q_{\text{new}}$、$k_{\text{new}}$、$v_{\text{new}}$，再将新的 Key/Value 追加到缓存：

$$
K_{\text{cache}} \leftarrow \operatorname{concat}(K_{\text{cache}}, k_{\text{new}})
$$

$$
V_{\text{cache}} \leftarrow \operatorname{concat}(V_{\text{cache}}, v_{\text{new}})
$$

最后用 $q_{\text{new}}$ 对完整缓存做注意力计算。

---

## 3. Prefill 与 Decode

![PrefillDecode.png](https://img.yumeko.site/file/blog/articles/1782559253455_PrefillDecode.webp)

自回归推理通常分成两个阶段：

| 阶段 | 输入 | 作用 |
|:--|:--|:--|
| Prefill | 完整 prompt | 一次性计算 prompt 的所有 KV |
| Decode | 每次一个新 token | 复用缓存，只计算新增 token |

Prefill 仍然是并行的，复杂度和完整 Attention 类似。Decode 阶段每次只处理一个新 token，KV Cache 的收益主要体现在这里。

---

## 4. 形状变化

以 Decoder-only 模型为例，单层缓存形状常见为：

| 张量 | 形状 |
|:--|:--|
| `k_cache` | `(batch, num_heads, cache_len, head_dim)` |
| `v_cache` | `(batch, num_heads, cache_len, head_dim)` |

当前新 token：

| 张量 | 形状 |
|:--|:--|
| `x` | `(batch, 1, d_model)` |
| `q_new` | `(batch, num_heads, 1, head_dim)` |
| `k_new` | `(batch, num_heads, 1, head_dim)` |
| `v_new` | `(batch, num_heads, 1, head_dim)` |

追加后：

| 张量 | 形状 |
|:--|:--|
| `k_all` | `(batch, num_heads, cache_len + 1, head_dim)` |
| `v_all` | `(batch, num_heads, cache_len + 1, head_dim)` |

注意力分数：

$$
\text{scores} = q_{\text{new}} K_{\text{all}}^\top
$$

| 张量 | 形状 |
|:--|:--|
| `scores` | `(batch, num_heads, 1, cache_len + 1)` |

---

## 5. 简化代码

```python
import torch
import torch.nn as nn


class KVCacheAttention(nn.Module):
    """带 KV Cache 的注意力层。"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out), k, v
```

---

## 6. 显存占用

KV Cache 的显存占用约为：

$$
\boxed{
\text{Memory}
= 2 \times L \times B \times T \times d_{\text{model}} \times \text{bytes}
}
$$

其中：

| 符号 | 含义 |
|:--|:--|
| $2$ | Key 和 Value |
| $L$ | 层数 |
| $B$ | batch size |
| $T$ | 缓存长度 |
| $d_{\text{model}}$ | 模型隐藏维度 |
| bytes | 数据类型字节数，如 FP16 为 2 |

这也是长上下文推理和高并发服务中，KV Cache 常常成为显存瓶颈的原因。

---

## 7. KV Cache 与 GQA/MQA

标准多头注意力中，每个 Query head 都有自己的 Key/Value head。GQA 和 MQA 通过共享 Key/Value 减少缓存：

| 结构 | Q 头数 | KV 头数 | KV Cache |
|:--|--:|--:|:--|
| MHA | $h$ | $h$ | 最大 |
| GQA | $h$ | $g$ | 减少 $h/g$ 倍 |
| MQA | $h$ | $1$ | 最小 |

因此 GQA/MQA 的核心收益之一就是降低推理阶段的 KV Cache 显存。

---

## 8. 常见坑

1. **以为 KV Cache 会减少模型参数。**  
   KV Cache 只缓存中间激活，不改变参数量。

2. **以为 KV Cache 主要用于训练。**  
   常规训练需要并行看完整序列，KV Cache 主要用于自回归推理。

3. **忘记位置偏移。**  
   使用位置编码时，新 token 的位置必须从 `cacheLen` 继续计数。

4. **无限增长缓存。**  
   实际推理中通常有最大上下文长度，超过后需要截断或滑动窗口策略。

---

## 9. 总结

KV Cache 的本质是：历史 token 的 $K,V$ 已经算过，不要重复计算。

它把自回归 decode 阶段从“每步重算整段上下文”改成“每步只算新增 token”，是 Transformer 推理优化中最基础也最重要的技术之一。

这篇文章暂时作为进阶推理优化内容保留，基础 Transformer 可以先不阅读。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
> - [[NeuralNetwork/Transformer/PositionalEncoding|Transformer 位置编码]]
