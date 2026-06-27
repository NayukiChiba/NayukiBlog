---
title: Transformer Block 结构
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - LayerNorm
  - FFN
description: 拆解 Transformer Block 中的 Attention、FFN、残差连接和 LayerNorm，并比较 Pre-LN 与 Post-LN。
image: https://img.yumeko.site/file/blog/TransformerBlock.png
status: published
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]。

![图0: Transformer Block 结构 Banner](https://img.yumeko.site/file/blog/TransformerBlock.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于 Transformer Block 结构技术博客封面。
> 设计概念：展示一个 Pre-LN Transformer Block 中 Attention、FFN、残差连接和 LayerNorm 的数据流。
> 中间是上下排列的 Attention 子层和 FFN 子层，旁边用弧形箭头表示 residual path。
> 配色：深蓝到暖金渐变，现代数据科学美学风格。
> 简洁无衬线标签，淡色网格背景，顶部留白供标题叠加。
> ```

## 1. Block 是 Transformer 的基本堆叠单元

Transformer 不是只做一次 Attention，而是重复堆叠多个 Block：

| 阶段 | 作用 |
|:--|:--|
| Embedding | 将 token id 转成向量 |
| Block 堆叠 | 逐层更新上下文表示 |
| Output Head | 映射到任务输出空间 |

每个 Block 通常包含：

| 部件 | 作用 |
|:--|:--|
| Multi-Head Attention | token 之间交换信息 |
| Feed Forward Network | 对每个 token 独立做非线性变换 |
| Residual Connection | 保留原输入并改善梯度传播 |
| LayerNorm | 稳定激活分布 |

---

## 2. Attention 子层

Attention 子层负责跨位置交互。输入形状：

| 张量 | 形状 |
|:--|:--|
| `x` | `(batch, seqLen, dModel)` |

输出形状仍然是：

| 张量 | 形状 |
|:--|:--|
| `attnOut` | `(batch, seqLen, dModel)` |

保持形状不变很重要，因为后面需要残差相加：

$$
x + \operatorname{Attention}(x)
$$

如果维度不一致，残差连接就无法直接成立。

---

## 3. FFN 子层

FFN 是逐位置的多层感知机。它不会让不同 token 互相通信，而是对每个 token 的表示做同一套非线性变换。

典型形式：

$$
\operatorname{FFN}(x) = W_2 \phi(W_1x + b_1) + b_2
$$

常见维度：

$$
d_{\text{ffn}} = 4 \times d_{\text{model}}
$$

例如 $d_{\text{model}}=512$ 时，FFN 中间层通常是 2048。

| 阶段 | 形状 |
|:--|:--|
| 输入 | `(batch, seqLen, 512)` |
| 第一层线性变换 | `(batch, seqLen, 2048)` |
| 激活函数 | `(batch, seqLen, 2048)` |
| 第二层线性变换 | `(batch, seqLen, 512)` |

Attention 负责“混合位置”，FFN 负责“加工每个位置的表示”。

---

## 4. 残差连接

残差连接将子层输入直接加到输出上：

$$
y = x + F(x)
$$

它有两个作用：

1. 保留原始信息，避免子层必须重新学习恒等映射。
2. 提供更直接的梯度路径，使深层模型更容易优化。

在 Transformer 中，Attention 和 FFN 后面都通常有残差：

$$
x' = x + \operatorname{Attention}(x)
$$

$$
y = x' + \operatorname{FFN}(x')
$$

---

## 5. LayerNorm 的位置

LayerNorm 可以放在子层之后，也可以放在子层之前。

![图1: Post-LN 与 Pre-LN 数据流对比](https://img.yumeko.site/file/blog/TransformerBlock/PreLnPostLn.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的对比图，展示 Post-LN 与 Pre-LN Transformer Block 的归一化位置差异。
> 画面左右两列：左列是 Post-LN，数据流为 x -> Sublayer -> Add -> LayerNorm；右列是 Pre-LN，数据流为 x -> LayerNorm -> Sublayer -> Add。
> 用不同颜色标出主分支、残差分支和 LayerNorm 节点，标签清晰标注 Post-LN、Pre-LN、Residual、LayerNorm。
> 白色背景，细线流程图，柔和蓝白配色，教科书插图风格。
> ```

### 5.1 Post-LN

原始 Transformer 使用 Post-LN：

$$
x' = \operatorname{LN}(x + \operatorname{Attention}(x))
$$

$$
y = \operatorname{LN}(x' + \operatorname{FFN}(x'))
$$

特点：

- 结构和原论文一致。
- 深层训练更容易不稳定。
- 通常更依赖学习率 warmup。

### 5.2 Pre-LN

现代深层 Transformer 更常用 Pre-LN：

$$
\boxed{
x' = x + \operatorname{Attention}(\operatorname{LN}(x))
}
$$

$$
\boxed{
y = x' + \operatorname{FFN}(\operatorname{LN}(x'))
}
$$

特点：

- 残差路径更直接。
- 深层训练更稳定。
- 是许多现代 Decoder-only 模型的常见选择。

---

## 6. Pre-LN 代码示例

```python
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Transformer 中的逐位置前馈网络。"""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block。"""

    def __init__(self, attention: nn.Module, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.attention = attention
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

---

## 7. Encoder Block 与 Decoder Block 的差异

| Block 类型 | 子层 |
|:--|:--|
| Encoder Block | Self-Attention、FFN |
| Encoder-Decoder 中的 Decoder Block | Masked Self-Attention、Cross-Attention、FFN |
| Decoder-only Block | Masked Self-Attention、FFN |

这也是为什么 Decoder-only 结构更简洁，适合大规模自回归语言建模。

---

## 8. 常见坑

1. **把 FFN 理解成跨 token 混合。**  
   FFN 是逐位置应用的，同一个 MLP 作用在每个 token 上。

2. **忽略残差前后形状必须一致。**  
   子层输出必须回到 `dModel`，否则不能与输入相加。

3. **把 Pre-LN 和 Post-LN 混写。**  
   两者不是简单换个位置，训练稳定性和梯度路径都会变化。

4. **只关注 Attention，忽略 FFN 参数量。**  
   在很多 Transformer 中，FFN 占据了相当大比例的参数和计算。

---

## 9. 总结

Transformer Block 可以理解成两步：Attention 让 token 互相交换信息，FFN 对每个 token 的新表示做非线性加工。

残差连接和 LayerNorm 则负责让这个结构可以稳定地堆得很深。

下一步可以继续阅读 [[NeuralNetwork/Transformer/PositionalEncoding|Transformer 位置编码]]。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
> - [[NeuralNetwork/Transformer/PositionalEncoding|Transformer 位置编码]]
> - [[NeuralNetwork/Training/BatchNormalization|BatchNorm 与归一化]]
