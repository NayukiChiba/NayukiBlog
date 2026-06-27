---
title: FlashAttention 原理
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - FlashAttention
  - 长序列
description: 从标准 Attention 的显存读写瓶颈出发，解释 FlashAttention 如何通过分块和在线 Softmax 精确计算注意力。
image: https://img.yumeko.site/file/blog/FlashAttention.png
status: draft
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]。

![图0: FlashAttention 原理 Banner](https://img.yumeko.site/file/blog/FlashAttention.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于 FlashAttention 原理技术博客封面。
> 设计概念：展示注意力矩阵被分块搬入高速 SRAM，避免完整写回显存。
> 左侧是 Q、K、V 矩阵，中间是分块 attention 计算网格，右侧是在线 softmax 累积输出。
> 配色：深蓝到暖金渐变，现代数据科学美学风格。
> 简洁无衬线标签，淡色网格背景，顶部留白供标题叠加。
> ```

## 1. 标准 Attention 的问题

标准注意力公式：

$$
\boxed{
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
}
$$

直接实现时通常会显式构造注意力矩阵。计算顺序是：先计算 $QK^\top$ 得到 `scores`，再对 `scores` 做 Softmax 得到 `attn`，最后计算 `attn @ V` 得到输出。

当序列长度很大时，`seqLen x seqLen` 的矩阵会占用大量显存。

例如：

| 序列长度 | 注意力矩阵大小 |
|:--|:--|
| 2048 | 约 4M 元素 |
| 32768 | 约 1B 元素 |
| 131072 | 约 17B 元素 |

问题不仅是计算量，还有显存读写。

---

## 2. 为什么瓶颈是 IO

GPU 上有不同层级的存储：

| 存储 | 特点 |
|:--|:--|
| HBM 显存 | 容量大，但访问比片上缓存慢 |
| SRAM 片上缓存 | 容量小，但访问快 |

标准 Attention 会把完整 `scores` 和 `attn` 写入 HBM，再读回来继续计算。长序列下，这些读写成本非常高。

FlashAttention 的核心不是改公式，而是减少 HBM 往返。

---

## 3. FlashAttention 的核心思想

![图1: FlashAttention 的分块计算与在线 Softmax](https://img.yumeko.site/file/blog/FlashAttention/BlockwiseSoftmax.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的科学示意图，解释 FlashAttention 如何分块读取 K、V 并在线更新 softmax 统计量。
> 画面左侧是 Q block，中间是多个 K/V block 依次进入 SRAM，右侧是输出 O 的累积更新。
> 标注 m_i、l_i、O_i 三个在线 softmax 状态，并用箭头表示每个 block 只短暂驻留在高速缓存中。
> 白色背景，细线流程图，柔和蓝白配色，教科书插图风格。
> ```

FlashAttention 做两件事：

1. **Tiling 分块计算**  
   将 Q、K、V 拆成小块，在 SRAM 中完成局部计算。

2. **Online Softmax 在线归一化**  
   不需要先保存完整注意力矩阵，也能得到和标准 Softmax 一样的结果。

最终只把输出 $O$ 写回 HBM，而不是保存完整的注意力矩阵。

---

## 4. 为什么分块后 Softmax 仍然正确

Softmax 的难点在于它需要全局归一化：

$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

如果分块处理，每块只能看到局部的 max 和 sum。Online Softmax 维护两个量：

| 变量 | 含义 |
|:--|:--|
| $m$ | 当前已处理元素的最大值 |
| $l$ | 当前归一化分母的累积和 |

当新块到来时，更新全局最大值：

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})
$$

旧的累积和需要按新的最大值重新缩放：

$$
l_{\text{new}}
= e^{m_{\text{old}} - m_{\text{new}}}l_{\text{old}}
+ e^{m_{\text{block}} - m_{\text{new}}}l_{\text{block}}
$$

这样就能在分块条件下得到和完整 Softmax 等价的归一化结果。

---

## 5. FlashAttention 不是近似算法

这一点很重要：FlashAttention 不是稀疏注意力，也不是低秩近似。

| 方法 | 是否改变数学结果 |
|:--|:--|
| 稀疏注意力 | 改变，只看部分 token |
| 低秩注意力 | 改变，用近似矩阵 |
| FlashAttention | 不改变，精确等价 |

FlashAttention 改的是计算顺序和内存访问方式，不改 Attention 公式。

---

## 6. PyTorch 中如何使用

PyTorch 2.x 提供了 `scaled_dot_product_attention`：

```python
import torch.nn.functional as F

out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```

在满足设备、数据类型和张量布局条件时，PyTorch 会自动选择高效内核。实际使用时应以 profiler 或日志确认是否走到了期望内核。

---

## 7. FlashAttention 与 KV Cache 的区别

两者都和注意力效率有关，但解决的问题不同。

| 技术 | 主要场景 | 解决问题 |
|:--|:--|:--|
| KV Cache | 自回归推理 | 避免重复计算历史 K/V |
| FlashAttention | 训练和长序列计算 | 减少注意力矩阵的显存读写 |

可以同时使用：

- Prefill 阶段：FlashAttention 加速整段 prompt 的注意力。
- Decode 阶段：KV Cache 复用历史 K/V。

---

## 8. 常见坑

1. **把 FlashAttention 当成近似注意力。**  
   它的输出与标准 Attention 等价，只是计算更省显存。

2. **以为所有情况下都会自动启用。**  
   是否启用取决于 PyTorch 版本、GPU、dtype、mask 类型和张量布局。

3. **忽略 dropout 和训练模式。**  
   训练中 dropout 会影响内核选择和结果复现。

4. **只看 FLOPs，不看 IO。**  
   FlashAttention 的价值主要来自减少 HBM 读写，而不只是减少乘加次数。

---

## 9. 总结

FlashAttention 的主线如下：

| 步骤 | 说明 |
|:--|:--|
| 1 | 标准 Attention 显式保存 $n \times n$ 矩阵 |
| 2 | 长序列下 HBM 读写成本巨大 |
| 3 | 将 $Q,K,V$ 分块放入 SRAM 计算 |
| 4 | 用 Online Softmax 保持精确归一化 |
| 5 | 只写回最终输出 |

它是长序列 Transformer 训练和推理中的关键底层优化。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/SelfAttentionMechanism|Self-Attention 机制详解]]
