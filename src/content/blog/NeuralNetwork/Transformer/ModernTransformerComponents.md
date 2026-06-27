---
title: 现代 Transformer 组件
date: 2026-06-27
category: 神经网络/Transformer
tags:
  - Transformer
  - GQA
  - RMSNorm
  - MoE
description: 解释 GQA/MQA、RMSNorm、SwiGLU、MoE 等现代 Transformer 中常见组件的作用与取舍。
image: https://img.yumeko.site/file/blog/ModernTransformerComponents.png
status: draft
---

> **前置阅读**：建议先阅读 [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]。

![图0: 现代 Transformer 组件 Banner](https://img.yumeko.site/file/blog/ModernTransformerComponents.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于现代 Transformer 组件技术博客封面。
> 设计概念：在一个 Transformer Block 周围展示 GQA、RMSNorm、SwiGLU、MoE 和 Pre-LN 等现代组件。
> 中间是主干 Block，周围以模块标签连接不同改进点，并标出它们分别影响推理显存、训练稳定性和模型容量。
> 配色：深蓝到暖金渐变，现代数据科学美学风格。
> 简洁无衬线标签，淡色网格背景，顶部留白供标题叠加。
> ```

## 1. 为什么 Transformer 还在不断变化

原始 Transformer 的核心结构已经很强，但大规模模型会暴露新的工程问题：

| 问题 | 典型改造 |
|:--|:--|
| KV Cache 显存太大 | GQA / MQA |
| LayerNorm 计算略重 | RMSNorm |
| FFN 表达能力不足 | SwiGLU |
| 参数量想变大但计算不能同比例增加 | MoE |
| 深层训练不稳定 | Pre-LN |

这些组件不是替代 Transformer，而是在 Transformer Block 内部做更适合大规模训练和推理的改造。

---

## 2. GQA 与 MQA

![图1: MHA、MQA 与 GQA 的头部分组对比](https://img.yumeko.site/file/blog/ModernTransformerComponents/GqaMqaComparison.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的对比图，展示 MHA、MQA 与 GQA 在 Query heads 和 Key/Value heads 上的对应关系。
> 画面分三列：MHA 中每个 Q head 有独立 K/V head；MQA 中所有 Q head 共享一个 K/V head；GQA 中多个 Q head 共享一组 K/V head。
> 用连线标出共享关系，关键标签包括 MHA、MQA、GQA、Q Heads、KV Heads、Cache Size。
> 白色背景，细线框图，柔和蓝白配色，教科书插图风格。
> ```

标准 Multi-Head Attention 中，每个 Query head 都有自己的 Key 和 Value：

| 类型 | 头数 |
|:--|:--|
| Query heads | $h$ |
| Key heads | $h$ |
| Value heads | $h$ |

这在推理时会带来很大的 KV Cache。

### 2.1 MQA

Multi-Query Attention 让所有 Query head 共享同一组 Key/Value：

| 类型 | 头数 |
|:--|:--|
| Query heads | $h$ |
| Key heads | $1$ |
| Value heads | $1$ |

优点是 KV Cache 最小，缺点是共享过强，可能影响表达能力。

### 2.2 GQA

Grouped-Query Attention 是折中方案：

| 类型 | 头数 |
|:--|:--|
| Query heads | $h$ |
| Key/Value heads | $g$ |

多个 Query head 共用一组 Key/Value。若 $h=32$、$g=8$，KV Cache 可减少 4 倍。

| 结构 | KV Cache | 表达能力 | 常见用途 |
|:--|:--|:--|:--|
| MHA | 大 | 强 | 标准结构 |
| MQA | 小 | 较弱 | 极限推理优化 |
| GQA | 中等 | 较强 | 现代大模型常用折中 |

---

## 3. RMSNorm

LayerNorm 计算：

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

RMSNorm 去掉均值中心化，只按均方根缩放：

$$
\boxed{
\operatorname{RMSNorm}(x)
= \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma
}
$$

特点：

- 少计算均值。
- 参数更少，通常没有 $\beta$。
- 在很多现代 Transformer 中表现稳定。

RMSNorm 的直觉是：只控制向量整体尺度，不强制把均值拉到 0。

---

## 4. SwiGLU

原始 Transformer 的 FFN 常用 ReLU：

$$
\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x)
$$

现代模型常用门控 FFN，例如 SwiGLU：

$$
\operatorname{SwiGLU}(x)
= \operatorname{Swish}(xW_1) \odot (xW_2)
$$

其中：

$$
\operatorname{Swish}(x)=x\sigma(x)
$$

可以把 SwiGLU 理解成：一条分支生成内容，另一条分支生成门控，二者逐元素相乘后再输出。

---

## 5. MoE：Mixture of Experts

标准 FFN 对每个 token 都使用同一套参数。MoE 将 FFN 换成多个专家：

$$
y = \sum_{i=1}^{E} g_i(x)\operatorname{Expert}_i(x)
$$

实际中通常只激活 Top-K 个专家：

例如总专家数 $E=8$ 时，每个 token 可以只选择 Top-2 个专家参与计算。

这样可以增加总参数量，同时让每个 token 的实际计算量保持较低。

### 5.1 MoE 的问题

MoE 最大的问题是路由不均衡。如果所有 token 都选择同一个专家，其他专家就浪费了。

因此通常需要负载均衡损失：

$$
\mathcal{L}_{aux} = E \sum_{i=1}^{E} f_i P_i
$$

其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是路由到专家 $i$ 的平均概率。

---

## 6. Pre-LN

Pre-LN 将 LayerNorm 放到子层之前：

$$
x' = x + F(\operatorname{LN}(x))
$$

它让残差路径更直接，深层训练通常更稳定。

Post-LN 则是：

$$
x' = \operatorname{LN}(x + F(x))
$$

可以简单记：

| 结构 | LayerNorm 位置 | 特点 |
|:--|:--|:--|
| Post-LN | 残差相加之后 | 原始结构，深层更难训 |
| Pre-LN | 子层计算之前 | 现代常用，更稳定 |

---

## 7. 这些组件分别解决什么

| 组件 | 解决的问题 | 代价 |
|:--|:--|:--|
| GQA/MQA | 降低 KV Cache 显存 | K/V 表达能力减少 |
| RMSNorm | 简化归一化计算 | 表达形式略有变化 |
| SwiGLU | 增强 FFN 表达 | 参数和计算增加 |
| MoE | 提升总参数量但保持稀疏计算 | 路由和负载均衡复杂 |
| Pre-LN | 稳定深层训练 | 有时最终性能需调参比较 |

---

## 8. 常见坑

1. **把这些组件当成必须全部使用。**  
   它们是工程取舍，不是所有模型都需要。

2. **只看参数量，不看激活参数量。**  
   MoE 的总参数量很大，但每个 token 只激活一部分专家。

3. **只看 GQA 减显存，不看质量影响。**  
   KV head 越少，共享越强，需要在速度和效果间权衡。

4. **以为 RMSNorm 只是更快的 LayerNorm。**  
   它去掉了均值中心化，数学形式确实不同。

---

## 9. 总结

现代 Transformer 的改造方向可以分成三类：

| 方向 | 代表组件 |
|:--|:--|
| 推理效率 | GQA、MQA、KV Cache |
| 训练稳定 | Pre-LN、RMSNorm |
| 表达能力 | SwiGLU、MoE |

这些组件共同目标很务实：让 Transformer 在更大规模、更长上下文、更高并发的场景下仍然可训练、可部署、可扩展。

---

> **相关文章**：
> - [[NeuralNetwork/Transformer/TransformerBlock|Transformer Block 结构]]
