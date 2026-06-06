---
title: Transformer 计算与实现技巧：从位置编码到 FlashAttention
date: 2026-06-05
category: 神经网络/Transformer
tags:
  - Transformer
  - 注意力机制
  - 推理优化
  - FlashAttention
  - 位置编码
description: 深入讲解 Transformer 的核心计算：Self-Attention 的时间/空间复杂度分析、KV Cache 推理加速原理、FlashAttention 的 IO 感知优化、Pre-LN vs Post-LN 的架构选择、RoPE 等主流位置编码方案对比，以及 MoE 的稀疏计算。
image: https://img.yumeko.site/file/blog/cover/1780734066880.webp
status: published
---

> **前置阅读**：本文假定读者熟悉 Transformer 和 Self-Attention 的基本概念。建议先阅读 [[NeuralNetwork/RNN/Attention|注意力机制详解]]。

## 1. Self-Attention 的计算复杂度

### 1.1 标准 Scaled Dot-Product Attention

给定输入序列 $\mathbf{X} \in \mathbb{R}^{n \times d}$（$n$ 为序列长度，$d$ 为隐藏维度），Self-Attention 的计算为：

$$
\boxed{\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}}
$$

其中 $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, $\mathbf{V} = \mathbf{X}\mathbf{W}^V$。

### 1.2 复杂度拆解

| 步骤 | 操作 | 复杂度 | 说明 |
|:--|:--|:--|:--|
| QKV 投影 | $\mathbf{X} \cdot \mathbf{W}$ | $O(n \cdot d^2)$ | 线性变换 |
| 注意力分数 | $\mathbf{Q}\mathbf{K}^\top$ | $O(n^2 \cdot d)$ | **序列长度的平方** |
| Softmax | 逐行 softmax | $O(n^2)$ | 指数 + 归一化 |
| 加权求和 | $\operatorname{softmax} \cdot \mathbf{V}$ | $O(n^2 \cdot d)$ | 与分数矩阵同规模 |
| 输出投影 | 线性变换 | $O(n \cdot d^2)$ | 标准 FFN |

**总复杂度**：当 $n \gg d$ 时，瓶颈在 $O(n^2 \cdot d)$。这也是长序列处理的根本挑战。

> [!WARNING] 长序列的 O(n²) 瓶颈
> - 当 $n = 2048$ 时，注意力矩阵大小为 $2048 \times 2048 = 4\text{M}$
> - 当 $n = 32768$（32K 上下文），注意力矩阵大小为 $32\text{K} \times 32\text{K} \approx 1\text{B}$
> - 当 $n = 131072$（128K 上下文），约为 $17\text{B}$ —— 单张 GPU 显存已无法容纳

### 1.3 Multi-Head Attention 的复杂度

Multi-Head Attention 将 $d$ 维拆分为 $h$ 个头，每个头维度 $d_k = d / h$：

- 每个头计算注意力：$O(n^2 \cdot d_k) = O(n^2 \cdot d / h)$
- $h$ 个头并行：$O(h \cdot n^2 \cdot d / h) = O(n^2 \cdot d)$

总计算量与单头注意力相同，但多头允许模型关注不同表示子空间。

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    标准 Multi-Head Attention 实现

    Args:
        dModel: 模型维度
        numHeads: 注意力头数
        dropout: Dropout 概率
    """

    def __init__(self, dModel=512, numHeads=8, dropout=0.1):
        super().__init__()
        assert dModel % numHeads == 0
        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads

        # QKV 联合投影
        self.wQKV = nn.Linear(dModel, 3 * dModel, bias=False)
        self.wO = nn.Linear(dModel, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 形状 (batch, n, dModel)
            mask: 形状 (batch, n, n) 或 (batch, 1, n, n)
        Returns:
            形状 (batch, n, dModel)
        """
        batch, n, _ = x.shape

        # 投影到 QKV: (batch, n, 3*dModel)
        qkv = self.wQKV(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆分为多头: (batch, numHeads, n, dK)
        q = q.view(batch, n, self.numHeads, self.dK).transpose(1, 2)
        k = k.view(batch, n, self.numHeads, self.dK).transpose(1, 2)
        v = v.view(batch, n, self.numHeads, self.dK).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.dK ** 0.5
        scores = (q @ k.transpose(-2, -1)) / scale  # (batch, h, n, n)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (batch, h, n, dK)

        # 合并多头: (batch, n, dModel)
        out = out.transpose(1, 2).contiguous().view(batch, n, self.dModel)
        return self.wO(out)
```

## 2. KV Cache：推理加速的核心技巧

### 2.1 问题

在自回归推理时，每生成一个 Token，Transformer 需要重新计算**整个序列**的 Attention。这意味着一级重复计算：

- 第 1 步：计算 1 个 Token 的 Attention
- 第 2 步：计算 2 个 Token 的 Attention（重复了第 1 步！）
- 第 $i$ 步：计算 $i$ 个 Token 的 Attention（重复了前 $i-1$ 步）

总计算量约为 $O(n^2 \cdot d)$ 中的重复部分为 $O(n^3 \cdot d)$。

### 2.2 KV Cache 原理

**核心思想**：之前已计算的 Key 和 Value 不需要重新计算，直接缓存下来，只对新 Token 计算 QKV。

```
推理第 0 步（Prefill）：
  输入整个 Prompt → 计算并缓存所有层的 K, V

推理第 1 步：
  输入新生成的 1 个 Token → Q = [1, d]，K_new = [1, d]，V_new = [1, d]
  K_cached = concat(K_cached, K_new)  → [prompt_len + 1, d]
  V_cached = concat(V_cached, V_new)  → [prompt_len + 1, d]
  Attention(Q_new, K_cached, V_cached)
```

![KVCache.png](https://img.yumeko.site/file/blog/articles/1780733951336.webp)

### 2.3 KV Cache 的内存分析

KV Cache 的显存占用：

$$
\text{Memory} = 2 \times \text{numLayers} \times \text{batchSize} \times \text{seqLen} \times \text{dModel} \times \text{dtypeBytes}
$$

对于 LLaMA-2 7B（$L=32$, $d=4096$, FP16）：
- 序列长度 2048，batch=1 → $2 \times 32 \times 1 \times 2048 \times 4096 \times 2 \approx 1\text{GB}$
- 序列长度 4096，batch=1 → $\approx 2\text{GB}$
- 高并发推理时 KV Cache 是**主要的显存瓶颈**，而非模型参数

```python
class KVCacheAttention(nn.Module):
    """
    带 KV Cache 的 Multi-Head Attention

    Args:
        dModel: 模型维度
        numHeads: 注意力头数
    """

    def __init__(self, dModel=512, numHeads=8):
        super().__init__()
        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads

        self.wQ = nn.Linear(dModel, dModel, bias=False)
        self.wK = nn.Linear(dModel, dModel, bias=False)
        self.wV = nn.Linear(dModel, dModel, bias=False)
        self.wO = nn.Linear(dModel, dModel)

    def forward(self, x, kCache=None, vCache=None):
        """
        Args:
            x: 当前输入（单 Token 或完整 Prompt），形状 (batch, n, dModel)
            kCache: 缓存的 Key，形状 (batch, numHeads, prevLen, dK) 或 None
            vCache: 缓存的 Value，形状 (batch, numHeads, prevLen, dK) 或 None
        Returns:
            output, kCacheNew, vCacheNew
        """
        batch, n, _ = x.shape

        q = self.wQ(x).view(batch, n, self.numHeads, self.dK).transpose(1, 2)
        k = self.wK(x).view(batch, n, self.numHeads, self.dK).transpose(1, 2)
        v = self.wV(x).view(batch, n, self.numHeads, self.dK).transpose(1, 2)

        # 如果有缓存，追加到现有 K/V
        if kCache is not None:
            k = torch.cat([kCache, k], dim=2)  # dim=2 是序列维度
            v = torch.cat([vCache, v], dim=2)

        scale = self.dK ** 0.5
        scores = (q @ k.transpose(-2, -1)) / scale
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, n, self.dModel)
        return self.wO(out), k, v  # 返回更新后的缓存
```

## 3. FlashAttention：IO 感知的精确注意力

### 3.1 瓶颈不是计算量，而是 IO

标准 Attention 需要将 $n \times n$ 的注意力矩阵写入 HBM（高带宽显存），再读出做 Softmax。当 $n$ 较大时，**IO 带宽成为瓶颈**，而不是 FLOPs。

| 操作 | 计算量 (FLOPs) | IO 量 | 瓶颈类型 |
|:--|:--|:--|:--|
| QK^T | $O(n^2 d)$ | $O(nd)$ 读 + $O(n^2)$ 写 | IO-bound |
| Softmax | $O(n^2)$ | $O(n^2)$ 读 + $O(n^2)$ 写 | IO-bound |
| Attention × V | $O(n^2 d)$ | $O(n^2)$ 读 + $O(nd)$ 写 | IO-bound |

### 3.2 FlashAttention 的核心思想

**Tiling（分块）**：将 $Q, K, V$ 分块，在 SRAM（片上缓存）中完成局部的 Softmax + 注意力计算，只将最终结果写回 HBM。

**Online Softmax**：用在线算法重写 Softmax，支持分块计算后合并：

标准 Softmax 需要两遍扫描（先算 max 归一化，再算 sum）：
$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

在线 Softmax 使用**修正因子**在分块处理后合并结果，只需一次扫描。

![FlashAttention.png](https://img.yumeko.site/file/blog/articles/1780734010500.webp)

### 3.3 使用 FlashAttention

PyTorch 2.0 起内置了 `scaled_dot_product_attention`，满足 CUDA 架构和 dtype 条件时自动调度到 FlashAttention 内核。也可直接使用 `flash-attn` 库。

```python
import torch.nn.functional as F

# PyTorch 2.0+ 内置（自动使用 FlashAttention）
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 或直接使用 flash-attn 库
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

> [!TIP] FlashAttention 的效果
> - 内存复杂度从 $O(n^2)$ 降低到 $O(n)$（不存储完整注意力矩阵）
> - 训练速度提升 2-4×
> - 支持更长序列（如 GPT-4 的 128K 上下文）
> - 数学上**精确等价**于标准 Attention（不是近似！）

### 3.4 FlashAttention-2/3 的改进

- **FlashAttention-2**：优化了 Q 在序列维度的并行策略，在 H100 上实现 2× 加速
- **FlashAttention-3**：针对 Hopper 架构（H100/H200）的异步执行优化，利用 TMA（Tensor Memory Accelerator）和 FP8

## 4. 位置编码方案

### 4.1 方案对比

| 位置编码 | 类型 | 可学习 | 外推能力 | 复杂度 | 代表模型 |
|:--|:--|:--|:--|:--|:--|
| Sinusoidal | 绝对 | 否 | 差 | $O(n)$ | 原始 Transformer |
| Learned | 绝对 | 是 | 差 | $O(n)$ | BERT, GPT-1 |
| RoPE | 相对隐式 | 否 | 中（可扩展） | $O(nd)$ | LLaMA, Qwen, Mistral |
| ALiBi | 相对显式 | 否 | 好 | $O(1)$ | BLOOM |

### 4.2 RoPE（Rotary Position Embedding）

当前最主流的位置编码方案。原理是将位置信息以旋转矩阵的形式编码到 Query 和 Key 中：

对于位置 $m$ 的 Query 向量 $\mathbf{q}_m \in \mathbb{R}^d$，第 $i$ 对维度 $(q_{2i}, q_{2i+1})$ 进行旋转变换：

$$
\begin{bmatrix} q_{2i}' \\ q_{2i+1}' \end{bmatrix} = \begin{bmatrix}
\cos(m \theta_i) & -\sin(m \theta_i) \\
\sin(m \theta_i) & \cos(m \theta_i)
\end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

其中 $\theta_i = b^{-\frac{2i}{d}}$（$b$ 通常为 10000）。

RoPE 的优雅之处：两个位置 $m, n$ 的 Attention Score 仅依赖于**相对位置** $m-n$：

$$
(\mathbf{R}_m \mathbf{q}_m)^\top (\mathbf{R}_n \mathbf{k}_n) = \mathbf{q}_m^\top \mathbf{R}_{n-m} \mathbf{k}_n
$$

```python
def applyRotaryEmbedding(x, cos, sin):
    """
    应用 RoPE 旋转位置编码

    Args:
        x: 输入张量，形状 (batch, numHeads, seqLen, dK)
        cos: 预计算的 cos 值，形状 (seqLen, dK)
        sin: 预计算的 sin 值，形状 (seqLen, dK)
    Returns:
        旋转后的张量
    """
    # 将特征维度两两分组 [x0, x1, x2, x3, ...]  →  [[x0, x1], [x2, x3], ...]
    xReshaped = x.reshape(*x.shape[:-1], -1, 2)

    # 旋转: [x, y] → [x*cos - y*sin, x*sin + y*cos]
    xRotated = torch.stack([
        xReshaped[..., 0] * cos - xReshaped[..., 1] * sin,
        xReshaped[..., 0] * sin + xReshaped[..., 1] * cos,
    ], dim=-1)

    return xRotated.flatten(-2)
```

## 5. Pre-LN vs Post-LN

Layer Normalization 的位置对训练稳定性有决定性影响。

### 5.1 两种架构

**Post-LN**（原始 Transformer）：

$$
\mathbf{x} \to \operatorname{Attention}(\mathbf{x}) \to \mathbf{x} + \operatorname{Attention}(\mathbf{x}) \to \operatorname{LN}(\mathbf{x} + \operatorname{Attention}(\mathbf{x})) \to \operatorname{FFN}(\dots)
$$

**Pre-LN**（现代 Transformer）：

$$
\mathbf{x} \to \operatorname{LN}(\mathbf{x}) \to \operatorname{Attention}(\operatorname{LN}(\mathbf{x})) \to \mathbf{x} + \operatorname{Attention}(\operatorname{LN}(\mathbf{x})) \to \operatorname{LN}(\dots) \to \operatorname{FFN}(\dots)
$$

### 5.2 对比

| 维度 | Post-LN | Pre-LN |
|:--|:--|:--|
| 梯度传播 | 残差路径中有 LN，可能阻碍梯度 | 残差路径中无 LN，梯度直通 |
| 训练稳定性 | 需要 warmup，深层难训练 | 无需 warmup，深层稳定 |
| 最终性能 | 理论上略好（有争议） | 实际应用中足够 |
| 现代采用 | 较少 | 主流（LLaMA, GPT-3+, Chinchilla） |

> [!TIP] 选择建议
> Pre-LN 是现代默认选择。原始 Post-LN Transformer 用 warmup 来缓解训练不稳定——
> 现在已经不需要了。详见 [[NeuralNetwork/Training/LearningRate|学习率策略详解]]。

## 6. MoE（Mixture of Experts）：稀疏计算

### 6.1 核心思想

标准 Transformer 的 FFN 层在推理时使用了**全部参数**。MoE 的 FFN 层包含 $E$ 个专家（Expert），每个 Token 只激活 Top-K 个（通常 K=1 或 2）：

$$
\boxed{\mathbf{y} = \sum_{i=1}^E g_i(\mathbf{x}) \cdot \operatorname{FFN}_i(\mathbf{x})}
$$

其中 $g(\mathbf{x}) = \operatorname{softmax}(\operatorname{TopK}(\mathbf{x} \cdot \mathbf{W}_g, K))$ 是**稀疏门控**。

### 6.2 计算优势

- **参数量**：$E \times$ FFN 参数（如 Mixtral 8×7B = 46.7B 总参数）
- **激活参数**：$\approx$ 1× FFN 参数（如 Mixtral 每次激活 12.9B 参数）
- **推理速度**：激活参数量量级（远小于总参数量）

### 6.3 Load Balancing Loss

为防止所有 Token 都选同一个专家，需要额外的负载均衡损失：

$$
\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^E f_i \cdot P_i
$$

其中 $f_i$ 是分配给专家 $i$ 的 Token 比例，$P_i$ 是路由到专家 $i$ 的平均概率。当分配均匀时，$f_i = 1/E$, $\mathcal{L}_{\text{aux}} = 1$。

## 7. 其他实用技巧

### 7.1 GQA / MQA（Query 分组 / 多查询注意力）

减少 KV Cache 显存的关键技术：

- **MHA**（Multi-Head Attention）：$h$ 个 Q head + $h$ 个 K head + $h$ 个 V head
- **GQA**（Grouped-Query Attention）：$h$ 个 Q head，但只有 $g$ 个 K/V head（$g \ll h$）。同组的 Q head 共享 K, V
- **MQA**（Multi-Query Attention）：所有 Q head 共享一个 K, V（$g=1$）

KV Cache 减小了 $h/g$ 倍。LLaMA-2 70B 使用 $g=8$（$h=64$），KV Cache 减少 8×。

### 7.2 RMSNorm

将 LayerNorm 简化为 RMSNorm（去除均值中心化，只做缩放）：

$$
\boxed{\operatorname{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}}
$$

计算量减少约 50%（无需计算均值），在 LLaMA 系列中使用。

### 7.3 SwiGLU 激活函数

现代 LLM 中常用的 FFN 激活函数，替代 ReLU：

$$
\boxed{\operatorname{SwiGLU}(\mathbf{x}) = \operatorname{Swish}(\mathbf{x}\mathbf{W}_1) \odot (\mathbf{x}\mathbf{W}_2)}
$$

其中 $\operatorname{Swish}(x) = x \cdot \sigma(x)$。SwiGLU 在 LLaMA、PaLM 等模型中均有使用。

---

> **相关文章**：
> - [[NeuralNetwork/RNN/Attention|注意力机制详解]]
> - [[NeuralNetwork/Training/LearningRate|学习率策略详解]]
> - [[NeuralNetwork/LLM/EmergentAbilities|大模型涌现能力详解]]
> - [[NeuralNetwork/LLM/RLHF|RLHF 算法详解]]
