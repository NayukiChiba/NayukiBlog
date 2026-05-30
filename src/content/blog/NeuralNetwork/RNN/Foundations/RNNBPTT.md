---
title: BPTT 详解：循环神经网络的反向传播
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
  - 优化
description: 从计算图展开到梯度连乘，深入理解 Through-Time Backpropagation 的数学原理、截断策略和梯度裁剪的有效性边界。
image: https://img.yumeko.site/file/blog/BackpropagationThroughTime.webp
status: draft
---

## 1. BPTT 的核心问题

### 1.1 普通反向传播 vs BPTT

在普通 MLP 中，反向传播（Backpropagation）沿着网络层从输出向输入传播梯度。网络是一个有向无环图（DAG），梯度计算路径是唯一且确定的。

在 RNN 中，情况复杂得多——同一个参数（如 $W_{hh}$）在**每个时间步**都被使用。因此，$W_{hh}$ 的总梯度 = 它在所有时间步产生的梯度之和。

**"Through Time"的含义**：梯度不仅穿过网络层，还**穿过时间**——从最后一步反向流到第一步。

### 1.2 展开计算图

RNN 的前向计算沿时间轴展开后，等价于一个**深度 = 序列长度**的"虚拟前馈网络"，且所有时间步的对应层共享参数：

![RNNBPTT.png](https://img.yumeko.site/file/articles/RNNBPTT/RNNBPTT.webp)

所有水平箭头（h_{t-1}→h_t）都使用同一个 W_hh
所有上升箭头（h_t→y_t）都使用同一个 W_hy
所有上升箭头（x_t→h_t）都使用同一个 W_xh

---

## 2. BPTT 的数学推导

### 2.1 前向传播的展开形式

对于长度为 $T$ 的序列，前向传播完全展开后：

$$
h_1 = \tanh(W_{hh} \cdot h_0 + W_{xh} \cdot x_1 + b_h)
$$
$$
h_2 = \tanh(W_{hh} \cdot h_1 + W_{xh} \cdot x_2 + b_h)
$$
$$
\vdots
$$
$$
h_T = \tanh(W_{hh} \cdot h_{T-1} + W_{xh} \cdot x_T + b_h)
$$

每个时间步的输出：

$$
y_t = W_{hy} \cdot h_t + b_y
$$

总损失（各时间步损失之和或平均）：

$$
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
$$

### 2.2 对 $W_{hh}$ 的梯度

![GradSum.png](https://img.yumeko.site/file/articles/RNNBPTT/GradSum.webp)

链式法则：$W_{hh}$ 在每个时间步都参与计算，所以：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

对于单个时间步 $t$，$L_t$ 通过 $h_t$ 依赖于 $W_{hh}$。但 $h_t$ 又通过 $W_{hh}$ 依赖于 $h_{t-1}$，后者又通过 $W_{hh}$ 依赖于 $h_{t-2}$……最终依赖于 $h_0$。

展开时间步 $t$ 的梯度：

$$
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

其中**连乘项**是关键：

$$
\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}
$$

这一项决定了从时间步 $k$ 到时间步 $t$ 的梯度是否能有效传递。

### 2.3 $\frac{\partial h_j}{\partial h_{j-1}}$ 的具体形式

对于 Vanilla RNN：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(\tanh'(W_{hh} \cdot h_{j-1} + W_{xh} \cdot x_j + b_h)\big) \cdot W_{hh}
$$

两个因素：
- $\text{diag}(\tanh'(\cdot))$：对角矩阵，对角线元素 $\in (0, 1]$，大部分远小于 1
- $W_{hh}$：训练过程中动态变化，范数可大可小

**连乘 $T$ 步后**：

- 如果每一步的谱范数 $< 1$：连乘趋近于 **0** → **梯度消失**
- 如果每一步的谱范数 $> 1$：连乘趋近于 **$\infty$** → **梯度爆炸**

### 2.4 对 $W_{xh}$ 和 $W_{hy}$ 的梯度

$W_{hy}$ 的梯度相对简单——每次使用 $W_{hy}$ 是独立的（不通过隐藏状态传递），不涉及时间轴连乘：

$$
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T
$$

$W_{xh}$ 的梯度类似 $W_{hh}$，也需要沿时间轴反向传播，但连乘只从 $t$ 到 $k$：

$$
\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{xh}}
$$

### 2.5 具体数值例子

以 Char-RNN 的配置为例（seq_len=100, hidden=256）：

- 假设 $\tanh'$ 平均值约 0.3，$||W_{hh}||$ 约 0.8
- 每一步的衰减因子 ≈ 0.3 × 0.8 = 0.24
- 100 步连乘：$0.24^{100} \approx 1.0 \times 10^{-62}$

即使 $W_{hh}$ 的特征值接近 1（比如 0.99），$\tanh'$ 仍然会压缩梯度：

- 若 $\tanh'$ 平均 = 0.5，每步衰减 = 0.5 × 0.99 = 0.495
- 100 步：$0.495^{100} \approx 3.0 \times 10^{-31}$
- 仍在有效计算精度以下（float32 约 $10^{-7}$ 以下就不可靠了）

**结论**：Vanilla RNN 中的梯度消失是结构性的——$\tanh$ 的导数被限制在 (0, 1]，即使 $W_{hh}$ 设计得再好，也无法消除 $\tanh$ 对梯度的压缩效应。

---

## 3. Truncated BPTT

### 3.1 基本思路

对于极长的序列（如一本书有几百万字符），完整 BPTT 在计算上不可行（需要存储整个序列的中间激活值，内存爆炸）。Truncated BPTT 的做法是：

1. 将长序列切分成固定长度 $k$ 的片段（如 100 步）
2. 在每个片段内做完整的 BPTT
3. 将前一个片段的最终隐藏状态传给下一个片段，但**截断**梯度流

```
片段1: [x_1, ..., x_100]  → BPTT → 更新参数
片段2: [x_101, ..., x_200] → BPTT → 更新参数
       ↑ h_100 传给这里，但梯度不反传回片段1
```

### 3.2 PyTorch 实现

```python
h, c = initHidden(batchSize)

for batch in dataloader:
    # batch: (B, 100)，已经切分为固定长度片段

    # 关键：截断梯度，不让梯度跨越片段边界
    h = h.detach()
    c = c.detach()

    output, (h, c) = model(batch, (h, c))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

`detach()` 是关键的 PyTorch 操作——它创建了一个新的 tensor，与原 tensor 共享数据但不共享计算图。梯度无法穿过 `detach()` 的边界。

### 3.3 Truncation Length 的选择

| 截断长度 | 优点 | 缺点 |
|----------|------|------|
| 短（20-50） | 内存小、计算快 | 无法学习超过长度的依赖 |
| 中（100-200） | **平衡点** | Char-RNN 使用 100 |
| 长（500+） | 更长依赖 | 内存大、梯度消失风险 |
| 完整 BPTT | 全部依赖 | 内存爆炸、序列长时不可行 |

### 3.4 Truncated BPTT 的局限性

截断 BPTT 有一个根本限制：模型**无法学习跨片段的长距离依赖**。如果序列中的关键信息（如 "France"）在第 50 步出现，而需要它的位置（如填空 "French"）在第 200 步，且截断长度为 100，这两个位置将分属不同片段，梯度无法跨越。

这就是为什么尽管有 Truncated BPTT，LSTM/GRU 的门控机制仍然至关重要——在片段内部（如 100 步），门控确保梯度不会消失。

---

## 4. 梯度裁剪

### 4.1 数学操作

梯度裁剪是应对梯度**爆炸**（而非消失）的标准方法：

$$
\text{if } ||g|| > \text{threshold}: \quad g = g \cdot \frac{\text{threshold}}{||g||}
$$

- $g$：所有参数的梯度拼接成的向量
- $||g||$：梯度的 L2 范数
- 不改变方向，只限制大小

### 4.2 为什么裁剪有效？

梯度爆炸时，某一步的 $||g||$ 可能是 100、1000 甚至更大。如果不加限制，参数更新量 $\eta \cdot g$ 会巨大，导致参数跳到极端值区域，Loss 变成 NaN。

裁剪将梯度的范数限制在安全范围内（如 5.0 或 1.0），确保每步的更新量是可控的。

### 4.3 裁剪不能解决梯度消失

这是一个常见的误解。梯度裁剪只限制梯度的**上限**，不能提升梯度的**下限**。如果梯度因为连乘趋近于 0（梯度消失），裁剪完全无能为力——0 放大多少倍还是 0。

| 问题 | 梯度裁剪有效？ | 真正解决方案 |
|------|:---:|------|
| 梯度爆炸 | ✅ **有效** | 梯度裁剪 |
| 梯度消失 | ❌ 无效 | LSTM/GRU 的门控机制 |

### 4.4 两个项目中的阈值选择

| 项目 | 阈值 | 原因 |
|------|:---:|------|
| Char-RNN | 5.0 | 模型较简单，支持三种 RNN 类型，Vanilla RNN 梯度波动更大 |
| EmotionClassification | 1.0 | BiLSTM + Attention 模型更深，梯度更容易爆炸，需要更保守的阈值 |

### 4.5 PyTorch 实现

```python
# 在整个模型上做梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 查看裁剪前后的梯度范数
totalNorm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=5.0
)
```

> 更多梯度裁剪实践参见 [[NeuralNetwork/Tips/Techniques/GradientClippingGuide|梯度裁剪指南]]。

---

## 5. LSTM 中的 BPTT

### 5.1 为什么 LSTM 的 BPTT 更稳定

在 LSTM 中，关键的梯度路径是细胞状态 $C_t$：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

对比 Vanilla RNN：

- Vanilla RNN：$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}$（矩阵乘法 + 非线性压缩）
- LSTM：$\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)$（逐元素乘法，无矩阵乘法，无非线性）

LSTM 的梯度路径**消除了 $W_{hh}$ 的矩阵乘法和 $\tanh$ 的非线性压缩**，只留下一个逐元素的遗忘门因子。当 $f_t[j] \approx 1$ 时，梯度可以几乎无损地穿过这一步。

### 5.2 GRU 中的类似机制

GRU 的 $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ 提供了类似的机制：

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t)
$$

当 $z_t[j] \approx 0$ 时，梯度无损传递。

---

## 6. 总结

| 概念 | 要点 |
|------|------|
| BPTT | 梯度沿时间轴反向传播，$\partial L / \partial W_{hh} = \sum_t$（各时间步贡献之和） |
| 梯度连乘 | $\prod \frac{\partial h_j}{\partial h_{j-1}}$ 决定了长距离梯度的存亡 |
| Truncated BPTT | 每 $k$ 步截断梯度流，平衡计算量与长期依赖学习 |
| 梯度裁剪 | 限制梯度范数上限，只解决爆炸，不解决消失 |
| LSTM 的优势 | $C_t$ 更新路径是线性的逐元素操作，梯度可无损传递 |

> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
> 梯度消失/爆炸的详细分析：[[NeuralNetwork/Tips/Troubleshooting/GradientExplodingVanishing|梯度消失与爆炸]]
