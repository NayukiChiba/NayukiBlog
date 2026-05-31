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
status: published
---

## 1. BPTT 的核心问题

### 1.1 普通反向传播 vs BPTT

在普通 MLP 中，反向传播（Backpropagation）沿着网络层从输出向输入传播梯度。网络是一个有向无环图（DAG），梯度计算路径是唯一且确定的。

在 RNN 中，情况复杂得多——同一个参数（如 $W_{hh}$）在**每个时间步**都被使用。因此，$W_{hh}$ 的总梯度 = 它在所有时间步产生的梯度之和。

**"Through Time"的含义**：梯度不仅穿过网络层，还**穿过时间**——从最后一步反向流到第一步。

### 1.2 展开计算图

RNN 的前向计算沿时间轴展开后，等价于一个**深度 = 序列长度**的"虚拟前馈网络"，且所有时间步的对应层共享参数：

![RNNBPTT.png](https://img.yumeko.site/file/articles/RNNBPTT/RNNBPTT.webp)

所有从 $h_{t-1}$ 到 $h_t$ 的横向连接共享同一个 $W_{hh}$，所有从 $h_t$ 到 $y_t$ 的纵向连接共享同一个 $W_{hy}$，所有从 $x_t$ 到 $h_t$ 的纵向连接共享同一个 $W_{xh}$。

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

- 如果每一步的谱范数 $< 1$：连乘趋近于 0，导致**梯度消失**
- 如果每一步的谱范数 $> 1$：连乘趋近于无穷大，导致**梯度爆炸**

### 2.4 对 $W_{xh}$ 和 $W_{hy}$ 的梯度

$W_{hy}$ 的梯度相对简单——每次使用 $W_{hy}$ 是独立的（不通过隐藏状态传递），不涉及时间轴连乘：

$$
\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T
$$

$W_{xh}$ 的梯度类似 $W_{hh}$，也需要沿时间轴反向传播，但连乘只从 $t$ 到 $k$：

$$
\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{xh}}
$$

### 2.5 具体数值推导

以下用一个简化的小规模例子，完整计算 $\frac{\partial L_3}{\partial W_{hh}}$，展示 BPTT 中梯度连乘的实际衰减过程。

**设定：** $T = 3$，$d_h = 2$，$d_x = 2$。使用与 RecurrentLayer 第 4 节相同的权重矩阵：

$$
W_{hh} = \begin{pmatrix}
0.1 & 0.4 \\
-0.2 & 0.6
\end{pmatrix}
$$

隐藏状态序列（取自手算结果）：

$$
h_1 = \begin{pmatrix} 0.462 \\ 0.664 \end{pmatrix}, \quad
h_2 = \begin{pmatrix} 0.012 \\ 0.466 \end{pmatrix}, \quad
h_3 = \begin{pmatrix} 0.188 \\ 0.277 \end{pmatrix}
$$

设损失仅来自第 3 步，且 $\frac{\partial L_3}{\partial h_3} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$（简化假设）。

**步骤 1：计算 $\frac{\partial h_3}{\partial h_2}$**

先求 $\tanh'$ 的对角矩阵。需要 $\tanh$ 激活之前的线性值 $a_3 = W_{hh} \cdot h_2 + W_{xh} \cdot x_3 + b_h$。设 $a_3 = \begin{pmatrix} 0.19 \\ 0.28 \end{pmatrix}$（与 $h_3 = \tanh(a_3)$ 的值一致），则：

$$
\tanh'(0.19) = 1 - \tanh^2(0.19) \approx 1 - 0.188^2 \approx 0.965
$$
$$
\tanh'(0.28) = 1 - \tanh^2(0.28) \approx 1 - 0.277^2 \approx 0.923
$$
$$
\text{diag}(\tanh'(a_3)) = \begin{pmatrix} 0.965 & 0 \\ 0 & 0.923 \end{pmatrix}
$$
$$
\frac{\partial h_3}{\partial h_2} = \text{diag}(\tanh'(a_3)) \cdot W_{hh}
= \begin{pmatrix} 0.965 & 0 \\ 0 & 0.923 \end{pmatrix} \cdot \begin{pmatrix} 0.1 & 0.4 \\ -0.2 & 0.6 \end{pmatrix}
= \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix}
$$

**步骤 2：计算 $\frac{\partial h_2}{\partial h_1}$**

同理，设 $a_2 = \begin{pmatrix} 0.012 \\ 0.506 \end{pmatrix}$：

$$
\tanh'(0.012) \approx 0.9999, \quad \tanh'(0.506) \approx 0.744
$$
$$
\text{diag}(\tanh'(a_2)) = \begin{pmatrix} 0.9999 & 0 \\ 0 & 0.744 \end{pmatrix}
$$
$$
\frac{\partial h_2}{\partial h_1} = \begin{pmatrix} 0.9999 & 0 \\ 0 & 0.744 \end{pmatrix} \cdot \begin{pmatrix} 0.1 & 0.4 \\ -0.2 & 0.6 \end{pmatrix}
= \begin{pmatrix} 0.100 & 0.400 \\ -0.149 & 0.446 \end{pmatrix}
$$

**步骤 3：计算连乘**

$$
\frac{\partial h_3}{\partial h_1} = \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}
= \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix} \cdot \begin{pmatrix} 0.100 & 0.400 \\ -0.149 & 0.446 \end{pmatrix}
= \begin{pmatrix} -0.048 & 0.211 \\ -0.101 & 0.173 \end{pmatrix}
$$

矩阵元素量级从第一步的约 $0.1-0.55$ 下降到约 $0.05-0.21$，两步连乘已衰减到原来的约一半。

**步骤 4：计算 $\frac{\partial h_k}{\partial W_{hh}}$**

对于 $k=1$（最远的时间步）：

$$
\frac{\partial h_1}{\partial W_{hh}} \text{ 是一个 } (2 \times 4) \text{ 的张量（对 } 2\times 2 \text{ 矩阵的导数）}
$$

取其中一个元素简化说明：对 $W_{hh}[0,0]$，$\frac{\partial h_1[0]}{\partial W_{hh}[0,0]} = \tanh'(a_1[0]) \cdot h_0[0] = 0$（因 $h_0 = 0$）。

对于 $k=2$：

$$
\frac{\partial h_2[0]}{\partial W_{hh}[0,0]} \approx 0.9999 \cdot h_1[0] = 0.9999 \times 0.462 \approx 0.462
$$

对于 $k=3$：

$$
\frac{\partial h_3[0]}{\partial W_{hh}[0,0]} \approx 0.965 \cdot h_2[0] = 0.965 \times 0.012 \approx 0.0116
$$

**步骤 5：汇总 $W_{hh}[0,0]$ 的梯度**

$$
\frac{\partial L_3}{\partial W_{hh}[0,0]} = \sum_{k=1}^{3} \frac{\partial L_3}{\partial h_3} \cdot \left( \prod_{j=k+1}^{3} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}[0,0]}
$$

- $k=1$：因 $h_0 = 0$，贡献为 $0$
- $k=2$：$[1, 1] \cdot \begin{pmatrix} 0.0965 & 0.386 \\ -0.185 & 0.554 \end{pmatrix} \cdot \begin{pmatrix} 0.462 \\ \dots \end{pmatrix}$（量级约 $0.05-0.4$）
- $k=3$：$[1, 1] \cdot \begin{pmatrix} 0.0116 \\ \dots \end{pmatrix}$（量级约 $0.01$）

**关键观察：** 即使在这个仅有 3 步的例子中，$k=2$（回传2步）的贡献已经衰减到约 $0.05-0.4$ 的量级，而 $k=3$ 的贡献约为 $0.01$。当 $T=100$ 时，$k=1$ 的贡献经过 99 步连乘，每个元素约 $(0.5)^{99} \times 0.5 \approx 10^{-31}$ 量级——在 float32 精度下完全不可用。

**结论：** Vanilla RNN 中的梯度消失是结构性的。$\tanh$ 的导数被限制在 $(0, 1]$ 区间，$W_{hh}$ 的重复乘法进一步压缩梯度。即使 $W_{hh}$ 设计得再好，$\tanh$ 的非线性压缩效应无法被消除。

### 2.6 公式汇总

为便于对照，将 BPTT 的核心公式集中列出。给定总损失 $L = \sum_{t=1}^{T} L_t$：

**对 $W_{hh}$ 的梯度：**

$$
\boxed{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}}
$$

其中每个时间步 $t$ 的贡献展开为对该参数在各时刻使用的偏导求和：

$$
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

**梯度连乘项（关键的衰减因子）：**

$$
\boxed{\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(\tanh'(W_{hh} \cdot h_{j-1} + W_{xh} \cdot x_j + b_h)\big) \cdot W_{hh}}
$$

**对 $W_{hy}$ 的梯度（无时间轴连乘，计算最简单）：**

$$
\boxed{\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T}
$$

**对 $W_{xh}$ 的梯度（结构与 $W_{hh}$ 相同）：**

$$
\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{xh}}
$$

**LSTM 中的对比：**

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t) \quad \text{vs} \quad \frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh') \cdot W_{hh}
$$

核心差异：LSTM 的梯度路径是逐元素对角矩阵（无矩阵乘法、无非线性压缩），Vanilla RNN 的梯度路径包含 $W_{hh}$ 矩阵乘法和 $\tanh'$ 压缩。

---

## 3. Truncated BPTT

### 3.1 基本思路

对于极长的序列（如一本书有几百万字符），完整 BPTT 在计算上不可行（需要存储整个序列的中间激活值，内存爆炸）。Truncated BPTT 的做法是：

1. 将长序列切分成固定长度 $k$ 的片段（如 100 步）
2. 在每个片段内做完整的 BPTT
3. 将前一个片段的最终隐藏状态传给下一个片段，但**截断**梯度流

具体而言：片段 1 的 $[x_1, \dots, x_{100}]$ 做完 BPTT 后更新参数，然后片段 2 的 $[x_{101}, \dots, x_{200}]$ 接收 $h_{100}$ 作为初始隐藏状态继续计算，但 $h_{100}$ 的计算图已被截断，损失对片段 2 的梯度不会反向传播回片段 1 的参数。

### 3.2 PyTorch 实现

核心操作是 `detach()`——它创建一个与原张量共享数据但脱离计算图的新张量。梯度无法穿过 `detach()` 的边界，从而实现了跨片段的梯度截断。

```python
h = initHidden(batchSize)

for batch in dataloader:
    h = h.detach()
    output, h = model(batch, h)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

在每次迭代开始时调用 `h.detach()`，使得当前片段的 BPTT 只能反向传播到当前片段的第一时间步，无法继续往前追溯到上一个片段。

### 3.3 Truncation Length 的选择

| 截断长度 | 优点 | 缺点 |
|----------|------|------|
| 短（20-50） | 内存小、计算快 | 无法学习超过长度的依赖 |
| 中（100-200） | **平衡点** | 字符级语言模型的常用设置 |
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

### 4.4 阈值选择经验

阈值的选择取决于模型规模和任务难度：
- 较浅的模型（如单层 Vanilla RNN）梯度波动相对温和，阈值可设大一些（如 5.0）
- 较深或含 Attention 的模型梯度更容易爆炸，需要更保守的阈值（如 1.0）
- 实际使用中，1.0 到 5.0 是最常见的范围

### 4.5 PyTorch 实现

`torch.nn.utils.clip_grad_norm_` 对整个模型的所有参数梯度做就地裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

该函数同时返回裁剪前的梯度总范数，可用于监控训练稳定性。如果返回的范数值持续增长或剧烈波动，说明梯度爆炸风险较高，应降低阈值或检查模型结构。

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

> 回到主文档：[[NeuralNetwork/RNN/Foundations/RNNOverview|RNN 详解主文档]]
> 梯度消失/爆炸的详细分析：[[NeuralNetwork/Tips/Troubleshooting/GradientExplodingVanishing|梯度消失与爆炸]]
