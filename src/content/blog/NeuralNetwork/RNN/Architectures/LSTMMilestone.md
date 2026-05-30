---
title: LSTM 里程碑：1997 年改变序列学习的发明
date: 2026-05-29
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: 回顾 Hochreiter 和 Schmidhuber 在 1997 年提出 LSTM 的历史背景、CEC 核心洞察、遗忘门的后续改进，以及 LSTM 如何在沉寂十余年后成为深度学习时代的基石。
image: https://img.yumeko.site/file/blog/LSTMMilestone.webp
status: draft
---

## 1. 诞生背景：梯度消失的理论分析

### 1.1 Hochreiter 的毕业论文 (1991)

Sepp Hochreiter 在 1991 年的毕业论文（导师：Jürgen Schmidhuber）中对 RNN 的梯度消失问题做了严格的数学分析。这是该问题首次被**系统性地研究**。

**核心发现**：在 Vanilla RNN 中，反向传播的误差信号随时间指数衰减——

$$
\left|\left| \frac{\partial L_T}{\partial h_1} \right|\right| \propto \prod_{t=2}^{T} \lambda_t
$$

其中 $\lambda_t$ 是 $\frac{\partial h_t}{\partial h_{t-1}}$ 的特征值。当 $|\lambda_t| < 1$ 时（在 Vanilla RNN 中几乎总是），连乘导致指数衰减。

这个分析为 LSTM 的设计提供了明确的方向：**创造一个 $\lambda_t \approx 1$ 的梯度路径**。

### 1.2 1990 年代中期的困境

到 1990 年代中期，RNN 研究陷入了困境：
- 理论分析已经证明了梯度消失的不可避免性
- Elman RNN 的实践经验证实了 10-20 步的实用上限
- 但尚无有效的解决方案

---

## 2. CEC：LSTM 的核心洞察

### 2.1 常量误差传送带

Hochreiter 和 Schmidhuber 在 1997 年的突破性想法是：**如果信息可以通过一个没有非线性、没有矩阵乘法的线性单元传递，梯度就能无损流动**。

他们把这个线性单元称为 **Constant Error Carousel（CEC，常量误差传送带）**。

CEC 的核心是：

$$
C_t = C_{t-1} + \text{(门控输入)}
$$

注意这里的更新方式是**加法**，而不是 Vanilla RNN 的矩阵乘法 + 非线性。对 $C_{t-1}$ 的梯度：

$$
\frac{\partial C_t}{\partial C_{t-1}} = 1
$$

当没有任何新信息写入时，梯度以 **1.0 的保真度**穿过这一时间步。

### 2.2 从 CEC 到细胞状态

CEC 就是现代 LSTM 中**细胞状态 $C_t$** 的前身。现代版本中，遗忘门的加入使得更新变为：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

CEC 的理想恒等映射（梯度=1）被放松为遗忘门控制的可调节映射（梯度= $f_t$，可学习）。这既保留了梯度无损流动的核心优势，又赋予了网络主动"遗忘"的能力。

---

## 3. 原始 LSTM（1997 年版）

### 3.1 结构：只有输入门和输出门

原始 LSTM 论文中的结构与现代 LSTM 有一个关键差异：**没有遗忘门**。

原始 LSTM 只有两个门：

| 门 | 作用 | 公式 |
|----|------|------|
| 输入门 | 控制写入 CEC 的信息量 | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| 输出门 | 控制从 CEC 读取的信息量 | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ |

细胞状态更新：$C_t = C_{t-1} + i_t \odot \tilde{C}_t$（纯加法，无法主动遗忘）

隐藏状态输出：$h_t = o_t \odot \tanh(C_t)$

### 3.2 原始 LSTM 如何处理"遗忘"？

没有遗忘门意味着 $C_t$ 只能**增长**（因为 $i_t \odot \tilde{C}_t$ 的加和）。对于无限长的序列，$C_t$ 会无限增长。

实践中通过以下方式缓解：
- 输入门 $i_t$ 在大多数时间步接近 0（自然"关闭"写入）
- $\tilde{C}_t$ 在 Tanh 后范围有限（-1 到 1）
- 输出门 $o_t$ 过滤 $C_t$ 中不需要暴露的内容

但这些是间接的补救措施，不是对遗忘的主动控制。

---

## 4. 遗忘门的加入（2000 年）

### 4.1 Gers 等人的关键改进

2000 年，Felix Gers（Schmidhuber 的学生）在博士论文中提出给 LSTM 加上**遗忘门**：

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

这个看似微小的改动，使 LSTM 能够**主动选择**遗忘旧的、不再需要的信息——这对于处理流式数据（无限长的序列）至关重要。

### 4.2 遗忘门的初始化策略

Gers 等人发现，遗忘门的偏置 $b_f$ 不应初始化为 0——这会让 $f_t$ 的初始值约 0.5，导致训练早期就大量遗忘信息。

更好的做法是将 $b_f$ 初始化为一个较大的正值（如 1.0），使得训练初期 $f_t \approx 0.73$（$\sigma(1) \approx 0.73$），倾向于保留信息。

```python
# PyTorch 中设置 LSTM 遗忘门偏置
lstm = nn.LSTM(input_size, hidden_size)
# 将 bias 中遗忘门对应的位置初始化为 1
# weight_ih 和 weight_hh 的排列: [i, f, C, o] × hidden_size
```

实际上 PyTorch 的默认 LSTM 初始化已经在内部做了类似的优化。

---

## 5. 沉寂与复兴（1997-2015）

![LSTMTimeline.png](https://img.yumeko.site/file/articles/LSTMMilestone/LSTMTimeline.webp)

### 5.1 为什么 LSTM 沉睡了十余年？

LSTM 在 1997 年发表后，并没有立即改变世界。直到 2010 年代才迎来广泛采用，原因包括：

- **计算资源不足**：1997 年的硬件难以训练大规模 LSTM
- **数据量不足**：LSTM 需要大量数据来学习有效的门控策略
- **优化技术不成熟**：Gradient clipping、更好的初始化、优化器（Adam）等都是后来发展的
- **学术惯性**：NLP 领域长期被基于规则和统计方法主导

### 5.2 复兴的关键节点

| 年份 | 事件 | 意义 |
|------|------|------|
| 2011 | Graves 等人用 LSTM 赢得手写识别比赛 | 首次大规模 LSTM 应用 |
| 2013 | Graves 等人用 LSTM 实现语音识别突破 | 在 TIMIT 上达到当时最优 |
| 2014 | Sutskever 用 LSTM Seq2Seq 做机器翻译 | 接近当时最好的统计机器翻译系统 |
| 2015 | Google 用 LSTM 做语音识别，降低 49% 词错率 | 工业级应用的标志 |
| 2016 | Google 神经机器翻译系统（GNMT）上线 | LSTM Seq2Seq 进入数亿用户的产品 |

到 2016 年，LSTM 已经成为序列学习的**事实标准**，从语音识别到机器翻译再到文本生成，无处不在。

---

## 6. 历史定位与影响

### 6.1 LSTM 的影响

即使在 Transformer 主导的 2020 年代，LSTM 的设计思想仍有持续影响：

1. **门控机制**成为神经网络设计的核心原语
2. **CEC 的设计思想**影响了后来的 Highway Networks 和 ResNet（跳跃连接）
3. **细胞状态/隐藏状态分离**启发了 Neural Turing Machine 和 Memory Networks 等可微分外部记忆
4. **Attention 最初就是为了增强 LSTM Seq2Seq 而设计的**

### 6.2 LSTM vs Transformer：并非替代而是演进

Transformer 在长序列、并行化方面有明显优势，但 LSTM 在小数据、流式处理、推理效率方面仍有竞争力。2020 年代，一些工作甚至重新发现了线性 RNN（如 Mamba、RWKV）的价值——它们在推理时比 Transformer 更高效。

LSTM 的核心理念——"为梯度提供一条无损传递的路径"——超越了具体架构，为深度学习设计提供了重要参考。

---

> LSTM 的详细数学解析：[[NeuralNetwork/RNN/Foundations/LongShortTermMemory|LSTM 详解]]
> 架构对比：[[NeuralNetwork/RNN/Architectures/ArchitectureComparison|RNN 架构对比]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
