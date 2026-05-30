---
title: GRU 详解：LSTM 的高效替代方案
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
description: 详细拆解 GRU 的重置门和更新门机制，理解 GRU 如何用更少的参数实现与 LSTM 接近的效果，并对比两者优劣和适用场景。
image: https://img.yumeko.site/file/blog/GatedRecurrentUnit.webp
status: draft
---

## 1. GRU 的诞生背景

### 1.1 Seq2Seq 的效率需求

2014 年，Cho 等人在开发 Seq2Seq（序列到序列）神经机器翻译模型时，需要一个高效的门控 RNN 单元。LSTM 虽然效果很好，但三个门 + 独立细胞状态的计算开销较大——在 Seq2Seq 中需要**两个** RNN（编码器和解码器），参数量和训练时间翻倍。

于是他们设计了 GRU（Gated Recurrent Unit）——在保留门控机制核心优势的前提下，大幅简化结构。

### 1.2 GRU 的简化策略
![LSTMVSGRU.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/LSTMVSGRU.webp)

对比 LSTM，GRU 做了两个关键简化：

| LSTM 组件 | GRU 的简化方式 |
|-----------|---------------|
| 遗忘门 + 输入门（2 个门） | 合并为 **更新门** $z_t$（1 个门） |
| 独立的细胞状态 $C_t$ + 隐藏状态 $h_t$ | 去掉 $C_t$，让 $h_t$ 同时承担"记忆"和"输出"两个角色 |
| 输出门 | 去掉，$h_t$ 本身就是对外输出 |

结果：GRU 只有 **2 个门**、**没有独立细胞状态**，参数量约为 LSTM 的 75%。

---

## 2. GRU 的完整公式系统

![GRUCell.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/GRUCell.webp)

### 2.1 重置门（Reset Gate）

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

- $\sigma$（Sigmoid）输出范围 (0, 1)
- 作用：控制在计算**候选状态**时，忽略多少过去的隐藏状态

$$
r_t[j] \approx 0 \text{：几乎完全忽略 } h_{t-1}[j] \text{（"重置"该维度）}
$$
$$
r_t[j] \approx 1 \text{：完整保留 } h_{t-1}[j] \text{（正常读取过去信息）}
$$

例如，在处理一个新句子时，重置门使模型在计算候选状态时降低上一句信息的影响力，避免上句的语法结构干扰下句的表示。

### 2.2 更新门（Update Gate）

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

- 作用：控制在"保留旧状态"和"写入新候选"之间如何平衡
- $z_t[j]$ 是该维度的"更新比例"
- $z_t[j] \approx 0$：几乎不更新（$h_t[j] \approx h_{t-1}[j]$，完全保留旧信息）
- $z_t[j] \approx 1$：完全更新（$h_t[j] \approx \tilde{h}_t[j]$，几乎完全采纳新内容）

**直觉**：更新门相当于合并了 LSTM 的遗忘门和输入门——"遗忘多少旧信息"和"写入多少新信息"用同一个标量控制，因为写入量 = 1 - 保留量。

### 2.3 候选隐藏状态

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

- 先用重置门 $r_t$ 对 $h_{t-1}$ 做**逐元素筛选**（决定读取过去的哪些部分）
- 将筛选后的 $h_{t-1}$ 与 $x_t$ 拼接
- 通过 $\tanh$ 产生候选值（范围 -1 ~ 1）

**与 LSTM 候选值的关键区别**：LSTM 在计算 $\tilde{C}_t$ 时使用**完整的** $h_{t-1}$（不经过筛选），依赖输入门 $i_t$ 来控制候选值的写入量；GRU 在计算 $\tilde{h}_t$ 时先通过重置门筛选 $h_{t-1}$，这影响了候选值本身的内容。

### 2.4 最终隐藏状态：线性插值

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

这是 GRU 最关键的设计。更新门 $z_t$ 在"保留"和"更新"之间做**线性插值**：

| $z_t$ | $h_t$ | 含义 |
|:---:|------|------|
| 0 | $h_{t-1}$ | 完全保留旧状态（信息不变） |
| 0.3 | $0.7 \cdot h_{t-1} + 0.3 \cdot \tilde{h}_t$ | 主要保留旧状态，少量更新 |
| 0.7 | $0.3 \cdot h_{t-1} + 0.7 \cdot \tilde{h}_t$ | 大量采纳新信息，少量保留旧信息 |
| 1 | $\tilde{h}_t$ | 完全覆盖为新候选值 |

**为什么这能缓解梯度消失？** 对 $h_{t-1}$ 求偏导：

$$
\frac{\partial h_t}{\partial h_{t-1}} \approx \text{diag}(1 - z_t)
$$

当 $z_t[j] \approx 0$（该维度几乎不更新）时，梯度几乎无损地从 $h_t$ 传到 $h_{t-1}$。对比 Vanilla RNN 的 $\text{diag}(\tanh') \cdot W_{hh}$（梯度被压缩和旋转），GRU 的梯度路径是一条**直通的、逐元素独立控制的线性路径**。

![GradinLSTM_GRU.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/GradinLSTM_GRU.webp)

---

## 3. 手算 GRU（与 LSTM 同输入对比）

![LSTMGRUEncode.png](https://img.yumeko.site/file/articles/GatedRecurrentUnit/LSTMGRUEncode.webp)

使用与 LSTM 章节相同简化的设置：input=2, hidden=2。

### 3.1 重置门

设 $W_r \cdot [h_0, x_1] + b_r = \begin{pmatrix} 0.3 \\ -0.5 \end{pmatrix}$：

$$
r_1 = \sigma\left( \begin{pmatrix} 0.3 \\ -0.5 \end{pmatrix} \right) \approx \begin{pmatrix} 0.57 \\ 0.38 \end{pmatrix}
$$

$r_1 = [0.57, 0.38]$ 意味着：维度 0 保留 57% 的过去信息，维度 1 只保留 38%（重置程度较高）。

### 3.2 候选隐藏状态

先筛选 $h_0$（$h_0 = [0, 0]$）：

$$
r_1 \odot h_0 = \begin{pmatrix} 0.57 \\ 0.38 \end{pmatrix} \odot \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

初始步 $h_0 = 0$，重置门的筛选没有意义。设 $W \cdot [r_1 \odot h_0, x_1] + b = \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix}$：

$$
\tilde{h}_1 = \tanh\left( \begin{pmatrix} 0.6 \\ 1.2 \end{pmatrix} \right) \approx \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
$$

### 3.3 更新门

设 $W_z \cdot [h_0, x_1] + b_z = \begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix}$：

$$
z_1 = \sigma\left( \begin{pmatrix} 0.4 \\ 0.4 \end{pmatrix} \right) \approx \begin{pmatrix} 0.60 \\ 0.60 \end{pmatrix}
$$

### 3.4 最终隐藏状态

$$
h_1 = (1 - 0.60) \odot \begin{pmatrix} 0 \\ 0 \end{pmatrix} + 0.60 \odot \begin{pmatrix} 0.537 \\ 0.834 \end{pmatrix}
= \begin{pmatrix} 0.322 \\ 0.500 \end{pmatrix}
$$

注意：因为 $h_0 = 0$，$h_1$ 完全由新写入的候选值决定（但被 $z_1 = 0.6$ 打了 6 折）。

**时间步 2**：$h_1 = [0.322, 0.500]$，$x_2 = [0, 1]$

设 $r_2 \approx [0.45, 0.65]$：

$$
r_2 \odot h_1 = \begin{pmatrix} 0.45 \times 0.322 \\ 0.65 \times 0.500 \end{pmatrix} = \begin{pmatrix} 0.145 \\ 0.325 \end{pmatrix}
$$

设 $\tilde{h}_2 \approx [0.200, 0.450]$，$z_2 \approx [0.40, 0.80]$：

$$
h_2 = \begin{pmatrix} 0.60 \\ 0.20 \end{pmatrix} \odot \begin{pmatrix} 0.322 \\ 0.500 \end{pmatrix}
+ \begin{pmatrix} 0.40 \\ 0.80 \end{pmatrix} \odot \begin{pmatrix} 0.200 \\ 0.450 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0.193 \\ 0.100 \end{pmatrix} + \begin{pmatrix} 0.080 \\ 0.360 \end{pmatrix}
= \begin{pmatrix} 0.273 \\ 0.460 \end{pmatrix}
$$

**与 LSTM 同输入对比**（LSTM §3.3 的结果 $C_2 = [0.047, 0.741]$）：

| 维度 | LSTM $C_2$ | GRU $h_2$ | 差异说明 |
|------|:---:|:---:|------|
| 0 | 0.047 | 0.273 | GRU 保留更多旧信息（更新门 z=0.40 → 保留 60%） |
| 1 | 0.741 | 0.460 | LSTM 累积更多（遗忘门 0.55 + 输入门 0.80 × 0.60） |

GRU 的输出在两个维度之间更"均衡"，LSTM 的维度间差异更大——这反映了 LSTM 更精细的门控能力。

---

## 4. GRU vs LSTM 完整对比

### 4.1 结构对比

| | LSTM | GRU |
|------|------|------|
| 门控数量 | 3（遗忘+输入+输出） | 2（重置+更新） |
| 状态数量 | 2（$C_t$ + $h_t$） | 1（$h_t$ 身兼两职） |
| 梯度传递路径 | $C_{t-1} \to C_t$（线性） | $h_{t-1} \to h_t$（线性插值） |
| 是否有输出门 | ✅（控制对外暴露多少） | ❌（$h_t$ 直接输出） |
| 参数量（单层） | $4 \times (h^2 + h \cdot i + h)$ | $3 \times (h^2 + h \cdot i + h)$ |

### 4.2 参数量具体对比

以 input=256, hidden=256 为例：

| 组件 | LSTM | GRU |
|------|------:|------:|
| weight_ih | 4×256×256 = 262,144 | 3×256×256 = 196,608 |
| weight_hh | 4×256×256 = 262,144 | 3×256×256 = 196,608 |
| bias | 4×256×2 = 2,048 | 3×256×2 = 1,536 |
| **单层合计** | **526,336** | **394,752** |
| GRU/LSTM 比例 | 100% | **75%** |

### 4.3 性能对比

| 维度 | LSTM | GRU | 推荐场景 |
|------|:---:|:---:|------|
| 长序列（>100步） | 略优 | 良好 | LSTM 的独立 CEC 有优势 |
| 短序列（<50步） | 良好 | 良好 | 两者差异不大 |
| 小数据集 | 可能过拟合 | **更好**（参数少） | GRU |
| 大数据集 | **略优** | 良好 | LSTM |
| 训练速度 | 较慢 | **较快** | GRU |
| 需精细门控 | **支持** | 有限 | LSTM（有独立输出门） |

### 4.4 社区共识

> "GRU 和 LSTM 在大多数任务上的差异很小。如果资源有限或数据量小，选 GRU；如果有大量数据和计算资源，LSTM 可能略占优势。不确定时，两个都试，选验证集上更好的那个。"
>
> — 来自社区实践经验总结

---

## 5. PyTorch 代码

```python
import torch.nn as nn

gru = nn.GRU(
    input_size=300,
    hidden_size=256,
    num_layers=2,
    dropout=0.5,
    bidirectional=True,
    batch_first=True,
)

# GRU 只需要 h_0（不需要 c_0）
B, L = 64, 128
h0 = torch.zeros(2 * 2, B, 256)  # 2层×2方向, B, 256

output, hn = gru(x, h0)
# output: (B, L, 512)
# hn: (4, B, 256)
```

与 LSTM 的接口几乎一样——只少了一个 `c_0` 参数和一个 `c_n` 返回值。

### 5.1 Char-RNN 中切换 GRU

```python
def createModel(rnnType, vocabSize, embeddingDim, hiddenDim, numLayers, dropout):
    if rnnType == "GRU":
        return CharGRU(vocabSize, embeddingDim, hiddenDim, numLayers, dropout)
    elif rnnType == "LSTM":
        return CharLSTM(vocabSize, embeddingDim, hiddenDim, numLayers, dropout)
    else:
        return CharRNN(vocabSize, embeddingDim, hiddenDim, numLayers, dropout)
```

---

## 6. 总结

GRU 是 LSTM 的简化版本——用两个门（重置+更新）+ 统一的隐藏状态，实现了接近 LSTM 的长距离依赖学习能力，同时减少了 25% 的参数量。

**选择建议**：
- 不确定时 → LSTM（更成熟的默认选择）
- 资源/时间受限时 → GRU
- 小数据集 → GRU（更不容易过拟合）
- 需要精细控制输出 → LSTM（有独立的输出门）

> 对比 LSTM 参见 [[NeuralNetwork/RNN/Foundations/LongShortTermMemory|LSTM 详解]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
