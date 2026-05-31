---
title: RNN 概述：为什么需要循环神经网络
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
description: 从全连接网络和 CNN 的局限出发，系统理解 RNN 处理序列数据的核心思想——隐藏状态、时序依赖、参数共享，以及 RNN 适合解决的五类序列问题。
image: https://img.yumeko.site/file/blog/RNNOverview.webp
status: published
---

## 1. 序列数据的本质特征

### 1.1 什么是序列数据？

序列数据是指**元素的顺序至关重要**的数据——改变顺序就改变了含义。这与表格数据（每列独立、行顺序无关）和图像数据（空间结构重要但尺寸固定）有本质区别。

| 数据类型 | 元素顺序重要？ | 输入尺寸 | 例子 |
|----------|:---:|------|------|
| 表格数据 | 否 | 固定 | 房价预测特征 |
| 图像数据 | 是（空间） | 固定 | 猫/狗分类 |
| 序列数据 | 是（时间/位置） | **可变** | 文本、语音、股票 |

### 1.2 序列数据的三个核心特征

**特征一：变长**

不同样本的序列长度不同：
- 短评论文本："很好"（2 词）
- 长评论文本："这个商品我买了一个月了用起来非常好..."（50+ 词）

任何要求固定输入尺寸的模型都必须做截断或填充，这都会引入信息损失或噪声。

**特征二：有序依赖**

序列中相邻的元素不是独立的。在自然语言中，"形容词修饰名词"这种关系依赖于词的位置。在时间序列中，今天的股价依赖于过去几天的走势。

**特征三：上下文依赖**

同一个词在不同上下文中含义完全不同：
- "这个**苹果**很甜"（水果）
- "这个**苹果**很好用"（电子产品）

要正确理解一个词，必须参考它的上下文——但上下文可能在前面很远的位置（"我出生在**法国**...（中间50个词）...我会说流利的**法语**"）。

### 1.3 序列数据的普遍性

序列数据是机器学习中**最常见**的数据类型之一，涵盖以下领域：

- **自然语言（词或字符序列）**：文本分类（情感、主题）、机器翻译（序列到序列）、文本生成（序列到序列，自回归）、命名实体识别（序列到标签序列）
- **语音（声波采样序列）**：语音识别、语音合成
- **时间序列（数值序列）**：股票价格预测、天气或温度预测、传感器异常检测
- **视频（帧序列）**：动作识别、视频描述
- **生物序列（DNA 或蛋白质）**：基因功能预测

---

## 2. 前馈网络的三大局限

### 2.1 局限一：固定输入尺寸

MLP 的第一层权重矩阵形状固定（如 `(128, 784)`），只能接收 784 维的向量。CNN 虽然通过卷积核的参数共享可以处理不同尺寸的输入，但全连接层仍然要求固定的特征图大小。

对于序列数据，这意味着：
- 截断会丢失长序列尾部的重要信息
- 填充会浪费计算在无意义的零值上，且引入了人为的噪声模式

### 2.2 局限二：无状态记忆

MLP 的每一次前向传播是**完全独立的**。处理第 $t$ 个词时，输入中不包含第 $t-1$ 个词的信息。以句子 "I love you" 为例：处理 "I" 时 MLP 独立计算，处理 "love" 时 MLP 再次独立计算（不知道前面出现了 "I"），处理 "you" 时同样独立计算（不知道前面是 "I love"）。三个词各自的前向传播之间不存在信息交换。

CNN 虽然在空间维度有感受野，但卷积核的滑动是在**同一张图**内，而不是在**时间维度**上。CNN 的前次输入不会以隐藏状态的形式影响当前前向传播。

![RNNVSMLP.png](https://img.yumeko.site/file/articles/RNNOverview/RNNVSMLP.webp)

### 2.3 局限三：无法建模变长依赖

即使把序列截断为固定长度输入 MLP，网络也很难学习远距离的依赖关系。在一个扁平的全连接层中，第 1 个位置和第 50 个位置的输入被当作**同等距离**的特征来处理——输入的位置信息完全丢失，两个相距 49 步的输入在模型中无法被区分。

对于一些依赖关系跨越几十甚至上百步的任务（如长文档理解），前馈网络无法有效处理。

---

## 3. RNN 的三大核心思想

### 3.1 隐藏状态：网络的"记忆"

RNN 的核心创新是引入一个**隐藏状态**（Hidden State）$h$——在处理序列的每个元素时，网络不仅产生输出，还会**更新一个状态向量**，这个向量概括了从序列开头到当前位置的所有输入信息。

![HiddenDim.png](https://img.yumeko.site/file/articles/RNNOverview/HiddenDim.webp)

处理完第 t 个元素后，$h_t$ 包含了 $x_1, x_2, ..., x_t$ 的所有信息的压缩表示。这个状态会传递给第 t+1 步，作为"历史信息"参与下一步的计算。

### 3.2 参数时间共享：一套参数处理无限长序列

RNN 在每个时间步使用**完全相同**的参数：

$$
h_t = f_W(x_t, h_{t-1})
$$

其中 $W$（包括 $W_{xh}, W_{hh}, W_{hy}$）在所有时间步 $t = 1, 2, ..., T$ 中复用。

这带来了三个好处：
- **参数量与序列长度无关**：10 步 or 10000 步，参数量完全一样
- **自动泛化到未见过的长度**：训练时看 50 词的句子，测试时能处理 100 词的句子
- **模式在时间维度迁移**：在第 3 步通过权重更新习得的模式可应用至第 300 步

![ParamReuse.png](https://img.yumeko.site/file/articles/RNNOverview/ParamReuse.webp)

### 3.3 循环计算图：时间维度的深度网络

RNN 的计算图不是一个简单的 DAG（有向无环图），而是包含**循环**的——隐藏状态的值既依赖于当前输入，也依赖于上一时刻自身的值。

![UnrolledRNN.png](https://img.yumeko.site/file/articles/RNNOverview/UnrolledRNN.webp)

当我们沿时间轴"展开"这个循环时，RNN 等价于一个**深度 = 序列长度**的前馈网络，且所有层的参数绑定在一起（权重共享）。

---

## 4. RNN 能解决的五类问题

RNN 的输入输出灵活性远超 MLP 和 CNN。根据输入和输出的序列长度关系，RNN 可以解决以下五类问题：

![RNNMode.png](https://img.yumeko.site/file/articles/RNNOverview/RNNMode.webp)

### 4.1 One-to-One（一对一）

标准的 MLP 模式。输入一个固定向量 $x \in \mathbb{R}^{d_x}$，输出一个固定向量 $y \in \mathbb{R}^{d_y}$，没有时间维度：

$$
h = \tanh(W_{xh} \cdot x + b_h), \quad y = W_{hy} \cdot h + b_y
$$

**应用**：图像分类（但这不是 RNN 的强项）

### 4.2 One-to-Many（一对多）

输入一个向量 $x$，输出一个序列 $(y_1, y_2, \dots, y_T)$。输入作为"条件"或"种子"，$x$ 在每一步都参与计算：

$$
h_1 = \tanh(W_{xh} \cdot x + b_h)
$$
$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x + b_h), \quad t = 2, \dots, T
$$
$$
y_t = W_{hy} \cdot h_t + b_y
$$

**应用**：
- 图像描述生成（Image Captioning）：$x$ 为图像特征，输出为描述文本
- 音乐生成：$x$ 为风格或音调条件，输出为音符序列

### 4.3 Many-to-One（多对一）

输入一个序列 $(x_1, x_2, \dots, x_T)$，输出一个向量 $y$。这是**序列分类**的标准模式。网络逐步读取序列并更新隐藏状态，最终只取 $h_T$ 产生分类输出：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h), \quad t = 1, \dots, T
$$
$$
y = W_{hy} \cdot h_T + b_y
$$

**应用**：
- 情感分析：输入为评论词序列，输出为正面或负面
- 文本主题分类：输入为文章，输出为类别

### 4.4 Many-to-Many（同步）

输入一个序列 $(x_1, \dots, x_T)$，输出一个等长的序列 $(y_1, \dots, y_T)$。每个输入位置 $t$ 对应一个输出标签 $y_t$：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h), \quad t = 1, \dots, T
$$
$$
y_t = W_{hy} \cdot h_t + b_y
$$

**应用**：
- 词性标注（POS Tagging）：每个词对应一个词性标签
- 命名实体识别（NER）：每个词对应 B/I/O 标签
- 视频帧分类：每帧对应一个动作类别

### 4.5 Many-to-Many（异步）

输入一个序列 $(x_1, \dots, x_T)$，输出一个不同长度的序列 $(y_1, \dots, y_S)$，其中 $S$ 和 $T$ 可以不同。这是**Seq2Seq的编码器-解码器**模式。编码器将输入序列压缩为上下文向量 $c$（通常取 $h_T$），解码器基于 $c$ 自回归地生成输出：

编码器：
$$
h_t^{\text{enc}} = \tanh(W_{hh}^{\text{enc}} \cdot h_{t-1}^{\text{enc}} + W_{xh}^{\text{enc}} \cdot x_t + b_h^{\text{enc}}), \quad c = h_T^{\text{enc}}
$$

解码器：
$$
h_1^{\text{dec}} = \tanh(W_{hh}^{\text{dec}} \cdot c + W_{xh}^{\text{dec}} \cdot y_0 + b_h^{\text{dec}})
$$
$$
h_s^{\text{dec}} = \tanh(W_{hh}^{\text{dec}} \cdot h_{s-1}^{\text{dec}} + W_{xh}^{\text{dec}} \cdot y_{s-1} + b_h^{\text{dec}}), \quad s = 2, \dots, S
$$
$$
y_s = W_{hy}^{\text{dec}} \cdot h_s^{\text{dec}} + b_y^{\text{dec}}
$$

**应用**：
- 机器翻译：中文句子映射为英文句子（长度通常不同）
- 文本摘要：长文压缩为短文
- 语言模型的自回归生成也属于此模式

---

## 5. RNN 发展时间线

| 年份 | 事件 | 意义 |
|------|------|------|
| 1986 | Rumelhart 等人提出 BPTT | 建立了 RNN 训练的理论基础 |
| 1990 | Elman 提出 Simple RNN | 奠定了现代 RNN 的基本结构 |
| 1997 | Hochreiter & Schmidhuber 提出 LSTM | 用门控机制大幅缓解梯度消失 |
| 1997 | Schuster & Paliwal 提出双向 RNN | RNN 可同时利用上文和下文两个方向的序列信息 |
| 2014 | Cho 等人提出 GRU | LSTM 的精简版，参数量减少 25% |
| 2014 | Sutskever 等人提出 Seq2Seq | 编码器-解码器统一了多种序列转换任务 |
| 2015 | Bahdanau 等人引入 Attention | 使模型可直接访问任意位置的序列信息，为 Transformer 铺路 |
| 2017 | Vaswani 等人提出 Transformer | 用自注意力取代循环，开启新时代 |

---

## 6. RNN vs MLP vs CNN：一张表看清差异

![RadarChart.png](https://img.yumeko.site/file/articles/RNNOverview/RadarChart.webp)

| 维度 | MLP | CNN | RNN |
|------|-----|-----|-----|
| 核心操作 | 矩阵乘法 | 卷积（滑动窗口） | 循环状态更新 |
| 输入类型 | 固定长度向量 | 固定尺寸张量 | **变长序列** |
| 参数共享 | 无（每层独立参数） | 空间维度（卷积核滑动） | **时间维度**（同参数处理所有步） |
| 感受野/记忆范围 | 全连接（看全部输入） | 局部到全局（层次化） | **逐步累积**（隐藏状态传递） |
| 并行性 | 高 | 高 | **低**（时序依赖） |
| 参数量 | $O(n_{in} \cdot n_{out})$ | $O(K^2 \cdot C_{in} \cdot C_{out})$ | $O(h^2 + h \cdot i)$ |
| 序列建模 | 差 | 差 | **设计初衷** |
| 典型应用 | 表格数据、简单分类 | 图像、视频 | **文本、语音、时序** |

---

## 7. 后续导航

本文是 RNN 系列的总览。深入理解各个组件请阅读：

- [[NeuralNetwork/RNN/Foundations/RecurrentLayer|循环层详解]] — Vanilla RNN 的数学本质
- [[NeuralNetwork/RNN/Foundations/LongShortTermMemory|LSTM 详解]] — 门控机制与细胞状态
- [[NeuralNetwork/RNN/Foundations/GatedRecurrentUnit|GRU 详解]] — LSTM 的高效替代
- [[NeuralNetwork/RNN/Foundations/RNNBPTT|BPTT 详解]] — 循环网络的反向传播
- [[NeuralNetwork/RNN/Foundations/EmbeddingLayer|嵌入层详解]] — 从离散符号到连续向量
- [[NeuralNetwork/RNN/Foundations/AttentionMechanism|注意力机制详解]] — 为 RNN 提供动态上下文聚焦能力

回到主文档：[[NeuralNetwork/RNN/Foundations/RNNOverview|RNN 详解主文档]]
