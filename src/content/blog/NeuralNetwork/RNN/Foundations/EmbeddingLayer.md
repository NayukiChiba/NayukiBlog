---
title: 嵌入层详解：从离散符号到稠密向量
date: 2026-05-28
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
  - NLP
description: 深入理解词嵌入和字符嵌入的数学本质、训练过程、padding_idx 的特殊处理以及嵌入维度选择的经验法则。
image: https://img.yumeko.site/file/blog/EmbeddingLayer.webp
status: published
---

## 1. 从离散到连续：为什么需要 Embedding？

![Dense.png](https://img.yumeko.site/file/articles/EmbeddingLayer/Dense.webp)

### 1.1 One-Hot 编码的三个死穴

处理文本的第一步是将字符或词转换为数值表示。设词表（vocabulary）为 $\mathcal{V}$，大小为 $V = |\mathcal{V}|$，每个词 $w_i$ 分配一个整数索引 $i \in \{0, 1, \dots, V-1\}$。

最朴素的向量化方式是 **one-hot 编码**——将索引 $i$ 映射为一个 $V$ 维的稀疏向量 $\mathbf{o}_i \in \{0, 1\}^V$：

$$
\mathbf{o}_i[j] = \begin{cases}
1 & \text{if } j = i \\
0 & \text{otherwise}
\end{cases}, \quad j \in \{0, 1, \dots, V-1\}
$$

该向量满足 $\sum_j \mathbf{o}_i[j] = 1$，$\|\mathbf{o}_i\|_0 = 1$——即只有一个位置是 1，其余全是 0。

这种表示有三个致命问题。

**死穴一：维度灾难**

中文词表可能包含 $V = 5\times 10^4$ 个词，每个词用 5 万维向量表示。以 float32（4 字节）存储，一个词占 $50000 \times 4 = 200\text{KB}$。一批 64 个句子、每句 128 个词的内存占用为：

$$
64 \times 128 \times 50000 \times 4 \text{ bytes} \approx 1.6\text{ GB}
$$

这仅是存储输入的开销，尚未计入后续的矩阵运算。

**死穴二：极度稀疏**

每个向量 $\mathbf{o}_i$ 的 $L_0$ 范数恒为 1，即 $V-1$ 个位置（99.998%）始终为零。任何与 one-hot 向量相乘的操作实质是在做稀疏查找，但底层硬件仍按稠密矩阵执行——大量计算资源浪费在与零相乘上。

**死穴三：无语义信息**

对于任意两个不同的词 $w_i \neq w_j$，其 one-hot 向量的内积恒为零：

$$
\mathbf{o}_i^T \cdot \mathbf{o}_j = 0 \quad (\forall i \neq j)
$$

余弦相似度则为 $\cos(\mathbf{o}_i, \mathbf{o}_j) = 0$。"国王"与"女王"的向量距离等于"国王"与"苹果"的距离。模型无法从输入表示中利用词与词之间的语义关系，所有关于词义的归纳必须从零学习。

### 1.2 Embedding 的解决方案

Embedding 的核心思想：将稀疏的 $V$ 维 one-hot 向量"压缩"为稠密的 $d_e$ 维实值向量（$d_e \ll V$）。

具体而言，引入一个嵌入矩阵 $E \in \mathbb{R}^{V \times d_e}$（可学习参数），词 $w_i$ 的嵌入向量 $\mathbf{e}_i$ 即为 $E$ 的第 $i$ 行：

$$
\mathbf{e}_i = E[i] \in \mathbb{R}^{d_e}
$$

对比两种表示：

- One-hot：$\mathbf{o}_i \in \mathbb{R}^{V}$，$\|\mathbf{o}_i\|_0 = 1$，每个维度对应"词表中某个特定词"
- Embedding：$\mathbf{e}_i \in \mathbb{R}^{d_e}$，$\|\mathbf{e}_i\|_0 = d_e$（稠密），每个维度没有预先定义的单一含义

稠密向量的优势在于**分布式表示（distributed representation）**——语义信息分布在 $d_e$ 个维度上，不同的激活模式可以编码指数级的信息量（理论上 $d_e$ 个维度可以表示 $2^{d_e}$ 种模式，远大于 $d_e$）。在实际训练中，语义相近的词会自然收敛到向量空间中相近的位置（由损失函数和梯度优化驱动）。

---

## 2. Embedding 的数学本质

### 2.1 查找表，而非矩阵乘法

从定义上看，给定索引序列 $\mathbf{x} = [x_1, x_2, \dots, x_T]$，嵌入操作输出：

$$
\text{Embedding}(\mathbf{x}) = [E[x_1], E[x_2], \dots, E[x_T]] \in \mathbb{R}^{T \times d_e}
$$

即直接从 $E$ 中取出对应索引的行，不做矩阵乘法。数学上这等价于将 one-hot 矩阵 $O \in \{0, 1\}^{T \times V}$（每行是一个 one-hot 向量）与 $E$ 相乘：

$$
\text{Embedding}(\mathbf{x}) = O \cdot E
$$

但在实现中，PyTorch 的 `nn.Embedding` 直接使用整数索引查表，避免了构造稀疏 one-hot 矩阵的计算和内存开销。

### 2.2 手算嵌入层：完整的数值例子

下面用一个极小规模的具体例子展示嵌入层的**查找过程**和**梯度更新过程**。假设：

- 词表大小 $V = 4$（四个词：$\langle\text{PAD}\rangle$, "猫", "狗", "鱼"）
- 嵌入维度 $d_e = 3$
- padding_idx = 0

#### 前向传播：查表

嵌入矩阵 $E \in \mathbb{R}^{4 \times 3}$ 的初始化（为便于手算使用小数值，实际 PyTorch 的初始化方式见下文）：

$$
E = \begin{pmatrix}
0.00 & 0.00 & 0.00 \\
0.50 & -0.30 & 0.80 \\
-0.20 & 0.70 & 0.10 \\
0.40 & -0.50 & -0.60
\end{pmatrix}
$$

- 第 0 行：$\langle\text{PAD}\rangle$（padding_idx，强制置零）
- 第 1 行："猫"
- 第 2 行："狗"
- 第 3 行："鱼"

输入一个批次（batch=1, seq_len=5）：

$$
\mathbf{x} = [1,\ 2,\ 3,\ 0,\ 1]
$$

对应的文本为 `["猫", "狗", "鱼", <PAD>, "猫"]`。

嵌入层的前向传播就是**逐位置查表**：

$$
\begin{aligned}
\text{位置 0: } &x_0 = 1 \quad \Rightarrow \quad E[1] = [0.50,\ -0.30,\ 0.80] &（\text{取出第 1 行}）\\
\text{位置 1: } &x_1 = 2 \quad \Rightarrow \quad E[2] = [-0.20,\ 0.70,\ 0.10] &（\text{取出第 2 行}）\\
\text{位置 2: } &x_2 = 3 \quad \Rightarrow \quad E[3] = [0.40,\ -0.50,\ -0.60] &（\text{取出第 3 行}）\\
\text{位置 3: } &x_3 = 0 \quad \Rightarrow \quad E[0] = [0.00,\ 0.00,\ 0.00] &（\text{PAD 位置强制输出零向量}）\\
\text{位置 4: } &x_4 = 1 \quad \Rightarrow \quad E[1] = [0.50,\ -0.30,\ 0.80] &（\text{再次查第 1 行}）
\end{aligned}
$$

最终输出张量形状为 $(1, 5, 3)$：

$$
\text{output} = \begin{pmatrix}
0.50 & -0.30 & 0.80 \\
-0.20 & 0.70 & 0.10 \\
0.40 & -0.50 & -0.60 \\
0.00 & 0.00 & 0.00 \\
0.50 & -0.30 & 0.80
\end{pmatrix}
$$

**关键观察**：
- 位置 0 和位置 4 的索引相同（都是 1），取出了**完全相同**的向量——同一行被查了两次
- 位置 3 是 PAD，输出全零——但注意这个零向量的计算图仍然连接着 $E[0]$，反向传播时梯度仍然会经过这里
- "猫"这个词不管出现在句子的第 1 个位置还是第 5 个位置，嵌入向量都一样——嵌入只关心"是哪个词"，不关心"在哪出现"

#### 反向传播：梯度更新

假设这批数据经过 RNN 后计算了损失 $L$，反向传播到嵌入层输出位置的梯度（$\frac{\partial L}{\partial \text{output}}$）为：

$$
\text{grad\_output} =
\begin{pmatrix}
0.10 & -0.20 & 0.05 \\
-0.15 & 0.08 & 0.12 \\
0.20 & -0.10 & -0.05 \\
0.08 & 0.03 & 0.02 \\
-0.05 & 0.15 & 0.10
\end{pmatrix}
$$

- 第 0 行：对位置 0 的梯度
- 第 1 行：对位置 1 的梯度
- 第 2 行：对位置 2 的梯度
- 第 3 行：对位置 3（PAD）的梯度
- 第 4 行：对位置 4 的梯度

嵌入层的反向传播是将每个位置的梯度**累加**回对应的嵌入矩阵行：

- **$E[0]$（$\langle\text{PAD}\rangle$）**：仅位置 3 用到 index=0，回传梯度为 $[0.08,\ 0.03,\ 0.02]$。但因为 `padding_idx=0`，此梯度在优化器更新前被**清零**。
- **$E[1]$（"猫"）**：位置 0 + 位置 4 都用到 index=1，梯度累加：
  $$
  \text{grad}[1] = [0.10,\ -0.20,\ 0.05] + [-0.05,\ 0.15,\ 0.10] = [0.05,\ -0.05,\ 0.15]
  $$
- **$E[2]$（"狗"）**：仅位置 1 用到 index=2，回传梯度为 $[-0.15,\ 0.08,\ 0.12]$
- **$E[3]$（"鱼"）**：仅位置 2 用到 index=3，回传梯度为 $[0.20,\ -0.10,\ -0.05]$

梯度汇总到嵌入矩阵的各行：

$$
\text{grad\_E} = \begin{pmatrix}
0.00 & 0.00 & 0.00 \\
0.05 & -0.05 & 0.15 \\
-0.15 & 0.08 & 0.12 \\
0.20 & -0.10 & -0.05
\end{pmatrix}
$$

- 第 0 行：padding_idx 清零
- 第 1 行："猫"在两个位置的梯度之和
- 第 2 行："狗"的梯度
- 第 3 行："鱼"的梯度

假设学习率 $\eta = 0.1$，用 SGD 更新后的嵌入矩阵：

$$
E_{\text{new}} = E - \eta \cdot \text{grad\_E}
$$

逐行计算：

$$
\begin{aligned}
E_{\text{new}}[0] &= [0.00,\ 0.00,\ 0.00] - 0.1 \times [0.00,\ 0.00,\ 0.00] = [0.00,\ 0.00,\ 0.00] \\
E_{\text{new}}[1] &= [0.50,\ -0.30,\ 0.80] - 0.1 \times [0.05,\ -0.05,\ 0.15] = [0.495,\ -0.295,\ 0.785] \\
E_{\text{new}}[2] &= [-0.20,\ 0.70,\ 0.10] - 0.1 \times [-0.15,\ 0.08,\ 0.12] = [-0.185,\ 0.692,\ 0.088] \\
E_{\text{new}}[3] &= [0.40,\ -0.50,\ -0.60] - 0.1 \times [0.20,\ -0.10,\ -0.05] = [0.38,\ -0.49,\ -0.595]
\end{aligned}
$$

这次更新中，"猫"的嵌入改变了 $[0.495,\ -0.295,\ 0.785]$、"狗"的嵌入变成了 $[-0.185,\ 0.692,\ 0.088]$、"鱼"变成了 $[0.38,\ -0.49,\ -0.595]$。变化量很小——这是因为嵌入维度只有 3，在实际训练中随着步数增加，语义相近的词会自然收敛到相近的位置。

#### PyTorch 的实际初始化方式

上面手算例子用的是简化的小数值。在实际 PyTorch 中，`nn.Embedding` 的初始化由 `reset_parameters()` 方法完成，源码位于 `torch/nn/modules/sparse.py`：

```python
def reset_parameters(self) -> None:
    nn.init.normal_(self.weight)
    self._fill_padding_idx_with_zero()
```

**第一步：`nn.init.normal_`** 将整个嵌入矩阵 $E$ 的每个元素从标准正态分布 $\mathcal{N}(0, 1)$ 中随机采样：

$$
E[i, j] \sim \mathcal{N}(0, 1), \quad \forall i \in [0, V-1], j \in [0, d_e-1]
$$

也就是说，初始化时 $E$ 中约 68% 的值落在 $[-1, 1]$ 之间，约 95% 落在 $[-2, 2]$ 之间。实际操作等价于：

```python
torch.randn(V, d_e)  # 从 N(0,1) 采样一个 (V, d_e) 的矩阵
```

以 $V=10000, d_e=300$ 为例，`randn` 生成的 300 万个值中：
- 约 204 万个在 $[-1, 1]$ 之间
- 约 285 万个在 $[-2, 2]$ 之间
- 极少数（约 300 个）的绝对值可能超过 3

**第二步：`_fill_padding_idx_with_zero()`** 将 `padding_idx` 指定的那一行强制置为零向量。这个操作发生在初始化阶段（而非前向传播时），所以即使从未跑过前向传播，PAD 行的权重在创建嵌入层的那一刻就已经是零了。

**为什么用 $\mathcal{N}(0, 1)$ 而不是其他分布？** 标准正态分布是深度学习中最常用的权重初始化分布。它的均值为 0（避免激活值偏移），标准差为 1（提供适中的初始梯度信号）。对于嵌入层来说，初始化值的大小并不关键——因为嵌入向量会在训练中通过梯度逐步调整到有意义的语义位置。初始值只是提供一个"起点"，训练过程负责将其移动到正确的位置。

可以在 Python 中直接验证：

```python
import torch
import torch.nn as nn

E = nn.Embedding(5, 3, padding_idx=0)

# 查看随机初始化的嵌入矩阵
print(E.weight)
# tensor([[ 0.0000,  0.0000,  0.0000],   ← padding_idx=0 已置零
#         [ 0.5410, -0.2934,  1.1783],   ← N(0,1) 随机值
#         [-1.2034,  0.7241, -0.1102],
#         [ 0.3892, -0.5123, -0.6015],
#         [-0.0843,  1.4420,  0.2617]], requires_grad=True)

# 验证分布特征
non_pad = E.weight[1:]  # 排除 PAD 行
print(f"均值: {non_pad.mean():.3f}")    # 接近 0
print(f"标准差: {non_pad.std():.3f}")   # 接近 1
```

注意 `requires_grad=True`——嵌入矩阵是模型的可训练参数，优化器会在 `backward()` 后自动更新它。

### 2.3 梯度的传递（理论总结）

Embedding 层虽然做的是"查表"操作，但它仍然是可训练的。反向传播时，损失 $L$ 对嵌入矩阵 $E$ 的梯度为：

$$
\frac{\partial L}{\partial E[i, :]} = \begin{cases}
\text{非零} & \text{if 词 } w_i \text{ 在本批数据中出现过} \\
\mathbf{0} & \text{otherwise}
\end{cases}
$$

只有被"查到"的行（即本批数据中实际出现的词）才会收到非零梯度，其余行的梯度恒为零。这个特性带来了几个实际影响：

- **低频词**获得的梯度更新次数少，嵌入向量可能不够优化，质量偏低
- **高频词**被反复更新，嵌入向量通常质量更高，语义表示更准确
- **$\langle\text{UNK}\rangle$ 词**（未知词标记）承载了所有词表外词的梯度信号——这会导致 UNK 的嵌入向量被多个完全不同语义的词拉扯，最终收敛到一个"均值"位置，缺乏区分能力

### 2.4 padding_idx 的特殊处理

在批处理中，不同长度的句子需要填充到统一长度。填充位置使用 $\langle\text{PAD}\rangle$ token（通常 index = 0）标记。PAD 不是真实的文本内容，其嵌入向量应当始终为零向量，且不应参与梯度更新。

PyTorch 的 `nn.Embedding` 通过 `padding_idx` 参数一步完成这两件事：

```python
embedding = nn.Embedding(
    vocabulary_size, embedding_dimension,
    padding_idx=pad_index  # 指定哪个索引是 PAD
)
```

**这个参数在底层做了两件事：**

1. **初始化时**：将嵌入矩阵第 `pad_index` 行的权重直接置为零向量（`weight.data[pad_index, :] = 0`），确保该位置从前向传播的第一刻起就输出零。
2. **每次反向传播后**：将第 `pad_index` 行的梯度清零（`weight.grad[pad_index, :] = 0`），即使某些计算路径给它分配了梯度，也会在优化器更新之前被抹掉。

换句话说，`padding_idx` 对应的行是一个"死行"——权重恒为零，梯度恒为零，永远不更新。

可以用以下代码验证这一行为：

```python
import torch
import torch.nn as nn

# 创建词表大小=5、嵌入维度=3、padding_idx=0 的嵌入层
E = nn.Embedding(5, 3, padding_idx=0)

# 查看初始权重：第 0 行已被强制初始化为零
print(E.weight.data[0])
# tensor([0., 0., 0.])

# 前向传播：输入包含 PAD (index=0) 和两个正常词
x = torch.tensor([0, 1, 2])
out = E(x)

# 第 0 个位置的输出是零向量——因为 index=0 对应 padding_idx
print(out[0])
# tensor([0., 0., 0.]) 不是随机值，而是严格为零

# 即使加入计算图，参与 loss，backward 后的梯度仍然为零
loss = out.sum()       # 把三个位置的嵌入求和作为 loss
loss.backward()        # 反向传播，计算每个参数的梯度

print(E.weight.grad[0])
# tensor([0., 0., 0.])  PAD 行梯度为零，不会更新

print(E.weight.grad[1])
# tensor([1., 1., 1.])  正常词的梯度非零，正常更新
```

逐行解释这段验证代码的要点：

- `E = nn.Embedding(5, 3, padding_idx=0)`：嵌入矩阵形状为 `(5, 3)`，index=0 被指定为 padding_idx。此时期望的行为是：无论输入中 index=0 出现多少次，对应的输出和梯度都是零。
- `x = torch.tensor([0, 1, 2])`：输入包含三个 token——PAD（0）、词 1、词 2。
- `out = E(x)`：查表操作，`E[0]` 返回零向量（因为 padding_idx 强制置零），`E[1]` 返回第 1 行的随机初始向量，`E[2]` 返回第 2 行的随机初始向量。输出形状为 `(3, 3)`。
- `loss = out.sum()`：将三个嵌入向量的所有分量求和作为标量 loss。由于 `out[0]` 全为零，它对 loss 的贡献为零——但计算图仍然连接着。
- `loss.backward()`：`out.sum()` 对 `out` 每个分量的梯度都是 1。PAD 位置的梯度 `[1., 1., 1.]` 会通过查表操作回传到 `E.weight[0]`——但 `padding_idx` 机制在 backward 的最后一步将其清零。
- 打印 `E.weight.grad[0]`：全零，验证 PAD 行不参与更新。打印 `E.weight.grad[1]`：`[1., 1., 1.]`，说明正常词正常接收梯度。

**为什么这个处理至关重要？** 假设不加 `padding_idx`，PAD token 也会获得随机的嵌入向量和正常的梯度。考虑一个实际场景：每条输入有 8 个有效词 + 120 个 PAD，每批 64 条数据。如果不处理 PAD，每批中有 64×120=7680 个 PAD token 参与梯度计算——它们携带的梯度噪声会干扰有效词的嵌入学习，并经由 RNN 层放大。`padding_idx` 将这些位置的梯度完全切断，PAD 就真正成为了"不存在"的 token。

---

## 3. 字符级嵌入 vs 词级嵌入

对同一个文本，有两种切分粒度："按词切分"和"按字符切分"。这两种方式在词表大小、序列长度、语义编码方式上截然不同。

### 3.1 Char-RNN：字符级嵌入

Char-RNN 以**单个字符**为最小单元。文本 "hello" 被切分为 `['h', 'e', 'l', 'l', 'o']` 五个字符。

根据 Char-RNN 项目的实际配置，字符词表通过频率过滤（`min_freq = 1`，即保留所有出现的字符）构建，最终词表大小约 65（包含大小写字母、标点和空格）。嵌入维度 $d_e = 256$，嵌入矩阵 $E \in \mathbb{R}^{\text{vocab\_size} \times 256}$，参数量约为：

$$
\text{vocab\_size} \times d_e \approx 65 \times 256 = 16640
$$

仅 1.6 万参数，可忽略不计。

**优点**：

- 词表极小（几十到几百个字符），嵌入层参数量微不足道
- **不存在 OOV（Out-of-Vocabulary）问题**：任何语言的文本归根结底都由这几十个字符组成，理论上可以处理任意输入
- 无需分词器：直接逐个字符切分即可，避免了中文分词的精度问题和英文子词切分（tokenization）的复杂性
- 天然捕获拼写和词形变化信息（英文的前缀、后缀、词根）

**缺点**：

- 序列长度剧增：一个 10 词的句子平均约 50 个字符，需要 50 个 RNN 时间步才能处理完——计算量和梯度路径长度随之增加
- 语义信息分散：一个词的语义不会直接出现在嵌入中，而是需要 RNN 在多个字符的隐藏状态上逐步累积——例如要理解"happy"，模型需要看到 `'h'`, `'a'`, `'p'`, `'p'`, `'y'` 的完整序列
- 生成任务中可能出现拼写错误：因为是逐字符生成，没有"词"的约束

### 3.2 EmotionClassification：词级嵌入

EmotionClassification 以**词**为最小单元。文本 "物流很快" 经过 jieba 分词后变为 `['物流', '很快']` 两个词。

根据项目实际配置，词表通过 jieba 分词构建并按词频过滤（`min_freq = 3`），词表大小约 1 万词。嵌入维度 $d_e = 300$，嵌入矩阵 $E \in \mathbb{R}^{\text{vocab\_size} \times 300}$，参数量约为：

$$
\text{vocab\_size} \times d_e \approx 10000 \times 300 = 3,000,000
$$

300 万参数占模型总参数的相当比例（以 LSTM hidden_size=256 + 双向 + 2 层为例，LSTM 参数量约为 $4 \times (300 \times 256 + 256^2) \times 2 \times 2 \approx 2.3\text{M}$），嵌入层和 LSTM 层的参数量在同一量级。

**优点**：

- 语义丰富：每个词有自己的独立嵌入，词的语义直接编码在向量中，模型无需从字符逐步累积
- 序列更短：一个 50 词的句子仅需 50 个 RNN 时间步，大幅缩短梯度路径
- 可加载预训练嵌入（Word2Vec、GloVe）加速收敛

**缺点**：

- 词表可能很大（数万到数十万），嵌入层参数量可观
- **OOV 问题**：词表外词只能映射到 $\langle\text{UNK}\rangle$，丢失了具体信息——"新垣结衣"和"埃隆·马斯克"都坍缩为同一个 UNK 向量
- 依赖分词器：中文分词（jieba）的错误会传递到下游；英文的子词切分（BPE, WordPiece）也需要额外处理
- 词频过滤（`min_freq`）会丢弃低频但可能信息量很高的词

### 3.3 选择指南

![Word2Vec.png](https://img.yumeko.site/file/articles/EmbeddingLayer/Word2Vec.webp)

| | 字符级 | 词级 |
|------|:---:|:---:|
| 词表大小 | 极小（<200） | 大（1万~10万+） |
| 序列长度 | 长（字符数） | 短（词数） |
| OOV 问题 | 无 | 有（映射到 UNK） |
| 需要分词 | 否 | 是 |
| 语义丰富度 | 低（需 RNN 累积） | 高（直接编码在嵌入中） |
| 拼写/形态信息 | 丰富 | 丢失（英文的词缀信息） |
| 适合任务 | 文本生成、拼写敏感任务 | 文本分类、情感分析、序列标注 |

选择的核心权衡：**序列长度 vs 语义密度**。字符级用更长的序列换取更小的词表和免分词的便利；词级用更大的词表和分词依赖换取更短的序列和更浓缩的语义。

---

## 4. 嵌入维度的选择

### 4.1 经验法则

嵌入维度 $d_e$ 是超参数，需要根据词表大小和任务复杂度选择：

| 词表大小 | 推荐 $d_e$ | 典型场景与理由 |
|----------|:---:|------|
| < 1,000 | 64-128 | 字符级任务，模式有限，大维度浪费 |
| 1,000-10,000 | 128-256 | Char-RNN（65 字符选 256）偏大但因词表极小，开销不高 |
| 10,000-100,000 | 200-300 | EmotionClassification 的 300 是业界常用值，覆盖大部分 NLP 任务 |
| 100,000+ | 300-512 | 大词表需要更多维度来区分词之间的细微语义差异 |

文献中有经验公式 $\text{embedding\_dim} \approx \sqrt[4]{\text{vocab\_size}}$。对于 $V = 10000$，$\sqrt[4]{10000} = 10$，这个值过于保守，仅作为理论下界的参考——在实际工程中，200-300 维是被广泛验证的有效区间。

### 4.2 维度太小和太大的后果

- **$d_e$ 太小**（如 16 维）：每个词只能编码 16 个自由度的语义信息，向量空间过于拥挤，不同词被迫"挤"在相近位置，成为模型的信息瓶颈。实验上表现为训练损失下降缓慢、最终准确率明显偏低。
- **$d_e$ 太大**（如 1024 维）：参数膨胀、训练变慢、小数据集上容易过拟合。更重要的是，边际收益迅速递减——从 300 维提升到 1024 维带来的语义编码增益通常远小于 3 倍的参数开销。

在大多数 NLP 任务中，200-300 维已经足够——这也是 Word2Vec、GloVe 等经典预训练嵌入的默认维度。

---

## 5. 预训练嵌入 vs 从零训练

### 5.1 从零训练（两个项目均采用此方式）

嵌入矩阵 $E$ 随机初始化（通常用 $\mathcal{N}(0, 1)$ 或 Xavier 初始化），然后在目标任务上与模型一起端到端训练。

**优点**：流程简单，嵌入直接适配目标任务的分布和损失函数。

**缺点**：需要大量标注数据才能学到高质量的嵌入；在小数据集上，随机初始化的嵌入质量远不如预训练嵌入，导致模型收敛慢、最终效果差。

### 5.2 加载预训练嵌入

使用在大规模无监督语料上预训练的嵌入（Word2Vec、GloVe、fastText），将预训练权重加载到 $E$ 中。

以 GloVe 为例，`glove.6B.300d.txt` 文件格式为每行一个词和其 300 维向量（空格分隔）。加载流程如下：

```python
# 第 1 步：加载预训练嵌入文件
# 文件格式: "the 0.418 0.24968 -0.41242 ..." (词 + 300 个浮点数)
pretrained = {}
with open("glove.6B.300d.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]                          # 词本身
        vector = [float(x) for x in parts[1:]]   # 300 个浮点数
        pretrained[word] = vector                # 存入字典: {str: list[float]}

# 第 2 步：按当前任务词表顺序，构建嵌入矩阵
# word2idx 是自己构建的映射: {"<PAD>": 0, "<UNK>": 1, "的": 2, ...}
embeddingMatrix = torch.zeros(vocabSize, 300)   # 先用全零占位
hitCount = 0                                     # 命中预训练的词数
for word, idx in word2idx.items():
    if word in pretrained:
        # 词在预训练文件中存在，用预训练向量填充对应行
        embeddingMatrix[idx] = torch.tensor(pretrained[word])
        hitCount += 1
    # else: 词在预训练中不存在，保持零向量（后续随机初始化）

# 第 3 步：创建 nn.Embedding 层，用预训练权重覆盖随机初始化的权重
embedding = nn.Embedding(vocabSize, 300)
embedding.weight.data.copy_(embeddingMatrix)  # 原地复制，不改变计算图

# 第 4 步：决定是否微调
embedding.weight.requires_grad = True   # True = 微调（嵌入随训练更新）
# embedding.weight.requires_grad = False  # False = 冻结（嵌入保持不变）
```

**逐段解释：**

**第 1 步——加载文件**：GloVe 的 `.txt` 文件是纯文本格式，每行第一个 token 是词，后面 300 个 token 是向量值。这里遍历每一行，提取词和对应的 300 个浮点数，存入 `pretrained` 字典。这个字典的键是词字符串，值是对应的 300 维向量（Python list）。GloVe 6B 版本包含约 40 万个词，字典约占用 $400000 \times 300 \times 4 \approx 480\text{MB}$ 内存。

**第 2 步——构建嵌入矩阵**：当前任务有自己的词表（通过分词和频率过滤得到的 `word2idx`），大小为 `vocabSize`。创建一个全零矩阵 `(vocabSize, 300)` 作为容器。然后遍历当前词表中的每个词：
- 如果该词在预训练字典中存在，取其向量填入嵌入矩阵的对应行；
- 如果不存在（预训练语料未覆盖或当前词是特殊 token 如 `<PAD>`, `<UNK>`），保持零向量。这些零向量的行随后可以在训练中从零学习，或单独做随机初始化。

**第 3 步——加载权重**：`nn.Embedding(vocabSize, 300)` 创建嵌入层时，PyTorch 会用 $\mathcal{N}(0, 1)$ 随机初始化 `weight`。`weight.data.copy_(embeddingMatrix)` 在 `.data`（不记录计算图）层面将预训练矩阵**原地复制**进去，覆盖随机值。使用 `.data` 是因为这是一个权重赋值操作，不需要被 autograd 追踪。

**第 4 步——微调开关**：`requires_grad = True` 表示嵌入矩阵是模型参数，优化器会更新它。如果设为 `False`，嵌入在整个训练过程中保持不变（冻结），仅作为固定的输入特征。需要特别注意：即使冻结，嵌入层内部 `padding_idx` 那行仍然保持为零（由 `padding_idx` 机制在每次 backward 后强制清零）。

**优点**：收敛速度更快（因初始嵌入已包含语义信息），小数据集上效果显著更好；fastText 的 OOV 词也能通过子词（$n$-gram）组合得到合理表示。

**缺点**：预训练嵌入文件体积大（GloVe 840B 版本约 5GB）；可能引入外部语料的分布偏差（预训练语料的领域特征会被带入目标任务）。

### 5.3 是否微调？

| 数据量 | 建议 |
|--------|------|
| 极少（< 10K 样本） | 冻结嵌入（`requires_grad = False`），仅训练 RNN 和分类层——避免嵌入在小数据上过拟合 |
| 中等（10K-100K） | 加载预训练 + 微调（`requires_grad = True`），学习率应设为模型其他部分的 0.1 倍 |
| 大量（> 100K） | 从零训练也可接受；如任务领域与预训练语料有显著差异，从零训练可能更优 |

---

## 6. PyTorch 代码详解

本节从头拆解 `nn.Embedding` 的完整用法——从构造、前向传播、到与上下游模块的衔接。

### 6.1 nn.Embedding 的核心参数

```python
import torch.nn as nn

# 词级嵌入 —— EmotionClassification 项目风格
wordEmbedding = nn.Embedding(
    num_embeddings=10000,    # ① 词表大小 V
    embedding_dim=300,       # ② 每个词的向量维度 d_e
    padding_idx=0,           # ③ PAD 索引
)

# 字符级嵌入 —— Char-RNN 项目风格
charEmbedding = nn.Embedding(
    num_embeddings=65,       # ① 字符种类数（约65种）
    embedding_dim=256,       # ② 每个字符的向量维度 d_e
    # 无 padding_idx —— 字符级使用滑动窗口，天然等长
)
```

**参数逐个解释：**

**① `num_embeddings`（词表大小 $V$）**：这是嵌入矩阵的行数，也是输入索引的取值范围。`nn.Embedding(V, d_e)` 内部维护一个形状为 `(V, d_e)` 的可训练权重矩阵。传入的索引必须在 $[0, V-1]$ 范围内，否则会抛出 `IndexError`。对于 EmotionClassification，$V$ 通常从几千到几万不等——取决于分词后按频率过滤的策略（`min_freq`）。Char-RNN 的字符种类数只有几十，$V$ 很小。

**② `embedding_dim`（嵌入维度 $d_e$）**：这是嵌入向量的长度，即每个词/字符被表示为多少维的实值向量。这个数字决定了嵌入层的表达能力上限和参数量：
- $d_e = 300$ 对应嵌入矩阵参数量 $10000 \times 300 = 3\text{M}$
- $d_e = 256$ 对应嵌入矩阵参数量 $65 \times 256 = 16.6\text{K}$
- $d_e$ 越大，每个词的表示越丰富——但参数量线性增长，且边际收益递减

**③ `padding_idx`（PAD 索引）**：指定哪个整数索引对应填充位置。设置为某个值（通常为 0）后：
- 前向时该行的输出恒为零向量
- 反向时该行的梯度恒为零向量
- 这一行在整个训练过程中**永远不更新**

EmotionClassification 设置了 `padding_idx=0` 是因为其词表的 index=0 固定预留给 `<PAD>`。Char-RNN 不设 `padding_idx` 是因为它按固定窗口切分文本，所有样本天然等长，没有填充的必要。

### 6.2 前向传播的完整过程

```python
# 假设 batch=2, 每条样本序列长度=5
inputIds = torch.tensor([
    [2, 5, 7, 0, 0],   # 样本 1: 3 个有效词 + 2 个 PAD
    [3, 1, 4, 8, 2],   # 样本 2: 5 个有效词
])
# inputIds 形状: (2, 5), dtype: torch.long (必须为整数类型)

embeddings = wordEmbedding(inputIds)
# embeddings 形状: (2, 5, 300)
```

**`nn.Embedding.forward()` 的底层执行逻辑：**

`wordEmbedding(inputIds)` 在 PyTorch 内部执行了以下操作：

1. **类型检查**：确认 `inputIds` 是整数张量（`torch.long` 或 `torch.int`）。浮点数索引会直接报错——因为查表需要整数下标。

2. **范围检查**（取决于 `_fill_padding_idx_with_zero` 等内部标记）：确保 `inputIds` 中的所有值在 $[0, V-1]$ 内。超出范围的值在不启用 `padding_idx` 的情况下会导致 `CUDA error: device-side assert triggered` 这类难以调试的报错。

3. **查表**：对 `inputIds` 中的每个整数 $i$，从嵌入矩阵 `weight` 中取出第 $i$ 行。本质上等价于 `weight[inputIds]`——高级索引（fancy indexing），输出形状为 `(*inputIds.shape, d_e)`。这里的 batch=2、seq_len=5，所以输出形状为 `(2, 5, 300)`。

4. **padding_idx 覆盖**：对于 `padding_idx` 指定的索引值（如 index=0），无论 `weight[0]` 当前存储的是什么值，输出中对应位置都会被替换为零向量。这个覆盖发生在计算图内部——因此虽然零向量不传递有效信息，autograd 仍然会处理计算图上的梯度链路。

**具体到这个输入张量的处理结果：**

- 样本 1 的第 1 个位置 index=2 取出 `weight[2]`，形状 `(300,)`，正常嵌入向量
- 样本 1 的第 4 个位置 index=0 取出 `weight[0]` 后被 `padding_idx` 机制覆盖为零向量 `[0., 0., ..., 0.]`
- 样本 2 的 5 个位置 index 都不为 0，各取对应行的正常嵌入向量

最终输出 `embeddings` 是一个形状为 `(2, 5, 300)` 的张量——每个 token 都被表示为一个 300 维的稠密向量。

### 6.3 嵌入层与上下游的衔接

嵌入层的输出直接送入后续网络。以 EmotionClassification 为例：

```python
class BaseRNN(nn.Module):
    def __init__(self, vocabulary_size, pad_index,
                 embedding_dimension=300, hidden_dimension=256):
        super().__init__()

        # 嵌入层：将整数索引转为稠密向量
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension,
            padding_idx=pad_index
        )

        # Dropout：对嵌入输出做正则化，防止过拟合
        # 为什么放在嵌入之后？
        # — Dropout 随机将部分嵌入维度置零，迫使 RNN 不依赖某几个特定维度
        self.dropout = nn.Dropout(0.5)

        # RNN 层：接收嵌入输出作为第一步的输入
        # input_size 必须等于 embedding_dim — 嵌入层的输出维度就是 RNN 的输入维度
        self.rnn = nn.LSTM(
            input_size=embedding_dimension,   # 300，与嵌入维度一致
            hidden_size=hidden_dimension,     # 256
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids, mask):
        # Step 1: Embedding
        # 输入: (B, L) 整数，输出: (B, L, 300) 稠密向量
        embedded = self.embedding(input_ids)

        # Step 2: Dropout
        # 训练时随机将 50% 的嵌入维度置零，推理时自动关闭
        embedded = self.dropout(embedded)

        # Step 3: RNN
        # 输入: (B, L, 300)，输出: (B, L, 512)  [256×2 因为双向]
        rnn_output, _ = self.rnn(embedded)

        # 后续进入 Attention 池化，分类器
        # ...
```

**关键衔接点：**

- **嵌入维度 $d_e$ 和 RNN 的 `input_size` 必须相等**。这是硬性约束——`nn.LSTM(input_size=d_e, ...)` 的第一个全连接层的权重矩阵形状为 `(4*hidden_size, d_e)`，如果维度不匹配，矩阵乘法直接报错。
- **Dropout 放在嵌入层之后**是最常见的做法。直接对嵌入向量做 Dropout（将整个维度的值随机置零）迫使后续网络不依赖个别维度，增强了泛化能力。注意 Dropout 作用在"特征维度"上（`nn.Dropout` 默认对所有元素独立置零），等价于随机遮蔽部分语义特征。
- **`batch_first=True`**：这个参数控制 RNN 输入输出的张量形状约定。设为 `True` 表示输入形状为 `(batch, seq_len, feature)`——这与嵌入层的输出形状 `(B, L, d_e)` 天然一致。如果不设此参数，PyTorch RNN 默认期望 `(seq_len, batch, feature)`，需要手动 `transpose`。

### 6.4 完整的最小示例（可运行）

```python
import torch
import torch.nn as nn

# 构建嵌入层
vocabSize = 100     # 词表 100 个词
embedDim = 16       # 每词用 16 维向量表示
padIdx = 0          # index=0 作为 PAD

embed = nn.Embedding(vocabSize, embedDim, padding_idx=padIdx)

# 构造输入
# batch=2, seq_len=4. 每个元素是 [0, vocabSize) 的整数
inputIds = torch.tensor([
    [2, 5, 7, 0],   # 句子 1: 最后一位是 PAD
    [3, 1, 4, 8],   # 句子 2: 无 PAD
], dtype=torch.long)

# 前向传播
out = embed(inputIds)
print(out.shape)
# torch.Size([2, 4, 16])
# batch=2, seq_len=4, 每词得到 16 维向量

# 验证 padding_idx
print(out[0, 3, :])   # 句子 1 第 4 个 token (index=0)
# tensor([0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0.])
# 16 维全零，验证 PAD 位置输出为零向量

print(out[0, 0, :])   # 句子 1 第 1 个 token (index=2)
# tensor([ 0.8234, -1.4472,  0.3319, ...])
# 16 维非零稠密向量，正常输出

# 查看嵌入矩阵参数
totalParams = sum(p.numel() for p in embed.parameters())
print(f"嵌入层参数量: {totalParams}")
# 嵌入层参数量: 1600  (= 100 × 16)

# 其中 index=0 那一行虽然计入参数量，但从未参与更新
# 实际有效参数量: 99 × 16 = 1584
```

这个最小示例覆盖了嵌入层的全部核心行为：查表、输出稠密向量、padding_idx 强制置零。理解这段代码中每一行的输入输出形状变化，就理解了嵌入层的全部。

> 权重初始化技巧参见 [[NeuralNetwork/Tips/Techniques/WeightInitialization|权重初始化指南]]。
> 回到主文档：[[NeuralNetwork/RNN/Foundations/RNNOverview|RNN 详解主文档]]
