---
title: 嵌入层详解：从离散符号到稠密向量
date: 2026-05-29
category: NeuralNetwork/RNN/Foundations
tags:
  - 深度学习
  - 基础
  - NLP
description: 深入理解词嵌入和字符嵌入的数学本质、训练过程、padding_idx 的特殊处理以及嵌入维度选择的经验法则。
image: https://img.yumeko.site/file/blog/EmbeddingLayer.webp
status: draft
---

## 1. 从离散到连续：为什么需要 Embedding？

![Dense.png](https://img.yumeko.site/file/articles/EmbeddingLayer/Dense.webp)

### 1.1 One-Hot 编码的三个死穴

处理文本的第一步是把字符/词转成数字。最朴素的方式是 one-hot 编码：

```
词表: ["PAD", "UNK", "我", "爱", "你", "机器学习", "深度学习", ...]
          0      1      2     3     4        5           6

"我":  [0, 0, 0, 1, 0, 0, 0, 0, ...]   (词表大小维，只有1位是1)
"爱":  [0, 0, 0, 0, 1, 0, 0, 0, ...]
"学习": [0, 0, 0, 0, 0, 0, 1, 0, ...]  (注意!"学习"不在词表中→UNK)
```

这种表示有三个致命问题：

**死穴一：维度灾难**

中文词表可能包含 5 万个词。每个词用 5 万维向量表示——假设用 float32 存储，一个词占 200KB。一批 64 个句子、每句 128 个词 = 64×128×50000×4 bytes ≈ 1.6 **GB** 仅用于存储输入！

**死穴二：极度稀疏**

每个向量只有 1 个位置是 1，其余 49,999 个位置全是 0。99.998% 的计算是在和零相乘——极其浪费。

**死穴三：无语义信息**

任意两个不同词的 one-hot 内积 = 0，余弦相似度 = 0。"国王"和"女王"的向量距离等于"国王"和"苹果"的距离。模型无法利用词与词之间的语义关系。

### 1.2 Embedding 的解决方案

把稀疏的高维 one-hot 向量"压缩"成稠密的低维向量：

```
One-hot: "我" → [0, 0, 1, 0, ..., 0]     (50000维，稀疏)
Embedding: "我" → [0.23, -0.45, 0.78, ...] (300维，稠密)
```

稠密向量的每个维度没有明确的"含义"对应某个特定特征——它们是通过训练**自动学习**出来的分布式表示。在 300 个维度上，不同的激活模式可以编码指数级的语义信息（$2^{300}$ 种模式）。

---

## 2. Embedding 的数学本质

### 2.1 就是一个查找表

Embedding 层在数学上极其简单——它就是一个**可学习的矩阵**：

```python
E = nn.Parameter(torch.randn(vocab_size, embedding_dim))
```

前向传播不是矩阵乘法，而是**整数索引查表**：

```python
def embedding_forward(indices, E):
    """
    indices: (batch, seq_len) — 每个元素是 [0, vocab_size) 的整数
    E: (vocab_size, embedding_dim) — 嵌入矩阵
    returns: (batch, seq_len, embedding_dim)
    """
    return E[indices]  # 取出对应行
```

数学上等价于：输入的 one-hot 向量（长度为 vocab_size）乘上嵌入矩阵 E。但在实现上直接查表，避免了 one-hot 矩阵乘法的计算浪费。

### 2.2 梯度的传递

虽然是"查表"操作，Embedding 层仍然是可训练的：

```python
# 假设 embedding 的输出用于后续计算，产生了 loss
loss.backward()

# E.grad 的形状: (vocab_size, embedding_dim)
# 只有被"查到"的行有非零梯度，其他行的梯度为 0
```

梯度只更新被使用的行——这意味着：
- 低频词被更新的次数少，嵌入向量可能不够优化
- 高频词被更新次数多，嵌入向量通常质量更高
- `<UNK>` 词承载了所有未见过的词的梯度信号

### 2.3 padding_idx 的特殊处理

在 batch 处理中，不同长度的句子需要填充到统一长度。填充位置用 `<PAD>` token（index=0）标记。PAD 的嵌入应当是**始终为零**且**不参与梯度更新**的：

```python
embedding = nn.Embedding(
    vocabSize, embeddingDim,
    padding_idx=0  # index=0 的嵌入始终为 0 向量
)

# 等价于每次前向后手动清零
# embedding.weight.data[0].zero_()
```

PyTorch 的 `padding_idx` 参数自动处理了这两件事——前向时强制输出零向量，反向时不更新该行。

```python
# 验证 padding_idx 的效果
E = nn.Embedding(5, 3, padding_idx=0)
x = torch.tensor([0, 1, 2])
out = E(x)
print(out[0])  # tensor([0., 0., 0.])  ← 强制为零

loss = out.sum()
loss.backward()
print(E.weight.grad[0])  # tensor([0., 0., 0.])  ← 梯度为零
```

---

## 3. 字符级嵌入 vs 词级嵌入

### 3.1 Char-RNN：字符级嵌入

```
词表: 65 个不同的字符（a-z, A-Z, 标点, 空格等）
嵌入维度: 256
嵌入矩阵: (65, 256)
参数量: 65 × 256 = 16,640
```

**优点**：
- 词表极小（几十到几百），参数量可忽略
- **不存在 OOV**：任何文本都由这几十个字符组成
- 不需要分词器：直接按字符切分

**缺点**：
- 序列更长：一个 10 词的句子 ≈ 50 个字符 → 需要 50 个时间步
- 语义信息分散：一个词的语义需要多个字符的隐藏状态累积才能获得
- 生成任务中拼写错误是可能的（因为逐字生成）

### 3.2 EmotionClassification：词级嵌入

```
词表: ~10,000 个词（按词频过滤，min_freq=3）
嵌入维度: 300
嵌入矩阵: (10000, 300)
参数量: 10000 × 300 = 3,000,000
```

**优点**：
- 语义丰富：每个词有自己的嵌入，词的语义直接编码在向量中
- 序列更短：一个 50 词的句子只需要 50 个时间步
- 可以加载预训练嵌入：Word2Vec、GloVe 等

**缺点**：
- 词表可能很大（数万到数十万），参数量可观
- **OOV 问题**：未见过的词只能映射到 `<UNK>`
- 需要分词器（中文用 jieba，英文用空格/子词）
- 词频过滤会丢弃低频但可能有用的词

### 3.3 选择指南

![Word2Vec.png](https://img.yumeko.site/file/articles/EmbeddingLayer/Word2Vec.webp)

| | 字符级 | 词级 |
|------|:---:|:---:|
| 词表大小 | 小（<200） | 大（1万~10万+） |
| 序列长度 | 长（字符数） | 短（词数） |
| OOV 问题 | 无 | 有（→ UNK） |
| 需要分词 | 否 | 是 |
| 语义丰富度 | 低（需累积） | 高（直接编码） |
| 拼写/形态信息 | 丰富 | 丢失（英文的词缀信息） |
| 适合任务 | 文本生成 | 文本分类、序列标注 |

---

## 4. 嵌入维度的选择

### 4.1 经验法则

| 词表大小 | 推荐嵌入维度 | 理由 |
|----------|:---:|------|
| <1,000 | 64-128 | 模式有限，不需要太大 |
| 1,000-10,000 | 128-256 | Char-RNN（65 字符→256 维）偏大但计算量小 |
| 10,000-100,000 | 200-300 | EmotionClassification 选 300，业界常见 |
| 100,000+ | 300-512 | 大词表需要更多维度来区分词 |

另一个经验公式：$\text{embedding\_dim} \approx \sqrt[4]{\text{vocab\_size}}$。对于 vocab_size=10,000，$\sqrt[4]{10000} = 10$（太小了，仅供参考下限）。

### 4.2 维度太小和太大的后果

- **太小**（如 16 维）：无法承载足够的语义信息，成为模型瓶颈
- **太大**（如 1024 维）：参数多、训练慢、容易过拟合，边际收益递减

对于 300 维的选择：Je在大多数 NLP 任务中，200-300 维已经足够，继续增加维度的收益迅速递减。

---

## 5. 预训练嵌入 vs 从零训练

### 5.1 从零训练（两个项目都是这种方式）

嵌入矩阵随机初始化，然后在目标任务上与模型一起端到端训练。

**优点**：简单，嵌入直接适配目标任务。
**缺点**：需要大量数据才能学到好的嵌入；小数据集上质量差。

### 5.2 加载预训练嵌入

使用在大规模语料上预训练的嵌入（Word2Vec、GloVe、fastText）：

```python
# 加载预训练的 GloVe 嵌入
pretrained = loadGlove("glove.6B.300d.txt")  # {word: vector}

# 构建嵌入矩阵
embeddingMatrix = torch.zeros(vocabSize, 300)
for word, idx in word2idx.items():
    if word in pretrained:
        embeddingMatrix[idx] = torch.tensor(pretrained[word])

# 创建 Embedding 层并加载预训练权重
embedding = nn.Embedding(vocabSize, 300)
embedding.weight.data.copy_(embeddingMatrix)
embedding.weight.requires_grad = True  # 允许微调
```

**优点**：收敛更快，小数据集上效果显著更好，OOV 词也能通过子词（fastText）获得合理表示。
**缺点**：预训练嵌入文件大（几百 MB），可能引入外部语料的偏见。

### 5.3 是否微调？

| 数据量 | 建议 |
|--------|------|
| 很少（<10K 样本） | 冻结嵌入（`requires_grad=False`），只训练其他层 |
| 中等（10K-100K） | 加载预训练 + 微调（`requires_grad=True`） |
| 大量（>100K） | 从零训练也 OK，预训练嵌入仍有帮助 |

---

## 6. PyTorch 代码详解

```python
import torch.nn as nn

# 词级嵌入（EmotionClassification 风格）
wordEmbedding = nn.Embedding(
    num_embeddings=10000,    # 词表大小
    embedding_dim=300,       # 嵌入维度
    padding_idx=0,           # PAD 索引
)

# 字符级嵌入（Char-RNN 风格）
charEmbedding = nn.Embedding(
    num_embeddings=65,       # 字符种类数
    embedding_dim=256,       # 嵌入维度
    # 字符级不需要 padding_idx（使用滑动窗口，无填充）
)

# 前向传播
inputIds = torch.tensor([[2, 5, 7, 0, 0],    # batch=2, seq_len=5
                          [3, 1, 4, 8, 2]])
embeddings = wordEmbedding(inputIds)         # (2, 5, 300)
# 第1句的PAD位置（index=0）自动输出零向量
```

> 权重初始化技巧参见 [[NeuralNetwork/Tips/Techniques/WeightInitialization|权重初始化指南]]。
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
