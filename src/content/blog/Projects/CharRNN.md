---
title: Char-RNN 字符级文本生成项目
date: 2026-06-26
category: 项目
tags:
  - PyTorch
  - RNN
  - LSTM
  - 文本生成
description: 基于 PyTorch 的字符级 RNN 文本生成项目，支持 RNN、LSTM、GRU 三种循环网络与完整训练、评估、生成流程。
image: https://img.yumeko.site/file/blog/cover/1782480926632_CharRNN.webp
status: published
---

# Char-RNN 字符级文本生成项目

> **前置阅读**：建议先阅读 [[NeuralNetwork/RNN/RNNOverview|RNN 总览]]、[[NeuralNetwork/RNN/VanillaRNN|Vanilla RNN]]、[[NeuralNetwork/RNN/LSTM|LSTM]] 与 [[NeuralNetwork/RNN/GRU|GRU]]。

::github[repo=NayukiChiba/Char-RNN]

## 1. 项目定位

`Char-RNN` 是一个字符级文本生成项目。它使用莎士比亚文本作为训练语料，将文本拆成字符序列，
训练模型根据前面的字符预测下一个字符。

项目支持三种循环网络：

| 模型 | 说明 |
|:--|:--|
| `RNN` | 最基础的循环神经网络，适合理解隐藏状态递推 |
| `LSTM` | 引入门控与细胞状态，缓解长序列梯度消失 |
| `GRU` | 参数更少的门控循环网络，训练速度通常更快 |

语言模型的核心目标是学习：

$$
\boxed{
P(x_t \mid x_1, x_2, \dots, x_{t-1})
}
$$

其中 $x_t$ 是第 $t$ 个字符。训练完成后，模型可以从一个 prompt 开始逐字符采样，生成新的文本。

---

## 2. 项目结构

```text
Char-RNN/
├── main.py
├── config/
│   ├── defaults.py
│   ├── datasets.py
│   └── paths.py
├── src/
│   ├── data/
│   │   ├── char_vocab.py
│   │   ├── process.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── gru.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── checkpoint.py
│   │   ├── logger.py
│   │   └── optim.py
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── inference/
│   │   └── inference.py
│   └── cli/
│       ├── parser.py
│       └── menu.py
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── tensorboard/
```

分层思路很清晰：

| 层级 | 职责 |
|:--|:--|
| `config/` | 集中管理模型、训练、数据与路径参数 |
| `src/data/` | 字符词表、文本预处理、DataLoader |
| `src/models/` | RNN、LSTM、GRU 三种模型定义 |
| `src/training/` | 训练器、优化器、日志、检查点 |
| `src/evaluation/` | 验证集和测试集 loss、PPL |
| `src/inference/` | temperature 与 top-k 采样生成 |
| `src/cli/` | 命令行参数和交互式菜单 |

---

## 3. 数据与字符词表

字符级语言模型不做分词，而是把文本中的每个字符映射为整数 ID。

设字符表为：

$$
\mathcal{V} = \{c_1, c_2, \dots, c_{|\mathcal{V}|}\}
$$

则编码函数为：

$$
\operatorname{encode}(c_i) = i
$$

解码函数为：

$$
\operatorname{decode}(i) = c_i
$$

训练样本由连续文本切片得到。若上下文长度为 $T$，一条样本可以写成：

$$
\text{input} = [x_1, x_2, \dots, x_T]
$$

$$
\text{target} = [x_2, x_3, \dots, x_{T+1}]
$$

也就是 target 总是 input 右移一位。

---

## 4. 模型结构

项目把三类模型封装为同一类语言模型接口。典型结构为：

1. 字符 ID 经过 Embedding 层变成稠密向量。
2. 序列输入 RNN/LSTM/GRU。
3. 每个时间步的隐藏状态经过线性层映射到词表大小。
4. 使用交叉熵预测下一个字符。

以隐藏状态递推表示，RNN 的核心是：

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

$$
\hat{y}_t = \operatorname{softmax}(W_{hy}h_t + b_y)
$$

训练损失为：

$$
\boxed{
\mathcal{L}
= -\frac{1}{T}\sum_{t=1}^{T}\log P(x_{t+1} \mid x_{\le t})
}
$$

困惑度（Perplexity）由 loss 指数化得到：

$$
\operatorname{PPL} = \exp(\mathcal{L})
$$

---

## 5. 训练与评估

训练流程包括：

1. 加载或预处理莎士比亚文本。
2. 构建字符表。
3. 按序列长度切分训练样本。
4. 创建 `RNN`、`LSTM` 或 `GRU` 模型。
5. 使用交叉熵训练下一个字符预测。
6. 使用梯度裁剪控制循环网络梯度爆炸。
7. 保存 `latest.pth` 和 `best.pth`。

梯度裁剪的直觉是限制梯度范数：

$$
\boxed{
g \leftarrow g \cdot \min\left(1, \frac{c}{\lVert g \rVert_2}\right)
}
$$

其中 $c$ 是裁剪阈值。循环网络在长序列上容易出现梯度爆炸，因此这个步骤很重要。

---

## 6. CLI 使用

安装依赖：

```bash
uv sync
```

训练默认 LSTM：

```bash
python main.py train
```

指定模型和超参数：

```bash
python main.py train --rnn_type GRU --epochs 30 --lr 0.002 --optimizer AdamW
```

从检查点恢复：

```bash
python main.py train --resume outputs/checkpoints/LSTM/latest.pth
```

评估模型：

```bash
python main.py eval --checkpoint outputs/checkpoints/LSTM/best.pth
```

生成文本：

```bash
python main.py generate --prompt "ROMEO:" --temperature 0.8
```

---

## 7. 生成策略

推理时模型输出每个字符的 logits。temperature 用于控制采样随机性：

$$
p_i =
\operatorname{softmax}\left(\frac{z_i}{\tau}\right)
$$

其中 $\tau$ 是温度参数：

| $\tau$ | 生成效果 |
|:--|:--|
| 较小 | 更保守，更接近高概率字符 |
| 约等于 $1$ | 保持模型原始分布 |
| 较大 | 更随机，更多探索 |

Top-K 采样只保留概率最高的 $K$ 个候选字符，可以减少低质量尾部采样。

---

## 8. 总结

`Char-RNN` 是一个适合理解序列建模基本闭环的项目：

| 模块 | 价值 |
|:--|:--|
| 字符级词表 | 避免复杂分词，突出语言模型本身 |
| RNN/LSTM/GRU 对比 | 观察不同循环结构的训练差异 |
| PPL 指标 | 用统一指标评估语言模型质量 |
| 检查点 | 支持恢复训练和选择最佳模型 |
| 采样生成 | 把概率模型转成可见文本输出 |

它是从 RNN 理论走向工程实现的好入口，也适合作为后续接入 Attention 或 Transformer 的对照基线。

---

> **相关文章**：
> - [[NeuralNetwork/RNN/RNNOverview|RNN 总览]]
> - [[NeuralNetwork/RNN/LSTM|LSTM]]
> - [[NeuralNetwork/RNN/GRU|GRU]]
> - [[Projects/Attention|手搓 Attention 项目]]
