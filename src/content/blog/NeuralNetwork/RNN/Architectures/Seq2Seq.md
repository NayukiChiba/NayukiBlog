---
title: Seq2Seq：编码器-解码器架构
date: 2026-05-29
category: NeuralNetwork/RNN/Architectures
tags:
  - 经典架构
  - 深度学习
description: 2014 年 Sutskever 等人提出的 Sequence-to-Sequence 学习框架。完整数学推导、从零 PyTorch 实现、Teacher Forcing 策略分析、信息瓶颈讨论及解码策略对比。
image: https://img.yumeko.site/file/blog/Seq2Seq.webp
status: draft
---

## 1. 问题定义：变长序列到变长序列的映射

### 1.1 标准 RNN 模式的局限

RNN 可以处理四种输入输出模式：One-to-One、One-to-Many、Many-to-One、Many-to-Many（同步）。但这些模式有一个共同约束——Many-to-Many（同步）要求输出序列长度严格等于输入序列长度。

许多重要的 AI 问题需要**变长序列到变长序列**的映射：

- **机器翻译**："How are you?"（3 词）映射为"你好吗？"（3 字），但通常情况下源语言和目标语言的词数不同
- **文本摘要**：500 词文章压缩为 50 词摘要
- **对话系统**：用户问题（不定长）生成系统回复（不定长）

这些任务的输入和输出序列长度通常不同，标准的同步 Many-to-Many RNN（每输入一步输出一步）无法直接套用。

### 1.2 Seq2Seq 的核心创新：编码然后解码

2014 年，Sutskever、Vinyals 和 Le（Google）以及 Cho 等人（Montreal）同时提出了 Sequence-to-Sequence 框架。核心思想是将问题分解为两个独立阶段：

1. **编码（Encode）**：用一个 RNN（编码器）将整个输入序列 $X = (x_1, x_2, \dots, x_T)$ 压缩为一个固定长度的**上下文向量（Context Vector）** $C$
2. **解码（Decode）**：用另一个 RNN（解码器）从 $C$ 出发，**自回归地**生成输出序列 $Y = (y_1, y_2, \dots, y_S)$

两个 RNN 是**完全独立**的网络——编码器和解码器各有自己的权重矩阵、嵌入层和输出投影。唯一的连接点是上下文向量 $C$——解码器的初始状态由 $C$ 决定。

---

## 2. 完整数学推导

### 2.1 编码器

![Seq2Seq.png](https://img.yumeko.site/file/articles/Seq2Seq/Seq2Seq.webp)

编码器是一个标准的 RNN（实践中通常使用 LSTM），逐时间步读取输入序列 $X = (x_1, x_2, \dots, x_T)$。

**输入嵌入**：每个源语言 token $x_t$（整数索引）先通过嵌入层转为稠密向量：

$$
e_t = \text{Embedding}(x_t) \in \mathbb{R}^{d_e}
$$

**RNN 前向**：编码器 LSTM 从 $t=1$ 到 $T$ 逐步处理，每一步更新隐藏状态和细胞状态：

对于 $t = 1, 2, \dots, T$：

$$
h_t^{\text{enc}}, C_t^{\text{enc}} = \text{LSTM}_{\text{enc}}(e_t, h_{t-1}^{\text{enc}}, C_{t-1}^{\text{enc}})
$$

初始状态 $h_0^{\text{enc}} = \mathbf{0}$，$C_0^{\text{enc}} = \mathbf{0}$。

**上下文向量提取**：编码完成后，取最后一步的隐藏状态（和细胞状态，如果使用 LSTM）作为上下文向量：

$$
h_{\text{context}} = h_T^{\text{enc}} \in \mathbb{R}^{d_h}, \quad C_{\text{context}} = C_T^{\text{enc}} \in \mathbb{R}^{d_h}
$$

对于非 LSTM 的编码器（如 GRU），上下文向量仅包含 $h_T^{\text{enc}}$。$h_{\text{context}}$ 的维度固定为 $d_h$，与输入序列长度 $T$ 无关——这是信息瓶颈的结构性根源。

**编码器的参数**：嵌入矩阵 $E_{\text{src}} \in \mathbb{R}^{V_{\text{src}} \times d_e}$，LSTM 的四套权重 $(W_f, W_i, W_o, W_C)$ 及其偏置。编码器参数总量约为 $\text{LSTM}_{\text{单向}} \approx 4 \times d_h \times (d_h + d_e)$。

### 2.2 解码器

解码器是另一个**独立的** LSTM（或 GRU），以编码器的上下文向量为初始状态，自回归地生成目标序列 $Y = (y_1, y_2, \dots, y_S)$。

**初始状态**：解码器的隐藏状态和细胞状态由编码器的上下文向量初始化：

$$
s_0^{\text{dec}} = h_T^{\text{enc}}, \quad C_0^{\text{dec}} = C_T^{\text{enc}}
$$

**自回归生成**：对于解码步 $t = 1, 2, \dots, S$：

**Step 1——目标词嵌入**：将上一时刻生成的词 $y_{t-1}$ 转为稠密向量。当 $t=1$ 时，$y_0$ 为特殊标记 $\langle\text{SOS}\rangle$（Start-of-Sequence）：

$$
e_t^{\text{dec}} = \text{Embedding}_{\text{trg}}(y_{t-1}) \in \mathbb{R}^{d_e}
$$

**Step 2——LSTM 前向**：

$$
s_t^{\text{dec}}, C_t^{\text{dec}} = \text{LSTM}_{\text{dec}}(e_t^{\text{dec}}, s_{t-1}^{\text{dec}}, C_{t-1}^{\text{dec}})
$$

**Step 3——输出投影**：将解码器隐藏状态映射到目标词表空间，产生每个目标词的 logits：

$$
\text{logits}_t = W_o \cdot s_t^{\text{dec}} + b_o \in \mathbb{R}^{V_{\text{trg}}}
$$

**Step 4——概率化**：Softmax 产生目标词的概率分布：

$$
p(y_t | y_{<t}, X) = \text{softmax}(\text{logits}_t)
$$

其中 $V_{\text{trg}}$ 是目标语言词表大小，$W_o \in \mathbb{R}^{V_{\text{trg}} \times d_h}$，$b_o \in \mathbb{R}^{V_{\text{trg}}}$。

**解码器参数**：目标语言嵌入矩阵 $E_{\text{trg}} \in \mathbb{R}^{V_{\text{trg}} \times d_e}$，另一套 LSTM 权重，输出投影 $W_o, b_o$。解码器参数总量与编码器相当（当 $V_{\text{src}} \approx V_{\text{trg}}$ 时）。

**生成终止条件**：当 $y_t = \langle\text{EOS}\rangle$（End-of-Sequence）或 $t$ 达到预设的最大解码长度 $S_{\text{max}}$ 时停止。

### 2.3 训练目标：交叉熵损失

在训练时，整个目标序列 $Y = (y_1, y_2, \dots, y_S)$ 是已知的。Seq2Seq 的训练目标是最小化每一步预测与真实目标词的交叉熵：

$$
\mathcal{L} = -\sum_{t=1}^{S} \log p(y_t^{\text{true}} | y_{<t}, X)
$$

$$
= -\sum_{t=1}^{S} \log \left( \text{softmax}(W_o \cdot s_t^{\text{dec}} + b_o)_{y_t^{\text{true}}} \right)
$$

损失沿 $S$ 个时间步求和后通过 BPTT 反向传播，梯度同时更新编码器和解码器的所有参数。

---

## 3. 从零实现 Seq2Seq

### 3.1 编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqEncoder(nn.Module):
    """Seq2Seq 编码器：将源语言序列编码为上下文向量。

    流程:
      src (B, T) -> Embedding -> LSTM -> (h_T, c_T) 作为上下文向量
    """

    def __init__(self, src_vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()
        # 源语言词嵌入: (src_vocab_size, embed_dim)
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 编码器 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, src):
        """
        Args:
            src: 源语言索引序列, (batch, src_len)
        Returns:
            hidden: 最终隐藏状态, (num_layers, batch, hidden_dim)
            cell:   最终细胞状态, (num_layers, batch, hidden_dim)
        """
        # (batch, src_len) -> (batch, src_len, embed_dim)
        embedded = self.dropout(self.embedding(src))

        # outputs: (batch, src_len, hidden_dim)  —— 所有时间步输出（本处不用）
        # hidden:  (num_layers, batch, hidden_dim) —— 最终隐藏状态
        # cell:    (num_layers, batch, hidden_dim) —— 最终细胞状态
        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell
```

### 3.2 解码器

解码器在训练时使用 Teacher Forcing——以概率 `teacher_forcing_ratio` 使用真实目标词作为下一步输入。

```python
class Seq2SeqDecoder(nn.Module):
    """Seq2Seq 解码器：从上下文向量自回归生成目标序列。

    每步流程:
      y_{t-1} -> Embedding -> LSTM -> Linear -> logits
    """

    def __init__(self, trg_vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size
        # 目标语言词嵌入: (trg_vocab_size, embed_dim)
        self.embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 解码器 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # 输出投影: hidden_dim -> trg_vocab_size
        self.fc = nn.Linear(hidden_dim, trg_vocab_size)

    def forward(self, trg, hidden, cell, teacher_forcing_ratio=0.5):
        """
        Args:
            trg:  目标语言索引序列, (batch, trg_len)
            hidden: 编码器传入的初始隐藏状态, (num_layers, batch, hidden_dim)
            cell:   编码器传入的初始细胞状态, (num_layers, batch, hidden_dim)
            teacher_forcing_ratio: Teacher Forcing 概率
        Returns:
            outputs: 每步预测, (batch, trg_len, trg_vocab_size)
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)

        # 第一步输入: <SOS> token = trg[:, 0]
        # (batch,)
        decoder_input = trg[:, 0]

        # 预分配输出张量
        # (batch, trg_len, trg_vocab_size)
        outputs = torch.zeros(
            batch_size, trg_len, self.trg_vocab_size, device=trg.device
        )

        for t in range(1, trg_len):
            # 嵌入上一时刻的输出
            # (batch,) -> (batch, embed_dim) -> (batch, 1, embed_dim)
            embedded = self.dropout(self.embedding(decoder_input))
            embedded = embedded.unsqueeze(1)

            # LSTM 一步前向
            # lstm_output: (batch, 1, hidden_dim)
            # hidden:      (num_layers, batch, hidden_dim)
            # cell:        (num_layers, batch, hidden_dim)
            lstm_output, (hidden, cell) = self.lstm(
                embedded, (hidden, cell)
            )

            # 输出投影到目标词表空间
            # (batch, 1, hidden_dim) -> (batch, hidden_dim) -> (batch, trg_vocab_size)
            prediction = self.fc(lstm_output.squeeze(1))
            outputs[:, t, :] = prediction

            # Teacher Forcing 决策
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            # 取预测的最高概率词索引, (batch,)
            top1 = prediction.argmax(-1)
            # 训练: 以概率 teacher_forcing_ratio 使用真实目标词
            # 推理: teacher_forcing_ratio=0，始终使用模型预测
            decoder_input = trg[:, t] if teacher_force else top1

        return outputs
```

### 3.3 完整 Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    """完整的 Sequence-to-Sequence 模型。

    前向流程:
      src -> Encoder -> (hidden, cell) -> Decoder(trg) -> outputs
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: 源语言, (batch, src_len)
            trg: 目标语言, (batch, trg_len)
        Returns:
            outputs: (batch, trg_len, trg_vocab_size)
        """
        # 编码: src -> (hidden, cell)
        hidden, cell = self.encoder(src)

        # 解码: (hidden, cell) + trg -> 自回归生成
        outputs = self.decoder(
            trg, hidden, cell, teacher_forcing_ratio
        )
        return outputs
```

### 3.4 训练循环

```python
def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0.0

    for src, trg in dataloader:
        # src: (batch, src_len), trg: (batch, trg_len)
        optimizer.zero_grad()

        # 前向传播
        # outputs: (batch, trg_len, trg_vocab_size)
        outputs = model(src, trg, teacher_forcing_ratio=0.5)

        # 计算损失: 忽略 <SOS> token (trg[:, 0])
        # outputs[:, 1:]:  去掉第一个预测（对应<SOS>的预测无意义）
        # trg[:, 1:]:      去掉第一个真实标签（<SOS>）
        output_dim = outputs.size(-1)
        loss = criterion(
            outputs[:, 1:].reshape(-1, output_dim),
            trg[:, 1:].reshape(-1)
        )

        loss.backward()
        # 梯度裁剪，防止 LSTM 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
```

**损失计算注意**：`outputs` 的第 0 列是模型在看到 `<SOS>` 后对第一个目标词的预测——这一步的预测应该与 `trg[:, 1]`（第一个真实目标词）对齐。因此需要偏移一位：`outputs[:, 1:]` 与 `trg[:, 1:]` 比较。

### 3.5 推理（Greedy Decoding）

```python
def greedy_decode(model, src, src_vocab, trg_vocab, max_len=50):
    """使用训练好的 Seq2Seq 模型进行贪心推理。

    Args:
        model:     训练好的 Seq2Seq 模型
        src:       源语言索引序列, (1, src_len)
        src_vocab: 源语言词表（用于后续可能的处理）
        trg_vocab: 目标语言词表（含 <SOS> 和 <EOS> 索引）
        max_len:   最大解码长度
    Returns:
        output_tokens: 生成的目标词索引列表
    """
    model.eval()
    sos_idx = trg_vocab['<SOS>']
    eos_idx = trg_vocab['<EOS>']

    with torch.no_grad():
        # 编码
        hidden, cell = model.encoder(src)

        # 解码第一步: 输入 <SOS>
        decoder_input = torch.tensor([sos_idx], device=src.device)

        output_tokens = []
        for _ in range(max_len):
            # (1,) -> (1, 1, embed_dim)
            embedded = model.decoder.dropout(
                model.decoder.embedding(decoder_input)
            ).unsqueeze(1)

            # LSTM 一步前向
            lstm_output, (hidden, cell) = model.decoder.lstm(
                embedded, (hidden, cell)
            )

            # 预测下一个词
            # (1, 1, trg_vocab_size) -> (1, trg_vocab_size)
            prediction = model.decoder.fc(lstm_output.squeeze(1))

            # 贪心选择最高概率词
            top1 = prediction.argmax(-1).item()  # 标量
            output_tokens.append(top1)

            if top1 == eos_idx:
                break

            # 下一步输入: 当前步的预测
            decoder_input = torch.tensor([top1], device=src.device)

    return output_tokens
```

---

## 4. Teacher Forcing：训练与推理的桥梁

![TeacherForcing.png](https://img.yumeko.site/file/articles/Seq2Seq/TeacherForcing.webp)

### 4.1 两套公式

**训练时（Teacher Forcing）**——使用真实目标词：

$$
s_t^{\text{dec}} = \text{LSTM}_{\text{dec}}(y_{t-1}^{\text{true}}, s_{t-1}^{\text{dec}})
$$

**推理时（自回归）**——使用模型自身预测：

$$
s_t^{\text{dec}} = \text{LSTM}_{\text{dec}}(\hat{y}_{t-1}, s_{t-1}^{\text{dec}})
$$

### 4.2 Teacher Forcing 的利弊

**优势**：训练更快更稳定。真实目标词提供"正确答案"作为下一步输入，避免了早期预测错误导致后续步骤连锁崩溃。

**劣势——曝光偏差（Exposure Bias）**：训练时解码器每一步都接收到完美的真实目标词；推理时却只能接收自身（可能不完美）的预测。这种训练和推理的输入分布差异导致模型在推理时可能因一步错误而逐步偏离，生成质量下降。

### 4.3 Scheduled Sampling

一种缓解曝光偏差的方法是 Scheduled Sampling——训练过程中逐步降低 Teacher Forcing 的概率：

$$
p_{\text{tf}}(\text{epoch}) = \max\left(p_{\text{min}}, p_{\text{initial}} \cdot \lambda^{\text{epoch}}\right)
$$

其中 $\lambda \in (0, 1)$ 是衰减率。训练早期使用高 Teacher Forcing 概率保证稳定收敛，后期降低概率让模型逐渐适应自回归模式。

```python
def get_teacher_forcing_ratio(epoch, initial=1.0, decay=0.95, min_ratio=0.2):
    """计算当前 epoch 的 Teacher Forcing 概率（指数衰减）。"""
    return max(min_ratio, initial * (decay ** epoch))
```

---

## 5. 上下文向量的信息瓶颈

### 5.1 长句翻译的性能退化

Seq2Seq 被提出后，研究者很快发现翻译质量随输入长度急剧下降。根本原因：固定长度为 $d_h$ 的上下文向量 $C$ 必须编码变长序列的全部信息。

![BlueScore.png](https://img.yumeko.site/file/articles/Seq2Seq/BlueScore.webp)

以 $d_h = 512$ 为例，$C \in \mathbb{R}^{512}$ 的容量是固定的。对于 5 词的短句，512 维足够编码语义、语法和词汇信息。对于 50 词的长句，512 维需要编码 10 倍的信息量——向量空间被"挤满"，前面的信息被后来的词覆盖。

实验数据（Bahdanau 2015）：BLEU 分数随句子长度增加单调下降——短句（少于 10 词）BLEU 30+，中句（10-20 词）BLEU 约 25，长句（30 词以上）BLEU 低于 20。

### 5.2 信息瓶颈的三个后果

1. **早期信息被覆盖**：RNN 的隐藏状态在每个时间步被重写，长句靠前位置的信息在经历数十次更新后被后来的词覆盖，上下文向量中几乎不含句首的语义
2. **消歧失败**：多义词语的消歧依赖完整上下文——但上下文在瓶颈处已被丢失，解码器无法区分"bank"是"银行"还是"河岸"
3. **缺乏直接对应**：目标序列中的每个词间接通过单一向量 $C$ 与源序列关联——输出 "dog" 时无法直接定位到源序列中的 "狗"，只能依赖 $C$ 中可能已被稀释的信息

这个瓶颈直接催生了 Attention 机制的核心思想——不再将所有信息压缩到一个向量，而是让解码器可以直接访问编码器的所有时间步输出。

---

## 6. 解码策略

### 6.1 贪心解码

每步选择概率最高的词：$\hat{y}_t = \arg\max_{y} p(y | \hat{y}_{<t}, X)$。

**优点**：速度快，每步只需一次 argmax。**缺点**：一旦某步选错，无法回头纠正。贪心解码追求的是局部最优（每步概率最大），不保证全局最优（整句概率最大）。

### 6.2 Beam Search

![BeamSearch.png](https://img.yumeko.site/file/articles/Seq2Seq/BeamSearch.webp)

维护 $k$（beam size）条最优候选序列，每步扩展并保留累积概率最高的 $k$ 条。以 $k=2$ 为例：

**第 1 步**：模型给出 3 个候选 $A(0.6)$、$B(0.3)$、$C(0.1)$，保留 $A, B$。

**第 2 步**：从 $A$ 和 $B$ 出发扩展，得到 $AA(0.2)$、$AB(0.5)$、$BA(0.4)$、$BB(0.1)$。计算累积对数概率（实际使用 $\log$ 避免浮点下溢）：

$$
\begin{aligned}
\log P(AB) &= \log 0.6 + \log 0.5 = -0.511 - 0.693 = -1.204 \\
\log P(BA) &= \log 0.3 + \log 0.4 = -1.204 - 0.916 = -2.120 \\
\log P(AA) &= \log 0.6 + \log 0.2 = -0.511 - 1.609 = -2.120 \\
\log P(BB) &= \log 0.3 + \log 0.1 = -1.204 - 2.303 = -3.507
\end{aligned}
$$

保留累积概率最高的两条 $AB$ 和 $BA$（$AA$ 与 $BA$ 概率相同，可任选其一或按词典序）。

**Beam Size 的选择**：
- $k=1$：等价于贪心解码
- $k=3-10$：常见设置，在质量和速度间取得良好平衡
- $k=100+$：搜索更全面，但计算成本高且收益递减

### 6.3 长度惩罚

Beam Search 天然偏向短序列——因为每步的概率都小于 1，短序列的累积概率通常高于长序列。实践中通过长度归一化来纠正：

$$
\text{score}(Y) = \frac{1}{|Y|^\alpha} \sum_{t=1}^{|Y|} \log p(y_t | y_{<t}, X)
$$

其中 $\alpha \in [0.6, 1.0]$ 控制惩罚强度。$\alpha=0$ 不惩罚（偏向短句），$\alpha=1$ 完全按长度归一化。

---

## 7. Seq2Seq 的历史意义

Seq2Seq 是 NLP 历史上具有深远影响的架构创新。它的意义超越了机器翻译本身：

- **统一了多种任务**：翻译、摘要、对话、问答等不同任务都可以用同一个框架处理——编码器编码输入，解码器生成输出
- **建立了编码-解码范式**：将复杂的序列转换问题分解为"理解"（编码）和"生成"（解码）两个子问题，这种分离设计影响了后续几乎所有生成模型
- **直接催生了 Attention**：信息瓶颈是 Seq2Seq 最大的结构性问题，解决它的尝试直接产生了注意力机制——深度学习最重要的创新之一
- **为预训练铺路**：Seq2Seq 证明了 RNN 可以从大规模平行语料中学习复杂的序列映射，为后来的预训练语言模型奠定了基础

> "Sequence to Sequence Learning with Neural Networks" 被引用了超过 24000 次（截至 2025 年），是深度学习历史上被引最多的论文之一。

---

> 克服信息瓶颈的方案：[[NeuralNetwork/RNN/Architectures/AttentionRNN|注意力 RNN]]
> 回到主文档：[[NeuralNetwork/RNN/RNN|RNN 详解主文档]]
