---
title: 手搓 Attention 项目
date: 2026-06-26
category: 项目
tags:
  - PyTorch
  - Attention
  - Transformer
  - GPT
description: 从缩放点积注意力开始，手写多头注意力、Transformer Block 与小型 GPT 中文文本生成模型。
image: https://img.yumeko.site/file/blog/cover/1782480856710_Attention.webp
status: published
---

# 手搓 Attention 项目

> **前置阅读**：本文默认读者熟悉 [[NeuralNetwork/RNN/Attention|注意力机制]] 与 [[NeuralNetwork/Transformer/TransformerOverview|Transformer 架构总览]]。

::github[repo=NayukiChiba/Attention]

## 1. 项目定位

`Attention` 是一个从零实现 Transformer 解码器的教学项目。项目不直接使用
`torch.nn.MultiheadAttention`，而是从最核心的缩放点积注意力开始，逐层搭建多头注意力、
Transformer Block、GPT 语言模型、训练循环、评估器和文本生成器。

核心目标很明确：把下面这个公式拆成可读、可测、可运行的 PyTorch 代码。

$$
\boxed{
\operatorname{Attention}(Q, K, V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
}
$$

其中 $Q$ 是查询矩阵，$K$ 是键矩阵，$V$ 是值矩阵，$d_k$ 是单个注意力头的键向量维度，
$M$ 是因果掩码和填充掩码组合后的 mask。

项目最终训练的是一个 GPT 风格的中文新闻文本生成模型，数据集使用 THUCNews。

---

## 2. 项目结构

```text
Attention/
├── main.py
├── config/
│   ├── defaults.py
│   └── paths.py
├── src/
│   ├── cli/
│   │   ├── menu.py
│   │   └── parser.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── tokenizer.py
│   ├── evaluate/
│   │   ├── evaluator.py
│   │   └── visualize.py
│   ├── inference/
│   │   └── generator.py
│   ├── model/
│   │   ├── embedding.py
│   │   ├── feedForward.py
│   │   ├── gpt.py
│   │   ├── mask.py
│   │   ├── multiHeadAttention.py
│   │   ├── scaledDotProductAttention.py
│   │   └── transformerBlock.py
│   └── train/
│       ├── checkpoint.py
│       ├── early_stopping.py
│       ├── logger.py
│       ├── optimizer.py
│       ├── scheduler.py
│       ├── trainer.py
│       └── utils.py
├── tests/
├── outputs/
└── pyproject.toml
```

这套结构按职责分层：

| 层级 | 目录 | 职责 |
|:--|:--|:--|
| 配置层 | `config/` | 管理路径、模型超参数、训练超参数、生成超参数 |
| 数据层 | `src/data/` | 下载、清洗、划分、分词、构建语言模型数据集 |
| 模型层 | `src/model/` | 从 Attention 到 GPT 的完整网络结构 |
| 训练层 | `src/train/` | 优化器、调度器、早停、检查点、日志和训练循环 |
| 评估层 | `src/evaluate/` | 测试集 loss、困惑度和注意力热力图 |
| 推理层 | `src/inference/` | Top-K、Top-P、温度、重复惩罚等生成策略 |
| 入口层 | `main.py` / `src/cli/` | 命令行子命令和交互式菜单 |

---

## 3. 模型实现说明

项目的模型实现按“基础组件、注意力层、Transformer Block、GPT 外壳”拆分。这样做的好处是每一层职责都很窄：
`mask.py` 只处理可见性约束，`scaledDotProductAttention.py` 只负责单头注意力计算，
`multiHeadAttention.py` 负责多头拆分与合并，`transformerBlock.py` 再组合 Attention、FFN、残差和 LayerNorm。
最外层的 `gpt.py` 负责堆叠模块并输出下一个 token 的 logits。

### 3.1 Mask：限制模型偷看未来

GPT 是自回归语言模型，当前位置 $t$ 只能看到 $0$ 到 $t$ 的 token，不能看到未来 token。
因此项目在 `mask.py` 中实现了因果掩码：

$$
M_{i,j} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

代码中使用布尔矩阵表示 mask：`True` 表示该位置需要被遮住，`False` 表示允许被关注。

填充掩码由输入 token 是否等于 `<PAD>` 决定：

$$
P_{b,j} = \mathbf{1}_{x_{b,j} = \text{PAD}}
$$

最终 mask 是因果掩码和填充掩码的逻辑或：

$$
\boxed{
M_{\text{final}} = M_{\text{causal}} \lor M_{\text{padding}}
}
$$

### 3.2 Scaled Dot-Product Attention

`ScaledDotProductAttention` 是整个项目最底层的核心。输入形状为：

| 张量 | 形状 |
|:--|:--|
| `query` | `(batch_size, num_attention_heads, seq_length_q, d_k)` |
| `key` | `(batch_size, num_attention_heads, seq_length_k, d_k)` |
| `value` | `(batch_size, num_attention_heads, seq_length_v, d_v)` |
| `mask` | `(batch_size, seq_length_q, seq_length_k)` |
| `output` | `(batch_size, num_attention_heads, seq_length_q, d_v)` |

计算步骤：

1. 用 `query @ key.transpose(-2, -1)` 得到注意力分数。
2. 除以 $\sqrt{d_k}$，避免 $QK^\top$ 数值过大导致 Softmax 饱和。
3. 将 mask 位置填成 $-\infty$。
4. 沿最后一维做 Softmax，得到注意力权重。
5. 用注意力权重乘以 $V$，得到加权后的上下文表示。

对应公式为：

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

$$
A = \operatorname{softmax}(S + M)
$$

$$
O = AV
$$

### 3.3 Multi-Head Attention

`MultiHeadAttention` 将输入投影为 $Q$、$K$、$V$，拆成多个头并行计算注意力，然后再拼回原始维度。

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

若模型维度为 $d_{\text{model}}$，注意力头数为 $h$，则单头维度为：

$$
\boxed{
d_{\text{head}} = \frac{d_{\text{model}}}{h}
}
$$

项目默认配置中：

| 参数 | 默认值 |
|:--|--:|
| $d_{\text{model}}$ | 384 |
| $h$ | 8 |
| $d_{\text{head}}$ | 48 |

多头注意力的完整输出为：

$$
\operatorname{MHA}(X)
= \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)W_O
$$

其中：

$$
\operatorname{head}_i
= \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})
$$

### 3.4 Transformer Block

`TransformerBlock` 支持 Pre-Norm 和 Post-Norm。默认使用 Pre-Norm，因为它在较深网络中通常更稳定。

Pre-Norm 结构：

$$
x' = x + \operatorname{MHA}(\operatorname{LN}(x))
$$

$$
y = x' + \operatorname{FFN}(\operatorname{LN}(x'))
$$

Post-Norm 结构：

$$
x' = \operatorname{LN}(x + \operatorname{MHA}(x))
$$

$$
y = \operatorname{LN}(x' + \operatorname{FFN}(x'))
$$

项目把 `norm_type` 暴露到配置中，方便对比两种 LayerNorm 放置方式。

### 3.5 GPT 模型

`GPT` 由三部分组成：

1. `GPTEmbedding`：Token Embedding + Position Encoding + Dropout。
2. `TransformerBlock` 堆叠：默认 $8$ 层。
3. `language_modeling_head`：将隐藏状态投影到词表大小。

前向传播形状变化：

| 阶段 | 形状 |
|:--|:--|
| `input_ids` | `(batch_size, seq_length)` |
| Embedding | `(batch_size, seq_length, embedding_dim)` |
| Transformer Blocks | `(batch_size, seq_length, embedding_dim)` |
| LM Head | `(batch_size, seq_length, vocab_size)` |

训练目标是预测下一个 token。给定序列：

$$
x = [x_1, x_2, \dots, x_T]
$$

模型学习：

$$
\boxed{
P(x_t \mid x_{<t})
}
$$

损失函数是忽略 `<PAD>` 的交叉熵：

$$
\mathcal{L}
= -\sum_{t=1}^{T}
\log P(x_t \mid x_{<t})
$$

---

## 4. 默认配置

### 4.1 模型配置

| 参数 | 默认值 | 说明 |
|:--|--:|:--|
| `vocab_size` | 运行时由 tokenizer 确定 | 字符级词表大小 |
| `context_length` | 128 | 最大上下文长度 |
| `embedding_dim` | 384 | 模型隐藏维度 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_layers` | 8 | Transformer Block 层数 |
| `ffn_hidden_dim` | 1536 | FFN 中间层维度 |
| `dropout_rate` | 0.1 | Dropout 比例 |
| `pos_encoding_type` | `sinusoidal` | 位置编码类型 |
| `activation` | `relu` | FFN 激活函数 |
| `norm_type` | `pre` | LayerNorm 位置 |
| `share_embedding_weights` | `True` | 输入嵌入与输出头权重共享 |

参数量大约为千万级，适合在个人显卡上做教学训练和小规模实验。

### 4.2 训练配置

| 参数 | 默认值 | 说明 |
|:--|--:|:--|
| `batch_size` | 32 | 批次大小 |
| `max_epochs` | 50 | 最大训练轮数 |
| `total_steps` | 1000000 | 最大训练步数 |
| `validation_interval` | 100 | 每隔多少 step 验证并保存 |
| `validation_batches` | 100 | 快速验证 batch 数 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `optimizer_type` | `adamw` | 优化器 |
| `learning_rate` | $3 \times 10^{-4}$ | 初始学习率 |
| `scheduler_type` | `cosine_warmup` | 学习率调度 |
| `warmup_steps` | 500 | 预热步数 |

训练过程中会保存：

| 文件 | 说明 |
|:--|:--|
| `outputs/checkpoints/best_model.pth` | 验证 loss 最优模型 |
| `outputs/checkpoints/last_model.pth` | 最新模型 |
| `outputs/logs/` | 训练日志 |
| `outputs/tensorboard/` | TensorBoard 日志 |
| `outputs/figures/training_curves.png` | 训练曲线 |
| `outputs/figures/attention_head_*.png` | 注意力热力图 |

---

## 5. 数据处理

项目使用 THUCNews 中文新闻数据集。原始数据按类别目录存放，预处理后统一保存为：

```text
D:/Datasets/Attention/
├── raw/
│   └── THUCNews/
├── interim/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── processed/
    └── vocab.json
```

每行数据格式为：

```text
label	text
```

数据处理流程：

1. 读取各新闻类别目录下的 `.txt` 文件。
2. 清理 BOM、零宽字符、控制字符和多余空白。
3. 根据 `min_text_length` 与 `max_text_length` 过滤或截断文本。
4. 按类别分组后划分训练集、验证集、测试集。
5. 仅使用训练集构建字符级词表，避免验证集和测试集信息泄露。

字符级分词器包含四个特殊 token：

| Token | ID | 作用 |
|:--|--:|:--|
| `<PAD>` | 0 | 序列补齐 |
| `<UNK>` | 1 | 未知字符 |
| `<BOS>` | 2 | 序列开始 |
| `<EOS>` | 3 | 序列结束 |

`NewsDataset` 返回语言模型训练所需的 `(input_ids, target_ids)`：

$$
\text{input} = [\text{BOS}, x_1, x_2, \dots, x_{T-1}]
$$

$$
\text{target} = [x_1, x_2, \dots, x_T, \text{EOS}]
$$

也就是 target 是 input 右移一位后的下一个 token 序列。

---

## 6. CLI 使用

### 6.1 安装依赖

```bash
uv sync
```

### 6.2 训练模型

```bash
python main.py train
```

常用调参示例：

```bash
python main.py train --batch-size 16 --context-length 256 --num-layers 6 --num-heads 6
```

从检查点恢复训练：

```bash
python main.py train --resume outputs/checkpoints/last_model.pth
```

### 6.3 评估模型

```bash
python main.py eval --checkpoint best_model.pth
```

评估验证集前若干 batch：

```bash
python main.py eval --checkpoint best_model.pth --split val --max-batches 50
```

### 6.4 生成文本

```bash
python main.py generate --checkpoint best_model.pth --prompt "今天" --max-tokens 100
```

调整采样策略：

```bash
python main.py generate --checkpoint best_model.pth --prompt "中国经济" --temperature 0.8 --top-k 50 --top-p 0.9
```

不提供 `--prompt` 时会进入交互式生成模式。

---

## 7. 训练循环

`Trainer` 是训练层唯一调度者。它负责把模型、数据、优化器、调度器、早停、检查点和日志串起来。

核心训练步骤：

1. 将模型移动到 `TrainingConfig.device`。
2. 创建优化器和学习率调度器。
3. 对每个 batch 执行前向传播。
4. 使用交叉熵计算语言模型损失。
5. 反向传播并裁剪梯度。
6. 更新优化器和学习率。
7. 每隔 `validation_interval` step 做快速验证。
8. 保存最佳模型和最新模型。
9. 训练结束后生成训练曲线和注意力热力图。

困惑度（Perplexity）由 loss 指数化得到：

$$
\boxed{
\operatorname{PPL} = \exp(\mathcal{L})
}
$$

PPL 越低，说明模型对下一个 token 的预测越确定。

---

## 8. 推理与采样

`TextGenerator` 逐 token 生成文本。每一步只取最后一个位置的 logits：

$$
z_t = f_\theta(x_{\le t})_{t}
$$

然后经过温度缩放：

$$
p_i =
\operatorname{softmax}\left(\frac{z_i}{\tau}\right)
$$

其中 $\tau$ 是 temperature。$\tau$ 越大，分布越平；$\tau$ 越小，生成越保守。

项目支持：

| 策略 | 说明 |
|:--|:--|
| Temperature | 控制随机性 |
| Top-K | 只保留概率最高的 $K$ 个 token |
| Top-P | 保留累计概率达到 $p$ 的最小 token 集合 |
| Repetition Penalty | 降低已出现 token 的概率 |
| No Repeat N-Gram | 屏蔽会形成重复 n-gram 的候选 token |

> [!NOTE] 关于 KV Cache
> 配置中保留了 `use_kv_cache` 选项，但当前生成器仍按完整上下文重新前向。
> 后续可以把每层的 $K,V$ 缓存下来，将单步生成从重复计算整段上下文改成只计算新增 token。

---

## 9. 项目设计取舍

### 9.1 为什么不用 `nn.MultiheadAttention`

项目的目标是教学透明，而不是最短代码。手写注意力能显式看到：

- $Q,K,V$ 如何由线性层投影得到。
- 多头如何从 `(B, L, D)` 拆成 `(B, H, L, D/H)`。
- mask 如何广播到注意力分数。
- Softmax 为什么沿最后一维归一化。
- 多头输出如何拼回 `embedding_dim`。

这些细节如果全部交给库函数，会更难定位张量形状错误。

### 9.2 为什么使用字符级 tokenizer

字符级 tokenizer 的优点是简单、确定、无额外分词依赖，适合教学项目。缺点也很明显：

| 维度 | 字符级 tokenizer |
|:--|:--|
| 实现复杂度 | 低 |
| OOV 问题 | 少 |
| 序列长度 | 偏长 |
| 语义粒度 | 较细 |
| 中文新闻生成 | 可用，但效率不如子词 tokenizer |

后续若追求生成质量，可以替换为 BPE、SentencePiece 或其他子词分词器。

### 9.3 为什么默认 Pre-Norm

Pre-Norm 把 LayerNorm 放在子层之前，残差路径更直接：

$$
x_{l+1} = x_l + F(\operatorname{LN}(x_l))
$$

这通常比 Post-Norm 更利于深层 Transformer 的梯度传播。项目仍保留 `post` 选项，方便做对照实验。

---

## 10. 可扩展方向

| 方向 | 修改位置 | 说明 |
|:--|:--|:--|
| 加入 KV Cache | `src/inference/generator.py`、`src/model/gpt.py` | 降低自回归生成重复计算 |
| 替换 tokenizer | `src/data/tokenizer.py`、`src/data/dataset.py` | 从字符级切换到 BPE 或 SentencePiece |
| 加入 Flash Attention | `src/model/scaledDotProductAttention.py` | 在保持接口不变的情况下优化注意力计算 |
| 支持分类任务 | `src/model/gpt.py` 新增分类头 | 复用 Transformer Backbone 做新闻分类 |
| 增加配置文件 | `config/` 与 CLI | 将命令行参数迁移到 TOML/YAML |
| 完善单元测试 | `tests/` | 覆盖模型形状、mask、采样和训练恢复 |

---

## 11. 总结

这个项目的价值不在于训练出一个大型语言模型，而在于把 Transformer 最核心的计算拆开：

| 模块 | 解决的问题 |
|:--|:--|
| `mask.py` | 让自回归模型不能偷看未来，也不关注 PAD |
| `scaledDotProductAttention.py` | 实现 $QK^\top$、缩放、Softmax、加权求和 |
| `multiHeadAttention.py` | 将注意力拆到多个子空间并行学习 |
| `transformerBlock.py` | 组合 Attention、FFN、残差和 LayerNorm |
| `gpt.py` | 堆叠 Decoder Block，输出下一个 token 的 logits |
| `trainer.py` | 完整训练循环、验证、早停和检查点 |
| `generator.py` | 将 logits 转成可控的文本生成过程 |

从学习路线看，它适合作为从 [[NeuralNetwork/RNN/Attention|RNN Attention]] 走向
[[NeuralNetwork/Transformer/TransformerOverview|Transformer]] 的工程桥梁：先看懂公式，再看懂张量形状，最后跑通一个能训练、能评估、能生成的最小 GPT。

---

> **相关文章**：
> - [[NeuralNetwork/RNN/Attention|注意力机制详解：从信息瓶颈到 Bahdanau 注意力]]
> - [[NeuralNetwork/Transformer/TransformerOverview|Transformer 架构总览]]
> - [[Projects/MachineLearning|机器学习全汇总项目]]
