---
title: 中文情感分类项目
date: 2026-06-26
category: 项目
tags:
  - PyTorch
  - RNN
  - NLP
  - 情感分析
description: 基于 RNN、LSTM、GRU 与 Attention 的中文情感二分类项目，覆盖预处理、训练、评估、可视化和推理。
image: https://img.yumeko.site/file/blog/cover/1782481118087_EmotionClassification.webp
status: published
---

# 中文情感分类项目

> **前置阅读**：建议先阅读 [[NeuralNetwork/RNN/EmbeddingLayer|Embedding 层]]、[[NeuralNetwork/RNN/BidirectionalRNN|双向 RNN]] 与 [[NeuralNetwork/RNN/Attention|注意力机制]]。

::github[repo=NayukiChiba/EmotionClassification]

## 1. 项目定位

`EmotionClassification` 是一个中文情感二分类项目，面向电商评论类数据集。输入是一段中文评论，输出是正面或负面情感标签。

项目目标不是只训练一个模型，而是把 NLP 分类任务的完整工程流程写清楚：

1. 读取评论 CSV。
2. 中文分词和停用词过滤。
3. 构建词表并缓存预处理结果。
4. 训练 RNN、LSTM 或 GRU。
5. 可选启用双向结构和 Attention。
6. 输出 Accuracy、Precision、Recall、F1、AUC。
7. 支持单条文本和 CSV 批量预测。

情感二分类本质上是在估计：

$$
\boxed{
P(y=1 \mid x_1, x_2, \dots, x_T)
}
$$

其中 $y=1$ 表示正面评论，$y=0$ 表示负面评论。

---

## 2. 项目结构

```text
EmotionClassification/
├── main.py
├── config/
│   ├── default.py
│   └── paths.py
├── datasets/
│   ├── raw/
│   └── processed/
├── src/
│   ├── cli/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   └── inference/
├── tests/
└── outputs/
    ├── checkpoints/
    ├── figures/
    ├── logs/
    └── models/
```

| 模块 | 职责 |
|:--|:--|
| `src/data/` | 文本清洗、分词、词表、Dataset 和 DataLoader |
| `src/models/` | RNN、LSTM、GRU、Attention 分类模型 |
| `src/train/` | 训练器、优化器、学习率调度、早停、checkpoint |
| `src/evaluate/` | 分类指标、混淆矩阵、ROC 曲线 |
| `src/inference/` | 单条预测和批量 CSV 预测 |
| `config/` | 路径、模型参数和训练参数 |

---

## 3. 数据格式

默认原始数据路径为：

```text
datasets/raw/online_shopping_10_cats.csv
```

CSV 至少需要包含两列：

| 列名 | 说明 |
|:--|:--|
| `text` | 评论文本 |
| `label` | 情感标签，`0` 为负面，`1` 为正面 |

预处理后会生成缓存：

```text
datasets/processed/
```

> [!WARNING] 缓存注意
> 如果修改了原始数据、最大序列长度、最小词频或分词策略，需要清理 `datasets/processed/` 后重新训练，避免复用旧缓存。

---

## 4. 文本到张量

中文评论首先经过分词，得到词序列：

$$
x = [w_1, w_2, \dots, w_T]
$$

词表把每个词映射为整数：

$$
\operatorname{id}(w_i) \in \{0, 1, \dots, |\mathcal{V}|-1\}
$$

Embedding 层将离散 ID 转为稠密向量：

$$
e_i = E[\operatorname{id}(w_i)]
$$

其中 $E \in \mathbb{R}^{|\mathcal{V}| \times d}$ 是可学习的嵌入矩阵。

---

## 5. 模型结构

基础结构可以概括为：

1. `Embedding`：将词 ID 转为向量。
2. `RNN/LSTM/GRU`：编码整段评论。
3. `Attention`：可选，对所有时间步加权聚合。
4. `Linear`：输出二分类 logits。
5. `Sigmoid` 或二分类交叉熵：得到情感概率。

如果启用 Attention，上下文向量为：

$$
\alpha_i =
\frac{\exp(s_i)}
{\sum_{j=1}^{T}\exp(s_j)}
$$

$$
\boxed{
c = \sum_{i=1}^{T}\alpha_i h_i
}
$$

其中 $h_i$ 是 RNN 在第 $i$ 个词位置的隐藏状态，$\alpha_i$ 是该词的注意力权重。

这种结构比只取最后一个隐藏状态更适合评论分类，因为情绪词可能出现在句子任意位置。

---

## 6. 评估指标

项目输出常见二分类指标：

| 指标 | 公式 | 说明 |
|:--|:--|:--|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | 整体预测正确率 |
| Precision | $\frac{TP}{TP+FP}$ | 预测为正面中有多少是真的正面 |
| Recall | $\frac{TP}{TP+FN}$ | 真实正面中有多少被找回 |
| F1 | $\frac{2PR}{P+R}$ | Precision 与 Recall 的调和平均 |
| AUC | ROC 曲线下的面积 | 衡量排序能力 |

对于评论情感分类，单看 Accuracy 不够。若数据集正负样本不均衡，F1 和 AUC 更能反映模型质量。

---

## 7. CLI 使用

安装依赖：

```bash
uv sync
```

训练 LSTM：

```bash
python main.py train --data datasets/raw/online_shopping_10_cats.csv --model lstm --epochs 20
```

指定批次和学习率：

```bash
python main.py train --data datasets/raw/online_shopping_10_cats.csv --model lstm --epochs 20 --batch_size 64 --learning_rate 0.01
```

评估模型：

```bash
python main.py eval --checkpoint outputs/models/best_model.pth
```

单条文本预测：

```bash
python main.py predict --checkpoint outputs/models/best_model.pth --text "物流很快，包装也很好"
```

批量预测：

```bash
python main.py predict --checkpoint outputs/models/best_model.pth --input datasets/raw/predict_samples.csv --output outputs/predictions.csv
```

---

## 8. 输出产物

典型输出包括：

| 路径 | 说明 |
|:--|:--|
| `outputs/models/best_model.pth` | 最佳模型 |
| `outputs/models/last_model.pth` | 最新模型 |
| `outputs/checkpoints/` | 训练检查点 |
| `outputs/figures/training_curve.png` | 训练曲线 |
| `outputs/figures/confusion_matrix.png` | 混淆矩阵 |
| `outputs/figures/roc_curve.png` | ROC 曲线 |
| `outputs/predictions.csv` | 批量预测结果 |

---

## 9. 总结

这个项目适合作为中文文本分类的标准模板：

| 主题 | 项目处理方式 |
|:--|:--|
| 中文分词 | jieba + 停用词过滤 |
| 序列编码 | Embedding + RNN/LSTM/GRU |
| 长距离信息 | 双向结构和 Attention |
| 训练稳定性 | 学习率调度、早停、梯度裁剪 |
| 结果解释 | 混淆矩阵、ROC、注意力权重 |
| 部署前准备 | 单条预测和 CSV 批量预测 |

它与 `Char-RNN` 的区别在于：`Char-RNN` 是生成任务，而本项目是判别任务；一个预测下一个字符，一个预测整段文本的情感标签。

---

> **相关文章**：
> - [[NeuralNetwork/RNN/EmbeddingLayer|Embedding 层]]
> - [[NeuralNetwork/RNN/BidirectionalRNN|双向 RNN]]
> - [[NeuralNetwork/RNN/Attention|注意力机制]]
> - [[Projects/CharRNN|Char-RNN 字符级文本生成项目]]
