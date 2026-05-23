---
title: Dropout 详解：随机失活防过拟合
date: 2026-05-07
category: NeuralNetwork/Theory
tags:
  - 基础
  - Dropout
  - 正则化
description: 理解 Dropout 的集成学习视角、概率设置策略，以及 train/eval 模式切换的关键细节。
image: https://img.yumeko.site/file/blog/Dropout.png
status: published
---

## 1. 过拟合是什么？

过拟合是指模型在训练数据上表现很好（甚至接近 100%），但在未见过的测试数据上表现很差。就像学生只会做见过的例题，换个新题就不会了。

## 2. Dropout 的原理

Dropout 是 2014 年由 Srivastava 等人提出的正则化技术，核心思想极为简单：

**训练时**：每个神经元以概率 $p$ 被随机"丢弃"（输出置为 0），以概率 $(1-p)$ 被保留。每个 mini-batch 都会随机丢弃不同的神经元，所以实际上每次训练的都是一个不同的"子网络"。

**测试/推理时**：所有神经元都参与运算（相当于集成了所有子网络），输出不做丢弃。

**为什么有效？**
- 强迫神经元独立学习有用特征，不能依赖特定同伴的存在
- 相当于训练了大量子网络并在测试时集成（ensemble）
- 破坏了神经元之间的共适应（co-adaptation）

直观理解：Dropout 就像考试复习时不让你依赖同桌的笔记——你必须自己把每个知识点都学好。

![Result.png](https://img.yumeko.site/file/articles/Dropout/Result.png)

## 3. 概率设置建议

| 层类型 | 推荐 Dropout 概率 |
| --- | :---: |
| 输入层 | $0.0 \sim 0.2$（通常不 dropout 输入） |
| 隐藏层（FC） | **0.5**（最常用的值） |
| 卷积层 | $0.0 \sim 0.2$（卷积层参数少，不太需要 dropout） |
| 输出层 | **0.0**（绝对不能 dropout 输出） |

卷积层不太需要 Dropout 的原因：卷积层通过权重共享已经大幅减少了参数，本身就不容易过拟合。而且卷积层提取的是空间特征，随机丢弃会破坏空间结构。

## 4. 代码示例

```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3136, 128)
        self.dropout = nn.Dropout(p=0.5)  # 50% 概率丢弃
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)   # 训练时随机丢弃，eval时不动
        x = self.fc2(x)
        return x
```

## 5. 关键细节：train/eval 模式切换

这是最常见的初学者陷阱（务必掌握 [[NeuralNetwork/Tips/Techniques/TrainEvalMode|模式切换]] 的正确时机）：

```python
model.train()   # Dropout 生效：随机丢弃神经元
model.eval()    # Dropout 不生效：所有神经元参与

# ❌ 常见错误：验证时忘了 eval()
model.train()   # 仍然是 train 模式！
with torch.no_grad():
    valOutput = model(valImages)  # Dropout 还在随机丢弃！

# ✅ 正确做法
model.eval()
with torch.no_grad():
    valOutput = model(valImages)
```

**忘记 `eval()` 的后果**：
- 预测结果不可复现（每次不同）
- 准确率显著下降
- 验证损失异常升高

## 6. Dropout 与 BatchNorm 共用的注意事项

在 `model.eval()` 模式下：

- Dropout 自动关闭
- BatchNorm 切换到全局统计量

所以只要正确调用 `model.eval()`，两者都能自动正常工作，不需要分别处理。

但有一个微妙之处：**Dropout 和 [[NeuralNetwork/Theory/BatchNormalization|BatchNorm]] 一起使用时，训练和测试的方差会有偏差**。在训练时，BN 层的输入经过了 Dropout（神经元随机置零），方差与测试时（无障碍通过）不同。这就是所谓的"方差偏移"问题。在实践中，只要 batch size 不是太小，这个问题影响有限。

## 7. 现代趋势

近年来，Dropout 在 CNN 中的使用有所减少，被 BatchNorm（自带轻微正则化）和数据增强（更直接的正则化）部分取代。但在全连接层中，Dropout 仍然是非常有效的正则化手段。

在 Transformer 架构中，Dropout 仍然被广泛使用（注意机制、FFN 层、嵌入层之后）。
