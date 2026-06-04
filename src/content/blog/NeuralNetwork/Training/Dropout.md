---
title: Dropout：从集成学习视角到实战配置
date: 2026-05-09
category: NeuralNetwork/Training
tags:
  - 基础
  - 深度学习
  - 高级教程
description: 深入理解 Dropout 的原理（集成学习视角、共适应破坏）、概率设置策略、train/eval 模式切换的关键细节，以及与 BatchNorm 共用时的注意事项。
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
| 隐藏层（全连接） | **0.5**（最常用的值） |
| 卷积层 | $0.0 \sim 0.2$（卷积层本身参数少，不太需要） |
| 输出层 | **0.0**（绝对不能 dropout 输出） |

卷积层不太需要 Dropout 的原因：卷积层通过权重共享已经大幅减少了参数，本身就不容易过拟合。而且卷积层提取的是空间特征，随机丢弃会破坏空间结构。

## 4. train/eval 模式切换

这是最常见的初学者陷阱（务必掌握 [[NeuralNetwork/Training/Reproducibility|模式切换]] 的正确时机）：

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

## 5. Dropout 与 BatchNorm 的交互

在 `model.eval()` 模式下：

- Dropout 自动关闭
- BatchNorm 切换到全局统计量

所以只要正确调用 `model.eval()`，两者都能自动正常工作，不需要分别处理。

但有一个微妙之处：**Dropout 和 [[NeuralNetwork/Training/BatchNormalization|BatchNorm]] 一起使用时，训练和测试的方差会有偏差**。在训练时，BN 层的输入经过了 Dropout（神经元随机置零），方差与测试时（无障碍通过）不同。这就是所谓的"方差偏移"问题。在实践中，只要 batch size 不是太小，这个问题影响有限。

如果你的网络交替使用 BN 和 Dropout 且效果不好，可以尝试将 Dropout 放在 BN 之前：

```
# 标准顺序
Linear → BN → ReLU → Dropout

# 如果遇到方差偏移问题，可尝试
Linear → Dropout → BN → ReLU
```

但这在实践中很少需要。

## 6. 代码示例

```python
import torch
import torch.nn as nn


class Classifier(nn.Module):
    """带 Dropout 的全连接分类器"""

    def __init__(self, inputDim: int = 512, numClasses: int = 10, dropoutP: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(inputDim, 256)
        self.dropout = nn.Dropout(p=dropoutP)  # 50% 概率丢弃
        self.fc2 = nn.Linear(256, numClasses)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)   # 训练时随机丢弃，eval 时不动
        x = self.fc2(x)
        return x


# 训练循环中的正确用法
model = Classifier()

# 训练阶段
model.train()
for images, labels in trainLoader:
    optimizer.zero_grad()
    outputs = model(images)       # Dropout 生效
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 验证阶段
model.eval()                       # 关键：切换到评估模式
with torch.no_grad():
    for images, labels in valLoader:
        outputs = model(images)   # Dropout 已关闭
        valLoss += criterion(outputs, labels).item()
```

## 7. 现代趋势

近年来，Dropout 在 CNN 中的使用有所减少，被以下替代方案部分取代：

- **BatchNorm**：自带轻微正则化效果
- **数据增强**：更直接的正则化手段
- **Weight Decay**：L2 正则化

但在全连接层中，Dropout 仍然是非常有效的正则化手段。在 Transformer 架构中，Dropout 同样被广泛使用（注意力机制、FFN 层、嵌入层之后）。
