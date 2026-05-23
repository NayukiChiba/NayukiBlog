---
title: Dropout 使用指南
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 高级教程
  - 深度学习
description: Dropout 概率怎么设？哪些层需要？与 BN 共用时注意什么？
image: https://img.yumeko.site/file/articles/NNTrainingTips/Dropout.png
status: published
---

## 1. 概率设置建议

| 层类型 | 推荐 Dropout 概率 |
| --- | :---: |
| 输入层 | $0.0 \sim 0.2$（通常不 dropout 输入） |
| 隐藏层（全连接） | **0.5**（最常用的值） |
| 卷积层 | $0.0 \sim 0.2$（卷积层本身参数少，不太需要） |
| 输出层 | **0.0**（绝对不能 dropout 输出） |

卷积层不太需要 [[NeuralNetwork/Theory/Dropout|Dropout]] 的原因：卷积层通过权重共享已经大幅减少了参数，且空间特征被随机丢弃会破坏结构。

## 2. [[NeuralNetwork/Tips/Techniques/TrainEvalMode|train/eval 模式]]

```python
model.train()   # Dropout 生效：随机丢弃神经元
model.eval()    # Dropout 不生效：所有神经元参与

# 重要：验证/测试时必须 eval()
model.eval()
with torch.no_grad():
    valOutput = model(valImages)
```

忘记 `eval()` 的后果：Dropout 继续随机丢弃 → 预测变成随机的 → 准确率大幅下降。

## 3. 与 BatchNorm 共用的注意事项

在 `model.eval()` 模式下：
- Dropout 自动关闭
- BatchNorm 切换到全局统计量

只要正确调用 `model.eval()`，两者都能自动正常工作。

但有一个微妙之处：**Dropout 和 BN 一起训练时，训练和测试的方差会有偏差**。BN 层的输入在训练时经过 Dropout（神经元随机置零），方差与测试时（无障碍通过）不同。实践中只要 batch size 不是太小，这个影响有限。

如果你的网络交替使用 BN 和 Dropout 且效果不好，可以尝试将 Dropout 放在 BN 之前：

```
# 标准
Linear → BN → ReLU → Dropout

# 如果遇到方差偏移问题
Linear → Dropout → BN → ReLU
```

但这在实践中很少需要。

## 4. PyTorch 中使用

```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)   # train: 随机丢弃; eval: 不动
        x = self.fc2(x)
        return x
```

## 5. 现代趋势

Dropout 在 CNN 中的使用近年来有所减少，被替代方案取代：
- **BatchNorm**：自带轻微正则化
- **数据增强**：更直接的正则化
- **Weight Decay**：L2 正则化

但在全连接层和 Transformer 中，Dropout 仍然广泛使用。



