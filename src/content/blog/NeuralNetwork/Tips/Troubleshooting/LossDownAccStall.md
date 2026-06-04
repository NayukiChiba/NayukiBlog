---
title: Loss 下降但 Acc 不上升怎么办？
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 高级教程
description: Loss 一直在降但准确率纹丝不动？诊断并解决这个常见的训练异常。
image: https://img.yumeko.site/file/blog/LossDownAccStall.png
status: published
---

## 现象描述

训练过程中观察到：

```
Epoch 10: Train Loss 0.45, Train Acc 75.2%, Val Acc 75.1%
Epoch 11: Train Loss 0.38, Train Acc 75.3%, Val Acc 75.0%
Epoch 12: Train Loss 0.32, Train Acc 75.1%, Val Acc 75.2%
```

Loss 在平滑下降（与 [[NeuralNetwork/Tips/Troubleshooting/LossNotDown|Loss不降]] 不同），但 Acc 几乎不变。

## 原因分析

Loss 下降而 Acc 不变，核心原因是模型在对"已经正确分类"的样本变得更加自信，但对错误样本的预测却没有改善。

具体来说：

1. **模型正在"打磨"正确样本**：CrossEntropyLoss 奖励模型对正确类别输出更高的概率。如果模型对已正确的样本从"60% 置信"提升到"90% 置信"，Loss 会下降，但 Acc 不变（它们本来就是对的）。

2. **边界样本仍被误判**：那些模棱两可的样本（比如潦草的数字 4 和 9），模型的预测概率从"48% 正确 vs 52% 错误"变为"45% 正确 vs 55% 错误"——Loss 可能微降，但 Acc 不升。

3. **[[NeuralNetwork/Techniques/LearningRateGuide|学习率]]过低**：模型陷入局部最优，参数更新幅度太小，无法跳出当前状态。

4. **模型容量饱和**：当前模型结构已无法从数据中学到更多有用的特征。

## 解决方案

### 方案 1：检查学习率（最常见原因）

学习率太小会导致模型只做微调，无法突破。尝试增大学习率：

```python
# 当前学习率可能太低
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 太小

# 尝试增大
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # 默认值
# 或使用 LR Finder 查找最优学习率
```

### 方案 2：添加 Label Smoothing

防止模型对正确样本过于自信（过于自信并不提升 Acc，但会降 Loss）：

```python
# 标准 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 添加 Label Smoothing（鼓励模型"谦虚"一点）
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Label Smoothing 让模型不要在已正确的样本上"精益求精"，而是把注意力放在边界样本上。

### 方案 3：使用 Focal Loss

Focal Loss 自动降低已分类良好样本的权重，让模型更关注难分类的样本：

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 正确类别的预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2.0)
```

$\gamma=2$ 时，如果一个样本的正确预测概率为 $0.9$，其 loss 会被乘以 $(1-0.9)^2=0.01$——几乎被忽略。但如果概率只有 $0.5$，loss 只被减为 $0.25$ 倍——仍然保留。

### 方案 4：检查数据标签是否正确

如果训练数据中有标注错误，模型会在正确样本上过度优化（Loss 降）但永远学不会错误标注的样本（Acc 不升）。

```python
# 随机抽查训练样本的标签
import random
indices = random.sample(range(len(trainSet)), 50)
for idx in indices:
    image, label = trainSet[idx]
    # 显示图像和标签，人工验证
```

### 方案 5：增加数据增强强度

更强的数据增强强迫模型面对更多变体，减少[[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]]：

```python
trainTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),           # 从 10 增加到 15
    transforms.ColorJitter(0.2, 0.2, 0.2),  # 添加颜色抖动
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ToTensor(),
])
```

### 方案 6：检查模型是否容量不足

如果模型太小，可能已经学到了它的上限：

```python
# 查看参数量
totalParams = sum(p.numel() for p in model.parameters())
print(f"总参数: {totalParams:,}")

# 如果参数量太少（比如 <100K for CIFAR-10），考虑增加
# Conv2d(32, 32, ...) → Conv2d(32, 64, ...)
```

### 方案 7：使用 Class-Balanced Sampling

如果某几类一直被误判，检查类别分布是否不均衡：

```python
from collections import Counter
labelCounts = Counter()
for _, label in trainLoader.dataset:
    labelCounts[label] += 1
print(f"类别分布: {labelCounts}")
# 如果某类样本极少，使用加权采样或 weighted loss
```

## 诊断流程

![Decision.png](https://img.yumeko.site/file/articles/LossDownAccStall/Decision.png)

