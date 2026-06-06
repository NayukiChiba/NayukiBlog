---
title: 过拟合怎么办？训练集 Acc 高验证集 Acc 低
date: 2026-05-23
category: 神经网络/问题排查
tags:
  - 高级教程
description: 训练集准确率 95%+，验证集却只有 70%？诊断并解决过拟合的完整方案。
image: https://img.yumeko.site/file/blog/cover/1780581828761.webp
status: published
---
## 现象描述

训练曲线呈现经典的"喇叭口"形态：

```
Epoch  5: Train Loss 0.21, Val Loss 0.45 | Train Acc 92%, Val Acc 80%
Epoch 10: Train Loss 0.08, Val Loss 0.62 | Train Acc 97%, Val Acc 78%
Epoch 15: Train Loss 0.03, Val Loss 0.85 | Train Acc 99%, Val Acc 76%
```

训练集指标持续改善，验证集指标先改善后恶化。

## 原因分析

过拟合意味着模型"背诵"了训练数据中的噪声和细节模式（与之相对的是[[NeuralNetwork/Troubleshooting/Underfitting|欠拟合]]），而非学习通用的分类规则。常见原因：

1. **模型容量过大**（相对数据量）
2. **训练数据太少**
3. **训练时间过长**（epoch 太多）
4. **缺少正则化手段**
5. **数据增强不足或缺失**

## 解决方案

### 方案 1：[[NeuralNetwork/Training/TrainingStability|Early Stopping]]（首选）

在验证集指标不再改善时自动停止训练：

```python
from cnnlib.training.earlyStopping import EarlyStopping

earlyStop = EarlyStopping(patience=10, minDelta=0.001, mode="max")
# mode="max" for accuracy, mode="min" for loss

for epoch in range(epochs):
    trainLoss, trainAcc = trainOneEpoch(...)
    valLoss, valAcc = validate(...)

    if earlyStop.step(valAcc):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 方案 2：增加 [[NeuralNetwork/Training/Dropout|Dropout]]

```python
# 在全连接层后添加 Dropout
model.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),       # 50% 丢弃率
    nn.Linear(256, num_classes)
)
```

### 方案 3：增强数据增强

数据增强是最有效的正则化手段之一——它直接扩大了训练集的多样性：

```python
# 从轻增强升级到强增强
trainTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),              # 增大旋转角度
    transforms.ColorJitter(0.3, 0.3, 0.3),     # 增加颜色抖动
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),  # 随机裁剪
    transforms.RandomErasing(p=0.3),            # 随机遮挡
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

### 方案 4：添加权重衰减（Weight Decay / L2 正则化）

```python
# AdamW 中设置 weight_decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4    # 从 0 增加到 1e-4
)
# 严重过拟合可尝试 weight_decay=1e-3 甚至 5e-3
```

### 方案 5：减小模型容量

如果数据量很小，用大模型必然过拟合：

```python
# 从大模型
model = ResNet50(num_classes=10)      # 25M 参数，数据只有 5000 张

# 换为小模型
model = LeNet(num_classes=10)         # 61K 参数，更适合小数据集
```

### 方案 6：使用 Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

防止模型对训练标签过于自信（自信到 100% 通常意味着过拟合）。

### 方案 7：添加 BatchNorm

BatchNorm 自带轻微正则化效果：

```python
# 在卷积后添加 BN
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),       # <- 添加 BN
    nn.ReLU(),
)
```

### 方案 8：收集更多数据

如果条件允许，这是最根本的解决方案。如果无法收集真实数据，可以考虑：
- 使用预训练模型（迁移学习）
- 对抗生成数据（GAN 数据增强）
- 半监督学习

## 诊断量化方法

```python
# 计算过拟合程度
overfitGap = trainAcc - valAcc
if overfitGap > 10.0:
    print(f"⚠️ 严重过拟合：Train-Val Gap = {overfitGap:.1f}%")
elif overfitGap > 5.0:
    print(f"⚡ 轻微过拟合：Train-Val Gap = {overfitGap:.1f}%")
else:
    print(f"✅ 泛化正常：Train-Val Gap = {overfitGap:.1f}%")
```

## 解决优先级

```
1. Early Stopping          -> 立竿见影，零成本
2. 加强数据增强            -> 效果最好
3. 添加/增大 weight_decay  -> 简单有效
4. 添加 Dropout            -> 针对 FC 层
5. Label Smoothing         -> 辅助手段
6. 减小模型容量            -> 可能影响最佳性能
7. 收集更多数据            -> 最根本但成本高
```
![Overfitting.png](https://img.yumeko.site/file/blog/articles/1780581594913.webp)