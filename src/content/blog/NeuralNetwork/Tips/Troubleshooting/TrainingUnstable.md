---
title: 训练不稳定（Loss 剧烈震荡）怎么办？
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 高级教程
description: Loss 曲线像过山车一样上下翻飞？系统排查训练不稳定的原因与稳定方案。
image: https://img.yumeko.site/file/blog/TrainingUnstable.png
status: published
---

## 现象描述

Loss 在训练过程中剧烈震荡，每个 batch 或每个 epoch 的 Loss 变化极大：

```
Epoch 1: Train Loss 0.85, Val Loss 0.92
Epoch 2: Train Loss 1.23, Val Loss 0.78   ← 反向？！ 
Epoch 3: Train Loss 0.56, Val Loss 1.15
Epoch 4: Train Loss 0.91, Val Loss 0.65
```

## 原因分析

1. **[[NeuralNetwork/Techniques/LearningRateGuide|学习率]]太大**（最常见）——参数更新步幅过大，在最优解附近弹来弹去，严重时可能产生[[NeuralNetwork/Tips/Troubleshooting/NanLoss|NaN]]
2. **Batch Size 太小**——每个 batch 的[[NeuralNetwork/Tips/Troubleshooting/GradientExplodingVanishing|梯度方向]]不稳定，不同 batch 之间差异大
3. **数据未充分打乱**——batch 的顺序影响了训练
4. **BatchNorm 在小 batch 下表现差**——统计量噪声大
5. **数据中存在异常值**——某些样本的 Loss 极大，拉偏了整体

## 解决方案

### 方案 1：降低学习率

```python
# 当前学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 尝试降低
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 如果仍然震荡，继续降
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
```

### 方案 2：增大 Batch Size

```python
# 当前：batch size 太小
trainLoader = DataLoader(dataset, batch_size=4, shuffle=True)

# 增大：更稳定的梯度估计
trainLoader = DataLoader(dataset, batch_size=64, shuffle=True)
```

注意：增大 batch size 可能需要同时增大学习率（线性缩放法则）。

### 方案 3：使用 SGD + Momentum 替代 Adam

Adam 的自适应学习率在某些情况下反而导致震荡。SGD + Momentum 的路径更平滑：

```python
# 从 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 换为 SGD + Momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

### 方案 4：添加学习率 Warmup

训练初期模型参数是随机的，梯度可能很大。Warmup 从前几轮用较小的学习率开始：

```python
def warmupLR(optimizer, currentStep, warmupSteps, targetLR):
    """线性 warmup：从 0 线性增长到目标学习率"""
    if currentStep < warmupSteps:
        lr = targetLR * (currentStep + 1) / warmupSteps
        for paramGroup in optimizer.param_groups:
            paramGroup['lr'] = lr

# 训练循环中使用
for batchIdx, (images, labels) in enumerate(trainLoader):
    warmupLR(optimizer, batchIdx, warmupSteps=500, targetLR=0.001)
    # ... 正常训练步骤
```

### 方案 5：使用梯度裁剪

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 方案 6：使用平滑的 Loss 记录

有时候"震荡"只是记录方式的问题——看 batch Loss 必然有噪声。记录移动平均版本：

```python
class SmoothedValue:
    def __init__(self, window=20):
        self.window = window
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    @property
    def avg(self):
        return sum(self.values) / len(self.values) if self.values else 0

lossSmoother = SmoothedValue(window=100)
for batch in trainLoader:
    loss = criterion(output, labels)
    lossSmoother.update(loss.item())
    # 用 lossSmoother.avg 而不是 loss.item() 来判断趋势
```

### 方案 7：检查数据

```python
# 检查是否有异常样本
for images, labels in trainLoader:
    if images.max() > 10 or images.min() < -10:
        print("⚠️ 数据中存在异常值")
    # 检查每个 batch 的 Loss
    outputs = model(images)
    losses = criterion(outputs, labels)
    # 如果某些样本的 loss 异常大，可能是标注错误
```

## 诊断方法：判断是否稳定收敛

值得区分"正常噪声"和"不稳定训练"：

```python
# 计算最近 N 个 epoch 的 Loss 标准差
recentLosses = history['valLoss'][-5:]
mean = sum(recentLosses) / len(recentLosses)
std = (sum((l - mean)**2 for l in recentLosses) / len(recentLosses)) ** 0.5

if std / abs(mean) > 0.3:  # 标准差超过均值的 30%
    print("⚠️ 训练不稳定")
```

## 解决优先级

```
1. 降低学习率                  → 最简单，先试
2. 增大 batch size             → 梯度更稳定
3. 添加梯度裁剪                → 保底手段
4. 添加 Warmup                 → 训练初期稳定
5. 换 SGD+Momentum             → Adam 不总是最优
6. 检查数据异常值              → 排除数据问题
```
![TrainingUnstable.png](https://img.yumeko.site/file/articles/TrainingUnstable/TrainingUnstable.png)