---
title: Early Stopping 详解
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 基础
  - 深度学习
description: 如何准确检测过拟合并自动停止训练？详解 Early Stopping 的实现设计与参数调优。
image: https://img.yumeko.site/file/articles/NNTrainingTips/EarlyStop.png
status: published
---

## 1. 过拟合的信号

训练过程中，如果你观察到：

- 训练损失持续下降
- 验证损失先下降，然后开始上升

这就是经典的[[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]]信号——模型开始"背诵"训练数据而非学习通用模式。

![LossCurve.png](https://img.yumeko.site/file/articles/EarlyStoppingGuide/LossCurve.png)

## 2. 基本 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, minDelta=0.001, mode='min'):
        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode        # 'min' for loss, 'max' for accuracy
        self.bestMetric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def step(self, metric):
        if self.mode == 'min':
            improved = metric < self.bestMetric - self.minDelta
        else:
            improved = metric > self.bestMetric + self.minDelta

        if improved:
            self.bestMetric = metric
            self.counter = 0
            return False  # 不停止
        else:
            self.counter += 1
            return self.counter >= self.patience  # patience 耗尽 → 停止
```

## 3. minDelta 的设计

`minDelta=0.001` 意味着验证损失必须下降至少 0.001 才算"有效改善"，避免随机波动被误判为改善。

```python
if valLoss < bestValLoss - self.minDelta:   # 改善了至少 minDelta
    bestValLoss = valLoss
```

如果 minDelta 太大，很多真正的改善会被忽略。如果太小，噪声波动会被当成改善。典型值：$10^{-3}$ 到 $10^{-4}$。

## 4. 多条件 Early Stopping

CNN 项目的 Early Stopping 使用了三种检查：

```python
def checkEarlyStop(self, trainLoss, valLoss, trainAcc, valAcc):
    # 条件 1：验证损失不再改善
    if self.patienceCounter >= self.patience:  # patience=3
        return True, "val_loss 不再改善"

    # 条件 2：过拟合检测（训练在改善但验证在恶化）
    if self.overfitCounter >= self.overfitPatience:  # overfitPatience=3
        return True, "检测到过拟合"

    # 条件 3：训练-验证准确率差距过大
    if trainAcc - valAcc > 10.0:
        return True, "Train-Val Acc 差距过大"

    return False, "继续训练"
```

**条件 1**：验证指标不再改善 → 模型已经学到了极限。

**条件 2**：训练损失还在降，但验证损失不再降 → 经典的过拟合模式。

**条件 3**：训练比验证高 10% 以上 → 严重过拟合，直接停。

## 5. 最佳模型保存

Early Stopping 常与[[NeuralNetwork/Tips/Techniques/CheckpointGuide|检查点]]保存配合：

```python
for epoch in range(epochs):
    trainLoss, trainAcc = trainOneEpoch(...)
    valLoss, valAcc = validate(...)

    # 保存最佳模型
    if valLoss < bestValLoss - minDelta:
        bestValLoss = valLoss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ 保存最佳模型 (val_loss={valLoss:.4f})")

    # Early Stopping 检查
    if earlyStop.step(valLoss):
        print(f"🛑 Early stopping at epoch {epoch}")
        break
```

**重要**：Early Stopping 停止训练后，应该恢复到最佳 epoch 的权重再做测试，否则可能用的是最后（已过拟合）的权重。

## 6. patience 的选择

| patience | 效果 |
| :---: | --- |
| 太小（如 2） | 可能过早停止——val loss 的小波动被当成停滞 |
| 适中（5-15） | 给模型足够时间跨过"高原期" |
| 太大（如 50） | 基本等于没开 Early Stopping |

经验值：小任务用 5-10，大任务用 10-15。



