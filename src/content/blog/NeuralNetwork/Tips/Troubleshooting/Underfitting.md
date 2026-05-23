---
title: 欠拟合怎么办？训练集和验证集 Acc 都低
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 训练问题
description: 训练集准确率上不去，验证集也差？诊断欠拟合的原因并找到对应的改进方向。
image: TODO
status: draft
---

## 现象描述

训练集和验证集的表现都差，且差距不大：

```
Epoch  5: Train Loss 0.85, Val Loss 0.87 | Train Acc 62%, Val Acc 61%
Epoch 10: Train Loss 0.80, Val Loss 0.83 | Train Acc 65%, Val Acc 64%
Epoch 20: Train Loss 0.75, Val Loss 0.79 | Train Acc 68%, Val Acc 67%
```

两个 Loss 都高，两个 Acc 都低，且没有拉开差距——[[NeuralNetwork/Tips/Troubleshooting/SlowConvergence|收敛慢]]。如果 Loss 完全不动，则是[[NeuralNetwork/Tips/Troubleshooting/LossNotDown|Loss不降]]的问题。模型连训练集都学不好，说明**模型容量不足以表达训练数据中的模式**。

[//]: # (TODO: 欠拟合训练曲线图 — 两条线都高且平坦)

## 原因分析

1. **模型容量太小**（最常见）：网络太浅或太窄，无法表达数据的复杂度
2. **训练不足**：epoch 太少或学习率太低
3. **特征提取不足**：数据预处理丢失了关键信息
4. **优化器选择不当**：用 SGD 但没调好 momentum，或学习率完全不对

## 解决方案

### 方案 1：增大模型容量

```python
# 当前：太小的网络
class TinyModel(nn.Module):
    def __init__(self):
        # Conv(1→8) → Conv(8→16) → FC(16*7*7→10)
        # 参数量: ~10K — 过于简单

# 改进：增加通道数和深度
class ImprovedModel(nn.Module):
    def __init__(self):
        # Conv(1→32) → Conv(32→64) → Conv(64→128)
        # → FC(128*3*3→256) → FC(256→10)
        # 参数量: ~500K — 更合理的容量
```

**实用规则**：模型参数量应该至少是数据集样本数的 1/10 到数倍之间。

### 方案 2：减少正则化强度

过强的正则化会导致欠拟合：

```python
# 当前：正则化太重
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
model.dropout = nn.Dropout(p=0.8)  # 丢弃太多

# 调整：降低正则化
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
model.dropout = nn.Dropout(p=0.2)  # 适度丢弃
```

### 方案 3：增加训练时间/调整学习率

```python
# 当前：epoch 太少
epochs = 10

# 改进：增加 epoch，配合学习率调度
epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### 方案 4：尝试更大的学习率

```python
# 当前学习率太小，更新太慢
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 尝试：用默认学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 方案 5：更换更复杂的架构

如果当前架构本身表达能力有限：

```python
# 从简单架构
model = LeNet(num_classes=10)        # 61K 参数

# 换为更强的架构
model = ResNet18(num_classes=10)     # 11M 参数
# 或
model = VGG16(num_classes=10)        # 138M 参数
```

### 方案 6：检查特征是否足够

确认输入数据保留了足够的信息：

```python
# 检查 Resize 是否把图像缩得太小
transform = transforms.Resize((8, 8))  # 太小！28×28 的 MNIST 缩到 8×8 丢失太多信息
transform = transforms.Resize((32, 32))  # 更好的选择
```

## 欠拟合 vs [[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]]：快速判断

| 指标 | 欠拟合 | 过拟合 |
| --- | :---: | :---: |
| Train Acc | 低 | 很高 |
| Val Acc | 低（≈ Train） | 明显低于 Train |
| Train-Val Gap | 小 | **大** |
| Loss | 高 | Train 很低，Val 升高 |
| 策略 | **增大容量** | **增加正则化** |

简单记忆：**欠拟合 → 加大模型，过拟合 → 加正则化。**

## 解决优先级

```
1. 增大模型容量（通道数/深度） → 最直接
2. 增加训练 epoch + 学习率调度 → 零成本
3. 调大学习率                  → 立竿见影
4. 减少正则化（weight_decay, dropout）
5. 换更复杂的架构
6. 检查数据预处理是否丢失信息
```
