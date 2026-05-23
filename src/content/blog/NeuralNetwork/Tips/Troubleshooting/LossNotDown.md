---
title: Loss 不下降怎么办？
date: 2026-05-23
category: NeuralNetwork/Tips/Troubleshooting
tags:
  - 高级教程
description: 训练了几个 epoch，Loss 纹丝不动？系统排查 Loss 不下降的 7 大常见原因及解决方案。
image: https://img.yumeko.site/file/blog/LossNotDown.png
status: published
---

## 现象描述

训练开始后，Loss 基本不变或变化极其微小：

```
Epoch 1: Train Loss 2.31, Train Acc 10.0%
Epoch 2: Train Loss 2.30, Train Acc 10.0%
Epoch 3: Train Loss 2.30, Train Acc 10.0%
```

Acc 也不变（如果是 10 分类问题，10% 约等于随机猜测）。

## 原因一：学习率太大

**现象**：Loss 在某个值附近震荡，偶尔飙高但不下降。

学习率太大导致参数更新幅度过大，每次都"跳过"了最优解，在最优解附近来回震荡，这是[[NeuralNetwork/Tips/Troubleshooting/TrainingUnstable|训练不稳]]的典型表现。

**解决**：逐步降低学习率。

```python
# 当前
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 尝试：降低 10 倍
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 再不行再降
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**诊断方法**：观察 Loss 是否在震荡——如果每个 batch 的 Loss 变化剧烈（比如从 1.0 跳到 5.0 再跳回 0.5），学习率很可能太大。

## 原因二：学习率太小

**现象**：Loss 极其缓慢地下降，每个 epoch 只降 0.001。

这和"太大"的症状相反——模型更新太慢，几乎停滞。

**解决**：增大学习率。

```python
# 尝试增大到 10 倍
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

## 原因三：数据未正确归一化

**现象**：输入的像素值在 $[0, 255]$ 而非 $[-1, 1]$ 或 $[0, 1]$ 范围。

未归一化的数据会导致[[NeuralNetwork/Tips/Troubleshooting/GradientExplodingVanishing|梯度问题]]（梯度爆炸），严重时产生[[NeuralNetwork/Tips/Troubleshooting/NanLoss|NaN]]，网络也无法学习。

```python
# 检查 DataLoader 输出的范围
for images, _ in trainLoader:
    print(f"像素范围: [{images.min():.2f}, {images.max():.2f}]")
    print(f"均值: {images.mean():.2f}, 标准差: {images.std():.2f}")
    break

# 正确输出应类似：
# 像素范围: [-2.0, 2.5]
# 均值: 0.0, 标准差: 1.0
```

**解决**：确保 Normalize 配置正确。

```python
# 错误的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 像素值在 [0, 1]
    # 忘记 Normalize！
])

# 正确的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST 的统计值
])
```

## 原因四：权重初始化不当

**现象**：训练开始时 Loss 异常大（如 CrossEntropyLoss > 10），然后不再下降。

这通常意味着初始化不当或忘记初始化，导致网络输出分布极度不平衡。

**诊断**：

```python
# 检查模型各层的初始输出范围
model.eval()
with torch.no_grad():
    for images, _ in trainLoader:
        output = model(images)
        print(f"输出范围: [{output.min():.2f}, {output.max():.2f}]")
        print(f"输出均值: {output.mean():.2f}, 标准差: {output.std():.2f}")
        break
```

对于 N 分类 CrossEntropyLoss，logits 的初始范围应在 $[-1, 1]$ 左右，太大或太小都说明初始化有问题。

**解决**：

```python
def initWeights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
```

## 原因五：数据标签与 Loss Function 不匹配

**现象**：Loss 值异常（非常高或为负数），Acc 接近随机。

常见情况：标签是 one-hot 编码但用了 CrossEntropyLoss（不需要 one-hot），或者标签范围超出了输出类别数。

```python
# 检查标签格式和范围
for _, labels in trainLoader:
    print(f"标签形状: {labels.shape}, 类型: {labels.dtype}")
    print(f"标签范围: [{labels.min()}, {labels.max()}]")
    # CrossEntropyLoss 需要 (batch,) 的 int 标签，范围 [0, num_classes-1]
    break
```

## 原因六：梯度消失

**现象**：Loss 一开始降一点然后就停滞。

深层网络或使用了 Tanh/Sigmoid 激活函数时容易出现。

**诊断**：检查各层的梯度范数。

```python
# 在 backward 之后检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        gradNorm = param.grad.norm().item()
        if gradNorm < 1e-7:
            print(f"⚠️ {name} 梯度接近 0: {gradNorm}")
```

**解决**：
- 使用 ReLU 替代 Tanh/Sigmoid
- 添加 BatchNorm
- 使用残差连接（ResNet）
- 降低网络深度

## 原因七：模型在训"错误的损失函数"

**现象**：Loss 下降但实际效果不对。

检查是否不小心在验证时用了训练模式的模型（Dropout 开启），或者 Loss 函数本身不适合任务。

## 快速排查清单

```
1. 数据归一化了吗？         → transforms.Normalize()
2. 学习率合适吗？          → 尝试 1e-4, 1e-3, 1e-2, 0.1
3. 权重初始化了吗？         → Kaiming/ReLU, Xavier/Tanh
4. 标签格式对吗？          → CrossEntropyLoss 用 int 标签
5. 输出范围正常吗？        → logits 应在 [-2, 2]
6. 梯度值正常吗？          → 梯度范数应在 [1e-6, 1e2]
```
![LossNotDown.png](https://img.yumeko.site/file/articles/LossNotDown/LossNotDown.png)

