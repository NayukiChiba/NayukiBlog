---
title: 深度学习项目训练流程总览
date: 2026-05-07
category: NeuralNetwork/Training
tags:
  - 基础
  - 深度学习
  - 总结
description: 从数据准备到模型部署，一篇理清深度学习项目的完整六阶段工程流程。
image: https://img.yumeko.site/file/blog/cover/1780581818737.webp
status: draft
---

从零开始做一个深度学习项目，应该遵循怎样的步骤？以下是基于 CNN 项目总结的标准流程。

## 阶段一：数据准备

### 1.1 获取数据

```python
from torchvision import datasets

trainSet = datasets.MNIST(root='./datasets', train=True, download=True)
testSet = datasets.MNIST(root='./datasets', train=False, download=True)
```

### 1.2 探索数据（必须先做！）

在写任何模型代码之前，先了解数据：
- **数据维度**：图像尺寸多大？通道数？
- **类别分布**：各类别样本数量均衡吗？
- **视觉检查**：随机看几张图，了解数据的"样子"
- **统计信息**：计算均值和标准差，用于后续标准化

### 1.3 定义预处理流水线

```python
# 训练集：预处理 + 数据增强
trainTransform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 验证/测试集：只预处理，不增强
testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 1.4 创建 DataLoader

```python
trainLoader = DataLoader(trainSet, batch_size=64,
                         shuffle=True, num_workers=4, pin_memory=True)
valLoader = DataLoader(valSet, batch_size=64,
                       shuffle=False, num_workers=4, pin_memory=True)
```

- `shuffle=True`：训练集打乱，防止模型学到样本顺序
- `num_workers=4`：子进程加载数据，加速训练
- `pin_memory=True`：GPU 训练时加速 CPU→GPU 传输

## 阶段二：模型定义

### 2.1 选择架构

| 任务类型 | 推荐架构 |
| --- | --- |
| 简单图像分类（MNIST） | 2-3 层 CNN + 1-2 层 FC |
| 中等图像分类（CIFAR-10） | ResNet-18/34 |
| 复杂图像分类（ImageNet） | ResNet-50/101, EfficientNet |
| 目标检测 | YOLO, Faster R-CNN |
| 图像分割 | U-Net, DeepLab |

### 2.2 验证输出形状

```python
dummy = torch.randn(1, 1, 28, 28)
output = model(dummy)
print(f"输入形状: {dummy.shape}")    # (1, 1, 28, 28)
print(f"输出形状: {output.shape}")    # (1, 10)
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 阶段三：训练配置

### 3.1 选择损失函数

| 任务类型 | 损失函数 |
| --- | --- |
| 多分类 | `CrossEntropyLoss` |
| 二分类 | `BCEWithLogitsLoss` |
| 回归 | `MSELoss` 或 `L1Loss` |

### 3.2 选择优化器

```python
# 快速实验
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 精细调优
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### 3.3 选择学习率调度器

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)
```

### 3.4 固定随机种子

```python
def setSeed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## 阶段四：训练循环

### 单个 batch 的 8 步

1. `model.train()`：启用 Dropout、设置 BN 为训练模式
2. 遍历 DataLoader：取 `batch_size` 个样本
3. `.to(device)`：数据从 CPU 移到 GPU
4. `optimizer.zero_grad()`：清空上轮累积的梯度
5. 前向传播：数据经过各层，输出 logits
6. 计算损失：logits 和真实标签对比
7. `loss.backward()`：自动计算所有参数的梯度
8. `optimizer.step()`：根据梯度和学习率更新参数

### 完整训练循环

```python
for epoch in range(1, epochs + 1):
    trainLoss, trainAcc = trainOneEpoch(...)
    valLoss, valAcc = validate(...)
    scheduler.step(valLoss)               # 学习率调度

    if valLoss < bestValLoss:             # 保存最佳模型
        bestValLoss = valLoss
        torch.save(model.state_dict(), 'best_model.pth')

    if earlyStop.step(valLoss):           # Early Stopping
        break
```

## 阶段五：评估与分析

- **测试集评估**：最终准确率和损失
- **混淆矩阵**：发现类别混淆模式
- **错误样本分析**：找出模型的薄弱点
- **训练曲线**：检查过拟合、判断是否收敛

## 阶段六：推理部署

```python
def predict(model, imagePath, transform, classNames, device):
    model.eval()
    image = Image.open(imagePath).convert('L')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    topProbs, topIdx = probs.topk(k=5, dim=1)
    return [(classNames[idx.item()], prob.item())
            for prob, idx in zip(topProbs[0], topIdx[0])]
```

## 完整检查清单

**数据准备**：
- [ ] 下载数据集，探索数据分布
- [ ] 定义预处理（ToTensor + Normalize）
- [ ] 定义数据增强（仅训练集）
- [ ] 创建 DataLoader

**模型定义**：
- [ ] 选择/设计架构
- [ ] 根据激活函数选择初始化
- [ ] 用虚拟输入验证输出形状

**训练配置**：
- [ ] 选择损失函数和[[NeuralNetwork/Training/Optimizers|优化器]]
- [ ] 配置学习率调度器
- [ ] 固定[[NeuralNetwork/Training/Reproducibility|可复现]]性（全局随机种子）

**训练循环**：
- [ ] 实现 `trainEpoch` + `validateEpoch`
- [ ] 保存[[NeuralNetwork/Training/TrainingStability|检查点]]（最佳模型和最近模型）
- [ ] 实现 Early Stopping

**评估部署**：
- [ ] 测试集评估 + 混淆矩阵
- [ ] 错误样本分析
- [ ] 单张/批量推理函数


