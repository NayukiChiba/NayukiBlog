---
title: ResNet 图像分类训练框架
date: 2026-06-26
category: 项目
tags:
  - PyTorch
  - ResNet
  - DenseNet
  - 图像分类
description: 基于 PyTorch 的图像分类训练框架，支持 ResNet-18、DenseNet-BC-100 与 CIFAR-10/CIFAR-100。
image: https://img.yumeko.site/file/blog/cover/1782481171186_ResNet__2_.webp
status: published
---

# ResNet 图像分类训练框架

> **前置阅读**：建议先阅读 [[NeuralNetwork/CNN/ResNet|ResNet]] 与 [[NeuralNetwork/CNN/CNNCoreLayers|CNN 核心层]]。

::github[repo=NayukiChiba/ResNet]

## 1. 项目定位

`ResNet` 是一个面向 CIFAR-10 和 CIFAR-100 的图像分类训练框架，支持 `ResNet-18` 与 `DenseNet-BC-100`。

项目重点是把现代 CNN 训练框架拆成明确模块：模型、数据、训练、评估、推理、配置和 CLI。

---

## 2. 项目结构

```text
ResNet/
├── main.py
├── models/
│   ├── resnet.py
│   └── densenet.py
├── data/
│   ├── transforms.py
│   └── dataLoader.py
├── training/
│   ├── loss.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── early_stopping.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── trainer.py
├── evaluation/
│   ├── metrics.py
│   ├── visualization.py
│   └── evaluator.py
├── inference/
│   └── inferencer.py
├── config/
│   ├── defaults.py
│   ├── paths.py
│   ├── datasets.py
│   └── training.py
└── cli/
    ├── train.py
    ├── evaluate.py
    └── inference.py
```

| 模块 | 职责 |
|:--|:--|
| `models/` | ResNet-18 与 DenseNet-BC-100 |
| `data/` | CIFAR 数据加载和 transform pipeline |
| `training/` | 损失、优化器、调度器、早停、checkpoint、Trainer |
| `evaluation/` | Top-K、混淆矩阵、类别精确率召回率 |
| `inference/` | 单张、批量、Top-K 推理和特征提取 |
| `cli/` | train、evaluate、inference 子命令 |

---

## 3. ResNet 的核心思想

普通深层网络直接学习映射 $H(x)$，ResNet 改为学习残差：

$$
\boxed{
H(x) = F(x) + x
}
$$

其中 $F(x)$ 是残差分支，$x$ 是 shortcut。这样做的直觉是：如果某些层不需要改变表示，
只要让 $F(x) \approx 0$，网络就能近似恒等映射。

BasicBlock 的形式为：

$$
y = \sigma(F(x; W) + x)
$$

当输入输出维度不一致时，需要使用投影 shortcut：

$$
y = \sigma(F(x; W) + W_sx)
$$

---

## 4. DenseNet 对照

项目同时实现了 `DenseNet-BC-100`。DenseNet 不是把输入相加，而是把前面所有层的输出拼接起来：

$$
\boxed{
x_l = H_l([x_0, x_1, \dots, x_{l-1}])
}
$$

ResNet 和 DenseNet 的区别：

| 维度 | ResNet | DenseNet |
|:--|:--|:--|
| 信息流 | 残差相加 | 特征拼接 |
| 目标 | 缓解深层退化 | 强化特征复用 |
| 通道变化 | 通常分 stage 扩张 | 随层数逐步增长 |
| 典型结构 | BasicBlock/Bottleneck | DenseBlock/Transition |

---

## 5. 训练组件

项目支持多种训练组件：

| 组件 | 支持项 |
|:--|:--|
| Loss | CrossEntropyLoss、FocalLoss |
| Optimizer | SGD、Adam、AdamW、RMSprop |
| Scheduler | CosineAnnealingLR、StepLR、ReduceLROnPlateau |
| Evaluation | Top-1、Top-5、混淆矩阵、各类别指标 |

Top-K 准确率定义为：

$$
\operatorname{Acc@K}
= \frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\{y_i \in \operatorname{TopK}(\hat{p}_i)\}
$$

当类别数较多时，例如 CIFAR-100，Top-5 比 Top-1 更能反映模型是否把真实类别排在高概率候选中。

---

## 6. CLI 使用

安装依赖：

```bash
uv sync
```

训练：

```bash
python main.py train --model resnet18 --dataset cifar100 --epochs 20
```

评估：

```bash
python main.py evaluate --model resnet18 --checkpoint outputs/checkpoints/resnet18/best.pth
```

推理：

```bash
python main.py inference --model resnet18 --checkpoint outputs/checkpoints/resnet18/best.pth --image cat.jpg
```

---

## 7. Python API

项目也支持直接用 Python API 组合训练流程：

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import buildLoaders
from models import resnet18
from training import CrossEntropyLoss, AdamW, Trainer

train_loader, val_loader = buildLoaders("cifar100", batch_size=128)
model = resnet18(num_classes=100)
criterion = CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

trainer = Trainer(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    model_name="resnet18", epochs=200, device="cuda",
)
trainer.fit()
```

---

## 8. 总结

这个项目适合用来理解“现代 CNN 训练框架”：

| 重点 | 说明 |
|:--|:--|
| ResNet | 学习残差映射，缓解深层网络退化 |
| DenseNet | 特征拼接和复用，形成另一种深层连接方式 |
| CIFAR-10/100 | 小规模图像分类实验基准 |
| Trainer | 将训练循环、评估、日志、checkpoint 封装起来 |
| Inferencer | 让训练结果可以直接用于单图或批量推理 |

它和 `ALL-CNN` 的关系是：`ALL-CNN` 更偏经典 CNN 架构总览，`ResNet` 更聚焦现代残差网络和 DenseNet 对照。

---

> **相关文章**：
> - [[NeuralNetwork/CNN/ResNet|ResNet]]
> - [[NeuralNetwork/CNN/CNNCoreLayers|CNN 核心层]]
> - [[Projects/ALLCNN|ALL-CNN 经典卷积网络项目]]
