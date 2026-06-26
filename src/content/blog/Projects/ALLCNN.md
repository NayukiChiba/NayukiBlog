---
title: ALL-CNN 经典卷积网络项目
date: 2026-06-26
category: 项目
tags:
  - PyTorch
  - CNN
  - 图像分类
  - 深度学习
description: 一个支持多种经典 CNN 架构与多数据集组合训练、评估、推理和基准测试的图像分类项目。
image: https://img.yumeko.site/file/blog/ALLCNN.png
status: draft
---

# ALL-CNN 经典卷积网络项目

> **前置阅读**：建议先阅读 [[NeuralNetwork/CNN/CNN-Overview|CNN 总览]]、[[NeuralNetwork/CNN/LeNet5|LeNet-5]]、[[NeuralNetwork/CNN/AlexNet|AlexNet]]、[[NeuralNetwork/CNN/VGG|VGG]]、[[NeuralNetwork/CNN/GoogLeNet|GoogLeNet]] 与 [[NeuralNetwork/CNN/NiN|NiN]]。

::github[repo=NayukiChiba/ALL-CNN]

## 1. 项目定位

`ALL-CNN` 是一个经典 CNN 架构学习项目。它把模型、数据集、训练、评估、推理和基准测试做成统一框架，
支持“多模型 × 多数据集”自由组合。

这个项目的核心价值是：用同一套训练管线对比不同 CNN 架构，而不是每个模型单独写一套脚本。

---

## 2. 项目结构

```text
ALL-CNN/
├── main.py
├── config/
│   ├── defaults.py
│   ├── paths.py
│   ├── data.py
│   └── training.py
├── cnnlib/
│   ├── cli/
│   ├── data/
│   ├── models/
│   ├── registry/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── experiments/
├── scripts/
├── tests/
├── docs/
└── outputs/
```

关键分层：

| 模块 | 说明 |
|:--|:--|
| `cnnlib/registry/` | 模型和数据集元数据注册表 |
| `cnnlib/models/` | LeNet、AlexNet、VGG、NiN、GoogLeNet 等 |
| `cnnlib/data/` | DataLoader 工厂和自动 transform |
| `cnnlib/training/` | Trainer、loss、optimizer、scheduler、checkpoint |
| `cnnlib/evaluation/` | Accuracy、混淆矩阵、各类别指标和可视化 |
| `cnnlib/inference/` | 单图和批量推理 |
| `cnnlib/experiments/` | 基准测试框架 |

---

## 3. 功能范围

### 3.1 已支持架构

| 架构 | 年份 | 输入尺寸 | 核心特点 |
|:--|:--|:--|:--|
| LeNet-5 | 1998 | $32 \times 32$ | 早期 CNN，卷积 + 池化 + 全连接 |
| AlexNet | 2012 | $224 \times 224$ | ReLU、Dropout、大规模 CNN |
| VGG-11 | 2015 | $224 \times 224$ | 多层 $3 \times 3$ 卷积堆叠 |
| VGG-13 | 2015 | $224 \times 224$ | 更深的 VGG 变体 |
| VGG-16 | 2015 | $224 \times 224$ | 经典 VGG 主力版本 |
| VGG-19 | 2015 | $224 \times 224$ | 更深的 VGG |
| NiN | 2014 | $32 \times 32$ | `mlpconv` 与全局平均池化 |
| GoogLeNet | 2015 | $224 \times 224$ | Inception v1，多尺度卷积 |

### 3.2 已支持数据集

| 数据集 | 类别数 | 图像特点 |
|:--|--:|:--|
| MNIST | 10 | 灰度手写数字 |
| Fashion-MNIST | 10 | 灰度服饰图像 |
| EMNIST | 47 | 字母和数字 |
| KMNIST | 10 | 日文假名 |
| CIFAR-10 | 10 | 彩色自然图像 |
| CIFAR-100 | 100 | 更细粒度自然图像 |
| SVHN | 10 | 街景门牌号 |
| STL-10 | 10 | 较高分辨率自然图像 |
| Caltech-101 | 101 | 物体类别 |
| Caltech-256 | 257 | 更多物体类别 |
| GTSRB | 43 | 交通标志 |
| Flowers-102 | 102 | 花卉分类 |

框架会根据模型输入要求和数据集特征自动构造 transform pipeline，包括通道转换、尺寸缩放、归一化和数据增强。

---

## 4. 注册表机制

项目使用注册表解耦 CLI 和具体模型实现。模型只需要注册元数据，训练入口根据名称查找即可。

抽象上看，模型注册表负责把 `modelName` 解析成对应的模型工厂，数据集注册表负责把
`datasetName` 解析成对应的数据集规格。CLI 只关心名称，具体创建逻辑交给注册表处理。

这种方式的好处是新增模型时不需要改 CLI 主分发逻辑，只需要补模型文件和注册信息。

---

## 5. 训练流程

图像分类训练目标是最小化交叉熵：

$$
\mathcal{L}
= -\sum_{k=1}^{C} y_k \log \hat{p}_k
$$

其中 $C$ 是类别数，$y_k$ 是真实标签的 one-hot 表示，$\hat{p}_k$ 是模型预测概率。

训练流程：

1. 根据 `--model` 和 `--dataset` 查询注册表。
2. 自动构建数据变换和 DataLoader。
3. 创建模型并移动到设备。
4. 创建优化器、学习率调度器和早停器。
5. 训练并在验证集上选择最佳 checkpoint。
6. 训练后自动评估测试集并生成图表。

---

## 6. CLI 使用

安装依赖：

```bash
uv sync
```

进入交互模式：

```bash
python main.py
```

训练任意模型与数据集组合：

```bash
python main.py --model vgg16 --dataset cifar10 train --epochs 50 --lr 0.01
```

评估模型：

```bash
python main.py --model vgg16 --dataset cifar10 eval --checkpoint outputs/vgg16/cifar10/checkpoints/best_model.pth
```

单张图片推理：

```bash
python main.py --model vgg16 --dataset cifar10 infer --checkpoint outputs/vgg16/cifar10/checkpoints/best_model.pth --image cat.jpg --top-k 5
```

运行基准测试：

```bash
python main.py --model all --dataset all benchmark --epochs 5
```

---

## 7. 输出产物

所有输出按 `{model}/{dataset}` 分层保存：

```text
outputs/
├── vgg16/
│   └── cifar10/
│       ├── checkpoints/
│       ├── logs/
│       └── visualizations/
└── benchmarks/
```

这种目录结构适合批量实验，因为不会把不同模型和数据集的产物混在一起。

---

## 8. 总结

`ALL-CNN` 的重点不是单个 CNN，而是一个可扩展的图像分类实验框架：

| 设计 | 价值 |
|:--|:--|
| 多模型支持 | 横向比较经典 CNN 架构 |
| 多数据集支持 | 测试模型在不同数据分布上的表现 |
| 注册表机制 | 新增模型和数据集更简单 |
| 自动 transform | 降低模型输入尺寸和通道适配成本 |
| benchmark | 系统比较参数量、准确率和推理时间 |

它适合作为 CNN 系列文章的工程总入口，也适合继续扩展 ResNet、DenseNet、MobileNet 等现代架构。

---

> **相关文章**：
> - [[NeuralNetwork/CNN/CNN-Overview|CNN 总览]]
> - [[NeuralNetwork/CNN/LeNet5|LeNet-5]]
> - [[NeuralNetwork/CNN/AlexNet|AlexNet]]
> - [[NeuralNetwork/CNN/VGG|VGG]]
> - [[NeuralNetwork/CNN/GoogLeNet|GoogLeNet]]
> - [[NeuralNetwork/CNN/NiN|NiN]]
