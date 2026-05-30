---
title: CNN 架构对比总结
date: 2026-05-23
category: NeuralNetwork/CNN/Architectures
tags:
  - 总结
description: 一张表纵览 LeNet 到 ResNet 的所有经典 CNN 架构，理解每一代的核心创新与演进脉络。
image: https://img.yumeko.site/file/blog/CNNArchitectureComparison.png
status: published
---

## 1. 演进脉络总览

![CNNArchitectureComparison.png](https://img.yumeko.site/file/articles/ArchitectureComparison/CNNArchitectureComparison.png)

## 2. 完整对比表

| 架构 | 年份 | 层数 | 参数量 | Top-5 错误率 | 核心创新 |
| --- | :---: | :---: | ---: | :---: | --- |
| LeNet-5 | 1998 | 5 | 61K | — | Conv→Pool→FC 基础模式 |
| AlexNet | 2012 | 8 | 62M | 15.3% | ReLU + Dropout + GPU 训练 |
| VGG-16 | 2014 | 16 | 138M | 7.3% | 全部 3×3 小卷积核 |
| VGG-19 | 2014 | 19 | 144M | 7.3% | 同 VGG-16，更深 |
| NiN | 2014 | 12 | ~7.5M | — | mlpconv + **GAP 替代 FC** |
| GoogLeNet | 2014 | 22 | 7M | 6.7% | Inception 多分支 + 1×1 降维 |
| ResNet-18 | 2015 | 18 | 11.7M | 10.7%* | 残差连接 |
| ResNet-34 | 2015 | 34 | 21.8M | 8.6%* | 残差连接 |
| ResNet-50 | 2015 | 50 | 25.6M | 5.3%* | Bottleneck + 残差连接 |
| ResNet-101 | 2015 | 101 | 44.5M | 4.4%* | 更深 |
| ResNet-152 | 2015 | 152 | 60.2M | 3.6%* | 极深残差网络 |

[//]: # (TODO: 带星号的数据为近似值，需要确认准确数字)

## 3. 关键创新演进

### 激活函数
```
Tanh (LeNet-5) → ReLU (AlexNet) → ReLU (所有现代架构)
```

### 归一化
```
无 (LeNet-5) → LRN (AlexNet) → BN (VGG, GoogLeNet, ResNet)
```

### 正则化
```
无 (LeNet-5) → Dropout (AlexNet, VGG) → BN + Dropout + Data Aug
```

### 池化
```
AvgPool (LeNet-5) → MaxPool (AlexNet, VGG, ResNet) → GAP (NiN, GoogLeNet, ResNet)
```

### 特征提取
```
单路串行 (LeNet-5~VGG) → 多分支并行 (GoogLeNet) → Skip Connection (ResNet)
```

### 分类器
```
多层 FC (LeNet-5, AlexNet, VGG) → GAP + 单层 FC (NiN, GoogLeNet, ResNet)
```

## 4. 参数量 vs 深度

![ParamsComparison.png](https://img.yumeko.site/file/articles/ArchitectureComparison/ParamsComparison.png)

注意 GoogLeNet 和 ResNet 在参数量远小于 VGG 的情况下达到了更深的深度和更好的效果——这是架构设计（GAP、Bottleneck、多分支）带来的效率提升。

## 5. 设计理念演进

| 时代 | 代表架构 | 设计理念 |
| --- | --- | --- |
| 前深度学习 | LeNet-5 | 简单的 Conv→Pool→FC 堆叠 |
| 深度学习初期 | AlexNet | 更大、更深、更强的 GPU |
| 结构探索期 | VGG, GoogLeNet | 精心设计的网络结构替代暴力堆叠 |
| 深度突破期 | ResNet | Skip connection 打破深度瓶颈 |

## 6. 选择建议

| 场景 | 推荐架构 | 原因 |
| --- | --- | --- |
| 教学/学习 | LeNet-5 → VGG-16 → ResNet-18 | 难度递增，覆盖核心思想 |
| 快速实验 | ResNet-18 | 轻量、收敛快、效果不错 |
| 迁移学习 | ResNet-50 | 预训练模型多、效果稳定 |
| 高精度需求 | ResNet-101/152 | 更深，适合大数据集 |
| 资源受限 | MobileNet / ShuffleNet | 本文未覆盖，但值得了解 |
| 理解 CNN 历史 | 全部按时间顺序 | 每个架构解决一个特定问题 |

每个架构都在解决前一代的核心瓶颈，从 LeNet-5 的基础范式到 ResNet 的残差革命，构成了 CNN 发展的一条清晰主线。
