---
title: AlexNet：深度学习革命的引爆者
date: 2026-05-23
category: 神经网络/CNN
tags:
  - 经典架构
description: AlexNet 在 2012 年以大幅优势赢得 ImageNet 竞赛，标志着深度学习时代的开始。本文详解其架构设计和多项关键创新。
image: https://img.yumeko.site/file/blog/cover/1780581687349.webp
status: published
---

## 1. 历史意义

2012 年，Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 提交的 AlexNet 在 ImageNet 大规模视觉识别挑战赛（ILSVRC）上以 **Top-5 错误率 15.3%** 夺冠，比第二名（26.2%）低了近 11 个百分点。

这一结果震惊了计算机视觉界。AlexNet 证明了：只要数据足够多、网络足够深、GPU 足够快，CNN 就能超越所有传统方法。深度学习的时代由此开启。

## 2. 完整架构

输入是 $224 \times 224 \times 3$ 的 RGB 图像（ImageNet 尺寸）。

| 层 | 操作 | 输出形状 | 参数量 |
| --- | --- | --- | ---:|
| Input | — | $(3, 224, 224)$ | 0 |
| Conv1 | Conv(11$\times$11, s=4) + ReLU | $(96, 55, 55)$ | 34,944 |
| Pool1 | MaxPool(3$\times$3, s=2) | $(96, 27, 27)$ | 0 |
| Conv2 | Conv(5$\times$5) + ReLU | $(256, 27, 27)$ | 614,656 |
| Pool2 | MaxPool(3$\times$3, s=2) | $(256, 13, 13)$ | 0 |
| Conv3 | Conv(3$\times$3) + ReLU | $(384, 13, 13)$ | 885,120 |
| Conv4 | Conv(3$\times$3) + ReLU | $(384, 13, 13)$ | 1,327,488 |
| Conv5 | Conv(3$\times$3) + ReLU | $(256, 13, 13)$ | 884,992 |
| Pool5 | MaxPool(3$\times$3, s=2) | $(256, 6, 6)$ | 0 |
| — | Flatten | $(9216,)$ | 0 |
| FC1 | Linear + ReLU + Dropout | $(4096,)$ | 37,752,832 |
| FC2 | Linear + ReLU + Dropout | $(4096,)$ | 16,781,312 |
| FC3 | Linear | $(1000,)$ | 4,097,000 |
| **总计** | | | **~62.4M** |

![AlexNet.jpg](https://img.yumeko.site/file/blog/articles/1780581145170.webp)


## 3. 关键创新

### ReLU 激活函数

AlexNet 是第一个大规模使用 ReLU 的 CNN。在此之前，Tanh 和 Sigmoid 是标准选择，但它们有严重的梯度消失问题。ReLU 的正数区域导数为常数 1，使得梯度能够有效传播到深层。

论文中报告：使用 ReLU 的 4 层 CNN 在 CIFAR-10 上达到 25% 训练误差所需的迭代次数，是使用 Tanh 的网络的 **1/6**。

### Dropout 正则化

AlexNet 在前两个全连接层（各有 4096 个神经元）使用了 [[NeuralNetwork/Training/Dropout|Dropout]](p=0.5) —— 这是 Dropout 在 CNN 中的首次大规模应用。没有 Dropout，如此庞大的全连接层会严重过拟合。

### 双 GPU 训练

原始 AlexNet 使用了 2 块 GTX 580 GPU（各 3GB 显存），将网络分成两半分别放在两块 GPU 上训练。这是工程上的巧妙设计——当时单块 GPU 显存不足以容纳全部 62M 参数。

我们项目中的实现是单 GPU 完整版（不做通道拆分）。

### Overlapping Pooling

传统的池化窗口不重叠（如 `kernel=2, stride=2`）。AlexNet 使用了重叠池化：`kernel=3, stride=2`。重叠池化能轻微缓解过拟合，论文报告 Top-1 错误率降低约 0.4%。

### Local Response Normalization（LRN）

在 Conv1 和 Conv2 的 ReLU 之后应用了 LRN——模仿生物神经元的"侧抑制"机制：强激活的神经元会抑制相邻神经元。

$$
b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\left(k + \alpha \displaystyle\sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a_{x,y}^{j})^2 \right)^\beta}
$$

LRN 后来被 BatchNorm 取代（BN 更有效），在现代架构中已很少使用。

## 4. 数据增强

AlexNet 使用了两种数据增强来防止过拟合：

**方式一：随机裁剪 + 水平翻转**

从 $256 \times 256$ 的原图中随机裁剪 $224 \times 224$ 的子区域，并随机水平翻转。这样将训练集扩大了 2048 倍（$(256-224)^2 \times 2$）。

**方式二：PCA 颜色增强**

对 RGB 像素值进行 PCA 主成分扰动，模拟光照和颜色变化：

$$
[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
$$

其中 $\mathbf{p}_i$ 和 $\lambda_i$ 是 RGB 像素值的第 $i$ 个特征向量和特征值，$\alpha_i$ 是随机高斯变量。这一技术将 Top-1 错误率降低了约 1%。

## 5. 权重初始化

```python
# Conv 层：Kaiming 初始化（配合 ReLU）
nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

# Linear 层：Xavier 初始化
nn.init.xavier_normal_(linear.weight)

# 偏置：全部初始化为 0
nn.init.constant_(module.bias, 0)
```

## 6. PyTorch 实现（简化版）

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
```

## 7. AlexNet 的局限

- **参数集中在 FC 层**：FC1 和 FC2 占了总参数的 87%
- **第一个卷积核过大**：$11 \times 11$ 卷积核参数多且感受野粗糙，后来的 [[NeuralNetwork/CNN/VGG|VGG]] 证明了堆叠小卷积核更好
- **需要大量显存**：原始版本需要双 GPU，训练成本高

AlexNet 与后续架构的全面对比见 [[NeuralNetwork/CNN/ArchitectureComparison|架构对比]]。

从 AlexNet 的 11x11 大卷积核到 [[NeuralNetwork/CNN/VGG|VGG]] 的全 3x3 小卷积核，卷积核设计理念发生了深刻变革。
