---
title: LeNet-5：CNN 的起点
date: 2026-05-23
category: NeuralNetwork/CNN/Architectures
tags:
  - LeNet
  - 经典架构
description: LeNet-5（1998）奠定了 CNN 的基础模式，本文逐层拆解其设计并对比现代改进。
image: https://img.yumeko.site/file/blog/LeNet5.png
status: published
---

## 1. 历史背景

LeNet-5 由 Yann LeCun 等人在 1998 年发表（论文《Gradient-Based Learning Applied to Document Recognition》），用于手写支票上的数字识别。它是第一个成功商用的 CNN，奠定了现代 CNN 的基础架构模式：**卷积 → 池化 → 全连接**。

尽管以今天的标准看非常简单（仅 ~61,000 参数），LeNet-5 的架构思想——局部连接、权重共享、层次化特征提取——至今仍是所有 CNN 的核心。

## 2. 完整架构

输入是 $32 \times 32$ 的灰度手写数字（MNIST 原图 $28 \times 28$，需要 padding/resize 到 $32 \times 32$）。

| 层名 | 操作 | 参数 | 输入形状 | 输出形状 |
|:---:| --- | --- | --- | --- |
| — | Resize | 28→32 | $(1,28,28)$ | $(1,32,32)$ |
| C1 | Conv2d + Tanh | in=1, out=6, k=5 | $(1,32,32)$ | $(6,28,28)$ |
| S2 | AvgPool2d | k=2, s=2 | $(6,28,28)$ | $(6,14,14)$ |
| C3 | Conv2d + Tanh | in=6, out=16, k=5 | $(6,14,14)$ | $(16,10,10)$ |
| S4 | AvgPool2d | k=2, s=2 | $(16,10,10)$ | $(16,5,5)$ |
| C5 | Conv2d + Tanh | in=16, out=120, k=5 | $(16,5,5)$ | $(120,1,1)$ |
| — | Flatten | — | $(120,1,1)$ | $(120,)$ |
| F6 | Linear + Tanh | in=120, out=84 | $(120,)$ | $(84,)$ |
| Output | Linear | in=84, out=10 | $(84,)$ | $(10,)$ |

![LeNet5.png](https://img.yumeko.site/file/articles/LeNet5/LeNet5.png)

### 参数计算

| 层 | 计算 | 参数量 |
| --- | --- | ---:|
| C1 | $1 \times 6 \times 5^2 + 6$ | 156 |
| C3 | $6 \times 16 \times 5^2 + 16$ | 2,416 |
| C5 | $16 \times 120 \times 5^2 + 120$ | 48,120 |
| F6 | $120 \times 84 + 84$ | 10,164 |
| Output | $84 \times 10 + 10$ | 850 |
| **总计** | | **~61,706** |

## 3. 关键设计细节

### C5 层的巧妙之处

C5 是一个 $5 \times 5$ 卷积，输入也是 $5 \times 5$，所以卷积核"滑不动"——每个核只产生一个输出值。这**等价于一个全连接层**，但保留为卷积形式是为了维持架构的卷积特征提取逻辑。这就是为什么尽管 C5 是卷积，输出却是 $(120, 1, 1)$。

### Tanh + Xavier 初始化

原始论文使用 [[NeuralNetwork/Theory/ActivationFunctions|激活函数]] Tanh（那时 ReLU 还未出现）。配合 Tanh，使用 **Xavier Uniform 初始化**是标准做法：

```python
for m in self.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

Xavier 确保信号在 Tanh 的活跃区间（$x$ 接近 0 时导数较大），而非饱和区间（$|x|$ 大时导数趋近 0）。
权重初始化内容请查看[[NeuralNetwork/Tips/Techniques/WeightInitialization|WeightInitialization]]
### Average Pooling 而非 Max Pooling

LeNet-5 使用平均[[NeuralNetwork/CNN/Foundations/PoolingLayer|池化]]。这在当时是自然选择——取局部平均值能平滑特征。但后来的实践证明 Max Pooling 保留最强信号通常效果更好，因此现代 CNN 几乎全部使用 Max Pooling。

### 无 Padding（Valid 卷积）

每次卷积后尺寸都会缩小。输入 $32 \times 32$，经过两次卷积+池化后变为 $5 \times 5$。现代 CNN 更倾向于使用 padding 保持尺寸不变，让池化专门负责下采样。

## 4. PyTorch 实现

```python
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # C1: (1, 32, 32) → (6, 28, 28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: (6, 28, 28) → (6, 14, 14)
        self.pool1 = nn.AvgPool2d(2, 2)
        # C3: (6, 14, 14) → (16, 10, 10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: (16, 10, 10) → (16, 5, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        # C5: (16, 5, 5) → (120, 1, 1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # F6: 120 → 84
        self.fc1 = nn.Linear(120, 84)
        # Output: 84 → num_classes
        self.fc2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.tanh(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 5. LeNet-5 与现代 CNN 的对比

| 方面 | LeNet-5（1998） | 现代 CNN |
| --- | :---: | :---: |
| 激活函数 | Tanh | **ReLU** / GELU |
| 池化方式 | 平均池化 | **最大池化** / Stride Conv |
| 归一化 | 无 | **BatchNorm** |
| 正则化 | 无 | **Dropout** |
| Padding | 无（valid 卷积） | **same 卷积** |
| 优化器 | 随机梯度下降 | **Adam** / AdamW |
| 参数量 | ~61K | 数百万至数十亿 |

## 6. 总结

LeNet-5 的意义不在于性能（今天的学生项目都能轻易超越），而在于它开创了 CNN 的范式。理解 LeNet-5 之后再看 [[NeuralNetwork/CNN/Architectures/AlexNet|AlexNet]]、VGG、ResNet，你会发现它们都是在 LeNet-5 的基础上 "加料"——换激活函数、加深、加 BN、改连接方式——但核心的 Conv→Pool→FC 模式从未改变。各架构的详细演进可参见 [[NeuralNetwork/CNN/Architectures/ArchitectureComparison|架构对比]]。

LeNet-5 的思想在后续架构中被继承和发扬，从 [[NeuralNetwork/CNN/Architectures/AlexNet|AlexNet]] 的 GPU 训练到 ResNet 的残差连接，Conv→Pool→FC 范式始终如一。
