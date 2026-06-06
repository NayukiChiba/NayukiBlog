---
title: 数据预处理与增强详解
date: 2026-05-07
category: 神经网络/CNN
tags:
  - 深度学习
  - 基础
  - 数据处理
description: 从 Resize 到 Normalize 再到数据增强，理解 CNN 输入预处理流水线的每一步。
image: https://img.yumeko.site/file/blog/cover/1780581723302.webp
status: published
---

## 1. 为什么需要预处理？

原始图像数据的数值范围是 $[0, 255]$（uint8）。直接把这么大的数值喂给神经网络会导致：

- **梯度爆炸**：损失函数对参数的梯度过大，参数更新幅度失控
- **收敛缓慢**：不同特征的数值范围不一致，优化路径曲折

正确的预处理让数据分布适应网络的期望——均值为 0、方差为 1 的标准分布附近。

## 2. 预处理三步曲

### 步骤 1：Resize（调整尺寸）

```
PIL Image (任意尺寸) -> Resize((H, W)) -> (H, W)
```

不同模型需要不同的输入尺寸：
- LeNet-5：$32 \times 32$（MNIST 原图是 $28 \times 28$，需要 resize）
- AlexNet / VGG / GoogLeNet / ResNet：$224 \times 224$
- MNIST-CNN：直接使用 $28 \times 28$

### 步骤 2：ToTensor（转为张量）

```
PIL Image (H, W), uint8 [0, 255]
    v
Tensor (1, H, W), float32 [0.0, 1.0]
```

做了两件事：
- 把 PIL 图像（HxW）转为 PyTorch 张量（1xHxW），增加通道维度
- 把像素值从 $[0, 255]$ 缩放到 $[0.0, 1.0]$（除以 255）

### 步骤 3：Normalize（标准化）

$$
x_{\text{norm}} = \frac{x - \mu}{\sigma}
$$

对于 MNIST：
- $\mu = 0.1307$，$\sigma = 0.3081$

这两个值是在整个 MNIST 训练集上统计出来的。标准化后，数据分布近似于均值为 0、标准差为 1 的标准正态分布。

**为什么不用 $[0, 1]$ 的均值 0.5？** 因为 MNIST 的实际像素分布不是均匀的——大部分像素是 0（背景黑色），只有少数像素有值（笔画白色），所以实际均值远小于 0.5。使用真实的统计值能更准确地标准化数据。

## 3. 数据集 Normalize 参数参考

| 数据集 | 通道数 | 均值 | 标准差 |
| --- | --- | --- | --- |
| MNIST | 1 | (0.1307,) | (0.3081,) |
| FashionMNIST | 1 | (0.2860,) | (0.3530,) |
| CIFAR-10 | 3 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| CIFAR-100 | 3 | (0.5071, 0.4867, 0.4408) | (0.2675, 0.2565, 0.2761) |
| ImageNet | 3 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |

## 4. 完整的预处理代码

```python
from torchvision import transforms

# 训练集预处理（含数据增强）
trainTransform = transforms.Compose([
    transforms.Resize((32, 32)),          # 调整尺寸
    transforms.RandomAffine(
        degrees=10, translate=(0.1, 0.1)  # 数据增强
    ),
    transforms.ToTensor(),                # 转张量 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 验证/测试集预处理（不做增强）
testTransform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

## 5. 数据增强（Data Augmentation）

### 核心原则

[[NeuralNetwork/Training/DataAugmentation|数据增强]]应该模拟真实场景中可能出现的变换。

| 任务类型 | 合适的增强 | 不合适的增强 |
| --- | --- | --- |
| 手写数字 | 旋转、平移、轻微缩放 | 水平翻转（6 和 9 会混淆） |
| 自然图像 | 随机裁剪、水平翻转、颜色抖动 | 垂直翻转（天空不会在下面） |
| 医学影像 | 旋转、平移、弹性变形 | 颜色抖动（染色有诊断意义） |
| 文本识别 | 平移、轻微旋转 | 水平翻转（文字会反过来） |

### 常用增强方法

```python
# 几何变换
transforms.RandomHorizontalFlip(p=0.5)       # 水平翻转
transforms.RandomVerticalFlip(p=0.5)         # 垂直翻转
transforms.RandomRotation(degrees=15)         # 随机旋转
transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))  # 仿射变换
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # 随机裁剪缩放

# 颜色变换
transforms.ColorJitter(brightness=0.2, contrast=0.2,
                       saturation=0.2, hue=0.1)
transforms.RandomGrayscale(p=0.1)

# 高级增强（需额外安装）
# transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))  # 随机遮挡
# transforms.AutoAugment()        # 自动搜索增强策略
# transforms.RandAugment()        # 简化版 AutoAugment
```

### 增强的"金科玉律"

1. **只在训练集上做增强**，验证集和测试集只用基础预处理
2. **增强后的样本必须仍然是合理的**——人类能正确分类，模型才可能学会
3. **从轻到重**：先从最保守的增强开始，逐步增加强度
4. **可视化检查**：训练前先看看增强后的图像是什么样的，确保没有引入错误

![RandomAffine.png](https://img.yumeko.site/file/blog/articles/1780581190385.webp)

## 6. 通道适配

当数据集和模型的通道数不匹配时，需要做通道转换：

```python
# 灰度数据集 + RGB 模型 -> 灰度转伪 RGB
if datasetChannels == 1 and modelChannels == 3:
    transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))

# RGB 数据集 + 灰度模型 -> RGB 转灰度
if datasetChannels == 3 and modelChannels == 1:
    transform = transforms.Grayscale(num_output_channels=1)
```

## 7. 预处理流水线验证

在开始训练前，验证预处理流水线的正确性：

```python
import torch

# 创建测试输入
dummyInput = torch.randn(1, 1, 28, 28)
print(f"输入范围: [{dummyInput.min():.2f}, {dummyInput.max():.2f}]")

# 检查 DataLoader 输出
for images, labels in trainLoader:
    print(f"Batch 形状: {images.shape}")   # (B, C, H, W)
    print(f"像素值范围: [{images.min():.2f}, {images.max():.2f}]")
    print(f"标签形状: {labels.shape}")      # (B,)
    break
```

关于各项增强策略的选择和使用场景，参见 [[NeuralNetwork/Training/DataAugmentation|数据增强指南]]。
