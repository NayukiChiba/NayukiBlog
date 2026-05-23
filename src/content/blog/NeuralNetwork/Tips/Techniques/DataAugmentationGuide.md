---
title: 数据增强策略选择指南
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 数据处理
  - 高级教程
description: 不同任务应该用哪些数据增强？哪些增强会适得其反？一份实用的策略选择指南。
image: https://img.yumeko.site/file/blog/DataAugmentationGuide.png
status: published
---

## 1. 核心原则

数据增强应该模拟真实场景中可能出现的变换，它是防止[[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]]最有效的正则化手段之一。

| 任务类型 | 合适的增强 | 不合适的增强 |
| --- | --- | --- |
| 手写数字 | 旋转、平移、轻微缩放 | 水平翻转（6 和 9 会混淆） |
| 自然图像分类 | 随机裁剪、水平翻转、颜色抖动 | 垂直翻转（天空不会在下面） |
| 医学影像 | 旋转、平移、弹性变形 | 颜色抖动（染色有诊断意义） |
| 文本识别 | 平移、轻微旋转 | 水平翻转（文字会反过来） |

## 2. 常用增强方法

### 几何变换

```python
from torchvision import transforms

# 翻转
transforms.RandomHorizontalFlip(p=0.5)       # 水平翻转（自然图像标配）
transforms.RandomVerticalFlip(p=0.5)         # 垂直翻转（卫星/显微镜图适用）

# 旋转
transforms.RandomRotation(degrees=15)         # ±15°（MNIST 用 ±10° 足够）

# 仿射变换
transforms.RandomAffine(
    degrees=10,              # 旋转 ±10°
    translate=(0.1, 0.1),    # 平移 10%
    scale=(0.9, 1.1)         # 缩放 90%~110%
)

# 随机裁剪（ImageNet 标配）
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
```

### 颜色/光照变换

```python
# 颜色抖动
transforms.ColorJitter(
    brightness=0.2,    # 亮度 ±20%
    contrast=0.2,      # 对比度 ±20%
    saturation=0.2,    # 饱和度 ±20%
    hue=0.1             # 色相 ±10%
)

# 随机灰度
transforms.RandomGrayscale(p=0.1)
```

### 高级增强

```python
# 随机遮挡（Cutout）
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
```

![DataAugment.png](https://img.yumeko.site/file/articles/DataAugmentationGuide/DataAugment.png)

## 3. 增强的"金科玉律"

1. **只在训练集上做增强**，验证集和测试集只用基础[[NeuralNetwork/CNN/Foundations/DataPreprocessing|数据预处理]]（ToTensor + Normalize）
2. **增强后的样本必须仍然合理**——人类能正确分类，模型才可能学会
3. **从轻到重**：先从最保守的增强开始，逐步增加强度
4. **可视化检查**：训练前先看增强后的图像，确保没有引入错误

## 4. MNIST-CNN 的数据增强

```python
transforms.RandomAffine(
    degrees=10,          # 旋转：±10°
    translate=(0.1, 0.1) # 平移：±10%
)
```

**为什么只有旋转和平移？** 手写数字最常见的变化就是书写角度（有人写得斜）和位置偏移（有人偏左）。这两种增强足够模拟真实场景，且不改变类别。

**为什么不做水平翻转？** 某些数字翻转后会变成另一个数字（如 6 ↔ 9 的某种写法），引入标签噪声。

## 5. ImageNet 风格的标准增强

```python
# 训练集标准增强（AlexNet/VGG/ResNet 通用）
trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 验证/测试集：只做 resize + center crop
valTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

## 6. 选择流程

```
任务类型 → 分析真实场景变化 → 选择对应增强 → 可视化验证 → 调整强度
```

不要盲目堆砌所有增强方法——每种增强都应该有明确的"模拟目标"。



