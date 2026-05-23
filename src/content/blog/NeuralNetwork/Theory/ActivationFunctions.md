---
title: 激活函数详解：为什么 CNN 需要非线性
date: 2026-05-07
category: NeuralNetwork/Theory
tags:
  - 基础
  - 深度学习
description: 从 ReLU 到 Swish，理解各激活函数的公式、优缺点、以及如何配合权重初始化使用。
image: https://img.yumeko.site/file/blog/ActivationFunctions.png
status: published
---

## 1. 为什么必须有激活函数？

卷积操作本质上是**线性变换**：

$$
\text{Output} = \text{Kernel} \otimes \text{Input} + \text{Bias}
$$

如果网络中只有线性的卷积和全连接层，无论堆多少层，最终的等效变换仍然是线性的：$y = Wx + b$。线性模型的表达能力极其有限——它只能学习线性可分的模式。

现实世界中的问题几乎都是**非线性**的。比如，"8" 这个数字由上圈和下圈组成——这种"部分组合"的关系，不是一条直线能画出来的。

## 2. ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

- 如果输入 $x > 0$，输出 $x$（保持不变）
- 如果输入 $x \le 0$，输出 $0$（彻底关闭）

**优点**：
- **计算简单**：只需要一个 max 操作，比 Tanh/Sigmoid 快很多
- **缓解梯度消失**：正数区域的导数是常数 $1$，梯度不会随着网络加深而指数衰减
- **稀疏激活**：约 50% 的神经元输出为 $0$，网络自动实现了稀疏表达

**缺点**：
- **"死亡 ReLU"**：如果某个神经元总是接收负值输入，它永远输出 $0$，梯度也是 $0$，从此再也不更新
- 使用较小的学习率可以缓解这个问题

```python
import torch.nn as nn

# PyTorch 中使用 ReLU
nn.ReLU(inplace=True)  # inplace=True 节省显存
```

## 3. LeakyReLU

解决 ReLU "死亡"问题的变体——负值区域给一个很小的斜率而不是直接截断为 0：

$$
\text{LeakyReLU}(x) = \max(0.01x, x)
$$

通常 slope 取 $0.01$。PReLU 则将这个斜率变为可学习参数。

## 4. Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- 输出范围：$(-1, 1)$，关于原点对称（零中心化）
- 在 $x$ 接近原点时近似线性，在两端饱和（导数趋近于 $0$）

**与 ReLU 的对比**：
- Tanh 在深层网络中更容易出现**梯度消失**（两端导数趋近于 $0$）
- 但 Tanh 的输出有正有负，零中心化有利于下一层的学习
- 原始 LeNet-5 使用 Tanh，现代 CNN 几乎全部使用 ReLU 或其变体

配合 [[NeuralNetwork/Theory/BatchNormalization|BatchNorm]] 可以有效缓解梯度消失问题，同时建议按照 [[NeuralNetwork/Tips/Techniques/WeightInitialization|权重初始化]] 指南选择合适的初始化策略。

## 5. Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- 输出范围：$(0, 1)$
- 历史上常用于二分类输出层
- 现在已被 ReLU 取代（隐藏层）和直接使用 BCEWithLogitsLoss（输出层）

**问题**：Sigmoid 在两端完全饱和，梯度几乎为 0，深层网络训练极其困难。这是 2012 年之前深层神经网络难以训练的主要原因之一。

## 6. Swish / SiLU

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

由 Google Brain 通过神经架构搜索发现。在深层网络上通常优于 ReLU，被 EfficientNet 等现代架构采用。

```python
# PyTorch 1.7+ 内置
nn.SiLU()  # 即 Swish
```

## 7. GELU

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

Transformer 架构（BERT, GPT, ViT）的标准激活函数。比 ReLU 更平滑，在负数区域不是完全截断而是"概率性"地通过。

```python
nn.GELU()
```

## 8. 激活函数对比总结

| 激活函数 | 公式 | 输出范围 | 使用场景 | 配合初始化 |
| --- | --- | --- | --- | --- |
| ReLU | $\max(0, x)$ | $[0, \infty)$ | CNN 默认选择 | **Kaiming** |
| LeakyReLU | $\max(0.01x, x)$ | $(-\infty, \infty)$ | ReLU 的替代 | Kaiming |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | LeNet-5, RNN | **Xavier** |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | 二分类输出层 | Xavier |
| Swish/SiLU | $x \cdot \sigma(x)$ | $(-\infty, \infty)$ | EfficientNet | Kaiming |
| GELU | $x \cdot \Phi(x)$ | $(-\infty, \infty)$ | Transformer/ViT | Kaiming |

![ActivationFunction.png](https://img.yumeko.site/file/articles/CNN/ActivationFunction.png)

![ActiFunc.png](https://img.yumeko.site/file/articles/ActivationFunctions/ActiFunc.png)

## 9. 激活函数与权重初始化的配合

这是初学者最容易忽略的点——激活函数的选择决定了应该用什么初始化方式：

| 激活函数 | 推荐初始化 | 原因 |
| --- | --- | --- |
| ReLU 族 | **Kaiming/He** | 考虑了 ReLU 的方差减半效应 |
| Tanh/Sigmoid | **Xavier/Glorot** | 让值落在活跃区而非饱和区 |
| Swish/GELU | Kaiming（近似可行） | 行为类似 ReLU |

```python
# ReLU + Kaiming
nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')

# Tanh + Xavier
nn.init.xavier_uniform_(conv.weight)
```

**PyTorch 的默认行为**：如果没有显式指定初始化方式，PyTorch 对 `nn.Linear` 和 `nn.Conv2d` 默认使用 Kaiming Uniform 初始化。这就是为什么很多教程不写初始化代码也能正常运行——但对于 Tanh 这类激活函数，默认的 Kaiming 不是最优的，需要手动改为 Xavier。
![FuncCoop.png](https://img.yumeko.site/file/articles/ActivationFunctions/FuncCoop.png)

## 10. 如何选择激活函数

- **CNN 隐藏层**：ReLU（最稳妥）
- **担心 Dead ReLU**：LeakyReLU
- **Transformer / ViT**：GELU
- **想要最新最好**：SiLU/Swish
- **二分类输出层**：直接用 BCEWithLogitsLoss，不需要手动 Sigmoid
- **多分类输出层**：直接用 CrossEntropyLoss，不需要手动 Softmax

激活函数的选择直接影响训练稳定性，具体配合方式参见 [[NeuralNetwork/Tips/Techniques/WeightInitialization|权重初始化]] 指南。
