---
title: 权重初始化：Xavier 与 Kaiming 详解
date: 2026-05-09
category: 神经网络/训练
tags:
  - 高级教程
  - 深度学习
description: 为什么初始化方式决定训练成败？从数学原理到代码实现，彻底理解 Xavier 和 Kaiming 初始化。
image: https://img.yumeko.site/file/blog/articles/1780581351535.webp
status: published
---

## 1. 为什么初始化很重要？

想象你在一个巨大的山谷里找最低点（优化损失函数）。如果你的初始位置在山顶附近，梯度很大，几步就能走下山；但如果你的初始位置在几千公里外的平原上——也就是参数初始值离谱地大或离谱地小——你要走很久才能到山脚下，甚至可能永远走不到。

**数学上的问题**：

- **初始化太大** → 激活值进入饱和区（Tanh/Sigmoid 两端导数趋近于 0）→ 梯度消失 → 学不动
- **初始化太小** → 信号在网络传播中越来越微弱 → 深层几乎收不到信号 → 同样学不动
- **初始化全为 0** → 所有神经元做完全相同的计算 → 对称性问题，等价于只有 1 个神经元

## 2. Xavier/Glorot 初始化

**适用[[NeuralNetwork/Training/ActivationFunctions|激活函数]]**：Tanh、Sigmoid

**核心思想**：让信号在前向传播和反向传播中，每一层的方差都保持一致。这样信号不会在深层衰减或爆炸。

假设一层有 $n_{\text{in}}$ 个输入神经元和 $n_{\text{out}}$ 个输出神经元。为了让输出的方差 ≈ 输入的方差：

$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

**Xavier Uniform**：在 $[-\text{limit}, \text{limit}]$ 范围内均匀采样：

$$
\text{limit} = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
$$

**Xavier Normal**：从正态分布 $\mathcal{N}(0, \text{std}^2)$ 采样：

$$
\text{std} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}
$$

**代码**（LeNet-5 使用）：

```python
for module in self.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

## 3. Kaiming/He 初始化

**适用激活函数**：ReLU 及其变体（LeakyReLU、ELU 等）

**为什么 ReLU 不能用 Xavier？**

ReLU 会把负值全部置为 0。从方差的角度看，经过 ReLU 后信号的方差会减半（一半的值被丢弃）。Xavier 没有考虑这个减半效应，导致使用 Xavier + ReLU 时信号逐渐衰减。

Kaiming 初始化将输出方差目标调整为考虑 ReLU 的减半效应：

$$
\text{std} = \sqrt{\frac{2}{n_{\text{in}}}} \quad \text{（Kaiming Normal）}
$$

$$
\text{limit} = \sqrt{\frac{6}{n_{\text{in}}}} \quad \text{（Kaiming Uniform）}
$$

注意：Kaiming 只用 $n_{\text{in}}$，而 Xavier 用 $(n_{\text{in}} + n_{\text{out}}) / 2$。对于 ReLU，只考虑前向传播的方差保持即可。

```python
# Kaiming 初始化
nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
```

**PyTorch 的默认行为**：如果没显式指定初始化，PyTorch 对 `nn.Linear` 和 `nn.Conv2d` 默认使用 Kaiming Uniform。这就是为什么很多教程不写初始化也能正常运行——但对于 Tanh 类激活函数，默认的 Kaiming 不是最优的，需要手动改为 Xavier。

![Comparison.png](https://img.yumeko.site/file/blog/articles/1780581553011.webp)

## 4. 选择对照表

| 激活函数 | 推荐初始化 | 原因 |
| --- | --- | --- |
| ReLU / LeakyReLU | **Kaiming/He** | 考虑了 ReLU 的方差减半效应 |
| Tanh / Sigmoid | **Xavier/Glorot** | 让值落在活跃区而非饱和区 |
| Swish / GELU | Kaiming（近似可行） | 行为类似 ReLU |
| 线性（无激活） | Xavier | 标准选择 |

## 5. CNN 项目中的初始化实践

| 模型 | 激活函数 | Conv 初始化 | Linear 初始化 |
| --- | --- | --- | --- |
| LeNet-5 | Tanh | Xavier Uniform | Xavier Uniform |
| AlexNet | ReLU | Kaiming Normal | Xavier Normal |
| VGG | ReLU | Kaiming Normal | Normal(0, 0.01) |
| NiN | ReLU | Kaiming Normal | N/A |
| GoogLeNet | ReLU | Kaiming Normal | Normal(0, 0.01) |

## 6. 完整的初始化函数

以下函数根据激活函数和 [[NeuralNetwork/Training/BatchNormalization|BN]] 层类型自动选择初始化方式：

```python
def initWeights(model, activation='relu'):
    """根据激活函数自动选择初始化方式"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if activation in ('relu', 'leaky_relu'):
                nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
            elif activation in ('tanh', 'sigmoid'):
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```



