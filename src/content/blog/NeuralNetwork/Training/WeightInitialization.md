---
title: 权重初始化指南：Xavier、Kaiming 与初始化检查
date: 2026-05-09
category: 神经网络/训练
tags:
  - 高级教程
  - 深度学习
description: 按激活函数和网络结构选择权重初始化方法，讲解 Xavier、Kaiming、Orthogonal、PyTorch 默认初始化和常见排查方法。
image: https://img.yumeko.site/file/blog/articles/1780581351535.webp
status: published
---

## 1. 快速选择：按激活函数决定初始化

初始化的目标很直接：让前向传播的激活值不要逐层变大或变小，也让反向传播的梯度不要逐层爆炸或消失。

最常用的选择规则如下：

| 网络结构 / 激活函数 | 推荐初始化 | PyTorch 函数 |
| --- | --- | --- |
| ReLU / LeakyReLU / SiLU / GELU | Kaiming / He | `nn.init.kaiming_normal_` |
| Tanh / Sigmoid | Xavier / Glorot | `nn.init.xavier_uniform_` |
| 普通线性层，无明确激活 | Xavier / Glorot | `nn.init.xavier_uniform_` |
| RNN 隐藏状态矩阵 | Orthogonal | `nn.init.orthogonal_` |
| BatchNorm / LayerNorm 缩放参数 | 常数 1 | `nn.init.ones_` |
| bias | 常数 0 | `nn.init.zeros_` |

如果只想要一个默认工程方案：

```python
# ReLU 系 CNN / MLP 的常用选择
nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
nn.init.zeros_(module.bias)
```

## 2. 初始化要解决什么问题

### 2.1 权重太大会怎样

权重太大时，每层输出的数值范围会不断变大。

对 `Tanh` 和 `Sigmoid` 来说，大输入会让激活函数进入饱和区：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

当 $x$ 很大或很小时，$\sigma'(x)$ 接近 0，梯度会变小，深层网络容易学不动。

### 2.2 权重太小会怎样

权重太小时，每层输出会不断接近 0。网络越深，信号越弱，反向传播时梯度也容易变小。

这类问题常表现为：

- loss 几乎不下降
- logits 初始值接近全 0
- 深层参数梯度非常小

### 2.3 为什么不能全部初始化为 0

如果同一层所有神经元权重都为 0，它们会得到完全相同的输入、输出和梯度。

这叫对称性问题。训练过程中这些神经元会一直保持相同，等价于只训练了一个神经元。

## 3. Xavier / Glorot 初始化：适合 Tanh 和 Sigmoid

### 3.1 适用场景

`Xavier` 初始化适合输出分布比较对称的激活函数：

- `Tanh`
- `Sigmoid`
- 无激活的普通线性层
- 早期 CNN，例如 LeNet-5

它不优先用于 ReLU，因为 ReLU 会丢掉一半负值，Xavier 没有专门补偿这个方差变化。

### 3.2 数学定义

设一层的输入维度为 $n_{\text{in}}$，输出维度为 $n_{\text{out}}$。

Xavier 的目标是让输入和输出的方差尽量一致：

$$
\text{Var}(W)=\frac{2}{n_{\text{in}}+n_{\text{out}}}
$$

### 3.3 Xavier Uniform

从均匀分布采样：

$$
W \sim U(-a,a)
$$

其中：

$$
a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
$$

PyTorch 写法：

```python
nn.init.xavier_uniform_(module.weight)
```

### 3.4 Xavier Normal

从正态分布采样：

$$
W \sim \mathcal{N}(0,\sigma^2)
$$

其中：

$$
\sigma=\sqrt{\frac{2}{n_{\text{in}}+n_{\text{out}}}}
$$

PyTorch 写法：

```python
nn.init.xavier_normal_(module.weight)
```

### 3.5 代码示例

```python
def initXavier(module: nn.Module) -> None:
    """初始化适合 Tanh / Sigmoid 的层。"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

## 4. Kaiming / He 初始化：适合 ReLU 系激活函数

### 4.1 适用场景

`Kaiming` 初始化适合 ReLU 及其变体：

- `ReLU`
- `LeakyReLU`
- `ELU`
- `SiLU` / `Swish`
- `GELU`
- 大多数现代 CNN 和 MLP

在 CNN 中，如果卷积层后面接 `ReLU`，通常优先使用 Kaiming 初始化。

### 4.2 为什么 ReLU 需要 Kaiming

ReLU 的定义是：

$$
\text{ReLU}(x)=\max(0,x)
$$

如果输入大致以 0 为中心，ReLU 会把一半负值变成 0。输出方差会下降。

Kaiming 初始化在权重方差中加入了补偿，让 ReLU 后的信号尺度更稳定。

![Comparison.png](https://img.yumeko.site/file/blog/articles/1780581553011.webp)

### 4.3 Kaiming Normal

最常见形式：

$$
W \sim \mathcal{N}(0,\sigma^2)
$$

其中：

$$
\sigma=\sqrt{\frac{2}{n_{\text{in}}}}
$$

PyTorch 写法：

```python
nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
```

### 4.4 Kaiming Uniform

从均匀分布采样：

$$
W \sim U(-a,a)
$$

其中：

$$
a=\sqrt{\frac{6}{n_{\text{in}}}}
$$

PyTorch 写法：

```python
nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
```

### 4.5 fan_in 和 fan_out 怎么选

`mode="fan_in"`：优先保持前向传播时激活值方差稳定。

`mode="fan_out"`：优先保持反向传播时梯度方差稳定。

工程上常用规则：

| 层类型 | 常见选择 | 说明 |
| --- | --- | --- |
| `Linear` | `fan_in` | 保持前向输出稳定 |
| `Conv2d` 分类模型 | `fan_out` | CNN 经典实现中常见 |
| 不确定 | `fan_in` | 简单、安全、常用 |

### 4.6 LeakyReLU 的 negative_slope

如果激活函数是 `LeakyReLU(negative_slope=0.1)`，初始化时也要写同样的斜率：

```python
nn.init.kaiming_normal_(
    module.weight,
    mode="fan_in",
    nonlinearity="leaky_relu",
    a=0.1,
)
```

`a` 写错不会立刻报错，但会让初始化方差和真实激活函数不匹配。

### 4.7 代码示例

```python
def initKaiming(module: nn.Module) -> None:
    """初始化适合 ReLU 系激活函数的层。"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

## 5. Orthogonal 初始化：适合 RNN 和深层线性变换

### 5.1 适用场景

`Orthogonal` 初始化会让权重矩阵尽量保持向量长度，常用于：

- RNN / LSTM / GRU 的 hidden-to-hidden 权重
- 很深的线性网络
- 希望减少梯度方向变形的场景

它不是 CNN 的默认选择，但在循环网络中很常见。

### 5.2 数学直觉

正交矩阵满足：

$$
W^\top W=I
$$

这意味着输入向量经过矩阵乘法后，长度不会被随意放大或压缩。

### 5.3 PyTorch 用法

```python
nn.init.orthogonal_(module.weight)
```

对 RNN 类模块，可以按参数名分别处理：

```python
def initRnnWeights(module: nn.Module) -> None:
    """初始化 RNN / LSTM / GRU 权重。"""
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
```

## 6. 常见层怎么初始化

### 6.1 Conv2d 和 Linear

这两类层通常按后续激活函数决定：

| 后续激活函数 | 推荐初始化 |
| --- | --- |
| ReLU / GELU / SiLU | Kaiming |
| Tanh / Sigmoid | Xavier |
| 无激活 | Xavier |

### 6.2 BatchNorm 和 LayerNorm

归一化层通常这样初始化：

```python
nn.init.ones_(module.weight)
nn.init.zeros_(module.bias)
```

含义是：初始化时不改变归一化后的分布，先让归一化层保持“恒等缩放”。

### 6.3 Embedding

`nn.Embedding` 通常使用正态分布或均匀分布初始化。对于 NLP 任务，更常见的做法是直接加载预训练词向量。

如果有 `padding_idx`，对应行通常需要保持为 0，避免 PAD token 引入有效语义。

```python
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

if embedding.padding_idx is not None:
    with torch.no_grad():
        embedding.weight[embedding.padding_idx].fill_(0)
```

### 6.4 输出层

输出层不一定要和中间层一样初始化。

常见做法：

- 分类输出层：可以沿用 Xavier 或 Kaiming
- 回归输出层：可使用较小标准差，避免初始预测值过大
- 强化学习策略头：常用较小初始化，让初始策略更平稳

示例：

```python
nn.init.normal_(outputLayer.weight, mean=0.0, std=0.01)
nn.init.zeros_(outputLayer.bias)
```

## 7. PyTorch 默认初始化

### 7.1 Linear 和 Conv2d 默认值

PyTorch 的 `nn.Linear` 和 `nn.Conv2d` 会在创建层时自动初始化参数。

因此很多模型不手动初始化也能训练。但默认初始化不一定适合所有结构，尤其是：

- 使用 `Tanh` / `Sigmoid` 的旧式网络
- 特别深的网络
- 自定义层
- 对复现实验要求很高的项目

### 7.2 什么时候必须手动初始化

建议显式初始化的情况：

- 复现论文
- 网络很深，训练不稳定
- 激活函数不是 ReLU 系
- 自己写了自定义层
- 训练一开始 loss 不动或直接 NaN

## 8. 完整初始化函数

### 8.1 通用初始化函数

下面这个函数覆盖常见 CNN / MLP 场景：

```python
import torch
import torch.nn as nn


def initWeights(model: nn.Module, activation: str = "relu") -> None:
    """
    根据激活函数初始化模型权重。

    Args:
        model (nn.Module): 需要初始化的模型
        activation (str): 主要激活函数，可选 relu、leaky_relu、tanh、sigmoid
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if activation in ("relu", "leaky_relu", "gelu", "silu"):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif activation in ("tanh", "sigmoid"):
                nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            if activation in ("relu", "leaky_relu", "gelu", "silu"):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif activation in ("tanh", "sigmoid"):
                nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

使用方式：

```python
model = MyModel()
initWeights(model, activation="relu")
```

### 8.2 CNN 项目中的选择

| 模型 | 激活函数 | Conv 初始化 | Linear 初始化 |
| --- | --- | --- | --- |
| LeNet-5 | Tanh | Xavier Uniform | Xavier Uniform |
| AlexNet | ReLU | Kaiming Normal | Xavier Normal |
| VGG | ReLU | Kaiming Normal | Normal(0, 0.01) |
| NiN | ReLU | Kaiming Normal | N/A |
| GoogLeNet | ReLU | Kaiming Normal | Normal(0, 0.01) |
| ResNet | ReLU | Kaiming Normal | Kaiming / Xavier |

## 9. 如何检查初始化是否合理

### 9.1 检查权重分布

初始化后可以打印每层权重的均值和标准差：

```python
def printWeightStats(model: nn.Module) -> None:
    """打印模型权重分布，检查初始化是否异常。"""
    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue

        meanValue = param.data.mean().item()
        stdValue = param.data.std().item()
        print(f"{name}: mean={meanValue:.4f}, std={stdValue:.4f}")
```

合理现象：

- 均值接近 0
- 标准差不是 0
- 不同层标准差可以不同，但不应离谱地大

### 9.2 检查第一批 logits

分类任务中，可以在训练前看一眼初始 logits：

```python
model.eval()
with torch.no_grad():
    logits = model(images)
    print(logits.mean().item(), logits.std().item())
```

如果 logits 初始值极大，例如标准差几十甚至上百，通常说明初始化或输入归一化有问题。

### 9.3 训练异常时优先排查

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| loss 从第一步就是 NaN | 权重过大或输入未归一化 | 降低初始化尺度，检查输入 |
| loss 几乎不动 | 权重过小或梯度消失 | 检查激活函数和初始化匹配 |
| ReLU 网络越深越难训 | 初始化不匹配 | 使用 Kaiming |
| Tanh 网络很快饱和 | 权重过大 | 使用 Xavier，减小输入尺度 |
| 所有神经元行为接近一致 | 对称性没有打破 | 不要把 weight 全部置 0 |

## 10. 总结

### 10.1 记住这几条

- ReLU 系激活函数优先用 `Kaiming`
- Tanh / Sigmoid 优先用 `Xavier`
- bias 通常初始化为 0
- BatchNorm / LayerNorm 的 weight 初始化为 1，bias 初始化为 0
- RNN 的 hidden-to-hidden 权重常用 `Orthogonal`

### 10.2 最容易犯的错

- 对 ReLU 网络使用过小的 Xavier 初始化，导致深层信号变弱
- 对 Tanh / Sigmoid 网络使用过大的初始化，导致激活饱和
- 手写自定义层后忘记初始化
- 把所有 weight 初始化为 0，导致神经元无法分化
