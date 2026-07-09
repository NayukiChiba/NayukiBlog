---
title: 损失函数完全指南：从交叉熵到均方误差
date: 2026-05-23
category: 神经网络/训练
tags:
  - 基础
  - 深度学习
description: 深入理解损失函数的数学原理、参数细节和常见陷阱——从交叉熵到均方误差，涵盖分类与回归任务的完整损失函数指南。
image: https://img.yumeko.site/file/blog/cover/1780581807243.webp
status: published
---

## 1. 快速定位

损失函数是一个“不满意度”的度量。模型预测得越好，损失值越低；预测得越差，损失值越高。训练的目标就是最小化这个损失值：

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(f_\theta(x), y)
$$

这篇文章按 PyTorch 损失函数类名组织。想查某个损失函数时，直接找对应大标题。

### 1.1 损失函数选择速查

| 任务类型 | 推荐损失函数 | 模型输出 | 标签格式 |
| --- | --- | --- | --- |
| 单标签多分类 | `CrossEntropyLoss` | logits，形状 `(N, C)` | 类别索引，形状 `(N,)` |
| 二分类 | `BCEWithLogitsLoss` | logits，形状 `(N,)` 或 `(N, 1)` | 0/1 浮点数 |
| 多标签分类 | `BCEWithLogitsLoss` | logits，形状 `(N, C)` | 0/1 多热向量 |
| 普通回归 | `MSELoss` | 连续值 | 连续值 |
| 有异常值的回归 | `L1Loss` 或 `SmoothL1Loss` | 连续值 | 连续值 |
| 概率分布匹配 | `KLDivLoss` | log-probability | probability |

### 1.2 损失函数之间的关系

分类任务关心概率分布之间的差异，回归任务关心连续值之间的距离。

从最大似然估计看，它们可以统一理解为“假设数据服从某种分布，然后最小化负对数似然”：

| 损失函数 | 对应假设 |
| --- | --- |
| `CrossEntropyLoss` | 标签服从分类分布 |
| `BCEWithLogitsLoss` | 每个标签服从伯努利分布 |
| `MSELoss` | 误差服从高斯分布 |
| `L1Loss` | 误差服从拉普拉斯分布 |

## 2. CrossEntropyLoss：单标签多分类

### 2.1 适用场景

`CrossEntropyLoss` 用于单标签多分类任务。每个样本只有一个正确类别，例如 MNIST 的 10 分类、ImageNet 的 1000 分类。

模型最后一层应该直接输出 logits，不要手动加 `Softmax`：

```python
criterion = nn.CrossEntropyLoss()

# logits: (batch_size, num_classes)，原始输出
# labels: (batch_size,)，类别索引
logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 0])

loss = criterion(logits, labels)
```

### 2.2 数学定义

#### 2.2.1 交叉熵公式

交叉熵来自信息论，用来衡量两个概率分布之间的差异：

$$
H(p, q) = -\sum_i p(i)\log q(i)
$$

其中 $p$ 是真实分布，$q$ 是模型预测分布。

当真实标签是 one-hot 时，只有正确类别 $y$ 那一项不为 0，所以公式退化为：

$$
\mathcal{L} = -\log q(y)
$$

直觉：模型给正确类别的概率越高，loss 越小；给正确类别的概率越低，loss 越大。

#### 2.2.2 Softmax + NLLLoss

PyTorch 的 `nn.CrossEntropyLoss()` 内部等价于：

```text
CrossEntropyLoss = LogSoftmax + NLLLoss
```

第一步，Softmax 把 logits 转成概率分布：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

第二步，负对数似然只取正确类别的概率：

$$
\text{NLL} = -\log(\text{softmax}(z_y))
$$

其中 $z_y$ 是正确类别对应的 logit。

#### 2.2.3 数值稳定性

`CrossEntropyLoss` 内部使用 log-sum-exp 技巧防止溢出：

$$
\log\left(\sum e^{z_j}\right)
= a + \log\left(\sum e^{z_j-a}\right)
$$

其中 $a = \max(z_j)$。这避免了 $e^{100}$ 这种大数直接参与计算。

### 2.3 数值例子

#### 2.3.1 三分类样本

某样本 logits 为 $[2.0, 1.0, 0.1]$，真实标签为类别 0：

```text
Softmax 分母 = e^2.0 + e^1.0 + e^0.1
             = 7.389 + 2.718 + 1.105
             = 11.212

类别 0 概率 = 7.389 / 11.212 = 0.659
类别 1 概率 = 2.718 / 11.212 = 0.242
类别 2 概率 = 1.105 / 11.212 = 0.099

Loss = -log(0.659) = 0.417
```

如果模型对类别 0 非常确信：

```text
类别 0 概率 ~= 0.982
Loss = -log(0.982) = 0.018
```

如果模型预测方向错了：

```text
类别 0 概率 ~= 0.090
Loss = -log(0.090) = 2.408
```

#### 2.3.2 损失曲线直觉

如果模型对正确类别的概率是 0.9，Loss $= -\log(0.9) \approx 0.105$。

如果模型对正确类别的概率是 0.01，Loss $= -\log(0.01) \approx 4.605$。

核心洞察：交叉熵对“自信地错误”惩罚极重。正确类别概率从 0.01 降到 0.001 时，loss 会从 4.6 跳到 6.9。

![CrossEntropyLoss.png](https://img.yumeko.site/file/blog/articles/1780581168582.webp)

### 2.4 参数详解

#### 2.4.1 weight：类别权重

`weight` 用来处理类别不平衡。给少数类更大的权重，可以让模型更关注少数类：

```python
# 假设类别 2 样本极少，给它 3 倍权重
classWeights = torch.tensor([1.0, 1.0, 3.0, 1.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=classWeights)
```

数学形式变为：

$$
\mathcal{L} =
-w_y \log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)
$$

如果某个类别样本数大约是平均水平的 1/5，可以先尝试给它 5 倍权重，再通过验证集调优。

#### 2.4.2 label_smoothing：标签平滑

Label Smoothing 是一种正则化技术，用来防止模型对预测过于自信：

$$
y_i^{\text{smooth}}
= (1-\alpha)y_i^{\text{one-hot}} + \frac{\alpha}{K}
$$

当 $\alpha=0.1$ 时，正确类别标签从 1.0 变为 0.9，其余类别从 0 变为 $0.1/K$。

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

它的作用是让模型不要把概率分布压得过于尖锐，通常能缓解 [[NeuralNetwork/Troubleshooting/Overfitting|过拟合]]。

#### 2.4.3 ignore_index：忽略标签

`ignore_index` 用来忽略特定标签，例如语义分割中的填充区域：

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

被忽略的样本不参与 loss 计算，也不参与梯度回传。

#### 2.4.4 reduction：聚合方式

| reduction | 输出 |
| --- | --- |
| `'mean'` | 所有样本 loss 的平均值，默认值 |
| `'sum'` | 所有样本 loss 求和 |
| `'none'` | 返回每个样本的 loss |

```python
criterion = nn.CrossEntropyLoss(reduction='none')
perSampleLoss = criterion(logits, labels)
```

### 2.5 常见陷阱

#### 2.5.1 传入 one-hot 标签

`CrossEntropyLoss` 要求标签是类别索引，不是 one-hot 向量：

```python
# 正确：类别索引，shape 为 (3,)
labels = torch.tensor([3, 7, 1])

# 错误：one-hot，shape 为 (3, 10)
labels = torch.tensor([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])
```

#### 2.5.2 手动做 Softmax

不要手动对 logits 做 `Softmax` 再传给 `CrossEntropyLoss`：

```python
# 正确
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# 错误：CrossEntropyLoss 内部已经包含 LogSoftmax
probs = F.softmax(logits, dim=1)
loss = criterion(probs, labels)
```

#### 2.5.3 忘记类别不平衡

如果整体准确率不错，但少数类准确率明显很差，优先检查类别分布，并尝试 `weight` 参数或重采样。

### 2.6 项目封装用法

```python
from cnnlib.training.loss import createLoss

# 标准多分类
lossFn = createLoss("cross_entropy")

# 带 Label Smoothing
lossFn = createLoss("cross_entropy", label_smoothing=0.1)

# 带类别权重
classWeights = torch.tensor([1.0, 1.0, 2.0])
lossFn = createLoss("cross_entropy", weight=classWeights)
```

## 3. BCEWithLogitsLoss：二分类与多标签分类

### 3.1 适用场景

`BCEWithLogitsLoss` 用于二分类和多标签分类。

二分类：每个样本只有一个 0/1 标签。

多标签分类：每个样本可以同时属于多个类别，例如一张图片既有“猫”，又有“室内”，还可能有“沙发”。

### 3.2 数学定义

#### 3.2.1 Sigmoid

二分类不需要 Softmax，而是对每个输出独立做 Sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

输出可以理解为该类别为 1 的概率。

#### 3.2.2 Binary Cross Entropy

BCE 的公式为：

$$
\mathcal{L}
= -\left[y\log p + (1-y)\log(1-p)\right]
$$

其中 $y \in \{0,1\}$，$p=\sigma(z)$。

当 $y=1$ 时，损失退化为 $-\log p$；当 $y=0$ 时，损失退化为 $-\log(1-p)$。

### 3.3 PyTorch 用法

#### 3.3.1 二分类

```python
criterion = nn.BCEWithLogitsLoss()

# logits: (batch_size,)，每个样本一个原始输出
# labels: (batch_size,)，0 或 1，需要是浮点数
logits = torch.randn(32)
labels = torch.randint(0, 2, (32,)).float()

loss = criterion(logits, labels)
```

#### 3.3.2 多标签分类

```python
criterion = nn.BCEWithLogitsLoss()

# logits: (batch_size, num_classes)
# labels: (batch_size, num_classes)，每个类别独立为 0 或 1
logits = torch.randn(8, 5)
labels = torch.randint(0, 2, (8, 5)).float()

loss = criterion(logits, labels)
```

### 3.4 为什么不用 Sigmoid + BCELoss

`BCEWithLogitsLoss` 把 Sigmoid 和 BCE 合并在一起，并使用数值稳定实现。

不要这样写：

```python
# 不推荐：可能遇到 log(0) 或梯度数值不稳定
probs = torch.sigmoid(logits)
loss = nn.BCELoss()(probs, labels)
```

优先这样写：

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)
```

### 3.5 参数详解

#### 3.5.1 pos_weight：正样本权重

`pos_weight` 用来处理正负样本不平衡。它只放大正样本项：

$$
\mathcal{L}
= -\left[w_p y\log p + (1-y)\log(1-p)\right]
$$

如果正样本很少，可以设置：

```python
posWeight = torch.tensor([3.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=posWeight)
```

#### 3.5.2 weight：样本权重

`weight` 是逐样本权重，常用于让不同样本对 loss 的贡献不同。

和 `pos_weight` 的区别：

| 参数 | 作用对象 |
| --- | --- |
| `pos_weight` | 只调整正样本项 |
| `weight` | 调整整个样本或元素的 loss |

### 3.6 常见陷阱

#### 3.6.1 标签 dtype 错误

`BCEWithLogitsLoss` 的标签必须是浮点数，不是整数类别索引：

```python
# 正确
labels = torch.tensor([0, 1, 1, 0]).float()

# 错误
labels = torch.tensor([0, 1, 1, 0])
```

#### 3.6.2 把多分类误写成 BCE

如果每个样本只有一个类别，并且类别互斥，用 `CrossEntropyLoss`。

如果每个样本可以同时属于多个类别，用 `BCEWithLogitsLoss`。

## 4. MSELoss：均方误差

### 4.1 适用场景

`MSELoss` 主要用于回归任务，例如预测价格、温度、坐标、评分等连续值。

它适合异常值较少、误差分布近似高斯的任务。

### 4.2 数学定义

#### 4.2.1 公式

均方误差的定义：

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值。

直觉：MSE 衡量预测值与真实值之间的平均平方距离。平方会放大大误差，误差为 2 的样本对 loss 的贡献是误差为 1 的样本的 4 倍。

#### 4.2.2 梯度行为

MSE 对预测值的梯度为：

$$
\frac{\partial \text{MSE}}{\partial \hat{y}_i}
= -\frac{2}{n}(y_i-\hat{y}_i)
$$

这意味着：

- 误差越大，梯度越大
- 误差接近 0 时，梯度也接近 0
- 收敛后期会自然减速

### 4.3 PyTorch 用法

```python
criterion = nn.MSELoss()

predictions = torch.tensor([2.5, 1.0, 3.0, 4.5])
targets = torch.tensor([3.0, 1.5, 2.5, 4.0])

loss = criterion(predictions, targets)
# = (0.25 + 0.25 + 0.25 + 0.25) / 4
# = 0.25
```

### 4.4 参数详解

#### 4.4.1 reduction

```python
# 默认：所有元素平均
criterion = nn.MSELoss(reduction='mean')

# 求和，梯度会随元素数量放大
criterion = nn.MSELoss(reduction='sum')

# 不聚合，返回逐元素 loss
criterion = nn.MSELoss(reduction='none')
perElementLoss = criterion(predictions, targets)
```

### 4.5 常见陷阱

#### 4.5.1 形状广播

`MSELoss` 要求 predictions 和 targets 形状完全一致。形状不一致时，PyTorch 可能广播，结果看似能跑，实际 loss 是错的：

```python
# 正确：形状匹配
predictions = model(x)       # (batch_size, 1)
targets = y                  # (batch_size, 1)
loss = criterion(predictions, targets)

# 错误：会发生广播
predictions = model(x)       # (batch_size, 1)
targets = y.squeeze()        # (batch_size,)
loss = criterion(predictions, targets)
```

#### 4.5.2 目标值尺度太大

如果 targets 的值域很大，例如房价预测中的几十万，loss 数值会非常大。建议对 targets 做标准化，让训练更稳定。

#### 4.5.3 回归输出层加了不合适的激活

普通回归任务最后一层通常直接线性输出。如果用 `Sigmoid` 或 `ReLU`，会限制预测范围，可能导致模型无法拟合真实值。

### 4.6 项目封装用法

```python
from cnnlib.training.loss import createLoss

lossFn = createLoss("mse")
```

## 5. L1Loss：平均绝对误差

### 5.1 适用场景

`L1Loss` 也叫 MAE，适合回归任务中存在异常值的情况。

相比 `MSELoss`，它不会把大误差平方放大，因此更鲁棒。

### 5.2 数学定义

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

它的梯度大小基本恒定，不会像 MSE 那样随误差增大而增大。

### 5.3 MSELoss vs L1Loss

| 对比项 | `MSELoss` | `L1Loss` |
| --- | --- | --- |
| 公式 | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$ | $\frac{1}{n}\sum\lvert y_i-\hat{y}_i\rvert$ |
| 对大误差 | 极度敏感 | 线性敏感 |
| 异常值鲁棒性 | 较弱 | 较强 |
| 收敛后期 | 梯度自然变小 | 梯度不随误差变小 |
| 常见问题 | 容易被异常值主导 | 可能在最优点附近震荡 |

### 5.4 PyTorch 用法

```python
criterion = nn.L1Loss()
loss = criterion(predictions, targets)
```

## 6. SmoothL1Loss：Huber Loss 的 PyTorch 版本

### 6.1 适用场景

`SmoothL1Loss` 是 `MSELoss` 和 `L1Loss` 的折中：

- 误差小时像 MSE，梯度平滑
- 误差大时像 L1，对异常值更鲁棒

它常用于目标检测中的边界框回归。

### 6.2 数学定义

Huber Loss 的形式为：

$$
\text{Huber}(y,\hat{y}) =
\begin{cases}
\frac{1}{2}(y-\hat{y})^2,
& |y-\hat{y}| \leq \delta \\[6pt]
\delta |y-\hat{y}| - \frac{1}{2}\delta^2,
& |y-\hat{y}| > \delta
\end{cases}
$$

其中 $\delta$ 控制从平方惩罚切换到线性惩罚的位置。

### 6.3 PyTorch 用法

```python
criterion = nn.SmoothL1Loss(beta=1.0)

# beta 控制切换点
# |误差| <= beta 时接近 MSE 行为
# |误差| > beta 时接近 L1 行为
loss = criterion(predictions, targets)
```

### 6.4 数值直觉

```text
误差为 0.5 时:  MSE=0.25,  L1=0.5,   Huber~=0.125
误差为 3.0 时:  MSE=9.0,   L1=3.0,   Huber~=2.5
误差为 10.0 时: MSE=100,   L1=10.0,  Huber~=9.5
```

Huber 对大误差的惩罚远小于 MSE，同时在小误差区域保留平滑梯度。

## 7. KLDivLoss：概率分布匹配

### 7.1 适用场景

`KLDivLoss` 用于让一个概率分布接近另一个概率分布。常见场景包括知识蒸馏、概率标签训练、语言模型中的分布约束等。

如果目标是普通多分类硬标签，优先使用 `CrossEntropyLoss`，不要为了“更数学”强行换成 `KLDivLoss`。

### 7.2 数学定义

KL 散度衡量目标分布 $P$ 和预测分布 $Q$ 的差异：

$$
D_{\text{KL}}(P \parallel Q)
= \sum_i P(i)\log\frac{P(i)}{Q(i)}
$$

它不是对称距离：

$$
D_{\text{KL}}(P \parallel Q)
\neq D_{\text{KL}}(Q \parallel P)
$$

### 7.3 PyTorch 用法

`KLDivLoss` 的输入通常是 log-probability，目标是 probability：

```python
criterion = nn.KLDivLoss(reduction='batchmean')

logProbs = F.log_softmax(studentLogits, dim=1)
targetProbs = F.softmax(teacherLogits, dim=1)

loss = criterion(logProbs, targetProbs)
```

### 7.4 常见陷阱

#### 7.4.1 输入不是 log-probability

`KLDivLoss` 的 input 不是普通概率，而是 log-probability。通常用 `F.log_softmax`。

#### 7.4.2 reduction 选择错误

KL 散度常用 `reduction='batchmean'`，这样更接近数学定义。默认 `mean` 会对所有元素平均，数值尺度可能和预期不同。

## 8. 总结

### 8.1 按任务选择

| 你要做什么 | 直接用 |
| --- | --- |
| 单标签多分类 | `CrossEntropyLoss` |
| 二分类 | `BCEWithLogitsLoss` |
| 多标签分类 | `BCEWithLogitsLoss` |
| 普通回归 | `MSELoss` |
| 有异常值的回归 | `L1Loss` 或 `SmoothL1Loss` |
| 分布匹配或蒸馏 | `KLDivLoss` |

### 8.2 最重要的坑

#### 8.2.1 CrossEntropyLoss

- 输入 logits，不要手动 Softmax
- 标签是类别索引，不是 one-hot
- 类别不平衡时考虑 `weight`

#### 8.2.2 BCEWithLogitsLoss

- 输入 logits，不要手动 Sigmoid
- 标签需要是浮点数
- 多标签分类用它，多分类不要误用它

#### 8.2.3 MSELoss

- predictions 和 targets 形状必须一致
- 回归输出层通常不加激活函数
- targets 值域很大时建议标准化
