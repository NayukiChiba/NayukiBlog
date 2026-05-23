---
title: Adam：自适应时刻估计
date: 2026-05-23
category: NeuralNetwork/Tips/Optimizers
tags:
  - Adam
  - 优化器
description: 拆解 Adam 的公式、参数含义、优点和常见陷阱，理解为什么它是现代默认优化器。
image: https://img.yumeko.site/file/blog/Optimizer.png
status: published
---

## 1. Adam 的核心思想

Adam（Adaptive Moment Estimation）结合了两种改进思路：

- **Momentum**：用梯度的指数移动平均（一阶矩）做平滑
- **自适应学习率**：用梯度平方的指数移动平均（二阶矩）调整每个参数的步长

结果：每个参数自动获得合适的学习率——梯度大、波动大的参数学习率自动变小；梯度小、稳定的参数学习率自动变大。更多优化器的分类和选择策略，参见 [[NeuralNetwork/Tips/Optimizers/OptimizerOverview|优化器概述]]。

## 2. Adam 公式拆解

**一阶矩（动量项）**：

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

**二阶矩（自适应项）**：

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

**偏差校正**：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**参数更新**：

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**偏差校正为什么必要？** 在最初的几步（$t$ 很小时），$m_t$ 和 $v_t$ 由于初始化为 0，会偏小。除以 $(1-\beta^t)$ 校正这个偏差。

![SGDvsAdam.png](https://img.yumeko.site/file/articles/Adam/SGDvsAdam.png)

## 3. 参数含义

| 参数 | 默认值 | 含义 |
| --- | :---: | --- |
| `lr` | 0.001 | 学习率 |
| `betas=(0.9, 0.999)` | (0.9, 0.999) | $\beta_1$（动量衰减）和 $\beta_2$（自适应衰减） |
| `eps` | 1e-8 | 防止除零 |
| `weight_decay` | 0 | 权重衰减（对 Adam 来说不是真正的 L2） |

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

## 4. Adam 的优点

- **开箱即用**：默认 `lr=0.001` 在大多数任务上表现良好
- **对初始学习率不敏感**：自适应机制自动调整
- **收敛快**：训练初期快速下降
- **适合稀疏梯度**：每个参数独立的学习率

## 5. 常见问题

### 问题 1：泛化不如 SGD

在一些大规模图像分类任务上，Adam 的最终精度略低于精心调参的 [[NeuralNetwork/Tips/Optimizers/SGD|SGD]]+Momentum。但这差距正在缩小（[[NeuralNetwork/Tips/Optimizers/AdamW|AdamW]] 改善了这一问题）。

### 问题 2：Weight Decay 不正确

Adam 中的 `weight_decay` 与 SGD 不同——它对自适应学习率后的参数做衰减，不是标准的 L2 正则化。这也是 AdamW 被提出的原因。

### 问题 3：收敛后可能不稳定

Adam 在训练后期可能"抖动"，因为它给每个参数分配的学习率会随时间变化。配合学习率调度（如 CosineAnnealing）可以缓解。

## 6. 何时用 Adam

- 快速实验和原型开发（首选）
- 小到中等规模的数据集
- 对超参数不敏感的场景
- Transformer / NLP 任务（Adam 是标配）
