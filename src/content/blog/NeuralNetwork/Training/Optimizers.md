---
title: 优化器完全指南：从 SGD 到 AdamW
date: 2026-05-23
category: NeuralNetwork/Training
tags:
  - 总结
  - 深度学习
  - 基础
  - 高级教程
description: 从基础梯度下降到 AdamW，完整理解优化器演进、公式原理、参数调优与最佳实践。
image: https://img.yumeko.site/file/blog/Optimizer.png
status: published
---

## 1. 优化器概述

### 1.1 优化器是什么？

优化器的任务：根据损失函数对模型参数的梯度，调整参数使损失越来越小。

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

其中 $\theta$ 是参数，$\eta$ 是学习率，$g_t = \nabla_\theta \mathcal{L}$ 是梯度。

### 1.2 梯度下降的三种变体

| 变体 | 每次更新使用的数据 | 特点 |
| --- | --- | --- |
| BGD（批量梯度下降） | 全部训练数据 | 梯度最准，但计算代价太高 |
| SGD（随机梯度下降） | 单个样本 | 更新快但噪声大 |
| Mini-batch SGD | 一个 batch | 平衡速度和稳定性，实际使用最多 |

深度学习中的 "SGD" 通常指 Mini-batch SGD。

![GD.png](https://img.yumeko.site/file/articles/OptimizerOverview/GD.png)

### 1.3 PyTorch 使用模式

```python
# 1. 创建优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 2. 训练循环
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()      # 清零梯度（PyTorch 默认累加）
        loss = criterion(model(x), y)
        loss.backward()            # 计算梯度
        optimizer.step()           # 更新参数
```

**为什么必须 `zero_grad()`？** PyTorch 默认累加梯度（`grad += new_grad`），这设计是为了支持 RNN 等需要梯度累积的场景。大多数情况下每次更新前需要清零。

### 1.4 优化器分类

| 类型       | 代表                                   | 特点|
| -------- | ------------------------------------ | ------------------------------------- |
| 基础 SGD   |SGD| 所有参数共享一个学习率 |
| 带动量      | SGD+Momentum| 积累历史梯度方向，加速收敛|
| 自适应学习率   | AdaGrad, RMSprop| 每个参数有自己的学习率|
| 动量 + 自适应 | Adam, AdamW      | 结合两者优势，现代默认选择 |

---

## 2. SGD + Momentum + Nesterov

### 2.1 朴素 SGD

$$
w_{t+1} = w_t - \eta \cdot \nabla \mathcal{L}(w_t)
$$

**问题**：SGD 对每个参数使用相同的学习率，如果损失曲面在某个方向陡峭、另一个方向平缓（"峡谷"形状），优化会来回震荡，收敛很慢。

### 2.2 SGD + Momentum

引入"速度"概念，让参数更新不仅依赖当前梯度，也依赖历史更新方向：

$$
v_t = \mu \cdot v_{t-1} + \eta \cdot \nabla \mathcal{L}(w_{t-1})
$$

$$
w_t = w_{t-1} - v_t
$$

**直观理解**：就像球滚下山坡——不仅受当前坡度（梯度）影响，还有惯性（历史动量）。惯性让优化路径更平滑，能快速通过平坦区域。

动量系数 $\mu=0.9$ 意味着当前更新方向中 90% 来自历史，10% 来自当前梯度。

![TODO: 有/无Momentum的SGD优化路径对比图，左图在峡谷中来回震荡，右图Momentum平滑加速下坡]

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9       # 标准动量值
)
```

### 2.3 Nesterov 加速梯度（NAG）

Nesterov Momentum 是动量的改进版：先按动量方向走一步，然后在那个位置计算梯度。

$$
v_t = \mu \cdot v_{t-1} + \eta \cdot \nabla \mathcal{L}(w_{t-1} - \mu \cdot v_{t-1})
$$

**直观理解**：普通动量是"先看再走"，Nesterov 是"先走再看再修正"——走到预估位置后，看那里的坡度有没有变化，再调整方向。

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)
```

### 2.4 SGD vs Adam

| | SGD + Momentum | Adam |
| --- | :---: | :---: |
| 收敛速度 | 较慢 | **较快** |
| 泛化能力 | **通常更好** | 有时略差 |
| 调参难度 | 需要仔细调 | 开箱即用 |
| 适合场景 | 大规模图像分类，追求 SOTA | 快速实验和小任务 |

**为什么 SGD 泛化更好？** 一种解释是：自适应优化器（如 Adam）找到的极小值比较"尖锐"，SGD 找到的更"平坦"——平坦的极小值泛化更好。

### 2.5 学习率调参

SGD 对学习率非常敏感：

| lr 值 | 可能效果 |
| :---: | --- |
| 0.1 | 大任务常用（配合 step decay） |
| 0.01 | 安全起步值 |
| 0.001 | 对于 SGD 来说可能太小 |

```python
# 典型 SGD 训练配置
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

### 2.6 Weight Decay

SGD 中 weight_decay 等效于 L2 正则化：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
```

对于 SGD，weight_decay 在梯度上加上 $w \cdot \text{weight\_decay}$ 项。

---

## 3. RMSprop

### 3.1 RMSprop 的设计动机

AdaGrad 给每个参数分配学习率 $\eta / \sqrt{G_t}$，其中 $G_t$ 是从训练开始以来所有梯度平方的累加和。问题是 $G_t$ 只增不减，学习率最终会降到 0。

RMSprop 的解决方案：用指数移动平均替代累加和，让"记忆"逐渐遗忘旧梯度：

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2
$$

### 3.2 参数更新公式

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t
$$

其中 $v_t$ 是梯度平方的指数移动平均，$\beta$ 通常设为 $0.9$（类比动量）。

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.9,      # 平滑系数（类似 beta）
    eps=1e-8,
    momentum=0.0,   # 可选，加动量
    weight_decay=0.0
)
```

### 3.3 RMSprop 的适用场景

- **RNN / LSTM**：RMSprop 是 RNN 训练的经典选择（Adam 出现之前）
- **强化学习**：非平稳目标（Q 值不断变化），RMSprop 的指数遗忘很合适
- **非平稳优化问题**：目标函数随时间改变的场景

### 3.4 RMSprop vs Adam

Adam 可以看作是 RMSprop + Momentum + Bias Correction 的组合。在 CNN 图像分类中，Adam/AdamW 通常优于 RMSprop。但在 RNN 和强化学习中，RMSprop（或带 momentum 的 RMSprop）仍有价值。

### 3.5 在 CNN 项目中使用

```python
from cnnlib.training.optimizer import createOptimizer

optimizer = createOptimizer(model, 'rmsprop', lr=0.001)
```

---

## 4. Adam

### 4.1 Adam 的核心思想

Adam（Adaptive Moment Estimation）结合了两种改进思路：

- **Momentum**：用梯度的指数移动平均（一阶矩）做平滑
- **自适应学习率**：用梯度平方的指数移动平均（二阶矩）调整每个参数的步长

结果：每个参数自动获得合适的学习率——梯度大、波动大的参数学习率自动变小；梯度小、稳定的参数学习率自动变大。

### 4.2 Adam 公式拆解

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

### 4.3 参数含义

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

### 4.4 Adam 的优点

- **开箱即用**：默认 `lr=0.001` 在大多数任务上表现良好
- **对初始学习率不敏感**：自适应机制自动调整
- **收敛快**：训练初期快速下降
- **适合稀疏梯度**：每个参数独立的学习率

### 4.5 常见问题

#### 问题 1：泛化不如 SGD

在一些大规模图像分类任务上，Adam 的最终精度略低于精心调参的 SGD+Momentum。但这差距正在缩小（AdamW 改善了这一问题）。

#### 问题 2：Weight Decay 不正确

Adam 中的 `weight_decay` 与 SGD 不同——它对自适应学习率后的参数做衰减，不是标准的 L2 正则化。这也是 AdamW 被提出的原因。

#### 问题 3：收敛后可能不稳定

Adam 在训练后期可能"抖动"，因为它给每个参数分配的学习率会随时间变化。配合学习率调度（如 CosineAnnealing）可以缓解。

### 4.6 何时用 Adam

- 快速实验和原型开发（首选）
- 小到中等规模的数据集
- 对超参数不敏感的场景
- Transformer / NLP 任务（Adam 是标配）

---

## 5. AdamW

### 5.1 Adam 的 Weight Decay 问题

在 Adam 中，`weight_decay` 参数是这样应用的：

- `Adam` 的做法：在梯度中加入 L2 正则化，后续自适应学习率会影响正则化项...

$$
g_t = \nabla L(w_t) + \lambda w_t
$$
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
$$
- `AdamW` 的做法：梯度不含正则化，权重衰减独立

$$
g_t = \nabla L(w_t)
$$

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
$$

$$
w_t = w_{t-1} - \eta\frac{\hat{m_t}}{\sqrt{\hat{v_t} + \epsilon}} - \lambda w_{t-1}
$$

**核心差异**：Adam 将 L2 正则化与自适应学习率耦合，导致正则化强度被学习率缩放。AdamW 将权重衰减解耦，直接对权重施加衰减。

### 5.2 AdamW 的优势

- **更强的正则化**：解耦后 weight_decay 效果更显著
- **泛化更好**：在 ImageNet 等大数据集上显著优于 Adam
- **更好的超参数解耦**：学习率和 weight_decay 可以独立调整

### 5.3 使用

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01    # 比 Adam 中的 weight_decay 更有效
)
```

**weight_decay 典型值**：AdamW 中常用 `0.01 ~ 0.1`（比 Adam 中的典型值大）。

### 5.4 Adam vs AdamW

| | Adam | AdamW |
| --- | --- | --- |
| 权重衰减方式 | 与梯度耦合（L2 正则化） | 解耦，直接衰减权重 |
| weight_decay 效果 | 较弱 | **较强** |
| 泛化能力 | 一般 | **更好** |
| 当代使用 | 逐渐被 AdamW 取代 | **首选** |

**建议**：新项目直接使用 AdamW。如果从 Adam 迁移，原先的 weight_decay 可能需要调大（因为解耦后效果变强）。

### 5.5 完整配置示例

```python
# ImageNet 风格训练配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300
)
```

---

## 6. 总结与选择建议

### 6.1 优化器演化路线

```
朴素 SGD
  └─ SGD + Momentum（引入动量，加速收敛）
      └─ SGD + Nesterov（"先走再看"，更智能的动量）
  └─ AdaGrad（参数自适应学习率，但学习率只降不升）
      └─ RMSprop（指数移动平均替代累加，解决 AdaGrad 缺陷）
          └─ Adam（RMSprop + Momentum + 偏差校正）
              └─ AdamW（解耦权重衰减，修复 Adam 正则化问题）
```

### 6.2 优化器选择速查

| 场景              | 推荐                                       |
| --------------- | ---------------------------------------- |
| 快速实验/原型         | **Adam** (lr=0.001)                      |
| CNN 图像分类        | **AdamW** (lr=0.001, weight_decay=1e-4)  |
| 追求最高精度          | **SGD+Momentum** (lr=0.01~0.1, momentum=0.9) |
| Transformer/大模型 | **AdamW**                                |
| RNN/LSTM        | **RMSprop** 或 **Adam**                   |
| 强化学习           | **RMSprop**                              |

### 6.3 CNN 项目支持的优化器

```python
from cnnlib.training.optimizer import createOptimizer

# Adam
optimizer = createOptimizer(model, 'adam', lr=0.001)

# AdamW
optimizer = createOptimizer(model, 'adamw', lr=0.001, weight_decay=1e-4)

# SGD
optimizer = createOptimizer(model, 'sgd', lr=0.01, momentum=0.9)

# RMSprop
optimizer = createOptimizer(model, 'rmsprop', lr=0.001)
```

### 6.4 核心要点

1. **SGD+Momentum** 仍是追求极限精度的首选，但需要仔细调参和配合学习率调度。
2. **Adam** 是快速实验和原型开发的最佳选择，开箱即用，收敛快。
3. **AdamW** 是现代默认推荐——修复了 Adam 的权重衰减问题，泛化更好，适合大多数 CNN 和 Transformer 任务。
4. **RMSprop** 在 RNN 和强化学习等非平稳优化场景中仍有独特价值。
5. 无论选择哪个优化器，配合合适的学习率调度（如 CosineAnnealingLR）都能显著提升效果。
