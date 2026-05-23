---
title: SGD + Momentum：经典的优化选择
date: 2026-05-23
category: NeuralNetwork/Tips/Optimizers
tags:
  - SGD
  - 优化器
description: 从朴素 SGD 到 Momentum 到 Nesterov，理解随机梯度下降的演进与最佳实践。
image: https://img.yumeko.site/file/blog/Optimizer.png
status: published
---

## 1. 朴素 SGD

$$
w_{t+1} = w_t - \eta \cdot \nabla \mathcal{L}(w_t)
$$

**问题**：SGD 对每个参数使用相同的学习率，如果损失曲面在某个方向陡峭、另一个方向平缓（"峡谷"形状），优化会来回震荡，收敛很慢。关于各类优化器的完整对比，参见 [[NeuralNetwork/Tips/Optimizers/OptimizerOverview|优化器概述]]。

## 2. SGD + Momentum

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

## 3. Nesterov 加速梯度（NAG）

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

## 4. SGD vs Adam

| | SGD + Momentum | Adam |
| --- | :---: | :---: |
| 收敛速度 | 较慢 | **较快** |
| 泛化能力 | **通常更好** | 有时略差 |
| 调参难度 | 需要仔细调 | 开箱即用 |
| 适合场景 | 大规模图像分类，追求 SOTA | 快速实验和小任务 |

**为什么 SGD 泛化更好？** 一种解释是：自适应优化器（如 [[NeuralNetwork/Tips/Optimizers/Adam|Adam]]）找到的极小值比较"尖锐"，SGD 找到的更"平坦"——平坦的极小值泛化更好。

## 5. 学习率调参

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

## 6. Weight Decay

SGD 中 weight_decay 等效于 L2 正则化：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
```

对于 SGD，weight_decay 在梯度上加上 $w \cdot \text{weight\_decay}$ 项。
