---
title: 梯度裁剪指南：防止梯度爆炸和 NaN
date: 2026-05-09
category: 神经网络/训练
tags:
  - 高级教程
  - 深度学习
description: 讲解梯度裁剪的作用、数学公式、clip_grad_norm_ 与 clip_grad_value_ 的区别、使用位置、阈值选择、梯度范数监控和常见错误。
status: published
---

## 1. 快速定位：什么时候需要梯度裁剪

梯度裁剪用于防止梯度爆炸。

如果训练中出现这些现象，优先考虑加梯度裁剪：

- loss 突然变成 NaN
- loss 剧烈震荡
- 梯度范数突然变得非常大
- RNN / LSTM / GRU 训练不稳定
- 使用较大学习率时训练容易崩

最常用写法：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

位置必须在 `loss.backward()` 之后、`optimizer.step()` 之前。

## 2. 梯度爆炸是什么

模型参数更新公式可以写成：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

其中 $\eta$ 是学习率，$g_t$ 是梯度。

如果梯度 $g_t$ 非常大，即使学习率不大，参数更新量也会非常大。严重时参数会变成极端值，后续计算产生 NaN。

RNN 类模型更容易出现梯度爆炸，因为反向传播需要沿时间步反复相乘，梯度可能在连乘中被放大。

## 3. clip_grad_norm_：按整体范数裁剪

### 3.1 数学定义

`clip_grad_norm_` 会先计算所有参数梯度的整体范数：

$$
\lVert g \rVert_2 =
\sqrt{\sum_i g_i^2}
$$

如果梯度范数超过阈值 $c$，就按比例缩小：

$$
g' = g \cdot \frac{c}{\lVert g \rVert_2}
$$

如果梯度范数没有超过阈值，则不修改：

$$
g'=g
$$

合在一起可以写成：

$$
g'=g\cdot\min\left(1,\frac{c}{\lVert g \rVert_2}\right)
$$

核心点：它保留梯度方向，只限制梯度长度。

### 3.2 PyTorch 用法

```python
total_grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

返回值 `total_grad_norm` 是裁剪前的总梯度范数，可以用于日志监控。

## 4. clip_grad_value_：按单个元素裁剪

### 4.1 数学定义

`clip_grad_value_` 会把每个梯度元素限制在固定范围内：

$$
g_i'=\max(-c,\min(g_i,c))
$$

它不关心整体梯度方向，只限制每个元素的取值。

### 4.2 PyTorch 用法

```python
torch.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=1.0,
)
```

### 4.3 两者怎么选

| 方法 | 推荐程度 | 特点 |
| --- | --- | --- |
| `clip_grad_norm_` | 常用 | 保留整体方向，只限制步长 |
| `clip_grad_value_` | 较少用 | 简单粗暴，可能改变梯度方向 |

一般优先使用 `clip_grad_norm_`。

## 5. 在训练循环中的正确位置

### 5.1 标准训练循环

```python
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    # 梯度裁剪必须放在 backward 之后、step 之前
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

### 5.2 使用 AMP 时的位置

如果使用自动混合精度，需要先 `unscale_`，再裁剪：

```python
scaler.scale(loss).backward()

# 先还原梯度尺度，再裁剪
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

否则裁剪的是放大后的梯度，阈值含义会不正确。

## 6. max_norm 怎么选

| `max_norm` | 适用情况 |
| --- | --- |
| `0.5` | 训练非常不稳定，先保守压住 |
| `1.0` | 常用默认值，RNN/LSTM 常见 |
| `5.0` | 中等限制，CNN 中可作为安全网 |
| `10.0` | 只拦截极端梯度 |
| 不设置 | 完全不保护 |

建议：

- RNN / LSTM：从 `1.0` 开始
- Transformer：常见取值 `1.0`
- CNN：一般可不开，出现 NaN 或震荡时再加
- 如果裁剪后训练明显变慢，可以逐步放宽到 `5.0`

## 7. 如何监控梯度范数

### 7.1 使用返回值

`clip_grad_norm_` 会返回裁剪前的范数：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

print(f"grad_norm={grad_norm:.4f}")
```

### 7.2 手动计算梯度范数

```python
def get_grad_norm(model: nn.Module) -> float:
    """计算模型所有参数的 L2 梯度范数。"""
    total_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        param_norm = parameter.grad.detach().data.norm(2).item()
        total_norm += param_norm ** 2

    return total_norm ** 0.5
```

如果梯度范数长期远大于 `max_norm`，说明训练本身可能不稳定。除了裁剪，还要检查学习率、输入归一化、损失函数和初始化。

## 8. 常见错误

### 8.1 放在 optimizer.step() 之后

错误写法：

```python
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

这时参数已经更新完了，裁剪没有意义。

### 8.2 忘记 optimizer.zero_grad()

如果梯度没有清零，多个 batch 的梯度会累积，梯度范数可能异常变大。

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 8.3 以为梯度裁剪能解决梯度消失

梯度裁剪只能限制梯度上限，不能放大接近 0 的梯度。

| 问题 | 梯度裁剪是否有效 |
| --- | --- |
| 梯度爆炸 | 有效 |
| 梯度消失 | 无效 |

梯度消失需要从激活函数、初始化、残差连接、归一化层等角度处理。

## 9. 总结

- 梯度裁剪主要解决梯度爆炸
- 常用 `clip_grad_norm_`，少用 `clip_grad_value_`
- 位置固定在 `loss.backward()` 后、`optimizer.step()` 前
- AMP 训练中先 `scaler.unscale_(optimizer)`，再裁剪
- `max_norm=1.0` 是常见起点
