---
title: Early Stopping 指南：防止过拟合的早停机制
date: 2026-05-09
category: 神经网络/训练
tags:
  - 高级教程
  - 基础
  - 深度学习
description: 讲解 Early Stopping 的适用场景、patience、min_delta、mode、最佳模型保存、多条件早停和常见错误。
status: published
---

## 1. 快速定位：Early Stopping 解决什么问题

Early Stopping 用来防止过拟合。

典型现象：

- 训练损失继续下降
- 验证损失不再下降，甚至开始上升
- 训练准确率继续上升
- 验证准确率长期停滞或下降

这说明模型正在继续记忆训练集，但泛化能力没有变好。

![LossCurve.png](https://img.yumeko.site/file/blog/articles/1780581571699.webp)

## 2. Early Stopping 的核心参数

### 2.1 monitor：监控什么指标

常见监控指标：

| 任务 | 推荐监控 |
| --- | --- |
| 分类任务 | `val_loss` 或 `val_acc` |
| 回归任务 | `val_loss` |
| 类别不平衡任务 | `val_f1`、`val_auc` |

优先监控验证集指标，不要监控训练集指标。

### 2.2 mode：指标越小越好还是越大越好

| `mode` | 适用指标 |
| --- | --- |
| `"min"` | `val_loss`、`val_error` |
| `"max"` | `val_acc`、`val_f1`、`val_auc` |

### 2.3 patience：允许多久不改善

`patience` 表示验证指标连续多少个 epoch 没有改善后停止。

| `patience` | 效果 |
| --- | --- |
| `2-3` | 很敏感，容易过早停止 |
| `5-10` | 小任务常用 |
| `10-20` | 大任务或波动较大的任务 |
| `50+` | 通常过大，接近没开 |

### 2.4 min_delta：多大变化才算改善

`min_delta` 用来过滤验证集噪声。

例如监控 `val_loss`：

```python
improved = val_loss < best_val_loss - min_delta
```

如果 `min_delta=1e-3`，验证损失至少下降 `0.001` 才算有效改善。

## 3. 基础实现

### 3.1 EarlyStopping 类

```python
class EarlyStopping:
    """根据验证指标判断是否提前停止训练。"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-3, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0

        if mode == "min":
            self.best_metric = float("inf")
        elif mode == "max":
            self.best_metric = float("-inf")
        else:
            raise ValueError("mode 必须是 'min' 或 'max'")

    def is_improved(self, metric: float) -> bool:
        """判断当前指标是否有效改善。"""
        if self.mode == "min":
            return metric < self.best_metric - self.min_delta

        return metric > self.best_metric + self.min_delta

    def step(self, metric: float) -> bool:
        """返回 True 表示应该停止训练。"""
        if self.is_improved(metric):
            self.best_metric = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
```

### 3.2 训练循环中使用

```python
early_stop = EarlyStopping(patience=10, min_delta=1e-3, mode="min")

for epoch in range(total_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    if early_stop.step(val_loss):
        print(f"Early Stopping 在 epoch {epoch} 触发")
        break
```

## 4. 必须配合保存最佳模型

### 4.1 为什么不能直接用最后一轮模型

Early Stopping 触发时，当前模型通常不是最佳模型。

例如：

```text
epoch 10: val_loss = 0.31  最佳
epoch 11: val_loss = 0.33
epoch 12: val_loss = 0.35
epoch 13: val_loss = 0.36  触发 early stopping
```

最终评估应该使用 epoch 10 的权重，而不是 epoch 13 的权重。

### 4.2 保存最佳模型

```python
best_val_loss = float("inf")

for epoch in range(total_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

    if early_stop.step(val_loss):
        break

model.load_state_dict(torch.load("best_model.pth", map_location=device))
```

更完整的保存和恢复方式见 [[NeuralNetwork/Training/Checkpointing|检查点管理指南]]。

## 5. ReduceLROnPlateau 和 Early Stopping 的配合

`ReduceLROnPlateau` 和 Early Stopping 都监控验证集指标，但目的不同：

| 机制 | 目的 |
| --- | --- |
| `ReduceLROnPlateau` | 降低学习率，继续训练 |
| `Early Stopping` | 停止训练 |

常见配置：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)

early_stop = EarlyStopping(
    patience=10,
    min_delta=1e-3,
    mode="min",
)
```

建议：`ReduceLROnPlateau.patience` 小于 `EarlyStopping.patience`。先给调度器降学习率的机会，再决定是否停止。

## 6. 多条件 Early Stopping

单独监控 `val_loss` 通常已经够用。

如果任务容易过拟合，可以加额外条件：

```python
def should_stop_by_gap(train_acc: float, val_acc: float, max_gap: float = 10.0) -> bool:
    """训练准确率和验证准确率差距过大时停止。"""
    return train_acc - val_acc > max_gap
```

常见条件：

| 条件 | 含义 |
| --- | --- |
| 验证指标不再改善 | 模型可能到达当前配置上限 |
| 训练好、验证差 | 过拟合 |
| train-val acc 差距过大 | 泛化能力明显变差 |

不要一开始就写太复杂的早停规则。先用 `val_loss + patience` 跑通，再按任务补充。

## 7. 参数选择建议

### 7.1 小数据集

```python
early_stop = EarlyStopping(patience=5, min_delta=1e-3, mode="min")
```

小数据集过拟合快，patience 可以小一些。

### 7.2 中大型数据集

```python
early_stop = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
```

训练曲线波动更明显，patience 不宜太小。

### 7.3 指标波动很大

如果验证指标抖动明显：

- 增大 `patience`
- 增大验证集规模
- 使用更平滑的监控指标
- 保存最佳模型，不要依赖最后一轮

## 8. 常见错误

### 8.1 监控训练集指标

错误：

```python
early_stop.step(train_loss)
```

训练集 loss 通常会持续下降，无法判断泛化能力。

应该监控验证集：

```python
early_stop.step(val_loss)
```

### 8.2 mode 写反

监控 `val_loss` 时应该用 `mode="min"`。

监控 `val_acc` 时应该用 `mode="max"`。

### 8.3 触发早停后不加载最佳模型

早停只是告诉你该停了，不代表当前权重最好。

停止后应加载最佳模型，再做测试集评估。

## 9. 总结

- Early Stopping 用于防止过拟合
- 核心参数是 `monitor`、`mode`、`patience`、`min_delta`
- 监控验证集指标，不要监控训练集指标
- 必须配合保存最佳模型
- 和 `ReduceLROnPlateau` 配合时，调度器 patience 应更小
