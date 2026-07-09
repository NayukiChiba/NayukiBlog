---
title: 检查点管理指南：保存、恢复和加载最佳模型
date: 2026-05-09
category: 神经网络/训练
tags:
  - 高级教程
  - 基础
  - 深度学习
description: 讲解 PyTorch 检查点应该保存什么、best 与 last 的区别、安全写入、恢复训练、部分加载权重和常见错误。
status: published
---

## 1. 快速定位：检查点解决什么问题

检查点用于保护训练进度。

它解决的问题：

- 训练中断后可以继续
- 保存验证集表现最好的模型
- 记录优化器和调度器状态
- 方便复现实验和对比模型

最少应该保存两类文件：

| 文件 | 用途 |
| --- | --- |
| `best_model.pth` | 验证指标最好的模型，用于最终评估或部署 |
| `last_checkpoint.pth` | 最近一次训练状态，用于恢复中断训练 |

## 2. 检查点应该保存什么

### 2.1 最小可恢复检查点

如果只想恢复训练，至少保存：

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
```

### 2.2 推荐保存内容

更完整的检查点建议包含：

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "scaler_state_dict": scaler.state_dict() if scaler else None,
    "best_metric": best_metric,
    "metrics": {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history,
    },
    "metadata": {
        "model_name": model_name,
        "dataset": dataset_name,
        "seed": seed,
    },
}
```

| 字段 | 是否必须 | 用途 |
| --- | --- | --- |
| `model_state_dict` | 必须 | 恢复模型参数 |
| `optimizer_state_dict` | 恢复训练时必须 | 恢复动量、Adam 状态等 |
| `scheduler_state_dict` | 有调度器时建议 | 恢复学习率进度 |
| `scaler_state_dict` | AMP 训练时建议 | 恢复混合精度缩放状态 |
| `epoch` | 必须 | 知道从哪一轮继续 |
| `best_metric` | 建议 | 继续保存最佳模型 |
| `metrics` | 建议 | 画图和复盘训练过程 |
| `metadata` | 建议 | 记录实验配置 |

## 3. best_model 和 last_checkpoint 的区别

### 3.1 best_model

`best_model.pth` 只在验证指标改善时更新。

用途：

- 最终测试
- 部署
- 和其他实验比较
- Early Stopping 后恢复最佳权重

### 3.2 last_checkpoint

`last_checkpoint.pth` 每个 epoch 都更新。

用途：

- 断点续训
- OOM、断电、进程崩溃后恢复
- 长时间训练任务保底

### 3.3 两者都要保存

只保存 `best_model.pth`，可能无法完整恢复训练，因为它通常不包含 optimizer 状态。

只保存 `last_checkpoint.pth`，最终可能拿到过拟合后的权重。

工程上建议两者都保存。

## 4. 保存检查点

### 4.1 保存最近检查点

```python
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> None:
    """保存可恢复训练的完整检查点。"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_metric": best_metric,
    }

    save_path = Path(checkpoint_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
```

### 4.2 保存最佳模型

```python
def save_best_model(
    best_model_path: str,
    model: nn.Module,
    epoch: int,
    metric: float,
) -> None:
    """保存验证集表现最好的模型。"""
    save_path = Path(best_model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
        },
        save_path,
    )
```

## 5. 安全写入：避免 checkpoint 写坏

如果进程在 `torch.save` 写入途中崩溃，文件可能损坏。

更稳的做法是先写临时文件，再原子替换：

```python
import os
from pathlib import Path
from typing import Any, Dict

import torch


def atomic_save(data: Dict[str, Any], save_path: str) -> None:
    """先写临时文件，再替换目标文件，降低文件损坏风险。"""
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(data, temp_path)
    os.replace(temp_path, target_path)
```

`os.replace` 在 Windows 和 Linux 上都可以替换已有文件。

## 6. 恢复训练

### 6.1 加载完整检查点

```python
from typing import Any, Optional

import torch
import torch.nn as nn


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> int:
    """加载检查点，返回下一轮训练的 epoch。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    return start_epoch
```

### 6.2 训练循环中使用

```python
start_epoch = 0

if resume_path is not None:
    start_epoch = load_checkpoint(
        checkpoint_path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
```

## 7. 加载最佳模型做测试

最终测试通常加载 `best_model.pth`：

```python
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

如果你只保存了 `model.state_dict()`：

```python
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
```

注意：测试前必须调用 `model.eval()`，否则 Dropout 和 BatchNorm 行为会不正确。

## 8. 模型结构变化后的部分加载

如果模型结构改了，例如分类头类别数变化，旧 checkpoint 可能无法完整加载。

可以只加载形状匹配的参数：

```python
def load_partial_state_dict(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """只加载名称和形状都匹配的参数。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    current_state = model.state_dict()

    filtered_state = {
        name: value
        for name, value in model_state.items()
        if name in current_state and current_state[name].shape == value.shape
    }

    current_state.update(filtered_state)
    model.load_state_dict(current_state)
```

这种方式适合微调和迁移学习，但不适合严格复现实验。

## 9. 检查点目录建议

推荐目录：

```text
output/
└── checkpoints/
    ├── best_model.pth
    ├── last_checkpoint.pth
    └── history.json
```

如果有多个实验，可以按实验名分目录：

```text
output/
└── checkpoints/
    ├── resnet18_cifar10_seed42/
    │   ├── best_model.pth
    │   └── last_checkpoint.pth
    └── vgg16_cifar10_seed42/
        ├── best_model.pth
        └── last_checkpoint.pth
```

路径应通过配置集中管理，避免散落在训练脚本里。

## 10. 常见错误

### 10.1 只保存 model.state_dict

只保存模型参数可以用于推理，但不能完整恢复训练。

恢复训练还需要 optimizer、scheduler、epoch 等状态。

### 10.2 恢复后 epoch 从错的位置开始

如果 checkpoint 保存的是当前 epoch，恢复时通常要从下一轮开始：

```python
start_epoch = checkpoint["epoch"] + 1
```

### 10.3 忘记保存 optimizer state

Adam、AdamW、SGD with momentum 都有内部状态。

如果恢复训练时不加载 optimizer state，优化轨迹会突然改变。

### 10.4 Early Stopping 后使用最后模型

Early Stopping 停止时，最后一个 epoch 通常不是最佳权重。

应该加载 `best_model.pth` 再做最终测试。

## 11. 总结

- `best_model.pth` 保存最佳权重，用于最终评估和部署
- `last_checkpoint.pth` 保存最近状态，用于断点续训
- 恢复训练要保存并加载 optimizer、scheduler、epoch
- 写 checkpoint 时建议使用临时文件加原子替换
- 模型结构变化后，可以按名称和形状部分加载
