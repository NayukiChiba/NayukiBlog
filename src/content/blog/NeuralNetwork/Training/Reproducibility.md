---
title: 实验可复现性：随机种子与模式切换
date: 2026-05-09
category: NeuralNetwork/Training
tags:
  - 基础
  - 深度学习
description: 深入探讨深度学习实验可复现性的两个关键保障：随机种子固定方案（含 cuDNN 配置、DataLoader 种子、可复现性级别）与训练/评估模式切换（model.train/eval 行为差异、最佳实践与易错场景）。
image: https://img.yumeko.site/file/blog/cover/1780581864650.webp
status: published
---

在深度学习实验中，可复现性（Reproducibility）是确保研究结论可靠的核心前提。两个最容易被忽视但至关重要的环节是：**随机种子的固定**和**训练/评估模式的正确切换**。前者保证相同的代码和数据在每次运行时产生相同的结果；后者保证模型在训练和推理时行为符合预期，Dropout 和 BatchNorm 等层的状态不会悄悄污染实验结果。本文将从这两个方面，提供一套完整的实验可复现性保障方案。

## 一、随机种子固定方案

### 1. 为什么不可复现是噩梦？

假设你在[[NeuralNetwork/Training/TrainingPipeline|训练流程]]中跑了一个实验，准确率 95%。第二天改了一行不相关的代码，准确率变成了 93%。是改出 Bug 了，还是随机性导致？

**没有固定随机种子，你永远分不清是代码变更还是随机波动。** 连[[NeuralNetwork/Training/TrainingStability|检查点]]都无法放心地比较和复用。

### 2. 完整的种子固定方案

```python
import random
import numpy as np
import torch

def setSeed(seed=42):
    random.seed(seed)                     # Python 内置 random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU（单卡）
    torch.cuda.manual_seed_all(seed)      # 所有 GPU（多卡）

    # 强制确定性算法（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 3. cudnn 配置解释

| 配置 | 设为 True | 设为 False |
| --- | --- | --- |
| `deterministic` | cuDNN 只用确定性算法，可复现 | 允许非确定性算法，可能更快 |
| `benchmark` | cuDNN 自动选最快算法（取决于输入尺寸） | 使用默认算法，固定行为 |

**性能代价**：开启 `deterministic=True` 后，某些卷积操作速度可能降低 10-30%。对于生产环境，可以在调试结束后关闭。

### 4. 数据划分的种子

```python
# 训练/验证划分也需要固定种子
generator = torch.Generator().manual_seed(42)
trainSet, valSet = random_split(
    fullDataset,
    [trainLen, valLen],
    generator=generator
)
```

确保每次划分的训练集/验证集完全一致，否则比较不同实验时会有偏差。

### 5. DataLoader 的 Worker 种子

```python
def workerInitFn(workerId):
    # 每个 worker 用不同的种子，但基于全局种子确定性地派生
    workerSeed = torch.initial_seed() % 2**32
    np.random.seed(workerSeed)

trainLoader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    worker_init_fn=workerInitFn
)
```

### 6. Dropout 和随机增强的种子

即使用了相同的数据集，如果 Dropout（训练时）和随机数据增强的种子不同，结果也会不同。这就是为什么需要全局固定种子。

### 7. 可复现性的"级别"

| 级别 | 措施 | 确定性 |
| --- | --- | :---: |
| 基本 | `torch.manual_seed` | 同硬件、同 PyTorch 版本 |
| 良好 | + `cudnn.deterministic=True` | 同 GPU 架构 |
| 严格 | + `cudnn.benchmark=False` + worker seed | 同 GPU 型号 + 驱动版本 |
| 绝对 | 需要锁定全部环境（Docker + 固定依赖版本） | 完全一致 |

对于研究和开发，"良好"级别通常足够。"严格"和"绝对"级别主要用于调试数值精度问题。

### 8. 检查可复现性

```python
# 跑两次训练，对比关键指标
def testReproducibility(trainFn, seed=42):
    setSeed(seed)
    metrics1 = trainFn()

    setSeed(seed)
    metrics2 = trainFn()

    for key in metrics1:
        if abs(metrics1[key] - metrics2[key]) > 1e-6:
            print(f"⚠️ 不可复现: {key}: {metrics1[key]} vs {metrics2[key]}")
```

## 二、训练/评估模式切换

### 1. 两种模式的行为差异

| 层 | `model.train()` | `model.eval()` |
| --- | :---: | :---: |
| Dropout | 随机丢弃神经元 | 不丢弃 |
| BatchNorm | 使用当前 batch 统计量 | 使用全局移动平均 |
| BatchNorm running stats | **更新** | 冻结 |

### 2. 最佳实践

建立清晰的模式：训练函数负责 `train()`，验证函数负责 `eval()`。

```python
# 训练阶段
def trainEpoch(model, dataloader, optimizer, criterion, device):
    model.train()                           # 1. 训练模式
    for images, labels in dataloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# 验证/测试阶段
@torch.no_grad()                            # 2. 不追踪梯度
def validateEpoch(model, dataloader, criterion, device):
    model.eval()                            # 1. 评估模式
    for images, labels in dataloader:
        output = model(images)
        # 不调用 backward/optimizer.step
```

**双重保险**：`model.eval()` + `torch.no_grad()` 同时使用。

- `model.eval()` → 负责 BN/Dropout 的正确行为
- `torch.no_grad()` → 关闭梯度追踪（节省显存、加速）

两者是独立的，不能互相替代。

### 3. 易错场景

```python
# ❌ 常见错误 1：验证时忘了 eval()
model.train()
with torch.no_grad():
    valOutput = model(valImages)  # Dropout 还在随机丢弃！

# ❌ 常见错误 2：推理时两者都忘了
output = model(testImages)  # Dropout 生效 + 梯度追踪，显存浪费

# ✅ 正确做法
model.eval()
with torch.no_grad():
    output = model(testImages)
```

### 4. 训练中临时评估

如果在训练过程中的某个环节需要评估（比如记录指标）：

```python
def trainEpoch(model, ...):
    model.train()
    # ... 训练步骤 ...

    # 临时评估
    wasTraining = model.training  # 记住当前模式
    model.eval()
    with torch.no_grad():
        valLoss = computeValLoss(model, valLoader, criterion, device)
    if wasTraining:
        model.train()  # 恢复训练模式
```

### 5. 检查当前模式

```python
# 检查模型是 train 还是 eval
print(model.training)  # True for train, False for eval
```

这个属性由 `model.train()` 和 `model.eval()` 管理。

### 6. 自定义层的模式切换

如果写了自定义 `nn.Module`，需要在 `train(mode)` 中正确响应模式切换：

```python
class CustomLayer(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        # 自定义层在模式切换时的特殊处理
        if not mode:
            # eval 模式：例如缓存某些计算结果
            self.cache = self.computeCache()
```

但对于绝大多数情况，只需要正确调用 `model.train()` / `model.eval()`，PyTorch 的默认行为就足够了。
