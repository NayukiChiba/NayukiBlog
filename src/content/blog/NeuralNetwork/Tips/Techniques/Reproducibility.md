---
title: 随机种子与实验可复现性
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 基础
description: 为什么你的实验结果每次都不一样？完整固定随机种子的方案及性能代价分析。
image: https://img.yumeko.site/file/blog/Reproducibility.png
status: published
---
## 1. 为什么不可复现是噩梦？

假设你在[[NeuralNetwork/Tips/Techniques/TrainingPipeline|训练流程]]中跑了一个实验，准确率 95%。第二天改了一行不相关的代码，准确率变成了 93%。是改出 Bug 了，还是随机性导致？

**没有固定随机种子，你永远分不清是代码变更还是随机波动。** 连[[NeuralNetwork/Tips/Techniques/CheckpointGuide|检查点]]都无法放心地比较和复用。

## 2. 完整的种子固定方案

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

## 3. cudnn 配置解释

| 配置 | 设为 True | 设为 False |
| --- | --- | --- |
| `deterministic` | cuDNN 只用确定性算法，可复现 | 允许非确定性算法，可能更快 |
| `benchmark` | cuDNN 自动选最快算法（取决于输入尺寸） | 使用默认算法，固定行为 |

**性能代价**：开启 `deterministic=True` 后，某些卷积操作速度可能降低 10-30%。对于生产环境，可以在调试结束后关闭。

## 4. 数据划分的种子

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

## 5. DataLoader 的 Worker 种子

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

## 6. Dropout 和随机增强的种子

即使用了相同的数据集，如果 Dropout（训练时）和随机数据增强的种子不同，结果也会不同。这就是为什么需要全局固定种子。

## 7. 可复现性的"级别"

| 级别 | 措施 | 确定性 |
| --- | --- | :---: |
| 基本 | `torch.manual_seed` | 同硬件、同 PyTorch 版本 |
| 良好 | + `cudnn.deterministic=True` | 同 GPU 架构 |
| 严格 | + `cudnn.benchmark=False` + worker seed | 同 GPU 型号 + 驱动版本 |
| 绝对 | 需要锁定全部环境（Docker + 固定依赖版本） | 完全一致 |

对于研究和开发，"良好"级别通常足够。"严格"和"绝对"级别主要用于调试数值精度问题。

## 8. 检查可复现性

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



