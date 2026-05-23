---
title: 检查点管理最佳实践
date: 2026-05-09
category: NeuralNetwork/Tips/Techniques
tags:
  - 检查点
  - 工程实践
description: 训练中断了怎么办？怎样安全地保存和恢复模型？详解检查点管理的最佳实践。
image: https://img.yumeko.site/file/articles/NNTrainingTips/Checkpoint.png
status: draft
---

## 1. 应该保存什么？

一个完整的检查点应包含恢复训练所需的一切：

```python
checkpoint = {
    'epoch': currentEpoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # 如果有
    'bestMetric': bestValAcc,
    'metrics': {
        'trainLoss': trainLosses,
        'valLoss': valLosses,
        'trainAcc': trainAccs,
        'valAcc': valAccs,
    },
    'metadata': {
        'modelName': 'resnet18',
        'dataset': 'cifar10',
        'timestamp': '2026-05-23 14:30:00',
    }
}

torch.save(checkpoint, 'checkpoint.pth')
```

**必须保存**：`model_state_dict`、`optimizer_state_dict`、`epoch`。
**建议保存**：`scheduler_state_dict`、`metrics`、超参数 metadata。

## 2. 两类检查点策略

| 类型 | 文件名 | 更新时机 | 用途 |
| --- | --- | --- | --- |
| 最佳模型 | `best_model.pth` | 验证指标创新高/新低时（与[[NeuralNetwork/Tips/Techniques/EarlyStoppingGuide|早停]]配合） | 最终部署、评估 |
| 最近模型 | `last_model.pth` | 每个 epoch 结束时 | 恢复中断的训练 |

```python
for epoch in range(epochs):
    trainLoss, trainAcc = trainOneEpoch(...)
    valLoss, valAcc = validate(...)

    # 保存最佳模型
    if valAcc > bestValAcc:
        bestValAcc = valAcc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'valAcc': valAcc,
        }, 'best_model.pth')

    # 保存最近模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainLoss': trainLoss, 'valLoss': valLoss,
    }, 'last_model.pth')
```

## 3. 安全写入（防止文件损坏）

```python
# 先写临时文件，再原子替换
tempPath = savePath + '.tmp'
torch.save(checkpoint, tempPath)
os.replace(tempPath, savePath)  # 原子操作（Windows 上也支持）
```

防止在写入过程中进程崩溃导致文件损坏。

## 4. 恢复[[NeuralNetwork/Tips/Techniques/TrainingPipeline|训练流程]]

```python
def resumeTraining(checkpointPath, model, optimizer, scheduler=None):
    checkpoint = torch.load(checkpointPath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    startEpoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始

    print(f"从 Epoch {startEpoch} 恢复训练")
    print(f"已记录最佳指标: {checkpoint.get('bestMetric', 'N/A')}")

    return startEpoch

# 使用
startEpoch = 1
if args.resume:
    startEpoch = resumeTraining(args.checkpoint, model, optimizer, scheduler)

for epoch in range(startEpoch, epochs + 1):
    # ... 正常训练
```

## 5. 检查点目录组织

![TODO: 推荐输出目录结构树状图，outputs/模型名/数据集名/下含checkpoints/logs/tensorboard子目录]

## 6. 版本兼容性注意事项

如果模型代码发生了变化（比如改了层的名称），旧的 checkpoint 可能无法加载。建议：

```python
checkpoint = torch.load(path)
modelState = checkpoint['model_state_dict']

# 过滤掉不匹配的键（在模型结构微调后很有用）
modelDict = model.state_dict()
filteredState = {k: v for k, v in modelState.items()
                 if k in modelDict and modelDict[k].shape == v.shape}

model.load_state_dict(filteredState, strict=False)
```



