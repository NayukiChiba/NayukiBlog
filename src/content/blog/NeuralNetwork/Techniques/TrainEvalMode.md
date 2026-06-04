---
title: 训练/评估模式切换的正确姿势
date: 2026-05-09
category: NeuralNetwork/Techniques
tags:
  - 基础
  - 深度学习
description: model.train() 和 model.eval() 到底改变了什么？为什么这是 Bug 高发区？
image: https://img.yumeko.site/file/blog/TrainEvalMode.png
status: published
---

## 1. 两种模式的行为差异

| 层 | `model.train()` | `model.eval()` |
| --- | :---: | :---: |
| Dropout | 随机丢弃神经元 | 不丢弃 |
| BatchNorm | 使用当前 batch 统计量 | 使用全局移动平均 |
| BatchNorm running stats | **更新** | 冻结 |

## 2. 最佳实践

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

## 3. 易错场景

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

## 4. 训练中临时评估

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

## 5. 检查当前模式

```python
# 检查模型是 train 还是 eval
print(model.training)  # True for train, False for eval
```

这个属性由 `model.train()` 和 `model.eval()` 管理。

## 6. 自定义层的模式切换

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



