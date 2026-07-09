---
title: BatchNorm 指南：公式、用法、训练推理差异与常见坑
date: 2026-05-09
category: 神经网络/训练
tags:
  - 基础
  - 深度学习
  - 高级教程
description: 清晰讲解 BatchNorm 的作用、数学公式、PyTorch 用法、train/eval 差异、小 batch 替代方案、冻结 BN 和常见排查方法。
image: https://img.yumeko.site/file/blog/cover/1780581701550.webp
status: published
---

## 1. 快速定位：BatchNorm 解决什么问题

Batch Normalization，简称 `BatchNorm` 或 `BN`，用于稳定每一层的输入分布。

它最常见的作用：

- 让训练更稳定
- 允许使用更大的学习率
- 减少对 [[NeuralNetwork/Training/WeightInitialization|权重初始化]] 的敏感性
- 在 CNN 中加速收敛
- 带来轻微正则化效果

最常见的使用位置：

```text
Conv2d -> BatchNorm2d -> ReLU
Linear -> BatchNorm1d -> ReLU
```

如果你只想写一个标准 CNN 卷积块，可以直接用这个顺序。

## 2. BatchNorm 公式：归一化再缩放平移

### 2.1 计算 batch 均值

对一个 mini-batch 中的某个通道或特征维度，先计算均值：

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

其中 $m$ 是参与统计的元素数量。

### 2.2 计算 batch 方差

再计算方差：

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

### 2.3 标准化

把输入变成均值接近 0、方差接近 1 的形式：

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$\epsilon$ 是防止除零的小常数，PyTorch 默认通常是 `1e-5`。

### 2.4 可学习的缩放和平移

最后用可学习参数恢复表达能力：

$$
y_i=\gamma\hat{x}_i+\beta
$$

其中：

- $\gamma$ 是缩放参数，对应 PyTorch 的 `weight`
- $\beta$ 是平移参数，对应 PyTorch 的 `bias`

这一步很重要。BatchNorm 不是强制所有层都只能输出标准正态分布，而是先标准化，再让网络自己学习合适的尺度和偏移。

![BatchNorm.png](https://img.yumeko.site/file/blog/articles/1780581356287.webp)

## 3. BatchNorm1d、BatchNorm2d、BatchNorm3d 怎么选

### 3.1 选择规则

| 输入类型 | 常用层 | 输入形状示例 |
| --- | --- | --- |
| MLP / 一维特征 | `nn.BatchNorm1d` | `(N, C)` |
| 一维序列特征 | `nn.BatchNorm1d` | `(N, C, L)` |
| CNN 图像特征 | `nn.BatchNorm2d` | `(N, C, H, W)` |
| 3D CNN / 视频 / 体数据 | `nn.BatchNorm3d` | `(N, C, D, H, W)` |

### 3.2 BatchNorm2d 统计哪些维度

对 `BatchNorm2d` 来说，输入形状通常是：

```text
(batch_size, channels, height, width)
```

它会对每个通道分别统计均值和方差，统计范围包括：

```text
batch_size、height、width
```

也就是说，每个通道有一组自己的 $\mu_B$、$\sigma_B^2$、$\gamma$、$\beta$。

## 4. 推荐放置位置：Conv -> BN -> ReLU

### 4.1 CNN 中的标准顺序

现代 CNN 中最常见的顺序是：

```text
Conv2d -> BatchNorm2d -> ReLU
```

对应代码：

```python
block = nn.Sequential(
    nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(outChannels),
    nn.ReLU(inplace=True),
)
```

注意：如果 `Conv2d` 后面立刻接 `BatchNorm2d`，卷积层通常可以设置 `bias=False`。因为 BN 自己有 $\beta$，卷积 bias 往往是冗余的。

### 4.2 MLP 中的标准顺序

MLP 中常见顺序是：

```text
Linear -> BatchNorm1d -> ReLU
```

对应代码：

```python
block = nn.Sequential(
    nn.Linear(inFeatures, hiddenFeatures, bias=False),
    nn.BatchNorm1d(hiddenFeatures),
    nn.ReLU(inplace=True),
)
```

### 4.3 为什么通常放在激活函数前

BN 的目标是稳定激活函数的输入。

如果先做 ReLU，再做 BN，BN 看到的是已经被截断的分布：负值都变成了 0。这样会改变 BN 的统计对象。

所以工程上优先使用：

```text
Conv / Linear -> BN -> Activation
```

## 5. 训练模式和推理模式的区别

### 5.1 train 模式

在 `model.train()` 下，BatchNorm 会：

- 使用当前 batch 的均值和方差做归一化
- 更新 `running_mean`
- 更新 `running_var`

```python
model.train()
output = model(trainImages)
```

### 5.2 eval 模式

在 `model.eval()` 下，BatchNorm 会：

- 不再使用当前 batch 的统计量
- 使用训练期间累计的 `running_mean`
- 使用训练期间累计的 `running_var`
- 不更新 running statistics

```python
model.eval()
with torch.no_grad():
    output = model(testImages)
```

验证和推理时必须调用 `model.eval()`。否则 BN 会继续使用当前 batch 的统计量，结果可能不稳定。

### 5.3 train/eval 行为对照

| 模式 | 归一化使用的统计量 | 是否更新 running stats |
| --- | --- | --- |
| `model.train()` | 当前 batch 的均值和方差 | 是 |
| `model.eval()` | `running_mean` 和 `running_var` | 否 |

### 5.4 PyTorch momentum 公式

PyTorch 中 running mean 的更新方式是：

$$
\text{running\_mean}_{new}
=(1-\text{momentum})\cdot \text{running\_mean}_{old}
+\text{momentum}\cdot \text{batch\_mean}
$$

`running_var` 同理。

默认 `momentum=0.1`，表示每次用 10% 的当前 batch 统计量更新 running statistics。

## 6. BatchNorm 的重要参数

### 6.1 num_features

`num_features` 是通道数或特征数：

```python
nn.BatchNorm1d(num_features=128)
nn.BatchNorm2d(num_features=64)
```

对 `BatchNorm2d(64)` 来说，输入的通道数必须是 64。

### 6.2 eps

`eps` 防止除以 0：

```python
nn.BatchNorm2d(64, eps=1e-5)
```

一般不需要修改。只有在混合精度训练、数值特别不稳定时才考虑调大。

### 6.3 momentum

`momentum` 控制 running statistics 更新速度：

```python
nn.BatchNorm2d(64, momentum=0.1)
```

- 值越大：越依赖最近 batch
- 值越小：running statistics 更新更慢、更平滑

### 6.4 affine

`affine=True` 时，BN 有可学习的 $\gamma$ 和 $\beta$：

```python
nn.BatchNorm2d(64, affine=True)
```

默认就是 `True`。大多数情况不要关。

### 6.5 track_running_stats

`track_running_stats=True` 时，BN 会维护 `running_mean` 和 `running_var`。

如果设为 `False`，训练和推理都会使用当前 batch 的统计量：

```python
nn.BatchNorm2d(64, track_running_stats=False)
```

这个选项不适合普通推理场景，除非你非常明确地需要 batch 依赖行为。

## 7. PyTorch 代码示例

### 7.1 构建卷积块

```python
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """标准 Conv-BN-ReLU 卷积块。"""

    def __init__(self, inChannels: int, outChannels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputTensor: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.block(inputTensor)
```

### 7.2 查看 BN 参数

```python
bn = nn.BatchNorm2d(64)

print(bn.weight.shape)        # gamma，形状为 (64,)
print(bn.bias.shape)          # beta，形状为 (64,)
print(bn.running_mean.shape)  # running mean，形状为 (64,)
print(bn.running_var.shape)   # running var，形状为 (64,)
```

### 7.3 验证时正确切换模式

```python
def validate(model: nn.Module, dataLoader, device: torch.device) -> float:
    """验证模型，确保 BatchNorm 和 Dropout 都切换到 eval 模式。"""
    model.eval()
    correctCount = 0
    totalCount = 0

    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correctCount += (predictions == labels).sum().item()
            totalCount += labels.size(0)

    return correctCount / totalCount
```

## 8. 小 batch 场景：什么时候别用 BatchNorm

### 8.1 BatchNorm 为什么怕小 batch

BatchNorm 依赖 batch 统计量。

当 `batch_size` 很小时，例如 1、2、4，当前 batch 的均值和方差噪声很大。BN 可能让训练变得不稳定。

典型表现：

- 训练 loss 波动很大
- 验证结果不稳定
- batch size 改变后精度明显变化

### 8.2 替代方案

| 方法 | 依赖 batch 维度 | 适用场景 |
| --- | --- | --- |
| `BatchNorm` | 是 | CNN，batch size 足够大 |
| `LayerNorm` | 否 | Transformer、RNN、MLP |
| `GroupNorm` | 否 | 小 batch CNN |
| `InstanceNorm` | 否 | 风格迁移、图像生成 |

小 batch CNN 中常用 `GroupNorm` 替代：

```python
# 原来
nn.BatchNorm2d(128)

# 小 batch 可替换为
nn.GroupNorm(num_groups=32, num_channels=128)
```

### 8.3 简单选择规则

- CNN 且 batch size 较大：用 `BatchNorm2d`
- CNN 但 batch size 很小：用 `GroupNorm`
- Transformer / RNN：用 `LayerNorm`
- 图像风格迁移：用 `InstanceNorm`

## 9. 冻结 BatchNorm：迁移学习常用

### 9.1 只冻结 running statistics

迁移学习时，如果新数据集很小，BN 的 running statistics 可能被少量数据带偏。可以让 BN 保持 eval 模式：

```python
def freezeBatchNormStats(model: nn.Module) -> None:
    """冻结 BatchNorm 的 running statistics。"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
```

这样做不会冻结 $\gamma$ 和 $\beta$，它们仍然可以训练。

### 9.2 完全冻结 BatchNorm

如果希望 BN 的 running statistics 和 affine 参数都不更新：

```python
def freezeBatchNorm(model: nn.Module) -> None:
    """完全冻结 BatchNorm。"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
```

### 9.3 注意 train() 会重新打开 BN

如果训练循环每个 epoch 都调用：

```python
model.train()
```

那么前面单独设置的 `module.eval()` 会被覆盖。需要在 `model.train()` 后再次冻结 BN：

```python
model.train()
freezeBatchNormStats(model)
```

## 10. BatchNorm 与 Dropout

### 10.1 推荐顺序

同时使用 BN 和 Dropout 时，常见顺序是：

```text
Conv / Linear -> BN -> Activation -> Dropout
```

例如：

```python
block = nn.Sequential(
    nn.Linear(inFeatures, hiddenFeatures, bias=False),
    nn.BatchNorm1d(hiddenFeatures),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
)
```

### 10.2 train/eval 必须一致切换

BN 和 Dropout 都受 `model.train()` / `model.eval()` 影响：

| 组件 | `model.train()` | `model.eval()` |
| --- | --- | --- |
| BatchNorm | 使用 batch 统计量，并更新 running stats | 使用 running stats |
| Dropout | 随机丢弃激活值 | 关闭随机丢弃 |

验证和推理时忘记 `model.eval()`，BN 和 Dropout 都会出错。

更多 Dropout 内容见 [[NeuralNetwork/Training/Dropout|Dropout 详解]]。

## 11. 常见问题排查

### 11.1 验证集准确率波动很大

优先检查：

- 验证前是否调用了 `model.eval()`
- batch size 是否太小
- 是否在迁移学习中错误更新了 BN running stats

### 11.2 batch size 从 32 改成 1 后效果明显变差

这是 BN 的常见问题。

处理方式：

- 推理时确保 `model.eval()`
- 训练时尽量增大 batch size
- 小 batch 训练改用 `GroupNorm`

### 11.3 迁移学习越训越差

如果目标数据集很小，BN 的 running statistics 可能被新数据集带偏。

可以尝试：

- 冻结 BN running statistics
- 降低学习率
- 只训练分类头

### 11.4 Conv 后面接 BN 还要 bias 吗

通常不需要。

因为：

$$
\text{BN}(Wx+b)
$$

会减去 batch 均值，卷积 bias 的影响通常会被抵消，而 BN 自己还有可学习的 $\beta$。

所以常见写法是：

```python
nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(outChannels)
```

## 12. 总结

### 12.1 记住这几条

- CNN 中常用 `Conv -> BN -> ReLU`
- 训练时 BN 使用当前 batch 统计量
- 推理时 BN 使用 `running_mean` 和 `running_var`
- 验证和推理必须调用 `model.eval()`
- 小 batch CNN 优先考虑 `GroupNorm`
- 迁移学习数据很少时，考虑冻结 BN

### 12.2 最容易犯的错

- 验证时忘记 `model.eval()`
- batch size 很小还强行使用 BN
- 冻结 BN 后又被 `model.train()` 重新打开
- Conv 后接 BN 仍保留不必要的 bias

