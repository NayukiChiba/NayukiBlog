---
title: 神经网络训练技巧
date: 2026-05-09
category: NeuralNetwork/Tips
tags:
  - PyTorch
  - 深度学习
  - 高级教程
description: 神经网络训练时候遇到各种问题如何解决？
image: https://img.yumeko.site/file/blog/NNTrainingTips.png
status: published
---
## 技巧一：权重初始化——Xavier 与 Kaiming

### 1.1 为什么初始化很重要？

想象你在一个巨大的山谷里找最低点（优化损失函数）。如果你的初始位置在山顶附近，梯度很大，几步就能走下山；但如果你的初始位置在几千公里外的平原上——也就是参数初始值离谱地大或离谱地小——你要走很久才能到山脚下，甚至可能永远走不到。

**数学上的问题**：

- **初始化太大** → 激活值进入饱和区（Tanh/Sigmoid 两端导数趋近于 0）→ 梯度消失 → 学不动
- **初始化太小** → 信号在网络传播中越来越微弱 → 深层几乎收不到信号 → 同样学不动
- **初始化全为 0** → 所有神经元做完全相同的计算 → 对称性问题，等价于只有 1 个神经元

### 1.2 Xavier/Glorot 初始化

**适用激活函数**：Tanh、Sigmoid

**核心思想**：让信号在前向传播和反向传播中，每一层的方差都保持一致。这样信号不会在深层衰减或爆炸。

**推导直觉**：

假设一层有 `n_in` 个输入神经元和 `n_out` 个输出神经元。如果每个权重独立地从同一分布中采样，为了让输出的方差 ≈ 输入的方差，需要：

```
Var(权重) = 2 / (n_in + n_out)
```

**Xavier Uniform 分布**：在 `[-limit, limit]` 范围内均匀采样，其中：

```
limit = √(6 / (n_in + n_out))
```

**Xavier Normal 分布**：从正态分布 `N(0, std²)` 采样，其中：

```
std = √(2 / (n_in + n_out))
```

**代码实例**（LeNet-5 项目）：

```python
# 对所有卷积层和全连接层使用 Xavier Uniform 初始化
for module in self.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        # 偏置初始化为 0
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

**LeNet-5 使用 Xavier + Tanh 为什么好？**

Tanh 的输出在 (-1, 1)，关于原点对称。Xavier 初始化确保了输入 Tanh 的值在接近原点的"活跃区"（导数较大），而不是在两端饱和区（导数接近 0）。这样：

- 前向传播时信号稳定传播
- 反向传播时梯度不会消失

### 1.3 Kaiming/He 初始化

**适用激活函数**：ReLU 及其变体（LeakyReLU、ELU 等）

**为什么 ReLU 不能用 Xavier？**

ReLU 有一个关键特性：它会把负值全部置为 0。从方差的角度看，经过 ReLU 后信号的方差会减半（因为一半的值被丢弃了）。Xavier 没有考虑这个减半效应，导致使用 Xavier + ReLU 时信号逐渐衰减。

**Kaiming 初始化**将输出方差目标调整为考虑 ReLU 的减半效应：

```
std = √(2 / n_in)       # Kaiming Normal
limit = √(6 / n_in)     # Kaiming Uniform
```

注意：Kaiming 只用 `n_in`，而 Xavier 用 `(n_in + n_out) / 2`。对于 ReLU，只考虑前向传播的方差保持即可。

**代码实例**：

```python
nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
```

**PyTorch 的默认行为**：

如果你没有显式指定初始化方式，PyTorch 对 `nn.Linear` 和 `nn.Conv2d` 默认使用 Kaiming Uniform 初始化。这就是为什么很多教程不写初始化代码也能正常运行——但对于 Tanh 这类激活函数，默认的 Kaiming 不是最优的，需要手动改为 Xavier。

### 1.4 选择对照表

| 激活函数         | 推荐初始化          | 原因                       |
| ---------------- | ------------------- | -------------------------- |
| ReLU / LeakyReLU | **Kaiming/He**      | 考虑了 ReLU 的方差减半效应 |
| Tanh / Sigmoid   | **Xavier/Glorot**   | 让值落在活跃区而非饱和区   |
| Swish / GELU     | Kaiming（近似可行） | 行为类似 ReLU              |
| 线性（无激活）   | Xavier              | 标准选择                   |

![ActivationInit.png](https://img.yumeko.site/file/articles/NNTrainingTips/ActivationInit.png)

---

## 技巧二：批量归一化（BatchNorm）的位置与注意事项

### 2.1 BatchNorm 应该放在哪里？

这是初学者经常困惑的问题。经过大量实验和论文论证，目前的主流共识是：

```
Conv2d → BatchNorm2d → ReLU     ← 推荐
Linear  → BatchNorm1d → ReLU    ← 推荐
```

**为什么 BN 在激活函数之前？**

BN 的核心目的是让激活函数的输入保持稳定的分布（均值为 β、方差为 γ²）。如果 BN 在 ReLU 之后，那它归一化的是一个被"截断"过的分布（ReLU 把负值都变成了 0），效果不如归一化原始的全范围分布。

**历史演进**：

- 原始 BatchNorm 论文：`Conv → ReLU → BN`（在激活之后）
- 后来发现 `Conv → BN → ReLU` 效果更好，成为了现代标准
- MNIST-CNN 项目采用的就是 `Conv → BN → ReLU` 的顺序

### 2.2 BatchNorm 的训练/测试差异（重要！）

这是最容易踩的坑之一。

```python
# 训练时
model.train()       # BN 使用当前 batch 的均值/方差
output = model(x)

# 测试/推理时
model.eval()        # BN 使用训练期间累积的移动平均均值/方差
with torch.no_grad():
    output = model(x)
```

**为什么测试时不能用当前 batch 的统计量？**

测试时可能只有一个样本（batch_size=1），此时 batch 的均值和方差没有统计意义。所以测试时使用训练过程中累积的全局统计量。

**错误示范**（常见 Bug）：

```python
# ❌ 错误：忘记切换到 eval 模式
model.train()  # 或者根本没设置
with torch.no_grad():
    test_output = model(x)  # BN 使用了错误的统计量！
```

这会导致测试准确率明显低于训练时的验证准确率，而且每次运行结果都不一样。

### 2.3 BatchNorm 的几个实用细节

**1. `eps` 参数**：防止分母为零的小常数，默认 `1e-5`。一般不需要修改。

**2. `momentum` 参数**：移动平均的更新速度。`running_mean = momentum × running_mean + (1-momentum) × batch_mean`。默认 0.1，PyTorch 的实现中 momentum 的定义和论文相反（PyTorch 的 0.1 ≈ 论文的 0.9）。一般也不需要修改。

**3. BatchNorm 的小 batch size 问题**：当 batch size 很小（比如 2 或 4）时，batch 统计量的噪声很大，BN 的效果会下降。替代方案：LayerNorm（Transformer 中常用）或 GroupNorm。

**4. BatchNorm 不适合的情况**：

- Batch size = 1（用 LayerNorm/InstanceNorm 替代）
- RNN 等序列模型（用 LayerNorm）
- 生成模型中对噪声敏感的任务

![BatchNorm.png](https://img.yumeko.site/file/articles/NNTrainingTips/BatchNorm.png)

---

## 技巧三：Dropout 的正确用法

### 3.1 Dropout 的概率设置

| 层类型       |                推荐 Dropout 概率                |
| ------------ | :---------------------------------------------: |
| 输入层       |        0.0 ~ 0.2（通常不 dropout 输入）         |
| 隐藏层（FC） |              **0.5**（最常用的值）              |
| 卷积层       | 0.0 ~ 0.2（卷积层本身参数少，不太需要 dropout） |
| 输出层       |          0.0（绝对不能 dropout 输出）           |

MNIST-CNN 只在第一个全连接层后使用 `Dropout(p=0.5)`，卷积层不加 dropout，这是标准做法。

### 3.2 `model.train()` 和 `model.eval()` 对 Dropout 的影响

```python
model.train()   # Dropout 生效：随机丢弃神经元
model.eval()    # Dropout 不生效：所有神经元参与，但输出乘以 (1-p)

# 实际上 PyTorch 的 eval 模式直接不做 dropout
# 数学上等价于保留所有神经元并缩放
```

**重要**：如果你在测试/验证时忘记调用 `model.eval()`，Dropout 会继续随机丢弃神经元，导致：

- 预测结果不可复现（每次不同）
- 准确率大幅下降
- 验证损失异常升高

### 3.3 Dropout 和 BatchNorm 共用时的细节

在 `model.eval()` 模式下：

- Dropout 自动关闭
- BatchNorm 切换到全局统计量

所以只要正确调用 `model.eval()`，两者都能自动正常工作。不需要分别处理。

![Dropout.png](https://img.yumeko.site/file/articles/NNTrainingTips/Dropout.png)

---

## 技巧四：数据增强的策略选择

### 4.1 基本原则

**核心原则**：数据增强应该模拟真实场景中可能出现的变换。

| 任务类型     | 合适的增强                   | 不合适的增强               |
| ------------ | ---------------------------- | -------------------------- |
| 手写数字     | 旋转、平移、轻微缩放         | 水平翻转（6 和 9 会混淆）  |
| 自然图像分类 | 随机裁剪、水平翻转、颜色抖动 | 垂直翻转（天空不会在下面） |
| 医学影像     | 旋转、平移、弹性变形         | 颜色抖动（染色有诊断意义） |
| 文本识别     | 平移、轻微旋转               | 水平翻转（文字会反过来）   |

### 4.2 MNIST-CNN 的数据增强

```python
transforms.RandomAffine(
    degrees=10,          # 旋转：±10°
    translate=(0.1, 0.1) # 平移：±10% 图像宽高
)
```

**为什么只有旋转和平移？**

手写数字最常见的变化就是书写角度（有人写斜）和位置偏移（有人偏左偏右）。这两种增强足够模拟真实场景，同时不会改变数字的类别（旋转 10° 的 "3" 还是 "3"）。

**为什么不做水平翻转？**

如果把 "6" 水平翻转，看起来就像反向的 "6"，在手写数字数据集中这很容易和 "6" 混淆。更重要的是，有些数字翻转后会变成另一个数字（比如某些风格的 "9" 翻转后像 "6"），这会引入标签噪声。

### 4.3 数据增强的"金科玉律"

1. **只在训练集上做增强**，验证集和测试集只用基础预处理（ToTensor + Normalize）
2. **增强后的样本必须仍然是合理的**——人类能正确分类，模型才可能学会
3. **从轻到重**：先从最保守的增强开始，逐步增加强度，观察验证准确率
4. **可视化检查**：训练前先看看增强后的图像是什么样的，确保没有引入错误

> **建议插图**：【图4】数据增强策略对比。展示同一张原始图像经过不同增强（旋转、平移、翻转、颜色抖动）后的效果，标注哪些适合手写数字，哪些不适合。

---

## 技巧五：学习率调度策略

### 5.1 学习率是最重要的超参数

有句经验之谈：如果你只能调一个超参数，那就调学习率。

- 学习率太大 → 损失上下跳，无法收敛
- 学习率太小 → 收敛太慢，可能陷入局部最优
- 学习率刚好 → 快速稳定收敛

### 5.2 常见调度策略

#### 策略 1：ReduceLROnPlateau（按需衰减）

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控指标越低越好（val_loss）
    factor=0.5,        # 衰减因子：新 lr = 旧 lr × 0.5
    patience=3,        # 容忍 3 个 epoch 不改善
    min_lr=1e-6        # 学习率下限
)

# 每个 epoch 后调用
scheduler.step(val_loss)  # 传入监控指标
```

**优点**：自适应，不需要预先知道训练多少 epoch
**缺点**：只在 loss 卡住时才降，可能反应不够快
**使用场景**：MNIST-CNN 项目采用此策略

#### 策略 2：StepLR（阶梯衰减）

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,      # 每 30 个 epoch
    gamma=0.1          # 衰减为原来的 0.1
)
```

**学习率变化**：Epoch 1-30: lr=0.001, Epoch 31-60: lr=0.0001, Epoch 61-90: lr=0.00001

#### 策略 3：CosineAnnealingLR（余弦退火）

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100           # 半个余弦周期的 epoch 数
)
```

**学习率变化**：从初始 lr 按余弦曲线平滑衰减到 0。训练后期 lr 非常小，帮助精确收敛。现代大模型训练（如 ViT、CLIP）普遍使用此策略。

### 5.3 策略对比

| 策略              | 适用场景            | 优点             | 缺点         |
| ----------------- | ------------------- | ---------------- | ------------ |
| ReduceLROnPlateau | 训练 epoch 数不确定 | 自适应，简单     | 被动等待     |
| StepLR            | 已知总 epoch 数     | 可预测，易理解   | 阶梯式突变   |
| CosineAnnealing   | 需要精细收敛        | 平滑衰减，效果好 | 需要预设周期 |
| Cosine + Warmup   | 大模型训练          | 最稳定           | 调参复杂     |

### 5.4 学习率 Warmup

**问题**：训练刚开始时，模型参数是随机的，梯度可能很大。如果直接用预设的大学习率，可能导致模型在最初几步就"跑偏"（发散）。

**方案**：前 N 个 epoch/batch 用较小的学习率，然后逐渐增加到目标学习率。

```
Warmup 阶段：lr 从 0 线性增加到 target_lr
正常阶段：使用上述调度策略衰减
```

*MNIST-CNN 和 LeNet-5 项目因为任务简单，都没有使用 warmup，但对于复杂任务（大模型、大批量），warmup 很关键。*

![SchedulingStrategy.png](https://img.yumeko.site/file/articles/NNTrainingTips/SchedulingStrategy.png)

---

## 技巧六：Early Stopping 防止过拟合

### 6.1 过拟合的信号

训练过程中，如果你观察到：

- 训练损失持续下降
- 验证损失先下降，然后开始上升

这就是经典的过拟合信号——模型开始"背诵"训练数据而非学习通用模式。

```
Loss
  ↑
  │    val_loss ───╲
  │              ╱    ╲___（开始上升）
  │    train_loss ──────────────（持续下降）
  │
  └──────────────────────────→ Epoch
           ↑ 最佳停止点
```

### 6.2 Early Stopping 的实现

LeNet-5 项目实现了一个比较完善的 Early Stopping 机制：

```python
def _checkEarlyStop(self, trainLoss, valLoss, trainAcc, valAcc):
    """检查三个条件"""

    # 条件 1：验证损失不再改善
    if self.patienceCounter >= self.patience:      # patience=3
        return True, "val_loss 不再改善"

    # 条件 2：过拟合检测
    if self.overfitCounter >= self.overfitPatience:  # overfitPatience=3
        return True, "检测到过拟合"

    # 条件 3：训练和验证准确率差距过大
    if trainAcc - valAcc > 10.0:
        return True, "训练-验证准确率差距过大"
```

**条件 1（验证损失不再改善）**：如果连续 3 个 epoch 验证损失都没有显著降低（改善小于 min_delta），说明模型已经学不到更好的泛化能力了，继续训练只是浪费时间。

**条件 2（过拟合检测）**：当训练损失还在下降，但验证损失不再下降（甚至上升）时，模型正在过拟合训练数据。

**条件 3（准确率差距过大）**：如果训练准确率比验证准确率高超过 10 个百分点，说明模型在训练集上表现远超验证集——严重的过拟合。

### 6.3 min_delta 的设计

```python
if valLoss < bestValLoss - self.minDelta:   # 改善了至少 minDelta
    bestValLoss = valLoss
```

`minDelta=1e-4` 意味着验证损失必须下降至少 0.0001 才算"有效改善"。这避免了因为随机波动而误判为改善。

![EarlyStop.png](https://img.yumeko.site/file/articles/NNTrainingTips/EarlyStop.png)

---

## 技巧七：梯度裁剪与监控

### 7.1 什么是梯度爆炸？

在反向传播过程中，梯度会从输出层逐层传回输入层。如果某些层的梯度特别大，经过多层累积后，浅层的梯度可能变得极大，导致参数更新步幅过大，模型参数突然变成 NaN。

### 7.2 梯度裁剪

```python
# 将梯度范数限制在阈值以内
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

常见阈值：1.0 或 5.0。这个技巧在 RNN/LSTM 中特别重要，在 CNN 中作为安全网使用。

*MNIST-CNN 和 LeNet-5 项目由于任务简单、网络浅，没有使用梯度裁剪，但在实际工程中建议加上。*

### 7.3 监控梯度大小（调试用）

使用 TensorBoard 或 wandb 记录梯度的范数，可以帮助诊断训练问题：

- 梯度范数突然飙升 → 数据有问题或学习率太大
- 梯度范数持续接近 0 → 梯度消失，可能需要调整初始化或激活函数
- 梯度范数在层间差异极大 → 网络结构或初始化有问题

---

## 技巧八：检查点（Checkpoint）管理

### 8.1 应该保存什么？

一个完整的检查点应包含恢复训练所需的一切：

```python
checkpoint = {
    'epoch': currentEpoch,              # 当前 epoch 数
    'model_state_dict': model.state_dict(),  # 模型权重
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'trainLosses': trainLosses,         # 训练历史
    'valLosses': valLosses,
    'trainAccs': trainAccs,
    'valAccs': valAccs,
}
```

### 8.2 两类检查点的策略

| 类型     | 文件名           | 更新时机          | 用途           |
| -------- | ---------------- | ----------------- | -------------- |
| 最佳模型 | `best_model.pth` | 验证损失创新低时  | 最终部署、评估 |
| 最近模型 | `last_model.pth` | 每个 epoch 结束时 | 恢复中断的训练 |

**保存最佳模型的条件**：

```python
if valLoss < bestValLoss - minDelta:
    bestValLoss = valLoss
    torch.save(model.state_dict(), 'best_model.pth')
    print(f"✅ 保存最佳模型 (val_loss={valLoss:.4f})")
```

### 8.3 安全写入

在多进程或可能中断的环境下，先写临时文件再重命名，防止写入一半时进程崩溃导致文件损坏：

```python
# 更安全的做法
tempPath = savePath + '.tmp'
torch.save(checkpoint, tempPath)
os.replace(tempPath, savePath)  # 原子操作
```

### 8.4 MNIST-CNN 的完整检查点管理

MNIST-CNN 实现了更完整的保存/恢复流程：

- **saveCheckpoint()**：保存完整训练状态到 `last_model.pth`
- **loadCheckpoint()**：恢复模型权重、优化器状态、epoch 数
- **训练中断恢复**：通过 `--resume` 参数从断点继续训练

![Checkpoint.png](https://img.yumeko.site/file/articles/NNTrainingTips/Checkpoint.png)

---

## 技巧九：训练/评估模式切换

### 9.1 为什么这是 Bug 高发区？

PyTorch 中 `model.train()` 和 `model.eval()` 的行为差异体现在多个层：

| 层                            |     model.train()     |   model.eval()   |
| ----------------------------- | :-------------------: | :--------------: |
| Dropout                       |    随机丢弃神经元     |      不丢弃      |
| BatchNorm                     | 使用当前 batch 统计量 | 使用全局移动平均 |
| BatchNorm 的 running_mean/var |       **更新**        |  冻结（不更新）  |

### 9.2 最佳实践

在项目中建立清晰的模式：

```python
# 训练阶段
def trainEpoch(model, dataloader, ...):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# 验证/测试阶段
@torch.no_grad()     # 第 1 层保护：不追踪梯度
def validateEpoch(model, dataloader, ...):
    model.eval()     # 第 2 层保护：BN/Dropout 正确行为
    for batch in dataloader:
        output = model(images)
        # 不调用 backward/optimizer.step
```

**双重保险**：`model.eval()` + `torch.no_grad()` 同时使用。

- `model.eval()` 负责 BN/Dropout 的行为
- `torch.no_grad()` 负责关闭梯度追踪（节省显存）

两者是独立的，不能互相替代。

### 9.3 易错场景

```python
# ❌ 常见错误 1：验证时忘了 eval()
model.train()  # 仍然是 train 模式！
with torch.no_grad():
    valOutput = model(valImages)  # Dropout 还在随机丢弃！

# ❌ 常见错误 2：推理时忘了两者
output = model(testImages)  # Dropout 生效 AND 梯度追踪，显存浪费

# ✅ 正确做法
model.eval()
with torch.no_grad():
    output = model(testImages)
```

---

## 技巧十：固定随机种子保证可复现性

### 10.1 为什么不可复现是噩梦？

假设你跑了一个实验，准确率 95%。第二天改了一行不相关的代码，准确率变成了 93%。是改出 Bug 了，还是随机性导致？

如果没有固定随机种子，你永远分不清是代码变更还是随机波动。

### 10.2 完整的种子固定方案

MNIST-CNN 的做法：

```python
import random
import numpy as np
import torch

def setSeed(seed=42):
    random.seed(seed)              # Python 内置 random
    np.random.seed(seed)           # NumPy
    torch.manual_seed(seed)        # PyTorch CPU
    torch.cuda.manual_seed(seed)   # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # 所有 GPU

    # 强制确定性算法（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**cudnn.deterministic 的代价**：某些 cuDNN 操作的非确定性实现更快。开启确定性后，速度可能略降，但每次运行结果完全一致。

**cudnn.benchmark**：设为 True 时，cuDNN 会根据输入尺寸自动选择最快的卷积算法。但如果输入尺寸变化（比如动态序列长度），每次选择的算法可能不同，引入随机性。设为 False 保证确定性。

### 10.3 数据划分的种子

```python
# 训练/验证划分也需要固定种子
generator = torch.Generator().manual_seed(seed)
trainSet, valSet = random_split(
    fullDataset, [trainLen, valLen],
    generator=generator
)
```

确保每次划分的训练集/验证集完全一致，否则比较不同实验的结果时会有偏差。