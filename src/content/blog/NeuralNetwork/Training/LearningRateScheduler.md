---
title: 学习率调度完全指南
date: 2026-05-23
category: 神经网络/训练
tags:
  - 高级教程
  - 深度学习
description: 学习率是最重要的超参数。详解 StepLR、ReduceLROnPlateau、CosineAnnealing、Warmup 的原理、使用方法和选择原则，包含完整训练代码示例。
image: https://img.yumeko.site/file/blog/articles/1780581370487.webp
status: published
---

## 1. 为什么调整学习率

有句经验之谈：如果你只能调一个超参数，那就调学习率。

- 学习率太大 -> 损失上下跳，无法收敛
- 学习率太小 -> [[NeuralNetwork/Troubleshooting/SlowConvergence|收敛太慢]]，可能陷入局部最优
- 学习率刚好 -> 快速稳定收敛

**为什么需要动态调整？** 训练初期离最优解远，需要大步快走（大 LR）。训练后期接近最优解，如果一直用大学习率，会在最优解附近来回震荡无法收敛。大步可能跨过头，需要小步微调（小 LR）。

从优化公式看，学习率调度器本质上是在控制每一步的步长：

$$
\theta_{t+1}=\theta_t-\eta_t\nabla_\theta L(\theta_t)
$$

其中 $\theta_t$ 是第 $t$ 步参数，$L$ 是损失函数，$\nabla_\theta L(\theta_t)$ 是当前梯度，$\eta_t$ 就是第 $t$ 步学习率。调度器要做的事，是把固定的 $\eta$ 变成随训练进度变化的序列 $\eta_t$。

![Comparision.png](https://img.yumeko.site/file/blog/articles/1780581550590.webp)

## 2. PyTorch 调度器使用模式

```python
# 创建调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    train(...)
    valLoss = validate(...)

    # 方式 1：按 epoch 调度（大多数调度器）
    scheduler.step()

    # 方式 2：按 metric 调度（ReduceLROnPlateau 专用）
    scheduler.step(valLoss)
```

**重要**：`scheduler.step()` 必须在 `optimizer.step()` 之后调用（每个 epoch 结束时），不是在每个 batch 之后。

### 2.1 各调度器直观对比

| 调度器 | 曲线形态 | 核心参数 |
| --- | --- | --- |
| ReduceLROnPlateau | 阶梯下降（触发式） | factor=0.5, patience=3 |
| StepLR | 阶梯下降（固定间隔） | step_size=30, gamma=0.1 |
| CosineAnnealing | 平滑余弦下降 | T_max=50 |
| CosineWarmRestarts | 余弦周期 + 重启 | T_0=10, T_mult=2 |
| OneCycleLR | 先升后降 | max_lr=0.01, pct_start=0.3 |

## 3. StepLR：阶梯式学习率衰减

### 3.1 工作原理

每 `step_size` 个 epoch，学习率乘以 `gamma`：

$$
\eta_e = \eta_0 \cdot \gamma^{\left\lfloor (e-1) / s \right\rfloor}
$$

其中 $\eta_0$ 是初始学习率，$e$ 是当前 epoch，$s$ 是 `step_size`，$\gamma$ 是 `gamma`。$\left\lfloor (e-1) / s \right\rfloor$ 表示当前 epoch 前已经经历了多少次完整的衰减间隔。

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,      # 每 30 个 epoch
    gamma=0.1          # 衰减为原来的 0.1
)
```

学习率变化（初始 lr=0.1）：

$$
\text{Epoch 1-30: } lr = 0.1 \quad
\text{Epoch 31-60: } lr = 0.01 \quad
\text{Epoch 61-90: } lr = 0.001
$$

![StepLR.png](https://img.yumeko.site/file/blog/articles/1780581580231.webp)

### 3.2 MultiStepLR：自定义衰减点

如果希望更灵活地控制衰减时机：

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 80, 120],  # 在这些 epoch 衰减
    gamma=0.1
)
```

这种方式在经典论文中很常见（如 ResNet 在 epoch 30 和 60 各衰减一次）。

MultiStepLR 的数学形式可以写成：

$$
\eta_e = \eta_0 \cdot \gamma^{N(e)}
$$

其中 $N(e)$ 表示到当前 epoch 为止已经经过的 `milestones` 数量：

$$
N(e)=\sum_{m \in \mathcal{M}} \mathbf{1}(e \ge m)
$$

$\mathcal{M}$ 是所有里程碑 epoch 的集合。相比 StepLR，MultiStepLR 只是把“固定间隔衰减”改成了“指定时间点衰减”。

### 3.3 参数选择

**step_size**：
- 太短：频繁衰减，训练后期学习率可能过早降到极小
- 太长：学习率保持高位太久，收敛不够精细
- 建议：总 epoch 的 1/3 到 1/5

**gamma**：
- 0.1：每次减为 1/10（经典选择）
- 0.5：每次减半（更温和）
- 0.3：折中

### 3.4 何时使用

- 训练 epoch 数是预先确定的
- 希望学习率曲线可预测
- 经典论文复现（很多论文使用 MultiStepLR）
- 数据集和任务比较熟悉，知道大概在什么阶段需要降学习率

### 3.5 与其他调度器对比

| | StepLR | ReduceLROnPlateau | CosineAnnealing |
| --- | :---: | :---: | :---: |
| 衰减方式 | 固定时间点 | 触发式 | 连续平滑 |
| 可预测性 | 高 | 低 | 高 |
| 调参难度 | 需选时间点 | 调 patience | 需选周期 |

## 4. ReduceLROnPlateau：自适应学习率衰减

### 4.1 工作原理

```
if val_loss 连续 patience 个 epoch 没有显著改善:
    lr = lr x factor
```

这是最"懒人友好"的调度器——不需要预设总 epoch 数，调度器自己判断时机。

数学上，ReduceLROnPlateau 不是只依赖 epoch 的函数，而是依赖验证指标序列。以 `mode='min'` 为例，记第 $e$ 个 epoch 的验证损失为 $m_e$，历史最优为 $b_e$：

$$
b_e=\min(m_1,m_2,\dots,m_e)
$$

按绝对阈值理解时，有效改善条件为：

$$
m_e < b_{e-1} - \delta
$$

如果连续 $p$ 个 epoch 都没有满足这个条件，那么触发学习率衰减：

$$
\eta_e = \max(\eta_{\min}, \eta_{e-1}\cdot f)
$$

其中 $p$ 是 `patience`，$\delta$ 是 `threshold`，$f$ 是 `factor`，$\eta_{\min}$ 是 `min_lr`。PyTorch 默认还支持相对阈值模式，此时改善条件会按比例判断。无论使用哪种阈值模式，它的曲线都无法在训练前完全确定，因为衰减时间取决于验证集表现。

### 4.2 参数详解

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # 'min' for loss, 'max' for accuracy
    factor=0.5,         # 衰减因子
    patience=3,         # 容忍多少个 epoch 不改善
    threshold=1e-4,     # 改善至少多少才算"有效改善"
    min_lr=1e-6,        # 学习率下限
    verbose=True        # 打印衰减信息
)

scheduler.step(valLoss)  # 必须传入监控指标
```

### 4.3 参数选择指南

**factor**：
- $0.5$（减半）：温和，常用
- $0.1$（减为 1/10）：激进，快速降低
- $0.3$：折中

**patience**：
- 太小（2-3）：容易因波动提前衰减
- 中等（5-10）：给模型时间跨过高原
- 太大：可能等太久

**mode**：
- `'min'`：监控 loss（越低越好）
- `'max'`：监控 accuracy（越高越好）

### 4.4 训练曲线效果

```
学习率变化（factor=0.5, patience=3）：
Epoch  1- 9: lr = 0.001   (初始)
Epoch 10-19: lr = 0.0005  (第7-9 epoch 没改善 -> 减半)
Epoch 20-24: lr = 0.00025 (第17-19 epoch 没改善 -> 再减半)
...
```

![ReduceLROnPlateau.png](https://img.yumeko.site/file/blog/articles/1780581576187.webp)

### 4.5 使用示例

```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                               factor=0.5, patience=3, min_lr=1e-6)

for epoch in range(epochs):
    trainLoss = train(...)
    valLoss = validate(...)

    scheduler.step(valLoss)  # <- 必须传入

    # 查看当前学习率
    currentLR = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: lr = {currentLR:.6f}")
```

### 4.6 优缺点

**优点**：自适应，无需预设 epoch；简单可靠；适合实验探索阶段。

**缺点**：只在卡住时才降，不能主动规划学习率曲线；可能反应偏慢。

### 4.7 与 Early Stopping 的区别

两者都监控验证指标，但目的不同：
- **ReduceLROnPlateau**：降学习率，继续训练
- **[[NeuralNetwork/Training/TrainingStability|Early Stopping]]**：停止训练，防止过拟合

通常两者配合使用，ReduceLROnPlateau 的 patience 应该小于 Early Stopping 的 patience。

## 5. CosineAnnealing：余弦退火详解

### 5.1 为什么用余弦曲线

StepLR 在衰减点学习率突变，导致训练在这些点出现短暂的震荡。余弦退火提供了一条平滑的衰减曲线，让学习率连续地、优雅地降低：

$$
lr_t = lr_{\min} + \frac{1}{2}(lr_{\max} - lr_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)
$$

其中 $T_{\max}$ 是半周期长度。

这个公式可以拆成两部分理解：

$$
\alpha_t=\frac{1}{2}\left(1+\cos\left(\frac{t}{T_{\max}}\pi\right)\right)
$$

$$
lr_t = lr_{\min} + (lr_{\max} - lr_{\min})\alpha_t
$$

$\alpha_t$ 是从 $1$ 平滑下降到 $0$ 的系数。训练开始时 $t=0$，$\cos(0)=1$，所以 $lr_t=lr_{\max}$；训练结束时 $t=T_{\max}$，$\cos(\pi)=-1$，所以 $lr_t=lr_{\min}$。

余弦曲线在开头和结尾变化都比较慢，中间变化更快。这意味着训练初期不会突然降学习率，训练后期也会更温和地接近最小学习率。

### 5.2 基本使用

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,       # 半个余弦周期长度
    eta_min=1e-6     # 最小学习率（默认 0）
)
```

学习率从初始值按余弦曲线平滑降到 `eta_min`。

### 5.3 带重启的余弦退火

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # 第一个周期长度
    T_mult=2,        # 后续周期倍增：10 -> 20 -> 40 -> 80 ...
    eta_min=1e-6
)
```

每个周期结束，学习率跳回初始值。周期逐渐变长，给模型初期"多探索"的机会、后期"精细收敛"的时间。

带重启的版本会把总训练过程拆成多个余弦周期。第 $i$ 个周期长度为：

$$
T_i = T_0 \cdot T_{\text{mult}}^i
$$

在某个周期内部，记已经走过的步数为 $t_{\text{cur}}$，则：

$$
\eta_t = \eta_{\min}
+ \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\frac{t_{\text{cur}}}{T_i}\pi\right)\right)
$$

当 $t_{\text{cur}}=T_i$ 时，这一轮退火结束；下一轮开始时 $t_{\text{cur}}$ 重置为 $0$，学习率回到 $\eta_{\max}$。`T_mult > 1` 时，后续周期越来越长，学习率重启越来越少。

**为什么需要重启？** 跳出局部最优——学习率突然变大给优化器一个"跳跃"的机会，可能跳出当前局部极小值。

### 5.4 学习率曲线对比

![Comparision.png](https://img.yumeko.site/file/blog/articles/1780581550590.webp)

### 5.5 为什么大模型训练常用

现代 ViT、CLIP、GPT 等模型几乎全部使用 Cosine + [[NeuralNetwork/Training/LearningRate|Warmup]]：

- **平滑性**：大模型对学习率突变更敏感，余弦曲线的平滑性避免了震荡
- **后期精细**：余弦最后阶段的极小学习率让模型精细收敛
- **可预测**：学习率曲线完全由 `T_max` 确定，方便对比实验

### 5.6 典型配置

```python
# 大模型标准配置
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                               weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-6
)
# 加上 Warmup：前 5 个 epoch 线性增长到 0.001

# 长时间训练配置（如 800 epoch）
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-6
)
# 周期：50 -> 100 -> 200 -> 400
```

## 6. Warmup：平稳起步的艺术

### 6.1 为什么需要 Warmup

训练开始时，模型参数是随机初始化的。此时：
- 梯度可能很大且方向不稳定
- 如果直接用预设的大学习率，参数更新幅度过大
- 模型可能在最初几步就"跑偏"（损失面很差的方向）

Warmup 的解决方案：前 N 步用较小的学习率，逐渐增加到目标学习率，让模型先"站稳脚跟"。

![WarmupVSCos.png](https://img.yumeko.site/file/blog/articles/1780581586761.webp)

### 6.2 Linear Warmup

最简单、最常用的 warmup 方式——学习率线性增长：

$$
\eta_t =
\begin{cases}
\eta_{\text{target}}\cdot \frac{t+1}{T_w}, & t < T_w \\
\eta_{\text{target}}, & t \ge T_w
\end{cases}
$$

其中 $T_w$ 是 warmup 总步数，$\eta_{\text{target}}$ 是目标学习率。线性 warmup 的含义很直接：每一步增加固定幅度，直到达到目标学习率。

```python
def linearWarmup(optimizer, currentStep, warmupSteps, targetLR):
    if currentStep < warmupSteps:
        lr = targetLR * (currentStep + 1) / warmupSteps
        for g in optimizer.param_groups:
            g['lr'] = lr

# 训练循环中
for batchIdx, (images, labels) in enumerate(trainLoader):
    linearWarmup(optimizer, batchIdx, warmupSteps=500, targetLR=0.001)
    # ... 正常训练步骤
```

### 6.3 Cosine Warmup

学习率按余弦曲线从 0 上升到目标值：

$$
\eta_t =
\begin{cases}
\frac{1}{2}\eta_{\text{target}}
\left(1-\cos\left(\frac{t}{T_w}\pi\right)\right), & t < T_w \\
\eta_{\text{target}}, & t \ge T_w
\end{cases}
$$

当 $t=0$ 时，$\eta_t=0$；当 $t=T_w$ 时，$\eta_t=\eta_{\text{target}}$。因为余弦函数两端斜率较小，所以 Cosine Warmup 在开始和结束时都更平滑。

```python
def cosineWarmup(optimizer, currentStep, warmupSteps, targetLR):
    if currentStep < warmupSteps:
        progress = currentStep / warmupSteps
        lr = targetLR * 0.5 * (1 - np.cos(np.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr
```

Cosine Warmup 比 Linear Warmup 更平滑（两端变化慢，中间快）。

### 6.4 Warmup 步数选择

| 数据集/模型规模 | 推荐 warmup 步数 |
| --- | :---: |
| 小型任务（MNIST, CIFAR-10） | 通常不需要，或 100-200 步 |
| 中型任务（ImageNet, ResNet-50） | 5 个 epoch 以内 |
| 大型任务（ViT, GPT） | 通常 2000-4000 步 |
| 超大 batch size 训练 | 必须 warmup，步数与 batch size 相关 |

经验法则：batch size 越大，warmup 越重要。使用 `batch_size=8192` 时，warmup 几乎是必须的。

### 6.5 Warmup + Decay 完整示例

Warmup + Cosine Decay 可以写成一个完整的分段函数：

$$
\eta_t =
\begin{cases}
\eta_{\max}\cdot \frac{t}{T_w}, & 0 \le t < T_w \\
\eta_{\min}
+ \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\frac{t-T_w}{T-T_w}\pi\right)\right),
& T_w \le t \le T
\end{cases}
$$

其中 $T_w$ 是 warmup 步数，$T$ 是总训练步数。第一段负责“起步”，第二段负责“退火”。这也是现代大模型训练中最常见的学习率曲线。

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmupEpochs, totalEpochs, minLR=1e-6):
        self.optimizer = optimizer
        self.warmupEpochs = warmupEpochs
        self.totalEpochs = totalEpochs
        self.minLR = minLR
        self.baseLRs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmupEpochs:
            # Warmup: 线性增长
            scale = (epoch + 1) / self.warmupEpochs
        else:
            # Cosine decay
            progress = (epoch - self.warmupEpochs) / (self.totalEpochs - self.warmupEpochs)
            scale = self.minLR + 0.5 * (1 - self.minLR) * (1 + np.cos(np.pi * progress))

        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = self.baseLRs[i] * scale
```

### 6.6 Warmup 的进阶知识

**不只在训练初期需要**：如果学习率调度使用了 CosineAnnealingWarmRestarts（周期性重启），每次学习率跳回高值时，也需要一个小规模的 warmup 来防止突然跳跃导致的震荡。

**Transformer 中的 warmup**：Transformer 对 warmup 特别敏感。原始 "Attention Is All You Need" 论文使用了 4000 步 warmup。不设 warmup 可能导致 Transformer 无法收敛。

## 7. 完整训练代码示例

### 7.1 Warmup + Cosine 组合调度器

现代训练流程通常 Warmup + 调度器组合使用：

```python
# 自定义 Warmup + Cosine 调度
class WarmupCosineLR:
    def __init__(self, optimizer, warmupSteps, totalSteps, minLR=1e-6):
        self.optimizer = optimizer
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
        self.baseLRs = [g['lr'] for g in optimizer.param_groups]

    def step(self, step):
        if step < self.warmupSteps:
            # Warmup 阶段：线性增长
            scale = step / self.warmupSteps
        else:
            # Cosine 阶段：余弦衰减
            progress = (step - self.warmupSteps) / (self.totalSteps - self.warmupSteps)
            scale = self.minLR + 0.5 * (1 - self.minLR) * (1 + np.cos(np.pi * progress))

        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = self.baseLRs[i] * scale
```

### 7.2 OneCycleLR：一周期策略

OneCycleLR 将学习率先升后降，将整个训练周期视为一个循环。常配合较大的 batch size 使用：

数学上可以近似理解为两段调度。设总步数为 $T$，上升阶段比例为 $r$，则上升阶段步数为 $T_{\text{up}}=rT$：

$$
\eta_t =
\begin{cases}
\text{anneal}(\eta_{\text{low}}, \eta_{\max}, t / T_{\text{up}}),
& 0 \le t < T_{\text{up}} \\
\text{anneal}(\eta_{\max}, \eta_{\text{final}},
(t-T_{\text{up}})/(T-T_{\text{up}})),
& T_{\text{up}} \le t \le T
\end{cases}
$$

其中 $\text{anneal}$ 可以是线性插值，也可以是余弦插值。PyTorch 默认使用余弦退火。直观上，OneCycleLR 先用逐渐变大的学习率加速探索，再用更小的最终学习率完成精细收敛。

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(trainLoader),
    epochs=30,
    pct_start=0.3    # 前 30% 时间升温
)
```

### 7.3 LR Finder：自动找最佳学习率

Leslie Smith 提出的方法：从一个很小的学习率开始，每个 batch 逐步增大，记录 Loss 变化。选择 Loss 下降最快的那段对应的学习率。

学习率通常按指数方式增长：

$$
\eta_t=\eta_{\text{start}}\cdot q^t
$$

如果希望在 $T$ 步内从 $\eta_{\text{start}}$ 增长到 $\eta_{\text{end}}$，增长系数可以设为：

$$
q=\left(\frac{\eta_{\text{end}}}{\eta_{\text{start}}}\right)^{1/(T-1)}
$$

观察损失函数 $L(\eta_t)$ 的变化时，通常不选择损失最低点对应的学习率，而选择损失开始快速下降、但还没有剧烈发散前的学习率。这样得到的值更适合作为训练初始学习率或 `max_lr`。

```python
# 简化的 LR Finder
def lrFinder(model, trainLoader, criterion, device):
    lrs = []
    losses = []
    lr = 1e-7
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for batch in trainLoader:
        lrs.append(lr)
        loss = trainOneBatch(model, batch, criterion, optimizer, device)
        losses.append(loss)

        lr *= 1.1  # 每 batch 增大 10%
        if lr > 1.0 or loss > 10 * min(losses):
            break
        for g in optimizer.param_groups:
            g['lr'] = lr

    # Plot losses vs lrs, pick lr where loss drops fastest
    return lrs, losses
```

### 7.4 CNN 项目调度器实践

```python
# 从 CNN 项目 cnnlib/training/scheduler.py
def createScheduler(optimizer, name, **kwargs):
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                 patience=3, min_lr=1e-6)
    elif name == 'step':
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=50)
    elif name == 'cosine_warm':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
```

使用方式：

```python
from cnnlib.training.scheduler import createScheduler

# ReduceLROnPlateau
scheduler = createScheduler(optimizer, 'plateau', factor=0.5, patience=3)

# StepLR
scheduler = createScheduler(optimizer, 'step', step_size=10, gamma=0.1)

# CosineAnnealingLR
scheduler = createScheduler(optimizer, 'cosine', T_max=50)

# CosineAnnealingWarmRestarts
scheduler = createScheduler(optimizer, 'cosine_warm', T_0=10, T_mult=1)
```

## 8. 选择原则与总结

### 8.1 策略对比

| 策略 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| ReduceLROnPlateau | epoch 数不确定 | 自适应，简单 | 被动等待 |
| StepLR | 已知总 epoch | 可预测 | 阶梯式突变 |
| CosineAnnealing | 需要精细收敛 | 平滑，效果好 | 需要预设周期 |
| Cosine + Warmup | 大模型训练 | 最稳定 | 调参复杂 |
| OneCycleLR | 快速训练 | 收敛极快 | 对 batch size 敏感 |

### 8.2 选择建议

| 场景 | 推荐 |
| --- | --- |
| epoch 数不确定，需要自动调整 | ReduceLROnPlateau |
| 已知训练总 epoch，简单可靠 | StepLR 或 MultiStepLR |
| 精细收敛，追求最优精度 | CosineAnnealing |
| 大模型，需要长时间训练 | CosineWarmRestarts |
| 快速训练，大 batch size | OneCycleLR |

### 8.3 总结

- **学习率是最重要的超参数**：调好学习率及其调度策略，往往比调模型结构更有效。
- **没有银弹**：不同任务、不同模型规模适合不同的调度策略。
- **Warmup 是标配**：现代大模型训练（Transformer、ViT 等）几乎都需要 Warmup，不设可能导致无法收敛。
- **Cosine 是主流**：从 ViT 到 GPT，CosineAnnealing 已成为大模型训练的事实标准。
- **小任务从简**：小型数据集和简单模型上，ReduceLROnPlateau 或 StepLR 往往足够。
- **实验优先**：不确定用哪个时，先用 ReduceLROnPlateau 跑通流程，再尝试 Cosine 追求更高精度。
