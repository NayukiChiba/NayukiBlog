---
title: 扩散模型推理详解：从 DDPM 采样到 Classifier-Free Guidance
date: 2026-06-05
category: 神经网络/扩散模型
tags:
  - 扩散模型
  - 生成模型
  - DDPM
  - DDIM
  - 推理加速
description: 系统讲解扩散模型的数学原理：前向扩散过程的噪声调度、反向去噪过程的参数化、DDPM 完整采样算法、DDIM 确定性加速采样、Classifier-Free Guidance 条件生成技术，以及主流采样器对比。
image: https://img.yumeko.site/file/blog/cover/1780668233580.webp
status: draft
---
> **前置阅读**：本文假定读者熟悉概率论基础。建议先阅读 [[Math/ProbabilityInequalities|概率不等式完全指南]]。

## 1. 问题的起点：扩散模型的直觉

扩散模型的核心直觉来自物理学中的**非平衡热力学**：

1. **前向过程（Forward Process）**：往一张清晰图片中逐步加噪声，最终变成完全的随机噪声
2. **反向过程（Reverse Process）**：学习如何从噪声中一步步去除噪声，恢复出清晰的图片

想象将一滴墨水滴入一杯清水中——墨水逐渐扩散（前向）。如果你知道扩散的规律，可以反向计算墨水的初始位置（反向）。扩散模型学习的正是这个反向过程。

> [!NOTE] 扩散模型 vs GAN vs VAE
> - GAN：对抗训练，生成质量高但训练不稳定
> - VAE：变分推断，训练稳定但生成质量较模糊
> - 扩散模型：逐步去噪，生成质量高且训练稳定，但推理慢（需要多步迭代）

## 2. 前向扩散过程

### 2.1 马尔可夫加噪链

给定一张真实图像 $\mathbf{x}_0 \sim q(\mathbf{x})$，前向过程通过 $T$ 步逐步加入高斯噪声：

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{1 - \beta_t} \,\mathbf{x}_{t-1}, \beta_t \mathbf{I} \right)
$$

其中 $\beta_t \in (0, 1)$ 是**方差调度（Variance Schedule）**，控制每步加入的噪声量。通常 $\beta_1 < \beta_2 < \dots < \beta_T$（越往后加噪越多）。

![图1: 前向扩散过程示意](https://img.yumeko.site/file/blog/DiffusionInference/ForwardProcess.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的科学示意图，展示扩散模型前向过程。
> 从左到右排列 6 个图像方块：x_0（清晰的猫照片）→ x_t1 → x_t2 → ... → x_T（纯高斯噪声）。
> 每个箭头标注 "q(x_t|x_{t-1})" 和噪声水平 β_t。
> 下方展示 β_t 从接近 0 增长到接近 1 的曲线图。
> 白色背景，柔和的蓝色调，教科书插图风格。
> ```

### 2.2 重参数化技巧：一步加噪

由于高斯分布的可加性，我们可以从 $\mathbf{x}_0$ 直接采样任意第 $t$ 步的 $\mathbf{x}_t$，无需逐步迭代：

记 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，则：

$$
\boxed{q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{\bar{\alpha}_t} \,\mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I} \right)}
$$

等价地：

$$
\boxed{\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})}
$$

> [!TIP] 关键直觉
> $t$ 很小 → $\bar{\alpha}_t \approx 1$ → $\mathbf{x}_t \approx \mathbf{x}_0$（信号占主导）
> $t$ 很大 → $\bar{\alpha}_t \approx 0$ → $\mathbf{x}_t \approx \boldsymbol{\epsilon}$（噪声占主导）
> $t=T$ → $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$（纯噪声）

```python
import numpy as np

def linearBetaSchedule(T=1000, betaStart=1e-4, betaEnd=0.02):
    """
    线性方差调度

    Args:
        T: 总步数
        betaStart: 起始 β 值
        betaEnd: 终止 β 值
    Returns:
        beta: 形状 (T,) 的噪声调度
        alphaBar: 形状 (T,) 的累积乘积
    """
    beta = np.linspace(betaStart, betaEnd, T)
    alpha = 1 - beta
    alphaBar = np.cumprod(alpha)
    return beta, alphaBar


def forwardDiffuse(x0, t, alphaBar):
    """
    前向扩散：从 x0 一步加到第 t 步

    Args:
        x0: 原始图像，形状 (batch, C, H, W)
        t: 时间步，形状 (batch,)
        alphaBar: 形状 (T,) 的累积 α_bar

    Returns:
        xt: 加噪后的图像
        noise: 加入的噪声（训练目标）
    """
    # 获取各样本对应的 alpha_bar
    alphaBarT = alphaBar[t]  # (batch,)
    alphaBarT = alphaBarT[:, None, None, None]  # reshape for broadcasting

    noise = np.random.randn(*x0.shape)
    xt = np.sqrt(alphaBarT) * x0 + np.sqrt(1 - alphaBarT) * noise
    return xt, noise
```

## 3. 反向去噪过程

### 3.1 反向条件分布

反向过程的目标：学习 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 来近似真实的反向分布 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$。

关键的数学事实：在给定 $\mathbf{x}_0$ 的条件下，反向条件分布 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 是**可解析计算的高斯分布**：

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I} \right)
$$

其中：

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

### 3.2 用噪声预测替代图像预测

实际训练中，我们不直接预测 $\mathbf{x}_0$，而是训练一个神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 来**预测加入的噪声**。将 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta)$ 代入 $\tilde{\boldsymbol{\mu}}_t$，可得：

$$
\boxed{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)}
$$

反向采样的一步为：

$$
\boxed{\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})}
$$

其中 $\sigma_t^2 = \tilde{\beta}_t$（在 DDPM 中通常取 $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta}_t$，两者差异很小）。

## 4. DDPM 采样算法

### 4.1 完整采样流程

```
算法: DDPM 采样（Ancestral Sampling）

输入: 噪声预测网络 ε_θ, 总步数 T
输出: 生成的图像 x_0

1. x_T ~ N(0, I)                    // 从纯噪声开始
2. for t = T, T-1, ..., 1:
3.     z ~ N(0, I) if t > 1 else 0  // 最后一步不加随机噪声
4.     x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
5. return x_0
```

### 4.2 Python 实现

```python
import torch
import torch.nn as nn

def ddimSampling(model, T, alphaBar, alpha, beta, shape, device):
    """
    DDPM 采样（完整 T 步去噪）

    Args:
        model: 噪声预测网络 ε_θ(xt, t)
        T: 总步数
        alphaBar: 累积 alpha_bar，形状 (T+1,)
        alpha: 单步 alpha，形状 (T+1,)
        beta: 单步 beta，形状 (T+1,)
        shape: 生成图像的形状 (batch, C, H, W)
        device: 设备
    Returns:
        生成的图像 x_0
    """
    model.eval()
    x = torch.randn(shape, device=device)  # x_T ~ N(0, I)

    for t in range(T, 0, -1):
        tTensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # 预测噪声
        with torch.no_grad():
            epsTheta = model(x, tTensor)  # ε_θ(x_t, t)

        # 系数计算
        sqrtAlphaT = torch.sqrt(alpha[t])
        sqrtOneMinusAlphaBarT = torch.sqrt(1 - alphaBar[t])

        # DDPM 均值
        mean = (x - (beta[t] / sqrtOneMinusAlphaBarT) * epsTheta) / sqrtAlphaT

        # 随机噪声（最后一步不加）
        if t > 1:
            sigma = torch.sqrt(beta[t])
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean

    return x
```

## 5. DDIM：确定性加速采样

### 5.1 动机

DDPM 的采样需要 $T$ 步（通常 $T=1000$），每步都要运行一次神经网络，推理速度极慢。DDIM（Denoising Diffusion Implicit Models）提出了一种**非马尔可夫**的采样方式，允许只用少量步骤（如 50-100 步）完成高质量采样。

### 5.2 DDIM 的数学原理

DDIM 的关键洞察：DDPM 的训练目标 $\mathcal{L}_{\text{simple}} = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$ **不依赖于**前向过程的马尔可夫性质，因此可以在推理时使用不同的（非马尔可夫）的反向过程。

DDIM 的采样公式：

$$
\boxed{\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的 } \mathbf{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}}
$$

其中：
- $\sigma_t = 0$ → **DDIM**（完全确定性，$\mathbf{x}_T$ 唯一确定 $\mathbf{x}_0$）
- $\sigma_t = \sqrt{(1 - \bar{\alpha}_{t-1})/(1 - \bar{\alpha}_t)} \sqrt{1 - \bar{\alpha}_t/\bar{\alpha}_{t-1}}$ → **DDPM**（随机采样）

> [!TIP] DDIM 为什么快？
> DDIM 可以选择 $T$ 的一个子序列（如等间隔取 50 步），在更少的步数下完成采样。
> 因为 DDIM 不依赖马尔可夫假设，跳步（非相邻时间步的采样）在数学上是合法的。

### 5.3 DDIM 跳跃采样

```python
def ddimSampling(model, alphaBar, steps=50, eta=0.0):
    """
    DDIM 加速采样

    Args:
        model: 噪声预测网络
        alphaBar: 形状 (T+1,) 的累积 alpha_bar（使用完整 1000 步调度）
        steps: 实际采样步数（<< 1000）
        eta: 随机性控制（0=DDIM确定, 1=DDPM随机）
    Returns:
        生成的图像
    """
    # 选择时间步子序列
    times = torch.linspace(1000, 1, steps, dtype=torch.long)

    x = torch.randn(batch, C, H, W)  # x_T

    for i in range(len(times) - 1):
        t = times[i]
        tPrev = times[i + 1]

        epsTheta = model(x, t)

        # 预测 x0
        alphaBarT = alphaBar[t]
        predX0 = (x - torch.sqrt(1 - alphaBarT) * epsTheta) / torch.sqrt(alphaBarT)

        # 确定 x_{t-1} 的方向
        alphaBarPrev = alphaBar[tPrev]
        sigma = eta * torch.sqrt((1 - alphaBarPrev) / (1 - alphaBarT)) \
                * torch.sqrt(1 - alphaBarT / alphaBarPrev)

        # DDIM 更新
        dirXt = torch.sqrt(1 - alphaBarPrev - sigma**2) * epsTheta
        x = torch.sqrt(alphaBarPrev) * predX0 + dirXt

        if eta > 0:
            x = x + sigma * torch.randn_like(x)

    return x
```

## 6. Classifier-Free Guidance（CFG）

### 6.1 动机

标准的无条件扩散模型生成的结果是随机的——你无法控制生成的内容。Classifier-Free Guidance（CFG）是当前最主流的条件生成方法，它不需要额外的分类器。

### 6.2 CFG 的核心公式

在推理时，CFG 将条件预测和无条件预测进行线性组合，以此"引导"生成朝向条件：

$$
\boxed{\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w \cdot \big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) \big)}
$$

简化形式：

$$
\boxed{\hat{\boldsymbol{\epsilon}}_\theta = (1 - w) \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})}
$$

其中：
- $\mathbf{c}$ 是条件（文本嵌入、类别标签等）
- $\emptyset$ 是空条件（无条件生成的噪声预测）
- $w \ge 1$ 是**引导尺度（Guidance Scale）**

> [!NOTE] Guidance Scale 的选择
> - $w = 1$：标准条件生成（无引导）
> - $w = 7.5$：Stable Diffusion 默认值——平衡生成质量与多样性
> - $w$ 太大（如 20+）：图像过度饱和、不自然，但更能遵循文本描述
> - $w = 0$：完全无条件生成

### 6.3 CFG 的实现

训练时随机将 10-20% 的样本的条件替换为空条件，使模型同时学习条件生成和无条件生成。推理时使用 CFG 公式组合两者。

```python
def classifierFreeGuidance(model, xt, t, textEmbedding, nullEmbedding, guidanceScale=7.5):
    """
    Classifier-Free Guidance 的噪声预测

    Args:
        model: 去噪网络
        xt: 当前噪声图像
        t: 时间步
        textEmbedding: 文本条件的嵌入
        nullEmbedding: 空文本的嵌入
        guidanceScale: CFG 引导尺度 w
    Returns:
        引导后的噪声预测
    """
    # 条件预测
    condPred = model(xt, t, textEmbedding)

    # 无条件预测
    uncondPred = model(xt, t, nullEmbedding)

    # CFG 线性组合
    return uncondPred + guidanceScale * (condPred - uncondPred)
```

### 6.4 CFG 的数学解释

Classifier-Free Guidance 可以理解为对条件分布的**温度调节**：

$$
p_w(\mathbf{x}_t \mid \mathbf{c}) \propto p(\mathbf{x}_t \mid \mathbf{c})^w \cdot p(\mathbf{x}_t)^{1-w}
$$

- $w > 1$ 使条件分布在采样中得到更高的"锐度"，生成结果更符合条件
- $w = 1$ 对应标准的条件采样

## 7. 主流采样器对比

| 采样器 | 类型 | 步数 | 随机性 | 特点 |
|:--|:--|:--|:--|:--|
| DDPM | 祖先采样 | 1000 | 随机 | 原始论文方法，慢但经典 |
| DDIM | 非马尔可夫 | 50-250 | 可选 | 加速采样，可插值 |
| PNDM | 数值方法 | 50 | 确定性 | 伪数值方法，质量好 |
| DPM-Solver | ODE 求解器 | 10-20 | 确定性 | 最快之一，数学优雅 |
| Euler Ancestral | 祖先采样 | 20-50 | 随机 | SDE 视角的简单实现 |
| UniPC | 多步预测 | 5-10 | 确定性 | 极快，精度高 |

> [!ABSTRACT] 选型建议
> - 日常使用：DDIM 50 步（质量与速度的最佳平衡）
> - 极致速度：DPM-Solver++ 2M 15 步 或 UniPC 5 步
> - 追求多样性：Euler Ancestral（保留随机性）

## 8. 完整的推理 Pipeline

以下是将各组件串联的完整推理流程，展示从文本 Prompt 到最终图像的端到端过程：逐步编码文本条件、从纯噪声初始化潜空间、CFG 引导去噪循环、最后解码到图像空间。

```python
@torch.no_grad()
def diffusionInferencePipeline(
    model,          # 去噪 UNet
    textEncoder,    # 文本编码器（如 CLIP）
    prompt,         # 文本提示词
    negativePrompt="",  # 负向提示词
    guidanceScale=7.5,  # CFG 引导尺度
    numSteps=50,        # 采样步数
    sampler="ddim",     # 采样器类型
    seed=42,            # 随机种子
):
    """
    完整的扩散模型推理流程

    Args:
        model: 去噪 UNet 模型
        textEncoder: 文本编码器
        prompt: 正向提示词
        negativePrompt: 负向提示词（CFG 用）
        guidanceScale: CFG 引导尺度
        numSteps: 采样步数
        sampler: 采样器名称
        seed: 随机种子
    Returns:
        生成的图像
    """
    torch.manual_seed(seed)

    textEmbed = textEncoder(prompt)
    nullEmbed = textEncoder(negativePrompt) if negativePrompt else textEncoder("")

    x = torch.randn(1, 4, 64, 64)  # 潜空间初始化（Stable Diffusion 示例维度）

    timesteps = torch.linspace(1000, 1, numSteps, dtype=torch.long)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        tNext = timesteps[i + 1]

        noisePredCond = model(x, t, textEmbed)
        noisePredUncond = model(x, t, nullEmbed)
        noisePred = noisePredUncond + guidanceScale * (noisePredCond - noisePredUncond)

        alphaBarT = alphaBar[t]
        alphaBarNext = alphaBar[tNext]

        x0Pred = (x - torch.sqrt(1 - alphaBarT) * noisePred) / torch.sqrt(alphaBarT)
        x = torch.sqrt(alphaBarNext) * x0Pred + torch.sqrt(1 - alphaBarNext) * noisePred

    image = vaeDecoder(x)
    return image
```

---

> **相关文章**：
> - [[Math/ProbabilityInequalities|概率不等式完全指南]]
> - [[NeuralNetwork/LLM/RLHF|RLHF 算法详解]]
> - [[NeuralNetwork/RNN/Attention|注意力机制详解]]
