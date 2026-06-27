---
title: RLHF 算法详解：从 PPO 到 DPO 的对齐技术演进
date: 2026-06-05
category: 神经网络/LLM
tags:
  - RLHF
  - 对齐
  - PPO
  - DPO
  - 大语言模型
description: 系统讲解 RLHF（基于人类反馈的强化学习）的完整三阶段流程：监督微调、奖励模型训练、PPO 强化学习优化，并深入对比 DPO 如何绕过奖励模型直接利用偏好数据，附完整的数学推导和 Python 实现示例。
image: https://img.yumeko.site/file/blog/articles/1780733778546.webp
status: draft
---

> **前置阅读**：本文假定读者熟悉 LLM 的基本概念。

## 1. 为什么需要 RLHF？

预训练的大语言模型是"自回归补全引擎"——它们学会了预测下一个 Token，但**没有学会"什么是有用的回答"**。

给定同样的 Prompt，模型可能：
- 给出有帮助的答案（desired）
- 给出有害/有偏见的回答（undesired）
- 编造不存在的事实（hallucination）
- 拒绝合理请求或接受不合理请求

RLHF（Reinforcement Learning from Human Feedback）的目标是**让模型的输出与人类偏好对齐**——让模型学会给出人类认为"好"的回答。

> [!NOTE] 对齐（Alignment）的三个维度
> Anthropic 的 HH 标准：
> - **Helpful（有帮助）**：回答准确、完整、有用
> - **Honest（诚实）**：不编造信息，承认不确定性
> - **Harmless（无害）**：不产生有害、歧视、危险内容

## 2. RLHF 的三阶段流程

RLHF 的核心流程包括三个步骤：

```
阶段 1: 监督微调（SFT）
  -> 用高质量对话数据微调预训练模型

阶段 2: 训练奖励模型（Reward Model, RM）
  -> 用人类偏好比较数据训练一个能"打分"的模型

阶段 3: 用 PPO 优化策略
  -> 以 RM 为奖励信号，用强化学习优化语言模型
```

![ThreeStageFlow.png](https://img.yumeko.site/file/blog/articles/1780733773680.webp)

## 3. 阶段 1：监督微调（Supervised Fine-Tuning, SFT）

### 3.1 目标

让预训练模型学会**对话格式**和基本的指令遵循能力。

### 3.2 数据格式

从人类标注者收集高质量的 (Prompt, Response) 对。Prompt 是用户问题，Response 是人类标注者写的理想回答。

```
数据集 D_SFT = {(x_i, y_i)} 其中：
- x_i: Prompt（用户输入）
- y_i: Response（人类标注的高质量回答）
```

### 3.3 训练目标

SFT 阶段使用标准的语言建模损失（交叉熵）：

$$
\boxed{\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]}
$$

这只优化 Response 部分的 Token，Prompt 部分不计算损失。

```python
def sftLoss(model, promptIds, responseIds):
    """
    计算 SFT 损失（仅优化 Response 部分）

    Args:
        model: 语言模型
        promptIds: 形状 (batch, prompt_len) 的 Prompt Token IDs
        responseIds: 形状 (batch, response_len) 的 Response Token IDs
    Returns:
        标量损失值
    """
    # 拼接成完整序列
    inputIds = torch.cat([promptIds, responseIds], dim=1)

    # 前向传播获取 logits
    logits = model(inputIds)  # (batch, seq_len, vocab_size)

    # 创建 mask：仅计算 Response 部分的损失
    promptLen = promptIds.shape[1]
    lossMask = torch.zeros_like(inputIds)
    lossMask[:, promptLen:] = 1  # Response 部分置 1

    # 交叉熵损失（带 mask）
    shiftLogits = logits[:, :-1, :].contiguous()
    shiftLabels = inputIds[:, 1:].contiguous()
    shiftMask = lossMask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shiftLogits.view(-1, shiftLogits.size(-1)),
        shiftLabels.view(-1),
        reduction='none'
    )
    loss = (loss * shiftMask.view(-1)).sum() / shiftMask.sum()
    return loss
```

> [!TIP] 为什么只优化 Response 部分？
> - Prompt 是已知条件，模型只需要学习"给定 Prompt 后如何生成好的 Response"
> - 让模型也优化 Prompt 部分相当于让它"学习提问"，不是本阶段目标

## 4. 阶段 2：训练奖励模型（Reward Model, RM）

### 4.1 为什么需要奖励模型？

我们不能让人类实时给模型的每个输出打分（太慢、太贵）。所以先用人类偏好数据训练一个**奖励模型**来模拟人类的打分偏好。

**奖励模型的本质**：奖励模型是一个**可微分的人类偏好代理**（differentiable proxy of human preferences）。这个定义包含三层含义：

- **代理（Proxy）**：奖励模型本身不直接优化语言模型——它只负责对回答质量进行打分。实际执行 RL 优化的是阶段 3 中的 PPO 算法。奖励模型充当的是"裁判"而非"教练"。
- **可微分（Differentiable）**：奖励模型输出的标量分数是连续且可导的。这使得它能够作为 RL 优化中的奖励信号 $r_\phi(x, y)$，通过策略梯度方法反向传播，驱动语言模型向高质量回答方向更新。
- **人类偏好（Human Preferences）**：奖励模型从人类标注的偏好排序数据中学习——它学会的是"判断哪个回答更好"（相对比较），而非"给回答打绝对分数"。

> [!IMPORTANT] 关于奖励模型的常见误解
> 以下理解是**错误**的，在笔试中需特别注意：
> - ❌ 奖励模型本身就是一个强化学习智能体 → 它不执行动作，不与环境交互。在 RLHF 中，**智能体是语言模型本身**，奖励模型仅是环境的一部分
> - ❌ 奖励模型的目标是学习模仿人类的写作风格 → 那是 SFT（阶段 1）的任务。RM 的目标是学习"评判回答质量的好坏"
> - ❌ 奖励模型直接修改基座模型的权重 → RM 在 PPO 阶段是**冻结**的，仅用于前向打分。真正更新权重的是策略模型 $\pi_\theta$

> [!TIP] 延伸阅读
> 关于奖励模型在 RLHF 中的角色定位，是华为 AI 岗校招笔试的高频考点。详见 [[Internship/HUAWEI/2026-03-18|华为2026-03-18 机试真题（AI方向）]] 选择题第 3 题。

### 4.2 偏好数据收集

对同一个 Prompt $x$，让模型生成两个回答 $y_A$ 和 $y_B$（或者用不同模型生成），人类标注者选择一个"更好的"：

```
Prompt x: "如何学习机器学习？"
Response y_A: "可以从吴恩达的课程开始，然后做 Kaggle 项目..."  <- 人类选了 A
Response y_B: "就去学呗，看书啊。"                              <- 人类没选 B
```

### 4.3 奖励模型的训练目标

奖励模型 $r_\phi(x, y)$ 输入 Prompt + Response，输出一个标量分数（越高越好）。

训练目标使用 Bradley-Terry 偏好模型——人类偏好 $y_a \succ y_b$ 的概率为：

$$
P(y_a \succ y_b \mid x) = \frac{\exp(r_\phi(x, y_a))}{\exp(r_\phi(x, y_a)) + \exp(r_\phi(x, y_b))} = \sigma\big(r_\phi(x, y_a) - r_\phi(x, y_b)\big)
$$

其中 $\sigma$ 是 Sigmoid 函数。最终奖励模型的损失函数为：

$$
\boxed{\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_a, y_b) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\big(r_\phi(x, y_a) - r_\phi(x, y_b)\big) \right]}
$$

直觉：如果人类选了 $y_a$（即 $r_\phi(x, y_a)$ 应当远大于 $r_\phi(x, y_b)$），那么 $\log \sigma(\text{差值})$ 应当接近 0（损失小）。

```python
def rewardModelLoss(rmModel, prompts, responseChosen, responseRejected):
    """
    计算奖励模型的 Pairwise 排序损失

    Args:
        rmModel: 奖励模型
        prompts: 形状 (batch, prompt_len) 的 Prompt
        responseChosen: 形状 (batch, resp_len) 的被选回答
        responseRejected: 形状 (batch, resp_len) 的未被选回答
    Returns:
        标量损失值
    """
    # 对两个回答分别打分
    scoreChosen = rmModel(prompts, responseChosen)     # (batch,)
    scoreRejected = rmModel(prompts, responseRejected)  # (batch,)

    # 分数差 -> Sigmoid -> 负对数似然
    diff = scoreChosen - scoreRejected
    loss = -F.logsigmoid(diff).mean()
    return loss
```

**奖励模型的实现细节**：
- 通常在 SFT 模型的基础上，将最后的 LM Head 替换为一个标量输出头（如一个线性层 `hidden_dim -> 1`）
- 奖励模型的最后一个 Token 的隐藏状态作为整条回答的表示，然后映射到标量分数

## 5. 阶段 3：用 PPO 优化策略

### 5.1 策略优化问题建模

现在有了奖励模型 $r_\phi$，希望优化语言模型（策略 $\pi_\theta$）以最大化来自奖励模型的预期奖励，同时约束策略不要偏离 SFT 模型太远。

完整的优化目标：

$$
\boxed{\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \cdot \operatorname{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\big) \right]}
$$

解释：
- 第一项 $r_\phi(x, y)$：最大化奖励模型给出的分数（"回答好"）
- 第二项 KL 散度惩罚：防止新策略与 SFT 模型偏差过远（防止模型"走火入魔"，例如为追求高分生成无意义文本）
- $\beta > 0$ 控制偏离程度

### 5.2 PPO（Proximal Policy Optimization）

PPO 是 RLHF 中最常用的 RL 算法。核心思想是通过 **clipping** 机制限制每次更新的幅度，防止策略崩溃：

$$
\boxed{\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \; \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]}
$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$ 是新旧策略的概率比，$\hat{A}_t$ 是**优势函数**（Advantage）。

**优势函数的计算**（在 RLHF 中使用 GAE, Generalized Advantage Estimation）：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是时序差分误差（TD Error），$V(s)$ 是价值函数。

> [!NOTE] 为什么在 RLHF 中使用 PPO 而非更简单的 RL 算法？
> 1. PPO 的 clipping 机制天然适合语言模型——限制策略变化幅度防止语言能力退化
> 2. PPO 在 NLP 任务上已被 TRL 等库广泛验证和实现
> 3. PPO 是策略梯度（Policy Gradient）方法，直接对概率分布优化，天然适合大词表场景

**为什么不选其他 RL 算法？**

| 算法 | 类型 | 能否用于 LLM？ | 原因 |
|:--|:--|:--:|:--|
| DQN | 值函数（Q-Learning） | ✗ | 为**离散小动作空间**设计（如 Atari 上下左右 4 个按钮）。LLM 每步要从 50000+ 词表中选 Token，为每个动作估算 Q 值不可行 |
| REINFORCE | 策略梯度 | 勉强 | 无 clipping，高方差，训练极不稳定。LLM 的序列长度和词表规模会放大方差 |
| A2C/A3C | Actor-Critic | 勉强 | 需要独立的 Critic 网络，额外复杂度；无 clipping 导致策略容易崩溃 |
| SAC | Off-policy Actor-Critic | ✗ | 连续动作空间设计，不匹配 Token 采样的离散性 |
| **PPO** | 策略梯度 + Clipping | ✅ | 策略梯度适配大词表 + clipping 防崩溃 + GAE 降方差，三合一匹配 LLM |

一个常见的笔试陷阱是把"最大化人类偏好得分"当成算法——这是 RLHF 的**目标**，不是实现方法。SFT 是 RLHF 的**前置步骤**，不是 RLHF 阶段本身。DQN 虽然也是强化学习算法，但设计场景完全不同，不能用于 LLM。

### 5.3 RLHF 中的 PPO 训练循环

```python
for batch in dataloader:
    prompts = batch['prompt']

    with torch.no_grad():
        responses = policyModel.generate(prompts)
        oldLogProbs = policyModel.logProb(prompts, responses)

    rewards = rewardModel(prompts, responses)

    klPenalty = computeKL(policyModel, sftModel, prompts, responses)
    rewards = rewards - beta * klPenalty

    advantages = computeGAE(rewards, valueModel, gamma=0.99, lam=0.95)

    for _ in range(ppoEpochs):
        newLogProbs = policyModel.logProb(prompts, responses)
        ratio = torch.exp(newLogProbs - oldLogProbs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policyLoss = -torch.min(surr1, surr2).mean()

        policyLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

    valueLoss = F.mse_loss(valueModel(prompts, responses), returns)
    valueLoss.backward()
    valueOptimizer.step()
```

## 6. DPO：绕过奖励模型的直接偏好优化

### 6.1 PPO 的问题

PPO 虽然有效，但引入了一系列工程复杂度：
- 需要训练一个独立的奖励模型
- 需要维护价值模型
- 需要在线采样（生成回答 -> 打分 -> 更新）
- 训练不稳定，对超参数敏感

### 6.2 DPO 的核心洞察

Rafailov et al. (2023) 发现：RLHF 的 KL 约束优化问题存在**闭式解**。最优策略 $\pi^*$ 可表示为：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

反解出奖励函数：$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$。

代入 Bradley-Terry 偏好模型中，$Z(x)$ 项互相抵消，得到 DPO 损失：

$$
\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_a, y_b) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_a \mid x)}{\pi_{\text{ref}}(y_a \mid x)} - \beta \log \frac{\pi_\theta(y_b \mid x)}{\pi_{\text{ref}}(y_b \mid x)} \right) \right]}
$$

**含义**：DPO 直接用偏好数据 $\{(x, y_a, y_b)\}$ 优化策略模型。不需要单独的奖励模型，不需要 PPO 采样——将 RLHF 化简为**一个分类问题**。

### 6.3 DPO 实现要点

```python
def dpoLoss(policyModel, refModel, prompts, responseChosen, responseRejected, beta=0.1):
    """
    计算 DPO 损失

    Args:
        policyModel: 正在优化的策略模型（pi_theta）
        refModel: 冻结的参考模型（通常是 SFT 模型，pi_ref）
        prompts: Prompt Token IDs
        responseChosen: 被选中的回答
        responseRejected: 被拒绝的回答
        beta: KL 惩罚系数
    Returns:
        标量损失值
    """
    # 计算策略模型下的对数概率
    policyChosenLogProb = policyModel.logProb(prompts, responseChosen)
    policyRejectedLogProb = policyModel.logProb(prompts, responseRejected)

    # 计算参考模型下的对数概率（冻结，不计算梯度）
    with torch.no_grad():
        refChosenLogProb = refModel.logProb(prompts, responseChosen)
        refRejectedLogProb = refModel.logProb(prompts, responseRejected)

    # 对数比
    chosenLogRatio = policyChosenLogProb - refChosenLogProb
    rejectedLogRatio = policyRejectedLogProb - refRejectedLogProb

    # DPO 损失
    loss = -F.logsigmoid(beta * (chosenLogRatio - rejectedLogRatio)).mean()
    return loss
```

## 7. PPO vs DPO 对比

| 维度 | PPO（RLHF） | DPO |
|:--|:--|:--|
| 是否需要奖励模型 | 是（阶段 2 独立训练） | 否（隐式奖励） |
| 是否需要在线采样 | 是（每步生成新回答） | 否（离线偏好数据即可） |
| 训练稳定性 | 敏感，需要调参 | 较稳定（纯监督学习） |
| 计算成本 | 高（4 个模型：策略、参考、奖励、价值） | 低（2 个模型：策略、参考） |
| 理论优雅性 | 较工程化 | 数学上更优雅 |
| 适用场景 | 有在线偏好收集能力 | 仅有离线偏好数据集 |
| 对齐效果 | 验证成熟 | 理论上等价，实际可比 |

## 8. RLHF 的前沿进展

- **RLAIF**（RL from AI Feedback）：用 AI 模型（如强 LLM）替代人类标注者提供偏好反馈，降低成本
- **KTO**（Kahneman-Tversky Optimization）：只需单一回答的"好/坏"标签，不需要配对比较
- **SimPO**：以生成序列的长度归一化概率作为隐式奖励，简化 DPO
- **SPIN**：自我博弈（Self-Play）迭代提升，模型自己生成偏好数据

---

> **相关文章**：
> - [[NeuralNetwork/RNN/Attention|注意力机制详解]]
> - [[NeuralNetwork/Transformer/TransformerOverview|Transformer 架构总览]]
