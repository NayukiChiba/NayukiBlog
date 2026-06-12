---
title: Q-learning 算法详解
date: 2026-06-06
category: 机器学习/强化学习
tags:
  - 强化学习
  - Q-learning
  - 时序差分
  - MDP
description: 从 MDP 基础出发，深入讲解 Q-learning 的更新公式、TD 误差、算法流程与具体计算示例，结合华为机试真题逐项分析。
image: https://img.yumeko.site/file/blog/cover/1780746817242.webp
status: published
---

> **前置阅读**：本文假定读者了解概率论基础和基本的监督学习概念。若对 MDP 背景不熟悉，建议先浏览第 2 节的 MDP 回顾。

## 1. 问题的起点

强化学习要解决一个根本问题：**智能体（agent）如何在与环境交互中学会做决策**。

想象你第一次玩一款电子游戏：你不知道哪个按钮做什么，不知道哪里有奖励。你只能**试错**——按一下按钮，看看发生了什么（奖励或惩罚），然后调整下一步的选择。经过足够多次尝试，你逐渐学会在什么状态下该按什么按钮。

Q-learning 就是这个"试错学习"过程的数学形式化。它是强化学习中最经典的**无模型（model-free）**算法之一，由 Watkins 于 1989 年提出。

> [!NOTE] 关键前提
> Q-learning 不需要环境模型（即不需要知道状态转移概率 $P(s' \mid s, a)$），
> 仅通过与环境交互获得的 $(s, a, r, s')$ 样本来学习。

## 2. MDP 基础回顾

Q-learning 建立在**马尔可夫决策过程**(Markov Decision Process, MDP)之上。一个有限 MDP 由五元组定义：

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
$$

| 符号 | 含义 | 示例 |
|:--|:--|:--|
| $\mathcal{S}$ | 状态集合 | 网格地图的每个格子 |
| $\mathcal{A}$ | 动作集合 | 上/下/左/右 |
| $P(s' \mid s, a)$ | 状态转移概率 | 执行"右移"有 80% 概率成功 |
| $R(s, a)$ | 即时奖励函数 | 到达终点 +10，撞墙 -1 |
| $\gamma \in [0,1]$ | 折扣因子 | $0.9$，权衡短期与长期回报 |

### 2.1 策略与价值函数

**策略** $\pi(a \mid s)$ 定义了在每个状态 $s$ 下选择各动作 $a$ 的概率。

**状态价值函数** $V^\pi(s)$：从状态 $s$ 出发，始终遵循策略 $\pi$ 的期望累积折扣回报：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \;\middle|\; S_0 = s \right]
$$

**动作价值函数** $Q^\pi(s, a)$：从状态 $s$ 出发，先执行动作 $a$，之后遵循策略 $\pi$ 的期望累积折扣回报：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \;\middle|\; S_0 = s, A_0 = a \right]
$$

### 2.2 Bellman 最优方程

最优 Q 函数 $Q^*(s, a)$ 满足 Bellman 最优方程：

$$
\boxed{Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \cdot \max_{a'} Q^*(s', a')}
$$

含义：最优动作价值 = 即时奖励 + 折扣后未来所有可能状态的最优价值的期望。

> [!TIP] 记忆技巧
> Bellman 最优方程是 Q-learning 更新公式的**理论源头**。Q-learning 的每次更新，
> 本质是用一个实际采样 $(s, a, r, s')$ 来**近似**这条方程。

## 3. Q-learning 核心思想

### 3.1 从 Bellman 方程到 TD 更新

Bellman 最优方程需要知道 $P(s' \mid s, a)$（环境模型），而 Q-learning 不需要。它用**单次实际采样**代替期望计算：

$$
\text{TD 目标} = r + \gamma \cdot \max_{a'} Q(s', a')
$$

然后用这个 TD 目标去修正当前的 $Q(s, a)$：

$$
\boxed{Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD 目标}} - \underbrace{Q(s, a)}_{\text{当前估计}} \right]}
$$

> [!NOTE] TD 方法的直觉
> 不等到 episode 结束再更新（Monte Carlo 方法），也不依赖环境模型（动态规划），
> 而是**每走一步就立即用下一步的估计来修正当前步**。这就是"时序差分"（Temporal Difference）的核心。

### 3.2 Off-policy 特性

Q-learning 是 **off-policy** 算法——这意味着：

- **行为策略（behavior policy）**：智能体实际用于探索的策略（如 $\varepsilon$-greedy）
- **目标策略（target policy）**：智能体正在学习的最优策略（由 $\max_{a'} Q(s', a')$ 体现）

两者可以不同。更新时使用的 $\max_{a'}$ 是"假设我们从 $s'$ 开始会选最优动作"，而不是"实际选了哪个动作"。这使得 Q-learning 可以从任意探索策略产生的数据中学习最优策略。

### 3.3 更新公式逐项解读

把公式拆开，每一步都有明确的数学含义：

$$
Q_{\text{new}}(s, a) = (1 - \alpha) \cdot Q_{\text{old}}(s, a) + \alpha \cdot \left( r + \gamma \max_{a'} Q(s', a') \right)
$$

上式是原公式的代数等价形式，它更直观地揭示了更新的本质：

> Q-learning 的每次更新 = **旧估计保留 $(1-\alpha)$ + TD 目标贡献 $\alpha$**

这是一种**指数移动平均（exponential moving average）**——$\alpha$ 越大，新数据权重越大，学习越快但不稳定；$\alpha$ 越小，旧经验保留越多，学习越慢但更稳定。

| 参数 | 作用 | 典型值 |
|:--|:--|:--|
| $\alpha$ | 学习率：控制"信任新数据"的程度 | $0.01 \sim 0.5$ |
| $\gamma$ | 折扣因子：控制"重视未来"的程度 | $0.9 \sim 0.99$ |

- $\gamma$ 接近 $0$ → 短视，只看眼前奖励
- $\gamma$ 接近 $1$ → 远见，看重长期累积回报

## 4. 算法流程

**输入**：学习率 $\alpha$、折扣因子 $\gamma$、探索率 $\varepsilon$、总 episode 数

**输出**：最优 Q 表

1. 初始化 $Q(s, a) = 0$ 对所有 $s \in \mathcal{S}, a \in \mathcal{A}$
2. 对每个 episode：
   - 初始化状态 $s$
   - 对 episode 中的每一步：
    > 1. 用 $\varepsilon$-greedy 策略选择动作 $a$（以 $\varepsilon$ 概率随机探索，$1-\varepsilon$ 概率选 $\max Q$）
    > 2. 执行动作 $a$，观测奖励 $r$ 和下一状态 $s'$
    > 3. 更新：$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a) \right]$
    > 4. $s \leftarrow s'$
   - 若 $s$ 为终态，结束本 episode

### 4.1 ε-greedy 探索策略

纯粹的 greedy（始终选 $\max Q$）会导致探索不足——智能体可能永远发现不了更好的路径。ε-greedy 是最简单的解决方式：

$$
a = \begin{cases}
\argmax_{a} Q(s, a) & \text{以概率 } 1 - \varepsilon \\
\text{随机选择一个动作} & \text{以概率 } \varepsilon
\end{cases}
$$

通常 $\varepsilon$ 从较大值（如 $1.0$）开始逐步衰减到较小值（如 $0.01$），实现"先探索、后利用"。

> [!ATTENTION] 注意
> ε 衰减过快可能导致探索不足，过早"锁定"在次优策略；衰减过慢则浪费探索时间。

## 5. 数值示例：笔试题详解

以下是 [[Internship/HUAWEI/2026-03-18|华为 2026 年 3 月机试]]中关于 Q-learning 的真题：

> **题目**：MDP 中 Q-learning 更新：$s_1 \to a_1 \to s_2$（终态），$r=3$，$\gamma=0.9$，$\alpha=0.1$，初始 $Q=0$。更新后 $Q(s_1,a_1)=$？

### 5.1 分析

这是最简单的单步更新场景。状态转移路径：$s_1$ 执行 $a_1$，获得奖励 $r=3$，转移到终态 $s_2$（episode 结束）。

关键信息汇总：

| 变量 | 值 | 说明 |
|:--|:--|:--|
| $Q_{\text{old}}(s_1, a_1)$ | $0$ | 初始 Q 值全零 |
| $r$ | $3$ | 本次动作获得的奖励 |
| $s'$ | $s_2$ | 下一状态 |
| $s_2$ 是否为终态 | 是 | 终态没有后续动作可选 |

### 5.2 代入计算

由于 $s_2$ 是终态，没有后续动作，因此 $\max_{a'} Q(s_2, a') = 0$：

$$
\begin{aligned}
Q_{\text{new}}(s_1, a_1) &= Q_{\text{old}}(s_1, a_1) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s_2, a') - Q_{\text{old}}(s_1, a_1) \right] \\[4pt]
&= 0 + 0.1 \times (3 + 0.9 \times 0 - 0) \\[4pt]
&= 0.1 \times 3 \\[4pt]
&= 0.3
\end{aligned}
$$

**答案：B. $0.3$**

### 5.3 常见错误

| 错误答案 | 可能的错误思路 |
|:--|:--|
| $0.4$ | 可能用了 $\alpha r + \alpha\gamma = 0.3 + 0.09 = 0.39 \approx 0.4$ |
| $0.2$ | 可能误用 $\alpha(r - \gamma) = 0.1 \times 2.1 = 0.21 \approx 0.2$ |
| $0.5$ | 可能用了 $\alpha r / (1 - \gamma) = 0.3 / 0.1 = 3$ 或其他错误公式 |

### 5.4 延伸思考

如果 $s_2$ **不是**终态怎么办？假设 $s_2$ 有两个可选动作 $a_1, a_2$，其 Q 值分别为 $0.5$ 和 $0.2$：

$$
\begin{aligned}
Q_{\text{new}}(s_1, a_1) &= 0 + 0.1 \times (3 + 0.9 \times \max(0.5, 0.2) - 0) \\
&= 0.1 \times (3 + 0.9 \times 0.5) \\
&= 0.1 \times 3.45 \\
&= 0.345
\end{aligned}
$$

TD 目标从 $3$ 变为了 $3.45$——多出来的 $0.45$ 正是未来回报的折现值。这就是 Q-learning "向后传播奖励"的机制：**终态的奖励通过一次次更新逐步向前传导**。

## 6. 完整的 GridWorld 示例

下面的例子展示 Q-learning 如何在一个 $3 \times 3$ 网格中学会找路径。

### 6.1 环境设置

考虑一个 $3 \times 3$ 的网格世界：

- **起点** $(0,0)$（左上角），**终点** $(2,2)$（右下角），奖励 $+10$
- **障碍物** $(1,1)$（正中间），不可穿越
- **动作**：上、下、左、右四个方向
- **每步奖励** $-0.1$（鼓励走最短路径）
- **撞墙/撞障碍物**：留在原地，奖励 $-1$
- **到达终点**：episode 结束

### 6.2 训练过程解读

初始时 Q 表全零，智能体随机游走。某次偶然到达终点附近：

- **episode 100**：智能体在 $(2,1)$ 执行"右移" → 到达终点 $(2,2)$，$r=10$

  $Q((2,1), \text{右}) = 0 + \alpha(10 + 0 - 0) = \alpha \times 10$

- **episode 150**：智能体在 $(2,0)$ 执行"右移" → 到 $(2,1)$，$r=-0.1$

  $Q((2,0), \text{右}) = 0 + \alpha(-0.1 + \gamma \cdot \max Q((2,1), \cdot) - 0) = \alpha(-0.1 + \gamma \cdot \alpha \cdot 10)$

终点的奖励 $10$ 通过 $\gamma$ 的折扣逐步向前传导：$(2,1) \to (2,0) \to (1,0) \to (0,0)$。经过足够多 episode 后，Q 表收敛，智能体学会走最短路径（绕开障碍物）。

## 7. 代码实现

```python
"""
Q-learning 算法实现

功能：
1. GridWorld 环境定义
2. ε-greedy 策略
3. Q-learning 训练与可视化
"""

import numpy as np
from typing import Tuple, List


class GridWorld:
    """简单的网格世界环境"""

    def __init__(self, size: int = 3):
        """
        Args:
            size: 网格边长
        """
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(1, 1)} if size == 3 else set()
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上/下/左/右
        self.state = None
        self.reset()

    def reset(self) -> Tuple[int, int]:
        """重置环境到起始状态"""
        self.state = self.start
        return self.state

    def step(self, actionIdx: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作

        Args:
            actionIdx: 动作索引 (0=上, 1=下, 2=左, 3=右)

        Returns:
            (nextState, reward, done)
        """
        dr, dc = self.actions[actionIdx]
        r, c = self.state
        nr, nc = r + dr, c + dc

        # 碰边界或障碍物则留在原地
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            self.state = (r, c)
            return self.state, -1.0, False
        if (nr, nc) in self.obstacles:
            self.state = (r, c)
            return self.state, -1.0, False

        self.state = (nr, nc)

        # 到达终点
        if self.state == self.goal:
            return self.state, 10.0, True
        return self.state, -0.1, False


def trainQlearning(
    env: GridWorld,
    numEpisodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsStart: float = 1.0,
    epsEnd: float = 0.01,
    epsDecay: float = 0.995,
) -> np.ndarray:
    """
    Q-learning 训练

    Args:
        env: GridWorld 环境实例
        numEpisodes: 训练 episode 数
        alpha: 学习率
        gamma: 折扣因子
        epsStart: 初始探索率
        epsEnd: 最终探索率
        epsDecay: 探索率衰减因子

    Returns:
        形状 (size, size, 4) 的 Q 表
    """
    size = env.size
    qTable = np.zeros((size, size, 4))  # Q(s, a)
    epsilon = epsStart

    for episode in range(numEpisodes):
        state = env.reset()
        done = False

        while not done:
            r, c = state

            # ε-greedy 选择动作
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = int(np.argmax(qTable[r, c]))

            nextState, reward, done = env.step(action)
            nr, nc = nextState

            # Q-learning 核心更新
            tdTarget = reward + gamma * np.max(qTable[nr, nc])
            tdError = tdTarget - qTable[r, c, action]
            qTable[r, c, action] += alpha * tdError

            state = nextState

        # 衰减探索率
        epsilon = max(epsEnd, epsilon * epsDecay)

    return qTable


def extractPolicy(qTable: np.ndarray) -> np.ndarray:
    """
    从 Q 表中提取最优策略

    Args:
        qTable: Q 表，形状 (size, size, 4)

    Returns:
        形状 (size, size) 的策略，值为最优动作索引
    """
    return np.argmax(qTable, axis=2)


def main():
    """训练并展示结果"""
    env = GridWorld(size=3)
    qTable = trainQlearning(env, numEpisodes=500)
    policy = extractPolicy(qTable)

    actionSymbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    print("最优策略（上/下/左/右）：")
    for r in range(env.size):
        row = []
        for c in range(env.size):
            if (r, c) == env.goal:
                row.append("G")
            elif (r, c) in env.obstacles:
                row.append("█")
            else:
                row.append(actionSymbols[policy[r, c]])
        print(" ".join(row))


if __name__ == "__main__":
    main()
```

运行上述代码，训练后的 Q 表将引导智能体从起点 $(0,0)$ 绕开障碍物到达终点 $(2,2)$。

## 8. 收敛性与讨论

### 8.1 收敛条件

Q-learning 在以下条件下能以概率 $1$ 收敛到 $Q^*$：

1. **所有状态-动作对被无限次访问**（要求充分的探索）
2. **学习率满足 Robbins-Monro 条件**：$\sum_{t=0}^\infty \alpha_t = \infty$，$\sum_{t=0}^\infty \alpha_t^2 < \infty$（如 $\alpha_t = 1/t$）
3. **环境是有限 MDP**

实践中常用常数 $\alpha$（如 $0.1$），虽不严格满足条件 2，但在平稳环境中依然有效。

### 8.2 局限性

| 问题 | 说明 | 解决方案 |
|:--|:--|:--|
| 维度灾难 | Q 表大小 = $\|\mathcal{S}\| \times \|\mathcal{A}\|$，状态空间大时不可行 | DQN（用神经网络近似 Q） |
| 离散动作 | 原生只支持离散动作空间 | DDPG、SAC 等连续动作算法 |
| 最大化偏差 | $\max$ 操作导致价值高估 | Double Q-learning |

> [!NOTE] 与 DQN 的关系
> Deep Q-Network（DQN）本质是 Q-learning + 深度神经网络。核心更新公式保持不变，
> 只是用网络参数 $\theta$ 替代 Q 表：$Q(s, a; \theta) \leftarrow r + \gamma \max_{a'} Q(s', a'; \theta^-)$。
> 详见 [[NeuralNetwork/RLHF|RLHF 中的强化学习]]。

## 9. 总结

| 要点 | 说明 |
|:--|:--|
| 核心公式 | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| 算法类型 | 无模型（model-free）、off-policy、时序差分（TD） |
| TD 误差 | $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$，衡量"预期 vs 实际"的偏差 |
| 终态处理 | $s'$ 为终态时 $\max_{a'} Q(s', a') = 0$，更新退化为 $Q \leftarrow Q + \alpha(r - Q)$ |
| $\alpha$ 作用 | 控制学习速度：大则快但不稳，小则慢但稳 |
| $\gamma$ 作用 | 控制远见程度：接近 0 短视，接近 1 远见 |
| 探索策略 | ε-greedy：以 ε 概率随机探索，$1-\varepsilon$ 概率利用 |
| 收敛前提 | 充分探索 + 合适的学习率衰减 |

> **相关文章**：
> - [[Internship/HUAWEI/2026-03-18|华为 2026-03-18 机试真题]]
> - [[NeuralNetwork/RLHF|RLHF：基于人类反馈的强化学习]]
> - [[MachineLearning/probabilistic/HMM|隐马尔可夫模型（HMM）]]
