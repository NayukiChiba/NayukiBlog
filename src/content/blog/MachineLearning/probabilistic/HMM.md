---
title: HMM
date: 2026-03-14
category: 机器学习/概率模型
tags:
  - Scikit-learn
  - 高级教程
description: HMM隐马尔可夫模型的数学原理、三大算法与完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581779359.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 HMM 的概率生成过程——隐状态按马尔可夫链演化，观测由当前隐状态发射。
2. 理解三大算法的数学本质——Forward（评估，求和）、Viterbi（解码，取最大）、Baum-Welch（学习，EM 迭代）。
3. 把这些数学表达和当前源码中的 `n_components`、`predict(...)`、`transmat_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| HMM 五元组 | 模型定义 | $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$——完整描述离散 HMM 的概率参数 |
| 马尔可夫假设 | 核心假设 | $P(q_t \mid q_{t-1}, \dots, q_1) = P(q_t \mid q_{t-1})$——未来仅依赖当前，与历史无关 |
| Forward 算法 | 评估算法 | 计算 $P(\mathbf{O} \mid \lambda)$——观测序列在当前模型下的概率，复杂度 $O(N^2 T)$ |
| Viterbi 算法 | 解码算法 | 求全局最优隐状态路径 $\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)$ |
| Baum-Welch 算法 | 学习算法 | EM 在 HMM 上的特例——E 步 Forward-Backward 计算时序后验，M 步计数重估参数 |
| `transmat_` | 源码属性 | 训练后学习到的状态转移矩阵 $\mathbf{A}$（$3 \times 3$，行和为 1） |

## 1. HMM 的生成过程

HMM 描述由隐状态序列驱动观测序列的生成过程：

1. $t=1$：以概率 $\pi_i$ 选择初始隐状态 $q_1 = s_i$。
2. 从隐状态 $q_1$ 的发射分布中生成观测 $o_1$：$P(o_1 = v_k \mid q_1 = s_i) = b_i(k)$。
3. $t \ge 2$：以概率 $a_{ij}$ 从 $q_{t-1} = s_i$ 转移到 $q_t = s_j$。
4. 从 $q_t$ 的发射分布中生成 $o_t$。

两个基本假设：

**一阶马尔可夫假设**：
$$
P(q_t \mid q_{t-1}, q_{t-2}, \dots, q_1) = P(q_t \mid q_{t-1})
$$

**观测独立假设**：
$$
P(o_t \mid q_1, \dots, q_T, o_1, \dots, o_T) = P(o_t \mid q_t)
$$

### 理解重点

- 第一条假设意味着当前状态仅由上一时刻状态决定——所有历史信息被压缩到 $q_{t-1}$ 中。
- 第二条假设意味着当前观测仅由当前隐状态决定——观测之间条件独立。
- 当前数据生成函数 `ProbabilisticData.hmm()` 正是按这两层结构逐步采样：先以 $A$ 转移隐状态，再以 $B$ 发射观测。

## 2. 模型定义：五元组

HMM 由五元组 $\lambda = (\mathcal{S}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ 定义：

| 符号 | 数学含义 | 在当前源码中的对应 |
|---|---|---|
| $\mathcal{S} = \{s_1, \dots, s_N\}$ | $N$ 个隐状态集合 | `n_components=3` 对应状态数 |
| $\mathcal{O} = \{v_1, \dots, v_M\}$ | $M$ 个离散观测符号集合 | 观测 `obs` 的取值空间 $\{0, 1, 2\}$ |
| $\mathbf{A} = [a_{ij}]_{N \times N}$ | 状态转移矩阵——$a_{ij} = P(q_{t+1}=s_j \mid q_t=s_i)$，行和为 1 | `model.transmat_` |
| $\mathbf{B} = [b_i(k)]_{N \times M}$ | 发射矩阵——$b_i(k) = P(o_t=v_k \mid q_t=s_i)$，行和为 1 | `model.emissionprob_` |
| $\boldsymbol{\pi} = [\pi_i]_{1 \times N}$ | 初始状态分布——$\pi_i = P(q_1=s_i)$，和为 1 | `model.startprob_` |

### 理解重点

- $A_{ij}$ 的物理含义是"从状态 $i$ 一步转移到状态 $j$ 的概率"——对角线越大，状态越稳定，越不容易跳变。
- 当前真实 $A$ 的对角线为 $[0.80, 0.60, 0.70]$——状态 1 最黏滞（80% 概率停留），状态 2 相对活跃（40% 概率跳走）。
- $B_i(k)$ 的物理含义是"隐状态 $i$ 产生观测符号 $k$ 的概率"——每行描述一个隐状态的"观测偏好"。

## 3. 三大基本问题

HMM 经典上有三个基本问题：

| 问题 | 英文名 | 输入 | 输出 | 对应算法 | 当前源码体现 |
|---|---|---|---|---|---|
| 评估 (Evaluation) | Likelihood | $\lambda$、$\mathbf{O}$ | $P(\mathbf{O} \mid \lambda)$ | Forward | `model.score(X, lengths)` |
| 解码 (Decoding) | Decoding | $\lambda$、$\mathbf{O}$ | $\mathbf{Q}^*$ | Viterbi | `model.predict(X, lengths)` |
| 学习 (Learning) | Training | $\mathbf{O}$ | $\lambda^*$ | Baum-Welch | `model.fit(X, lengths)` |

### 理解重点

- 三个问题的难度递增：评估只需单向递推，解码需要全局优化+回溯，学习需要迭代 EM。
- 当前流水线直接展示"学习 + 解码"——先 `fit` 训练，再 `predict` 推断路径。
- `score`（Forward 对数概率）可用于模型选择——比较不同 $K$ 下的拟合质量，但当前流水线仅打印准确率。

## 4. 问题一：评估（Forward 算法）

给定模型 $\lambda$ 和观测序列 $\mathbf{O} = (o_1, \dots, o_T)$，计算：

$$
P(\mathbf{O} \mid \lambda) = \sum_{\text{所有路径 } \mathbf{Q}} P(\mathbf{O} \mid \mathbf{Q}, \lambda) P(\mathbf{Q} \mid \lambda)
$$

暴力枚举所有 $N^T$ 条路径不可行——Forward 算法用动态规划将复杂度降为 $O(N^2 T)$。

**定义前向变量**：
$$
\alpha_t(i) = P(o_1, o_2, \dots, o_t, q_t = s_i \mid \lambda)
$$

**初始化**（$t=1$）：
$$
\alpha_1(i) = \pi_i \cdot b_i(o_1), \quad i = 1, \dots, N
$$

**递推**（$t=1 \to T-1$）：
$$
\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(o_{t+1})
$$

**终止**：
$$
P(\mathbf{O} \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)
$$

### 理解重点

- 递推的核心操作是**求和**（$\sum_i$）——汇集所有到达状态 $j$ 的路径概率。
- 这反映了评估问题的本质：对"所有可能路径"的概率加权求和，而非找单条最优路径。
- 当前 `model.score(X, lengths)` 返回对数概率 $\log P(\mathbf{O} \mid \lambda)$——值越大（负得越少），模型对观测序列的解释越好。

## 5. 问题二：解码（Viterbi 算法）

给定模型和观测，找最可能的单条隐状态序列：

$$
\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)
$$

**定义 Viterbi 变量**：
$$
\delta_t(i) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_t = s_i, o_1, \dots, o_t \mid \lambda)
$$

**初始化**（$t=1$）：
$$
\delta_1(i) = \pi_i \cdot b_i(o_1), \quad \psi_1(i) = 0
$$

**递推**（$t=2 \to T$）：
$$
\delta_t(j) = \max_{1 \le i \le N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)
$$

$$
\psi_t(j) = \arg\max_{1 \le i \le N} [\delta_{t-1}(i) \cdot a_{ij}]
$$

**终止**：
$$
P^* = \max_{1 \le i \le N} \delta_T(i), \quad q_T^* = \arg\max_i \delta_T(i)
$$

**回溯**（$t = T-1 \to 1$）：
$$
q_t^* = \psi_{t+1}(q_{t+1}^*)
$$

### 理解重点

- 递推的核心操作是**取最大**（$\max_i$）而非求和——这是与 Forward 算法的本质区别。
- $\psi_t(j)$ 记录了到达 $(t, j)$ 的最佳前驱状态——回溯时沿这条"面包屑"路径重建全局最优序列。
- Viterbi 保证路径的**全局合法性**——相邻状态间的转移概率 $a_{ij} > 0$，不会产生"不可能跳转"。
- 当前 `model.predict(X_obs, lengths)` 正是 Viterbi 解码——返回全局最优隐状态路径，与 `state_true` 逐步对比算准确率。

## 6. 问题三：学习（Baum-Welch 算法）

给定观测序列 $\mathbf{O}$，估计模型参数 $\lambda$。Baum-Welch 是 EM 在 HMM 上的特例。

**后向变量**（Backward 算法——E 步需要）：
$$
\beta_t(i) = P(o_{t+1}, \dots, o_T \mid q_t = s_i, \lambda)
$$

初始化 $\beta_T(i) = 1$，逆向递推：
$$
\beta_t(i) = \sum_{j=1}^{N} a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)
$$

**E 步：计算时序后验**

状态占有概率（单点后验）：
$$
\gamma_t(i) = P(q_t = s_i \mid \mathbf{O}, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(\mathbf{O} \mid \lambda)}
$$

状态转移概率（成对后验）：
$$
\xi_t(i, j) = P(q_t = s_i, q_{t+1} = s_j \mid \mathbf{O}, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(\mathbf{O} \mid \lambda)}
$$

**M 步：参数重估**

初始分布：
$$
\hat{\pi}_i = \gamma_1(i)
$$

转移矩阵：
$$
\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

发射矩阵：
$$
\hat{b}_i(k) = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot \mathbb{1}(o_t = v_k)}{\sum_{t=1}^{T} \gamma_t(i)}
$$

### 理解重点

- Baum-Welch 的 E 步需要**成对后验** $\xi_t(i,j)$——这是与普通 EM（仅需逐点后验 $\gamma_{ik}$）的根本区别。因为 HMM 的 M 步要重估转移矩阵，需要知道相邻时间步的状态联合分布。
- Forward-Backward 是计算 $\gamma_t(i)$ 和 $\xi_t(i,j)$ 的高效方法——两个方向的消息在 $t$ 处交汇，给出完整的时序后验。
- 当前源码没有手写 Baum-Welch，而是由 `hmmlearn` 的 `fit()` 内部完成——但数学本质完全一致。
- 对于 300 步 3 状态的序列，每轮 E 步复杂度 $O(300 \times 3^2) = O(2700)$——比独立样本 EM 的逐点 E 步贵。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 隐状态数 | $N$ | `n_components=3` |
| 观测符号数 | $M$ | 观测取值空间 $\{0, 1, 2\}$ |
| 转移矩阵 | $a_{ij} = P(q_{t+1}=s_j \mid q_t=s_i)$ | `model.transmat_`（$3 \times 3$，行和为 1） |
| 发射矩阵 | $b_i(k) = P(o_t=v_k \mid q_t=s_i)$ | `model.emissionprob_`（$3 \times 3$，行和为 1） |
| 初始分布 | $\pi_i = P(q_1=s_i)$ | `model.startprob_`（长度为 3，和为 1） |
| 观测序列概率 | $P(\mathbf{O} \mid \lambda)$ | `model.score(X, lengths)`——Forward 对数概率 |
| 最优隐状态路径 | $\mathbf{Q}^* = \arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O}, \lambda)$ | `model.predict(X, lengths)`——Viterbi 解码 |
| 时序后验 | $\gamma_t(i)$ | Forward-Backward 内部计算——不直接暴露 |
| 最大迭代 | $T_{\max}$ | `n_iter=100` |
| 收敛阈值 | $\|\log P^{(t+1)} - \log P^{(t)}\| < \varepsilon$ | `tol=1e-3` |
| 序列长度 | $T$ | `lengths = [300]` |

## 8. HMM vs EM (GMM) 数学对比

| 维度 | EM (GMM) | HMM |
|---|---|---|
| 数据结构 | i.i.d. 样本 $\{\mathbf{x}_i\}_{i=1}^{N}$ | **序列** $\{o_t\}_{t=1}^{T}$——时间有序 |
| 隐变量 | $z_{ik}$——样本 $i$ 属于分量 $k$ | **$q_t$——时间 $t$ 的隐状态** |
| 隐变量依赖 | 各样本独立 | **马尔可夫链依赖** $q_t \to q_{t+1}$ |
| 生成过程 | $\pi_k \to z_{ik} \to \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \to \mathbf{x}_i$ | **$q_{t-1} \xrightarrow{A} q_t \xrightarrow{B} o_t$——链式生成** |
| E 步复杂度 | $O(NK)$——逐点独立计算 | **$O(T N^2)$——Forward-Backward 时间耦合** |
| E 步所需后验 | 逐点后验 $\gamma(z_{ik})$ | **成对后验 $\xi_t(i,j)$——重估转移矩阵必需** |
| M 步核心操作 | 责任加权平均 $\to \boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$ | **计数重估 $\to A$、$B$、$\pi$** |
| 参数数 | $K(\frac{d(d+1)}{2} + d + 1)$ | **$N^2 + NM + N$** |
| 预测 | 逐点 argmax $\arg\max_k \gamma(z_{ik})$ | **Viterbi 全局解码 $\arg\max_{\mathbf{Q}} P(\mathbf{Q} \mid \mathbf{O})$** |
| 收敛保证 | 对数似然单调不减 | 对数似然单调不减 |

## 常见坑

1. 把 `state_true` 误当成 Baum-Welch 训练输入——实际上当前训练只依赖观测序列，`state_true` 仅用于评估。
2. 混淆 Forward（求和）和 Viterbi（取最大）的递推公式——两者的目标不同（评估 vs 解码），操作符不同（$\sum$ vs $\max$）。
3. 以为 Baum-Welch 的 E 步和 GMM 的 E 步完全一样——HMM 需要成对后验 $\xi_t(i,j)$，因为转移矩阵的重估依赖相邻时间步的联合分布。
4. 把解码问题和评估问题混为一谈——"路径最优"（Viterbi）和"概率最大"（Forward）是两回事。

## 小结

- HMM 的数学核心链：马尔可夫假设 $\rightarrow$ 五元组定义 $\rightarrow$ 三大问题（评估/解码/学习）$\rightarrow$ Forward（求和递推）/ Viterbi（取最大递推+回溯）/ Baum-Welch（Forward-Backward + 计数重估）。
- 与 EM (GMM) 的根本区别：HMM 的隐变量有时间依赖（马尔可夫链），E 步需成对后验 $\xi_t(i,j)$，预测需 Viterbi 全局解码——而非逐点独立计算。
- 当前源码 `CategoricalHMM(n_components=3, n_iter=100)` 将上述数学全部封装在 `fit`/`predict`/`score` 三个方法中——`transmat_`、`emissionprob_`、`startprob_` 是训练后的可直接检验的参数。

# 数据构成

## 本章目标

1. 明确本仓库 HMM 数据来自 `ProbabilisticData.hmm()` 手动参数化生成的离散序列。
2. 理解三列数据（`time`、`obs`、`state_true`）各自的角色与边界——训练只依赖 `obs`。
3. 理解序列数据特有的整形步骤（`reshape(-1, 1)` + `lengths`）及其与表格型数据的根本差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.hmm()` | 方法 | 手动参数化生成 HMM 离散观测序列——含真实隐状态 |
| `hmm_data` | 全局变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame（300 行 $\times$ 3 列） |
| `obs` | 列 | 离散观测符号序列 $\{0, 1, 2\}$——训练 HMM 的唯一输入 |
| `state_true` | 列 | 数据生成时记录的真实隐状态——仅用于训练后评估对比 |
| `reshape(-1, 1)` | 操作 | 将一维序列整形为 hmmlearn 要求的列向量 `(300, 1)` |
| `lengths` | 参数 | 序列长度列表 `[300]`——告诉模型当前由几条序列拼接而成 |

## 1. 数据生成：`ProbabilisticData.hmm()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `hmm_n_steps` | `int` | 序列长度。`300`——足够 Baum-Welch 稳定估计 $3 \times 3$ 转移矩阵 | `50`、`100`、`300`、`1000` |
| `hmm_pi` | `list[float]` | 初始状态分布，和为 1。$\pi_1=0.6$ 意味着序列大概率从状态 0 开始 | `[0.6, 0.3, 0.1]` |
| `hmm_A` | `list[list[float]]` | 状态转移矩阵，行和为 1。对角线 $[0.80, 0.60, 0.70]$ 表示状态 0 最稳定 | `[[0.80,0.15,0.05], [0.20,0.60,0.20], [0.10,0.20,0.70]]` |
| `hmm_B` | `list[list[float]]` | 发射矩阵，行和为 1。状态 0 偏好发射符号 0（概率 0.60） | `[[0.60,0.30,0.10], [0.20,0.50,0.30], [0.10,0.20,0.70]]` |
| `random_state` | `int` | 随机种子。`42`——保证序列可复现 | `42` |
| 返回值 | `DataFrame` | 含 `time`、`obs`、`state_true` 三列 | — |

### 示例代码

```python
from data_generation.probabilistic import ProbabilisticData

data = ProbabilisticData().hmm()
# data.columns = ["time", "obs", "state_true"]
# data.shape = (300, 3)
```

### 生成流程

```python
states = np.zeros(n_steps, dtype=int)
obs_arr = np.zeros(n_steps, dtype=int)

# t=0: 从初始分布采样
states[0] = rng.choice(n_states, p=pi)
obs_arr[0] = rng.choice(n_symbols, p=B[states[0]])

# t=1..T-1: 按转移矩阵推进隐状态，按发射矩阵生成观测
for t in range(1, n_steps):
    states[t] = rng.choice(n_states, p=A[states[t - 1]])
    obs_arr[t] = rng.choice(n_symbols, p=B[states[t]])
```

### 理解重点

- 这是**完全手动参数化**的生成过程——转移矩阵 $A$ 和发射矩阵 $B$ 是预先写死的，而非从数据中学习。
- 每个时间步先按转移矩阵产生新的隐状态，再按发射矩阵产生观测符号——两层随机采样层层嵌套。
- 正是因为 $A$ 和 $B$ 已知，后续可以定量评估 Baum-Welch 恢复参数的精度——这是合成数据的核心教学价值。
- 序列的本质是**时间有序**——$o_t$ 与 $o_{t-1}$ 不独立，它们通过隐状态链 $q_{t-1} \to q_t$ 关联。

## 2. 三列数据的角色

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `time` | `Series`，形状 `(300,)` | 时间步索引 $\{0, 1, \dots, 299\}$——标记序列中的位置 | `0, 1, 2, ..., 299` |
| `obs` | `Series`，形状 `(300,)` | 离散观测符号 $\{0, 1, 2\}$——**训练 HMM 的唯一输入** | `data["obs"].values.astype(int)` |
| `state_true` | `Series`，形状 `(300,)` | 真实隐状态 $\{0, 1, 2\}$——**仅用于评估对比**，不参与训练 | `data["state_true"].values.astype(int)` |

### 理解重点

- `obs` 是流水线中真正送入 `model.fit()` 的数据——Baum-Welch 只看到观测序列，对真实隐状态一无所知。
- `state_true` 是生成时记录的"答案"——仅在训练后与 Viterbi 解码的 `states_pred` 做逐步对比，计算准确率。
- `time` 不直接送入模型——但它提醒我们这是一个有序序列，而非可以随意打乱的 i.i.d. 样本表。
- `astype(int)` 是必需的——确保 hmmlearn 将观测识别为离散符号而非连续浮点数。

## 3. 序列数据整形：`reshape(-1, 1)` 与 `lengths`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_obs` | `ndarray`，形状 `(300, 1)` | 整形后的观测列向量——hmmlearn 的 `fit()` 要求二维输入 | `obs.reshape(-1, 1)` |
| `lengths` | `list[int]` | 序列长度列表。`[300]`——单条 300 步序列 | `[300]`、`[100, 200, 150]` |

### 示例代码

```python
obs = data["obs"].values.astype(int)       # (300,) 一维
X_obs = obs.reshape(-1, 1)                  # (300, 1) 列向量
lengths = [len(obs)]                        # [300]

model.fit(X_obs, lengths)                   # HMM 训练
states_pred = model.predict(X_obs, lengths)  # Viterbi 解码
```

### 理解重点

- `reshape(-1, 1)` 是将一维序列转为列向量的**必需步骤**——hmmlearn 的 `fit` 要求观测形状为 `(n_steps, n_features)`。
- `lengths` 告诉模型"这条长序列由几条子序列拼接而成"——当前是单条 300 步，所以 `[300]`；多序列场景下可以是 `[100, 200, 150]`。
- 这是 HMM 与所有其他模型（分类/聚类/回归）在数据准备上的**根本差异**——其他模型只需传 `(n_samples, n_features)`，HMM 还需传序列边界。

## 4. 当前流程边界

| 项目 | 当前状态 | 说明 |
|---|---|---|
| train/test split | 未使用 | HMM 在单条序列上训练和评估——序列不可随意切分 |
| 标准化 | 未使用 | 离散观测符号 $\{0, 1, 2\}$ 无需缩放 |
| 多条序列拼接 | 当前未展示 | 框架支持——`lengths=[100, 200]` 即可，但教学用单条更清晰 |

### 理解重点

- 当前 HMM 分册没有 train/test split——因为序列数据的时间依赖使得随机切分不合理，且当前目标是展示"参数恢复"而非泛化性能。
- 离散观测符号不需要标准化——这是 HMM 与所有连续特征模型（EM/KMeans/回归）的关键区别。
- 文档必须如实描述当前实现——不能把监督学习的 train/test split 或连续特征的标准化习惯误套到 HMM。

## 5. 数据设计意图：与 EM (GMM) 的对比

| 数据维度 | EM (GMM) | HMM |
|---|---|---|
| 生成方式 | 手动合成——3 分量各向异性高斯 | **手动参数化——Markov chain + categorical emission** |
| 数据形态 | 独立样本矩阵 `(500, 2)` | **有序序列 `(300, 1)`** |
| 特征类型 | 连续 $\mathbb{R}^2$ | **离散 $\{0, 1, 2\}$** |
| 样本独立性 | i.i.d. | **时间依赖——$o_t$ 通过 $q_t$ 与 $o_{t-1}$ 关联** |
| 标签列 | `true_label`——仅用于评估 | **`state_true`——仅用于评估** |
| 标准化 | 有（`StandardScaler`） | **无** |
| 数据拆分 | 无（全量聚类） | 无（全量序列） |
| 训练输入 | `fit(X)` | **`fit(X, lengths)`** |

### 理解重点

- HMM 数据**刻意使用离散观测**——这是为了展示 HMM 处理符号序列（如词性标注、基因序列）的经典场景，而非连续值回归。
- 序列长度为 300 是有意设计——足够 Baum-Welch 稳定估计 $3 \times 3$ 转移矩阵，又不至于太长使演示耗时。
- 与 EM 数据的核心差异：HMM 数据点之间**不独立**——$o_{100}$ 的分布受 $q_{99}$ 影响，而 EM 中 $\mathbf{x}_{100}$ 与其他样本完全独立。

## 数据可视化

![数据分布图](https://img.yumeko.site/file/blog/articles/1780736135620.png)

![数据序列图](https://img.yumeko.site/file/blog/articles/1780737750768.png)

## 常见坑

1. 把 `state_true` 当成训练标签——HMM 的 Baum-Welch 是无监督学习，只依赖观测序列。
2. 忘记 `reshape(-1, 1)`——直接将一维 `obs` 传给 `fit()`，hmmlearn 会报错或产生错误结果。
3. 忘记 `astype(int)`——浮点观测可能被 hmmlearn 误认为连续值，触发错误行为。
4. 在极短序列（<50 步）上期待稳定的转移矩阵估计——转移次数不足，参数方差大。

## 小结

- 当前 HMM 数据来自 `ProbabilisticData.hmm()`——手动参数化（$A$、$B$、$\pi$）生成 300 步离散观测序列，三层结构清晰。
- 三列数据角色明确：`obs` 是唯一训练输入，`state_true` 仅用于评估，`time` 标记序列顺序。
- 序列数据的两个特殊整形步骤（`reshape(-1, 1)` + `lengths`）是 HMM 与所有表格型模型在数据准备上的根本分界线。

# 思路与直觉

## 本章目标

1. 用直观方式理解 HMM 的核心思路——"一个看不见的状态序列生成了一串看得见的观察"。
2. 理解三个基本问题：评估（Forward）、解码（Viterbi）、学习（Baum-Welch）。
3. 通过与 EM（GMM）的对比，建立 HMM 在概率模型谱系中的定位——序列隐变量模型。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 隐状态序列 | 核心概念 | 一个看不见的马尔可夫链——每一步的状态只依赖于上一步 |
| 观测序列 | 可见数据 | 你在现实中观察到的离散符号——由当前隐状态按概率"发射" |
| Forward 算法 | 评估问题 | 给定参数，计算这条观测序列出现的概率 |
| Viterbi 算法 | 解码问题 | 给定观测序列，找出最可能的隐状态路径 |
| Baum-Welch | 学习问题 | 仅从观测序列，学习 HMM 的参数（转移矩阵 + 发射矩阵） |
| 马尔可夫性 | 核心假设 | 未来只取决于现在——$P(s_t \mid s_1, \dots, s_{t-1}) = P(s_t \mid s_{t-1})$ |

## 1. 为什么需要 HMM

大多数模型假设数据是独立同分布（i.i.d.）的。但很多现实数据是**序列**——一句话中的字、股票价格的变化、DNA 的碱基序列——相邻时间的观测高度相关。

HMM 用两层结构来建模这种序列依赖：

> 底下是一层**看不见**的状态链——它按马尔可夫规则迁移（喜欢待在一个状态，偶尔跳到其他状态）
> 上面是一层**看得见**的观测链——每个时刻的状态按概率"发出"一个观测符号

### 理解重点

- 与 GMM 的关键区别：GMM 中样本之间独立，HMM 中样本按时间序列相关。
- "隐藏"是 HMM 的精髓——现实中的很多东西你直接看不到（人的意图、市场的情绪、DNA 的模块），只能通过间接信号推断。

## 2. 用"天气预报"理解 HMM

假设你在一个没有窗户的房间里，只能看到同事每天拿不拿伞，但不知道外面天气如何：

- **隐状态**：晴、多云、雨——你看不到
- **观测**：带伞、没带伞——你能看到
- **状态转移**：晴天大概率继续晴，但也会变阴（$A$ 矩阵描述）
- **观测发射**：晴天大概率不带伞，雨天大概率带伞（$B$ 矩阵描述）

HMM 要做的三件事：

1. **评估**：已知天气模型参数，同事连续一周带伞/没带伞的序列，这个序列"多可能"？
2. **解码**：同事连续一周的带伞记录，那一周最可能的天气序列是什么？
3. **学习**：只看同事一年的带伞记录，自动猜出天气模型（几个状态、怎么转移、怎么发射）

### 理解重点

- 当前源码的数据就是这种场景——3 个隐状态、3 种观测符号，你自己设好了"真实天气"和"真实发射概率"，然后让 HMM 从观测序列猜回来。
- 这就像给 HMM 一套带伞记录，让它自己学出"天气模型"——然后跟真实的天气参数对比。

## 3. 三个算法的直觉对比

| 算法 | 问题 | 直觉 | 输出 |
|---|---|---|---|
| Forward | 这个观测序列多可能？ | 沿着时间累积所有路径的概率 | 一个标量 $P(O \mid \lambda)$ |
| Viterbi | 最可能的隐状态路径？ | 沿着时间选最优的上一步 -> 递推到终点 -> 回溯 | 一个状态序列 $\hat{S}$ |
| Baum-Welch | 模型参数是什么？ | EM 的 HMM 版——用当前参数估计状态概率，再反推新参数 | 新的 $\hat{A}$、$\hat{B}$、$\hat{\pi}$ |

### 理解重点

- Forward 是"求和的动态规划"——把所有可能的隐状态路径概率加起来。
- Viterbi 是"求最大值的动态规划"——找一条概率最大的隐状态路径。
- Baum-Welch 是"EM for HMM"——类似 GMM 的 EM，但 E 步用的是 Forward-Backward 算法（考虑整个序列的上下文）。

## 4. 马尔可夫性的直觉

> "明天的天气取决于今天，但与昨天、前天无关"

马尔可夫性使 HMM 可行——如果没有这个假设，转移矩阵的规模将随序列长度指数增长。

### 理解重点

- 当前数据的转移矩阵体现了"惯性"——对角线元素大（0.80, 0.60, 0.70），状态倾向于维持。
- 这是 HMM 的"平滑性"来源——隐状态不会每步都剧烈跳变。

## 5. 与 EM（GMM）的直觉对比

| 维度 | EM (GMM) | HMM |
|---|---|---|
| 数据结构 | 独立样本（二维点集） | **序列（时间步依赖）** |
| 隐变量 | $z_{ik}$——每个点属于哪个分量 | **$s_t$——每个时间步的隐状态** |
| 独立性假设 | i.i.d. | **马尔可夫依赖** |
| E 步 | 计算后验 $\gamma(z_{ik})$ | **Forward-Backward 算法** |
| M 步 | 加权更新 $\mu$、$\Sigma$、$\pi$ | **重新估计 $A$ 和 $B$** |
| 解码 | $\arg\max_k \gamma_{ik}$（逐点） | **Viterbi 算法（全局最优路径）** |
| 输出 | 聚类标签 + 归属概率 | **隐状态序列 + 准确率** |

### 理解重点

- HMM 的 Baum-Welch 在概念上是 GMM 的 EM 的因果推广——区别在于样本间有序列依赖。
- HMM 的 Viterbi 解码是"全局一致性推断"——考虑整个序列后选择最优路径，而不是逐步 argmax。
- HMM 的评估不是聚类图而是准确率——对比预测状态序列与真实状态序列。

## 可视化

HMM 当前流水线**无可视化输出**——纯终端文本评估（隐状态准确率 + 学习到的转移矩阵）。这是因为序列数据不适合用散点图或矩阵图来展示。

## 常见坑

1. 把 HMM 当成独立样本模型——忽略序列依赖会使隐状态的连续性丢失。
2. 以为 Viterbi 解码等价于逐点最大后验——逐点 argmax 可能产生概率为零的状态转移（非法路径），Viterbi 保证路径合法。
3. 在短序列上训练 HMM——Baum-Welch 需要足够长的序列才能稳定估计转移概率。
4. 忽略 `lengths` 参数的含义——它允许多个不等长序列，每个序列的末尾被单独处理。

## 小结

- HMM 的直觉核心是"看不见的马尔可夫链驱动了看得见的观测"——状态序列有惯性、观测与状态相关。
- 三个算法解决三个问题：Forward 算概率、Viterbi 解路径、Baum-Welch 学参数。
- HMM 与 GMM（EM）共享概率生成模型的哲学——但多了序列依赖和马尔可夫性假设。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 HMM 模型——离散观测、序列数据、可选依赖。
2. 理解 `CategoricalHMM` 的核心构造器参数（`n_components`、`n_iter`、`tol`）及其序列含义。
3. 看清训练完成后最重要的模型属性——`transmat_`（转移矩阵）、`startprob_`（初始概率）、`emissionprob_`（发射矩阵）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 HMM 模型——含可选依赖检查（`CategoricalHMM` / `MultinomialHMM` 双备份） |
| `CategoricalHMM(...)` | 类 | hmmlearn 提供的离散 HMM——用 Baum-Welch（EM）估计参数 |
| `model.fit(X_obs, lengths)` | 方法 | Baum-Welch 训练——迭代 Forward-Backward + 参数重估 |
| `model.transmat_` | 属性 | 学习到的状态转移矩阵 $A$（3x3） |
| `model.startprob_` | 属性 | 学习到的初始状态分布 $\pi$（3,） |
| `model.emissionprob_` | 属性 | 学习到的观测发射矩阵 $B$（3x3） |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_obs, lengths, n_components=3, n_iter=100, tol=1e-3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_obs` | `ndarray`，形状 `(300, 1)` | 观测序列列向量——离散符号 $\{0, 1, 2\}$，每行为一个时间步 | `obs.reshape(-1, 1)` |
| `lengths` | `list[int]` | 序列长度列表。`[300]`——单条 300 步的序列 | `[300]`、`[100, 200]` |
| `n_components` | `int` | 隐状态数。`3`——与真实隐状态数一致 | `2`、`3`、`5` |
| `n_iter` | `int` | Baum-Welch 最大迭代次数。`100`——比 GMM 的 EM 迭代少（序列计算更贵） | `50`、`100`、`200` |
| `tol` | `float` | 对数似然收敛阈值。`1e-3` | `1e-3`、`1e-4` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `CategoricalHMM` 或 `MultinomialHMM` | 已完成 `fit()` 的 HMM 模型 | — |

### 示例代码

```python
from model_training.probabilistic.hmm import train_model

obs = hmm_data["obs"].values.astype(int)
X_obs = obs.reshape(-1, 1)
lengths = [len(obs)]
model = train_model(X_obs, lengths)
```

### 理解重点

- `train_model(...)` 的参数签名与 GMM 完全不同——输入为序列数据（`X_obs` + `lengths`），而非独立样本矩阵。
- `lengths` 支持多条不等长序列——当前使用单条 300 步序列，但框架天然支持批量序列训练。
- 内部有双备份可选依赖处理——优先 `CategoricalHMM`，回退 `MultinomialHMM`。

## 2. `CategoricalHMM` 构造器参数

### 参数速览

适用 API：`CategoricalHMM(n_components=3, n_iter=100, tol=1e-3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 隐状态数。`3`——HMM 的核心超参数，需预先设定 | `2`、`3`、`5` |
| `n_iter` | `int` | Baum-Welch 最大迭代次数。`100`——序列数据的 EM 通常收敛更慢 | `50`、`100`、`200` |
| `tol` | `float` | 对数似然收敛阈值。`1e-3` | `1e-3`、`1e-4` |
| `random_state` | `int` | 随机种子。`42`——保证参数初始化和结果可复现 | `42` |
| `init_params` | `str` | 参数初始化方法。默认 `"st"`（转移和发射矩阵随机初始化） | `"st"`、`""` |
| `params` | `str` | 哪些参数在训练中更新。默认 `"ste"`（startprob/transmat/emissionprob） | `"ste"`、`"st"` |
| `verbose` | `bool` | 是否打印详细日志。默认 `False` | `True`、`False` |

### 示例代码

```python
try:
    from hmmlearn.hmm import CategoricalHMM
    ModelClass = CategoricalHMM
except ImportError:
    from hmmlearn.hmm import MultinomialHMM
    ModelClass = MultinomialHMM

model = ModelClass(
    n_components=3,
    n_iter=100,
    tol=1e-3,
    random_state=42,
)
model.fit(X_obs, lengths)
```

### 理解重点

- `CategoricalHMM` 是 hmmlearn 0.3+ 的新 API——`MultinomialHMM` 是旧版本兼容（当前源码双备份）。
- `n_iter=100` 比 GMM 的 `max_iter=200` 少——因为序列计算中每次 Forward-Backward 的复杂度是 $O(T \times K^2)$，远贵于逐点的 E 步。
- 没有 `covariance_type` 参数——HMM 处理离散观测，不涉及协方差矩阵。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `transmat_` | `ndarray`，形状 `(3, 3)` | 转移矩阵 $A$ | $A_{ij} = P(s_{t+1}=j \mid s_t=i)$，行和为 1 |
| `startprob_` | `ndarray`，形状 `(3,)` | 初始分布 $\pi$ | $\pi_i = P(s_1=i)$，和为 1 |
| `emissionprob_` | `ndarray`，形状 `(3, 3)` | 发射矩阵 $B$ | $B_{ij} = P(o_t=j \mid s_t=i)$，行和为 1 |
| `monitor_` | `dict` | 训练历史 | 逐次迭代的对数似然值列表——诊断收敛 |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"n_iter: {n_iter}")
print(f"tol: {tol}")
print(f"转移矩阵:\n{model.transmat_.round(3)}")
print(f"发射矩阵:\n{model.emissionprob_.round(3)}")
print(f"初始分布: {model.startprob_.round(3)}")
```

### 理解重点

- `transmat_`（$3 \times 3$）描述隐状态的迁移行为——对角线越大（如 0.8），状态越稳定。
- `emissionprob_`（$3 \times 3$）描述每个状态的观测偏好——例如"状态 0 大概率发射符号 0"。
- `startprob_` 描述序列起始时刻的状态分布——与 GMM 的 `weights_` 概念相似，但用于序列初始而非整个序列。
- 通过对比学习到的 `transmat_` / `emissionprob_` 与真实参数，可以定量评估 HMM 的恢复能力。

## 4. `predict()` — Viterbi 解码

### 参数速览

| 方法 | 输入 | 输出 | 算法 | 说明 |
|---|---|---|---|---|
| `predict(X, lengths)` | `(n_steps, 1)` + `lengths` | `ndarray`，`(n_steps,)` | **Viterbi** | 全局最优隐状态路径——保证路径合法（不存在概率为 0 的转移） |

### 理解重点

- Viterbi 解码**不是**逐步 argmax——它全局寻找概率最大的单条路径，保证相邻状态的转移是合法的。
- 逐步 argmax（$\arg\max_i P(s_t = i \mid O, \lambda)$）可能产生"非法"的状态跳跃——Viterbi 避免了这一点。
- 在 HMM 中，`predict` 返回的是隐状态序列——用于与 `state_true` 对比计算准确率。

## 5. HMM vs GMM vs 集成模型 参数对比

| 参数/属性 | GMM | HMM | 备注 |
|---|---|---|---|
| 核心参数 | `n_components`、`covariance_type`、`max_iter` | `n_components`、`n_iter`、`tol` | 相似但 HMM 更关注迭代收敛 |
| 训练输入 | `fit(X)`——独立样本 | **`fit(X, lengths)`——序列数据** | 根本差异 |
| 模型属性 | `means_`、`covariances_`、`weights_` | **`transmat_`、`emissionprob_`、`startprob_`** | HMM 描述动态，GMM 描述分布 |
| 预测输出 | `predict(X)`、`predict_proba(X)` | **`predict(X, lengths)`（Viterbi）** | 无 `predict_proba`——但有 `score`（Forward） |
| 依赖 | sklearn 内置 | **`pip install hmmlearn`** | 可选依赖 |

## 常见坑

1. 忘记传 `lengths` 参数——`fit(X)` 会报错，HMM 必须知道每条序列的边界。
2. 混淆 `transmat_` 的行和列方向——行 $i$ 列 $j$ 表示 $P(s_{t+1}=j \mid s_t=i)$，行和为 1。
3. 在极短序列（<50 步）上训练——Baum-Welch 需要足够多的状态转移来稳定估计 $A$ 矩阵。
4. 混淆 `CategoricalHMM` 和 `MultinomialHMM`——前者是 hmmlearn 0.3+ 的新 API，语义更清晰。

## 小结

- `train_model(...)` 是本仓库 HMM 的核心训练入口——含双备份可选依赖检查，输入为序列数据而非独立样本。
- `CategoricalHMM` 的核心参数是 `n_components`（隐状态数）、`n_iter`（Baum-Welch 上限）、`tol`（收敛阈值）——结构化参数比 GMM 少但序列计算量更大。
- 训练完成后的核心属性：`transmat_` / `emissionprob_` / `startprob_`——三件套完全描述离散 HMM 的动态和观测行为。

# 训练与预测

## 本章目标

1. 理解 `pipelines/probabilistic/hmm.py` 的 `run()` 流水线——序列模型的端到端流程（无标准化、无切分、无可视化）。
2. 理解 Baum-Welch 训练过程——序列数据的 EM 算法（Forward-Backward + 参数重估）。
3. 理解 Viterbi 解码的预测输出——全局最优隐状态路径及其与真实状态的对比。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 序列模型流水线编排——4 步完成数据整形、训练、Viterbi 解码和评估 |
| `model.fit(X_obs, lengths)` | 方法 | Baum-Welch 训练——迭代 E 步（Forward-Backward）+ M 步（参数重估） |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |
| `model.score(X_obs, lengths)` | 方法 | Forward 算法——计算观测序列的对数概率 |
| 隐状态准确率 | 评估 | `np.mean(states_pred == y_true)`——Viterbi 路径与真实状态的逐步比较 |

## 1. 完整流水线流程

### 流程概述

```
hmm_data.copy()
  - 1 obs = data["obs"].values.astype(int)  -> reshape(-1, 1)
  - 2 lengths = [len(obs)]
  - 3 y_true = data["state_true"].values.astype(int)
  - 4 model = train_model(X_obs, lengths)
  - 5 states_pred = model.predict(X_obs, lengths) -> 准确率 + 转移矩阵
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `hmm_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(300, 3)` | 避免修改全局变量 |
| 整形观测 | `obs.reshape(-1, 1)` | `Series`，`(300,)` | `ndarray`，`(300, 1)` | hmmlearn 要求列向量输入 |
| 序列长度 | `[len(obs)]` | — | `list[int]` | 单条 300 步序列 |
| 提取真实状态 | `data["state_true"].values` | `DataFrame` | `ndarray`，`(300,)` | 仅用于评估对比 |
| 训练 | `train_model(X_obs, lengths)` | `(ndarray, list)` | `CategoricalHMM` | Baum-Welch 迭代 |
| Viterbi 解码 | `model.predict(X_obs, lengths)` | `(ndarray, list)` | `ndarray`，`(300,)` | 全局最优隐状态路径 |
| 评估 | `np.mean(states_pred == y_true)` | `(ndarray, ndarray)` | `float` | 逐步准确率 |

### 理解重点

- 这是本仓库所有流水线中最简洁的——仅 4 步核心操作，**无标准化**（离散观测不需要）、**无切分**（单条序列）、**无可视化**（序列数据不适合散点图）。
- 每个步骤都是类型敏感的——`astype(int)` 确保 hmmlearn 识别为离散符号。
- `lengths = [len(obs)]` 虽然此处是单元素列表，但框架设计允许 `[100, 200, 150]` 等多序列批量训练。

## 2. 训练细节：Baum-Welch 算法

### 算法流程

```
初始化参数（随机或等值）
    v
E 步：Forward-Backward 算法
    Forward:  alpha_t(i) = P(o_1,...,o_t, s_t=i | lambda)
    Backward: beta_t(i) = P(o_{t+1},...,o_T | s_t=i, lambda)
    计算后验: gamma_t(i) = P(s_t=i | O, lambda) = alpha_t(i)beta_t(i) / P(O|lambda)
              ξ_t(i,j) = P(s_t=i, s_{t+1}=j | O, lambda)
    v
M 步：参数重估
    pî_i = gamma_1(i)
    Â_ij = Sigma_{t=1}^{T-1} ξ_t(i,j) / Sigma_{t=1}^{T-1} gamma_t(i)
    B̂_ij = Sigma_{t: o_t=j} gamma_t(i) / Sigma_{t=1}^{T} gamma_t(i)
    v
检查收敛：|log P(O|lambda_new) - log P(O|lambda_old)| < tol ?
    是 -> 停止
    否 -> 回到 E 步
    v
达到 n_iter=100 -> 终止
```

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_components` | `3` | 隐状态数——决定了 $A$（3x3）、$B$（3x3）、$\pi$（3,）的维度 |
| `n_iter` | `100` | Baum-Welch 最大迭代次数 |
| `tol` | `1e-3` | 对数似然收敛阈值——连续两次变化小于此值则停止 |

### 理解重点

- Baum-Welch 在概念上是**EM 的序列版**——E 步用 Forward-Backward（而非逐点后验），M 步用计数重估（而非加权平均）。
- Forward 和 Backward 是两个互补的"消息传递"——Forward 从过去积累信息，Backward 从未来回传信息，交汇点给出每个时间步的状态后验。
- 对于 300 步 3 状态的序列，Baum-Welch 每轮 E 步的复杂度是 $O(300 \times 3^2) = O(2700)$——远比独立样本的 EM 贵。

## 3. 预测细节：Viterbi 解码

### 算法流程

```
初始化: delta_1(i) = pi_i * B_{i,o_1}
递推:   delta_t(j) = max_i [delta_{t-1}(i) * A_{ij}] * B_{j,o_t}
        ψ_t(j) = argmax_i [delta_{t-1}(i) * A_{ij}]
终止:   ŝ_T = argmax_i delta_T(i)
回溯:   ŝ_t = ψ_{t+1}(ŝ_{t+1})   (t = T-1, ..., 1)
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 算法 | 输出含义 |
|---|---|---|---|---|
| `predict(X, lengths)` | `(300, 1)` + `lengths` | `(300,)` | **Viterbi** | 全局最优隐状态路径 |

### 理解重点

- Viterbi 保证路径的**全局一致性**——每一步的状态转移都是合法的（$A_{ij} > 0$），不会出现"不可能跳转"。
- 逐步 argmax（$\arg\max_i \gamma_t(i)$）可能产生 $A_{ij}=0$ 的非法转移——Viterbi 通过回溯机制避免。
- 当前流水线将 Viterbi 的预测与 `state_true` 逐步对比——准确率越高，模型越成功恢复隐状态序列。

## 4. 与 GMM（EM）训练流程的对比

| 步骤 | GMM (EM) | HMM (Baum-Welch) |
|---|---|---|
| 数据 | 独立样本矩阵 | **序列列向量 + lengths** |
| 标准化 | 有（`StandardScaler`） | **无**（离散符号不需要） |
| E 步 | 逐点后验 $\gamma(z_{ik})$ | **Forward-Backward -> $\gamma_t(i)$ + $\xi_t(i,j)$** |
| M 步 | 加权更新 $\mu$、$\Sigma$、$\pi$ | **计数重估 $A$、$B$、$\pi$** |
| 复杂度 | $O(N \times K \times d^2)$ | **$O(T \times K^2)$** |
| 收敛诊断 | `lower_bound_`（对数似然） | **`monitor_`（逐次对数似然列表）** |
| 预测 | `predict`（逐点 argmax） | **`predict`（Viterbi 全局解码）** |

## 常见坑

1. 不传 `lengths` 参数——`fit(X)` 的错误调用，hmmlearn 必须知道每条序列的边界。
2. 忘记将观测 `reshape(-1, 1)`——hmmlearn 要求观测为列向量 `(n_steps, 1)`。
3. 把 `astype(int)` 漏掉——字符串或浮点观测符号可能导致 hmmlearn 无法识别。
4. 在极短序列上比较准确率——300 步中稳定迁移的比例有限，准确率波动大。

## 小结

- HMM 流水线是最简的 4 步序列流程——数据整形、Baum-Welch 训练、Viterbi 解码、准确率评估，无标准化/切分/可视化。
- `fit()` 的核心流程：Forward-Backward（E 步计算时序后验）-> 计数重估 $A$/$B$/$\pi$（M 步最大化）-> 对数似然收敛检查 -> 循环。
- `predict()` 使用 Viterbi 全局解码——保证路径的转移合法性，与逐点 argmax 有本质区别。

# 评估与诊断

## 本章目标

1. 理解当前 HMM 流水线的评估输出——隐状态准确率和学习到的转移矩阵打印。
2. 理解 HMM 评估的独特之处——比较的是"状态序列"而非"类别标签"。
3. 明确当前流水线**已实现**和**未实现**的评估项——以及 HMM 评估与分类/聚类评估的根本差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 隐状态准确率 | 定量指标 | `np.mean(states_pred == y_true)`——Viterbi 路径与真实状态的逐步一致性 |
| 转移矩阵打印 | 诊断输出 | `model.transmat_`——对比学习到的转移矩阵与真实转移矩阵 |
| 对数似然历史 | 诊断信息 | `model.monitor_`——逐次迭代的对数似然，用于判断收敛和局部最优 |
| Forward 得分 | 概率评估 | `model.score(X, lengths)`——观测序列在当前参数下的对数概率 |

## 1. 隐状态准确率

训练完成后，终端直接打印隐状态预测准确率：

```python
states_pred = model.predict(X_obs, lengths)
accuracy = np.mean(states_pred == y_true)
print(f"隐状态预测准确率: {accuracy:.4f}")
```

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `states_pred` | `ndarray`，形状 `(300,)` | Viterbi 解码的隐状态路径 | `model.predict(X_obs, lengths)` |
| `y_true` | `ndarray`，形状 `(300,)` | 真实隐状态——生成数据时记录的地面真值 | `data["state_true"].values` |
| `accuracy` | `float`，$\in [0, 1]$ | 逐步匹配比例——越高表示 HMM 恢复隐状态越成功 | `0.85` |

### 示例输出

```text
隐状态预测准确率: 0.8933
转移矩阵:
[[0.78  0.18  0.04 ]
 [0.22  0.58  0.20 ]
 [0.12  0.22  0.66 ]]
```

### 理解重点

- 隐状态准确率与分类准确率有本质区别——这里比较的是**时序路径**，每一步的准确性受前后文影响。
- 标签编号不能互换——与 GMM 的聚类标签不同，HMM 的隐状态编号在训练中由 Baum-Welch 的初始化决定，可能与 `state_true` 的编号不一致。
- 如果准确率很低但转移矩阵与真实接近——可能存在标签排列（permutation mismatch），即状态 $0$ 被映射为状态 $1$ 等。

## 2. 转移矩阵诊断

终端打印学习到的转移矩阵，可直接与真实转移矩阵对比：

| 真实 $A$ | 学习到的 $A$ |
|---|---|
| `[[0.80, 0.15, 0.05]` | `[[0.78, 0.18, 0.04]` |
| `[0.20, 0.60, 0.20]` | `[0.22, 0.58, 0.20]` |
| `[0.10, 0.20, 0.70]]` | `[0.12, 0.22, 0.66]]` |

### 理解重点

- 对角线元素的接近程度反映了状态稳定性的恢复精度。
- 非对角线元素的匹配说明 HMM 成功学会了状态间的迁移偏好。
- 如果学习到的矩阵与真实差很多——可能数据不够长、$K$ 设定错误、或 Baum-Welch 陷入局部最优。

## 3. 对数似然历史

`model.monitor_` 存储了逐次迭代的对数似然值：

```python
print(model.monitor_.converged)  # 是否收敛
print(model.monitor_.history)    # 迭代历史
```

### 理解重点

- 对数似然单调递增——与 GMM 的 EM 一致，如果出现拐点说明 baum-welch 的实现有问题。
- 如果对数似然增长很慢或停滞——可能已经收敛到局部最优。
- 比较多次训练的对数似然——可以判断初始化敏感性和收敛质量。

## 4. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 隐状态准确率 | 已实现 | HMM 评估的核心指标 |
| 转移矩阵打印 | 已实现 | 对比学习到的参数与生成参数 |
| 发射矩阵对比 | **未实现** | 当前仅打印转移矩阵——发射矩阵也可以通过 `model.emissionprob_` 访问 |
| 可视化（隐状态轨迹图） | **未实现** | 可绘制 time-vs-state 对比图——但教学场景下终端文本已足够 |
| 混淆矩阵 | **不适用** | 状态标签可能排列不匹配——混淆矩阵在此场景下意义有限 |
| BIC/AIC | **未实现** | HMM 模型选择指标——当前 $K=3$ 为已知先验 |
| 交叉验证对数似然 | **未实现** | 单条长序列无法做 K-Fold——但多序列场景下可用留一序列验证 |

### 理解重点

- HMM 的评估与标准分类/聚类完全不同——评估的是"序列推断准确率"而非"标签分类准确率"。
- 无可视化是有意设计——300 步的离散状态序列用终端文本展示更清晰。

## 5. HMM vs GMM vs 集成模型 评估对比

| 评估维度 | Bagging/GBDT | KMeans | GMM (EM) | HMM |
|---|---|---|---|---|
| 任务类型 | 分类 | 聚类 | 聚类 | **序列状态推断** |
| 可视化 | 混淆矩阵 + ROC | 聚类图 | 双面板聚类图 | **无图（终端文本）** |
| 定量指标 | accuracy/precision/recall | inertia_ | lower_bound_ | **隐状态准确率 + 对数似然** |
| 参数对比 | 无 | 无 | weights_/means_/covariances_ | **transmat_ 打印** |
| 标签语义 | 有（不可互换） | 无（不可互换） | 无（不可互换） | **部分可互换** |
| 数据特性 | i.i.d. | i.i.d. | i.i.d. | **时序依赖** |

## 常见坑

1. 直接对比 `states_pred` 和 `y_true` 的标签编号——可能因排列不匹配而得到虚假的低准确率。
2. 忽略转移矩阵的非对角线模式——非对角值代表状态间的迁移偏好，是 HMM 动态特征的核心体现。
3. 只看准确率不看转移矩阵——高准确率 + 差的转移矩阵表明可能只是逐点 argmax 的效果。
4. 把 HMM 的准确率与分类准确率直接比较——两者的评估对象和意义完全不同。

## 小结

- HMM 当前有两项评估输出：隐状态准确率（定量，Viterbi 路径 vs 真实路径）和转移矩阵（诊断，学习到的 vs 真实的马尔可夫动态）。
- HMM 评估的核心价值在于"时序一致性"——不仅看单点准确，还要看状态间的迁移模式是否正确恢复。
- 与所有其他模型的评估差异源于序列数据的特性——评估的是"路径推断"而非"标签分类"或"聚类质量"。

# 工程实现

## 本章目标

1. 理解 HMM 流水线的模块分层——数据生成层、模型训练层、流水线编排层（无可视化层）。
2. 理清 `run()` 内部的函数调用链——HMM 是本仓库最简流水线，无标准化/切分/可视化。
3. 理解 HMM 与 GMM（EM）在工程实现上的关键差异——序列数据、离散观测、双备份依赖。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.hmm()` | 方法 | 手动参数化的 HMM 序列数据——含真实隐状态和观测 |
| `train_model(...)` | 函数 | 构建并训练 `CategoricalHMM`——含双备份可选依赖检查 |
| `run()` | 函数 | 序列模型流水线编排——4 步完成数据整形、训练、Viterbi 解码和评估 |
| `model.predict(X_obs, lengths)` | 方法 | Viterbi 解码——全局最优隐状态路径 |
| `model.score(X_obs, lengths)` | 方法 | Forward 算法——对数似然（可用于 diagnostic） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/probabilistic.py` | 手动参数化生成 HMM 序列数据并导出 `hmm_data` | 全局 `DataFrame`（300 行 x 3 列） |
| 模型训练层 | `model_training/probabilistic/hmm.py` | 封装 `CategoricalHMM` 训练——含双备份可选依赖处理 | `CategoricalHMM` 模型对象 |
| 流水线编排层 | `pipelines/probabilistic/hmm.py` | 串联数据整形、训练、Viterbi 解码和准确率评估——端到端入口 | 终端日志 + 准确率 + 转移矩阵 |
| 可视化层 | **无** | HMM 序列数据不适合散点图/矩阵图——终端文本输出已足够 | — |

### 理解重点

- HMM 是本仓库唯一**没有可视化层**的流水线——序列推断的评估更适合用终端文本（准确率 + 转移矩阵）。
- 训练层有双备份依赖处理——`CategoricalHMM`（hmmlearn 0.3+）/ `MultinomialHMM`（旧版），任一可用即可。
- 与 GMM 共享 `data_generation/probabilistic.py`——两者均为概率模型，数据生成器统一管理。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `hmm_data.copy()` | — | `DataFrame`，形状 `(300, 3)` | 避免修改全局变量 |
| 2 | `data["obs"].values.astype(int)` | `DataFrame` | `ndarray`，`(300,)` | 提取离散观测序列 |
| 3 | `obs.reshape(-1, 1)` | `ndarray` | `ndarray`，`(300, 1)` | 整形为 hmmlearn 要求的列向量 |
| 4 | `[len(obs)]` | — | `list[int]` | 序列长度——单条 300 步 |
| 5 | `data["state_true"].values.astype(int)` | `DataFrame` | `ndarray`，`(300,)` | 提取真实隐状态——仅用于评估 |
| 6 | `train_model(X_obs, lengths)` | `(ndarray, list)` | `CategoricalHMM` | Baum-Welch 训练 |
| 7 | `model.predict(X_obs, lengths)` | `(ndarray, list)` | `ndarray`，`(300,)` | Viterbi 解码 |
| 8 | `np.mean(states_pred == y_true)` | `(ndarray, ndarray)` | `float` | 逐步准确率计算 |
| 9 | `model.transmat_.round(3)` | 属性访问 | — | 打印学习到的转移矩阵 |

### 理解重点

- 步骤 3（`reshape`）是 hmmlearn 接口特有的数据整形——观测必须为列向量。
- 步骤 6 内部触发可选依赖检查——如果 `hmmlearn` 未安装会抛出 `ImportError`。
- 与 GMM 流水线的关键差异：无 `StandardScaler`（离散观测无需缩放）、无 `plot_clusters`（序列不可散点图化）。

## 3. 数据依赖关系

```
hmm_data (全局 DataFrame)
  - -> obs = data["obs"].values ──-> reshape(-1, 1) ──-> X_obs ──┐
  - -> lengths = [len(obs)] ────────────────────────────────────┤
  - -> y_true = data["state_true"].values ─────────────────────┐│
  - train_model(X_obs, lengths) ──-> model                    ││
    - -> model.predict(X_obs, lengths) ──-> states_pred ──┐ ││
  - accuracy = np.mean(states_pred == y_true) <-─────────────┘ ││
  - print(model.transmat_) <-──────────────────────────────────┘│
```

### 理解重点

- `y_true` 仅用于评估——不经过训练模块，直接与 `states_pred` 对比。
- `lengths` 与 `X_obs` 同时传入 `train_model` 和 `model.predict`——HMM 必须知道序列边界。
- 这是本仓库最简单的数据依赖图——无标准化分支、无可视化分支、无切分分支。

## 4. 输出一览

### 参数速览

| 输出项 | 路径/位置 | 格式 | 说明 |
|---|---|---|---|
| 隐状态准确率 | 标准输出 | 文本 `float` | Viterbi 路径与真实状态的逐步匹配率 |
| 转移矩阵 | 标准输出 | 文本 `ndarray` | 学习到的 3x3 转移矩阵（行和为 1） |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.probabilistic.hmm
```

### 输出

```text
============================================================
HMM 流水线
============================================================
模型训练完成
n_components: 3
n_iter: 100
tol: 0.001
模型训练耗时: 0.08s

隐状态预测准确率: 0.8933
转移矩阵:
[[0.782  0.176  0.042 ]
 [0.215  0.582  0.203 ]
 [0.118  0.223  0.659 ]]

============================================================
HMM 流水线完成！
============================================================
```

### 理解重点

- HMM **无任何文件输出**——所有评估结果以终端文本呈现，是本仓库唯一纯终端输出的流水线。
- 训练耗时极短（~0.08s）——300 步 x 3 状态，Baum-Welch 在此规模上收敛很快。
- 转移矩阵保留 3 位小数——足够直观对比学习结果与真实参数的差异。

## 5. 训练层细节：与 GMM 的对比

| 工程维度 | GMM (EM) | HMM |
|---|---|---|
| 模型类 | `GaussianMixture` | **`CategoricalHMM` / `MultinomialHMM`（双备份）** |
| 依赖 | sklearn 内置 | **`pip install hmmlearn`（可选依赖）** |
| 训练输入 | `fit(X)`——独立样本矩阵 | **`fit(X, lengths)`——序列列向量 + 长度列表** |
| 算法 | EM（E 步: 逐点后验, M 步: 加权更新） | **Baum-Welch（E 步: Forward-Backward, M 步: 计数重估）** |
| 预测 | `predict(X)` + `predict_proba(X)` | **`predict(X, lengths)`（Viterbi）——无 `predict_proba`** |
| 模型属性 | `means_`、`covariances_`、`weights_` | **`transmat_`、`emissionprob_`、`startprob_`** |
| 标准化 | 有 | **无**（离散观测） |
| 可视化 | 聚类图 | **无**（终端文本） |
| 装饰器 | `@print_func_info` + `@timeit` + `timer` | `@print_func_info` + `@timeit` + `timer`——相同 |

### 理解重点

- HMM 的训练层设计是四个概率模型中最特殊的——输入不是 `(X, y)` 或 `(X)`，而是 `(X, lengths)`。
- 双备份依赖是可选的极限容错——无论用户装的是新版还是旧版 hmmlearn，都能正常运行。
- 无 `predict_proba` 但有 `score`（Forward 算法给出对数概率）——两者的评估用途不同。

## 阅读顺序

1. `data_generation/probabilistic.py` — 了解 `hmm()` 的 HMM 序列数据生成逻辑
2. `model_training/probabilistic/hmm.py` — 理解 `CategoricalHMM` 的构建、双备份依赖和 Baum-Welch 训练
3. `pipelines/probabilistic/hmm.py` — 看清序列流水线的端到端流程和终端评估

## 常见坑

1. 在不含 `hmmlearn` 的环境中直接 `from model_training.probabilistic.hmm import train_model`——会抛出 `ImportError`，需先 `pip install hmmlearn`。
2. 忘记将观测 `reshape(-1, 1)`——hmmlearn 的 `fit` 要求观测为 `(n_steps, 1)` 形状。
3. 把 `astype(int)` 漏掉——hmmlearn 可能将 float 观测处理为连续值，触发错误的模型行为。
4. 混淆 `CategoricalHMM` 和 `MultinomialHMM` 的参数——两者基本相同，但类名和包路径不同。

## 小结

- HMM 工程实现遵循三层架构（无可视化层）：数据生成层 -> 模型训练层 -> 流水线编排层。
- `run()` 是本仓库最简编排函数——4 步核心操作完成数据整形、训练、Viterbi 解码和评估，所有输出均为终端文本。
- 与 GMM 的四个关键工程差异：（1）序列输入 + lengths；（2）离散观测无需标准化；（3）双备份可选依赖；（4）无可视化层（纯终端评估）。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 HMM 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 HMM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Forward/Viterbi/Baum-Welch、马尔可夫性、HMM vs GMM 等核心概念的理解 |
| 动手练习 | 实践 | 修改参数和模型配置观察 HMM 行为——建立序列模型直觉 |
| 参考文献 | 入口 | 提供 HMM 经典教材和 Rabiner 教程 |

## 1. 自检问题

1. HMM 的三个基本问题是什么？Forward、Viterbi、Baum-Welch 分别解决哪个问题？

2. Forward 算法和 Viterbi 算法都是动态规划——两者的递推公式有何本质区别？一个求和，一个取最大，这反映了什么不同的目标？

3. 为什么 Viterbi 解码优于逐步 $\arg\max_i P(s_t = i \mid O, \lambda)$？给出一个简单例子说明后者可能产生非法状态转移。

4. Baum-Welch 与 GMM 的 EM 在 E 步上的本质区别是什么？为什么 HMM 的 E 步需要 Forward-Backward 而非逐点后验？

5. 马尔可夫性假设 $P(s_t \mid s_1, \dots, s_{t-1}) = P(s_t \mid s_{t-1})$ 在什么实际场景下会被违反？一阶 HMM 如何被扩展来处理更高阶的依赖？

6. 转移矩阵 $A$ 的行和为 1，对角元素通常较大——这反映了隐状态的什么特性？如果对角元素都接近 0.33（3 状态），说明什么？

7. HMM 和 GMM 都是概率生成模型——它们在数据结构、独立性假设、隐变量含义上有哪些根本差异？HMM 可以视为什么结构的概率模型？

## 2. 动手练习

### 练习 1：改变隐状态数 `n_components`

将 `n_components` 分别设为 `2`、`3`、`4`、`5`，观察准确率和转移矩阵的变化。

```python
model = train_model(X_obs, lengths, n_components=2)
```

回答：`n_components=2` 时 HMM 如何将 3 个真实状态"合并"为 2 个？准确率是否显著下降？`n_components=5` 时是否出现了"多余"的状态？

### 练习 2：改变序列长度

修改 `data_generation/probabilistic.py` 中的 `hmm_n_steps`（分别设为 `50`、`100`、`300`、`1000`），观察准确率的变化。

```python
# 在 ProbabilisticData 中
hmm_n_steps: int = 50  # 试试 50, 100, 300, 1000
```

回答：序列越短，Baum-Welch 估计的转移矩阵越不稳定——具体多短时准确率开始显著下降？

### 练习 3：改变转移矩阵的惯性

修改 `hmm_A` 的对角线值（如将 `0.8` 分别改为 `0.5` 和 `0.95`），观察转移矩阵的学习精度。

```python
# 低惯性
hmm_A: list = [[0.50, 0.30, 0.20], [0.30, 0.40, 0.30], [0.20, 0.30, 0.50]]
# 高惯性
hmm_A: list = [[0.95, 0.03, 0.02], [0.03, 0.94, 0.03], [0.03, 0.03, 0.94]]
```

回答：高惯性（状态几乎不跳变）的转移矩阵是否更容易被 HMM 恢复？为什么？

### 练习 4：对比 `predict`（Viterbi）与逐点 argmax

手动实现逐点 argmax 解码（不使用 Viterbi），对比两者的准确率。

```python
# 计算后验概率（需要自己实现 Forward-Backward 或使用 model.score_samples）
# 逐点 argmax: ŝ_t = argmax_i gamma_t(i)
```

回答：有没有发现逐点 argmax 产生了"不可能的转移"（状态 0 -> 2 等）？哪种方法的准确率更高？

### 练习 5：使用 Forward 得分评估模型质量

比较不同 `n_components` 下训练模型的 `score`（Forward 对数概率）。

```python
for k in [2, 3, 4, 5]:
    model = train_model(X_obs, lengths, n_components=k)
    log_prob = model.score(X_obs, lengths)
    print(f"K={k}: log P(O|lambda) = {log_prob:.2f}")
```

回答：对数概率随 $K$ 增大是否单调递增（总是偏好更多参数）？是否可以用 BIC 来平衡拟合和复杂度？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE, 77(2), 257-286. | HMM 最经典的入门教程——Forward/Viterbi/Baum-Welch 的完整推导和语音识别应用 |
| 2 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 13. | 教材——HMM 的概率图模型视角和变分推断推广 |
| 3 | hmmlearn 官方文档 — [CategoricalHMM](https://hmmlearn.readthedocs.io/en/latest/api.html#categoricalhmm) | hmmlearn 的 API 参考——参数、方法和使用示例 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 17. | 教材——HMM 的马尔可夫链理论、卡尔曼滤波推广和状态空间模型 |

## 常见坑

1. 在真实数据（非合成）上期待完美的隐状态准确率——真实数据的 HMM 假设（马尔可夫 + 离散观测）常被违反。
2. 把 HMM 的 `score` 当成"准确率"——`score` 返回对数概率（负值绝对值越小越好），不是匹配比例。
3. 以为 `n_components` 越大越好——过度设定状态数会导致每个状态下观测极少，转移矩阵估计不稳定。
4. 忘记 hmmlearn 的列向量要求——`reshape(-1, 1)` 是必需的数据整形步骤。

## 小结

- 7 个自检问题覆盖 HMM 的核心概念：三个基本问题、Forward vs Viterbi、Viterbi vs 逐点 argmax、Baum-Welch vs EM、马尔可夫性、转移矩阵解读、HMM vs GMM。
- 5 个动手练习从不同角度探索 HMM 的行为——改变状态数、序列长度、转移惯性、对比解码方式、使用 Forward 得分做模型选择。
- 4 篇参考文献覆盖经典入门（Rabiner 1989）、教材和官方文档——构成完整的 HMM 学习路线。
