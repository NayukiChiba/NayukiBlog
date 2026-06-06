---
title: 概率不等式完全指南：从马尔可夫到切尔诺夫
date: 2026-06-05
category: 数学
tags:
  - 概率论
  - 不等式
  - 集中不等式
  - 机器学习理论
description: 系统讲解概率论中最核心的六大不等式：马尔可夫、切比雪夫、切尔诺夫、霍夫丁、琴生、柯西-施瓦茨。从每条不等式的证明、直观含义、紧性分析到应用场景，以假设-结论对照表串联不等式之间的层级关系。
image: https://img.yumeko.site/file/blog/cover/1780668202244.webp
status: published
---

## 1. 为什么需要概率不等式？

概率论中，我们经常需要回答这类问题：

> 一枚硬币抛 $n$ 次，正面频率偏离 $0.5$ 超过 $0.1$ 的概率有多大？

如果知道精确分布（二项分布），可以直接算。但更常见的情形是：**只知道期望和方差**，或者**只知道变量有界**，甚至**只知道期望存在**。概率不等式就是在这些"部分信息"下，给出概率上界的工具。

$$
\text{已知条件越少} \;\longrightarrow\; \text{需要的假设越弱} \;\longrightarrow\; \text{给出的界越松}
$$

本文从最基础的马氏不等式出发，逐步收紧条件，引入越来越强的集中不等式。

## 2. 马尔可夫不等式：最弱假设，最基础的界

### 2.1 定理与证明

设 $X$ 是**非负随机变量**（$X \ge 0$ a.s.），$E[X] < \infty$。则对任意 $a > 0$：

$$
\boxed{P(X \ge a) \le \frac{E[X]}{a}}
$$

**证明**（期望拆分法）：

$$
\begin{aligned}
E[X] &= E[X \cdot \mathbf{1}_{\{X \ge a\}}] + E[X \cdot \mathbf{1}_{\{X < a\}}] \\
     &\ge E[X \cdot \mathbf{1}_{\{X \ge a\}}] \quad\text{（第二项 } X \ge 0\text{）} \\
     &\ge E[a \cdot \mathbf{1}_{\{X \ge a\}}] \quad\text{（在 } \{X \ge a\} \text{ 上 } X \ge a\text{）} \\
     &= a \cdot P(X \ge a)
\end{aligned}
$$

两端同除以 $a$ 即得。 ∎

### 2.2 直觉与局限性

**直觉**：一个非负随机变量，其期望 $E[X]$ 确定了"概率质量"的总额。如果 $P(X \ge a)$ 很大，那一部分质量至少是 $a \cdot P(X \ge a)$，它不能超过总质量 $E[X]$。

**马尔可夫不等式的松紧**：以指数分布 $X \sim \text{Exp}(\lambda)$（$E[X] = 1/\lambda$）为例：

| $a$ | 马尔可夫界 | 精确值 $e^{-\lambda a}$ | 比值 |
|:--|:--|:--|:--|
| $E[X]$ | $1.000$ | $0.368$ | 2.7$\times$ |
| $3E[X]$ | $0.333$ | $0.050$ | 6.7$\times$ |
| $5E[X]$ | $0.200$ | $0.007$ | 29$\times$ |

马尔可夫界是**绝对安全的上界**，但远不是紧的。其普适性（仅需 $X \ge 0$ 和 $E[X]$ 存在）的代价是保守性。

![MarkovBound.png](https://img.yumeko.site/file/blog/articles/1780733458877.webp)

### 2.3 经典考题

**华为 2026 机试真题**：设 $X$ 是非负整数随机变量，求证 $P(X \ge 1) \le E[X]$。

直接应用马尔可夫不等式（$a = 1$）：$P(X \ge 1) \le E[X]/1 = E[X]$。

**延伸**：由 $E[X] = \sum_{k=1}^{\infty} P(X \ge k) \ge P(X \ge 1)$ 可验证一致。

---

## 3. 切比雪夫不等式：引入方差

### 3.1 定理与推导

设随机变量 $Y$ 具有有限期望 $\mu = E[Y]$ 和有限方差 $\sigma^2 = \operatorname{Var}(Y)$。则对任意 $\varepsilon > 0$：

$$
\boxed{P(|Y - \mu| \ge \varepsilon) \le \frac{\sigma^2}{\varepsilon^2}}
$$

等价形式（以标准差 $k\sigma$ 为单位）：

$$
P(|Y - \mu| \ge k\sigma) \le \frac{1}{k^2}
$$

**证明思路**：切比雪夫 = 马尔可夫 + 平方变换。

令 $X = (Y - \mu)^2 \ge 0$，则 $E[X] = \sigma^2$。对 $X$ 应用马尔可夫不等式，取 $a = \varepsilon^2$：

$$
P(|Y - \mu| \ge \varepsilon) = P((Y - \mu)^2 \ge \varepsilon^2) \le \frac{E[(Y - \mu)^2]}{\varepsilon^2} = \frac{\sigma^2}{\varepsilon^2}
$$

两步即得。 ∎

![ChebyshevDerivation.png](https://img.yumeko.site/file/blog/articles/1780733462950.webp)

### 3.2 紧性分析

切比雪夫不等式的 $1/k^2$ 衰减率在"仅知方差"的前提下**不可改进**。

**构造达到界的分布**：对任意 $k > 0$，取离散分布：

$$
Y = \begin{cases}
\mu + k\sigma & \text{以概率 } \frac{1}{2k^2} \\[4pt]
\mu & \text{以概率 } 1 - \frac{1}{k^2} \\[4pt]
\mu - k\sigma & \text{以概率 } \frac{1}{2k^2}
\end{cases}
$$

验证 $E[Y] = \mu$，$\operatorname{Var}(Y) = \sigma^2$，且 $P(|Y - \mu| \ge k\sigma) = \frac{1}{k^2}$——恰好达到切比雪夫界。

### 3.3 与正态分布对比

| $k$ | 切比雪夫界 | 正态精确值 | 差距 |
|:--|:--|:--|:--|
| $2$ | $0.250$ | $0.046$ | 5.4$\times$ |
| $3$ | $0.111$ | $0.0027$ | 41$\times$ |
| $4$ | $0.0625$ | $6.3\times 10^{-5}$ | ~1000$\times$ |

正态分布有指数衰减的尾部（来自矩母函数），切比雪夫只有多项式衰减——这是"仅知方差"必须付出的代价。

### 3.4 核心应用：样本均值的集中性

设 $X_1, \dots, X_n$ i.i.d.，$E[X_i] = \mu$，$\operatorname{Var}(X_i) = \sigma^2$。样本均值 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ 满足：

$$
E[\bar{X}_n] = \mu,\qquad \operatorname{Var}(\bar{X}_n) = \frac{\sigma^2}{n}
$$

应用切比雪夫：

$$
\boxed{P(|\bar{X}_n - \mu| \ge \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2}}
$$

当 $n \to \infty$ 时右侧 $\to 0$——这就是 [[Math/LawOfLargeNumbers|弱大数定律]] 的入口，仅用了切比雪夫不等式和三行推导。

---

## 4. 切尔诺夫界：指数化马尔可夫

### 4.1 核心思想

马尔可夫不等式太松，因为它只用了 $E[X]$。**切尔诺夫界**（Chernoff bound）的核心技巧是：先对随机变量取指数变换，再应用马尔可夫不等式，最后对变换参数优化。

设 $X$ 是任意随机变量（不要求非负）。对任意 $t > 0$：

$$
P(X \ge a) = P(e^{tX} \ge e^{ta}) \le \frac{E[e^{tX}]}{e^{ta}} = e^{-ta} M_X(t)
$$

其中 $M_X(t) = E[e^{tX}]$ 是 $X$ 的**矩母函数**（Moment Generating Function）。对 $t > 0$ 取最小值得最紧界：

$$
\boxed{P(X \ge a) \le \inf_{t > 0} \ e^{-ta} M_X(t)}
$$

**三步曲**：指数化 $\rightarrow$ 马尔可夫 $\rightarrow$ 优化 $t$。

![ChernoffBound.png](https://img.yumeko.site/file/blog/articles/1780733466800.webp)

### 4.2 独立和的切尔诺夫界

设 $S_n = \sum_{i=1}^n X_i$，$X_i$ 相互独立。则由矩母函数的乘积性质 $M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t)$：

$$
\boxed{P(S_n - E[S_n] \ge \delta) \le \inf_{t > 0} \ e^{-t(\mu + \delta)} \prod_{i=1}^n E[e^{tX_i}]}
$$

### 4.3 正态分布的切尔诺夫界（紧的！）

设 $X_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$，$S_n = \sum_{i=1}^n X_i$：

矩母函数：$E[e^{t(X_i - \mu)}] = e^{\sigma^2 t^2 / 2}$

$$
P(\bar{X}_n - \mu \ge \varepsilon) \le \inf_{t>0} \exp\left(-t n\varepsilon + \frac{n\sigma^2 t^2}{2}\right)
$$

对 $t$ 求导得最优 $t^* = \varepsilon / \sigma^2$，代入：

$$
\boxed{P(\bar{X}_n - \mu \ge \varepsilon) \le \exp\left(-\frac{n\varepsilon^2}{2\sigma^2}\right)}
$$

**对比切比雪夫**：切比雪夫给出 $O(1/n)$ 的衰减，切尔诺夫给出 $O(e^{-n})$ 的衰减——对于正态分布，切尔诺夫达到了**指数级紧度**。

### 4.4 伯努利分布的切尔诺夫界

设 $X_i \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$，$\bar{X}_n$ 是样本均值。矩母函数：$E[e^{tX_i}] = 1 - p + p e^t$。

经过优化推导（利用 $1 + x \le e^x$），得到常用的**加法形式**：

$$
\boxed{P(\bar{X}_n \ge p + \varepsilon) \le \exp\left(-n \cdot D_{\text{KL}}(p + \varepsilon \,\|\, p)\right)}
$$

其中 $D_{\text{KL}}(q \| p) = q \ln\frac{q}{p} + (1-q)\ln\frac{1-q}{1-p}$ 是 KL 散度。

---

## 5. 霍夫丁不等式：有界变量的集中

### 5.1 定理陈述

设 $X_1, \dots, X_n$ 是独立随机变量，$X_i \in [a_i, b_i]$（有界性假设）。令 $S_n = \sum_{i=1}^n X_i$。则对任意 $\varepsilon > 0$：

$$
\boxed{P(S_n - E[S_n] \ge \varepsilon) \le \exp\left(-\frac{2\varepsilon^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)}
$$

对样本均值的等价形式（假设 $X_i \in [a, b]$ 同界）：

$$
\boxed{P(|\bar{X}_n - \mu| \ge \varepsilon) \le 2 \exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right)}
$$

### 5.2 与切比雪夫的对比

同样是有界变量，切比雪夫给出：
$$
P(|\bar{X}_n - \mu| \ge \varepsilon) \le \frac{\operatorname{Var}(X_1)}{n\varepsilon^2} \propto \frac{1}{n}
$$

霍夫丁给出：
$$
P(|\bar{X}_n - \mu| \ge \varepsilon) \le 2\exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right) \propto e^{-n}
$$

**指数级 vs 多项式级**——这是"仅知方差"和"知有界性"之间巨大的信息差距。

### 5.3 证明思路

霍夫丁不等式的证明核心是**霍夫丁引理**（Hoeffding's Lemma）：

> 若 $X \in [a, b]$ 且 $E[X] = 0$，则 $E[e^{tX}] \le \exp\left(\frac{t^2(b-a)^2}{8}\right)$。

然后对 $S_n - E[S_n]$ 应用切尔诺夫界，利用独立性将矩母函数拆分为乘积，再对每个因子应用霍夫丁引理，最后优化 $t$。

### 5.4 机器学习中的应用

霍夫丁不等式是 PAC 学习理论（Probably Approximately Correct）的基石。训练误差 $\hat{R}(h)$ 与泛化误差 $R(h)$ 之间的差距，如果假设空间 $\mathcal{H}$ 有限（$|\mathcal{H}| = M$），通过联合界（union bound）+ 霍夫丁不等式：

$$
P\left(\sup_{h \in \mathcal{H}} |\hat{R}(h) - R(h)| \ge \varepsilon\right) \le 2M \exp(-2n\varepsilon^2)
$$

反解 $n$ 即得样本复杂度下界：
$$
n \ge \frac{1}{2\varepsilon^2} \ln\frac{2M}{\delta} \quad\text{（以概率 } 1-\delta \text{ 保证 } \varepsilon\text{-泛化）}
$$

---

## 6. 琴生不等式：凸函数的期望

### 6.1 定理

设 $f$ 是**凸函数**（convex），$X$ 是随机变量（$E[|X|] < \infty$，且 $X$ 取值在 $f$ 的定义域内）。则：

$$
\boxed{f(E[X]) \le E[f(X)]}
$$

若 $f$ 是**凹函数**，不等式方向反转：$E[f(X)] \le f(E[X])$。

### 6.2 概率论中的经典应用

**（1）方差非负**：$f(x) = x^2$ 是凸函数 $\rightarrow$ $E[X]^2 \le E[X^2]$，即 $\operatorname{Var}(X) = E[X^2] - E[X]^2 \ge 0$。

**（2）KL 散度非负**：$f(x) = -\ln x$ 是凸函数 $\rightarrow$ $D_{\text{KL}}(P \| Q) = E_P[-\ln(q/p)] \ge -\ln(E_P[q/p]) = -\ln(1) = 0$。

**（3）EM 算法的收敛性**：EM 算法的 E 步构造了一个凹函数的下界，M 步最大化该下界——琴生不等式保证了每次迭代后似然函数不降。

**（4）算术平均 $\geq$ 几何平均**：$f(x) = -\ln x$ 凸 $\rightarrow$ $-\ln E[X] \le E[-\ln X]$ $\rightarrow$ $E[X] \ge \exp(E[\ln X])$。

### 6.3 与马尔可夫不等式的关系

琴生不等式也可用于推导马尔可夫不等式（对于非负 $X$，取 $f(x) = x \cdot \mathbf{1}_{\{x \ge a\}}$），但这不是其主要用途。琴生的真正威力在于建立期望和变换之间的不等式关系，这在信息论和优化中无处不在。

---

## 7. 柯西-施瓦茨不等式：二阶矩的界

### 7.1 概率形式

对任意两个具有有限二阶矩的随机变量 $X, Y$：

$$
\boxed{\left|E[XY]\right| \le \sqrt{E[X^2] \cdot E[Y^2]}}
$$

等号成立当且仅当 $X$ 和 $Y$ 几乎必然线性相关（$Y = cX$ a.s. 或 $X = cY$ a.s.）。

### 7.2 核心推论

**（1）协方差的界**：取 $X' = X - E[X]$，$Y' = Y - E[Y]$：

$$
|\operatorname{Cov}(X, Y)| \le \sqrt{\operatorname{Var}(X) \cdot \operatorname{Var}(Y)} = \sigma_X \sigma_Y
$$

**（2）相关系数的范围**：由上式直接推出 $|\rho_{XY}| = \frac{|\operatorname{Cov}(X,Y)|}{\sigma_X \sigma_Y} \le 1$。

**（3）马尔可夫不等式的加强版**（Paley-Zygmund 不等式）：柯西-施瓦茨给出下界方向的控制：

$$
P(X > 0) \ge \frac{E[X]^2}{E[X^2]}
$$

这有时被称为"反集中不等式"——它不仅说概率不能太大（马氏），还说在有一定二阶矩信息时，概率也**不能太小**。

### 7.3 与其他不等式的关系

柯西-施瓦茨是 Hölder 不等式在 $p = q = 2$ 时的特例。Hölder 的一般形式：

$$
E[|XY|] \le E[|X|^p]^{1/p} \cdot E[|Y|^q]^{1/q}, \quad \frac{1}{p} + \frac{1}{q} = 1
$$

---

## 8. 六大不等式对照总表

### 8.1 假设 — 结论 — 衰减速度

| 不等式 | 所需假设 | 对 $P(\vert\bar{X}_n - \mu\vert \ge \varepsilon)$ 的界 | 衰减速度 | 典型应用 |
|:--|:--|:--|:--|:--|
| **马尔可夫** | $X \ge 0$，$E[X]$ 存在 | $\frac{E[X]}{a}$（单变量） | $O(1/a)$ | 最基础的概率界 |
| **切比雪夫** | $\operatorname{Var}(X)$ 存在 | $\frac{\sigma^2}{n\varepsilon^2}$ | $O(1/n)$ | WLLN、样本量估计 |
| **切尔诺夫** | 矩母函数 $E[e^{tX}]$ 存在 | $\exp\left(-\frac{n\varepsilon^2}{2\sigma^2}\right)$（正态） | $O(e^{-n})$ | 指数级紧的尾部界 |
| **霍夫丁** | $X_i \in [a_i, b_i]$ 有界 | $2\exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right)$ | $O(e^{-n})$ | PAC 学习、Bandit |
| **琴生** | $f$ 凸，$E[X]$ 存在 | — | — | EM、信息论 |
| **柯西-施瓦茨** | $E[X^2], E[Y^2]$ 存在 | — | — | 相关系数、方差分解 |

### 8.2 不等式层级关系

六条不等式之间有清晰的推导与加强关系：

1. **马尔可夫不等式**是源头——仅需 $X \ge 0$ 和 $E[X]$ 存在，所有其他不等式都直接或间接源于它。
2. **切比雪夫不等式**由马尔可夫通过平方变换 $(X - \mu)^2$ 导出，引入方差信息但保持 $O(1/\varepsilon^2)$ 的衰减速度。
3. **切尔诺夫界**由马尔可夫通过指数变换 $e^{tX}$ 并优化 $t$ 导出，在矩母函数存在的条件下达到 $O(e^{-n})$ 的指数级衰减。
4. **霍夫丁不等式**是切尔诺夫界在有界独立变量上的应用（通过霍夫丁引理控制每个因子的矩母函数），同样达到 $O(e^{-n})$。
5. **琴生不等式**独立于上述链条——它不直接给出概率上界，而是建立 $f(E[X])$ 与 $E[f(X)]$ 的关系，是 EM 算法和信息论的基石。
6. **柯西-施瓦茨不等式**利用二阶矩信息给出 $|E[XY]| \le \sqrt{E[X^2]E[Y^2]}$，是相关性理论和 Paley-Zygmund 反集中不等式的起点。

**核心规律**：马氏给出 $O(1/a)$（最松），切氏给出 $O(1/\varepsilon^2)$，切尔诺夫和霍夫丁在更强假设下给出 $O(e^{-n})$ 的指数级衰减。假设越强，界越紧——这是概率不等式的基本 trade-off。

![InequalityComparison (2).png](https://img.yumeko.site/file/blog/articles/1780733477211.webp)


| 你手中的信息 | 用什么不等式 |
|:--|:--|
| 只知道期望 | 马尔可夫 |
| 知道期望 + 方差 | 切比雪夫 |
| 知道矩母函数 + 独立 | 切尔诺夫 |
| 知道有界 + 独立 | 霍夫丁 |
| 需要凸/凹函数关系 | 琴生 |
| 需要二阶矩的界 | 柯西-施瓦茨 |

---

> **相关文章**：
> - [[Math/LawOfLargeNumbers|大数定律详解]]——这些不等式是证明大数定律的核心工具
> - [[Math/MultivariateStatistics|多元统计分析]]——协方差矩阵与多元推断
