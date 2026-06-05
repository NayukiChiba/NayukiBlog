---
title: 大数定律详解：从伯努利到柯尔莫哥洛夫
date: 2026-06-04
category: 数学
tags:
  - 概率论
  - 数理统计
  - 收敛
  - 大数定律
description: 系统讲解六种大数定律——伯努利、辛钦、切比雪夫、马尔可夫（弱大数定律）与博雷尔、柯尔莫哥洛夫（强大数定律）——的数学定义、假设条件、证明思路及相互关系。厘清"依概率收敛"与"几乎必然收敛"的本质区别。
image: https://img.yumeko.site/file/blog/LawOfLargeNumbers.png
status: draft
---

> **前置阅读**：大数定律的证明依赖多项概率不等式。建议先阅读 [[Math/ProbabilityInequalities|概率不等式完全指南]]，了解马尔可夫、切比雪夫等基础工具。

![图0: 大数定律概览——从随机波动到确定性收敛](https://img.yumeko.site/file/blog/LawOfLargeNumbers/LLNBanner.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A wide banner image (2.35:1 aspect ratio) for a blog post about the Law of Large Numbers. Design concept: Left side shows a chaotic, noisy scatter of individual random variables (small translucent dots fluctuating wildly). Moving right, the dots coalesce into a running average line that gradually smooths out and converges toward a horizontal dashed line labeled "μ" (the expected value). The rightmost portion shows the sample mean tightly hugging μ. Two labels: "n = 10" (left, jittery), "n = 100" (middle, smoothing), "n = 10,000" (right, flat at μ). Color palette: deep navy to warm gold gradient, modern data-science aesthetic. Clean sans-serif labels, faint grid background. Mathematical textbook meets infographic style. Leave space at top for title overlay.
> ```

## 1. 直觉：为什么需要大数定律？

抛一枚均匀硬币 $n$ 次，正面朝上的频率记为 $\bar{X}_n$。直觉告诉我们：

$$
\bar{X}_n \xrightarrow{n \to \infty} \frac{1}{2}
$$

但这里的"箭头"是什么意思？是指 $\bar{X}_n$ 和 $0.5$ 的差趋于 $0$？还是一定会趋于 $0$？还是"几乎一定"趋于 $0$？

**大数定律**（Law of Large Numbers, LLN）就是把这种直觉精确化的数学定理。但大数定律不是**一个**定理——它是一个**定理家族**，随着对随机变量假设的逐步放松（从 Bernoulli 的"独立同分布、二阶矩存在"到辛钦的"独立同分布、仅一阶矩存在"），结论的强度也从弱收敛过渡到强收敛。

本文从历史最悠久的伯努利大数定律出发，逐步放松条件，引出六种经典形式，并说明它们之间的蕴含关系。

## 2. 前置知识：随机变量的收敛模式

理解两个大数定律的区别，必须先理解随机变量序列的四种收敛方式。设 $\{X_n\}$ 是随机变量序列，$X$ 是随机变量。

### 2.1 依概率收敛（Convergence in Probability）

$$
X_n \xrightarrow{P} X \quad\iff\quad
\forall \varepsilon > 0,\ \lim_{n \to \infty} P(|X_n - X| > \varepsilon) = 0
$$

**含义**：随着 $n$ 增大，$X_n$ 偏离 $X$ 超过任意小量 $\varepsilon$ 的**概率**趋于 $0$。注意——它不保证对于某个具体的 $\omega$（样本点），$X_n(\omega)$ 一定趋于 $X(\omega)$。

**经典反例**：构造一个"游荡的 1"序列：

$$
\begin{aligned}
X_1 &= \mathbf{1}_{[0,1]} \\
X_2 &= \mathbf{1}_{[0,1/2]},\quad X_3 = \mathbf{1}_{[1/2,1]} \\
X_4 &= \mathbf{1}_{[0,1/4]},\quad X_5 = \mathbf{1}_{[1/4,1/2]},\quad X_6 = \mathbf{1}_{[1/2,3/4]},\quad X_7 = \mathbf{1}_{[3/4,1]} \\
&\vdots
\end{aligned}
$$

$X_n \xrightarrow{P} 0$（$1$ 的区间越来越短），但对于**每一个**具体的 $\omega \in [0,1]$，$X_n(\omega)$ 都会无限多次取到 $1$——永远不收敛。

![图1: "游荡1"反例——依概率收敛但不几乎必然收敛](https://img.yumeko.site/file/blog/LawOfLargeNumbers/WanderingOne.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A mathematical visualization of the "wandering 1" counterexample showing convergence in probability but not almost sure convergence. Four small panels stacked vertically, each showing a unit interval [0,1] on the x-axis. Panel 1 (top): "Step 1" — a single red bar spanning the entire [0,1] interval (X₁ = 1 everywhere). Panel 2: "Step 2-3" — two narrower red bars each of width 1/2 (X₂ on left half, X₃ on right half). Panel 3: "Step 4-7" — four even narrower red bars each of width 1/4. Panel 4: "Step 8-15" — eight very thin red bars each of width 1/8. Below the panels, show a sample path plot for a fixed ω (e.g., ω = 0.37): a timeline from n=1 to n=20 where the indicator value keeps returning to 1 infinitely often, never settling to 0. Label: "For any fixed ω, Xₙ(ω) does NOT converge to 0". Clean academic figure, white background, thin axis lines. Mathematical textbook style.
> ```

### 2.2 几乎必然收敛（Almost Sure Convergence）

$$
X_n \xrightarrow{a.s.} X \quad\iff\quad
P\left( \lim_{n \to \infty} X_n = X \right) = 1
$$

或等价地：

$$
P\left( \{\omega : X_n(\omega) \to X(\omega)\} \right) = 1
$$

**含义**：除了一个零测集外，对**几乎每一个**样本点 $\omega$，数列 $X_n(\omega)$ 都按照通常的数列收敛方式趋于 $X(\omega)$。

### 2.3 两种收敛的关系

$$
\boxed{X_n \xrightarrow{a.s.} X \;\Longrightarrow\; X_n \xrightarrow{P} X}
$$

几乎必然收敛 ⇒ 依概率收敛，但**反过来不成立**。上面的"游荡 1"就是反例。

### 2.4 依分布收敛（Convergence in Distribution）

$$
X_n \xrightarrow{D} X \quad\iff\quad
F_{X_n}(x) \to F_X(x) \text{ 对所有 } F_X \text{ 的连续点 } x
$$

这是**最弱**的收敛模式——只要求分布函数趋同，不要求随机变量本身靠近。中心极限定理就是依分布收敛的典型例子。

![图2: 四种收敛模式的关系——从强到弱的蕴含链](https://img.yumeko.site/file/blog/LawOfLargeNumbers/ConvergenceModes.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A clean hierarchical diagram showing the relationships between four modes of convergence in probability theory. Four labeled boxes arranged with arrows showing implication directions:
> Top: "Almost Sure Convergence (a.s.)" — strongest, arrow pointing down to "Convergence in Probability (P)".
> "Convergence in Probability (P)" — middle, with two outgoing arrows: one to "Convergence in Distribution (D)" (directly below), and a dashed arrow to "Lᵖ Convergence" (to the right, with condition label "dominated / uniform integrability").
> "Convergence in Distribution (D)" — at bottom, weakest mode.
> Color code the boxes: a.s. in dark green (strongest), P in blue, Lᵖ in orange, D in gray (weakest).
> Add small text annotations: "WLLN uses this" under P box, "SLLN uses this" under a.s. box, "CLT uses this" under D box.
> Also show: a.s. → P arrow labeled "always", P → D arrow labeled "always", P → a.s. arrow with red X mark labeled "counterexample: wandering 1", D → P arrow with red X mark labeled "counterexample".
> Clean vector diagram, white background, mathematical illustration style.
> ```

---

这四种收敛模式中，大数定律涉及的恰好是**前两种**：

| | 弱大数定律（WLLN） | 强大数定律（SLLN） |
|:--|:--|:--|
| 收敛模式 | 依概率收敛 | 几乎必然收敛 |

## 3. 弱大数定律（WLLN）——四种经典形式

弱大数定律断言 $\bar{X}_n \xrightarrow{P} \mu$（依概率收敛）。但不同形式对随机变量施加的条件不同——从历史上 Bernoulli 的最强条件逐步放松到 Khinchin 的最弱条件。

### 3.1 伯努利大数定律（1713）——最古老的形式

**定理**：设 $S_n \sim \text{Binomial}(n, p)$，即 $n$ 次独立伯努利试验的成功次数。则频率 $\frac{S_n}{n}$ 依概率收敛到 $p$：

$$
\boxed{\frac{S_n}{n} \xrightarrow{P} p}
$$

**历史意义**：Jakob Bernoulli 在遗作《推测术》（*Ars Conjectandi*, 1713）中首次证明了这个定理，这是概率论从赌博计算走向数学科学的分水岭。Bernoulli 花了 20 年才完成证明——当时没有切比雪夫不等式，证明困难得多。

**现代证明**（借助切比雪夫）：$S_n$ 是 $n$ 个 i.i.d. $\text{Bernoulli}(p)$ 之和。$E[S_n/n] = p$，$\operatorname{Var}(S_n/n) = p(1-p)/n$。切比雪夫不等式直接给出：

$$
P\left(\left|\frac{S_n}{n} - p\right| \ge \varepsilon\right) \le \frac{p(1-p)}{n\varepsilon^2} \to 0
$$

这就是最早的大数定律——只针对伯努利试验，但开启了整个领域。

> 关于切比雪夫不等式的详细证明和更多不等式，参见 [[Math/ProbabilityInequalities|概率不等式完全指南]]。

### 3.2 切比雪夫大数定律（WLLN，方差有限形式）

**定理**：设 $\{X_n\}$ 是**两两不相关**的随机变量序列，满足：
- $E[X_i] = \mu_i$（不必相等）
- $\operatorname{Var}(X_i) \le C < \infty$（一致有界方差）

令 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$，$\bar{\mu}_n = \frac{1}{n}\sum_{i=1}^n \mu_i$（均值序列）。则：

$$
\boxed{\bar{X}_n - \bar{\mu}_n \xrightarrow{P} 0}
$$

当 $E[X_i] \equiv \mu$（等期望）时，简化为 $\bar{X}_n \xrightarrow{P} \mu$。

**证明**（仅三行）：

$$
P\left( \left| \bar{X}_n - \bar{\mu}_n \right| \ge \varepsilon \right)
\le \frac{\operatorname{Var}(\bar{X}_n)}{\varepsilon^2}
\le \frac{C}{n\varepsilon^2} \xrightarrow{n \to \infty} 0
$$

仅用了两两不相关（交叉协方差为零保证 $\operatorname{Var}(\bar{X}_n) = \frac{1}{n^2}\sum \operatorname{Var}(X_i)$）和切比雪夫不等式。证明虽简洁，但**假设了方差存在且一致有界**。

### 3.3 辛钦大数定律（Khinchin's WLLN，仅需一阶矩）

**定理**：设 $\{X_n\}$ 是**独立同分布**（i.i.d.）的随机变量序列，且 **$E|X_1| < \infty$**（期望存在）。令 $\mu = E[X_1]$，则：

$$
\boxed{\bar{X}_n \xrightarrow{P} \mu}
$$

**关键改进**：辛钦形式**不要求方差存在**——仅需一阶绝对矩有限。例如，$X_i \sim \text{Cauchy}$ 就不满足条件（$E|X| = \infty$），样本均值确实不收敛。但对于 $X_i \sim t_2$（$t$ 分布，自由度 2），$E|X|$ 存在而方差不存在——切比雪夫形式无法处理，辛钦形式却可以。

**证明思路**（不同于切比雪夫）：使用**特征函数方法**（characteristic function）。由于 i.i.d.，$\bar{X}_n$ 的特征函数为：

$$
\phi_{\bar{X}_n}(t) = \left[\phi_{X_1}\left(\frac{t}{n}\right)\right]^n
$$

由 $E|X_1| < \infty$，特征函数在 $0$ 附近可微：$\phi_{X_1}(t) = 1 + i\mu t + o(t)$。代入：

$$
\phi_{\bar{X}_n}(t) = \left[1 + \frac{i\mu t}{n} + o\left(\frac{1}{n}\right)\right]^n \to e^{i\mu t}
$$

$e^{i\mu t}$ 是退化分布 $\delta_\mu$ 的特征函数。由 Lévy 连续性定理，$\bar{X}_n \xrightarrow{D} \mu$（依分布收敛到常数），而依分布收敛到常数等价于依概率收敛（因为极限是常数时两种收敛等价）。证毕。

**对比切比雪夫**：辛钦用特征函数避开了对方差的需求，但引入了"独立同分布"条件。切比雪夫不要求独立性（仅两两不相关），但要求方差一致有界。两者各有优势。

### 3.4 马尔可夫大数定律（Markov's WLLN，最弱方差条件）

**定理**：设 $\{X_n\}$ 是任意随机变量序列（不要求独立、不要求同分布），满足：

$$
\frac{1}{n^2}\operatorname{Var}\left(\sum_{i=1}^n X_i\right) \to 0 \quad \text{as } n \to \infty
$$

则 $\bar{X}_n - E[\bar{X}_n] \xrightarrow{P} 0$。

**含义**：马尔可夫形式把切比雪夫的"一致有界方差"条件放松到了"平均方差趋于零"——这是方差条件下 WLLN 成立的最弱充分条件。

**证明**：直接对 $\bar{X}_n$ 应用切比雪夫不等式（注意这里不要求两两不相关——$\operatorname{Var}(\sum X_i)$ 本身就包含了协方差项，条件是直接对这个量施加的）。

### 3.5 四种弱大数定律对比

| 形式 | 提出者 | 年份 | 条件 | 特色 |
|:--|:--|:--|:--|:--|
| **伯努利** | J. Bernoulli | 1713 | i.i.d. Bernoulli，二阶矩自然存在 | 最早、最具体、最强条件 |
| **切比雪夫** | P. Chebyshev | 1867 | 两两不相关 + 方差一致有界 | 不需要独立性，不需要同分布 |
| **辛钦** | A. Khinchin | 1929 | i.i.d. + $E\vert X_1\vert < \infty$ | 仅需一阶矩，最弱的矩条件 |
| **马尔可夫** | A. Markov | ~1900 | $\frac{1}{n^2}\operatorname{Var}(\sum X_i) \to 0$ | 最弱的方差条件，不要求独立性和同分布 |

**蕴含关系**（同分布时）：

$$
\text{伯努利} \subset \text{切比雪夫} \subset \text{马尔可夫} \qquad\text{（方差条件层层放松）}
$$

$$
\text{切比雪夫} \;\longleftrightarrow\; \text{辛钦} \qquad\text{（方差 vs 独立性，各有取舍，互不蕴含）}
$$

---

## 4. 强大数定律（SLLN）——两种经典形式

强大数定律断言 $\bar{X}_n \xrightarrow{a.s.} \mu$（几乎必然收敛）。比 WLLN 强，证明更复杂。

### 4.1 柯尔莫哥洛夫强大数定律（Kolmogorov's SLLN，1933）

**定理**：设 $\{X_n\}$ 是**独立同分布**（i.i.d.）的随机变量序列，且 $E|X_1| < \infty$（期望存在）。令 $\mu = E[X_1]$，则：

$$
\boxed{\bar{X}_n \xrightarrow{a.s.} \mu}
$$

即：

$$
P\left( \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} X_i = \mu \right) = 1
$$

**注意**：柯尔莫哥洛夫 SLLN 的条件与辛钦 WLLN **完全一致**（i.i.d. + 一阶矩有限），但结论更强——从依概率收敛升级到了几乎必然收敛。这是柯尔莫哥洛夫 1933 年名著《概率论基础》的巅峰成果之一，开创了现代概率论的公理化时代。

### 4.2 证明概要

SLLN 的完整证明需要**柯尔莫哥洛夫不等式**和 **Borel-Cantelli 引理**。

**关键工具——柯尔莫哥洛夫不等式**：

对于独立随机变量 $Y_1, \dots, Y_n$（均值为 $0$），设 $S_k = \sum_{i=1}^{k} Y_i$：

$$
P\left( \max_{1 \le k \le n} |S_k| \ge \varepsilon \right)
\le \frac{\operatorname{Var}(S_n)}{\varepsilon^2}
$$

这与切比雪夫不等式神似，但它控制的是**整个部分和路径的最大偏离**，而非单个点——这正是从"依概率"跨越到"几乎必然"的关键。

![图4: 切比雪夫 vs 柯尔莫哥洛夫不等式——控制范围对比](https://img.yumeko.site/file/blog/LawOfLargeNumbers/InequalityComparison.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A two-panel visual comparison of Chebyshev's inequality versus Kolmogorov's inequality. Each panel shows the same random walk S₁, S₂, ..., Sₙ as a blue zigzag path over time (x-axis: step k = 1..n, y-axis: cumulative sum value).
> Left panel: "Chebyshev: controls ONE point" — a vertical dashed line at k = n (last step only), with a red shaded horizontal band around 0 of width ±ε. The inequality bounds P(|Sₙ| ≥ ε). Most of the path is unchecked (gray).
> Right panel: "Kolmogorov: controls the ENTIRE path" — a red shaded horizontal band of width ±ε spanning the full time range (k = 1 to n). The inequality bounds P(max₁≤ₖ≤ₙ |Sₖ| ≥ ε). Annotate: if any part of the blue path exits the band, it's detected. Label: "Stronger control → enables a.s. convergence proof".
> The right panel's red band is visually more "protective" — it catches deviations anywhere along the path, not just at the end.
> Clean academic figure style, white background, thin coordinate axes. Mathematical textbook illustration.
> ```

**证明策略**（截断法）：

1. **截断**：将 $X_i$ 分解为 $X_i = X_i \mathbf{1}_{|X_i| \le i} + X_i \mathbf{1}_{|X_i| > i}$（有界部分 + 尾部）
2. 对有界部分应用柯尔莫哥洛夫不等式，通过 Borel-Cantelli 引理证明几乎必然收敛
3. 对尾部证明其影响随 $n \to \infty$ 而消失

涉及的核心不等式链：

$$
\sum_{i=1}^{\infty} \frac{\operatorname{Var}(X_i \mathbf{1}_{|X_i| \le i})}{i^2} < \infty
\;\Longrightarrow\; \text{几乎必然收敛}
$$

这个条件是确保"足够快"收敛的关键。

### 4.3 博雷尔强大数定律（Borel's SLLN，1909）

**定理**（柯尔莫哥洛夫 SLLN 的 Bernoulli 特例）：设 $S_n \sim \text{Binomial}(n, p)$，则：

$$
\boxed{\frac{S_n}{n} \xrightarrow{a.s.} p}
$$

**历史意义**：Émile Borel 在 1909 年首次证明了 SLLN（甚至早于柯尔莫哥洛夫的公理化概率论）。Borel 证明了：不仅频率收敛于概率（WLLN），而且**以概率 1**，极限就是 $p$。

柯尔莫哥洛夫 1933 年的贡献是将 Borel 结果从 Bernoulli 推广到任意 i.i.d. 序列——条件相同（i.i.d. + $E|X| < \infty$），结论相同（几乎必然收敛），但适用范围扩大了无数倍。

### 4.4 SLLN vs WLLN：一个直观例子

考虑独立随机变量序列（注意：**不是同分布**）：

$$
X_n = \begin{cases}
n^2 & \text{以概率 } \frac{1}{n^2} \\
0 & \text{以概率 } 1 - \frac{1}{n^2}
\end{cases}
$$

- $E[X_n] = n^2 \cdot \frac{1}{n^2} = 1$
- $\operatorname{Var}(X_n) = E[X_n^2] - 1^2 = n^4 \cdot \frac{1}{n^2} - 1 = n^2 - 1$

方差无界 → 切比雪夫形式不适用。辛钦形式要求 i.i.d.，此例不是同分布。**WLLN 可能成立也可能不成立**，取决于更精细的分析。

关键启示：**SLLN 对条件更敏感**——它对尾部的衰减速率有实质性要求（需要 $E|X_1| < \infty$ 或更强的矩条件），有时在 WLLN 能"侥幸"成立的边缘地带，SLLN 不成立。

![图3: WLLN vs SLLN——模拟对比，抛硬币 $n=10, 100, 1000, 10000$](https://img.yumeko.site/file/blog/LawOfLargeNumbers/LLNComparison.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A two-row multi-panel figure simulating coin flips to illustrate WLLN vs SLLN.
> Top row (4 panels): "WLLN — Probability View" — each panel shows a histogram/distribution of X̄ₙ values from 10,000 independent simulation runs at a given n. n = 10 (wide bell), n = 100 (narrower), n = 1000 (very narrow), n = 10000 (spike at μ). The distribution tightens around μ = 0.5, illustrating P(|X̄ₙ - μ| > ε) → 0.
> Bottom row (1 wide panel spanning full width): "SLLN — Path View" — a single plot showing 5 overlaid sample paths (colored lines) of the running average X̄ₙ from n=1 to n=1000. All paths start jittery but smooth out and converge to the horizontal dashed line μ = 0.5. Shade the region beyond n ≈ 500 with light green to show "once close, stays close." Label: "P(lim X̄ₙ = μ) = 1 — almost every path converges."
> Professional data visualization style, white background, clean axes. Color palette: blue distributions, warm-colored paths.
> ```

---

## 5. 六种大数定律对照总表

### 5.1 弱大数定律（依概率收敛）

| 形式 | 提出者 | 年份 | 条件 | 证明工具 |
|:--|:--|:--|:--|:--|
| **伯努利 WLLN** | J. Bernoulli | 1713 | i.i.d. Bernoulli($p$) | 初等方法 / 切比雪夫（现代） |
| **切比雪夫 WLLN** | P. Chebyshev | 1867 | 两两不相关 + $\operatorname{Var}$ 一致有界 | 切比雪夫不等式 |
| **辛钦 WLLN** | A. Khinchin | 1929 | i.i.d. + $E\vert X_1\vert < \infty$ | 特征函数 + Lévy 连续性定理 |
| **马尔可夫 WLLN** | A. Markov | ~1900 | $\frac{1}{n^2}\operatorname{Var}(\sum X_i) \to 0$ | 切比雪夫不等式 |

### 5.2 强大数定律（几乎必然收敛）

| 形式 | 提出者 | 年份 | 条件 | 证明工具 |
|:--|:--|:--|:--|:--|
| **博雷尔 SLLN** | É. Borel | 1909 | i.i.d. Bernoulli($p$) | Borel-Cantelli（初等形式） |
| **柯尔莫哥洛夫 SLLN** | A. Kolmogorov | 1933 | i.i.d. + $E\vert X_1\vert < \infty$ | 柯尔莫哥洛夫不等式 + Borel-Cantelli + 截断法 |

### 5.3 关键蕴含关系

**条件由具体到一般**：伯努利大数定律是 i.i.d. Bernoulli 的特例，而柯尔莫哥洛夫 SLLN 和辛钦 WLLN 将其推广到任意 i.i.d. 序列（仅需 $E|X_1| < \infty$）。

**收敛模式由弱到强**：SLLN（几乎必然收敛）蕴含 WLLN（依概率收敛），但反之不成立。条件完全相同的辛钦 WLLN 和柯尔莫哥洛夫 SLLN（均为 i.i.d. + $E|X|<\infty$）体现了这种强弱关系——前者只能保证 $\bar{X}_n \xrightarrow{P} \mu$，后者保证 $\bar{X}_n \xrightarrow{a.s.} \mu$。

**Bernoulli 情形下的具体对应**：伯努利 WLLN 在加强收敛模式后成为博雷尔 SLLN；辛钦 WLLN 在加强收敛模式后成为柯尔莫哥洛夫 SLLN。即——同样的随机对象（i.i.d. Bernoulli 之和），随着证明工具从切比雪夫升级到 Borel-Cantelli，结论从依概率收敛升级到几乎必然收敛。

## 6. 关键区别再辨析

| 维度 | 弱大数定律（WLLN） | 强大数定律（SLLN） |
|:--|:--|:--|
| **收敛模式** | 依概率收敛（$X_n \xrightarrow{P} \mu$） | 几乎必然收敛（$X_n \xrightarrow{a.s.} \mu$） |
| **含义** | 偏差超过 $\varepsilon$ 的概率趋于 $0$ | 对几乎所有样本路径，序列确定性地趋于 $\mu$ |
| **能否预测远期？** | ✗ 不能——概率小不代表不会发生 | ✓ 能——几乎每条路径最终都稳定 |
| **最弱条件（i.i.d.）** | 辛钦：$E\vert X_1\vert < \infty$ | 柯尔莫哥洛夫：$E\vert X_1\vert < \infty$ |
| **核心不等式** | 切比雪夫不等式 | 柯尔莫哥洛夫不等式 |
| **证明难度** | ⭐ 简单 | ⭐⭐⭐ 复杂（需要测度论工具） |

**用一句话记住**：

- **WLLN**："样本均值偏离真值的**可能性**趋于零。"
- **SLLN**："样本均值偏离真值的**实际发生**趋于零——而且一旦 $n$ 足够大，它就再也不会回来了。"

---

## 7. 总结

| 要点 | 说明 |
|:--|:--|
| 大数定律是一个定理家族 | Bernoulli → Chebyshev → Khinchin → Markov（WLLN 线）；Borel → Kolmogorov（SLLN 线） |
| WLLN 的定义 | $\bar{X}_n \xrightarrow{P} \mu$，即 $P(\vert\bar{X}_n - \mu\vert > \varepsilon) \to 0$ |
| SLLN 的定义 | $\bar{X}_n \xrightarrow{a.s.} \mu$，即 $P(\lim \bar{X}_n = \mu) = 1$ |
| 辛钦 WLLN vs 柯尔莫哥洛夫 SLLN | 条件**完全相同**（i.i.d. + $E\vert X\vert < \infty$），结论一弱一强——这是 SLLN 比 WLLN 强的终极证据 |
| 核心不等式 | 切比雪夫 → WLLN；柯尔莫哥洛夫 → SLLN |
| SLLN ⇒ WLLN | 几乎必然收敛蕴含依概率收敛，反之不成立（"游荡 1"反例） |
| 辛钦 vs 切比雪夫 | 辛钦不要方差但要独立同分布；切比雪夫不要独立/同分布但要方差 |

---

> **相关文章**：
> - [[Math/ProbabilityInequalities|概率不等式完全指南]]——马尔可夫、切比雪夫、切尔诺夫、霍夫丁等证明工具
> - [[Math/MultivariateStatistics|多元统计分析]]——当随机变量从标量扩展到向量时，大数定律仍然成立（分量性）
