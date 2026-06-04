---
title: 大数定律详解：从弱大数律到强大数律
date: 2026-06-04
category: Math
tags:
  - 概率论
  - 数理统计
  - 收敛
description: 系统讲解弱大数定律（WLLN）与强大数定律（SLLN）的数学定义、收敛方式差异、证明思路及经典反例，厘清"依概率收敛"与"几乎必然收敛"的本质区别。
image: https://img.yumeko.site/file/blog/LawOfLargeNumbers.png
status: draft
---

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

**大数定律**（Law of Large Numbers, LLN）就是把这种直觉精确化的数学定理。它回答：样本均值在什么意义下、以什么方式收敛到期望值。

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

几乎必然收敛 ⇒ 依概率收敛，但**反过来不成立**。上面的"游荡 1"就是反例：依概率收敛到 $0$，但并不几乎必然收敛。

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

## 3. 弱大数定律（WLLN）

### 3.1 定理陈述（切比雪夫形式）

设 $\{X_n\}$ 是**两两不相关**的随机变量序列，满足：

- $E[X_i] = \mu$（等期望，但非必须——不等期望时只需假设均值收敛）
- $\operatorname{Var}(X_i) \le C < \infty$（一致有界方差）

令 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$，则：

$$
\boxed{\bar{X}_n \xrightarrow{P} \mu}
$$

即：对任意 $\varepsilon > 0$，

$$
\lim_{n \to \infty} P\left( \left| \bar{X}_n - \mu \right| \ge \varepsilon \right) = 0
$$

### 3.2 证明（切比雪夫不等式）

这是最简洁的概率论证明之一，仅靠两个不等式：

**第一步**：样本均值的期望和方差

$$
E[\bar{X}_n] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \mu
$$

由于两两不相关（协方差为零）：

$$
\operatorname{Var}(\bar{X}_n) = \frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(X_i)
\le \frac{nC}{n^2} = \frac{C}{n}
$$

**第二步**：应用切比雪夫不等式

切比雪夫不等式：$P(|Y - E[Y]| \ge \varepsilon) \le \operatorname{Var}(Y) / \varepsilon^2$

对 $Y = \bar{X}_n$：

$$
P\left( \left| \bar{X}_n - \mu \right| \ge \varepsilon \right)
\le \frac{\operatorname{Var}(\bar{X}_n)}{\varepsilon^2}
\le \frac{C}{n\varepsilon^2} \xrightarrow{n \to \infty} 0
$$

证毕。只用了一个不等式和三行推导。

![图3: 弱大数定律证明流程——切比雪夫不等式三步推导](https://img.yumeko.site/file/blog/LawOfLargeNumbers/WLLNProof.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A three-step proof flowchart for the Weak Law of Large Numbers (Chebyshev form). Three rectangular boxes connected by downward arrows:
> Box 1: "Step 1: Compute E[X̄ₙ] = μ, Var(X̄ₙ) ≤ C/n" with a small inset showing the variance calculation: Var(X̄ₙ) = (1/n²)ΣVar(Xᵢ) ≤ C/n.
> Box 2: "Step 2: Apply Chebyshev's Inequality" with the formula: P(|X̄ₙ - μ| ≥ ε) ≤ Var(X̄ₙ)/ε².
> Box 3: "Step 3: Take limit" with: P(|X̄ₙ - μ| ≥ ε) ≤ C/(nε²) → 0 as n → ∞.
> Below Box 3: a checkmark "∎ QED" in green.
> Right side: a small inset graph showing a decay curve P(|X̄ₙ - μ| ≥ ε) vs n, exponentially approaching 0.
> Clean academic flowchart style with soft blue and white color scheme. Math formulas in LaTeX style. White background.
> ```

### 3.3 弱大数定律的局限性

上面的证明假设方差存在且一致有界。但弱大数律在更弱的条件下也成立——比如**辛钦大数定律**只要求独立同分布且期望存在，不要求方差存在。

然而，无论条件多弱，WLLN 始终只能保证**依概率收敛**——这意味着：

- 对任意 $\varepsilon$，$P(|\bar{X}_n - \mu| > \varepsilon) \to 0$
- 但**不能保证**对于某个具体的随机试验序列，样本均值一定收敛

## 4. 强大数定律（SLLN）

### 4.1 定理陈述（柯尔莫哥洛夫形式）

设 $\{X_n\}$ 是**独立同分布**（i.i.d.）的随机变量序列，且 $E|X_1| < \infty$（期望存在）。令 $\mu = E[X_1]$，则：

$$
\boxed{\bar{X}_n \xrightarrow{a.s.} \mu}
$$

即：

$$
P\left( \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} X_i = \mu \right) = 1
$$

### 4.2 证明概要

SLLN 的完整证明需要柯尔莫哥洛夫不等式和 Borel-Cantelli 引理，篇幅较长。这里给出策略级概述。

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

1. **截断**：将 $X_i$ 分解为 $X_i = X_i \mathbf{1}_{|X_i| \le n} + X_i \mathbf{1}_{|X_i| > n}$（有界部分 + 尾部）
2. 对有界部分应用柯尔莫哥洛夫不等式，通过 Borel-Cantelli 引理证明几乎必然收敛
3. 对尾部证明其影响随 $n \to \infty$ 而消失

涉及的核心不等式链：

$$
\sum_{i=1}^{\infty} \frac{\operatorname{Var}(X_i \mathbf{1}_{|X_i| \le i})}{i^2} < \infty
\;\Longrightarrow\; \text{几乎必然收敛}
$$

这个条件是确保"足够快"收敛的关键。

### 4.3 SLLN vs WLLN：一个直观例子

下面构造一个符合 WLLN 但**不满足** SLLN 的例子，以体会两者的本质差异。

考虑独立随机变量序列：

$$
X_n = \begin{cases}
n^2 & \text{以概率 } \frac{1}{n^2} \\
0 & \text{以概率 } 1 - \frac{1}{n^2}
\end{cases}
$$

- $E[X_n] = n^2 \cdot \frac{1}{n^2} = 1$
- $\operatorname{Var}(X_n) = E[X_n^2] - 1^2 = n^4 \cdot \frac{1}{n^2} - 1 = n^2 - 1$

方差无界 → 切比雪夫形式的 WLLN 不直接适用。但辛钦形式要求 i.i.d.，此例不是同分布，因此 WLLN 可能成立也可能不成立（取决于具体构造）。

关键启示：**SLLN 对条件更敏感**——它对尾部的衰减速率有实质性要求（需要 $E|X_1| < \infty$ 或更强的矩条件），而 WLLN 有时可以在更弱的条件下侥幸成立。

## 5. 关键区别再辨析

| 维度 | 弱大数定律（WLLN） | 强大数定律（SLLN） |
|:--|:--|:--|
| **收敛模式** | 依概率收敛（$X_n \xrightarrow{P} \mu$） | 几乎必然收敛（$X_n \xrightarrow{a.s.} \mu$） |
| **含义** | 偏差超过 $\varepsilon$ 的概率趋于 $0$ | 对几乎所有样本路径，序列确定性地趋于 $\mu$ |
| **能否预测远期？** | ✗ 不能——概率小不代表不会发生 | ✓ 能——几乎每条路径最终都稳定 |
| **典型条件** | 方差有界 + 不相关（切比雪夫）<br> 或 i.i.d. + 期望存在（辛钦） | i.i.d. + $E|X_1| < \infty$（柯尔莫哥洛夫） |
| **核心不等式** | 切比雪夫不等式 | 柯尔莫哥洛夫不等式 + Borel-Cantelli 引理 |
| **证明难度** | ⭐ 简单（三行） | ⭐⭐⭐ 复杂（需要测度论工具） |

### 用一句话记住

- **WLLN**："样本均值偏离真值的**可能性**趋于零。"
- **SLLN**："样本均值偏离真值的**实际发生**趋于零——而且一旦 $n$ 足够大，它就再也不会回来了。"

![图5: WLLN vs SLLN——模拟对比，抛硬币 $n=10, 100, 1000, 10000$](https://img.yumeko.site/file/blog/LawOfLargeNumbers/LLNComparison.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A two-row multi-panel figure simulating coin flips to illustrate WLLN vs SLLN.
> Top row (4 panels): "WLLN — Probability View" — each panel shows a histogram/distribution of X̄ₙ values from 10,000 independent simulation runs at a given n. n = 10 (wide bell), n = 100 (narrower), n = 1000 (very narrow), n = 10000 (spike at μ). The distribution tightens around μ = 0.5, illustrating P(|X̄ₙ - μ| > ε) → 0.
> Bottom row (1 wide panel spanning full width): "SLLN — Path View" — a single plot showing 5 overlaid sample paths (colored lines) of the running average X̄ₙ from n=1 to n=1000. All paths start jittery but smooth out and converge to the horizontal dashed line μ = 0.5. Shade the region beyond n ≈ 500 with light green to show "once close, stays close." Label: "P(lim X̄ₙ = μ) = 1 — almost every path converges."
> Professional data visualization style, white background, clean axes. Color palette: blue distributions, warm-colored paths.
> ```

## 6. 总结

| 要点 | 说明 |
|:--|:--|
| WLLN 的定义 | $\bar{X}_n \xrightarrow{P} \mu$，即 $P(\vert\bar{X}_n - \mu\vert > \varepsilon) \to 0$ |
| SLLN 的定义 | $\bar{X}_n \xrightarrow{a.s.} \mu$，即 $P(\lim \bar{X}_n = \mu) = 1$ |
| 核心区别 | 收敛模式不同——依概率 vs 几乎必然 |
| WLLN 的证明 | 切比雪夫不等式，三行即得 |
| SLLN 的证明 | 需要柯尔莫哥洛夫不等式 + Borel-Cantelli 引理 |
| SLLN ⇒ WLLN | 几乎必然收敛蕴含依概率收敛，反之不成立 |
| "游荡 1"反例 | 依概率收敛到 0 但并非几乎必然收敛 |
