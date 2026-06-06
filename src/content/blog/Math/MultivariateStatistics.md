---
title: 多元统计分析：分布、计算与推断
date: 2026-06-05
category: 数学
tags:
  - 多元统计
  - 概率论
  - 假设检验
  - 分布
description: 系统讲解多元统计分析的完整框架：四大核心分布（多元正态、Wishart、Hotelling T²、Wilks Λ）、协方差矩阵的计算与几何解释、以及基于这些分布的统计推断方法（均值检验、MANOVA、置信椭球）。从一元到多元的自然推广，配以完整数值示例。
image: https://img.yumeko.site/file/blog/cover/1780667956153.webp
status: published
---

> **前置阅读**：本文假定读者熟悉协方差矩阵的基本概念。若需回顾马尔可夫不等式、切比雪夫不等式等概率工具，请参阅 [[Math/ProbabilityInequalities|概率不等式完全指南]] 和 [[Math/LawOfLargeNumbers|大数定律详解]]。

## 第一部分：分布

### 1. 为什么需要多元分布？

一元统计有一套成熟的分布体系，每一项都有精确的多元对应：

| 一元问题 | 一元分布 | **多元分布** | 多元问题 |
|:--|:--|:--|:--|
| 方差估计 | $\chi^2$ | **Wishart** $\mathcal{W}_d$ | 协方差矩阵估计 |
| 均值推断（$\sigma$ 未知） | $t$ | **Hotelling** $T^2$ | 均值向量推断 |
| 方差比较 / ANOVA | $F$ | **Wilks** $\Lambda$ | 协方差矩阵比较 / MANOVA |

这一切建立在**多元正态分布**之上，正如一元正态是一元统计的基石。

---

### 2. 多元正态分布

#### 2.1 定义

称 $d$ 维随机向量 $\mathbf{X} = (X_1, \dots, X_d)^\top$ 服从**多元正态分布**，记作 $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$，若其联合密度为：

$$
\boxed{f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)}
$$

- $\boldsymbol{\mu} = E[\mathbf{X}] \in \mathbb{R}^d$：均值向量
- $\Sigma = \operatorname{Cov}(\mathbf{X}) \in \mathbb{R}^{d \times d}$：协方差矩阵（对称正定）

等价地，可用**特征函数**定义（不要求 $\Sigma$ 可逆）：

$$
\phi_{\mathbf{X}}(\mathbf{t}) = E[e^{i\mathbf{t}^\top \mathbf{X}}] = \exp\left( i \mathbf{t}^\top \boldsymbol{\mu} - \frac{1}{2} \mathbf{t}^\top \Sigma \mathbf{t} \right)
$$

#### 2.2 核心性质

**（1）线性变换保持正态性**

设 $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$，$\mathbf{A} \in \mathbb{R}^{m \times d}$，$\mathbf{b} \in \mathbb{R}^m$，则：

$$
\boxed{\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \ \mathbf{A}\Sigma\mathbf{A}^\top)}
$$

**（2）白化变换**

令 $\mathbf{Z} = \Sigma^{-1/2}(\mathbf{X} - \boldsymbol{\mu})$，则 $\mathbf{Z} \sim \mathcal{N}_d(\mathbf{0}, \mathbf{I}_d)$——各分量独立标准正态。这是将相关变量"去相关"的核心技术。

**（3）边缘分布仍是正态**

任意子向量 $(X_{i_1}, \dots, X_{i_k})^\top$ 服从 $k$ 维正态，均值和协方差从 $\boldsymbol{\mu}$ 和 $\Sigma$ 中取对应子块。

**（4）独立性 $\iff$ 不相关性**

对多元正态，这是**充要条件**（一般分布只有"独立 ⇒ 不相关"）：

$$
X_i \perp\!\!\!\perp X_j \quad\iff\quad \Sigma_{ij} = 0
$$

**（5）条件分布也是正态**

将 $\mathbf{X}$ 分块为 $(\mathbf{X}_1^\top, \mathbf{X}_2^\top)^\top$，对应 $\boldsymbol{\mu}$ 和 $\Sigma$ 分块：

$$
\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}
$$

则 $\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2$ 的条件分布为：

$$
\boxed{\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}\left(\boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2),\ \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)}
$$

条件均值是 $\mathbf{x}_2$ 的**线性回归**，条件协方差（Schur 补）不依赖 $\mathbf{x}_2$ 的具体值。

**（6）二次型服从 $\chi^2$**

$$
\boxed{(\mathbf{X} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2_d}
$$

由白化变换直接推出：$\mathbf{Z}^\top\mathbf{Z} = \sum_{i=1}^d Z_i^2$，$Z_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$。

#### 2.3 一元到多元的直觉映射

| 一元 $\mathcal{N}(\mu, \sigma^2)$ | 多元 $\mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$ |
|:--|:--|
| 密度指数：$-\frac{(x-\mu)^2}{2\sigma^2}$ | $-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |
| 标准化：$Z = \frac{X-\mu}{\sigma}$ | 白化：$\mathbf{Z} = \Sigma^{-1/2}(\mathbf{X}-\boldsymbol{\mu})$ |
| $Z^2 \sim \chi^2_1$ | $\mathbf{Z}^\top\mathbf{Z} \sim \chi^2_d$ |
| $\bar{x} \perp\!\!\!\perp s^2$ | $\bar{\mathbf{x}} \perp\!\!\!\perp \hat{\Sigma}$ |

---

### 3. Wishart 分布：多元 $\chi^2$

#### 3.1 从一元到多元

一元：从正态总体抽样，样本方差满足 $\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$。等价于 $\sum (x_i - \bar{x})^2 \sim \sigma^2 \cdot \chi^2_{n-1}$。

多元推广：将标量平方和替换为**外积之和**：

$$
\mathbf{S} = \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top \in \mathbb{R}^{d \times d}
$$

#### 3.2 定义

设 $\mathbf{Z}_1, \dots, \mathbf{Z}_m \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}_d(\mathbf{0}, \Sigma)$（$m \ge d$）。定义：

$$
\mathbf{W} = \sum_{i=1}^m \mathbf{Z}_i \mathbf{Z}_i^\top
$$

则 $\mathbf{W}$ 服从自由度为 $m$、尺度矩阵为 $\Sigma$ 的 **Wishart 分布**：

$$
\boxed{\mathbf{W} \sim \mathcal{W}_d(m, \Sigma)}
$$

#### 3.3 与样本协方差矩阵的关系

给定 $\mathbf{x}_1, \dots, \mathbf{x}_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$：

$$
\boxed{\mathbf{S} = \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top \sim \mathcal{W}_d(n-1,\ \Sigma)}
$$

自由度为 $n-1$（损失了 1 个自由度给 $\bar{\mathbf{x}}$），而非 $n$。

#### 3.4 核心性质

- 期望：$E[\mathbf{W}] = m\Sigma$
- 线性变换：$\mathbf{A}\mathbf{W}\mathbf{A}^\top \sim \mathcal{W}_p(m, \mathbf{A}\Sigma\mathbf{A}^\top)$
- 对角元边缘：$\mathbf{W}_{ii} \sim \sigma_{ii} \cdot \chi^2_m$
- 可加性：$\mathbf{W}_1 \sim \mathcal{W}_d(m_1, \Sigma) \perp\!\!\!\perp \mathbf{W}_2 \sim \mathcal{W}_d(m_2, \Sigma) \implies \mathbf{W}_1 + \mathbf{W}_2 \sim \mathcal{W}_d(m_1+m_2, \Sigma)$

#### 3.5 一元退化验证

$d=1$ 时：$\mathbf{W}$ 退化为标量 $W = \sum Z_i^2$，$Z_i \sim \mathcal{N}(0, \sigma^2)$：

$$
W \sim \mathcal{W}_1(m, [\sigma^2]) \quad\iff\quad \frac{W}{\sigma^2} \sim \chi^2_m
$$

Wishart 就是"多元的 $\chi^2$"。

---

### 4. Hotelling $T^2$ 分布：多元 $t$

#### 4.1 定义

设 $\mathbf{Z} \sim \mathcal{N}_d(\mathbf{0}, \Sigma)$ 与 $\mathbf{W} \sim \mathcal{W}_d(m, \Sigma)$ 相互独立（$m \ge d$）。定义：

$$
\boxed{T^2 = m \cdot \mathbf{Z}^\top \mathbf{W}^{-1} \mathbf{Z}}
$$

称 $T^2$ 服从自由度为 $(d, m)$ 的 **Hotelling $T^2$ 分布**：

$$
T^2 \sim T^2(d, m)
$$

#### 4.2 与 $F$ 分布的精确关系

$$
\boxed{\frac{m - d + 1}{d \cdot m} \cdot T^2(d, m) \sim F_{d,\ m-d+1}}
$$

**一元退化的具体验证**（$d=1$）：$Z \sim \mathcal{N}(0, \sigma^2)$，$W \sim \mathcal{W}_1(m, [\sigma^2]) \iff W/\sigma^2 \sim \chi^2_m$：

$$
T^2 = \frac{m Z^2}{W} = \frac{m \cdot \chi^2_1}{\chi^2_m} \quad\iff\quad \frac{T^2}{m} = \frac{\chi^2_1/1}{\chi^2_m/m} \sim F_{1,m}
$$

与公式 $\frac{m-1+1}{1 \cdot m}T^2 = \frac{T^2}{m} \sim F_{1,m}$ 完全一致。

---

### 5. Wilks $\Lambda$ 分布：多元 $F$

#### 5.1 定义

设 $\mathbf{B} \sim \mathcal{W}_d(p, \Sigma)$（组间离差）与 $\mathbf{W} \sim \mathcal{W}_d(q, \Sigma)$（组内离差）相互独立。定义：

$$
\boxed{\Lambda = \frac{|\mathbf{W}|}{|\mathbf{B} + \mathbf{W}|} = \frac{1}{|\mathbf{I} + \mathbf{W}^{-1}\mathbf{B}|}}
$$

称 $\Lambda$ 服从自由度为 $(d, p, q)$ 的 **Wilks $\Lambda$ 分布**：

$$
\Lambda \sim \Lambda(d, p, q)
$$

#### 5.2 直觉

$\mathbf{W}$ = 组内离差矩阵（"噪声"），$\mathbf{B} + \mathbf{W} = \mathbf{T}$ = 总离差矩阵（"信号 + 噪声"）。

- $H_0$ 成立（各组均值相等）→ $\mathbf{B}$ 很小 → $|\mathbf{B}+\mathbf{W}| \approx |\mathbf{W}|$ → **$\Lambda \approx 1$**
- $H_0$ 不成立 → $\mathbf{B}$ 很大 → $|\mathbf{B}+\mathbf{W}| \gg |\mathbf{W}|$ → **$\Lambda \ll 1$**

$\Lambda$ 越小，越拒绝 $H_0$。这是一元 ANOVA 中 "$F$ 值越大越显著"的多元对应（$\Lambda$ 本质上是 $1/(1+F)$ 的推广）。

#### 5.3 精确 $F$ 转化（特定参数下）

| 参数条件 | 转化关系 |
|:--|:--|
| $d=1$ | $\frac{1-\Lambda}{\Lambda} \cdot \frac{q}{p} \sim F_{p, q}$（退化为 ANOVA $F$） |
| $p=1$ | $\frac{1-\Lambda}{\Lambda} \cdot \frac{q-d+1}{d} \sim F_{d, q-d+1}$（退化为 Hotelling $T^2$） |
| $d=2$ | $\frac{1-\sqrt{\Lambda}}{\sqrt{\Lambda}} \cdot \frac{q-1}{p} \sim F_{2p, 2(q-1)}$ |
| $p=2$ | $\frac{1-\sqrt{\Lambda}}{\sqrt{\Lambda}} \cdot \frac{q-d+1}{d} \sim F_{2d, 2(q-d+1)}$ |

对于一般参数，使用 **Bartlett $\chi^2$ 近似**（大样本）：

$$
-\left(q - \frac{d + p + 1}{2}\right) \ln \Lambda \xrightarrow{\text{approx}} \chi^2_{dp}
$$

---

## 第二部分：计算

### 6. 协方差矩阵的计算

#### 6.1 矩阵形式的样本协方差

给定 $n$ 个 $d$ 维样本，排成数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$（每行一个样本）。样本均值 $\bar{\mathbf{x}} = \frac{1}{n}\mathbf{X}^\top \mathbf{1}_n$。

**中心化**：$\tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}_n \bar{\mathbf{x}}^\top$（每行减去均值向量）

**样本协方差矩阵**（$d \times d$）：

$$
\boxed{\hat{\Sigma} = \frac{1}{n-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}}
$$

**逐元素展开**：$\hat{\Sigma}_{ij} = \frac{1}{n-1} \sum_{k=1}^n (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$

> 除以 $n$ = 总体协方差（MLE，有偏）；除以 $n-1$ = 样本协方差（无偏）。大样本下差异可忽略。

#### 6.2 完整数值示例

3 个二维样本：$\{(0,0),\ (1,1),\ (2,2)\}$

**步骤 1**：$\bar{x}_1 = \bar{x}_2 = 1$，$\bar{\mathbf{x}} = (1,1)^\top$

**步骤 2**：中心化
$$
\tilde{\mathbf{X}} = \begin{bmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix}
$$

**步骤 3**：矩阵乘法
$$
\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} \begin{bmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}
$$

**步骤 4**：除以自由度

| 方法 | 公式 | 结果 | $\operatorname{tr}$ |
|:--|:--|:--|:--|
| 总体协方差（MLE） | $\frac{1}{3}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ | $\begin{bmatrix} 2/3 & 2/3 \\ 2/3 & 2/3 \end{bmatrix}$ | $4/3$ |
| 样本协方差（无偏） | $\frac{1}{2}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ | $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ | $2$ |

#### 6.3 相关系数矩阵

$$
\rho_{ij} = \frac{\hat{\Sigma}_{ij}}{\sqrt{\hat{\Sigma}_{ii} \cdot \hat{\Sigma}_{jj}}}, \qquad
\mathbf{P} = \mathbf{D}^{-1/2} \, \hat{\Sigma} \, \mathbf{D}^{-1/2}
$$

其中 $\mathbf{D} = \operatorname{diag}(\hat{\Sigma}_{11}, \dots, \hat{\Sigma}_{dd})$。上例中 $\rho_{12} = 1$（完全正相关——因为 $X_1 = X_2$）。

#### 6.4 几何解释：协方差椭球与 PCA

$\hat{\Sigma}$ 在几何上定义了一个**等 Mahalanobis 距离椭球**：

$$
\{\mathbf{x} : (\mathbf{x} - \bar{\mathbf{x}})^\top \hat{\Sigma}^{-1} (\mathbf{x} - \bar{\mathbf{x}}) = c^2\}
$$

对 $\hat{\Sigma}$ 做特征分解 $\hat{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$：
- **主轴方向** = 特征向量 $\mathbf{v}_1, \dots, \mathbf{v}_d$
- **半轴长度** = $\sqrt{\lambda_i} \cdot c$
- 最大特征值方向 = 数据方差最大的方向 → **第一主成分**

**Mahalanobis 距离**以协方差矩阵为度量，考虑了各维度的方差和相关性：

$$
D_M(\mathbf{x}, \bar{\mathbf{x}}) = \sqrt{(\mathbf{x} - \bar{\mathbf{x}})^\top \hat{\Sigma}^{-1} (\mathbf{x} - \bar{\mathbf{x}})}
$$

沿方差大的方向"放宽容忍度"，沿方差小的方向"严格要求"——异常检测和判别分析的核心工具。

![CovarianceEllipsoid.png](https://img.yumeko.site/file/blog/articles/1780733243957.webp)

#### 6.5 NumPy 实现

```python
import numpy as np


def sampleCovariance(X: np.ndarray, bias: bool = False) -> np.ndarray:
    """
    计算样本协方差矩阵

    Args:
        X: 形状 (n, d)，每行一个样本
        bias: True→除以 n（MLE），False→除以 n-1（无偏）
    Returns:
        形状 (d, d) 的协方差矩阵
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)          # 中心化
    divisor = n if bias else n - 1
    return (Xc.T @ Xc) / divisor


def covToCorr(S: np.ndarray) -> np.ndarray:
    """协方差矩阵 → 相关系数矩阵"""
    std = np.sqrt(np.diag(S))
    return S / np.outer(std, std)


# === 示例 ===
X = np.array([[0, 0], [1, 1], [2, 2]])

Spop = sampleCovariance(X, bias=True)   # MLE
Ssam = sampleCovariance(X, bias=False)  # 无偏
R = covToCorr(Spop)

# PCA via eigendecomposition
evals, evecs = np.linalg.eigh(Spop)
print(f"特征值: {evals}")                          # [0, 1.333]
print(f"第一主成分方向: {evecs[:, -1]}")           # ≈ [0.707, 0.707]
print(f"解释方差比: {evals[-1] / evals.sum():.2%}") # 100%
```

#### 6.6 多元正态的模拟：Cholesky 分解

要生成 $\mathcal{N}_d(\boldsymbol{\mu}, \Sigma)$ 的样本：

1. 生成 $d$ 个独立标准正态 $\mathbf{Z} \sim \mathcal{N}_d(\mathbf{0}, \mathbf{I}_d)$
2. Cholesky 分解 $\Sigma = \mathbf{L}\mathbf{L}^\top$（$\mathbf{L}$ 下三角）
3. 变换：$\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$

验证：$\operatorname{Cov}(\mathbf{X}) = \mathbf{L} \cdot \mathbf{I}_d \cdot \mathbf{L}^\top = \Sigma$。

```python
def generateMVN(mu, Sigma, n):
    """生成 n 个 N(mu, Sigma) 样本"""
    d = len(mu)
    L = np.linalg.cholesky(Sigma)           # Cholesky: Sigma = L @ L.T
    Z = np.random.randn(n, d)               # N(0, I)
    return mu + Z @ L.T                     # mu + LZ
```

---

## 第三部分：数理统计推断

### 7. Hotelling $T^2$ 检验：多元均值的推断

#### 7.1 单样本检验

**问题**：检验 $H_0: \boldsymbol{\mu} = \boldsymbol{\mu}_0$ vs $H_1: \boldsymbol{\mu} \neq \boldsymbol{\mu}_0$。

**检验统计量**：

$$
\boxed{T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top \hat{\Sigma}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)}
$$

在 $H_0$ 下：$\bar{\mathbf{x}} \sim \mathcal{N}_d(\boldsymbol{\mu}_0, \Sigma/n)$，$(n-1)\hat{\Sigma} \sim \mathcal{W}_d(n-1, \Sigma)$，两者独立。因此：

$$
T^2 \sim T^2(d, n-1) \quad\iff\quad \boxed{\frac{n-d}{d(n-1)} T^2 \sim F_{d,\ n-d}}
$$

**拒绝域**（显著性水平 $\alpha$）：$T^2 > \frac{d(n-1)}{n-d} F_{d, n-d}(\alpha)$。

**$100(1-\alpha)\%$ 置信椭球**：满足以下条件的所有 $\boldsymbol{\mu}$ 构成均值向量的置信区域：

$$
n(\bar{\mathbf{x}} - \boldsymbol{\mu})^\top \hat{\Sigma}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}) \le \frac{d(n-1)}{n-d} F_{d, n-d}(\alpha)
$$

**为什么不用 $d$ 次一元 $t$ 检验？** 多重比较导致总 I 类错误率膨胀。Hotelling $T^2$ 通过 $\hat{\Sigma}^{-1}$ 考虑了各维度的相关性，一次性检验整个均值向量，在多元正态下是**一致最优检验**（似然比检验）。

#### 7.2 双样本检验（等协方差假设）

**问题**：$H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2$。

**假设**：$\Sigma_1 = \Sigma_2 = \Sigma$（可用 Box's M 检验验证）。

**合并协方差估计**：

$$
\mathbf{S}_{\text{pooled}} = \frac{(n_1-1)\hat{\Sigma}_1 + (n_2-1)\hat{\Sigma}_2}{n_1 + n_2 - 2}
$$

**检验统计量**：

$$
T^2 = \frac{n_1 n_2}{n_1 + n_2} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)^\top \mathbf{S}_{\text{pooled}}^{-1} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)
$$

在 $H_0$ 下：

$$
\boxed{\frac{n_1 + n_2 - d - 1}{d(n_1 + n_2 - 2)} T^2 \sim F_{d,\ n_1 + n_2 - d - 1}}
$$

### 8. MANOVA：多元方差分析

#### 8.1 模型设定

$k$ 组，第 $j$ 组有 $n_j$ 个 $d$ 维观测 $\mathbf{x}_{ji} \in \mathbb{R}^d$。总样本量 $n = \sum_{j=1}^k n_j$。

**假设**：$H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2 = \dots = \boldsymbol{\mu}_k$（$k$ 组均值向量全部相等）

#### 8.2 离差矩阵分解

与一元 ANOVA 的平方和分解完全平行——每一项从标量变成矩阵：

![MANOVADecomposition.png](https://img.yumeko.site/file/blog/articles/1780733241636.webp)

$$
\boxed{\mathbf{T} = \mathbf{B} + \mathbf{W}}
$$

| 矩阵 | 定义 | 自由度 | 含义 |
|:--|:--|:--|:--|
| $\mathbf{T}$（总） | $\sum_{j}\sum_{i} (\mathbf{x}_{ji} - \bar{\mathbf{x}})(\mathbf{x}_{ji} - \bar{\mathbf{x}})^\top$ | $n-1$ | 总变异 |
| $\mathbf{B}$（组间） | $\sum_{j} n_j (\bar{\mathbf{x}}_j - \bar{\mathbf{x}})(\bar{\mathbf{x}}_j - \bar{\mathbf{x}})^\top$ | $k-1$ | 组间变异 |
| $\mathbf{W}$（组内） | $\sum_{j}\sum_{i} (\mathbf{x}_{ji} - \bar{\mathbf{x}}_j)(\mathbf{x}_{ji} - \bar{\mathbf{x}}_j)^\top$ | $n-k$ | 组内变异（误差） |

#### 8.3 检验统计量：Wilks $\Lambda$

$$
\Lambda = \frac{|\mathbf{W}|}{|\mathbf{T}|} = \frac{|\mathbf{W}|}{|\mathbf{B} + \mathbf{W}|} \sim \Lambda(d, k-1, n-k)
$$

**决策**：
- 精确：若 $d$ 或 $p = k-1$ 较小，用精确 $F$ 转化（见 5.3 节）
- 大样本：Bartlett $\chi^2$ 近似

```python
from scipy.stats import chi2
import numpy as np

def wilksLambdaTest(B, W, d, p, q):
    """
    MANOVA Wilks Lambda 检验

    Args:
        B: 组间离差矩阵 (d×d)
        W: 组内离差矩阵 (d×d)
        d: 变量维度
        p: 组间自由度 = k-1
        q: 组内自由度 = n-k
    Returns:
        (Lambda, approx_F, p_value)
    """
    Lambda = np.linalg.det(W) / np.linalg.det(B + W)

    # Bartlett χ² 近似
    bartlett = -(q - (d + p + 1)/2) * np.log(Lambda)
    df = d * p
    pValue = 1 - chi2.cdf(bartlett, df)

    return Lambda, bartlett, pValue
```

> **注意**：除 Wilks $\Lambda$ 外尚有三种等价的 MANOVA 统计量——Pillai's trace、Hotelling-Lawley trace、Roy's greatest root——在大多数场景下结论一致。

### 9. 多元正态的假设检验

#### 9.1 均值向量检验

| 场景 | 检验方法 | 统计量 |
|:--|:--|:--|
| 单样本（$\Sigma$ 已知） | $\chi^2$ 检验 | $n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top\Sigma^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0) \sim \chi^2_d$ |
| 单样本（$\Sigma$ 未知） | Hotelling $T^2$ | 见 7.1 |
| 双样本（$\Sigma_1 = \Sigma_2$） | Hotelling $T^2$ | 见 7.2 |
| 多样本（$k \ge 2$） | MANOVA / Wilks $\Lambda$ | 见 8 |

#### 9.2 协方差矩阵检验

| 检验 | $H_0$ | 统计量 |
|:--|:--|:--|
| 球性检验（Bartlett） | $\Sigma = \sigma^2 \mathbf{I}$ | 似然比 $\to \chi^2$ |
| 等协方差（Box's M） | $\Sigma_1 = \Sigma_2 = \dots = \Sigma_k$ | Box's M $\to \chi^2$ 或 $F$ |
| 独立性检验 | $\Sigma$ 为对角阵 | 似然比 $\to \chi^2$ |

#### 9.3 似然比检验（LRT）的一般框架

在多元正态下，大多数假设检验都是似然比检验的特例。LRT 统计量为：

$$
\lambda = \frac{\sup_{H_0} L(\boldsymbol{\mu}, \Sigma)}{\sup_{H_1} L(\boldsymbol{\mu}, \Sigma)}
$$

在 $H_0$ 下，$-2\ln\lambda \xrightarrow{D} \chi^2_{\nu}$，其中 $\nu$ = 自由参数数量之差（Wilks 定理）。

Wilks $\Lambda$ 就是 MANOVA 的 LRT 统计量（经过合适的幂变换）——$\Lambda = \lambda^{2/n}$。

---

## 第四部分：总结

![DistributionRelations.png](https://img.yumeko.site/file/blog/articles/1780733240792.webp)

所有多元分布在 $d=1$ 时精确退化为一元对应分布，这是验证推导正确性的试金石：

- **Wishart $\mathcal{W}_d(m, \Sigma)$** 当 $d=1$ 时退化为 $\sigma^2 \cdot \chi^2_m$。这是因为 $\mathbf{W}$ 退化为标量 $W = \sum Z_i^2$，$Z_i \sim \mathcal{N}(0, \sigma^2)$，而 $W/\sigma^2 \sim \chi^2_m$。
- **Hotelling $T^2(d, m)$** 当 $d=1$ 时退化为 $t^2_m = F(1, m)$。由 $T^2 = m Z^2 / W$，$Z^2/\sigma^2 \sim \chi^2_1$，$W/\sigma^2 \sim \chi^2_m$，两者独立，故 $T^2/m \sim F_{1,m}$。
- **Wilks $\Lambda(d, p, q)$** 当 $d=1$ 时退化为 $1/(1 + \frac{p}{q}F_{p,q})$，等价于一元 ANOVA 的 $F$ 检验。当 $p=1$ 时进一步退化为 Hotelling $T^2$（两组比较）。

三者的退化链：Wishart（协方差矩阵的分布）加上正态向量构造出 Hotelling $T^2$（均值向量的检验），Hotelling $T^2$ 的多组推广即为 Wilks $\Lambda$（MANOVA）。$d=1$ 时这条链收敛为一元统计的标准工具链：$\chi^2 \to t^2 \to F$。

### 10.2 多元统计推断工具速查

| 你想做什么 | 用什么方法 | 核心统计量 |
|:--|:--|:--|
| 估计协方差矩阵 | 样本协方差 | $\hat{\Sigma} = \frac{1}{n-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}$ |
| 检验均值向量 = 指定值 | Hotelling $T^2$ 单样本 | $T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top \hat{\Sigma}^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)$ |
| 检验两组均值向量相等 | Hotelling $T^2$ 双样本 | 合并协方差 + $T^2$ |
| 检验 $k$ 组均值向量相等 | MANOVA | Wilks $\Lambda = \vert\mathbf{W}\vert / \vert\mathbf{T}\vert$ |
| 降维 / 可视化 | PCA | $\hat{\Sigma}$ 的特征分解 |
| 异常检测 | Mahalanobis 距离 | $D_M = \sqrt{(\mathbf{x} - \bar{\mathbf{x}})^\top \hat{\Sigma}^{-1}(\mathbf{x} - \bar{\mathbf{x}})}$ |
| 多元正态模拟 | Cholesky 分解 | $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$，$\Sigma = \mathbf{L}\mathbf{L}^\top$ |

### 10.3 关键直觉

1. **多元正态是基石**——正如一元正态生出了 $\chi^2$、$t$、$F$，多元正态生出了 Wishart、$T^2$、$\Lambda$
2. **Wishart = 多元 $\chi^2$**——从标量方差到矩阵协方差，自由度概念不变
3. **Hotelling $T^2$ = 多元 $t^2$**——Mahalanobis 距离的平方，可精确转化为 $F$ 得到 $p$ 值
4. **Wilks $\Lambda$ = 多元 ANOVA $F$**——组内离差占总离差的比例
5. **所有多元分布退化为一元**——$d=1$ 是验证一切推导正确性的试金石
6. **$\hat{\Sigma}^{-1}$ 的去相关作用**——这是理解 Hotelling $T^2$ 和 MANOVA 优势的关键：它们考虑了变量间的相关性，避免了多重比较的信息重复

---

> **相关文章**：
> - [[Math/ProbabilityInequalities|概率不等式完全指南]]——多元统计的证明依赖马尔可夫、切比雪夫等不等式控制尾部
> - [[Math/LawOfLargeNumbers|大数定律详解]]——多元大数定律是各分量大数定律的直接推论；样本协方差矩阵的相合性依赖 SLLN
> - [[Math/CubicSplineInterpolation|三次样条插值详解]]——数值分析中的插值方法
