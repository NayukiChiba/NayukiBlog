---
title: 距离、向量范数与矩阵范数：从定义到应用
date: 2026-06-07
category: 数学
tags:
  - 线性代数
  - 范数
  - 距离
  - 正则化
  - 数值计算
description: 从度量空间公理出发，系统讲解向量范数（L0/L1/L2/L^inf/Lp）、矩阵范数（Frobenius、算子范数、核范数）的定义、计算方法、几何直觉和机器学习应用，附完整 Python 示例。
image: https://img.yumeko.site/file/blog/VectorMatrixNorm.png
status: draft
---

> **前置阅读**：本文假定读者熟悉线性代数基本概念。范数与矩阵分解密切相关，可参阅 [[Math/MatrixDecomposition|矩阵分解完全指南]]。

![图0: 范数与距离 Banner](https://img.yumeko.site/file/blog/VectorMatrixNorm.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于技术博客封面。
> 设计概念：从左到右展示三个层次的概念——
> 距离（两点间的线段标注 d(x,y)）、向量范数（箭头 x，旁注 ||x||_p 的不同 p 值下的几何形状）、
> 矩阵范数（矩阵块，旁注 ||A||_F 和 ||A||_2）。
> 配色：深蓝到紫色渐变，现代数学教科书风格。
> 顶部留白供标题叠加。
> ```

## 1. 问题的起点

在机器学习中，「距离」和「大小」无处不在：

- KNN 分类：靠「最近邻」投票，但什么是「近」？
- 正则化：L1 和 L2 惩罚项到底有什么不同？
- 矩阵条件数：如何衡量矩阵求逆的数值稳定性？
- 低秩近似：截断 SVD 的误差用什么范数度量？

这些问题的答案都指向同一个核心概念——**范数**（norm）。范数是「长度」概念的推广，定义了向量和矩阵的「大小」，进而定义了空间中两点之间的「距离」。

## 2. 距离：度量空间公理

### 2.1 定义

在数学上，「距离」不是一个可以随意定义的概念——它必须满足四条公理。设集合 $X$ 上的函数 $d: X \times X \to \mathbb{R}_{\ge 0}$，若满足以下条件，则称 $(X, d)$ 为**度量空间**（metric space）：

**非负性**：$d(x, y) \ge 0$，且 $d(x, y) = 0 \iff x = y$

**对称性**：$d(x, y) = d(y, x)$

**三角不等式**：$d(x, z) \le d(x, y) + d(y, z)$

$$
\boxed{d(x, z) \le d(x, y) + d(y, z)}
$$

三角不等式是距离定义中最关键的公理——它保证「直走不绕路」，即经过中间点的路径不会比直达更短。

### 2.2 由范数诱导的距离

最常见的距离由范数诱导：

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|
$$

只要 $\|\cdot\|$ 是范数（满足下一节的条件），上式定义的 $d$ 自动满足度量空间四条公理。

但并非所有距离都由范数诱导。例如离散度量 $d(x,y) = \mathbf{1}_{\{x \neq y\}}$（两点相同距离为 0，不同为 1）就是一个合法的距离，但不存在对应的范数。

## 3. 向量范数

### 3.1 范数的三条公理

向量空间 $V$ 上的函数 $\|\cdot\|: V \to \mathbb{R}_{\ge 0}$ 称为**范数**，当且仅当：

**绝对齐次性**：$\|\alpha \mathbf{x}\| = |\alpha| \cdot \|\mathbf{x}\|$

**三角不等式**：$\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$

**正定性**：$\|\mathbf{x}\| \ge 0$，且 $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$

### 3.2 Lp 范数族

对于 $\mathbb{R}^n$ 中的向量 $\mathbf{x} = (x_1, x_2, \dots, x_n)^\top$，Lp 范数（p-范数）定义为：

$$
\boxed{\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}}, \quad p \ge 1
$$

**特殊的 p 值**产生了几种最常用的范数：

#### L0 「范数」

严格来说不是范数（不满足绝对齐次性），但极其常用：

$$
\|\mathbf{x}\|_0 = \text{非零元素的个数}
$$

例如 $\mathbf{x} = (0, -3, 0, 5)^\top$ 的 L0 范数为 2。

#### L1 范数（曼哈顿距离 / 出租车范数）

$$
\boxed{\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|}
$$

对应 $p=1$。得名于曼哈顿街区网格状道路——只能沿坐标轴方向移动。

**数值示例**：$\mathbf{x} = (3, -4)^\top$，$\|\mathbf{x}\|_1 = |3| + |-4| = 7$。

#### L2 范数（欧几里得范数）

$$
\boxed{\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}}
$$

对应 $p=2$。这是我们最熟悉的「直线距离」。$\mathbf{x} = (3, -4)^\top$，$\|\mathbf{x}\|_2 = \sqrt{9 + 16} = 5$。

#### L^inf 范数（最大范数 / Chebyshev 范数）

$$
\boxed{\|\mathbf{x}\|_\infty = \max_{1 \le i \le n} |x_i|}
$$

当 $p \to \infty$ 时，Lp 范数的极限就是各分量绝对值的最大值。$\mathbf{x} = (3, -4)^\top$，$\|\mathbf{x}\|_\infty = \max(3, 4) = 4$。

![图1: Lp 范数单位球](https://img.yumeko.site/file/blog/VectorMatrixNorm/LpUnitBalls.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的对比图，在同一个 2D 坐标系中绘制四种 Lp 范数的单位球（满足 ||x||_p = 1 的点集）：
> L0.5（凹星形，虚线）、L1（菱形）、L2（圆形）、L^inf（正方形）。
> 用不同颜色区分，图例标注。坐标轴从 -1.5 到 1.5。
> 白色背景，细网格线，学术教科书风格。
> ```

### 3.3 几何直觉：单位球

范数的**单位球** $\{\mathbf{x}: \|\mathbf{x}\|_p = 1\}$ 直观展示了每种范数的几何性质：

| p 值 | 单位球形状 | 特点 |
|:--|:--|:--|
| $p=0.5$ | 凹星形 | 不是范数（违反三角不等式），但 L0.5 在压缩感知中仍有应用 |
| $p=1$ | **菱形**（旋转正方形） | 角在坐标轴上，稀疏性强 |
| $p=1.5$ | 圆角菱形 | L1 和 L2 的过渡 |
| $p=2$ | **圆形** | 各向同性，旋转不变 |
| $p=\infty$ | **正方形** | 边平行于坐标轴 |

> [!TIP] L1 为什么产生稀疏解？
> L1 单位球的「尖角」在坐标轴上。在约束优化中，目标函数的等高线最容易和这些尖角相切，而尖角对应的坐标就是「某些分量为零」——这正是稀疏性的来源。相比之下，L2 的圆形球面光滑，等高线可以相切于球面的任意位置，通常所有分量都不为零。

### 3.4 Lp 范数的计算示例

对 $\mathbf{x} = (1, -2, 3, -4)^\top$：

| 范数 | 计算过程 | 结果 |
|:--|:--|:--:|
| $\|\mathbf{x}\|_0$ | 非零元素：1, -2, 3, -4 共 4 个 | 4 |
| $\|\mathbf{x}\|_1$ | $1 + 2 + 3 + 4$ | 10 |
| $\|\mathbf{x}\|_2$ | $\sqrt{1 + 4 + 9 + 16} = \sqrt{30}$ | $\approx 5.48$ |
| $\|\mathbf{x}\|_3$ | $(1 + 8 + 27 + 64)^{1/3} = 100^{1/3}$ | $\approx 4.64$ |
| $\|\mathbf{x}\|_\infty$ | $\max(1, 2, 3, 4)$ | 4 |

观察：$\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1$，这是普遍成立的范数不等式链。

### 3.5 范数等价性

在有限维空间 $\mathbb{R}^n$ 中，所有范数都是**等价**的——存在正常数 $c, C$ 使得：

$$
c\|\mathbf{x}\|_a \le \|\mathbf{x}\|_b \le C\|\mathbf{x}\|_a, \quad \forall \mathbf{x} \in \mathbb{R}^n
$$

最常见的等价不等式链：

$$
\boxed{\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\,\|\mathbf{x}\|_2 \le n\,\|\mathbf{x}\|_\infty}
$$

> [!NOTE] 范数等价的实际意义
> 在分析算法收敛性时，可以选用最方便的范数进行推导，然后利用等价性转换到目标范数。例如：证明梯度下降时先用 $\|\cdot\|_2$ 推导，再用等价性推出 $\|\cdot\|_\infty$ 的界。

## 4. 矩阵范数

矩阵范数是向量范数的自然推广。对于 $m \times n$ 矩阵空间，范数必须额外满足**次乘性**（submultiplicativity）：

$$
\|AB\| \le \|A\| \cdot \|B\|
$$

这一性质保证了当 $\|A\| < 1$ 时，Neumann 级数 $(I - A)^{-1} = \sum_{k=0}^\infty A^k$ 收敛。

### 4.1 Frobenius 范数（逐元素 L2）

将矩阵当作一个长向量，计算其 L2 范数：

$$
\boxed{\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{\operatorname{tr}(A^\top A)} = \sqrt{\sum_{i=1}^{\min(m,n)} \sigma_i^2}}
$$

Frobenius 范数是实践中最常用的矩阵范数之一——计算简单（直接平方和开方），且是**酉不变**的：$\|UAV\|_F = \|A\|_F$（$U, V$ 为正交矩阵）。

**数值示例**：

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\|A\|_F = \sqrt{1^2 + 2^2 + 3^2 + 4^2} = \sqrt{30} \approx 5.48
$$

### 4.2 算子范数（诱导范数）

算子范数由向量范数「诱导」而来，衡量矩阵作为线性变换的最大拉伸倍数：

$$
\boxed{\|A\|_p = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_p}
$$

**算子范数的直观意义**：在单位 Lp 球面上，被 $A$ 映射后的所有向量中最长的那个的长度。

#### 矩阵 L1 范数（最大绝对列和）

当 $p=1$ 时，算子范数退化为一个简单的闭式：

$$
\boxed{\|A\|_1 = \max_{1 \le j \le n} \sum_{i=1}^{m} |a_{ij}|}
$$

即**各列绝对值之和的最大值**。

**推导直觉**：$\|A\mathbf{x}\|_1 = \sum_i |\sum_j a_{ij} x_j| \le \sum_j |x_j| \sum_i |a_{ij}| \le \|\mathbf{x}\|_1 \cdot \max_j \sum_i |a_{ij}|$。取 $\mathbf{x} = \mathbf{e}_j$（第 $j$ 个标准基向量）达到上界。

**数值示例**：$A = \begin{bmatrix} 1 & -2 \\ 3 & 4 \end{bmatrix}$

列和：第 1 列 $|1|+|3| = 4$，第 2 列 $|-2|+|4| = 6$。$\|A\|_1 = \max(4, 6) = 6$。

#### 矩阵 L^inf 范数（最大绝对行和）

当 $p=\infty$ 时，算子范数也退化为简单闭式：

$$
\boxed{\|A\|_\infty = \max_{1 \le i \le m} \sum_{j=1}^{n} |a_{ij}|}
$$

即**各行绝对值之和的最大值**。

**数值示例**：$A = \begin{bmatrix} 1 & -2 \\ 3 & 4 \end{bmatrix}$

行和：第 1 行 $|1|+|-2| = 3$，第 2 行 $|3|+|4| = 7$。$\|A\|_\infty = \max(3, 7) = 7$。

> [!TIP] L1 和 L^inf 的对称性
> 矩阵的 L1 范数是**列和**最大值，L^inf 范数是**行和**最大值。这不难记——下标 1 对应竖着的列（像数字 1），下标 ^inf 对应横着的行。

#### 矩阵 L2 范数（谱范数）

当 $p=2$ 时，算子范数等于**最大奇异值**：

$$
\boxed{\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^\top A)}}
$$

**推导**：$\|A\|_2^2 = \max_{\|\mathbf{x}\|_2=1} \|A\mathbf{x}\|_2^2 = \max \mathbf{x}^\top A^\top A \mathbf{x} = \lambda_{\max}(A^\top A) = \sigma_{\max}^2(A)$。最后一个等号用了 Rayleigh 商的极值性质。

**数值示例**：$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$

计算 $A^\top A = \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}$，特征值 $\lambda_{\max} = 15 + \sqrt{221} \approx 29.87$。

$$
\|A\|_2 = \sigma_{\max}(A) \approx \sqrt{29.87} \approx 5.46
$$

**验证**：$\|A\|_2 \le \|A\|_F = 5.48$（谱范数不超过 Frobenius 范数）✓。

### 4.3 核范数（Nuclear Norm）

核范数是所有奇异值之和：

$$
\boxed{\|A\|_* = \sum_{i=1}^{\min(m,n)} \sigma_i(A)}
$$

核范数是秩函数的**最紧凸松弛**——在矩阵补全（matrix completion）问题中，用核范数替代秩做正则化，使非凸的秩最小化问题变为可解的凸优化问题。

**数值示例**：$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$

奇异值 $\sigma_1 \approx 5.46$，$\sigma_2 \approx 0.37$。

$$
\|A\|_* = 5.46 + 0.37 \approx 5.83
$$

### 4.4 各种矩阵范数的对比

对同一个矩阵 $A = \begin{bmatrix} 1 & 2 & 0 \\ -1 & 3 & 1 \end{bmatrix}$：

| 范数 | 计算 | 结果 |
|:--|:--|:--:|
| $\|A\|_F$ | $\sqrt{1+4+0+1+9+1} = \sqrt{16}$ | 4 |
| $\|A\|_1$（列和最大） | $\max(2, 5, 1)$ | 5 |
| $\|A\|_\infty$（行和最大） | $\max(3, 5)$ | 5 |
| $\|A\|_2$（谱范数） | 最大奇异值 | $\approx 3.72$ |
| $\|A\|_*$（核范数） | 奇异值之和 | $\approx 4.82$ |

### 4.5 范数不等式链

矩阵的几种常用范数满足以下不等式：

$$
\boxed{\|A\|_2 \le \|A\|_F \le \sqrt{r}\,\|A\|_2 \le \|A\|_* \le \sqrt{r}\,\|A\|_F}
$$

其中 $r = \operatorname{rank}(A)$。这些不等式在矩阵分析中反复出现，值得记住前两个：

- $\|A\|_2 \le \|A\|_F$：最大奇异值不超过所有奇异值的平方和开方
- $\|A\|_F \le \sqrt{r}\|A\|_2$：所有奇异值的平方和不超过 $r$ 倍最大奇异值平方

## 5. 在机器学习中的应用

### 5.1 L1 正则化（Lasso）→ 稀疏模型

L1 正则化在损失函数上添加权重向量的 L1 范数：

$$
\mathcal{L}_{\text{Lasso}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|_1
$$

L1 惩罚倾向于产生**稀疏解**（很多权重恰好为零），自动实现特征选择。这是 §3.3 中 L1 单位球「尖角」效应的直接应用。

### 5.2 L2 正则化（Ridge）→ 权重衰减

L2 正则化添加权重向量的 L2 范数平方：

$$
\mathcal{L}_{\text{Ridge}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|_2^2
$$

L2 惩罚倾向于让所有权重**均匀缩小**（但不为零），有效防止过拟合。在梯度下降中等价于 weight decay。

### 5.3 矩阵条件数 → 数值稳定性

矩阵条件数定义为最大与最小奇异值之比：

$$
\boxed{\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \|A\|_2 \cdot \|A^{-1}\|_2}
$$

条件数越大，线性方程组 $Ax = b$ 对 $b$ 的微小扰动越敏感，数值求解越不稳定。判断依据：

- $\kappa \approx 1$：良态（well-conditioned），如正交矩阵
- $\kappa \gg 1$：病态（ill-conditioned），求解误差被放大 $\kappa$ 倍

### 5.4 截断 SVD 的误差度量

截断 SVD $A_k$ 的近似误差恰好用奇异值的 Frobenius 范数（或谱范数）表示：

$$
\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2, \qquad \|A - A_k\|_2 = \sigma_{k+1}
$$

Eckart-Young-Mirsky 定理保证这是秩-$k$ 近似在两种范数下的最优结果。

### 5.5 距离度量与 KNN

KNN 算法的核心就是距离。不同范数诱导不同距离，适用于不同场景：

| 距离 | 公式 | 适用场景 |
|:--|:--|:--|
| 欧几里得（L2） | $\|\mathbf{x} - \mathbf{y}\|_2$ | 通用，各向同性数据 |
| 曼哈顿（L1） | $\|\mathbf{x} - \mathbf{y}\|_1$ | 高维稀疏数据（文本 TF-IDF） |
| Chebyshev（L^inf） | $\max_i |x_i - y_i|$ | 最大分量差异敏感的场景 |
| 余弦距离 | $1 - \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$ | 方向敏感、模长不敏感的文本/图像特征 |

## 6. Python 实现汇总

```python
"""
向量范数与矩阵范数计算汇总

使用 NumPy 和 SciPy 计算各种常用范数
"""

import numpy as np
from scipy import linalg


def computeVectorNorms(x: np.ndarray):
    """
    计算向量的各种范数

    Args:
        x: 一维向量
    Returns:
        包含各范数值的字典
    """
    results = {}
    # L0：非零元素个数
    results['L0'] = np.count_nonzero(x)
    # L1：绝对值之和
    results['L1'] = np.linalg.norm(x, ord=1)
    # L2：欧几里得范数
    results['L2'] = np.linalg.norm(x, ord=2)
    # L^inf：最大绝对值
    results['Linf'] = np.linalg.norm(x, ord=np.inf)
    # 自定义 L3
    results['L3'] = np.linalg.norm(x, ord=3)
    return results


def computeMatrixNorms(A: np.ndarray):
    """
    计算矩阵的各种范数

    Args:
        A: 二维矩阵
    Returns:
        包含各范数值的字典
    """
    results = {}
    # Frobenius 范数（ord='fro'）
    results['Frobenius'] = np.linalg.norm(A, ord='fro')
    # L1 算子范数：最大列和
    results['L1_operator'] = np.linalg.norm(A, ord=1)
    # L2 算子范数（谱范数）：最大奇异值
    results['L2_spectral'] = np.linalg.norm(A, ord=2)
    # L^inf 算子范数：最大行和
    results['Linf_operator'] = np.linalg.norm(A, ord=np.inf)
    # 核范数：奇异值之和
    singularValues = linalg.svdvals(A)
    results['Nuclear'] = np.sum(singularValues)
    return results


def computeConditionNumber(A: np.ndarray):
    """
    计算矩阵的 L2 条件数

    Args:
        A: 方阵
    Returns:
        条件数 κ(A) = σ_max / σ_min
    """
    sv = linalg.svdvals(A)
    # 避免除以零
    sv = sv[sv > 1e-15]
    return sv[0] / sv[-1] if len(sv) > 0 else np.inf


def main():
    """演示所有范数计算"""
    np.set_printoptions(precision=4, suppress=True)

    # 测试向量
    x = np.array([1, -2, 3, -4])
    print("向量 x =", x)
    print("向量范数:", computeVectorNorms(x))

    # 测试矩阵
    A = np.array([[1, 2, 0],
                   [-1, 3, 1]])
    print("\n矩阵 A =\n", A)
    print("矩阵范数:", computeMatrixNorms(A))
    print("条件数 κ(A):", computeConditionNumber(A))

    # 验证范数不等式
    frob = np.linalg.norm(A, 'fro')
    spectral = np.linalg.norm(A, 2)
    print(f"\n范数不等式验证：\|A\|_2 = {spectral:.4f} ≤ \|A\|_F = {frob:.4f}")


if __name__ == "__main__":
    main()
```

## 7. 总结

| 层次 | 核心概念 | 关键公式 | ML 典型应用 |
|:--|:--|:--|:--|
| 距离 | 度量空间四条公理 | $d(x,y) = \|x - y\|$ | KNN、聚类、度量学习 |
| 向量 L1 | 绝对值之和 | $\|x\|_1 = \sum \|x_i\|$ | Lasso 正则化、稀疏编码 |
| 向量 L2 | 欧几里得长度 | $\|x\|_2 = \sqrt{\sum x_i^2}$ | Ridge 正则化、梯度下降 |
| 向量 L^inf | 最大分量 | $\|x\|_\infty = \max \|x_i\|$ | 鲁棒性分析 |
| 矩阵 Frobenius | 逐元素 L2 | $\|A\|_F = \sqrt{\sum a_{ij}^2}$ | 矩阵近似误差 |
| 矩阵谱范数 | 最大奇异值 | $\|A\|_2 = \sigma_{\max}$ | 条件数、收敛性分析 |
| 矩阵核范数 | 奇异值之和 | $\|A\|_* = \sum \sigma_i$ | 矩阵补全、低秩恢复 |

> [!ABSTRACT] 关键直觉
> - **范数 = 大小的度量**。向量范数量化向量的长度，矩阵范数量化矩阵的「作用强度」
> - **L1 产生稀疏，L2 产生平滑**——这是机器学习正则化选择的核心依据
> - **算子范数 = 最大拉伸倍数**——衡量矩阵作为线性变换的「极端效应」
> - **核范数 = 秩的凸替代**——在需要低秩结构时使用，如推荐系统、图像修复

---

> **相关文章**：
> - [[Math/MatrixDecomposition|矩阵分解完全指南]]
> - [[Math/MultivariateStatistics|多元统计分析基础]]
> - [[Math/ProbabilityInequalities|概率不等式完全指南]]
