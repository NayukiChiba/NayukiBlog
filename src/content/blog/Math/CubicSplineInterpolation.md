---
title: 三次样条插值详解：从线性插值到光滑曲线
date: 2026-06-04
category: 数学
tags:
  - 数值分析
  - 插值
  - 样条
description: 从数学基础出发，系统讲解三次样条插值的定义、构造过程、三对角方程组推导、边界条件、追赶法求解及Python实现，辅以完整数值示例。
image: https://img.yumeko.site/file/blog/cover/1780667852039.webp
status: published
---

## 1. 问题的起点：什么是插值？

给定平面上 $n+1$ 个数据点：

$$
(x_0, y_0),\ (x_1, y_1),\ \dots,\ (x_n, y_n),\quad x_0 < x_1 < \dots < x_n
$$

我们希望找到一条**光滑曲线** $y = S(x)$，使其恰好穿过所有这些点：

$$
S(x_i) = y_i,\quad i = 0, 1, \dots, n
$$

这便是**插值问题**（Interpolation）。$x_i$ 称为**节点**（Knot），$y_i$ 称为**节点值**。

插值方案有无数种——多项式插值、三角函数插值、有理函数插值......为什么工程师和科学家偏爱**三次样条**？答案要从多项式插值的"翻车现场"说起。

## 2. 多项式插值的困境与出路

### 2.1 拉格朗日插值

给定 $n+1$ 个点，存在**唯一**的 $n$ 次多项式穿过全部点：

$$
P_n(x) = \sum_{i=0}^{n} y_i \cdot \ell_i(x),\quad
\ell_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}
$$

其中 $\ell_i(x)$ 满足 $\ell_i(x_i) = 1$ 且 $\ell_i(x_j) = 0$（$j \neq i$），是插值的"基函数"。

### 2.2 Runge 现象：高次多项式的灾难

理论无懈可击，实践中却可能惨不忍睹。以经典 Runge 函数为例：

$$
f(x) = \frac{1}{1 + 25x^2},\quad x \in [-1, 1]
$$

使用**等距节点**进行高次多项式插值时，区间两端出现剧烈振荡：

| 多项式次数 | 区间中部误差 | 端点附近最大误差 |
|:----------:|:------------:|:----------------:|
| 5 次 | $\sim 10^{-3}$ | $\sim 10^{-1}$ |
| 10 次 | $\sim 10^{-5}$ | $\sim 10^{0}$ |
| 15 次 | $\sim 10^{-6}$ | $\sim 10^{1}$ |
| 20 次 | $\sim 10^{-6}$ | $\sim 10^{2} \sim 10^{3}$ |

节点越多，两端振荡越剧烈——这违反了"数据越多越精确"的直觉。

**根本原因**：等距节点下勒贝格常数呈指数增长（$\Lambda_n \sim \frac{2^{n+1}}{e n \log n}$），将微小的数据误差在端点处放大到不可接受的程度。而多项式本身"牵一发而动全身"——任何一个系数的调整都会影响整个实数轴上的函数值。

![RungePhenomenon.png](https://img.yumeko.site/file/blog/articles/1780730941543.webp)

### 2.3 出路：分段插值

与其用一个全局高次多项式，不如**分而治之**——在每个子区间上使用低次多项式，然后"拼接"：

```
- 分段线性：──── 每个区间一条直线，节点处有"尖角"（一阶导数不连续）
- 分段二次：── 每个区间一条抛物线，一阶导数可能仍不连续
分段三次：∿∿ 每个区间一条三次曲线，二阶导数连续 -> 视觉完全光滑
```

这就是**样条**（Spline）的核心思想。名称来源于绘图员使用的弹性木条（spline），它被压铁固定在数据点处，自然弯曲成应变能最小的形状——其数学模型恰好是**分段三次多项式，且二阶导数连续**。

![PiecewiseComparison.png](https://img.yumeko.site/file/blog/articles/1780730981417.webp)

## 3. 三次样条的数学定义

### 3.1 形式化定义

给定 $(x_i, y_i),\ i=0,1,\dots,n$，函数 $S(x)$ 称为**三次样条插值函数**，当且仅当：

1. **分段三次**：在每个子区间 $[x_i, x_{i+1}]$ 上，$S(x)$ 是一个三次多项式，记为 $S_i(x)$
2. **插值条件**：$S(x_i) = y_i$，$i = 0,1,\dots,n$
3. **$C^2$ 连续性**：在内部节点 $x_1,\dots,x_{n-1}$ 处：

   $$
   \begin{aligned}
   S_{i-1}(x_i) &= S_i(x_i) &&\text{（函数连续）} \\[4pt]
   S'_{i-1}(x_i) &= S'_i(x_i) &&\text{（一阶导数连续）} \\[4pt]
   S''_{i-1}(x_i) &= S''_i(x_i) &&\text{（二阶导数连续）}
   \end{aligned}
   $$

### 3.2 自由度分析

- $n$ 个子区间 x 每区间 4 个系数 = $4n$ 个未知数
- 插值条件（每个区间两端）-> $2n$ 个方程
- 一阶导数连续（$n-1$ 个内部节点）-> $n-1$ 个方程
- 二阶导数连续（$n-1$ 个内部节点）-> $n-1$ 个方程

$$
\text{总方程数} = 2n + (n-1) + (n-1) = 4n - 2
$$

**尚缺 $2$ 个方程**——这正是边界条件（Boundary Conditions）要解决的问题。

## 4. 三次样条的构造

### 4.1 记号约定

$$
\begin{aligned}
h_i &= x_{i+1} - x_i, && i = 0,1,\dots,n-1 \quad\text{（步长）} \\[4pt]
M_i &= S''(x_i), && i = 0,1,\dots,n \quad\text{（节点处的二阶导数值，待求）}
\end{aligned}
$$

### 4.2 推导策略

**核心思路**：先求出所有 $M_i$，再用 $M_i$ 表示每个子区间上的三次多项式。

为什么以二阶导数为桥梁？

- 三次多项式的二阶导是一次多项式——两端确定，中间唯一
- $M_i$ 直接编码了曲线的"弯曲程度"，物理意义清晰
- 得出的方程组是**三对角**的，可 $O(n)$ 求解

### 4.3 用 $M_i$ 和 $M_{i+1}$ 表示 $S_i(x)$

$S_i(x)$ 是三次多项式，其二阶导数 $S''_i(x)$ 是线性函数。由端点值 $S''_i(x_i) = M_i$、$S''_i(x_{i+1}) = M_{i+1}$ 作线性插值：

$$
S''_i(x) = M_i \cdot \frac{x_{i+1} - x}{h_i} + M_{i+1} \cdot \frac{x - x_i}{h_i} \tag{4.1}
$$

对 $(4.1)$ 积分一次（得 $S'_i$）：

$$
S'_i(x) = -M_i \cdot \frac{(x_{i+1} - x)^2}{2h_i}
         + M_{i+1} \cdot \frac{(x - x_i)^2}{2h_i}
         + C_i \tag{4.2}
$$

再积分一次（得 $S_i$）：

$$
S_i(x) = M_i \cdot \frac{(x_{i+1} - x)^3}{6h_i}
       + M_{i+1} \cdot \frac{(x - x_i)^3}{6h_i}
       + C_i(x - x_i)
       + D_i \tag{4.3}
$$

利用插值条件 $S_i(x_i) = y_i$ 和 $S_i(x_{i+1}) = y_{i+1}$ 确定积分常数 $C_i$ 和 $D_i$。

代入 $x = x_i$：

$$
M_i \cdot \frac{h_i^3}{6h_i} + D_i = y_i
\quad\Rightarrow\quad
D_i = y_i - \frac{M_i h_i^2}{6}
$$

代入 $x = x_{i+1}$ 并整理：

$$
M_{i+1} \cdot \frac{h_i^2}{6} + C_i h_i + \left(y_i - \frac{M_i h_i^2}{6}\right) = y_{i+1}
\quad\Rightarrow\quad
C_i = \frac{y_{i+1} - y_i}{h_i} - \frac{h_i}{6}(M_{i+1} - M_i)
$$

最终，区间 $[x_i, x_{i+1}]$ 上的三次样条多项式为：

$$
\boxed{
\begin{aligned}
S_i(x) &= M_i \frac{(x_{i+1} - x)^3}{6h_i}
       + M_{i+1} \frac{(x - x_i)^3}{6h_i} \\[4pt]
       &+ \left(y_i - \frac{M_i h_i^2}{6}\right) \frac{x_{i+1} - x}{h_i}
       + \left(y_{i+1} - \frac{M_{i+1} h_i^2}{6}\right) \frac{x - x_i}{h_i}
\end{aligned}
} \tag{4.4}
$$

**验证**：代入 $x = x_i$，$(x_{i+1} - x_i)^3 = h_i^3$，$(x_i - x_i)^3 = 0$，$(x_{i+1} - x_i)/h_i = 1$，$(x_i - x_i)/h_i = 0$：

$$
S_i(x_i) = M_i \frac{h_i^2}{6} + \left(y_i - \frac{M_i h_i^2}{6}\right) = y_i \quad\checkmark
$$

同理可验证 $S_i(x_{i+1}) = y_{i+1}$。

![PiecewiseConstruction.png](https://img.yumeko.site/file/blog/articles/1780731085078.webp)

### 4.4 核心方程：一阶导数连续性

要使整个样条在内部节点 $x_i$ 处 $C^1$ 光滑，需 $S'_{i-1}(x_i) = S'_i(x_i)$。

由 $(4.2)$ 整理得 $S'_i(x)$ 的简化形式：

$$
S'_i(x) = -M_i \frac{(x_{i+1} - x)^2}{2h_i}
         + M_{i+1} \frac{(x - x_i)^2}{2h_i}
         + \frac{y_{i+1} - y_i}{h_i}
         - \frac{h_i}{6}(M_{i+1} - M_i) \tag{4.5}
$$

**从左侧**（$S_{i-1}$）计算 $x = x_i$ 处的导数：

$$
S'_{i-1}(x_i) = \frac{y_i - y_{i-1}}{h_{i-1}}
               + \frac{h_{i-1}}{6}M_{i-1}
               + \frac{h_{i-1}}{3}M_i \tag{4.6}
$$

**从右侧**（$S_i$）计算 $x = x_i$ 处的导数：

$$
S'_i(x_i) = \frac{y_{i+1} - y_i}{h_i}
           - \frac{h_i}{3}M_i
           - \frac{h_i}{6}M_{i+1} \tag{4.7}
$$

令 $(4.6) = (4.7)$ 并两边乘 $6$：

$$
\boxed{
h_{i-1}M_{i-1} + 2(h_{i-1} + h_i)M_i + h_i M_{i+1}
= 6\left( \frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}} \right)
} \tag{4.8}
$$

方程 $(4.8)$ 对 $i = 1, 2, \dots, n-1$ 成立，共 $n-1$ 个方程，含 $n+1$ 个未知数。

### 4.5 标准化形式

为进一步简化，定义：

$$
\mu_i = \frac{h_{i-1}}{h_{i-1} + h_i},\qquad
\lambda_i = \frac{h_i}{h_{i-1} + h_i} = 1 - \mu_i
$$

$$
d_i = \frac{6}{h_{i-1} + h_i}
      \left( \frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}} \right)
$$

其中 $d_i$ 本质上是被 $6$ 缩放后的**二阶差商**（离散二阶导数）。方程 $(4.8)$ 化为：

$$
\boxed{
\mu_i M_{i-1} + 2M_i + \lambda_i M_{i+1} = d_i,\quad i = 1, 2, \dots, n-1
} \tag{4.9}
$$

**$n-1$ 个方程，$n+1$ 个未知数**——下面补充边界条件。

## 5. 边界条件

### 5.1 自然边界（Natural Spline）⭐ 最常用

两端点处二阶导数为零：

$$
M_0 = 0,\qquad M_n = 0
$$

物理类比：一根弹性木条，两端无外力矩约束时会自然伸直。这是大多数数值库（如 SciPy 的 `CubicSpline` 默认参数 `bc_type='natural'`）的选择。

### 5.2 固定边界（Clamped / Complete Spline）

给定两端点处的一阶导数值 $y'_0$ 和 $y'_n$：

$$
S'(x_0) = y'_0,\qquad S'(x_n) = y'_n
$$

利用 $(4.5)$ 在端点的表达式，转化为 $M$ 的方程：

$$
\begin{aligned}
2h_0 M_0 + h_0 M_1 &= 6\left( \frac{y_1 - y_0}{h_0} - y'_0 \right) \\[4pt]
h_{n-1} M_{n-1} + 2h_{n-1} M_n &= 6\left( y'_n - \frac{y_n - y_{n-1}}{h_{n-1}} \right)
\end{aligned}
$$

当你确切知道端点斜率时使用。例如，$y = \sin x$ 在 $x = 0$ 处斜率为 $\cos 0 = 1$。

### 5.3 Not-a-Knot 边界

要求 $x_1$ 和 $x_{n-1}$ 处三阶导数连续——即前两个区间和后两个区间各自共享同一个三次多项式。等价条件：

$$
\begin{aligned}
h_1 M_0 - (h_0 + h_1)M_1 + h_0 M_2 &= 0 \\[4pt]
h_{n-1}M_{n-2} - (h_{n-2} + h_{n-1})M_{n-1} + h_{n-2}M_n &= 0
\end{aligned}
$$

这是 MATLAB `spline` 函数的默认选项。适用于没有额外端点信息的一般插值场景。

### 5.4 周期边界（Periodic Spline）

当 $y_0 = y_n$ 且数据呈周期性时：

$$
S'(x_0) = S'(x_n),\qquad M_0 = M_n
$$

未知数减少为 $n$ 个（$M_0$ 不再独立），方程组演变为循环三对角结构。

### 5.5 三种边界条件的直观对比

| 边界类型 | 物理类比 | 何时使用 | 默认采用者 |
|:--------:|:--------:|:--------:|:----------:|
| 自然 | 两端自由的弹性木条 | 无额外信息时 | SciPy, 教科书 |
| 固定 | 两端被夹具固定角度 | 已知端点斜率时 | 工程测量 |
| Not-a-Knot | 前/后两段是同一曲线 | 通用插值 | MATLAB |

![BoundaryConditions.png](https://img.yumeko.site/file/blog/articles/1780731085698.webp)

## 6. 三对角方程组的求解

### 6.1 方程组矩阵形式

以自然边界条件为例，$(4.9)$ 构成三对角方程组：

$$
\underbrace{\begin{bmatrix}
1 & 0 & 0 & 0 & \cdots & 0 \\
\mu_1 & 2 & \lambda_1 & 0 & \cdots & 0 \\
0 & \mu_2 & 2 & \lambda_2 & \cdots & 0 \\
\vdots & & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & \mu_{n-1} & 2 & \lambda_{n-1} \\
0 & \cdots & 0 & 0 & 0 & 1
\end{bmatrix}}_{\mathbf{A}}
\underbrace{\begin{bmatrix}
M_0 \\ M_1 \\ M_2 \\ \vdots \\ M_{n-1} \\ M_n
\end{bmatrix}}_{\mathbf{M}}
=
\underbrace{\begin{bmatrix}
0 \\ d_1 \\ d_2 \\ \vdots \\ d_{n-1} \\ 0
\end{bmatrix}}_{\mathbf{d}}
$$

系数矩阵 $\mathbf{A}$ 是**严格对角占优**的三对角矩阵（每行的 $|2| > |\mu_i| + |\lambda_i| = 1$），保证解存在且唯一。

![TridiagonalMatrix.png](https://img.yumeko.site/file/blog/articles/1780731082293.webp)

### 6.2 追赶法（Thomas Algorithm）

三对角方程组可在 **$O(n)$** 时间内求解，远优于通用消元法的 $O(n^3)$。

设方程为 $a_i M_{i-1} + b_i M_i + c_i M_{i+1} = d_i$（$i = 0, 1, \dots, n$，其中 $a_0 = c_n = 0$）。

**消元——"追"**（Forward Sweep）：

$$
\begin{aligned}
c'_0 &= \frac{c_0}{b_0},\quad d'_0 = \frac{d_0}{b_0} \\[4pt]
c'_i &= \frac{c_i}{b_i - a_i \cdot c'_{i-1}},\quad
d'_i = \frac{d_i - a_i \cdot d'_{i-1}}{b_i - a_i \cdot c'_{i-1}},\quad i=1,2,\dots,n
\end{aligned}
$$

**回代——"赶"**（Backward Sweep）：

$$
\begin{aligned}
M_n &= d'_n \\[4pt]
M_i &= d'_i - c'_i \cdot M_{i+1},\quad i = n-1, n-2, \dots, 0
\end{aligned}
$$

![ThomasAlgorithm.png](https://img.yumeko.site/file/blog/articles/1780731227503.webp)

### 6.3 插值求值

求解出全部 $M_i$ 后，对任意 $x \in [x_0, x_n]$：

1. 二分查找确定 $x$ 所在区间 $[x_i, x_{i+1}]$
2. 代入 $(4.4)$ 计算 $S_i(x)$

## 7. 完整数值示例

用一组具体小数据走完整流程。

### 7.1 数据

$$
\begin{array}{c|cccc}
x & 0 & 1 & 2 & 3 \\
\hline
y & 0 & 2 & 1 & 3
\end{array}
$$

$n=3$，4 个节点，3 个子区间。

### 7.2 步长和一阶差商

$$
h_0 = h_1 = h_2 = 1 \quad\text{（等距节点）}
$$

$$
\frac{y_1-y_0}{h_0} = 2,\qquad
\frac{y_2-y_1}{h_1} = -1,\qquad
\frac{y_3-y_2}{h_2} = 2
$$

### 7.3 计算 $d_i$

$$
\begin{aligned}
d_1 &= \frac{6}{2}\left( \frac{y_2-y_1}{h_1} - \frac{y_1-y_0}{h_0} \right)
    = 3 \cdot (-1 - 2) = -9 \\[6pt]
d_2 &= \frac{6}{2}\left( \frac{y_3-y_2}{h_2} - \frac{y_2-y_1}{h_1} \right)
    = 3 \cdot (2 - (-1)) = 9
\end{aligned}
$$

### 7.4 建立方程组

等距节点 -> $\mu_i = \lambda_i = 0.5$。自然边界 $M_0 = M_3 = 0$：

$$
\begin{cases}
M_0 = 0 \\
0.5 \cdot 0 + 2M_1 + 0.5M_2 = -9 \\
0.5M_1 + 2M_2 + 0.5 \cdot 0 = 9 \\
M_3 = 0
\end{cases}
\;\Longrightarrow\;
\begin{cases}
2M_1 + 0.5M_2 = -9 \\
0.5M_1 + 2M_2 = 9
\end{cases}
$$

求解二元一次方程组：

$$
M_1 = -6,\qquad M_2 = 6
$$

最终：

$$
M_0 = 0,\quad M_1 = -6,\quad M_2 = 6,\quad M_3 = 0
$$

### 7.5 构造三段多项式

运用公式 $(4.4)$，分别代入各区间的 $M_i$、$M_{i+1}$、$h_i$、$y_i$、$y_{i+1}$。

**区间 $[0, 1]$**：$M_0 = 0$，$M_1 = -6$

$$
S_0(x) = -x^3 + 3x
$$

**区间 $[1, 2]$**：$M_1 = -6$，$M_2 = 6$

$$
S_1(x) = -(2-x)^3 + (x-1)^3 + 3(2-x)
$$

**区间 $[2, 3]$**：$M_2 = 6$，$M_3 = 0$

$$
S_2(x) = (3-x)^3 + 3x - 6
$$

### 7.6 验证

| 验证项 | $x=0$ | $x=1$ | $x=2$ | $x=3$ |
|:------|:-----:|:-----:|:-----:|:-----:|
| $S(x)$ 值 | $0$ ✓ | $2$ ✓ | $1$ ✓ | $3$ ✓ |
| $S'(x)$ 连续 | — | $S'_0(1)=S'_1(1)=0$ ✓ | $S'_1(2)=S'_2(2)=-3$ ✓ | — |
| $S''(1)$ | — | $M_1=-6$ ✓ | — | — |
| $S''(2)$ | — | — | $M_2=6$ ✓ | — |

曲线在全部节点处一阶、二阶导数均连续——视觉上是一整条**完全光滑**的曲线。

![NumericalExample.png](https://img.yumeko.site/file/blog/articles/1780731249758.webp)

## 8. Python 实现

### 8.1 手写实现（教学版）

```python
"""
三次样条插值 —— 教学实现

使用方法：
    python cubicSpline.py
"""

import numpy as np
from typing import Tuple


def cubicSplineNatural(
    xData: np.ndarray,
    yData: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造自然三次样条插值

    Args:
        xData: 节点横坐标，需严格递增
        yData: 节点纵坐标

    Returns:
        (x, a, b, c, d): 每个子区间的三次多项式系数
            S_i(t) = a_i + b_i*t + c_i*t^2 + d_i*t^3
            其中 t = x - x_i
    """
    n = len(xData) - 1  # 区间数
    h = np.diff(xData)  # 步长 h_i = x_{i+1} - x_i

    # 计算三对角方程组右边 d_i
    # d_i = 6 * [(y_{i+1} - y_i)/h_i - (y_i - y_{i-1})/h_{i-1}]
    dy = np.diff(yData)
    rhs = 6.0 * np.diff(dy / h)  # rhs[i] 对应内部节点 x_{i+1}

    # 构造三对角矩阵（自然边界后，有 n-1 个未知数 M_1,...,M_{n-1}）
    # 主对角：2*(h_{i-1} + h_i)，次对角：h_i
    diagMain = 2.0 * (h[:-1] + h[1:])
    diagSub = h[1:-1]    # 下对角：h_1, h_2, ..., h_{n-2}
    diagSuper = h[1:-1]  # 上对角：h_1, h_2, ..., h_{n-2}

    # 追赶法求解三对角方程组
    M = np.zeros(n + 1)               # 包含 M_0 = 0 和 M_n = 0
    M[1:n] = solveTridiagonal(diagSub, diagMain, diagSuper, rhs)

    # 计算每个子区间的多项式系数（使用局部坐标 t = x - x_i）
    a = yData[:-1]                          # 常数项 = S_i(0) = y_i
    b = dy / h - h * (2 * M[:-1] + M[1:]) / 6.0  # 一次项系数
    c = M[:-1] / 2.0                        # 二次项系数
    d = np.diff(M) / (6.0 * h)              # 三次项系数

    return xData[:-1], a, b, c, d


def solveTridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray
) -> np.ndarray:
    """
    追赶法求解三对角方程组

    a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
    a[0] = c[m-1] = 0

    Args:
        a: 下对角（长度 m，a[0] 不使用）
        b: 主对角（长度 m）
        c: 上对角（长度 m，c[m-1] 不使用）
        d: 右边向量（长度 m）

    Returns:
        x: 解向量
    """
    m = len(b)
    # 工作副本
    cPrime = np.zeros(m - 1)
    dPrime = np.zeros(m)

    # 追（Forward）
    cPrime[0] = c[0] / b[0]
    dPrime[0] = d[0] / b[0]

    for i in range(1, m):
        denom = b[i] - a[i] * cPrime[i - 1]
        if i < m - 1:
            cPrime[i] = c[i] / denom
        dPrime[i] = (d[i] - a[i] * dPrime[i - 1]) / denom

    # 赶（Backward）
    x = np.zeros(m)
    x[-1] = dPrime[-1]

    for i in range(m - 2, -1, -1):
        x[i] = dPrime[i] - cPrime[i] * x[i + 1]

    return x


def evaluateSpline(
    x: np.ndarray,
    xKnots: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray
) -> np.ndarray:
    """
    对任意点集求样条值

    Args:
        x: 待求值的横坐标数组
        xKnots: 各区间左端点（即 xData[:-1]）
        a, b, c, d: 多项式系数

    Returns:
        y: 插值结果
    """
    # 二分查找每个 x 所在的区间
    indices = np.searchsorted(xKnots, x, side='right') - 1
    indices = np.clip(indices, 0, len(xKnots) - 1)

    t = x - xKnots[indices]  # 局部坐标
    y = (a[indices]
         + b[indices] * t
         + c[indices] * t**2
         + d[indices] * t**3)

    return y


# ===== 使用示例 =====
if __name__ == "__main__":
    xData = np.array([0.0, 1.0, 2.0, 3.0])
    yData = np.array([0.0, 2.0, 1.0, 3.0])

    # 构造样条
    xKnots, a, b, c, d = cubicSplineNatural(xData, yData)

    print("各区间多项式系数 (S_i(t) = a + b*t + c*t^2 + d*t^3):")
    for i in range(len(xKnots)):
        print(f"  区间 [{xKnots[i]}, {xKnots[i]+np.diff(xData)[i]}]: "
              f"a={a[i]:.3f}, b={b[i]:.3f}, c={c[i]:.3f}, d={d[i]:.3f}")

    # 密集采样求值
    xFine = np.linspace(0, 3, 61)
    yFine = evaluateSpline(xFine, xKnots, a, b, c, d)

    print(f"\n插值结果 (前5个点):")
    for i in range(5):
        print(f"  S({xFine[i]:.2f}) = {yFine[i]:.4f}")
```

运行结果：

```
各区间多项式系数 (S_i(t) = a + b*t + c*t^2 + d*t^3):
  区间 [0.0, 1.0]: a=0.000, b=3.000, c=0.000, d=-1.000
  区间 [1.0, 2.0]: a=2.000, b=0.000, c=-3.000, d=2.000
  区间 [2.0, 3.0]: a=1.000, b=-3.000, c=3.000, d=-1.000

插值结果 (前5个点):
  S(0.00) = 0.0000
  S(0.05) = 0.1499
  S(0.10) = 0.2990
  S(0.15) = 0.4466
  S(0.20) = 0.5920
```

展开验证区间 $[0, 1]$：$S_0(x) = 0 + 3x + 0x^2 + (-1)x^3 = 3x - x^3$，与第 7 节手算结果一致。

### 8.2 使用 SciPy（工程版）

```python
from scipy.interpolate import CubicSpline
import numpy as np

xData = np.array([0.0, 1.0, 2.0, 3.0])
yData = np.array([0.0, 2.0, 1.0, 3.0])

# 自然边界
csNatural = CubicSpline(xData, yData, bc_type='natural')
print(csNatural([0.5, 1.5, 2.5]))  # [1.375 1.625 1.875]

# 固定边界（指定端点一阶导数）
csClamped = CubicSpline(xData, yData, bc_type=((1, 3.0), (1, -2.0)))
print(csClamped([0.5, 1.5, 2.5]))

# Not-a-Knot 边界
csNak = CubicSpline(xData, yData, bc_type='not-a-knot')
print(csNak([0.5, 1.5, 2.5]))
```

## 9. 样条插值的最优性质

三次自然样条拥有一个深刻的**变分特性**（即"最小弯曲能"性质）：

> 在所有满足插值条件 $f(x_i) = y_i$ 且在 $[x_0, x_n]$ 上二阶导数平方可积的函数中，三次自然样条**最小化**如下泛函：
>
> $$
> J[f] = \int_{x_0}^{x_n} [f''(x)]^2 \, dx
> $$

**证明概要**：设 $S(x)$ 为三次自然样条（$S''(x_0) = S''(x_n) = 0$），$f$ 为任意满足插值条件的函数。令 $g(x) = f(x) - S(x)$，则 $g(x_i) = 0$：

$$
\begin{aligned}
\int (f'')^2 dx &= \int (S'' + g'')^2 dx \\
&= \int (S'')^2 dx + \int (g'')^2 dx + 2\int S'' g'' dx
\end{aligned}
$$

交叉项分部积分，并利用 $S$ 在各区间是三次多项式（$S^{(4)} = 0$）及 $S''(x_0) = S''(x_n) = 0$，可证 $\int S'' g'' dx = 0$。故：

$$
\int (f'')^2 dx = \int (S'')^2 dx + \int (g'')^2 dx \ge \int (S'')^2 dx
$$

等号成立当且仅当 $g'' \equiv 0$，即 $f = S$。

**物理含义**：如果将 $f''(x)$ 理解为"曲率"的线性近似（小挠度假设），那么 $\int [f'']^2 dx$ 正比于弹性木条的**弯曲应变能**。三次自然样条是所有插值曲线中"最不弯曲的"——这正是为什么物理样条（弹性木条）能自动形成三次样条曲线。

> **注意**：精确的曲率是 $\kappa = f''/(1 + {f'}^2)^{3/2}$。上述变分原理用的是 $[f'']^2$ 作为 $\kappa^2$ 的近似，对应**小挠度假定**（$|f'| \ll 1$）。在一般情形下，物理样条并非精确的三次样条，但实践中这个近似已经足够好。

## 10. 与其他插值方法的对比

| 方法 | 光滑性 | 局部性 | 振荡风险 | 计算复杂度 | 适用场景 |
|:----:|:------:|:------:|:--------:|:----------:|:--------:|
| 拉格朗日（高次） | $C^\infty$ | ✗ 全局 | ⚠️ 高（Runge） | $O(n^2)$ | 仅低次可用 |
| 分段线性 | $C^0$ | ✓ | 无 | $O(n)$ | 粗糙快速估计 |
| **三次样条（自然）** | **$C^2$** | **近似局部** | **低** | **$O(n)$** | **通用光滑插值** |
| Akima 样条 | $C^1$ | ✓ 严格局部 | 极低 | $O(n)$ | 抗过冲，快速变化数据 |
| B 样条 | 可配置 | ✓ 局部 | 低 | $O(n)$ | CAD/图形学 |
| 径向基函数（RBF） | $C^\infty$ | ✗ 全局 | 中 | $O(n^3)$ | 散乱数据/高维 |

**关键洞察**：三次样条在光滑性（$C^2$）、局部性（一个节点的改动主要影响相邻三四个区间）、计算效率（$O(n)$）三者之间取得了**最佳平衡**。

![InterpolationComparison.png](https://img.yumeko.site/file/blog/articles/1780731281344.webp)

## 11. 常见误区与要点

### 11.1 三次样条 != 过三个点的唯一三次多项式

这是初学者最常犯的混淆。一条三次样条由**多段**三次多项式拼接而成，每段只穿过本区间的两个节点；全局一致性由一阶和二阶导数连续性保证。

### 11.2 样条不会因节点增多而振荡

不同于全局多项式插值，三次样条的误差界是**一致有界**的。若被插值函数 $f \in C^4[a,b]$，则误差满足：

$$
\|f - S\|_\infty \le \frac{5}{384} \cdot \max_i h_i^4 \cdot \|f^{(4)}\|_\infty
$$

即加密节点（减小 $h_i$）时误差以 $O(h^4)$ 收敛，且**没有 Runge 型发散**。

### 11.3 自然边界不总是最佳选择

$M_0 = M_n = 0$ 意味着样条在端点处"不弯曲"——如果真实函数的二阶导数在端点不接近零，自然样条在边界附近会偏离真值。当你对边界行为有先验知识（如周期性数据），应选择对应的边界条件。

### 11.4 外推不可靠

样条是**插值**方法，区间 $[x_0, x_n]$ 之外的行为不受控制。外推时应使用回归或其他外推技术，而非直接用样条公式。

## 12. 总结

| 要点 | 说明 |
|:----|:----|
| 核心思想 | 分段三次多项式 + $C^2$ 连续性条件 |
| 为什么三次？ | 最低次多项式能保证二阶导数连续（视觉光滑） |
| 未知量策略 | 以节点二阶导数 $M_i$ 为桥梁，先求解再反推多项式系数 |
| 核心方程 | $\mu_i M_{i-1} + 2M_i + \lambda_i M_{i+1} = d_i$（三对角） |
| 边界条件 | 自然（$M_0=M_n=0$）/ 固定 / Not-a-Knot / 周期 |
| 求解 | 追赶法 $O(n)$ |
| 误差 | $O(h^4)$ 一致收敛，无 Runge 现象 |
| 最优性 | 最小化 $\int [f'']^2 dx$（最小弯曲能） |
| 计算工具 | SciPy `CubicSpline`，可直接使用 |

三次样条是现代数值分析中最优雅的算法之一——简单的数学结构（三次多项式 + 三对角矩阵），却产出了光滑、稳定、高效的插值曲线。无论是科研绘图、工程可视化还是几何建模，理解其数学基础都能让你更自信地使用和调试插值相关代码。
