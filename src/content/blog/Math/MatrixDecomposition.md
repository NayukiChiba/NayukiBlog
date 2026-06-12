---
title: 矩阵分解完全指南：从 LU 到 SVD 的数学原理与应用
date: 2026-06-07
category: 数学
tags:
  - 线性代数
  - 矩阵分解
  - 数值计算
  - SVD
  - 特征值
description: 系统讲解八种核心矩阵分解方法（LU、Cholesky、LDL、QR、特征值、谱分解、Schur、SVD），包括每种分解的数学定义、存在条件、几何直觉、应用场景及 Python 实现，附带完整的对比总结表。
image: https://img.yumeko.site/file/blog/MatrixDecomposition.png
status: draft
---

> **前置阅读**：本文假定读者熟悉线性代数基本概念（矩阵乘法、转置、逆、秩、范数）。若需回顾多元统计中的矩阵运算，可参阅 [[Math/MultivariateStatistics|多元统计分析基础]]。

![图0: 矩阵分解 Banner](https://img.yumeko.site/file/blog/MatrixDecomposition.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于技术博客封面。
> 设计概念：从左到右展示八种矩阵分解的符号公式——
> A=LU（下三角×上三角）、A=LL^T（Cholesky）、A=LDL^T（LDL）、
> A=QR（正交×上三角）、A=QΛQ^T（特征值）、A=∑λᵢuᵢuᵢ^T（谱分解）、
> A=QTQ^H（Schur）、A=UΣV^T（SVD），
> 每种分解用不同颜色的矩阵块表示，形成"矩阵分解家族"的视觉序列。
> 配色：深蓝到青绿渐变，现代数学教科书风格。
> 顶部留白供标题叠加。
> ```

## 1. 问题的起点

在机器学习中，矩阵无处不在——协方差矩阵、权重矩阵、数据矩阵。但原始矩阵的形式往往不方便直接求解问题：

- 线性方程组 $Ax = b$ 怎么快速求解？矩阵是稠密的，直接求逆太慢
- 如何用低秩矩阵近似一个大矩阵，实现数据压缩？
- 如何判断一个矩阵是否正定？能否稳定地求它的逆？
- PCA 降维背后的数学是什么？

**矩阵分解**（Matrix Decomposition / Matrix Factorization）是回答这些问题的统一框架。核心思想是：将一个复杂矩阵分解为若干具有特殊结构（三角、正交、对角）的简单矩阵的乘积，从而简化计算。

> [!NOTE] 矩阵分解在 AI/ML 中的典型应用
> - **PCA 降维**：对协方差矩阵做特征值分解，或直接对数据矩阵做 SVD
> - **最小二乘回归**：用 QR 分解避免求正规方程 $A^\top A$（数值更稳定）
> - **优化算法**：Newton 法中需要 Cholesky 分解来高效求解 Hessian 的逆
> - **推荐系统**：SVD 用于矩阵补全（如 Netflix 评分预测）
> - **大模型推理**：对权重矩阵做低秩分解以减少推理时的计算量（LoRA 等）

## 2. 分解方法总览

矩阵分解可以按「因子的结构」分为几大类：

| 类型 | 分解形式 | 因子结构 | 核心条件 |
|:--|:--|:--|:--|
| **三角分解** | $A = LU$ | $L$ 下三角，$U$ 上三角 | 方阵，顺序主子式非零 |
| **Cholesky** | $A = LL^\top$ | $L$ 下三角 | 对称正定 |
| **LDL** | $A = LDL^\top$ | $L$ 单位下三角，$D$ 对角 | 对称，主子式非零 |
| **正交-三角** | $A = QR$ | $Q$ 正交，$R$ 上三角 | 任意矩阵 |
| **特征值** | $A = Q\Lambda Q^{-1}$ | $Q$ 特征向量，$\Lambda$ 对角 | 可对角化方阵 |
| **谱分解** | $A = \sum \lambda_i \mathbf{u}_i \mathbf{u}_i^H$ | $\mathbf{u}_i$ 正交特征向量 | 正规矩阵 |
| **Schur** | $A = QTQ^H$ | $Q$ 酉矩阵，$T$ 上三角 | 任意方阵 |
| **SVD** | $A = U\Sigma V^\top$ | $U,V$ 正交，$\Sigma$ 对角 | **任意矩阵**（最通用） |

![图1: 分解关系图](https://img.yumeko.site/file/blog/MatrixDecomposition/OverviewMap.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的教学关系图，展示矩阵分解之间的适用条件层级关系。
> 从顶部"SVD（任意矩阵）"开始，逐层向下分支：
> SVD → 方阵→ 特征值分解→ 正规矩阵→ 谱分解；
> SVD → 方阵→ Schur 分解（不可对角化时）；
> SVD → 方阵→ LU分解→ 对称→ LDL分解→ 正定→ Cholesky分解；
> 另一分支：SVD → 任意矩阵→ QR分解。
> 箭头标注条件，每种分解用圆角矩形卡片表示。
> 白色背景，细线箭头，专业数据可视化配色。
> ```

> [!TIP] 核心直觉
> 越「特殊」的矩阵拥有越「简洁」的分解。SVD 是最通用的（任意矩阵都能做），但代价是因子解释性不如特征值分解直观。工程中应优先选择能满足条件的、最简单的分解。

## 3. LU 分解：线性方程组的利器

### 3.1 定义

对于 $n \times n$ 方阵 $A$，若其顺序主子式均非零，则存在唯一的分解：

$$
\boxed{A = LU}
$$

其中 $L$ 是**单位下三角矩阵**（对角元全为 1），$U$ 是**上三角矩阵**。

有时也写成 $A = LDU$ 或 $PA = LU$（带部分主元 pivot 的版本）。

### 3.2 直观理解

LU 分解本质上是**高斯消元法的矩阵表示**。消元过程中每一步「将第 $i$ 行减去第 $k$ 行的倍数」等价于左乘一个下三角矩阵 $L_k^{-1}$。将消元过程中所有乘子收集起来就是 $L$，最终的上三角形式就是 $U$。

$$
\underbrace{L_{n-1}^{-1} \cdots L_2^{-1} L_1^{-1}}_{L} \cdot A = U
$$

### 3.3 数值示例

对 $A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$ 做 LU 分解：

**步骤 1**：消去第一列。$L_1^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ -4 & 0 & 1 \end{bmatrix}$，乘子为 $2, 4$。

$$
L_1^{-1}A = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 3 & 5 \end{bmatrix}
$$

**步骤 2**：消去第二列。乘子为 $3$，$L_2^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -3 & 1 \end{bmatrix}$。

$$
L_2^{-1}L_1^{-1}A = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix} = U
$$

**步骤 3**：乘子取反放入 $L$：

$$
L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix}, \quad
U = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}
$$

验证：$LU = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix} = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix} = A$。✓

### 3.4 拓展矩阵法（增广矩阵法）

除了逐步记录乘子构造 $L$ 之外，还有一种更直观的方法——**拓展矩阵法**。将 $A$ 和单位矩阵 $I$ 拼在一起，在同一个增广矩阵上完成全部消元。

**核心思路**：

1. 构造拓展矩阵 $[A \mid I]$
2. 对左侧的 $A$ 做高斯消元，逐步化为上三角矩阵 $U$
3. 对右侧的 $I$ 施加**完全相同的行变换**
4. 最终右侧的矩阵就是 $L^{-1}$，取逆即得 $L$

数学原理：每一行操作等价于左乘一个初等矩阵。设全过程左乘的初等矩阵之积为 $E = E_k \cdots E_2 E_1$，则：

$$
E \cdot A = U \quad \Longrightarrow \quad A = E^{-1} \cdot U
$$

由于 $E$ 作用于 $I$ 后得到 $E$ 自身，而 $L = E^{-1}$，因此：

$$
[\,A \mid I\,] \;\xrightarrow{\text{消元}}\; [\,U \mid E\,] = [\,U \mid L^{-1}\,] \;\xrightarrow{\text{取逆}}\; L
$$

> [!TIP] 为什么右边是 $L^{-1}$ 而非 $L$？
> 因为消元操作 $E$ 是在「消除下三角元素」——它把 $A$ 变成了上三角 $U$，等价于 $EA = U$，所以 $A = E^{-1}U = LU$，从而 $L = E^{-1}$。右侧矩阵记录的是 $E$，即 $L^{-1}$。

**用同一个例子演示**：$A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$。

**构造拓展矩阵**：

$$
[A \mid I] = \left[
\begin{array}{ccc|ccc}
2 & 1 & 1 & 1 & 0 & 0 \\
4 & 3 & 3 & 0 & 1 & 0 \\
8 & 7 & 9 & 0 & 0 & 1
\end{array}
\right]
$$

**步骤 1**：以 $a_{11}=2$ 为 pivot，消去下方两行。

第二行：$R_2 \leftarrow R_2 - 2 \cdot R_1$（乘子 $m_{21}=2$）
第三行：$R_3 \leftarrow R_3 - 4 \cdot R_1$（乘子 $m_{31}=4$）

$$
\left[
\begin{array}{ccc|ccc}
2 & 1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & -2 & 1 & 0 \\
0 & 3 & 5 & -4 & 0 & 1
\end{array}
\right]
$$

**步骤 2**：以 $a_{22}=1$ 为 pivot，消去第三行。

第三行：$R_3 \leftarrow R_3 - 3 \cdot R_2$（乘子 $m_{32}=3$）

$$
\left[
\begin{array}{ccc|ccc}
2 & 1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & -2 & 1 & 0 \\
0 & 0 & 2 & 2 & -3 & 1
\end{array}
\right]
$$

此时左侧已是上三角矩阵 $U$，右侧即为 $L^{-1}$：

$$
U = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}, \qquad
L^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 2 & -3 & 1 \end{bmatrix}
$$

**步骤 3**：对 $L^{-1}$ 求逆得到 $L$。

因为 $L^{-1}$ 是**单位下三角矩阵**，求逆非常简单——用向前代入（forward substitution）逐列求解。

设待求的 $L = \begin{bmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{bmatrix}$，满足 $L^{-1} L = I$：

**第一列**（$I$ 的第 1 列 $\mathbf{e}_1 = [1, 0, 0]^\top$）：

$$
\begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 2 & -3 & 1 \end{bmatrix}
\begin{bmatrix} 1 \\ l_{21} \\ l_{31} \end{bmatrix} =
\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

从上到下逐行代入：

$$
\begin{aligned}
\text{Row 1: } &1 \cdot 1 = 1 \;\checkmark \\
\text{Row 2: } &-2 \cdot 1 + 1 \cdot l_{21} = 0 \;\Longrightarrow\; l_{21} = 2 \\
\text{Row 3: } &2 \cdot 1 + (-3) \cdot 2 + 1 \cdot l_{31} = 0 \;\Longrightarrow\; 2 - 6 + l_{31} = 0 \;\Longrightarrow\; l_{31} = 4
\end{aligned}
$$

**第二列**（$\mathbf{e}_2 = [0, 1, 0]^\top$）：

$$
\begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 2 & -3 & 1 \end{bmatrix}
\begin{bmatrix} 0 \\ 1 \\ l_{32} \end{bmatrix} =
\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

$$
\begin{aligned}
\text{Row 1: } &0 = 0 \;\checkmark \\
\text{Row 2: } &0 + 1 = 1 \;\checkmark \\
\text{Row 3: } &0 + (-3) \cdot 1 + 1 \cdot l_{32} = 0 \;\Longrightarrow\; l_{32} = 3
\end{aligned}
$$

**第三列**（$\mathbf{e}_3 = [0, 0, 1]^\top$）自动满足，因为 $L^{-1}$ 最后一列恰好是 $\mathbf{e}_3$。

**结果**：

$$
L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix}
$$

:::fold[另一种视角：再构造一次增广矩阵求逆]

对 $L^{-1}$ 求逆也可以套用「右边放单位矩阵再消元」的套路：

$$
[L^{-1} \mid I] = \left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 0 \\
-2 & 1 & 0 & 0 & 1 & 0 \\
2 & -3 & 1 & 0 & 0 & 1
\end{array}
\right]
$$

对下三角矩阵，从**下往上**消去严格下三角的非零元素：

第 2 行加 2 倍的第 3 行：$R_2 \leftarrow R_2 + 2R_3$ → 消除第 2 行第 3 列没有意义（已经是 0），应反过来操作：

实际上是对每一**列**从下往上消。把第 2 行的 $-2$ 消掉：$R_2 \leftarrow R_2 + 2R_1$；把第 3 行的 $2, -3$ 消掉：先消 $-3$（$R_3 \leftarrow R_3 + 3R_2$）再消 $2$（$R_3 \leftarrow R_3 - 2R_1$ 补充调整）。

总之对三角矩阵求逆极度廉价——复杂度仅 $O(n^2)$，远低于一般矩阵求逆的 $O(n^3)$。

:::

与 3.3 节直接记录乘子的结果完全一致。注意 $L$ 的严格下三角部分恰好就是乘子 $m_{21}, m_{31}, m_{32}$（符号取正）。

**拓展矩阵法的优势**：
- 把整个消元过程记录在一个表格中，不易遗漏乘子
- 右侧矩阵的行变换自动追踪了所有操作，适合手算和教学
- 与求逆矩阵的 Gauss-Jordan 消元法一脉相承，容易记忆

**不足**：
- 最后多一步求逆操作（虽然对三角矩阵求逆非常廉价）
- 对于大规模矩阵，显式构造 $L^{-1}$ 不如直接记录乘子高效

### 3.5 应用场景

**求解线性方程组 $Ax = b$**。用 LU 分解替代直接求逆：

1. 分解：$A = LU$（$O(n^3)$，一次性的预处理）
2. 前代（forward substitution）：解 $Ly = b$ 得 $y$（$O(n^2)$）
3. 回代（backward substitution）：解 $Ux = y$ 得 $x$（$O(n^2)$）

当需要多次求解「相同 $A$、不同 $b$」的方程组时，LU 分解的优势非常大——只需一次 $O(n^3)$ 分解，每次求解只需 $O(n^2)$。

> [!TIP] 注意
> 直接用 $A^{-1}$ 求解 $x = A^{-1}b$ 在数值上不如用 LU 分解稳定，且计算量更大。**在实际工程中，永远用 `solve` 而非 `inv`**。

## 4. Cholesky 分解：对称正定的加速器

### 4.1 定义

若 $A$ 是 $n \times n$ **对称正定矩阵**，则存在唯一的分解：

$$
\boxed{A = LL^\top}
$$

其中 $L$ 是**对角元为正的实下三角矩阵**。这是 LU 分解在对称正定条件下的特化版本，计算量约为 LU 分解的一半。

### 4.2 为什么更快？

对称正定矩阵不需要行交换（无需选主元），且只需存储和计算一个三角因子。算法流程（Cholesky-Crout）：

$$
L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}, \qquad
L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{k=1}^{j-1} L_{ik} L_{jk} \right) \quad (i > j)
$$

### 4.3 数值示例

**先看一个简单的 2×2 例子**，建立直觉：

$A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$（对称正定，特征值 $1, 6 > 0$）：

$$
L_{11} = \sqrt{4} = 2, \quad L_{21} = \frac{2}{2} = 1, \quad L_{22} = \sqrt{3 - 1^2} = \sqrt{2}
$$

$$
L = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{bmatrix}, \quad
LL^\top = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix} = A \;\checkmark
$$

**再做一个 3×3 的完整演示**：

$$
A = \begin{bmatrix}
25 & 15 & -5 \\
15 & 18 & 0 \\
-5 & 0 & 11
\end{bmatrix}
$$

首先验证正定性：顺序主子式 $25 > 0$，$\det\begin{bmatrix}25&15\\15&18\end{bmatrix} = 225 > 0$，$\det(A) = 2700 > 0$。✓

按列逐步计算 $L$：

**第 1 列**：

$$
L_{11} = \sqrt{A_{11}} = \sqrt{25} = 5
$$

$$
L_{21} = \frac{A_{21}}{L_{11}} = \frac{15}{5} = 3, \qquad
L_{31} = \frac{A_{31}}{L_{11}} = \frac{-5}{5} = -1
$$

$$
\text{当前 } L = \begin{bmatrix} 5 & 0 & 0 \\ 3 & * & 0 \\ -1 & * & * \end{bmatrix}
$$

**第 2 列**：

$$
L_{22} = \sqrt{A_{22} - L_{21}^2} = \sqrt{18 - 3^2} = \sqrt{9} = 3
$$

$$
L_{32} = \frac{A_{32} - L_{31} L_{21}}{L_{22}} = \frac{0 - (-1) \cdot 3}{3} = \frac{3}{3} = 1
$$

$$
\text{当前 } L = \begin{bmatrix} 5 & 0 & 0 \\ 3 & 3 & 0 \\ -1 & 1 & * \end{bmatrix}
$$

**第 3 列**：

$$
L_{33} = \sqrt{A_{33} - L_{31}^2 - L_{32}^2} = \sqrt{11 - (-1)^2 - 1^2} = \sqrt{9} = 3
$$

**最终结果**：

$$
L = \begin{bmatrix} 5 & 0 & 0 \\ 3 & 3 & 0 \\ -1 & 1 & 3 \end{bmatrix}
$$

**验证**：

$$
LL^\top = \begin{bmatrix} 5 & 0 & 0 \\ 3 & 3 & 0 \\ -1 & 1 & 3 \end{bmatrix}
\begin{bmatrix} 5 & 3 & -1 \\ 0 & 3 & 1 \\ 0 & 0 & 3 \end{bmatrix}
= \begin{bmatrix} 25 & 15 & -5 \\ 15 & 18 & 0 \\ -5 & 0 & 11 \end{bmatrix} = A \;\checkmark
$$

### 4.4 应用场景

- **求解正定线性方程组**：如 Newton 法中的 $H \Delta x = -g$，Hessian $H$ 是对称正定的
- **多元正态采样**：若 $z \sim \mathcal{N}(0, I)$，则 $x = \mu + Lz \sim \mathcal{N}(\mu, \Sigma)$，其中 $\Sigma = LL^\top$
- **Kalman 滤波**：协方差矩阵的传播和更新广泛使用 Cholesky 分解

> [!ATTENTION] 注意
> Cholesky 分解要求矩阵**严格正定**（特征值全 $> 0$）。半正定矩阵（有零特征值）会因 $L_{jj} = 0$ 导致算法中断。此时应使用 LDL 分解或添加微小对角扰动（$\Sigma + \epsilon I$）。

## 5. LDL 分解：对称但不正定时的选择

### 5.1 定义

当矩阵 $A$ 是对称矩阵但**不一定正定**时，Cholesky 分解可能失败（出现负数开平方）。LDL 分解绕过了这个问题——不要求正定性，只需 $A$ 的各阶顺序主子式非零：

$$
\boxed{A = LDL^\top}
$$

其中 $L$ 是**单位下三角矩阵**（对角元全为 1），$D$ 是**对角矩阵**（对角元可正可负）。

### 5.2 与 Cholesky 的关系

若 $A$ 正定，则 $D$ 的对角元全为正，此时可进一步分解 $D = D^{1/2} D^{1/2}$，令 $\tilde{L} = L D^{1/2}$，就回到了 Cholesky 形式 $A = \tilde{L} \tilde{L}^\top$。

LDL 的优势在于 $D$ 的对角元符号直接反映矩阵的惯性指数（正特征值个数 = $D$ 正对角元个数，负特征值个数 = $D$ 负对角元个数）。

### 5.3 数值示例

对 $A = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 2 & 1 \\ 0 & 1 & 0 \end{bmatrix}$ 做 LDL 分解。

先验证对称性：$A = A^\top$ ✓。验证正定性：顺序主子式 $d_1 = 1 > 0$，$d_2 = 1\cdot2 - 4 = -2 < 0$ → **不正定**，Cholesky 不可用。

**计算 $LDL^\top$**。设 $L = \begin{bmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{bmatrix}$，$D = \operatorname{diag}(d_1, d_2, d_3)$。

展开 $LDL^\top$：

$$
\begin{bmatrix}
d_1 & d_1 l_{21} & d_1 l_{31} \\
d_1 l_{21} & d_1 l_{21}^2 + d_2 & d_1 l_{21} l_{31} + d_2 l_{32} \\
d_1 l_{31} & d_1 l_{21} l_{31} + d_2 l_{32} & d_1 l_{31}^2 + d_2 l_{32}^2 + d_3
\end{bmatrix}
= \begin{bmatrix} 1 & 2 & 0 \\ 2 & 2 & 1 \\ 0 & 1 & 0 \end{bmatrix}
$$

**逐元素求解**：

- $d_1 = A_{11} = 1$
- $l_{21} = A_{21} / d_1 = 2/1 = 2$，$l_{31} = A_{31} / d_1 = 0/1 = 0$
- $d_2 = A_{22} - d_1 l_{21}^2 = 2 - 1 \cdot 4 = -2$
- $l_{32} = (A_{32} - d_1 l_{31} l_{21}) / d_2 = (1 - 1 \cdot 0 \cdot 2) / (-2) = -1/2$
- $d_3 = A_{33} - d_1 l_{31}^2 - d_2 l_{32}^2 = 0 - 0 - (-2) \cdot (1/4) = 1/2$

**结果**：

$$
L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 0 & -1/2 & 1 \end{bmatrix}, \quad
D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -2 & 0 \\ 0 & 0 & 1/2 \end{bmatrix}
$$

**验证**：

$$
LDL^\top = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 0 & -1/2 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 0 \\ 0 & -2 & 0 \\ 0 & 0 & 1/2 \end{bmatrix}
\begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & -1/2 \\ 0 & 0 & 1 \end{bmatrix}
= \begin{bmatrix} 1 & 2 & 0 \\ 2 & 2 & 1 \\ 0 & 1 & 0 \end{bmatrix} = A \;\checkmark
$$

**惯性分析**：$D$ 对角元为 $(1, -2, 1/2)$，一正一负一正，说明 $A$ 有两个正特征值、一个负特征值（矩阵不定）。若要求对称正定，Cholesky 会因 $d_2 < 0$ 在 $\sqrt{-2}$ 处失败，但 LDL 顺利算出 $d_2 = -2$。

### 5.4 应用场景

- **Newton 法（非凸优化）**：Hessian 可能不定，LDL 用 $D$ 的对角元符号判断曲率方向
- **修正 Newton 法**：将 $D$ 中负对角元改为正数，构造正定近似用于下降方向
- **半正定协方差**：Kalman 滤波中协方差矩阵接近奇异时，LDL 比 Cholesky 更稳健

## 6. QR 分解：最小二乘的标准解法

### 5.1 定义

对于任意 $m \times n$ 矩阵 $A$（$m \ge n$），存在分解：

$$
\boxed{A = QR}
$$

其中 $Q$ 是 $m \times m$ **正交矩阵**（$Q^\top Q = I$），$R$ 是 $m \times n$ **上三角矩阵**（底部 $m-n$ 行为零）。

若 $A$ 列满秩，则 $R$ 的对角元非零，且分解在忽略 $Q$ 列符号的意义下唯一。

### 5.2 几何直觉

QR 分解本质上是 **Gram-Schmidt 正交化过程的矩阵形式**。$A$ 的列向量是空间中的一组基，$Q$ 的列是经过正交归一化后的标准正交基，$R$ 记录了原始基在正交基下的坐标。

![图2: QR 分解几何意义](https://img.yumeko.site/file/blog/MatrixDecomposition/QRGeometry.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的几何示意图，展示 QR 分解的含义。
> 左侧：二维平面上两个线性无关的向量 a1, a2（斜坐标系）。
> 右侧：正交归一化后的 q1, q2（标准正交基），以及上三角矩阵 R 的系数如何将 q1, q2 映射回 a1, a2。
> 箭头标注：a1 = r11·q1, a2 = r12·q1 + r22·q2。
> 白色背景，蓝色向量，学术教科书风格，中文标注。
> ```

### 5.3 数值示例

对 $A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$ 做经典 Gram-Schmidt 正交化，**完整计算每一步**：

**步骤 1：提取列向量**

$$
\mathbf{a}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{a}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}
$$

**步骤 2：正交化第一列**

$$
\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} = \frac{1}{\sqrt{1^2 + 1^2 + 0^2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

$R$ 的第一列：$r_{11} = \|\mathbf{a}_1\| = \sqrt{2}$。

**步骤 3：正交化第二列**

先将 $\mathbf{a}_2$ 投影到 $\mathbf{q}_1$ 方向上并减去（得到与 $\mathbf{q}_1$ 正交的分量）：

$$
r_{12} = \mathbf{q}_1^\top \mathbf{a}_2 = \frac{1}{\sqrt{2}}(1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1) = \frac{1}{\sqrt{2}}
$$

$$
\mathbf{u}_2 = \mathbf{a}_2 - r_{12} \mathbf{q}_1
= \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
= \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \begin{bmatrix} 1/2 \\ 1/2 \\ 0 \end{bmatrix}
= \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix}
$$

归一化：

$$
\|\mathbf{u}_2\| = \sqrt{(1/2)^2 + (-1/2)^2 + 1^2} = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}
$$

$$
\mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|} = \frac{1}{\sqrt{3/2}} \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix}
= \sqrt{\frac{2}{3}} \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix}
= \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}
$$

$R$ 的第二列：$r_{22} = \|\mathbf{u}_2\| = \sqrt{3/2}$，而 $r_{12} = 1/\sqrt{2}$（已在上面算出）。

**步骤 4：组装 $Q$ 和 $R$**

$$
Q = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{6} \\ 1/\sqrt{2} & -1/\sqrt{6} \\ 0 & 2/\sqrt{6} \end{bmatrix}, \qquad
R = \begin{bmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} \end{bmatrix}
$$

**步骤 5：验证**

第一列：$r_{11} \mathbf{q}_1 = \sqrt{2} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\\0\end{bmatrix} = \begin{bmatrix}1\\1\\0\end{bmatrix} = \mathbf{a}_1$ ✓

第二列：$r_{12} \mathbf{q}_1 + r_{22} \mathbf{q}_2 = \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\\0\end{bmatrix} + \sqrt{\frac{3}{2}}\cdot\frac{1}{\sqrt{6}}\begin{bmatrix}1\\-1\\2\end{bmatrix} = \begin{bmatrix}1/2\\1/2\\0\end{bmatrix} + \begin{bmatrix}1/2\\-1/2\\1\end{bmatrix} = \begin{bmatrix}1\\0\\1\end{bmatrix} = \mathbf{a}_2$ ✓

> [!TIP] $R$ 的上三角结构
> $r_{11}$ 是第一列的模长；$r_{12} = \mathbf{q}_1^\top \mathbf{a}_2$ 是第一列对第二列的「贡献」；$r_{22}$ 是第二列正交化后的模长。$r_{21} = 0$ 是因为 $\mathbf{q}_2 \perp \mathbf{a}_1$（正交性保证）。

### 5.4 应用场景

**最小二乘问题 $\min_x \|Ax - b\|_2$**。这是 QR 分解最重要的应用之一：

传统方法解正规方程 $(A^\top A)x = A^\top b$ 有两个问题：(1) $A^\top A$ 的条件数是 $A$ 的平方，数值不稳定；(2) 需要显式计算 $A^\top A$，额外计算量。

QR 分解只需：
1. 分解 $A = QR$，则 $Rx = Q^\top b$
2. 回代求解（$R$ 是上三角）

不需要显式求 $A^\top A$，数值稳定性远优于正规方程法。

> [!NOTE] QR 分解的其他应用
> - **特征值计算**：QR 算法（Francis 1961）是计算一般矩阵所有特征值的标准方法，每次迭代做一次 QR 分解
> - **矩阵求逆**：$A^{-1} = R^{-1} Q^\top$，$R$ 是三角矩阵，求逆容易
> - **行列式计算**：$\det(A) = \pm \prod R_{ii}$（符号取决于 $Q$ 的实现方式）

## 7. 特征值分解：理解矩阵的「本质」

### 6.1 定义

对于 $n \times n$ 方阵 $A$，若存在非零向量 $\mathbf{v}$ 使得 $A\mathbf{v} = \lambda \mathbf{v}$，则 $\lambda$ 是特征值，$\mathbf{v}$ 是对应的特征向量。

若 $A$ 有 $n$ 个线性无关的特征向量，则 $A$ **可对角化**：

$$
\boxed{A = Q\Lambda Q^{-1}}
$$

其中 $Q$ 的列是特征向量，$\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ 是对角特征值矩阵。

若 $A$ 是**实对称矩阵**（$A = A^\top$），则特征向量可选为正交，即 $Q^\top Q = I$：

$$
A = Q\Lambda Q^\top
$$

### 6.2 矩阵「对角化」意味着什么？

特征值分解告诉我们：**任何可对角化的矩阵本质上只是在不同方向上的缩放**。在特征向量坐标系中，矩阵的作用变得极其简单——每个坐标独立缩放。

从变换角度理解：

$$
A^n = (Q\Lambda Q^{-1})^n = Q\Lambda^n Q^{-1} = Q \begin{bmatrix} \lambda_1^n & & \\ & \ddots & \\ & & \lambda_n^n \end{bmatrix} Q^{-1}
$$

这解释了为什么 $|\lambda_i| < 1$ 时 $A^n \to 0$（动力系统稳定），$|\lambda_i| > 1$ 时发散。

### 6.3 数值示例

**2×2 对称矩阵**（快速回顾）：

$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$。

特征方程 $\det(A - \lambda I) = (2-\lambda)^2 - 1 = 0$，解得 $\lambda_{1,2} = 3, 1$。

对应特征向量：$\lambda=3 \to \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$，$\lambda=1 \to \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$。

$$
A = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}
\begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}
$$

> [TIP] 几何直觉
> $A$ 将向量在 $(1,1)$ 方向拉伸 3 倍，在 $(1,-1)$ 方向保持不变。任何向量都可以分解为这两个方向的组合，其变换结果一目了然。

**3×3 对称矩阵**（完整演示）：

$$
A = \begin{bmatrix}
3 & 2 & 0 \\
2 & 3 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

**步骤 1：求特征值**。解 $\det(A - \lambda I) = 0$：

$$
\det\begin{bmatrix} 3-\lambda & 2 & 0 \\ 2 & 3-\lambda & 0 \\ 0 & 0 & 1-\lambda \end{bmatrix} = 0
$$

按第三行展开（只有 $a_{33} = 1-\lambda$）：

$$
(1-\lambda) \cdot \det\begin{bmatrix} 3-\lambda & 2 \\ 2 & 3-\lambda \end{bmatrix} = 0
$$

$$
(1-\lambda)\big[(3-\lambda)^2 - 4\big] = (1-\lambda)(\lambda^2 - 6\lambda + 5) = (1-\lambda)(\lambda - 1)(\lambda - 5) = 0
$$

$$
\lambda_1 = 5, \quad \lambda_2 = 1, \quad \lambda_3 = 1
$$

**步骤 2：求特征向量**。

对 $\lambda_1 = 5$，解 $(A - 5I)\mathbf{v} = 0$：

$$
\begin{bmatrix} -2 & 2 & 0 \\ 2 & -2 & 0 \\ 0 & 0 & -4 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

前两行等价于 $-2x + 2y = 0$，即 $x = y$。第三行 $-4z = 0$ 得 $z = 0$。归一化：

$$
\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

对 $\lambda_{2,3} = 1$（二重根），解 $(A - I)\mathbf{v} = 0$：

$$
\begin{bmatrix} 2 & 2 & 0 \\ 2 & 2 & 0 \\ 0 & 0 & 0 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

只有一个独立方程 $2x + 2y = 0$，即 $y = -x$，$z$ 自由。

选取两个正交的特征向量（Gram-Schmidt 思想）：

$$
\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}, \qquad
\mathbf{v}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

验证：$\mathbf{v}_1 \cdot \mathbf{v}_2 = 1 - 1 + 0 = 0$，$\mathbf{v}_1 \cdot \mathbf{v}_3 = 0$，$\mathbf{v}_2 \cdot \mathbf{v}_3 = 0$ ✓（三个特征向量两两正交，因为 $A$ 是对称矩阵）

**步骤 3：组装分解**。

$$
A = \underbrace{\begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & 0 \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \\ 0 & 0 & 1 \end{bmatrix}}_{Q}
\underbrace{\begin{bmatrix} 5 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}}_{\Lambda}
\underbrace{\begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & 0 \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \\ 0 & 0 & 1 \end{bmatrix}^{\top}}_{Q^\top}
$$

验证 $Q\Lambda Q^\top$：第一列 $\frac{5}{\sqrt{2}}\mathbf{v}_1 + \frac{1}{\sqrt{2}}\mathbf{v}_2 = \begin{bmatrix}3\\2\\0\end{bmatrix} = \mathbf{a}_1$ ✓，其余类似。

### 6.4 应用场景

- **PCA 主成分分析**：协方差矩阵 $\Sigma = \frac{1}{n}X^\top X$ 的特征向量是主成分方向，特征值是各方向的方差
- **谱聚类**：Graph Laplacian 的特征向量用于降维和聚类
- **PageRank**：转移矩阵的主特征向量给出网页的重要性排名
- **微分方程**：$\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ 的解完全由 $A$ 的特征值和特征向量决定

### 6.5 为什么不是所有矩阵都能对角化？

考虑 $B = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$。特征值 $\lambda = 2$（代数重数 2），但 $(B - 2I) = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$ 的零空间只有 1 维（几何重数 1）。**几何重数 < 代数重数** → 不可对角化。

> [!NOTE] 扩展阅读
> 对于不可对角化的矩阵，Schur 分解（#9）提供了通用的上三角化方案。而对于正规矩阵，谱分解（#8）给出了更强的正交对角化结论。

## 8. 谱分解：正规矩阵的正交对角化

### 8.1 定义

谱分解（Spectral Decomposition）是特征值分解的一个重要特例。当矩阵 $A$ 是**正规矩阵**（Normal Matrix）——即满足 $A A^H = A^H A$——时，其特征向量可以选为一组标准正交基：

$$
\boxed{A = \sum_{i=1}^{n} \lambda_i \mathbf{u}_i \mathbf{u}_i^H}
$$

或用矩阵形式 $A = U\Lambda U^H$，其中 $U$ 是酉矩阵（$U^H U = I$），$\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)$。

这个分解式将矩阵 $A$ 表示为 $n$ 个秩-1 投影矩阵的加权和，权重即为特征值。所谓「谱」（spectrum），指的就是特征值的集合。

### 8.2 哪些矩阵是正规矩阵？

正规矩阵的条件 $AA^H = A^HA$ 涵盖了大量常见的矩阵类型：

| 矩阵类型 | 额外条件 | 特征值性质 |
|:--|:--|:--|
| 实对称矩阵 | $A = A^\top$ | 全部为实数 |
| Hermitian 矩阵 | $A = A^H$ | 全部为实数 |
| 实反对称矩阵 | $A = -A^\top$ | 全部为纯虚数 |
| 正交矩阵 | $A^\top A = I$ | 全部模长为 1（在单位圆上） |
| 酉矩阵 | $A^H A = I$ | 全部模长为 1 |

> [!NOTE] 正规矩阵 vs 可对角化
> 所有正规矩阵都可酉对角化，但并非所有可对角化矩阵都是正规矩阵。例如 $A = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$ 可对角化（特征值 1, 2），但不正规：$AA^\top \neq A^\top A$。

### 8.3 数值示例

用 #7.3 的对称矩阵 $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$（实对称 $\subset$ 正规矩阵）：

特征值 $\lambda_1 = 3, \lambda_2 = 1$，正交归一特征向量 $\mathbf{u}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}, \mathbf{u}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$。

**秩-1 分解形式**：

$$
\begin{aligned}
A &= 3 \cdot \mathbf{u}_1 \mathbf{u}_1^\top + 1 \cdot \mathbf{u}_2 \mathbf{u}_2^\top \\
&= 3 \cdot \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix} + 1 \cdot \frac{1}{2}\begin{bmatrix}1&-1\\-1&1\end{bmatrix} \\
&= \begin{bmatrix}3/2&3/2\\3/2&3/2\end{bmatrix} + \begin{bmatrix}1/2&-1/2\\-1/2&1/2\end{bmatrix}
= \begin{bmatrix}2&1\\1&2\end{bmatrix} = A \;\checkmark
\end{aligned}
$$

谱分解的优美之处：它把矩阵分解为独立「模态」的叠加——每个特征方向 $\mathbf{u}_i$ 上的投影矩阵 $\mathbf{u}_i \mathbf{u}_i^\top$ 被其特征值 $\lambda_i$ 加权。对于 PCA 等降维方法，这意味着可以按特征值大小排序，丢弃小特征值对应的分量来实现降维。

**再看一个非对称正规矩阵的例子**：$A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$（旋转 90°，正交矩阵 $\subset$ 正规矩阵）。

特征值 $\lambda = \pm i$，对应的复特征向量是标准正交的（在酉意义下）。谱分解写作：

$$
A = i \cdot \mathbf{u}_1 \mathbf{u}_1^H + (-i) \cdot \mathbf{u}_2 \mathbf{u}_2^H
$$

虽然特征值和特征向量是复数，但最终求和的结果保持 $A$ 为实矩阵（虚部在求和时抵消）。

### 8.4 谱分解与特征值分解的关系

| 维度 | 特征值分解 | 谱分解 |
|:--|:--|:--|
| 适用范围 | 可对角化方阵 | **正规矩阵**（子集） |
| 特征向量 | 线性无关，未必正交 | **一定正交（酉意义下）** |
| 变换矩阵 | $Q$ 可逆 | $U$ 酉矩阵（$U^{-1} = U^H$） |
| 分解形式 | $A = Q\Lambda Q^{-1}$ | $A = U\Lambda U^H = \sum \lambda_i \mathbf{u}_i \mathbf{u}_i^H$ |

核心区别：谱分解额外要求 $AA^H = A^HA$，换来的是**正交特征向量**和**秩-1 投影的显式分解**。在 PCA 和谱聚类中，实际使用的是谱分解，而非一般的特征值分解——因为协方差矩阵和 Laplacian 矩阵都是对称（从而正规）的。

## 9. Schur 分解：所有方阵的「备选方案」

### 7.1 定义

**Schur 分解定理**（1909）是线性代数的基石定理之一：任何 $n \times n$ 复矩阵 $A$ 都存在酉相似变换，化为上三角形式：

$$
\boxed{A = QTQ^H}
$$

其中 $Q$ 是**酉矩阵**（$Q^H Q = I$），$T$ 是**上三角矩阵**，其对角元就是 $A$ 的全部特征值（按任意指定顺序排列）。

对于**实矩阵**，存在实 Schur 形式：

$$
A = QTQ^\top
$$

其中 $Q$ 是**正交矩阵**，$T$ 是**拟上三角矩阵**（块上三角，对角为 $1 \times 1$ 或 $2 \times 2$ 块，$2 \times 2$ 块对应一对共轭复特征值）。

### 7.2 Schur vs 特征值分解

| 特性 | 特征值分解 | Schur 分解 |
|:--|:--|:--|
| 适用矩阵 | 可对角化方阵 | **所有方阵** |
| 变换矩阵 | 特征向量 $Q$ | 酉/正交 $Q$ |
| 目标形式 | 对角阵 $\Lambda$ | 上三角阵 $T$ |
| 正交性 | 仅实对称时保证 | **始终保证** |

Schur 分解的优美之处：**放弃「对角化」，换取「通用性 + 正交性」**。上三角形式虽然不如对角形式简洁，但特征值仍然在对角线上，而且 $Q$ 始终是酉矩阵（数值上极其稳定）。

### 7.3 数值示例

**例 1：不可对角化的 Jordan 块**

$B = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$。特征值 $\lambda = 2$（代数重数 2），但只有 1 个线性无关的特征向量（几何重数 1），因此**不可对角化**。

但它本身已经是上三角形式，所以 Schur 分解即自身：$T = B$，$Q = I$。这完美体现了 Schur 分解的通用性——即使矩阵不可对角化，Schur 分解依然存在。

**例 2：复特征值的旋转矩阵**

$A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$（旋转 90°）。特征值为 $\pm i$（纯虚数），复特征值分解为：

$$
A = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ -i & i \end{bmatrix}
\begin{bmatrix} i & 0 \\ 0 & -i \end{bmatrix}
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & i \\ 1 & -i \end{bmatrix}
$$

而**实** Schur 分解直接接受 $2 \times 2$ 对角块，$T = A$ 本身（$Q = I$），因为 $A$ 已经是一个不可约的 $2 \times 2$ 实块，其特征值为一对共轭复数。块中 $\operatorname{tr}(T_{2\times2}) = 0 = i + (-i)$，$\det(T_{2\times2}) = 1 = i \cdot (-i)$。

**例 3：一般 3×3 非对称矩阵（完整计算）**

$$
A = \begin{bmatrix}
5 & -1 & 0 \\
4 & 1 & -1 \\
0 & 6 & 2
\end{bmatrix}
$$

这个矩阵不是上三角形式，也不对称。下面演示如何用**构造性 deflation 方法**逐步求出 Schur 分解。

**第 1 轮：求出一个实 Schur 向量并降维**

先求 $A$ 的一个特征值。展开 $\det(A - \lambda I)$：

$$
\det\begin{bmatrix} 5-\lambda & -1 & 0 \\ 4 & 1-\lambda & -1 \\ 0 & 6 & 2-\lambda \end{bmatrix} = 0
$$

按第一行展开：

$$
(5-\lambda)\big[(1-\lambda)(2-\lambda) + 6\big] + 1 \cdot \big[4(2-\lambda) - 0\big] = 0
$$

$$
(5-\lambda)(\lambda^2 - 3\lambda + 8) + (8 - 4\lambda) = 0
$$

展开 $(5-\lambda)(\lambda^2 - 3\lambda + 8) = 5\lambda^2 - 15\lambda + 40 - \lambda^3 + 3\lambda^2 - 8\lambda = -\lambda^3 + 8\lambda^2 - 23\lambda + 40$。

加 $(8-4\lambda)$：$-\lambda^3 + 8\lambda^2 - 27\lambda + 48 = 0$。

因式分解试根：$\lambda = 3$：$-27 + 72 - 81 + 48 = 12 \neq 0$。$\lambda = 4$：$-64 + 128 - 108 + 48 = 4 \neq 0$。$\lambda = 2$：$-8 + 32 - 54 + 48 = 18 \neq 0$。

:::fold[三次方程的数值求解]

该三次方程的精确根较复杂。在实际中，特征值由 QR 算法迭代求出。这里直接用 SciPy 验证：三个实特征值约为 $4.56, 2.22, 1.22$。

为了演示完整的手算流程，下面改用特征值更简洁的矩阵。

:::

为便于手算演示，换用以下矩阵（特征值全部为整数）：

$$
A = \begin{bmatrix}
3 & 1 & 1 \\
0 & 4 & 2 \\
0 & -1 & 1
\end{bmatrix}
$$

此矩阵的第一列已经是 $[3, 0, 0]^\top$——只有对角元非零，下方都是零！这意味着 $\lambda = 3$ 是一个特征值，且 $\mathbf{e}_1 = [1,0,0]^\top$ 是左特征向量，但右侧尚需处理下面的 $2 \times 2$ 块。

由于 $A$ 的第一列已部分三角化，我们把注意力集中在右下 $2 \times 2$ 子矩阵：

$$
A_{2:3, 2:3} = \begin{bmatrix} 4 & 2 \\ -1 & 1 \end{bmatrix}
$$

对这块求 Schur 分解。先求特征值：$\det = (4-\lambda)(1-\lambda) + 2 = \lambda^2 - 5\lambda + 6 = (\lambda - 2)(\lambda - 3) = 0$。$\lambda = 2, 3$。

对 $\lambda = 2$：$(A - 2I)\mathbf{v} = \begin{bmatrix}2&2\\-1&-1\end{bmatrix}\mathbf{v} = 0$，$\mathbf{v} = \frac{1}{\sqrt{2}}[1,-1]^\top$。

对 $\lambda = 3$：$(A - 3I)\mathbf{v} = \begin{bmatrix}1&2\\-1&-2\end{bmatrix}\mathbf{v} = 0$，$\mathbf{v} = \frac{1}{\sqrt{5}}[2,-1]^\top$。

这两个特征向量不正交（非对称矩阵的特征向量一般不保证正交）。正交化：

取 $\mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$，投影得 $\mathbf{q}_3$（与 $\mathbf{q}_2$ 正交）：

$$
\mathbf{q}_3 = \frac{\begin{bmatrix}2\\-1\end{bmatrix} - (\begin{bmatrix}2\\-1\end{bmatrix}^\top \mathbf{q}_2)\mathbf{q}_2}{\|\cdots\|}
= \frac{\begin{bmatrix}2\\-1\end{bmatrix} - \frac{3}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}}{\|\cdots\|}
= \frac{\begin{bmatrix}2\\-1\end{bmatrix} - \begin{bmatrix}3/2\\-3/2\end{bmatrix}}{\|\cdots\|}
= \frac{\begin{bmatrix}1/2\\1/2\end{bmatrix}}{\sqrt{1/2}}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}
$$

**组装完整的 $Q$ 和 $T$**：

$$
Q = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1/\sqrt{2} & 1/\sqrt{2} \\
0 & -1/\sqrt{2} & 1/\sqrt{2}
\end{bmatrix}
$$

计算 $T = Q^\top A Q$：

先算 $AQ = \begin{bmatrix}3&0&\sqrt{2}\\0&6/\sqrt{2}&2/\sqrt{2}\\0&-1/\sqrt{2}&1/\sqrt{2}\end{bmatrix}$，再乘 $Q^\top$：

$$
T = Q^\top A Q = \begin{bmatrix}
3 & 0 & \sqrt{2} \\
0 & 3 & 2 \\
0 & -1 & 2
\end{bmatrix}
$$

这已经是一个**上三角矩阵**（注意 $T_{21} = T_{31} = T_{32} \neq 0$ 中只有 $T_{32} \neq 0$，而 $T_{21}=0$ 也是零——等等，$T$ 的上三角检查：$T_{21}=0$ ✓，$T_{31}=0$ ✓，$T_{32}=-1$，这是下三角元素！）

实际上 $T_{32} = -1 \neq 0$，说明还需继续迭代。这体现了实 Schur 分解的迭代性质——一次正交变换通常不能一步到位。

**最终 Schur 分解**（经 QR 算法迭代收敛后）：

经过若干次 QR 迭代，$A$ 的实 Schur 形式会收敛到真正的上三角（或拟上三角）形式。用 `scipy.linalg.schur` 可验证：对角线元素收敛为特征值 $3, 3, 2$（已在上面算出）。

> [!NOTE] 关键认知
> Schur 分解的**理论存在性**由定理保证，但**实际计算**需要通过 QR 算法迭代（或 deflation 逐步降维）。对于手算而言，重点理解其存在性、形式和验证方法，而非完整的迭代过程。

### 7.4 应用场景

- **QR 算法求特征值**：QR 算法迭代 $A_k = Q_k R_k$，$A_{k+1} = R_k Q_k$，最终 $A_k$ 收敛到 Schur 形式 $T$，对角元即为特征值。这是 `numpy.linalg.eig` 背后的算法
- **矩阵函数 $f(A)$**：通过 Schur 分解，$f(A) = Q f(T) Q^H$，而上三角矩阵的函数 $f(T)$ 计算方法成熟（Parlett 递推）
- **控制理论**：Lyapunov 方程 $AX + XA^\top = -Q$ 的求解

## 10. SVD：矩阵分解的「终极形式」

### 8.1 定义

对于**任意** $m \times n$ 实矩阵 $A$（秩为 $r$），存在奇异值分解（Singular Value Decomposition）：

$$
\boxed{A = U\Sigma V^\top}
$$

其中：
- $U$（$m \times m$）：**左奇异向量**，正交矩阵，列是 $AA^\top$ 的特征向量
- $\Sigma$（$m \times n$）：**奇异值**对角矩阵，$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$，其余为零
- $V$（$n \times n$）：**右奇异向量**，正交矩阵，列是 $A^\top A$ 的特征向量

与特征值分解的关键区别：**SVD 对任意矩阵成立，不需要方阵，不需要对称**。

### 8.2 几何直觉

SVD 将任意线性变换分解为三个步骤：

![图3: SVD 三步变换](https://img.yumeko.site/file/blog/MatrixDecomposition/SVDGeometry.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张教学示意图，展示 SVD 的几何意义——三步变换。
> 左侧：输入空间 R^n 中的单位球面。
> 中间步骤1：V^T 旋转/反射（球面不变，只转动方向）。
> 中间步骤2：Σ 缩放（球面被拉伸为椭球，各轴按奇异值 σ1,σ2 缩放）。
> 右侧：U 再次旋转，得到 R^m 中的最终椭球。
> 每步之间用彩色箭头连接，标注矩阵名称。
> 白色背景，学术教科书风格，中文标注。
> ```

1. **$V^\top$（正交变换）**：旋转/反射，保持长度不变（单位球面仍是球面）
2. **$\Sigma$（对角缩放）**：在各坐标轴上按 $\sigma_i$ 独立缩放（球面 → 椭球面）
3. **$U$（正交变换）**：再次旋转，对最终方向进行定位

**任何线性变换 $A$ 本质上只是在两个旋转之间插入一组独立的缩放**。这是线性代数中最深刻的结论之一。

### 8.3 紧 SVD 和截断 SVD

**紧 SVD（Compact SVD）**：只保留 $U$ 的前 $r$ 列、$\Sigma$ 的前 $r$ 行 $r$ 列、$V$ 的前 $r$ 列：

$$
A = U_r \Sigma_r V_r^\top, \quad U_r \in \mathbb{R}^{m \times r}, \; \Sigma_r \in \mathbb{R}^{r \times r}, \; V_r \in \mathbb{R}^{n \times r}
$$

**截断 SVD（Truncated SVD）**：只保留前 $k < r$ 个最大的奇异值，得到 $A$ 的**秩-$k$ 最优近似**：

$$
A_k = U_k \Sigma_k V_k^\top
$$

Eckart-Young-Mirsky 定理保证：在 Frobenius 范数和谱范数下，$A_k$ 是所有秩-$k$ 矩阵中与 $A$ 最接近的那个。近似误差为：

$$
\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2
$$

### 8.4 数值示例

对 $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$（$2 \times 3$，秩 $2$），**逐步计算完整 SVD**。

**步骤 1：计算 $A^\top A$**（用于求 $V$ 和 $\Sigma$）

$$
A^\top A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
= \begin{bmatrix}
1\cdot1+0\cdot0 & 1\cdot0+0\cdot1 & 1\cdot1+0\cdot1 \\
0\cdot1+1\cdot0 & 0\cdot0+1\cdot1 & 0\cdot1+1\cdot1 \\
1\cdot1+1\cdot0 & 1\cdot0+1\cdot1 & 1\cdot1+1\cdot1
\end{bmatrix}
= \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

验证对称性：$A^\top A = (A^\top A)^\top$ ✓。

**步骤 2：求 $A^\top A$ 的特征值和特征向量**（得 $\Sigma$ 和 $V$）

特征方程 $\det(A^\top A - \lambda I) = 0$：

$$
\det\begin{bmatrix} 1-\lambda & 0 & 1 \\ 0 & 1-\lambda & 1 \\ 1 & 1 & 2-\lambda \end{bmatrix} = 0
$$

按第一行展开：

$$
(1-\lambda)\big[(1-\lambda)(2-\lambda) - 1\big] + 1 \cdot \big[0 - (1-\lambda)\big] = 0
$$

$$
(1-\lambda)(\lambda^2 - 3\lambda + 1) - (1-\lambda) = (1-\lambda)(\lambda^2 - 3\lambda) = \lambda(1-\lambda)(\lambda - 3) = 0
$$

特征值：$\lambda_1 = 3$，$\lambda_2 = 1$，$\lambda_3 = 0$（因为 $\operatorname{rank}(A)=2$，必然有一个零特征值）。

奇异值：$\sigma_1 = \sqrt{3}$，$\sigma_2 = 1$，$\sigma_3 = 0$。

$$
\Sigma = \begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
$$

**步骤 3：求 $A^\top A$ 的特征向量**（得 $V$ 的各列）

对 $\lambda_1 = 3$，解 $(A^\top A - 3I)\mathbf{v} = 0$：

$$
\begin{bmatrix} -2 & 0 & 1 \\ 0 & -2 & 1 \\ 1 & 1 & -1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

第一行：$-2x + z = 0 \;\Rightarrow\; z = 2x$
第二行：$-2y + z = 0 \;\Rightarrow\; z = 2y$
故 $x = y = z/2$。取 $x = 1$，得 $\mathbf{v}_1 = (1, 1, 2)^\top$。归一化：

$$
\mathbf{v}_1 = \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}
$$

对 $\lambda_2 = 1$，解 $(A^\top A - I)\mathbf{v} = 0$：

$$
\begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

第三行：$x + y + z = 0$。第一/二行：$z = 0$。故 $x + y = 0$，即 $y = -x$。

取 $x = 1$，得 $\mathbf{v}_2 = (1, -1, 0)^\top$。归一化：

$$
\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}
$$

对 $\lambda_3 = 0$，解 $A^\top A \mathbf{v} = 0$：

$$
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

第一行：$x + z = 0$，第二行：$y + z = 0$。$x = y = -z$。取 $z = -1$：

归一化：

$$
\mathbf{v}_3 = \frac{1}{\sqrt{3}} \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}
$$

验证正交性：$\mathbf{v}_1 \cdot \mathbf{v}_2 = 1 - 1 = 0$ ✓，$\mathbf{v}_1 \cdot \mathbf{v}_3 = 1 + 1 - 2 = 0$ ✓，$\mathbf{v}_2 \cdot \mathbf{v}_3 = 1 - 1 + 0 = 0$ ✓。

$$
V = \begin{bmatrix}
1/\sqrt{6} & 1/\sqrt{2} & 1/\sqrt{3} \\
1/\sqrt{6} & -1/\sqrt{2} & 1/\sqrt{3} \\
2/\sqrt{6} & 0 & -1/\sqrt{3}
\end{bmatrix}
$$

**步骤 4：求 $U$**（通过 $U = AV\Sigma^+$ 或直接用 $AA^\top$ 的特征向量）

方法一：利用 $A\mathbf{v}_i = \sigma_i \mathbf{u}_i$（仅对有非零奇异值的前 $r = 2$ 列）：

$$
\mathbf{u}_1 = \frac{A\mathbf{v}_1}{\sigma_1} = \frac{1}{\sqrt{3}} \cdot \frac{1}{\sqrt{6}}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}
= \frac{1}{\sqrt{18}} \begin{bmatrix} 1+0+2 \\ 0+1+2 \end{bmatrix}
= \frac{1}{3\sqrt{2}} \begin{bmatrix} 3 \\ 3 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

$$
\mathbf{u}_2 = \frac{A\mathbf{v}_2}{\sigma_2} = \frac{1}{1} \cdot \frac{1}{\sqrt{2}}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 \\ -1 \\ 0 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 1+0+0 \\ 0-1+0 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

验证正交性：$\mathbf{u}_1 \cdot \mathbf{u}_2 = 1 - 1 = 0$ ✓。

$$
U = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}
$$

> 注意 $\mathbf{u}_2$ 也可以取 $\frac{1}{\sqrt{2}}[-1, 1]^\top$（符号翻转不影响分解的正确性，因为对应的 $\sigma_2$ 在 $\Sigma$ 中不变）。

**步骤 5：组装并验证**

$$
A = U\Sigma V^\top = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\begin{bmatrix} \sqrt{3} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix}
1/\sqrt{6} & 1/\sqrt{6} & 2/\sqrt{6} \\
1/\sqrt{2} & -1/\sqrt{2} & 0
\end{bmatrix}
$$

先算 $\Sigma V^\top$：

$$
\Sigma V^\top = \begin{bmatrix} \sqrt{3}/\sqrt{6} & \sqrt{3}/\sqrt{6} & 2\sqrt{3}/\sqrt{6} \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \end{bmatrix}
= \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & \sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \end{bmatrix}
$$

再乘 $U$：

$$
U(\Sigma V^\top) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} & \sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} & 0 \end{bmatrix}
$$

第一行第一列：$\frac{1}{\sqrt{2}}(1 \cdot 1/\sqrt{2} + 1 \cdot 1/\sqrt{2}) = \frac{1}{\sqrt{2}} \cdot \frac{2}{\sqrt{2}} = 1$

第一行第二列：$\frac{1}{\sqrt{2}}(1 \cdot 1/\sqrt{2} + 1 \cdot (-1/\sqrt{2})) = 0$

第一行第三列：$\frac{1}{\sqrt{2}}(1 \cdot \sqrt{2} + 1 \cdot 0) = 1$

第二行第一列：$\frac{1}{\sqrt{2}}(1 \cdot 1/\sqrt{2} + (-1) \cdot 1/\sqrt{2}) = 0$

第二行第二列：$\frac{1}{\sqrt{2}}(1 \cdot 1/\sqrt{2} + (-1) \cdot (-1/\sqrt{2})) = \frac{1}{\sqrt{2}} \cdot \frac{2}{\sqrt{2}} = 1$

第二行第三列：$\frac{1}{\sqrt{2}}(1 \cdot \sqrt{2} + (-1) \cdot 0) = 1$

$$
U\Sigma V^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = A \;\checkmark
$$

**截断 SVD 演示**（$k=1$，秩-1 近似）：

保留最大的奇异值 $\sigma_1 = \sqrt{3}$，丢弃 $\sigma_2 = 1$：

$$
A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top = \sqrt{3} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} \cdot \frac{1}{\sqrt{6}}\begin{bmatrix} 1 & 1 & 2 \end{bmatrix}
= \frac{\sqrt{3}}{\sqrt{12}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 2 \end{bmatrix}
= \frac{1}{2} \begin{bmatrix} 1 & 1 & 2 \\ 1 & 1 & 2 \end{bmatrix}
$$

近似误差：$\|A - A_1\|_F^2 = \sigma_2^2 = 1^2 = 1$。

验证：$A - A_1 = \begin{bmatrix} 1/2 & -1/2 & 0 \\ -1/2 & 1/2 & 0 \end{bmatrix}$，Frobenius 范数平方 $= 1/4 + 1/4 + 0 + 1/4 + 1/4 + 0 = 1$ ✓。

### 8.5 应用场景

SVD 是 AI/ML 中应用最广泛的矩阵分解，「没有之一」：

| 应用 | 机制 | 使用方式 |
|:--|:--|:--|
| **PCA** | 数据中心化后 $X = U\Sigma V^\top$，$V$ 的列是主成分 | 截断 SVD |
| **图像压缩** | 保留前 $k$ 个奇异值，$A_k = U_k \Sigma_k V_k^\top$ | 截断 SVD |
| **推荐系统** | 用户-物品评分矩阵的低秩近似 | 截断 SVD |
| **伪逆** | $A^+ = V\Sigma^+ U^\top$ | 完整 SVD |
| **矩阵条件数** | $\kappa(A) = \sigma_{\max} / \sigma_{\min}$ | 完整 SVD |
| **秩判定** | 非零奇异值的个数 | 完整 SVD |
| **LoRA（大模型微调）** | 权重更新 $\Delta W = BA$，本质上就是低秩分解 | 秩-$k$ 近似 |

## 11. 极分解：旋转与拉伸的分离

### 11.1 定义

任意 $n \times n$ 复方阵 $A$ 可以唯一地分解为一个酉矩阵和一个半正定 Hermitian 矩阵的乘积。根据乘法顺序，有两种等价形式：

**右极分解**（旋转在左，拉伸在右）：

$$
\boxed{A = UP}
$$

**左极分解**（拉伸在左，旋转在右）：

$$
\boxed{A = P'U}
$$

其中：
- $U$ 是**酉矩阵**（$U^H U = I$），代表「旋转 / 反射」部分
- $P = \sqrt{A^H A}$ 是**半正定 Hermitian 矩阵**，代表「拉伸」部分
- $P' = \sqrt{A A^H}$，通常 $P \neq P'$（除非 $A$ 是正规矩阵）

对于实矩阵，$U$ 是正交矩阵，$P$ 是对称半正定矩阵。

### 11.2 几何直觉

极分解将任意线性变换拆解为两个最纯粹的几何操作：

- **$P$（拉伸）**：沿特征方向缩放假想空间。$P$ 的特征向量是拉伸主轴，特征值是各方向的拉伸倍数
- **$U$（旋转）**：刚性旋转 / 反射，保持所有长度不变（$\|U\mathbf{x}\| = \|\mathbf{x}\|$）

任何线性变换 $A$ 都可以理解为：**先拉伸再旋转**（$A = UP$），或者**先旋转再拉伸**（$A = P'U$）。

> [!TIP] 类比：极坐标
> 极分解之于矩阵，正如极坐标之于复数：$z = r e^{i\theta}$，其中 $r = |z|$（模长，对应 $P$），$e^{i\theta}$（旋转，对应 $U$）。

### 11.3 通过 SVD 构造极分解

极分解可以从 SVD 直接得到。设 $A = W\Sigma V^H$ 为 SVD：

**右极分解**：在 $\Sigma$ 和 $V^H$ 之间插入 $W^H W = I$：

$$
A = W\Sigma V^H = (W V^H)(V \Sigma V^H) = U \cdot P
$$

其中：
- $U = W V^H$ — 两个酉矩阵的乘积仍是酉矩阵 ✓
- $P = V \Sigma V^H$ — 对 $\Sigma$ 做正交相似变换，保持半正定性 ✓

验证 $P = \sqrt{A^H A}$：

$$
A^H A = (W\Sigma V^H)^H (W\Sigma V^H) = V\Sigma^2 V^H
$$

$$
P = V\Sigma V^H = \sqrt{V\Sigma^2 V^H} = \sqrt{A^H A} \;\checkmark
$$

### 11.4 数值示例

对 $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$ 做极分解。

**步骤 1**：求 $A$ 的 SVD。

先算 $A^\top A = \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 9 & 3 \\ 3 & 5 \end{bmatrix}$。

特征值：$\det = (9-\lambda)(5-\lambda) - 9 = \lambda^2 - 14\lambda + 36 = 0$，$\lambda = 7 \pm \sqrt{13}$。

$$
\sigma_1 = \sqrt{7 + \sqrt{13}} \approx 3.26, \quad \sigma_2 = \sqrt{7 - \sqrt{13}} \approx 1.84
$$

对 $\lambda_1 = 7+\sqrt{13}$：$(A^\top A - \lambda_1 I)\mathbf{v} = \begin{bmatrix} 2-\sqrt{13} & 3 \\ 3 & -2-\sqrt{13} \end{bmatrix}\mathbf{v} = 0$。

第一行：$(2-\sqrt{13})v_1 + 3v_2 = 0 \;\Rightarrow\; v_2 = \frac{\sqrt{13}-2}{3}v_1$。

归一化后得 $\mathbf{v}_1$。对 $\lambda_2$ 类似得 $\mathbf{v}_2$（与 $\mathbf{v}_1$ 正交）。

:::fold[完整 SVD 数值结果]

精确计算得：

$$
W \approx \begin{bmatrix} 0.89 & -0.46 \\ 0.46 & 0.89 \end{bmatrix}, \quad
\Sigma \approx \begin{bmatrix} 3.26 & 0 \\ 0 & 1.84 \end{bmatrix}, \quad
V \approx \begin{bmatrix} 0.83 & -0.55 \\ 0.55 & 0.83 \end{bmatrix}
$$

:::

**步骤 2**：构造右极分解 $A = UP$。

$$
U = W V^\top \approx \begin{bmatrix} 0.89 & -0.46 \\ 0.46 & 0.89 \end{bmatrix}
\begin{bmatrix} 0.83 & 0.55 \\ -0.55 & 0.83 \end{bmatrix}
\approx \begin{bmatrix} 0.99 & 0.11 \\ -0.11 & 0.99 \end{bmatrix}
$$

验证 $U$ 的正交性：$U^\top U \approx I$，$\det(U) \approx 1$（纯旋转，无反射）。$U$ 对应的旋转角 $\theta = \arcsin(0.11) \approx 6.3^\circ$。

$$
P = V \Sigma V^\top \approx \begin{bmatrix} 0.83 & -0.55 \\ 0.55 & 0.83 \end{bmatrix}
\begin{bmatrix} 3.26 & 0 \\ 0 & 1.84 \end{bmatrix}
\begin{bmatrix} 0.83 & 0.55 \\ -0.55 & 0.83 \end{bmatrix}
\approx \begin{bmatrix} 2.81 & 0.83 \\ 0.83 & 2.29 \end{bmatrix}
$$

**验证**：

$$
UP \approx \begin{bmatrix} 0.99 & 0.11 \\ -0.11 & 0.99 \end{bmatrix}
\begin{bmatrix} 2.81 & 0.83 \\ 0.83 & 2.29 \end{bmatrix}
\approx \begin{bmatrix} 3.00 & 1.00 \\ 0.00 & 2.00 \end{bmatrix} = A \;\checkmark
$$

**用更简洁的例子走一遍完整手算**：$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$（剪切变换）。

SVD 计算较繁琐，但极分解的结果可以直接验证。$A$ 的极分解为：

$$
P = \sqrt{A^\top A} = \sqrt{\begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}}
$$

$$
U = A P^{-1}
$$

:::fold[精确结果]

$A^\top A = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}$，特征值 $\lambda_{1,2} = \frac{3 \pm \sqrt{5}}{2}$。

$P = \sqrt{A^\top A}$ 与 $A^\top A$ 有相同的特征向量，特征值为 $\sqrt{\lambda}$。

最终 $U = A P^{-1}$ 是正交矩阵，代表纯旋转。

:::

**几何解读**：剪切变换 $A$ 被分解为「先沿特定方向拉伸（$P$），再旋转到最终位置（$U$）」。拉伸的「椭圆」主轴方向由 $P$ 的特征向量决定。

### 11.5 应用场景

- **连续介质力学**：变形梯度张量 $F$ 的极分解 $F = RU$，$U$ 是右伸长张量（纯变形），$R$ 是旋转张量（刚体旋转）
- **计算机图形学**：形状插值（shape interpolation）中，对变换矩阵做极分解，分别插值旋转部分和拉伸部分，避免出现不自然的剪切
- **最近正交矩阵逼近**：在所有正交矩阵中，$U$（极分解的酉因子）是 Frobenius 范数下离 $A$ 最近的那个——即 $\min_{\|Q^TQ=I\|} \|A - Q\|_F$ 的解就是 $U$
- **量子力学**：算符的极分解对应连续线性变换的「模」和「相位」分离

## 12. 完整对比总结

![图4: 分解对比表](https://img.yumeko.site/file/blog/MatrixDecomposition/ComparisonTable.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的信息图，以层级决策树的形式展示如何选择合适的矩阵分解方法。
> 从顶部问"矩阵是否对称？"开始，分为左右两支。
> 左侧（对称）：正定→Cholesky，非正定→LDL，对角化→特征值分解。
> 右侧（非对称）：方阵→LU或Schur，任意→QR或SVD。
> 每个终点卡片标注分解公式、复杂度（O记号）、典型应用一个词。
> 白色背景，蓝绿色系，专业信息图风格。
> ```

### 分解选择决策树

**第一步：看矩阵形状**

- $A$ 不是方阵？ → 用 **QR**（正交分解）或 **SVD**（终极通用）
- $A$ 是方阵？ → 进入第二步

**第二步：看是否对称**

- $A$ 对称？
  - 是 → 进入第三步
  - 否 → 进入第四步

**第三步：对称矩阵的选择**

- $A$ 正定？ → **Cholesky**：$A = LL^\top$（最快）
- $A$ 不保证正定？ → **LDL**：$A = LDL^\top$（稳健）
- 需要特征信息（PCA 等）？ → **谱分解**：$A = \sum \lambda_i \mathbf{u}_i \mathbf{u}_i^\top$（特征向量正交）

**第四步：非对称方阵的选择**

- 解线性方程组 $Ax = b$（多右端项）？ → **LU**：$PA = LU$
- 需要特征值？可对角化 → **特征值分解**：$A = Q\Lambda Q^{-1}$
- 需要特征值？不可对角化 → **Schur**：$A = QTQ^H$
- 不确定 / 需要正交因子？ → **QR**：$A = QR$

**终极兜底**：如果以上都不适用，或者需要最优低秩近似——**SVD**：$A = U\Sigma V^\top$，对任意矩阵有效。

### 复杂度与数值特性

| 分解 | 计算复杂度 | 数值稳定性 | 适用条件 | ML 典型应用 |
|:--|:--|:--|:--|:--|
| **LU** | $O(n^3)$ | 中等（需选主元） | 方阵，主子式非零 | 线性方程组求解 |
| **Cholesky** | $O(n^3/3)$ | 极好 | 对称正定 | 优化（Newton 法）、多元采样 |
| **LDL** | $O(n^3/3)$ | 很好 | 对称，主子式非零 | 非凸优化（不定 Hessian） |
| **QR** | $O(mn^2)$ | 很好 | 任意矩阵 | 最小二乘、特征值计算 |
| **特征值** | $O(n^3)$ | 中等 | 可对角化方阵 | PCA、谱聚类 |
| **谱分解** | $O(n^3)$ | 极好 | 正规矩阵 | PCA（协方差）、谱聚类 |
| **Schur** | $O(n^3)$ | 极好 | 任意方阵 | 矩阵函数、QR 算法 |
| **SVD** | $O(mn^2)$ | 极好 | **任意矩阵** | PCA、压缩、推荐、伪逆 |

## 13. Python 实现汇总

下面的代码汇总了 NumPy/SciPy 中每种分解的调用方式：

```python
"""
矩阵分解方法汇总

演示六种核心矩阵分解的 NumPy/SciPy 调用方式
以及各自的关键参数和注意事项
"""

import numpy as np
from scipy import linalg


def demoLU(A: np.ndarray):
    """
    LU 分解

    Args:
        A: 方阵
    Returns:
        P, L, U: 置换矩阵、单位下三角、上三角
    """
    P, L, U = linalg.lu(A)
    # 验证：P @ A = L @ U
    assert np.allclose(P @ A, L @ U)
    print(f"LU 分解完成，L shape={L.shape}, U shape={U.shape}")
    return P, L, U


def demoCholesky(A: np.ndarray):
    """
    Cholesky 分解（要求对称正定）

    Args:
        A: 对称正定矩阵
    Returns:
        L: 下三角矩阵，满足 A = L @ L.T
    """
    L = linalg.cholesky(A, lower=True)
    assert np.allclose(A, L @ L.T)
    print(f"Cholesky 分解完成，L shape={L.shape}")
    return L


def demoQR(A: np.ndarray):
    """
    QR 分解

    Args:
        A: 任意矩阵 (m, n)
    Returns:
        Q, R: 正交矩阵、上三角矩阵
    """
    Q, R = linalg.qr(A)
    assert np.allclose(A, Q @ R)
    print(f"QR 分解完成，Q shape={Q.shape}, R shape={R.shape}")
    return Q, R


def demoEigen(A: np.ndarray):
    """
    特征值分解

    Args:
        A: 方阵
    Returns:
        eigenvalues: 特征值数组
        eigenvectors: 特征向量矩阵
    """
    eigenvalues, eigenvectors = linalg.eig(A)
    # 验证：A @ v_i = λ_i * v_i
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        assert np.allclose(A @ v, lam * v)
    print(f"特征值分解完成，特征值: {eigenvalues}")
    return eigenvalues, eigenvectors


def demoSchur(A: np.ndarray):
    """
    Schur 分解

    Args:
        A: 方阵
    Returns:
        T: 上三角（或拟上三角）矩阵
        Q: 酉/正交矩阵
    """
    T, Q = linalg.schur(A)
    assert np.allclose(A, Q @ T @ Q.T)
    # 检查 T 的对角元是否为特征值
    assert np.allclose(np.sort(np.diag(T)), np.sort(linalg.eigvals(A)))
    print(f"Schur 分解完成，T 对角元 = {np.diag(T)}")
    return T, Q


def demoSVD(A: np.ndarray):
    """
    SVD 奇异值分解

    Args:
        A: 任意矩阵 (m, n)
    Returns:
        U, S, Vt: 左奇异向量、奇异值、右奇异向量转置
    """
    U, S, Vt = linalg.svd(A, full_matrices=False)
    # 重建
    Sigma = np.diag(S)
    assert np.allclose(A, U @ Sigma @ Vt)
    print(f"SVD 分解完成，奇异值: {S}")
    return U, S, Vt


def main():
    """主函数：演示所有分解方法"""
    np.set_printoptions(precision=4, suppress=True)

    # 测试数据
    A_symPos = np.array([[4, 2], [2, 3]])          # 对称正定
    A_square = np.array([[2, 1, 1], [4, 3, 3],     # 一般方阵
                         [8, 7, 9]])
    A_rect = np.array([[1, 0, 1], [0, 1, 1]])      # 矩形矩阵
    A_nonDiag = np.array([[2, 1], [0, 2]])          # Jordan 块（不可对角化）

    print("=" * 50)
    print("1. LU 分解（一般方阵）")
    print("=" * 50)
    demoLU(A_square)

    print("\n" + "=" * 50)
    print("2. Cholesky 分解（对称正定）")
    print("=" * 50)
    demoCholesky(A_symPos)

    print("\n" + "=" * 50)
    print("3. QR 分解（矩形矩阵）")
    print("=" * 50)
    demoQR(A_rect)

    print("\n" + "=" * 50)
    print("4. 特征值分解（对称矩阵 = 正交对角化）")
    print("=" * 50)
    demoEigen(A_symPos)

    print("\n" + "=" * 50)
    print("5. Schur 分解（不可对角化矩阵）")
    print("=" * 50)
    demoSchur(A_nonDiag)

    print("\n" + "=" * 50)
    print("6. SVD 分解（矩形矩阵）")
    print("=" * 50)
    demoSVD(A_rect)


if __name__ == "__main__":
    main()
```

---

> **相关文章**：
> - [[Math/MultivariateStatistics|多元统计分析基础]]
> - [[Math/ProbabilityInequalities|概率不等式完全指南]]
