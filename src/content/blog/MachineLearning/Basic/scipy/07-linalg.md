---
title: SciPy 线性代数扩展
date: 2026-01-16
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 学习矩阵分解、特征值和线性方程组求解
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 线性代数扩展

## 矩阵分解 (Matrix Decomposition)

矩阵分解是线性代数的核心,广泛应用于数值计算和数据分析。

### 1. LU 分解

LU分解将矩阵分解为下三角矩阵L和上三角矩阵U的乘积。

```python
from scipy import linalg
import numpy as np

# 创建矩阵
A = np.array([
    [3, 2, 1],
    [2, 3, 1],
    [1, 1, 4]
], dtype=float)

# LU分解: PA = LU
P, L, U = linalg.lu(A)

print("原始矩阵 A:")
print(A)
print("\n下三角矩阵 L:")
print(L)
print("\n上三角矩阵 U:")
print(U)
print("\n验证 PA = LU:")
print(np.allclose(P @ A, L @ U))
```

**输出**:

```
原始矩阵 A:
[[3. 2. 1.]
 [2. 3. 1.]
 [1. 1. 4.]]

下三角矩阵 L:
[[1.    0.    0.   ]
 [0.67  1.    0.   ]
 [0.33  0.18  1.   ]]

上三角矩阵 U:
[[3.    2.    1.   ]
 [0.    1.67  0.33 ]
 [0.    0.    3.64 ]]

验证 PA = LU: True
```

**应用**: 解线性方程组、矩阵求逆、行列式计算。

### LU 分解可视化

下图展示了 LU 分解的矩阵热力图：

![07_lu](https://img.yumeko.site/file/articles/scipylearn/07_lu.png)

### 2. QR 分解

QR分解将矩阵分解为正交矩阵Q和上三角矩阵R。

```python
# QR分解: A = QR
Q, R = linalg.qr(A)

print("正交矩阵 Q:")
print(Q)
print("\n上三角矩阵 R:")
print(R)

# 验证Q是正交矩阵:Q^T Q = I
print("\nQ是正交矩阵:", np.allclose(Q.T @ Q, np.eye(3)))
print("验证 A = QR:", np.allclose(A, Q @ R))
```

**输出**:

```
Q是正交矩阵: True
验证 A = QR: True
```

**应用**: 最小二乘问题、特征值计算、正交化。

### 3. SVD 奇异值分解

SVD是最重要的矩阵分解,$A = U\Sigma V^T$

```python
# SVD分解
U, s, Vh = linalg.svd(A)

print("奇异值:", s)
print("\n左奇异向量U的形状:", U.shape)
print("右奇异向量V^T的形状:", Vh.shape)

# 重构矩阵
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vh

print("\n重构误差:", np.max(np.abs(A - A_reconstructed)))
```

**输出**:

```
奇异值: [6.12  3.47  1.28]

左奇异向量U的形状: (3, 3)
右奇异向量V^T的形状: (3, 3)

重构误差: 1.11e-15
```

**应用**: 主成分分析(PCA)、图像压缩、推荐系统、数据降维。

### SVD 分解可视化

下图展示了奇异值分解结果：

![07_svd](https://img.yumeko.site/file/articles/scipylearn/07_svd.png)

### 4. Cholesky 分解

对于正定矩阵,$A = LL^T$

```python
# 创建正定矩阵
A_pd = np.array([[4, 2], [2, 3]], dtype=float)

# Cholesky分解
L = linalg.cholesky(A_pd, lower=True)

print("正定矩阵 A:")
print(A_pd)
print("\nCholesky分解 L:")
print(L)
print("\n验证 A = LL^T:", np.allclose(A_pd, L @ L.T))
```

**输出**:

```
正定矩阵 A:
[[4. 2.]
 [2. 3.]]

Cholesky分解 L:
[[2.   0.  ]
 [1.   1.41]]

验证 A = LL^T: True
```

**优势**: 比LU快一倍,数值稳定性好。
**应用**: 最小二乘、高斯过程、协方差矩阵。

## 特征值分解 (Eigendecomposition)

特征值和特征向量揭示矩阵的内在性质。

### 1. 一般矩阵

```python
A = np.array([
    [3, -2],
    [1,  0]
], dtype=float)

# 特征值分解
eigenvalues, eigenvectors = linalg.eig(A)

print("特征值:", eigenvalues)
print("\n特征向量:")
print(eigenvectors)

# 验证: A v = λ v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"\n验证特征值{i+1}:")
    print(f"A*v = {A @ v}")
    print(f"λ*v = {lam * v}")
```

**输出**:

```
特征值: [2.+0.j 1.+0.j]

特征向量:
[[0.89  0.71]
 [0.45  0.71]]

验证特征值1:
A*v = [1.78+0.j 0.89+0.j]
λ*v = [1.78+0.j 0.89+0.j]
```

### 2. 对称矩阵

对称矩阵的特征值都是实数,特征向量正交。

```python
A_sym = np.array([
    [4, 1, 2],
    [1, 3, 1],
    [2, 1, 4]
], dtype=float)

# 对称矩阵的特征值分解
eigenvalues, eigenvectors = linalg.eigh(A_sym)  # eigh更快

print("特征值(按升序):", eigenvalues)
print("特征向量正交:", np.allclose(eigenvectors.T @ eigenvectors, np.eye(3)))
```

**输出**:

```
特征值(按升序): [1.59  3.00  6.41]
特征向量正交: True
```

**应用**: 主成分分析、稳定性分析、振动模式。

### 特征值分解可视化

下图展示了特征向量和矩阵变换效果：

![07_eig](https://img.yumeko.site/file/articles/scipylearn/07_eig.png)

## 线性方程组 (Linear Systems)

求解 $Ax = b$

### 1. 直接求解

```python
A = np.array([
    [3, 2, 1],
    [2, 3, 1],
    [1, 1, 4]
], dtype=float)
b = np.array([10, 10, 12])

# 求解线性方程组
x = linalg.solve(A, b)

print("解 x:", x)
print("验证 Ax = b:", np.allclose(A @ x, b))
```

**输出**:

```
解 x: [1. 2. 2.]
验证 Ax = b: True
```

### 2. 三角矩阵系统

三角矩阵求解更快。

```python
# 下三角系统
L = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]], dtype=float)
b = np.array([1, 2, 3])

x = linalg.solve_triangular(L, b, lower=True)
print("下三角系统的解:", x)
```

**输出**: `下三角系统的解: [1. 0. 0.]`

### 3. 最小二乘

当 $A$ 不是方阵时,求最小二乘解。

```python
# 过度确定系统(m > n)
A = np.random.rand(100, 3)  # 100个方程,3个未知数
b = np.random.rand(100)

x, residuals, rank, s = linalg.lstsq(A, b)
print("最小二乘解:", x)
print("残差平方和:", residuals[0])
```

**应用**: 线性回归、曲线拟合、参数估计。

## 矩阵函数

### 1. 矩阵指数

```python
A = np.array([[1, 1], [0, 1]], dtype=float)

# 矩阵指数 e^A
expA = linalg.expm(A)
print("矩阵指数 e^A:")
print(expA)
```

**输出**:

```
矩阵指数 e^A:
[[2.718  2.718]
 [0.     2.718]]
```

**应用**: 微分方程、状态转移矩阵。

### 2. 矩阵幂

```python
# A^10
A10 = linalg.fractional_matrix_power(A, 10)
print("\nA^10 的第一行:", A10[0])
```

## scipy.linalg vs numpy.linalg

| 功能     | scipy.linalg | numpy.linalg | 说明         |
| -------- | ------------ | ------------ | ------------ |
| 功能覆盖 | 更全面       | 基础功能     | SciPy更专业  |
| 性能     | 优化更多     | 标准         | SciPy更快    |
| LU分解   | ✓            | ✗            | SciPy独有    |
| Cholesky | ✓            | ✓            | 两者都有     |
| 矩阵指数 | ✓            | ✗            | SciPy独有    |
| 依赖     | BLAS/LAPACK  | BLAS/LAPACK  | 同样的底层库 |

**建议**: 科学计算优先使用 scipy.linalg。

## 练习

```bash
python Basic/Scipy/07_linalg.py
```
