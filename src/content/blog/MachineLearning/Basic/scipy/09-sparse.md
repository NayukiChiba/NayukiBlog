---
title: SciPy 稀疏矩阵
date: 2026-01-17
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 学习稀疏矩阵格式、操作和线性代数运算
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 稀疏矩阵

## 稀疏矩阵基础

稀疏矩阵是大部分元素为0的矩阵，通过只存储非零元素来节省内存。

### 为什么需要稀疏矩阵？

```python
import numpy as np
from scipy import sparse

# 创建一个大型稀疏矩阵
n = 10000
density = 0.001  # 0.1%的元素非零

# 密集矩阵存储
dense = np.random.rand(n, n)
dense[dense > density] = 0
dense_memory = dense.nbytes / 1024**2  # MB

# 稀疏矩阵存储
sparse_matrix = sparse.csr_matrix(dense)
sparse_memory = (sparse_matrix.data.nbytes +
                 sparse_matrix.indices.nbytes +
                 sparse_matrix.indptr.nbytes) / 1024**2

print(f"矩阵大小: {n}x{n}")
print(f"非零元素比例: {density*100:.1f}%")
print(f"密集存储: {dense_memory:.1f} MB")
print(f"稀疏存储: {sparse_memory:.1f} MB")
print(f"内存节省: {(1-sparse_memory/dense_memory)*100:.1f}%")
```

**输出**:

```
矩阵大小: 10000x10000
非零元素比例: 0.1%
密集存储: 762.9 MB
稀疏存储: 1.2 MB
内存节省: 99.8%
```

## 稀疏矩阵格式

| 格式 | 全称   | 适用场景         | 优点       | 缺点       |
| ---- | ------ | ---------------- | ---------- | ---------- |
| CSR  | 压缩行 | 行切片、矩阵乘法 | 快速行访问 | 慢列访问   |
| CSC  | 压缩列 | 列切片、转置     | 快速列访问 | 慢行访问   |
| COO  | 坐标   | 构建、转换       | 灵活构建   | 不支持切片 |
| LIL  | 链表   | 增量构建         | 易修改     | 计算慢     |
| DOK  | 字典   | 随机访问         | 灵活       | 内存开销大 |

## 创建稀疏矩阵

### 1. 从COO格式创建

```python
# COO格式:通过(row, col, data)三元组指定
 row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

# 创建3x3稀疏矩阵
coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

print("稀疏矩阵 (COO):")
print(coo.toarray())
print(f"\n非零元素: {coo.nnz}")
print(f"稀疏度: {1 - coo.nnz / (3*3):.2%}")
```

**输出**:

```
稀疏矩阵 (COO):
[[1 0 2]
 [0 0 3]
 [4 5 6]]

非零元素: 6
稀疏度: 33.33%
```

### 2. 从密集矩阵转换

```python
# 密集矩阵
dense = np.array([
    [1, 0, 0, 2],
    [0, 0, 3, 0],
    [4, 0, 0, 0]
])

# 转换为CSR格式
csr = sparse.csr_matrix(dense)
print("转换为CSR:")
print(csr)
print("\n还原为密集:")
print(csr.toarray())
```

### 3. 特殊矩阵

```python
# 单位矩阵
I = sparse.identity(5, format='csr')
print("稀疏单位矩阵:")
print(I.toarray())

# 对角矩阵
diag = sparse.diags([1, 2, 3], [0, 1, -1], shape=(5, 5))
print("\n对角矩阵:")
print(diag.toarray())

# 随机稀疏矩阵
random_sparse = sparse.random(1000, 1000, density=0.01, format='csr')
print(f"\n随机稀疏矩阵: {random_sparse.shape}")
print(f"非零元素: {random_sparse.nnz}")
```

**输出**:

```
稀疏单位矩阵:
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

对角矩阵:
[[1. 2. 0. 0. 0.]
 [3. 1. 2. 0. 0.]
 [0. 3. 1. 2. 0.]
 [0. 0. 3. 1. 2.]
 [0. 0. 0. 3. 1.]]

随机稀疏矩阵: (1000, 1000)
非零元素: 10000
```

### 稀疏矩阵结构可视化

下图展示了稀疏矩阵的结构 (spy 图)：

![09_create](https://img.yumeko.site/file/articles/scipylearn/09_create.png)

## 稀疏矩阵操作

### 1. 基本运算

```python
A = sparse.csr_matrix([[1, 0, 2], [0, 3, 0]])
B = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]])

# 加法
C = A + B
print("加法:")
print(C.toarray())

# 数乘
D = A * 2
print("\n数乘:")
print(D.toarray())

# 矩阵乘法
E = A @ A.T
print("\n矩阵乘法:")
print(E.toarray())
```

**输出**:

```
加法:
[[2 1 2]
 [0 4 1]]

数乘:
[[2 0 4]
 [0 6 0]]

矩阵乘法:
[[5 0]
 [0 9]]
```

### 稀疏矩阵运算可视化

下图展示了稀疏矩阵加法结果：

![09_ops](https://img.yumeko.site/file/articles/scipylearn/09_ops.png)

### 2. 切片和索引

```python
A = sparse.csr_matrix([
    [1, 0, 2, 0],
    [0, 3, 0, 4],
    [5, 0, 6, 0]
])

# 行切片(CSR高效)
row = A[1, :]
print("第2行:", row.toarray())

# 列切片(需要转换为CSC)
A_csc = A.tocsc()
col = A_csc[:, 2]
print("第3列:", col.toarray().T)

# 子矩阵
submatrix = A[0:2, 1:3]
print("\n子矩阵:")
print(submatrix.toarray())
```

**输出**:

```
第2行: [[0 3 0 4]]
第3列: [[2 0 6]]

子矩阵:
[[0 2]
 [3 0]]
```

## 稀疏线性代数

### 1. 解线性方程组

```python
from scipy.sparse import linalg as splinalg

# 创建稀疏系统 Ax = b
n = 1000
A = sparse.random(n, n, density=0.01, format='csr')
A = A + sparse.identity(n) * 10  # 保证对角占优
b = np.random.rand(n)

# 直接求解
import time
start = time.time()
x = splinalg.spsolve(A, b)
solve_time = time.time() - start

print(f"矩阵大小: {n}x{n}")
print(f"非零元素: {A.nnz}")
print(f"求解时间: {solve_time:.4f}秒")
print(f"残差: {np.linalg.norm(A @ x - b):.2e}")
```

**输出**:

```
矩阵大小: 1000x1000
非零元素: 11000
求解时间: 0.0234秒
残差: 3.45e-14
```

### 稀疏线性代数可视化

下图展示了三对角矩阵的结构和线性方程组解：

![09_linalg](https://img.yumeko.site/file/articles/scipylearn/09_linalg.png)

### 2. 迭代求解器

对于大型稀疏系统，迭代方法更快。

```python
# 共轭梯度法
x, info = splinalg.cg(A, b, tol=1e-6)
print(f"CG求解状态: {info}")
print(f"残差: {np.linalg.norm(A @ x - b):.2e}")

# BiCGSTAB(非对称矩阵)
x, info = splinalg.bicgstab(A, b)
print(f"\nBiCGSTAB求解状态: {info}")
```

**info 说明**:

- `0`: 成功收敛
- $> 0$: 达到最大迭代次数
- $< 0$: 非法输入或分解失败

### 3. 特征值问题

```python
# 计算最大的10个特征值
A_sym = A + A.T  # 对称化
eigenvalues, eigenvectors = splinalg.eigsh(A_sym, k=10, which='LM')

print("最大的10个特征值:")
print(eigenvalues)
```

**which 参数**:

- `'LM'`: 最大模
- `'SM'`: 最小模
- `'LA'`: 最大代数值
- `'SA'`: 最小代数值

## 性能对比

```python
# 密集 vs 稀疏矩阵乘法
n = 1000
density = 0.01

A_sparse = sparse.random(n, n, density=density, format='csr')
B_sparse = sparse.random(n, n, density=density, format='csr')

A_dense = A_sparse.toarray()
B_dense = B_sparse.toarray()

# 稀疏矩阵乘法
start = time.time()
C_sparse = A_sparse @ B_sparse
sparse_time = time.time() - start

# 密集矩阵乘法
start = time.time()
C_dense = A_dense @ B_dense
dense_time = time.time() - start

print(f"稀疏矩阵乘法: {sparse_time:.4f}秒")
print(f"密集矩阵乘法: {dense_time:.4f}秒")
print(f"加速比: {dense_time/sparse_time:.1f}x")
```

**输出**:

```
稀疏矩阵乘法: 0.0023秒
密集矩阵乘法: 0.1245秒
加速比: 54.1x
```

## 应用场景

1. **图论**: 邻接矩阵(大部分节点未连接)
2. **机器学习**: 文本特征矩阵(TF-IDF)
3. **科学计算**: 有限元分析、偏微分方程
4. **推荐系统**: 用户-物品矩阵(大部分空值)

### 内存效率可视化

下图展示了密集矩阵与稀疏矩阵的内存对比：

![09_efficiency](https://img.yumeko.site/file/articles/scipylearn/09_efficiency.png)

## 练习

```bash
python Basic/Scipy/09_sparse.py
```
