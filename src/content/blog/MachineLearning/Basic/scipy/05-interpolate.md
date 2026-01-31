---
title: SciPy 插值方法
date: 2026-01-15
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 学习一维、二维插值和样条插值方法
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 插值方法

## 插值基础

插值是在已知数据点之间估算未知值的方法，广泛应用于数据平滑、上采样和曲线绘制。

## 一维插值 (1D Interpolation)

### 1. 线性插值

```python
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# 已知数据点
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 2, 5, 3])

# 创建线性插值函数
f_linear = interpolate.interp1d(x, y, kind='linear')

# 在更密集的点上插值
x_new = np.linspace(0, 5, 50)
y_linear = f_linear(x_new)

print("原始数据点:", len(x))
print("插值后数据点:", len(x_new))
print("在x=2.5处的插值: y =", f_linear(2.5))
```

**输出**:

```
原始数据点: 6
插值后数据点: 50
在x=2.5处的插值: y = 3.0
```

**原理**: 相邻两点间用直线连接，$y = y_1 + \frac{y_2-y_1}{x_2-x_1}(x-x_1)$

**优点**: 简单快速，连续
**缺点**: 不光滑，在节点处有折角

### 2. 三次样条插值

```python
# 三次样条插值(更平滑)
f_cubic = interpolate.interp1d(x, y, kind='cubic')
y_cubic = f_cubic(x_new)

print("在x=2.5处的三次插值: y =", f_cubic(2.5))
```

**输出**: `在x=2.5处的三次插值: y = 2.734`

**优点**: 平滑，二阶导数连续
**应用**: 曲线绘制、动画路径、信号重建

### 3. 不同插值方法对比

```python
# 可用的插值方法
kinds = ['linear', 'nearest', 'quadratic', 'cubic']

for kind in kinds:
    f = interpolate.interp1d(x, y, kind=kind)
    y_interp = f(2.5)
    print(f"{kind:12s} 插值: y(2.5) = {y_interp:.4f}")
```

**输出**:

```
linear       插值: y(2.5) = 3.0000
nearest      插值: y(2.5) = 4.0000
quadratic    插值: y(2.5) = 2.8750
cubic        插值: y(2.5) = 2.7344
```

**kind 参数说明**:

- `'nearest'`: 最近邻插值(阶梯状)
- `'linear'`: 线性插值(分段直线)
- `'quadratic'`: 二次插值
- `'cubic'`: 三次插值(推荐)
- `'slinear', 'zero', 'previous', 'next'`: 其他方法

### 一维插值可视化

下图展示了线性插值与三次插值的对比：

![05_interp1d](https://img.yumeko.site/file/articles/scipylearn/05_interp1d.png)

## 样条插值 (Spline)

样条插值提供更多控制选项，特别适合需要外推或自定义平滑度的情况。

### 1. B样条插值

```python
# 创建B样条表示
tck = interpolate.splrep(x, y, s=0)  # s=0表示精确通过所有点

# 计算插值点
x_new = np.linspace(0, 5, 100)
y_new = interpolate.splev(x_new, tck)

print("B样条在x=2.5处:", interpolate.splev(2.5, tck))

# 计算导数
y_deriv = interpolate.splev(x_new, tck, der=1)  # 一阶导数
print("x=2.5处的斜率:", interpolate.splev(2.5, tck, der=1))
```

**输出**:

```
B样条在x=2.5处: 2.7344
x=2.5处的斜率: -0.4688
```

**参数 s (平滑因子)**:

- $s = 0$: 精确通过所有点(默认)
- $s > 0$: 允许偏差，曲线更平滑

### 2. 平滑样条

```python
# 带噪声的数据
y_noisy = y + np.random.normal(0, 0.5, len(y))

# 精确拟合(s=0)
tck_exact = interpolate.splrep(x, y_noisy, s=0)
y_exact = interpolate.splev(x_new, tck_exact)

# 平滑拟合(s>0)
tck_smooth = interpolate.splrep(x, y_noisy, s=5)
y_smooth = interpolate.splev(x_new, tck_smooth)

print("精确拟合在噪声数据上会产生振荡")
print("平滑拟合可以去除噪声影响")
```

**应用**: 噪声数据处理、趋势分析。

### 样条插值可视化

下图展示了样条插值曲线及其导数：

![05_spline](https://img.yumeko.site/file/articles/scipylearn/05_spline.png)

## 二维插值 (2D Interpolation)

用于网格数据或图像的插值。

### 1. 规则网格插值

```python
# 创建规则网格数据
x = np.linspace(0, 4, 5)
y = np.linspace(0, 4, 5)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # 高度值

# 创建插值函数
interp_func = interpolate.RegularGridInterpolator((x, y), Z)

# 在任意点插值
points = np.array([[1.5, 2.3], [2.7, 3.1]])
values = interp_func(points)

print("网格大小:", Z.shape)
print("插值点:", points)
print("插值结果:", values)
```

**输出**:

```
网格大小: (5, 5)
插值点: [[1.5 2.3]
 [2.7 3.1]]
插值结果: [-0.6234  0.4521]
```

**应用**: 地形数据、图像缩放、科学可视化。

### 2. 不规则点插值

```python
# 不规则分布的点
np.random.seed(42)
points = np.random.rand(20, 2) * 4
values = np.sin(points[:, 0]) * np.cos(points[:, 1])

# 创建规则网格
grid_x, grid_y = np.mgrid[0:4:100j, 0:4:100j]

# 使用griddata插值
from scipy.interpolate import griddata
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

print("不规则点数量:", len(points))
print("插值网格大小:", grid_z.shape)
```

**输出**:

```
不规则点数量: 20
插值网格大小: (100, 100)
```

**method 参数**:

- `'nearest'`: 最近邻
- `'linear'`: 线性(快)
- `'cubic'`: 三次(慢但平滑)

### 二维插值可视化

下图展示了二维网格插值结果：

![05_interp2d](https://img.yumeko.site/file/articles/scipylearn/05_interp2d.png)

## 径向基函数插值 (RBF)

RBF适用于多维散点数据的插值，特别是高维情况。

```python
from scipy.interpolate import RBFInterpolator

# 散点数据
points = np.random.rand(30, 2) * 10
values = np.sin(points[:, 0]) + np.cos(points[:, 1])

# 创建RBF插值器
rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')

# 在新点插值
new_points = np.array([[2.5, 3.7], [5.1, 8.2]])
interp_values = rbf(new_points)

print("RBF插值结果:", interp_values)
```

**常用核函数**:

- `'thin_plate_spline'`: 薄板样条(推荐)
- `'linear'`: 线性
- `'cubic'`: 三次
- `'gaussian'`: 高斯

**优势**:

- 支持任意维度
- 不需要规则网格
- 自然平滑

**应用**: 地理信息系统、形状变形、数据拟合。

### RBF 插值可视化

下图展示了径向基函数插值结果：

![05_rbf](https://img.yumeko.site/file/articles/scipylearn/05_rbf.png)

## 插值方法选择指南

| 场景       | 推荐方法                   | 理由       |
| ---------- | -------------------------- | ---------- |
| 简单1D数据 | `interp1d(kind='cubic')`   | 平滑且快速 |
| 带噪声数据 | `splrep(s>0)`              | 可控平滑度 |
| 规则2D网格 | `RegularGridInterpolator`  | 高效       |
| 不规则2D点 | `griddata(method='cubic')` | 通用性强   |
| 高维散点   | `RBFInterpolator`          | 任意维度   |
| 需要导数   | `splrep/splev`             | 可计算导数 |

## 练习

```bash
python Basic/Scipy/05_interpolate.py
```
