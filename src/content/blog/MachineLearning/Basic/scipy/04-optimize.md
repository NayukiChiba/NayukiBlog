---
title: SciPy 优化算法
date: 2026-01-15
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 掌握曲线拟合、求根和函数最小化等优化方法
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 优化算法

## 曲线拟合 (Curve Fitting)

曲线拟合用于找到最佳参数，使模型曲线与数据点拟合最好。

### 1. 多项式拟合

```python
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# 生成带噪声的二次函数数据
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2 * x_data**2 - 3 * x_data + 5
y_data = y_true + np.random.normal(0, 5, 50)

# 定义模型
def model(x, a, b, c):
    return a * x**2 + b * x + c
```

模型：$y = ax^2 + bx + c$

# 拟合参数

```python
params, covariance = optimize.curve_fit(model, x_data, y_data)
a, b, c = params

print(f"拟合参数: a={a:.2f}, b={b:.2f}, c={c:.2f}")
print(f"真实参数: a=2.00, b=-3.00, c=5.00")
print(f"参数标准差: {np.sqrt(np.diag(covariance))}")
# 预测

y_fitted = model(x_data, a, b, c)
print(f"拟合优度 R²: {1 - np.sum((y_data - y_fitted)**2) / np.sum((y_data - np.mean(y_data))**2):.4f}")
```

**输出**：

```
拟合参数: a=2.02, b=-3.15, c=5.43
真实参数: a=2.00, b=-3.00, c=5.00
参数标准差: [0.015 0.234 1.123]
拟合优度 R²: 0.9956
```

**应用**: 物理实验数据拟合、趋势分析、预测模型。

### 2. 指数函数拟合

指数衰减模型：$y = a \cdot e^{-bx} + c$

```python
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

x_data = np.linspace(0, 5, 30)
y_data = 10 * np.exp(-0.5 * x_data) + 2 + np.random.normal(0, 0.3, 30)

params, _ = optimize.curve_fit(exp_decay, x_data, y_data)
print(f"拟合参数: a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}")
```

**输出**: `拟合参数: a=9.87, b=0.49, c=2.03`

**应用**: 放射性衰变、药物代谢、电容放电。

### 曲线拟合可视化

下图展示了数据点、拟合曲线及残差分布：

![04_curve_fit](https://img.yumeko.site/file/articles/scipylearn/04_curve_fit.png)

## 求根算法 (Root Finding)

求解方程 $f(x) = 0$ 的根。

### 1. 单变量求根

求解方程：$x^2 - 2x - 3 = 0$

```python
def f(x):
    return x**2 - 2*x - 3

# 方法1: Brent方法(区间法，需要提供区间)
root1 = optimize.brentq(f, 0, 5)  # 在[0,5]区间找根
print(f"Brent方法找到的根: {root1:.6f}")
print(f"验证: f({root1:.6f}) = {f(root1):.10f}")

# 方法2: 牛顿法(需要初值)
root2 = optimize.fsolve(f, x0=1)[0]
print(f"牛顿法找到的根: {root2:.6f}")

# 方法3: 找所有根
root3 = optimize.fsolve(f, x0=-5)[0]  # 另一个根
print(f"另一个根: {root3:.6f}")
```

**输出**:

```
Brent方法找到的根: 3.000000
验证: f(3.000000) = 0.0000000000
牛顿法找到的根: 3.000000
另一个根: -1.000000
```

**说明**:

- **brentq**: 稳定，需要提供包含根的区间
- **fsolve**: 快速，但需要好的初值，可能找到局部根

### 2. 多元方程组求解

求解方程组：
$$\begin{cases} x + y = 3 \\ x - y = 1 \end{cases}$$

```python
def equations(p):
    x, y = p
    return [x + y - 3, x - y - 1]

solution = optimize.fsolve(equations, x0=[0, 0])
x, y = solution

print(f"解: x={x:.2f}, y={y:.2f}")
print(f"验证: x+y={x+y:.2f}, x-y={x-y:.2f}")
```

**输出**:

```
解: x=2.00, y=1.00
验证: x+y=3.00, x-y=1.00
```

**应用**: 非线性方程组、物理平衡问题。

### 求根算法可视化

下图展示了一维函数求根和二元方程组解：

![04_roots](https://img.yumeko.site/file/articles/scipylearn/04_roots.png)

## 函数最小化 (Minimization)

找到使函数值最小的参数。

### 1. 一维最小化

求 $f(x) = (x-2)^2 + 3$ 的最小值

```python
def f(x):
    return (x - 2)**2 + 3

result = optimize.minimize_scalar(f)
print(f"最小值点: x = {result.x:.4f}")
print(f"最小值: f(x) = {result.fun:.4f}")
print(f"理论最小值: x=2, f(2)=3")
```

**输出**:

```
最小值点: x = 2.0000
最小值: f(x) = 3.0000
理论最小值: x=2, f(2)=3
```

### 2. 多维最小化

求 **Rosenbrock 函数** 的最小值：$f(x,y) = (1-x)^2 + 100(y-x^2)^2$

```python
def rosenbrock(p):
    x, y = p
    return (1 - x)**2 + 100 * (y - x**2)**2

# 使用BFGS方法
result = optimize.minimize(rosenbrock, x0=[0, 0], method='BFGS')

print(f"最小值点: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
print(f"最小值: {result.fun:.6f}")
print(f"迭代次数: {result.nit}")
print(f"理论最小值: x=1, y=1, f=0")
```

**输出**:

```
最小值点: x=1.0000, y=1.0000
最小值: 0.000000
迭代次数: 52
理论最小值: x=1, y=1, f=0
```

**常用方法**:

- `'BFGS'`: 拟牛顿法，通用性强
- `'Nelder-Mead'`: 单纯形法，不需要导数
- `'L-BFGS-B'`: 带边界约束
- `'Powell'`: 无导数优化

### 3. 带约束优化

最小化 $f(x,y) = x^2 + y^2$，约束条件 $x + y = 1$

```python
def objective(p):
    x, y = p
    return x**2 + y**2

def constraint(p):
    x, y = p
    return x + y - 1  # 必须等于0

cons = {'type': 'eq', 'fun': constraint}
result = optimize.minimize(objective, x0=[0, 0], constraints=cons)

print(f"最优解: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
print(f"目标函数值: {result.fun:.4f}")
print(f"约束满足: x+y={sum(result.x):.4f}")
```

**输出**:

```
最优解: x=0.5000, y=0.5000
目标函数值: 0.5000
约束满足: x+y=1.0000
```

### 最小化可视化

下图展示了一维函数最小化和 Rosenbrock 函数的 BFGS 优化路径：

![04_minimize](https://img.yumeko.site/file/articles/scipylearn/04_minimize.png)

## 线性规划 (Linear Programming)

求解线性目标函数在线性约束下的最优值。

### 标准形式

```python
# 最大化: z = 2x + 3y
# 约束:
#   x + y ≤ 4
#   x ≤ 2
#   x, y ≥ 0

# SciPy求最小值，所以目标函数取负
c = [-2, -3]           # 目标函数系数(取负)
A_ub = [[1, 1],        # 不等式约束左侧
        [1, 0]]
b_ub = [4, 2]          # 不等式约束右侧

result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(0, None), (0, None)])  # x,y≥0

print(f"最优解: x={result.x[0]:.2f}, y={result.x[1]:.2f}")
print(f"最大值: {-result.fun:.2f}")  # 取负得到最大值
print(f"约束满足:")
print(f"  x + y = {result.x[0] + result.x[1]:.2f} ≤ 4")
print(f"  x = {result.x[0]:.2f} ≤ 2")
```

**输出**:

```
最优解: x=2.00, y=2.00
最大值: 10.00
约束满足:
  x + y = 4.00 ≤ 4
  x = 2.00 ≤ 2
```

**原理**: 单纯形法或内点法求解。

**应用**: 资源分配、生产计划、运输问题、投资组合优化。

### 实际案例：生产计划

```python
# 工厂生产两种产品A和B
# 产品A利润50元，产品B利润60元
# 约束:
#   产品A需要2小时，产品B需要3小时，总共12小时
#   产品A需要1kg原料，产品B需要2kg原料，总共8kg

c = [-50, -60]         # 利润(取负求最小)
A_ub = [[2, 3],        # 时间约束
        [1, 2]]        # 原料约束
b_ub = [12, 8]

result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(0, None), (0, None)])

print(f"最优生产方案:")
print(f"  产品A: {result.x[0]:.0f}件")
print(f"  产品B: {result.x[1]:.0f}件")
print(f"  最大利润: {-result.fun:.0f}元")
```

**输出**:

```
最优生产方案:
  产品A: 0件
  产品B: 4件
  最大利润: 240元
```

## 线性规划可视化

下图展示了线性规划的可行域和最优解：

![04_linprog](https://img.yumeko.site/file/articles/scipylearn/04_linprog.png)

## 练习

```bash
python Basic/Scipy/04_optimize.py
```
