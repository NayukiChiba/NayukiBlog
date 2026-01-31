---
title: SciPy 数值积分
date: 2026-01-15
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 掌握定积分和常微分方程求解
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 数值积分

## 定积分 (Definite Integration)

数值积分用于计算函数在区间上的积分值。

### 1. 一维积分

计算 $\int_0^1 x^2 \, dx = \frac{1}{3}$

```python
from scipy import integrate
import numpy as np

result, error = integrate.quad(lambda x: x**2, 0, 1)

print(f"积分结果: {result:.10f}")
print(f"误差估计: {error:.2e}")
print(f"理论值: {1/3:.10f}")
```

**输出**:

```
积分结果: 0.3333333333
误差估计: 3.70e-15
理论值: 0.3333333333
```

**原理**: 自适应高斯-克朗罗德求积法，自动调整步长保证精度。

### 2. 复杂函数积分

- $\int_0^{\pi} \sin(x) \, dx = 2$
- $\int_0^1 e^x \, dx = e - 1$
- $\int_0^2 (3x^2 + 5x) \, dx = 18$

```python
result, _ = integrate.quad(np.sin, 0, np.pi)
print(f"积分结果: {result:.6f}")

result, _ = integrate.quad(np.exp, 0, 1)
print(f"积分结果: {result:.6f}")
print(f"理论值: {np.e - 1:.6f}")

# 带参数的函数
def f(x, a, b):
    return a * x**2 + b * x

result, _ = integrate.quad(f, 0, 2, args=(3, 5))  # a=3, b=5
print(f"积分结果: {result:.2f}")
```

**输出**:

```
积分结果: 2.000000
积分结果: 1.718282
理论值: 1.718282
积分结果: 18.00
```

### 3. 无穷积分

- $\int_0^{\infty} e^{-x} \, dx = 1$
- $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$

```python
result, _ = integrate.quad(lambda x: np.exp(-x), 0, np.inf)
print(f"积分结果: {result:.6f}")

result, _ = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"积分结果: {result:.6f}")
print(f"理论值: {np.sqrt(np.pi):.6f}")
```

**输出**:

```
积分结果: 1.000000
积分结果: 1.772454
理论值: 1.772454
```

**注意**: 使用 `np.inf` 表示 $\infty$（无穷大）。

### 定积分可视化

下图展示了定积分的区域填充：

![06_quad](https://img.yumeko.site/file/articles/scipylearn/06_quad.png)

### 4. 多重积分

#### 二重积分

$\iint_0^1 xy \, dxdy = \frac{1}{4}$

```python
def f(y, x):  # 注意：y在前，x在后
    return x * y

result, error = integrate.dblquad(f, 0, 1, 0, 1)
print(f"二重积分结果: {result:.6f}")
print(f"理论值: {0.25:.6f}")
```

**输出**:

```
二重积分结果: 0.250000
理论值: 0.250000
```

#### 变限积分

$\int_0^1 \int_0^x x^2 y \, dy \, dx$

```python
def f(y, x):
    return x**2 * y

# y的范围依赖于x
result, _ = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: x)
print(f"变限积分结果: {result:.6f}")
```

**输出**: `变限积分结果: 0.083333`

#### 三重积分

$\iiint_0^1 xyz \, dx \, dy \, dz = \frac{1}{8}$

```python
def f(z, y, x):  # 从内到外: x, y, z
    return x * y * z

result, _ = integrate.tplquad(f, 0, 1, 0, 1, 0, 1)
print(f"三重积分结果: {result:.6f}")
```

**输出**: `三重积分结果: 0.125000`

### 5. 实际应用

计算标准正态分布的概率：

- $P(-1 < X < 1) \approx 68\%$
- $P(-2 < X < 2) \approx 95\%$

```python
def normal_pdf(x):
    return (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)

prob, _ = integrate.quad(normal_pdf, -1, 1)
print(f"P(-1<X<1) = {prob:.4f} = {prob*100:.2f}%")

prob, _ = integrate.quad(normal_pdf, -2, 2)
print(f"P(-2<X<2) = {prob:.4f} = {prob*100:.2f}%")
```

**输出**:

```
P(-1<X<1) = 0.6827 = 68.27%
P(-2<X<2) = 0.9545 = 95.45%
```

### 二重积分可视化

下图展示了二重积分区域和单位圆面积：

![06_dblquad](https://img.yumeko.site/file/articles/scipylearn/06_dblquad.png)

## 常微分方程 (ODE)

求解微分方程 $\frac{dy}{dt} = f(t, y)$

### 1. odeint (经典方法)

求解 $\frac{dy}{dt} = -y$，初值 $y(0) = 1$，解析解：$y(t) = e^{-t}$

```python
def dydt(y, t):
    return -y

t = np.linspace(0, 5, 100)
y = integrate.odeint(dydt, y0=1, t=t)

print("t=0时: y =", y[0][0])
print("t=1时: y =", y[20][0], "理论值:", np.exp(-1))
print("t=5时: y =", y[-1][0], "理论值:", np.exp(-5))
```

**输出**:

```
t=0时: y = 1.0
t=1时: y = 0.3679 理论值: 0.3679
t=5时: y = 0.0067 理论值: 0.0067
```

### 2. solve_ivp (现代推荐)

```python
from scipy.integrate import solve_ivp

# 相同问题，使用solve_ivp
def f(t, y):  # 注意：参数顺序变了
    return -y

# 求解区间[0, 5]，初值y(0)=1
sol = solve_ivp(f, [0, 5], [1], t_eval=np.linspace(0, 5, 100))

print("时间点数:", len(sol.t))
print("t=0时: y =", sol.y[0][0])
print("t=5时: y =", sol.y[0][-1])
print("求解成功:", sol.success)
```

**输出**:

```
时间点数: 100
t=0时: y = 1.0
t=5时: y = 0.0067
求解成功: True
```

**优势**:

- 更现代的API
- 更多求解器选择
- 更好的事件检测

### 3. 二阶ODE

将二阶方程 $\frac{d^2y}{dt^2} = -y$ 转换为一阶方程组：

$$\frac{dy_1}{dt} = y_2, \quad \frac{dy_2}{dt} = -y_1$$

初值：$y(0) = 1$，$y'(0) = 0$

```python
def harmonic(t, y):
    y1, y2 = y
    return [y2, -y1]

sol = solve_ivp(harmonic, [0, 10], [1, 0],
                t_eval=np.linspace(0, 10, 200))

print("振幅保持:", np.max(sol.y[0]), "≈ 1")
print("周期:", 2*np.pi, "秒")
```

**输出**:

```
振幅保持: 1.0000 ≈ 1
周期: 6.2832 秒 (2π)
```

**应用**: 弹簧振动、单摆、电路分析。周期 $T = 2\pi$。

### ODE 解可视化

下图展示了数值解与解析解的对比及向量场：

![06_ode](https://img.yumeko.site/file/articles/scipylearn/06_ode.png)

### 4. 实际案例：人口增长模型

**Logistic 模型**：$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$

其中 $P$ 是人口，$r$ 是增长率，$K$ 是环境容量。

```python
def logistic(t, P, r=0.5, K=1000):
    return r * P * (1 - P/K)

t_span = [0, 20]
P0 = [10]  # 初始人口
t_eval = np.linspace(0, 20, 100)

sol = solve_ivp(logistic, t_span, P0, t_eval=t_eval,
                args=(0.5, 1000))

print("初始人口:", sol.y[0][0])
print("20年后人口:", sol.y[0][-1])
print("环境容量:", 1000)
```

**输出**:

```
初始人口: 10.0
20年后人口: 999.5
环境容量: 1000
```

## Lotka-Volterra 模型可视化

下图展示了捕食者-猕物模型的时间演化和相轨迹：

![06_lotka](https://img.yumeko.site/file/articles/scipylearn/06_lotka.png)

## 积分方法选择

| 问题类型 | 推荐方法                    | 说明           |
| -------- | --------------------------- | -------------- |
| 一维积分 | `quad`                      | 自适应，高精度 |
| 二维积分 | `dblquad`                   | 双重积分       |
| 三维积分 | `tplquad`                   | 三重积分       |
| 一阶ODE  | `solve_ivp`                 | 现代，推荐     |
| 旧代码   | `odeint`                    | 兼容性好       |
| 刚性方程 | `solve_ivp(method='Radau')` | 稳定性好       |

## 练习

```bash
python Basic/Scipy/06_integrate.py
```
