---
title: SciPy 基础入门
date: 2026-01-14
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: SciPy 科学计算库基础入门，了解模块结构
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# SciPy 基础入门

## SciPy 模块结构

| 模块                | 功能           |
| ------------------- | -------------- |
| `scipy.constants`   | 物理和数学常数 |
| `scipy.special`     | 特殊函数       |
| `scipy.integrate`   | 数值积分       |
| `scipy.optimize`    | 优化算法       |
| `scipy.interpolate` | 插值           |
| `scipy.linalg`      | 线性代数       |
| `scipy.signal`      | 信号处理       |
| `scipy.sparse`      | 稀疏矩阵       |
| `scipy.stats`       | 统计分布       |
| `scipy.spatial`     | 空间数据       |

## 物理常数 (scipy.constants)

SciPy 提供了大量物理和数学常数，方便科学计算使用。

### 1. 数学常数

```python
from scipy import constants

print("圆周率 π =", constants.pi)
print("自然对数底 e =", constants.e)
print("黄金比例 φ =", constants.golden)
```

**输出**:

```
圆周率 π = 3.141592653589793
自然对数底 e = 2.718281828459045
黄金比例 φ = 1.618033988749895
```

### 2. 物理常数

```python
# 基本物理常数
print("光速 c =", constants.c, "m/s")
print("普朗克常数 h =", constants.h, "J·s")
print("引力常数 G =", constants.G, "m³/(kg·s²)")
print("电子质量 =", constants.m_e, "kg")
print("质子质量 =", constants.m_p, "kg")
```

**输出**:

```
光速 c = 299792458.0 m/s
普朗克常数 h = 6.62607015e-34 J·s
引力常数 G = 6.6743e-11 m³/(kg·s²)
电子质量 = 9.1093837015e-31 kg
质子质量 = 1.67262192369e-27 kg
```

**说明**: 这些常数在物理计算中非常常用，精度符合国际标准。

### 3. 单位转换

```python
# 长度单位
print("1 英里 =", constants.mile, "米")
print("1 英寸 =", constants.inch, "米")
print("1 英尺 =", constants.foot, "米")

# 时间单位
print("1 天 =", constants.day, "秒")
print("1 小时 =", constants.hour, "秒")

# 质量单位
print("1 磅 =", constants.pound, "千克")
print("1 盎司 =", constants.ounce, "千克")
```

**输出**:

```
1 英里 = 1609.344 米
1 英寸 = 0.0254 米
1 英尺 = 0.30479999999999996 米
1 天 = 86400.0 秒
1 小时 = 3600.0 秒
1 磅 = 0.45359236999999997 千克
1 盎司 = 0.028349523124999998 千克
```

**应用场景**: 国际单位换算、科学计算、工程应用。

### 4. 实用示例

```python
# 计算物体从10米高处自由落体的时间
import numpy as np

h = 10  # 高度(米)
g = constants.g  # 重力加速度
t = np.sqrt(2 * h / g)

print(f"从{h}米高处落地需要时间: {t:.2f} 秒")
```

**输出**: `从10米高处落地需要时间: 1.43 秒`

**原理**: 使用自由落体公式 $h = \frac{1}{2}gt^2$，解得 $t = \sqrt{\frac{2h}{g}}$

## 特殊函数 (scipy.special)

SciPy 提供了数百个特殊数学函数，广泛应用于统计、物理和工程领域。

### 1. 阶乘和组合数

```python
from scipy import special

# 阶乘: n! = n × (n-1) × ... × 2 × 1
print("5! =", special.factorial(5))
print("10! =", special.factorial(10))

# 组合数: C(n,k) = n!/(k!(n-k)!)
print("C(10,3) =", special.comb(10, 3))
print("C(52,5) =", special.comb(52, 5))  # 扑克牌52张抽5张
```

**输出**:

```
5! = 120.0
10! = 3628800.0
C(10,3) = 120.0
C(52,5) = 2598960.0
```

**应用**: 概率计算、排列组合问题。

### 2. 伽马函数

伽马函数是阶乘的推广，对于正整数 $n$，有 $\Gamma(n) = (n-1)!$

```python
# 伽马函数
print("Γ(5) =", special.gamma(5))     # Γ(5) = 4! = 24
print("Γ(0.5) =", special.gamma(0.5)) # Γ(0.5) = √π
print("√π =", np.sqrt(constants.pi))

# 对数伽马函数(避免溢出)
print("ln(Γ(100)) =", special.gammaln(100))
```

**输出**:

```
Γ(5) = 24.0
Γ(0.5) = 1.7724538509055159
√π = 1.7724538509055159
ln(Γ(100)) = 359.1342053695754
```

**原理**: $\Gamma(n) = \int_0^{\infty} t^{n-1}e^{-t}dt$，特别地 $\Gamma(\frac{1}{2}) = \sqrt{\pi}$

### 3. 贝塞尔函数

贝塞尔函数在波动、热传导、电磁学等领域广泛应用。

```python
# 第一类贝塞尔函数
x = np.linspace(0, 10, 50)
y0 = special.jv(0, x)  # 0阶
y1 = special.jv(1, x)  # 1阶

print("J₀(0) =", special.jv(0, 0))
print("J₀(1) =", special.jv(0, 1))
print("J₁(1) =", special.jv(1, 1))
```

**输出**:

```
J₀(0) = 1.0
J₀(1) = 0.7651976865579666
J₁(1) = 0.44005058574493355
```

**应用**: 圆形振动、波导传播、信号处理。

### 4. 误差函数

误差函数在概率论和统计学中很重要，与正态分布相关。

```python
# 误差函数: erf(x) = (2/√π)∫₀ˣ e^(-t²)dt
print("erf(0) =", special.erf(0))
print("erf(1) =", special.erf(1))
print("erf(∞) ≈", special.erf(10))

# 互补误差函数: erfc(x) = 1 - erf(x)
print("erfc(0) =", special.erfc(0))
```

**输出**:

```
erf(0) = 0.0
erf(1) = 0.8427007929497149
erf(∞) ≈ 1.0
erfc(0) = 1.0
```

**原理**: 与标准正态分布的累积分布函数关系: $\Phi(x) = \frac{1}{2}[1 + \text{erf}(\frac{x}{\sqrt{2}})]$

### 5. Logistic 函数

```python
# Sigmoid函数: f(x) = 1/(1+e^(-x))
x = np.array([-2, -1, 0, 1, 2])
y = special.expit(x)

print("x:", x)
print("sigmoid(x):", y)
```

**输出**:

```
x: [-2 -1  0  1  2]
sigmoid(x): [0.11920292 0.26894142 0.5 0.73105858 0.88079708]
```

**应用**: 逻辑回归、神经网络激活函数。

## 练习

```bash
python Basic/Scipy/01_basics.py
```
