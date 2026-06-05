---
title: SciPy 全指南
date: 2026-01-17
category: 机器学习/基础
tags:
  - Python
  - 基础
description: SciPy 库完整学习指南，涵盖常数、统计、优化、插值、积分、线性代数、信号处理、稀疏矩阵和空间计算等核心模块。
image: https://img.yumeko.site/file/blog/cover/1780581871582.webp
status: published
---

# SciPy 概览

## 本章目标

1. 了解 SciPy 的整体模块结构与各子模块的功能定位
2. 掌握 `scipy.constants` 中物理常数与单位换算的使用
3. 掌握 `scipy.special` 中常用特殊函数（阶乘、组合数、伽马、贝塞尔）
4. 会查询 SciPy 与 NumPy 的版本信息

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `scipy.constants.pi` / `.c` / `.h` / `.k` / `.N_A` / `.e` | 常量 | 数学与物理常数 |
| `scipy.constants.mile` / `.inch` / `.pound` | 常量 | 单位换算因子 |
| `special.factorial(...)` | 函数 | 阶乘 $n!$ |
| `special.comb(...)` | 函数 | 组合数 $C(n, k)$ |
| `special.perm(...)` | 函数 | 排列数 $P(n, k)$ |
| `special.gamma(...)` | 函数 | 伽马函数 $\Gamma(z)$ |
| `special.jv(...)` | 函数 | 第一类贝塞尔函数 $J_v(x)$ |
| `scipy.__version__` | 属性 | SciPy 版本号 |

## 1. SciPy 模块总览

SciPy 不是单一模块，而是**子模块集合**——按需导入，各子模块相对独立：

| 子模块 | 功能定位 | 对应章节 |
|---|---|---|
| `scipy.constants` | 数学/物理常数、单位换算 | ch01（本章） |
| `scipy.special` | 特殊数学函数（伽马、贝塞尔等） | ch01（本章） |
| `scipy.stats` | 概率分布、描述统计、假设检验 | ch02、ch03 |
| `scipy.optimize` | 曲线拟合、求根、最优化、线性规划 | ch04 |
| `scipy.interpolate` | 一维/多维插值、样条、RBF | ch05 |
| `scipy.integrate` | 数值积分、常微分方程 | ch06 |
| `scipy.linalg` | 线性代数（LU/QR/SVD/特征值） | ch07 |
| `scipy.signal` | 信号处理（滤波、卷积、FFT） | ch08 |
| `scipy.sparse` | 稀疏矩阵（CSR/CSC/COO） | ch09 |
| `scipy.spatial` | 空间数据（KDTree/凸包/Voronoi） | ch10 |

#### 理解重点

- 导入子模块而非顶层：`from scipy import optimize`，不要 `import scipy; scipy.optimize.minimize(...)`
- `scipy.linalg` 比 `numpy.linalg` 更全面，部分函数效率更高
- 各子模块依赖 NumPy 但不互相依赖——只学需要的部分即可

## 2. 物理常数与单位换算

### `scipy.constants` — 物理常数

#### 作用

提供 CODATA 标准物理常数和数学常数。所有常数都是**标量浮点数**，可直接参与 NumPy 运算。

#### 常用物理常数

| 名称 | 含义 | 数值（SI 单位） |
|---|---|---|
| `constants.pi` | 圆周率 $\pi$ | `3.141592653589793` |
| `constants.c` | 真空光速 | `299792458.0` m/s |
| `constants.h` | 普朗克常数 | `6.62607015e-34` J·s |
| `constants.hbar` | 约化普朗克常数 $\hbar$ | `h / (2\pi)` |
| `constants.k` | 玻尔兹曼常数 | `1.380649e-23` J/K |
| `constants.N_A` | 阿伏伽德罗常数 | `6.02214076e+23` |
| `constants.G` | 引力常数 | `6.67430e-11` |
| `constants.g` | 重力加速度 | `9.80665` m/s² |
| `constants.e` | 基本电荷（**非欧拉数！**） | `1.602176634e-19` C |
| `constants.R` | 理想气体常数 | `8.31446261815324` |

### `scipy.constants` — 单位换算

#### 常用单位→SI 换算因子

| 名称 | 含义 | 换算值 |
|---|---|---|
| `constants.mile` | 英里 → 米 | `1609.344` |
| `constants.inch` | 英寸 → 米 | `0.0254` |
| `constants.foot` | 英尺 → 米 | `0.3048` |
| `constants.pound` | 磅 → 千克 | `0.45359237` |
| `constants.minute` | 分钟 → 秒 | `60.0` |
| `constants.hour` | 小时 → 秒 | `3600.0` |
| `constants.degree` | 度 → 弧度 | $\pi / 180$ |

### 综合示例

#### 示例代码

```python
from scipy import constants

print(f"π = {constants.pi}")
print(f"光速 c = {constants.c} m/s")
print(f"普朗克 h = {constants.h} J·s")
print(f"玻尔兹曼 k = {constants.k} J/K")
print(f"阿伏伽德罗 N_A = {constants.N_A}")

print(f"\n1 英里 = {constants.mile} 米")
print(f"1 英寸 = {constants.inch} 米")
print(f"1 磅 = {constants.pound} 千克")

# 搜索与查找
print(f"\n查找 Planck 相关常数:")
for key in constants.find("Planck"):
    print(f"  {key}: {constants.value(key)}")
```

#### 输出

```text
π = 3.141592653589793
光速 c = 299792458.0 m/s
普朗克 h = 6.62607015e-34 J·s
玻尔兹曼 k = 1.380649e-23 J/K
阿伏伽德罗 N_A = 6.02214076e+23

1 英里 = 1609.344 米
1 英寸 = 0.0254 米
1 磅 = 0.45359237 千克

查找 Planck 相关常数:
  Planck length: 1.616255e-35
  Planck mass: 2.176434e-08
  Planck temperature: 1.416784e+32
  Planck time: 5.391247e-44
```

#### 理解重点

- 全部常数按 **SI 单位制**给出——物理计算可直接相乘除
- `constants.find('planck')` 搜索相关常数；`constants.value('speed of light in vacuum')` 按 CODATA 标准名查找
- **`constants.e` 是基本电荷，不是欧拉数！** 欧拉数用 `math.e` 或 `numpy.e`

## 3. 特殊函数

### `special.factorial`

#### 作用

计算阶乘 $n! = 1 \times 2 \times \cdots \times n$。`exact=True` 返回 Python 整数精确值，默认返回浮点数（大 n 会溢出）。

#### 重点方法

```python
special.factorial(n, exact=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int`、`array_like` | 非负整数 | `5`、`[1, 2, 3]` |
| `exact` | `bool` | `True` 返回精确整数，`False` 返回浮点数，默认为 `False` | `True` |

### `special.comb`

#### 作用

计算组合数 $C(N, k) = \binom{N}{k} = \frac{N!}{k!(N-k)!}$。支持 `exact=True` 精确整数和 `repetition=True` 允许重复选择。

#### 重点方法

```python
special.comb(N, k, exact=False, repetition=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int`、`array_like` | 总元素数 | `10` |
| `k` | `int`、`array_like` | 选取数 | `3` |
| `exact` | `bool` | `True` 返回精确整数，默认为 `False` | `True` |
| `repetition` | `bool` | `True` 允许重复选择（$C(N+k-1, k)$），默认为 `False` | `True` |

### `special.perm`

#### 作用

计算排列数 $P(N, k) = \frac{N!}{(N-k)!}$。

#### 重点方法

```python
special.perm(N, k, exact=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int`、`array_like` | 总元素数 | `10` |
| `k` | `int`、`array_like` | 选取数 | `3` |
| `exact` | `bool` | `True` 返回精确整数，默认为 `False` | `True` |

### `special.gamma`

#### 作用

计算伽马函数 $\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$。对正整数 $n$ 有 $\Gamma(n) = (n-1)!$。支持复数输入。

#### 重点方法

```python
special.gamma(z)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `z` | `array_like` | 实数或复数输入 | `5`、`0.5`、`[1, 2, 3]` |

### `special.jv`

#### 作用

计算第一类贝塞尔函数 $J_v(x)$，是贝塞尔微分方程的正则解。$v$ 为阶数，$x$ 为自变量（可为复数）。

#### 重点方法

```python
special.jv(v, x)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `v` | `array_like` | 贝塞尔函数的阶数（可为浮点） | `0`、`1.5` |
| `x` | `array_like` | 自变量（实数或复数） | `1.0`、`[0, 1, 2]` |

### 其他常用特殊函数速览

| 函数 | 含义 | 示例 |
|---|---|---|
| `special.gammaln(z)` | $\ln\Gamma(z)$，避免溢出 | `special.gammaln(1000)` |
| `special.beta(a, b)` | 贝塔函数 $B(a, b)$ | `special.beta(2, 3)` |
| `special.yv(v, x)` | 第二类贝塞尔函数 $Y_v(x)$ | `special.yv(0, 1)` |
| `special.erf(x)` | 误差函数 $\operatorname{erf}(x)$ | `special.erf(1)` |
| `special.expit(x)` | sigmoid 函数 $\frac{1}{1+e^{-x}}$ | `special.expit(0)` |
| `special.logit(x)` | sigmoid 的逆函数 | `special.logit(0.5)` |
| `special.softmax(x)` | softmax 激活，数值稳定 | `special.softmax([1, 2, 3])` |

### 综合示例

#### 示例代码

```python
from scipy import special
import math

# 阶乘与组合数
print(f"5! = {special.factorial(5)}")
print(f"5! (精确) = {special.factorial(5, exact=True)}")
print(f"C(10, 3) = {special.comb(10, 3)}")
print(f"P(10, 3) = {special.perm(10, 3)}")

# 伽马函数
print(f"\nΓ(5) (=4!) = {special.gamma(5)}")
print(f"Γ(0.5) (=√π) = {special.gamma(0.5)}")
print(f"对比 √π = {math.sqrt(math.pi)}")

# 贝塞尔函数
print(f"\nJ_0(1) = {special.jv(0, 1):.6f}")
print(f"J_1(1) = {special.jv(1, 1):.6f}")

# 机器学习常用
print(f"\nexpit(0) = {special.expit(0)}")
print(f"softmax([1,2,3]) = {special.softmax([1, 2, 3])}")
print(f"logit(0.5) = {special.logit(0.5)}")
```

#### 输出

```text
5! = 120.0
5! (精确) = 120
C(10, 3) = 120.0
P(10, 3) = 720.0

Γ(5) (=4!) = 24.0
Γ(0.5) (=√π) = 1.7724538509055159
对比 √π = 1.7724538509055159

J_0(1) = 0.765198
J_1(1) = 0.440051

expit(0) = 0.5
softmax([1,2,3]) = [0.09003057 0.24472847 0.66524096]
logit(0.5) = 0.0
```

#### 理解重点

- `factorial(n, exact=True)` 返回整数精确值——大 n 用 `gammaln(n+1)` 避免溢出
- `special.expit` 是 sigmoid 的数值稳定实现，`special.softmax` 自动处理 overflow——机器学习中优先使用
- 贝塞尔函数 $J_v(x)$ 和 $Y_v(x)$ 是振动问题的标准解——$v$ 可为非整数

## 4. 版本查询

### `scipy.__version__`

#### 作用

SciPy 版本号字符串。配合 `numpy.__version__` 做环境诊断。

#### 示例代码

```python
import scipy
import numpy as np

print(f"SciPy 版本: {scipy.__version__}")
print(f"NumPy 版本: {np.__version__}")
```

#### 输出

```text
SciPy 版本: 1.11.4
NumPy 版本: 1.26.2
```

#### 理解重点

- SciPy 与 NumPy 有版本兼容矩阵——升级前查官方说明
- 版本号遵循 `major.minor.patch` 语义

## 常见坑

1. 不要写 `import scipy` 然后 `scipy.optimize.minimize(...)`——旧版顶层不自动导入子模块，需 `from scipy import optimize`
2. `scipy.constants.e` 是**基本电荷**（$1.6 \times 10^{-19}$ C），不是欧拉数——欧拉数用 `math.e` 或 `numpy.e`
3. `special.factorial(n)` 对大 n 会溢出（`exact=False` 时）——改用 `gammaln(n+1)` 取对数
4. `special.comb(N, k)` 同理——大数用 `exact=True` 或 `gammaln`
5. `exact=True` 返回 Python `int` 而非 NumPy 数组——批量计算时注意类型不一致

## 小结

- SciPy 是**建立在 NumPy 之上**的科学计算生态，按子模块组织——按需导入
- 常数（`constants`）+ 特殊函数（`special`）是最轻量的入口——物理计算和组合数学的基础
- 后续章节逐一深入核心子模块：统计、优化、插值、积分、线代、信号、稀疏、空间

# SciPy 统计

## 本章目标

1. 掌握 `scipy.stats` 中常见概率分布的创建与使用（正态、二项、泊松）
2. 理解分布对象的统一接口：`.pdf` / `.cdf` / `.ppf` / `.rvs`
3. 掌握描述性统计函数（变异系数、偏度、峰度、众数）
4. 理解百分位数与四分位距的计算方法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `stats.norm(loc, scale)` | 构造器 | 创建正态分布对象 |
| `stats.binom(n, p)` | 构造器 | 创建二项分布对象 |
| `stats.poisson(mu)` | 构造器 | 创建泊松分布对象 |
| `dist.pdf(x)` / `dist.pmf(k)` | 方法 | 概率密度/质量函数 |
| `dist.cdf(x)` / `dist.ppf(q)` | 方法 | 累积分布/分位数函数 |
| `dist.rvs(size)` | 方法 | 生成随机样本 |
| `stats.variation(a)` | 函数 | 变异系数 CV = $\sigma / \mu$ |
| `stats.skew(a)` | 函数 | 偏度 |
| `stats.kurtosis(a)` | 函数 | 峰度（Fisher 定义） |
| `stats.mode(a)` | 函数 | 众数 |
| `stats.norm.fit(data)` | 函数 | 最大似然估计拟合正态分布参数 |

## 1. 概率分布对象

SciPy 的 `stats` 模块提供 100+ 种概率分布，每种都是"冻结分布"对象。创建时固定参数，随后通过统一接口（`.pdf` / `.cdf` / `.ppf` / `.rvs`）操作。

### `stats.norm`

#### 作用

创建正态（高斯）分布对象。概率密度函数：

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

#### 重点方法

```python
stats.norm(loc=0, scale=1)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `loc` | `float` | 均值 $\mu$，默认为 `0` | `100` |
| `scale` | `float` | 标准差 $\sigma$，默认为 `1` | `15` |

### `stats.binom`

#### 作用

创建二项分布对象。$n$ 次独立 Bernoulli 试验中成功次数的分布。概率质量函数：

$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

#### 重点方法

```python
stats.binom(n, p)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int` | 试验次数 | `10` |
| `p` | `float` | 每次试验的成功概率 | `0.5` |

### `stats.poisson`

#### 作用

创建泊松分布对象。描述单位时间内随机事件发生次数的分布。概率质量函数：

$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

#### 重点方法

```python
stats.poisson(mu)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `mu` | `float` | 期望值 $\lambda$ | `3` |

### 分布对象的统一接口

创建分布后，以下方法对所有分布统一可用：

| 方法 | 类型 | 含义 |
|---|---|---|
| `dist.pdf(x)` | 连续 | 概率密度函数 $f(x)$ |
| `dist.pmf(k)` | 离散 | 概率质量函数 $P(X=k)$ |
| `dist.cdf(x)` | 全部 | 累积分布 $P(X \le x)$ |
| `dist.ppf(q)` | 全部 | 分位数函数（CDF 逆函数） |
| `dist.rvs(size)` | 全部 | 生成随机样本 |
| `dist.mean()` | 全部 | 理论均值 |
| `dist.var()` | 全部 | 理论方差 |

### 综合示例

#### 示例代码

```python
from scipy import stats

# 正态分布
norm = stats.norm(loc=0, scale=1)
print(f"pdf(0): {norm.pdf(0):.4f}")
print(f"cdf(0): {norm.cdf(0):.4f}")
print(f"ppf(0.95): {norm.ppf(0.95):.4f}")
print(f"rvs(5): {norm.rvs(size=5)}")

# 二项分布
binom = stats.binom(n=10, p=0.5)
print(f"\npmf(5): {binom.pmf(5):.4f}")
print(f"cdf(5): {binom.cdf(5):.4f}")
print(f"mean: {binom.mean()}, var: {binom.var()}")

# 泊松分布
poisson = stats.poisson(mu=3)
print(f"\npmf(3): {poisson.pmf(3):.4f}")
print(f"mean: {poisson.mean()}")
```

#### 输出

```text
pdf(0): 0.3989
cdf(0): 0.5000
ppf(0.95): 1.6449
rvs(5): [ 0.4967 -0.1383  0.6477  1.5230 -0.2342]

pmf(5): 0.2461
cdf(5): 0.6230
mean: 5.0, var: 2.5

pmf(3): 0.2240
mean: 3.0
```

#### 理解重点

- `norm.pdf(0) = 0.3989 = 1/\sqrt{2\pi}$——标准正态在均值处的密度值
- `norm.ppf(0.95) = 1.6449` 是 95% 单侧置信区间的常用分位数
- 二项分布 $B(10, 0.5)$ 的期望值 = $np = 5$，方差 = $np(1-p) = 2.5$
- 泊松分布的期望值和方差都等于 $\lambda$
- 连续分布用 `.pdf()`，离散分布用 `.pmf()`——不可混用

## 2. 描述性统计

### `stats.variation`

#### 作用

计算变异系数 $CV = \sigma / \mu$，消除量纲影响后比较不同数据集的离散程度。

#### 重点方法

```python
stats.variation(a)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `np.random.normal(100, 15, 100)` |

### `stats.skew`

#### 作用

计算偏度——衡量分布的不对称程度和方向。0 为对称，正值右偏（右尾更长），负值左偏。

#### 重点方法

```python
stats.skew(a)
```

### `stats.kurtosis`

#### 作用

计算峰度——衡量分布的"尖峭"程度。默认使用 Fisher 定义（正态分布峰度 = 0），正值比正态更尖峭，负值更平坦。

#### 重点方法

```python
stats.kurtosis(a, fisher=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `data` |
| `fisher` | `bool` | `True` Fisher 定义（正态=0），`False` Pearson 定义（正态=3），默认为 `True` | `False` |

### `stats.mode`

#### 作用

计算数据中出现频率最高的值（众数）。需指定 `keepdims=True`（新版 SciPy 要求）。

#### 重点方法

```python
stats.mode(a, keepdims=True)
```

### `stats.norm.fit`

#### 作用

从数据中通过最大似然估计拟合正态分布参数，返回 $(\hat{\mu}, \hat{\sigma})$。

#### 重点方法

```python
stats.norm.fit(data)
```

### 综合示例

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)
data = np.random.normal(100, 15, 100)

# 集中趋势
print(f"均值: {np.mean(data):.2f}")
print(f"中位数: {np.median(data):.2f}")
print(f"众数: {stats.mode(data.astype(int), keepdims=True)[0][0]}")

# 离散程度
print(f"标准差: {np.std(data):.2f}")
print(f"变异系数: {stats.variation(data):.4f}")

# 分布形态
print(f"偏度: {stats.skew(data):.4f}")
print(f"峰度: {stats.kurtosis(data):.4f}")

# MLE 拟合
muHat, sigmaHat = stats.norm.fit(data)
print(f"MLE 拟合: N({muHat:.1f}, {sigmaHat:.1f}²)")
```

#### 输出

```text
均值: 99.73
中位数: 100.19
众数: 86
标准差: 14.03
变异系数: 0.1406
偏度: -0.1442
峰度: -0.2058
MLE 拟合: N(99.7, 14.0²)
```

#### 理解重点

- 变异系数 0.14 表示标准差约为均值的 14%——无量纲，可跨数据集比较
- 偏度 ≈ -0.14 接近 0——分布近似对称，与正态假设一致
- 峰度 ≈ -0.21 接近 0——分布形态接近正态（Fisher 定义下正态峰度=0）
- `stats.norm.fit(data)` 返回 $(\hat{\mu}, \hat{\sigma})$，不是 $(\mu, \sigma^2)$

## 3. 百分位数与四分位距

### `np.percentile`

#### 作用

计算数据的第 $q$ 百分位数。四分位距 $IQR = Q3 - Q1$ 是稳健的离散度指标（不受极端值影响）。

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数据 | `data` |
| `q` | `float`、`list[float]` | 百分位数 (0-100) | `[25, 50, 75, 90, 95]` |

#### 示例代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, 100)

for p in [25, 50, 75, 90, 95]:
    print(f"P{p}: {np.percentile(data, p):.2f}")

q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
print(f"\nQ1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
```

#### 输出

```text
P25: 90.47
P50: 100.19
P75: 109.07
P90: 117.35
P95: 123.64

Q1=90.47, Q3=109.07, IQR=18.60
```

#### 理解重点

- P50 = 中位数 ≈ 100.19，接近理论值 $\mu=100$
- IQR ≈ 18.60，理论值 $2 \times 0.6745\sigma \approx 20.24$，样本值合理
- 异常值检测规则：$< Q1 - 1.5 \times IQR$ 或 $> Q3 + 1.5 \times IQR$ 的点视为异常
- 百分位数比均值/标准差更稳健——常用于箱线图和风险分析（VaR）

## 常见坑

1. 连续分布用 `.pdf()`，离散分布用 `.pmf()`——调错会报 `AttributeError`
2. `stats.mode` 新版要求显式传 `keepdims=True`——否则会有 `DeprecationWarning`
3. `stats.kurtosis` 默认 Fisher 定义（正态=0），Pearson 定义需 `fisher=False`（正态=3）
4. `stats.norm.fit` 返回 `(μ, σ)` 不是 `(μ, σ²)`——标准差不是方差
5. `dist.rvs()` 每次调用结果不同——复现需提前 `np.random.seed(seed)`

## 小结

- `scipy.stats` 提供 100+ 概率分布，统一接口 `pdf/pmf` → `cdf` → `ppf` → `rvs`
- 描述性统计从三个维度刻画数据：集中趋势、离散程度、分布形态
- 百分位数和 IQR 是稳健统计量——不受极端值影响，广泛用于数据分析和异常检测

# SciPy 假设检验

## 本章目标

1. 掌握三种 t 检验的适用场景与使用方法（单样本、独立、配对）
2. 理解卡方检验在拟合优度和独立性检验中的应用
3. 学会使用单因素方差分析（ANOVA）比较多组均值
4. 了解 Mann-Whitney U 和 Wilcoxon 等非参数检验方法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `stats.ttest_1samp(a, popmean)` | 函数 | 单样本 t 检验 |
| `stats.ttest_ind(a, b)` | 函数 | 独立双样本 t 检验 |
| `stats.ttest_rel(a, b)` | 函数 | 配对 t 检验 |
| `stats.chisquare(f_obs, f_exp)` | 函数 | 卡方拟合优度检验 |
| `stats.chi2_contingency(observed)` | 函数 | 卡方独立性检验（列联表） |
| `stats.f_oneway(*groups)` | 函数 | 单因素方差分析 |
| `stats.mannwhitneyu(x, y)` | 函数 | Mann-Whitney U 非参数检验 |
| `stats.wilcoxon(x, y)` | 函数 | Wilcoxon 符号秩检验 |

所有检验函数返回 `(statistic, pvalue)` 元组。p < 0.05 通常在 $\alpha=0.05$ 水平拒绝原假设。

## 1. t 检验

### `stats.ttest_1samp` / `stats.ttest_ind` / `stats.ttest_rel`

#### 作用

三种 t 检验覆盖不同的实验设计：

- **单样本** `ttest_1samp`：检验样本均值是否等于某个假设值 $H_0: \mu = \mu_0$
- **独立双样本** `ttest_ind`：检验两个独立样本的均值是否相等 $H_0: \mu_1 = \mu_2$
- **配对** `ttest_rel`：检验配对样本（前后测）的均值差是否为零 $H_0: \mu_d = 0$

#### 重点方法

```python
stats.ttest_1samp(a, popmean)
stats.ttest_ind(a, b, equal_var=True)
stats.ttest_rel(a, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 第一组样本数据 | `np.random.normal(105, 15, 30)` |
| `popmean` | `float` | 单样本检验的假设总体均值 | `100` |
| `b` | `array_like` | 第二组样本数据 | `np.random.normal(110, 15, 30)` |
| `equal_var` | `bool` | 是否假设等方差；`False` 使用 Welch's t 检验，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# 单样本 t 检验：H0: μ = 100
sample = np.random.normal(105, 15, 30)
t1, p1 = stats.ttest_1samp(sample, 100)
print(f"单样本: t={t1:.4f}, p={p1:.4f}, 均值={sample.mean():.2f}")

# 独立样本 t 检验
g1 = np.random.normal(100, 15, 30)
g2 = np.random.normal(110, 15, 30)
t2, p2 = stats.ttest_ind(g1, g2)
print(f"独立双样本: t={t2:.4f}, p={p2:.4f}")
print(f"  组1均值={g1.mean():.2f}, 组2均值={g2.mean():.2f}")

# 配对 t 检验
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
t3, p3 = stats.ttest_rel(before, after)
print(f"配对: t={t3:.4f}, p={p3:.4f}")
print(f"  前测均值={before.mean():.2f}, 后测均值={after.mean():.2f}")
```

#### 输出

```text
单样本: t=2.6789, p=0.0122, 均值=107.49
独立双样本: t=-2.6961, p=0.0093
  组1均值=97.71, 组2均值=108.79
配对: t=-6.5025, p=0.0000
  前测均值=98.60, 后测均值=103.66
```

#### 理解重点

- 单样本 p=0.012 < 0.05，拒绝 $H_0$，样本均值显著不等于 100
- 独立双样本 p=0.009 < 0.05，两组均值存在显著差异
- 配对 p ≈ 0，效果最显著——配对设计通过消除个体差异，检验力更强
- t 统计量的符号反映方向：正值表示样本均值 > 假设值

## 2. 卡方检验

### `stats.chisquare` / `stats.chi2_contingency`

#### 作用

- **拟合优度检验** `chisquare`：检验观察频数是否符合期望分布
- **独立性检验** `chi2_contingency`：检验两个分类变量是否独立

卡方统计量：$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$，期望频数每格应 ≥ 5。

#### 重点方法

```python
stats.chisquare(f_obs, f_exp=None)
stats.chi2_contingency(observed)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f_obs` | `array_like` | 观察频数 | `[45, 35, 20]` |
| `f_exp` | `array_like` 或 `None` | 期望频数；`None` 时默认为均匀分布 | `[40, 40, 20]` |
| `observed` | `array_like` | 列联表（二维数组），行=变量1，列=变量2 | `[[30, 20], [25, 25]]` |

#### 示例代码

```python
import numpy as np
from scipy import stats

# 拟合优度检验
observed = np.array([45, 35, 20])
expected = np.array([40, 40, 20])
chi2, p1 = stats.chisquare(observed, f_exp=expected)
print(f"拟合优度: χ²={chi2:.4f}, p={p1:.4f}")

# 独立性检验
ct = np.array([[30, 20], [25, 25]])
chi2i, p2, dof, exp = stats.chi2_contingency(ct)
print(f"独立性: χ²={chi2i:.4f}, p={p2:.4f}, 自由度={dof}")
```

#### 输出

```text
拟合优度: χ²=0.9375, p=0.6256
独立性: χ²=0.6494, p=0.4204, 自由度=1
```

#### 理解重点

- 拟合优度 p=0.626 > 0.05，不能拒绝 $H_0$——观察频数与期望频数无显著差异
- 独立性 p=0.420 > 0.05，两个分类变量之间无显著关联
- 自由度 = (行数−1)×(列数−1)，2×2 列联表自由度为 1
- `chi2_contingency` 返回 4 个值：`(χ², p, dof, expected)`——其中 `expected` 是期望频数矩阵

## 3. 单因素方差分析

### `stats.f_oneway`

#### 作用

检验 3 组或更多组的均值是否全部相等。$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$，$H_1$：至少有一组不同。F = 组间方差 / 组内方差。

#### 重点方法

```python
stats.f_oneway(*groups)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*groups` | `array_like`（可变参数） | 各组样本数据，每组为一个数组 | `group1, group2, group3` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)
g1 = np.random.normal(100, 10, 20)
g2 = np.random.normal(105, 10, 20)
g3 = np.random.normal(110, 10, 20)

fStat, p = stats.f_oneway(g1, g2, g3)
print(f"组均值: {g1.mean():.2f}, {g2.mean():.2f}, {g3.mean():.2f}")
print(f"F={fStat:.4f}, p={p:.4f}")
```

#### 输出

```text
组均值: 99.48, 103.82, 111.36
F=7.5090, p=0.0013
```

#### 理解重点

- F=7.51, p=0.0013 < 0.05，拒绝 $H_0$——至少有一组均值与其他组不同
- ANOVA 只告诉你"存在差异"，不告诉"哪两组有差异"——需要事后检验（如 Tukey HSD）
- F 统计量越大说明组间差异相对组内差异越大
- 要求各组近似正态、方差齐性——违反时考虑非参数方法（Kruskal-Wallis）

## 4. 非参数检验

### `stats.mannwhitneyu` / `stats.wilcoxon`

#### 作用

不假设数据服从特定分布，适用于数据不满足正态性假设的情况：

- **Mann-Whitney U**：独立样本 t 检验的非参数替代，基于秩的比较
- **Wilcoxon 符号秩检验**：配对 t 检验的非参数替代，基于差值的秩

#### 重点方法

```python
stats.mannwhitneyu(x, y, alternative='two-sided')
stats.wilcoxon(x, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 第一组数据 | `np.random.normal(100, 15, 20)` |
| `y` | `array_like` | 第二组数据 | `np.random.normal(110, 15, 20)` |
| `alternative` | `str` | 备择假设方向：`'two-sided'` / `'less'` / `'greater'`，默认为 `'two-sided'` | `'greater'` |

#### 示例代码

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Mann-Whitney U
x1 = np.random.normal(100, 15, 20)
x2 = np.random.normal(110, 15, 20)
uStat, p1 = stats.mannwhitneyu(x1, x2)
print(f"Mann-Whitney U: U={uStat:.0f}, p={p1:.4f}")

# Wilcoxon
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(5, 3, 20)
wStat, p2 = stats.wilcoxon(before, after)
print(f"Wilcoxon: 统计量={wStat:.0f}, p={p2:.4f}")
```

#### 输出

```text
Mann-Whitney U: U=108, p=0.0107
Wilcoxon: 统计量=3, p=0.0000
```

#### 理解重点

- Mann-Whitney U p=0.011 < 0.05，两组分布存在显著差异
- Wilcoxon p ≈ 0，配对样本的前后差异极其显著
- 非参数检验更稳健但统计效力通常略低于参数检验
- 样本量小或数据明显偏态时优先使用非参数方法

## 常见坑

1. p 值不是效应大小——p < 0.05 只说明差异"显著"，不代表差异"大"，大样本下微小差异也能显著
2. `ttest_ind` 默认 `equal_var=True`——若方差不齐需设 `False`（Welch's t 检验）
3. 卡方检验每格期望频数应 ≥ 5——否则使用 Fisher 精确检验
4. ANOVA 只判断"是否有差异"——不告诉具体哪两组不同，需要事后多重比较
5. 多次检验会膨胀 I 类错误率——需 Bonferroni 等校正

## 小结

- t 检验比较均值：单样本 vs 假设值、独立双样本、配对前后测——三种场景三种函数
- 卡方检验处理分类数据：拟合优度（观察 vs 期望）和独立性（列联表）
- ANOVA 是 t 检验在多组场景的推广——检验多组均值是否全部相等
- 非参数检验（Mann-Whitney U / Wilcoxon）在数据不满足正态假设时使用，更稳健
- 假设检验核心流程：建立 $H_0$ → 选择检验方法 → 计算统计量和 p 值 → 根据 $\alpha$ 决策

# SciPy 优化

## 本章目标

1. 掌握 `curve_fit` 进行非线性曲线拟合
2. 学会使用 `brentq` 和 `fsolve` 求解方程与方程组
3. 理解一维和多维最小化方法（`minimize_scalar` / `minimize`）
4. 了解线性规划 `linprog` 的问题建模与求解

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `optimize.curve_fit(f, xdata, ydata)` | 函数 | 非线性最小二乘曲线拟合 |
| `optimize.brentq(f, a, b)` | 函数 | 区间求根（Brent 方法，一维） |
| `optimize.fsolve(func, x0)` | 函数 | 方程组求根（多维） |
| `optimize.minimize_scalar(fun)` | 函数 | 一维标量函数最小化 |
| `optimize.minimize(fun, x0, method)` | 函数 | 多维函数最小化 |
| `optimize.linprog(c, A_ub, b_ub)` | 函数 | 线性规划 |

## 1. 曲线拟合

### `optimize.curve_fit`

#### 作用

使用非线性最小二乘法将自定义模型函数拟合到数据。返回 `(popt, pcov)`：最优参数和协方差矩阵。参数标准误 = `np.sqrt(np.diag(pcov))`。

模型函数的第一个参数必须是自变量 x，后续参数为待拟合参数。

#### 重点方法

```python
optimize.curve_fit(f, xdata, ydata, p0=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f` | `callable` | 模型函数，签名为 `f(x, *params)` | `lambda x, a, b, c: a*x**2 + b*x + c` |
| `xdata` | `array_like` | 自变量数据 | `np.linspace(0, 10, 50)` |
| `ydata` | `array_like` | 因变量数据 | 含噪声的二次函数值 |
| `p0` | `array_like` 或 `None` | 参数初始猜测；`None` 时默认为全 1 | `[1, 1, 1]` |

#### 示例代码

```python
import numpy as np
from scipy import optimize

def model(x, a, b, c):
    return a * x**2 + b * x + c

np.random.seed(42)
xData = np.linspace(0, 10, 50)
yData = 2 * xData**2 + 3 * xData + 5 + np.random.normal(0, 5, 50)

params, cov = optimize.curve_fit(model, xData, yData)
print(f"真实参数: a=2, b=3, c=5")
print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.4f}")
print(f"标准误: {np.sqrt(np.diag(cov))}")
```

#### 输出

```text
真实参数: a=2, b=3, c=5
拟合参数: a=2.0144, b=2.8485, c=5.4753
标准误: [0.0417 0.3987 0.7717]
```

#### 理解重点

- 拟合参数 a≈2.01, b≈2.85, c≈5.48 接近真实值 (2, 3, 5)
- 标准误反映参数估计的不确定性：a 的标准误最小（0.04），c 最大（0.77）
- 高次项系数估计更精确——其对 y 的影响更大
- 复杂模型需提供合理的 `p0`，否则可能收敛到局部最优

## 2. 求根算法

### `optimize.brentq` / `optimize.fsolve`

#### 作用

- **brentq**：在区间 [a, b] 内求根，要求 f(a)·f(b) < 0，保证收敛，仅限一维
- **fsolve**：从初始点 x0 出发用牛顿类方法求根，支持多元方程组

#### 重点方法

```python
optimize.brentq(f, a, b)
optimize.fsolve(func, x0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `f` | `callable` | 一维目标函数 f(x)，须 f(a)·f(b) < 0 | `lambda x: x**2 - 4` |
| `a` | `float` | 搜索区间左端点 | `0` |
| `b` | `float` | 搜索区间右端点 | `3` |
| `func` | `callable` | 返回残差向量；多元时返回 `[r1, r2, ...]` | `lambda p: [p[0]+p[1]-3, p[0]-p[1]-1]` |
| `x0` | `array_like` | 初始猜测（一维传标量，多维传 list） | `[0, 0]` |

#### 示例代码

```python
from scipy import optimize

# 一维求根：f(x) = x² - 4
def f(x):
    return x**2 - 4

root = optimize.brentq(f, 0, 3)
print(f"brentq 求根 [0,3]: x = {root:.6f}, f(x) = {f(root):.2e}")

# fsolve 求根
root2 = optimize.fsolve(f, x0=1)[0]
print(f"fsolve 求根 (x0=1): x = {root2:.6f}")

# 多元方程组：x+y=3, x-y=1
def equations(p):
    x, y = p
    return [x + y - 3, x - y - 1]

sol = optimize.fsolve(equations, x0=[0, 0])
print(f"方程组解: x={sol[0]:.1f}, y={sol[1]:.1f}")
```

#### 输出

```text
brentq 求根 [0,3]: x = 2.000000, f(x) = 0.00e+00
fsolve 求根 (x0=1): x = 2.000000
方程组解: x=2.0, y=1.0
```

#### 理解重点

- `brentq` 精确找到 x=2（x²−4=0 的正根），残差达机器精度
- `fsolve` 不同 x0 可能找不同根——x0=−1 会找到 x=−2
- 方程组 x+y=3, x−y=1 的解为 (2, 1)
- `brentq` 适合一维且已知根区间；`fsolve` 适合多维或不知区间

## 3. 最小化

### `optimize.minimize_scalar` / `optimize.minimize`

#### 作用

- **minimize_scalar**：一维标量函数最小化，无需提供梯度
- **minimize**：多维函数最小化，需指定初始点 `x0` 和优化方法

常用方法：`'BFGS'`（拟牛顿）、`'Nelder-Mead'`（单纯形）、`'L-BFGS-B'`（支持边界约束）。

#### 重点方法

```python
optimize.minimize_scalar(fun)
optimize.minimize(fun, x0, method='BFGS')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fun` | `callable` | 目标函数 | `lambda x: (x-3)**2 + 2` |
| `x0` | `array_like` | 初始点（多维时） | `[0.0, 0.0]` |
| `method` | `str` | 优化算法：`'BFGS'` / `'Nelder-Mead'` / `'L-BFGS-B'` 等 | `'BFGS'` |

#### 示例代码

```python
import numpy as np
from scipy import optimize

# 一维最小化：f(x) = (x-3)² + 2
def f(x):
    return (x - 3)**2 + 2

r1 = optimize.minimize_scalar(f)
print(f"一维: x={r1.x:.6f}, f(x)={r1.fun:.6f}")

# 多维最小化：Rosenbrock 函数
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

r2 = optimize.minimize(rosenbrock, np.array([0.0, 0.0]), method='BFGS')
print(f"Rosenbrock: x={r2.x}, f={r2.fun:.6f}, 迭代={r2.nit}")
```

#### 输出

```text
一维: x=3.000000, f(x)=2.000000
Rosenbrock: x=[1. 1.], f=0.000000, 迭代=34
```

#### 理解重点

- (x−3)²+2 的最小值点为 x=3，最小值=2——`minimize_scalar` 精确找到
- Rosenbrock 函数的全局最小值在 (1, 1)，值为 0——其"香蕉形"山谷使优化困难
- BFGS 算法约 34 次迭代找到最优解
- 不同方法适用于不同问题：无约束用 BFGS，有界约束用 L-BFGS-B

## 4. 线性规划

### `optimize.linprog`

#### 作用

求解标准形式的线性规划问题：$\min \mathbf{c}^T \mathbf{x}$，约束 $A_{ub}\mathbf{x} \leq b_{ub}$，$A_{eq}\mathbf{x} = b_{eq}$，$x \geq 0$。

最大化问题需对目标系数取负，结果也需取负。

#### 重点方法

```python
optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                 bounds=None, method='highs')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `c` | `array_like` | 目标函数系数（最小化） | `[-2, -3]`（取负以最大化） |
| `A_ub` | `array_like` | 不等式约束左侧矩阵 | `[[1, 1], [1, 0], [0, 1]]` |
| `b_ub` | `array_like` | 不等式约束右侧向量 | `[4, 2, 3]` |
| `bounds` | `tuple` 或 `None` | 变量边界；`None` 表示 $x \geq 0$ | `(0, None)` |
| `method` | `str` | 求解算法，默认为 `'highs'` | `'highs'` |

#### 示例代码

```python
from scipy import optimize

# 最大化 z = 2x + 3y
# 约束: x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0
# linprog 求最小化，目标取负

c = [-2, -3]
A_ub = [[1, 1], [1, 0], [0, 1]]
b_ub = [4, 2, 3]

result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
print(f"最优解: x={result.x[0]:.1f}, y={result.x[1]:.1f}")
print(f"最大值: z={-result.fun:.1f}")
```

#### 输出

```text
最优解: x=1.0, y=3.0
最大值: z=11.0
```

#### 理解重点

- 最优解 (1, 3)，最大值 z = 2×1 + 3×3 = 11
- 最优解一定出现在可行域的顶点上（线性规划基本定理）
- `linprog` 返回的 `result.fun` 是最小化结果（即 −11），取负得最大值 11
- 可行域顶点: (0,0), (2,0), (2,2), (1,3), (0,3)——逐个计算 z 可验证 (1,3) 最优

## 常见坑

1. `curve_fit` 对复杂模型需提供合理的 `p0`——否则可能不收敛或收敛到局部最优
2. `brentq` 要求区间端点异号——f(a)·f(b) < 0，否则报错
3. `fsolve` 不同 `x0` 可能找到不同的根——多根情况需多次尝试不同初始值
4. `linprog` 是最小化——最大化问题必须对目标系数取负，最终结果也取负
5. `minimize` 方法选择：无约束用 BFGS，有界约束用 L-BFGS-B，非光滑用 Nelder-Mead

## 小结

- `curve_fit` 通过非线性最小二乘拟合自定义模型，返回最优参数和协方差矩阵
- `brentq`（区间法）和 `fsolve`（牛顿法）用于求解方程和方程组——各有适用场景
- `minimize_scalar` / `minimize` 覆盖从一维到多维的函数最小化——`method` 参数选择关键
- `linprog` 求解线性规划问题——关键是将实际问题转化为标准形式
- 优化问题的核心：选择合适方法 → 提供好的初始值 → 理解约束条件

# SciPy 插值

## 本章目标

1. 掌握 `interp1d` / `make_interp_spline` 进行一维插值
2. 学会使用 `splrep` / `splev` 进行三次样条插值及求导
3. 理解 `RegularGridInterpolator` 的规则网格多维插值
4. 了解 `RBFInterpolator` 径向基函数插值处理散点数据

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `interpolate.interp1d(x, y, kind)` | 构造器 | 一维插值函数（线性/三次等） |
| `interpolate.make_interp_spline(x, y, k)` | 函数 | B 样条插值（推荐替代 interp1d） |
| `interpolate.splrep(x, y, s)` | 函数 | 三次样条拟合，返回 (t, c, k) |
| `interpolate.splev(x, tck, der)` | 函数 | 计算样条值或导数 |
| `interpolate.RegularGridInterpolator(pts, vals)` | 构造器 | 规则网格多维插值 |
| `interpolate.RBFInterpolator(y, d, kernel)` | 构造器 | 径向基函数散点插值 |

## 1. 一维插值

### `interpolate.interp1d`

#### 作用

根据已知数据点构建插值函数，返回可调用对象。`kind` 参数控制插值类型：`'linear'`（线性）、`'cubic'`（三次）、`'quadratic'`（二次）等。

#### 重点方法

```python
interpolate.interp1d(x, y, kind='linear', bounds_error=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 已知数据点的 x 坐标（须严格递增） | `[0, 1, 2, 3, 4, 5]` |
| `y` | `array_like` | 已知数据点的 y 坐标 | `x**2` |
| `kind` | `str` | 插值类型：`'linear'` / `'cubic'` / `'quadratic'` 等，默认为 `'linear'` | `'cubic'` |
| `bounds_error` | `bool` | 查询点超出 x 范围时是否报错，默认为 `True` | `False` |
| `fill_value` | `str` 或 `array_like` | 超出范围时的填充值；`'extrapolate'` 允许外推 | `'extrapolate'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.array([0, 1, 2, 3, 4, 5])
y = x**2

fLin = interpolate.interp1d(x, y, kind='linear')
fCub = interpolate.interp1d(x, y, kind='cubic')

print(f"线性 f(2.5)={fLin(2.5):.4f}")
print(f"三次 f(2.5)={fCub(2.5):.4f}")
print(f"真实 2.5²={2.5**2}")
```

#### 输出

```text
线性 f(2.5)=6.5000
三次 f(2.5)=6.2500
真实 2.5²=6.25
```

#### 理解重点

- 线性插值在 x=2.5 得到 6.5（两端点中点），存在误差
- 三次插值得到 6.25——与真实值完全一致，因为二次函数被三次多项式精确拟合
- SciPy 1.10+ 推荐 `make_interp_spline` 替代 `interp1d` 进行样条插值
- 线性插值速度快但不光滑；三次插值更平滑但计算量稍大

## 2. 样条插值

### `interpolate.splrep` / `interpolate.splev`

#### 作用

`splrep` 拟合三次样条曲线，返回 `(t, c, k)` 元组（节点、系数、阶数）。`splev` 利用 tck 计算任意点的插值值或导数（`der` 参数控制导数阶数）。

$s=0$ 表示精确插值（通过所有数据点），$s > 0$ 允许平滑拟合。

#### 重点方法

```python
interpolate.splrep(x, y, s=0)
interpolate.splev(x, tck, der=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 数据点 x 坐标（须严格递增） | `[0, 1, 2, 3, 4, 5]` |
| `y` | `array_like` | 数据点 y 坐标 | `np.sin(x)` |
| `s` | `float` | 平滑因子；`0` = 精确插值，越大越平滑，默认为 `len(x)` 相关 | `0` |
| `tck` | `tuple` | `splrep` 返回的样条表示 `(节点, 系数, 阶数)` | `tck` |
| `der` | `int` | 导数阶数：`0`=值，`1`=一阶导，`2`=二阶导，默认为 `0` | `1` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.array([0, 1, 2, 3, 4, 5])
y = np.sin(x)

tck = interpolate.splrep(x, y, s=0)

xNew = np.array([0.5, 1.5, 2.5, 3.5])
yInterp = interpolate.splev(xNew, tck)
yTrue = np.sin(xNew)

for xi, yi, yt in zip(xNew, yInterp, yTrue):
    print(f"x={xi}: 插值={yi:.4f}, 真实={yt:.4f}, 误差={abs(yi-yt):.6f}")
```

#### 输出

```text
x=0.5: 插值=0.4783, 真实=0.4794, 误差=0.001133
x=1.5: 插值=0.9972, 真实=0.9975, 误差=0.000249
x=2.5: 插值=0.5989, 真实=0.5985, 误差=0.000459
x=3.5: 插值=-0.3519, 真实=-0.3508, 误差=0.001145
```

#### 理解重点

- 样条插值误差在 0.001 量级，远小于线性插值
- `splev(x, tck, der=1)` 可直接计算一阶导数
- `s=0` 要求样条精确通过所有数据点；增大 `s` 牺牲精度换取平滑度
- 样条的"分段"特性使其处理长序列数据时不会出现高次多项式的 Runge 现象

## 3. 二维插值

### `interpolate.RegularGridInterpolator`

#### 作用

用于规则网格（矩形网格）上的多维插值。输入为各轴的一维坐标和值数组，返回可调用对象用于任意查询点上的插值。替代已废弃的 `interp2d`。

#### 重点方法

```python
interpolate.RegularGridInterpolator(points, values, method='linear')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `tuple[array_like]` | 各轴的坐标元组 | `(x, y)` |
| `values` | `array_like` | 网格上的函数值，形状需与网格维度匹配 | `Z.T` |
| `method` | `str` | 插值方法：`'linear'` / `'nearest'` 等，默认为 `'linear'` | `'nearest'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

x = np.arange(0, 5)
y = np.arange(0, 5)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

interp = interpolate.RegularGridInterpolator((x, y), Z.T)

points = np.array([[1.5, 2.5], [2.5, 3.5]])
values = interp(points)

for p, v in zip(points, values):
    trueV = np.sin(p[0]) + np.cos(p[1])
    print(f"点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={trueV:.4f}")
```

#### 输出

```text
点(1.5, 2.5): 插值=0.5773, 真实=0.1955
点(2.5, 3.5): 插值=-0.3409, 真实=-0.2720
```

#### 理解重点

- 二维线性插值在粗 5×5 网格上精度有限——增加网格密度可提高精度
- `Z.T`（转置）是因为 `meshgrid` 和 `RegularGridInterpolator` 对轴顺序的约定不同
- `RegularGridInterpolator` 是 `interp2d` 的官方替代——后者已在 SciPy 1.10+ 废弃
- 支持三维及以上——传入更多轴坐标即可

## 4. 径向基函数（RBF）插值

### `interpolate.RBFInterpolator`

#### 作用

用于**非规则分布**的散点数据插值，不要求数据在网格上。核函数 `kernel` 控制插值的平滑特性。适合地理数据、气象数据等空间不规则采样场景。

#### 重点方法

```python
interpolate.RBFInterpolator(y, d, kernel='thin_plate_spline')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y` | `array_like` | 已知散点坐标，形状 `(n, d)` | `np.column_stack([x, y])` |
| `d` | `array_like` | 已知散点处的函数值，形状 `(n,)` | `np.sin(x) + np.cos(y)` |
| `kernel` | `str` | 核函数：`'thin_plate_spline'` / `'multiquadric'` / `'gaussian'` 等 | `'thin_plate_spline'` |

#### 示例代码

```python
import numpy as np
from scipy import interpolate

np.random.seed(42)

x = np.random.rand(10) * 4
y = np.random.rand(10) * 4
z = np.sin(x) + np.cos(y)

rbf = interpolate.RBFInterpolator(
    np.column_stack([x, y]), z, kernel='thin_plate_spline'
)

testPts = np.array([[1.0, 1.0], [2.0, 2.0]])
values = rbf(testPts)

for p, v in zip(testPts, values):
    trueV = np.sin(p[0]) + np.cos(p[1])
    print(f"点({p[0]}, {p[1]}): 插值={v:.4f}, 真实={trueV:.4f}")
```

#### 输出

```text
点(1.0, 1.0): 插值=1.3827, 真实=1.3818
点(2.0, 2.0): 插值=0.4932, 真实=0.4931
```

#### 理解重点

- RBF 插值精度很高——误差在 0.001 量级以内
- 薄板样条核产生全局光滑的插值曲面，适合大多数场景
- RBF 的优势：不要求数据在规则网格上，可处理任意散点分布
- 散点数据量大时计算量为 $O(n^3)$——大数据集考虑局部插值方法

## 常见坑

1. `interp1d` 默认 `bounds_error=True`——查询点超出范围会报错，可设 `fill_value='extrapolate'`
2. `interp2d` 已废弃——SciPy 1.10+ 推荐 `RegularGridInterpolator` 替代
3. `splrep` 要求 x 数据严格递增——否则报错
4. `RegularGridInterpolator` 轴顺序——`meshgrid` 生成的 Z 可能需要转置
5. RBF 大数据计算慢——`RBFInterpolator` 对 n > 数千的数据集计算缓慢

## 小结

- `interp1d` 是最基础的一维插值工具——`kind` 控制插值阶数
- `splrep` / `splev` 提供三次样条插值，支持求导——适合光滑曲线拟合
- `RegularGridInterpolator` 用于规则网格多维插值——替代已废弃的 `interp2d`
- `RBFInterpolator` 处理非规则散点数据——核函数选择影响插值特性
- 插值方法选择取决于：数据分布（规则/散乱）、精度要求、计算效率

# SciPy 积分

## 本章目标

1. 掌握 `quad` 计算定积分（有限区间与无穷区间）
2. 学会使用 `dblquad` 计算二重积分
3. 理解 `odeint` 求解一阶常微分方程
4. 了解 `odeint` 求解 ODE 方程组（Lotka-Volterra 模型）

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `integrate.quad(func, a, b)` | 函数 | 一维定积分（自适应 Gauss 求积法） |
| `integrate.dblquad(func, a, b, gfun, hfun)` | 函数 | 二重积分 |
| `integrate.odeint(func, y0, t)` | 函数 | 常微分方程数值求解 |
| `integrate.solve_ivp(fun, t_span, y0)` | 函数 | ODE 初值问题（新版推荐） |

`quad` 返回 `(result, error)`，`odeint` 返回解数组形状为 `(len(t), len(y0))`。

## 1. 定积分

### `integrate.quad`

#### 作用

使用自适应 Gauss 求积法计算定积分。支持有限区间 $[a, b]$ 和无穷区间（`a=-np.inf`、`b=np.inf`）。返回 `(result, error)`，`error` 为绝对误差上界。

#### 重点方法

```python
integrate.quad(func, a, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 被积函数 `func(x)` | `lambda x: x**2` |
| `a` | `float` | 积分下限 | `0`、`-np.inf` |
| `b` | `float` | 积分上限 | `1`、`np.inf` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# ∫₀¹ x² dx
r1, e1 = integrate.quad(lambda x: x**2, 0, 1)
print(f"∫₀¹ x² dx = {r1:.6f} (误差: {e1:.2e})")
print(f"解析解: 1/3 = {1/3:.6f}")

# ∫₀ᵠ sin(x) dx
r2, e2 = integrate.quad(np.sin, 0, np.pi)
print(f"∫₀ᵠ sin(x) dx = {r2:.6f} (解析解: 2)")

# 无穷积分 ∫e^(-x²) dx
r3, e3 = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
print(f"∫e^(-x²) dx = {r3:.6f} (解析解: √π = {np.sqrt(np.pi):.6f})")
```

#### 输出

```text
∫₀¹ x² dx = 0.333333 (误差: 3.70e-15)
解析解: 1/3 = 0.333333
∫₀ᵠ sin(x) dx = 2.000000 (解析解: 2)
∫e^(-x²) dx = 1.772454 (解析解: √π = 1.772454)
```

#### 理解重点

- 三个积分的数值解与解析解完全一致——误差在 $10^{-14} \sim 10^{-15}$ 量级
- `quad` 能自动处理无穷区间——通过变量替换将无穷积分转化为有限区间
- 高斯积分 $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$ 是概率论的基础结果
- 误差估计值远小于结果——说明数值积分的可靠性很高

## 2. 二重积分

### `integrate.dblquad`

#### 作用

计算 $\iint f(y, x) \,dy\,dx$ 形式的二重积分。**注意**：被积函数的参数顺序是 `func(y, x)`（内层积分变量在前）。y 的积分范围可以是 x 的函数，用于处理非矩形区域。

#### 重点方法

```python
integrate.dblquad(func, a, b, gfun, hfun)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 被积函数 `func(y, x)`——注意 y 在前 | `lambda y, x: x*y` |
| `a` | `float` | x 的积分下限 | `0` |
| `b` | `float` | x 的积分上限 | `1` |
| `gfun` | `callable` | y 的下限函数 `gfun(x)` | `lambda x: 0` |
| `hfun` | `callable` | y 的上限函数 `hfun(x)` | `lambda x: 2` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# 矩形区域: ∬xy dA, [0,1]×[0,2]
r1, e1 = integrate.dblquad(
    lambda y, x: x * y, 0, 1,
    lambda x: 0, lambda x: 2
)
print(f"∬xy dA = {r1:.6f} (解析解: 1)")

# 圆形区域: 单位圆面积
r2, e2 = integrate.dblquad(
    lambda y, x: 1, -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2)
)
print(f"单位圆面积 = {r2:.6f} (解析解: π = {np.pi:.6f})")
```

#### 输出

```text
∬xy dA = 1.000000 (解析解: 1)
单位圆面积 = 3.141593 (解析解: π = 3.141593)
```

#### 理解重点

- 矩形区域积分 $\int_0^1\int_0^2 xy\,dy\,dx = \frac{1}{2} \times 2 = 1$
- 圆形区域通过变积分上下限（$y = \pm\sqrt{1-x^2}$）实现非矩形区域积分
- `dblquad` 的参数顺序容易混淆：`func(y, x)` 中 y 是内层积分变量
- 单位圆面积 = π——验证了积分的正确性

## 3. 常微分方程（ODE）

### `integrate.odeint`

#### 作用

使用 LSODA 算法求解初值问题 $dy/dt = f(y, t)$。内部自适应步长，自动在刚性和非刚性方法之间切换。函数签名为 `func(y, t)`（注意 y 在前，与 `solve_ivp` 相反）。

#### 重点方法

```python
integrate.odeint(func, y0, t)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `callable` | 微分方程右端函数 `func(y, t)`，返回 `dy/dt` | `lambda y, t: -y` |
| `y0` | `array_like` | 初始条件 | `1` 或 `[10, 5]` |
| `t` | `array_like` | 求解的时间点数组——只影响输出点，不影响内部步长 | `np.linspace(0, 5, 100)` |

#### 示例代码

```python
import numpy as np
from scipy import integrate

# 一阶 ODE: dy/dt = -y, y(0) = 1  →  解析解: y = e^(-t)
def dydt(y, t):
    return -y

t = np.linspace(0, 5, 6)
y = integrate.odeint(dydt, 1, t)

print("t      数值解      解析解")
for ti, yi in zip(t, y.flatten()):
    print(f"{ti:.1f}    {yi:.6f}    {np.exp(-ti):.6f}")
```

#### 输出

```text
t      数值解      解析解
0.0    1.000000    1.000000
1.0    0.367879    0.367879
2.0    0.135335    0.135335
3.0    0.049787    0.049787
4.0    0.018316    0.018316
5.0    0.006738    0.006738
```

#### 理解重点

- dy/dt = −y 描述指数衰减——解析解 $y = e^{-t}$，数值解与解析解完全一致
- `odeint` 内部自适应步长——用户指定的 `t` 只影响输出点密度
- LSODA 算法自动在刚性和非刚性方法间切换——适应性极强
- 函数签名是 `func(y, t)`，新版 `solve_ivp` 的签名是 `fun(t, y)`——两者相反

## 4. ODE 方程组

### `integrate.odeint`（向量化）

#### 作用

`odeint` 同样可求解方程组——`func` 返回向量，`y0` 为向量。Lotka-Volterra 方程是经典的捕食者-猎物模型。

$$
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta xy \quad \text{(猎物)} \\[4pt]
\frac{dy}{dt} &= \delta xy - \gamma y \quad \text{(捕食者)}
\end{aligned}
$$

#### 示例代码

```python
import numpy as np
from scipy import integrate

alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5

def lotkaVolterra(state, t):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

t = np.linspace(0, 40, 500)
state0 = [10, 5]
solution = integrate.odeint(lotkaVolterra, state0, t)

print(f"Lotka-Volterra 模型 (α={alpha}, β={beta}, δ={delta}, γ={gamma})")
print(f"初始: 猎物={state0[0]}, 捕食者={state0[1]}")
print(f"解形状: {solution.shape}")
print(f"猎物 [t=0,20,40]: {solution[0,0]:.1f}, {solution[250,0]:.1f}, {solution[-1,0]:.1f}")
```

#### 输出

```text
Lotka-Volterra 模型 (α=1.0, β=0.1, δ=0.075, γ=1.5)
初始: 猎物=10, 捕食者=5
解形状: (500, 2)
猎物 [t=0,20,40]: 10.0, 8.5, 10.0
```

#### 理解重点

- 猎物增多 → 捕食者增多 → 猎物减少 → 捕食者减少 → 猎物增多（循环）
- 相空间轨迹呈闭合环——系统具有周期性（守恒量存在）
- `odeint` 返回形状 `(500, 2)`：`solution[:, 0]` 是猎物，`solution[:, 1]` 是捕食者
- 方程组通过向量化的 `func` 和 `y0` 实现——扩展到更高维同样简单

## 常见坑

1. `dblquad` 参数顺序：被积函数是 `func(y, x)` 不是 `func(x, y)`——内层积分变量在前
2. `odeint` 的 `func` 签名：参数顺序是 `func(y, t)`，`solve_ivp` 则是 `fun(t, y)`
3. 无穷积分的收敛性：`quad` 对不收敛积分可能返回错误结果——需检查 error 值
4. `odeint` 刚性问题：极端情况可能需要调整容差参数
5. 时间点密度：`odeint` 的 `t` 数组只影响输出，不影响内部计算步长

## 小结

- `quad` 高精度计算一维定积分——支持有限区间和无穷区间
- `dblquad` 计算二重积分——支持非矩形区域（y 的范围可以是 x 的函数）
- `odeint` 使用 LSODA 算法求解 ODE 初值问题——自适应步长和方法切换
- ODE 方程组通过向量化的 `func` 和 `y0` 实现——适用于物理、生态等多维动力系统
- 数值积分的核心：理解函数签名的参数顺序、检查误差估计、选择合适的求解器

# SciPy 线性代数

## 本章目标

1. 掌握 LU 分解及其在矩阵运算中的作用
2. 学会使用 QR 分解处理非方阵
3. 理解 SVD 奇异值分解的原理与矩阵重构
4. 掌握特征值分解及其几何意义
5. 学会使用 `linalg.solve` 求解线性方程组

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `linalg.lu(A)` | 函数 | LU 分解（A = PLU） |
| `linalg.qr(A)` | 函数 | QR 分解（A = QR） |
| `linalg.svd(A)` | 函数 | 奇异值分解（A = UΣVᴴ） |
| `linalg.eig(A)` | 函数 | 特征值与特征向量 |
| `linalg.solve(A, b)` | 函数 | 线性方程组求解（Ax = b） |

`scipy.linalg` 比 `numpy.linalg` 功能更全面，部分函数效率更高。

## 1. LU 分解

### `linalg.lu`

#### 作用

将矩阵分解为 A = PLU（置换矩阵 × 下三角 × 上三角）。P 用于行交换以保证数值稳定性（选主元策略）。LU 分解是高斯消元法的矩阵形式，分解一次后可高效求解多个右端向量的方程组。

#### 重点方法

```python
linalg.lu(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的方阵 | `[[1,2,3],[4,5,6],[7,8,10]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
P, L, U = linalg.lu(A)

print(f"原矩阵 A:\n{A}")
print(f"\nL (下三角):\n{np.round(L, 4)}")
print(f"\nU (上三角):\n{np.round(U, 4)}")
print(f"\n验证 P@L@U:\n{np.round(P @ L @ U, 4)}")
```

#### 输出

```text
原矩阵 A:
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8 10]]

L (下三角):
[[1.     0.     0.    ]
 [0.5714 1.     0.    ]
 [0.1429 0.5    1.    ]]

U (上三角):
[[7.     8.     10.    ]
 [0.     0.4286 0.2857]
 [0.     0.     0.5   ]]

验证 P@L@U:
[[ 1.  2.  3.]
 [ 4.  5.  6.]
 [ 7.  8. 10.]]
```

#### 理解重点

- P 将第 3 行移到第 1 行（选主元），保证 L 的元素绝对值 ≤ 1
- L 的对角线全为 1（单位下三角），U 的对角线是主元
- P@L@U 精确重构原矩阵 A——验证分解正确性
- LU 分解一次后可高效求解多个右端向量的方程组

## 2. QR 分解

### `linalg.qr`

#### 作用

将矩阵分解为 A = QR（正交矩阵 × 上三角矩阵）。Q 的列向量两两正交（$Q^T Q = I$），R 是上三角矩阵。适用于非方阵，常用于最小二乘问题和特征值算法。

#### 重点方法

```python
linalg.qr(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的矩阵（可为非方阵） | `[[1,2],[3,4],[5,6]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2

Q, R = linalg.qr(A)

print(f"原矩阵 A (3x2):\n{A}")
print(f"\nR (上三角):\n{np.round(R, 4)}")
print(f"\n验证 Q@R:\n{np.round(Q @ R, 4)}")
print(f"Q^T @ Q 是否≈I: {np.allclose(Q.T @ Q, np.eye(3))}")
```

#### 输出

```text
原矩阵 A (3x2):
[[1 2]
 [3 4]
 [5 6]]

R (上三角):
[[-5.9161 -7.4370]
 [ 0.0000  0.8281]
 [ 0.0000  0.0000]]

验证 Q@R:
[[1. 2.]
 [3. 4.]
 [5. 6.]]
Q^T @ Q 是否≈I: True
```

#### 理解重点

- A 是 3×2 矩阵，Q 是 3×3 正交矩阵，R 是 3×2 上三角矩阵
- R 的前 2 行是上三角，第 3 行全零（因为 A 只有 2 列）
- Q 的列向量两两正交且模为 1
- QR 分解是 Gram-Schmidt 正交化过程的矩阵形式

## 3. SVD 分解

### `linalg.svd`

#### 作用

奇异值分解将矩阵分解为 $A = U\Sigma V^H$。U 和 V 是正交矩阵，$\Sigma$ 的对角元素为奇异值（非负且递减）。SVD 适用于任意形状的矩阵——是最通用的矩阵分解方法。奇异值反映矩阵的"信息量"，用于降维、压缩和伪逆计算。

#### 重点方法

```python
linalg.svd(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的矩阵（任意形状） | `[[1,2,3],[4,5,6]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3

U, s, Vh = linalg.svd(A)

print(f"原矩阵 A (2x3):\n{A}")
print(f"\n奇异值 s: {np.round(s, 4)}")

# 重构
S = np.zeros_like(A, dtype=float)
S[:len(s), :len(s)] = np.diag(s)
reconstructed = U @ S @ Vh
print(f"\n重构 U@S@Vh:\n{np.round(reconstructed, 4)}")
```

#### 输出

```text
原矩阵 A (2x3):
[[1 2 3]
 [4 5 6]]

奇异值 s: [9.5080 0.7729]

重构 U@S@Vh:
[[1. 2. 3.]
 [4. 5. 6.]]
```

#### 理解重点

- `linalg.svd` 返回的 `s` 是一维数组（奇异值），需手动构造对角矩阵 $\Sigma$
- 奇异值 9.51 >> 0.77——矩阵的主要信息集中在第一个奇异值方向
- 保留前 k 个最大奇异值可实现矩阵的低秩近似（数据压缩）
- U@S@Vh 精确重构原矩阵——验证分解正确性

## 4. 特征值与特征向量

### `linalg.eig`

#### 作用

计算方阵的特征值和特征向量。特征方程 $Av = \lambda v$：矩阵乘以特征向量等于特征值缩放特征向量。特征值可能是复数（即使矩阵元素全为实数）。特征向量按列排列，第 i 列对应第 i 个特征值。

#### 重点方法

```python
linalg.eig(A)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 待分解的方阵 | `[[4,2],[1,3]]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

A = np.array([[4, 2], [1, 3]])

eigvals, eigvecs = linalg.eig(A)

print(f"矩阵 A:\n{A}")
print(f"\n特征值: {eigvals}")

print("\n验证 Av = λv:")
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    lhs = A @ v
    rhs = lam * v
    print(f"  λ={lam:.1f}: A@v={np.round(lhs, 4)}, λ*v={np.round(rhs, 4)}")
```

#### 输出

```text
矩阵 A:
[[4 2]
 [1 3]]

特征值: [5.+0.j 2.+0.j]

验证 Av = λv:
  λ=5.0+0.0j: A@v=[4.4721 2.2361], λ*v=[4.4721 2.2361]
  λ=2.0+0.0j: A@v=[-1.4142  1.4142], λ*v=[-1.4142  1.4142]
```

#### 理解重点

- 矩阵 A 的特征值为 5 和 2（特征方程 $\det(A-\lambda I)=0 \to \lambda^2-7\lambda+10=0$）
- Av = λv 验证成功：矩阵作用在特征向量上只改变长度不改变方向
- 特征值以复数返回（即使是实数也带 `+0.j`），使用 `.real` 提取实部
- 几何意义：特征向量是矩阵变换下方向不变的"轴"，特征值是沿该轴的缩放因子

## 5. 线性方程组求解

### `linalg.solve`

#### 作用

直接求解 Ax = b。比 `np.linalg.inv(A) @ b` 更高效和数值稳定（内部使用 LU 分解）。要求 A 是方阵且非奇异。对于超定方程组（m > n），应使用 `linalg.lstsq` 求最小二乘解。

#### 重点方法

```python
linalg.solve(A, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `A` | `array_like` | 系数矩阵（方阵） | `[[3, 1], [1, 2]]` |
| `b` | `array_like` | 右端向量 | `[9, 8]` |

#### 示例代码

```python
import numpy as np
from scipy import linalg

# 3x + y = 9, x + 2y = 8
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = linalg.solve(A, b)

print(f"方程组: 3x+y=9, x+2y=8")
print(f"解: x={x[0]:.1f}, y={x[1]:.1f}")
print(f"验证 A@x = {A @ x}")
```

#### 输出

```text
方程组: 3x+y=9, x+2y=8
解: x=2.0, y=3.0
验证 A@x = [9. 8.]
```

#### 理解重点

- 解为 x=2, y=3，代入验证：3×2+3=9, 2+2×3=8
- `linalg.solve` 比显式求逆再相乘更快且数值更稳定
- 验证 A@x = b 是检验解正确性的标准方法
- 对于大规模稀疏方程组，使用 `scipy.sparse.linalg.spsolve`

## 常见坑

1. `eig` 返回复数——即使矩阵是实数，特征值也以复数形式返回，需 `.real` 提取
2. `svd` 的 s 是一维数组——不是对角矩阵，重构时需手动构造 $\Sigma$ 矩阵
3. `solve` 要求方阵——A 必须是方阵且非奇异，非方阵用 `lstsq`
4. `scipy.linalg` vs `numpy.linalg`——SciPy 版本功能更多（如 LU 分解），且有时更快
5. 数值精度——矩阵条件数大时，分解和求解的精度会下降

## 小结

- LU 分解（A=PLU）是高斯消元的矩阵形式——用于高效求解线性方程组
- QR 分解（A=QR）产生正交矩阵和上三角矩阵——用于最小二乘和特征值算法
- SVD（$A=U\Sigma V^H$）是最通用的矩阵分解——奇异值反映信息结构，用于降维和压缩
- 特征值分解揭示矩阵的内在几何特性——特征向量是变换的不变方向
- `linalg.solve` 是求解线性方程组的首选——比求逆矩阵更高效稳定

# SciPy 信号处理

## 本章目标

1. 掌握 Butterworth 滤波器的设计与零相位滤波
2. 学会使用 `signal.convolve` 进行信号卷积运算
3. 理解 FFT（快速傅里叶变换）的频域分析方法
4. 掌握 `signal.find_peaks` 进行信号峰值检测

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `signal.butter(N, Wn, btype, fs)` | 函数 | 设计 Butterworth 滤波器 |
| `signal.filtfilt(b, a, x)` | 函数 | 零相位滤波（前后各滤一次） |
| `signal.convolve(in1, in2, mode)` | 函数 | 信号卷积 |
| `fft.fft(x)` | 函数 | 快速傅里叶变换 |
| `fft.fftfreq(n, d)` | 函数 | 生成对应的频率轴 |
| `signal.find_peaks(x, height, distance)` | 函数 | 一维信号峰值检测 |

## 1. 滤波器设计

### `signal.butter` / `signal.filtfilt`

#### 作用

`signal.butter` 设计 Butterworth 滤波器，返回系数 `(b, a)`。Butterworth 滤波器的特点是通带内频率响应最大平坦（无纹波）。`signal.filtfilt` 进行零相位滤波——前后各滤一次消除相位延迟，但不能用于实时处理。

#### 重点方法

```python
signal.butter(N, Wn, btype='low', fs=None)
signal.filtfilt(b, a, x)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int` | 滤波器阶数 | `4` |
| `Wn` | `float`、`list[float]` | 截止频率；指定 `fs` 时单位为 Hz，否则为归一化频率 (0~1) | `10` |
| `btype` | `str` | 滤波器类型：`'low'` / `'high'` / `'band'`，默认为 `'low'` | `'low'` |
| `fs` | `float` 或 `None` | 采样频率（Hz），默认为 `None` | `1000` |
| `b` / `a` | `ndarray` | 滤波器系数（`butter` 的返回值） | `b, a` |
| `x` | `array_like` | 待滤波信号 | 带噪声的正弦波 |

#### 示例代码

```python
import numpy as np
from scipy import signal

np.random.seed(42)
t = np.linspace(0, 1, 1000)
clean = np.sin(2 * np.pi * 5 * t)         # 5Hz 正弦波
noise = 0.5 * np.random.randn(len(t))
noisy = clean + noise

# Butterworth 低通滤波器（4阶，截止频率 10Hz）
b, a = signal.butter(4, 10, btype='low', fs=1000)
filtered = signal.filtfilt(b, a, noisy)

print(f"信号长度: {len(t)}, 采样率: 1000 Hz")
print(f"噪声信号标准差: {np.std(noisy):.4f}")
print(f"滤波后标准差: {np.std(filtered):.4f}")
print(f"纯信号标准差 (1/√2): {1/np.sqrt(2):.4f}")
```

#### 输出

```text
信号长度: 1000, 采样率: 1000 Hz
噪声信号标准差: 0.8956
滤波后标准差: 0.7066
纯信号标准差 (1/√2): 0.7071
```

#### 理解重点

- 5Hz 信号叠加噪声后标准差约 0.90，滤波后降至约 0.71（接近纯正弦波的 $1/\sqrt{2} \approx 0.707$）
- 截止频率 10Hz 保留了 5Hz 信号分量，滤除了大部分高频噪声
- `filtfilt` 比 `lfilter` 多一次反向滤波——消除相位延迟，但不能实时使用
- 滤波器阶数越高过渡带越陡峭，但可能引入更多振铃效应

## 2. 卷积运算

### `signal.convolve`

#### 作用

计算两个信号的卷积。滤波本质上就是信号与滤波器核的卷积。`mode` 控制输出长度和边界处理方式。

#### 重点方法

```python
signal.convolve(in1, in2, mode='full')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `in1` | `array_like` | 输入信号 | `[1, 2, 3, 4, 5]` |
| `in2` | `array_like` | 卷积核（或第二个信号） | `[1, 0, -1]` |
| `mode` | `str` | 输出模式：`'full'`（完整，长度 n1+n2−1）/ `'same'`（与 in1 等长）/ `'valid'`（无边界效应），默认为 `'full'` | `'same'` |

#### 示例代码

```python
import numpy as np
from scipy import signal

x = np.array([1, 2, 3, 4, 5])
h = np.array([1, 0, -1])  # 差分算子

yFull = signal.convolve(x, h, mode='full')
ySame = signal.convolve(x, h, mode='same')

print(f"信号 x: {x}")
print(f"核 h: {h}")
print(f"卷积 (full): {yFull}")
print(f"卷积 (same): {ySame}")
```

#### 输出

```text
信号 x: [1 2 3 4 5]
核 h: [ 1  0 -1]
卷积 (full): [ 1  2  2  2  2 -4 -5]
卷积 (same): [ 2  2  2  2 -4]
```

#### 理解重点

- 核 `[1, 0, -1]` 是差分算子——卷积结果近似反映信号的变化率
- `full` 模式输出长度 5+3−1=7，包含边界效应
- `same` 模式输出长度与输入相同（5），截取中间部分
- 卷积满足交换律和结合律：`convolve(x, h)` = `convolve(h, x)`

## 3. 傅里叶变换

### `fft.fft` / `fft.fftfreq`

#### 作用

`fft.fft` 将时域信号变换到频域，返回复数频谱。`fft.fftfreq` 生成对应的频率轴。频谱的幅度 $|Y(f)|$ 反映各频率成分的强度。FFT 是 $O(n\log n)$ 算法，远快于 DFT 的 $O(n^2)$。

#### 重点方法

```python
fft.fft(x)
fft.fftfreq(n, d=1.0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 时域信号 | 5Hz + 50Hz 混合正弦波 |
| `n` | `int` | 采样点数（`fftfreq` 的第一个参数） | `1000` |
| `d` | `float` | 采样间隔 = 1/采样率，默认为 `1.0` | `1/1000` |

#### 示例代码

```python
import numpy as np
from scipy import fft, signal

fs = 1000
t = np.linspace(0, 1, fs)
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

yf = fft.fft(sig)
xf = fft.fftfreq(len(t), 1 / fs)

# 找正频率部分的峰值
magnitude = np.abs(yf[:len(t) // 2])
peaks, _ = signal.find_peaks(magnitude, height=100)
peakFreqs = xf[:len(t) // 2][peaks]

print(f"信号: sin(2π·5t) + 0.5·sin(2π·50t)")
print(f"检测到的频率峰值: {peakFreqs} Hz")
```

#### 输出

```text
信号: sin(2π·5t) + 0.5·sin(2π·50t)
检测到的频率峰值: [ 5. 50.] Hz
```

#### 理解重点

- FFT 准确检测到两个频率成分：5Hz（幅度 1.0）和 50Hz（幅度 0.5）
- 实信号的 FFT 结果共轭对称——只需分析前 N/2 个点（正频率部分）
- 频率分辨率 = fs / N = 1000 / 1000 = 1Hz
- FFT 结合 `find_peaks` 可自动提取频率成分——广泛用于音频分析、振动诊断

## 4. 峰值检测

### `signal.find_peaks`

#### 作用

在一维信号中检测局部极大值。`height` 设置最小高度阈值，`distance` 设置相邻峰值间的最小采样点距离。返回 `(peaks, properties)`：峰值索引和属性字典（含高度、突出度、半高宽等）。

#### 重点方法

```python
signal.find_peaks(x, height=None, distance=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 待检测的一维信号 | `np.sin(x) + 噪声` |
| `height` | `float` 或 `None` | 最小峰值高度阈值，默认为 `None` | `0.5` |
| `distance` | `int` 或 `None` | 相邻峰值最小间距（采样点数），默认为 `None` | `10` |

#### 示例代码

```python
import numpy as np
from scipy import signal

np.random.seed(42)
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x) + 0.1 * np.random.randn(len(x))

peaks, props = signal.find_peaks(y, height=0.5, distance=10)

print(f"信号点数: {len(x)}")
print(f"检测到 {len(peaks)} 个峰值")
print(f"峰值位置: {peaks}")
print(f"峰值高度: {np.round(props['peak_heights'], 4)}")
```

#### 输出

```text
信号点数: 100
检测到 2 个峰值
峰值位置: [12 37]
峰值高度: [1.0675 0.9692]
```

#### 理解重点

- sin(x) 在 [0, 4π] 有 2 个正峰值——`find_peaks` 全部检测到
- `height=0.5` 过滤掉了小于 0.5 的峰值（负峰值不会被检测到）
- `distance=10` 确保峰值间至少间隔 10 个采样点——避免噪声导致的虚假峰
- 找极小值需对信号取负：`find_peaks(-y)`
- `props` 字典还包含 `prominences`（突出度）和 `widths`（半高宽）

## 常见坑

1. `filtfilt` vs `lfilter`：`filtfilt` 零相位但不能实时使用；`lfilter` 有相位延迟但支持在线处理
2. 滤波器截止频率单位：指定 `fs` 时 `Wn` 单位为 Hz；不指定时 `Wn` 是归一化频率（0~1，1 对应奈奎斯特频率）
3. FFT 频谱对称性：实信号的 FFT 结果共轭对称——只需分析前 N/2 个点
4. `find_peaks` 只找极大值：找极小值需 `find_peaks(-y)`
5. 卷积 `mode` 选择：`'full'` 有边界效应，`'same'` 截断可能丢失信息，`'valid'` 最短但无边界问题

## 小结

- Butterworth 滤波器通带最大平坦——`filtfilt` 实现零相位滤波
- `signal.convolve` 计算信号卷积——`mode` 控制输出长度和边界处理
- FFT 将时域信号变换到频域——结合 `find_peaks` 可自动提取频率成分
- `find_peaks` 通过 `height` 和 `distance` 参数灵活控制峰值检测灵敏度
- 信号处理核心流程：时域观察 → 频域分析 → 滤波/特征提取 → 验证

# SciPy 稀疏矩阵

## 本章目标

1. 掌握 CSR 和 COO 格式的稀疏矩阵创建方法
2. 学会稀疏矩阵的基本运算与格式转换
3. 理解稀疏线性代数求解器 `spsolve` 的使用
4. 了解稀疏矩阵在内存效率上的优势

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `sparse.csr_matrix(data)` | 构造器 | 创建 CSR 格式（压缩行）稀疏矩阵 |
| `sparse.coo_matrix((data, (row, col)))` | 构造器 | 创建 COO 格式（坐标）稀疏矩阵 |
| `sparse.random(m, n, density)` | 函数 | 创建指定密度的随机稀疏矩阵 |
| `sparse.diags(diagonals, offsets, shape)` | 函数 | 创建对角稀疏矩阵 |
| `sparse.linalg.spsolve(A, b)` | 函数 | 稀疏线性方程组求解 |

CSR 适合计算（行切片、矩阵乘法），COO 适合构建（逐元素添加）。

## 1. 稀疏矩阵创建

### `sparse.csr_matrix` / `sparse.coo_matrix`

#### 作用

- **CSR（Compressed Sparse Row）**：按行压缩存储，存储三个数组：`data`（非零值）、`indices`（列索引）、`indptr`（行指针）。适合行切片和矩阵-向量乘法
- **COO（Coordinate）**：坐标格式，用 `(row, col, data)` 三元组存储。适合构建稀疏矩阵，构建完成后转为 CSR 计算更高效

#### 重点方法

```python
sparse.csr_matrix(arg1, shape=None)
sparse.coo_matrix((data, (row, col)), shape=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg1` | `array_like` | 密集矩阵或其他稀疏数据 | `np.eye(4)` |
| `data` | `array_like` | 非零元素值（COO 的第一参数） | `[1, 2, 3, 4]` |
| `row` | `array_like` | 非零元素行索引 | `[0, 1, 2, 3]` |
| `col` | `array_like` | 非零元素列索引 | `[0, 1, 2, 3]` |
| `shape` | `tuple[int, int]` | 矩阵形状；可省略（从索引推断） | `(4, 4)` |

#### 示例代码

```python
import numpy as np
from scipy import sparse

# 从密集矩阵创建 CSR
dense = np.diag([1, 2, 3, 4])
csr = sparse.csr_matrix(dense)

print(f"密集矩阵:\n{dense}")
print(f"\nCSR 非零元素: {csr.data}")
print(f"CSR 列索引: {csr.indices}")

# 以 COO 格式手动创建
row = np.array([0, 1, 2, 3])
col = np.array([0, 1, 2, 3])
data = np.array([1, 2, 3, 4])
coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))

print(f"\nCOO 格式:\n{coo}")
print(f"COO 非零元素: {coo.nnz}")
```

#### 输出

```text
密集矩阵:
[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]

CSR 非零元素: [1 2 3 4]
CSR 列索引: [0 1 2 3]

COO 格式:
  (0, 0)	1
  (1, 1)	2
  (2, 2)	3
  (3, 3)	4
COO 非零元素: 4
```

#### 理解重点

- 4×4 矩阵有 16 个元素，只有 4 个非零——稀疏率 75%
- CSR 和 COO 存储相同信息，但 CSR 更适合计算（矩阵乘法 $O(nnz)$），COO 更适合构建
- `coo.tocsr()` 和 `csr.tocoo()` 可在格式之间快速转换
- `.toarray()` 将稀疏矩阵转回密集 NumPy 数组

## 2. 稀疏矩阵操作

### `sparse.random` / 矩阵运算

#### 作用

`sparse.random` 生成指定密度的随机稀疏矩阵。稀疏矩阵支持加法、乘法等运算，结果保持稀疏格式。`.nnz` 属性返回非零元素数量。

#### 重点方法

```python
sparse.random(m, n, density=0.01, format='coo')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `m` | `int` | 矩阵行数 | `20` |
| `n` | `int` | 矩阵列数 | `20` |
| `density` | `float` | 非零元素占比，默认为 `0.01` | `0.1` |
| `format` | `str` | 输出格式：`'csr'` / `'csc'` / `'coo'`，默认为 `'coo'` | `'csr'` |

#### 示例代码

```python
import numpy as np
from scipy import sparse

np.random.seed(42)
A = sparse.random(20, 20, density=0.1, format='csr')
print(f"随机稀疏矩阵 A (密度=0.1):")
print(f"  形状: {A.shape}")
print(f"  非零元素: {A.nnz}")
print(f"  实际密度: {A.nnz / (A.shape[0]*A.shape[1]):.2f}")

# 矩阵运算：A + I
B = sparse.eye(20, format='csr')
C = A + B
print(f"\nA + I 非零元素: {C.nnz}")

# 转为密集矩阵
dense = A.toarray()
print(f"密集矩阵 shape: {dense.shape}")
```

#### 输出

```text
随机稀疏矩阵 A (密度=0.1):
  形状: (20, 20)
  非零元素: 40
  实际密度: 0.10

A + I 非零元素: 56

密集矩阵 shape: (20, 20)
```

#### 理解重点

- 20×20 矩阵 10% 密度 → 约 40 个非零元素
- A + I（加单位矩阵）后非零元素增加到 56——对角线上部分零位被填充
- 稀疏矩阵运算保持稀疏格式——不会自动转为密集矩阵
- `sparse.eye(n)` 创建稀疏单位矩阵——比 `np.eye(n)` 节省大量内存

## 3. 稀疏线性代数

### `sparse.linalg.spsolve` / `sparse.diags`

#### 作用

`spsolve` 利用矩阵稀疏结构高效求解 Ax = b。`sparse.diags` 创建对角稀疏矩阵——三对角矩阵是最常见的稀疏结构，广泛用于有限差分法。稀疏求解的时间复杂度远低于密集求解的 $O(n^3)$。

#### 重点方法

```python
sparse.diags(diagonals, offsets, shape, format='csr')
sparse.linalg.spsolve(A, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `diagonals` | `list[array_like]` | 各对角线上的值 | `[[-1, 2, -1]]` |
| `offsets` | `list[int]` | 对角线偏移：`0`=主对角线，`-1`=下对角，`1`=上对角 | `[-1, 0, 1]` |
| `shape` | `tuple[int, int]` | 矩阵形状 | `(100, 100)` |
| `A` | `sparse matrix` | 稀疏系数矩阵（方阵） | 三对角 CSR 矩阵 |
| `b` | `array_like` | 右端向量 | `np.ones(100)` |

#### 示例代码

```python
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

# 创建三对角系统 [-1, 2, -1]（一维拉普拉斯算子）
n = 100
A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
b = np.ones(n)

x = splinalg.spsolve(A, b)

print(f"矩阵: {n}x{n}, 非零元素: {A.nnz}")
print(f"解范数: {np.linalg.norm(x):.4f}")
print(f"残差: {np.linalg.norm(A @ x - b):.2e}")
```

#### 输出

```text
矩阵: 100x100, 非零元素: 298
解范数: 29.0115
残差: 2.15e-14
```

#### 理解重点

- 100×100 三对角矩阵只有 298 个非零元素（主对角 100 + 上下各 99）——远少于密集的 10000
- `spsolve` 利用三对角结构，时间复杂度 $O(n)$（Thomas 算法）——密集求解需 $O(n^3)$
- 残差 ≈ $10^{-14}$（机器精度）——验证了求解的正确性
- 三对角矩阵 `[-1, 2, -1]` 是一维拉普拉斯算子的离散形式——广泛用于热传导、扩散方程

## 4. 稀疏矩阵内存效率

### 内存对比

#### 作用

稀疏矩阵的核心优势是内存节省和计算加速。密集矩阵内存 = $n^2 \times 8$ 字节（float64），稀疏矩阵内存 ≈ $nnz \times 16$ 字节。当密度低于约 10% 时，稀疏格式在内存和速度上都有显著优势。

#### 示例代码

```python
n = 1000
density = 0.01
nnz = int(n * n * density)

# 密集矩阵内存 (float64 = 8 bytes)
denseMem = n * n * 8

# 稀疏矩阵内存 (COO: data(float64) + row(int32) + col(int32))
sparseMem = nnz * (8 + 4 + 4)

print(f"矩阵: {n}x{n}, 密度: {density*100}%")
print(f"密集矩阵: {denseMem / 1024 / 1024:.2f} MB")
print(f"稀疏矩阵: {sparseMem / 1024:.2f} KB")
print(f"节省: {(1 - sparseMem / denseMem) * 100:.1f}%")
```

#### 输出

```text
矩阵: 1000x1000, 密度: 1.0%
密集矩阵: 7.63 MB
稀疏矩阵: 156.25 KB
节省: 98.0%
```

#### 理解重点

- 1000×1000 密度 1% 的矩阵——稀疏仅需 156KB，密集需 7.63MB，节省 98%
- 随着矩阵规模增大，节省比例不变（由密度决定），但绝对值差距急剧增大
- 5000×5000 密集需 ~190MB，稀疏仅 ~3.8MB
- 实际场景（推荐系统用户-物品矩阵、NLP 词-文档矩阵）密度往往不到 0.1%——稀疏存储是唯一可行方案

## 常见坑

1. 格式选择：CSR 适合行操作和矩阵乘法，CSC 适合列操作，COO 适合构建——选错影响性能
2. 逐元素赋值低效：不要用 `A[i,j] = v` 逐个赋值——先收集坐标再一次性创建 COO
3. `toarray()` 内存爆炸：大规模稀疏矩阵转密集可能导致内存溢出
4. 稀疏 × 密集 = 密集：`sparse @ dense` 返回密集矩阵，可能抵消稀疏内存优势
5. `spsolve` 要求方阵：非方阵最小二乘应使用 `sparse.linalg.lsqr`

## 小结

- CSR 和 COO 是最常用的稀疏矩阵格式——各有适用场景
- 稀疏矩阵支持加法、乘法等基本运算——结果保持稀疏格式
- `spsolve` 利用矩阵稀疏结构高效求解——残差达机器精度
- 稀疏存储在低密度场景下可节省 98%+ 的内存——处理大规模数据的关键技术
- 原则：密度 < 10% 优先稀疏；构建用 COO，计算用 CSR/CSC

# SciPy 空间数据与距离计算

## 本章目标

1. 掌握常见距离度量（欧氏、曼哈顿、切比雪夫、余弦）及距离矩阵计算
2. 学会使用 KD 树进行高效最近邻搜索
3. 理解凸包（Convex Hull）的计算与属性
4. 掌握 Voronoi 图的构建与区域分析
5. 了解 Delaunay 三角剖分及其与 Voronoi 图的对偶关系

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `distance.euclidean(u, v)` | 函数 | 欧氏距离（L2 范数） |
| `distance.cdist(XA, XB, metric)` | 函数 | 成对距离矩阵 |
| `spatial.KDTree(data)` | 构造器 | 构建 KD 树空间索引 |
| `tree.query(x, k)` | 方法 | K 最近邻查询 |
| `spatial.ConvexHull(points)` | 构造器 | 凸包计算 |
| `spatial.Voronoi(points)` | 构造器 | Voronoi 图 |
| `spatial.Delaunay(points)` | 构造器 | Delaunay 三角剖分 |

## 1. 距离计算

### `distance.euclidean` / `distance.cdist`

#### 作用

`distance.euclidean` 计算欧氏距离（L2 范数）。`distance.cdist` 计算两组点之间的成对距离矩阵。另有 `cityblock`（曼哈顿 L1）、`chebyshev`（L∞）、`cosine`（1−余弦相似度）等度量。

#### 重点方法

```python
distance.euclidean(u, v)
distance.cdist(XA, XB, metric='euclidean')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `u` / `v` | `array_like` | 两个向量 | `[1, 2, 3]`, `[4, 5, 6]` |
| `XA` | `array_like` | 第一组点，形状 `(m, d)` | `[[0,0],[1,0],[0,1],[1,1]]` |
| `XB` | `array_like` | 第二组点，形状 `(n, d)` | 同上 |
| `metric` | `str` | 距离度量：`'euclidean'` / `'cityblock'` / `'cosine'` 等，默认为 `'euclidean'` | `'cosine'` |

#### 示例代码

```python
import numpy as np
from scipy.spatial import distance

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"向量 a: {a}, 向量 b: {b}")
print(f"欧氏距离: {distance.euclidean(a, b):.4f}")
print(f"曼哈顿距离: {distance.cityblock(a, b):.4f}")
print(f"切比雪夫距离: {distance.chebyshev(a, b):.4f}")
print(f"余弦距离: {distance.cosine(a, b):.4f}")

# 距离矩阵
pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
dMat = distance.cdist(pts, pts, 'euclidean')
print(f"\n距离矩阵:\n{np.round(dMat, 4)}")
```

#### 输出

```text
向量 a: [1 2 3], 向量 b: [4 5 6]
欧氏距离: 5.1962
曼哈顿距离: 9.0000
切比雪夫距离: 3.0000
余弦距离: 0.0254

距离矩阵:
[[0.     1.     1.     1.4142]
 [1.     0.     1.4142 1.    ]
 [1.     1.4142 0.     1.    ]
 [1.4142 1.     1.     0.    ]]
```

#### 理解重点

- 欧氏距离 = $\sqrt{3^2+3^2+3^2} = \sqrt{27} \approx 5.196$——直线距离
- 曼哈顿距离 = |3|+|3|+|3| = 9——沿坐标轴走（如城市街区）
- 切比雪夫距离 = max(3,3,3) = 3——各维度最大差值
- 余弦距离 ≈ 0.025——两向量方向几乎一致（余弦相似度 ≈ 0.975）
- 距离矩阵对称，对角线为 0；(0,0) 到 (1,1) 的距离为 $\sqrt{2} \approx 1.414$

## 2. KD 树

### `spatial.KDTree`

#### 作用

KD 树将数据空间递归二分，实现高效空间查询。查询时间复杂度 $O(\log n)$，远优于暴力搜索的 $O(n)$。适用于低维空间（通常 d < 20），高维时性能退化。

#### 重点方法

```python
tree = spatial.KDTree(data, leafsize=10)
tree.query(x, k=1)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like` | 点集数据，形状 `(n, d)` | 100 个随机二维点 |
| `leafsize` | `int` | 叶节点最大点数，默认为 `10` | `10` |
| `x` | `array_like` | 查询点 | `[5, 5]` |
| `k` | `int` | 最近邻个数，默认为 `1` | `5` |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(100, 2) * 10

tree = spatial.KDTree(points)
print(f"点集大小: {len(points)}")

# 最近邻
queryPt = [5, 5]
dist, idx = tree.query(queryPt)
print(f"查询点: {queryPt}")
print(f"最近邻: {points[idx]} (距离: {dist:.4f})")

# K=5 最近邻
dists, idxs = tree.query(queryPt, k=5)
print("\n5 个最近邻:")
for d, i in zip(dists, idxs):
    print(f"  {points[i]} (距离: {d:.4f})")
```

#### 输出

```text
点集大小: 100
查询点: [5, 5]
最近邻: [4.9785 5.0266] (距离: 0.0351)

5 个最近邻:
  [4.9785 5.0266] (距离: 0.0351)
  [5.2716 5.1034] (距离: 0.2867)
  [4.6477 5.1877] (距离: 0.3980)
  [5.4408 5.0297] (距离: 0.4414)
  [4.6399 4.7096] (距离: 0.4646)
```

#### 理解重点

- KD 树将 100 个点组织成树结构——查询最近邻只需访问少数节点
- 最近邻距离 ≈ 0.035——在 [0,10]×[0,10] 区域内 100 个点分布比较密集
- K=5 查询返回按距离排序的 5 个最近邻
- 构建时间 $O(n\log n)$，查询 $O(\log n)$——适合反复查询场景
- `query_ball_point(x, r)` 可查询半径 r 内的所有点

## 3. 凸包

### `spatial.ConvexHull`

#### 作用

计算点集的凸包（包围所有点的最小凸多边形）。`hull.vertices` 返回凸包顶点的索引。**注意**：二维中 `hull.volume` 返回面积，`hull.area` 返回周长——SciPy 使用通用 N 维术语。

#### 重点方法

```python
spatial.ConvexHull(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 点集数据，形状 `(n, d)` | 30 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(30, 2)

hull = spatial.ConvexHull(points)

print(f"点数: {len(points)}")
print(f"凸包顶点数: {len(hull.vertices)}")
print(f"凸包顶点索引: {hull.vertices}")
print(f"凸包面积 (volume): {hull.volume:.4f}")
```

#### 输出

```text
点数: 30
凸包顶点数: 8
凸包顶点索引: [16  1  3 22 14 23 15 27]
凸包面积 (volume): 0.8014
```

#### 理解重点

- 30 个随机点中约 8 个位于凸包边界上——其余点在凸包内部
- 凸包面积接近 1——点在 [0,1]×[0,1] 均匀分布，凸包几乎覆盖整个正方形
- 二维中 `hull.volume` = 面积，`hull.area` = 周长
- `ConvexHull` 基于 Qhull 库——时间复杂度 $O(n\log n)$

## 4. Voronoi 图

### `spatial.Voronoi`

#### 作用

计算 Voronoi 图——将空间划分为每个种子点的最近邻区域。每个 Voronoi 区域内的所有位置，到对应种子点的距离比到其他任何种子点都近。

#### 重点方法

```python
spatial.Voronoi(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 种子点集，形状 `(n, 2)` | 10 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(10, 2)

vor = spatial.Voronoi(points)

print(f"种子点数: {len(points)}")
print(f"Voronoi 顶点数: {len(vor.vertices)}")
print(f"Voronoi 区域数: {len(vor.regions)}")

print("\n点对应的区域:")
for i, regionIdx in enumerate(vor.point_region):
    print(f"  点 {i} -> 区域 {regionIdx}")
```

#### 输出

```text
种子点数: 10
Voronoi 顶点数: 13
Voronoi 区域数: 11

点对应的区域:
  点 0 -> 区域 1
  点 1 -> 区域 3
  点 2 -> 区域 2
  点 3 -> 区域 8
  点 4 -> 区域 5
  点 5 -> 区域 10
  点 6 -> 区域 7
  点 7 -> 区域 4
  点 8 -> 区域 6
  点 9 -> 区域 9
```

#### 理解重点

- 10 个种子点产生 11 个区域（含空区域），13 个 Voronoi 顶点
- `vor.regions` 中包含 −1 的区域延伸到无穷远（边界点）
- 每个 Voronoi 区域内所有位置到对应种子点的距离最近
- 广泛用于：最近邻区域划分、选址问题、GIS、晶体结构分析

## 5. Delaunay 三角剖分

### `spatial.Delaunay`

#### 作用

计算 Delaunay 三角剖分——将点集连接成不重叠的三角形，最大化最小角（避免狭长三角形）。与 Voronoi 图互为对偶：Voronoi 的每条边垂直平分对应的 Delaunay 边。

#### 重点方法

```python
spatial.Delaunay(points)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `points` | `array_like` | 点集数据，形状 `(n, 2)` | 15 个随机二维点 |

#### 示例代码

```python
import numpy as np
from scipy import spatial

np.random.seed(42)
points = np.random.rand(15, 2)

tri = spatial.Delaunay(points)

print(f"点数: {len(points)}")
print(f"三角形数: {len(tri.simplices)}")

print("\n前 3 个三角形顶点索引:")
for i, simplex in enumerate(tri.simplices[:3]):
    print(f"  三角形 {i}: {simplex}")
```

#### 输出

```text
点数: 15
三角形数: 20

前 3 个三角形顶点索引:
  三角形 0: [13  3  7]
  三角形 1: [ 2  7 10]
  三角形 2: [ 7  3  2]
```

#### 理解重点

- 15 个点生成 20 个三角形——符合 Euler 公式：三角形数 ≈ 2n − h − 2
- Delaunay 满足"空圆性质"：每个三角形的外接圆内不包含其他点
- 与 Voronoi 图的对偶关系：Delaunay 两点相连 ⇔ 它们的 Voronoi 区域共享边
- 应用场景：有限元网格生成、地形建模、三维重建、路径规划
- `tri.find_simplex(point)` 可查找某个点位于哪个三角形内

## 常见坑

1. `cosine` 返回距离不是相似度：`distance.cosine` 返回 $1 - \cos\theta$，范围 [0, 2]——不是余弦相似度
2. `hull.volume` 在二维中是面积：二维中 `volume`=面积、`area`=周长——容易混淆
3. KD 树不适合高维：维度超过 ~20 时 KD 树退化为暴力搜索——应使用 Ball Tree
4. Voronoi 无穷区域：边界点的 Voronoi 区域延伸到无穷远——`regions` 中包含 −1
5. `cdist` 内存：n 个点的距离矩阵大小为 $n^2$——大规模点集可能内存不足

## 小结

- `distance` 模块提供丰富的距离度量——`cdist` 高效计算成对距离矩阵
- KD 树将最近邻搜索从 $O(n)$ 加速到 $O(\log n)$——空间索引的核心数据结构
- 凸包是包围点集的最小凸多边形——用于形状分析和碰撞检测
- Voronoi 图将空间划分为最近邻区域——广泛用于选址和区域分析
- Delaunay 三角剖分与 Voronoi 图互为对偶——有限元网格生成的基础算法
- 空间数据核心流程：选择距离度量 → 构建空间索引 → 执行空间查询/分析

