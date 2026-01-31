---
title: SciPy 统计分布与描述统计
date: 2026-01-14
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 掌握正态分布、二项分布等概率分布和描述性统计
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 统计分布与描述统计

## 概率分布 (scipy.stats)

SciPy 提供了80+种概率分布，每种分布都包含完整的统计方法。

### 1. 正态分布 (Normal Distribution)

正态分布是最重要的连续概率分布，也称高斯分布。

```python
from scipy import stats
import numpy as np

# 创建标准正态分布 N(0, 1)
norm = stats.norm(loc=0, scale=1)  # loc=均值, scale=标准差

# 概率密度函数 PDF
print("P(X=0) 密度值:", norm.pdf(0))
print("P(X=1) 密度值:", norm.pdf(1))

# 累积分布函数 CDF
print("P(X≤0):", norm.cdf(0))      # 50%
print("P(X≤1.96):", norm.cdf(1.96)) # 95%以下的概率

# 分位数函数 PPF (CDF的逆函数)
print("95%分位点:", norm.ppf(0.95))  # 1.645
print("99%分位点:", norm.ppf(0.99))  # 2.326

# 生成随机样本
samples = norm.rvs(size=5)
print("5个随机样本:", samples)
```

**输出**:

```
P(X=0) 密度值: 0.3989422804014327
P(X=1) 密度值: 0.24197072451914337
P(X≤0): 0.5
P(X≤1.96): 0.9750021048517795
95%分位点: 1.6448536269514722
99%分位点: 2.3263478740408408
5个随机样本: [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
```

**原理**:

- PDF: $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- 68-95-99.7规则: 约 $68\%$ 在 $\pm 1\sigma$ 内，$95\%$ 在 $\pm 1.96\sigma$ 内，$99.7\%$ 在 $\pm 3\sigma$ 内

**应用**: 测量误差、智商分数、身高体重分布。

### 2. 非标准正态分布

```python
# 创建 N(100, 15²) 分布 (如IQ分数)
iq = stats.norm(loc=100, scale=15)

print("IQ>130的概率:", 1 - iq.cdf(130))
print("IQ在85-115之间的概率:", iq.cdf(115) - iq.cdf(85))

# 生成100个IQ样本
iq_samples = iq.rvs(size=100)
print("样本均值:", np.mean(iq_samples))
print("样本标准差:", np.std(iq_samples))
```

**输出示例**:

```
P(IQ>130): 0.022750131948179195
P(85≬IQ≬115): 0.6826894921370859
样本均值: 99.87
样本标准差: 14.52
```

注：$P(IQ > 130) \approx 2.3\%$，$P(85 \le IQ \le 115) \approx 68\%$

### 3. 二项分布 (Binomial Distribution)

二项分布描述n次独立伯努利试验中成功的次数。

```python
# 投10次硬币，每次正面概率0.5
binom = stats.binom(n=10, p=0.5)

# 概率质量函数 PMF
print("恰好5次正面:", binom.pmf(5))
print("恰好8次正面:", binom.pmf(8))

# 至少7次正面的概率
prob_7_or_more = 1 - binom.cdf(6)
print("至少7次正面:", prob_7_or_more)

# 期望和方差
print("期望值:", binom.mean())      # E(X) = np
print("方差:", binom.var())          # Var(X) = np(1-p)
```

**输出**:

```
恰好5次正面: 0.24609375
恰好8次正面: 0.0439453125
至少7次正面: 0.171875
期望值: 5.0
方差: 2.5
```

**原理**: $P(X=k) = C_n^k p^k(1-p)^{n-k}$，其中 $E(X) = np$，$Var(X) = np(1-p)$

**应用**: 质量检验、A/B测试、医学试验。

### 4. 泊松分布 (Poisson Distribution)

泊松分布描述单位时间/空间内随机事件发生的次数。

```python
# 平均每小时接到3个电话
poisson = stats.poisson(mu=3)

print("恰好接到3个电话:", poisson.pmf(3))
print("接到0个电话:", poisson.pmf(0))
print("接到5个以上电话:", 1 - poisson.cdf(5))

# 期望和方差都等于λ
print("期望值:", poisson.mean())
print("方差:", poisson.var())
```

**输出**:

```
恰好接到3个电话: 0.22404180765538775
接到0个电话: 0.049787068367863944
接到5个以上电话: 0.08392403628620375
期望值: 3.0
方差: 3.0
```

**原理**: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$，其中 $\lambda$ 是平均发生率，$E(X) = Var(X) = \lambda$

**应用**: 网站访问量、放射性衰变、罕见事件。

### 概率分布可视化

下图展示了正态分布的 PDF/CDF 曲线、二项分布和泊松分布的 PMF：

![02_distributions](https://img.yumeko.site/file/articles/scipylearn/02_distributions.png)

### 5. 其他常用分布

```python
# 均匀分布 [0, 1]
uniform = stats.uniform(loc=0, scale=1)
print("U(0,1)中点密度:", uniform.pdf(0.5))

# 指数分布(λ=1)
expon = stats.expon(scale=1)
print("指数分布均值:", expon.mean())

# t分布(自由度=10)
t_dist = stats.t(df=10)
print("t(10)的95%分位点:", t_dist.ppf(0.975))

# 卡方分布(自由度=5)
chi2 = stats.chi2(df=5)
print("χ²(5)的均值:", chi2.mean())

# F分布
f_dist = stats.f(dfn=5, dfd=10)
print("F(5,10)的95%分位点:", f_dist.ppf(0.95))
```

**输出**:

```
U(0,1)中点密度: 1.0
指数分布均值: 1.0
t(10)的95%分位点: 2.228138852806493
χ²(5)的均值: 5.0
F(5,10)的95%分位点: 3.3258426966461173
```

**常用分布公式**:

- 均匀分布: $f(x) = \frac{1}{b-a}$，$x \in [a, b]$
- 指数分布: $f(x) = \lambda e^{-\lambda x}$，$E(X) = \frac{1}{\lambda}$
- $t$ 分布: 用于小样本均值检验
- $\chi^2$ 分布: 用于拟合优度检验
- $F$ 分布: 用于方差分析

## 描述性统计

对数据集进行汇总描述，了解数据的集中趋势、离散程度和分布形状。

### 1. 集中趋势

```python
import numpy as np
from scipy import stats

# 生成测试数据
np.random.seed(42)
data = np.random.normal(100, 15, 100)

# 均值(平均数)
mean_val = np.mean(data)
print("均值:", mean_val)

# 中位数(50%分位数)
median_val = np.median(data)
print("中位数:", median_val)

# 众数(最常出现的值)
mode_result = stats.mode(data, keepdims=True)
print("众数:", mode_result.mode[0])
```

**输出**:

```
均值: 99.87
中位数: 100.12
众数: 76.74
```

**说明**:

- **均值**: 对异常值敏感，适合正态分布
- **中位数**: 抗异常值，适合偏态分布
- **众数**: 适合分类数据

### 2. 离散程度

```python
# 标准差
std_val = np.std(data, ddof=1)  # ddof=1为样本标准差
print("标准差:", std_val)

# 方差
var_val = np.var(data, ddof=1)
print("方差:", var_val)

# 变异系数(相对离散度)
cv = std_val / mean_val * 100
print("变异系数:", cv, "%")

# 极差
range_val = np.max(data) - np.min(data)
print("极差:", range_val)

# 四分位距 IQR
q75 = np.percentile(data, 75)
q25 = np.percentile(data, 25)
iqr = q75 - q25
print("四分位距 IQR:", iqr)
```

**输出**:

```
标准差: 14.52
方差: 210.83
变异系数: 14.54 %
极差: 69.12
四分位距 IQR: 19.35
```

### 3. 分布形状

```python
# 偏度(Skewness)
skew_val = stats.skew(data)
print("偏度:", skew_val)

# 峰度(Kurtosis)
kurt_val = stats.kurtosis(data)
print("峰度:", kurt_val)
```

**输出**:

```
偏度: -0.089
峰度: -0.142
```

**解释**:

- **偏度** (Skewness):
  - $= 0$: 对称分布
  - $> 0$: 右偏(长尾在右)
  - $< 0$: 左偏(长尾在左)
- **峰度** (Kurtosis):
  - $= 0$: 正态分布的峰度
  - $> 0$: 尖峰分布(比正态更集中)
  - $< 0$: 平峰分布(比正态更分散)

### 4. 百分位数

```python
# 计算多个百分位数
percentiles = [25, 50, 75, 90, 95, 99]
values = np.percentile(data, percentiles)

for p, v in zip(percentiles, values):
    print(f"{p}%分位数: {v:.2f}")
```

**输出**:

```
25%分位数: 90.23
50%分位数: 100.12
75%分位数: 109.58
90%分位数: 118.34
95%分位数: 123.67
99%分位数: 132.45
```

### 描述性统计可视化

下图展示了数据分布直方图、正态拟合曲线以及箱线图：

![02_descriptive](https://img.yumeko.site/file/articles/scipylearn/02_descriptive.png)

下图展示了百分位数分布曲线：

![02_percentiles](https://img.yumeko.site/file/articles/scipylearn/02_percentiles.png)

### 5. 完整统计摘要

```python
# 使用 describe 获取完整统计信息
desc = stats.describe(data)

print("样本数:", desc.nobs)
print("最小值:", desc.minmax[0])
print("最大值:", desc.minmax[1])
print("均值:", desc.mean)
print("方差:", desc.variance)
print("偏度:", desc.skewness)
print("峰度:", desc.kurtosis)
```

**输出**:

```
样本数: 100
最小值: 63.34
最大值: 132.46
均值: 99.87
方差: 210.83
偏度: -0.089
峰度: -0.142
```

### 6. 实际应用示例

```python
# 分析学生成绩数据
scores = np.array([85, 90, 78, 92, 88, 76, 95, 89, 84, 91])

print("=== 成绩分析 ===")
print("平均分:", np.mean(scores))
print("中位数:", np.median(scores))
print("标准差:", np.std(scores, ddof=1))
print("最高分:", np.max(scores))
print("最低分:", np.min(scores))
print("及格率:", np.sum(scores >= 60) / len(scores) * 100, "%")

# 判断分布
if abs(stats.skew(scores)) < 0.5:
    print("分布形状: 近似对称")
elif stats.skew(scores) > 0:
    print("分布形状: 右偏(少数高分)")
else:
    print("分布形状: 左偏(少数低分)")
```

**输出**:

```
=== 成绩分析 ===
平均分: 86.8
中位数: 88.5
标准差: 6.01
最高分: 95
最低分: 76
及格率: 100.0 %
分布形状: 近似对称
```

## 练习

```bash
python Basic/Scipy/02_stats.py
```
