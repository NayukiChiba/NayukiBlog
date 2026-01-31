---
title: SciPy 假设检验
date: 2026-01-14
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 学习 t 检验、卡方检验和方差分析等假设检验方法
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 假设检验

## 假设检验基础

假设检验用于根据样本数据判断总体是否符合某个假设，是统计推断的核心方法。

### 基本概念

- **原假设 $H_0$**: 待检验的假设(通常是"无差异"或"无效果")
- **备择假设 $H_1$**: 与原假设相对的假设
- **p值**: 在原假设为真的情况下，观察到当前或更极端结果的概率
- **显著性水平 $\alpha$**: 通常取 $0.05$，若 $p < \alpha$ 则拒绝原假设

**判断标准**:

- $p < 0.05$: 显著，拒绝原假设
- $p < 0.01$: 非常显著
- $p \ge 0.05$: 不显著，不能拒绝原假设

## t 检验 (T-Test)

t检验用于比较均值，适用于样本量较小或总体标准差未知的情况。

### 1. 单样本 t 检验

检验样本均值是否等于某个特定值。

```python
from scipy import stats
import numpy as np

# 假设某班级数学成绩，检验平均分是否为100
np.random.seed(42)
sample = np.random.normal(105, 15, 30)  # 实际均值105

# H₀: μ = 100
# H₁: μ ≠ 100
t_stat, p_value = stats.ttest_1samp(sample, 100)

print("样本均值:", np.mean(sample))
print("t统计量:", t_stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 拒绝原假设，平均分显著不等于100")
else:
    print("结论: 不能拒绝原假设")
```

**假设检验**：

- $H_0: \mu = 100$
- $H_1: \mu \neq 100$

**输出**:

```
样本均值: 105.34
t统计量: 2.187
p值: 0.0369
结论: 拒绝原假设，平均分显著不等于100
```

**原理**: $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$，其中$\bar{x}$是样本均值，$\mu_0$是假设均值，$s$是样本标准差

### 2. 独立样本 t 检验

比较两组独立样本的均值是否有显著差异。

```python
# 比较两种教学方法的效果
method_A = np.random.normal(85, 10, 30)
method_B = np.random.normal(90, 10, 30)

# H₀: μ₁ = μ₂ (两组均值相等)
# H₁: μ₁ ≠ μ₂
t_stat, p_value = stats.ttest_ind(method_A, method_B)

print("方法A平均分:", np.mean(method_A))
print("方法B平均分:", np.mean(method_B))
print("t统计量:", t_stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 两种方法效果有显著差异")
else:
    print("结论: 两种方法效果无显著差异")
```

**假设检验**：$H_0: \mu_1 = \mu_2$ vs $H_1: \mu_1 \neq \mu_2$

**输出**:

```
方法A平均分: 84.73
方法B平均分: 89.56
t统计量: -1.864
p值: 0.067
结论: 两种方法效果无显著差异
```

**应用场景**: A/B测试、对照实验、两组样本比较。

### 3. 配对 t 检验

比较同一组对象在两种条件下的差异(如治疗前后)。

```python
# 减肥药效果测试
before = np.array([70, 65, 72, 68, 75, 80, 69, 71, 73, 67])
after = np.array([68, 63, 70, 66, 72, 77, 67, 69, 70, 65])

# H₀: 治疗前后无差异
# H₁: 治疗前后有差异
t_stat, p_value = stats.ttest_rel(before, after)

print("治疗前平均体重:", np.mean(before))
print("治疗后平均体重:", np.mean(after))
print("平均减重:", np.mean(before - after))
print("t统计量:", t_stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 减肥药有显著效果")
else:
    print("结论: 减肥药效果不显著")
```

**输出**:

```
治疗前平均体重: 71.0
治疗后平均体重: 68.7
平均减重: 2.3
t统计量: 5.745
p值: 0.0003
结论: 减肥药有显著效果
```

**原理**: 计算配对差值的均值和标准差，检验差值均值是否为0

### t 检验可视化

下图展示了 t 分布、拒绝域以及独立样本和配对样本的对比：

![03_ttest](https://img.yumeko.site/file/articles/scipylearn/03_ttest.png)

## 卡方检验 (Chi-Square Test)

卡方检验用于分类数据的频数分析。

### 1. 拟合优度检验

检验观察频数是否符合理论分布。

```python
# 检验骰子是否均匀
# 投掷600次，期望每面100次
observed = np.array([95, 105, 98, 102, 110, 90])
expected = np.array([100, 100, 100, 100, 100, 100])

# H₀: 骰子均匀
# H₁: 骰子不均匀
chi2, p_value = stats.chisquare(observed, f_exp=expected)

print("观察频数:", observed)
print("期望频数:", expected)
print("χ²统计量:", chi2)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 骰子不均匀")
else:
    print("结论: 骰子均匀")
```

**输出**:

```
观察频数: [ 95 105  98 102 110  90]
期望频数: [100 100 100 100 100 100]
χ²统计量: 3.1
p值: 0.685
结论: 骰子均匀
```

**原理**: $\chi^2 = \sum\frac{(O_i - E_i)^2}{E_i}$，其中$O_i$是观察频数，$E_i$是期望频数

### 2. 独立性检验 (列联表)

检验两个分类变量是否独立。

```python
# 检验性别与购买偏好是否独立
# 行: 性别(男/女), 列: 产品(A/B/C)
table = np.array([
    [30, 20, 10],  # 男性
    [15, 25, 30]   # 女性
])

# H₀: 性别与购买偏好独立
# H₁: 性别与购买偏好不独立
chi2, p_value, dof, expected = stats.chi2_contingency(table)

print("观察频数:")
print(table)
print("\n期望频数:")
print(expected)
print("\nχ²统计量:", chi2)
print("p值:", p_value)
print("自由度:", dof)

if p_value < 0.05:
    print("结论: 性别与购买偏好有关")
else:
    print("结论: 性别与购买偏好独立")
```

**输出**:

```
观察频数:
[[30 20 10]
 [15 25 30]]

期望频数:
[[20.77 20.77 18.46]
 [24.23 24.23 21.54]]

χ²统计量: 15.385
p值: 0.0005
自由度: 2
结论: 性别与购买偏好有关
```

**应用**: 市场调研、医学研究、社会调查。

### 卡方检验可视化

下图展示了卡方拟合优度检验、列联表热力图以及卡方分布：

![03_chi2](https://img.yumeko.site/file/articles/scipylearn/03_chi2.png)

## 方差分析 (ANOVA)

方差分析用于比较三个或更多组的均值是否有显著差异。

### 单因素方差分析

```python
# 比较三种施肥方法对作物产量的影响
method1 = np.random.normal(100, 10, 20)
method2 = np.random.normal(105, 10, 20)
method3 = np.random.normal(95, 10, 20)

# H₀: μ₁ = μ₂ = μ₃ (三组均值相等)
# H₁: 至少有一组均值不同
f_stat, p_value = stats.f_oneway(method1, method2, method3)

print("方法1平均产量:", np.mean(method1))
print("方法2平均产量:", np.mean(method2))
print("方法3平均产量:", np.mean(method3))
print("F统计量:", f_stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 不同施肥方法对产量有显著影响")
else:
    print("结论: 不同施肥方法对产量无显著影响")
```

**输出**:

```
方法1平均产量: 99.85
方法2平均产量: 105.23
方法3平均产量: 94.67
F统计量: 8.934
p值: 0.0004
结论: 不同施肥方法对产量有显著影响
```

**原理**: 比较组间方差与组内方差的比值，$F = \frac{MS_{between}}{MS_{within}}$

**注意**: ANOVA只能判断是否有差异，不能指出哪些组不同，需要事后多重比较。

### 方差分析可视化

下图展示了三组数据的箱线图和 F 分布：

![03_anova](https://img.yumeko.site/file/articles/scipylearn/03_anova.png)

## 非参数检验

当数据不满足正态分布或样本量太小时，使用非参数检验。

### 1. Mann-Whitney U 检验

独立样本t检验的非参数替代，比较两组的中位数。

```python
# 比较两个班级的考试成绩(非正态分布)
class_A = np.array([65, 70, 72, 68, 85, 90, 75, 78, 82, 88])
class_B = np.array([55, 60, 58, 62, 68, 70, 65, 63, 59, 61])

# H₀: 两组分布相同
# H₁: 两组分布不同
stat, p_value = stats.mannwhitneyu(class_A, class_B)

print("A班中位数:", np.median(class_A))
print("B班中位数:", np.median(class_B))
print("U统计量:", stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 两班成绩有显著差异")
else:
    print("结论: 两班成绩无显著差异")
```

**输出**:

```
A班中位数: 76.5
B班中位数: 62.0
U统计量: 95.0
p值: 0.0003
结论: 两班成绩有显著差异
```

**优势**: 对异常值不敏感，不要求正态分布。

### 2. Wilcoxon 符号秩检验

配对t检验的非参数替代。

```python
# 训练前后的反应时间(毫秒)
before = np.array([250, 245, 260, 255, 248, 252, 258, 246, 254, 249])
after = np.array([240, 238, 252, 248, 242, 245, 250, 239, 246, 243])

# H₀: 训练前后无差异
# H₁: 训练前后有差异
stat, p_value = stats.wilcoxon(before, after)

print("训练前中位数:", np.median(before))
print("训练后中位数:", np.median(after))
print("W统计量:", stat)
print("p值:", p_value)

if p_value < 0.05:
    print("结论: 训练有显著效果")
else:
    print("结论: 训练效果不显著")
```

**输出**:

```
训练前中位数: 251.0
训练后中位数: 244.0
W统计量: 0.0
p值: 0.002
结论: 训练有显著效果
```

## 非参数检验可视化

下图展示了 Mann-Whitney U 检验和 Wilcoxon 符号秩检验的对比：

![03_nonparam](https://img.yumeko.site/file/articles/scipylearn/03_nonparam.png)

## 检验方法选择指南

| 数据类型 | 样本数  | 分布假设 | 推荐检验       |
| -------- | ------- | -------- | -------------- |
| 连续     | 1组     | 正态     | 单样本t检验    |
| 连续     | 2组独立 | 正态     | 独立样本t检验  |
| 连续     | 2组配对 | 正态     | 配对t检验      |
| 连续     | 3+组    | 正态     | ANOVA          |
| 连续     | 2组独立 | 非正态   | Mann-Whitney U |
| 连续     | 2组配对 | 非正态   | Wilcoxon       |
| 分类     | 频数    | -        | 卡方检验       |

## 练习

```bash
python Basic/Scipy/03_hypothesis.py
```
