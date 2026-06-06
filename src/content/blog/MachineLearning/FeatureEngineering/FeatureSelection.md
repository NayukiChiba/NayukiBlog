---
title: 机器学习特征选择方法完全指南：过滤式、包裹式与嵌入式
date: 2026-06-05
category: 机器学习/特征工程
tags:
  - 特征选择
  - 机器学习
  - 基础
  - scikit-learn
description: 系统讲解机器学习中三大类特征选择方法：过滤式（方差、相关系数、卡方检验、互信息）、包裹式（RFE、前向/后向搜索）和嵌入式（Lasso、树模型重要性），附完整 sklearn 代码实现和方法选型决策树。
image: https://img.yumeko.site/file/blog/cover/1780668043913.webp
status: published
---

## 1. 问题的起点：为什么需要特征选择？

给定一个包含 $d$ 个特征的数据集，直觉上"特征越多信息越多"。但实际上：

- **维度灾难**：随着特征数增加，样本密度指数级下降，模型需要指数级更多的数据才能保持相同的泛化能力
- **噪声特征**：无关特征或冗余特征不仅不会帮助模型，还会引入噪声、降低泛化性能
- **计算开销**：训练时间通常与特征数正相关，高维数据训练成本极高
- **可解释性**：更少的特征意味着更简单的模型，更容易解释和部署

特征选择的目标是：从 $d$ 个原始特征中选出一个最优子集 $S \subseteq \{1, 2, \dots, d\}$，使得模型在验证集上的性能最大化（或至少不显著下降）。

> [!NOTE] 特征选择 vs 降维
> 特征选择（Feature Selection）保留原始特征的子集，可解释性强；
> 降维（如 PCA）将原始特征映射到新的低维空间，新特征不可解释。
> 详见 [[MachineLearning/dimensionality/PCA|PCA 主成分分析]]。

## 2. 三大方法论总览

特征选择方法可分为三大类：

| 类别 | 核心思想 | 与模型的关系 | 计算成本 | 典型方法 |
|:--|:--|:--|:--|:--|
| 过滤式（Filter） | 基于统计指标独立评分 | 独立于模型 | 低 | 方差、相关系数、卡方检验、互信息 |
| 包裹式（Wrapper） | 以模型性能为评价标准搜索子集 | 依赖模型 | 高 | RFE、前向选择、后向消除 |
| 嵌入式（Embedded） | 在模型训练过程中完成选择 | 内置机制 | 中 | Lasso L1 正则化、树模型重要性 |

**选型直觉**：
- 数据量极大、特征极多 -> 先用 Filter 快速筛一遍 -> 再用 Embedded 精筛
- 小数据集、强解释性要求 -> Wrapper（但注意过拟合风险）
- 线性模型场景 -> Embedded（Lasso 一步到位）

## 3. 过滤式方法（Filter Methods）

过滤式方法使用统计指标对每个特征独立打分，然后按阈值或 Top-K 选择。

### 3.1 方差选择（Variance Threshold）

最原始的过滤式方法：如果某个特征在几乎所有样本中取值相同（方差接近 0），它无法提供区分信息。

$$
\operatorname{Var}(X_j) = \frac{1}{n}\sum_{i=1}^n (x_{ij} - \bar{x}_j)^2
$$

```python
from sklearn.feature_selection import VarianceThreshold

# 移除方差低于阈值的特征（对伯努利变量，threshold=p*(1-p)）
selector = VarianceThreshold(threshold=0.01)
XSelected = selector.fit_transform(X)
```

**适用场景**：预处理的第一步，快速剔除常量特征和近常量特征。

### 3.2 相关系数（Pearson Correlation）

衡量每个特征与目标变量之间的**线性相关性**：

$$
r_j = \frac{\operatorname{Cov}(X_j, y)}{\sigma_{X_j} \sigma_y} = \frac{\sum_{i=1}^n (x_{ij} - \bar{x}_j)(y_i - \bar{y})}{\sqrt{\sum_i (x_{ij} - \bar{x}_j)^2 \sum_i (y_i - \bar{y})^2}}
$$

$|r_j|$ 越大，线性相关性越强。选 $|r_j|$ 最大的 Top-K 个特征。

> [!WARNING] 注意
> Pearson 相关系数只能捕捉**线性**关系。如果特征与目标之间存在非线性关系（如 $y = x^2$），Pearson 可能接近 0 但特征仍然重要。

```python
import numpy as np

def pearsonSelection(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    基于 Pearson 相关系数选择 Top-K 特征

    Args:
        X: 形状 (n, d) 的特征矩阵
        y: 形状 (n,) 的目标向量
        k: 要选择的特征数
    Returns:
        形状 (k,) 的选中特征的索引
    """
    n, d = X.shape
    correlations = np.zeros(d)

    for j in range(d):
        xj = X[:, j]
        # 计算 Pearson 相关系数
        num = np.sum((xj - xj.mean()) * (y - y.mean()))
        den = np.sqrt(np.sum((xj - xj.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
        correlations[j] = np.abs(num / den) if den > 0 else 0

    return np.argsort(correlations)[::-1][:k]
```

### 3.3 卡方检验（Chi-Square Test）

用于**分类任务 + 离散/分类特征**。检验特征与标签之间的独立性。

对于特征 $X_j$ 有 $r$ 个取值和标签 $y$ 有 $c$ 个类别，构建 $r \times c$ 列联表。卡方统计量：

$$
\chi^2 = \sum_{i=1}^r \sum_{k=1}^c \frac{(O_{ik} - E_{ik})^2}{E_{ik}}
$$

其中 $O_{ik}$ 是观测频数，$E_{ik} = \frac{\text{第 i 行合计} \times \text{第 k 列合计}}{\text{总计}}$ 是期望频数。$\chi^2$ 越大，特征与标签的关联越强。

```python
from sklearn.feature_selection import chi2, SelectKBest

# chi2 要求特征值非负
selector = SelectKBest(chi2, k=10)
XSelected = selector.fit_transform(X, y)
```

### 3.4 互信息（Mutual Information）

最通用的过滤式指标，可捕捉**任意形式**的依赖关系（线性 + 非线性）：

$$
I(X_j; y) = \sum_{x \in \mathcal{X}_j} \sum_{c \in \mathcal{Y}} P(X_j=x, y=c) \log \frac{P(X_j=x, y=c)}{P(X_j=x) P(y=c)}
$$

互信息本质上衡量"知道 $X_j$ 后对 $y$ 的不确定性减少了多少"。$I(X_j; y) = 0$ 当且仅当 $X_j$ 与 $y$ 独立。

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# 分类任务
miScores = mutual_info_classif(X, y, random_state=42)

# 回归任务
miScores = mutual_info_regression(X, y, random_state=42)
```

> [!TIP] 方法选择速查
> - 连续特征 x 连续目标（回归） -> Pearson 相关系数 / 互信息回归版
> - 离散特征 x 离散标签（分类） -> 卡方检验 / 互信息分类版
> - 未知关系类型 -> 互信息（最通用但计算慢）

### 3.5 过滤式方法的优缺点

**优点**：计算快、与模型无关、不易过拟合、可解释性强。

**缺点**：不考虑特征之间的交互（如 $X_1$ 和 $X_2$ 各自不重要但组合起来重要）；不考虑特征与特定模型的匹配程度。

## 4. 包裹式方法（Wrapper Methods）

包裹式方法以**目标模型的性能**作为评价标准，在特征子集空间中搜索。

### 4.1 递归特征消除（RFE）

RFE 的核心思路是**反复训练模型 -> 剔除最不重要特征 -> 再训练**：

1. 用全部 $d$ 个特征训练模型
2. 根据模型提供的特征重要性（如线性模型的系数绝对值 $|w_j|$）排序
3. 剔除最不重要的 $s$ 个特征（通常 $s=1$）
4. 重复第 1-3 步，直到剩余 $k$ 个特征

$$
\text{第 t 轮: } S_t = S_{t-1} \setminus \left\{\arg\min_{j \in S_{t-1}} |w_j|\right\}
$$

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=10, step=1)
XSelected = selector.fit_transform(X, y)

# 查看各特征是否被选中
print(selector.support_)   # 布尔数组
print(selector.ranking_)   # 排名（1=最优）
```

**带交叉验证的 RFECV**：自动通过交叉验证选择最优特征数：

```python
from sklearn.feature_selection import RFECV

selector = RFECV(estimator, step=1, cv=5, scoring='accuracy')
XSelected = selector.fit_transform(X, y)
print(f"最优特征数: {selector.n_features_}")
```

### 4.2 前向选择（Forward Selection）

从空集开始，每步加入当前最优的单个特征：

1. $S_0 = \emptyset$
2. 对每个候选特征 $j \notin S_t$，用 $S_t \cup \{j\}$ 训练模型并评估
3. 选取提升最大的特征加入：$S_{t+1} = S_t \cup \{j^*\}$
4. 直到达到目标特征数 $k$ 或性能不再提升

### 4.3 后向消除（Backward Elimination）

从全集开始，每步删除当前最差的单个特征（RFE 的特例，`step=1`）：

1. $S_0 = \{1, 2, \dots, d\}$
2. 对每个特征 $j \in S_t$，用 $S_t \setminus \{j\}$ 训练模型并评估
3. 移除影响最小的特征：$S_{t+1} = S_t \setminus \{j^*\}$
4. 直到达到目标特征数 $k$

### 4.4 包裹式方法的优缺点

**优点**：考虑特征与模型的交互，通常能找到对特定模型最优的子集。

**缺点**：计算成本极高（每步需要重新训练模型 $O(d)$ 到 $O(d^2)$ 次）；容易过拟合（当 $n$ 较小时尤其危险）。

## 5. 嵌入式方法（Embedded Methods）

嵌入式方法在模型训练过程中**自动**完成特征选择，是实际生产中最常用的方式。

### 5.1 Lasso（L1 正则化）

Lasso 在线性模型的损失函数中添加 L1 惩罚项：

$$
\boxed{\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n} \|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}}
$$

L1 惩罚的几何特性使得最优解是**稀疏的**——许多系数 $\beta_j$ 恰好为 0。$\lambda$ 控制稀疏程度：

- $\lambda = 0$ -> 普通最小二乘，所有系数非零
- $\lambda \to \infty$ -> 所有系数为零
- $\lambda$ 适中 -> 只有部分系数非零

```python
from sklearn.linear_model import LassoCV

# LassoCV 通过交叉验证自动选 lambda
model = LassoCV(cv=5, random_state=42)
model.fit(X, y)

# 非零系数对应的特征
selectedFeatures = np.where(model.coef_ != 0)[0]
print(f"选中 {len(selectedFeatures)} 个特征")
```

> [!NOTE] L1 vs L2 正则化
> L2 正则化（Ridge）收缩系数但不设为零，不具备特征选择功能。
> Lasso 的稀疏性来源于 L1 球的"尖角"——最优解容易落在坐标轴上。
> 详见 [[MachineLearning/regression/Regularization|正则化回归]]。

### 5.2 决策树/随机森林特征重要性

决策树在每次分裂时选择能最大程度降低不纯度的特征。将其在所有节点上的降低量按权重累积，即得到该特征的重要性分数：

$$
\text{Importance}(X_j) = \sum_{t: \text{split on } j} \frac{n_t}{n} \cdot \Delta \text{Impurity}(t)
$$

其中 $n_t$ 是到达节点 $t$ 的样本数，$\Delta \text{Impurity}(t)$ 是该次分裂减少的不纯度（Gini 或 Entropy）。

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 获取特征重要性
importances = model.feature_importances_

# 可视化
import matplotlib.pyplot as plt

def plotFeatureImportance(importances, featureNames, topK=20):
    """
    绘制特征重要性排序图

    Args:
        importances: 形状 (d,) 的重要性分数
        featureNames: 特征名称列表
        topK: 显示前 K 个重要特征
    """
    indices = np.argsort(importances)[::-1][:topK]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [featureNames[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('特征重要性')
    plt.title(f'Top {topK} 特征重要性排序')
    plt.tight_layout()
    plt.show()
```

### 5.3 基于特征重要性的特征选择

scikit-learn 提供了通用的 `SelectFromModel`，可配合任何有 `coef_` 或 `feature_importances_` 属性的模型：

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# 基于 GBDT 的重要性自动选择
selector = SelectFromModel(
    GradientBoostingClassifier(random_state=42),
    threshold='median'  # 以中位数为阈值
)
XSelected = selector.fit_transform(X, y)
```

### 5.4 嵌入式方法的优缺点

**优点**：计算高效（选择和训练同步完成）、考虑特征交互（树模型）、抗过拟合能力较好。

**缺点**：选择结果依赖于所选模型（Lasso 选出的特征不一定对随机森林最优）；树模型对高基数类别特征有偏。

## 6. 方法选型决策

根据以下场景依次判断：

1. **特征数 > 10,000？** -> 先用方差阈值 + 互信息快速筛到 1000 以下
2. **需要严格的特征独立性检验？** -> 过滤式（互信息 / 卡方检验）
3. **使用线性模型？** -> 嵌入式（Lasso L1 正则化）
4. **使用树模型（RF/GBDT/XGBoost）？** -> 嵌入式（特征重要性 + SelectFromModel）
5. **特征数 < 50 且对最优子集有强需求？** -> 包裹式（RFECV / 前向选择）
6. **通用方案** -> Filter（互信息）筛到约 100 -> Embedded（树模型）精筛到约 20

## 7. 综合对比

| 维度 | Filter | Wrapper | Embedded |
|:--|:--|:--|:--|
| 计算速度 | 快 | 慢 | 中 |
| 过拟合风险 | 低 | 高 | 中 |
| 考虑特征交互 | 否 | 是 | 是 |
| 与模型无关 | 是 | 否 | 否 |
| 处理高维数据 | 好 | 差 | 中 |
| 典型方法 | 互信息、卡方 | RFE、前向选择 | Lasso、树重要性 |
| 推荐场景 | 预处理初筛 | 小数据集精筛 | 日常建模首选 |

## 8. 实战流程建议

```python
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 完整的特征选择 Pipeline
pipeline = Pipeline([
    # 第 1 步：去除低方差特征（d -> d1）
    ('variance_filter', VarianceThreshold(threshold=0.01)),

    # 第 2 步：互信息初筛（d1 -> d2）
    ('mutual_info', SelectKBest(mutual_info_classif, k=50)),

    # 第 3 步：嵌入式精筛（d2 -> d3）
    ('embedding', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold='median'
    )),
])

XSelected = pipeline.fit_transform(X, y)
print(f"原始特征数: {X.shape[1]}")
print(f"选择后特征数: {XSelected.shape[1]}")
```

---

> **相关文章**：
> - [[MachineLearning/dimensionality/PCA|PCA 主成分分析]]
> - [[MachineLearning/regression/Regularization|正则化回归]]
> - [[NeuralNetwork/Training/Dropout|Dropout 正则化]]
