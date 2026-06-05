---
title: LDA 线性判别分析
date: 2026-05-06
category: 机器学习/降维
tags:
  - Scikit-learn
description: LDA线性判别分析的数学原理、Fisher判别准则与完整工程实现。
image: https://img.yumeko.site/file/blog/cover/1780581784800.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 LDA 的优化目标——最大化类间散度与类内散度之比（Fisher 判别准则）。
2. 理解类内散度矩阵 $\mathbf{S}_W$ 和类间散度矩阵 $\mathbf{S}_B$ 的构造与含义。
3. 理解广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ 如何给出判别方向，以及 $K-1$ 维上限的秩论证。
4. 理解 `solver='svd'` 的求解路径，以及与 PCA 在优化目标上的根本区别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 类内散度矩阵 $\mathbf{S}_W$ | 数学对象 | 衡量同一类别内部样本围绕类均值的分散程度——LDA 希望将其最小化 |
| 类间散度矩阵 $\mathbf{S}_B$ | 数学对象 | 衡量各类别均值围绕全局均值的分散程度——LDA 希望将其最大化 |
| Fisher 判别准则 $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ | 优化目标 | 广义瑞利商——同时最大化类间分离度与最小化类内散布 |
| 广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ | 求解形式 | 判别方向是 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 的特征向量，对应最大的 $q$ 个特征值 |
| $K-1$ 维上限 | 理论约束 | $\text{rank}(\mathbf{S}_B) \leq K-1$，因此 $K$ 类 LDA 最多提取 $K-1$ 个判别方向 |
| `solver='svd'` | 工程求解 | 通过 SVD 分解直接求解，无需显式计算 $\mathbf{S}_W^{-1}$，数值稳定性更好 |

## 1. LDA 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$ 及其类别标签 $y_i \in \{1, \dots, K\}$，LDA 寻找投影方向 $\mathbf{w}$，使得投影后类间散度最大、类内散度最小。

### 二分类 Fisher 准则

将数据投影到方向 $\mathbf{w}$ 后，第 $k$ 类的投影均值和投影散度为：

$$
\tilde{\mu}_k = \mathbf{w}^T \boldsymbol{\mu}_k, \quad \tilde{s}_k^2 = \sum_{\mathbf{x}_i \in C_k} (\mathbf{w}^T \mathbf{x}_i - \tilde{\mu}_k)^2 = \mathbf{w}^T \mathbf{S}_k \mathbf{w}
$$

Fisher 准则定义为类间距离与类内散度之比：

$$
J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

### 理解重点

- 分子是类间距离的平方——不同类别的投影中心应尽可能远离。
- 分母是各类内部的投影散度之和——同类别样本投影后应尽可能聚集。
- 这个比值越大，方向 $\mathbf{w}$ 的判别能力越强。当前源码中 `LinearDiscriminantAnalysis` 寻找的就是最大化此比值的 $\mathbf{w}$。

## 2. 散度矩阵

### 类内散度矩阵（Within-Class Scatter Matrix）

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
$$

其中 $\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ 为第 $k$ 类的均值向量，$C_k$ 为第 $k$ 类的样本集合。

### 类间散度矩阵（Between-Class Scatter Matrix）

$$
\mathbf{S}_B = \sum_{k=1}^{K} N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
$$

其中 $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$ 为全局均值向量，$N_k = |C_k|$ 为第 $k$ 类样本数。

### 理解重点

- $\mathbf{S}_W$ 汇总了各类内部的协方差结构——它是 $K$ 个类内协方差矩阵的加权和（在等协方差假设下，各类协方差相同）。
- $\mathbf{S}_B$ 汇总了类中心之间的方差——它的秩不超过 $K-1$，因为 $K$ 个类中心满足一个线性关系（均值的加权平均等于全局均值）。
- 当前 Wine 数据集（$d=13, K=3$）下，$\mathbf{S}_W$ 是 $13 \times 13$ 矩阵，$\mathbf{S}_B$ 的秩为 2。

## 3. 广义瑞利商与广义特征值问题

### 从优化到特征值问题

最大化 $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ 等价于求解广义特征值问题：

$$
\boxed{\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}}
$$

若 $\mathbf{S}_W$ 可逆，即化为标准特征值问题：

$$
\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}
$$

判别方向取 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 最大的 $q$ 个特征值对应的特征向量，其中 $q \leq K-1$。

### 拉格朗日推导

以 $\mathbf{w}^T \mathbf{S}_W \mathbf{w} = 1$ 为约束，最大化 $\mathbf{w}^T \mathbf{S}_B \mathbf{w}$：

$$
\mathcal{L} = \mathbf{w}^T \mathbf{S}_B \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{S}_W \mathbf{w} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{S}_B \mathbf{w} - 2\lambda \mathbf{S}_W \mathbf{w} = 0 \quad\Rightarrow\quad \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}
$$

### 理解重点

- 这不是经验规则——判别方向是严格优化问题的解析结果。
- 特征值 $\lambda$ 就是该判别方向对应的 Fisher 准则值 $J(\mathbf{w})$——特征值越大，该方向的判别能力越强。
- `explained_variance_ratio_` 就是各特征值占总特征值之和的比例。

## 4. 二分类闭式解

二分类时 $\text{rank}(\mathbf{S}_B) = 1$，存在闭式解：

$$
\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

### 理解重点

- 二分类 LDA 的判别方向非常直观：就是"类中心差异"经"类内协方差结构"修正后的方向。
- $\mathbf{S}_W^{-1}$ 的作用是白化——消除特征间的相关性，使几何距离在各方向上等价。
- 这是理解多分类 LDA 的最佳起点：多分类只是将"一对多"的概念推广到多个判别方向。

## 5. 多分类推广与 $K-1$ 维上限

### 秩论证

$\mathbf{S}_B$ 是 $K$ 个秩-1 外积 $\{(\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T\}_{k=1}^{K}$ 的加权和，且 $\sum_k N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu}) = \mathbf{0}$，因此：

$$
\text{rank}(\mathbf{S}_B) \leq K - 1
$$

进而 $\text{rank}(\mathbf{S}_W^{-1}\mathbf{S}_B) \leq K-1$，最多只有 $K-1$ 个非零特征值。

### 对当前源码的影响

| 类别数 $K$ | 最大判别方向数 $K-1$ | 当前 `n_components` |
|---|---|---|
| 2 | 1 | — |
| 3（Wine） | 2 | `2`（达理论上限） |
| 4 | 3 | — |
| 10 | 9 | — |

### 理解重点

- 当前 Wine 数据（$K=3$）下 `n_components=2` 不是随意选择——它恰好达到理论上限。
- 这是区分 LDA 与 PCA 的核心数学特征之一：PCA 维数无类别限制，LDA 受 $K-1$ 约束。
- 当前流水线只输出 2D 图（不输出 3D 图）的数学根源即在于此。

## 6. SVD 求解器

当前源码使用 `solver='svd'`，其求解路径为：

1. 对类内散度 $\mathbf{S}_W$ 做 Cholesky 分解或直接取逆的平方根
2. 将广义特征值问题转化为普通特征值问题
3. 通过 SVD 求解，避免显式计算 $\mathbf{S}_W^{-1}\mathbf{S}_B$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `solver` | `str` | 求解器选择。`'svd'`（默认）通过 SVD 求解，无需显式计算散度矩阵逆；`'eigen'` 直接做特征分解；`'lsqr'` 使用最小二乘 | `'svd'`、`'eigen'`、`'lsqr'` |

### 理解重点

- `solver='svd'` 是 scikit-learn 的默认选择——数值稳定性好，且不要求 $\mathbf{S}_W$ 满秩。
- 不同求解器最显著的工程差异是 `explained_variance_ratio_` 是否可用——`'svd'` 支持此属性，`'lsqr'` 不支持。
- 当前源码用 `hasattr(model, "explained_variance_ratio_")` 做保护式输出，正因求解器差异。

## 7. LDA 与 PCA 的数学对比

| 维度 | PCA | LDA |
|---|---|---|
| 监督方式 | 无监督 | 有监督（需要 $y$） |
| 优化目标 | $\max \mathbf{w}^T \mathbf{S}_T \mathbf{w}$（最大化投影方差） | $\max \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$（最大化类间/类内比） |
| 核心矩阵 | 总散度矩阵 $\mathbf{S}_T = \sum_i (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$ | $\mathbf{S}_W$ 和 $\mathbf{S}_B$，满足 $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ |
| 降维上限 | 最多 $\min(d, N)$ 维 | 最多 $K-1$ 维 |
| 标签参与 | 否 | 是（定义类结构） |
| 适用场景 | 数据压缩、无监督可视化、去噪 | 分类预处理、判别式降维、特征提取 |

### 理解重点

- $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ 表明总方差可分解为"类内方差 + 类间方差"——PCA 最大化总方差，LDA 最大化类间方差占类内方差的比例。
- 这是两者数学本质差异的集中体现：PCA 不关心类别，LDA 以类别为核心。

## 8. 标准化对 LDA 的数学必要性

LDA 的核心操作涉及散度矩阵的计算：

$$
\mathbf{S}_W = \sum_k \sum_{\mathbf{x} \in C_k} (\mathbf{x} - \boldsymbol{\mu}_k)(\mathbf{x} - \boldsymbol{\mu}_k)^T
$$

若特征 $x_1$ 的量纲是 $x_2$ 的 100 倍，则 $x_1$ 方向的方差将主导散度矩阵——判别方向被尺度最大的特征绑架。

### 理解重点

- 标准化后每个特征对散度矩阵的贡献均等——判别方向反映真实的类别可分性结构。
- Wine 数据集中 `alcohol`（~13）和 `proline`（~746）的数值范围差异巨大，不标准化将导致 `proline` 主导全部判别方向。
- 这与此前所有基于距离/散度的算法（KMeans、DBSCAN、SVC）的逻辑完全一致——标准化是几何意义的前置条件。

## 9. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 类内散度矩阵 | $\mathbf{S}_W$ | `LinearDiscriminantAnalysis` 内部计算 |
| 类间散度矩阵 | $\mathbf{S}_B$ | `LinearDiscriminantAnalysis` 内部计算 |
| Fisher 判别准则 | $J(\mathbf{w}) = \mathbf{w}^T\mathbf{S}_B\mathbf{w} / \mathbf{w}^T\mathbf{S}_W\mathbf{w}$ | LDA 优化核心 |
| 广义特征值问题 | $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ | `solver` 内部求解 |
| 判别方向数 | $q \leq K-1$ | `n_components=2` |
| 求解器 | — | `solver='svd'` |
| 判别方向 | $\mathbf{w}_1, \dots, \mathbf{w}_q$ | `model.scalings_` |
| 解释方差比 | $\lambda_j / \sum_i \lambda_i$ | `model.explained_variance_ratio_`（若 solver 支持） |
| 类均值 | $\boldsymbol{\mu}_k$ | `model.means_` |
| 先验概率 | $\pi_k = N_k / N$ | `model.priors_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |
| 投影 | $\mathbf{X}\mathbf{W}$ | `model.transform(X)` |

## 常见坑

1. 混淆 PCA 与 LDA 的优化目标——PCA 最大化投影方差（无监督），LDA 最大化类间/类内散度比（有监督）。
2. 忽略 $K-1$ 维上限，误以为 LDA 可以像 PCA 一样自由增加输出维度。
3. 把 `explained_variance_ratio_` 当成所有求解器都支持的属性——`lsqr` 求解器不提供此属性。
4. 在不标准化的数据上运行——不同量纲的特征绑架散度矩阵计算。

## 小结

- LDA 的数学核心链：类内/类间散度矩阵 $\mathbf{S}_W, \mathbf{S}_B$ → Fisher 准则 $\max \mathbf{w}^T\mathbf{S}_B\mathbf{w} / \mathbf{w}^T\mathbf{S}_W\mathbf{w}$ → 广义特征值问题 $\mathbf{S}_B\mathbf{w} = \lambda \mathbf{S}_W\mathbf{w}$ → $\mathbf{S}_W^{-1}\mathbf{S}_B$ 特征分解 → 取最大 $q \leq K-1$ 个特征向量作为判别方向。
- $K-1$ 维上限来自 $\text{rank}(\mathbf{S}_B) \leq K-1$ 的秩论证——这是 LDA 区别于 PCA 最核心的数学约束。
- 当前源码 `LinearDiscriminantAnalysis(n_components=2, solver='svd')` 针对 Wine 数据（$K=3$）是最经典的监督降维配置。

# 数据构成

## 本章目标

1. 明确本仓库 LDA 数据来自 `DimensionalityData.lda()` 加载的 Wine 真实数据集。
2. 明确特征列与 `label` 在当前流水线中的角色差异——这是有监督降维，`label` 参与训练。
3. 明确标准化发生在什么位置，以及为什么它对基于散度矩阵的 LDA 至关重要。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `DimensionalityData.lda()` | 方法 | 加载 LDA 使用的 Wine 真实数据集 |
| `load_wine(as_frame=True)` | 函数 | scikit-learn 提供的红酒化学成分数据集加载器 |
| `lda_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `label` | 列名 | 3 分类标签——既参与 LDA 训练（定义类间/类内散度），也用于可视化着色 |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——散度矩阵计算的前置条件 |

## 1. 数据生成：`DimensionalityData.lda()`

当前 LDA 数据来自 `DimensionalityData.lda()`，底层调用 `sklearn.datasets.load_wine()`。

### 参数速览

适用函数：`load_wine(as_frame=True)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `as_frame` | `bool` | 是否返回带列名的 `DataFrame`。当前设为 `True` | `True`、`False` |
| 返回值 | `Bunch` 或 `(DataFrame, Series)` | `as_frame=True` 时返回含 `frame`（DataFrame）和 `target`（Series）的 Bunch 对象 | — |

### 示例代码

```python
data = load_wine(as_frame=True)
df = data.frame.copy().rename(columns={"target": "label"})
```

### 理解重点

- Wine 数据集是 UCI 经典真实数据集——178 个红酒样本，13 种化学成分特征，3 个葡萄品种类别。
- 标签列在源码中被统一重命名为 `label`——这使降维分册中的数据接口更一致（PCA 和 LDA 都使用此命名）。
- 与 PCA 使用的 `make_classification` 合成数据不同，LDA 使用真实数据——类别差异明显，适合展示监督降维的判别效果。

## 2. 特征列与 `label` 的角色

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(178, 13)$ | 含 13 个连续特征的特征矩阵，列名为 `alcohol`、`malic_acid`、`ash` 等 | `data.drop(columns=["label"])` |
| `y` | `ndarray`，形状 $(178,)$ | 类别标签 $y_i \in \{0, 1, 2\}$——**参与 LDA 训练**，定义类间/类内散度结构 | `data["label"].values` |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `label` 参与 LDA 的 `fit()`——它被用于计算各类均值 $\boldsymbol{\mu}_k$、类内散度 $\mathbf{S}_W$ 和类间散度 $\mathbf{S}_B$。
- 这与 PCA 分册有根本区别——PCA 的 `label` 仅用于可视化着色，不参与训练。LDA 的 `label` 既参与训练，也用于着色。
- Wine 数据恰好有 $K=3$ 个类别——这意味着 LDA 最多提取 $K-1=2$ 个判别方向，恰好可以完整展示在 2D 平面上。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(178, 13)$ | 去掉 `label` 后的全量特征矩阵 | `X` |
| 返回值 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- LDA 依赖散度矩阵的计算——$\mathbf{S}_W$ 和 $\mathbf{S}_B$ 的元素是平方和与叉积，对特征尺度高度敏感。
- Wine 数据的特征量纲差异巨大——`alcohol` 取值约 11-15，`proline` 取值约 278-1680。不标准化将使 `proline` 主导全部判别方向。
- 当前流水线没有 train/test split——直接在全量数据上 `fit_transform`。这是教学型简化，目标是展示整体判别结构而非评估泛化能力。

## 4. Wine 数据集特征清单

| 特征名 | 含义 | 典型量纲范围 |
|---|---|---|
| `alcohol` | 酒精含量 | 11.0 – 14.8 |
| `malic_acid` | 苹果酸 | 0.7 – 5.8 |
| `ash` | 灰分 | 1.4 – 3.2 |
| `alcalinity_of_ash` | 灰分碱度 | 10.6 – 30.0 |
| `magnesium` | 镁含量 | 70 – 162 |
| `total_phenols` | 总酚 | 0.98 – 3.88 |
| `flavanoids` | 黄酮类 | 0.34 – 5.08 |
| `nonflavanoid_phenols` | 非黄酮类酚 | 0.13 – 0.66 |
| `proanthocyanins` | 原花青素 | 0.41 – 3.58 |
| `color_intensity` | 颜色强度 | 1.3 – 13.0 |
| `hue` | 色调 | 0.48 – 1.71 |
| `od280/od315_of_diluted_wines` | 稀释酒 OD280/OD315 | 1.27 – 4.00 |
| `proline` | 脯氨酸 | 278 – 1680 |

### 理解重点

- 13 个特征中 `proline` 的量纲范围比 `nonflavanoid_phenols` 大约 3000 倍——不标准化的后果非常直观。
- 这三类分别对应意大利同一地区三种不同品种的葡萄酒——类别间化学成分确实存在系统差异，适合展示 LDA 的判别能力。

## 数据可视化

![类别分布图](https://img.yumeko.site/file/articles/ML/lda/data_class_distribution.png)

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/lda/data_correlation.png)

![二维特征空间](https://img.yumeko.site/file/articles/ML/lda/data_feature_space_2d.png)

## 常见坑

1. 把 `label` 当成 PCA 那样仅用于着色的辅助列——LDA 中 `label` 是训练输入，定义散度结构。
2. 忽略标准化——Wine 数据特征量纲差异巨大，散度矩阵计算被 `proline` 等大尺度特征主导。
3. 期望当前流水线有 train/test split——当前实现为教学型简化，直接在全量数据上训练和投影。
4. 误以为可以训练 `n_components=3` 的 LDA——$K=3$ 类数据最多 2 个判别方向。

## 小结

- 当前 LDA 数据来自 `load_wine(as_frame=True)`：178 个样本 × 13 个连续特征 × 3 个类别。
- 数据流为：`load_wine` → DataFrame（13 特征 + `label`）→ 全量标准化。
- `label` 既参与训练（定义 $\mathbf{S}_W$ 和 $\mathbf{S}_B$）也参与可视化（着色）——这是有监督降维与无监督降维在数据处理上的根本差异。
- Wine 真实数据集类别差异明显、特征量纲丰富——是展示 LDA 判别能力的理想教学数据。

# 思路与直觉

## 本章目标

1. 用直观方式理解 LDA 的判别式降维思路——"找一个方向，让不同类尽量分开、同类尽量靠拢"。
2. 理解为什么 LDA 需要标签参与训练，而 PCA 不需要——这是两种降维哲学的根本分歧。
3. 通过与 PCA 的对比，建立 LDA 在降维算法图中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 判别式降维 | 核心直觉 | 投影后不同类别的中心尽量远，同类样本尽量近——将"分类友好"作为降维目标 |
| 类间散度 | 概念 | 衡量不同类别中心之间的分离程度——LDA 想要最大化它 |
| 类内散度 | 概念 | 衡量同一类别内部样本的分散程度——LDA 想要最小化它 |
| 判别方向 | 结果 | 最大化类间散度与类内散度之比的投影方向 |
| `label` | 监督信息 | LDA 训练必需的类别标签——没有它就无法定义"类间"和"类内" |
| $K-1$ 维上限 | 约束 | $K$ 个类别最多只能提取 $K-1$ 个判别方向——3 类数据恰好 2D |

## 1. 为什么需要 LDA

LDA 回答的问题是：

> 如果要在低维空间里最好地区分不同类别，应该往哪个方向投影？

PCA 找的是"方差最大"的方向——它不关心类别，只关心数据整体散布。LDA 找的是"类别最可分"的方向——它利用标签信息，主动寻找让类别更清晰的低维表示。

### 理解重点

- LDA 把降维问题转化成了"让类别在低维空间中更容易分开"——这在分类任务中非常有价值。
- 这个直觉在日常中也有类比——看地图时旋转角度让行政区划更清晰，就是在做类似的事。
- 正因为需要知道"类别是什么"，LDA 必须有监督（需要标签），这是它和 PCA 最根本的区别。

## 2. 用"类间拉远、类内压紧"理解优化目标

LDA 的工作方式可以想象成：

1. **看看各类的中心在哪**——用标签统计每类的均值
2. **看看类内有多散**——用标签统计每类内部样本的分布
3. **猜一个投影方向**——把高维数据投影到一条线上
4. **算一下投影后类中心多远、类内多紧**——这就是 Fisher 准则的值
5. **换个方向再试**——不断调整直到找到"类间最远、类内最紧"的方向

### 理解重点

- 这就像给三群鸟拍照——找一个角度让三群鸟在照片里看起来距离最远、每群最紧密，而不是拍出鸟群整体的展翅范围。
- LDA 关心的不是"数据变化最大的方向"（PCA），而是"类别差异最明显的方向"。
- 每一步"换个方向"在数学上就是广义特征值问题——数学原理章节会展开推导，这里只需建立直觉。

## 3. 为什么 Wine 数据特别适合讲 LDA

当前数据来自 Wine 真实数据集（178 样本 × 13 特征 × 3 类）：

- 三类葡萄品种的化学成分确实存在系统差异——类间散度天然较大
- 13 个特征包含多种量纲——标准化后散度矩阵才具有几何意义
- 3 个类别恰好对应 2 个判别方向——可以完整展示在 2D 平面上

### 理解重点

- 如果类别本来就高度重叠（类间散度 ≈ 类内散度），LDA 找到的投影方向也难以让图像明显分开。
- Wine 数据集的类别差异在化学成分上确实存在——这使得 LDA 的判别效果直观可感。
- 使用真实数据集（而非 `make_classification` 合成数据）增加了教学的真实感——读者可以看到 LDA 在真实化学测量上的表现。

## 4. 为什么 `label` 在这里是真正的训练输入

LDA 的优化目标（类间散度 / 类内散度）天然依赖类别标签：

- 没有标签 → 无法定义"哪是类内、哪是类间"
- 没有类结构和类结构 → Fisher 准则无从计算
- 没有 Fisher 准则 → LDA 退化成无目标的降维

### 理解重点

- 在 PCA 分册中，`label` 只是着色用的"参考信息"——删掉它 PCA 照样训练。
- 在 LDA 分册中，`label` 是训练必需的"监督信号"——删掉它 LDA 无法工作。
- 这是两套降维哲学的分水岭：PCA 是"数据怎么说"，LDA 是"标签怎么引导"。

## 5. 为什么当前只输出 2D LDA 图

Wine 数据有 3 个类别：

- LDA 最多提取 $K-1 = 2$ 个判别方向
- `n_components=2` 已经达到理论上限——不可能再训练 `n_components=3`
- 因此 2D 图既是"全部判别信息的完整展示"，也是"数学约束的必然结果"

### 理解重点

- 这不是工程上的偷懒——PCA 可以随意扩展到 3D/4D，LDA 做不到。
- 对 Wine 数据而言，2 个判别方向已经捕获了全部类别可分信息——不存在"丢掉的第 3 个判别方向"。
- 这个 $K-1$ 约束是 LDA 最需要建立直觉的数学性质——它直接决定了你能降多少维。

## 6. 与 PCA 的直觉对比

| 维度 | PCA | LDA |
|---|---|---|
| 核心问题 | 哪个方向方差最大？ | 哪个方向类别最可分？ |
| 标签 | 不需要——只用特征 | 必须——标签定义类别结构 |
| 投影效果 | 保留最多信息，但类别可能混在一起 | 保留最多类别差异，但可能丢失部分方差结构 |
| 维数限制 | 最多到 $\min(d, N)$ | 最多到 $K-1$ |
| 理想场景 | 数据压缩、去噪、探索性可视化 | 分类预处理、判别式特征提取 |
| 类比 | 给整群鸟拍"展翅全景" | 给三群鸟拍"分群特写" |

### 理解重点

- PCA 和 LDA 不是"谁更好"——它们的目标根本不同。PCA 回答"数据怎么分布"，LDA 回答"类别怎么区分"。
- 如果任务是分类，LDA 通常更合适——它主动利用标签信息。如果任务是压缩，PCA 更合适——它不依赖标签。
- 一个有用的对照实验：在 Wine 数据上分别运行 PCA（2D）和 LDA（2D），观察同一份数据在两种目标下的投影差异。

## 可视化

![二维降维结果](https://img.yumeko.site/file/articles/ML/lda/dimensionality_2d.png)

## 常见坑

1. 把 LDA 当成"带标签的 PCA"——两者优化目标根本不同（类间/类内比 vs 方差）。
2. 忽略 $K-1$ 维上限，期望像 PCA 一样自由增加输出维数。
3. 忘记 `label` 在 LDA 中是真正的训练输入——删掉它算法无法运行。
4. 在 PCA 和 LDA 之间做优劣比较而不考虑任务需求——它们是不同目标的工具。

## 小结

- LDA 的直觉核心是有监督判别式降维：用标签定义类间/类内散度 → 找最大化两者之比的方向 → 投影后类别更可分。
- Wine 真实数据集（3 类 13 特征）类别差异明显——是展示 LDA 判别能力的最佳教学数据。
- LDA 与 PCA 在直觉上截然不同：一个以类别为核心找"最可分"方向，一个以数据为核心找"最大方差"方向——选哪个取决于任务目标，而非算法优劣。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LinearDiscriminantAnalysis`。
2. 理解 `LinearDiscriminantAnalysis` 的核心构造器参数（`n_components`、`solver`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`explained_variance_ratio_`（判别方向贡献）、`scalings_`（判别向量）、`means_`（类均值）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 模型，打印解释方差比日志 |
| `LinearDiscriminantAnalysis(...)` | 类 | scikit-learn 提供的线性判别分析器——求解广义特征值问题以找到判别方向 |
| `model.fit(X_train, y_train)` | 方法 | 学习判别方向——有监督，标签用于计算类均值、类内散度和类间散度 |
| `model.explained_variance_ratio_` | 属性 | 各判别方向的特征值占比——反映每个方向的相对判别能力 |
| `model.scalings_` | 属性 | 判别向量矩阵——将原始特征空间映射到判别子空间的线性变换 |
| `model.transform(X)` | 方法 | 将数据投影到判别子空间——生成降维后的坐标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_components=2, solver='svd')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(178, 13)$ | 标准化后的特征矩阵，传入 `LinearDiscriminantAnalysis.fit()` | `X_scaled` |
| `y_train` | `array_like`，形状 $(178,)$ | 类别标签 $\{0, 1, 2\}$——LDA 训练必需的监督信息，用于定义类间/类内散度 | `y` |
| `n_components` | `int` | 保留的判别方向数。当前设为 `2`——$K=3$ 类数据的理论上限 | `1`、`2` |
| `solver` | `str` | 求解器。`'svd'`（默认）通过 SVD 求解，无需显式计算 $\mathbf{S}_W^{-1}$；`'eigen'` 直接特征分解；`'lsqr'` 最小二乘 | `'svd'`、`'eigen'`、`'lsqr'` |
| 返回值 | `LinearDiscriminantAnalysis` | 已完成 `fit()` 的模型对象，含 `explained_variance_ratio_`、`scalings_`、`means_` 等属性 | — |

### 示例代码

```python
from model_training.dimensionality.lda import train_model

model = train_model(X_scaled, y, n_components=2)
```

### 理解重点

- 和 PCA 分册不同，`train_model(...)` **必须有 `y_train` 参数**——LDA 是有监督降维，标签用于定义类内散度 $\mathbf{S}_W$ 和类间散度 $\mathbf{S}_B$。
- `n_components=2` 不是随意选的默认值——Wine 数据 $K=3$ 类，理论上限恰好就是 $K-1=2$。
- `train_model(...)` 是对 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的薄封装——算法本体是 scikit-learn 基于 SVD 的高效实现。

## 2. `LinearDiscriminantAnalysis` 构造器参数

### 参数速览

适用 API：`LinearDiscriminantAnalysis(n_components=2, solver='svd')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的判别方向数。必须 $\leq K-1$。当前设为 `2` | `1`、`2` |
| `solver` | `str` | 求解器选择。`'svd'`（默认）通过 SVD 求解，数值稳定性好，支持 `explained_variance_ratio_`；`'eigen'` 直接特征分解；`'lsqr'` 最小二乘，不支持 `explained_variance_ratio_` | `'svd'`、`'eigen'`、`'lsqr'` |
| `priors` | `array_like` 或 `None` | 类先验概率。`None`（默认）使用各类样本频率 $N_k/N$ | `None`、`[0.3, 0.3, 0.4]` |
| `shrinkage` | `float` 或 `str` 或 `None` | 收缩参数——仅在 `solver='lsqr'` 或 `'eigen'` 时可用，用于正则化 $\mathbf{S}_W$ 估计 | `None`、`'auto'`、`0.5` |
| `tol` | `float` | 特征值筛选的数值容忍度——仅 `solver='eigen'` 时使用。默认 `1e-4` | `1e-3`、`1e-4` |
| `covariance_estimator` | `CovarianceEstimator` 或 `None` | 协方差估计器——scikit-learn 1.2+ 新增参数 | `None` |

### 示例代码

```python
model = LinearDiscriminantAnalysis(
    n_components=2,
    solver="svd",
)
model.fit(X_train, y_train)
```

### 理解重点

- LDA 的核心参数是 `n_components`——它直接决定了降维后的维数，但受 $K-1$ 上限约束。
- `solver='svd'` 是 scikit-learn 的默认黄金标准——数值稳定、不经由显式矩阵求逆、且支持 `explained_variance_ratio_`。
- LDA 的 `fit()` 是解析求解（广义特征值分解）——与 KMeans 的迭代优化和 DBSCAN 的密度扩展在计算特征上截然不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `explained_variance_ratio_` | `ndarray`，形状 `(n_components,)` | $\lambda_j / \sum_i \lambda_i$ | 各判别方向的特征值占比——反映每个方向对类别分离的相对贡献。仅 `solver='svd'` 和 `'eigen'` 时可用 |
| `scalings_` | `ndarray`，形状 `(n_features, n_components)` | 判别向量 $\mathbf{w}_1, \dots, \mathbf{w}_q$ | 将 13 维特征映射到 2 维判别子空间的线性变换矩阵 |
| `means_` | `ndarray`，形状 `(n_classes, n_features)` | $\boldsymbol{\mu}_k$ | 各类在原始特征空间中的均值向量——当前为 $3 \times 13$ 矩阵 |
| `priors_` | `ndarray`，形状 `(n_classes,)` | $\pi_k = N_k/N$ | 各类的先验概率 |
| `classes_` | `ndarray`，形状 `(n_classes,)` | 类别标签 | 训练数据中出现的类别标签——当前为 `[0, 1, 2]` |
| `xbar_` | `ndarray`，形状 `(n_features,)` | $\boldsymbol{\mu}$ | 全局均值向量——$\mathbf{S}_B$ 计算的基准 |

### 示例代码

```python
print(f"n_components: {n_components}")
if hasattr(model, "explained_variance_ratio_"):
    print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
    print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- `explained_variance_ratio_` 是理解判别方向相对重要性的关键——第一个方向通常捕获绝大部分类别分离能力。
- `scalings_` 是 LDA 区别于 PCA 的标志性属性——PCA 有 `components_`（主成分方向），LDA 有 `scalings_`（判别方向）。名称不同，数学含义也不同。
- `explained_variance_ratio_` 的条件可用性（`hasattr` 检查）反映了不同求解器的工程差异——`svd` 和 `eigen` 支持，`lsqr` 不支持。

## 4. `transform()` ：从模型训练到降维输出的桥梁

### 参数速览

适用方法：`model.transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n, 13)` | 经过同一 `scaler` 标准化后的特征矩阵 | `X_scaled` |
| 返回值 | `ndarray`，形状 `(n, 2)` | 投影到判别子空间后的坐标——$\mathbf{X} \cdot \text{scalings\_}$ | `X_transformed` |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- `fit()` 学习判别方向（`scalings_`），`transform()` 执行投影——两者分离的设计使模型可以对新数据重复投影。
- 流水线中 `X_transformed` 是 `plot_dimensionality(...)` 的直接输入——它是训练和可视化的桥梁。
- 与 PCA 的 `transform()` 语法相同、语义不同——PCA 投影到方差最大方向，LDA 投影到类别最可分方向。

## 5. 训练阶段的工程封装

除了 `LinearDiscriminantAnalysis(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察判别方向学习耗时——通常极快（毫秒级） |
| `n_components` 日志 | 确认当前判别方向数 |
| `explained_variance_ratio_` 日志 | 若求解器支持，打印各方向解释占比和累计值 |
| `timer(...)` 上下文 | 单独测量 `fit()` 阶段的耗时 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过条件判断保护性地输出解释比例。
- `explained_variance_ratio_` 的条件打印（`hasattr` 检查）是 LDA 特有的工程边界处理——不同求解器的属性可用性不同。
- LDA 不打印簇数量（不是聚类）、不打印准确率（当前目标是降维而非分类）——与监督分类和聚类分册的输出各有侧重。

## 常见坑

1. 误以为 `train_model(...)` 不需要传 `y_train`——LDA 是有监督算法，标签是训练必需的输入。
2. 忽略 `n_components` 受 $K-1$ 约束——对 3 类数据传 `n_components=3` 会报错。
3. 把 `explained_variance_ratio_` 当成所有求解器都支持的属性——`lsqr` 求解器没有此属性。
4. 把 `scalings_` 当成 PCA 的 `components_`——两者优化目标不同，方向含义不同。

## 小结

- `train_model(...)` 是本仓库 LDA 的核心训练入口，是对 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的薄封装。
- `LinearDiscriminantAnalysis` 的核心参数是 `n_components`（判别方向数，受 $K-1$ 约束）和 `solver`（求解路径，决定属性可用性）。
- 训练完成后的核心属性：`explained_variance_ratio_`（方向贡献）、`scalings_`（判别向量）、`means_`（类均值）——三者分别回答了"哪个方向更重要""怎么投影""各类在哪"。
- LDA 有 `scalings_`、有 `explained_variance_ratio_`、有 `transform()` 方法、标签必需参与 `fit()`——这四点构成了它与 PCA 最核心的工程差异。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 LDA 流水线从数据复制到 2D 判别图输出的完整步骤。
2. 理解 LDA 有监督训练、无切分、`transform()` 为输出的工程特征——与 PCA 既有相似又有本质差异。
3. 理解 `label` 在当前流程中的双重角色——既是训练输入，也是可视化着色依据。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `lda_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `data.drop(columns=["label"])` | 操作 | 去掉标签列，保留 13 个特征作为训练输入 |
| `StandardScaler().fit_transform(X)` | 方法 | 对全量特征做一致性标准化——散度矩阵计算的前置条件 |
| `train_model(X_scaled, y, n_components=2)` | 函数 | 训练 LDA 模型——有监督，标签参与判别方向学习 |
| `model.transform(X_scaled)` | 方法 | 将 13 维特征投影到 2 维判别子空间——生成降维坐标 |
| `plot_dimensionality(...)` | 函数 | 绘制降维后的 2D 散点图（按类别着色） |

## 1. 流水线起点：复制数据并拆出特征与标签

### 示例代码

```python
data = lda_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `lda_data`。
- `label` 被单独保存为 `y`——它**既参与后续 `train_model()`**，也用于最终的 `plot_dimensionality(...)` 着色。
- 与 PCA 流水线最关键的区别：`y` 在这里是训练输入，PCA 的 `y` 仅用于着色。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(178, 13)$ | 去掉 `label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- LDA 流水线**没有**训练/测试切分——当前实现为教学型简化，直接在全量数据上训练和投影。
- `fit_transform` 直接在全量数据上计算统计量并变换——目标是展示整体判别结构而非评估泛化能力。
- 标准化是必须的——散度矩阵 $\mathbf{S}_W$ 和 $\mathbf{S}_B$ 的元素对特征尺度高度敏感。Wine 数据中 `proline`（~278-1680）的量纲远大于 `nonflavanoid_phenols`（~0.13-0.66）。

## 3. 训练阶段：`fit()` 学习判别方向

### 参数速览

适用 API：`train_model(X_scaled, y, n_components=2)` → `model.fit(X_scaled, y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(178, 13)$ | 标准化后的特征矩阵——LDA 的特征输入 | `X_scaled` |
| `y` | `ndarray`，形状 $(178,)$ | 类别标签 $\{0, 1, 2\}$——**训练必需的监督信息** | `y` |
| `n_components` | `int` | 保留的判别方向数。当前为 `2`（$K=3$ 类数据的上限） | `2` |

### 示例代码

```python
model = train_model(X_scaled, y, n_components=2)
```

### 理解重点

- `LinearDiscriminantAnalysis.fit(X_scaled, y)` 内部流程：计算各类均值 $\boldsymbol{\mu}_k$ → 构造 $\mathbf{S}_W$ 和 $\mathbf{S}_B$ → 通过 SVD 求解广义特征值问题 → 取最大的 2 个特征值对应的特征向量作为判别方向 → 存入 `scalings_`。
- 这**既有训练也有模型**——LDA 产出一个可复用的投影矩阵（`scalings_`），可以对任意新数据做 `transform()`。
- 与 PCA 对比：`fit()` 都产出一个投影矩阵，但 PCA 不需要 `y`（无监督），LDA 必须有 `y`（有监督）。

## 4. 投影阶段：`transform()` 将数据降到 2D

### 参数速览

适用方法：`model.transform(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(178, 13)$ | 需要降维的特征矩阵——当前为训练数据本身 | `X_scaled` |
| 返回值 | `ndarray`，形状 $(178, 2)$ | 投影到判别子空间后的 2D 坐标 | `X_transformed` |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- `transform()` 是 LDA 流水线的"预测"步骤——它不是预测类别标签，而是预测低维坐标。
- 降维后的 2D 坐标就是判别子空间中的位置——同类样本应聚集，不同类样本应分离。
- 当前流水线对训练数据本身做 `transform()`（全量投影），目的是展示整体判别结构——这与"对新样本做投影"在数学上完全相同。

## 5. 2D 判别图：可视化如何接入流水线

### 参数速览

适用函数：`plot_dimensionality(X_transformed, y=y, explained_variance_ratio=evr, title=..., dataset_name=..., model_name=..., mode='2d')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_transformed` | `ndarray`，形状 $(178, 2)$ | LDA 降维后的 2D 坐标 | `X_transformed` |
| `y` | `ndarray`，形状 $(178,)$ | 类别标签——用于散点着色和图例 | `y` |
| `explained_variance_ratio` | `ndarray` 或 `None` | 各判别方向贡献占比——若存在则标注在坐标轴上 | `evr` |
| `mode` | `str` | 输出模式。当前为 `'2d'`——LDA 只输出 2D 图 | `'2d'` |

### 示例代码

```python
evr = (
    model.explained_variance_ratio_
    if hasattr(model, "explained_variance_ratio_")
    else None
)
plot_dimensionality(
    X_transformed,
    y=y,
    explained_variance_ratio=evr,
    title="LDA 降维 (2D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="2d",
)
```

### 理解重点

- `plot_dimensionality(...)` 是当前 LDA 分册唯一的可视化函数——与分类分册的四类评估（混淆矩阵+ROC+决策边界+学习曲线）完全不同。
- `explained_variance_ratio` 的条件传递（`if hasattr` → `evr` 或 `None`）是 LDA 特有的工程模式——不同求解器对此属性的支持不同。
- 图中 `y` 既是训练标签也是着色依据——它在当前分册中有双重作用。坐标轴标签会包含解释占比（如 `LD1 (78.5%)`）。

## 6. 用伪代码看完整流程

```python
data = lda_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values

X_scaled = StandardScaler().fit_transform(X)

model = train_model(X_scaled, y, n_components=2)
X_transformed = model.transform(X_scaled)

plot_dimensionality(X_transformed, y=y, explained_variance_ratio=evr, mode="2d")
```

### 理解重点

- 当前 LDA 流水线的主线非常清楚：取数 → 标准化 → 有监督训练 → 判别投影 → 2D 可视化。
- 这条链路里最关键的中间变量是：`X_scaled`（标准化特征）、训练后的 `model`（含 `scalings_`）、二维投影结果 `X_transformed` 和标签 `y`。
- 与 PCA 流水线的步骤形式极其相似（StandardScaler → fit → transform → plot），但 `fit()` 是否传 `y` 是两套流程的本质分野。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/articles/ML/lda/learning_curve.png)

## 常见坑

1. 把 LDA 流水线写成 PCA 那种"标签仅用于着色"的无监督流程——`y` 在 LDA 中是训练输入。
2. 期望当前流水线有 `train_test_split`——当前实现为教学型简化，直接在全量数据上训练和投影。
3. 忘记 `transform()` 才是生成降维坐标的步骤——`fit()` 只学习判别方向，不生成投影坐标。
4. 把 `explained_variance_ratio_` 当成一定存在的属性——需用 `hasattr` 做保护式判断。

## 小结

- 当前 LDA 流水线非常清晰：复制数据 → 拆出 `X` 和 `y` → 全量标准化 → `fit(X, y)` 学习判别方向 → `transform(X)` 投影到 2D → 判别散点图。
- 与 PCA 流水线的核心差异：`y` 参与训练（有监督）、降维上限 $K-1$（非 $\min(d,N)$）、优化目标不同（类间/类内比 vs 方差）。
- 与分类分册的核心差异：输出是低维坐标而非类别预测、可视化是降维散点图而非混淆矩阵/ROC、无 `predict()`（用 `transform()` 替代）。

# 评估与诊断

## 本章目标

1. 明确当前仓库 LDA 实现的评估手段——2D 判别投影图 + `explained_variance_ratio_` 日志。
2. 理解判别投影图应观察什么——类别分离度、类内紧凑度、判别方向贡献。
3. 理解 LDA 评估与 PCA 评估和分类评估的本质差异——关注类别可分性而非方差保留率或分类精度。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_dimensionality(...)` | 函数 | 绘制降维后的 2D 判别散点图（按类别着色，轴标注解释占比） |
| `explained_variance_ratio_` | 指标 | 各判别方向的特征值占比——反映每个方向对类别分离的相对贡献 |
| 累计解释方差 | 派生量 | 前 $q$ 个判别方向总共捕获的类别分离信息比例——$q=2$ 时为 100%（达理论上限） |
| 类别分离度 | 视觉诊断 | 不同颜色在 2D 空间中是否明显分开——LDA 效果的最直观判断 |
| 类内紧凑度 | 视觉诊断 | 同色样本是否聚集在一起——类内散度被压缩的程度 |

## 1. 当前仓库的评估入口

LDA 的评估与分类分册截然不同——没有混淆矩阵、没有 ROC 曲线、没有准确率。评估依赖两种手段：

1. **2D 判别投影图** —— 将 13 维数据投影到 2 维判别子空间，按类别着色，直观观察类别分离效果
2. **`explained_variance_ratio_` 日志** —— 打印各判别方向的特征值占比和累计值（若求解器支持）

### 示例代码

```python
plot_dimensionality(
    X_transformed,
    y=y,
    explained_variance_ratio=evr,
    title="LDA 降维 (2D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="2d",
)
```

### 理解重点

- 降维评估不能像分类那样用 `accuracy`——LDA 的目标是让类别在低维空间中更可分，而非直接预测类别。
- 当前评估方法对教学场景来说足够直观——读者可以一眼看出 3 类葡萄酒在判别空间中是否形成了清晰的分离。
- `explained_variance_ratio_` 是最简明的定量参考——它说明了每个判别方向对类别分离的贡献占比。

## 2. 2D 判别投影图能观察什么

### 参数速览

适用函数：`plot_dimensionality(X_transformed, y=y, explained_variance_ratio=evr, mode='2d')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_transformed` | `ndarray`，形状 `(178, 2)` | LDA 降维后的 2D 坐标——判别子空间中的样本位置 | `X_transformed` |
| `y` | `ndarray`，形状 `(178,)` | 类别标签——用于散点着色和图例区分 | `y` |
| `explained_variance_ratio` | `ndarray` 或 `None` | 各判别方向贡献占比——若存在则用于轴标签（如 `LD1 (78.5%)`） | `evr` |

### 理解重点

- 图中每个点代表一个红酒样本，颜色代表葡萄品种（3 类）。
- 坐标轴标签格式为 `LD1 (xx.x%)`——`LD` 表示 Linear Discriminant，括号内为该方向的特征值占比。
- 观察重点：
  - **类别分离度**——不同颜色的点群是否在空间中明显分开？理想情况下三类应形成三个不重叠的簇
  - **类内紧凑度**——同色样本是否紧密聚集？类内散度被压缩得越好，同色点越集中
  - **判别方向贡献**——LD1 通常捕获大部分分离信息（如 70-80%），LD2 捕获剩余部分
  - **边界清晰度**——三个簇之间的边界是否清晰，是否存在明显的重叠区域
- 与 PCA 2D 图的关键区别：LDA 图上的类别分离效果通常显著优于 PCA——因为 LDA 主动以类别可分性为优化目标。

## 3. `explained_variance_ratio_` 能告诉我们什么

### Wine 数据的典型输出

运行当前流水线，终端通常输出类似：

```
解释方差比: [0.6875 0.3125]
累计解释方差: 1.0000
```

### 理解重点

- 累计解释方差为 1.0000（100%）并非偶然——$K=3$ 类数据最多 2 个非零特征值，`n_components=2` 保留了全部判别信息。
- 第一个判别方向（LD1）通常捕获 60-80% 的类别分离能力——它在散点图中对应横轴，往往是最能拉开类别的方向。
- 第二个判别方向（LD2）捕获剩余 20-40%——与 LD1 正交，提供额外的类别分离维度。
- 这不同于 PCA 的 `explained_variance_ratio_`——PCA 的分量是"方差占比"，LDA 的分量是"判别能力占比"。

## 4. LDA 与 PCA 在相同数据上的评估对比

| 评估维度 | PCA（`n_components=2`） | LDA（`n_components=2`） |
|---|---|---|
| 图标题 | `PCA 降维 (2D)` | `LDA 降维 (2D)` |
| 轴标注 | `PC1 (xx.x%)` / `PC2 (xx.x%)` | `LD1 (xx.x%)` / `LD2 (xx.x%)` |
| 类别分离度 | 通常较弱——PCA 不利用标签信息 | 通常较强——LDA 主动最大化类间/类内比 |
| 累计解释比例含义 | 保留了多少总方差 | 保留了多少类别分离能力 |
| 能否达 100% | 通常不能——2D 不可能捕获全部方差 | Wine 数据上必然 100%——$K-1=2$ 已取全部非零特征值 |

### 理解重点

- 在 Wine 数据上做 PCA（2D）vs LDA（2D）的对照实验，是理解两种降维哲学差异的最直观方式。
- 典型结果：PCA 2D 图上三类可能有明显重叠，而 LDA 2D 图上三类分离清晰——这是一份数据在两种目标下的投影差异。
- 这不表示 LDA 比 PCA"更好"——它只表示 LDA 在"类别分离"这个特定目标上更有效。

## 5. 当前实现中尚未纳入的评估手段

| 评估手段 | 说明 | 当前未使用的原因 |
|---|---|---|
| 降维后分类器精度对比 | 在 LDA 投影后的 2D 特征上训练分类器，与原始 13 维特征的分类精度对比 | 当前侧重展示判别结构本身，而非下游分类应用 |
| 混淆矩阵 / ROC 曲线 | LDA 可直接作为分类器使用（`predict()` 方法），但当前流水线未调用 | 当前分册定位为降维而非分类 |
| 3D LDA 可视化 | $K \geq 4$ 类数据可降至 3D | Wine 数据 $K=3$，最多 2D——不受此限制影响 |
| 与 PCA 的量化对比表 | 在同一数据上对比两种降维后分类精度的差异 | 教学型仓库优先展示视觉对比而非数值表格 |

### 理解重点

- 当前流水线不显式计算这些指标——文档可以提到它们是扩展方向，但不能写成"当前源码已在计算"。
- LDA 实际上是一个完整的分类器（基于高斯判别分析），但在当前仓库中它被用作降维工具——这是设计上的有意定位。
- 对于教学型仓库，2D 判别投影图 + `explained_variance_ratio_` 日志已经能有效支撑 LDA 的诊断需求。

## 评估图表

![二维降维结果](https://img.yumeko.site/file/articles/ML/lda/dimensionality_2d.png)

## 常见坑

1. 把 PCA 的评估框架（"方差保留了多少"）搬到 LDA——LDA 关注类别分离能力而非方差保留。
2. 看到累计解释方差为 100% 就认为模型过拟合——这是 $K-1=2$ 时 `n_components=2` 取全非零特征值的数学必然。
3. 把 LDA 2D 图上的类别分离度当成分类精度的替代品——分离度好不等于分类决策边界最优。
4. 忽略 `explained_variance_ratio_` 可能为空的情况——`lsqr` 求解器不提供此属性。

## 小结

- 当前仓库对 LDA 的评估简洁而直观：2D 判别投影图看类别分离效果，`explained_variance_ratio_` 日志看各判别方向贡献。
- LDA 没有混淆矩阵、没有 ROC、没有准确率——这些是监督分类的评估手段。当前分册关注的是"降维后类别是否更可分"。
- LDA 评估与 PCA 评估的核心差异：前者问"类别分开了吗"，后者问"方差保留了多少"——两种目标、两种观察重点。

# 工程实现

## 本章目标

1. 从工程角度看清 LDA 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和降维可视化分别负责什么。
3. 理解 LDA 工程实现与 PCA 在架构上的相似性与关键差异——有监督 `fit()`、`transform()` 投影、$K-1$ 维约束。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.lda()` 加载 Wine 真实数据集 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `lda_data` |
| 训练封装 | `model_training/dimensionality/lda.py` | 构建并训练 `LinearDiscriminantAnalysis`，打印解释方差比日志 |
| 流水线入口 | `pipelines/dimensionality/lda.py` | 组织数据拆分、标准化、训练、投影与降维可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制降维后的 2D 散点图（按类别着色，轴标注解释占比） |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.dimensionality.lda
```

### 理解重点

- 这个命令串起当前 LDA 分册中最核心的工程流程。
- 依次完成：数据复制 → 拆出 `X` 和 `y` → 全量标准化 → LDA `fit(X, y)`（学习判别方向）→ `transform(X)`（投影到 2D）→ 判别散点图。
- 对大多数读者来说，`pipelines/dimensionality/lda.py` 是理解工程实现的最佳起点——代码量少、流程清晰。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与标签
    data = lda_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values

    # 2. 全量标准化——无切分（教学型简化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 有监督训练——fit() 学习判别方向（y 是训练输入）
    model = train_model(X_scaled, y, n_components=2)

    # 4. 判别投影——transform() 将 13 维降到 2 维
    X_transformed = model.transform(X_scaled)

    # 5. 单一可视化（2D 判别散点图）
    evr = (
        model.explained_variance_ratio_
        if hasattr(model, "explained_variance_ratio_")
        else None
    )
    plot_dimensionality(
        X_transformed,
        y=y,
        explained_variance_ratio=evr,
        title="LDA 降维 (2D)",
        dataset_name=DATASET,
        model_name=MODEL,
        mode="2d",
    )
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的广义特征值求解在 `LinearDiscriminantAnalysis.fit()` 中。
- 数据流是单向的：数据 → 标准化 → 判别方向学习 → 投影 → 2D 散点图。
- 与分类流水线的核心差异：
  - **无 `predict()` 调用**——LDA 在这里是降维工具，输出是 `transform()` 而非类别标签
  - **无 `predict_proba`**——当前流水线不涉及分类概率
  - **单一可视化**（`plot_dimensionality`）而非四类（混淆矩阵+ROC+决策边界+学习曲线）
- 与 PCA 流水线的核心差异：
  - **`fit()` 传入 `y`**——LDA 是有监督的，PCA 是无监督的
  - **`n_components` 受 $K-1$ 约束**——不能像 PCA 那样自由扩展

## 3. 训练模块负责什么

`model_training/dimensionality/lda.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `LinearDiscriminantAnalysis(n_components=2, solver='svd')` 实例
2. 调用 `model.fit(X_train, y_train)`——学习判别方向（有监督，标签用于计算散度矩阵）
3. 打印 `n_components`、`explained_variance_ratio_` 和累计解释方差（若求解器支持）
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, y_train, n_components=2, solver='svd')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，传入 `LDA.fit()` | `X_scaled` |
| `y_train` | `array_like` | 类别标签 $\{0, 1, 2\}$——LDA 训练必需的监督信息 | `y` |
| `n_components` | `int` | 保留的判别方向数。默认 `2` | `1`、`2` |
| `solver` | `str` | 求解器。默认 `'svd'` | `'svd'`、`'eigen'`、`'lsqr'` |
| 返回值 | `LinearDiscriminantAnalysis` | 已完成 `fit()` 的模型对象，含 `scalings_`、`explained_variance_ratio_` 等 | — |

### 理解重点

- LDA 的 `fit()` 是有监督的——接收 `y_train` 参数。这与 PCA 分册形成鲜明对比，与分类分册一致。
- 训练日志中的 `explained_variance_ratio_` 是 LDA 独有的统计输出——它反映判别方向贡献，PCA 的对应物含义不同（方差占比 vs 判别能力占比）。
- `@print_func_info` 和 `@timeit` 装饰器提供函数标题和耗时——增强了教学型仓库的可读性。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 降维可视化 | `plot_dimensionality(...)` | `X_transformed`（投影坐标）、`y`（着色标签）、`explained_variance_ratio`（轴标注） | 2D 判别散点图（PNG），坐标轴标注解释占比 |

### 理解重点

- `plot_dimensionality(...)` 是当前 LDA 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- `explained_variance_ratio` 参数用于在坐标轴上标注各判别方向的贡献占比（如 `LD1 (68.8%)`）。
- 支持 `mode='2d'` 和 `mode='3d'` 两种模式——但 LDA 当前只使用 2D 模式（受 $K-1=2$ 约束）。
- 与 PCA 共用同一个可视化函数——区别在于传入的数据含义不同（判别投影 vs 主成分投影），以及轴标签前缀不同（`LD` vs `PC`）。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `lda_data` | `data_generation/dimensionality.py` | `pipelines/dimensionality/lda.py` |
| `y` | `data["label"]` 提取 | `train_model`（训练输入）、`plot_dimensionality`（着色） |
| `X_scaled` | `StandardScaler` | `train_model`、`model.transform` |
| `model`（含 `scalings_`、`explained_variance_ratio_`） | `train_model(...)` | `model.transform`、终端日志 |
| `X_transformed` | `model.transform(X_scaled)` | `plot_dimensionality` |
| 图片产物 | `plot_dimensionality(...)` | `outputs/lda/` 目录 |

### 理解重点

- 数据依赖关系中有 6 个节点——比 PCA 流水线多了 `y` 流向 `train_model` 这一条边（有监督的核心差异）。
- 比分类流水线少了 `train_test_split`、`predict`、`predict_proba`、PCA、ROC 评估、学习曲线等节点。
- `y` 的流向是扇出的——同时流入 `train_model`（训练）和 `plot_dimensionality`（着色）。这是 LDA 流水线数据流的最关键特征。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `LDA 降维流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`n_components`、`explained_variance_ratio_`（各方向 + 累计）、函数耗时 | 查看判别方向学习耗时和各方向贡献占比 |
| 降维图 | `outputs/lda/lda_dim_2d.png` | 2D 判别散点图——Wine 3 类在判别子空间中的分布 |

### 理解重点

- 输出比分类分册少得多——只有 2 类（日志 + 1 张图），而非 5 类（日志 + 4 张图）。
- `explained_variance_ratio_` 是 LDA 独有的日志输出——它在 PCA 的日志中也存在但含义不同（判别能力占比 vs 方差占比）。
- 2D 判别散点图是最核心的教学产出——它直接展示了有监督降维将高维类别差异映射到低维空间的效果。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/dimensionality/lda.py` — 入口，代码量少，流程清晰
2. 再看 `model_training/dimensionality/lda.py` — 训练封装，理解有监督 `fit(X, y)` 和条件性日志输出
3. 再看 `result_visualization/dimensionality_plot.py` — 降维散点图绘制逻辑（含 `explained_variance_ratio` 轴标注）
4. 最后回到 `data_generation/dimensionality.py` — 理解 Wine 数据集加载和标签重命名

### 理解重点

- 从入口看整体流程，再下钻到训练和可视化细节，阅读成本最低。
- LDA 的调用链与 PCA 几乎一致（同一套模块分层），差异集中在 `fit()` 是否传 `y` 和 `n_components` 的上限约束。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/articles/ML/lda/result_display.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的广义特征值求解在 `LinearDiscriminantAnalysis.fit()` 中。
2. 期待当前流水线有 `train_test_split` 或 `predict()` 调用——LDA 在这里是降维工具，输出低维坐标而非类别标签。
3. 忽略 `explained_variance_ratio_` 的条件可用性——`lsqr` 求解器不提供此属性。
4. 把 `label` 的流向写成仅到可视化——它同时流入 `train_model`（训练输入）。
5. 忘记 `n_components=2` 是 $K-1$ 约束的结果而非随意选择——与 PCA 的 `n_components` 无此约束形成对比。

## 小结

- 当前 LDA 工程实现采用与 PCA 一致的模块分层：数据生成 → 训练封装（有监督）→ 流水线编排 → 单一可视化（判别散点图）。
- `run()` 负责串联，`train_model(...)` 负责判别方向学习（`fit(X, y)`），`plot_dimensionality(...)` 负责降维可视化。
- LDA 在工程上最不同于 PCA 的地方：`fit()` 有 `y` 参数（有监督）、`n_components` 受 $K-1$ 约束、`explained_variance_ratio_` 含义不同（判别能力 vs 方差）。
- LDA 在工程上最不同于分类算法的地方：输出是 `transform()` 而非 `predict()`、单一降维散点图而非多类评估图——这是由降维定位决定的。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 LDA 实现。
2. 给出继续深入阅读线性判别分析与相关数据工具的可靠入口。

## 自检题

1. 为什么 LDA 的 `fit()` 必须接收 `y` 参数，而 PCA 不需要？`y` 在散度矩阵 $\mathbf{S}_W$ 和 $\mathbf{S}_B$ 的构造中分别扮演什么角色？
2. 为什么 Wine 数据（$K=3$）的 LDA 最多只能降到 2 维？这个 $K-1$ 上限从 $\mathbf{S}_B$ 的秩如何推导出来？
3. `solver='svd'` 与 `solver='eigen'` 和 `solver='lsqr'` 有什么关键差异？为什么当前源码用 `hasattr` 检查 `explained_variance_ratio_` 是否存在？
4. 为什么 LDA 的 `explained_variance_ratio_` 在 Wine 数据上，当 `n_components=2` 时累计值必然为 100%？这与 PCA 的同名属性在含义上有何本质区别？
5. 为什么 LDA 2D 投影图上的类别分离效果通常优于 PCA 2D 投影图？两者的优化目标分别是什么？
6. 为什么标准化对 LDA 是硬性要求？Wine 数据中哪个特征的量纲范围最极端？不标准化会有什么后果？
7. LDA 既可以降维也可以分类——当前仓库将它定位为哪种？`transform()` 和 `predict()` 分别做什么？

## 练习方向

### 1. 改变 `solver`

- 把 `solver='svd'` 改成 `'eigen'` 和 `'lsqr'`
- 观察变化：
  - `explained_variance_ratio_`——在 `'lsqr'` 下是否消失？`'eigen'` 下的值与 `'svd'` 是否一致？
  - 2D 判别图——不同求解器下散点分布是否相同？（应基本一致，因优化目标相同）
  - 训练耗时——三种求解器的速度差异
- 核心理解：不同求解器的求解路径不同，但目标相同——`explained_variance_ratio_` 的可用性是选择求解器时需考虑的关键工程因素

### 2. 改变 `n_components`

- 把 `n_components=2` 改成 `1`
- 观察变化：
  - 2D 图变为 1D 图——丢失了多少类别分离信息？
  - `explained_variance_ratio_`——第一个判别方向占了多大比例？
  - 尝试设 `n_components=3`——是否会报错？错误信息说明了什么？
- 核心理解：$K-1$ 是刚性约束——`n_components=2` 是 3 类数据下的最优配置

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察 2D 判别图的变化——类别分离效果是否显著变差或偏斜？
- 对比：原始数据中各特征的数值范围差异（`proline` ~278-1680 vs `nonflavanoid_phenols` ~0.13-0.66）
- 体会：标准化后各特征对散度矩阵贡献均等——判别方向才反映真实的类别可分性结构

### 4. 使用不同特征子集

- 在流水线中临时只保留部分特征列，例如只保留 3-4 个特征，再重新训练
- 观察变化：
  - 类别在 2D 判别空间中的分离效果是否明显变化
  - `explained_variance_ratio_` 的分布是否变化
- 体会：哪些化学特征对葡萄品种的区分更重要——这能帮助建立对 Wine 数据结构的直观认识

### 5. 对比 PCA 在相同 Wine 数据上的效果

- 用 PCA（`n_components=2`）在相同 Wine 数据上降维并生成 2D 散点图
- 对比变化：
  - 类别分离度——LDA 的类间分离通常显著优于 PCA
  - 坐标轴含义——`LD1/LD2`（判别方向）vs `PC1/PC2`（主成分方向）
  - `explained_variance_ratio_` 含义——"判别能力占比" vs "方差占比"
  - 累计解释方差——LDA 在 `n_components=2` 时必为 100%，PCA 通常 < 100%
- 核心理解：LDA 和 PCA 不是"谁更好"——同一份数据在两种优化目标下给出不同的低维表示。分类预处理选 LDA，数据探索选 PCA

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`LinearDiscriminantAnalysis` | 完整构造器参数（`n_components`、`solver`、`priors`、`shrinkage`、`tol`、`covariance_estimator`）、属性（`explained_variance_ratio_`、`scalings_`、`means_`、`priors_`、`classes_`、`xbar_`）与方法（`fit`、`transform`、`predict`、`predict_proba`、`predict_log_proba`、`decision_function`）说明 |
| 2 | scikit-learn 官方文档：`load_wine` | Wine 数据集的加载方式、特征说明和类别信息——`as_frame` 参数、`return_X_y` 参数 |
| 3 | scikit-learn 用户指南：LDA and QDA | 线性判别分析和二次判别分析的原理、`solver` 选择指南、收缩（shrinkage）正则化、与 PCA 的使用场景对比 |
| 4 | Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems*. Annals of Eugenics. | LDA 原始论文——Fisher 判别准则、类间/类内散度、Iris 数据上的经典应用——奠定线性判别分析数学基础的里程碑工作 |

- scikit-learn `LinearDiscriminantAnalysis`：https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- scikit-learn `load_wine`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
- scikit-learn 用户指南 LDA and QDA：https://scikit-learn.org/stable/modules/lda_qda.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 LDA 分册的核心内容：
  - LDA 是有监督降维——`fit()` 必须接收 `y`，标签用于定义类间/类内散度结构
  - Fisher 判别准则最大化类间散度与类内散度之比——这是 LDA 与 PCA（最大化方差）最根本的数学差异
  - $K-1$ 维上限来自 $\text{rank}(\mathbf{S}_B) \leq K-1$——Wine 数据 $K=3$，最多 2 个判别方向
  - `explained_variance_ratio_` 在 LDA 中表示"判别能力占比"，在 PCA 中表示"方差占比"——同名异义
  - `solver='svd'` 是默认选择——数值稳定性好，且支持 `explained_variance_ratio_`
  - 标准化对基于散度矩阵的 LDA 是硬性要求——Wine 数据特征量纲差异达数千倍
  - LDA 既有 `transform()`（降维投影）也有 `predict()`（分类预测）——当前仓库定位为降维工具
  - LDA 与 PCA 不是"谁更好"的关系——分类场景选 LDA，数据压缩/探索选 PCA
