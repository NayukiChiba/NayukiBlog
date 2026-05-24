---
title: DecisionTreeRegression
date: 2026-02-21
category: MachineLearning/Regression
tags:
  - Python
  - 基础
description: 决策树回归的数学原理、CART递归分裂与完整工程实现。
image: https://img.yumeko.site/file/blog/DecisionTreeRegression.png
status: published
---

# 数学原理

## 本章目标

1. 理解决策树回归的数学本质——递归二分特征空间，每次选择使平方误差最小的特征和阈值。
2. 理解叶子节点为什么输出局部常数（区域均值）——这是平方误差最小化的自然结果。
3. 把这些数学表达和当前源码中的 `max_depth`、`min_samples_split`、`min_samples_leaf` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 区域划分 | 数学过程 | 将特征空间递归二分为若干矩形子区域 $R_1, R_2, \dots, R_M$ |
| 平方误差最小化 | 分割准则 | $\min_{j,s} \left[\sum_{\mathbf{x}_i \in R_1} (y_i - \hat{c}_1)^2 + \sum_{\mathbf{x}_i \in R_2} (y_i - \hat{c}_2)^2\right]$——选择最优特征 $j$ 和阈值 $s$ |
| 局部常数预测 | 叶子输出 | $\hat{c}_m = \frac{1}{|R_m|} \sum_{\mathbf{x}_i \in R_m} y_i$——区域 $R_m$ 内所有样本目标值的均值 |
| 复杂度控制 | 正则化 | `max_depth`、`min_samples_split`、`min_samples_leaf`——防止树过深、区域过细 |

## 1. 区域划分的数学形式

决策树回归将特征空间划分为 $M$ 个互不相交的矩形区域 $R_1, R_2, \dots, R_M$。预测函数为：

$$
f(\mathbf{x}) = \sum_{m=1}^{M} \hat{c}_m \cdot \mathbb{1}(\mathbf{x} \in R_m)
$$

其中 $\hat{c}_m$ 是区域 $R_m$ 的预测值（常数），$\mathbb{1}(\cdot)$ 是指示函数。

对于特征 $j$ 和分割点 $s$，定义左右子区域：

$$
R_1(j, s) = \{\mathbf{x} \mid x_j \le s\}, \quad R_2(j, s) = \{\mathbf{x} \mid x_j > s\}
$$

划分过程是**递归**的——对 $R_1$ 和 $R_2$ 各自继续寻找最优 $(j, s)$ 分裂，直到满足停止条件。

### 理解重点

- 决策树的每一步分裂都是**轴对齐**（axis-aligned）的——每次只用一个特征的一个阈值切一刀。
- 这意味着决策边界总是由垂直于坐标轴的超平面组成——不能直接产生斜线分割。
- 数学上的区域划分，在工程上对应树的节点不断生成左右子节点——直到触及 `max_depth` 或 `min_samples_split` 等约束。

## 2. 分割准则：平方误差最小化

回归树的核心目标是最小化每个区域内的平方误差和。在第 $m$ 个节点，选择最优 $(j, s)$ 使得分裂后的总平方误差最小：

$$
\min_{j, s} \left[\sum_{\mathbf{x}_i \in R_1(j,s)} (y_i - \hat{c}_1)^2 + \sum_{\mathbf{x}_i \in R_2(j,s)} (y_i - \hat{c}_2)^2\right]
$$

其中：

$$
\hat{c}_1 = \frac{1}{|R_1|} \sum_{\mathbf{x}_i \in R_1} y_i, \quad \hat{c}_2 = \frac{1}{|R_2|} \sum_{\mathbf{x}_i \in R_2} y_i
$$

### 理解重点

- 回归树的分裂准则与分类树有本质区别——分类树用基尼系数或信息增益（衡量类别纯度），回归树用平方误差（衡量数值离散度）。
- 每次分裂的目标是让左右两边的目标值各自更"集中"——也就是让区域内方差尽可能小。
- 这也解释了为什么叶子节点预测值自然取区域均值——对于平方损失，均值是最优的常数预测。

## 3. 叶子节点预测值：局部常数

一旦样本落入某个叶子节点（区域 $R_m$），预测值固定为该区域内所有训练样本目标值的均值：

$$
\hat{c}_m = \frac{1}{|R_m|} \sum_{\mathbf{x}_i \in R_m} y_i
$$

整个模型的预测函数因此呈现**分段常数**形态：

$$
f(\mathbf{x}) = \hat{c}_{m(\mathbf{x})}, \quad m(\mathbf{x}) = \text{包含 } \mathbf{x} \text{ 的叶子区域}
$$

### 理解重点

- 决策树回归**不是**在拟合一条连续曲线——它在每个叶子区域输出一个常数，整体预测函数是阶梯状的。
- 与线性回归的对比：线性回归对任意 $\mathbf{x}$ 输出 $\boldsymbol{\beta}^T \mathbf{x}$（全局连续函数），决策树回归输出不同区域的局部均值（分段常数）。
- 树越深、叶子越多，分段越细——模型越灵活但也越容易过拟合。

## 4. 训练复杂度与搜索策略

对每个特征 $j$，将其取值排序后遍历可能的分割点，整体搜索成本为：

$$
O(d \cdot N \log N)
$$

其中 $d$ 是特征数（当前 8），$N$ 是样本数（训练集约 16512）。

每层分裂后样本被二分，递归深度受 `max_depth` 限制，总复杂度约为 $O(d \cdot N \log N \cdot \text{depth})$。

### 理解重点

- 回归树的训练不是遍历所有可能的分裂组合——而是对每个特征单独排序、单独搜索最优阈值，再选全局最优。
- California Housing 有 20640 样本 × 8 特征——在此规模上决策树训练极快（秒级），这也是树模型的一大工程优势。
- 当前源码中同时使用 `@timeit` 和 `timer` 打印耗时——对于这个数据规模，训练耗时通常在毫秒到秒级。

## 5. 复杂度控制与正则化

不加约束的决策树会一直分裂到每个叶子只有 1 个样本——完美拟合训练数据但毫无泛化能力。当前源码通过三个超参数约束树的生长：

| 约束项 | 数学含义 | 当前取值 |
|---|---|---|
| `max_depth` | 树的最大深度——限制从根到叶子的最长路径长度 | `6` |
| `min_samples_split` | 节点继续分裂所需的最小样本数——若当前节点样本数小于此值，停止分裂 | `6` |
| `min_samples_leaf` | 叶子节点允许的最少样本数——分裂后任一子节点样本数少于此值则拒绝该分裂 | `3` |

此外，理论上还有代价复杂度剪枝（cost-complexity pruning）：

$$
C_\alpha(T) = \sum_{m=1}^{|T|} N_m \cdot \text{MSE}_m + \alpha |T|
$$

其中 $|T|$ 是叶子节点数，$\alpha$ 是复杂度惩罚系数。但**当前源码未使用 `ccp_alpha`**——复杂度控制完全通过上述三个超参数。

### 理解重点

- 三个超参数从不同角度阻止树的过度生长：`max_depth` 限制深度上限，`min_samples_split` 限制何时还能切，`min_samples_leaf` 限制叶子不能太小。
- 当前默认值 `(6, 6, 3)` 是中等保守的配置——在 California Housing 上既有足够的非线性拟合能力，又不过度分裂。
- 理论上常见的 `ccp_alpha` 剪枝在 scikit-learn 中可用但当前未启用——文档必须区分"理论上常见的控制方式"和"当前实现实际使用了什么"。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 区域划分 | $R_1(j,s), R_2(j,s)$ | `DecisionTreeRegressor` 内部节点分裂——`max_depth=6` 限制层数 |
| 分割准则 | $\min_{j,s} \sum (y_i - \hat{c})^2$ | `criterion="squared_error"`（默认，源码未显式写出） |
| 叶子预测值 | $\hat{c}_m = \text{mean}(y_i : \mathbf{x}_i \in R_m)$ | `model.predict(X)` 返回各叶子区域均值 |
| 特征搜索 | 对每个 $j$ 排序后遍历 $s$ | scikit-learn 内部 Cython 实现——`random_state=42` 保证可复现 |
| 最大深度 | $\text{depth}_{\max}$ | `max_depth=6` |
| 最小分裂样本数 | $N_{\min}^{\text{split}}$ | `min_samples_split=6` |
| 最小叶子样本数 | $N_{\min}^{\text{leaf}}$ | `min_samples_leaf=3` |
| 叶节点数 | $|T|$ | `model.get_n_leaves()` |
| 树深度 | $\text{depth}(T)$ | `model.get_depth()` |
| 特征重要性 | $\text{imp}_j$ | `model.feature_importances_` |

## 7. 决策树回归 vs 线性回归 数学对比

| 维度 | 线性回归 | 决策树回归 |
|---|---|---|
| 模型形式 | $f(\mathbf{x}) = \boldsymbol{\beta}^T \mathbf{x} + \beta_0$——全局线性函数 | **$f(\mathbf{x}) = \sum_m \hat{c}_m \mathbb{1}(\mathbf{x} \in R_m)$——分段常数** |
| 目标函数 | $\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$——闭式解或梯度下降 | **$\min_{j,s} \sum_{R_1,R_2} (y_i - \hat{c})^2$——贪心搜索** |
| 假设 | 全局线性关系 | **无条件分布假设** |
| 特征交互 | 需手工构造交互项 | **自然通过条件分支捕获** |
| 非线性处理 | 需基函数展开或特征工程 | **天然支持——分裂即是非线性** |
| 参数数 | $d + 1$（系数 + 截距） | **随树深度指数增长——叶子数** |
| 过拟合风险 | 低（参数少） | **高（需显式约束深度和叶子大小）** |
| 对特征尺度敏感 | 是——需标准化 | **否——仅依赖相对排序** |

## 常见坑

1. 把回归树的分裂准则与分类树混淆——回归用平方误差（MSE），分类用基尼系数或熵。
2. 以为树越深越好——未约束的树会在训练集上完美拟合但泛化极差。
3. 把叶子预测值理解为"该区域的线性拟合"——决策树回归输出的是常数（均值），不是局部线性函数。
4. 忽略 `min_samples_split` 和 `min_samples_leaf` 的联合作用——只看 `max_depth` 不足以控制复杂度。

## 小结

- 决策树回归的数学核心链：递归二分特征空间 → 平方误差最小化选择 $(j, s)$ → 叶子输出区域均值 $\hat{c}_m$ → `max_depth`/`min_samples_split`/`min_samples_leaf` 控制复杂度。
- 与线性回归的根本区别：分段常数 vs 全局线性，无条件假设 vs 线性假设，天然处理非线性 vs 需特征工程。
- 当前源码 `DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3)` 是展示回归树核心数学最经典的中等复杂度配置。

# 数据构成

## 本章目标

1. 明确本仓库决策树回归数据来自 `fetch_california_housing` 真实数据集——非手动合成。
2. 理解 8 个特征与标签 `price` 的角色，以及为何不需要标准化。
3. 明确训练/测试集切分方式（`randomSplit`，`test_size=0.2`）及其与 SVR/线性回归的预处理差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory.loadDecisionTreeRegressionDataset()` | 方法 | 加载 California Housing 真实回归数据——标签列重命名为 `price` |
| `fetch_california_housing(as_frame=True)` | 函数 | scikit-learn 提供的加州房价数据集加载器——20640 样本 × 8 特征 |
| `price` | 列 | 回归目标列——加州街区房价中位数（单位：$100k） |
| `train_test_split` | 函数 | 随机切分训练/测试集——`test_size=0.2, random_state=42` |

## 1. 数据生成：`RegressionDatasetFactory.loadDecisionTreeRegressionDataset()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `as_frame` | `bool` | `True`——直接返回带列名的 `DataFrame` | `True` |
| 返回值 | `DataFrame` | 含 8 个特征列 + `price` 标签列的完整数据表，形状 `(20640, 9)` | — |

### 示例代码

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={"MedHouseVal": "price"})
# df.shape = (20640, 9)
```

### 理解重点

- 当前分册使用的是**真实数据集**而非手动合成数据——这使回归树的非线性分裂能力有实际意义，而非仅展示数学性质。
- 标签列 `MedHouseVal` 在源码中被重命名为 `price`——与其他回归分册（线性回归、SVR）保持标签列名统一。
- 20640 个样本对于树模型来说非常充裕——足够展示不同复杂度约束下的过拟合/欠拟合行为。
- 因为是真实数据，数据中的非线性关系和特征交互更复杂——这正是决策树回归相对于线性回归的优势场景。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 特征列 | `DataFrame`，形状 `(20640, 8)` | 8 个连续特征——涵盖收入、房屋属性、人口和地理位置 | `data.drop(columns=["price"])` |
| `price` | `Series`，形状 `(20640,)` | 目标变量——加州街区房价中位数（$100k），范围约 $[0.15, 5.0]$ | `data["price"]` |

California Housing 的 8 个特征：

| 特征名 | 含义 | 单位 |
|---|---|---|
| `MedInc` | 街区收入中位数 | $10k |
| `HouseAge` | 房屋年龄中位数 | 年 |
| `AveRooms` | 平均房间数 | 间 |
| `AveBedrms` | 平均卧室数 | 间 |
| `Population` | 街区人口 | 人 |
| `AveOccup` | 平均居住人数 | 人/户 |
| `Latitude` | 纬度 | 度 |
| `Longitude` | 经度 | 度 |

### 理解重点

- `Latitude` 和 `Longitude` 是地理位置特征——树模型可以自然切出"北加州 vs 南加州"或"沿海 vs 内陆"这样的空间模式。
- `MedInc`（收入中位数）通常是最重要的分裂特征——房价与收入高度相关，树会优先在收入维度切分。
- 8 个特征覆盖了经济（MedInc）、物理（HouseAge/AveRooms）、人口（Population/AveOccup）和地理（Latitude/Longitude）四个维度——特征类型多样化，适合展示决策树的多维分裂行为。

## 3. 数据切分

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比。`0.2`——20640 × 0.2 ≈ 4128 个测试样本 | `0.2` |
| `random_state` | `int` | 随机种子。`42`——保证可复现划分 | `42` |
| 返回值 | `tuple` | `(X_train, X_test, y_train, y_test)`——训练集约 16512 样本，测试集约 4128 样本 | — |

### 示例代码

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 当前使用随机切分（`randomSplit`）而非分层切分——回归目标 `price` 是连续值，无法分层。
- 与线性回归/SVR 的切分方式完全一致——使用相同的 `test_size=0.2` 和 `random_state=42`。
- 切分后训练集约 16512 样本——对于决策树来说非常充足，学习曲线可以看出随样本量增加得分趋于稳定。

## 4. 为什么不需要标准化

### 参数速览

| 项目 | 当前状态 | 原因 |
|---|---|---|
| 标准化 | **未使用** | 树模型基于特征阈值的相对排序分裂——不依赖距离或内积 |
| 缺失值处理 | **未使用** | California Housing 无缺失值 |

### 理解重点

- 决策树的分裂准则是"$x_j \le s$"——只关心特征值的相对排序，不关心绝对尺度。将特征统一放大 10 倍，分裂阈值也放大 10 倍，树结构完全不变。
- 这与线性回归和 SVR 形成鲜明对比——线性回归的系数估计和梯度下降对特征尺度敏感，SVR 的 RBF 核依赖欧氏距离。
- 但标准化并非对树模型"毫无影响"——如果特征间数量级差异极大（如一个特征范围 0.001-0.01，另一个 0-100000），在与其他模型对比时标准化可以统一预处理管道。当前分册因只涉及树模型，不标准化是正确选择。

## 5. 数据设计意图：与线性回归/SVR 的对比

| 数据维度 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 数据来源 | 手动合成——`面积`/`房间数`/`房龄` → `price` | `make_friedman1`——10 特征非线性 | **`fetch_california_housing`——真实加州房价** |
| 样本数 | 200 | 200 | **20640** |
| 特征维度 | 3 | 10 | **8** |
| 标签类型 | 连续（手动公式 + 噪声） | 连续（Friedman 函数 + 噪声） | **连续（真实房价中位数）** |
| 标准化 | 有（`StandardScaler`） | 有（`StandardScaler`） | **无** |
| 数据拆分 | 随机切分 | 随机切分 | 随机切分 |
| 设计意图 | 展示线性关系 + 系数解释 | 展示核方法非线性拟合 | **展示真实数据非线性 + 特征交互 + 复杂度控制** |

### 理解重点

- 线性回归用公式 `price = 2*面积 + 10*房间数 - 3*房龄 + noise` 手动合成——数据中的关系是精确线性的，适合展示系数解释。
- SVR 用 `make_friedman1`——非线性函数 + 噪声，适合展示核方法的非线性拟合能力。
- 决策树回归用真实 California Housing——数据中的关系未知且复杂，既有非线性又有特征交互，正是树模型的天然优势场景。
- 三种数据设计形成递进：手动线性 → 合成非线性 → 真实复杂——覆盖了回归问题的典型数据形态。

## 数据可视化

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/decision_tree_regression/data_correlation.png)

![特征与目标变量关系](https://img.yumeko.site/file/articles/ML/decision_tree_regression/data_feature_vs_price.png)

## 常见坑

1. 看到回归任务就默认写入标准化步骤——当前决策树回归源码**没有**标准化，且这是正确的设计决策。
2. 期待在数据中找到精确的线性公式——California Housing 是真实数据，关系复杂且含噪声，不存在简洁的生成公式。
3. 忽略 `random_state=42` 的作用——它保证了每次运行的数据切分完全一致，是实验可复现的基础。

## 小结

- 当前决策树回归数据来自 `fetch_california_housing(as_frame=True)`——20640 样本 × 8 特征的真实加州房价数据集，标签列为 `price`。
- 数据流为：加载 → 列重命名 → 随机切分（`test_size=0.2`）→ 直接训练（无标准化）。
- 不标准化的设计意图明确——树模型仅依赖特征阈值的相对排序，与线性回归/SVR 的预处理形成有意义的工程对比。

# 思路与直觉

## 本章目标

1. 理解决策树回归的核心直觉——不是拟合一条全局曲线，而是不断提问"哪个特征超过哪个阈值"，把样本分到越来越纯的区域。
2. 理解为什么 California Housing 真实数据特别适合展示树模型的非线性与特征交互能力。
3. 建立"区域划分 → 局部常数 → 复杂度约束"的直觉链，并与线性回归形成对比。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 递归分裂 | 核心机制 | 每次选一个特征和一个阈值，把当前区域一分为二——不断重复直到满足停止条件 |
| 局部常数预测 | 预测方式 | 落入同一叶子的样本共享同一个预测值（该区域训练样本目标值的均值） |
| `max_depth` | 复杂度约束 | 限制树最多分裂多少层——防止无限生长 |
| `feature_importances_` | 模型解释 | 量化各特征在分裂过程中对误差降低的总贡献 |

## 1. 决策树回归想做什么

决策树回归的核心目标不是学一条全局公式，而是不断提问"$x_j \le s$？"，把特征空间切成越来越小的矩形区域，每个区域用一个常数值（区域内目标均值）作为预测。

### 理解重点

- 线性回归问"所有特征一起贡献多少"——输出是 $\sum \beta_j x_j$ 的加权和。
- 决策树回归问"先看收入高不高，再看位置在哪"——输出是一系列 if-else 规则引导下的局部均值。
- 这种"问答式"结构使决策树天然适合处理"不同区域表现不同"的数据模式。

## 2. 为什么 California Housing 特别适合讲决策树回归

California Housing 数据中的房价由多种异质因素共同决定：收入水平、地理位置、房屋年龄、人口密度等。这些因素与房价的关系通常不是简单线性的——例如"沿海高收入区"和"内陆农业区"的房价形成机制完全不同。

### 理解重点

- 如果房价与所有特征的关系都是线性的，线性回归的系数解释更清晰——不需要树模型。
- 但真实房价数据中，"位置 + 收入"的交互、"老房子在好地段仍贵"等模式很难用单一线性公式表达。
- 决策树可以自然地先按 `Latitude` 切出"南加州 vs 北加州"，再按 `MedInc` 细分"高收入区 vs 低收入区"——这正是树模型直觉的核心。

## 3. 如何理解"局部常数预测"

决策树回归的预测函数是一条分段常数函数——不像线性回归那样输出连续变化的数值。

| 对比项 | 线性回归 | 决策树回归 |
|---|---|---|
| 预测形态 | 连续超平面——输入微小变化，输出也微小变化 | **阶梯状——同一叶子内输入变化不影响输出** |
| 输入-输出关系 | 全局统一公式 | **局部独立规则** |
| 极端值外推 | 线性外推 | **叶子边界外无外推能力** |

### 理解重点

- 树的预测值只在分裂边界处跳变——同一条"if-else"路径走到同一个叶子的样本，输出相同。
- 这是优点的代价：树能拟合复杂非线性，但也失去了线性模型的外推能力和平滑性。
- 叶子节点数越多，分段越细，预测函数越接近连续——但过拟合风险也越大。

## 4. 为什么树模型适合处理非线性和特征交互

树模型通过条件分支自然表达"如果 A 且 B，则..."的交互逻辑：

1. 第一层按 `MedInc` 切——"高收入区"和"低收入区"分开。
2. 第二层对"高收入区"按 `Latitude` 再切——"沿海高收入"和"内陆高收入"分开。
3. 第三层对"沿海高收入"按 `HouseAge` 再切——"新房贵区"和"老房区"分开。

这种"先按一个特征切，再按另一个特征细分"的模式，本质上就是在自动发现特征交互。

### 理解重点

- 在线性回归中，`MedInc × Latitude` 的交互需要**手工构造**交互项。
- 在决策树中，"先切 MedInc，再切 Latitude"自动实现了这一交互——不需要特征工程。
- 树的深度越大，能表达的交互阶数越高——深度为 3 的树最多能表达三阶交互。

## 5. 复杂度约束的直觉：为什么不放任树自由生长

不设约束的决策树会一直分裂到每个叶子只有一个样本——在训练集上误差为零，但预测新样本时表现极差。

| 约束 | 直觉 | 过松的症状 |
|---|---|---|
| `max_depth=6` | "最多问 6 个问题" | 树太深——对训练集中偶然模式过度敏感 |
| `min_samples_split=6` | "节点里至少 6 个样本才继续问" | 叶子太细——单个异常样本就能长出一个分支 |
| `min_samples_leaf=3` | "每个答案至少覆盖 3 个样本" | 叶子不可靠——基于极少量样本的均值波动大 |

### 理解重点

- 三个约束从不同角度"早停"树的生长——就像考试时限制答题步骤数，防止钻牛角尖。
- 当前默认值 `(6, 6, 3)` 是中等保守配置——在 California Housing 上既有足够的灵活性又不会过拟合。
- 这些约束是回归树"偏差-方差权衡"的工程杠杆——约束越紧，偏差越大但方差越小。

## 6. 直觉如何映射到训练日志与可视化

### 理解重点

- 训练日志中的**树深度**和**叶子节点数**是结构复杂度的直接反映——深度 6 意味着最多 $2^6 = 64$ 个叶子（实际通常少于上限）。
- **特征重要性图**告诉你"模型主要看什么"——在 California Housing 上，`MedInc` 和 `Latitude`/`Longitude` 通常排在最前。
- **残差图**告诉你"哪里没拟合好"——如果残差在地理位置上有系统偏移，说明树的空间切分可能不够细。
- **学习曲线**告诉你"样本量够不够"——如果训练得分远高于验证得分，说明过拟合；两者都低则欠拟合。

## 可视化

![决策树结构](https://img.yumeko.site/file/articles/ML/decision_tree_regression/tree_structure.png)

## 常见坑

1. 把决策树回归误解成"自动拟合一条更复杂的曲线"——它实际上是在做分段常数近似，而非连续函数拟合。
2. 只看特征重要性，不看树深度和叶子节点数——错过模型复杂度的关键信息。
3. 误以为树模型一定比线性模型好——树在非线性数据上有优势，但在近似线性的数据上可能不如线性回归稳定。

## 小结

- 决策树回归的核心直觉：递归二分空间 → 每个区域用均值预测 → 复杂度约束防止过拟合。
- California Housing 真实数据天然适合展示树的非线性与特征交互能力——地理位置切分 + 收入细分正是树模型的直觉优势。
- 建立"区域划分 → 局部常数 → 复杂度约束 → 残差/重要性/学习曲线"的完整直觉链后，再看模型构建与评估会水到渠成。

# 模型构建

## 本章目标

1. 明确 `trainDecisionTreeRegressionModel(...)` 如何构建并训练 `DecisionTreeRegressor`。
2. 理解三个复杂度超参数（`max_depth`、`min_samples_split`、`min_samples_leaf`）的默认值与作用。
3. 看清训练完成后最重要的模型属性——`feature_importances_`、`get_depth()`、`get_n_leaves()`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练一个 `sklearn.tree.DecisionTreeRegressor` 模型 |
| `DecisionTreeRegressor(...)` | 类 | scikit-learn 提供的决策树回归器——CART 算法 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上递归生长决策树——每次选最优 $(j, s)$ 分裂 |
| `model.get_depth()` | 方法 | 返回当前树的最大深度——复杂度诊断 |
| `model.get_n_leaves()` | 方法 | 返回当前树的叶子节点总数——分段数量诊断 |
| `model.feature_importances_` | 属性 | 各特征在分裂中对误差降低的累积贡献 |

## 1. `trainDecisionTreeRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainDecisionTreeRegressionModel(XTrain, yTrain, randomState=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(16512, 8)` | 训练特征矩阵——California Housing 的 8 个特征 | `X_train` |
| `yTrain` | `ndarray`，形状 `(16512,)` | 训练目标值——房价中位数 | `y_train` |
| `randomState` | `int` | 随机种子。`42`——保证分裂过程中的随机性可复现 | `42` |
| 返回值 | `DecisionTreeRegressor` | 已完成 `fit()` 的回归树模型 | — |

### 示例代码

```python
from src.mlAlgorithms.training.regression.regressionModels import (
    trainDecisionTreeRegressionModel,
)

model = trainDecisionTreeRegressionModel(X_train, y_train)
```

### 理解重点

- 当前训练入口返回的是**单个**决策树模型——不是集成（如随机森林、GBDT），更强调单棵树的结构可解释性。
- 函数内部不切分数据——训练/测试切分在流水线层完成，训练层只负责接收训练数据并拟合。
- `randomState=42` 保证分裂过程中的随机性可复现——虽然 CART 的 $(j, s)$ 搜索是确定性的，但某些内部实现细节涉及随机性。

## 2. `DecisionTreeRegressor` 的构造器参数

### 参数速览

适用 API：`DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_depth` | `int` | 树的最大深度。`6`——限制从根到叶子的最长路径为 6 层 | `3`、`6`、`10`、`None` |
| `min_samples_split` | `int` | 节点继续分裂所需的最小样本数。`6`——节点样本数 < 6 时停止分裂 | `2`、`6`、`20` |
| `min_samples_leaf` | `int` | 叶子节点最少样本数。`3`——分裂后任一子节点样本数 < 3 则拒绝该分裂 | `1`、`3`、`10` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `criterion` | `str` | 分裂准则。默认 `"squared_error"`——当前源码未显式写出，使用默认值 | `"squared_error"`、`"friedman_mse"`、`"absolute_error"` |

### 示例代码

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train)
```

### 理解重点

- 当前源码没有显式设置 `criterion`——使用 scikit-learn 默认的 `"squared_error"`（平方误差）。这是回归树最经典的分裂准则。
- `max_depth=6` 是中等深度——在 8 维特征空间中最深可产生 $2^6 = 64$ 个叶子（实际通常远少于此，因为 `min_samples_split` 和 `min_samples_leaf` 会提前停止）。
- 与分类树的关键区别：回归树没有 `class_weight`、`criterion="gini"` 等分类特有参数。

## 3. 三个复杂度超参数的联合作用

### 理解重点

- `max_depth` 是最直观的上限约束——无论样本多充足，根到叶子的路径不超过 6 层。
- `min_samples_split` 是"是否继续切"的门槛——节点样本数太小说明已经切得够细，继续切意义不大。
- `min_samples_leaf` 是"切了之后叶子是否可靠"的门槛——如果切完后某边只剩 1-2 个样本，这个分裂的统计意义存疑。
- 三者联合作用：即使 `max_depth` 还有余量，如果样本数不满足 `min_samples_split` 或 `min_samples_leaf`，树也会提前停止。

## 4. 训练完成后的关键属性

### 参数速览

| 属性/方法 | 类型 | 含义 | 示例取值 |
|---|---|---|---|
| `get_depth()` | `int` | 实际树深度——≤ `max_depth` | `6` |
| `get_n_leaves()` | `int` | 叶子节点总数——即分段预测函数的常数段数 | `20`~`50` |
| `feature_importances_` | `ndarray`，形状 `(8,)` | 各特征在分裂中对平方误差降低的累积贡献，和为 1 | `[0.45, 0.05, 0.03, ...]` |
| `tree_` | 内部对象 | Cython 实现的树结构——含节点信息、阈值、左右子节点索引等 | — |

### 示例代码

```python
print(f"树深度: {model.get_depth()}")
print(f"叶子节点数: {model.get_n_leaves()}")
print(f"特征重要性: {model.feature_importances_}")
```

### 理解重点

- `get_depth()` 和 `get_n_leaves()` 是训练后最先关注的结构性指标——深度表示模型复杂度，叶子数表示分段数。
- `feature_importances_` 反映特征在分裂中的相对贡献——值越大，该特征在树中被用于分裂的次数越多、每次分裂的误差降低越大。
- 与线性回归的 `coef_` 有本质区别：`feature_importances_` 衡量的是"分裂贡献"而非"线性效应大小和方向"。

## 5. 决策树回归 vs 线性回归 vs SVR 模型参数对比

| 参数/属性 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 核心超参数 | 无（仅 `fit_intercept`） | `C`、`epsilon`、`kernel`、`gamma` | **`max_depth`、`min_samples_split`、`min_samples_leaf`** |
| 训练输入 | `fit(X, y)` | `fit(X, y)` | `fit(X, y)` |
| 模型属性 | `coef_`、`intercept_` | `support_vectors_`、`dual_coef_` | **`feature_importances_`、`get_depth()`、`get_n_leaves()`** |
| 预测输出 | 连续值 | 连续值 | 连续值（分段常数） |
| 标准化 | 有 | 有 | **无** |
| 依赖 | sklearn 内置 | sklearn 内置 | sklearn 内置 |

## 常见坑

1. 只看 `max_depth` 不看 `min_samples_split` 和 `min_samples_leaf`——实际树结构是由三者联合决定的。
2. 把 `feature_importances_` 解读为"特征对目标的正负影响"——重要性只衡量分裂贡献，不表示影响方向。
3. 在极深树（`max_depth=None`）上期待稳定的特征重要性——树过深时重要性可能分散到多个相关特征。

## 小结

- `trainDecisionTreeRegressionModel(...)` 是本仓库决策树回归的核心训练入口——对 `DecisionTreeRegressor` 的薄封装，传递三个复杂度超参数。
- `DecisionTreeRegressor(max_depth=6, min_samples_split=6, min_samples_leaf=3)` 是当前默认配置——中等保守，在 California Housing 上平衡了灵活性与稳定性。
- 训练完成后的核心属性：`get_depth()`（复杂度）、`get_n_leaves()`（分段数）、`feature_importances_`（特征贡献）——三者构成回归树的结构诊断三件套。

# 训练与预测

## 本章目标

1. 理解决策树回归流水线的完整执行顺序——从数据加载到四类可视化输出。
2. 理解决策树的训练过程——递归寻找最优 $(j, s)$ 分裂直到满足停止条件。
3. 理解 `predict` 的预测方式——输入沿树从根走到叶，输出该叶子的局部常数。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练决策树回归模型——递归二分特征空间 |
| `model.fit(X_train, y_train)` | 方法 | CART 算法——每次遍历所有特征的候选分裂点，选平方误差最小的 $(j, s)$ |
| `model.predict(X_test)` | 方法 | 对新样本沿树从根走到叶——返回叶子节点训练集目标均值 |
| `plot_residuals(...)` | 函数 | 绘制预测-真实散点图 + 残差分布图 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证 R² 随样本量变化的曲线 |
| `plot_tree_structure(...)` | 函数 | 绘制决策树的可视化结构图 |

## 1. 完整流水线流程

### 流程概述

```
fetch_california_housing(as_frame=True)
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ model = trainDecisionTreeRegressionModel(X_train, y_train)
    ├─ ④ y_pred = model.predict(X_test)
    ├─ ⑤ plot_residuals(y_test, y_pred)
    ├─ ⑥ plot_feature_importance(model, feature_names)
    ├─ ⑦ plot_learning_curve(DecisionTreeRegressor(...), X_train, y_train, scoring="r2")
    └─ ⑧ plot_tree_structure(model, feature_names)
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `fetch_california_housing` | — | `DataFrame`，`(20640, 9)` | 真实加州房价数据 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X` `(20640, 8)`、`y` `(20640,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`、`y` | `X_train`、`X_test`、`y_train`、`y_test` | `test_size=0.2`，无标准化 |
| 训练 | `trainDecisionTreeRegressionModel` | `X_train`、`y_train` | `DecisionTreeRegressor` | CART 递归分裂 |
| 预测 | `model.predict` | `X_test` | `y_pred` `(4128,)` | 叶子局部常数 |
| 残差图 | `plot_residuals` | `y_test`、`y_pred` | PNG 图像 | 误差分布诊断 |
| 特征重要性 | `plot_feature_importance` | `model`、`feature_names` | PNG 图像 | 特征贡献排名 |
| 学习曲线 | `plot_learning_curve` | 新 `DecisionTreeRegressor`、`X_train`、`y_train` | PNG 图像 | 样本量-得分趋势 |
| 树结构图 | `plot_tree_structure` | `model`、`feature_names` | PNG 图像 | 树的分裂结构可视化 |

### 理解重点

- 当前流水线**无标准化步骤**——决策树基于特征阈值的相对排序分裂，特征尺度不影响分裂选择。
- 学习曲线传入的是**新的** `DecisionTreeRegressor(...)` 实例，而非已训练的 `model`——因为学习曲线内部需要在不同训练子集上重新拟合。
- 树结构图是本流水线特有的可视化——大多数回归模型（线性回归、SVR）没有结构图。

## 2. 训练细节：CART 递归分裂

### 算法流程

```
从根节点开始（包含全部训练样本）
    ↓
对当前节点：
    ① 检查停止条件（深度 ≥ max_depth？样本数 < min_samples_split？）
       是 → 创建叶子节点，预测值 = 区域内样本 y 的均值
       否 → 继续
    ② 对每个特征 j：
        排序所有样本的 x_j 取值
        遍历候选分割点 s
        计算分裂后的平方误差降低：Δ = MSE(parent) - (n₁/N)·MSE(child₁) - (n₂/N)·MSE(child₂)
    ③ 选 Δ 最大的 (j*, s*)
    ④ 按 (j*, s*) 将样本分为左子节点和右子节点
    ⑤ 对左右子节点递归执行 ①-④
    ↓
达到停止条件 → 树生长完成
```

### 理解重点

- CART（Classification and Regression Tree）是二叉分裂——每个节点恰好产生两个子节点，不会同时分裂出多路。
- 平方误差降低 $\Delta$ 等价于"分裂后方差减少量"——CART 贪婪地选择每步方差降低最大的分裂。
- 停止条件是**预剪枝**（pre-pruning）——在生长过程中提前阻止，而非长成后再剪（post-pruning）。

## 3. 预测细节：从根走到叶

对测试样本 $\mathbf{x}$：

```
从根节点开始
    ↓
while 当前节点不是叶子:
    if xⱼ ≤ s:  去左子节点
    else:        去右子节点
    ↓
到达叶子 → 返回该叶子的预测值（训练时该区域样本 y 的均值）
```

### 理解重点

- 预测只需沿树走一条路径——复杂度为 $O(\text{depth})$，极快。
- 同一叶子内的所有测试样本得到完全相同的预测值——这就是"分段常数"在预测端的体现。
- 树模型天然支持缺失值处理（寻找替代分裂），但当前 California Housing 数据无缺失值，此能力未展示。

## 4. 与线性回归训练流程的对比

| 步骤 | 线性回归 | 决策树回归 |
|---|---|---|
| 数据 | 手动合成 $(200, 3)$ | **真实数据 $(20640, 8)$** |
| 标准化 | 有（`StandardScaler`） | **无** |
| 训练算法 | 闭式解（正规方程）或梯度下降 | **CART 贪心递归分裂** |
| 训练复杂度 | $O(d^3 + Nd^2)$（闭式解） | **$O(d \cdot N \log N)$** |
| 模型结构 | 系数向量 $\boldsymbol{\beta}$ | **二叉树——节点 + 阈值 + 叶子值** |
| 预测 | $\hat{y} = \mathbf{x}^T \boldsymbol{\beta}$ | **沿树走到叶子 → 返回叶子均值** |
| 预测复杂度 | $O(d)$ | **$O(\text{depth})$** |
| 评估可视化 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构图** |

## 常见坑

1. 在决策树流水线中引入标准化——树模型不需要，且标准化不会改变树结构（只会改变阈值的数值表示）。
2. 忘记额外保存 `feature_names`——`feature_importances_` 只返回数值数组，没有特征名。
3. 把已训练的 `model` 直接传给学习曲线——学习曲线需要在不同子集上重新训练，必须传未训练的模型实例。

## 小结

- 决策树回归流水线为 8 步：加载 → 拆分 → 切分 → 训练 → 预测 → 残差图 → 特征重要性 → 学习曲线 → 树结构图——无标准化。
- `fit()` 的核心流程：检查停止条件 → 遍历特征和阈值 → 选平方误差降低最大的 $(j, s)$ → 递归分裂左右子节点 → 直到触达约束。
- `predict()` 极为高效——沿树走一条路径（$O(\text{depth})$），到达叶子后输出该区域的训练均值。

# 评估与诊断

## 本章目标

1. 理解当前决策树回归流水线的四类评估输出——残差图、特征重要性图、学习曲线和树结构图。
2. 理解每类评估分别诊断什么问题，以及如何交叉解读。
3. 明确当前流水线**已实现**和**未实现**的评估项——回归树评估与分类树评估的本质差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 生成预测-真实散点图 + 残差分布图——诊断拟合质量和系统偏差 |
| `plot_feature_importance(...)` | 函数 | 生成特征重要性柱状图——揭示哪些特征主导分裂决策 |
| `plot_learning_curve(...)` | 函数 | 生成训练/验证 R² 曲线——诊断过拟合/欠拟合和样本量充足性 |
| `plot_tree_structure(...)` | 函数 | 绘制决策树可视化结构——展示完整的分裂路径和叶子值 |
| `model.feature_importances_` | 属性 | 各特征对平方误差降低的累积贡献——特征重要性图的底层数据 |

## 1. 残差图

### 参数速览

适用函数：`plot_residuals(y_true, y_pred, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `ndarray`，形状 `(4128,)` | 测试集真实房价 | `y_test` |
| `y_pred` | `ndarray`，形状 `(4128,)` | 模型预测房价 | `model.predict(X_test)` |
| `title` | `str` | 图标题 | `"决策树回归 残差分析"` |
| `dataset_name` | `str` | 输出目录名 | `"decision_tree_regression"` |
| `model_name` | `str` | 输出文件名前缀 | `"decision_tree"` |

### 示例代码

```python
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# 左图: 预测值 vs 真实值散点图（对角线 = 完美预测）
ax1.scatter(y_test, y_pred, alpha=0.6)

# 右图: 残差 vs 预测值散点图（红线 = 零残差）
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color="r", linestyle="--")
```

### 输出

![残差图](https://img.yumeko.site/file/articles/ML/decision_tree_regression/residual_plot.png)

### 理解重点

- 左图（预测 vs 真实）：点越贴近对角线 $y = \hat{y}$，预测越准确。对决策树回归，由于分段常数特性，预测值会呈现离散的水平条带。
- 右图（残差 vs 预测）：残差应围绕 0 随机分布。若残差随预测值增大而发散（漏斗形），说明模型对高价区预测不稳定。若残差整体偏正或偏负，说明模型存在系统性高估或低估。
- 决策树特有的诊断信号：残差图中出现明显的"阶梯"模式——同一叶子内的样本有相同的预测值但不同的真实值，残差在叶子内呈现垂直线段。

## 2. 特征重要性图

### 参数速览

适用函数：`plot_feature_importance(model, feature_names, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `DecisionTreeRegressor` | 已训练的回归树——提供 `feature_importances_` | `model` |
| `feature_names` | `list[str]` | 特征名列表——为重要性值提供可读标签 | `["MedInc", "HouseAge", ...]` |
| `top_n` | `int` 或 `None` | 展示前 N 个重要特征。`None`——展示全部 | `None`、`5` |
| `title` | `str` | 图标题 | `"决策树回归 特征重要性"` |

### 示例代码

```python
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
```

### 输出

![特征重要性](https://img.yumeko.site/file/articles/ML/decision_tree_regression/feature_importance.png)

### 理解重点

- 特征重要性反映的是**分裂贡献**——特征在树中每次分裂带来的平方误差降低的累积量。值越大，该特征在模型中越重要。
- 在 California Housing 上，`MedInc`（收入中位数）通常是重要性最高的特征——收入是房价最直接的预测因子。`Latitude` 和 `Longitude`（地理位置）通常排第二、第三。
- 重要性**不表示影响方向**——"MedInc 重要性 0.5"不意味着收入增加房价一定增加（虽然直觉上确实如此），只看重要性数值无法判断正负效应。
- 如果少数几个特征占据了绝大部分重要性（如 80%+），说明树主要依赖这些特征分裂——其他特征的分裂贡献很小。

## 3. 学习曲线

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, scoring, cv, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `DecisionTreeRegressor` | **未训练的**新模型实例——学习曲线内部会多次 fit | `DecisionTreeRegressor(max_depth=6, random_state=42)` |
| `X` | `ndarray`，形状 `(16512, 8)` | 训练集特征 | `X_train` |
| `y` | `ndarray`，形状 `(16512,)` | 训练集目标 | `y_train` |
| `scoring` | `str` | 评分指标。`"r2"`——R² 决定系数 | `"r2"`、`"neg_mean_squared_error"` |
| `cv` | `int` | 交叉验证折数。默认 `5` | `5` |
| `train_sizes` | `ndarray` | 训练样本比例序列。默认 10 个点从 0.1 到 1.0 | `np.linspace(0.1, 1.0, 10)` |

### 示例代码

```python
plot_learning_curve(
    DecisionTreeRegressor(max_depth=6, random_state=42),
    X_train,
    y_train,
    scoring="r2",
    title="决策树回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 输出

![学习曲线](https://img.yumeko.site/file/articles/ML/decision_tree_regression/learning_curve.png)

### 理解重点

- 学习曲线用不同规模的训练子集重复训练和验证——揭示模型性能随样本量增加的变化趋势。
- **过拟合信号**：训练得分远高于验证得分（两条曲线之间有明显间隙）——树过于复杂，学到了训练集中的噪声模式。
- **欠拟合信号**：训练得分和验证得分都很低且接近——树约束过紧，连训练数据中的基本模式都未充分学习。
- **理想状态**：两条曲线逐渐收敛到一个较高的得分——样本量充足，模型复杂度适中。
- 当前 `max_depth=6` 的配置在 California Housing 的 16512 训练样本上通常表现为轻微过拟合——训练 R² 约 0.7~0.8，验证 R² 约 0.6~0.7。

## 4. 树结构图

### 理解重点

- 树结构图将训练完成的决策树可视化——每个节点显示分裂特征、阈值、平方误差、样本数和预测值。
- 叶子节点的 `value` 即为该区域的局部常数预测值——可以直接读出"这个叶子预测的房价是多少"。
- 从根到叶子的路径就是一条完整的 if-else 规则——例如"MedInc ≤ 5.0 → Latitude ≤ 34.0 → HouseAge ≤ 25.0 → 预测房价 2.5"。
- 结构图可以直观判断树是否过深、某些分支是否基于极少样本等结构性问题。

## 5. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 残差图 | 已实现 | 回归模型的核心诊断工具——比单一数值指标更丰富 |
| 特征重要性图 | 已实现 | 树模型独有的解释性优势——线性模型看不出来 |
| 学习曲线 | 已实现 | 诊断过拟合/欠拟合和样本量充足性 |
| 树结构图 | 已实现 | 展示完整的分裂路径——树模型独有 |
| MSE / MAE / RMSE / R² 数值打印 | **未实现** | 当前流水线侧重图形化诊断而非数值指标 |
| 交叉验证 R² | **未实现** | 学习曲线内部使用了 CV，但未单独输出 CV 均值 |
| 超参数搜索 | **未实现** | 未使用 `GridSearchCV` 或 `RandomizedSearchCV` |

### 理解重点

- 当前评估设计强调**图形化诊断**而非数值指标——残差图比单个 MSE 数字更能揭示误差的分布和模式。
- 树结构图是决策树回归独有的评估资产——线性回归和 SVR 没有等效的可视化。
- 评估章节必须以源码为准——不能将"常见的回归评估手段"写成"当前已实现"。

## 6. 决策树回归 vs 分类决策树 vs 线性回归 评估对比

| 评估维度 | 分类决策树 | 线性回归 | 决策树回归 |
|---|---|---|---|
| 任务类型 | 分类 | 回归 | 回归 |
| 核心可视化 | 混淆矩阵 + ROC + 决策边界 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构图** |
| 定量指标 | accuracy / precision / recall | 无显式打印 | **无显式打印** |
| 模型解释 | 特征重要性 + 树结构 | 系数 $\beta_j$ | **特征重要性 + 树结构** |
| 诊断重点 | 类别区分度 | 线性假设验证 | **过拟合/欠拟合 + 特征贡献** |
| 独有诊断 | 决策边界 + ROC 曲线 | 系数显著性 | **树结构可视化——完整规则路径** |

## 常见坑

1. 只看特征重要性不看残差图——重要性高不意味着预测准确，特征重要但分裂不够细仍会导致高误差。
2. 只看残差图不看学习曲线——残差图只展示单次测试集上的表现，学习曲线揭示泛化趋势。
3. 期待树结构图像线性回归系数一样"简洁"——深度为 6 的树可能有几十个节点，结构图可能较大。
4. 把特征重要性解读为"对目标的正负影响"——重要性只衡量分裂贡献大小，不表示方向。

## 小结

- 当前决策树回归有四类评估输出：残差图（误差分布）、特征重要性图（特征贡献）、学习曲线（泛化趋势）、树结构图（分裂路径）——四者交叉解读才构成完整诊断。
- 残差图揭示"误差分布是否健康"，特征重要性图揭示"模型依赖哪些特征"，学习曲线揭示"复杂度与样本量是否匹配"，树结构图揭示"具体的分裂规则"。
- 与分类决策树的评估差异源于任务性质——回归评估关注误差分布和连续性，分类评估关注类别区分和混淆模式。

# 工程实现

## 本章目标

1. 理解决策树回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到四类结果图落盘的完整调用链。
3. 理解决策树回归与线性回归、SVR 在工程实现上的关键差异——无标准化、树结构图、特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadDecisionTreeRegressionDataset()` 加载 California Housing |
| `trainDecisionTreeRegressionModel(...)` | 函数 | 构建并训练 `DecisionTreeRegressor`——回归树训练的唯一入口 |
| `PipelineSpec` | 数据类 | 声明式流水线配置——关联数据集、训练器、预处理、可视化 |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行数据加载、预处理、训练、评估 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_feature_importance(...)` | 函数 | 特征重要性图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制 |
| `plot_tree_structure(...)` | 函数 | 树结构图绘制 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | 调用 `fetch_california_housing`，标签列重命名为 `price` | `DataFrame`，形状 `(20640, 9)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.decision_tree", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainDecisionTreeRegressionModel(...)`——构建并训练 `DecisionTreeRegressor` | `DecisionTreeRegressor` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.decision_tree", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载数据 → 预处理 → 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、特征重要性图、学习曲线、树结构图 | PNG 图像文件 |

### 理解重点

- 当前代码库采用**声明式流水线**架构——`PipelineSpec` 描述"要用什么数据、什么模型、什么预处理、什么可视化"，运行器根据 Spec 执行。
- 决策树回归的预处理为 `None`——在 `PipelineSpec` 中 scaler 位置为空。这是正确的：树模型不需要标准化。
- 诊断可视化列表为 `["featureImportance"]` + `["learningCurve", "treeStructure"]`——比其他回归模型多出特征重要性和树结构两项。

## 2. `PipelineSpec` 配置详情

### 参数速览

```python
PipelineSpec(
    "regression.decision_tree",        # pipeline ID
    TaskType.REGRESSION,               # 任务类型
    "regression.decision_tree",        # dataset ID
    RunnerType.REGRESSION,             # 运行器类型
    trainDecisionTreeRegressionModel,  # 训练函数
    None,                              # 预处理 —— 无标准化
    "randomSplit",                     # 切分策略
    "default",                         # 后处理
    "regression",                      # 输出目录前缀
    "regression",                      # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],             # 训练后诊断可视化
    ["learningCurve", "treeStructure"],# 学习可视化
    "decision_tree_regression",        # 结果存储子目录
)
```

### 理解重点

- `None` scaler 是决策树回归与其他回归模型的关键工程差异——线性回归和 SVR 使用 `"standardScaler"`。
- 后处理 `"default"` 表示不进行特征选择或降维等额外处理——直接使用全部 8 个特征。
- `"treeStructure"` 是决策树独有的学习可视化——分类决策树也有，但其他回归模型（线性回归、SVR）没有。

## 3. 数据依赖关系

```
fetch_california_housing(as_frame=True)
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ X_train, y_train ──→ model.fit() ──→ model
    │       │       │
    │       │       └─→ plot_learning_curve(new DecisionTreeRegressor(), X_train, y_train)
    │       │
    │       └─→ X_test ──→ model.predict() ──→ y_pred
    │               │
    │               └─→ plot_residuals(y_test, y_pred)
    │
    ├─→ model ──→ plot_feature_importance(model, feature_names)
    ├─→ model ──→ plot_tree_structure(model, feature_names)
    │
    └─→ feature_names ──→ 特征重要性图 + 树结构图
```

### 理解重点

- `y_test` 仅用于评估——不参与训练，只在残差图中与 `y_pred` 对比。
- `feature_names` 是关键中间变量——必须在切分前从 `X.columns` 保存，因为后续特征重要性图和树结构图都需要特征名。
- 学习曲线使用**独立**的 `DecisionTreeRegressor(...)` 实例——不共享已训练 `model` 的状态。

## 4. 运行器层的执行链

### 参数速览

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadDecisionTreeRegressionDataset()` | 加载 `(20640, 9)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分——无标准化 |
| 5 | 调用 `trainDecisionTreeRegressionModel(X_train, y_train)` | CART 递归分裂训练 |
| 6 | `model.predict(X_test)` | 获取测试集预测值 |
| 7 | `plot_residuals(y_test, y_pred)` | 残差诊断图 |
| 8 | `plot_feature_importance(model, feature_names)` | 特征贡献图 |
| 9 | `plot_learning_curve(new DecisionTreeRegressor(...), X_train, y_train, scoring="r2")` | 学习曲线 |
| 10 | `plot_tree_structure(model, feature_names)` | 树结构可视化 |

### 理解重点

- 运行器层是纯粹的**编排者**——不自己造数据、不实现模型、不画图，只按顺序调用各层组件。
- 步骤 9 的学习曲线创建了一个新的 `DecisionTreeRegressor` 实例（使用 `_buildLearningCurveFactory` 工厂函数），参数与主训练模型一致。
- 步骤 10 的树结构图是本流水线的特有步骤——其他回归流水线（线性回归、SVR）没有这一输出。

## 5. 决策树回归 vs 线性回归 vs SVR 工程对比

| 工程维度 | 线性回归 | SVR | 决策树回归 |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainSvrRegressionModel` | **`trainDecisionTreeRegressionModel`** |
| 模型类 | `LinearRegression` | `SVR` | **`DecisionTreeRegressor`** |
| 预处理 | `standardScaler` | `standardScaler` | **`None`** |
| 切分策略 | `randomSplit` | `randomSplit` | `randomSplit` |
| 训练后诊断 | `["featureImportance"]` | `[]` | **`["featureImportance"]`** |
| 学习可视化 | `["learningCurve"]` | `["learningCurve"]` | **`["learningCurve", "treeStructure"]`** |
| 数据量 | 200（手动合成） | 200（`make_friedman1`） | **20640（真实数据）** |
| 超参数数 | 0 | 4 | **3** |

## 常见坑

1. 误以为运行器层直接导入可视化函数——实际上是通过诊断/学习可视化列表配置，由运行器根据列表动态调用。
2. 找不到决策树特有的 `treeStructure` 可视化在其他回归模型中出现——它是树模型独有的结构可视化。
3. 混淆 `PipelineSpec` 中的多个 ID 字段——`pipelineId`、`datasetId`、结果目录名是三个不同的标识符。

## 小结

- 决策树回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- 与线性回归/SVR 的四个关键工程差异：（1）预处理为 `None`；（2）多了 `treeStructure` 可视化；（3）数据规模大一个量级；（4）超参数从系数/支持向量变为树深度/叶子约束。
- 数据依赖图的核心节点：`feature_names`（贯穿重要性图和结构图）、`model`（分裂结果）、`y_pred`（残差图）。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对决策树回归核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索回归树的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对平方误差分裂、局部常数预测、复杂度约束、树 vs 线性回归等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数和对比模型配置观察回归树行为——建立树模型直觉 |
| 参考文献 | 入口 | 提供决策树回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. 决策树回归的分裂准则是什么？为什么用平方误差而非基尼系数或信息增益？

2. 叶子节点的预测值为什么取区域内目标值的均值？从平方损失最小化的角度解释。

3. `max_depth`、`min_samples_split`、`min_samples_leaf` 分别从什么角度限制树的生长？如果三者同时设得很宽松（如 `max_depth=None, min_samples_split=2, min_samples_leaf=1`），会发生什么？

4. 决策树回归的预测函数为什么是分段常数形态？这种形态在处理房价这种连续值目标时有什么优缺点？

5. 为什么决策树回归不需要标准化？从分裂准则的数学形式给出解释。

6. `feature_importances_` 衡量的是什么？与线性回归的 `coef_` 在含义上有何根本区别？

7. 决策树回归和线性回归在处理非线性和特征交互方面各有什么优势和劣势？什么场景下应优先选树模型？

## 2. 动手练习

### 练习 1：改变 `max_depth`

将 `max_depth` 分别设为 `2`、`4`、`6`、`10`、`None`，观察树结构、残差图和学习曲线的变化。

```python
# 在 trainDecisionTreeRegressionModel 中修改
model = DecisionTreeRegressor(
    max_depth=2,  # 试试 2, 4, 6, 10, None
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42,
)
```

回答：`max_depth=2` 时树是否欠拟合（训练和验证 R² 都低）？`max_depth=None` 时是否明显过拟合（训练 R² 远高于验证 R²）？叶子节点数随深度如何变化？

### 练习 2：改变 `min_samples_leaf`

将 `min_samples_leaf` 分别设为 `1`、`3`、`10`、`50`，保持其他参数不变。

```python
model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=1,  # 试试 1, 3, 10, 50
    random_state=42,
)
```

回答：`min_samples_leaf=1` 时叶子节点数是否大幅增加？残差图中是否出现了更多极端预测？`min_samples_leaf=50` 时树是否变得过于保守？

### 练习 3：对比特征重要性在不同深度下的变化

分别记录 `max_depth=3` 和 `max_depth=10` 时的特征重要性排名。

```python
importances = model.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

回答：哪些特征在两个深度下都排在最前面？`max_depth` 增大后是否出现了新的重要特征？为什么深度会影响特征重要性的分布？

### 练习 4：对比决策树回归与线性回归在 California Housing 上的表现

在同一数据上分别训练线性回归和决策树回归，对比残差图。

```python
from sklearn.linear_model import LinearRegression

# 注意：线性回归需要标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 对比决策树
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

回答：残差图的表现有何不同？线性回归的预测值 vs 真实值图是否呈现更连续的分布？决策树的预测值是否呈现离散的分段特征？哪种模型的 R² 更高？

### 练习 5：手动计算一个叶子的预测值

用 `model.tree_` 的底层属性找到任意一个叶子节点，提取该叶子内的训练样本索引，验证该叶子的预测值是否等于这些样本目标值的均值。

```python
tree = model.tree_
# 找到叶子节点（children_left[i] == children_right[i] == -1）
leaf_indices = [i for i in range(tree.node_count) 
                if tree.children_left[i] == -1]
# 选择一个叶子
leaf_id = leaf_indices[0]
# 获取该叶子的预测值
leaf_value = tree.value[leaf_id].flatten()[0]
print(f"叶子 {leaf_id} 的预测值: {leaf_value:.4f}")
```

回答：叶子的预测值是否确实等于落到该叶子的训练样本的 `y` 均值？如果偏差较大，可能是什么原因？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth. | CART 算法的原始专著——分类树与回归树的完整理论体系和算法推导 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 9. | 教材——树模型的偏差-方差分析、剪枝策略和与集成方法的衔接 |
| 3 | scikit-learn 官方文档 — [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) | scikit-learn 的 API 参考——所有构造器参数、属性和方法的详细说明 |
| 4 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 8. | 入门教材——树模型的基础直觉、R/Python 实现和与线性模型的对比 |

## 常见坑

1. 把回归树的分裂准则与分类树混淆——回归用平方误差（MSE），分类用基尼系数或熵。
2. 在未设 `random_state` 的情况下对比不同实验——树的分裂可能因随机性而不同，实验结果不可复现。
3. 只用 R² 评估模型——树模型的残差图和结构图能揭示数值指标无法反映的局部拟合问题。
4. 把 `feature_importances_` 解读为"特征对目标的正负影响方向和大小"——重要性只看分裂贡献，不表示方向也不等效于线性系数。

## 小结

- 7 个自检问题覆盖决策树回归的核心概念：平方误差分裂、局部常数预测、三重复杂度约束、无标准化原因、特征重要性含义、与线性回归对比。
- 5 个动手练习从不同角度探索回归树的行为——改变深度和叶子约束、对比特征重要性、与线性回归横向对比、验证叶子预测值的数学本质。
- 4 篇参考文献覆盖 CART 原始专著（Breiman 1984）、两本经典教材和 scikit-learn 官方文档——构成完整的回归树学习路线。
