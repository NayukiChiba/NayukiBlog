---
title: KMeans K 均值聚类
date: 2026-05-05
category: 机器学习/聚类
tags:
  - Scikit-learn
description: KMeans聚类的数学原理、参数调优、标准化必要性及完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581783193.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 KMeans 的优化目标——最小化簇内平方和（Inertia / WCSS）。
2. 理解分配-更新两步骤的交替迭代机制，以及为什么算法保证收敛（到局部最优）。
3. 理解 `k-means++` 初始化策略的数学动机——如何减少不良局部最优的风险。
4. 理解 `inertia_` 作为损失函数的含义及其随 $k$ 增大单调递减的性质。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 簇内平方和 $\sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 优化目标 | KMeans 尝试最小化的损失函数——质心代表簇成员越紧密 |
| 分配步骤（E-step） | 迭代步骤 | 固定质心，将每个点分配给最近的质心 |
| 更新步骤（M-step） | 迭代步骤 | 固定分配，将每个质心更新为簇内所有点的均值 |
| `k-means++` | 初始化策略 | 加权随机选择初始质心，使其尽可能分散——显著减少不良局部最优 |
| `n_init` | 鲁棒机制 | 多次运行取最佳结果——以计算量换局部最优质量的提升 |
| `inertia_` | 收敛指标 | 最终簇内平方和——用于评估聚类紧密度和肘部法则选 $k$ |

## 1. KMeans 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$ 和簇数 $K$，KMeans 将数据划分为 $K$ 个不相交的集合 $C_1, C_2, \dots, C_K$，最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）：

$$
\min_{C_1, \dots, C_K} \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

其中 $\boldsymbol{\mu}_k$ 是第 $k$ 个簇的质心（该簇内所有点的均值）：

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### 理解重点

- 这是一个组合优化问题——同时优化质心位置和分配方案。直接求解是 NP-hard，因此使用交替迭代的启发式方法。
- 目标函数使用平方欧氏距离——这隐含假设簇在各方向上的方差相近（各向同性），因此 KMeans 偏好球形簇。
- `inertia_` 就是优化目标在收敛处的值——它是 KMeans 训练完成后最重要的标量输出。

## 2. 分配-更新交替迭代

KMeans 使用 EM 风格的交替最小化来逼近最优解。

### 分配步骤（Assignment Step）

固定 $K$ 个质心 $\{\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K\}$，将每个样本分配给最近的质心：

$$
C_k = \{\mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 \leq \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2, \; \forall j \neq k\}
$$

### 更新步骤（Update Step）

固定簇分配，重新计算每个簇的质心为簇内所有点的均值：

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i
$$

### 收敛性

每一步都保证不增加目标函数值——分配步取最近距离，更新步的均值为该簇 SSE 的全局最小化点。由于只有有限种分配方式，算法在有限步内收敛到**局部最优**。

### 理解重点

- 每次迭代目标函数单调不增——这是一个收敛保证，但收敛到的是局部最优而非全局最优。
- 最终结果高度依赖初始质心的选择——这正是 `k-means++` 和 `n_init` 存在的理由。
- 当前源码 `max_iter=300` 设置了迭代上限，防止在病态数据上无限循环。

## 3. `k-means++` 初始化

随机选择初始质心容易导致不良局部最优（例如两个质心落在同一簇内）。`k-means++` 通过加权随机采样使初始质心尽可能分散：

1. 从数据中随机选择第一个质心
2. 对每个点 $\mathbf{x}_i$，计算其到已选质心的最小平方距离 $D(\mathbf{x}_i)^2$
3. 以概率 $\frac{D(\mathbf{x}_i)^2}{\sum_j D(\mathbf{x}_j)^2}$ 选择下一个质心——距离已有质心越远的点越可能被选中
4. 重复 2-3 直到选满 $K$ 个质心

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `init` | `str` | 初始化方法。`'k-means++'`（默认）使用加权随机采样策略；`'random'` 从数据中纯随机选择 $K$ 个点 | `'k-means++'`、`'random'` |
| `n_init` | `int` 或 `'auto'` | 使用不同初始质心运行 KMeans 的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |

### 理解重点

- `k-means++` 是 KMeans 从"频繁得到差结果"到"实践中稳定可靠"的关键改进——它将选到不良初始质心的概率降低了多个数量级。
- `n_init=10` 是额外保险——以约 10 倍计算量换取更好的局部最优。对当前 400 样本 2 维 4 簇数据，10 次运行几乎总能找到正确的聚类结构。

## 4. 质心、标签与 `inertia_`

训练完成后，KMeans 生成三项核心输出：

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `cluster_centers_` | `ndarray`，形状 `(n_clusters, n_features)` | $\boldsymbol{\mu}_k$ | 每个簇的质心坐标——当前为 $4 \times 2$ 矩阵 |
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个样本所属簇的编号 $\{0, 1, 2, 3\}$ |
| `inertia_` | `float` | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 最终簇内平方和——值越小表示簇越紧凑 |

### 理解重点

- `cluster_centers_` 是 KMeans 区别于 DBSCAN 的标志性属性——KMeans 有显式质心，DBSCAN 没有。
- `inertia_` 随 $K$ 增大**单调递减**——当 $K=N$ 时惯性为 0（每个点自成簇）。因此它不能直接用于选择最优 $K$，需配合肘部法则（Elbow Method）使用。
- 当前源码打印 `inertia_` 到 4 位小数——这是一项聚类紧密度的定量参考。

## 5. 标准化对 KMeans 的数学必要性

KMeans 的核心操作是计算点到质心的欧氏距离：

$$
\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2 = \sum_{j=1}^{d} (x_{ij} - \mu_{kj})^2
$$

若特征 $x_1$ 的取值量纲是 $x_2$ 的 100 倍，则 $x_1$ 的差异平方将主导整个距离计算——聚类结果实际上只由 $x_1$ 决定，$x_2$ 的贡献被淹没。

### 理解重点

- 标准化后每个特征对距离的贡献均等——聚类结果反映所有特征维度的信息。
- 对 KMeans 而言标准化是必须的——它直接依赖于距离度量的几何意义。
- 这与 DBSCAN 和 SVC（RBF 核）的逻辑完全一致——任何基于距离度量的算法都需要标准化。

## 6. 为什么适合 `make_blobs` 数据

`make_blobs` 从各向同性高斯分布 $\mathcal{N}(\mathbf{c}_k, \sigma^2 \mathbf{I})$ 采样生成簇：

- 簇内样本在质心周围球形散布——与 KMeans 的平方欧氏距离假设完美匹配
- 各簇方差统一（`cluster_std=0.8`）——避免了 KMeans 在方差差异大时偏向大方差簇的问题
- 4 个质心分布在二维平面的不同象限——分配步骤容易做出正确判断

### 理解重点

- `make_blobs` 是为 KMeans "量身定制"的数据——它满足了 KMeans 的所有隐假设（球形、等方差）。
- 这种设计在教学上有意为之——先在理想数据上展示算法优势，再通过练习引导理解边界条件。
- 对比 DBSCAN 的 `make_moons`——不同聚类算法需要不同的数据形态来展示各自最强的一面。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 预设簇数 | $K$ | `n_clusters=4` |
| 优化目标 | $\min \sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | KMeans 算法核心 |
| 分配步骤 | $C_k = \{\mathbf{x}_i : \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2 = k\}$ | KMeans 内部迭代 |
| 更新步骤 | $\boldsymbol{\mu}_k = \frac{1}{\vert C_k \vert} \sum_{\mathbf{x} \in C_k} \mathbf{x}$ | KMeans 内部迭代 |
| 质心初始化 | `k-means++` 加权采样 | `init='k-means++'` |
| 多轮初始化 | 运行 $n$ 次取惯性最小者 | `n_init=10` |
| 最大迭代次数 | — | `max_iter=300` |
| 质心坐标 | $\boldsymbol{\mu}_k$ | `model.cluster_centers_` |
| 簇分配标签 | $\{0, 1, \dots, K-1\}$ | `model.labels_` |
| 簇内平方和 | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | `model.inertia_` |
| 迭代次数 | — | `model.n_iter_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 不理解 KMeans 收敛到局部最优而非全局——`k-means++` 和 `n_init` 旨在缓解而非根除。
2. 用 `inertia_` 直接比较不同 $K$ 的模型——惯性随 $K$ 单调递减，需配合肘部法则或轮廓系数。
3. 在不标准化的数据上运行——距离计算被量纲绑架，聚类结果由尺度最大的特征主导。
4. 混淆 `labels_` 编号与 `true_label` 编号——簇标签是任意的，0 不一定对应真实标签 0。

## 小结

- KMeans 的数学核心链：簇内平方和 $\min\sum\|\mathbf{x}-\boldsymbol{\mu}\|^2$ -> 分配-更新交替迭代 -> `k-means++` 加权初始化 -> `n_init` 多轮择优 -> 收敛到局部最优。
- KMeans 有显式质心（`cluster_centers_`）和可量化的损失（`inertia_`）——这是它区别于 DBSCAN 最核心的数学特征。
- 当前源码 `KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)` 是针对 `make_blobs` 球形高斯簇的最经典配置。

# 数据构成

## 本章目标

1. 明确本仓库 KMeans 数据来自 `make_blobs(...)` 构造的球形高斯簇数据。
2. 明确特征列与 `true_label` 在当前流水线中的角色差异——这是无监督聚类，`true_label` 仅用于结果对照。
3. 明确标准化发生在什么位置，以及为什么它对基于距离的 KMeans 至关重要。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClusteringData.kmeans()` | 方法 | 生成 KMeans 使用的二维球形 blob 聚类数据 |
| `make_blobs(...)` | 函数 | scikit-learn 提供的各向同性高斯簇数据生成器 |
| `kmeans_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `true_label` | 列名 | 真实簇标签——仅用于与预测结果视觉对照，不参与 `fit()` |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——距离度量的前置条件 |

## 1. 数据生成：`make_blobs()`

当前 KMeans 数据来自 `ClusteringData.kmeans()`，底层调用 `sklearn.datasets.make_blobs()`。

### 参数速览

适用函数：`make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。默认 400，4 个簇各约 100 个样本 | `400`、`500` |
| `centers` | `int` 或 `ndarray` | 簇的数量（传入整数时随机生成质心坐标）或质心坐标矩阵。默认 `4` | `4`、`3`、`[[0,0],[5,5]]` |
| `cluster_std` | `float` 或 `array` | 各簇的标准差。`0.8` 使簇内样本适度分散——标准差越大簇越松散、越难聚类 | `0.5`、`0.8`、`1.5` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| `shuffle` | `bool` | 是否打乱样本顺序。默认 `True` | `True` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(400, 2)$，$y$ 取值 $\{0, 1, 2, 3\}$ | — |

### 示例代码

```python
X, y = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)
data = DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})
```

### 理解重点

- `make_blobs` 生成的是各向同性高斯簇——簇内点在质心周围球形散布，与 KMeans 的平方欧氏距离优化目标完美匹配。
- `cluster_std=0.8` 在"足够紧凑可清晰聚类"和"适度分散有真实感"之间取得了教学平衡。
- 4 个簇的质心随机分布在二维平面的不同区域——确保簇间距离 >> 簇内散布，分配步骤易于判断。

## 2. 特征列与 `true_label` 的角色

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["true_label"])` |
| `y_true` | `ndarray` | 真实簇标签 $y_i \in \{0, 1, 2, 3\}$，**仅用于结果对照**，不参与 KMeans 的 `fit()` | `data["true_label"].values` |

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `true_label` 不传入 `model.fit()`——KMeans 是无监督算法，`fit(X)` 只接收特征。
- `true_label` 的唯一目的是在 `plot_clusters(...)` 中与 `model.labels_` 做左右对照——帮助读者判断算法是否恢复了真实的 4 簇结构。
- 簇标签编号（0, 1, 2, 3）在 `labels_` 和 `true_label` 之间不一定对应——这是聚类评估的正常情况。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 去掉 `true_label` 后的二维特征矩阵 | `X` |
| 返回值 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- KMeans 依赖欧氏距离到质心：$\|\mathbf{x} - \boldsymbol{\mu}_k\|^2 = \sum_j (x_j - \mu_{kj})^2$。如果特征量纲不同，距离计算被尺度主导。
- 标准化后各特征平等贡献于分配决策——质心的位置和形状在几何上才有意义。
- 与 DBSCAN 流水线一致——无切分，在全量数据上直接 `fit_transform`。

## 数据可视化

![原始数据散点图](https://img.yumeko.site/file/blog/articles/1780737759347.png)

![真实标签散点图](https://img.yumeko.site/file/blog/articles/1780737770006.png)

![聚类分布图](https://img.yumeko.site/file/blog/articles/1780737760634.png)

![特征相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

## 常见坑

1. 把 `true_label` 当成训练标签传入 `model.fit()`——KMeans 是无监督算法，不接受标签参数。
2. 忽略标准化——距离度量被特征量纲绑架，聚类结果失真。
3. 期望 `labels_` 的簇编号（0, 1, 2, 3）与 `true_label` 完全对应——标签编号是任意的。

## 小结

- 当前 KMeans 数据来自 `make_blobs(n_samples=400, centers=4, cluster_std=0.8)`：2 个连续特征、4 个各向同性高斯簇。
- 数据流为：`make_blobs` -> DataFrame（`x1`、`x2` + `true_label`）-> 剥离 `true_label` -> 全量标准化。
- `true_label` 仅用于结果对照——这是无监督聚类与有监督分类在数据处理上的根本差异。
- `make_blobs` 的球形高斯假设与 KMeans 的平方欧氏距离优化天然匹配——这使其成为 KMeans 教学的理想基准数据。

# 思路与直觉

## 本章目标

1. 用直观方式理解 KMeans 的中心式聚类思路——"先猜质心位置，再反复逼近"。
2. 理解为什么 KMeans 在 `make_blobs` 球形数据上效果极好，但在弯曲结构上会失败。
3. 通过与 DBSCAN 的对比，建立 KMeans 在聚类算法图中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 中心式聚类 | 核心直觉 | 每个簇由一个质心代表，样本归属于最近的质心 |
| 交替逼近 | 迭代机制 | 反复"分点->移心->分点->移心"直到稳定 |
| 预设 $K$ | 前置约束 | 必须在运行前指定期望的簇数——算法不会自己决定 |
| 球形偏好 | 隐假设 | KMeans 的簇边界是 Voronoi 多边形——每个簇被最近质心"吸引" |
| `make_blobs` | 理想数据 | 各簇围绕质心球形散布——完全符合 KMeans 的模型假设 |

## 1. 为什么需要 KMeans

KMeans 是最直观的聚类方法之一。它的想法很简单：

> 如果数据中包含 $K$ 个自然的分组，每个组应该围绕某个中心聚集。找到这 $K$ 个中心，也就找到了这 $K$ 个组。

### 理解重点

- KMeans 把聚类问题转化成了"找 $K$ 个最有代表性的中心"——每个点只需回答"离哪个中心最近"。
- 这个直觉在日常中随处可见——城市群围绕核心城市、卫星围绕行星——只不过 KMeans 用数学严格化了这个过程。
- 正因为假设了"中心代表"，KMeans 偏好各向同性（球形）的簇——中心到各方向的吸引力均等。

## 2. 用"反复调整"理解迭代过程

KMeans 的工作方式可以想象成：

1. **随便猜 $K$ 个中心**——可以在数据点中随机选
2. **分地盘**——每个点归离自己最近的中心管辖
3. **挪中心**——每个中心移到它所辖区域内所有点的正中间
4. **重新分地盘**——因为中心挪了，有些点该换归属
5. **继续挪**——重复"分地盘->挪中心"，直到中心不再大幅移动

### 理解重点

- 这就像"一座城市、多所学校"的学区调整——先划片区（分配），再根据学生分布调整学校位置（更新），反复多次直到稳定。
- 每一步都会让"学生到学校的总走路距离"减少（或至少不增加）——目标函数单调递减。
- 但最终结果取决于"第一稿学校选址"——在不同位置起步，可能收敛到不同结果。

## 3. 为什么 KMeans 在球形数据上表现好

当前数据来自 `make_blobs(n_samples=400, centers=4, cluster_std=0.8)`：

- 4 个簇各自围绕一个质心球形散布——这正是 KMeans 假设的数据形态
- 各簇方差相近（均为 `cluster_std=0.8`）——到中心的距离在各簇间可比
- 簇间距离远大于簇内散布——"分地盘"时几乎不会产生歧义

### 理解重点

- `make_blobs` 的每个簇从 $\mathcal{N}(\mathbf{c}_k, \sigma^2 \mathbf{I})$ 采样——每个维度方差相同，形成完美的圆形散点（二维下）或球形散点（高维下）。
- 这与 KMeans 基于平方欧氏距离的 Voronoi 划分天然吻合——每个点到哪个中心最近一目了然。
- 这就是为什么 KMeans 在这类数据上几乎总是正确——数据形态与算法假设高度一致。

## 4. `k-means++` 的直觉：让中心先散开

如果一开始 $K$ 个质心都落在同一个真正的簇里（随机选择的倒霉情况），那反复迭代也只是在那一个簇内纠缠——其他簇被完全忽略。

`k-means++` 的策略可以直观理解为：

- **已经选中的中心附近，不再轻易选下一个**——给远离现有中心的点更高的选中概率
- 这样选出的 $K$ 个初始中心在数据中分布更均匀——每个真实簇至少抓到一个初始中心的机会大幅提高

### 理解重点

- 这就像在公园里均匀摆放摊位——已经有人在某处摆了摊位，下一个就该往还有空档的地方去。
- `k-means++` 不是选"离当前中心最远"那个点（那会被孤立噪声点欺骗），而是"距离与概率挂钩"——远者优先但不独享。

## 5. 与 DBSCAN 的直觉对比

| 维度 | KMeans | DBSCAN |
|---|---|---|
| 核心问题 | 离哪个中心最近？ | 哪些区域足够密集且彼此连通？ |
| 簇的形状假设 | 球形/凸形（Voronoi 划分） | 任意形状（仅需密度连通） |
| 簇数 | 必须预设 $K$ | 由密度结构自动决定 |
| 噪声处理 | 强制分配——每个点必属一簇 | 天然识别为标签 $-1$ |
| 质心 | 有（`cluster_centers_`） | 无 |
| 预测新样本 | 支持（`predict(X_new)`） | sklearn 不支持 |
| 理想数据 | `make_blobs`（球形高斯簇） | `make_moons`（弯曲不规则结构） |

### 理解重点

- KMeans 和 DBSCAN 不是"谁更强"——它们分别适合形状截然不同的数据。
- 将 KMeans 用于 `make_moons`：Voronoi 直线边界会沿月牙弧形切开——不是 KMeans 有问题，而是数据形态不匹配算法假设。
- 将 DBSCAN 用于 `make_blobs`：表现通常也不错（只要能找到合适的 `eps`），但 KMeans 更简洁高效——此时 KMeans 是更好的选择。

## 可视化

![聚类结果图](https://img.yumeko.site/file/blog/articles/1780736373148.png)

## 常见坑

1. 不加思考地使用 KMeans 处理任意形状的数据——数据是月牙形时 Voronoi 直线边界必然出错。
2. 忘记预设 $K$ 是 KMeans 的刚性约束——选错 $K$ 等于强制拆分或合并真实簇。
3. 把每次运行得到的不同 `labels_` 编号理解为模型不稳定——编号是任意的，簇的结构才重要。
4. 不标准化数据——让特征量纲差异主导质心距离计算。

## 小结

- KMeans 的直觉核心是中心式聚类：预设 $K$ 个质心 -> 反复"分配-更新"交替 -> 收敛到局部最优。
- `make_blobs` 球形高斯簇与 KMeans 的模型假设完美匹配——这是展示其优势的最佳教学数据。
- KMeans 与 DBSCAN 在直觉上截然相反：一个从全局中心出发划分，一个从局部密度出发连通——选哪个取决于数据形态，而非算法优劣。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KMeans`。
2. 理解 `KMeans` 的核心构造器参数（`n_clusters`、`init`、`n_init`、`max_iter`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`cluster_centers_`（质心）、`labels_`（簇分配）、`inertia_`（簇内平方和）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.cluster.KMeans` 模型，打印 `inertia_` 日志 |
| `KMeans(...)` | 类 | scikit-learn 提供的 K 均值聚类器——基于交替最小化簇内平方和 |
| `model.fit(X_train)` | 方法 | 执行分配-更新交替迭代直至收敛——无监督，不传入标签 |
| `model.cluster_centers_` | 属性 | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性属性 |
| `model.inertia_` | 属性 | 收敛时的簇内平方和——衡量簇紧密度，用于肘部法则选 $K$ |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，形状 $(400, 2)$，传入 `KMeans.fit()` | `X_scaled` |
| `n_clusters` | `int` | 预设簇数 $K$。当前设为 `4`，与 `make_blobs(centers=4)` 一致 | `3`、`4`、`5` |
| `init` | `str` | 质心初始化策略。`'k-means++'`（默认）使用加权随机采样使初始质心分散 | `'k-means++'`、`'random'` |
| `n_init` | `int` | 不同初始质心下独立运行 KMeans 的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300`——对 400 样本通常远不需要这么多 | `100`、`300` |
| `random_state` | `int` | 随机种子，保证质心初始化和结果可复现。默认 `42` | `42` |
| 返回值 | `KMeans` | 已完成 `fit()` 的模型对象，含 `cluster_centers_`、`labels_`、`inertia_` | — |

### 示例代码

```python
from model_training.clustering.kmeans import train_model

model = train_model(X_scaled)
```

### 理解重点

- 和监督学习分册不同，`train_model(...)` **没有 `y_train` 参数**——KMeans 是无监督算法。
- `n_clusters=4` 是必须预设的前置条件——KMeans 不会自己决定簇数，需要用户根据领域知识或肘部法则选定。
- `train_model(...)` 是对 `sklearn.cluster.KMeans` 的薄封装——算法本体是 scikit-learn 基于 Lloyd 算法的高效实现。

## 2. `KMeans` 构造器参数

### 参数速览

适用 API：`KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_clusters` | `int` | 预设簇数 $K$。当前设为 `4`。这是 KMeans 最重要的参数——选错直接导致错误合并或分裂 | `3`、`4`、`5`、`8` |
| `init` | `str` | 质心初始化方法。`'k-means++'`（默认）加权随机采样；`'random'` 纯随机选 $K$ 个点。scikit-learn 1.2+ 还支持传入 `ndarray` 自定义初始质心 | `'k-means++'`、`'random'` |
| `n_init` | `int` 或 `'auto'` | 不同初始质心下的独立运行次数，返回 `inertia_` 最小的结果。`'auto'`（scikit-learn 1.4+ 默认）在 `init='k-means++'` 时自动设为 `1`。当前显式设为 `10` | `1`、`10`、`'auto'` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300`——大多数数据远在此之前就收敛了 | `100`、`300`、`500` |
| `tol` | `float` | 收敛容忍度——当质心位移的 Frobenius 范数相对变化小于此值时停止迭代。默认 `1e-4` | `1e-3`、`1e-4` |
| `algorithm` | `str` | 计算后端。`'lloyd'`（经典 EM 风格）、`'elkan'`（三角不等式加速，适合密集数据）、`'auto'`。默认 `'lloyd'` | `'lloyd'`、`'elkan'` |
| `random_state` | `int` | 随机种子，保证质心初始化可复现。当前设为 `42` | `42` |

### 示例代码

```python
model = KMeans(
    n_clusters=4,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
)
model.fit(X_train)
```

### 理解重点

- KMeans 的核心参数是 `n_clusters`——它必须在训练前确定，算法不会自行推断。这是 KMeans 与 DBSCAN 最根本的工程差异。
- `init='k-means++'` 是默认值的黄金标准——在绝大多数情况下比 `'random'` 收敛更快、结果更好。
- `n_init=10`（当前显式设为 10）是 scikit-learn 较旧版本的默认值——用计算量换取更可靠的局部最优。
- KMeans 的 `fit()` 是迭代优化（分配-更新交替），这与 DBSCAN 和 GaussianNB（一步式）在计算特征上截然不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `cluster_centers_` | `ndarray`，形状 `(n_clusters, n_features)` | $\boldsymbol{\mu}_k$ | $K$ 个簇的质心坐标——当前为 $4 \times 2$ 矩阵 |
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个样本所属簇的编号 $\{0, 1, 2, 3\}$ |
| `inertia_` | `float` | $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 收敛时的簇内平方和——值越小簇越紧凑 |
| `n_iter_` | `int` | 收敛所用迭代次数 | 反映收敛速度——通常远小于 `max_iter=300` |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |

### 示例代码

```python
print(f"n_clusters: {n_clusters}")
print(f"inertia: {model.inertia_:.4f}")
print(f"质心坐标:\n{model.cluster_centers_}")
```

### 理解重点

- `cluster_centers_` 是 KMeans 最有教学意义的属性——它把"中心式聚类"这一直觉直接映射为可观察的坐标。
- `inertia_` 是 KMeans 的核心输出——它量化了整个聚类的紧密度。当前源码打印到 4 位小数。
- 与 DBSCAN 的关键对比：KMeans 有 `cluster_centers_` 和 `inertia_`，DBSCAN 有 `core_sample_indices_` 和 `-1` 噪声标签——前者反映"中心在哪、多紧"，后者反映"核心在哪、谁被排除"。

## 4. 训练阶段的工程封装

除了 `KMeans(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察迭代优化耗时——通常极快（毫秒级） |
| `n_clusters` 日志 | 确认预设簇数 |
| `inertia_` 日志 | 观察最终簇内平方和——聚类紧密度定量参考 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键统计量。
- `inertia_` 是 KMeans 独有的日志输出——它直接反映聚类紧密度，是肘部法则的基础。
- KMeans 不打印簇数量（因为 $K$ 是预设的），不打印噪声点数量（因为每个点都被强制分配）——这两点与 DBSCAN 形成鲜明对比。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——KMeans 是无监督算法，不接受标签。
2. 忽略 `n_clusters` 是必须预设的前置约束——选错 $K$ 意味着预设的分组方式与数据真实结构不匹配。
3. 把不同运行中 `labels_` 编号的变化理解为模型不稳定——编号是任意的，重要的是簇的结构和边界。
4. 忘记 `inertia_` 随 $K$ 单调递减——不能直接用它选择 $K$，需配合肘部法则。

## 小结

- `train_model(...)` 是本仓库 KMeans 的核心训练入口，是对 `sklearn.cluster.KMeans` 的薄封装。
- `KMeans` 的核心参数是 `n_clusters`（预设 $K$）、`init`（初始化策略）、`n_init`（多轮择优次数）——三者共同决定了聚类的质量和稳定性。
- 训练完成后的核心属性：`cluster_centers_`（质心坐标）、`labels_`（簇分配）、`inertia_`（簇内平方和）——三者分别回答了"中心在哪""谁属于谁""有多紧"。
- KMeans 有 `cluster_centers_` 和 `inertia_`、有 `predict()` 方法、但需要预设 $K$——这三点构成了它与 DBSCAN 最核心的工程差异。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 KMeans 流水线从数据复制到聚类输出的完整步骤。
2. 理解 KMeans 无训练/测试切分、但拥有 `predict()` 方法的工程特征——与 DBSCAN 有本质差异。
3. 理解 `fit()` 即训练、`labels_` + `cluster_centers_` + `inertia_` 三者共同构成聚类输出的流程特征。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `kmeans_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `StandardScaler` | 类 | 对全量特征做一致性标准化——欧氏距离计算的前置条件 |
| `train_model(...)` | 函数 | 调用 `KMeans.fit()` 执行分配-更新交替迭代，返回模型对象 |
| `model.fit(X_scaled)` | 方法 | 在标准化特征上执行 KMeans 聚类——迭代优化质心位置以最小化簇内平方和 |
| `model.labels_` | 属性 | 每个样本的簇分配标签——KMeans 强制每个点必属一簇，无噪声标记 |
| `model.cluster_centers_` | 属性 | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性输出 |
| `model.inertia_` | 属性 | 收敛时的簇内平方和——衡量聚类紧密度，用于肘部法则选 $K$ |

## 1. 流水线起点：复制数据并拆出特征与对照标签

### 示例代码

```python
data = kmeans_data.copy()
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `kmeans_data`。
- `true_label` 被单独保存为 `y_true`——它**不参与**后续的 `fit()`，只在最后的 `plot_clusters(...)` 中作为对照显示。
- 这是聚类与分类最根本的工程差异——没有"标签=训练目标"的概念。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 2)$ | 去掉 `true_label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- KMeans 流水线**没有**训练/测试切分——无监督聚类不需要验证集。
- `fit_transform` 直接在全量数据上计算统计量并变换——不存在测试集数据泄露的风险。
- 标准化是必须的——欧氏距离 $\|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ 的几何意义依赖于各维度尺度一致。

## 3. 模型训练：`fit()` 执行分配-更新交替迭代

### 参数速览

适用 API：`train_model(X_scaled)` -> `model.fit(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征矩阵——KMeans 的唯一输入 | `X_scaled` |
| 无 `y` 参数 | — | KMeans 是无监督算法——`fit(X)` 不接受标签 | — |

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- `KMeans.fit(X_scaled)` 内部流程：`k-means++` 初始化 $K$ 个质心 -> 分配每个点到最近质心 -> 更新质心为簇内均值 -> 重复直到收敛或达到 `max_iter`。
- 与 DBSCAN 不同——KMeans 的 `fit()` 是迭代优化过程（有 `n_iter_` 属性记录迭代次数），而非一次性密度扩展。
- 与分类模型比较：分类流程是 `fit(X, y)` -> `predict(X_test)`，KMeans 流程是 `fit(X)` -> `labels_` + `cluster_centers_` + `inertia_`。

## 4. 获取聚类结果

### 参数速览

| 属性名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 $(400,)$ | 每个样本的簇分配标签 $\{0, 1, 2, 3\}$——KMeans 强制分配，无噪声标记 | `model.labels_` |
| `cluster_centers_` | `ndarray`，形状 $(4, 2)$ | $K$ 个簇的质心坐标——KMeans 区别于 DBSCAN 的标志性属性 | `model.cluster_centers_` |
| `inertia_` | `float` | 收敛时的簇内平方和 $\sum_k \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$ | 当前打印到 4 位小数 |
| `n_iter_` | `int` | 收敛所用迭代次数——通常远小于 `max_iter=300` | `model.n_iter_` |

### 示例代码

```python
labels_pred = model.labels_
centers = model.cluster_centers_
```

### 理解重点

- `model.labels_` 对训练样本的簇分配——每个点必属一簇，没有噪声标记。这是 KMeans 与 DBSCAN（有 $-1$ 噪声标签）的关键差异。
- `model.cluster_centers_` 是 KMeans 最有教学意义的输出——它把"中心式聚类"这一直觉直接映射为可观察的坐标。
- `model.inertia_` 在训练日志中打印——它是 KMeans 独有的定量输出，DBSCAN 没有对应物。

## 5. KMeans 的 `predict()` 方法

KMeans 与 DBSCAN 的一个重要工程差异：KMeans **支持**对新样本的簇归属预测。

### 参数速览

适用 API：`model.predict(X_new)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_new` | `array_like`，形状 `(n, 2)` | 经过**同一 `scaler` 标准化后**的新样本特征 | `X_new_scaled` |
| 返回值 | `ndarray`，形状 `(n,)` | 每个新样本的簇编号 $\{0, 1, 2, 3\}$——归属最近质心 | — |

### 示例代码

```python
# 新样本必须先经过训练阶段相同的 scaler 变换
X_new_scaled = scaler.transform(X_new)
new_labels = model.predict(X_new_scaled)
```

### 理解重点

- `predict()` 的原理很简单——计算新样本到 $K$ 个质心的距离，返回最近质心的编号。不需要重新迭代。
- 当前流水线没有演示这一步——教学重点在聚类本身，而非对新样本的预测。
- 这与 DBSCAN 形成鲜明对比——sklearn 的 DBSCAN 根本没有 `predict()` 方法。

## 6. 聚类结果可视化

### 参数速览

适用函数：`plot_clusters(X_scaled, labels_pred=model.labels_, labels_true=y_true, centers=model.cluster_centers_, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征，用于散点图的坐标 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 $(400,)$ | KMeans 的预测簇标签——来自 `model.labels_` | `model.labels_` |
| `labels_true` | `ndarray`，形状 $(400,)$ | 真实簇标签（仅用于视觉对照） | `y_true` |
| `centers` | `ndarray`，形状 $(4, 2)$ | 质心坐标——以红色 `X` 标记显示在预测图中 | `model.cluster_centers_` |

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    centers=model.cluster_centers_,
    title="KMeans 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `plot_clusters(...)` 是当前 KMeans 分册唯一的可视化函数——与分类分册的四类评估（混淆矩阵+ROC+决策边界+学习曲线）完全不同。
- `centers` 参数是 KMeans 独有的——DBSCAN 调用 `plot_clusters` 时不传此参数。红色 `X` 标记直观展示每个簇的中心位置。
- 双侧对照布局：左侧显示 `labels_pred`（KMeans 聚类结果 + 质心标记），右侧显示 `labels_true`（真实簇标签）——帮助读者判断算法是否恢复了真实的 4 簇结构。

## 常见坑

1. 期望当前流水线有训练/测试切分——无监督聚类不划分训练集和验证集。
2. 把 `true_label` 当成 `fit()` 的输入——它仅用于最终的可视化对照。
3. 忘记 `predict()` 前需要对新样本做**同一 `scaler` 的 `transform`**——新样本必须经过与训练数据相同的标准化。
4. 把 KMeans 的 `predict()` 与分类模型的 `predict()` 混为一谈——KMeans 只是返回最近质心的编号，没有概率输出。

## 小结

- 当前 KMeans 流水线极为简洁：复制数据 -> 剥离 `true_label` -> 全量标准化 -> `fit(X)` 交替迭代 -> `labels_` + `cluster_centers_` + `inertia_` 三输出 -> 可视化对照（含质心标记）。
- 与 DBSCAN 的核心差异：KMeans 有 `cluster_centers_`（DBSCAN 没有）、有 `inertia_`（DBSCAN 没有）、有 `predict()`（DBSCAN 没有）、强制分配无噪声点（DBSCAN 有 $-1$ 噪声）、$K$ 必须预设（DBSCAN 由密度自动决定）。
- 与分类分册的核心差异：无切分（无 `train_test_split`）、无监督标签、无 `predict_proba`、无混淆矩阵/ROC/学习曲线。

# 评估与诊断

## 本章目标

1. 明确当前仓库 KMeans 实现的评估手段——聚类对照散点图（含质心标记）+ `inertia_` 日志。
2. 理解 KMeans 特有的评估逻辑——`inertia_` 作为簇紧密度指标、肘部法则作为选 $K$ 的方法。
3. 理解轮廓系数等常见聚类指标的原理，以及它们未在流水线中显式计算的原因。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_clusters(...)` | 函数 | 绘制预测簇标签与真实标签的左右对照散点图（含质心 `X` 标记） |
| `inertia_` | 指标 | 最终簇内平方和——衡量簇紧密度，值越小簇越紧凑 |
| 肘部法则（Elbow Method） | 诊断方法 | 绘制 `inertia_` 随 $K$ 变化的曲线，拐点即为推荐的 $K$ |
| 轮廓系数（Silhouette Score） | 指标 | 量化聚类紧密度与分离度——$s_i \in [-1, 1]$，越接近 1 越好 |
| 质心位置对照 | 诊断 | 观察 `cluster_centers_` 是否落在各簇的几何中心——判断收敛质量 |

## 1. 当前仓库的评估入口

KMeans 的评估与分类分册截然不同——没有混淆矩阵、没有 ROC 曲线、没有决策边界、没有学习曲线。评估依赖两种手段：

1. **聚类对照散点图** —— 左右并列显示算法聚类结果（含红色 `X` 质心标记）与真实标签，视觉上判断簇结构恢复质量
2. **`inertia_` 日志** —— 训练完成后打印的簇内平方和，量化聚类紧密度

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    centers=model.cluster_centers_,
    title="KMeans 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 聚类评估不能像分类那样用 `y_pred == y_test` 计算准确率——簇标签编号（0, 1, 2, 3）与真实标签编号不一定对应，且聚类关注的是簇结构而非逐样本正确率。
- 当前评估方法对教学场景来说足够直观——读者可以一眼看出 KMeans 是否恢复了 4 个球形簇，质心标记是否落在各簇几何中心。
- `inertia_` 是最简明的定量参考——配合肘部法则可以辅助判断 $K$ 的选择是否合理。

## 2. 聚类对照散点图能观察什么

### 参数速览

适用函数：`plot_clusters(X, labels_pred, labels_true, centers, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `ndarray`，形状 `(n_samples, 2)` | 标准化后的全量特征矩阵 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 `(n_samples,)` | KMeans 聚类结果——来自 `model.labels_` | `model.labels_` |
| `labels_true` | `ndarray`，形状 `(n_samples,)` | 真实簇标签——来自 `make_blobs` 生成时的 `y` | `y_true` |
| `centers` | `ndarray`，形状 `(n_clusters, 2)` | 质心坐标——以红色 `X` 标记显示 | `model.cluster_centers_` |

### 理解重点

- 左侧图展示 KMeans 的聚类结果——用不同颜色标记不同簇，红色 `X` 标记各簇质心位置。
- 右侧图展示真实簇标签——为读者提供"正确答案"的视觉基准。
- 对比重点：
  - 簇的边界是否符合 Voronoi 划分预期——直线边界将平面划分为 $K$ 个凸区域
  - 质心是否落在各簇的几何中心——偏离说明收敛到局部最优或 $K$ 设置不当
  - 预测簇与真实簇在空间分布上的一致性——颜色编号可以不同，但空间分组应对应
- 与 DBSCAN 的可视化差异：KMeans 有红色 `X` 质心标记，无边界的噪声点——这反映了"中心式聚类"与"密度聚类"在可视化上的根本区别。

## 3. `inertia_` 与肘部法则

### `inertia_` 的含义

`inertia_`（簇内平方和）是 KMeans 的核心定量输出：

$$
\text{inertia} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

- 值越小 -> 簇内样本离质心越近 -> 簇越紧凑
- 但 `inertia_` **随 $K$ 增大单调递减**——当 $K = N$（每个点自成簇）时 `inertia_ = 0`。因此不能直接用 `inertia_` 比较不同 $K$ 的模型

### 肘部法则

肘部法则是 KMeans 选 $K$ 的经典方法：

1. 对 $K = 1, 2, 3, \dots, K_{\max}$ 分别训练 KMeans
2. 记录每个 $K$ 对应的 `inertia_`
3. 绘制 $K$-`inertia_` 曲线——拐点（"肘部"）即为推荐的 $K$

### 理解重点

- 肘部法则用"边际收益递减"的直觉选 $K$——当增加一个簇带来的紧密度提升显著变小时，就是合理的 $K$。
- 当前流水线直接使用 `n_clusters=4`（与 `make_blobs(centers=4)` 一致），没有演示肘部法则——但文档应说明这是选 $K$ 的标准方法。
- `inertia_` 是 KMeans 独有的诊断指标——DBSCAN 没有对应物（DBSCAN 用 `n_clusters` 和 `n_noise` 做诊断）。

## 4. 当前实现中尚未纳入但常见的聚类指标

| 指标 | 公式 / 含义 | 当前未使用的原因 |
|---|---|---|
| 轮廓系数（Silhouette Score） | $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$，其中 $a_i$ 为点到同簇其他点的平均距离（紧密度），$b_i$ 为点到最近邻簇的平均距离（分离度）。$s_i \in [-1, 1]$，越接近 1 越好 | 当前侧重直观教学而非量化评估，且轮廓系数需要额外计算步骤 |
| 调整兰德指数（ARI） | 度量两个聚类分配之间的相似度（共识），扣除随机期望。ARI $\in [-1, 1]$，1 表示完全一致 | 需要真实标签——当前 `true_label` 确实存在，但流水线选择直接用视觉对照而非数值指标 |
| Davies-Bouldin 指数 | 簇间相似度均值（越小越好）——衡量簇内紧密度与簇间分离度的比值 | 无需真实标签——但当前分册的教学重点是理解 KMeans 机制，而非指标对比 |
| Calinski-Harabasz 指数 | 簇间离散度与簇内离散度的比值（越大越好）——基于方差分析的思路 | 同 Davies-Bouldin——教学型仓库优先展示直观的视觉诊断 |

### 理解重点

- 聚类评估体系与分类评估有本质区别——分类可以直接比对预测标签与真实标签，而聚类的"正确性"更多体现在簇结构的合理性（紧密度 + 分离度）。
- 当前流水线不显式计算这些指标——文档可以提到它们是扩展方向，但不能写成"当前源码已在计算"。
- 对于教学型仓库，散点图视觉对照（含质心标记）+ `inertia_` 日志 + 肘部法则概念已经能有效支撑 KMeans 的诊断需求。

## 5. KMeans 评估与 DBSCAN 评估的关键差异

| 维度 | KMeans | DBSCAN |
|---|---|---|
| 核心定量指标 | `inertia_`（簇内平方和） | `n_clusters`、`n_noise` |
| 选参方法 | 肘部法则（$K$-`inertia_` 曲线） | `eps`/`min_samples` 联合调试 |
| 可视化特征 | 红色 `X` 质心标记 | 噪声点特殊着色（$-1$） |
| 簇数来源 | 预设 $K$（用户指定） | 自动决定（密度结构驱动） |
| 噪声概念 | 无——每个点强制归属一簇 | 有——噪声点标记为 $-1$ |

### 理解重点

- KMeans 的评估围绕"簇紧密度"展开（`inertia_`、肘部法则），DBSCAN 的评估围绕"密度结构恢复"展开（簇数量、噪声比例）。
- 两种评估哲学的差异源于算法本质：KMeans 优化几何紧密度，DBSCAN 检测密度连通区域。
- 一个关注"中心在哪、有多紧"，另一个关注"核心在哪、谁被排除"。

## 评估图表

![聚类分布图](https://img.yumeko.site/file/blog/articles/1780736373148.png)

## 常见坑

1. 直接用 `inertia_` 比较不同 $K$ 的模型——`inertia_` 随 $K$ 单调递减，需配合肘部法则使用。
2. 把分类的评估框架（混淆矩阵、ROC、accuracy）搬到聚类评估——两者评估哲学根本不同。
3. 看到 `labels_` 编号与 `true_label` 不对应就认为模型失败——簇标签编号是任意的，空间分组对应才重要。
4. 忽略质心在散点图中的位置——质心偏离簇几何中心是收敛到局部最优的信号。

## 小结

- 当前仓库对 KMeans 的评估简洁而直观：聚类对照散点图（含质心标记）看簇结构恢复质量，`inertia_` 日志看聚类紧密度。
- KMeans 没有混淆矩阵、没有 ROC、没有准确率——这些是监督分类的评估手段，不适用于无监督聚类。
- `inertia_` 是 KMeans 的核心诊断指标——配合肘部法则可以在无真实标签的情况下辅助选择 $K$。
- KMeans 评估与 DBSCAN 评估的核心差异：前者关注"紧密度"（`inertia_`），后者关注"密度结构"（`n_clusters`/`n_noise`）。

# 工程实现

## 本章目标

1. 从工程角度看清 KMeans 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和聚类可视化分别负责什么。
3. 理解 KMeans 工程实现与 DBSCAN 在架构上的关键差异——有 `cluster_centers_`、有 `inertia_`、有 `predict()`。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.kmeans()` 生成球形 blob 聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `kmeans_data` |
| 训练封装 | `model_training/clustering/kmeans.py` | 构建并训练 `KMeans`，打印 `inertia_` 日志 |
| 流水线入口 | `pipelines/clustering/kmeans.py` | 组织数据拆分、标准化、训练与聚类可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签与真实标签的左右对照散点图（含质心标记） |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.clustering.kmeans
```

### 理解重点

- 这个命令串起当前 KMeans 分册中最核心的工程流程。
- 依次完成：数据复制 -> 剥离 `true_label` -> 全量标准化 -> KMeans `fit()`（分配-更新交替迭代）-> `inertia_` 日志 -> 对照散点图（含质心标记）。
- 对大多数读者来说，`pipelines/clustering/kmeans.py` 是理解工程实现的最佳起点——代码量少、流程清晰。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与对照标签
    data = kmeans_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    # 2. 全量标准化——无切分（无监督聚类不需要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. KMeans 训练——fit() 执行分配-更新交替迭代
    model = train_model(X_scaled)

    # 4. 单一可视化（左右对照散点图 + 质心标记）
    plot_clusters(
        X_scaled,
        labels_pred=model.labels_,
        labels_true=y_true,
        centers=model.cluster_centers_,
        title="KMeans 聚类分布",
        dataset_name=DATASET,
        model_name=MODEL,
    )
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的分配-更新迭代在 `KMeans.fit()` 中。
- 数据流是单向的：数据 -> 标准化 -> KMeans 迭代优化 -> `labels_` + `cluster_centers_` + `inertia_` -> 对照散点图（含质心标记）。
- 与分类流水线的核心差异：
  - **无 `train_test_split`**——无监督聚类不划分训练/测试集
  - **无 `predict()` 调用**——流水线直接使用 `model.labels_`（虽然 KMeans 支持 `predict()`，但教学流水线不演示）
  - **无 `predict_proba`**——KMeans 不产生概率
  - **单一可视化**（`plot_clusters`）而非四类（混淆矩阵+ROC+决策边界+学习曲线）
- 与 DBSCAN 流水线的差异：
  - `plot_clusters` 传入了 `centers=model.cluster_centers_`——DBSCAN 不传此参数
  - 训练日志打印 `inertia_`——DBSCAN 打印 `n_clusters` 和 `n_noise`

## 3. 训练模块负责什么

`model_training/clustering/kmeans.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)` 实例
2. 调用 `model.fit(X_train)`——分配-更新交替迭代（仅传特征，不传标签）
3. 打印 `n_clusters` 和 `inertia_` 日志
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的全量特征矩阵，传入 `KMeans.fit()` | `X_scaled` |
| `n_clusters` | `int` | 预设簇数 $K$。默认 `4`，与 `make_blobs(centers=4)` 一致 | `3`、`4`、`5` |
| `init` | `str` | 质心初始化策略。`'k-means++'` 加权随机采样 | `'k-means++'`、`'random'` |
| `n_init` | `int` | 不同初始质心下独立运行的次数，返回 `inertia_` 最小的结果。默认 `10` | `1`、`10`、`20` |
| `max_iter` | `int` | 单次运行的最大迭代次数。默认 `300` | `100`、`300` |
| `random_state` | `int` | 随机种子，保证可复现。默认 `42` | `42` |
| 返回值 | `KMeans` | 已完成 `fit()` 的模型对象，含 `cluster_centers_`、`labels_`、`inertia_` | — |

### 理解重点

- KMeans 的 `fit()` 是无监督的——不接收标签参数。这与分类分册中所有 `train_model` 都有 `y_train` 参数形成鲜明对比。
- 训练日志中的 `inertia_` 是 KMeans 独有的统计输出——它直接反映聚类紧密度，DBSCAN 没有对应物。
- `@print_func_info` 和 `@timeit` 装饰器提供函数标题和耗时——增强了教学型仓库的可读性。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 聚类对照图 | `plot_clusters(...)` | `X_scaled`、`labels_pred`（`model.labels_`）、`labels_true`（`y_true`）、`centers`（`model.cluster_centers_`） | 左右对照散点图（PNG），含红色 `X` 质心标记 |

### 理解重点

- `plot_clusters(...)` 是当前 KMeans 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- `centers` 参数是 KMeans 调用的特有参数——红色 `X` 标记直观展示每个簇的中心位置。DBSCAN 调用时不传此参数。
- 左右对照布局：左侧为 KMeans 聚类结果（含质心标记），右侧为真实标签——这种设计在教学上非常直观。
- 不涉及 PCA 降维——原始数据本身就是二维的，可以直接用作散点图坐标。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `kmeans_data` | `data_generation/clustering.py` | `pipelines/clustering/kmeans.py` |
| `y_true` | `data["true_label"]` 提取 | `plot_clusters`（仅对照用） |
| `X_scaled` | `StandardScaler` | `train_model`、`plot_clusters` |
| `model`（含 `labels_`、`cluster_centers_`、`inertia_`） | `train_model(...)` | `plot_clusters`、终端日志 |
| 图片产物 | `plot_clusters(...)` | `outputs/kmeans/` 目录 |

### 理解重点

- 数据依赖关系极为简洁——只有 5 个节点，单向流动无循环依赖。
- 比分类流水线少了 `train_test_split`、`predict`、`predict_proba`、PCA、ROC 评估、学习曲线等 6+ 个节点。
- `cluster_centers_` 的流向是 KMeans 数据流独有的——它从 `train_model` 产出，流入 `plot_clusters` 作为红色 `X` 标记。
- `y_true` 的流向是单向的——从数据到可视化，不经过模型训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `KMeans 聚类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`n_clusters`、`inertia_`（4 位小数） | 查看迭代优化耗时和聚类紧密度 |
| 聚类对照图 | `outputs/kmeans/cluster_plot.png` | 左右对照：KMeans 聚类结果（含红色 `X` 质心） vs 真实 4 簇标签 |

### 理解重点

- 输出比分类分册少得多——只有 2 类（日志 + 1 张图），而非 5 类（日志 + 4 张图）。
- `inertia_` 是 KMeans 独有的日志输出——它在 DBSCAN 的训练日志中不存在（DBSCAN 打印 `n_clusters` 和 `n_noise`）。
- 聚类对照图（含质心 `X` 标记）是最核心的教学产出——它直接展示了中心式聚类的效果和与真实结构的吻合度。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/clustering/kmeans.py` — 入口，代码量少，流程清晰
2. 再看 `model_training/clustering/kmeans.py` — 训练封装，理解无监督 `fit()` 和 `inertia_` 日志输出
3. 再看 `result_visualization/cluster_plot.py` — 聚类对照散点图绘制逻辑（含 `centers` 参数处理）
4. 最后回到 `data_generation/clustering.py` — 理解 `make_blobs` 球形高斯簇数据生成

### 理解重点

- 从入口看整体流程，再下钻到训练和可视化细节，阅读成本最低。
- KMeans 的调用链比分类分册短得多——这本身就是无监督聚类简洁性的体现。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/blog/articles/1780736373148.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的分配-更新迭代在 `KMeans.fit()` 中。
2. 期待当前流水线有 `train_test_split`——无监督聚类不需要。
3. 忽略 `inertia_` 的日志输出——它是理解聚类紧密度的最直接依据。
4. 把 `true_label` 当成参与训练的数据流——它的流向是"数据 -> 可视化"，从未进入模型。
5. 忘记 `centers` 参数是 KMeans 调用 `plot_clusters` 的特有参数——DBSCAN 不传此参数。

## 小结

- 当前 KMeans 工程实现采用极简的模块分层：数据生成 -> 训练封装（无监督）-> 流水线编排 -> 单一可视化（对照散点图 + 质心标记）。
- `run()` 负责串联，`train_model(...)` 负责交替迭代优化（仅 `fit(X)`），`plot_clusters(...)` 负责视觉对照（含 `centers` 参数）。
- KMeans 在工程上最不同于 DBSCAN 的地方：有 `cluster_centers_`（传入 `plot_clusters` 的 `centers`）、有 `inertia_`（训练日志输出）、有 `predict()`（虽然流水线未演示）、无噪声点概念。
- KMeans 在工程上最不同于分类算法的地方：无切分、无监督 `fit()`、单一可视化（对照散点图）——这是由无监督聚类本质决定的。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KMeans 实现。
2. 给出继续深入阅读 K 均值聚类与相关数据工具的可靠入口。

## 自检题

1. 为什么 KMeans 流水线没有 `train_test_split`？无监督聚类为什么不需要训练/测试切分？
2. 为什么 `n_clusters` 必须在训练前预设？如果把 `n_clusters=4` 改成 `3` 或 `5`，聚类结果会发生什么变化？
3. `k-means++` 初始化与纯随机初始化有什么本质区别？为什么 `k-means++` 能显著减少不良局部最优的风险？
4. 为什么 `inertia_` 随 $K$ 增大单调递减？为什么不能直接用 `inertia_` 选择最优 $K$？肘部法则的原理是什么？
5. 为什么 KMeans 在 `make_blobs` 球形高斯簇数据上表现极好，但在 `make_moons` 弯月数据上会失败？Voronoi 划分与数据形态不匹配的后果是什么？
6. 为什么标准化对 KMeans 是硬性要求？如果去掉标准化，欧氏距离计算会被哪个特征主导？
7. 为什么 KMeans 的评估不包含混淆矩阵、ROC 曲线和准确率？聚类评估与分类评估的根本差异是什么？

## 练习方向

### 1. 改变 `n_clusters`

- 把 `n_clusters=4` 改成 `2`、`3`、`4`、`5`、`8`
- 观察变化：
  - $K=2$——哪些真实簇被合并了？Voronoi 边界如何错误地穿过真实簇？
  - $K=5$——哪个真实簇被错误分裂了？新增的质心落在什么位置？
  - $K=8$——簇过度分裂的视觉特征是什么？`inertia_` 是否显著下降？
  - 聚类对照图——左右两侧的视觉差异是否随 $K$ 偏离真实值而急剧增大？
- 核心理解：$K$ 是 KMeans 最重要的参数——选错 $K$ 意味着预设的分组方式与数据真实结构不匹配

### 2. 改变 `init` 初始化策略

- 把 `init='k-means++'` 改成 `init='random'`
- 同时可以去掉 `random_state` 固定，多次运行观察结果的波动
- 观察变化：
  - `inertia_`——`random` 初始化是否经常得到更高的 `inertia_`（更差的局部最优）？
  - `n_iter_`——`random` 初始化是否需要更多迭代才能收敛？
  - 质心位置——是否有时出现两个质心挤在同一真实簇内的情况？
- 核心理解：`k-means++` 是 KMeans 从"频繁得到差结果"到"实践中稳定可靠"的关键改进

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察聚类结果的变化——当 `x1` 和 `x2` 的数值范围不同时，聚类是否被尺度更大的特征主导？
- 对比：原始数据坐标与标准化数据坐标的实际数值范围差异
- 体会：标准化后各特征平等贡献于距离计算——质心的位置和簇的边界在几何上才有意义

### 4. 改变 `cluster_std`

- 修改 `make_blobs(cluster_std=...)` 的 `cluster_std` 参数（`0.3`、`0.8`、`1.5`、`2.5`）
- 观察变化：
  - 低标准差（`0.3`）——簇极度紧凑，聚类几乎完美，`inertia_` 极小
  - 高标准差（`2.5`）——簇间边界模糊，部分点跨越 Voronoi 边界被错误分配
  - `inertia_` 随 `cluster_std` 增大而增大的趋势——簇越松散，紧密度越差
- 核心理解：KMeans 假设簇内方差相近——当 `cluster_std` 过大导致簇间重叠时，Voronoi 边界不再准确

### 5. 对比 DBSCAN

- 用 DBSCAN（`eps=0.3, min_samples=5`）在同一 `make_blobs` 数据上聚类，观察密度聚类在球形数据上的表现
- 用 KMeans（`n_clusters=2`）在 DBSCAN 的 `make_moons` 弯月数据上聚类，观察 Voronoi 直线边界如何沿月牙弧形错误切分
- 对比变化：
  - KMeans 在 `make_blobs` 上更简洁高效——这是它的理想数据
  - DBSCAN 在 `make_moons` 上能沿弯月密度扩展——这是 KMeans 做不到的
  - 两种算法的输出属性差异——KMeans 有 `cluster_centers_` + `inertia_`，DBSCAN 有噪声标签 $-1$
- 核心理解：聚类算法没有万能方案——算法选择必须匹配数据的结构特征。KMeans 和 DBSCAN 不是"谁更强"，而是分别适合形状截然不同的数据

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`KMeans` | 完整构造器参数（`n_clusters`、`init`、`n_init`、`max_iter`、`tol`、`algorithm`、`random_state`）、属性（`cluster_centers_`、`labels_`、`inertia_`、`n_iter_`）与方法（`fit`、`predict`、`fit_predict`、`fit_transform`、`transform`）说明 |
| 2 | scikit-learn 官方文档：`make_blobs` | 各向同性高斯簇数据生成器的 `n_samples`、`centers`、`cluster_std`、`n_features`、`shuffle`、`random_state` 等参数说明 |
| 3 | scikit-learn 用户指南：Clustering -> K-means | KMeans 算法原理、`k-means++` 初始化、肘部法则选 $K$、不同数据形态上的局限性与与其他聚类算法的使用场景对比 |
| 4 | Arthur, D. and Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA 2007. | k-means++ 原始论文——加权随机采样初始化策略的理论分析、近似比证明（$\Theta(\log K)$ 竞争比）和实验验证 |

- scikit-learn `KMeans`：https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- scikit-learn `make_blobs`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- scikit-learn 用户指南 Clustering：https://scikit-learn.org/stable/modules/clustering.html#k-means

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 KMeans 分册的核心内容：
  - 无监督聚类不需要训练/测试切分——`fit()` 在全量数据上执行，`labels_` 直接输出
  - $K$ 必须预设——这是 KMeans 最刚性的约束，选错 $K$ 意味着强制拆分或合并真实簇
  - `k-means++` 通过加权随机采样使初始质心分散——显著降低收敛到不良局部最优的概率
  - `inertia_` 随 $K$ 单调递减——不能直接用于选 $K$，需配合肘部法则
  - KMeans 偏好球形簇——Voronoi 划分的直线边界在弯月等非凸结构上必然出错
  - 标准化对基于欧氏距离的 KMeans 是硬性要求——距离计算不能被特征量纲绑架
  - KMeans 的 `fit()` 不接收标签——与分类分册的 `fit(X, y)` 有本质区别
  - KMeans 有 `predict()`——新样本只需找最近质心，DBSCAN 没有这个能力
  - `cluster_centers_` 是 KMeans 区别于 DBSCAN 的标志性属性——中心式聚类有显式质心
  - 聚类评估不同于分类评估——没有 accuracy/混淆矩阵/ROC，依赖散点图视觉对照和 `inertia_` 定量诊断
  - KMeans 与 DBSCAN 不是"谁更好"的关系——球形数据选 KMeans，弯曲不规则数据选 DBSCAN
