---
title: DBSCAN 密度聚类
date: 2026-05-01
category: 机器学习/聚类
tags:
  - Scikit-learn
description: DBSCAN密度聚类的数学原理、参数调优与工程实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581728739.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 DBSCAN 如何用 $\epsilon$ 邻域和密度关系定义簇——而非像 KMeans 那样依赖质心。
2. 理解核心点、边界点、噪声点的数学定义及其与 `eps` 和 `min_samples` 的关系。
3. 理解密度直达、密度可达、密度相连三种关系如何将散点组织成簇。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| $\epsilon$ 邻域 $N_{\epsilon}(\mathbf{x})$ | 基础定义 | 以 $\mathbf{x}$ 为中心、$\epsilon$ 为半径的超球体内所有点 |
| 核心点 | 点类型 | $\vert N_{\epsilon}(\mathbf{x}) \vert \geq \text{MinPts}$——邻域内点数达到阈值，可以向外扩展簇 |
| 边界点 | 点类型 | 自身非核心点，但落在某核心点的 $\epsilon$ 邻域内 |
| 噪声点 | 点类型 | 既非核心点也不属于任何簇——`labels_ == -1` |
| 密度直达 | 关系 | 核心点向其 $\epsilon$ 邻域内任意点的单向一步关系 |
| 密度可达 | 关系 | 通过有限步密度直达串联而成的传递关系 |
| 密度相连 | 关系 | 两点通过同一核心点桥接——这是簇的连通性基础 |

## 1. $\epsilon$ 邻域与点类型

给定数据集 $D = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$、邻域半径 $\epsilon > 0$ 和最小邻域点数 $\text{MinPts}$。

### $\epsilon$ 邻域

点 $\mathbf{x}$ 的 $\epsilon$ 邻域定义为：

$$
N_{\epsilon}(\mathbf{x}) = \{\mathbf{x}' \in D \mid d(\mathbf{x}, \mathbf{x}') \leq \epsilon\}
$$

其中 $d(\cdot, \cdot)$ 是距离度量——当前源码默认使用欧氏距离 $d(\mathbf{x}, \mathbf{x}') = \|\mathbf{x} - \mathbf{x}'\|_2$。

### 核心点（Core Point）

若点 $\mathbf{x}$ 的 $\epsilon$ 邻域内包含至少 $\text{MinPts}$ 个样本（含自身），则 $\mathbf{x}$ 为核心点：

$$
|N_{\epsilon}(\mathbf{x})| \geq \text{MinPts}
$$

### 边界点（Border Point）

点 $\mathbf{x}$ 不是核心点（$|N_{\epsilon}(\mathbf{x})| < \text{MinPts}$），但位于某个核心点的 $\epsilon$ 邻域内。

### 噪声点（Noise Point）

点 $\mathbf{x}$ 既不是核心点，也不属于任何核心点扩展出的簇。

### 理解重点

- `eps` 控制"多近算邻居"——$\epsilon$ 越大越宽松，越小越严格。
- `min_samples` 控制"多密才算核心"——$\text{MinPts}$ 越大，成为核心点的门槛越高。
- 核心点是簇扩展的"种子"——只有核心点能向外扩展，边界点只能被包含，噪声点被排除。

## 2. 三种密度关系

### 密度直达（Directly Density-Reachable）

$\mathbf{x}'$ 从 $\mathbf{x}$ 密度直达，当且仅当：

1. $\mathbf{x}$ 是核心点
2. $\mathbf{x}' \in N_{\epsilon}(\mathbf{x})$

密度直达**不对称**——如果 $\mathbf{x}'$ 是边界点（非核心点），则 $\mathbf{x}$ 不能从 $\mathbf{x}'$ 密度直达。

### 密度可达（Density-Reachable）

$\mathbf{x}_n$ 从 $\mathbf{x}_1$ 密度可达，当存在一条点链 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n$，使得对每个 $i$，$\mathbf{x}_{i+1}$ 从 $\mathbf{x}_i$ 密度直达。

密度可达**不对称**——边界点可以被核心点密度可达，但反过来不成立。

### 密度相连（Density-Connected）

$\mathbf{x}$ 和 $\mathbf{x}'$ 密度相连，当存在点 $\mathbf{o}$，使得 $\mathbf{x}$ 和 $\mathbf{x}'$ 都从 $\mathbf{o}$ 密度可达。

密度相连是**对称的**——这是簇定义的连通性基础。

### 理解重点

- 密度直达是"一步扩展"（微观），密度可达是"沿链扩展"（中观），密度相连是"桥接扩展"（宏观）。
- 只有密度相连关系是对称的——这正是 DBSCAN 能把点归入同一个簇的数学保证。

## 3. 簇的数学定义

基于以上关系，DBSCAN 定义的簇 $C$ 满足两个性质：

1. **最大性（Maximality）**：若 $\mathbf{x} \in C$ 且 $\mathbf{x}'$ 从 $\mathbf{x}$ 密度可达，则 $\mathbf{x}' \in C$。
2. **连通性（Connectivity）**：$C$ 中任意两点都是密度相连的。

不属于任何簇的点被标记为噪声（标签 $-1$）。

### 理解重点

- 最大性保证簇"收齐"所有能连通到的点——不会遗漏。
- 连通性保证簇内部的点在密度上是连通的——不会错误合并。
- 噪声点不是算法失败——它是 DBSCAN 设计的固有输出，对应数据中密度不足以形成簇的离群点。

## 4. 参数 `eps` 与 `min_samples`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `eps` | `float` | $\epsilon$ 邻域半径。$\epsilon \uparrow$ → 更多点被纳入邻域，大簇倾向、噪声点减少；$\epsilon \downarrow$ → 邻居门槛变严，小簇倾向、噪声点增多。默认 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。值越大，成为核心点的门槛越高，簇更保守、噪声可能更多。默认 `5` | `3`、`5`、`10`、`20` |

### 理解重点

- `eps` 和 `min_samples` 是联动参数——不能孤立调参。增大 `eps` 同时可能需要增大 `min_samples` 以避免过度合并。
- 对于二维数据，`min_samples` 的经验值通常是 $2 \times d$ 到 $2 \times d + 1$（$d$ 为特征维度）——当前 `min_samples=5` 对二维数据是合理起点。
- 当前 `eps=0.3` 是针对标准化后双月牙数据的选择——两月牙内侧最小距离约 0.5，0.3 的 $\epsilon$ 小于此间距，避免两月牙被错误连成一个簇。

## 5. 距离度量 `metric`

### 参数速览

适用 API：`DBSCAN(metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `metric` | `str` 或 `callable` | 距离度量方式。`'euclidean'`（默认）使用欧氏距离 $d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum (x_j - z_j)^2}$；`'manhattan'` 使用曼哈顿距离 $d(\mathbf{x}, \mathbf{z}) = \sum \vert x_j - z_j \vert$；`'cosine'` 使用余弦距离 | `'euclidean'`、`'manhattan'`、`'cosine'` |

### 理解重点

- 距离度量的选择直接影响 $\epsilon$ 邻域的形状——欧氏距离产生超球邻域，曼哈顿距离产生超菱面邻域。
- 当前源码使用默认的 `'euclidean'`，与标准化后的二维特征匹配。
- 不同度量下相同的 `eps` 值对应不同的实际邻域范围——切换度量时需重新调整 `eps`。

## 6. 标准化对 DBSCAN 的数学必要性

`eps` 是一个在特征空间中定义邻域半径的绝对数值。如果特征 $x_1$ 取值在 $[-2, 2]$ 而 $x_2$ 取值在 $[-100, 100]$，则：

- $\epsilon = 0.3$ 对 $x_1$ 来说覆盖了其取值范围的 7.5%
- 但同样的 $\epsilon = 0.3$ 对 $x_2$ 来说仅覆盖了其取值范围的 0.15%

这意味着 $\epsilon$ 邻域在不同维度上的实际含义不同——距离计算被量纲绑架。

### 理解重点

- 标准化后每个特征均值为 0、方差为 1，$\epsilon$ 在所有维度上的意义一致。
- 对于 DBSCAN 而言，标准化不是可选的优化手段——它是 `eps` 参数几何意义正确的前提。
- 这与 SVC（RBF 核距离敏感）的逻辑一致——任何基于距离度量的方法都需要标准化。

## 7. 为什么适合双月牙数据

`make_moons` 生成的双月牙数据具有以下数学特征：

- 两个月牙内部的点密度较高且均匀——满足核心点的判定条件
- 两个月牙之间的最小间距（约 0.5 标准化单位）大于 `eps=0.3`——密度扩展不会跨月牙跳跃
- 月牙内部沿弧形方向密度连通——单个月牙内的任意两点可以通过密度可达/密度相连关系归入同一簇

### 理解重点

- DBSCAN 的密度扩展天然适合月牙的弯曲形状——不需要任何全局形状假设。
- KMeans 依赖到中心的欧氏距离划分，会将弯月沿中心连线切分成两个半球形区域——这是算法本质差异。

## 8. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| $\epsilon$ 邻域半径 | $\epsilon$ | `eps=0.3` |
| 最小邻域点数 | $\text{MinPts}$ | `min_samples=5` |
| 距离度量 | $d(\mathbf{x}, \mathbf{x}')$ | `metric='euclidean'` |
| 核心点判定 | $\vert N_{\epsilon}(\mathbf{x}) \vert \geq \text{MinPts}$ | DBSCAN 算法内部逻辑 |
| 密度直达 | $\mathbf{x}' \in N_{\epsilon}(\mathbf{x})$，$\mathbf{x}$ 为核心点 | DBSCAN 算法扩展步骤 |
| 密度可达链 | $\mathbf{x}_1 \to \mathbf{x}_2 \to \dots \to \mathbf{x}_n$ | DBSCAN 的 BFS/DFS 扩展 |
| 簇标签 | $\{0, 1, \dots, k-1\}$ | `model.labels_` |
| 噪声标签 | $-1$ | `labels_ == -1` |
| 簇数量 | $k$ | `n_clusters = len(set(labels_)) - (1 if -1 in labels_ else 0)` |
| 噪声点数量 | — | `n_noise = (labels_ == -1).sum()` |
| 核心点索引 | — | `model.core_sample_indices_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 把 DBSCAN 当成中心式聚类——它没有簇中心（无 `cluster_centers_`），簇由密度连通关系定义。
2. 孤立调 `eps` 而忽略 `min_samples`——两者联动，增大一个时通常需调整另一个。
3. 看到噪声点（`labels_ == -1`）就认为模型失败——噪声识别是 DBSCAN 的核心设计，不是 bug。
4. 不标准化数据——`eps` 是绝对数值，在未标准化的数据上其几何意义被量纲扭曲。
5. 期望 DBSCAN 能像 KMeans 一样预测新点的簇归属——sklearn 的 DBSCAN 没有 `predict()` 方法，只能对训练数据做 `fit_predict`。

## 小结

- DBSCAN 的数学核心链：$\epsilon$ 邻域 $\to$ 核心点判定 $\to$ 密度直达/可达/相连 $\to$ 最大性 + 连通性定义簇 $\to$ 噪声点为 $-1$。
- `eps` 和 `min_samples` 联合决定点类型的划分和簇的形态——这是 DBSCAN 仅有的两个核心超参数。
- 当前源码 `DBSCAN(eps=0.3, min_samples=5, metric='euclidean')` 是二维标准化双月牙数据的合理配置——`eps` 小于月牙间距，`min_samples` 匹配二维特征的经验建议。

# 数据构成

## 本章目标

1. 明确本仓库 DBSCAN 数据来自 `make_moons(...)` 构造的双月牙聚类数据。
2. 明确特征列与 `true_label` 在当前流水线中的角色差异——这是无监督聚类，`true_label` 仅用于结果对照。
3. 明确标准化发生在什么位置，以及为什么它对基于距离的 `eps` 邻域判定至关重要。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClusteringData.dbscan()` | 方法 | 生成 DBSCAN 使用的二维双月牙聚类数据 |
| `make_moons(...)` | 函数 | scikit-learn 提供的双月牙数据生成器 |
| `dbscan_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `true_label` | 列名 | 真实簇标签——仅用于与预测结果视觉对照，不参与 `fit()` |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——`eps` 邻域判定的前置条件 |

## 1. 数据生成：`make_moons()`

当前 DBSCAN 数据来自 `ClusteringData.dbscan()`，底层调用 `sklearn.datasets.make_moons()`。

### 参数速览

适用函数：`make_moons(n_samples=400, noise=0.08, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。默认 400，两个月牙各约 200 个样本 | `400`、`500` |
| `noise` | `float` | 添加到 x 和 y 坐标上的高斯噪声标准差。`0` 表示完全平滑的月牙弧线，`0.08` 使样本轻微偏离理想弧线 | `0.08`、`0.05`、`0.12` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| `shuffle` | `bool` | 是否打乱样本顺序。默认 `True` | `True` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(400, 2)$，$y$ 取值 $\{0, 1\}$ | — |

### 示例代码

```python
X, y = make_moons(
    n_samples=400,
    noise=0.08,
    random_state=42,
)
data = DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_label": y})
```

### 理解重点

- 双月牙数据是展示 DBSCAN 优势的经典基准——两个月牙弧形弯曲、互不连通，无法用球形簇或线性边界有效分离。
- `noise=0.08` 在保持月牙弧形结构可辨识的前提下，增加了一定的局部密度波动——少量点可能落在两个月牙之间的间隙中。
- 只包含 $x_1$、$x_2$ 两个特征，非常适合二维散点图直观展示聚类效果。

## 2. 特征列与 `true_label` 的角色

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["true_label"])` |
| `y_true` | `ndarray` | 真实簇标签 $y_i \in \{0, 1\}$，**仅用于结果对照**，不参与 DBSCAN 的 `fit()` | `data["true_label"].values` |

### 示例代码

```python
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- 这是分类分册与聚类分册的核心差异——`true_label` 不传入 `model.fit()`。
- DBSCAN 是无监督算法：`fit(X)` 只接收特征矩阵，不接收标签。
- `true_label` 的唯一目的是在 `plot_clusters(...)` 中与 `model.labels_` 做左右对照——帮助读者判断算法是否恢复了真实的簇结构。

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

- DBSCAN 使用 `eps` 定义 $\epsilon$ 邻域半径——这是一个绝对数。如果特征量纲不同，同样的 `eps` 在不同维度上代表完全不同的邻域范围。
- 标准化后每个特征具有相同的尺度，`eps=0.3` 在所有维度上含义一致。
- 与分类分册不同，DBSCAN 流水线没有训练/测试切分——因为无监督聚类不需要验证集，标准化在全量数据上执行。

## 数据可视化

![原始数据散点图](https://img.yumeko.site/file/articles/ML/dbscan/data_raw_scatter.png)

![真实标签散点图](https://img.yumeko.site/file/articles/ML/dbscan/data_true_label_scatter.png)

![聚类分布图](https://img.yumeko.site/file/articles/ML/dbscan/data_cluster_distribution.png)

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/dbscan/data_correlation.png)

## 常见坑

1. 把 `true_label` 当成训练标签传入 `model.fit()`——DBSCAN 是无监督算法，不接受标签参数。
2. 误以为 `true_label` 和分类分册中的 `y_train` 有相同角色——一个是无监督对照，一个是有监督训练目标。
3. 忽略标准化——`eps` 是绝对数值，不标准化的数据会让邻域判定在不同维度上含义不一致。
4. 看到双月牙效果很好，就误以为 DBSCAN 在所有密度分布上都同样稳定。

## 小结

- 当前 DBSCAN 数据来自 `make_moons(n_samples=400, noise=0.08)`：2 个连续特征、双月牙弧形结构。
- 数据流为：`make_moons` → DataFrame（`x1`、`x2` + `true_label`）→ 剥离 `true_label` → 全量标准化。
- `true_label` 仅用于结果对照——这是无监督聚类与有监督分类在数据处理上的最根本差异。

# 思路与直觉

## 本章目标

1. 用直观方式理解 DBSCAN 的密度聚类思路——从"哪里密集就从哪里向外扩展"而非"先定中心再划分"。
2. 理解为什么 DBSCAN 在双月牙数据上能恢复真实簇结构，而中心式聚类方法会失败。
3. 通过对比 KMeans，建立 DBSCAN 在聚类算法图中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 密度聚类 | 核心直觉 | 不预设簇形状，而是把"局部密集且连通"的区域识别为簇 |
| 核心点扩展 | 生长机制 | 只有周围足够密集的点才有资格向外"生长"出簇 |
| 噪声识别 | 天然输出 | 不落入任何密集区域的点自动被标为噪声——无需额外后处理 |
| 双月牙数据 | 教学示例 | 最能体现 DBSCAN 相对中心式聚类优势的非球形数据 |
| KMeans | 对比算法 | 擅长球形簇，在弯曲结构中会将月牙沿 Voronoi 边界切开 |

## 1. 为什么需要密度聚类

大多数聚类算法（如 KMeans）的核心思路是：先假设簇是围绕某个中心的球形区域，然后把每个点分配给最近的中心。

但当簇的形状弯曲、拉长或不规则时，"距离最近中心"的划分方式会切出违反直觉的结果——例如把一条弯月从中间斩断。

DBSCAN 换了另一种思路：

- **不找中心**——找高密度区域
- **不从全局划分**——从局部连通出发
- **不预设簇数**——让密度结构自己决定有多少个簇

### 理解重点

- 密度聚类的出发点是"哪里人多就往哪里聚"，而不是"谁离我近我跟谁"。
- 这种直觉在物理上很自然——银河系、城市群、社交网络社区也都是密度驱动的聚类。
- 因为不依赖中心假设，DBSCAN 天然适合任意形状的簇。

## 2. 为什么 DBSCAN 在双月牙数据上表现好

当前数据是两个弯曲的月牙形簇，内侧相对。这种结构有四个特征：

- 月牙**内部**密度较高且均匀——容易形成核心点
- 月牙**之间**有明显的低密度间隙——密度扩展不会跨月牙跳跃
- 月牙形状**弯曲**——无法用球形区域描述
- 维度只有 **2**——便于直观观察

### 理解重点

- 如果 `eps=0.3` 小于两个月牙之间的最小间距（标准化后约 0.5），密度扩展会自然地止步于月牙边缘——这正是正确的聚类结果。
- KMeans 会沿两个球形簇的 Voronoi 边界将弯月切分成两个"半球"——这不是 KMeans 的错误，而是它的设计假设（球形簇）与双月牙数据不匹配的必然结果。
- DBSCAN 不需要"知道"数据是月牙形的——它只关心局部密度和连通性，形状信息是被密度结构间接表达的。

## 3. 用"从核心点向外生长"理解算法

DBSCAN 的过程可以直观地想象为：

1. **找种子**：遍历所有点，找到满足"周围足够密集"的核心点
2. **向外生**：从核心点出发，把 $\epsilon$ 邻域内所有点拉入当前簇
3. **继续扩**：如果新拉入的点中还有核心点，就继续向外扩散
4. **止于边界**：当扩展到边界点（邻域不够密）时停止——边界点本身不能继续扩展
5. **收尾**：遍历结束后，没有被任何核心点触达的点标为噪声

### 理解重点

- DBSCAN 更像"菌落生长"——从密集中心向外扩散，直到遇到密度不达标的边缘。
- KMeans 更像"行政区划"——先划定中心点，再按最近原则分配。
- 两种直觉对应了两类截然不同的聚类哲学。

## 4. 噪声是设计，不是失败

DBSCAN 会把不满足密度连通条件的点标为噪声（`labels_ == -1`）。

### 理解重点

- 在 KMeans 中，每个点都会被强行分配到一个簇——即使它离所有中心都很远。而 DBSCAN 允许点"不属于任何簇"。
- 噪声识别是 DBSCAN 的天然输出——这在异常检测、离群点分析等场景中本身就是有价值的信息。
- 如果噪声点过多（如超过 20%），通常说明 `eps` 太小或 `min_samples` 太大——参数需要调整，但机制本身是合理的。

## 5. 与 KMeans 的直觉对比

| 维度 | DBSCAN | KMeans |
|---|---|---|
| 核心问题 | 哪些点是局部密集且彼此连通的？ | 每个点离哪个中心最近？ |
| 簇的形状假设 | 任意形状（仅需密度连通） | 球形/凸形（基于到中心的距离） |
| 簇数 | 由密度结构自动决定 | 必须预设 $k$ |
| 噪声处理 | 天然识别为标签 $-1$ | 强制分配——每个点都属于某个簇 |
| 对标准化的依赖 | 强——`eps` 是绝对数值 | 强——距离是划分依据 |
| 对新样本的扩展 | sklearn 不支持 `predict()` | 支持 `predict(X_new)` |

### 理解重点

- DBSCAN 和 KMeans 不是"谁更好"的关系——它们分别精通不同类型的数据。
- 球形数据选 KMeans，弯曲/不规则形状选 DBSCAN，密度差异极大的数据两者都可能不适用。
- 当前仓库选择 `make_moons` 作为 DBSCAN 的教学数据，正是为了展示这种差异。

## 可视化

![聚类结果图](https://img.yumeko.site/file/articles/ML/dbscan/cluster_plot.png)

## 常见坑

1. 因为当前双月牙示例效果很好，就认为 DBSCAN 适合所有聚类数据——它假设各簇密度相近。
2. 把噪声点理解成算法出错——噪声识别是 DBSCAN 的核心设计，不是 bug。
3. 只调 `eps` 而忽略 `min_samples`——两者是联动参数，单独调一个往往达不到预期效果。
4. 不区分"局部密度连通"与"全局几何距离接近"——两个点可能几何上很近，但被低密度区域阻隔，DBSCAN 不会将它们归入同一簇。

## 小结

- DBSCAN 的直觉核心是密度聚类：从高密度的核心点出发，沿密度连通关系逐步扩展出簇——不预设簇数和簇形状。
- 双月牙数据与这种直觉高度匹配——两个月牙内部密集、月牙之间有间隙、形状弯曲无规则。
- 噪声识别是 DBSCAN 区别于 KMeans 的关键特征——它允许点"不属于任何簇"，这在很多实际场景中是优势而非缺陷。
- 理解 DBSCAN 与 KMeans 的直觉差异，是理解"何时选 DBSCAN、何时选 KMeans"的前提。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `DBSCAN`。
2. 理解 `DBSCAN` 的核心构造器参数（`eps`、`min_samples`、`metric`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`labels_`、`core_sample_indices_`、以及衍生的簇数量和噪声点数量。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.cluster.DBSCAN` 模型，打印聚类统计日志 |
| `DBSCAN(...)` | 类 | scikit-learn 提供的密度聚类器——基于 $\epsilon$ 邻域和密度连通关系 |
| `model.fit(X_train)` | 方法 | 在训练数据上执行密度聚类——注意无监督：只传特征不传标签 |
| `model.labels_` | 属性 | 每个训练样本的簇分配结果，噪声点标记为 $-1$ |
| `model.core_sample_indices_` | 属性 | 核心点在训练数组中的索引位置 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，形状 $(400, 2)$，传入 `DBSCAN.fit()` | `X_scaled` |
| `eps` | `float` | $\epsilon$ 邻域半径。$\epsilon \uparrow$ → 更大邻域、更多核心点、簇数减少。默认 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。值越大，成为核心点的门槛越高。默认 `5` | `3`、`5`、`10` |
| `metric` | `str` | 距离度量方式。默认 `'euclidean'`（欧氏距离 $d = \sqrt{\sum (x_j - z_j)^2}$） | `'euclidean'`、`'manhattan'` |
| 返回值 | `DBSCAN` | 已完成 `fit()` 的模型对象，含 `labels_`、`core_sample_indices_` 等属性 | — |

### 示例代码

```python
from model_training.clustering.dbscan import train_model

model = train_model(X_scaled)
```

### 理解重点

- 当前入口只负责构建一个 `DBSCAN` 并 `fit`——没有参数网格搜索或多度量对比。
- 和监督学习分册的 `train_model` 不同，这里**没有 `y_train` 参数**——DBSCAN 是无监督算法。
- `train_model(...)` 是对 `sklearn.cluster.DBSCAN` 的薄封装——算法本体是 scikit-learn 的 C++ 实现。

## 2. `DBSCAN` 构造器参数

### 参数速览

适用 API：`DBSCAN(eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `eps` | `float` | $\epsilon$ 邻域半径。决定了"多近算邻居"。默认 `0.5`，当前源码显式设为 `0.3` | `0.2`、`0.3`、`0.5`、`1.0` |
| `min_samples` | `int` | 核心点判定阈值 $\text{MinPts}$。决定了"多密算高密度区域"。$d$ 维数据建议 $\geq d+1$，默认 `5` | `3`、`5`、`10`、`20` |
| `metric` | `str` 或 `callable` | 距离度量。`'euclidean'`（欧氏距离）、`'manhattan'`（曼哈顿距离）、`'cosine'`（余弦距离）等。默认 `'euclidean'` | `'euclidean'`、`'manhattan'`、`'cosine'` |
| `algorithm` | `str` | 最近邻搜索算法。`'auto'` 自动选择最优；`'ball_tree'` 球树；`'kd_tree'` KD 树；`'brute'` 暴力搜索。默认 `'auto'` | `'auto'`、`'ball_tree'`、`'kd_tree'`、`'brute'` |
| `leaf_size` | `int` | BallTree 或 KDTree 的叶子节点大小。对构建索引速度和查询速度有影响，不影响聚类结果。默认 `30` | `20`、`30`、`50` |
| `p` | `float` | Minkowski 距离的指数参数。`p=2` 等价于欧氏距离，`p=1` 等价于曼哈顿距离。仅当 `metric='minkowski'` 时生效。默认 `2` | `1`、`2` |
| `n_jobs` | `int` 或 `None` | 并行计算的作业数。`None` 表示 1，`-1` 表示使用所有 CPU。默认 `None` | `None`、`-1`、`4` |

### 示例代码

```python
model = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric="euclidean",
)
model.fit(X_train)
```

### 理解重点

- DBSCAN 的核心参数是 `eps` 和 `min_samples`——两者联合决定了点类型的划分（核心/边界/噪声）和最终的簇结构。
- `eps` 的默认值是 `0.5`，但当前源码显式设为 `0.3`——这是针对标准化后双月牙数据的定制选择（月牙间距约 0.5，`0.3 < 0.5` 避免跨月牙连接）。
- `min_samples=5` 对二维数据是一个合理起点——经验规则 $2d$ 到 $2d+1$，即二维下 4~5。
- `algorithm='auto'`（默认）会根据数据量和特征维度自动选择最近邻搜索方式——对当前 400 样本 2 维数据，通常选用 KD 树。
- 与分类模型（逻辑回归、SVC）的关键差异：DBSCAN 的 `fit()` 不接受标签参数 `y`——`fit(X)` 而非 `fit(X, y)`。

## 3. 训练完成后的关键属性与统计量

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 `(n_samples,)` | 簇分配标签 | 每个训练样本所属簇的编号 $\{0, 1, \dots, k-1\}$，噪声点为 $-1$ |
| `core_sample_indices_` | `ndarray`，形状 `(n_core_samples,)` | 核心点索引集 | 所有核心点在原始训练数组中的下标位置 |
| `components_` | `ndarray`，形状 `(n_core_samples, n_features)` | 核心点特征值 | 所有核心点的特征向量——仅在内存高效模式下可用 |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |

### 衍生统计量

| 统计量 | 计算方式 | 说明 |
|---|---|---|
| `n_clusters` | `len(set(labels_)) - (1 if -1 in labels_ else 0)` | 排除噪声后的簇数量——当前期望为 2（两个月牙） |
| `n_noise` | `(labels_ == -1).sum()` | 被标记为噪声的样本数量——反映数据中密度不足的点规模 |

### 示例代码

```python
labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"簇数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")
```

### 理解重点

- `labels_` 是 DBSCAN 最重要的输出——它不是单个预测值，而是对所有 400 个训练样本的簇分配。
- `-1` 是 DBSCAN 的独特标签——与其他分类器的 `classes_` 不同，它专门标记不满足密度连通条件的噪声点。
- DBSCAN **没有** `cluster_centers_`（KMeans 有）——因为密度聚类不依赖簇中心。
- DBSCAN **没有** `predict()` 方法——sklearn 的 DBSCAN 只能对训练数据本身做聚类，不能预测新样本。这是一个常见的工程限制。

## 4. 训练阶段的工程封装

除了 `DBSCAN(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察密度聚类执行时间 |
| `eps` / `min_samples` 日志 | 确认当前参数配置 |
| `簇数量` 日志 | 快速查看算法发现的簇数——与真实类别数对比 |
| `噪声点数量` 日志 | 观察被识别为离群点的样本规模 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出聚类统计量。
- `簇数量` 和 `噪声点数量` 是 DBSCAN 独有的日志输出——它们直接反映算法的聚类行为。
- 这一层封装把"构建模型""训练模型""打印统计"收在一个函数里，方便文档和流水线复用。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——DBSCAN 是无监督算法，不接受标签。
2. 误以为 DBSCAN 训练完成后能得到簇中心——它没有 `cluster_centers_` 属性。
3. 期望能用 `model.predict(X_new)` 预测新样本——sklearn 的 DBSCAN 不支持，需结合 `NearestNeighbors` 等后处理。
4. 只看 `labels_`，却忽略 `-1`（噪声）、`n_clusters` 和 `n_noise` 才是理解聚类行为的关键统计量。
5. 忘记当前 `X_train` 应该是标准化后的特征——`eps` 是绝对数值，未经标准化的数据会让邻域判定失真。

## 小结

- `train_model(...)` 是本仓库 DBSCAN 的核心训练入口，是对 `sklearn.cluster.DBSCAN` 的薄封装。
- `DBSCAN` 的核心参数是 `eps`（$\epsilon$ 邻域半径）和 `min_samples`（核心点阈值）——两者联合决定簇的形态和噪声规模。
- 训练完成后的关键属性：`labels_`（含噪声标签 $-1$）、`core_sample_indices_`（核心点索引）——通过它们推导 `n_clusters` 和 `n_noise`。
- DBSCAN 没有簇中心、没有 `predict()`、不接收 `y` 标签——这三个"没有"是它与分类模型和 KMeans 最核心的工程差异。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 DBSCAN 流水线从数据复制到聚类输出的完整步骤。
2. 理解 DBSCAN 无训练/测试切分、无 `predict()` 的工程特征——与分类分册有本质差异。
3. 理解 `fit()` 即聚类、`labels_` 即输出的流程特征。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `dbscan_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `StandardScaler` | 类 | 对全量特征做一致性标准化——`eps` 邻域判定的前置条件 |
| `train_model(...)` | 函数 | 调用 `DBSCAN.fit()` 执行密度聚类，返回模型对象 |
| `model.fit(X_scaled)` | 方法 | 在标准化特征上执行密度聚类——标签生成、簇分配一步到位 |
| `model.labels_` | 属性 | 聚类结果的唯一输出——每个样本被分配到簇编号或 $-1$（噪声） |

## 1. 流水线起点：复制数据并拆出特征与对照标签

### 示例代码

```python
data = dbscan_data.copy()
y_true = data["true_label"].values
X = data.drop(columns=["true_label"])
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `dbscan_data`。
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

- DBSCAN 流水线**没有**训练/测试切分——无监督聚类不需要验证集。
- `fit_transform` 直接在全量数据上计算统计量并变换——不存在测试集数据泄露的风险。
- 标准化是必须的——`eps=0.3` 作为绝对距离阈值，其几何意义依赖于各维度尺度一致。

## 3. 密度聚类：`fit()` 即训练 + 预测

### 参数速览

适用 API：`train_model(X_scaled)` → `model.fit(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征矩阵——DBSCAN 的唯一输入 | `X_scaled` |
| 无 `y` 参数 | — | DBSCAN 是无监督算法——`fit(X)` 不接受标签 | — |

### 示例代码

```python
model = train_model(X_scaled)
```

### 理解重点

- `DBSCAN.fit(X_scaled)` 内部流程：遍历所有点 → 对每个点计算 $\epsilon$ 邻域 → 判定核心/边界/噪声 → 沿密度可达关系 BFS/DFS 扩展簇 → 生成 `labels_`。
- 这**既是训练也是预测**——DBSCAN 没有分离的 `fit()` + `predict()` 两阶段。对于新样本，sklearn 的 DBSCAN 无法直接预测簇归属。
- 与分类模型比较：分类流程是 `fit(X, y)` → `predict(X_test)`，DBSCAN 流程是 `fit(X)` → `labels_`。

## 4. 获取聚类结果

### 参数速览

| 属性名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `labels_` | `ndarray`，形状 $(400,)$ | 每个样本的簇分配标签，$\{-1, 0, 1, \dots, k-1\}$ | `model.labels_` |
| `n_clusters` | `int` | 排除 $-1$ 后的簇数量 | 当前期望为 `2` |
| `n_noise` | `int` | 标签为 $-1$ 的噪声点数量 | 取决于 `noise=0.08` 下落入月牙间隙的样本数 |

### 示例代码

```python
labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
```

### 理解重点

- `model.labels_` 是 DBSCAN 的唯一输出——它就是这个聚类算法的"预测结果"。
- `-1` 标签是 DBSCAN 特定的噪声标记——与分类模型中的 `predict` 输出不同，噪声点不属于任何类别。
- 没有 `model.predict(X_test)`——这是 DBSCAN 在预测能力上的天然限制（sklearn 实现）。新样本的簇归属需通过其他方式推断（如最近邻搜索）。

## 5. 聚类结果可视化

### 参数速览

适用函数：`plot_clusters(X_scaled, labels_pred=model.labels_, labels_true=y_true, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 2)$ | 标准化后的全量特征，用于散点图的坐标 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 $(400,)$ | DBSCAN 的预测簇标签（含 $-1$ 噪声） | `model.labels_` |
| `labels_true` | `ndarray`，形状 $(400,)$ | 真实簇标签（仅用于视觉对照） | `y_true` |

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    title="DBSCAN 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `plot_clusters(...)` 是当前 DBSCAN 分册唯一的可视化函数——与分类分册的四类评估（混淆矩阵+ROC+决策边界+学习曲线）完全不同。
- 双侧对照布局：左侧显示 `labels_pred`（算法聚类结果），右侧显示 `labels_true`（真实簇标签）——帮助读者直观判断算法是否恢复了真实结构。
- `labels_pred` 中的噪声点（$-1$）通常以特殊颜色（如黑色或灰色）标记，便于识别。

## 常见坑

1. 期望当前流水线有训练/测试切分——无监督聚类不划分训练集和验证集。
2. 误以为 `model.predict(X_new)` 可用——sklearn 的 DBSCAN 不支持 `predict()`，`fit()` 即得到全部聚类结果。
3. 把 `true_label` 当成 `fit()` 的输入——它仅用于最终的可视化对照。
4. 把 DBSCAN 的训练流程理解成"先生成模型，再用于预测新数据"——它是直接对输入数据进行标记，没有分离的训练和推理阶段。

## 小结

- 当前 DBSCAN 流水线极为简洁：复制数据 → 剥离 `true_label` → 全量标准化 → `fit(X)` 密度聚类 → `labels_` 直接作为聚类输出 → 可视化对照。
- 与分类分册的核心差异：无切分（无 `train_test_split`）、无监督标签、无 `predict()`（`fit()` 即输出）、无概率输出、无混淆矩阵/ROC/学习曲线。
- 这种简洁性源于 DBSCAN 的算法特性——它是直接对数据点做密度连通分析，而非训练一个可泛化的判别函数。

# 评估与诊断

## 本章目标

1. 明确当前仓库 DBSCAN 实现的评估手段——聚类对照散点图 + 簇数量/噪声点数量日志。
2. 理解聚类评估与分类评估的本质差异——无标签无监督，评价的是簇结构恢复质量而非分类正确率。
3. 理解轮廓系数等常见聚类指标的原理，以及它们未在流水线中显式计算的原因。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_clusters(...)` | 函数 | 绘制预测簇标签与真实标签的左右对照散点图 |
| `n_clusters` | 统计量 | 算法实际发现的簇数量——与真实簇数（2）对比 |
| `n_noise` | 统计量 | 被标记为噪声的样本数——反映密度结构中的"灰区" |
| 簇标签对应关系 | 诊断 | 评估预测簇编号与真实标签之间是否直观对应 |
| 轮廓系数（Silhouette Score） | 指标 | 量化聚类紧密度与分离度——当前未在流水线中显式计算 |

## 1. 当前仓库的评估入口

DBSCAN 的评估与分类分册截然不同——没有混淆矩阵、没有 ROC 曲线、没有决策边界、没有学习曲线。评估依赖两种手段：

1. **聚类对照散点图** —— 左右并列显示算法聚类结果与真实标签，视觉上判断簇结构恢复质量
2. **聚类统计日志** —— 簇数量 `n_clusters` 和噪声点数量 `n_noise`

### 示例代码

```python
plot_clusters(
    X_scaled,
    labels_pred=model.labels_,
    labels_true=y_true,
    title="DBSCAN 聚类分布",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 聚类评估不能像分类那样用 `y_pred == y_test` 计算准确率——因为簇标签编号（0, 1）与真实标签编号不一定对应，且两者含义不同。
- 当前评估方法对教学场景来说足够直观——读者可以一眼看出密度扩展是否恢复了两个月牙结构。
- `n_clusters` 和 `n_noise` 这两个统计量是最简明的定量诊断——它们直接反映了 `eps`/`min_samples` 配置的合理性。

## 2. 聚类对照散点图能观察什么

### 参数速览

适用函数：`plot_clusters(X, labels_pred, labels_true, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `ndarray`，形状 `(n_samples, 2)` | 标准化后的全量特征矩阵 | `X_scaled` |
| `labels_pred` | `ndarray`，形状 `(n_samples,)` | 算法聚类结果——来自 `model.labels_`，含噪声标记 $-1$ | `model.labels_` |
| `labels_true` | `ndarray`，形状 `(n_samples,)` | 真实簇标签——来自 `make_moons` 生成时的 `y` | `y_true` |

### 理解重点

- 左侧图展示 DBSCAN 的聚类结果——用不同颜色标记不同簇，噪声点通常为特殊颜色（如黑色）。
- 右侧图展示真实簇标签——为读者提供"正确答案"的视觉基准。
- 对比重点：
  - 簇的形状是否与真实月牙一致
  - 两个月牙是否被正确识别为两个独立的簇
  - 噪声点（左侧）的数量与位置——是否合理（落在月牙间隙中）还是过多（参数不当）

## 3. 聚类统计日志能观察什么

### 参数速览

| 统计量 | 来源 | 理想值 | 异常信号 |
|---|---|---|---|
| `n_clusters` | `len(set(labels_)) - (1 if -1 in labels_ else 0)` | `2`（两个月牙各一簇） | `1` 表示两个簇被错误合并（`eps` 过大）；`3+` 表示过度分裂（`eps` 过小或 `min_samples` 过大） |
| `n_noise` | `(labels_ == -1).sum()` | 少量（通常 `< 5%`） | 过多（`> 20%`）表示 `eps` 过小或 `min_samples` 过大——大量正常点被误判为噪声 |

### 理解重点

- 这两个统计量构成了 DBSCAN 的最简单有效诊断——不需要任何额外计算。
- 如果 `n_clusters != 2` 且并非数据生成有问题，那就是参数配置不当。
- 过量噪声点通常意味着密度参数与数据的实际密度结构存在系统偏差。

## 4. 当前实现中尚未纳入但常见的聚类指标

| 指标 | 公式 / 含义 | 当前未使用的原因 |
|---|---|---|
| 轮廓系数（Silhouette Score） | $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$，其中 $a_i$ 为点到同簇其他点的平均距离（紧密度），$b_i$ 为点到最近邻簇的平均距离（分离度）。$s_i \in [-1, 1]$ | 当前侧重直观教学而非量化评估，且轮廓系数对噪声点（$-1$）需特殊处理 |
| 调整兰德指数（ARI） | 度量两个聚类分配之间的相似度（共识），扣除随机期望。ARI $\in [-1, 1]$，1 表示完全一致 | 需要真实标签——当前 `true_label` 确实存在，但流水线选择直接用视觉对照而非数值指标 |
| 同质性/完整性/ V-measure | 同质性 $h$：每个簇只包含单个类别的成员。完整性 $c$：每个类别的所有成员被分配到同一个簇。V-measure = $2hc/(h+c)$ | 同 ARI——当前侧重教学直观性 |
| Davies-Bouldin 指数 | 簇间相似度均值（越小越好）——衡量簇内紧密度与簇间分离度的比值 | 无需真实标签——但当前分册的教学重点是理解密度聚类机制，而非指标对比 |

### 理解重点

- 聚类评估体系与分类评估有本质区别——分类可以直接比对预测标签与真实标签，而聚类的"正确性"更多体现在簇结构的合理性上。
- 当前流水线不显式计算这些指标——文档可以提到它们是扩展方向，但不能写成"当前源码已在计算"。
- 对于教学型仓库，散点图视觉对照 + `n_clusters`/`n_noise` 日志已经能有效支撑 DBSCAN 的诊断需求。

## 评估图表

![聚类分布图](https://img.yumeko.site/file/articles/ML/dbscan/cluster_plot.png)

## 常见坑

1. 把分类的评估框架（混淆矩阵、ROC、accuracy）搬到聚类评估——两者评估哲学根本不同。
2. 看到 `n_clusters` 不等于真实簇数就认为模型失败——需结合 `eps`/`min_samples` 配置和数据噪声水平综合判断。
3. 把噪声点当成评估中的"错误"——噪声是 DBSCAN 的设计输出，不是误分类。
4. 忽略左右对照图中预测簇颜色与真实簇颜色的不对应——簇标签编号是任意的，0 和 1 可能互换。

## 小结

- 当前仓库对 DBSCAN 的评估简洁而直观：聚类对照散点图看簇形状恢复质量，`n_clusters` 和 `n_noise` 统计量看参数配置合理性。
- DBSCAN 没有混淆矩阵、没有 ROC、没有准确率——这些是监督分类的评估手段，不适用于无监督聚类。
- 噪声点不是评估中的扣分项——它是 DBSCAN 的核心输出，噪声过多或过少才需要关注参数配置。
- 聚类评估的核心问题是"算法是否恢复了数据的真实密度结构"——而非"预测标签是否与某个参考答案完全一致"。

# 工程实现

## 本章目标

1. 从工程角度看清 DBSCAN 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和聚类可视化分别负责什么。
3. 理解 DBSCAN 工程实现与分类算法在架构上的关键差异——无切分、无 `predict`、单一可视化。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/clustering.py` | `ClusteringData.dbscan()` 生成双月牙聚类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `dbscan_data` |
| 训练封装 | `model_training/clustering/dbscan.py` | 构建并训练 `DBSCAN`，打印聚类统计日志 |
| 流水线入口 | `pipelines/clustering/dbscan.py` | 组织数据拆分、标准化、训练与聚类可视化 |
| 聚类结果可视化 | `result_visualization/cluster_plot.py` | 绘制预测簇标签与真实标签的左右对照散点图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.clustering.dbscan
```

### 理解重点

- 这个命令串起当前 DBSCAN 分册中最核心的工程流程。
- 依次完成：数据复制 → 剥离 `true_label` → 全量标准化 → DBSCAN `fit()`（密度扩展）→ 聚类统计 → 对照散点图。
- 对大多数读者来说，`pipelines/clustering/dbscan.py` 是理解工程实现的最佳起点——代码量少、流程清晰。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与对照标签
    data = dbscan_data.copy()
    y_true = data["true_label"].values
    X = data.drop(columns=["true_label"])

    # 2. 全量标准化——无切分（无监督聚类不需要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 密度聚类——fit() 即得到全部结果（无 predict）
    model = train_model(X_scaled)

    # 4. 单一可视化（左右对照散点图）
    plot_clusters(
        X_scaled,
        labels_pred=model.labels_,
        labels_true=y_true,
        title="DBSCAN 聚类分布",
        dataset_name=DATASET,
        model_name=MODEL,
    )
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的密度扩展在 `DBSCAN.fit()` 中。
- 数据流是单向的：数据 → 标准化 → 密度扩展 → `labels_` → 对照散点图。
- 与分类流水线的核心差异：
  - **无 `train_test_split`**——无监督聚类不划分训练/测试集
  - **无 `predict()` 调用**——`fit()` 即输出 `labels_`
  - **无 `predict_proba`**——DBSCAN 不产生概率
  - **单一可视化**（`plot_clusters`）而非四类（混淆矩阵+ROC+决策边界+学习曲线）

## 3. 训练模块负责什么

`model_training/clustering/dbscan.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `DBSCAN(eps=0.3, min_samples=5, metric='euclidean')` 实例
2. 调用 `model.fit(X_train)`——密度聚类（仅传特征，不传标签）
3. 从 `labels_` 推导 `n_clusters` 和 `n_noise` 并打印日志
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, eps=0.3, min_samples=5, metric='euclidean')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的全量特征矩阵，传入 `DBSCAN.fit()` | `X_scaled` |
| `eps` | `float` | $\epsilon$ 邻域半径。默认 `0.3` | `0.2`、`0.3`、`0.5` |
| `min_samples` | `int` | 核心点阈值。默认 `5` | `3`、`5`、`10` |
| `metric` | `str` | 距离度量。默认 `'euclidean'` | `'euclidean'`、`'manhattan'` |
| 返回值 | `DBSCAN` | 已完成 `fit()` 的模型对象，含 `labels_`、`core_sample_indices_` | — |

### 理解重点

- DBSCAN 的 `fit()` 是无监督的——不接收标签参数。这与分类分册中所有 `train_model` 都有 `y_train` 参数形成鲜明对比。
- 训练日志中的 `n_clusters` 和 `n_noise` 是 DBSCAN 独有的统计输出——它们直接反映密度参数配置的合理性。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 聚类对照图 | `plot_clusters(...)` | `X_scaled`、`labels_pred`（`model.labels_`）、`labels_true`（`y_true`） | 左右对照散点图（PNG） |

### 理解重点

- `plot_clusters(...)` 是当前 DBSCAN 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- 左右对照布局：左侧为算法聚类结果（噪声点特殊着色），右侧为真实标签——这种设计在教学上非常直观。
- 不涉及 PCA 降维——原始数据本身就是二维的，可以直接用作散点图坐标。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `dbscan_data` | `data_generation/clustering.py` | `pipelines/clustering/dbscan.py` |
| `y_true` | `data["true_label"]` 提取 | `plot_clusters`（仅对照用） |
| `X_scaled` | `StandardScaler` | `train_model`、`plot_clusters` |
| `model`（含 `labels_`） | `train_model(...)` | `plot_clusters` |
| 图片产物 | `plot_clusters(...)` | `outputs/dbscan/` 目录 |

### 理解重点

- 数据依赖关系极为简洁——只有 5 个节点，单向流动无循环依赖。
- 比分类流水线少了 `train_test_split`、`predict`、`predict_proba`、PCA、ROC 评估、学习曲线等 6+ 个节点。
- `y_true` 的流向是单向的——从数据到可视化，不经过模型训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `DBSCAN 聚类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`eps`、`min_samples`、`簇数量`、`噪声点数量` | 查看密度扩展耗时、参数配置和聚类统计量 |
| 聚类对照图 | `outputs/dbscan/cluster_plot.png` | 左右对照：DBSCAN 聚类结果 vs 真实双月牙标签 |

### 理解重点

- 输出比分类分册少得多——只有 2 类（日志 + 1 张图），而非 5 类（日志 + 4 张图）。
- `簇数量` 和 `噪声点数量` 是 DBSCAN 独有的日志输出——它们在其他算法的训练日志中不存在。
- 聚类对照图是最核心的教学产出——它直接展示了密度聚类的效果和与真实结构的吻合度。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/clustering/dbscan.py` — 入口，代码量少，流程清晰
2. 再看 `model_training/clustering/dbscan.py` — 训练封装，理解无监督 `fit()` 和日志输出
3. 再看 `result_visualization/cluster_plot.py` — 聚类对照散点图绘制逻辑
4. 最后回到 `data_generation/clustering.py` — 理解 `make_moons` 双月牙数据生成

### 理解重点

- 从入口看整体流程，再下钻到训练和可视化细节，阅读成本最低。
- DBSCAN 的调用链比分类分册短得多——这本身就是密度聚类简洁性的体现。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/articles/ML/dbscan/cluster_plot.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的密度扩展在 `DBSCAN.fit()` 中。
2. 期待当前流水线有 `train_test_split` 或 `predict()` 调用——无监督聚类不需要这些。
3. 忽略 `n_clusters` 和 `n_noise` 的日志输出——它们是理解参数配置是否合理的最直接依据。
4. 把 `true_label` 当成参与训练的数据流——它的流向是"数据 → 可视化"，从未进入模型。

## 小结

- 当前 DBSCAN 工程实现采用极简的模块分层：数据生成 → 训练封装（无监督）→ 流水线编排 → 单一可视化（对照散点图）。
- `run()` 负责串联，`train_model(...)` 负责密度扩展（仅 `fit(X)`），`plot_clusters(...)` 负责视觉对照。
- DBSCAN 在工程上最不同于分类算法的地方：无切分、无监督 `fit()`、无 `predict()`、单一可视化（对照散点图）——这是由无监督聚类和 sklearn 的 DBSCAN 实现特性共同决定的。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 DBSCAN 实现。
2. 给出继续深入阅读密度聚类与相关数据工具的可靠入口。

## 自检题

1. 为什么 DBSCAN 流水线没有 `train_test_split`？无监督聚类为什么不需要训练/测试切分？
2. 为什么 `eps=0.3` 和 `min_samples=5` 在当前标准化后的双月牙数据上能正确分离两个月牙？如果 `eps` 增大到 `0.6` 会怎样？减小到 `0.1` 会怎样？
3. DBSCAN 的 `fit()` 与分类模型的 `fit(X, y)` 在参数签名和输出上有何本质差异？为什么 sklearn 的 DBSCAN 没有 `predict()` 方法？
4. 为什么 `model.labels_` 中会出现 $-1$？它与分类模型中的误分类样本有何本质区别？
5. 为什么 DBSCAN 的评估不包含混淆矩阵、ROC 曲线和准确率？聚类评估与分类评估的根本差异是什么？
6. 为什么 `true_label` 在当前流水线中不传入 `model.fit()`？它在流程中扮演什么角色？
7. 为什么标准化对 DBSCAN 是必要的？如果去掉标准化，`eps=0.3` 的几何意义会发生什么变化？

## 练习方向

### 1. 改变 `eps`

- 把 `eps=0.3` 改成 `0.1`、`0.2`、`0.3`、`0.5`、`0.7`
- 观察变化：
  - $n\_clusters$——`eps` 过大时两个月牙被错误合并为一个簇；`eps` 过小时每个高密度微区域都变成独立簇
  - $n\_noise$——`eps` 越小噪声点越多，直到几乎所有点都变成噪声
  - 聚类对照图——左右的视觉差异直接反映参数是否合理
- 核心理解：`eps` 是最需要精心调整的参数——它决定了"多大范围内的邻居算同一密度区域"

### 2. 改变 `min_samples`

- 把 `min_samples=5` 改成 `2`、`5`、`10`、`20`、`50`
- 观察变化：
  - 核心点的比例——`min_samples` 越大，越少的点满足核心条件
  - 簇的数量和噪声点数量——高 `min_samples` 时更多点被判定为噪声
  - 边界效应——对于 `min_samples=2`，几乎所有点都成为核心点
- 核心理解：`min_samples` 与 `eps` 联动——`eps` 增大时一般需要相应增大 `min_samples` 以避免合并过度

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察 `eps=0.3` 下的聚类结果变化
- 对比：原始数据坐标与标准化数据坐标的实际数值范围差异
- 体会：`eps` 是绝对数值——不标准化的数据让邻域判定在不同维度上含义不等同

### 4. 改变噪声水平

- 修改 `make_moons(noise=...)` 的 `noise` 参数（`0.0`、`0.05`、`0.08`、`0.15`、`0.25`）
- 观察变化：
  - 无噪声（`0.0`）——两个月牙完美分离，几乎没有噪声点
  - 高噪声（`0.25`）——两个月牙之间的间隙被部分填充，密度扩展可能跨过间隙
  - `n_noise` 随噪声增大而变化的趋势
- 核心理解：DBSCAN 的密度假设在低噪声数据上最有效，但适当噪声（`0.08`）下的表现展示了算法的鲁棒性

### 5. 对比 KMeans

- 用 `KMeans(n_clusters=2)` 在同一双月牙数据上聚类
- 对比变化：
  - 簇边界的形状——KMeans 以直线（Voronoi 边界）划分，将弯月切分；DBSCAN 沿月牙弧形密度扩展
  - 噪声处理——KMeans 强制分配每个点，DBSCAN 噪声点单独标记
  - 对非球形簇的适应性——这是两者最根本的差异
- 核心理解：聚类算法没有万能方案——算法选择必须匹配数据的结构特征

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`DBSCAN` | 完整构造器参数（`eps`、`min_samples`、`metric`、`algorithm`、`leaf_size`、`p`、`n_jobs`）、属性（`labels_`、`core_sample_indices_`）与方法说明 |
| 2 | scikit-learn 官方文档：`make_moons` | 双月牙数据生成器的 `n_samples`、`noise`、`shuffle`、`random_state` 等参数说明 |
| 3 | scikit-learn 用户指南：Clustering → DBSCAN | 密度聚类原理、`eps`/`min_samples` 选参方法、不同密度数据上的局限性与与其他聚类算法的使用场景对比 |
| 4 | Ester, M., Kriegel, H.-P., Sander, J., and Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*. KDD-96. | DBSCAN 原始论文——$\epsilon$ 邻域、核心点/边界点/噪声点概念、密度可达/密度相连关系和算法伪代码的源头 |

- scikit-learn `DBSCAN`：https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- scikit-learn `make_moons`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
- scikit-learn 用户指南 Clustering：https://scikit-learn.org/stable/modules/clustering.html#dbscan

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 DBSCAN 分册的核心内容：
  - 无监督聚类不需要训练/测试切分——`fit()` 在全量数据上执行，`labels_` 直接输出
  - `eps` 和 `min_samples` 联动决定核心点/边界点/噪声点的划分——改变其中一个通常需要调整另一个
  - DBSCAN 的 `fit()` 不接收标签——与分类分册的 `fit(X, y)` 有本质区别
  - sklearn 的 DBSCAN 没有 `predict()` 方法——它只能对训练数据本身做标记，不能预测新样本
  - 噪声点（$-1$）是 DBSCAN 的核心输出而非错误——噪声过多或过少才暗示参数不当
  - 聚类评估不同于分类评估——没有 accuracy/混淆矩阵/ROC，依赖散点图视觉对照和 `n_clusters`/`n_noise` 统计量
  - 标准化对基于距离的 `eps` 邻域判定是硬性要求——`eps` 是绝对数值，其几何意义依赖各维度尺度一致
  - DBSCAN 与 KMeans 不是"谁更好"的关系——球形数据选 KMeans，弯曲不规则数据选 DBSCAN
