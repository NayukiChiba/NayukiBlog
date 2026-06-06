---
title: EM 与 GMM
date: 2026-04-14
category: 机器学习/概率模型
tags:
  - Scikit-learn
  - 高级教程
description: 高斯混合模型的数学原理、EM算法迭代与完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581764332.webp
status: published
---

# 数学原理

## 本章目标

1. 理解高斯混合模型（GMM）的生成过程——$\pi_k$ 选分量 $\rightarrow$ $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ 生成样本。
2. 理解 EM 算法的两步迭代——E 步（计算责任）和 M 步（最大化参数）。
3. 理解对数似然的下界保证——EM 保证对数似然单调不减。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 高斯混合模型 | 生成模型 | $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$——$K$ 个高斯分布的加权和 |
| 隐变量 $z_{ik}$ | 概率框架 | 指示样本 $i$ 是否由分量 $k$ 生成——GMM 的"未观测变量" |
| E 步 | 期望计算 | 计算后验责任 $\gamma(z_{ik})$——"在当前参数下，样本 $i$ 属于分量 $k$ 的概率" |
| M 步 | 参数最大化 | 用责任加权更新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$——最大化完全数据对数似然的期望 |
| 对数似然下界 | 收敛保证 | $\log p(\mathbf{X} \mid \Theta)$ 在每次 EM 迭代中单调不减 |
| 协方差类型 | 模型假设 | `full`（完全协方差）允许椭圆形簇——比 KMeans 的球面假设更灵活 |

## 1. 高斯混合模型的生成过程

GMM 假设数据由 $K=3$ 个高斯分量按以下过程生成：

1. 以概率 $\pi_k$ 选择一个高斯分量：
   $$
   p(z_k = 1) = \pi_k, \quad \sum_{k=1}^{K} \pi_k = 1
   $$
2. 从所选分量的高斯分布中采样：
   $$
   p(\mathbf{x} \mid z_k = 1) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
   $$

边缘分布为：
$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### 理解重点

- $\pi_k$ 是**混合权重**——$\pi_k \ge 0$，$\sum_k \pi_k = 1$。当前数据 $\pi = [0.5, 0.3, 0.2]$。
- $\boldsymbol{\mu}_k = [\mu_{k1}, \mu_{k2}]^T$ 是第 $k$ 个分量的均值（2 维）。
- $\boldsymbol{\Sigma}_k$ 是 $2 \times 2$ 的协方差矩阵——`covariance_type="full"` 允许每个分量的协方差各不相同。

## 2. 最大似然的挑战

直接最大化对数似然：
$$
\log p(\mathbf{X} \mid \Theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

困难在于：$\log \sum$ 内部有求和——导数为零的方程没有闭式解，因为隐变量 $z_{ik}$ 未被观测。

### 理解重点

- 如果有标签（知道每个样本属于哪个分量），参数估计简化为加权样本均值和协方差——有闭式解。
- 无标签时，EM 通过**迭代猜测**（E 步）和**用猜测更新参数**（M 步）来绕过这个困难。

## 3. E 步：计算后验责任

给定当前参数 $\Theta^{(t)}$，计算每个样本属于每个分量的后验概率（责任）：

$$
\gamma(z_{ik})^{(t+1)} = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}
{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

- $\gamma(z_{ik}) \in [0, 1]$，且 $\sum_k \gamma(z_{ik}) = 1$——每个样本对各分量的责任和为 1
- 高斯密度：$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$

### 理解重点

- 责任 $\gamma(z_{ik})$ 就是**软赋值**——样本 $i$ 对三个分量各有部分归属。
- 与 KMeans 的硬赋值对比：KMeans 输出 $\gamma_{ik} \in \{0, 1\}$，EM 输出 $\gamma_{ik} \in [0, 1]$。
- 当前 `covariance_type="full"` 使 $\boldsymbol{\Sigma}_k$ 可以是任意正定矩阵——每个分量的高斯密度是倾斜的椭圆形。

## 4. M 步：最大化参数

用 E 步计算的责任 $\gamma_{ik}$ 作为权重，重新估计参数：

**有效样本数**：
$$
N_k = \sum_{i=1}^{N} \gamma(z_{ik})
$$

**均值更新**：
$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik}) \mathbf{x}_i
$$

**协方差更新**（`covariance_type="full"`）：
$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik}) (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^T
$$

**混合权重更新**：
$$
\pi_k^{(t+1)} = \frac{N_k}{N}
$$

### 理解重点

- 每个参数更新都是**责任加权**——$\gamma_{ik}$ 越大的样本对分量 $k$ 的参数更新贡献越大。
- 这相当于"软计数"——不是每个点固定属于一个分量，而是按比例贡献于多个分量。
- `full` 协方差给每个分量最大自由度——可以学习任意方向的椭圆形状。

## 5. 对数似然的单调性

EM 算法保证对数似然在每次迭代中**单调不减**：
$$
\log p(\mathbf{X} \mid \Theta^{(t+1)}) \ge \log p(\mathbf{X} \mid \Theta^{(t)})
$$

这是因为 EM 实际上在最大化对数似然的一个**下界**函数（ELBO）：
$$
\mathcal{L}(\Theta, q) = \sum_i \sum_k \gamma_{ik} \log \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\gamma_{ik}} \le \log p(\mathbf{X} \mid \Theta)
$$

### 理解重点

- 对数似然单调不减是 EM 收敛的保证——但只保证收敛到**局部最大值**，不保证全局最优。
- 当前源码中 `model.lower_bound_` 记录了收敛时的对数似然下界值。
- 在实际中，初始化的均值和协方差可能会使 EM 收敛到不同的局部最优——这类似于 KMeans 的 `n_init`。

## 6. 协方差类型对比

| `covariance_type` | 协方差约束 | 簇形状 | 参数数（$K$ 分量、$d$ 维） |
|---|---|---|---|
| `full` | 无约束 | 任意椭圆 | $K \times \frac{d(d+1)}{2}$ |
| `tied` | 所有分量共享 | 相同椭圆 | $\frac{d(d+1)}{2}$ |
| `diag` | 对角矩阵 | 轴对齐椭圆 | $K \times d$ |
| `spherical` | $\sigma_k^2 \mathbf{I}$ | 球形（同 KMeans） | $K \times 1$ |

当前源码使用 `full`——每个分量有独立的 $2 \times 2$ 协方差矩阵（3 个参数每个）。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 生成模型 | $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | `GaussianMixture(n_components=3, covariance_type="full")` |
| 隐变量 | $z_{ik}$ | 内部矩阵——E 步计算 |
| 后验责任 | $\gamma(z_{ik})$ | `model.predict_proba(X)` |
| 混合权重 | $\pi_k$ | `model.weights_` |
| 分量均值 | $\boldsymbol{\mu}_k$ | `model.means_` |
| 分量协方差 | $\boldsymbol{\Sigma}_k$ | `model.covariances_` |
| 对数似然下界 | $\log p(\mathbf{X} \mid \Theta)$ | `model.lower_bound_` |
| 最大迭代 | $t_{\max}$ | `max_iter=200` |
| 收敛判断 | $\|\Theta^{(t+1)} - \Theta^{(t)}\| < \epsilon$ | 内部自动判断 |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 8. EM vs KMeans 数学对比

| 维度 | KMeans | EM (GMM) |
|---|---|---|
| 目标函数 | $\min \sum_{i} \sum_{k} r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ | $\max \sum_i \log \sum_k \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ |
| 赋值 | 硬赋值 $r_{ik} \in \{0, 1\}$ | 软赋值 $\gamma_{ik} \in [0, 1]$ |
| 簇形状 | 球形（等距离衰减各向同性） | 椭圆形（全协方差各向异性） |
| 不确定性 | 无 | 有——$1 - \max_k \gamma_{ik}$ 量化置信度 |
| 参数数 | $K \times d$（均值） | $K \times (1 + d + d(d+1)/2)$（权重 + 均值 + 协方差） |

## 常见坑

1. 混淆 EM 与 KMeans——EM 输出概率归属（软聚类），KMeans 输出确定归属（硬聚类）。
2. 在 `covariance_type="spherical"` 下期待椭圆形簇——球形协方差等价于 KMeans 的簇形状假设。
3. 忽略 EM 收敛到局部最优的风险——不同初始化可能导致不同的聚类结果。
4. 认为 `max_iter=200` 不够——200 次对于 2 维 3 分量数据通常足够收敛。

## 小结

- EM 算法的数学核心链：GMM 生成模型 $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ $\rightarrow$ 隐变量 $\rightarrow$ E 步计算责任 $\gamma(z_{ik})$ $\rightarrow$ M 步责任加权更新参数 $\rightarrow$ 对数似然单调递增 $\rightarrow$ 局部收敛。
- 与 KMeans 的根本区别：概率软赋值（$\gamma_{ik}$ 连续）vs 距离硬赋值（$r_{ik}$ 离散）、椭圆协方差 vs 球形距离。
- 当前源码 `GaussianMixture(n_components=3, covariance_type="full", max_iter=200)` 是 GMM 最灵活的教学配置——允许每个分量有独立的全协方差矩阵。

# 数据构成

## 本章目标

1. 明确本仓库 EM 数据来自 `ProbabilisticData.em()` 手动合成的 3 分量高斯混合数据。
2. 理解为何手动合成非球形数据——各分量具有不同的均值和标准差，充分展示 GMM 全协方差的建模能力。
3. 明确当前流程中 `true_label` 的角色——仅用于评估对比，**不参与 EM 训练**。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.em()` | 方法 | 手动合成 3 分量非球形 GMM 数据 |
| `numpy.random.RandomState` | 类 | 种子随机数生成——保证数据可复现 |
| `em_means` | 属性 | 3 个分量的均值——$\{[0,0], [4,4], [-3,4]\}$ |
| `em_stds` | 属性 | 3 个分量的标准差——各维度不同，生成非球形簇 |
| `em_weights` | 属性 | 3 个分量的混合权重——$[0.5, 0.3, 0.2]$ |
| `true_label` | 列 | 真实分量标签——仅用于评估对比，EM 训练时不可见 |

## 1. 数据生成：`ProbabilisticData.em()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。`500`——适中规模，EM 可在秒级收敛 | `500`、`1000` |
| `em_n_components` | `int` | 高斯分量数。`3`——充分展示 GMM 的多分量建模 | `2`、`3`、`5` |
| `em_means` | `list[list]` | 各分量的 2 维均值。$\{[0,0], [4,4], [-3,4]\}$——分量间有显著间距 | 任意 `list[list[float]]` |
| `em_stds` | `list[list]` | 各分量的 2 维标准差。$\{[0.8,0.5], [0.6,1.0], [1.2,0.7]\}$——各维度不同，生成非球形簇 | 任意 `list[list[float]]` |
| `em_weights` | `list[float]` | 混合权重。$[0.5, 0.3, 0.2]$——分量不等权，更贴近实际情况 | 任意和为 1 的 `list[float]` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `DataFrame` | 含 `x1`、`x2`、`true_label` 三列 | — |

### 示例代码

```python
from data_generation.probabilistic import ProbabilisticData

data = ProbabilisticData().em()
# data.columns = ["x1", "x2", "true_label"]
# data.shape = (500, 3)
```

### 生成流程

```python
# 1. 按混合权重分配各分量样本数
counts = rng.multinomial(500, [0.5, 0.3, 0.2])
# counts ~= [250, 150, 100]

# 2. 各分量独立生成样本
for k in range(3):
    X_k = rng.randn(counts[k], 2) * stds[k] + means[k]

# 3. 合并后随机打乱
X = np.vstack([X_0, X_1, X_2])
idx = rng.permutation(500)
```

### 理解重点

- 这是**手动合成**的数据——不是 `make_blobs` 生成的等权球形簇。每个分量有独立的均值和各向异性的标准差。
- 分量 1（`[0,0]`，权重 0.5）是最大最密的分量——标准差 $[0.8, 0.5]$，沿 $x_1$ 方向更宽。
- 分量 2（`[4,4]`，权重 0.3）较紧凑——标准差 $[0.6, 1.0]$，沿 $x_2$ 方向更宽。
- 分量 3（`[-3,4]`，权重 0.2）最小最散——标准差 $[1.2, 0.7]$，沿 $x_1$ 方向最分散。
- 这种非球形设计使得 `covariance_type="full"` 的 GMM 能正确建模椭圆簇——而 KMeans（球面聚类）则无法精确刻画。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(500, 2)` | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["true_label"])` |
| `true_label` | `Series`，形状 `(500,)` | 真实分量标签 $\{0, 1, 2\}$——**仅用于评估对比**，不参与训练 | `data["true_label"].values` |

### 理解重点

- `true_label` 是生成数据时记录的真实分量标号——EM 算法**完全不使用**这一列。
- 在流水线中，`y_true = data["true_label"].values` 在标准化前即被提取——它不参与后续任何计算，仅传入 `plot_clusters` 做可视化对比。
- 这与 KMeans/DBSCAN 分册中的 `true_label` 角色完全一致——聚类算法对标签"视而不见"。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(500, 2)` | 全量特征矩阵 | `X` |
| 返回值 | `ndarray`，形状 `(500, 2)` | Z-score 标准化后的特征 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- 当前流水线使用 `fit_transform` 在全量数据上做标准化——**不分训练/测试集**。这是聚类的标准做法。
- 与集成分类（有 `fit_transform`/`transform` 分离）形成对比——聚类没有"将训练集的统计量应用于测试集"的概念。
- 标准化对 EM 至关重要——高斯密度中的马氏距离 $(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ 对特征尺度敏感。未标准化的特征可能导致某个维度主导协方差矩阵。

## 4. 数据设计意图：与 KMeans/DBSCAN 的对比

| 数据维度 | KMeans | DBSCAN | EM (GMM) |
|---|---|---|---|
| 生成方式 | `make_blobs`（等权球形簇） | `make_blobs` + 均匀噪声 | **手动合成（不等权非球形）** |
| 簇形状 | 球形——适合 KMeans | 球形 + 噪声——适合 DBSCAN | **非球形椭圆——适合 GMM** |
| 样本数 | 500 | 500 | 500 |
| 特征维度 | 2 | 2 | 2 |
| 簇数 | 3 | 3 | 3 |
| 噪声点 | 无 | 有（均匀噪声） | 无 |
| 混合权重 | 等权 | 等权 | **不等权 $[0.5, 0.3, 0.2]$** |

### 理解重点

- EM 数据**刻意不球对称**——每个分量的 $x_1$ 和 $x_2$ 标准差各不相同，簇形状为拉伸的椭圆。
- 这种设计是为了展示 GMM 相对于 KMeans 的核心优势——`covariance_type="full"` 能正确建模椭圆形簇，而 KMeans 的球面假设无法处理。
- 不等权设计也更贴近实际数据——真实数据中的"簇"通常大小不匀。

## 数据可视化

![聚类分布图](https://img.yumeko.site/file/blog/articles/1780737746297.png)

## 常见坑

1. 不标准化就直接调用 `GaussianMixture`——不同维度尺度差异会导致某个维度主导协方差估计。
2. 把 `true_label` 当成训练标签——EM 是无监督算法，标签只用于评估。
3. 在球形数据（`make_blobs`）上对 GMM 使用 `covariance_type="full"`——参数过多可能过拟合，`spherical` 更合适。
4. 忽略数据打乱步骤——本数据集在生成后已随机打乱，但若自行构造数据，不打乱会影响可视化判断。

## 小结

- 当前 EM 数据来自手动合成的 3 分量 GMM：均值 $\{[0,0], [4,4], [-3,4]\}$，标准差各异，权重 $[0.5, 0.3, 0.2]$——非球形、不等权。
- 数据流为：手动采样 -> DataFrame（`x1`、`x2` + `true_label`）-> 提取 `true_label` 用于评估 -> 全量标准化。
- 非球形不等权的设计意图是展示 GMM 全协方差建模相对于 KMeans 球面聚类的核心优势。

# 思路与直觉

## 本章目标

1. 用直观方式理解 EM 算法的核心思路——"猜分属，再调整"的循环迭代。
2. 理解软赋值（责任）与硬赋值（标签）的直觉差异——概率归属 vs 确定归属。
3. 通过与 KMeans 的对比，建立 GMM 在聚类谱系中的定位——概率生成模型。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 鸡生蛋蛋生鸡 | 核心困境 | 知道样本属于哪个分量 -> 可以算分量参数；知道分量参数 -> 可以判断样本属于哪个分量 |
| E 步（责任分配） | 迭代操作 | "在当前参数下，这个点大概是哪个分量生成的？"——给每个点对各分量的归属概率 |
| M 步（参数更新） | 迭代操作 | "按这个归属概率，每个分量的最佳中心在哪？椭圆该长什么样？" |
| 软赋值 | 核心特征 | 每个点对 3 个分量各有归属概率——$\gamma_{ik} \in [0,1]$，和为 1 |
| 对数似然下界 | 收敛保证 | 每次迭代后"数据在当前参数下的概率"都在上升——不会变差 |

## 1. 为什么需要 EM

假设你要把 500 个点分成 3 个椭圆簇——但你没有标签。如果你知道每个点的分量归属：

*可以很容易算出每个分量的中心（加权平均）和形状（加权协方差）。*

反过来，如果你知道 3 个分量的中心和椭圆形状：

*可以很容易算出每个点属于哪个分量（看它在哪个椭圆的正下方）。*

但两样都没有——这就是"鸡生蛋蛋生鸡"。

EM 的解决思路：先猜参数 -> 凭参数猜归属 -> 凭归属更新参数 -> 凭新参数重新猜归属 -> ... 直到稳定。

## 2. 用"分类垃圾"理解 EM

想象你把 500 颗不同颜色的珠子混在一起，想让机器自动按颜色分成 3 堆：

1. **先乱猜 3 堆的中心和范围**（初始化 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$）
2. **看每颗珠子离哪堆更近**（E 步——但 EM 不是"最近的堆"，而是"最可能来自哪堆"——考虑到每堆的大小和形状）
3. **重新计算每堆的中心和范围**（M 步——加权平均，因为有些珠子可能"有点属于堆 A，也有点属于堆 B"）
4. **重复 2-3 步**直到每堆的中心不再移动

### 理解重点

- EM 比 KMeans 多考虑了**形状**——如果一个簇是细长的椭圆，靠近椭圆长轴末端的点可能离另一个簇的球心更近，但它实际上属于这个椭圆簇。
- 这就是马氏距离 vs 欧氏距离的核心差异——EM 衡量的是"在椭圆坐标系下的距离"。

## 3. 软赋值 vs 硬赋值的直觉

**KMeans（硬赋值）**：
> "这个点离 A 最近，它就是 A 的人，跟 B 和 C 一点关系都没有。"

**EM（软赋值）**：
> "这个点 70% 可能是 A 的，20% 可能是 B 的，10% 可能是 C 的——但我不确定，所以三个分量都参考它，只是权重不同。"

### 理解重点

- 软赋值在边界模糊处最有价值——两个椭圆重叠区域的点，KMeans 可能武断地划给一方，EM 会"分担责任"。
- `model.predict_proba(X)` 返回的就是这个软赋值矩阵——`(500, 3)`，每行和为 1.0。
- 不确定性 = $1 - \max_k \gamma_{ik}$——值越大说明这个点"在分量间摇摆不定"。

## 4. E 步和 M 步的直觉对比

| 步骤 | 直觉 | 输入 | 输出 |
|---|---|---|---|
| E 步 | "在当前参数下，猜测每个点属于各分量" | 当前 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$ | 责任矩阵 $\gamma_{ik}$ |
| M 步 | "根据每个点的分量归属，重新计算参数" | 责任矩阵 $\gamma_{ik}$ | 新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$ |

### 理解重点

- E 步用的是当前的模型参数——"模型现在长这样，你说这些点是谁的？"
- M 步用的是 E 步的输出——"既然你们觉得这些点大概是 A 的，那 A 的中心应该在...这个位置，椭圆应该长...这个形状。"
- 每次迭代，$p(\text{数据} \mid \text{参数})$ 都在增加——EM 永远不会"变差"。

## 5. 用"猜考试分数"理解协方差类型

3 个班的学生考了两门试（数学和物理）：

- **`spherical`**：假设每个班的数学和物理分数独立且方差相同——等价于 KMeans
- **`diag`**：假设每个班的数学和物理分数独立但方差不同——横平竖直的椭圆
- **`full`**：每个班的数学和物理分数可能相关（数学好的物理也好）——任意方向的椭圆

当前数据使用 `full`——因为分量 1 在 $x_1$ 方向更宽（数学分差大），分量 2 在 $x_2$ 方向更宽（物理分差大）。

## 6. EM vs KMeans 直觉对比

| 维度 | KMeans | EM (GMM) |
|---|---|---|
| 核心问题 | 如何把点分成 K 个球形簇？ | **这些点最可能由哪 K 个椭圆高斯生成？** |
| 赋值 | 硬——每个点只属于一个簇 | **软——每个点对各分量有归属概率** |
| 距离度量 | 欧氏距离（到质心） | **马氏距离（考虑协方差的加权距离）** |
| 簇形状 | 圆形（各向同性） | **任意椭圆（各向异性）** |
| 不确定性 | 无——每个点只有一个标签 | **有——边界模糊的点有混合归属** |
| 输出 | `labels_`（整数标签） | **`labels_` intersect `predict_proba()`（概率归属）** |
| 极限关系 | — | KMeans 是 GMM 在 `spherical` 协方差 + 硬赋值下的极限 |

### 理解重点

- GMM 是 KMeans 的"概率升级版"——保持了"把点分组"的目标，但增加了椭圆形状和软归属。
- 损失：GMM 的参数更多（`full` 下每个分量要估计 3 个协方差参数），在小样本上可能过拟合。
- 收益：GMM 能正确拟合非球形簇，并提供归属概率的置信度。

## 可视化

![聚类分布图](https://img.yumeko.site/file/blog/articles/1780737746297.png)

## 常见坑

1. 把 EM 当成 KMeans 的"精确版"——EM 收敛更慢、对初始化更敏感，不保证全局最优。
2. 认为软赋值"更高级所以总是更好"——如果数据确实是球形等权簇（`make_blobs`），GMM 的全协方差反而包含冗余参数。
3. 忽略协方差类型的含义——在 `spherical` 下 GMM 退化为 KMeans 的概率版，椭圆簇建模能力为零。
4. 期待 EM 每次收敛到相同结果——EM 是局部优化，不同初始化的结果可能不同。

## 小结

- EM 的直觉核心是"猜-改"循环：用当前参数猜归属（E 步）-> 用归属更新参数（M 步）-> 重复。每步保证数据似然不降。
- 软赋值是 EM 与 KMeans 最核心的直觉差异——概率归属允许"一个点同时部分属于多个分量"。
- 全协方差（`full`）使 GMM 能拟合任意方向的椭圆——这是相对于 KMeans 球面簇假设的最大优势。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GaussianMixture`——无监督，不需要 `y_train` 参数。
2. 理解 `GaussianMixture` 的核心构造器参数（`n_components`、`covariance_type`、`max_iter`）及其概率含义。
3. 看清训练完成后最重要的模型属性——`weights_`（混合权重）、`means_`（分量均值）、`covariances_`（分量协方差）、`lower_bound_`（对数似然下界）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.mixture.GaussianMixture` 模型——EM 算法 |
| `GaussianMixture(...)` | 类 | scikit-learn 提供的 GMM 实现——通过 EM 迭代估计分量参数 |
| `model.fit(X)` | 方法 | EM 迭代训练——最多 200 次 E-M 循环，收敛则提前停止 |
| `model.weights_` | 属性 | 混合权重 $\pi_k$——3 个分量的先验概率 |
| `model.means_` | 属性 | 分量均值 $\boldsymbol{\mu}_k$——3 个椭圆中心的坐标 |
| `model.covariances_` | 属性 | 分量协方差 $\boldsymbol{\Sigma}_k$——3 个椭圆的形状和方向 |
| `model.lower_bound_` | 属性 | 对数似然下界——训练收敛的诊断指标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_components=3, covariance_type="full", max_iter=200, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(500, 2)` | 标准化后的特征矩阵——注意无 `y_train` 参数，EM 是无监督学习 | `X_scaled` |
| `n_components` | `int` | 高斯分量数。`3`——与真实分量数一致，是已知先验知识 | `2`、`3`、`5` |
| `covariance_type` | `str` | 协方差类型。`"full"`——每个分量有独立的完全协方差矩阵 | `"full"`、`"tied"`、`"diag"`、`"spherical"` |
| `max_iter` | `int` | EM 最大迭代次数。`200`——足够 2 维 3 分量数据收敛 | `100`、`200`、`500` |
| `random_state` | `int` | 随机种子，保证初始化可复现。`42` | `42` |
| 返回值 | `GaussianMixture` | 已完成 `fit()` 的模型对象——含 `weights_`、`means_`、`covariances_` | — |

### 示例代码

```python
from model_training.probabilistic.em import train_model

model = train_model(X_scaled)
```

### 理解重点

- `train_model(...)` 是**无监督训练**——没有 `y_train` 参数。这是 EM 与集成分类模型（Bagging/GBDT）的最根本差异。
- `n_components=3` 需要作为先验知识给定——与 KMeans 的 `n_clusters=3` 相同。在实际应用中，$K$ 需要通过 BIC 或交叉验证选择。
- `covariance_type="full"` 是 GMM 最灵活的配置——允许 3 个分量各自由学习形状。

## 2. `GaussianMixture` 构造器参数

### 参数速览

适用 API：`GaussianMixture(n_components=3, covariance_type="full", max_iter=200, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 高斯分量数。默认 `3`——EM 算法的 $K$ 值，需预先指定 | `2`、`3`、`5` |
| `covariance_type` | `str` | 协方差约束类型。`"full"`——每个分量有独立的完全协方差矩阵 | `"full"`、`"tied"`、`"diag"`、`"spherical"` |
| `tol` | `float` | 收敛阈值。默认 `1e-3`——对数似然变化小于此值则停止 | `1e-3`、`1e-4` |
| `max_iter` | `int` | EM 最大迭代次数。`200`——安全上限，2 维 3 分量数据通常在 100 次内收敛 | `100`、`200`、`500` |
| `n_init` | `int` | 随机初始化的次数。默认 `1`——只用一次 k-means 初始化 | `1`、`5`、`10` |
| `init_params` | `str` | 初始化方法。默认 `"kmeans"`——先用 KMeans 聚类作为初始参数 | `"kmeans"`、`"random"` |
| `reg_covar` | `float` | 协方差对角线的非负正则化。默认 `1e-6`——防止协方差奇异 | `1e-6`、`1e-4` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `verbose` | `int` | 日志级别。默认 `0` | `0`、`1`、`2` |

### 示例代码

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=3,
    covariance_type="full",
    max_iter=200,
    random_state=42,
)
model.fit(X_scaled)
```

### 理解重点

- `covariance_type` 是 GMM 最关键的选择——`full` 参数最多但最灵活；`spherical` 参数最少但退化为 KMeans-like。
- `init_params="kmeans"`（默认）使用 KMeans 聚类结果初始化 GMM 均值和协方差——这提供了一个"合理的起点"。
- `reg_covar=1e-6` 是一个数值稳定技巧——在协方差对角线上加一个小值，防止矩阵奇异。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `weights_` | `ndarray`，形状 `(3,)` | 混合权重 $\pi_k$ | 3 个分量的先验概率——和为 1，反映各分量的体量 |
| `means_` | `ndarray`，形状 `(3, 2)` | 分量均值 $\boldsymbol{\mu}_k$ | 3 个椭圆中心在 2 维空间中的坐标 |
| `covariances_` | `ndarray`，形状 `(3, 2, 2)` | 分量协方差 $\boldsymbol{\Sigma}_k$ | 3 个 2x2 协方差矩阵——描述各椭圆的形状和方向 |
| `lower_bound_` | `float` | 对数似然下界 $\log p(\mathbf{X} \mid \Theta)$ | 收敛时当前参数下的数据对数似然 |
| `converged_` | `bool` | EM 是否收敛 | `True` 表示在 `max_iter` 内达到容差收敛 |
| `n_iter_` | `int` | 实际 EM 迭代次数 | 可能小于 `max_iter`（提前收敛） |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"covariance_type: {covariance_type}")
print(f"log-likelihood: {model.lower_bound_:.4f}")
print(f"混合权重: {model.weights_}")
print(f"分量均值:\n{model.means_}")
print(f"是否收敛: {model.converged_}")
```

### 理解重点

- `weights_`、`means_`、`covariances_` 是 GMM 的"三件套"——完全描述了 $K$ 个高斯分量的混合模型。
- `lower_bound_` 是 EM 训练的诊断指标——值越大（越接近 0），模型对数据的拟合越好。
- `covariances_` 的 `(3, 2, 2)` 形状——3 个分量，每个有一个 $2 \times 2$ 协方差矩阵。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2\}$ | 硬预测——取 $\gamma_{ik}$ 最大的分量索引 |
| `predict_proba(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n, 3)` | 软预测——每个样本对 3 个分量的归属概率，行和为 1.0 |

### 理解重点

- `predict()` 是对软赋值做硬截断——等价于 $\arg\max_k \gamma_{ik}$。与 KMeans 的 `predict()` 输出格式一致。
- `predict_proba()` 是 GMM 独有的输出——直接返回 E 步计算的责任矩阵 $\gamma_{ik}$，提供了归属不确定性信息。
- 与集成分类的 `predict_proba` 不同——这里的概率是"属于哪个高斯分量"，不是"属于哪个类别"。

## 常见坑

1. 在 `n_components` 未知的情况下盲目设定——GMM 需要预先知道分量数，可借助 BIC/AIC 选择。
2. 混淆 `predict()` 和 `predict_proba()` 的输出——前者是硬标签（与 KMeans 相同），后者是软归属（GMM 独有）。
3. 在 `covariance_type="spherical"` 下期待椭圆簇——此时 GMM 退化为概率版 KMeans。
4. 忽略 `lower_bound_` 的符号——它是对数似然，始终为负数（密度小于 1），越接近 0 拟合越好。

## 小结

- `train_model(...)` 是本仓库 EM 的核心训练入口——是对 `sklearn.mixture.GaussianMixture` 的薄封装，无监督（无 `y_train`）。
- `GaussianMixture` 的核心参数是 `n_components`（分量数）、`covariance_type`（协方差约束）、`max_iter`（迭代上限）——三者共同决定模型的灵活性和收敛行为。
- 训练完成后的核心属性：`weights_` / `means_` / `covariances_`（三件套描述 GMM）和 `lower_bound_`（收敛诊断）——构成了完整的概率模型描述。

# 训练与预测

## 本章目标

1. 理解 `pipelines/probabilistic/em.py` 的 `run()` 流水线——无监督聚类下的端到端流程（无训练/测试切分）。
2. 理解 EM 算法的 `fit()` 训练过程——E 步计算责任 + M 步更新参数，对数似然单调不减。
3. 理解 `predict()` 和 `predict_proba()` 的输出差异——硬标签 vs 软归属。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 无监督聚类流水线编排——5 步串联标准化、EM 训练、预测和聚类可视化 |
| `model.fit(X_scaled)` | 方法 | EM 迭代训练——E 步（计算责任）+ M 步（更新参数），直到收敛或达到 `max_iter` |
| `model.predict(X_scaled)` | 方法 | 硬聚类标签——对每个样本取 $\gamma_{ik}$ 最大分量的索引 |
| `model.predict_proba(X_scaled)` | 方法 | 软归属——每个样本对 3 个分量的概率，返回后验责任矩阵 $\gamma_{ik}$ |
| `plot_clusters(X_scaled, labels_pred, labels_true, ...)` | 函数 | 双面板对比——预测标签 vs 真实标签的聚类分布 |

## 1. 完整流水线流程

### 流程概述

```
em_data.copy()
  - 1 y_true = data["true_label"].values  # 仅用于评估对比
  - 2 X = data.drop(columns=["true_label"])
  - 3 X_scaled = scaler.fit_transform(X)  # 全量标准化
  - 4 model = train_model(X_scaled)        # 无 y_train
  - 5 labels_pred = model.predict(X_scaled) -> plot_clusters
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `em_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(500, 3)` | 避免修改全局变量 |
| 提取真实标签 | `data["true_label"].values` | `DataFrame` | `ndarray`，`(500,)` | 仅用于评估对比——EM 训练时不使用 |
| 分离 X | `data.drop(columns=["true_label"])` | `DataFrame` | `DataFrame`，`(500, 2)` | 特征 `x1`、`x2` |
| 标准化 | `scaler.fit_transform(X)` | `DataFrame` | `ndarray`，`(500, 2)` | 全量数据 Z-score——无训练/测试切分 |
| 训练 | `train_model(X_scaled)` | `ndarray` | `GaussianMixture` | EM 迭代——无监督，无 `y_train` |
| 预测 | `model.predict(X_scaled)` | `ndarray` | `ndarray`，`(500,)` | 硬聚类标签 $\{0, 1, 2\}$ |
| 可视化 | `plot_clusters(X_scaled, labels_pred, y_true, ...)` | `(ndarray, ndarray, ndarray)` | PNG 文件 | 双面板聚类分布对比 |

### 理解重点

- 流水线只有 5 步——比分类流水线更简洁（无 `train_test_split`、无 `stratify`）。
- `y_true` 在步骤 1 就提取完毕——全程不参与训练，只在最后传入 `plot_clusters` 做可视化对比。
- 标准化使用 `fit_transform`（全量数据一次性完成）——聚类没有"将训练统计量应用于测试集"的需求。

## 2. 训练细节：`model.fit(X_scaled)`

### EM 迭代流程

```
初始化（KMeans 聚类 -> 初始均值和协方差）
    v
E 步：对每个样本 i 和分量 k，计算责任 gamma(z_ik)
    gamma(z_ik) = pi_k * N(x_i|mu_k, Sigma_k) / Sigma_j pi_j * N(x_i|mu_j, Sigma_j)
    v
M 步：用 gamma(z_ik) 作为权重，更新参数
    mu_k = Sigma_i gamma(z_ik) * x_i / N_k
    Sigma_k = Sigma_i gamma(z_ik) * (x_i - mu_k)(x_i - mu_k)^T / N_k
    pi_k = N_k / N
    v
检查收敛：|log p(X|Theta_new) - log p(X|Theta_old)| < tol ？
    是 -> 停止
    否 -> 回到 E 步
    v
达到 max_iter=200 -> 终止
```

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_components` | `3` | 高斯分量数——EM 的预设 K，决定了责任矩阵列数 |
| `covariance_type` | `"full"` | 每个分量学习独立的 2x2 协方差矩阵 |
| `max_iter` | `200` | E-M 循环最大次数——通常远小于此即收敛 |
| `tol` | `1e-3`（默认） | 对数似然变化阈值——连续两次小于此值则收敛 |
| `init_params` | `"kmeans"`（默认） | 初始参数来自 KMeans 聚类——提供较好的起点 |
| `reg_covar` | `1e-6`（默认） | 协方差对角线的正则化——防止奇异矩阵 |

### 理解重点

- EM 训练是**最小化对数似然的负数**——每次迭代保证数据似然不降，但只收敛到局部最优。
- `init_params="kmeans"` 意味着初始均值来自 KMeans 聚类——这比随机初始化收敛更快且更稳定。
- 对于 2 维 3 分量数据，EM 通常会在 50-100 次迭代内收敛——远小于 `max_iter=200`。

## 3. 预测细节

### `model.predict(X_scaled)` — 硬聚类

对每个样本 $i$，返回后验概率最大的分量索引：
$$
\hat{y}_i = \arg\max_k \gamma(z_{ik}) = \arg\max_k p(z_{ik} = 1 \mid \mathbf{x}_i, \Theta)
$$

### `model.predict_proba(X_scaled)` — 软聚类

直接返回后验责任矩阵：
$$
[\gamma(z_{ik})]_{N \times K} = \left[\frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}\right]
$$

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 2)` | `(n,)` | 硬聚类标签——$\{0, 1, 2\}$ |
| `predict_proba(X)` | `(n, 2)` | `(n, 3)` | 软归属概率——行和为 1.0，每列对应一个分量 |

### 理解重点

- `predict()` 的输出与 KMeans 的 `predict()` 格式完全一致——都是 $\{0, 1, 2\}$ 的整数标签。
- `predict_proba()` 是 GMM 独有的——KMeans 没有此方法。它提供了每个样本对各分量归属的"确定性"。
- 高不确定性样本：如果某样本的 $\max_k \gamma_{ik} < 0.6$，说明它在两个分量边界处"摇摆不定"。

## 4. 与 KMeans 训练流程的对比

| 步骤 | KMeans | EM (GMM) |
|---|---|---|
| 初始化 | k-means++（质心） | KMeans 聚类（均值和协方差） |
| 赋值 | 硬——最小欧氏距离 | **软——最大后验概率 $\gamma_{ik}$** |
| 更新 | 算术平均（等权更新质心） | **加权平均（$\gamma_{ik}$ 加权更新均值和协方差）** |
| 收敛条件 | 标签不再变化 | **对数似然变化 < tol** |
| 标准化 | fit_transform（全量） | fit_transform（全量）——相同 |
| 训练数据 | X_scaled（无 y） | X_scaled（无 y）——相同 |
| 迭代上限 | `max_iter=300` | `max_iter=200` |

## 常见坑

1. 没有标准化就直接 `fit`——不同特征尺度导致协方差估计偏向尺度大的维度。
2. 在 `n_components` 不匹配真实分量数时硬用——分量数需要作为先验知识或通过 BIC 选择。
3. 忽略 EM 的局部最优风险——不同的 `random_state` 可能给出不同的聚类结果。
4. 混淆 `predict` 和 `predict_proba` 的用途——评估聚类效果用前者，分析归属不确定性用后者。

## 小结

- EM 流水线仅有 5 步——是最简洁的聚类流水线之一：提取标签（仅供评估）-> 分离特征 -> 标准化 -> 训练 -> 预测 -> 可视化。
- `fit()` 的核心流程：KMeans 初始化 -> E 步（计算责任 $\gamma_{ik}$）-> M 步（责任加权更新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$、$\pi_k$）-> 检查对数似然收敛 -> 循环。
- `predict()` 和 `predict_proba()` 分别提供硬聚类标签和软归属概率——后者是 GMM 区别于 KMeans 的关键输出。

# 评估与诊断

## 本章目标

1. 理解当前 EM 流水线的评估输出——聚类分布图（预测 vs 真实标签双面板对比）和对数似然日志。
2. 理解聚类分布图中的诊断信号——软边界、椭圆形状、误分模式。
3. 明确当前流水线**已实现**和**未实现**的评估项——以及 EM 评估与分类评估的根本差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 聚类分布图 | 图表 | 双面板对比——左：EM 预测标签（`labels_pred`）；右：真实分量标签（`true_label`） |
| 对数似然日志 | 终端文本 | `lower_bound_`——收敛时数据在当前 GMM 参数下的对数似然 |
| 软归属不确定性 | 诊断信息 | $1 - \max_k \gamma_{ik}$——定量衡量每个样本在分量间的摇摆程度 |

## 1. 聚类分布图

`plot_clusters(X_scaled, labels_pred=labels_pred, labels_true=y_true, ...)` 绘制双面板聚类对比图。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 `(500, 2)` | 标准化后的 2 维特征 | `scaler.fit_transform(X)` |
| `labels_pred` | `ndarray`，形状 `(500,)` | EM 预测的硬聚类标签——$\arg\max_k \gamma_{ik}$ | `model.predict(X_scaled)` |
| `labels_true` | `ndarray`，形状 `(500,)` | 真实分量标签——**仅用于评估对比，不参与训练** | `data["true_label"].values` |
| `title` | `str` | 图表标题 | `"EM (GMM) 聚类分布"` |
| `dataset_name` | `str` | 数据集名称——决定输出路径 | `"em"` |
| `model_name` | `str` | 模型名称——决定输出路径 | `"gmm"` |

### 示例代码

```python
y_true = data["true_label"].values
labels_pred = model.predict(X_scaled)
plot_clusters(
    X_scaled,
    labels_pred=labels_pred,
    labels_true=y_true,
    title="EM (GMM) 聚类分布",
    dataset_name="em",
    model_name="gmm",
)
```

### 输出

![聚类分布图](https://img.yumeko.site/file/blog/articles/1780737746297.png)

### 理解重点

- 双面板对比是聚类评估的核心手段——左面板显示"EM 认为的 3 个簇"，右面板显示"真实的 3 个分量"。
- 理想情况：左右面板的簇边界和颜色分布基本一致，说明 EM 成功恢复了 GMM 的分量结构。
- GMM 的特色：边界模糊区域的点——左侧可能将"摇摆不定的点"划到某个分量，这是硬截断的副作用。通过 `predict_proba` 可查看这些点的真实归属不确定性。
- 与 KMeans 聚类图的对比：KMeans 的左面板分界线是 Voronoi 直线，EM 的边界是曲线（因为马氏距离度量）。

## 2. 对数似然下界

训练完成后，终端打印 `lower_bound_`：

```text
模型训练完成
n_components: 3
covariance_type: full
log-likelihood: -2.1457
```

### 理解重点

- `lower_bound_` 是 EM 在收敛点的对数似然估计——始终为负数（概率密度 < 1 时 $\log < 0$）。
- 值越接近 0，模型拟合越好——但要注意过拟合风险（参数越多，似然越高）。
- `lower_bound_` 在每次 EM 迭代中**单调递增**——如果观察到下降，说明程序存在 bug。

## 3. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 聚类分布图 | 已实现 | 无监督聚类评估的核心可视化 |
| 对数似然日志 | 已实现 | EM 收敛的定量诊断指标 |
| BIC / AIC | **未实现** | 模型选择指标——用于选择 $n\_components$。当前 $K=3$ 为已知先验 |
| 归属不确定性分析 | **部分实现** | `predict_proba` 提供数据但未做可视化——可在练习中探索 |
| 混淆矩阵 | **不适用** | `labels_pred` 的分量标签可能与 `true_label` 交换——聚类标签没有内在语义 |
| 轮廓系数 | **未实现** | 适用于硬聚类评估——与 GMM 的软赋值哲学不完全匹配 |
| 交叉验证对数似然 | **未实现** | 教学场景下单次训练的对数似然足够 |

### 理解重点

- 聚类评估与分类评估的根本差异：**聚类标签没有内在语义**——EM 的输出可能是把分量 1 标为 $0$，分量 2 标为 $1$，而真实标签的编号可能正好相反。混淆矩阵在此场景下无意义。
- BIC（贝叶斯信息准则）和 AIC（赤池信息准则）是 GMM 选择最优 $K$ 的标准方法——它们在对数似然的基础上加上了参数数量的惩罚项。

## 4. 软归属的诊断价值

虽然流水线中未显式使用 `predict_proba`，但软归属矩阵 $\gamma_{ik}$ 提供了重要的诊断信息：

- **高不确定性样本**：$\max_k \gamma_{ik} < 0.6$——这些点在分量边界处，说明分量间有重叠
- **误分模式**：高不确定性样本集中的区域——可能是真实分量交界处，也可能是 EM 初始化不佳导致的"错误边界"
- **分量体量估计**：$\sum_i \gamma_{ik} / N \approx \pi_k$——与 `model.weights_` 应一致

### 理解重点

- 软归属是 GMM 评估相对于 KMeans 的独特优势——不仅知道"点被分到哪"，还知道"分的把握有多大"。
- 高不确定性样本在决策应用中需要特殊处理——可以拒绝决策或要求人工标注。

## 5. EM vs KMeans vs DBSCAN 评估对比

| 评估维度 | KMeans | DBSCAN | EM (GMM) |
|---|---|---|---|
| 聚类可视化 | 聚类图（`labels_pred`） | 聚类图（`labels_pred` + 噪声点） | **双面板对比图**（`labels_pred` + `true_label`） |
| 定量指标 | `inertia_` | 噪声点数量 + 簇数 | **`lower_bound_`（对数似然）** |
| 软归属 | 无 | 无 | **`predict_proba()`** |
| 簇形状诊断 | Voronoi 多边形 | 任意形状簇 | **椭圆簇——观察协方差矩阵** |
| 标签语义 | 无 | 无 | 无——标签编号可以交换 |
| 噪声处理 | 无（所有点归入簇） | 核心特征（label=-1） | **无——每个点都有完整的归属概率** |

## 常见坑

1. 直接对比 `labels_pred` 和 `true_label` 的标签编号——聚类标签没有语义，编号可以互换。
2. 认为 `lower_bound_` 越大越好（无上限）——过参数化（增大 $K$ 或使用 `full` 协方差）总会增加似然，需要 BIC 平衡。
3. 忽略高不确定性样本——它们在分析中可能表示"分量间重叠区"或"数据不满足 GMM 假设"。
4. 把聚类分布图中的边界曲线当成确定性分界线——GMM 的分界线是 $\gamma_{ik}$ 最大分量切换的位置，边界处的软归属需额外关注。

## 小结

- EM 当前有两项评估输出：聚类分布图（预测 vs 真实标签双面板对比）和对数似然下界日志（收敛定量指标）。
- GMM 评估的独特价值在于软归属——`predict_proba()` 提供每个样本对各分量的归属概率，可用于不确定性分析和边界模糊诊断。
- 与分类评估的根本差异：聚类标签无内在语义，不能直接用准确率/混淆矩阵评判——聚类图的眼睛对比是最直观的评估方式。

# 工程实现

## 本章目标

1. 理解 EM 流水线的模块分层——数据生成层、模型训练层、流水线编排层、聚类可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径——注意无监督特征（无 `y_train`、无切分）。
3. 理解 EM 与 KMeans/DBSCAN 在工程实现上的异同——同为聚类，但模型内部结构完全不同。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ProbabilisticData.em()` | 方法 | 手动合成 3 分量非球形 GMM 数据 |
| `train_model(...)` | 函数 | 构建并训练 `GaussianMixture`——无监督，无 `y_train` 参数 |
| `run()` | 函数 | 无监督聚类流水线编排——5 步串联标准化、EM 训练、预测和可视化 |
| `plot_clusters(...)` | 函数 | 绘制双面板聚类分布对比图——预测标签 vs 真实标签 |
| `model.predict(X)` | 方法 | 硬聚类标签 |
| `model.predict_proba(X)` | 方法 | 软归属——后验责任矩阵 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/probabilistic.py` | 手动合成 3 分量非球形 GMM 数据并导出 `em_data` | 全局 `DataFrame`（500 行 x 3 列） |
| 模型训练层 | `model_training/probabilistic/em.py` | 封装 `GaussianMixture` 训练——含装饰器 | `GaussianMixture` 模型对象 |
| 流水线编排层 | `pipelines/probabilistic/em.py` | 串联标准化、EM 训练、预测和聚类可视化——端到端入口 | 终端日志 + 聚类分布图 |
| 可视化层 | `result_visualization/cluster_plot.py` | 生成双面板聚类分布对比图 | 1 个 PNG 文件 |

### 理解重点

- EM 的模块分层与 KMeans/DBSCAN 的结构完全一致——数据生成 -> 训练封装 -> 流水线编排 -> 聚类可视化。
- 训练层使用 `@print_func_info` + `@timeit` + `timer`——与 GBDT/LightGBM/XGBoost 的装饰器风格一致。
- 与集成分类的最关键区别：**训练层不接收 `y_train`**——EM 是无监督学习。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `em_data.copy()` | — | `DataFrame`，形状 `(500, 3)` | 避免修改全局变量 |
| 2 | `data["true_label"].values` | `DataFrame` | `ndarray`，`(500,)` | 提取真实标签——仅供评估对比 |
| 3 | `data.drop(columns=["true_label"])` | `DataFrame` | `DataFrame`，`(500, 2)` | 分离 2 维特征 X |
| 4 | `scaler.fit_transform(X)` | `DataFrame` | `ndarray`，`(500, 2)` | 全量数据 Z-score 标准化 |
| 5 | `train_model(X_scaled)` | `ndarray` | `GaussianMixture` | EM 迭代训练——无 `y_train` |
| 6 | `model.predict(X_scaled)` | `ndarray` | `ndarray`，`(500,)` | 硬聚类标签 |
| 7 | `plot_clusters(X_scaled, labels_pred, y_true, ...)` | `(ndarray, ndarray, ndarray)` | PNG 文件 | 双面板聚类对比图 |

### 理解重点

- 步骤 2-3 顺序不可交换——必须先提取 `true_label`，再 `drop`。如果先 `drop`，`true_label` 将丢失。
- 步骤 5 无 `y_train` 参数——这是 EM 与集成分类训练函数的根本差异。
- 与 KMeans 流水线唯一的区别：`plot_clusters` 多传了一个 `labels_true` 参数以实现双面板对比。

## 3. 数据依赖关系

```
em_data (全局 DataFrame)
  - -> y_true = data["true_label"].values ──-> 仅供评估 ──────────┐
  - -> X = data.drop(columns=["true_label"])                      │
    - -> scaler.fit_transform(X) ──-> X_scaled ──┐             │
      - train_model(X_scaled) ──-> model        │             │
        - -> model.predict(X_scaled) ──-> labels_pred ──┐  │
      - plot_clusters(X_scaled, labels_pred, y_true, ...) <-┘  │
```

### 理解重点

- `y_true` 是一个独立的横向数据流——从数据提取阶段直接流向可视化，完全不经过训练和预测。
- 没有 `train_test_split` 分支——聚类在整个数据集上训练和评估。
- 与 KMeans 的数据依赖图结构一致——只是 `train_model` 的输入参数不同（无 `y_train`）。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 聚类分布图 | `outputs/gmm/data_cluster_distribution.png` | PNG | 双面板对比——左：EM 预测标签 / 右：真实分量标签 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 对数似然下界 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.probabilistic.em
```

### 输出

```text
============================================================
EM (GMM) 聚类流水线
============================================================
模型训练完成
n_components: 3
covariance_type: full
log-likelihood: -2.1457
模型训练耗时: 0.15s

============================================================
EM (GMM) 流水线完成！
============================================================
```

### 理解重点

- EM 输出 1 个 PNG 文件——与 KMeans/DBSCAN 相同（都是聚类图），但多了 `labels_true` 对比面板。
- 训练耗时通常极短（亚秒级）——500 样本 x 2 维 x 3 分量，EM 收敛很快。
- 终端日志打印 `log-likelihood`——这是 EM 独有的诊断输出，KMeans 和 DBSCAN 都没有。

## 5. 训练层细节：与 KMeans 的对比

| 工程维度 | KMeans | EM (GMM) |
|---|---|---|
| 模型类 | `KMeans` | **`GaussianMixture`** |
| 核心参数 | `n_clusters`、`init`、`n_init` | **`n_components`、`covariance_type`、`max_iter`** |
| 训练输入 | `fit(X)`——无 `y` | `fit(X)`——无 `y` |
| 预测输出 | `predict(X)` ☑ `predict_proba` ☒ | `predict(X)` ☑ `predict_proba(X)` ☑ |
| 模型属性 | `cluster_centers_`、`inertia_`、`labels_` | **`means_`、`covariances_`、`weights_`、`lower_bound_`** |
| 装饰器 | 无 | `@print_func_info` + `@timeit` + `timer` |
| 日志 | `n_clusters`、`inertia_` | **`n_components`、`covariance_type`、`log-likelihood`** |

### 理解重点

- EM 的参数体系比 KMeans 多一个关键维度——`covariance_type` 控制簇形状的灵活性。
- EM 的输出比 KMeans 更丰富——多概率输出（`predict_proba`）和概率模型组件（`means_`、`covariances_`、`weights_`）。
- EM 的训练初始化依赖 KMeans（`init_params="kmeans"`）——两者在工程上是合作关系。

## 阅读顺序

1. `data_generation/probabilistic.py` — 了解 `em()` 的 GMM 数据合成逻辑
2. `model_training/probabilistic/em.py` — 理解 `GaussianMixture` 的构建和 EM 迭代训练
3. `pipelines/probabilistic/em.py` — 看清无监督聚类端到端流程
4. `result_visualization/cluster_plot.py` — 了解聚类双面板对比图实现

## 常见坑

1. 在调用 `drop("true_label")` 之前未提取 `y_true`——`true_label` 列被丢弃后将无法用于可视化对比。
2. 把 `train_model` 当成有监督训练——它接收的参数只有 `X_train`，无 `y_train`。
3. 直接修改 `em_data` 而不先 `copy()`——污染全局变量。
4. 在测试集上使用 `fit_transform`——EM 的聚类场景下没有测试集概念，但如果在其他场景误用，仍然会造成信息泄露。

## 小结

- EM 工程实现遵循本仓库标准四层架构：数据生成层 -> 模型训练层 -> 流水线编排层 -> 可视化层（聚类图模块）。
- `run()` 是极简编排函数——5 步完成标签提取、特征分离、标准化、训练、预测和可视化。
- 与 KMeans/DBSCAN 的核心工程共同点：同为无监督聚类（无 `y_train`、无 `train_test_split`）；核心差异：EM 有更丰富的概率输出（`predict_proba`、`means_`、`covariances_`、`weights_`、`lower_bound_`）。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 EM 算法核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 GMM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 E 步/M 步、软赋值、协方差类型、EM vs KMeans 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 GMM 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 EM 算法原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. EM 算法的 E 步和 M 步分别完成什么任务？为什么说 EM 的每次迭代都保证了数据对数似然单调不减？

2. GMM 的软赋值（后验责任 $\gamma_{ik}$）与 KMeans 的硬赋值（最近质心归属）有何本质区别？在什么场景下软赋值的优势最为明显？

3. `covariance_type` 的四个选项（`full`、`tied`、`diag`、`spherical`）分别对应什么约束？为什么当前数据使用 `full` 而非 `spherical`？

4. `lower_bound_` 的含义是什么？为什么它始终为负数？如果连续两次训练的 `lower_bound_` 差异很大，可能是什么原因？

5. GMM 的混合权重 $\pi_k$、均值 $\boldsymbol{\mu}_k$、协方差 $\boldsymbol{\Sigma}_k$ 三者在 M 步的更新公式是什么？为什么每个都使用了责任加权？

6. 如果 `n_components` 被设为不等于 3 的值（如 2 或 5），GMM 的聚类结果会怎样？如何用 BIC 来辅助选择最优 $K$？

7. EM 和 KMeans 在初始化、参数更新方式、收敛条件上有哪些本质差异？KMeans 可以视为 GMM 的哪种特殊情况？

## 2. 动手练习

### 练习 1：改变协方差类型

将 `covariance_type` 分别设为 `"full"`、`"tied"`、`"diag"`、`"spherical"`，观察聚类分布图的变化。

```python
model = train_model(X_scaled, covariance_type="spherical")
```

回答：`spherical` 下 GMM 的椭圆建模能力完全丢失——聚类分布图是否退化为与 KMeans 类似？`tied`（所有分量共享协方差）的簇形状与 `full` 有何差异？

### 练习 2：错误设定分量数 `n_components`

将 `n_components` 分别设为 `2`、`3`、`4`、`5`、`7`，观察聚类分布图和 `lower_bound_` 的变化。

```python
model = train_model(X_scaled, n_components=2)
```

回答：`n_components=2` 时 EM 如何将 3 个真实分量"合并"为 2 个簇？`n_components=7` 时是否出现了多余的空分量？`lower_bound_` 随 $K$ 增大是否单调递增？

### 练习 3：改变随机种子观察局部最优

将 `random_state` 分别设为 `0`、`1`、`42`、`99`、`123`，观察聚类分布图的变化。

```python
model = train_model(X_scaled, random_state=0)
```

回答：不同的随机种子是否导致显著不同的聚类结果？哪些种子下 EM 收敛到了"不好的"局部最优？如何通过增加 `n_init` 来缓解此问题？

### 练习 4：分析软归属

提取 `predict_proba` 的输出，找到不确定性最高的样本。

```python
probas = model.predict_proba(X_scaled)
uncertainty = 1 - probas.max(axis=1)
top_uncertain = np.argsort(uncertainty)[-10:]  # 最不确定的 10 个点
```

回答：高不确定性样本在二维空间中的位置在哪？它们是否位于两个真实分量之间的"重叠带"？

### 练习 5：对比 GMM 与 KMeans 的聚类差异

在相同数据上分别训练 KMeans（`KMeans(n_clusters=3)`）和 GMM（`GaussianMixture(n_components=3, covariance_type="full")`），对比聚类分布图。

```python
from sklearn.cluster import KMeans

model_km = KMeans(n_clusters=3, random_state=42)
model_km.fit(X_scaled)
labels_km = model_km.predict(X_scaled)
```

回答：在非球形分量（如分量 1 的 $x_1$ 标准差 0.8 vs $x_2$ 标准差 0.5）的区域，KMeans 的硬边界是否显得"不合理"？GMM 是否成功捕捉了椭圆的各向异性？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*. Journal of the Royal Statistical Society, Series B, 39(1), 1-38. | EM 算法的原始论文——E 步/M 步形式化和收敛性证明 |
| 2 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 9. | 标准教材——GMM 和 EM 算法的完整推导和实例 |
| 3 | scikit-learn 官方文档 — [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 11. | 概率视角教材——EM 算法作为变分推断的特例，含 BIC/AIC 模型选择 |

## 常见坑

1. 把 `n_components` 设得过大——过多的分量会导致某些分量的协方差矩阵退化为奇异矩阵（`reg_covar` 是最后防线）。
2. 在不做标准化的数据上跑 EM——不同特征尺度会导致协方差矩阵被大尺度特征主导。
3. 认为 `lower_bound_` 可以跨数据集比较——对数似然依赖于数据规模和维度，不同数据集间的值不可直接对比。
4. 忽略 EM 的局部最优特性——单次训练的结果可能只是一个局部最优解，`n_init > 1` 可降低此风险。

## 小结

- 7 个自检问题覆盖 EM 算法的核心概念：E 步/M 步、软赋值、协方差类型、`lower_bound_`、参数更新公式、$K$ 选择、与 KMeans 对比。
- 5 个动手练习从不同角度探索 GMM 的行为——改变协方差类型、错误设定分量数、测试随机种子、分析软归属、对比 KMeans。
- 4 篇参考文献从原始论文（Dempster et al. 1977）$\rightarrow$ 教材（Bishop + Murphy）$\rightarrow$ 官方文档构成完整的阅读路线。
