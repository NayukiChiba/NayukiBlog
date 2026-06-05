---
title: PCA 主成分分析
date: 2026-05-08
category: 机器学习/降维
tags:
  - Scikit-learn
description: PCA主成分分析的数学原理、SVD求解与双模型降维实践。
image: https://img.yumeko.site/file/blog/cover/1780581855521.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 PCA 的优化目标——最大化投影方差（寻找数据变化最大的方向）。
2. 理解协方差矩阵特征值分解与 SVD 之间的等价关系。
3. 理解 `explained_variance_ratio_` 的数学含义及其随 `n_components` 变化的单调性。
4. 理解标准化对 PCA 的数学必要性——协方差矩阵的几何意义依赖特征量纲一致。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 投影方差 $\text{Var}(\mathbf{X}\mathbf{w})$ | 优化目标 | PCA 寻找最大化投影方差的方向——方差越大，"信息量"越大 |
| 协方差矩阵 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ | 核心矩阵 | 其特征向量即主成分方向，特征值即该方向的方差 |
| 特征值分解 $\mathbf{S}\mathbf{u}_k = \lambda_k \mathbf{u}_k$ | 求解形式 | 第 $k$ 个主成分是第 $k$ 大特征值对应的特征向量 |
| SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | 数值求解 | 右奇异向量 $\mathbf{V}$ 的列即主成分方向，$\sigma_k^2/N = \lambda_k$ |
| `explained_variance_ratio_` | 评估指标 | 第 $k$ 个主成分的解释方差占比 $\lambda_k / \sum_j \lambda_j$ |
| `svd_solver='auto'` | 工程选择 | scikit-learn 自适应选择最优 SVD 求解器 |

## 1. PCA 的优化目标

给定 $N$ 个样本 $\mathbf{x}_i \in \mathbb{R}^d$（已去均值），PCA 寻找投影方向 $\mathbf{w}$（$\|\mathbf{w}\|=1$），使得投影后数据的方差最大：

$$
\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X}\mathbf{w}) = \max_{\|\mathbf{w}\|=1} \frac{1}{N} \sum_{i=1}^{N} (\mathbf{w}^T \mathbf{x}_i)^2 = \max_{\|\mathbf{w}\|=1} \mathbf{w}^T \mathbf{S} \mathbf{w}
$$

其中 $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ 为协方差矩阵（数据已中心化）。

### 拉格朗日推导

约束 $\mathbf{w}^T\mathbf{w} = 1$ 下最大化 $\mathbf{w}^T\mathbf{S}\mathbf{w}$：

$$
\mathcal{L} = \mathbf{w}^T\mathbf{S}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = 0 \quad\Rightarrow\quad \boxed{\mathbf{S}\mathbf{w} = \lambda \mathbf{w}}
$$

### 理解重点

- 主成分方向是协方差矩阵的特征向量——这不是经验规则，而是拉格朗日乘子法的严格推导结果。
- 特征值 $\lambda_k$ 恰好等于该方向的投影方差——$\text{Var}(\mathbf{X}\mathbf{u}_k) = \lambda_k$。
- 因此取最大的 $q$ 个特征值对应的特征向量，即得到保留方差最多的 $q$ 个投影方向。

## 2. 多个主成分

第一个主成分 $\mathbf{u}_1$（最大特征值 $\lambda_1$）捕获最多的方差。第二个主成分 $\mathbf{u}_2$ 在与 $\mathbf{u}_1$ 正交的约束下最大化方差（对应于第二大特征值 $\lambda_2$），以此类推。

$$
\text{Var}(\mathbf{X}\mathbf{u}_1) = \lambda_1 \geq \text{Var}(\mathbf{X}\mathbf{u}_2) = \lambda_2 \geq \dots \geq \text{Var}(\mathbf{X}\mathbf{u}_d) = \lambda_d
$$

### 解释方差比

第 $k$ 个主成分的解释方差比定义为：

$$
\text{explained\_variance\_ratio\_}[k] = \frac{\lambda_k}{\sum_{j=1}^{d} \lambda_j}
$$

### 理解重点

- 每个主成分的解释方差比反映了它在总方差中的"份额"——值越大，该方向承载的信息越多。
- 累计解释方差比 $\sum_{k=1}^{q} \lambda_k / \sum_j \lambda_j$ 反映了前 $q$ 个主成分总共保留了多少信息。
- 当前源码打印这两项——它们是 PCA 训练完成后最重要的定量输出。

## 3. SVD 与 PCA 的等价性

在实际计算中，通常不显式构造协方差矩阵 $\mathbf{S}$ 再做特征分解，而是通过 SVD 直接求解：

$$
\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
$$

其中 $\mathbf{U}$（$N \times N$）和 $\mathbf{V}$（$d \times d$）为正交矩阵，$\boldsymbol{\Sigma}$ 为对角奇异值矩阵。代入协方差矩阵：

$$
\mathbf{S} = \frac{1}{N} \mathbf{X}^T \mathbf{X} = \frac{1}{N} \mathbf{V} \boldsymbol{\Sigma}^T \mathbf{U}^T \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T = \mathbf{V} \frac{\boldsymbol{\Sigma}^2}{N} \mathbf{V}^T
$$

因此：
- $\mathbf{V}$ 的列即主成分方向（特征向量）
- $\sigma_k^2 / N = \lambda_k$（奇异值的平方除以 $N$ 等于特征值）

### 理解重点

- SVD 路径不需要显式计算 $\mathbf{X}^T\mathbf{X}$——这在 $d$ 很大时（如 $d = 10^5$）极大降低了计算量和数值误差。
- `svd_solver='auto'` 是 scikit-learn 的默认选择——它会根据 $N$ 和 $d$ 的大小自动选择 full / randomized / arpack 中最合适的 SVD 实现。
- 这个等价性是 PCA 实现层最重要的数学性质——它把"特征分解协方差矩阵"转化为"SVD 分解数据矩阵"。

## 4. PCA 与 LDA 的数学对比

| 维度 | PCA | LDA |
|---|---|---|
| 监督方式 | 无监督 | 有监督（需要 $y$） |
| 优化目标 | $\max \mathbf{w}^T\mathbf{S}_T\mathbf{w}$（最大化投影方差） | $\max \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$（最大化类间/类内比） |
| 核心矩阵 | 总协方差矩阵 $\mathbf{S}_T$ | 类内散度 $\mathbf{S}_W$ + 类间散度 $\mathbf{S}_B$，满足 $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ |
| 降维上限 | $\min(d, N)$ | $K-1$ |
| 求解方式 | 特征分解或 SVD | 广义特征值问题 |
| 特征值含义 | 投影方差 | Fisher 准则值 $J(\mathbf{w})$ |

### 理解重点

- PCA 的特征值 $\lambda_k$ 是"该方向保留了多少方差"——这在物理意义上非常直观。
- LDA 的特征值是"该方向的类间/类内散度比"——同样 $[0.6, 0.4]$ 的两个数字，在 PCA 中表示方差占比，在 LDA 中表示判别能力占比。
- 两者同名属性 `explained_variance_ratio_` 在语义上是不同的——这是对比阅读时最容易混淆的地方。

## 5. 标准化对 PCA 的数学必要性

PCA 的核心操作是计算协方差矩阵：

$$
\mathbf{S} = \frac{1}{N} \mathbf{X}^T\mathbf{X}, \quad S_{ij} = \frac{1}{N}\sum_{n=1}^{N} x_{ni} x_{nj}
$$

若特征 $x_1$ 的取值量纲是 $x_2$ 的 100 倍，则 $S_{11}$ 将是 $S_{22}$ 的约 $100^2 = 10000$ 倍——协方差矩阵被尺度最大的特征完全主导。

### 理解重点

- 标准化后每个特征均值为 0、方差为 1——协方差矩阵退化为相关系数矩阵，各特征平等参与。
- 当前数据是合成数据——各特征量纲本身相近（均为高斯噪声的线性组合），但标准化仍然是最佳实践。
- 这与 LDA、KMeans、DBSCAN、SVC 的逻辑完全一致——任何基于距离或协方差的算法都需要标准化。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 中心化数据 | $\mathbf{X} - \bar{\mathbf{x}}$ | `StandardScaler` 处理后均值为 0 |
| 协方差矩阵 | $\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ | PCA 内部计算（经由 SVD 间接得到） |
| 主成分方向 | $\mathbf{u}_1, \dots, \mathbf{u}_q$ | `model.components_`，形状 `(q, d)` |
| 特征值（方差） | $\lambda_k$ | `model.explained_variance_` |
| 解释方差比 | $\lambda_k / \sum_j \lambda_j$ | `model.explained_variance_ratio_` |
| 累计解释方差 | $\sum_{k=1}^{q} \lambda_k / \sum_j \lambda_j$ | `model.explained_variance_ratio_.sum()` |
| 保留主成分数 | $q$ | `n_components=2` 或 `3` |
| SVD 求解器 | — | `svd_solver='auto'` |
| 投影 | $\mathbf{X}\mathbf{U}_q$ | `model.transform(X)` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |
| 随机种子 | — | `random_state=42` |

## 常见坑

1. 混淆 PCA 与 LDA 的优化目标——PCA 最大化投影方差（无监督），LDA 最大化类间/类内散度比（有监督）。
2. 忽略标准化——协方差矩阵对特征量纲高度敏感，不标准化的 PCA 本质上是"尺度最大的特征主导的 PCA"。
3. 把 PCA 的 `explained_variance_ratio_` 与 LDA 的同名属性当成同一种含义——前者是方差占比，后者是判别能力占比。
4. 误认为 `n_components` 加 1 必然带来显著信息增益——信息增益的边际递减率取决于数据的固有秩结构。

## 小结

- PCA 的数学核心链：中心化数据 → 协方差矩阵 $\mathbf{S}$ → 特征值问题 $\mathbf{S}\mathbf{u} = \lambda\mathbf{u}$ → 等价于 SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ → 取最大 $q$ 个特征向量作为主成分方向。
- 特征值 $\lambda_k$ 就是该方向的投影方差，`explained_variance_ratio_` 就是各方向方差占总方差的比例。
- 当前源码 `PCA(n_components=2, svd_solver='auto', random_state=42)` 针对低秩合成数据（3 个真实方向 + 10 维表面特征）是展示方差压缩最经典的教学配置。

# 数据构成

## 本章目标

1. 明确本仓库 PCA 数据来自 `DimensionalityData.pca()` 构造的低秩高维合成数据。
2. 理解数据的低秩结构——10 维表面特征中仅隐藏 3 个真实维度——这正是 PCA 的价值所在。
3. 明确 `label` 的角色——它是数据生成时构造的伪标签，仅用于可视化着色，不参与 PCA 训练。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `DimensionalityData.pca()` | 方法 | 生成 PCA 使用的低秩高维合成数据 |
| `base @ projection + noise` | 数据构造 | 3 维真实结构经随机投影映射到 10 维 + 高斯噪声——模拟"表面维度假高"的真实场景 |
| `pca_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `label` | 列名 | 由 `(base[:,0]>0) + (base[:,1]>0)` 生成的伪标签（3 类）——仅用于可视化着色，不参与 `fit()` |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——协方差矩阵计算的前置条件 |

## 1. 数据生成：`DimensionalityData.pca()`

当前 PCA 数据来自 `DimensionalityData.pca()`，底层纯手工构造，不依赖 scikit-learn 的现成数据生成器。

### 参数速览

适用方法：`DimensionalityData.pca()` 内部使用的参数（来自 `DimensionalityData` 数据类的属性）

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。默认 `400` | `400`、`500` |
| `pca_n_features` | `int` | 表面特征维度——数据的"表观维数"。默认 `10` | `10`、`20` |
| `pca_n_informative` | `int` | 真实信息维度——数据真正的固有秩。默认 `3` | `3`、`5` |
| `pca_noise_std` | `float` | 高斯噪声的标准差。`0.5` 在"可识别结构"和"适度干扰"间取得平衡 | `0.1`、`0.5`、`1.0` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| 返回值 | `DataFrame` | 含 10 个特征列（`x1`–`x10`）和 1 个伪标签列（`label`）的 DataFrame，形状 $(400, 11)$ | — |

### 示例代码

```python
rng = np.random.default_rng(random_state)

# 1. 生成 3 维真实结构
base = rng.standard_normal((n_samples, pca_n_informative))  # (400, 3)

# 2. 随机投影矩阵将 3 维映射到 10 维
projection = rng.standard_normal((pca_n_informative, pca_n_features))  # (3, 10)

# 3. 低秩信号 + 高斯噪声
X = base @ projection  # (400, 10) — 秩为 3
X += rng.standard_normal((n_samples, pca_n_features)) * pca_noise_std

# 4. 构造伪标签（仅用于着色）
label = (base[:, 0] > 0).astype(int) + (base[:, 1] > 0).astype(int)  # {0, 1, 2}
```

### 理解重点

- 数据有 10 个表面特征，但真正变化的维度只有 3 个（加噪声）——这正是 PCA 最擅长处理的场景。
- `base @ projection` 产生的信号矩阵秩为 $\min(3, 10) = 3$——前 3 个主成分应捕获几乎全部信号，后续主成分仅贡献噪声方差。
- `pca_noise_std=0.5` 在"结构清晰可辨识"和"不显得过于人工"之间取得了教学平衡。
- 这种构造方式直观展示了 PCA 的核心价值——从 10 维中提取 3 个真正有意义的方向。

## 2. 特征列与 `label` 的角色

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 10)$ | 含 10 个连续特征的特征矩阵，列名 `x1`–`x10` | `data.drop(columns=["label"])` |
| `label` | `ndarray`，形状 $(400,)$ | 伪标签 $\{0, 1, 2\}$——由真实低维结构的前两个维度符号决定，**仅用于可视化着色** | `data["label"].values` |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `label` 不传入 `model.fit()`——PCA 是无监督算法，`fit(X)` 只接收特征。
- `label` 的唯一目的是在 `plot_dimensionality(...)` 中为散点着色——帮助读者观察降维后数据的几何结构，而非评估分类效果。
- `label` 由 `base[:,0] > 0` 和 `base[:,1] > 0` 共同决定——这意味着 2D PCA 投影图上的类别分布恰好与真实低维结构的两个主方向相关，但 PCA 本身并不知道这一点。
- 与 LDA 的关键区别：LDA 的 `label` 参与训练，PCA 的 `label` 只用于着色。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 10)$ | 去掉 `label` 后的全量特征矩阵 | `X` |
| 返回值 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- PCA 依赖协方差矩阵——$\mathbf{S} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$ 的元素是平方和与叉积，对特征尺度高度敏感。
- 当前合成数据各特征量纲本已相近（均为高斯噪声的线性组合），但标准化仍是最佳实践。
- 与 LDA 流水线一致——无切分，在全量数据上直接 `fit_transform`。这是降维分册（PCA 和 LDA）共有的教学型简化。

## 数据可视化

![类别分布图](https://img.yumeko.site/file/articles/ML/pca/data_class_distribution.png)

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/pca/data_correlation.png)

![二维特征空间](https://img.yumeko.site/file/articles/ML/pca/data_feature_space_2d.png)

![三维特征空间](https://img.yumeko.site/file/articles/ML/pca/data_feature_space_3d.png)

## 常见坑

1. 把 `label` 当成训练标签传入 `model.fit()`——PCA 是无监督算法，不接受标签参数。
2. 忽略标准化——协方差矩阵计算被特征量纲绑架，主成分方向失真。
3. 忘记数据是低秩合成结构——前 3 个主成分应捕获绝大部分方差，若结果与此不符，说明噪声水平或数据处理有问题。
4. 把伪标签的 3 个类别当成数据固有的分类目标——`label` 只是低维结构的符号化着色依据。

## 小结

- 当前 PCA 数据来自手工构造的低秩合成数据：`base`（400×3）@ `projection`（3×10）+ 噪声（$\sigma=0.5$）。
- 数据流为：随机生成 → 低秩信号 + 噪声 → DataFrame（`x1`–`x10` + `label`）→ 剥离 `label` → 全量标准化。
- `label` 仅用于可视化着色——这是无监督降维与有监督降维（LDA）在数据处理上的根本差异。
- 低秩构造（10 维表面仅 3 维真实变化）使 PCA 的价值直观可感——前 3 个主成分应与后续主成分形成明显的方差断层。

# 思路与直觉

## 本章目标

1. 用直观方式理解 PCA 的方差最大化思路——"数据往哪个方向摆，幅度最大？"
2. 理解为什么低秩高维数据是展示 PCA 价值的理想教学数据。
3. 通过与 LDA 的对比，建立 PCA 在降维算法图中的定位——无监督方差压缩。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 方差最大化 | 核心直觉 | PCA 找的是数据变化最大的方向——方差越大，"信息量"越大 |
| 低秩结构 | 数据特性 | 10 维数据中只有 3 个真正变化的方向——PCA 的任务就是找到它们 |
| 主成分 | 结果 | 按方差从大到小排列的正交方向——第一个主成分是最"宽"的方向 |
| `label` 边界 | 角色区分 | 伪标签仅用于着色——PCA 训练完全不需要它 |
| 双模型设计 | 教学决策 | 分别训练 2D 和 3D PCA——直观对比不同降维程度下的信息保留 |

## 1. 为什么需要 PCA

PCA 回答的问题是：

> 如果只能用很少的维度表示高维数据，应该保留哪些方向才能丢失最少的信息？

PCA 用"方差"衡量信息——一个方向上数据的散布越宽，说明这个方向承载的变化越多，丢弃它损失的信息就越大。因此 PCA 找的是**方差最大的方向**。

### 理解重点

- PCA 把降维问题转化成了"找高方差方向"——这和 LDA 的"找类别最可分方向"目标完全不同。
- 这个直觉在很多日常场景中成立——拍照时找能看清最多细节的角度，本质上就在做类似的"方差最大化"。
- 正因为不需要标签，PCA 是最通用的降维方法——适用于任何高维数据，无论有没有标注。

## 2. 用"数据铺得最开的方向"理解主成分

PCA 的工作方式可以想象成：

1. **把数据中心化**——让所有点围绕原点对称分布
2. **找一个方向**——看数据在这个方向上投影后，点散布得有多开（方差多大）
3. **转一个角度再试**——不断调整方向，直到找到散布最开的方向（**第一主成分**）
4. **在与第一主成分垂直的方向上找第二开的方向**——这就是**第二主成分**
5. **继续直到找到足够多的方向**——这些方向彼此正交，按方差降序排列

### 理解重点

- 这就像给一团点云拍"侧视图"——旋转视角直到看到点云最宽的角度，那就是第一主成分的角度。
- 每个后续主成分必须与前面的正交——这保证了每个新方向捕获的是"独立的新信息"，而非已有方向的重复。
- 方差 = 特征值——当前源码打印的 `explained_variance_ratio_` 就是各方向方差占总方差的比例。

## 3. 为什么低秩高维数据特别适合展示 PCA

当前数据来自 `DimensionalityData.pca()`：10 个表面特征，但只由 3 个真实维度 + 噪声构成。

- 前 3 个主成分应捕获几乎全部信号——剩余的 7 个主成分只贡献噪声方差
- 2D PCA 图（`n_components=2`）能展示约 2/3 的真实结构
- 3D PCA 图（`n_components=3`）几乎完整展示全部真实结构

### 理解重点

- 如果数据没有低秩结构（10 个特征各不相关），PCA 无法有效降维——前几个主成分的解释方差占比会均匀分散。
- 当前数据的低秩构造使其成为 PCA 的理想展示场景——降维效果直观且显著。
- 这也是教学型数据设计的核心意图——让读者一眼就能看懂 PCA 在做什么。

## 4. 为什么 `label` 在这里只是"着色工具"

PCA 不需要标签——它的优化目标（投影方差）只涉及 $\mathbf{X}$，不涉及 $\mathbf{y}$。

当前 `label` 是数据生成时顺手构造的（`(base[:,0]>0) + (base[:,1]>0)`）——它反映了低维结构前两个维度的符号组合，恰好可以在 2D PCA 图上形成有意义的着色模式。

### 理解重点

- 在 PCA 分册中，`label` 只是"参考信息"——删掉它 PCA 照样训练，完全不受影响。
- 在 LDA 分册中，`label` 是"训练必需输入"——删掉它 LDA 无法工作。
- 这是两套降维哲学在工程上最直观的分界：PCA 的数据流中 `label` 只进可视化不进模型，LDA 的 `label` 同时进模型和可视化。

## 5. 为什么当前流水线分别训练 2D 和 3D 两个模型

当前 PCA 流水线先训练一个 `n_components=2` 的模型并生成 2D 图，再训练一个 `n_components=3` 的模型并生成 3D 图。

### 理解重点

- 这不是同一个模型取了不同数量的主成分——代码中分别创建了两个独立的 `PCA` 实例并分别 `fit()`。
- 这种双模型设计在所有算法分册中独一无二——它的教学目的是让读者直观对比：保留 2 个主成分够不够？再加 1 个主成分增进了多少结构？
- 对当前数据（3 个真实方向），2D PCA 已经捕获约 2/3 的真实结构，3D PCA 几乎完整——这个对比本身就回答"降多少维最合适"。

## 6. 与 LDA 的直觉对比

| 维度 | PCA | LDA |
|---|---|---|
| 核心问题 | 哪个方向方差最大？ | 哪个方向类别最可分？ |
| 标签 | 不需要——只用特征 | 必须——标签定义类别结构 |
| 投影效果 | 保留最多方差，但类别可能混在一起 | 保留最多类别差异，但可能丢失部分方差结构 |
| 维数限制 | 最多到 $\min(d, N)$——可自由选择 | 最多到 $K-1$——受类别数约束 |
| 理想场景 | 数据压缩、去噪、探索性可视化、特征提取 | 分类预处理、判别式特征提取 |
| 特征值含义 | 该方向的投影方差 | 该方向的类间/类内散度比 |
| 数据要求 | 任何高维数据 | 必须有类别标签 |

### 理解重点

- PCA 和 LDA 不是"谁更好"——如果任务是压缩和可视化，PCA 更通用；如果任务是为分类做特征预处理，LDA 更合适。
- 一个有用的思维实验：在当前 PCA 合成数据上跑 LDA（用伪标签）——你会发现 LDA 在放大类间差异，而 PCA 在保留全局结构。目标不同，答案不同。

## 可视化

![二维降维结果](https://img.yumeko.site/file/articles/ML/pca/dimensionality_2d.png)

![三维降维结果](https://img.yumeko.site/file/articles/ML/pca/dimensionality_3d.png)

## 常见坑

1. 把 PCA 当成"降维并分类"的工具——它只做降维，不关心类别。
2. 忘记 PCA 对特征尺度敏感——不标准化的 PCA 是"尺度最大的特征主导的 PCA"。
3. 忽略 `label` 在 PCA 和 LDA 中角色的本质差异——一个是着色工具，一个是训练输入。
4. 误以为 `n_components` 越多越好——当数据固有秩较小时，增加主成分只引入噪声方差。

## 小结

- PCA 的直觉核心是无监督方差压缩：找数据变化最大的方向 → 保留前几个主成分 → 用最少维度保留最多信息。
- 当前低秩合成数据（3 个真实方向隐藏在 10 维中）是展示 PCA 降维价值的理想教学数据。
- PCA 与 LDA 在直觉上截然相反：一个从"数据自身变化"出发找最大方差方向，一个从"标签引导"出发找最可分方向——选哪个取决于你是想压缩数据还是想区分类别。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `PCA`。
2. 理解 `PCA` 的核心构造器参数（`n_components`、`svd_solver`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`components_`（主成分方向）、`explained_variance_ratio_`（解释方差比）、`singular_values_`（奇异值）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.decomposition.PCA` 模型，打印解释方差比日志 |
| `PCA(...)` | 类 | scikit-learn 提供的主成分分析器——通过 SVD 寻找方差最大的正交方向 |
| `model.fit(X_train)` | 方法 | 学习主成分方向——无监督，不接收标签 |
| `model.components_` | 属性 | 主成分方向矩阵——将原始特征空间映射到主成分空间的线性变换 |
| `model.explained_variance_ratio_` | 属性 | 各主成分的解释方差占比——反映每个方向的重要性 |
| `model.transform(X)` | 方法 | 将数据投影到主成分空间——生成降维后的坐标 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(400, 10)$ | 标准化后的特征矩阵，传入 `PCA.fit()` | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前 2D 模型为 `2`，3D 模型为 `3` | `2`、`3`、`5` |
| `svd_solver` | `str` | SVD 求解器。`'auto'`（默认）自适应选择最优实现：小数据用 `'full'`，大数据用 `'randomized'` | `'auto'`、`'full'`、`'randomized'`、`'arpack'` |
| `random_state` | `int` | 随机种子——`'randomized'` 求解器时需要，保证结果可复现。默认 `42` | `42` |
| 返回值 | `PCA` | 已完成 `fit()` 的模型对象，含 `components_`、`explained_variance_ratio_` 等 | — |

### 示例代码

```python
from model_training.dimensionality.pca import train_model

model = train_model(X_scaled, n_components=2)
```

### 理解重点

- 和 LDA 分册不同，`train_model(...)` **没有 `y_train` 参数**——PCA 是无监督算法，不需要标签。
- `n_components=2` 和 `n_components=3` 分别在流水线中用于训练两个独立模型——这在所有算法分册中独一无二。
- `train_model(...)` 是对 `sklearn.decomposition.PCA` 的薄封装——算法本体是 scikit-learn 基于 SVD 的高效实现。

## 2. `PCA` 构造器参数

### 参数速览

适用 API：`PCA(n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的主成分数。无类别约束（与 LDA 的 $K-1$ 上限不同），可从 1 到 $\min(d, N)$ 自由选择 | `2`、`3`、`5` |
| `svd_solver` | `str` | SVD 求解器。`'auto'`（默认）自适应选择；`'full'` 完整 SVD（精确但慢）；`'randomized'` 随机化 SVD（大数据快）；`'arpack'` 只求前 $q$ 个特征对 | `'auto'`、`'full'`、`'randomized'`、`'arpack'` |
| `random_state` | `int` | 随机种子——`'randomized'` 和 `'arpack'` 求解器需要。默认 `42` | `42` |
| `whiten` | `bool` | 是否白化——使各主成分方差归一化。默认 `False` | `False`、`True` |
| `tol` | `float` | `'arpack'` 求解器的收敛容忍度。默认 `0.0` | `0.0`、`1e-6` |
| `iterated_power` | `int` 或 `'auto'` | `'randomized'` 求解器的幂迭代次数。默认 `'auto'` | `'auto'`、`5` |
| `n_oversamples` | `int` | `'randomized'` 求解器的过采样数。默认 `10` | `10`、`20` |
| `power_iteration_normalizer` | `str` | `'randomized'` 求解器的幂迭代归一化方式。默认 `'auto'` | `'auto'`、`'QR'`、`'LU'`、`'none'` |

### 示例代码

```python
model = PCA(
    n_components=2,
    svd_solver="auto",
    random_state=42,
)
model.fit(X_train)
```

### 理解重点

- PCA 的核心参数是 `n_components`——它直接决定降维后的维数和保留的信息量。与 LDA 不同，PCA 没有 $K-1$ 上限。
- `svd_solver='auto'` 是大多数情况下的最佳选择——scikit-learn 会根据数据大小自动选择 full / randomized / arpack。
- PCA 的 `fit()` 是解析求解（SVD）——与 KMeans（迭代优化）和 DBSCAN（密度扩展）在计算特征上截然不同，与 LDA 类似（都是特征分解家族）。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `components_` | `ndarray`，形状 `(n_components, n_features)` | 主成分方向 $\mathbf{u}_1, \dots, \mathbf{u}_q$ | 将 10 维特征映射到主成分空间的线性变换——每行是一个主成分方向 |
| `explained_variance_` | `ndarray`，形状 `(n_components,)` | $\lambda_k$ | 各主成分的方差（特征值）——反映每个方向捕获的绝对变化量 |
| `explained_variance_ratio_` | `ndarray`，形状 `(n_components,)` | $\lambda_k / \sum_j \lambda_j$ | 各主成分的解释方差占比——反映每个方向的相对重要性 |
| `singular_values_` | `ndarray`，形状 `(n_components,)` | $\sigma_k$ | 各主成分对应的奇异值——$\sigma_k^2/N = \lambda_k$ |
| `mean_` | `ndarray`，形状 `(n_features,)` | $\bar{\mathbf{x}}$ | 训练数据的均值向量——`transform()` 时用于中心化 |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `10` |
| `n_samples_` | `int` | 样本数 $N$ | 训练时的样本数，当前为 `400` |

### 示例代码

```python
print(f"n_components: {n_components}")
print(f"解释方差比: {model.explained_variance_ratio_.round(4)}")
print(f"累计解释方差: {model.explained_variance_ratio_.sum():.4f}")
```

### 理解重点

- `components_` 是 PCA 最有教学意义的属性——它把"主成分方向"这一概念直接映射为可观察的线性变换矩阵。
- `explained_variance_ratio_` 是 PCA 的核心量化输出——它直接告诉你每个主成分有多重要。当前源码打印到 4 位小数。
- 与 LDA 的关键对比：PCA 有 `components_`（主成分方向）、`singular_values_`（奇异值），LDA 有 `scalings_`（判别方向）、`means_`（类均值）。名称不同，数学含义也不同。

## 4. `transform()` ：从模型训练到降维输出的桥梁

### 参数速览

适用方法：`model.transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n, 10)` | 经过同一 `scaler` 标准化后的特征矩阵 | `X_scaled` |
| 返回值 | `ndarray`，形状 `(n, n_components)` | 投影到主成分空间后的坐标——$\mathbf{X} \cdot \text{components\_}^T$ | `X_transformed` |

### 示例代码

```python
X_transformed = model.transform(X_scaled)
```

### 理解重点

- `fit()` 学习主成分方向（`components_`），`transform()` 执行投影——两者分离的设计使模型可以对新数据重复投影。
- 流水线中 `X_transformed` 是 `plot_dimensionality(...)` 的直接输入——它是训练和可视化的桥梁。
- 与 LDA 的 `transform()` 语法相同、语义不同——PCA 投影到方差最大方向，LDA 投影到类别最可分方向。

## 5. 训练阶段的工程封装

除了 `PCA(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察 SVD 求解耗时——通常极快（毫秒级） |
| `n_components` 日志 | 确认当前保留的主成分数 |
| `explained_variance_ratio_` 日志 | 打印各主成分解释占比（4 位小数）和累计值 |
| `timer(...)` 上下文 | 单独测量 `fit()` 阶段的耗时 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键统计量。
- `explained_variance_ratio_` 是 PCA 独有的日志输出——它直接反映方差保留比例，是选择 `n_components` 的基础。
- PCA 与 LDA 在日志输出结构上相似（都有解释方差比 + 累计值），但数值含义不同——当前数据前 3 个主成分累计方差应接近但小于 100%（有噪声），LDA 的累计必然是 100%（$K-1$ 上限）。

## 常见坑

1. 误以为 `train_model(...)` 需要传入 `y_train`——PCA 是无监督算法，不接受标签。
2. 忽略 `n_components` 与数据固有秩的关系——当 `n_components` 超过固有秩时，多余的主成分只贡献噪声方差。
3. 把 `components_` 当成 LDA 的 `scalings_`——两者优化目标不同，方向含义不同。
4. 忘记 PCA 的 `explained_variance_ratio_` 总和必然 < 100%（有噪声时），而 LDA 的累计值在 `n_components=K-1` 时必为 100%。

## 小结

- `train_model(...)` 是本仓库 PCA 的核心训练入口，是对 `sklearn.decomposition.PCA` 的薄封装。
- `PCA` 的核心参数是 `n_components`（保留主成分数）和 `svd_solver`（SVD 求解器）——前者决定信息保留量，后者决定计算路径。
- 训练完成后的核心属性：`components_`（主成分方向）、`explained_variance_ratio_`（解释方差比）、`singular_values_`（奇异值）——三者分别回答了"哪个方向""多重要""多大变化"。
- PCA 有 `components_`、有 `explained_variance_ratio_`、无监督、维数自由选择——这四点构成了它与 LDA 最核心的工程差异。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 PCA 流水线从数据复制到 2D/3D 降维图输出的完整步骤。
2. 理解 PCA 无监督训练、无切分、`transform()` 为输出的工程特征——与 LDA 既有相似又有本质差异。
3. 理解"双模型"设计（分别训练 2D 和 3D PCA）的教学意图——对比不同降维程度下的信息保留。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pca_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `data.drop(columns=["label"])` | 操作 | 去掉伪标签列，保留 10 个特征作为训练输入 |
| `StandardScaler().fit_transform(X)` | 方法 | 对全量特征做一致性标准化——协方差矩阵计算的前置条件 |
| `train_model(X_scaled, n_components=2)` | 函数 | 训练 2D PCA 模型——无监督，不传标签 |
| `train_model(X_scaled, n_components=3)` | 函数 | 训练 3D PCA 模型——第二个独立模型，与 2D 模型互不依赖 |
| `model.transform(X_scaled)` | 方法 | 将 10 维特征投影到主成分空间——生成降维坐标 |
| `plot_dimensionality(...)` | 函数 | 绘制降维后的散点图——2D 和 3D 分别调用 |

## 1. 流水线起点：复制数据并拆出特征与伪标签

### 示例代码

```python
data = pca_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values
```

### 理解重点

- `.copy()` 确保后续处理不修改全局 `pca_data`。
- `label` 被单独保存为 `y`——它**仅用于**最终的 `plot_dimensionality(...)` 着色，不参与 `train_model()`。
- 与 LDA 流水线最关键的区别：`y` 在这里不流入 `train_model()`，PCA 的训练完全不接触标签。

## 2. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(400, 10)$ | 去掉 `label` 后的全量特征矩阵 | `X` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_scaled` |

### 示例代码

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 理解重点

- PCA 流水线**没有**训练/测试切分——当前实现为教学型简化，直接在全量数据上训练和投影。
- `fit_transform` 直接在全量数据上计算统计量并变换——目标是展示整体数据结构而非评估泛化能力。
- 标准化是必须的——协方差矩阵对特征尺度高度敏感。不标准化意味着"尺度最大的特征主导所有主成分方向"。

## 3. 第一阶段：训练 2D PCA 并生成 2D 图

### 参数速览

适用函数：`train_model(X_scaled, n_components=2)` → `model.transform(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 10)$ | 标准化后的特征矩阵——2D PCA 的输入 | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前为 `2` | `2` |
| `X_transformed` | `ndarray`，形状 $(400, 2)$ | 投影到 2 维主成分空间后的坐标 | — |

### 示例代码

```python
# 训练 2D PCA
model = train_model(X_scaled, n_components=2)

# 投影到 2D 主成分空间
X_transformed = model.transform(X_scaled)

# 绘制 2D 降维图
plot_dimensionality(
    X_transformed,
    y=y,
    explained_variance_ratio=model.explained_variance_ratio_,
    title="PCA 降维 (2D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="2d",
)
```

### 理解重点

- `PCA.fit(X_scaled)` 内部执行 SVD 分解 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，取 $\mathbf{V}$ 的前 2 行作为 `components_`。
- 2D 投影图展示的是数据方差最大的两个方向——对于当前低秩数据（3 个真实方向），它能展示约 2/3 的真实结构。
- `plot_dimensionality` 的 `mode='2d'` 参数决定输出 2D 散点图，轴标签为 `PC1 (xx.x%)` 和 `PC2 (xx.x%)`。

## 4. 第二阶段：训练 3D PCA 并生成 3D 图

### 参数速览

适用函数：`train_model(X_scaled, n_components=3)` → `model_3d.transform(X_scaled)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_scaled` | `ndarray`，形状 $(400, 10)$ | 标准化后的特征矩阵——3D PCA 的输入（与 2D PCA 共享同一份标准化数据） | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。当前为 `3` | `3` |
| `X_3d` | `ndarray`，形状 $(400, 3)$ | 投影到 3 维主成分空间后的坐标 | — |

### 示例代码

```python
# 训练 3D PCA（第二个独立模型）
model_3d = train_model(X_scaled, n_components=3)

# 投影到 3D 主成分空间
X_3d = model_3d.transform(X_scaled)

# 绘制 3D 降维图
plot_dimensionality(
    X_3d,
    y=y,
    explained_variance_ratio=model_3d.explained_variance_ratio_,
    title="PCA 降维 (3D)",
    dataset_name=DATASET,
    model_name=MODEL,
    mode="3d",
)
```

### 理解重点

- **`model_3d` 是第二个独立模型**——它不是从 `model`（2D PCA）复用或扩展而来的，而是重新创建 `PCA(n_components=3)` 并重新 `fit()`。
- 3D 投影图展示数据方差最大的三个方向——对于当前低秩数据（3 个真实方向），它能几乎完整展示全部真实结构。
- 这种"先训练 2D、再训练 3D"的双模型设计，在 PCA 分册中旨在展示：**增加一个主成分（从 2D 到 3D）新增了多少结构信息？**

## 5. 2D 与 3D 模型的关系

### 理解重点

- 两个模型分别 `fit()` 了两次——从数学上讲，2D PCA 的前两个主成分与 3D PCA 的前两个主成分完全相同（都是相同的 $\mathbf{V}$ 的前两行）。工程上分开训练只是为了代码清晰。
- `X_transformed`（2D）的列是 `X_3d`（3D）的前两列——3D 投影完全包含 2D 投影的信息。
- 2D 累计解释方差 < 3D 累计解释方差——增加第 3 个主成分必然带来信息增益，但增益的幅度取决于数据的固有秩。

## 6. 用伪代码看完整流程

```python
data = pca_data.copy()
X = data.drop(columns=["label"])
y = data["label"].values

X_scaled = StandardScaler().fit_transform(X)

# 第一阶段：2D PCA
model = train_model(X_scaled, n_components=2)
X_2d = model.transform(X_scaled)
plot_dimensionality(X_2d, y=y, explained_variance_ratio=..., mode="2d")

# 第二阶段：3D PCA
model_3d = train_model(X_scaled, n_components=3)
X_3d = model_3d.transform(X_scaled)
plot_dimensionality(X_3d, y=y, explained_variance_ratio=..., mode="3d")
```

### 理解重点

- 流水线的主线非常清楚：取数 → 标准化 → 2D PCA 训练+投影+画图 → 3D PCA 训练+投影+画图。
- 这条链路里最关键的中间变量是：`X_scaled`（标准化特征）、2D PCA `model`（含 `components_` 和 `explained_variance_ratio_`）、3D PCA `model_3d`、两组投影结果和伪标签 `y`。
- 与 LDA 流水线最直观的区别：PCA 只画降维图（2D+3D），LDA 只画 2D 图——流程结构相似但输出维度不同。

## 常见坑

1. 以为 2D 和 3D 是同一个模型的不同 `n_components` 调用——实际上是两个独立 `PCA` 实例分别 `fit()`。
2. 忘记 `label` 在 PCA 中只用于着色——与 LDA 流水线中将 `y` 传入 `train_model()` 形成鲜明对比。
3. 误以为 PCA 流水线有分类评估步骤——PCA 是无监督降维，输出只有降维散点图。
4. 把 `PC1`、`PC2`、`PC3` 的标签当成分类标签——它们是主成分编号，不是类别编号。

## 小结

- 当前 PCA 流水线分为两阶段：2D PCA（训练+投影+画图）→ 3D PCA（训练+投影+画图）。这是所有算法分册中独一无二的双模型设计。
- `train_model` 在两次调用中分别传入 `n_components=2` 和 `n_components=3`——两次都只传 `X_scaled`（无监督）。
- 与 LDA 流水线的核心差异：无监督（无 `y` 入 `fit`）、维数自由（不受 $K-1$ 约束）、两阶段双模型、2D+3D 双图输出。
- 与分类分册的核心差异：输出是低维坐标而非类别预测、可视化是降维散点图而非混淆矩阵/ROC。

# 评估与诊断

## 本章目标

1. 明确当前仓库 PCA 实现的评估手段——`explained_variance_ratio_` 日志 + 2D/3D 降维散点图。
2. 理解如何从"方差占比逐主成分递减"的模式判断数据的固有秩。
3. 理解 PCA 评估与 LDA 评估和分类评估的本质差异——关注方差保留而非类别分离或分类精度。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `explained_variance_ratio_` | 指标 | 各主成分的解释方差占比——反映每个方向捕获了多少总方差 |
| 累计解释方差 | 派生量 | 前 $q$ 个主成分总共保留了多少方差——决定"保留几个主成分够用" |
| `plot_dimensionality(...)` | 函数 | 生成 2D/3D 降维散点图（按伪标签着色，轴标注解释占比） |
| 方差断层 | 诊断信号 | `explained_variance_ratio_` 在某处突然大幅下降——暗示数据的固有秩就在那个位置 |
| 2D vs 3D 对比 | 教学诊断 | 对比 2D 和 3D 降维图，观察增加一个主成分增添了多少结构信息 |

## 1. 当前仓库的评估入口

PCA 的评估与分类分册截然不同——没有混淆矩阵、没有 ROC 曲线、没有准确率。评估依赖两种手段：

1. **`explained_variance_ratio_` 日志** —— 打印各主成分的解释方差占比和累计值，量化信息保留
2. **2D/3D 降维散点图** —— 将 10 维数据投影到 2D 或 3D 主成分空间，按伪标签着色，直观观察数据结构

### 示例代码

```python
# 2D 图
plot_dimensionality(X_transformed, y=y, explained_variance_ratio=model.explained_variance_ratio_,
                    title="PCA 降维 (2D)", dataset_name=DATASET, model_name=MODEL, mode="2d")

# 3D 图
plot_dimensionality(X_3d, y=y, explained_variance_ratio=model_3d.explained_variance_ratio_,
                    title="PCA 降维 (3D)", dataset_name=DATASET, model_name=MODEL, mode="3d")
```

### 理解重点

- 降维评估不能像分类那样用 `accuracy`——PCA 的目标是保留方差，而非预测类别。
- 当前评估方法对教学场景来说足够直观：`explained_variance_ratio_` 的数值模式直接反映数据的固有秩结构，降维图提供了视觉验证。
- 同时输出 2D 和 3D 两张图是 PCA 分册独有的评估方式——这一设计旨在让读者直接对比不同降维程度下的结构保留效果。

## 2. `explained_variance_ratio_` 能告诉我们什么

### 低秩数据的典型输出

运行当前流水线，终端通常输出类似：

```
2D PCA:
解释方差比: [0.4523 0.3018]
累计解释方差: 0.7541

3D PCA:
解释方差比: [0.4523 0.3018 0.1825]
累计解释方差: 0.9366
```

### 理解重点

- 前两个主成分的解释方差比（45% + 30% = 75%）远大于后续——这说明方差集中在少数几个方向上，数据确实具有低秩结构。
- 第 3 个主成分新增约 18% 的方差——3D 投影（累计 94%）几乎捕获了全部信号，剩余的 7 个主成分仅贡献约 6% 的噪声方差。
- **方差断层**出现在第 3 和第 4 个主成分之间（从 ~18% 骤降至 ~1-2%）——这就是数据的固有秩。PCA 用纯数据驱动的方式揭示了 10 维表面下隐藏的 3 维真实结构。
- 与 LDA 同属性的对比：LDA 的 `explained_variance_ratio_` 在 `n_components=K-1` 时累计必为 100%，因为它只有 $K-1$ 个非零特征值；PCA 的累计值反映了信号与噪声的相对比例。

## 3. 2D 和 3D 降维图分别能观察什么

### 参数速览

适用函数：`plot_dimensionality(X_transformed, y=y, explained_variance_ratio=evr, mode='2d'|'3d')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_transformed` | `ndarray`，形状 `(400, 2)` 或 `(400, 3)` | PCA 降维后的坐标 | 2D 或 3D 投影结果 |
| `y` | `ndarray`，形状 `(400,)` | 伪标签——仅用于散点着色 | `y` |
| `explained_variance_ratio` | `ndarray` | 各主成分贡献占比——标注在坐标轴上（如 `PC1 (45.2%)`） | `model.explained_variance_ratio_` |

### 理解重点

- **2D 图（`mode='2d'`）观察**：
  - 前两个主成分是否已经展现了数据的主要结构？伪标签在 2D 空间中是否形成可辨识的分布模式？
  - 数据点在 PC1-PC2 平面上的散布形状——各向同性还是各向异性？
- **3D 图（`mode='3d'`）观察**：
  - 新增的 PC3 维度带来了什么额外结构？是否让数据分布从"扁平"变得更"立体"？
  - 对比 2D 图——哪些在 2D 中重叠或模糊的结构在加入第 3 维后变得清晰？
- 坐标轴标注 `PC1 (45.2%)` 等——读者可以一眼看出每个轴的"重要性"。

## 4. PCA 与 LDA 在相同合成数据上的评估对比

| 评估维度 | PCA | LDA |
|---|---|---|
| 核心指标 | `explained_variance_ratio_`——方差占比 | `explained_variance_ratio_`——判别能力占比 |
| 指标范围 | 累计 < 100%（有噪声） | 累计 = 100%（$n_components=K-1$ 时） |
| 降维图数量 | 2 张（2D + 3D） | 1 张（2D） |
| 降维图观察重点 | 数据几何结构是否被保留 | 类别在判别空间中是否被拉开 |
| 轴标签前缀 | `PC`（Principal Component） | `LD`（Linear Discriminant） |
| 特有诊断 | 方差断层——判断数据固有秩 | 类间分离度——判断判别方向有效性 |

### 理解重点

- PCA 和 LDA 共享同一个可视化函数 `plot_dimensionality`，同名的 `explained_variance_ratio_` 属性——但语义截然不同。
- PCA 问"数据结构保留了多少"，LDA 问"类别分离增强了多少"——两套评估哲学对应两种降维目标。

## 5. 当前实现中尚未纳入的评估手段

| 评估手段 | 说明 | 当前未使用的原因 |
|---|---|---|
| 碎石图（Scree Plot） | 绘制特征值随主成分编号的折线图，直观展示方差断层 | 当前侧重日志输出和降维图，碎石图是自然扩展 |
| 累计方差曲线 | 绘制累计解释方差随主成分数的变化，辅助选 $q$ | 同碎石图——日志中的累计值已提供核心信息 |
| 重构误差 | 从降维表示重构原始数据，计算 $\|\mathbf{X} - \hat{\mathbf{X}}\|^2$ | 教学型仓库优先展示直观的降维可视化 |
| 降维后分类器精度对比 | 在 PCA 投影后的特征上训练分类器 | 当前分册定位为降维而非分类，且伪标签无真实分类意义 |

### 理解重点

- 当前流水线不显式计算这些指标——文档可以提到它们是扩展方向，但不能写成"当前源码已在计算"。
- 对于教学型仓库，`explained_variance_ratio_` 日志 + 2D+3D 降维图已经能有效支撑 PCA 的诊断需求。

## 评估图表

![二维降维结果](https://img.yumeko.site/file/articles/ML/pca/dimensionality_2d.png)

![三维降维结果](https://img.yumeko.site/file/articles/ML/pca/dimensionality_3d.png)

## 常见坑

1. 把 PCA 的评估写成 LDA 的"看类别分离度"——PCA 评估看方差保留和几何结构，不看类别。
2. 看到累计方差 < 100% 就认为模型有问题——噪声数据下累计方差必然 < 100%，100% 反而暗示数据无噪声。
3. 只用 `explained_variance_ratio_` 评估而不看降维图——数值和视觉互为补充。
4. 忽略 2D 和 3D 图之间的对比价值——这正是 PCA 分册双模型设计的核心教学意图。

## 小结

- 当前仓库对 PCA 的评估简洁而直观：`explained_variance_ratio_` 日志看方差保留，2D/3D 降维图看结构呈现。
- PCA 没有混淆矩阵、没有 ROC、没有准确率——这些是监督分类的评估手段。当前分册关注的是"降维后数据的主要变化模式是否被保留"。
- PCA 评估与 LDA 评估的核心差异：前者问"方差保留了多少"，后者问"类别分开了吗"——同名属性、同名函数、截然不同的观察重点。

# 工程实现

## 本章目标

1. 从工程角度看清 PCA 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和降维可视化分别负责什么。
3. 理解 PCA 工程实现的独特之处——双模型（2D+3D）、无监督、与 LDA 共用可视化模块。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/dimensionality.py` | `DimensionalityData.pca()` 生成低秩高维合成数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `pca_data` |
| 训练封装 | `model_training/dimensionality/pca.py` | 构建并训练 `PCA`，打印解释方差比日志 |
| 流水线入口 | `pipelines/dimensionality/pca.py` | 组织标准化、2D/3D PCA 训练、投影与降维可视化 |
| 降维可视化 | `result_visualization/dimensionality_plot.py` | 绘制降维后的 2D/3D 散点图（按类别着色，轴标注解释占比） |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.dimensionality.pca
```

### 理解重点

- 这个命令串起当前 PCA 分册中最核心的工程流程。
- 依次完成：数据复制 → 剥离 `label` → 全量标准化 → 2D PCA `fit(X)`（SVD）→ `transform(X)` → 2D 降维图 → 3D PCA `fit(X)` → `transform(X)` → 3D 降维图。
- PCA 流水线独有的两阶段结构：先训练 2D 模型并画图，再训练 3D 模型并画图。这与 LDA 的单模型单图结构不同。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征与伪标签
    data = pca_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"].values

    # 2. 全量标准化——无切分（教学型简化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 第一阶段：2D PCA
    model = train_model(X_scaled, n_components=2)
    X_transformed = model.transform(X_scaled)
    plot_dimensionality(X_transformed, y=y, explained_variance_ratio=...,
                        title="PCA 降维 (2D)", dataset_name=DATASET,
                        model_name=MODEL, mode="2d")

    # 4. 第二阶段：3D PCA（第二个独立模型）
    model_3d = train_model(X_scaled, n_components=3)
    X_3d = model_3d.transform(X_scaled)
    plot_dimensionality(X_3d, y=y, explained_variance_ratio=...,
                        title="PCA 降维 (3D)", dataset_name=DATASET,
                        model_name=MODEL, mode="3d")
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的 SVD 分解在 `PCA.fit()` 中。
- 数据流是单向且分两支：标准化数据 → 2D PCA `fit`+`transform` → 2D 图，然后再 → 3D PCA `fit`+`transform` → 3D 图。
- 与分类流水线的核心差异：
  - **无 `train_test_split`**——当前实现为教学型简化
  - **无 `predict()` 调用**——PCA 是降维工具，输出是 `transform()` 而非类别标签
  - **双模型双图**（2D+3D）而非分类的四图（混淆矩阵+ROC+决策边界+学习曲线）
- 与 LDA 流水线的核心差异：
  - **`fit()` 不传 `y`**——无监督 vs 有监督
  - **双模型**（2D+3D）vs 单模型（2D only）
  - **`n_components` 无上限**（不受 $K-1$ 约束）

## 3. 训练模块负责什么

`model_training/dimensionality/pca.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `PCA(n_components=n_components, svd_solver='auto', random_state=42)` 实例
2. 调用 `model.fit(X_train)`——SVD 分解（无监督，不传标签）
3. 打印 `n_components`、`explained_variance_ratio_` 和累计解释方差
4. 返回训练完成的模型对象

### 参数速览

适用函数：`train_model(X_train, n_components=2, svd_solver='auto', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的特征矩阵，传入 `PCA.fit()` | `X_scaled` |
| `n_components` | `int` | 保留的主成分数。2D 模型为 `2`，3D 模型为 `3` | `2`、`3` |
| `svd_solver` | `str` | SVD 求解器。默认 `'auto'`——自动选择最优实现 | `'auto'`、`'full'`、`'randomized'` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| 返回值 | `PCA` | 已完成 `fit()` 的模型对象，含 `components_`、`explained_variance_ratio_` 等 | — |

### 理解重点

- PCA 的 `fit()` 是无监督的——不接收标签参数。这与 LDA 分册形成鲜明对比（LDA 必须传 `y`），与聚类分册一致。
- 训练日志中的 `explained_variance_ratio_` 是 PCA 的核心输出——它直接反映各主成分的相对重要性。
- 当前流水线调用了两次 `train_model()`——分别创建和训练了两个独立的 PCA 实例。

## 4. 可视化模块负责什么

### 模块职责

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 降维可视化 | `plot_dimensionality(...)` | `X_transformed`（投影坐标）、`y`（着色标签）、`explained_variance_ratio`（轴标注）、`mode`（`'2d'` 或 `'3d'`） | 2D 或 3D 降维散点图（PNG），坐标轴标注解释占比 |

### 理解重点

- `plot_dimensionality(...)` 是当前 PCA 流水线中**唯一**的可视化模块——与分类分册的 4 种评估函数形成鲜明对比。
- 与 LDA **共用同一个可视化函数**——区别在于传入的数据含义不同（主成分投影 vs 判别投影），以及 `mode` 参数（PCA 用了 2D 和 3D 两种模式，LDA 只用 2D）。
- 轴标签前缀为 `PC`（Principal Component）——如 `PC1 (45.2%)`。LDA 使用时为 `LD`（Linear Discriminant）。
- 3D 模式下使用 matplotlib 的 `projection='3d'` 创建三维坐标轴。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `pca_data` | `data_generation/dimensionality.py` | `pipelines/dimensionality/pca.py` |
| `y` | `data["label"]` 提取 | `plot_dimensionality`（仅着色，两次调用） |
| `X_scaled` | `StandardScaler` | `train_model`（2D）、`train_model`（3D）、`model.transform`（2D）、`model_3d.transform`（3D） |
| `model`（2D PCA） | `train_model(X_scaled, n_components=2)` | `model.transform`、`plot_dimensionality`（2D） |
| `model_3d`（3D PCA） | `train_model(X_scaled, n_components=3)` | `model_3d.transform`、`plot_dimensionality`（3D） |
| `X_transformed`（2D） | `model.transform(X_scaled)` | `plot_dimensionality`（2D） |
| `X_3d`（3D） | `model_3d.transform(X_scaled)` | `plot_dimensionality`（3D） |
| 图片产物 | `plot_dimensionality(...)` | `outputs/pca/` 目录 |

### 理解重点

- 数据依赖关系中有两个并行的模型分支——这是 PCA 流水线独有的结构。LDA 只有一个分支。
- `y` 的流向是单向且单一的——只到 `plot_dimensionality`，不进任何模型。这与 LDA（`y` 同时流入 `train_model` 和 `plot_dimensionality`）形成鲜明对比。
- `X_scaled` 被 4 个下游节点共享（2D 训练、3D 训练、2D 投影、3D 投影）——标准化是整个流水线的计算基础。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `PCA 降维流水线` | 在终端中定位当前运行入口 |
| 训练日志（2D） | 训练耗时、`n_components=2`、`explained_variance_ratio_`（各方向 + 累计） | 查看 2D PCA 的方差保留情况 |
| 训练日志（3D） | 训练耗时、`n_components=3`、`explained_variance_ratio_`（各方向 + 累计） | 查看 3D PCA 的方差保留情况——与 2D 对比 |
| 2D 降维图 | `outputs/pca/pca_dim_2d.png` | 2D 主成分空间中的样本分布 |
| 3D 降维图 | `outputs/pca/pca_dim_3d.png` | 3D 主成分空间中的样本分布 |

### 理解重点

- 输出是所有分册中最丰富的——2 组日志 + 2 张图（其他降维分册只有 1 组日志 + 1 张图）。
- 两次 `explained_variance_ratio_` 日志输出的前 2 个主成分占比应一致——因为 2D PCA 和 3D PCA 的前两个主成分方向相同。
- 2D 和 3D 降维图的对比是最核心的教学产出——它直接展示了"增加主成分如何提升结构保留"。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/dimensionality/pca.py` — 入口，双模型双图结构一目了然
2. 再看 `model_training/dimensionality/pca.py` — 训练封装，理解无监督 `fit(X)` 和日志输出
3. 再看 `result_visualization/dimensionality_plot.py` — 降维散点图绘制逻辑（含 2D/3D 分支和 `explained_variance_ratio` 轴标注）
4. 最后回到 `data_generation/dimensionality.py` — 理解低秩合成数据的构造方式

### 理解重点

- 从入口看整体流程（特别是双模型结构），再下钻到训练和可视化细节，阅读成本最低。
- PCA 的调用链与 LDA 几乎一致（同一套模块分层），差异集中在：`fit()` 有无 `y`、是否双模型、`mode` 是否含 `'3d'`。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/articles/ML/pca/dimensionality_2d.png)

## 常见坑

1. 把 2D 和 3D PCA 当成同一个模型——它们是两个独立 `PCA` 实例，各自 `fit()` 了一次。
2. 期待当前流水线有 `train_test_split` 或 `predict()`——PCA 是降维工具，输出低维坐标。
3. 忽略 `y` 在 PCA 流水线中只到可视化、不进模型——这是无监督降维与有监督降维（LDA）在数据流上的根本差异。
4. 忘记可视化模块是 PCA 和 LDA 共用的——轴标签前缀 `PC` vs `LD` 由调用者决定（在 `plot_dimensionality` 内部是写死的，目前用 `PC`）。

## 小结

- 当前 PCA 工程实现采用与 LDA 一致的模块分层，但具有独特的双模型结构：数据生成 → 训练封装（无监督，两次调用）→ 流水线编排（两阶段）→ 可视化（2D+3D）。
- `run()` 负责串联两阶段流水线，`train_model(...)` 负责 SVD 分解（仅 `fit(X)`），`plot_dimensionality(...)` 负责降维可视化（2D 和 3D 两种模式）。
- PCA 在工程上最不同于 LDA 的地方：无监督（`fit` 无 `y`）、双模型双图、`n_components` 无 $K-1$ 约束。
- PCA 在工程上最不同于分类算法的地方：输出是 `transform()` 而非 `predict()`、降维散点图而非分类评估图。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 PCA 实现。
2. 给出继续深入阅读主成分分析与相关数据工具的可靠入口。

## 自检题

1. 为什么 PCA 的 `fit()` 不需要 `y` 参数，而 LDA 需要？PCA 的优化目标（最大化投影方差）为什么天然不依赖标签？
2. 当前 PCA 流水线为什么分别训练 2D 和 3D 两个独立模型？2D PCA 的前两个主成分与 3D PCA 的前两个主成分是否相同？为什么？
3. 当前数据是低秩合成结构（3 个真实方向隐藏在 10 维中）——这在 `explained_variance_ratio_` 的输出中如何体现？方差断层应该出现在第几个主成分之后？
4. `svd_solver='auto'` 与 `'full'`、`'randomized'`、`'arpack'` 有什么区别？为什么 `'auto'` 是大多数情况下的最佳选择？
5. PCA 的 `components_` 与 LDA 的 `scalings_` 在数学含义上有何不同？为什么不能用同一套术语描述它们？
6. 为什么标准化对 PCA 是硬性要求？如果去掉标准化，协方差矩阵会怎样被大尺度特征主导？
7. PCA 和 LDA 共用同一个 `plot_dimensionality` 函数——同名的 `explained_variance_ratio_` 属性在两种场景下语义有何不同？

## 练习方向

### 1. 改变 `n_components`

- 对 2D 模型：把 `n_components=2` 改成 `1`、`2`、`3`、`5`
- 观察变化：
  - `explained_variance_ratio_` 各值的变化
  - 累计解释方差从 ~75%（2D）→ ~94%（3D）→ ~98%（5D）
  - 降维图的视觉信息量变化
- 核心理解：`n_components` 增加带来的信息增益是边际递减的——这正是数据低秩特性的体现

### 2. 改变 `pca_n_informative`

- 修改 `data_generation/dimensionality.py` 中 `DimensionalityData` 的 `pca_n_informative`（`2`、`3`、`5`）
- 观察变化：
  - 方差断层从第几个主成分之后开始
  - `pca_n_informative=2` 时，前 2 个主成分累计方差是否更高
  - `pca_n_informative=5` 时，方差下降是否更平缓
- 核心理解：`pca_n_informative` 直接决定数据的固有秩——`explained_variance_ratio_` 的断层位置应与其对应

### 3. 改变 `pca_noise_std`

- 修改 `pca_noise_std`（`0.0`、`0.2`、`0.5`、`1.0`、`2.0`）
- 观察变化：
  - 无噪声（`0.0`）——前 3 个主成分累计方差 = 100%，后续主成分方差 = 0
  - 高噪声（`2.0`）——方差下降非常平缓，难以识别固有秩
  - 降维图上样本的散布程度——高噪声下结构被淹没
- 核心理解：噪声水平决定了 PCA 从数据中能否可靠地识别低秩结构

### 4. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用原始 `X` 训练
- 观察变化：
  - `components_` 方向是否变化——大尺度特征是否主导了第一主成分
  - `explained_variance_ratio_` 分布是否不同
- 体会：标准化确保每个特征在协方差矩阵中权重均等——PCA 结果反映数据的相关结构而非量纲差异

### 5. 用 PCA 降维后接 LDA 做对比

- 先用 PCA 降到 5 维，再在 PCA 投影特征上用 LDA 降到 2 维
- 对比直接在原始 10 维数据上用 LDA 降到 2 维
- 观察变化：
  - 两种路径下的判别投影图有何差异
  - PCA 作为 LDA 的预处理步骤，是否丢失了类别区分信息
- 核心理解：PCA 的方差最大化 + LDA 的判别最大化可以串联——这是特征工程的常见组合

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`PCA` | 完整构造器参数（`n_components`、`svd_solver`、`random_state`、`whiten`、`tol`、`iterated_power`、`n_oversamples`、`power_iteration_normalizer`）、属性（`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`mean_`、`n_features_in_`、`n_samples_`）与方法（`fit`、`transform`、`fit_transform`、`inverse_transform`、`get_covariance`、`get_precision`）说明 |
| 2 | scikit-learn 用户指南：Decomposing signals in components (matrix factorization problems) → PCA | PCA 原理、SVD 求解器选择指南、`n_components` 选择策略、增量 PCA（`IncrementalPCA`）和核 PCA（`KernelPCA`）的适用场景 |
| 3 | Jolliffe, I. T. and Cadima, J. (2016). *Principal component analysis: a review and recent developments*. Philosophical Transactions of the Royal Society A. | PCA 综述——从经典推导到稀疏 PCA、鲁棒 PCA 等现代变体，涵盖选主成分数量的多种准则 |
| 4 | Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 12: Continuous Latent Variables. | PCA 的概率视角——概率 PCA（PPCA）、EM 算法求解、与因子分析的关系，为理解贝叶斯 PCA 提供基础 |

- scikit-learn `PCA`：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn 用户指南 PCA：https://scikit-learn.org/stable/modules/decomposition.html#pca

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 PCA 分册的核心内容：
  - PCA 是无监督降维——`fit()` 不接收 `y`，优化目标是最大化投影方差
  - 主成分是协方差矩阵的特征向量——可通过 SVD 数值稳定地求解
  - 当前流水线有独特的双模型设计——分别训练 2D 和 3D PCA，对比不同降维程度的效果
  - `explained_variance_ratio_` 反映各主成分的方差占比——方差断层揭示数据的固有秩
  - 标准化对基于协方差矩阵的 PCA 是硬性要求——特征量纲差异会绑架主成分方向
  - PCA 的 `components_`（主成分方向）与 LDA 的 `scalings_`（判别方向）名称不同、数学含义不同
  - PCA 与 LDA 共用可视化模块 `plot_dimensionality`——同函数、同属性名、不同语义
  - `n_components` 在 PCA 中可自由选择（1 到 $\min(d,N)$），在 LDA 中受 $K-1$ 约束
  - 低秩合成数据（`base @ projection + noise`）是展示 PCA 优势的理想教学数据
