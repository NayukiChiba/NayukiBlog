---
title: 正则化回归
date: 2026-02-21
category: 机器学习/回归
tags:
  - Scikit-learn
description: 正则化回归的数学原理、L1/L2惩罚机制与多模型工程实现。
image: https://img.yumeko.site/file/blog/cover/1780581862162.webp
status: published
---

# 数学原理

## 本章目标

1. 理解正则化为什么在 OLS 损失函数中加入参数惩罚项。
2. 理解 Ridge（L2）、Lasso（L1）、ElasticNet（L1+L2）三种目标函数的数学差异。
3. 将数学公式中的 $\lambda$、$\rho$ 与源码中的 `alpha`、`l1_ratio` 精确对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| L2 正则化（Ridge） | 惩罚项 | $\lambda \|\mathbf{w}\|_2^2$——平方惩罚，系数整体收缩 |
| L1 正则化（Lasso） | 惩罚项 | $\lambda \|\mathbf{w}\|_1$——绝对值惩罚，系数可精确归零 |
| ElasticNet | 惩罚项 | $\lambda[\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2]$——L1+L2 混合 |
| `alpha` | 超参数 | 对应数学中的 $\lambda$——控制正则化总强度 |
| `l1_ratio` | 超参数 | 对应数学中的 $\rho$——控制 L1 在 ElasticNet 中的占比 |

## 1. 从 OLS 到正则化：为什么需要惩罚项

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| OLS 损失 | 数学表达式 | $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$——仅最小化残差平方和 | — |
| 正则化损失 | 数学表达式 | OLS 损失 $+ \lambda \cdot$ 惩罚项——在拟合与复杂度之间权衡 | — |
| $\lambda$ | 标量 | 正则化强度——$\lambda=0$ 退化为 OLS，$\lambda \to \infty$ 系数全部归零 | `alpha=0.15`（Lasso） |

### 理解重点

- OLS 的唯一目标是让训练误差最小——在高维或共线场景下，系数可能剧烈波动。
- 正则化在 OLS 损失上叠加系数惩罚——"你可以拟合数据，但要为系数过大付出代价"。
- 当前源码中 Lasso/Ridge/ElasticNet 的 `alpha` 值不同（0.15 / 2.0 / 0.2），说明不同惩罚类型需要不同的强度来展示典型行为。

## 2. Ridge 回归（L2 正则化）

### 参数速览

适用模型：`Ridge(alpha=2.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{Ridge}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$ | — |
| $\lambda$ | 标量 | L2 正则化强度 | `alpha=2.0` |
| $\|\mathbf{w}\|_2^2$ | 标量 | $\sum_{j=1}^d w_j^2$——所有系数的平方和 | — |
| 闭式解 | 数学表达式 | $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | — |

### 理解重点

- L2 惩罚对每个系数施加平方代价——大系数付出更大的代价，因此所有系数被**均匀收缩**。
- 闭式解中 $\lambda\mathbf{I}$ 使 $\mathbf{X}^T\mathbf{X}$ 始终可逆——这是 Ridge 处理共线性的数学基础。
- 平方惩罚在零点可微——系数可以趋于零但**不会精确为零**。
- 贝叶斯视角：Ridge 等价于对 $\mathbf{w}$ 施加高斯先验 $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \frac{\sigma^2}{\lambda}\mathbf{I})$ 后的 MAP 估计。

## 3. Lasso 回归（L1 正则化）

### 参数速览

适用模型：`Lasso(alpha=0.15, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{Lasso}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_1$ | — |
| $\lambda$ | 标量 | L1 正则化强度 | `alpha=0.15` |
| $\|\mathbf{w}\|_1$ | 标量 | $\sum_{j=1}^d \|w_j\|$——所有系数的绝对值之和 | — |
| 次梯度 | 数学表达式 | $\partial \|w_j\| = \text{sign}(w_j)$ 当 $w_j \neq 0$；$\partial \|w_j\| \in [-1, 1]$ 当 $w_j = 0$ | — |

### 理解重点

- L1 惩罚在原点**不可微**——正是这个不可微性使得系数可以被精确驱动到零。
- 次梯度条件：当某个 $w_j$ 对降低残差的贡献不足以抵消 $\lambda$ 时，$w_j$ 被置为零——自动特征选择。
- Lasso 没有闭式解——scikit-learn 使用坐标下降法迭代求解。
- 贝叶斯视角：Lasso 等价于对 $\mathbf{w}$ 施加拉普拉斯先验 $P(w_j) \propto \exp(-\lambda\|w_j\|)$ 后的 MAP 估计。
- 拉普拉斯分布在零点有尖峰——比高斯分布更倾向于让参数恰好为零。

## 4. ElasticNet（弹性网）

### 参数速览

适用模型：`ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 目标函数 | 数学表达式 | $\mathcal{L}_{\text{EN}} = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda[\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2]$ | — |
| $\lambda$ | 标量 | 总正则化强度 | `alpha=0.2` |
| $\rho$ | 标量 | L1 占比——$\rho=1$ 退化为 Lasso，$\rho=0$ 退化为 Ridge | `l1_ratio=0.5` |

### 理解重点

- ElasticNet 通过 $\rho$ 在 L1 和 L2 之间插值——兼具稀疏性和共线性稳定性。
- 纯 Lasso 在相关特征组中可能随机只选一个——ElasticNet 的 L2 分量鼓励同组特征共享权重。
- $\rho=0.5$ 表示当前实现在稀疏性和稳定性之间取折中——不极端偏向任一侧。
- 也是坐标下降求解——`max_iter=10000` 确保在 $\rho$ 非极端值时充分收敛。

## 5. L1 vs L2 行为对比

| 特性 | L2（Ridge） | L1（Lasso） |
|---|---|---|
| 惩罚项 | $\sum w_j^2$ | $\sum \|w_j\|$ |
| 原点可微性 | 可微 | **不可微** |
| 系数归零 | 否——仅收缩 | **是——可精确归零** |
| 闭式解 | 有 | 无（需迭代） |
| 共线性处理 | 系数分摊到同组特征 | 可能只选部分特征 |
| 贝叶斯先验 | 高斯分布 | 拉普拉斯分布 |
| 特征选择 | 不擅长 | **擅长** |

### 理解重点

- L1 产生稀疏解的根本原因是原点不可微——不是"惩罚得更狠"，而是惩罚的**形状**不同。
- L2 的圆形等高线容易与损失等高线在任意点相切——系数非零但较小。
- L1 的菱形等高线容易在坐标轴尖点处与损失等高线相切——系数精确为零。

## 6. 数学概念与代码实现的映射

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 特征矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times d}$ | `X_train_s`——标准化后形状 `(353, 21)` |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train`——形状 `(353,)` |
| 系数向量 | $\mathbf{w} \in \mathbb{R}^d$ | `model.coef_`——形状 `(21,)` |
| 截距 | $b$ | `model.intercept_`——标量 |
| L2 惩罚强度 | $\lambda$（Ridge） | `Ridge(alpha=2.0)` |
| L1 惩罚强度 | $\lambda$（Lasso） | `Lasso(alpha=0.15)` |
| 总惩罚强度 | $\lambda$（ElasticNet） | `ElasticNet(alpha=0.2)` |
| L1 混合比例 | $\rho$ | `ElasticNet(l1_ratio=0.5)` |
| 系数绝对值 < $10^{-3}$ | $\{j : \|w_j\| < 10^{-3}\}$ | `np.sum(np.abs(coef) < 1e-3)` |
| 标准化 | $z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$ | `StandardScaler().fit_transform(X_train)` |

## 7. 正则化回归 vs 线性回归 数学对比

| 数学维度 | 线性回归 | 正则化回归 |
|---|---|---|
| 损失函数 | $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ | OLS 损失 $+ \lambda \cdot R(\mathbf{w})$ |
| 优化问题 | 无约束最小化 | **带惩罚的约束最小化** |
| 求解方法 | SVD 闭式解（一步到位） | **坐标下降 / 闭式解（Ridge）+ 迭代（Lasso/EN）** |
| 唯一解 | 是（当 $\mathbf{X}$ 满秩） | **是——惩罚项使问题强凸（Ridge/EN）或凸（Lasso）** |
| 对共线性的数值稳定性 | 低——$\mathbf{X}^T\mathbf{X}$ 可能接近奇异 | **高——$\lambda\mathbf{I}$ 改善条件数（Ridge/EN）** |
| 特征选择 | 无——所有系数非零 | **有——L1 罚项可将系数精确归零** |
| 超参数数量 | 0 | **1~2（$\alpha$ + 可选的 l1_ratio）** |
| 尺度敏感性 | 不敏感——闭式解是尺度等变的 | **敏感——惩罚项对系数量级敏感，必须标准化** |

## 常见坑

1. 把 `alpha` 理解成学习率或迭代轮数——它在这里是正则化强度 $\lambda$，越大惩罚越重。
2. 认为"正则化一定让预测更准"——正则化在偏差和方差之间权衡，过强的正则化会导致欠拟合。
3. 只记公式不记代码映射——`alpha` $\leftrightarrow$ $\lambda$、`l1_ratio` $\leftrightarrow$ $\rho$、`np.sum(np.abs(coef) < 1e-3)` $\leftrightarrow$ 稀疏性。

## 小结

- 正则化回归的数学本质是在 OLS 损失上叠加系数惩罚——从无约束优化变为带惩罚优化。
- L2 惩罚产生收缩（Ridge），L1 惩罚产生稀疏（Lasso），L1+L2 混合产生折中（ElasticNet）。
- L1 产生稀疏解的根本原因是绝对值函数在原点不可微——而非"惩罚更重"。
- 数学公式中的 $\lambda$、$\rho$ 直接映射到源码中的 `alpha`、`l1_ratio`——理解这个映射是读懂后续章节的前提。

# 数据构成

## 本章目标

1. 明确正则化回归数据的来源——`loadRegularizationDataset()` 在 diabetes 基础上构造三层特征。
2. 理解三层特征结构（原始医学 -> 共线特征 -> 纯噪声）的设计意图。
3. 理解标准化在数据层的执行边界——`StandardScaler` 由流水线层而非数据层执行。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadRegularizationDataset()` | 方法 | 加载 diabetes 并追加共线特征和纯噪声特征——返回 `(442, 22)` DataFrame |
| `load_diabetes(as_frame=True)` | 函数 | scikit-learn 提供的糖尿病回归数据集——10 个医学特征 |
| `bmi_corr` / `bp_corr` / `s5_corr` | 列名 | 人为制造多重共线性的相关特征——与原始列相关系数约 0.9 |
| `noise_1` ~ `noise_8` | 列名 | 纯随机噪声特征——用于观察 L1 稀疏化效果 |
| `price` | 列名 | 回归目标列——由 diabetes 的 `target` 重命名而来 |

## 1. 数据入口：`loadRegularizationDataset()`

### 参数速览

适用函数：`loadRegularizationDataset()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `randomState` | `int` | 随机种子——保证共线噪声和纯噪声的可复现性 | `42` |
| 返回值 | `DataFrame` | 形状 `(442, 22)`——10 原始 + 3 共线 + 8 噪声 + 1 标签 | — |

### 示例代码

```python
def loadRegularizationDataset(self) -> DataFrame:
    rng = np.random.RandomState(self.randomState)
    data = load_diabetes(as_frame=True).frame.copy().rename(
        columns={"target": "price"}
    )
    # 追加共线特征
    data["bmi_corr"] = data["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    data["bp_corr"] = data["bp"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    data["s5_corr"] = data["s5"] * 0.9 + rng.normal(scale=0.02, size=len(data))
    # 追加纯噪声特征
    for index in range(8):
        data[f"noise_{index + 1}"] = rng.normal(size=len(data))
    return data
```

### 理解重点

- 基础数据来自 scikit-learn 的真实糖尿病数据集——10 个标准化后的医学特征，442 个样本。
- 原始列名 `target` 被重命名为 `price`——保持与仓库其他回归分册的标签列名一致。
- 共线特征通过 `原始值 x 0.9 + 微小噪声` 构造——与原始列的相关系数约 0.9，刻意制造多重共线性。

## 2. 三层特征结构

### 参数速览

| 特征层 | 列名 | 数量 | 设计意图 |
|---|---|---|---|
| 原始医学特征 | `age`, `sex`, `bmi`, `bp`, `s1`~`s6` | 10 | 提供真实回归信号——来自真实医学数据 |
| 共线特征 | `bmi_corr`, `bp_corr`, `s5_corr` | 3 | 人为制造多重共线性——观察 Ridge 的收缩 vs Lasso 的筛选 |
| 纯噪声特征 | `noise_1` ~ `noise_8` | 8 | 完全无预测能力的随机列——观察 Lasso 是否将其系数压到零 |
| 标签列 | `price` | 1 | 糖尿病病情进展量化指标 |

### 理解重点

- 三层结构是刻意设计的——每一层测试正则化的不同能力：共线层测试稳定性，噪声层测试稀疏性。
- `bmi_corr` 与 `bmi` 高度相关（r ~= 0.9）——OLS 会在这两个特征之间难以分配系数，而 Ridge 会均匀分摊，Lasso 可能只保留一个。
- `noise_*` 理论上不应有任何非零系数——观察 Lasso 的 `near_zero` 计数可以直接验证 L1 的稀疏化效果。
- 与线性回归的合成数据不同——正则化回归使用真实数据 + 人工干扰，更接近实际应用场景。

## 3. 特征切分与标准化边界

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比 | `0.2` |
| `random_state` | `int` | 切分随机种子 | `42` |
| 训练集形状 | — | `X_train`: `(353, 21)`, `y_train`: `(353,)` | — |
| 测试集形状 | — | `X_test`: `(89, 21)`, `y_test`: `(89,)` | — |
| `StandardScaler` | 预处理 | 仅在 `X_train` 上 `fit_transform`，`X_test` 仅 `transform` | — |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 标准化不在数据层执行——数据层只返回原始 DataFrame，标准化由流水线（运行器层）负责。
- 正则化回归**必须标准化**——L1/L2 惩罚对系数量级敏感，未标准化的特征会导致惩罚不均匀。
- 这与线性回归和决策树回归形成关键差异——它们不需要标准化，正则化回归是回归分册中首个强制标准化的模型。
- `StandardScaler` 仅在训练集上 `fit`——测试集使用训练集的均值和标准差做 `transform`，避免数据泄露。

## 4. 数据特征总览

### 参数速览

| 属性 | 值 |
|---|---|
| 样本总数 | 442 |
| 特征总数 | 21（10 原始 + 3 共线 + 8 噪声） |
| 训练样本数 | 353（80%） |
| 测试样本数 | 89（20%） |
| 标签列名 | `price` |
| 是否有缺失值 | 否——diabetes 数据集已预处理 |
| 数据来源 | `sklearn.datasets.load_diabetes` |

### 理解重点

- 442 样本 vs 21 特征——样本量远大于特征数，但共线和噪声特征的存在使得纯 OLS 仍然不稳定。
- 与线性回归的 200 样本 / 3 特征相比——正则化回归的数据规模和复杂度明显更高。
- 与决策树回归的 20640 样本 / 8 特征相比——正则化回归样本量较小但特征结构更复杂（共线 + 噪声层）。

## 5. 数据设计意图：与线性回归/决策树回归的对比

| 数据维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 数据来源 | 手工合成 | 真实数据（California Housing） | **真实数据 + 人工干扰（diabetes + 共线 + 噪声）** |
| 样本量 | 200 | 20640 | **442** |
| 特征数 | 3 | 8 | **21（10 + 3 + 8）** |
| 特征关系 | 完全独立 | 自然相关 | **刻意构造共线 + 纯噪声** |
| 标签 | `price = 2x面积 + 10x房间 - 3x房龄 + $\varepsilon$` | 加州房价中位数 | **糖尿病病情进展（真实医学指标）** |
| 标准化 | 否 | 否 | **是——强制要求** |
| 设计意图 | 透明验证 OLS 恢复精度 | 非线性树结构演示 | **观察 L1 稀疏化 + L2 收缩 + 共线性处理** |

## 常见坑

1. 忘记 diabetes 的 `target` 已被重命名为 `price`——拆分标签时写错列名。
2. 只关注原始 10 个医学特征——忽略 `bmi_corr`/`bp_corr`/`s5_corr` 和 `noise_*` 是正则化行为展示的核心。
3. 在切分前对全量数据做标准化——造成测试集信息泄露到训练过程。
4. 认为"数据层已标准化"——标准化实际在运行器层执行，数据层返回的是原始值。

## 小结

- 正则化回归数据由三层构成：10 个真实医学特征 + 3 个人工共线特征 + 8 个纯噪声特征 = 21 维。
- 三层结构各有设计意图：原始特征提供信号，共线特征测试收缩/筛选，噪声特征验证稀疏化。
- 标准化是正则化回归的强制预处理——与线性回归和决策树回归形成关键工程差异。
- 数据量（442 样本 / 21 特征）介于线性回归和决策树回归之间——复杂度适中，适合观察正则化行为。

# 思路与直觉

## 本章目标

1. 理解为什么正则化回归的数据被刻意设计成三层结构——每一层暴露 OLS 的不同弱点。
2. 建立 Ridge 整体收缩、Lasso 选择性清零、ElasticNet 折中的直觉。
3. 理解 L1 产生稀疏解的几何直觉——菱形约束区域 vs 圆形约束区域。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 共线性问题 | 概念 | 两个特征高度相关时 OLS 系数不稳定——Ridge 分摊解决，Lasso 选择解决 |
| 高维噪声问题 | 概念 | 无意义特征被 OLS 分到非零权重——L1 惩罚将其驱动到零 |
| L2 收缩直觉 | 概念 | 平方惩罚->所有系数变小但不归零——"都保留，但都温和" |
| L1 稀疏直觉 | 概念 | 绝对值惩罚->不重要系数精确归零——"宁可删掉一部分" |
| 菱形 vs 圆形 | 几何直觉 | L1 约束区域是菱形（尖点在坐标轴）-> 稀疏解；L2 是圆形 -> 非稀疏解 |

## 1. 为什么当前数据特别适合讲正则化

当前分册的数据不是随机选的——它刻意构造了两类会让 OLS 暴露弱点的问题。

### 参数速览

| 问题类型 | 数据体现 | OLS 的表现 | 正则化的对策 |
|---|---|---|---|
| 多重共线性 | `bmi_corr` = `bmi` x 0.9 + epsilon | 系数在 `bmi` 和 `bmi_corr` 间剧烈摇摆 | Ridge 分摊权重；Lasso 可能只选一个 |
| 无关特征 | `noise_1` ~ `noise_8`（纯随机） | 给噪声分配非零系数，过拟合训练集 | Lasso/ElasticNet 将噪声系数压到零 |

### 理解重点

- 如果没有 `*_corr` 特征，三种模型处理共线性的差异就无从观察。
- 如果没有 `noise_*` 特征，Lasso 的稀疏化能力就缺乏直观验证。
- 这份数据是为"展示正则化行为差异"而设计的——每一层特征都在测试正则化的一个特定能力。

## 2. OLS 在面对共线性和噪声时为什么会不稳

### 参数速览

| 维度 | OLS | 正则化回归 |
|---|---|---|
| 训练目标 | $\min \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$ | $\min \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \lambda R(\mathbf{w})$ |
| 对系数的态度 | 不关心系数大小——只要拟合误差小 | **系数大小有代价——越大惩罚越重** |
| 共线性下的行为 | 系数方差大——`bmi` 和 `bmi_corr` 的系数随机摇摆 | Ridge 分摊稳定；Lasso 选择其一 |
| 噪声特征处理 | 可能给噪声分到非零权重 | **Lasso/EN 将噪声系数压到接近零** |

### 理解重点

- OLS 是"自由的"——只要降低训练误差，系数多大都行。
- 正则化是"有约束的"——系数每增大一点都要付出 $\lambda$ 的代价。
- 在面对高度相关的两个特征时，OLS 不知道权重该分给谁——Ridge 说"都分一点"，Lasso 说"只留一个"。

## 3. Ridge 的直觉：都保留，但整体缩小

### 参数速览

适用模型：`Ridge(alpha=2.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L2 惩罚强度——越大收缩越狠 | `2.0` |
| 预期行为 | — | 所有系数非零但整体偏小；`bmi` 和 `bmi_corr` 权重分摊 | — |
| 近零系数数 | — | 通常为 0（所有系数都非零） | — |

### 理解重点

- Ridge 的理念是"这些特征也许都有用，但别让任何一个膨胀得太夸张"。
- 面对 `bmi` 和 `bmi_corr` 这类高度相关特征——Ridge 会把权重均匀分摊，两者都保留非零系数。
- L2 的平方惩罚在零点平滑可微——系数趋近零时惩罚梯度也趋近零，因此缺乏"清零"的动力。
- 可以理解为：Ridge 对所有特征持"保留态度"，通过温和收缩来控制过拟合。

## 4. Lasso 的直觉：宁可删掉一些特征

### 参数速览

适用模型：`Lasso(alpha=0.15, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L1 惩罚强度——越大清零越激进 | `0.15` |
| `max_iter` | `int` | 坐标下降最大迭代次数 | `10000` |
| 预期行为 | — | 部分系数精确归零；`noise_*` 系数应接近零；`bmi` 和 `bmi_corr` 可能只保留一个 | — |
| 近零系数数 | — | 应 > 0——噪声特征被清零的直接证据 | — |

### 理解重点

- Lasso 的理念是"如果某个特征贡献不大，直接删掉它"。
- 对 `noise_1` ~ `noise_8`——因为它们完全不含信号，Lasso 会将其系数精确压到零。
- 对 `bmi` 和 `bmi_corr`——Lasso 可能只保留其中一个（通常是信号更强的原始列），另一个清零。
- L1 的绝对值惩罚在零点处不可微——次梯度包含 `[-1, 1]` 区间，为清零提供了数学条件。

## 5. L1 vs L2 的几何直觉：为什么 L1 稀疏而 L2 不稀疏

这是理解正则化回归最关键的直觉。

### 参数速览

| 几何维度 | L2（Ridge） | L1（Lasso） |
|---|---|---|
| 约束区域形状 | **圆形**（$\sum w_j^2 \leq t$） | **菱形**（$\sum \|w_j\| \leq t$） |
| 与坐标轴的交点 | 圆弧——光滑，无尖点 | **尖点**——在坐标轴上有顶点 |
| 损失等高线首次接触位置 | 通常在圆弧的非轴点 | **更可能在菱形尖点（坐标轴上）** |
| 结果 | 系数非零但收缩 | **部分系数精确为零** |

### 理解重点

- 将约束区域想象成一个"围墙"——损失等高线从原点向外扩张，首次碰到围墙的位置就是解。
- L2 的围墙是圆形的——圆弧光滑无尖角，损失等高线很少恰好在坐标轴上首次接触围墙 -> 系数非零。
- L1 的围墙是菱形的——菱形的尖点在坐标轴上，损失等高线有较大概率在尖点处接触 -> 对应系数为零。
- 这就是为什么 L1 能产生稀疏解的本质原因——不是"惩罚更重"，而是"约束区域的形状不同"。

## 6. ElasticNet 的直觉：既想筛选，也想保持稳定

### 参数速览

适用模型：`ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | 总正则化强度 | `0.2` |
| `l1_ratio` | `float` | L1 占比——0=纯 Ridge，1=纯 Lasso | `0.5` |
| 预期行为 | — | 兼具稀疏性和稳定性——噪声特征被清零，共线特征权重分摊 | — |
| 近零系数数 | — | 介于 Ridge（0）和 Lasso（较多）之间 | — |

### 理解重点

- 纯 Lasso 的问题：面对一组高度相关特征，可能随机只选一个——不够稳定。
- 纯 Ridge 的问题：保留所有特征，包括噪声——不够稀疏。
- ElasticNet 的 L2 分量鼓励同组特征共享权重（稳定性），L1 分量推动噪声特征归零（稀疏性）——取两者之长。
- `l1_ratio=0.5` 表示当前实现处于中性折中——不偏向稀疏也不偏向收缩。

## 7. 如何从训练日志中验证直觉

### 参数速览

| 观察项 | 正常表现 | 异常信号 |
|---|---|---|
| Lasso 的 `near_zero` | 显著 > 0——`noise_*` 被清零 | 若 = 0——alpha 太小，L1 未生效 |
| Ridge 的 `near_zero` | 通常 = 0——所有系数非零 | 若 > 0——alpha 过大，过度收缩 |
| ElasticNet 的 `near_zero` | 介于 Lasso 和 Ridge 之间 | 若接近 Lasso——`l1_ratio` 偏高 |
| `bmi` vs `bmi_corr` 系数 | Ridge 两者均非零；Lasso 可能只有一个；EN 两者非零但较小 | 若 OLS 风格（两者都大）——alpha 太小 |
| `noise_*` 系数 | Lasso/EN 接近零；Ridge 非零但很小 | 若 Lasso 的 `noise_*` 仍很大——alpha 不够 |

### 理解重点

- 系数结构本身就是正则化回归的"成绩单"——不仅看预测精度，还要看系数是否符合预期行为。
- `near_zero` 是最直观的诊断指标——它直接量化了 L1 的稀疏化效果。
- 三个模型的行为差异应该在日志中清晰可见——如果三者系数几乎一样，说明 alpha 设置有问题。

## 常见坑

1. 把"Lasso 产生零系数"等同于"Lasso 预测更准"——稀疏性 != 准确性，两者需分开评估。
2. 只看 `near_zero` 总数不看具体哪些特征被清零——`noise_*` 被清零是好的，`bmi` 被清零则可能过度正则化。
3. 忽略 `l1_ratio=0.5` 的含义——它不是"一半 L1 一半 L2 的强度"，而是"混合比例各占一半"。

## 小结

- 正则化回归的核心直觉：通过约束系数大小来换取稳定性和可解释性。
- Ridge 倾向"整体温和收缩"，Lasso 倾向"选择性清零"，ElasticNet 在两者间折中。
- L1 产生稀疏解的根本原因是约束区域的几何形状（菱形尖点）——而非惩罚更重。
- 当前数据的三层结构（原始/共线/噪声）就是为展示这些直觉差异而设计的。

# 模型构建

## 本章目标

1. 理解 `trainRegularizationModels(...)` 如何一次性构建并训练三个正则化模型。
2. 理解 Lasso、Ridge、ElasticNet 的超参数及其默认值的选取理由。
3. 理解 `coef_`、`intercept_` 和近零系数计数在模型构建层的角色。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainRegularizationModels(...)` | 函数 | 构建并训练 Lasso、Ridge、ElasticNet——返回 `dict[str, 模型]` |
| `Lasso(alpha=0.15)` | 类 | L1 正则化线性模型——坐标下降求解 |
| `Ridge(alpha=2.0)` | 类 | L2 正则化线性模型——闭式解 |
| `ElasticNet(alpha=0.2, l1_ratio=0.5)` | 类 | L1+L2 混合正则化——坐标下降求解 |
| `model.coef_` | 属性 | 21 维系数向量——正则化的核心输出 |
| `np.sum(np.abs(coef) < 1e-3)` | 派生量 | 近零系数计数——量化稀疏化程度 |

## 1. `trainRegularizationModels(...)` 的函数签名

### 参数速览

适用函数：`trainRegularizationModels(XTrain, yTrain, randomState=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(353, 21)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `yTrain` | `ndarray`，形状 `(353,)` | 训练目标值 | `y_train` |
| `randomState` | `int` | 随机种子——保证坐标下降可复现 | `42` |
| 返回值 | `dict[str, 模型]` | `{"lasso": Lasso, "ridge": Ridge, "elasticnet": ElasticNet}` | — |

### 示例代码

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def trainRegularizationModels(XTrain, yTrain, randomState: int = 42):
    models = {
        "lasso": Lasso(alpha=0.15, max_iter=10000, random_state=randomState),
        "ridge": Ridge(alpha=2.0, random_state=randomState),
        "elasticnet": ElasticNet(
            alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=randomState
        ),
    }
    for model in models.values():
        model.fit(XTrain, yTrain)
    return models
```

### 理解重点

- 这是本仓库**唯一返回多个模型的训练函数**——其他回归训练函数（线性回归、决策树、SVR）都只返回单个模型。
- 三个模型共享同一份标准化后的训练数据——确保对比的公平性。
- 函数签名比 `trainLinearRegressionModel` 多了 `randomState` 参数——Lasso 和 ElasticNet 的坐标下降涉及随机性。

## 2. Lasso 的构造器参数

### 参数速览

适用 API：`Lasso(alpha=0.15, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L1 正则化强度——越大清零越激进 | `0.15` |
| `max_iter` | `int` | 坐标下降最大迭代次数——防止未收敛 | `10000` |
| `random_state` | `int` | 坐标下降的随机种子——保证可复现 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |
| `coef_` | `ndarray`，形状 `(21,)` | 训练后的系数向量——部分可能精确为零 | — |

### 理解重点

- `alpha=0.15` 是经过调试的取值——在 diabetes 数据上既能展示稀疏化，又不至于把所有系数清零。
- `max_iter=10000` 远大于默认值（1000）——确保在 21 维特征 + 坐标下降下充分收敛。
- Lasso 没有 `l1_ratio` 参数——它是纯 L1 惩罚，与 ElasticNet 不同。

## 3. Ridge 的构造器参数

### 参数速览

适用 API：`Ridge(alpha=2.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | L2 正则化强度——越大收缩越狠 | `2.0` |
| `random_state` | `int` | 随机种子——Ridge 使用闭式解，此参数仅影响特定求解器 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |
| `solver` | `str` | 求解器——默认 `'auto'`，根据数据自动选择 | `'auto'`（默认） |

### 理解重点

- `alpha=2.0` 明显大于 Lasso 的 `0.15`——Ridge 使用平方惩罚，需要更大的 alpha 才能产生等量的收缩效果。
- Ridge 有闭式解 $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$——不需要 `max_iter`。
- Ridge 的 `random_state` 仅在 `solver='sag'` 或 `'saga'` 时生效——当前使用默认 `'auto'`，通常选择闭式求解器。

## 4. ElasticNet 的构造器参数

### 参数速览

适用 API：`ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=10000, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `alpha` | `float` | 总正则化强度——越大惩罚越重 | `0.2` |
| `l1_ratio` | `float` | L1 占比——0 = 纯 Ridge，1 = 纯 Lasso | `0.5` |
| `max_iter` | `int` | 坐标下降最大迭代次数 | `10000` |
| `random_state` | `int` | 坐标下降的随机种子 | `42` |
| `fit_intercept` | `bool` | 是否拟合截距——默认 `True` | `True`（默认） |

### 理解重点

- `l1_ratio=0.5` 使 ElasticNet 处于 L1 和 L2 的正中间——不是极端偏向任一侧。
- `alpha=0.2` 介于 Lasso（0.15）和 Ridge（2.0）之间——总强度适中。
- 坐标下降同时处理 L1 和 L2 分量——`max_iter=10000` 与 Lasso 一致，确保充分收敛。
- ElasticNet 的关键优势：在 `bmi`/`bmi_corr` 这种共线特征对上，比纯 Lasso 更稳定（L2 分量分摊权重），比纯 Ridge 更稀疏（L1 分量清零噪声）。

## 5. 训练后的关键属性

### 参数速览

| 属性 | 类型 | Lasso | Ridge | ElasticNet |
|---|---|---|---|---|
| `coef_` | `ndarray(21,)` | 部分系数精确为 0 | 所有系数非零但收缩 | 介于两者之间 |
| `intercept_` | `float` | 截距项 | 截距项 | 截距项 |
| `alpha` | `float` | `0.15` | `2.0` | `0.2` |
| `l1_ratio` | `float` | —（无此属性） | —（无此属性） | `0.5` |
| `n_iter_` | `int` | 实际迭代次数 | —（闭式解） | 实际迭代次数 |

### 理解重点

- `coef_` 是正则化回归最重要的输出——不仅是预测参数，更是特征选择的直接证据。
- Lasso 的 `coef_` 中精确为零的位置对应被淘汰的特征——这是 L1 正则化的核心价值。
- ElasticNet 的 `coef_` 中零的数量取决于 `l1_ratio`——越接近 1 越像 Lasso，越接近 0 越像 Ridge。
- `intercept_` 不受正则化惩罚——只有系数向量被惩罚，截距项始终自由。

## 6. 正则化回归 vs 线性回归 vs 决策树回归 模型构建对比

| 模型维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | **`Lasso` / `Ridge` / `ElasticNet`** |
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | **`trainRegularizationModels`** |
| 返回值 | 单个模型 | 单个模型 | **`dict`——三个模型** |
| 超参数数 | 0 | 3 | **Lasso: 1, Ridge: 1, EN: 2** |
| `random_state` | 不需要 | 需要 | **需要（Lasso/EN）** |
| 核心属性 | `coef_`, `intercept_` | `feature_importances_`, `tree_` | **`coef_`, `intercept_` + 近零计数** |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）/ 闭式解（Ridge）** |
| 是否需要标准化 | 否 | 否 | **是——强制要求** |

## 常见坑

1. 误以为 `trainRegularizationModels` 只训练一个模型——它返回 `dict`，是三个模型的容器。
2. 将三个模型的 `alpha` 值直接比较大小——Lasso 的 0.15 和 Ridge 的 2.0 产生不同的惩罚效果（L1 vs L2），不能直接比较数值。
3. 期待 Ridge 也有 `max_iter` 参数——Ridge 使用闭式解，不需要迭代。
4. 忘记 ElasticNet 有 `l1_ratio` 而 Lasso 没有——Lasso 是纯 L1，不需要混合比例。

## 小结

- `trainRegularizationModels(...)` 是本仓库唯一返回多模型的训练函数——一次性构建 Lasso、Ridge、ElasticNet 三个模型。
- 三个模型的超参数各有侧重：Lasso 强调稀疏化（alpha=0.15），Ridge 强调收缩（alpha=2.0），ElasticNet 折中（alpha=0.2, l1_ratio=0.5）。
- `coef_` 不仅是预测参数，更是正则化回归的"成绩单"——观测系数结构比观测预测分数更重要。
- 标准化是模型构建的隐含前提——`trainRegularizationModels` 的输入必须是标准化后的数据。

# 训练与预测

## 本章目标

1. 理解正则化回归流水线的完整执行顺序——从数据加载到残差图输出。
2. 理解三种模型的训练过程——坐标下降（Lasso/EN）vs 闭式解（Ridge）。
3. 理解预测阶段的统一循环——三模型共用同一份标准化测试数据。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadRegularizationDataset()` | 方法 | 加载 diabetes + 共线 + 噪声特征——返回 `(442, 22)` DataFrame |
| `StandardScaler` | 预处理 | Z-score 标准化——正则化回归训练前的强制步骤 |
| `trainRegularizationModels(...)` | 函数 | 构建并 `fit` 三个正则化模型——返回模型字典 |
| `model.predict(X_test_s)` | 方法 | 对标准化测试集做回归预测——$\hat{y} = \mathbf{X}\mathbf{w} + b$ |
| `plot_residuals(...)` | 函数 | 为每个模型绘制残差诊断图 |

## 1. 完整流水线流程

### 流程概述

```
loadRegularizationDataset()
  - 1 X = data.drop(columns=["price"]), y = data["price"]
  - 2 X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
  - 3 scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train)
  - 4 X_test_s = scaler.transform(X_test)
  - 5 models = trainRegularizationModels(X_train_s, y_train)
  - 6 for name, model in models.items():
    - y_pred = model.predict(X_test_s)
    - plot_residuals(y_test, y_pred, ...)
  - 7 plot_feature_importance(model, feature_names)  — 对每个模型
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadRegularizationDataset` | — | `DataFrame`，`(442, 22)` | diabetes + 共线 + 噪声 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X(442,21)`, `y(442,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`, `y` | `X_train(353,21)`, `X_test(89,21)` | `test_size=0.2` |
| 标准化 | `StandardScaler` | `X_train`, `X_test` | `X_train_s`, `X_test_s` | **正则化回归必需** |
| 训练 | `trainRegularizationModels` | `X_train_s`, `y_train` | `dict[lasso/ridge/elasticnet]` | 一次训练三个模型 |
| 预测 | `model.predict` | `X_test_s` | 各模型 `y_pred(89,)` | 循环三次 |
| 残差图 | `plot_residuals` | `y_test`, `y_pred` | PNG 图像 | 每个模型一张 |
| 特征重要性 | `plot_feature_importance` | `model`, `feature_names` | PNG 图像 | 系数柱状图 |

### 理解重点

- 正则化回归流水线比线性回归多两步——标准化（34）和特征重要性图（7），但没有学习曲线。
- 标准化在切分**之后**执行——先在训练集上 `fit_transform`，再对测试集仅 `transform`。
- 三个模型共享完全相同的训练/测试数据——对比的公平性由统一的切分和标准化保证。

## 2. 标准化：正则化回归训练的关键前置步骤

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `ndarray(353, 21)` | 未标准化的训练特征 | 原始 diabetes + 构造特征 |
| `X_train_s` | `ndarray(353, 21)` | 标准化后——每列均值 0、标准差 1 | — |
| `X_test_s` | `ndarray(89, 21)` | 使用训练集统计量标准化 | — |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # 计算 mu, sigma 并变换
X_test_s = scaler.transform(X_test)         # 仅变换——使用训练集的 mu, sigma
```

### 理解重点

- 正则化回归**必须标准化**——L1/L2 惩罚对系数量级敏感。未标准化的特征会导致惩罚不均匀：量级大的特征被过度惩罚，量级小的特征惩罚不足。
- 这是正则化回归与线性回归、决策树回归最关键的工程差异——两者在 `PipelineSpec` 中的预处理分别为 `None` 和 `"standardScaler"`。
- `fit_transform` vs `transform` 的区别是防止数据泄露——测试集的均值和标准差不应影响模型。

## 3. 训练细节：三种算法，三种求解路径

### 参数速览

| 模型 | 求解算法 | 是否有闭式解 | 是否需要迭代 | 收敛判断 |
|---|---|---|---|---|
| Ridge | SVD / 闭式解 | **是**——$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$ | 否 | 不需要 |
| Lasso | 坐标下降 | 否 | **是**——`max_iter=10000` | 对偶间隙 < `tol` |
| ElasticNet | 坐标下降 | 否 | **是**——`max_iter=10000` | 对偶间隙 < `tol` |

### 理解重点

- Ridge 训练是**瞬时**的——21 维特征的闭式解计算量极小。
- Lasso 和 ElasticNet 是**迭代**的——坐标下降逐维度优化，`max_iter=10000` 确保充分收敛。
- 三种模型都保证找到全局最优解——Ridge 的目标函数是强凸的，Lasso/EN 是凸的（坐标下降收敛到全局最优）。

## 4. 预测细节：统一的线性公式

三种模型的预测公式完全相同：

$$
\hat{y} = \mathbf{X}_{\text{test}} \mathbf{w} + b
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_test_s` | `ndarray(89, 21)` | 标准化后的测试特征 | — |
| `model.coef_` | `ndarray(21,)` | 各模型的系数向量——形态因正则化策略而异 | Lasso：部分零 / Ridge：全非零 |
| `model.intercept_` | `float` | 截距项——不受正则化惩罚 | — |
| `y_pred` | `ndarray(89,)` | 预测值 | — |

### 理解重点

- 预测公式与普通线性回归完全一致——都是 $\mathbf{X}\mathbf{w} + b$ 的矩阵乘法。
- 差异不在预测公式，而在 $\mathbf{w}$ 的形态——Lasso 的 $\mathbf{w}$ 有零分量，Ridge 的 $\mathbf{w}$ 整体收缩。
- 预测复杂度 $O(N_{\text{test}} \cdot d) = O(89 \times 21)$——几乎瞬时。

## 5. 多模型预测循环

### 示例代码

```python
models = trainRegularizationModels(X_train_s, y_train, randomState=42)

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    plot_residuals(
        y_test, y_pred,
        title=f"{name} 残差分析",
        dataset_name="regularization",
        model_name=name,
    )
    plot_feature_importance(
        model, feature_names,
        title=f"{name} 系数",
        dataset_name="regularization",
        model_name=name,
    )
```

### 理解重点

- 循环结构是正则化回归流水线的独特特征——其他回归模型只训练和评估一个模型。
- 每个模型都生成独立的残差图和系数图——便于横向对比三种正则化策略的差异。
- `model_name=name` 使输出文件自动按模型名命名（`lasso_residual.png`、`ridge_coefficients.png` 等）。

## 6. 正则化回归 vs 线性回归 vs 决策树回归 训练对比

| 训练维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 数据 | 合成 `(200, 3)` | 真实 `(20640, 8)` | **真实+构造 `(442, 21)`** |
| 标准化 | 无 | 无 | **`StandardScaler`——强制** |
| 训练算法 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）+ 闭式解（Ridge）** |
| 训练模型数 | 1 | 1 | **3（并行训练）** |
| 收敛判断 | 不需要 | `max_depth` 等早停 | **`max_iter=10000`（Lasso/EN）** |
| 预测 | $\hat{y} = \mathbf{X}\mathbf{w} + b$ | 沿树走到叶子返回均值 | **$\hat{y} = \mathbf{X}\mathbf{w} + b$（同线性回归）** |
| 评估可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 + 学习曲线 + 树结构 | **残差图 + 特征重要性——无学习曲线** |

## 常见坑

1. 训练时传 `X_train_s`，预测时却传了未标准化的 `X_test`——预测结果会严重偏离。
2. 忘记标准化必须在切分之后——先标准化再切分会导致测试集信息泄露。
3. 期待正则化回归也有学习曲线——当前 `PipelineSpec` 中学可视化列表为 `[]`，无学习曲线。
4. 以为 Lasso 和 Ridge 的 `alpha` 可以直接比较——`alpha=0.15`（Lasso）和 `alpha=2.0`（Ridge）量级不同，因为 L1 和 L2 的惩罚尺度不同。

## 小结

- 正则化回归流水线为 8 步：加载 -> 拆分 -> 切分 -> 标准化 -> 训练三模型 -> 循环预测 -> 残差图 -> 系数图。
- 标准化是正则化回归区别于线性回归和决策树回归的最关键工程差异——惩罚项对尺度敏感。
- 训练阶段一次产出三个模型——Ridge 用闭式解瞬间完成，Lasso/EN 用坐标下降迭代求解。
- 预测阶段与线性回归完全相同（$\mathbf{X}\mathbf{w} + b$）——差异在 $\mathbf{w}$ 的形态而非预测公式。

# 评估与诊断

## 本章目标

1. 理解当前正则化回归的两类评估输出——残差图和系数图（特征重要性）。
2. 理解近零系数计数作为正则化回归独有诊断指标的价值。
3. 明确当前流水线**已实现**和**未实现**的评估项。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 生成预测-真实散点图 + 残差分布图——诊断拟合质量 |
| `plot_feature_importance(...)` | 函数 | 生成系数柱状图——诊断稀疏性和特征权重分布 |
| `residuals = y_true - y_pred` | 派生量 | 衡量每个样本的预测误差 |
| `np.sum(np.abs(coef) < 1e-3)` | 派生量 | 近零系数计数——正则化回归独有的稀疏性诊断 |

## 1. 残差图

### 参数速览

适用函数：`plot_residuals(y_test, y_pred, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `ndarray`，形状 `(89,)` | 测试集真实值 | 糖尿病病情进展 |
| `y_pred` | `ndarray`，形状 `(89,)` | 模型预测值 | 各模型分别预测 |
| `title` | `str` | 图标题 | `"Lasso 残差分析"` |
| `dataset_name` | `str` | 输出目录名 | `"regularization"` |
| `model_name` | `str` | 输出文件名前缀——区分三模型 | `"lasso"`, `"ridge"`, `"elasticnet"` |

### 示例代码

```python
y_pred = model.predict(X_test_s)
residuals = y_test - y_pred

# 左图: 预测值 vs 真实值散点图
ax1.scatter(y_test, y_pred, alpha=0.6)

# 右图: 残差 vs 预测值散点图
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color="r", linestyle="--")
```

### 理解重点

- 每个模型生成独立的残差图——可以直接对比三种正则化策略的预测误差分布。
- 左图看整体拟合：点越贴近对角线，预测越准确。在 diabetes 数据上，三种模型的散点分布差异通常不大——因为底层都是线性模型。
- 右图看系统偏差：残差围绕 0 随机分布为健康状态。若残差呈 U 形或漏斗形，说明线性假设可能不足。
- 残差图回答的是"预测误差分布如何"——与系数图回答的"模型如何分配权重"是互补关系。

## 2. 系数图（特征重要性）

### 参数速览

适用函数：`plot_feature_importance(model, feature_names, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | 已训练的模型 | 含 `coef_` 属性——系数向量 | Lasso / Ridge / ElasticNet |
| `feature_names` | `list[str]` | 21 个特征名——与 `coef_` 一一对应 | `["age", "sex", ..., "noise_8"]` |
| `title` | `str` | 图标题 | `"Lasso 系数"` |
| `dataset_name` | `str` | 输出目录名 | `"regularization"` |
| `model_name` | `str` | 输出文件名前缀 | `"lasso"` |

### 理解重点

- 系数图是正则化回归**最重要的诊断工具**——比残差图更具模型特异性。
- Lasso 的系数图中应能看到部分柱状条高度为零——被清零的特征一目了然。
- Ridge 的系数图中所有柱状条非零但整体较小——收缩效果通过系数量级体现。
- ElasticNet 的系数图介于两者之间——部分清零 + 部分收缩。

## 3. 近零系数计数：正则化回归独有的诊断指标

### 参数速览

| 指标 | 计算方式 | 诊断含义 |
|---|---|---|
| `near_zero` | `np.sum(np.abs(model.coef_) < 1e-3)` | 系数绝对值小于 0.001 的特征数 |
| 阈值 `1e-3` | 工程容差——非严格等于零 | 避免将极小但非零的系数误判为零 |

### 示例代码

```python
for name, model in models.items():
    coef = model.coef_
    near_zero = np.sum(np.abs(coef) < 1e-3)
    print(f"{name} 近零系数: {near_zero}/{len(coef)}")
    for f, c in zip(feature_names, coef):
        print(f"  {f}: {c:.3f}")
```

### 理解重点

- 近零系数计数直接量化 L1 的稀疏化效果——Lasso 的 `near_zero` 应 > 0，Ridge 通常 = 0。
- 用 `< 1e-3` 而非 `== 0`——坐标下降结果可能存在极小但非精确零的系数。
- 重点关注 `noise_*` 特征的系数是否被清零——它们理论上不应有任何非零权重。
- 对于 `bmi`/`bmi_corr` 共线对——Lasso 可能清零其中之一，Ridge 两者均保留。

## 4. 三模型对比：残差图 + 系数图 + 近零计数的联合解读

### 参数速览

| 诊断维度 | Lasso | Ridge | ElasticNet |
|---|---|---|---|
| 近零系数数 | **高**——`noise_*` 被清零 | **低**——通常为 0 | **中**——介于两者之间 |
| `noise_*` 系数 | 接近零 | 非零但很小 | 接近零 |
| `bmi` vs `bmi_corr` | 可能只保留一个 | 两者均非零，权重分摊 | 两者均非零但较小 |
| 残差图 | 与 Ridge 相近——底层都是线性模型 | 与 Lasso 相近 | 与两者相近 |
| 可解释性 | **最高**——系数表简洁 | 中——所有特征都参与 | 较高——兼具筛选和分摊 |

### 理解重点

- 三种模型在 diabetes 数据上的残差图差异通常不大——因为它们都是线性模型，预测能力相近。
- 真正的差异在**系数结构**——Lasso 的系数表比 Ridge 简洁得多（噪声被清零）。
- 正则化回归的诊断核心不是"哪个模型预测更准"，而是"哪个模型以更合理的系数结构达到相近的预测精度"。
- 如果 Lasso 的预测精度与 Ridge 相近但系数更稀疏——则 Lasso 更可取（奥卡姆剃刀原则）。

## 5. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 残差图 | 已实现 | 回归模型的核心诊断——每个模型独立生成 |
| 系数图（特征重要性） | 已实现 | 正则化回归最关键的诊断——系数结构可视化 |
| 系数打印 + 近零计数 | 已实现 | 训练日志中打印——可对照系数图验证 |
| MSE / MAE / RMSE / R^2 数值打印 | **未实现** | 当前流水线侧重图形化诊断而非数值指标 |
| 学习曲线 | **未实现** | `PipelineSpec` 中学可视化列表为 `[]` |
| 交叉验证 | **未实现** | 无 `cross_val_score` 等调用 |

### 理解重点

- 与线性回归和决策树回归不同——正则化回归**没有学习曲线**（`PipelineSpec` 中学可视化为 `[]`）。
- 评估设计聚焦于系数结构对比——残差图看预测误差，系数图 + 近零计数看正则化行为。
- 正则化回归独有的评估优势是可以直接观察"特征选择"——其他回归模型没有 L1 清零机制。

## 6. 正则化回归 vs 线性回归 vs 决策树回归 评估对比

| 评估维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 核心可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 + 学习曲线 + 树结构 | **残差图 + 特征重要性——无学习曲线** |
| 模型解释 | `coef_` + `intercept_` | `feature_importances_` | **`coef_` + 近零计数——可验证特征选择** |
| 独有诊断 | 系数与真实公式对照 | 树结构可视化 | **近零系数计数——L1 稀疏化量化** |
| 定量指标 | 无显式打印 | 无显式打印 | 无显式打印 |
| 对比模式 | 单模型诊断 | 单模型诊断 | **三模型横向对比——Lasso vs Ridge vs EN** |

## 常见坑

1. 只看残差图不看系数图——正则化回归的核心诊断在系数结构，残差图只是辅助。
2. 把"系数更稀疏"等同于"预测更准"——稀疏性与准确性是正交的评估维度。
3. 期待 `near_zero` 中 `noise_*` 全部精确为零——坐标下降可能产生 `1e-5` 量级的极小系数，用 `1e-3` 阈值统一判断。
4. 忽略三模型对比视角——单独看一个模型的系数没有意义，必须并排对比 Lasso/Ridge/ElasticNet。

## 小结

- 正则化回归有两类评估输出（残差图 + 系数图）+ 一项日志输出（近零系数计数）——三者构成完整的诊断体系。
- 系数图是正则化回归最重要的诊断工具——比残差图更能体现三种正则化策略的行为差异。
- 近零系数计数是正则化回归独有的诊断指标——直接量化 L1 的稀疏化效果。
- 正则化回归没有学习曲线——这是 `PipelineSpec` 的明确配置，与线性回归和决策树回归不同。

# 工程实现

## 本章目标

1. 理解正则化回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链，特别是多模型循环分支。
3. 理解正则化回归与线性回归、决策树回归在工程实现上的关键差异——标准化 + 多模型 + 无学习曲线。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadRegularizationDataset()` 构造 diabetes + 共线 + 噪声 |
| `trainRegularizationModels(...)` | 函数 | 构建并训练三个正则化模型——返回 `dict` |
| `PipelineSpec` | 数据类 | 声明式流水线配置——`multiModel=True` 标记多模型模式 |
| `RegressionRunner` | 类 | 回归流水线运行器——多模型分支下循环评估每个模型 |
| `plot_residuals(...)` | 函数 | 残差图绘制——每个模型独立调用 |
| `plot_feature_importance(...)` | 函数 | 系数柱状图绘制——每个模型独立调用 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadRegularizationDataset()`——加载 diabetes 并追加共线和噪声 | `DataFrame`，形状 `(442, 22)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.regularization", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainRegularizationModels(...)`——构建三模型 `dict` 并 `fit` | `dict[str, 模型]` |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.regularization", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec -> 加载 -> 标准化 -> 训练 -> 循环评估 -> 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、系数图 | PNG 图像文件 |

### 理解重点

- 正则化回归的工程结构比线性回归多两样：标准化和多模型循环——其余结构完全一致。
- 与决策树回归的差异：决策树有学习曲线和树结构可视化，正则化回归有标准化和多模型但无学习曲线。
- `multiModel=True` 是运行器层的关键分支标志——告诉 `RegressionRunner` 训练返回的是 `dict` 而非单个模型。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.regularization",           # pipeline ID
    TaskType.REGRESSION,                   # 任务类型
    "regression.regularization",           # dataset ID
    RunnerType.REGRESSION,                 # 运行器类型
    trainRegularizationModels,             # 训练函数——返回 dict
    "standardScaler",                      # 预处理——正则化强制标准化
    "randomSplit",                         # 切分策略
    "default",                             # 后处理
    "regression",                          # 输出目录前缀
    "regression",                          # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],                 # 训练后诊断可视化——系数图
    [],                                    # 学习可视化——无学习曲线
    "ridge",                               # 结果存储子目录
    metadata={"multiModel": True},         # 多模型标记——运行器据此分支
)
```

### 理解重点

- `"standardScaler"` 是正则化回归与线性回归、决策树回归在 `PipelineSpec` 层面的关键差异——前者为 `"standardScaler"`，后两者为 `None`。
- `metadata={"multiModel": True}` 告诉运行器训练函数返回的是模型字典——运行器会循环评估每个模型。
- `[]` 学习可视化列表为空——正则化回归不生成学习曲线，这在回归分册中是独特的。
- `"ridge"` 是结果存储子目录名——但实际输出包含 lasso/ridge/elasticnet 三个模型的各自文件。

## 3. 数据依赖关系

```
loadRegularizationDataset()
  - -> X = data.drop(columns=["price"])
  - -> y = data["price"]
  - -> feature_names = list(X.columns)
  - -> train_test_split(test_size=0.2)
      - -> StandardScaler().fit_transform(X_train) ──-> X_train_s
      - -> StandardScaler().transform(X_test) ──-> X_test_s
      - -> trainRegularizationModels(X_train_s, y_train)
          - -> models["lasso"] = Lasso(...).fit()
          - -> models["ridge"] = Ridge(...).fit()
          - -> models["elasticnet"] = ElasticNet(...).fit()
      - -> for name, model in models.items():
          - -> y_pred = model.predict(X_test_s)
          - -> plot_residuals(y_test, y_pred)
          - -> plot_feature_importance(model, feature_names)
```

### 理解重点

- 标准化分支（`fit_transform` / `transform`）是正则化回归独有的——线性回归和决策树回归的数据依赖图中没有这一分支。
- 训练分支产出三个模型——后续所有可视化步骤循环执行三次。
- 没有学习曲线分支——`plot_learning_curve` 不出现在此数据依赖图中。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadRegularizationDataset()` | 加载 `(442, 22)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分 |
| 5 | `StandardScaler().fit_transform(X_train)` | 标准化训练集——**正则化回归独有** |
| 6 | `StandardScaler().transform(X_test)` | 标准化测试集 |
| 7 | 调用 `trainRegularizationModels(X_train_s, y_train)` | 训练三个模型——返回 `dict` |
| 8 | 检测 `multiModel=True` -> 进入多模型循环 | **多模型分支——其他回归模型无此步骤** |
| 9 | 循环内：`model.predict(X_test_s)` | 每个模型独立预测 |
| 10 | 循环内：`plot_residuals(y_test, y_pred)` | 每个模型独立生成残差图 |
| 11 | 循环内：`plot_feature_importance(model, feature_names)` | 每个模型独立生成系数图 |

### 理解重点

- 步骤 5-6 是正则化回归与线性回归/决策树回归在运行器层的根本差异——多了标准化步骤。
- 步骤 8 是多模型检测分支——运行器检查 `metadata["multiModel"]`，若为 `True` 则对 `models.items()` 循环评估。
- 步骤 9-11 在循环内执行三次——相比线性回归的单模型路径，多出了两次 `predict` 和两次可视化调用。

## 5. 正则化回归 vs 线性回归 vs 决策树回归 工程对比

| 工程维度 | 线性回归 | 决策树回归 | 正则化回归 |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | **`trainRegularizationModels`** |
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | **`Lasso`, `Ridge`, `ElasticNet`** |
| 训练函数行数 | 3 行 | ~5 行 | **~10 行** |
| 预处理 | `None` | `None` | **`standardScaler`** |
| 超参数数 | 0 | 3 | **Lasso: 1, Ridge: 1, EN: 2** |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | **`["featureImportance"]`** |
| 学习可视化 | `["learningCurve"]` | `["learningCurve", "treeStructure"]` | **`[]`——无学习曲线** |
| 数据量 | 200（手工合成） | 20640（真实数据） | **442（真实 + 构造）** |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | **坐标下降（Lasso/EN）+ 闭式解（Ridge）** |
| PipelineSpec 元数据 | 无 | 无 | **`{"multiModel": True}`** |

## 常见坑

1. 误以为运行器对所有回归模型统一处理——`multiModel=True` 导致运行器走多模型循环分支，与单模型路径不同。
2. 把 `"ridge"`（输出子目录名）理解成只输出 Ridge 的结果——实际三个模型都有独立输出。
3. 期待正则化回归也有学习曲线——`PipelineSpec` 中学可视化列表明确为 `[]`。
4. 忽略标准化在运行器层而非数据层执行——数据层返回原始值，运行器负责 `StandardScaler` 调用。

## 小结

- 正则化回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按 `multiModel=True` 分支执行。
- 与线性回归/决策树回归的工程差异：（1）唯一使用 `standardScaler` 预处理；（2）唯一返回多模型 `dict`；（3）唯一没有学习曲线；（4）唯一使用 `multiModel` 元数据。
- 标准化 + 多模型循环是正则化回归工程实现的两大核心特征——理解这两点就理解了正则化回归的工程全貌。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对正则化回归核心概念的理解程度。
2. 通过动手练习在代码层面验证 Lasso、Ridge、ElasticNet 的行为差异。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 L1/L2 惩罚、稀疏性、标准化必要性、共线性处理等核心概念的理解 |
| 动手练习 | 实践 | 修改 alpha、l1_ratio、关闭共线/噪声特征——观察三种正则化模型的行为变化 |
| 参考文献 | 入口 | 提供正则化回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. L1 正则化和 L2 正则化的数学惩罚项分别是什么？为什么 L1 能产生稀疏解而 L2 不能？

2. `alpha` 和 `l1_ratio` 分别控制什么？`l1_ratio=0` 和 `l1_ratio=1` 时 ElasticNet 分别退化成什么模型？

3. 为什么正则化回归**必须**标准化而线性回归和决策树回归不需要？从惩罚项对尺度的敏感性解释。

4. 当前数据为什么刻意构造 `bmi_corr`/`bp_corr`/`s5_corr` 和 `noise_1`~`noise_8`？每层特征分别测试正则化的什么能力？

5. `np.sum(np.abs(coef) < 1e-3)` 为什么用 `< 1e-3` 而非 `== 0`？Lasso、Ridge、ElasticNet 的 `near_zero` 预期分别是多少？

6. Lasso 和 Ridge 在面对 `bmi` 和 `bmi_corr` 这对高度相关特征时，系数分配策略有何不同？哪种策略更稳定？

7. 当前正则化回归流水线为什么没有学习曲线？这与 `PipelineSpec` 中的哪个配置字段对应？

## 2. 动手练习

### 练习 1：改变 Lasso 的 alpha

修改 `trainRegularizationModels` 中 Lasso 的 `alpha` 值，分别设为 `0.01`、`0.15`（默认）、`1.0`、`5.0`。

```python
# 在 trainRegularizationModels 中修改
models = {
    "lasso": Lasso(alpha=0.01, max_iter=10000, random_state=randomState),
    # 试试 0.01, 0.15, 1.0, 5.0
}
```

回答：`alpha=0.01` 时近零系数数量是否接近 0？`alpha=5.0` 时是否几乎所有系数都被清零？`noise_*` 系数在哪个 alpha 值开始被清零？

### 练习 2：改变 ElasticNet 的 l1_ratio

将 `l1_ratio` 分别设为 `0.1`、`0.5`（默认）、`0.9`。

```python
# 在 trainRegularizationModels 中修改
"elasticnet": ElasticNet(alpha=0.2, l1_ratio=0.1, max_iter=10000, random_state=randomState),
# 试试 0.1, 0.5, 0.9
```

回答：`l1_ratio=0.1` 时 ElasticNet 的系数分布是否接近 Ridge？`l1_ratio=0.9` 时是否接近 Lasso？`bmi` 和 `bmi_corr` 的系数分配随 `l1_ratio` 如何变化？

### 练习 3：关闭噪声特征

修改 `loadRegularizationDataset` 中噪声特征的数量为 `0`。

```python
# 在 loadRegularizationDataset 中修改循环终止条件
for index in range(0):  # 原为 8——改为 0
    data[f"noise_{index + 1}"] = rng.normal(size=len(data))
```

回答：没有 `noise_*` 特征后，Lasso 的稀疏化优势是否变得不明显？三种模型的近零系数数量是否趋同？残差图是否受影响？

### 练习 4：关闭共线特征

修改 `loadRegularizationDataset` 中共线特征的构造逻辑——注释掉 `bmi_corr`/`bp_corr`/`s5_corr` 的追加。

```python
# 注释掉以下三行
# data["bmi_corr"] = data["bmi"] * 0.9 + rng.normal(scale=0.02, size=len(data))
# data["bp_corr"] = data["bp"] * 0.9 + rng.normal(scale=0.02, size=len(data))
# data["s5_corr"] = data["s5"] * 0.9 + rng.normal(scale=0.02, size=len(data))
```

回答：没有共线特征后，Ridge 和 Lasso 的系数差异是否变小？哪种模型的行为变化最明显？

### 练习 5：手动计算 R^2 并就系数图对比

在流水线预测循环中手动计算并打印 R^2：

```python
from sklearn.metrics import r2_score, mean_squared_error

for name, model in models.items():
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - R^2: {r2:.4f}, MSE: {mse:.4f}")
```

回答：三个模型的 R^2 差距是否显著？"系数更稀疏"和"R^2 更高"是否同时发生？数值指标与残差图的视觉判断是否一致？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3. | 经典教材——线性回归、岭回归与 Lasso 的完整数学推导，含 LAR 算法 |
| 2 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 6. | 入门教材——线性模型选择与正则化的基础直觉和 R/Python 实现 |
| 3 | Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society, Series B, 58(1), 267-288. | Lasso 原始论文——L1 正则化产生稀疏解的理论基础和算法 |
| 4 | Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*. Journal of the Royal Statistical Society, Series B, 67(2), 301-320. | ElasticNet 原始论文——L1+L2 混合正则化的提出与理论分析 |
| 5 | scikit-learn 官方文档 — [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) / [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) / [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) | scikit-learn 的 API 参考——构造器参数、属性和使用示例 |

## 常见坑

1. 修改 `alpha` 后只运行不看 `near_zero`——alpha 变化的核心效果体现在近零系数数量，而非残差图。
2. 调参时同时改多个参数——每次只改一个变量，才能确定是哪个参数导致的行为变化。
3. 关闭噪声/共线特征后忘记恢复——建议在修改前用 `git stash` 保存原始状态。
4. 只看一个模型的系数不看三模型对比——正则化回归的诊断价值在于"对比"，单独看一个模型意义有限。
5. 在未标准化的数据上开启正则化——若跳过了 `StandardScaler`，系数形态会完全不可预期。

## 小结

- 7 个自检问题覆盖正则化回归的核心概念：L1/L2 惩罚、稀疏性几何直觉、标准化必要性、三层数据结构、近零系数、共线性处理和 PipelineSpec 配置。
- 5 个动手练习从不同角度探索正则化行为——调 $\alpha$、调 l1_ratio、关闭噪声、关闭共线、计算数值指标。
- 5 篇参考文献覆盖 ESL、ISLR 两本经典教材、Lasso 和 ElasticNet 两篇原始论文、scikit-learn 官方 API 文档——构成完整的正则化回归学习路线。
