---
title: 线性回归实现
date: 2026-03-15
category: 机器学习/回归
tags:
  - Python
  - Scikit-learn
description: 从数学原理、损失函数、正规方程推导，到使用Python和scikit-learn实现数据生成、探索、可视化、预处理、训练、评估及结果可视化的完整线性回归项目指南。
image: https://img.yumeko.site/file/blog/cover/1780581793200.webp
status: published
---

# 数学原理

## 本章目标

1. 理解线性回归的模型形式——目标值是特征的线性组合加上截距和噪声。
2. 理解最小二乘法（OLS）的目标函数、正规方程闭式解及其成立条件。
3. 理解极大似然估计视角——高斯噪声假设下 OLS 等价于 MLE。
4. 把这些数学表达和当前源码中的 `coef_`、`intercept_` 对应起来。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性模型 | 模型形式 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$——用线性函数拟合连续值目标 |
| OLS | 优化目标 | $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$——最小化残差平方和 |
| 正规方程 | 闭式解 | $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$——OLS 的解析解 |
| MLE | 概率视角 | 高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 下最大化似然等价于最小化 RSS |
| `coef_` / `intercept_` | 源码属性 | 训练后 $\mathbf{w}$ 和 $b$ 在工程中的直接映射 |

## 1. 模型定义

线性回归假设目标变量 $y$ 与特征 $\mathbf{x} = (x_1, \dots, x_d)^T$ 之间存在线性关系：

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b
$$

引入扩展向量 $\tilde{\mathbf{x}} = (1, x_1, \dots, x_d)^T$，$\tilde{\mathbf{w}} = (b, w_1, \dots, w_d)^T$，可统一写为：

$$
\hat{y} = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}
$$

其中 $\mathbf{w} \in \mathbb{R}^d$ 是系数向量，$b \in \mathbb{R}$ 是截距。

### 理解重点

- 线性回归预测的本质是"特征乘系数，再加截距"——每个特征独立贡献，最终求和。
- 当前数据中的 `面积`、`房间数`、`房龄` 分别乘上自己的系数再相加，加上截距得到预测房价。
- 训练完成后，`model.coef_` 和 `model.intercept_` 就是学到的 $\mathbf{w}$ 和 $b$。

## 2. 最小二乘法（OLS）

对 $N$ 个训练样本，定义残差平方和（RSS）为损失函数：

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{N} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2 = \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ 是扩展设计矩阵（第一列全为 1），$\mathbf{y} \in \mathbb{R}^N$ 是目标向量。

OLS 的目标是找到使 $\mathcal{L}$ 最小的 $\tilde{\mathbf{w}}$。

### 理解重点

- OLS 的目标非常直接——让所有样本的预测误差平方和尽可能小。
- 平方惩罚意味着大误差会受到不成比例的重罚——一个偏离 10 的样本比十个偏离 1 的样本对损失函数的贡献更大。
- 当前代码没有手写这个损失函数——`LinearRegression()` 内部求解的正是这个问题。

## 3. 正规方程：闭式解

展开损失函数并求导：

$$
\mathcal{L} = \mathbf{y}^T\mathbf{y} - 2\tilde{\mathbf{w}}^T\mathbf{X}^T\mathbf{y} + \tilde{\mathbf{w}}^T\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{w}}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\tilde{\mathbf{w}} = 0
$$

得到**正规方程**（Normal Equation）：

$$
\boxed{\tilde{\mathbf{w}}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$

### 理解重点

- 正规方程给出了 OLS 的**闭式解**——不需要迭代，一次矩阵运算即可得到最优参数。
- 对于 $d=3$、$N=160$（训练集大小），$\mathbf{X}^T\mathbf{X}$ 是 $4 \times 4$ 矩阵——求逆计算几乎瞬间完成。
- 当前仓库没有手写矩阵求逆——scikit-learn 使用 `scipy.linalg.lstsq`（基于 SVD）求解，数值更稳定。

## 4. 正规方程的成立条件

$\mathbf{X}^T\mathbf{X}$ 必须可逆。当：

- 特征数 $d$ 大于样本数 $N$
- 特征间存在精确线性关系（多重共线性）

时，$\mathbf{X}^T\mathbf{X}$ 奇异或近似奇异，正规方程数值不稳定。

### 理解重点

- 当前数据只有 3 个特征、160 个训练样本——$N \gg d$，$\mathbf{X}^T\mathbf{X}$ 通常满秩。
- 特征之间相对独立（面积、房间数、房龄来自独立均匀采样）——无严重共线性。
- 当特征高度相关或 $d > N$ 时，需要正则化（Ridge/Lasso）——这是后续正则化分册要解决的问题。

## 5. 极大似然估计视角

从概率视角看，假设目标值由线性函数加高斯噪声生成：

$$
y_i = \mathbf{w}^T \mathbf{x}_i + b + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

单个样本的似然：

$$
P(y_i \mid \mathbf{x}_i, \mathbf{w}, b) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2}{2\sigma^2}\right)
$$

对数似然：

$$
\ln L = -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i - b)^2
$$

最大化 $\ln L$ 等价于最小化 $\sum (y_i - \hat{y}_i)^2$——即 OLS。

### 理解重点

- 当前数据生成函数**确实含有高斯噪声** `rng.normal(0, 10, size=n_samples)`——MLE 的高斯假设与数据生成过程完全一致。
- OLS 不只是一个代数技巧——它是高斯噪声假设下最自然的参数估计方法。
- 理解这层关系，有助于后续过渡到正则化（从 MLE 到 MAP）和贝叶斯线性回归。

## 6. 常见评估指标（理论层）

| 指标 | 公式 | 含义 |
|---|---|---|
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | 均方误差——预测误差平方的平均 |
| RMSE | $\sqrt{\text{MSE}}$ | 均方根误差——与目标同量纲 |
| MAE | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | 平均绝对误差——对异常值不敏感 |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数——模型解释的目标方差比例，$\le 1$，越接近 1 越好 |

### 理解重点

- 这些是线性回归理论上最常用的评估指标——但**当前流水线未显式打印**任何数值指标。
- 当前评估侧重图形化诊断（残差图 + 学习曲线）——学习曲线内部使用 $R^2$ 作为评分。
- 区分"理论上常见"与"当前实现真实输出"——不可将公式表当成已实现的功能。

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 线性模型 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ | `LinearRegression()` |
| 系数向量 | $\mathbf{w} = (w_1, w_2, w_3)^T$ | `model.coef_`——长度为 3 的数组 |
| 截距 | $b$ | `model.intercept_`——标量 |
| 设计矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ | `X_train`（扩展列由 scikit-learn 内部处理） |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train` |
| OLS 求解 | $\tilde{\mathbf{w}}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | `model.fit(X_train, y_train)`——基于 SVD 的数值求解 |
| 预测 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ | `model.predict(X_test)` |
| 残差 | $e_i = y_i - \hat{y}_i$ | `y_test - y_pred`——残差图的数据源 |
| 训练样本数 | $N$ | `n_samples=200`——`test_size=0.2` 后训练集约 160 |
| 特征维度 | $d$ | `3`——面积、房间数、房龄 |

## 8. 线性回归 vs 决策树回归 数学对比

| 维度 | 线性回归 | 决策树回归 |
|---|---|---|
| 模型形式 | $\hat{y} = \mathbf{w}^T \mathbf{x} + b$——全局线性 | **$\hat{y} = \sum_m \hat{c}_m \mathbb{1}(\mathbf{x} \in R_m)$——分段常数** |
| 目标函数 | $\min \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2$——凸优化 | **$\min_{j,s} \sum_{R_1,R_2} (y_i - \hat{c})^2$——贪心搜索** |
| 求解方式 | 闭式解（正规方程或 SVD） | **贪心递归——无闭式解** |
| 参数数 | $d + 1$（系数 + 截距）——固定 | **叶子节点数——随数据增长** |
| 非线性处理 | 需基函数展开或特征工程 | **天然支持——递归分裂即是非线性** |
| 特征交互 | 需手工构造交互项 | **自然通过条件分支捕获** |
| 可解释性 | 极强——一个系数一个影响方向 | 中等——重要性无方向 |
| 外推能力 | 有——可线性外推 | **无——叶子边界外预测为常数** |

## 常见坑

1. 把正规方程当成当前源码显式写出的训练逻辑——实际上仓库调用的是 scikit-learn 基于 SVD 的实现。
2. 忽略高斯噪声假设与 OLS 的关系——只把最小二乘当成纯公式，错过了概率建模的统一视角。
3. 把理论上的 MSE/MAE/RMSE/$R^2$ 指标表误读成当前流水线已打印输出——实际只用了图形化评估。
4. 看到当前实现没有标准化就认为"线性回归永远不需要标准化"——当使用梯度下降求解或正则化时标准化是必需的。

## 小结

- 线性回归的数学核心链：线性模型 $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ → OLS $\min \|\mathbf{y} - \mathbf{X}\tilde{\mathbf{w}}\|^2$ → 正规方程闭式解 → MLE 概率解释 → `coef_`/`intercept_` 工程映射。
- 与决策树回归的根本区别：全局线性函数 vs 分段常数，闭式解 vs 贪心搜索，系数可解释 vs 重要性无方向。
- 当前源码 `LinearRegression()` 是 OLS 的最简教学实现——无超参数、无标准化、关系透明，是回归学习的逻辑起点。

# 数据构成

## 本章目标

1. 明确本仓库线性回归数据来自 `RegressionDatasetFactory.loadLinearRegressionDataset()` 手工合成的线性房价数据。
2. 理解显式生成公式——`price = 2×面积 + 10×房间数 - 3×房龄 + N(0,10²) + 50`——与训练结果之间的对照关系。
3. 明确训练/测试集切分方式，以及当前实现**没有标准化**这一关键事实。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory.loadLinearRegressionDataset()` | 方法 | 手工合成 3 特征线性房价数据——含真实生成公式 |
| `面积` | 列 | 房屋面积特征——范围 $[20, 80]$，正向影响房价 |
| `房间数` | 列 | 房屋房间数量特征——范围 $[1, 5]$，正向影响房价 |
| `房龄` | 列 | 房屋年龄特征——范围 $[1, 20]$，负向影响房价 |
| `price` | 列 | 回归目标——由显式线性公式加高斯噪声生成 |

## 1. 数据生成：`RegressionDatasetFactory.loadLinearRegressionDataset()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `nSamples` | `int` | 样本数。`200`——适中规模，OLS 可瞬间求解 | `200`、`500`、`1000` |
| `randomState` | `int` | 随机种子。`42`——保证数据可复现 | `42` |
| `area` | `ndarray`，形状 `(200,)` | 面积——$\mathcal{U}(20, 80)$ | `rng.uniform(20, 80, ...)` |
| `rooms` | `ndarray`，形状 `(200,)` | 房间数——$\mathcal{U}(1, 5)$ | `rng.uniform(1, 5, ...)` |
| `age` | `ndarray`，形状 `(200,)` | 房龄——$\mathcal{U}(1, 20)$ | `rng.uniform(1, 20, ...)` |
| 返回值 | `DataFrame` | 含 `面积`、`房间数`、`房龄`、`price` 四列 | — |

### 示例代码

```python
rng = np.random.RandomState(42)
area = rng.uniform(20, 80, size=200)
rooms = rng.uniform(1, 5, size=200)
age = rng.uniform(1, 20, size=200)
noise = rng.normal(0, 10, size=200)
price = 2 * area + 10 * rooms - 3 * age + noise + 50
```

### 生成公式

$$
\text{price} = 2 \times \text{面积} + 10 \times \text{房间数} - 3 \times \text{房龄} + \epsilon + 50, \quad \epsilon \sim \mathcal{N}(0, 10^2)
$$

### 理解重点

- 这是**完全手动合成**的数据——生成公式是显式写出的，不是从真实世界采集的。
- 这种设计的最大教学价值：训练得到的 `coef_` 和 `intercept_` 可以直接与真实系数 `[2, 10, -3]` 和截距 `50` 对照——透明地验证 OLS 的正确性。
- 高斯噪声的标准差为 10——相对于房价范围（约 30~300），噪声水平适中，训练结果会接近但不会精确等于真实公式。
- 三个特征来自独立的均匀分布——特征间无共线性，$\mathbf{X}^T\mathbf{X}$ 条件良好，OLS 求解稳定。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `面积` | `Series`，形状 `(200,)` | 房屋面积，范围 $[20, 80]$ | `data["面积"]` |
| `房间数` | `Series`，形状 `(200,)` | 房间数量，范围 $[1, 5]$ | `data["房间数"]` |
| `房龄` | `Series`，形状 `(200,)` | 房屋年龄，范围 $[1, 20]$ | `data["房龄"]` |
| `price` | `Series`，形状 `(200,)` | 目标变量——由显式公式生成 | `data["price"]` |

各特征的真实影响：

| 特征 | 真实系数 | 影响方向 | 单位影响的房价变化 |
|---|---|---|---|
| `面积` | `+2` | 正——面积越大，房价越高 | 面积每增加 1 单位，房价增加 2 |
| `房间数` | `+10` | 正——房间越多，房价越高 | 房间数每增加 1，房价增加 10 |
| `房龄` | `-3` | 负——房龄越大，房价越低 | 房龄每增加 1 年，房价减少 3 |
| 截距 | `+50` | — | 所有特征取 0 时的基线房价 |

### 理解重点

- 因为真实系数已知，训练后可以直接验证：`面积` 的系数是否接近 `2`？`房龄` 是否接近 `-3`？——这是手工合成数据的核心诊断价值。
- 特征使用中文命名——这在本仓库中独树一帜，训练日志直接显示 `面积`、`房间数`、`房龄`，可读性极强。
- 三个特征量纲不同但量级接近（20~80 vs 1~5 vs 1~20）——不标准化时系数的绝对值仍有可比性。

## 3. 数据切分

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比。`0.2`——200 × 0.2 = 40 个测试样本 | `0.2` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `tuple` | `(X_train, X_test, y_train, y_test)`——训练集 160 样本，测试集 40 样本 | — |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 当前使用随机切分（`randomSplit`）——回归目标 `price` 是连续值，无法分层。
- 训练集 160 样本对 3 特征线性回归绰绰有余——$N \gg d$，参数估计稳定。
- 与决策树回归/SVR 使用相同的切分配置——保证不同模型在同一数据划分上可比。

## 4. 为什么当前实现没有标准化

### 参数速览

| 项目 | 当前状态 | 原因 |
|---|---|---|
| 标准化 | **未使用** | 数据量纲直观、关系简单——正规方程/SVD 求解无需标准化；当前实现聚焦系数可解释性 |

### 理解重点

- 线性回归的闭式解（正规方程/SVD）在数学上不需要标准化——解是精确的，不依赖特征尺度。
- 但标准化**在以下场景非常重要**：（1）使用梯度下降求解时加速收敛；（2）正则化（Ridge/Lasso）需要统一惩罚尺度；（3）比较不同量纲特征的系数大小时。
- 当前实现不标准化是因为数据本身量纲接近且关系简单——这是有意保留的最简教学配置，不代表"线性回归永远不需要标准化"。

## 5. 数据设计意图：与决策树回归/SVR 的对比

| 数据维度 | 线性回归 | 决策树回归 | SVR |
|---|---|---|---|
| 数据来源 | 手工合成——显式线性公式 | California Housing 真实数据 | `make_friedman1`——非线性合成 |
| 样本数 | 200 | 20640 | 200 |
| 特征维度 | 3 | 8 | 10 |
| 真实关系 | **已知——`2×面积 + 10×房间数 - 3×房龄 + 50`** | 未知——真实世界复杂关系 | 已知——Friedman 非线性函数 |
| 噪声 | 显式高斯 $N(0,10^2)$ | 真实世界噪声 | 显式高斯噪声 |
| 标准化 | 无 | 无 | 有（`StandardScaler`） |
| 设计意图 | **系数可验证——关系透明的教学基线** | 真实数据非线性 + 特征交互 | 核方法非线性拟合 |

### 理解重点

- 三种数据设计形成清晰的递进：线性回归用显式线性公式（关系完全透明）→ 决策树回归用真实数据（关系复杂未知）→ SVR 用合成非线性（关系已知但非线性）。
- 线性回归数据的核心价值在于"已知答案"——训练后可以定量验证 OLS 恢复了多少真实信号。
- 200 样本是有意的小规模——足以展示 OLS 的基本行为，又保持训练和可视化的即时性。

## 数据可视化

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/linear_regression/data_correlation.png)

![特征与目标变量关系](https://img.yumeko.site/file/articles/ML/linear_regression/data_feature_vs_price.png)

## 常见坑

1. 把当前数据误认为真实房价数据集——它是按显式公式合成的教学数据，真实关系完全已知。
2. 看到回归任务就默认写入标准化步骤——当前源码**没有**标准化，且这是正确的设计选择。
3. 忽略噪声项的存在，误以为训练后系数一定会精确等于 `2`、`10`、`-3`——噪声 $\sigma=10$ 意味着训练结果会有偏差。
4. 期待在 200 样本上得到极端精确的系数估计——样本量有限，系数波动在统计上是正常的。

## 小结

- 当前线性回归数据来自 `RegressionDatasetFactory.loadLinearRegressionDataset()`——手工合成 3 特征 200 样本的线性房价数据，生成公式完全透明。
- 数据流为：独立均匀采样 → 线性组合 + 高斯噪声 → DataFrame（`面积`/`房间数`/`房龄`+`price`）→ 随机切分（`test_size=0.2`）→ 直接训练（无标准化）。
- 透明的关系设计使线性回归分册成为"系数可验证"的教学基线——训练结果可直接与真实公式对照，这是所有其他回归分册不具备的核心优势。

# 思路与直觉

## 本章目标

1. 理解线性回归的核心直觉——用一条直线/超平面近似特征与目标的关系，系数直接表达影响方向与大小。
2. 理解为什么当前手工合成的透明数据是建立回归直觉的最佳起点。
3. 建立"系数正负 → 影响方向，系数大小 → 影响强度，截距 → 基线"的直觉链。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `LinearRegression` | 模型 | 用线性函数拟合连续值目标——回归建模的起点 |
| `coef_` | 属性 | 各特征的线性系数——正值表示正向影响，负值表示负向影响 |
| `intercept_` | 属性 | 截距——所有特征取 0 时的基线预测值 |
| 真实生成公式 | 数据 | `price = 2×面积 + 10×房间数 - 3×房龄 + noise + 50`——训练后可直接对照 |

## 1. 线性回归想做什么

线性回归的核心目标是用最简单的数学形式——特征加权求和——去近似连续值目标。它不寻找复杂的非线性模式，而是在所有可能的直线/超平面中找到使预测误差平方和最小的那一个。

### 理解重点

- 如果数据关系本身接近线性，线性回归能用极简的模型给出不错的拟合——这就是"奥卡姆剃刀"在回归建模中的体现。
- 相比决策树的分段常数、SVR 的核映射，线性回归最大的优势是**结果可直接解释**——一个系数对应一个特征的影响。
- 当前这份手工合成的数据正是为了最大化这种"简单但可解释"的优势——关系是精确线性的（仅叠加了可控噪声）。

## 2. 为什么当前合成数据是建立直觉的最佳起点

当前数据的生成公式完全透明：

$$
\text{price} = 2 \times \text{面积} + 10 \times \text{房间数} - 3 \times \text{房龄} + \epsilon + 50
$$

这意味着训练前你就知道"正确答案"——训练后可以直接验证模型是否学到了这些系数。

### 理解重点

- 在真实数据（如 California Housing）上训练线性回归——你永远不知道"真实系数"是什么，只能间接评估。
- 在当前合成数据上——你可以直接对比 `coef_` 和 `[2, 10, -3]`，定量衡量 OLS 的恢复精度。
- 这种"有标准答案"的设计是建立回归直觉的最有效方式——先在有答案的数据上理解模型行为，再迁移到真实数据。

## 3. 如何理解系数的正负和大小

| 系数现象 | 直观含义 | 当前数据示例 |
|---|---|---|
| 系数为正 | 特征增大 → 预测值增大 | `面积` 系数 ≈ 2——面积越大，房价越高 |
| 系数为负 | 特征增大 → 预测值减小 | `房龄` 系数 ≈ -3——房龄越大，房价越低 |
| 系数绝对值大 | 该特征的单位变化对预测值影响大 | `房间数` 系数 ≈ 10——每多一个房间，房价多 10 单位 |
| 系数接近 0 | 该特征对预测值几乎无影响 | 如果加入无关特征，其系数应接近 0 |

### 理解重点

- 系数的正负直接告诉你"这个特征对目标的影响方向"——这是线性回归独有的可解释性优势。决策树的特征重要性只告诉你"用了多少次"，不告诉你方向。
- 系数的大小表示"在其他特征不变时，该特征变化一个单位，预测值变化多少"——这是经济学中"边际效应"的数学表达。
- 当前数据中 `房间数` 的系数（10）远大于 `面积`（2）——意味着"多一个房间"比"面积大一单位"对房价的推升效果强 5 倍。

## 4. 如何理解截距

截距 $b$ 是所有特征都取 0 时的预测值。在当前数据中，真实截距是 50。

### 理解重点

- 截距不是"某个特征的重要程度"——它是整个线性关系的基线。即使所有特征都为 0，房价也有一个基础值。
- 在当前房价语境中，截距 50 可以理解为"地段/土地的固有价值"——即使面积为零、没有房间、房龄为零，土地本身仍有价值。
- 训练得到的截距通常接近但不精确等于 50——因为噪声 $\sigma=10$ 和有限样本使得估计存在统计波动。

## 5. 直觉如何映射到训练日志

训练完成后，终端日志最核心的输出是：

```
截距(intercept): XX.XX
斜率(coefficients):
  面积: X.XX
  房间数: X.XX
  房龄: X.XX
```

### 理解重点

- 如果 `房龄` 的系数被学成**正数**——说明模型结果与真实关系矛盾（真实是 -3），需要检查数据或样本波动。
- 如果三项系数的正负号都正确（面积正、房间数正、房龄负）——说明 OLS 成功捕捉了数据中的真实信号方向。
- 系数值与真实值 `[2, 10, -3]` 的偏差主要来自噪声项——噪声越大、样本越少，偏差通常越大。

## 6. 与决策树回归的直觉对比

| 直觉维度 | 线性回归 | 决策树回归 |
|---|---|---|
| 模型思维 | "所有特征一起贡献一个加权和" | **"先问收入高不高，再问位置在哪"——if-else 链** |
| 预测形态 | 连续超平面——输入微小变化，输出也微小变化 | **分段常数——同一叶子的样本输出相同** |
| 关系表达 | 全局统一公式 | **局部独立规则** |
| 解释方式 | 系数正负与大小 | **特征重要性排名** |
| 适合场景 | 关系近线性、需要系数解释 | **非线性、特征交互复杂、不需要系数方向** |
| 学习难度 | 极低——近乎本能 | 中等——需要理解深度和叶子约束 |

### 理解重点

- 线性回归是"一条公式打天下"——简单、稳定、可解释，但假设强。
- 决策树回归是"不同区域不同规则"——灵活、可处理非线性，但可解释性弱于系数。
- 当前仓库先讲线性回归，正是因为它的直觉最容易建立——理解"系数是什么"之后，再理解"为什么树不需要系数"会更有对比感。

## 常见坑

1. 把系数大小直接跨不同量纲特征比较——当前三个特征量纲接近所以可以直接比，但在量纲悬殊的数据上需先标准化再比较。
2. 把截距误解成"某个特征的权重"——截距是独立的基线项，不与任何特征挂钩。
3. 看到训练结果和真实公式不完全一致就误以为模型出错——噪声和有限样本导致的偏差是统计学习的固有属性，不是 bug。

## 小结

- 线性回归的核心直觉：加权求和 → 最小化误差 → 得到一组可解释的系数——每个系数直接告诉你特征对目标的影响方向与大小。
- 当前手工合成数据是建立回归直觉的最佳起点——真实关系完全透明，训练后可直接验证 OLS 是否恢复了正确答案。
- 建立"系数正负 → 方向，系数大小 → 强度，截距 → 基线"的直觉链之后，再看数学原理、训练流程和评估会非常顺畅。

# 模型构建

## 本章目标

1. 明确 `trainLinearRegressionModel(...)` 如何构建并训练 `LinearRegression`——本仓库最简训练函数。
2. 理解 `coef_` 和 `intercept_` 的含义及其与真实生成公式的对照关系。
3. 看清 `feature_names` 的处理逻辑——如何让训练日志中的系数与中文列名一一对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练一个 `sklearn.linear_model.LinearRegression` 模型——最简薄封装 |
| `LinearRegression()` | 类 | scikit-learn 提供的普通最小二乘线性回归器——无超参数 |
| `model.fit(X_train, y_train)` | 方法 | 基于 SVD 求解 OLS——返回 `coef_` 和 `intercept_` |
| `model.coef_` | 属性 | 各特征对应的线性系数 $\mathbf{w}$——形状 `(3,)` |
| `model.intercept_` | 属性 | 截距 $b$——标量 |

## 1. `trainLinearRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainLinearRegressionModel(XTrain, yTrain)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(160, 3)` | 训练特征矩阵——面积、房间数、房龄 | `X_train` |
| `yTrain` | `ndarray`，形状 `(160,)` | 训练目标值——房价 | `y_train` |
| 返回值 | `LinearRegression` | 已完成 `fit()` 的模型对象——含 `coef_` 和 `intercept_` | — |

### 示例代码

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
# model.coef_      ≈ [2.0, 10.0, -3.0]
# model.intercept_ ≈ 50.0
```

### 理解重点

- 这是本仓库**最简训练函数**——没有超参数、没有装饰器、没有耗时统计，仅 3 行代码。
- 与决策树回归的 `trainDecisionTreeRegressionModel` 形成鲜明对比——后者有 3 个复杂度超参数。
- `LinearRegression()` 的无参设计是因为 OLS 无需调参——最优解由数据通过 SVD 唯一确定。

## 2. `LinearRegression()` 的构造器参数

### 参数速览

适用 API：`LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fit_intercept` | `bool` | 是否拟合截距。默认 `True`——当前源码未显式写出 | `True` |
| `copy_X` | `bool` | 是否复制输入数据。默认 `True` | `True` |
| `n_jobs` | `int` 或 `None` | 并行计算线程数。默认 `None`——单线程 | `None`、`-1` |

### 理解重点

- 当前源码使用全部默认参数——`LinearRegression()` 无参构造是 scikit-learn 中最简洁的模型之一。
- `fit_intercept=True` 意味着模型会学习截距 $b$——不需要手动在数据中加一列 1。
- 没有 `random_state` 参数——因为 OLS 的解是确定性的（给定相同数据，结果永远相同），不存在随机性。

## 3. 训练完成后的关键属性

### 参数速览

| 属性 | 类型 | 数学含义 | 示例取值 |
|---|---|---|---|
| `coef_` | `ndarray`，形状 `(3,)` | 系数向量 $\mathbf{w} = [w_1, w_2, w_3]$ | `[2.03, 9.87, -2.94]`（接近 $[2, 10, -3]$） |
| `intercept_` | `float` | 截距 $b$ | `51.23`（接近 $50$） |
| `rank_` | `int` | 设计矩阵 $\mathbf{X}$ 的秩 | `3`（= 特征数，满秩） |
| `singular_` | `ndarray` | $\mathbf{X}$ 的奇异值 | 内部使用——通常不需要关注 |

### 示例代码

```python
print(f"截距(intercept): {model.intercept_:.2f}")
print("斜率(coefficients):")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

### 输出

```text
截距(intercept): 51.23
斜率(coefficients):
  面积: 2.03
  房间数: 9.87
  房龄: -2.94
```

### 理解重点

- `coef_` 的值应与真实系数 `[2, 10, -3]` 接近——正负方向应完全一致，数值因噪声而有小幅偏差。
- `intercept_` 应接近 `50`——偏差同样来自噪声和有限样本。
- 三个系数的正负号正确（面积+、房间数+、房龄-）比数值精确更重要——方向正确说明模型学到了真实的数据模式。

## 4. `feature_names` 的处理

`trainLinearRegressionModel` 在打印系数日志时，需要特征名来提升可读性。当前源码在流水线层处理特征名：

### 示例代码

```python
feature_names = list(X.columns)  # ["面积", "房间数", "房龄"]
# 训练后将系数与特征名一一对应打印
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name}: {coef:.2f}")
```

### 理解重点

- 特征名处理在流水线层面而非训练函数内部——训练函数只关心数值矩阵，不关心列名。
- 中文列名（`面积`、`房间数`、`房龄`）在日志中直接显示——比英文列名更具可读性。
- `feature_names` 是贯穿流水线的重要中间变量——在训练日志和后续可视化的标题中都会用到。

## 5. 线性回归 vs 决策树回归 模型参数对比

| 参数/属性 | 线性回归 | 决策树回归 |
|---|---|---|
| 构造器参数 | `fit_intercept`（1 个可选） | **`max_depth`、`min_samples_split`、`min_samples_leaf`、`random_state`（4 个）** |
| 训练方式 | SVD 闭式解——确定性 | **CART 贪心递归——含随机性** |
| 核心属性 | `coef_`、`intercept_` | **`feature_importances_`、`get_depth()`、`get_n_leaves()`** |
| 属性数量 | 4（含 `rank_`、`singular_`） | **多个——`tree_` 含完整节点结构** |
| 超参数调优 | 不需要（无超参数） | **需要——深度和叶子约束直接影响泛化** |
| 训练耗时 | 极短（$O(d^3 + Nd^2)$，$d=3$ 时毫秒级） | 短（$O(d \cdot N \log N)$） |
| 预测输出 | 连续值（线性函数） | 连续值（分段常数） |

## 常见坑

1. 期待 `LinearRegression()` 有丰富的超参数——它是 scikit-learn 中最简模型之一，仅 `fit_intercept` 一个实质参数。
2. 把 `coef_` 的返回值顺序搞错——`coef_[0]` 对应 `X` 的第一列，需与 `feature_names` 对齐。
3. 忽略 `feature_names` 的作用——数组输入时日志只会显示 `Feature_0, Feature_1, ...`，丧失可读性。

## 小结

- `trainLinearRegressionModel(...)` 是本仓库最简训练函数——仅 3 行，对 `LinearRegression()` 做最薄的调用封装。
- `LinearRegression()` 使用 SVD 求解 OLS——无超参数、无随机性、确定性输出——`coef_` 和 `intercept_` 是唯一的训练结果。
- 与决策树回归的模型构建形成清晰对比：线性回归追求"极简 + 可解释"，决策树回归追求"灵活 + 需约束"。

# 训练与预测

## 本章目标

1. 理解线性回归流水线的完整执行顺序——从数据加载到残差图和学习曲线输出。
2. 理解 OLS 的训练过程——SVD 闭式求解，无需迭代，无收敛判断。
3. 理解 `predict` 的预测方式——简单的矩阵乘法 $\hat{y} = \mathbf{X}\mathbf{w} + b$。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练线性回归模型——基于 SVD 的闭式求解 |
| `model.fit(X_train, y_train)` | 方法 | 求解 $\min_{\mathbf{w},b} \|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|^2$——一次计算完成 |
| `model.predict(X_test)` | 方法 | 对测试集做矩阵乘法预测——$\hat{y} = \mathbf{X}\mathbf{w} + b$ |
| `plot_residuals(...)` | 函数 | 绘制预测-真实散点图 + 残差分布图 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证 R² 随样本量变化的曲线 |

## 1. 完整流水线流程

### 流程概述

```
loadLinearRegressionDataset()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ model = trainLinearRegressionModel(X_train, y_train)
    ├─ ④ y_pred = model.predict(X_test)
    ├─ ⑤ plot_residuals(y_test, y_pred)
    └─ ⑥ plot_learning_curve(LinearRegression(), X_train, y_train, scoring="r2")
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadLinearRegressionDataset` | — | `DataFrame`，`(200, 4)` | 手工合成线性房价 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X` `(200, 3)`、`y` `(200,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`、`y` | `X_train` `(160, 3)`、`X_test` `(40, 3)` | `test_size=0.2`，无标准化 |
| 训练 | `trainLinearRegressionModel` | `X_train`、`y_train` | `LinearRegression` | SVD 闭式求解——瞬间完成 |
| 预测 | `model.predict` | `X_test` | `y_pred` `(40,)` | 矩阵乘法 |
| 残差图 | `plot_residuals` | `y_test`、`y_pred` | PNG 图像 | 误差分布诊断 |
| 学习曲线 | `plot_learning_curve` | 新 `LinearRegression()`、`X_train`、`y_train` | PNG 图像 | 样本量-得分趋势 |

### 理解重点

- 这是本仓库**最简流水线**——仅 6 步，无标准化、无特征重要性、无树结构，聚焦于系数解释和残差诊断。
- 训练步骤耗时极短（毫秒级）——3 个特征 × 160 样本的 SVD 求解计算量极小。
- 与决策树回归流水线的对比：决策树多出特征重要性和树结构图两步，训练为贪心递归而非闭式求解。

## 2. 训练细节：SVD 闭式求解

### 算法流程

```
输入 X_train (160, 3), y_train (160,)
    ↓
① 构建设计矩阵: X̃ = [1, X_train] → (160, 4)
② 对 X̃ 做奇异值分解: X̃ = U Σ V^T
③ 计算: w̃* = V Σ^{-1} U^T y_train
④ 返回: coef_ = w̃*[1:], intercept_ = w̃*[0]
```

### 理解重点

- scikit-learn 的 `LinearRegression` 使用 `scipy.linalg.lstsq`（基于 SVD 或 QR 分解）求解——比直接计算 $(\mathbf{X}^T\mathbf{X})^{-1}$ 的数值稳定性更好。
- 训练是**一次性**的——没有迭代、没有收敛判断、没有 `n_iter` 或 `tol` 参数。
- 这是 OLS 与所有迭代式训练算法（EM、Baum-Welch、梯度下降）的根本区别——OLS 保证找到全局最优解，且一步到位。

## 3. 预测细节：矩阵乘法

对测试样本矩阵 $\mathbf{X}_{\text{test}}$：

$$
\hat{\mathbf{y}} = \mathbf{X}_{\text{test}} \mathbf{w} + b = \tilde{\mathbf{X}}_{\text{test}} \tilde{\mathbf{w}}
$$

### 理解重点

- 预测完全不涉及训练数据——模型参数 $\mathbf{w}$ 和 $b$ 已经固化在 `coef_` 和 `intercept_` 中。
- 预测复杂度为 $O(N_{\text{test}} \cdot d) = O(40 \times 3)$——几乎瞬时。
- 与决策树回归的预测对比：线性回归做矩阵乘法（全局统一公式），决策树沿树走到叶子（局部 if-else 路径）。

## 4. 与决策树回归训练流程的对比

| 步骤 | 线性回归 | 决策树回归 |
|---|---|---|
| 数据 | 手工合成 `(200, 3)` | 真实数据 `(20640, 8)` |
| 标准化 | 无 | 无 |
| 训练算法 | SVD 闭式解——一次性完成 | CART 贪心递归——逐层分裂 |
| 训练复杂度 | $O(d^3 + Nd^2)$——极快 | $O(d \cdot N \log N)$——快 |
| 是否需要 `random_state` | 否——确定性解 | **是——分裂涉及随机性** |
| 收敛判断 | 不需要——闭式解一次到位 | **需要 `max_depth`/`min_samples_split` 等早停** |
| 预测 | $\hat{y} = \mathbf{X}\mathbf{w} + b$（矩阵乘法） | **沿树走到叶子 → 返回叶子均值** |
| 评估可视化 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构** |

## 常见坑

1. 在 `LinearRegression()` 上期待看到 `n_iter` 或训练耗时——它是一次性闭式求解，没有迭代过程。
2. 把 `plot_learning_curve` 传入已训练的 `model`——学习曲线需要未训练的模型实例做交叉验证。
3. 在 200 样本上期待看到学习曲线中训练/验证得分的巨大差异——线性回归参数少（4 个），小样本下也不容易过拟合。

## 小结

- 线性回归流水线为最简 6 步：加载 → 拆分 → 切分 → 训练 → 预测 → 残差图 + 学习曲线——无标准化、无特征重要性、无树结构。
- `fit()` 的核心是 SVD 闭式求解——一次计算，无迭代，无收敛判断，是 OLS 区别于所有迭代式算法的最本质特征。
- `predict()` 是简单的矩阵乘法——测试样本与固定参数做线性组合，计算量极小。

# 评估与诊断

## 本章目标

1. 理解当前线性回归流水线的两类评估输出——残差图和学习曲线。
2. 理解每类评估分别诊断什么——残差图看误差分布，学习曲线看泛化趋势。
3. 明确当前流水线**已实现**和**未实现**的评估项——线性回归评估以图形化诊断为主。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 生成预测-真实散点图 + 残差分布图——诊断拟合质量和系统偏差 |
| `plot_learning_curve(...)` | 函数 | 生成训练/验证 R² 曲线——诊断样本量充足性和泛化趋势 |
| `residuals = y_true - y_pred` | 派生量 | 衡量每个样本的预测误差——残差图的数据源 |
| `scoring='r2'` | 参数 | 学习曲线使用的评分指标——$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |

## 1. 残差图

### 参数速览

适用函数：`plot_residuals(y_true, y_pred, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `ndarray`，形状 `(40,)` | 测试集真实房价 | `y_test` |
| `y_pred` | `ndarray`，形状 `(40,)` | 模型预测房价 | `model.predict(X_test)` |
| `title` | `str` | 图标题 | `"线性回归 残差分析"` |
| `dataset_name` | `str` | 输出目录名 | `"linear_regression"` |
| `model_name` | `str` | 输出文件名前缀 | `"linear_regression"` |

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

![残差图](https://img.yumeko.site/file/articles/ML/linear_regression/residual_plot.png)

### 理解重点

- 左图（预测 vs 真实）：点越贴近对角线 $y = \hat{y}$，预测越准确。对于线性回归，在数据关系近线性的前提下，点应大致沿对角线分布。
- 右图（残差 vs 预测）：残差应围绕 0 随机分布。若残差呈现明显的曲线趋势（如 U 形），说明线性假设可能不足——数据中存在非线性关系需要捕捉。
- 当前合成数据的关系是精确线性的（仅叠加高斯噪声），残差图通常表现规整——残差随机散开，无系统模式。这也是为什么线性回归在该数据上是最合适的模型。
- 若残差随预测值增大而发散（漏斗形），说明存在异方差——不同预测值区间的误差方差不一致。

## 2. 学习曲线

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, scoring, cv, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `LinearRegression` | **未训练的**新模型实例——学习曲线内部会多次 fit | `LinearRegression()` |
| `X` | `ndarray`，形状 `(160, 3)` | 训练集特征 | `X_train` |
| `y` | `ndarray`，形状 `(160,)` | 训练集目标 | `y_train` |
| `scoring` | `str` | 评分指标。`"r2"`——R² 决定系数 | `"r2"`、`"neg_mean_squared_error"` |
| `cv` | `int` | 交叉验证折数。默认 `5` | `5` |
| `train_sizes` | `ndarray` | 训练样本比例序列。默认 10 个点从 0.1 到 1.0 | `np.linspace(0.1, 1.0, 10)` |

### 示例代码

```python
plot_learning_curve(
    LinearRegression(),
    X_train,
    y_train,
    scoring="r2",
    title="线性回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 输出

![学习曲线](https://img.yumeko.site/file/articles/ML/linear_regression/learning_curve.png)

### 理解重点

- 对于线性回归在当前数据上，学习曲线通常表现良好——因为 3 特征线性模型参数极少（4 个），160 训练样本远多于参数数，不容易过拟合。
- 训练和验证得分应较快收敛到相近的值——两条曲线之间不应有大的间隙。
- 如果样本量极少（如 < 20），验证曲线可能出现大幅波动——这是小样本下 CV 估计不稳定的典型表现。
- 学习曲线和残差图回答不同问题：残差图回答"这次预测误差分布如何"，学习曲线回答"样本量变化时性能如何变化"。

## 3. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 残差图 | 已实现 | 回归模型的核心诊断工具——可视化误差分布 |
| 学习曲线 | 已实现 | 诊断样本量充足性和泛化趋势 |
| 系数打印 | 已实现 | 训练日志中打印 `coef_` 和 `intercept_`——可对照真实公式 |
| MSE / MAE / RMSE / R² 数值打印 | **未实现** | 当前流水线侧重图形化诊断而非数值指标 |
| 交叉验证 R² 均值 | **未实现** | 学习曲线内部使用了 CV，但未单独输出 CV 均值 |
| 特征重要性图 | **不适用** | 线性回归的 `coef_` 直接表达了特征影响——不需要重要性图 |

### 理解重点

- 当前评估设计强调**图形化诊断 + 系数对照**——残差图看误差模式，系数对照验证 OLS 恢复精度。
- 线性回归独有的评估优势是可以直接对照 `coef_` 与真实公式——这比任何数值指标都更直观地验证模型正确性。
- 评估章节必须以源码为准——不能将常见的回归指标（MSE/MAE）写成当前已实现。

## 4. 线性回归 vs 决策树回归 评估对比

| 评估维度 | 线性回归 | 决策树回归 |
|---|---|---|
| 核心可视化 | 残差图 + 学习曲线 | **残差图 + 特征重要性 + 学习曲线 + 树结构** |
| 模型解释 | `coef_` + `intercept_`——直接对照真实公式 | **`feature_importances_`——分裂贡献排名** |
| 独有诊断 | **系数方向与大小的可验证性** | **树结构可视化——完整规则路径** |
| 过拟合风险 | 低——参数极少 | **高——需显式深度约束** |
| 定量指标 | 无显式打印 | 无显式打印 |
| 教学价值 | 残差 + 系数的联合解读 | 结构 + 重要性 + 残差的交叉诊断 |

## 常见坑

1. 只看残差图不看学习曲线——残差图只反映单次测试集表现，学习曲线揭示泛化趋势。
2. 在关系近线性的合成数据上期待看到复杂诊断信号——线性回归在此数据上表现良好是正常的，不需要"调优"。
3. 误以为当前流水线已输出 MSE/MAE/R² 数值——实际上仅打印了系数，数值指标需自行添加。
4. 忽略了线性回归独有的"系数可验证"评估优势——对照 `coef_` 与 `[2, 10, -3]` 是其他任何回归模型都没有的评估手段。

## 小结

- 当前线性回归有两类评估输出（残差图 + 学习曲线）+ 一项日志输出（系数打印）——三者构成完整的诊断体系。
- 残差图揭示"误差分布是否健康"，学习曲线揭示"泛化趋势是否稳定"，系数打印揭示"模型是否学到了正确的方向与大小"。
- 线性回归独有的评估优势：`coef_` 和 `intercept_` 可以直接与透明生成公式对照——这是所有回归分册中唯一能进行"有标准答案验证"的评估。

# 工程实现

## 本章目标

1. 理解线性回归流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链。
3. 理解线性回归与决策树回归在工程实现上的关键差异——最简训练函数、无标准化、无可视化差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadLinearRegressionDataset()` 手工合成线性房价数据 |
| `trainLinearRegressionModel(...)` | 函数 | 构建并训练 `LinearRegression`——本仓库最简训练函数（3 行） |
| `PipelineSpec` | 数据类 | 声明式流水线配置——关联数据集、训练器、预处理、可视化 |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行各阶段 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadLinearRegressionDataset()`——按显式线性公式生成 200 样本 | `DataFrame`，形状 `(200, 4)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.linear_regression", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainLinearRegressionModel(...)`——构建 `LinearRegression()` 并 fit | `LinearRegression` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.linear_regression", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载数据 → 预处理（无）→ 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、学习曲线 | PNG 图像文件 |

### 理解重点

- 线性回归是当前代码库中**工程结构最简**的回归流水线——训练函数仅 3 行，预处理为 `None`，评估可视化仅 2 项。
- 与决策树回归的差异：决策树多了 `featureImportance` 和 `treeStructure` 两项可视化，训练函数更复杂（3 个超参数 + 结构日志）。
- 这种极简设计是有意的——线性回归作为回归学习的起点，工程结构越简单越利于理解核心调用链。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.linear_regression",     # pipeline ID
    TaskType.REGRESSION,                # 任务类型
    "regression.linear_regression",     # dataset ID
    RunnerType.REGRESSION,              # 运行器类型
    trainLinearRegressionModel,         # 训练函数
    None,                               # 预处理 —— 无标准化
    "randomSplit",                      # 切分策略
    "default",                          # 后处理
    "regression",                       # 输出目录前缀
    "regression",                       # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    ["featureImportance"],              # 训练后诊断可视化
    ["learningCurve"],                  # 学习可视化
    "linear_regression",                # 结果存储子目录
)
```

### 理解重点

- `None` 预处理是线性回归与其他回归模型（SVR、正则化）的关键工程差异——但需区分：当前选择是因为数据量纲接近且关系简单，不代表所有场景下线性回归都不需要标准化。
- `["featureImportance"]` 在训练后可视化中——对线性回归来说，"特征重要性"即 `coef_` 的绝对值可视化（系数柱状图）。
- `["learningCurve"]` 使用 `_buildLearningCurveFactory` 工厂函数——传入新的 `LinearRegression()` 实例做 CV。

## 3. 数据依赖关系

```
loadLinearRegressionDataset()
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ X_train, y_train ──→ model.fit() ──→ model (coef_, intercept_)
    │       │       │
    │       │       └─→ plot_learning_curve(LinearRegression(), X_train, y_train)
    │       │
    │       └─→ X_test ──→ model.predict() ──→ y_pred
    │               │
    │               └─→ plot_residuals(y_test, y_pred)
    │
    └─→ model ──→ 终端日志: coef_ + intercept_ 打印
```

### 理解重点

- 数据依赖图比决策树回归更简单——没有 `featureImportance` 和 `treeStructure` 两条分支。
- `y_test` 仅用于评估对比——不参与训练，只在残差图中与 `y_pred` 对比。
- `model` 的核心输出是 `coef_` 和 `intercept_`——终端日志直接打印，是流水线最核心的训练产物。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadLinearRegressionDataset()` | 加载 `(200, 4)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分——无标准化 |
| 5 | 调用 `trainLinearRegressionModel(X_train, y_train)` | SVD 闭式求解——打印 `coef_` 和 `intercept_` |
| 6 | `model.predict(X_test)` | 获取测试集预测值 |
| 7 | `plot_residuals(y_test, y_pred)` |  残差诊断图 |
| 8 | `plot_learning_curve(LinearRegression(), X_train, y_train, scoring="r2")` | 学习曲线 |

### 理解重点

- 步骤 5 是本仓库最短的训练步骤——3 行代码，SVD 闭式解，瞬间完成，无迭代日志。
- 步骤 8 的学习曲线使用新 `LinearRegression()` 实例——`_buildLearningCurveFactory("regression.linear_regression")` 返回的工厂函数。
- 与决策树回归的执行链对比：少了 `plot_feature_importance` 和 `plot_tree_structure` 两个步骤。

## 5. 线性回归 vs 决策树回归 vs SVR 工程对比

| 工程维度 | 线性回归 | 决策树回归 | SVR |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainDecisionTreeRegressionModel` | `trainSvrRegressionModel` |
| 模型类 | `LinearRegression` | `DecisionTreeRegressor` | `SVR` |
| 训练函数行数 | **3 行** | ~5 行 | ~4 行 |
| 预处理 | `None` | `None` | `standardScaler` |
| 超参数数 | 0 | 3 | 4 |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | `[]` |
| 学习可视化 | `["learningCurve"]` | `["learningCurve", "treeStructure"]` | `["learningCurve"]` |
| 数据量 | 200（手工合成） | 20640（真实数据） | 200（合成非线性） |
| 训练方式 | SVD 闭式解 | CART 贪心递归 | 凸优化（SMO 类算法） |

## 常见坑

1. 误以为运行器层直接导入可视化函数——实际上是通过诊断/学习可视化列表配置，由运行器根据列表动态调用。
2. 将 `PipelineSpec` 中的旧路径引用（如 `data_generation/`）当成当前代码库的实际结构——实际代码在 `src/mlAlgorithms/` 下。
3. 把 `trainLinearRegressionModel` 的极简实现误解为功能缺失——3 行代码是因为 `LinearRegression()` 无需超参数，这是设计上的有意简洁。

## 小结

- 线性回归工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- 与决策树回归/SVR 的工程差异：（1）训练函数最简（3 行）；（2）预处理为 `None`（无标准化）；（3）可视化最少（无 treeStructure）；（4）数据为手工合成小规模。
- 这种极简设计使线性回归成为理解整个回归流水线架构的最佳入口——先看懂最简单的，再对比复杂的。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对线性回归核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 OLS 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 OLS、正规方程、MLE、系数解释、残差图等核心概念的理解 |
| 动手练习 | 实践 | 修改噪声、样本量、新增无关特征观察 OLS 行为——建立线性模型直觉 |
| 参考文献 | 入口 | 提供线性回归经典教材和 scikit-learn 官方文档 |

## 1. 自检问题

1. 线性回归的模型形式是什么？`coef_` 和 `intercept_` 分别对应数学公式中的什么？

2. OLS 的损失函数是什么？为什么选择平方误差而非绝对误差？从高斯噪声假设的 MLE 视角解释。

3. 正规方程 $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ 在什么条件下会失效？scikit-learn 实际使用什么方法来避免这个问题？

4. 为什么当前线性回归流水线没有标准化？在什么场景下线性回归需要标准化？

5. 当前手工合成数据的真实公式是什么？如果训练后 `房龄` 的系数变成了正数，可能是什么原因？

6. 残差图和学习曲线分别更适合诊断什么问题？为什么两者需要结合来看？

7. 线性回归和决策树回归在处理特征交互方面有什么区别？"多一个房间且面积大一倍"这种交互效应在线性回归中如何表达？

## 2. 动手练习

### 练习 1：改变噪声水平

修改 `RegressionDatasetFactory` 中数据生成的噪声标准差（当前为 10），分别设为 `1`、`10`、`50`。

```python
# 在 loadLinearRegressionDataset 中修改 noise 的标准差
noise = rng.normal(0, 1, size=200)   # 低噪声——试试 1, 10, 50
```

回答：噪声 $\sigma=1$ 时系数是否几乎精确等于真实值 `[2, 10, -3]`？噪声 $\sigma=50$ 时系数偏移多少？残差图在三种噪声水平下有何不同？

### 练习 2：改变样本量

将 `nSamples` 分别设为 `20`、`100`、`500`、`2000`。

```python
# 在 RegressionDatasetFactory 中修改
nSamples: int = 20  # 试试 20, 100, 500, 2000
```

回答：`nSamples=20` 时系数估计是否极不稳定？学习曲线的验证得分波动是否随样本量增大而减小？$N=2000$ 时系数是否几乎完美恢复真实值？

### 练习 3：对照真实公式验证训练结果

运行默认流水线后，将控制台输出的系数与真实公式对照：

```python
# 真实公式: price = 2*面积 + 10*房间数 - 3*房龄 + noise + 50
# 训练输出:
#   面积: X.XX  (真实: 2)
#   房间数: X.XX (真实: 10)
#   房龄: X.XX  (真实: -3)
#   截距: X.XX  (真实: 50)
```

回答：三项系数的正负号是否全部正确？数值偏差最大的特征是什么？截距偏差了多少？

### 练习 4：新增一个无关特征

在数据生成中增加一个完全随机的噪声列，观察 OLS 如何处理无关特征。

```python
noise_feature = rng.normal(0, 5, size=200)
# 添加到 DataFrame 中，重新训练
```

回答：无关特征的系数是否接近 0？加入无关特征后，原有三个特征的系数是否发生明显变化？学习曲线是否受影响？

### 练习 5：手动计算 R² 并与残差图对照

在流水线中手动计算并打印测试集 $R^2$：

```python
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集 R²: {r2:.4f}")
print(f"测试集 MSE: {mse:.4f}")
```

回答：$R^2$ 是否接近 1.0？MSE 是否与噪声方差 $\sigma^2=100$ 量级一致？数值指标与残差图的视觉判断是否一致？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 3. | 经典教材——线性回归的完整理论：OLS、子集选择、岭回归与 Lasso 的数学推导 |
| 2 | James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer. Chapter 3. | 入门教材——线性回归的基础直觉、R/Python 实现和与 KNN 的对比 |
| 3 | Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis*. Wiley. | 线性回归专著——从一元到多元、诊断、影响分析和共线性处理的全面覆盖 |
| 4 | scikit-learn 官方文档 — [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) | scikit-learn 的 API 参考——所有构造器参数、属性和方法的详细说明 |

## 常见坑

1. 在合成数据上期待系数精确等于真实值——噪声 $\sigma=10$ 和有限样本 200 意味着系数必然存在统计波动，这是 OLS 的固有属性而非 bug。
2. 只改噪声不改样本量——噪声和样本量对系数估计精度的联合影响才完整反映 OLS 行为。
3. 只看系数不看残差图——系数正负正确但残差图有系统偏差，可能意味着模型设定本身有问题。
4. 手动修改源码后忘记还原——建议在修改前用 `git stash` 保存原始状态，便于对比。

## 小结

- 7 个自检问题覆盖线性回归的核心概念：OLS、正规方程、MLE、标准化场景、系数验证、残差与学习曲线、特征交互。
- 5 个动手练习从不同角度探索 OLS 行为——改变噪声、改变样本量、对照真实公式、加入无关特征、计算数值指标。
- 4 篇参考文献覆盖两本经典教材（ESL、ISLR）、线性回归专著和 scikit-learn 官方文档——构成完整的线性回归学习路线。
