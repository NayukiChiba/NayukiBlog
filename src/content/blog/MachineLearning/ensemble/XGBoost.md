---
title: XGBoost
date: 2026-05-22
category: 机器学习/集成学习
tags:
  - Scikit-learn
  - 高级教程
description: XGBoost的数学原理、二阶泰勒展开与显式正则化工程实现。
image: https://img.yumeko.site/file/blog/cover/1780581910420.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 XGBoost 与 GBDT 共享的数学基础——加法模型、梯度提升、收缩步长。
2. 理解 XGBoost 独有的数学创新——二阶泰勒展开（Hessian）、显式正则化目标函数、分位数加权草图。
3. 理解 XGBoost 的回归目标（MSE）与分类目标（交叉熵）在数学形式上的差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 二阶泰勒展开 | 目标函数近似 | 使用 Hessian（二阶导数）比仅用梯度更精确地近似损失变化 |
| 正则化目标函数 | 模型正则化 | $\Omega(f) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|_1$——剪枝 + 权重收缩 |
| 分裂增益公式 | 分裂决策 | 精确计算每次分裂的损失下降——直接最大化增益 |
| 加权分位数草图 | 近似分裂点搜索 | 用二阶梯度加权的分位数确定候选分裂点——比等频分桶更高效 |
| 稀疏感知分裂 | 缺失值处理 | 自动学习缺失值的最优分裂方向 |
| 列块并行 | 计算加速 | 预排序后按列分块——在分裂搜索层面并行 |

## 1. 加法模型与目标函数

### 加法模型

与 GBDT 一致：

$$
\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(\mathbf{x}_i), \quad f_m \in \mathcal{F}
$$

其中 $\mathcal{F}$ 是回归树空间，$f_m$ 是第 $m$ 棵树。

### 正则化目标函数（XGBoost 独有）

XGBoost 的核心创新——在损失函数外显式加入正则项：

$$
\text{Obj}(\Theta) = \sum_{i=1}^{N} \ell(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(f_m)
$$

其中单棵树的正则项为：

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \|\mathbf{w}\|^2 + \alpha \|\mathbf{w}\|_1
$$

- $T$：叶子节点数，$\gamma$（`gamma`）控制分裂的"代价"——分裂增益必须超过 $\gamma$ 才执行
- $\mathbf{w}$：叶子权重的向量，$\lambda$（`reg_lambda=1.0`）做 L2 收缩，$\alpha$（`reg_alpha=0.0`）做 L1 稀疏

### 理解重点

- GBDT（sklearn）的"正则化"主要是学习率收缩——XGBoost 在此基础上加入显式的 L1/L2 惩罚项。
- `gamma=0.0` 表示当前源码不要求分裂有最低增益——调大 `gamma` 是防止过拟合的有效手段。
- `reg_lambda=1.0`（L2 默认开启）是 XGBoost 泛化性能好的重要原因——它对叶子权重做持续的收缩约束。

## 2. 二阶泰勒展开（XGBoost 独有）

在第 $m$ 轮，XGBoost 对损失函数做二阶泰勒展开：

$$
\text{Obj}^{(m)} \approx \sum_{i=1}^{N} \left[ \ell(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)
$$

其中：

$$
g_i = \frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}, \quad
h_i = \frac{\partial^2 \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i^2} \bigg|_{\hat{y}=\hat{y}^{(m-1)}}
$$

### 回归（MSE）下的 $g_i$ 和 $h_i$

对于当前回归任务，$\ell = \frac{1}{2}(y_i - \hat{y}_i)^2$：

$$
g_i = \hat{y}_i - y_i, \quad h_i = 1
$$

### 理解重点

- 二阶泰勒展开是 XGBoost 最核心的数学创新——Hessian $h_i$ 提供了损失函数曲率信息，使目标函数近似比 GBDT 的一阶近似更精确。
- 在 MSE 回归下，$h_i = 1$（常数），二阶信息退化——但 XGBoost 的框架对任意可微损失函数都适用。
- 对于分类（对数损失），$h_i = p_i(1-p_i)$——此时二阶信息提供了预测不确定性的加权。

## 3. 叶子权重的闭式解

将目标函数按叶子重组，对第 $j$ 个叶子：

$$
\text{Obj}_{\text{leaf}}^{(m)} = \sum_{i \in I_j} \left( g_i w_j + \frac{1}{2} h_i w_j^2 \right) + \frac{1}{2} \lambda w_j^2 + \alpha |w_j|
$$

L1 正则化 $\alpha=0.0$ 时（当前源码），对 $w_j$ 求导为零得最优权重：

$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

代入得最优叶子对应的目标函数值：

$$
\text{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

### 理解重点

- 叶子权重的闭式解存在，是因为 XGBoost 的二次近似目标函数——GBDT（sklearn）没有这样的闭式解。
- `reg_lambda=1.0` 在分母中——它抑制大权重，防止单片叶子主导预测。
- 在 MSE 回归中，$w_j^* = -\frac{\sum g_i}{\vert I_j\vert + \lambda}$——即该叶子内残差均值的 L2 压缩版。

## 4. 分裂增益公式

给定一个叶子节点，将其分裂为左右子节点 $L$ 和 $R$，分裂增益为：

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

其中 $G = \sum_{i \in I} g_i$，$H = \sum_{i \in I} h_i$。

当 $\text{Gain} > 0$ 时执行分裂；`gamma` 增大要求更高的最小增益——做预剪枝。

### 理解重点

- 分裂增益公式使 XGBoost 能**精确评估**每次候选分裂的效果——最大化增益等价于最小化目标函数。
- $\gamma=0.0$（当前源码）意味着只要增益为正就分裂——这是最小限制。
- 这个公式也是特征重要性的计算基础——特征在所有分裂中的增益累加。

## 5. 加权分位数草图

XGBoost 寻找候选分裂点时，不用简单的等频分桶（直方图），而是用二阶梯度加权分位数：

按 $h_i$ 加权排序后取分位数——样本的 Hessian 越大，在分位数计算中的权重越大。

### 理解重点

- $h_i$ 反映了样本对损失函数的"重要性"——Hessian 大的样本，损失在该点变化剧烈，分裂点应该更精确地考虑它们。
- 在 MSE 回归中 $h_i=1$，加权分位数退化为等频分位数——此时近似分裂点搜索与直方图分桶等价。
- 在分类场景下，$h_i = p_i(1-p_i)$——接近决策边界（$p \approx 0.5$）的样本有更大权重。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $\hat{y}_i^{(M)} = \sum_{m=1}^{M} f_m(\mathbf{x}_i)$ | `XGBRegressor(n_estimators=300, learning_rate=0.05)` |
| 正则化目标 | $\text{Obj} = \sum\ell + \sum\Omega(f)$ | `reg_lambda=1.0, reg_alpha=0.0, gamma=0.0` |
| 梯度（MSE） | $g_i = \hat{y}_i - y_i$ | 内部自动计算 |
| Hessian（MSE） | $h_i = 1$ | 内部自动计算 |
| 叶子权重闭式解 | $w_j^* = -\frac{G_j}{H_j + \lambda}$ | 内部自动计算 |
| 分裂增益 | $\text{Gain} = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$ | 内部自动计算 |
| 行采样 | 随机子集 | `subsample=0.9` |
| 列采样 | 随机特征子集 | `colsample_bytree=0.9` |
| 最小叶子权重和 | $\sum_{i \in I_j} h_i \ge$ `min_child_weight` | `min_child_weight=1` |
| 学习率收缩 | $\eta \cdot f_m$ | `learning_rate=0.05` |
| 列块并行 | 预排序后按列分块 | `n_jobs=-1` |

## 7. XGBoost vs GBDT vs LightGBM 数学对比

| 维度 | GBDT (sklearn) | LightGBM | XGBoost |
|---|---|---|---|
| 目标函数近似 | 一阶（仅梯度） | 一阶（仅梯度） | **二阶（梯度 + Hessian）** |
| 正则化 | 学习率收缩 | 学习率收缩 | **学习率 + L1 + L2 + gamma 剪枝** |
| 叶子权重 | 逐点线搜索 | 逐点线搜索 | **闭式解**（二次近似） |
| 分裂点搜索 | 预排序 $\rightarrow$ 逐一计算 | 直方图分桶 | **加权分位数草图** |
| 缺失值 | 不支持 | 不支持 | **稀疏感知——自动学习最优方向** |
| 并行 | 无 | 直方图构建级 | **列块级** |
| 树生长 | Level-wise | Leaf-wise | Level-wise（近似） |

## 常见坑

1. 在 MSE 回归场景下，$h_i=1$ 是常数——XGBoost 的二阶展开近似退化为与牛顿法而非梯度下降对应的形式，理解这一点很重要。
2. 忽略 `reg_lambda=1.0` 的默认值——XGBoost 默认 L2 正则化已开启，与 GBDT/LightGBM 的默认行为不同。
3. 混淆 `min_child_weight=1` 与 `min_samples_leaf`——前者是 Hessian 和的最小值（MSE 下等价于叶子最小样本数），非样本计数。
4. 认为 `gamma` 和 `reg_lambda` 功能重叠——`gamma` 做分裂级剪枝（分裂是否值得），`reg_lambda` 做权重级收缩（叶子值是否过大）。

## 小结

- XGBoost 的数学核心链：加法模型 $\rightarrow$ 二阶泰勒展开（$g_i + \frac{1}{2}h_i f^2$）$\rightarrow$ 正则化目标（$+\gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|_1$）$\rightarrow$ 叶子权重闭式解 $w_j^* = -\frac{G_j}{H_j+\lambda}$ $\rightarrow$ 分裂增益公式 $\rightarrow$ 精确剪枝。
- 与 GBDT/LightGBM 的最关键区别：二阶展开 + 显式正则化项——前者提供更精确的目标近似，后者提供更强的过拟合控制。
- 当前源码 `XGBRegressor(n_estimators=300, max_depth=6, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)` 是回归任务的经典配置——L2 默认开启、无 L1 稀疏、无最低分裂增益。

# 数据构成

## 本章目标

1. 明确本仓库 XGBoost 数据来自 `EnsembleData.xgboost()` 返回的加州房价真实数据集。
2. 理解为什么选择真实回归数据——20640 条记录充分展示 XGBoost 的工程实力和正则化优势。
3. 明确当前流程中的训练/测试切分——注意无标准化（树模型天然尺度不敏感），无分层抽样（回归无类别）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.xgboost()` | 静态方法 | 返回加州房价真实数据集 |
| `fetch_california_housing(...)` | 函数 | scikit-learn 提供的真实世界加州房价数据集加载器 |
| `xgboost_data` | 变量 | 在 `data_generation/__init__.py` 中导出的全局 DataFrame（20640 $\times$ 9） |
| `price` | 目标列 | 连续值回归目标——加州地区房屋中位价（单位：10 万美元） |
| `train_test_split` | 函数 | 训练/测试切分——无 `stratify` 参数（回归任务） |

## 1. 数据生成：`EnsembleData.xgboost()`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `as_frame` | `bool` | `True`——返回 Pandas DataFrame，含特征名和目标列 | `True`、`False` |
| 返回值 | `Bunch` | sklearn Bunch 对象——`.frame` 属性为 DataFrame，目标列名 `MedHouseVal` | — |

### 特征列表

| 特征名 | 全称 | 类型 | 说明 |
|---|---|---|---|
| `MedInc` | Median Income | 连续 | 街区收入中位数（万美元） |
| `HouseAge` | House Age | 连续 | 房屋年龄中位数（年） |
| `AveRooms` | Average Rooms | 连续 | 每户平均房间数 |
| `AveBedrms` | Average Bedrooms | 连续 | 每户平均卧室数 |
| `Population` | Population | 连续 | 街区人口 |
| `AveOccup` | Average Occupancy | 连续 | 每户平均居住人数 |
| `Latitude` | Latitude | 连续 | 纬度 |
| `Longitude` | Longitude | 连续 | 经度 |

### 目标列

| 目标名 | 全称 | 类型 | 取值范围 |
|---|---|---|---|
| `price` | Median House Value | 连续 | 约 $[0.15, 5.0]$（单位：10 万美元） |

### 示例代码

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame.rename(columns={"MedHouseVal": "price"})
# df.shape = (20640, 9)  # 8 特征 + 1 目标 price
```

### 理解重点

- 这是本仓库集成学习分册中唯一的**真实数据集**——非合成生成，含 20640 条记录，8 个特征各有现实含义。
- 目标 `price` 是**连续值**——这是回归任务，不是分类。与 Bagging/GBDT/LightGBM 的离散标签形成根本区别。
- `n_samples` 参数对此方法无效——真实数据集的行数固定为 20640。
- 与所有其他集成模型的数据设计意图不同：这里不追求"展示方差缩减"或"偏差缩减"，而是展示 XGBoost 在真实工业表格数据上的综合表现。

## 2. 特征列与目标列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(20640, 8)` | 含 8 个连续特征的特征矩阵 | `data.drop(columns=["price"])` |
| `y` | `Series`，形状 `(20640,)` | 连续回归目标——房屋中位价 | `data["price"]` |

### 理解重点

- `price` 是回归监督目标——参与 `model.fit()` 和残差分析评估。
- 特征全为连续值——无类别变量，无需独热编码。
- 与 Bagging/GBDT/LightGBM 的标签列不同：这里没有 `stratify`，没有 `predict_proba`，没有混淆矩阵。

## 3. 训练/测试切分

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 `(20640, 8)` | 全量特征矩阵 | `X` |
| `y` | `Series`，形状 `(20640,)` | 全量目标 | `y` |
| `test_size` | `float` | 测试集比例。`0.2`——4128 个测试样本 | `0.2`、`0.3` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `(DataFrame, DataFrame, Series, Series)` | `X_train`（16512 样本）、`X_test`（4128 样本）及对应目标 | — |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 理解重点

- 当前流水线**有**训练/测试切分——与所有集成模型一致。
- **无** `stratify` 参数——回归任务没有类别概念，目标值连续分布。
- **无** `StandardScaler`——树模型基于分裂点比较（$x_j < \text{threshold}$），对特征的线性缩放不敏感。这是与 Bagging/GBDT/LightGBM 流水线的关键差异。

## 4. 数据设计意图：与其他集成模型的对比

| 数据维度 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 数据类型 | 合成 | 合成 | 合成 | **真实** |
| 任务 | 二分类 | 三分类 | 四分类 | **回归** |
| 样本数 | 500 | 500 | 1000 | **20640** |
| 特征维度 | 2 | 8 | 20 | **8** |
| 标签类型 | $\{0,1\}$ | $\{0,1,2\}$ | $\{0,1,2,3\}$ | **连续 $\mathbb{R}$** |
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |

### 理解重点

- XGBoost 是四个集成模型中唯一使用真实数据、唯一做回归任务的——这使得它在本仓库集成学习分册中具有独特地位。
- 20640 个样本远超其他集成模型——XGBoost 的列块并行和加权分位数草图在此规模上开始发挥优势。
- 无标准化的设计意味着流水线少了一个步骤——体现了树模型"免预处理"的工程便利。

## 数据可视化

![特征相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

## 常见坑

1. 在回归数据上使用 `stratify`——只有分类任务才有分层抽样的概念，回归目标连续分布。
2. 对树模型做 `StandardScaler`——非必需操作，不会提升模型性能（树分裂只依赖相对顺序）。
3. 修改 `EnsembleData.n_samples` 期望影响数据量——`xgboost()` 使用真实数据集，`n_samples` 对其无效。
4. 混淆目标列名——原始名为 `MedHouseVal`，在 `EnsembleData.xgboost()` 中被重命名为 `price`。

## 小结

- 当前 XGBoost 数据来自 `fetch_california_housing(as_frame=True)`：加州真实房价数据，20640 样本 x 8 特征，目标为连续房价。
- 数据流为：`fetch_california_housing` -> 重命名目标列 -> 训练/测试切分（无标准化、无分层）。
- 真实数据 + 回归任务的设计意图是展示 XGBoost 在工业表格回归场景下的工程成熟度——正则化、大规模数据、缺失值稀疏感知。

# 思路与直觉

## 本章目标

1. 用直观方式理解 XGBoost 相对于 GBDT 的核心创新——二阶信息、显式正则化、稀疏感知。
2. 理解为什么 XGBoost 在 Kaggle 等比赛中统治表格数据——从精确目标近似到强正则化的全链路优化。
3. 通过与 GBDT 和 LightGBM 的对比，建立 XGBoost 在 Boosting 谱系中的定位——最"数学精确"的 Boosting 实现。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 二阶泰勒展开 | 目标近似 | 不仅知道方向（梯度），还知道曲率（Hessian）——迈出的每一步更"精准" |
| 显式正则化 | 防过拟合 | L1 剪枝无用权重 + L2 收缩所有权重 + gamma 阻止不值得的分裂 |
| 加权分位数草图 | 分裂点搜索 | 按"样本对损失的重要程度"加权分桶——不简单等分 |
| 稀疏感知 | 缺失值处理 | 自动学习缺失值该走左子树还是右子树 |
| 列块并行 | 加速 | 预排序后按特征（列）分块——各特征的分裂搜索可并行 |

## 1. 为什么需要 XGBoost

GBDT 是一阶方法——每棵树只看梯度（斜率），不知道损失函数弯曲的程度。XGBoost 多看一眼 Hessian（曲率），迈的步子更准。

> GBDT：知道下坡的方向（梯度），迈一步试
> XGBoost：知道下坡的方向（梯度）**和**坡度变化率（Hessian），一次迈到最优点附近

### 理解重点

- 二阶信息在"损失函数不是均匀弯曲"时最有用——分类的对数损失曲率随概率变化（$h_i = p(1-p)$），XGBoost 能自适应。
- 在 MSE 回归中 $h_i=1$，二阶信息退化——但 XGBoost 仍通过正则化项（$\lambda$、$\gamma$）提供优势。
- XGBoost 不创造新的 Boosting 范式——它让 Boosting 的每一步更科学、更精确。

## 2. 用"考试加分题"理解二阶泰勒展开

一阶方法（GBDT）像：
> "你上次错了 10 分，这次再补 10 分的课"

二阶方法（XGBoost）像：
> "你上次错了 10 分，而且错误集中在几何题（高曲率），这次专门补几何 8 分、代数 2 分"

### 理解重点

- Hessian $h_i$ 区分了"错了但好纠正"和"错了但难纠正"的样本——在分类中，决策边界附近的样本 $h_i$ 大（$p \approx 0.5$），XGBoost 更关注它们。
- 这使得 XGBoost 的分裂点更"聪明"——不是在特征空间里均匀搜索，而是在"损失变化最剧烈"的区域密集搜索。

## 3. 用"交通规则"理解显式正则化

GBDT 只有一个正则化手段：学习率收缩（开慢点）。XGBoost 有三种：

- **`gamma`（最小分裂增益）**：绿灯信号——分裂增益必须超过 $\gamma$ 才允许通过
- **`reg_lambda`（L2）**：限速牌——叶子权重太大就压一压
- **`reg_alpha`（L1）**：单行道——不重要的叶子权重直接压到 0

### 理解重点

- `reg_lambda=1.0` 默认开启（不像 GBDT/LightGBM 默认关闭正则化）——XGBoost 天生"谨慎"。
- 三种正则化作用在不同层级：gamma 管分裂是否值得，lambda 管叶子值是否过大，alpha 管无关叶子是否置零。
- 这就像从"只有油门（学习率）"升级到"油门 + 刹车（lambda）+ 红绿灯（gamma）+ 路障（alpha）"。

## 4. 用"重要客户的专属服务"理解加权分位数

普通分桶方法把数据均分到 256 个桶——所有客户一视同仁。

XGBoost 的加权分位数按 Hessian 加权——大客户（Hessian 大的样本）权重更高，在他们密集的区域分桶更细。

### 理解重点

- 在 MSE 回归中 $h_i=1$，所有样本等权——加权分位数退化，与 LightGBM 的直方图效果相近。
- 但在分类中，$h_i = p_i(1-p_i)$——决策边界附近的样本有更大的 Hessian，XGBoost 在这些区域放更多桶。
- 这就是为什么 XGBoost 在某些分类任务上比 LightGBM 精度更高——分裂点搜索更"关注重点区域"。

## 5. 用"缺考处理"理解稀疏感知

一个学生缺考了数学——应该给他 0 分还是用其他科目估计？

XGBoost 的稀疏感知：自动学习——在这个特征上，缺失值应该走左子树还是右子树，哪个收益大就走哪边。

### 理解重点

- 这对于工业数据极其重要——真实数据总有缺失值，XGBoost 不需要预填充。
- 当前加州房价数据完整无缺失——但 XGBoost 的稀疏处理能力内建于算法。

## 6. 与 GBDT 和 LightGBM 的直觉对比

| 维度 | GBDT | LightGBM | XGBoost |
|---|---|---|---|
| 核心直觉 | 接力纠错——后面补前面 | 更快接力——直方图 + 重点培养 | **更准接力——二阶信息 + 三重重正则化** |
| 步子精度 | 一阶（只看梯度方向） | 一阶（只看梯度方向） | **二阶（看梯度 + 曲率）** |
| 防过拟合 | 学习率控制步长 | 学习率 + num_leaves 限制 | **学习率 + lambda L2 + alpha L1 + gamma 剪枝** |
| 分裂点搜索 | 逐个值排序比较 | 等频直方图分桶 | **Hessian 加权分位数分桶** |
| 缺失值 | 需预填充 | 需预填充 | **自动学习最优方向** |
| 训练速度 | 慢 | **最快** | 中等 |
| 精度 | 基准 | 略优（调参后） | **略优（调参后）** |

### 理解重点

- XGBoost 定位是"最严谨的 Boosting"——每个细节都有数学依据（二阶展开、闭式解、正则化理论）。
- LightGBM 定位是"最快的 Boosting"——牺牲一些数学精确性换取工程速度。
- 三者在实际比赛中的精度差异很小——通常 XGBoost ~= LightGBM > sklearn GBDT，差距主要来自调参。

## 可视化

![残差分析图](https://img.yumeko.site/file/blog/articles/1780736282426.png)

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

## 常见坑

1. 以为 XGBoost 一定比 LightGBM "好"——在超大样本上 LightGBM 更快，在小样本上 XGBoost 的二阶优势被数据不足稀释。
2. 在 MSE 回归中期待二阶展开带来巨大提升——MSE 的 Hessian 是常数，二阶优势在此场景不显著。
3. 忽略 `reg_lambda=1.0` 的默认值——与 GBDT/LightGBM 不同，XGBoost 的 L2 正则化默认开启。
4. 把 XGBoost 当成"不需要调参"——gamma、min_child_weight 等参数对精度的敏感度不低。

## 小结

- XGBoost 的直觉核心是"用更精确的数学做 Boosting"：二阶泰勒展开（看曲率）+ 显式正则化（三重防过拟合）+ 加权分位数（重点区域精分）+ 稀疏感知（自动处理缺失）。
- 加州房价真实回归数据是展示 XGBoost 工业级鲁棒性的最佳场景——正则化在真实数据上比在干净合成数据上更有价值。
- XGBoost 与 LightGBM 不是"谁更强"，而是精度（XGBoost）vs 速度（LightGBM）的权衡——两者在 Kaggle 上常打成平手。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `XGBRegressor`——注意这是回归模型，非分类。
2. 理解 `XGBRegressor` 的核心构造器参数（`n_estimators`、`max_depth`、`gamma`、`reg_lambda`、`min_child_weight`）及其与 GBDT/LightGBM 的差异。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`n_estimators_`（实际树数）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `xgboost.XGBRegressor` 回归模型——含可选依赖检查 |
| `XGBRegressor(...)` | 类 | XGBoost 的 scikit-learn 兼容回归接口——二阶泰勒展开 + 显式正则化 |
| `model.fit(X_train, y_train)` | 方法 | 训练 300 棵回归树——二阶目标近似 + 加权分位数草图 + 列块并行 |
| `model.feature_importances_` | 属性 | 8 个特征的重要性分数——基于分裂增益累加 |
| `model.predict(X)` | 方法 | 300 棵树加权累加——输出连续房价预测值 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(16512, 8)` | 训练特征矩阵（**无标准化**——树模型天然尺度不敏感） | `X_train` |
| `y_train` | `array_like`，形状 `(16512,)` | 连续回归目标——房屋中位价 | `y_train` |
| `n_estimators` | `int` | 弱学习器数量。当前 `300`——与 LightGBM 一致 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——每次只修正残差的 5% | `0.01`、`0.05`、`0.1` |
| `max_depth` | `int` | 树的最大深度。`6`——深于 GBDT（3），浅于完全生长 | `3`、`6`、`10` |
| `min_child_weight` | `int` | 叶子节点的最小 Hessian 和。`1`——MSE 下等价于最小样本数 | `1`、`5`、`10` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 训练样本 | `0.5`、`0.9`、`1.0` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征（~=7/8） | `0.3`、`0.9`、`1.0` |
| `gamma` | `float` | 分裂所需的最小损失下降。`0.0`——不设最低增益门槛 | `0.0`、`0.1`、`1.0` |
| `reg_alpha` | `float` | L1 正则化系数。`0.0`——不启用 L1 稀疏 | `0.0`、`0.1`、`1.0` |
| `reg_lambda` | `float` | L2 正则化系数。`1.0`——**默认开启**，抑制叶子权重过大 | `0.0`、`1.0`、`10.0` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `XGBRegressor` | 已完成 `fit()` 的回归模型对象 | — |

### 示例代码

```python
from model_training.ensemble.xgboost import train_model

model = train_model(X_train, y_train)
```

### 理解重点

- `train_model(...)` 是有监督回归训练——`y_train` 是连续值房价，不是离散类别标签。
- XGBoost 的 `max_depth=6` 深于 GBDT（3）但远浅于 Bagging 的完全生长树——在偏差和方差间取平衡。
- `reg_lambda=1.0` 是 XGBoost 独有的默认值——其他 Boosting 实现默认不开启 L2 正则化。
- `min_child_weight=1` 在回归中等于"每个叶子至少 1 个样本"——因为 Hessian 恒为 1。实际上相当于 `min_samples_leaf=1`。

## 2. `XGBRegressor` 构造器参数

### 参数速览

适用 API：`XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.9, colsample_bytree=0.9, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, random_state=42, n_jobs=-1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 弱学习器数量。`300`——步数更多但每步更小 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率。`0.05`——越小越需更多树 | `0.01`、`0.05`、`0.1` |
| `max_depth` | `int` | 树的最大深度。`6`——适中深度，防止过拟合 | `3`、`6`、`10` |
| `min_child_weight` | `int` | 叶子节点的最小 Hessian 和。`1` | `1`、`5`、`10` |
| `subsample` | `float` | 行采样比例。`0.9` | `0.5`、`0.8`、`0.9` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——8 个特征中约 7 个用于每棵树 | `0.3`、`0.8`、`0.9` |
| `gamma` | `float` | 分裂最小增益。`0.0`——不设门槛 | `0.0`、`0.1`、`1.0` |
| `reg_alpha` | `float` | L1 正则化。`0.0`——不启用 L1 稀疏 | `0.0`、`0.1` |
| `reg_lambda` | `float` | L2 正则化。`1.0`——**默认开启**，抑制大权重 | `0.0`、`1.0` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| `n_jobs` | `int` | 并行线程数。`-1` 使用所有 CPU——列块并行 | `-1`、`1`、`4` |
| `verbosity` | `int` | 日志级别。默认 `1`（warning） | `0`、`1`、`2` |

### 示例代码

```python
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("请先 pip install xgboost")

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
```

### 理解重点

- XGBoost 的参数列表是四个集成模型中最长的——体现了它在正则化和精确控制上的设计理念。
- `gamma` 是 XGBoost 独有的预剪枝参数——区别于 `max_depth`（硬深度限制）和 `min_child_weight`（叶子样本数限制）。
- 三重正则化（gamma + reg_lambda + reg_alpha）作用于不同层级——gamma 控分裂是否发生，lambda 控叶子权重是否过大，alpha 控无关权重是否置零。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 说明 |
|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(8,)` | 8 个特征的重要性分数——基于分裂增益累加（`gain`） |
| `n_estimators_` | `int` | 实际训练的树数量——等于 `n_estimators=300` |
| `n_features_in_` | `int` | 特征维度——当前为 `8` |
| `best_iteration_` | `int` | 早停最优迭代轮次（启用 `early_stopping_rounds` 时可用） |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"max_depth: {max_depth}")
print(f"min_child_weight: {min_child_weight}")
print(f"subsample: {subsample}")
print(f"colsample_bytree: {colsample_bytree}")
print(f"gamma: {gamma}")
print(f"reg_alpha: {reg_alpha}")
print(f"reg_lambda: {reg_lambda}")
print(f"特征重要性: {model.feature_importances_}")
```

### 理解重点

- `feature_importances_` 默认使用 `gain`（分裂增益累加）——与 LightGBM 一致，不同于 sklearn GBDT 的 impurity 下降量。
- 在加州房价数据上，`MedInc`（收入中位数）通常是最重要的特征——收入是房价的主要驱动力，符合直觉。
- XGBoost 没有 `predict_proba`——回归输出为连续值，不是概率分布。

## 4. `predict()` — 预测连续值

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n,)`，连续值 | 300 棵树加权累加——直接输出房价预测值 |

### 理解重点

- `predict()` 返回连续实数——即房屋中位价的预测值（单位：10 万美元）。
- 与分类集成模型不同——没有 `predict_proba`，没有 softmax，没有 argmax。
- 预测值 = $\sum_{m=1}^{300} \eta \cdot f_m(\mathbf{x})$——300 棵树的加权累加。

## 5. XGBoost vs GBDT vs LightGBM 参数对比

| 参数 | GBDT (sklearn) | LightGBM | XGBoost |
|---|---|---|---|
| 任务 | 分类 | 分类 | **回归** |
| `n_estimators` | 200 | 300 | 300 |
| `learning_rate` | 0.1 | 0.05 | 0.05 |
| 复杂度控制 | `max_depth=3` | `num_leaves=31` | **`max_depth=6`** |
| 最小叶子 | — | `min_child_samples=20` | **`min_child_weight=1`** |
| 行采样 | `subsample=1.0` | `subsample=0.9` | `subsample=0.9` |
| 列采样 | 无 | `colsample_bytree=0.9` | `colsample_bytree=0.9` |
| 分裂门槛 | — | — | **`gamma=0.0`** |
| L1 正则化 | — | — | **`reg_alpha=0.0`** |
| L2 正则化 | — | — | **`reg_lambda=1.0`** |
| 依赖 | sklearn 内置 | `pip install lightgbm` | `pip install xgboost` |

## 常见坑

1. 把 `min_child_weight=1` 理解成"最小样本数为 1"——对非 MSE 损失函数，Hessian 不是常数，两者不等价。
2. 忘记 `reg_lambda=1.0` 默认开启——如果感觉模型欠拟合，尝试降为 0.0。
3. 把 `gamma` 和 `reg_alpha` 功能混淆——gamma 做分裂级剪枝，alpha 做权重级稀疏化。
4. 在新环境中直接 `from model_training.ensemble.xgboost import train_model`——需先 `pip install xgboost`。

## 小结

- `train_model(...)` 是本仓库 XGBoost 的核心训练入口，是对 `xgboost.XGBRegressor` 的薄封装——含可选依赖检查和 12 个可配置参数。
- `XGBRegressor` 的核心参数体系是四个集成模型中最丰富的——`n_estimators`（树数量）、`learning_rate`（学习率）、`max_depth`（深度）、`min_child_weight`（最小 Hessian 和）、`gamma`（分裂门槛）、`reg_lambda`（L2）、`reg_alpha`（L1）——构成三层正则化体系。
- 训练完成后核心属性：`feature_importances_`（8 个特征按增益排序）——是回归场景下理解特征贡献的关键诊断工具。

# 训练与预测

## 本章目标

1. 理解 `pipelines/ensemble/xgboost.py` 的 `run()` 流水线——回归任务下的端到端流程（无标准化、无分层抽样）。
2. 理解 XGBoost 的 `fit()` 训练过程——二阶目标近似 + 显式正则化 + 加权分位数草图。
3. 理解回归预测的输出——连续房价预测值与残差分析。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 回归流水线编排——6 步串联数据拆分、训练、预测和两项评估 |
| `model.fit(X_train, y_train)` | 方法 | 训练 300 棵二阶近似正则化回归树——列块并行 + 加权分位数草图 |
| `model.predict(X_test)` | 方法 | 300 棵树加权累加——输出连续房价预测值 |
| `plot_residuals(y_test, y_pred, ...)` | 函数 | 绘制预测残差散点图和分布图——回归专用评估 |
| `plot_feature_importance(model, feature_names, ...)` | 函数 | 绘制特征重要性柱状图 |

## 1. 完整流水线流程

### 流程概述

```
xgboost_data.copy()
  - 1 X = data.drop(columns=["price"]), y = data["price"]
  - 2 feature_names = list(X.columns)
  - 3 X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
  - 4 model = train_model(X_train, y_train)  # 无标准化，含 ImportError 检查
  - 5 y_pred = model.predict(X_test)
  - 6 两项评估可视化
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `xgboost_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(20640, 9)` | 避免修改全局变量 |
| 分离 X/y | `data.drop(columns=["price"])` + `data["price"]` | `DataFrame` | `(DataFrame, Series)` | 特征 8 列 + 连续目标 1 列 |
| 提取特征名 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 供特征重要性图表使用 |
| 切分数据 | `train_test_split(test_size=0.2)` | `(X, y)` | `(X_train, X_test, y_train, y_test)` | 16512 训练 / 4128 测试 |
| 训练 | `train_model(X_train, y_train)` | `(DataFrame, Series)` | `XGBRegressor` | 300 棵二阶正则化树 |
| 预测 | `model.predict(X_test)` | `DataFrame`，`(4128, 8)` | `ndarray`，`(4128,)` | 连续房价预测值 |
| 残差图 | `plot_residuals(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 残差散点图 + 分布图 |
| 特征重要性 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 8 个特征排序柱状图 |

### 理解重点

- 这是四个集成模型中最简洁的流水线——6 步（vs Bagging 7 步、GBDT 9 步、LightGBM 7 步），少了标准化步骤。
- 与分类集成流水线的关键差异：无 `StandardScaler`、无 `stratify`、无 `predict_proba`、无混淆矩阵、无 ROC。
- 目标列名为 `price`（不是 `label`）——这是回归任务与分类任务在命名上的明确区分。

## 2. 训练细节：`model.fit(X_train, y_train)`

### 训练过程（300 棵树串行，含列块并行）

1. **第 1 棵树**：在原始房价标签上训练——初始预测为训练集均值
2. **第 $m$ 棵树**（$m = 2, \dots, 300$）：计算一阶梯度 $g_i$ 和二阶 Hessian $h_i$（回归下 $h_i=1$），对目标函数做二阶泰勒展开
3. **分裂点搜索**：对 8 个特征分别用加权分位数草图找候选分裂点，计算分裂增益 $\text{Gain} = \frac{1}{2}[\dots] - \gamma$，选最大增益分裂
4. **列采样**：`colsample_bytree=0.9`——每棵树随机选约 7 个特征
5. **行采样**：`subsample=0.9`——每轮随机保留 90% 样本
6. **正则化约束**：$w_j^* = -\frac{G_j}{H_j + \lambda}$（叶子权重 L2 压缩）+ $\gamma$ 门槛检查
7. **学习率收缩**：每棵树的输出乘以 `learning_rate=0.05`

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_estimators` | `300` | 串行训练的弱学习器数量 |
| `learning_rate` | `0.05` | 每棵树输出的收缩乘数 |
| `max_depth` | `6` | 每棵树的最大深度——可以分裂 6 次（最多 64 个叶子） |
| `min_child_weight` | `1` | 叶子节点的最小 Hessian 和——回归下等价于最小样本数 1 |
| `subsample` | `0.9` | 行采样比例——每轮随机保留 90% 训练样本 |
| `colsample_bytree` | `0.9` | 列采样比例——每棵树随机选 90% 特征（~=7/8） |
| `gamma` | `0.0` | 分裂最低增益——当前不设门槛 |
| `reg_lambda` | `1.0` | L2 正则化——压缩叶子权重 |
| `reg_alpha` | `0.0` | L1 正则化——当前不启用 |
| `n_jobs` | `-1` | 列块并行——各特征分裂点搜索可并行 |

### 理解重点

- XGBoost 的训练**在概念上**仍是串行 Boosting——但每棵树内部的列块分裂搜索是并行的（`n_jobs=-1`）。
- `reg_lambda=1.0` 使得每片叶子的权重被压缩——$w_j^* = -\frac{G_j}{H_j + 1}$，分母恒加 1 防止权重过大。
- 在回归任务中 $h_i=1$，Hessian 恒为常数——二阶展开的信息增量为零，但闭式解和正则化仍有效。

## 3. 预测细节

### `model.predict(X_test)` — 输出连续值

```
300 棵树加权累加（每棵 x learning_rate）
    -> 连续实数（房价预测值，单位：10 万美元）
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 8)` | `(n,)` | 连续房价预测值——$\in \mathbb{R}$ |

### 理解重点

- 与分类模型的根本不同：`predict()` 返回连续实数，不是类别标号。
- 没有 `predict_proba()`——回归模型只输出一个标量预测值。
- 预测值 = 训练集初始均值 + $\sum_{m=1}^{300} 0.05 \times f_m(\mathbf{x})$。

## 4. 与 Bagging/GBDT/LightGBM 流水线对比

| 步骤 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |
| `predict_proba` | 有（条件检查） | 有 | 有 | **无** |
| 混淆矩阵 | 有 | 有 | 有 | **无** |
| ROC 曲线 | 有（条件可用） | 有 | 有 | **无** |
| 残差图 | 无 | 无 | 无 | **有** |
| 学习曲线 | 无 | 有 | 无 | 无 |

### 理解重点

- XGBoost 流水线与其他集成模型的差异根源于任务类型——回归 vs 分类导致评估手段完全不同。
- 残差图是回归模型的标准诊断——它回答"预测值和真实值的偏差在哪些区域较大、有没有系统偏差"。

## 常见坑

1. 在回归场景下调用 `model.predict_proba()`——`XGBRegressor` 没有此方法，只有 `predict()`。
2. 误以为需要标准化——树模型基于分裂点比较，对特征尺度不变，标准化既非必须也无帮助。
3. 在 `train_test_split` 中传入 `stratify=y`——回归任务的连续目标没有类别可分层。
4. 在缺少 `xgboost` 的环境中直接运行流水线——会触发 `ImportError`。

## 小结

- XGBoost 流水线是最简洁的集成模型流水线——6 步完成数据拆分、训练、预测和两项评估，无标准化、无分层。
- `fit()` 的核心流程：二阶泰勒展开 $g_i + \frac{1}{2}h_i f^2$ -> 正则化目标 + 叶子权重闭式解 $w_j^* = -\frac{G_j}{H_j+\lambda}$ -> 加权分位数草图 + 列块并行 -> 300 棵树串行累加。
- `predict()` 输出连续实数——与分类集成模型的 softmax + argmax 预测路径在本质上不同。

# 评估与诊断

## 本章目标

1. 理解当前 XGBoost 流水线的两项评估输出——残差分析图和特征重要性。
2. 理解残差分析在回归任务中的诊断价值——比混淆矩阵更直接地反映预测误差。
3. 明确当前流水线**已实现**和**未实现**的评估项——以及回归评估与分类评估的根本差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 残差分析图 | 图表 | 残差散点图（y_true vs y_pred）+ 残差分布直方图——检验预测是否存在系统偏差 |
| 特征重要性 | 图表 | 8 个特征按分裂增益排序——揭示房价的核心驱动因素 |
| 终端日志 | 文本 | 训练完成时打印 9 项超参数和训练耗时 |

## 1. 残差分析图

`plot_residuals(y_test, y_pred, title="XGBoost 残差分析", ...)` 绘制残差诊断图。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(4128,)` | 测试集真实房价 | 来自 `train_test_split` |
| `y_pred` | `ndarray`，形状 `(4128,)` | 模型预测房价——300 棵树加权累加 | `model.predict(X_test)` |
| `title` | `str` | 图表标题 | `"XGBoost 残差分析"` |
| `dataset_name` | `str` | 数据集名称——决定输出路径 | `"xgboost"` |
| `model_name` | `str` | 模型名称——决定输出路径 | `"xgboost"` |

### 示例代码

```python
y_pred = model.predict(X_test)
plot_residuals(
    y_test, y_pred,
    title="XGBoost 残差分析",
    dataset_name="xgboost",
    model_name="xgboost",
)
```

### 输出

![残差分析图](https://img.yumeko.site/file/blog/articles/1780736282426.png)

### 理解重点

- 残差 = $y_{\text{true}} - y_{\text{pred}}$——正值表示模型低估了房价，负值表示高估。
- 残差分析图通常包含两个子图：
  - **残差散点图**：横轴为预测值，纵轴为残差——理想情况下残差随机分布在零线上下且无明显模式
  - **残差分布直方图**：残差的分布——理想情况下接近正态分布、中心在 0
- 常见异常模式：残差随预测值增大而增大（异方差性——对高房价区间预测不准确）、残差在某些区域系统偏高或偏低（模型欠拟合）。
- 与分类混淆矩阵的对比：混淆矩阵看"错分了多少"，残差图看"偏差有多大及其分布模式"。

## 2. 特征重要性

`plot_feature_importance(model, feature_names=feature_names, ...)` 绘制特征重要性柱状图。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `XGBRegressor` | 已训练的 XGBoost 模型——含 `feature_importances_` 属性 | `model` |
| `feature_names` | `list[str]` | 特征名列表 | `["MedInc", "HouseAge", ...]` |
| `title` | `str` | 图表标题 | `"XGBoost 特征重要性"` |
| `dataset_name` | `str` | 数据集名称 | `"xgboost"` |
| `model_name` | `str` | 模型名称 | `"xgboost"` |

### 示例代码

```python
feature_names = list(X.columns)
plot_feature_importance(
    model,
    feature_names=feature_names,
    title="XGBoost 特征重要性",
    dataset_name="xgboost",
    model_name="xgboost",
)
```

### 输出

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

### 理解重点

- XGBoost 的特征重要性基于**分裂增益**（`gain`）——每次分裂带来的损失下降量，按特征累计。
- 在加州房价数据上，`MedInc`（收入中位数）通常是最重要的特征——收入是房价的核心驱动因素，符合经济学直觉。
- `Latitude` 和 `Longitude`（地理位置）通常也具有高重要性——房价与地理位置强相关。
- 与 GBDT/LightGBM 的特征重要性含义相同——但 XGBoost 使用二阶增益公式计算。

## 3. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 残差分析图 | 已实现 | 回归评估的核心诊断工具 |
| 特征重要性 | 已实现 | XGBoost 的 `feature_importances_`——训练后自动可用 |
| 混淆矩阵 | **不适用** | 回归任务——无类别可混淆 |
| ROC 曲线 | **不适用** | 回归模型无 `predict_proba`，无 one-vs-rest 概念 |
| 学习曲线 | **未实现** | GBDT 分册已展示——XGBoost 不重复相同诊断 |
| $R^2$ / MAE / MSE 打印 | **未实现** | 可从残差图直接估计——图表比单个标量数值更具诊断价值 |
| 交叉验证 | **未实现** | 当前专注于单次切分评估——留出法在教学场景下足够 |

### 理解重点

- XGBoost 的评估体系与其他集成模型有根本差异——回归 vs 分类导致评估工具完全不同。
- 残差分析图是回归评估的"混淆矩阵等价物"——它回答"预测哪里错了、为什么错"，而非简单统计错分数量。
- 不打印 $R^2$ 等标量指标是有意设计——教学场景下残差分布的视觉诊断比单个数值更能揭示问题。

## 4. XGBoost vs 其他集成模型评估对比

| 评估维度 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 任务类型 | 分类 | 分类 | 分类 | **回归** |
| 混淆矩阵 | 2x2 | 3x3 | 4x4 | **不适用** |
| ROC 曲线 | 条件可用 | 始终可用 | 始终可用 | **不适用** |
| 残差分析 | 无 | 无 | 无 | **有** |
| 特征重要性 | 无 | 8 特征排序 | 20 特征排序 | **8 特征排序** |
| 学习曲线 | 无 | 有 | 无 | 无 |
| OOB 得分 | 有 | 无 | 无 | 无 |

### 理解重点

- 残差图 vs 混淆矩阵代表了回归评估与分类评估的本质差异——前者分析误差的幅度和模式，后者分析错分的类型和数量。
- 特征重要性在四个模型中的含义一致（分裂增益），但 XGBoost 使用其二阶增益公式计算。

## 5. 残差图的诊断重点

### 健康的残差图

- 残差随机散布在零线上下，无明显趋势
- 残差主要集中在 +/-0.5（半个价格单位）以内
- 没有明显的异方差性（高预测区残差显著增大）

### 需要关注的异常信号

- **漏斗形**：残差随预测值增大而扩大——模型对高房价区间的预测不稳定
- **U 形分布**：残差在中等预测区偏大——模型过度平滑了边界
- **偏移分布**：残差均值显著偏离 0——模型存在系统性高估或低估

## 常见坑

1. 用分类评估指标来评判回归模型——回归没有 accuracy/precision/recall 概念。
2. 仅看残差均值而忽略分布模式——均值接近 0 也可能存在结构性错误（高估低价房、低估高价房）。
3. 把特征重要性当成因果推断——`MedInc` 重要说明它与房价相关，但"涨工资"不一定直接"推高房价"。
4. 在回归数据上期待 ROC 曲线——回归模型没有 `predict_proba`。

## 小结

- XGBoost 当前有两项评估输出：残差分析图（回归诊断核心）和特征重要性（8 特征增益排序）。
- 残差图是回归评估的标准工具——它直接显示预测误差的大小、分布和模式，比单指标更信息丰富。
- 与分类集成模型的评估差异源于任务本质：回归分析误差幅度和模式，分类统计错分的类型和频次。

# 工程实现

## 本章目标

1. 理解 XGBoost 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径——注意回归任务的无标准化、无分层特点。
3. 理解 XGBoost 与其他集成模型在工程实现上的关键差异——可选依赖、回归评估、无数据预处理。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.xgboost()` | 静态方法 | 返回加州房价真实数据集——`fetch_california_housing` |
| `train_model(...)` | 函数 | 构建并训练 `XGBRegressor`——含可选依赖检查和 12 个可配置参数 |
| `run()` | 函数 | 回归流水线编排——6 步串联数据拆分、训练、预测和两项评估 |
| `plot_residuals(...)` | 函数 | 绘制残差散点图和分布图——回归专用 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` -> `data_generation/__init__.py` | 加载加州房价真实数据并导出 `xgboost_data` | 全局 `DataFrame`（20640 行 x 9 列） |
| 模型训练层 | `model_training/ensemble/xgboost.py` | 封装 `XGBRegressor` 训练——含 `ImportError` 处理 + 装饰器 | `XGBRegressor` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/xgboost.py` | 串联数据拆分、训练、预测和两项评估——端到端入口 | 终端日志 + 调用两个可视化函数 |
| 可视化层 | `result_visualization/residual_plot.py`、`feature_importance.py` | 生成两项评估图表 | 2 个 PNG 文件 |

### 理解重点

- XGBoost 的可视化层使用 `residual_plot.py`（回归专用）替代了 `confusion_matrix.py` 和 `roc_curve.py`（分类专用）。
- 训练层有三重保护：`try/except ImportError`（可选依赖）+ `@print_func_info`（调用日志）+ `@timeit`（耗时日志）。
- 与其他集成模型的核心工程差异：（1）无标准化步骤；（2）无分层抽样；（3）使用残差图替代混淆矩阵/ROC。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `xgboost_data.copy()` | — | `DataFrame`，形状 `(20640, 9)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["price"])` | `DataFrame` | `DataFrame`，形状 `(20640, 8)` | 分离 8 维特征 X |
| 3 | `data["price"]` | `DataFrame` | `Series`，形状 `(20640,)` | 分离连续回归目标 y |
| 4 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 提取特征名——供特征重要性图表使用 |
| 5 | `train_test_split(X, y, test_size=0.2)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 训练/测试切分（无 stratification） |
| 6 | `train_model(X_train, y_train)` | `(DataFrame, Series)` | `XGBRegressor` | 训练 300 棵二阶正则化树 |
| 7 | `model.predict(X_test)` | `DataFrame`，`(4128, 8)` | `ndarray`，`(4128,)` | 连续房价预测值 |
| 8 | `plot_residuals(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 残差散点图 + 分布图 |
| 9 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 8 个特征重要性排序柱状图 |

### 理解重点

- 步骤 5 无 `stratify=y` 参数——回归任务的连续目标没有类别可分层。
- 步骤 6 无标准化——树模型天然对特征缩放不敏感，跳过预处理环节。
- 步骤 8 使用残差分析替代分类的混淆矩阵/ROC——回归评估的根本差异。
- XGBoost 的流水线是最简洁的——6 步 vs Bagging 7 步、GBDT 9 步、LightGBM 7 步。

## 3. 数据依赖关系

```
xgboost_data (全局 DataFrame)
  - -> X = data.drop(columns=["price"])  ──-> feature_names = list(X.columns) ──┐
  - -> y = data["price"]                                                        │
  - -> train_test_split(X, y, test_size=0.2)                                    │
    - -> X_train (16512, 8) ──────────────────────────────────────┐          │
    - -> y_train (16512,) ────────────────────────┐               │          │
    - -> X_test (4128, 8) ────────────────────┐   │               │          │
    - -> y_test (4128,) ───────────────┐      │   │               │          │
    - train_model(X_train, y_train) ──-> model │   │               │          │
      - -> model.predict(X_test) ──-> y_pred─┘   │               │          │
      - -> model.feature_importances_ ──-> + feature_names ──────┘          │
      - plot_residuals(y_test, y_pred, ...) <-─────┘                         │
      - plot_feature_importance(model, feature_names, ...) <-────────────────┘
```

### 理解重点

- XGBoost 的数据依赖图是最简洁的——无 `StandardScaler` 分支、无 `plot_learning_curve` 分支、无 `predict_proba` 分支。
- `y_train` 仅参与训练，`y_test` 仅参与残差分析——没有混淆矩阵和 ROC 的数据需求。
- `feature_names` 与 `feature_importances_` 交汇于特征重要性可视化——流程清晰。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 残差分析图 | `outputs/xgboost/residual_plot.png` | PNG | 残差散点图（预测值 vs 残差）+ 残差分布直方图 |
| 特征重要性 | `outputs/xgboost/feature_importance.png` | PNG | 8 个特征的重要性排序柱状图 |
| 终端日志 | 标准输出 | 文本 | 9 项训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.xgboost
```

### 输出

```text
============================================================
XGBoost 回归流水线
============================================================
模型训练完成
n_estimators: 300
learning_rate: 0.05
max_depth: 6
min_child_weight: 1
subsample: 0.9
colsample_bytree: 0.9
gamma: 0.0
reg_alpha: 0.0
reg_lambda: 1.0
模型训练耗时: 3.11s

============================================================
XGBoost 流水线完成！
============================================================
```

### 理解重点

- XGBoost 输出 2 个 PNG 文件——四个集成模型中最少（Bagging 2 个、GBDT 4 个、LightGBM 3 个）。
- 训练耗时比 GBDT（~2s）长——因为 20640 个样本远多于 GBDT 的 500 个，但 `n_jobs=-1` 列块并行在一定程度上抵消了数据规模增长。
- 终端日志打印 9 项超参数——四个模型中最多，体现 XGBoost 参数体系的丰富程度。

## 5. 训练层细节：与其他集成模型的对比

| 工程维度 | Bagging | GBDT | LightGBM | XGBoost |
|---|---|---|---|---|
| 任务 | 分类 | 分类 | 分类 | **回归** |
| 模型类 | `BaggingClassifier` | `GradientBoostingClassifier` | `LGBMClassifier` | **`XGBRegressor`** |
| 依赖 | sklearn 内置 | sklearn 内置 | `pip install lightgbm` | **`pip install xgboost`** |
| 导入保护 | `try/except TypeError` | 无 | `try/except ImportError` | `try/except ImportError` |
| 装饰器 | 无 | `timer` | `@print_func_info` + `@timeit` + `timer` | `@print_func_info` + `@timeit` + `timer` |
| 标准化 | 有 | 有 | 有 | **无** |
| 分层抽样 | 有 | 有 | 有 | **无** |
| 评估项 | 混淆矩阵 + ROC | 混淆矩阵 + ROC + 特征重要性 + 学习曲线 | 混淆矩阵 + ROC + 特征重要性 | **残差图 + 特征重要性** |
| 超参数数 | 5 | 4 | 6 | **9** |

### 理解重点

- XGBoost 的训练层参数是四个模型中最丰富的——从 `gamma` 到 `reg_lambda`，体现更精细的控制粒度。
- 无标准化和无分层的设计使得 XGBoost 的流水线最简洁——树模型的工程便利性在此充分体现。
- XGBoost 与 LightGBM 共享可选的依赖处理模式——两者都不是 sklearn 原生，需要 `try/except` 保护。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解 `xgboost()` 的数据加载逻辑（加州房价真实数据）
2. `model_training/ensemble/xgboost.py` — 理解 `XGBRegressor` 的构建、可选依赖和二阶训练
3. `pipelines/ensemble/xgboost.py` — 看清端到端回归流程和两项评估的串联
4. `result_visualization/residual_plot.py` — 了解残差分析图实现
5. `result_visualization/feature_importance.py` — 了解特征重要性图表实现

## 常见坑

1. 在不含 `xgboost` 的环境中直接 `from model_training.ensemble.xgboost import train_model`——会触发 `ImportError`，需先 `pip install xgboost`。
2. 在回归数据上传递 `stratify=y`——回归无类别可分层，会直接报错。
3. 直接修改 `xgboost_data` 而不先 `copy()`——会污染其他模块引用的同一全局变量。
4. 期望 XGBoost 流水线输出混淆矩阵——回归任务没有混淆矩阵概念。

## 小结

- XGBoost 工程实现遵循本仓库标准四层架构：数据生成层 -> 模型训练层 -> 流水线编排层 -> 可视化层（含 2 个模块）。
- `run()` 是四个集成模型中最简洁的编排函数——6 步完成数据拆分、训练、预测和两项评估，无预处理步骤。
- 与其他集成模型的三个关键工程差异：（1）回归任务——无分类评估；（2）真实数据——无需标准化；（3）参数体系最丰富——9 项可配置超参数。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 XGBoost 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 XGBoost 的行为——注意回归任务的特点。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对二阶泰勒展开、显式正则化、加权分位数、XGBoost vs GBDT/LightGBM 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 XGBoost 行为变化——建立回归参数-效果的直觉 |
| 参考文献 | 入口 | 提供 XGBoost 原始论文、官方文档和扩展阅读 |

## 1. 自检问题

1. XGBoost 的二阶泰勒展开引入了 Hessian $h_i$——在 MSE 回归任务中 $h_i$ 的值是什么？这对 XGBoost 的二阶优势有何影响？

2. XGBoost 的正则化目标函数包含三个正则化项：$\gamma T$ + $\frac{1}{2}\lambda\|\mathbf{w}\|^2$ + $\alpha\|\mathbf{w}\|_1$。这三个项分别作用于什么层级？在过拟合时应该如何调整？

3. 叶子权重的闭式解 $w_j^* = -\frac{G_j}{H_j + \lambda}$ 中，$\lambda$ 的作用是什么？为什么 GBDT（sklearn）没有类似的闭式解？

4. 分裂增益公式 $\text{Gain} = \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$ 中，$\gamma$ 和 $\lambda$ 分担了什么不同的角色？

5. XGBoost 的加权分位数草图与 LightGBM 的直方图分桶有何不同？在 MSE 回归下（$h_i=1$），两者是否等价？

6. `min_child_weight` 在回归任务中（$h_i=1$）等价于什么？在分类任务中（$h_i = p_i(1-p_i)$），同一个 `min_child_weight=1` 的含义有何不同？

7. XGBoost 与其他三个集成模型（Bagging/GBDT/LightGBM）在任务类型、数据规模、评估体系上的核心差异有哪些？

## 2. 动手练习

### 练习 1：调整正则化系数 `reg_lambda`

将 `reg_lambda` 分别设为 `0.0`、`0.1`、`1.0`（默认）、`10.0`、`100.0`，观察残差图和特征重要性的变化。

```python
model = train_model(X_train, y_train, reg_lambda=0.0)
```

回答：`reg_lambda=0.0`（关闭 L2）时，残差的分布是否变大？`reg_lambda=100.0` 是否导致欠拟合（残差系统性增宽）？

### 练习 2：调整分裂门槛 `gamma`

将 `gamma` 分别设为 `0.0`、`0.1`、`1.0`、`5.0`，观察树结构的变化。

```python
model = train_model(X_train, y_train, gamma=1.0)
```

回答：`gamma` 增大后，树的数量是否减少？`gamma=5.0` 时是否出现明显的欠拟合？

### 练习 3：改变树深度 `max_depth`

将 `max_depth` 分别设为 `2`、`4`、`6`、`10`、`15`，观察残差图变化。

```python
model = train_model(X_train, y_train, max_depth=2)
```

回答：`max_depth=2` 的残差图是否出现系统性偏差？`max_depth=15` 是否在真实数据上过拟合？

### 练习 4：对比 XGBoost 与 GBDT 在回归任务上的表现

使用相同的加州房价数据，分别训练 XGBoost 和 sklearn `GradientBoostingRegressor`，对比残差。

```python
from sklearn.ensemble import GradientBoostingRegressor

model_gbdt = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
model_gbdt.fit(X_train, y_train)
y_pred_gbdt = model_gbdt.predict(X_test)
```

回答：XGBoost 的预测残差是否比 GBDT 更小？正则化（`reg_lambda=1.0`）是否带来了泛化提升？

### 练习 5：改变采样比例

将 `subsample` 和 `colsample_bytree` 分别设为更低的值（如 `0.5`），观察训练耗时和残差的变化。

```python
model = train_model(X_train, y_train, subsample=0.5, colsample_bytree=0.5)
```

回答：采样比例降至 0.5 后，训练是否明显加速？残差是否显著增大？这种采样比例在什么场景下可能有用？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. | XGBoost 原始论文——二阶泰勒展开、正则化目标函数和系统设计的完整推导 |
| 2 | XGBoost 官方文档 — [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) | 全部参数的官方说明和调参指南 |
| 3 | scikit-learn 兼容接口 — [XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) | XGBoost 回归模型的 scikit-learn API 参考 |
| 4 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. | GBDT 的理论基础——XGBoost 在此基础上引入二阶展开和显式正则化 |

## 常见坑

1. 在 MSE 回归中期待二阶展开带来巨大提升——MSE 的 Hessian 是常数 1，二阶优势主要体现于分类（$h_i = p_i(1-p_i)$ 非均匀）。
2. 忽略 `reg_lambda=1.0` 的默认值——与 GBDT/LightGBM 不同，XGBoost 的 L2 默认开启，调参时应优先调整它。
3. 混淆 `min_child_weight` 与 `min_samples_leaf`——两者仅在 Hessian 恒为 1（MSE 回归）时等价。
4. 在 `train_test_split` 中误传入 `stratify=y`——回归无类别可分层。

## 小结

- 7 个自检问题覆盖 XGBoost 的核心创新：二阶泰勒展开、三重正则化、闭式解、加权分位数、`min_child_weight` 含义、与其他集成模型对比。
- 5 个动手练习从不同角度探索 XGBoost 的行为——调整 L2 正则化、gamma 门槛、树深度、对比 GBDT 回归、改变采样比例。
- 4 篇参考文献从原始论文（Chen & Guestrin 2016）$\rightarrow$ 官方文档 $\rightarrow$ API 参考 $\rightarrow$ GBDT 理论基础构成完整的阅读路线。
