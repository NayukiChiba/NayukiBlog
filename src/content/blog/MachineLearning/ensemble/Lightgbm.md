---
title: LightGBM
date: 2026-05-21
category: 机器学习/集成学习
tags:
  - Scikit-learn
description: LightGBM的数学原理、Leaf-wise生长与直方图加速及完整实现。
image: https://img.yumeko.site/file/blog/cover/1780581787373.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 LightGBM 与 GBDT 共享的数学基础——加法模型、负梯度拟合、多类对数损失。
2. 理解 LightGBM 独有的工程优化：Leaf-wise 生长策略、直方图算法、GOSS 采样、EFB 特征捆绑。
3. 理解 LightGBM 的参数选择如何在数学上影响模型——`num_leaves` vs `max_depth`、`learning_rate` 收缩。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 加法模型 | 数学框架 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$——GBDT 系列共享的建模方式 |
| 负梯度 | 优化理论 | 每棵树拟合 $\tilde{y}_i^{(m)} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$——函数空间的梯度下降 |
| Leaf-wise 生长 | 树生长策略 | 每次选择损失下降最多的叶子分裂——更快的收敛速度和更深的树 |
| 直方图算法 | 加速技术 | 连续特征离散化为 $k$ 个 bins——分割点搜索从 $O(n\log n)$ 降到 $O(k)$ |
| GOSS | 采样策略 | 保留所有大梯度样本 + 从小梯度样本中随机采样——在信息损失很小的前提下加速训练 |
| EFB | 降维技术 | 将互斥特征捆绑为一个特征——减少直方图构建开销 |

## 1. GBDT 数学基础（与 LightGBM 共享）

LightGBM 在数学框架上与 GBDT 完全一致——都是加法模型 + 负梯度拟合。

### 加法模型

$$
F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x}; \Theta_m)
$$

其中 $h_m$ 是第 $m$ 棵回归树，$\nu$ 是学习率（`learning_rate`），$\Theta_m$ 是树的结构参数。

### 多类对数损失

对于 $K=4$ 类分类问题，使用多类对数损失（交叉熵）：

$$
L(\{y_i\}, \{F(\mathbf{x}_i)\}) = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log p_k(\mathbf{x}_i)
$$

其中 $p_k(\mathbf{x}_i) = \frac{\exp(F_k(\mathbf{x}_i))}{\sum_{j=1}^{K} \exp(F_j(\mathbf{x}_i))}$（softmax），$y_{ik}$ 是 one-hot 编码。

### 负梯度（残差近似）

第 $m$ 轮对第 $k$ 类的负梯度：

$$
\tilde{y}_{ik}^{(m)} = -\left[\frac{\partial L}{\partial F_k(\mathbf{x}_i)}\right]_{F=F^{(m-1)}} = y_{ik} - p_k^{(m-1)}(\mathbf{x}_i)
$$

即**真实概率与当前预测概率之差**——新树拟合这个差值。

### 理解重点

- LightGBM 在数学上等价于 GBDT——差异全在工程实现，不在数学框架。
- 负梯度 $\tilde{y}_i$ 在分类场景下恰好是"残差概率"——当前预测的 softmax 概率与真实 one-hot 的偏差。
- 学习率 $\nu=0.05$ 表示每棵树只修正残差概率的 5%——防止单棵树修正过猛。

## 2. Leaf-wise 生长（LightGBM 独有）

### Level-wise（sklearn GBDT）的局限

传统 GBDT 按层生长（Level-wise）：每层所有叶子同时分裂——不分"重要"和"不重要的"叶子。

### Leaf-wise 策略

LightGBM 按叶子生长（Leaf-wise）：在所有叶子中，选择**分裂后损失下降最多**的叶子进行分裂。

数学上：设叶子的分裂增益为 $\Delta L_j$，选择

$$
j^* = \arg\max_j \Delta L_j
$$

重复此过程直到叶子数达到 `num_leaves=31`。

### 参数关系

| Leaf-wise 关键参数 | 数学含义 |
|---|---|
| `num_leaves=31` | 最大叶子数——复杂度上限 |
| `max_depth=-1` | 不限制深度——Leaf-wise 树可能很深但叶子数固定 |

### 理解重点

- Leaf-wise 使 Loss 下降更高效——同等叶子数下，Leaf-wise 树的损失低于 Level-wise 树。
- 代价是可能生成极深的树（深度 $\gg \log_2(\text{num\_leaves})$）——因此需要 `min_child_samples=20` 等正则化手段防止叶子过小。
- 与 Bagging 的完全生长树不同——Leaf-wise 仍受 `num_leaves` 限制，不会无限生长。

## 3. 直方图算法

### 传统方法：预排序

sklearn GBDT 对每个特征的每个分裂点，排序后逐一计算损失——复杂度 $O(n_{\text{unique}})$。

### LightGBM：直方图分桶

将连续特征值离散化为 $k$ 个 bins（直方图桶），只在桶边界搜索分裂点——复杂度 $O(k)$，$k \ll n_{\text{unique}}$。

数学上：

$$
\text{bin}(x_j) = \lfloor k \cdot \frac{x_j - x_{\min}}{x_{\max} - x_{\min}} \rfloor
$$

### 理解重点

- 直方图加速是 LightGBM 快于 sklearn GBDT 3-5 倍的核心原因——在大数据上差距更大。
- 分桶带来轻微的正则化效果——离散化后的分割点更粗糙，有助于防止过拟合。
- 代价是牺牲了极细粒度的分割点——但在实践中，256 个桶通常足够（默认 `max_bin=255`）。

## 4. GOSS（Gradient-based One-Side Sampling）

### 动机

在 Boosting 中，大梯度样本（$|\tilde{y}_i|$ 大）对训练更重要——它们是"还没学好的样本"。

### GOSS 策略

1. 按梯度绝对值 $|\tilde{y}_i|$ 排序所有样本
2. 保留前 $a \times 100\%$ 的大梯度样本（不采样）
3. 从剩余小梯度样本中随机采样 $b \times 100\%$
4. 为小梯度样本乘以权重 $\frac{1-a}{b}$ 以补偿

当前源码 `subsample=0.9`（全局采样）——未显式启用 GOSS（需要分别设置 `top_rate` 和 `other_rate`）。但 `subsample` 机制与 GOSS 的思想一致：利用梯度信息偏向保留重要样本。

### 理解重点

- GOSS 使得 LightGBM 在保持训练精度的前提下，减少了参与分裂计算的样本数。
- 梯度是样本"重要性"的天然代理——大梯度样本是当前模型处理不好的样本。

## 5. EFB（Exclusive Feature Bundling）

### 动机

高维稀疏数据中，许多特征互斥（不会同时为非零值）。EFB 将互斥特征捆绑为一个特征，减少直方图构建开销。

对于当前 20 维稠密数据，EFB 的收益有限——但这是 LightGBM 在处理稀疏高维数据时的关键加速手段。

### 理解重点

- EFB 本质上是一个图着色问题——将互斥特征（冲突少的特征）分到同一组，每组构建一个共享直方图。
- 在当前数据上 `n_features=20`，EFB 的收益不大——但数据维度提升到数千维时，EFB 的降维效果显著。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ | `LGBMClassifier(n_estimators=300, learning_rate=0.05)` |
| 多类对数损失 | $L = -\sum_i \sum_k y_{ik} \log p_k(\mathbf{x}_i)$ | `objective='multiclass'`（内部默认） |
| 负梯度 | $\tilde{y}_{ik} = y_{ik} - p_k(\mathbf{x}_i)$ | 内部自动计算 |
| Leaf-wise 生长 | $\arg\max_j \Delta L_j$ | `num_leaves=31, max_depth=-1` |
| 直方图分桶 | $\text{bin}(x) = \lfloor k \cdot (x - x_{\min})/(x_{\max} - x_{\min})\rfloor$ | `max_bin=255`（内部默认） |
| 行采样 | 按梯度采样 | `subsample=0.9` |
| 列采样 | 随机选择特征子集 | `colsample_bytree=0.9` |
| Softmax 概率 | $p_k = \exp(F_k)/\sum_j \exp(F_j)$ | `model.predict_proba(X)` |
| 学习率收缩 | $\nu \cdot h_m$ | `learning_rate=0.05` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 7. LightGBM vs GBDT 数学对比

| 维度 | GBDT (sklearn) | LightGBM |
|---|---|---|
| 加法模型 | $F_M = \sum \nu h_m$ | $F_M = \sum \nu h_m$——相同 |
| 损失函数 | 多类对数损失 | 多类对数损失——相同 |
| 负梯度 | $\tilde{y} = y - p$ | $\tilde{y} = y - p$——相同 |
| 树生长策略 | Level-wise（按层） | Leaf-wise（按叶子）——**不同** |
| 分裂点搜索 | 预排序 -> 逐一计算 | 直方图分桶 -> 桶边界搜索——**不同** |
| 样本采样 | 随机子采样 | GOSS（梯度加权采样）——**不同** |
| 特征降维 | 无 | EFB（互斥特征捆绑）——**不同** |
| 树复杂度控制 | `max_depth=3` | `num_leaves=31`——**不同** |

### 理解重点

- LightGBM 在数学主链上与 GBDT 完全相同——差异全在算法实现的四个环节：生长策略、分裂点搜索、样本采样、特征处理。
- 这四个差异使得 LightGBM 在训练速度上有数量级优势——但预测精度与调好参的 GBDT 通常相当。

## 常见坑

1. 把 `max_depth=-1` 当成"树可以无限大"——Leaf-wise 生长下，`num_leaves` 才是真正的复杂度上限。
2. 把 GOSS 当成普通的随机子采样——GOSS 保留所有大梯度样本，不是均匀随机采样。
3. 以为 EFB 总是有效——在稠密低维数据上，特征间几乎没有互斥关系，EFB 收益极小。
4. 忽略学习率与树数量的耦合——$\nu$ 和 $M$ 共同决定总修正量 $M \times \nu$。

## 小结

- LightGBM 的数学核心链与 GBDT 完全一致：加法模型 + 负梯度拟合 + 多类对数损失 + softmax 输出。
- LightGBM 的工程优化链：Leaf-wise 生长（损失下降更高效）-> 直方图分桶（分裂搜索加速）-> GOSS（梯度加权采样）-> EFB（互斥特征捆绑）——四项优化在不改变数学框架的前提下大幅提升训练速度。
- 当前源码 `LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9)` 是轻量级高维数据的经典配置。

# 数据构成

## 本章目标

1. 明确本仓库 LightGBM 数据来自 `EnsembleData.lightgbm()` 构造的高维多类别分类数据。
2. 理解为什么选择高维数据——`n_features=20`（含 7 个纯噪声特征）充分展示 LightGBM 的直方图加速和 GOSS 采样优势。
3. 明确当前流程中的训练/测试切分（分层抽样）和标准化顺序。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.lightgbm()` | 方法 | 生成 LightGBM 使用的高维多类别分类数据 |
| `make_classification(...)` | 函数 | scikit-learn 提供的合成分类数据生成器 |
| `lightgbm_data` | 变量 | 在 `data_generation/__init__.py` 中导出的全局 DataFrame |
| `lgbm_class_sep` | 参数 | 类别间隔 `0.6`——较小的间隔提高分类难度，体现 Boosting 的偏差缩减能力 |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——训练集拟合、测试集变换 |

## 1. 数据生成：`EnsembleData.lightgbm()`

当前 LightGBM 数据来自 `EnsembleData.lightgbm()`，底层调用 `sklearn.datasets.make_classification()`。

### 参数速览

适用函数：`make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=5, n_classes=4, n_clusters_per_class=1, class_sep=0.6, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。`1000`——比 GBDT 多一倍，提供更充足的训练信号，同时展示 LightGBM 的直方图加速优势 | `500`、`1000`、`2000` |
| `n_features` | `int` | 总特征数。`20`——高维设置，体现 LightGBM 处理大规模特征的能力 | `8`、`20`、`50` |
| `n_informative` | `int` | 有效特征数。`8`——与类标签真正相关的独立信号 | `4`、`8` |
| `n_redundant` | `int` | 冗余特征数。`5`——由有效特征线性组合生成 | `2`、`5` |
| `n_classes` | `int` | 类别数。`4`——多分类 $\{0, 1, 2, 3\}$ | `3`、`4` |
| `n_clusters_per_class` | `int` | 每类的簇数。`1`——每类一个高斯簇 | `1`、`2` |
| `class_sep` | `float` | 类别间隔。`0.6`——较小的间隔使类别重叠较多，分类难度较高 | `0.3`、`0.6`、`1.5` |
| `random_state` | `int` | 随机种子，保证数据可复现。`42` | `42` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(1000, 20)$，$y$ 取值 $\{0, 1, 2, 3\}$ | — |

### 示例代码

```python
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=5,
    n_repeated=0,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=0.6,
    random_state=42,
)
columns = [f"x{i + 1}" for i in range(20)]
data = DataFrame(X, columns=columns)
data["label"] = y
```

### 理解重点

- `n_features=20` 是 LightGBM 独有设计——比 GBDT 的 8 维高 2.5 倍。含 8 个有效特征 + 5 个冗余特征 + 7 个纯噪声特征（$20 - 8 - 5 = 7$）。
- `class_sep=0.6` 低于 GBDT 的 `0.7`——类别间隔更小意味着更高的分类难度和更模糊的类别边界。
- `n_samples=1000` 是 GBDT 的两倍——更大数据量使 LightGBM 的直方图加速优势更加显著。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(1000, 20)$ | 含 20 个连续特征的特征矩阵，列名 `x1`~`x20` | `data.drop(columns=["label"])` |
| `y` | `Series`，形状 $(1000,)$ | 四分类标签 $\{0, 1, 2, 3\}$——参与 LightGBM 训练和评估 | `data["label"]` |

### 特征构成

| 特征范围 | 数量 | 类型 | 说明 |
|---|---|---|---|
| `x1` ~ `x8` | 8 | 有效特征 | 由 `make_classification` 生成的独立信号——与标签直接相关 |
| `x9` ~ `x13` | 5 | 冗余特征 | 由 `x1`~`x8` 线性组合生成——提供重复信息 |
| `x14` ~ `x20` | 7 | 噪声特征 | 随机生成——与标签无任何关联 |

### 理解重点

- 特征重要性的期望排序：`x1`~`x8` > `x9`~`x13` > `x14`~`x20`——这是验证 LightGBM 特征选择能力的关键诊断。
- `label` 是四分类监督标签——与 GBDT 的三分类和 Bagging 的二分类形成难度梯度。
- 与 GBDT 的数据对比：类别多 1 个（4 vs 3），特征多 12 个（20 vs 8），样本多 500 个（1000 vs 500）。

## 3. 训练/测试切分与标准化

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(1000, 20)$ | 全量特征矩阵 | `X` |
| `y` | `Series`，形状 $(1000,)$ | 全量标签 | `y` |
| `test_size` | `float` | 测试集比例。`0.2` | `0.2`、`0.3` |
| `stratify` | `array_like` | 分层抽样依据——确保训练/测试集中类别比例一致 | `y` |
| `random_state` | `int` | 随机种子。`42` | `42` |
| 返回值 | `(DataFrame, DataFrame, Series, Series)` | `X_train`（800 样本）、`X_test`（200 样本）及对应标签 | — |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 当前流水线**有**训练/测试切分——与集成分类系列（Bagging/GBDT）一致。
- `stratify=y` 确保 200 个测试样本中 4 个类别的分布比例与原始数据一致——在多分类场景下尤其重要。
- 标准化采用监督学习标准做法：`fit_transform` 在训练集上计算 $\mu$ 和 $\sigma$，`transform` 在测试集上使用相同统计量。

## 4. 数据设计意图：与 GBDT 的对比

| 数据维度 | GBDT | LightGBM | 设计意图 |
|---|---|---|---|
| 样本数 | 500 | **1000** | 更大数据量——展示直方图加速优势 |
| 特征维度 | 8 (4+2+2) | **20 (8+5+7)** | 更高维——展示 EFB 和列采样的价值 |
| 类别数 | 3 | **4** | 更多类别——提高多分类复杂度 |
| 类别间隔 | 0.7 | **0.6** | 更难分类——展示偏差缩减的必要性 |
| 噪声特征数 | 2 | **7** | 更多噪声——展示特征重要性筛选能力 |

### 理解重点

- LightGBM 的数据设计在所有维度上都比 GBDT"更大更难"——这是有意为之，因为 LightGBM 的工程优化使其在高维数据上仍有显著的训练速度优势。
- 20 维中 12 个非独立有效信号——特征重要性图表天然区分有效特征与冗余/噪声特征。

## 数据可视化

![特征相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

## 常见坑

1. 把 `class_sep=0.6` 当成数据缺陷——低间隔是有意设计，分类难度过低无法体现偏差缩减的价值。
2. 忽略 `stratify=y` 的重要性——四分类数据上不设分层抽样可能导致测试集中某类别过少。
3. 在测试集上 `fit_transform` 而非 `transform`——标准信息泄露。
4. 忘记 `lightgbm_data` 是模块级全局变量——直接修改会污染其他模块。

## 小结

- 当前 LightGBM 数据来自 `make_classification(n_samples=1000, n_features=20, n_informative=8, n_classes=4, class_sep=0.6)`：20 个连续特征、四分类、高维较高难度。
- 数据流为：`make_classification` -> DataFrame（`x1`~`x20` + `label`）-> 分层训练/测试切分 -> 训练集拟合标准化器 / 测试集变换。
- `n_features=20` 和 `class_sep=0.6` 的设计意图是充分展示 LightGBM 在高维中等难度数据上的直方图加速速度和特征筛选能力。

# 思路与直觉

## 本章目标

1. 用直观方式理解 LightGBM 相对于 GBDT 的三项核心创新——Leaf-wise 生长、直方图加速、GOSS 采样。
2. 理解为什么 LightGBM 比 GBDT 更快——不在数学框架，而在工程实现。
3. 通过与 GBDT 的对比，建立 LightGBM 在 Boosting 谱系中的定位——高效的 GBDT 工程实现。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Leaf-wise 生长 | 树生长策略 | 不按层分裂所有叶子——只分裂"最值得长"的那一片叶子 |
| 直方图分桶 | 加速技术 | 连续值 -> 离散桶——分裂点从逐个值扫描降为桶边界扫描 |
| GOSS | 采样策略 | 梯度大的样本全保留（没学好），梯度小的样本随机采样（已学好） |
| EFB | 降维技术 | 把互斥特征打包——减少特征扫描次数 |
| 特征重要性 | 附加收益 | 基于分裂增益自动排序——20 个特征中区分有效 vs 噪声 |

## 1. 为什么需要 LightGBM

GBDT（sklearn）有一个工程瓶颈：**训练慢**。当数据量或特征维度增大时，Level-wise 生长 + 预排序分裂点搜索的计算成本急剧上升。

LightGBM 的思路很直接：

> GBDT 的数学框架完全保留——只是用 Leaf-wise 替代 Level-wise、用直方图替代预排序、用 GOSS 替代随机采样。同样的数学，快 10 倍的训练。

### 理解重点

- LightGBM 不创造新的数学——它在 GBDT 之上做了**纯工程优化**。预测结果与调好参的 sklearn GBDT 几乎一致。
- 三大优化瞄准三个计算瓶颈：（1）树怎么长——Leaf-wise；（2）分裂点怎么找——直方图；（3）样本怎么用——GOSS。
- 当前数据 1000 样本 x 20 特征时，LightGBM 训练约 0.4s，GBDT 约 2s——差距约 5 倍。数据越大差距越明显。

## 2. 用"重点培养"理解 Leaf-wise 生长

Level-wise（GBDT）像学校统一教学——每层（深度）所有学生（叶子）同时上课（分裂）。

Leaf-wise（LightGBM）像精英教育——找最有潜力的学生（损失下降最多的叶子）专门辅导（分裂），直到叶子数达到 `num_leaves=31`。

### 理解重点

- Leaf-wise 在同等叶子数下，对复杂边界的拟合更精细——因为它优先把计算资源投入到"最有价值"的区域。
- 代价是树可能非常深——一片叶子连续分裂 7 次就深达 7 层。因此需要 `min_child_samples=20` 等正则化防止单个叶子过小。
- 与 Bagging 的完全生长树不同——Leaf-wise 有明确的 `num_leaves` 上限，不是无限生长。

## 3. 用"大约取整"理解直方图

预排序（sklearn GBDT）像拿着精确的尺子——每个数据点的值都要排序、逐一比较。

直方图（LightGBM）像把尺子刻度简化——只保留 255 个整厘米刻度，按近似值分桶后只在这些整数边界比较。

### 理解重点

- 离散化损失了微小的精度——但换来了巨大的速度提升。在实践中，255 个桶的分割精度通常足够。
- 直方图还有意外的好处——离散化本身是一种正则化，分割更粗糙反而有助于防止过拟合。
- 这就是为什么 LightGBM 在 Kaggle 等比赛中常比调好参的 GBDT 效果更好——快只是其一，直方图正则化是其二。

## 4. 用"补差距"理解 GOSS

一个班级里：
- **成绩差的学生**（大梯度 $\vert\tilde{y}_i\vert$ 大）——需要老师重点关注，全部保留
- **成绩好的学生**（小梯度 $\vert\tilde{y}_i\vert$ 小）——已经学得不错，随机抽查几个即可

GOSS 就是这个策略——100% 保留大梯度样本 + 随机抽样小梯度样本 -> 训练样本减少但信息损失很小。

### 理解重点

- 梯度大小天然衡量"学习紧急程度"——大梯度 = 预测偏差大 = 还没学好。
- 随机丢弃"已学好"的样本几乎不影响下一棵树的训练——因为它们对负梯度的贡献已经很小。
- 当前源码 `subsample=0.9` 是均匀概率采样（非严格 GOSS），但其思想一致：减少参与训练的数据量以加速。

## 5. 与 Bagging 和 GBDT 的直觉对比

| 维度 | Bagging | GBDT | LightGBM |
|---|---|---|---|
| 核心问题 | 如何让"太敏感"的专家冷静？ | 如何让"太迟钝"的新手变聪明？ | 如何让"太迟钝"的新手**快速**变聪明？ |
| 训练方式 | 平行——各学各的 | 串行——后人补前人的错 | 串行——更快地补前人的错 |
| 基学习器 | 强学习器（完全生长树） | 弱学习器（浅层树 `max_depth=3`） | 弱学习器（Leaf-wise 树 `num_leaves=31`） |
| 核心收益 | 降方差 | 降偏差 | 降偏差——但训练快 10 倍 |
| 速度瓶颈 | 无（天然并行） | 预排序分裂点搜索 | 直方图分桶 + GOSS——大幅加速 |
| 诊断工具 | OOB 得分 | 特征重要性 + 学习曲线 | 特征重要性 |
| 理想场景 | 高噪声 + 少特征 | 中等维度 + 中等噪声 | **高维 + 大规模数据** |

### 理解重点

- LightGBM 在定位上不是"Boosting 的替代方案"——它是 GBDT 的**高效实现**。数学相同，工程不同。
- 三者的互补关系：Bagging 降方差（平行）、GBDT 降偏差（串行）、LightGBM 更快地降偏差（串行 + 直方图 + GOSS + EFB）。

## 6. 特征重要性在高维数据上的直觉

20 个特征中有 8 个有效、5 个冗余、7 个纯噪声——特征重要性图表天然展示了 Boosting 的"自动特征选择"能力。

### 理解重点

- 冗余特征（`x9`~`x13`）的重要性通常低于有效特征——因为信息已经被有效特征提供。
- 噪声特征（`x14`~`x20`）的重要性应该最低——它们不提供任何真实信息。
- 这就是 Boosting 优于单一决策树的地方——300 棵树的累加使得噪声特征的重要性被稀释，有效特征的重要性被放大。

## 可视化

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

## 常见坑

1. 把 LightGBM 当成"GBDT 的上位替代"——在极小数据（<100 样本）上，直方图分桶的离散化损失可能反而导致精度低于精调 GBDT。
2. 以为 `num_leaves` 越大越好——Leaf-wise 生长下叶子过多会生成极深的树，容易过拟合。
3. 忽略 `max_depth=-1` 的潜在风险——Leaf-wise 树可能深达 10-15 层，单叶子样本极少。
4. 在低维数据上过度推崇 LightGBM——`n_features=2` 时直方图加速优势不明显。

## 小结

- LightGBM 的直觉核心是"更快地做 GBDT"——Leaf-wise 生长（重点培养）+ 直方图分桶（大约取整）+ GOSS（补差距）+ EFB（打包互斥特征）四项工程优化在不改变数学框架的前提下大幅加速。
- 当前 20 特征四分类数据是展示 LightGBM 在高维场景下速度优势的最佳教学配置——1000 样本时训练比 GBDT 快约 5 倍。
- LightGBM 与 GBDT 在直觉上是同一家族——都靠串行接力把粗糙变精细（降偏差），LightGBM 只是让每一次接力跑得更快。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LGBMClassifier`（含可选依赖检查）。
2. 理解 `LGBMClassifier` 的核心构造器参数（`n_estimators`、`learning_rate`、`num_leaves`、`subsample`、`colsample_bytree`）及其与 GBDT 的差异。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`n_estimators_`（实际树数）、`predict_proba`（概率输出）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `lightgbm.LGBMClassifier` 模型——含可选依赖检查 |
| `LGBMClassifier(...)` | 类 | LightGBM 的 scikit-learn 兼容接口——Leaf-wise 生长 + 直方图加速的 GBDT 实现 |
| `model.fit(X_train, y_train)` | 方法 | 串行训练 300 棵 Leaf-wise 直方图回归树——支持 GOSS + 直方图分桶 |
| `model.feature_importances_` | 属性 | 特征重要性——基于分裂增益累加，和为 1.0 |
| `model.predict(X)` | 方法 | 加法模型输出——300 棵树加权累加 -> softmax -> argmax |
| `model.predict_proba(X)` | 方法 | 概率输出——300 棵树加权累加 -> softmax |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(800, 20)` | 标准化后的训练特征矩阵，传入 `LGBMClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(800,)` | 训练标签 $\{0, 1, 2, 3\}$——四分类监督信息 | `y_train` |
| `n_estimators` | `int` | 弱学习器数量。当前 `300`——步数更多但每步更小（`learning_rate=0.05`） | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——每次只修正残差的 5% | `0.01`、`0.05`、`0.1` |
| `num_leaves` | `int` | 每棵树的最大叶子数。`31`——Leaf-wise 生长的复杂度上限 | `15`、`31`、`63`、`127` |
| `max_depth` | `int` | 树的最大深度。`-1`——不限制深度，Leaf-wise 由 `num_leaves` 控制复杂度 | `-1`、`3`、`7` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 的训练样本 | `0.5`、`0.9`、`1.0` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征 | `0.5`、`0.9`、`1.0` |
| `random_state` | `int` | 随机种子，保证采样和训练可复现。`42` | `42` |
| 返回值 | `LGBMClassifier` | 已完成 `fit()` 的模型对象，含 `feature_importances_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.lightgbm import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。四分类任务下，y 取值 $\{0, 1, 2, 3\}$。
- `n_estimators=300` + `learning_rate=0.05` 是教学平衡选择——总修正量 $300 \times 0.05 = 15$，比 GBDT 的 $200 \times 0.1 = 20$ 更精细。
- `train_model(...)` 内部有 `try/except ImportError` 保护——`lightgbm` 不是 sklearn 标准组件，需要单独安装。

## 2. `LGBMClassifier` 构造器参数

### 参数速览

适用 API：`LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 弱学习器数量。默认 `300`——核心超参数，需与 `learning_rate` 配对 | `100`、`300`、`500` |
| `learning_rate` | `float` | 学习率（收缩因子）。`0.05`——越小越需更多树，但泛化更好 | `0.01`、`0.05`、`0.1` |
| `num_leaves` | `int` | 每棵树的最大叶子数。`31`——LightGBM 独有，替代 `max_depth` 控制复杂度 | `15`、`31`、`63` |
| `max_depth` | `int` | 树的最大深度。`-1`——不限制，Leaf-wise 生长由 `num_leaves` 控制 | `-1`、`3`、`7` |
| `subsample` | `float` | 行采样比例。`0.9`——每轮迭代随机保留 90% 样本 | `0.5`、`0.8`、`0.9` |
| `colsample_bytree` | `float` | 列采样比例。`0.9`——每棵树随机选择 90% 的特征 | `0.5`、`0.8`、`0.9` |
| `subsample_freq` | `int` | 行采样频率。默认 `0`（每轮都采样） | `0`、`5` |
| `reg_alpha` | `float` | L1 正则化系数。默认 `0.0` | `0.0`、`0.1` |
| `reg_lambda` | `float` | L2 正则化系数。默认 `0.0` | `0.0`、`0.1` |
| `min_child_samples` | `int` | 叶子节点的最小样本数。默认 `20` | `5`、`20`、`50` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| `n_jobs` | `int` | 并行线程数。`-1` 使用所有 CPU——直方图构建和特征扫描级并行，非基学习器级并行 | `-1`、`1`、`4` |
| `verbosity` | `int` | 日志详细程度。默认 `1` | `0`、`1` |

### 示例代码

```python
try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("请先 pip install lightgbm")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
```

### 理解重点

- `num_leaves` 是 LightGBM 最独特的参数——它替代 GBDT 的 `max_depth` 作为复杂度控制手段。Leaf-wise 生长从根节点开始，每次选择损失下降最多的叶子分裂，直至达到 `num_leaves`。
- `max_depth=-1` 不是"无限生长"——它表示不设置硬深度上限，但 `num_leaves=31` 已经限定了最大叶子数。
- `n_jobs=-1` 利用了 LightGBM 的并行机制——在直方图构建和特征扫描层面并行，而非基学习器级并行（Boosting 仍串行）。
- 与 GBDT（sklearn）的三个关键参数差异：（1）`num_leaves` 替代 `max_depth`；（2）多了 `colsample_bytree`；（3）`max_depth=-1` 而非 `max_depth=3`。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 说明 |
|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(20,)` | 20 个特征的重要性分数——基于分裂增益，和为 1.0 |
| `n_estimators_` | `int` | 实际训练的树数量——等于 `n_estimators=300` |
| `classes_` | `ndarray`，形状 `(4,)` | 类别标签——四分类 `[0, 1, 2, 3]` |
| `n_features_in_` | `int` | 特征维度——当前为 `20` |
| `best_iteration_` | `int` | 早停最优迭代轮次（当启用 `early_stopping_rounds` 时） |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"num_leaves: {num_leaves}")
print(f"max_depth: {max_depth}")
print(f"subsample: {subsample}")
print(f"colsample_bytree: {colsample_bytree}")
print(f"特征重要性 (前 5): {model.feature_importances_[:5]}")
```

### 理解重点

- `feature_importances_` 是 GBDT 系列共有的诊断属性——它提供"哪些特征在真正起作用"的自动诊断能力。
- 20 维特征中，预期 `x1`~`x8`（8 个有效特征）的重要性显著高于 `x9`~`x13`（5 个冗余特征）和 `x14`~`x20`（7 个纯噪声特征）。
- LightGBM 的 `feature_importances_` 默认使用 `gain`（分裂增益）——不同于 sklearn GBDT 默认的 impurity 下降量。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 20)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2, 3\}$ | 300 棵树加权累加 -> softmax -> argmax |
| `predict_proba(X)` | `array_like`，形状 `(n, 20)` | `ndarray`，形状 `(n, 4)` | 300 棵树加权累加 -> softmax——4 类概率和为 1.0 |

### 理解重点

- `predict()` 是加法模型输出——逐棵树累加（每棵乘以 `learning_rate`），取 softmax 概率最大的类别。
- `predict_proba()` 与 `predict()` 共享前段计算——最后一步分支（`predict` 取 argmax，`predict_proba` 返回概率分布）。
- 与 Bagging 的投票聚合不同——LightGBM 是加权累加，每棵树的权重由 `learning_rate` 隐式决定。

## 常见坑

1. `num_leaves` 设得过大（如 `num_leaves=1024`）——Leaf-wise 生长可能生成极深的树，导致单棵过拟合。
2. 把 `max_depth=-1` 当成"不设防"——`num_leaves` 才是真正的复杂度控制参数，但过深的分支仍可能只含极少数样本。
3. `learning_rate=0.05` 且 `n_estimators=100`——总修正量仅 5，可能欠拟合。
4. 混淆 LightGBM 的 `subsample` 与 Bagging 的 `max_samples`——LightGBM 是均匀随机采样（非 Bootstrap 有放回），且与 GOSS 的思想相通。
5. 忘记 `lightgbm` 是可选依赖——在新环境运行前必须 `pip install lightgbm`。

## 小结

- `train_model(...)` 是本仓库 LightGBM 的核心训练入口，是对 `lightgbm.LGBMClassifier` 的薄封装——含可选依赖检查。
- `LGBMClassifier` 的核心参数是 `n_estimators`（树数量）、`learning_rate`（学习率）、`num_leaves`（Leaf-wise 叶子数）、`subsample`（行采样）、`colsample_bytree`（列采样）——五者共同决定偏差缩减的程度和训练速度。
- 训练完成后的核心属性：`feature_importances_`（自动特征选择诊断）——20 维数据下区分有效/冗余/噪声特征的关键工具。

# 训练与预测

## 本章目标

1. 理解 `pipelines/ensemble/lightgbm.py` 的 `run()` 流水线——从数据加载到评估的完整 7 步流程。
2. 理解 LightGBM 的 `fit()` 训练过程与 GBDT 的本质区别——Leaf-wise 生长、直方图加速、行/列双重采样。
3. 理解 `predict()` 和 `predict_proba()` 的输出——多类别 softmax 聚合。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | 端到端流水线编排——7 步串联数据准备、标准化、训练、预测和三项评估 |
| `model.fit(X_train_s, y_train)` | 方法 | 串行训练 300 棵 Leaf-wise 直方图回归树——行/列双采样 + 直方图加速 |
| `model.predict(X_test_s)` | 方法 | 300 棵树加权累加 -> softmax -> argmax——输出类别 $\{0, 1, 2, 3\}$ |
| `model.predict_proba(X_test_s)` | 方法 | 300 棵树加权累加 -> softmax——输出 4 类概率分布 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集计算统计量，`transform` 在测试集使用相同统计量 |

## 1. 完整流水线流程

### 流程概述

```
lightgbm_data.copy()
  - 1 X = data.drop(columns=["label"]), y = data["label"]
  - 2 feature_names = list(X.columns)
  - 3 X_train, X_test, y_train, y_test = train_test_split(test_size=0.2, stratify=y)
  - 4 X_train_s = scaler.fit_transform(X_train), X_test_s = scaler.transform(X_test)
  - 5 model = train_model(X_train_s, y_train)  # 含 ImportError 检查
  - 6 y_pred = model.predict(X_test_s)
  - 7 三项可视化评估
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 复制数据 | `lightgbm_data.copy()` | 全局 `DataFrame` | 本地 `DataFrame`，`(1000, 21)` | 避免修改全局变量 |
| 分离 X/y | `data.drop(columns=["label"])` + `data["label"]` | `DataFrame` | `(DataFrame, Series)` | 特征 20 列 + 标签 1 列 |
| 提取特征名 | `list(X.columns)` | `DataFrame` | `list[str]` | 供特征重要性图表使用 |
| 切分数据 | `train_test_split(test_size=0.2, stratify=y)` | `(X, y)` | `(X_train, X_test, y_train, y_test)` | 800 训练 / 200 测试 |
| 标准化 | `scaler.fit_transform(X_train)` + `transform(X_test)` | `(DataFrame, DataFrame)` | `(ndarray, ndarray)` | 训练集计算 $\mu$/$\sigma$，测试集应用 |
| 训练 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `LGBMClassifier` | 300 棵直方图树串行训练——使用 `n_jobs=-1` 特征级并行 |
| 预测 | `model.predict(X_test_s)` | `ndarray`，`(200, 20)` | `ndarray`，`(200,)` | 类别 $\{0, 1, 2, 3\}$ |
| 概率输出 | `model.predict_proba(X_test_s)` | `ndarray`，`(200, 20)` | `ndarray`，`(200, 4)` | 每类 softmax 概率 |

### 理解重点

- 与 GBDT 流水线的关键差异：步骤 5 的训练内部使用了 `num_leaves`（非 `max_depth`）控制复杂度，训练速度显著更快。
- 与 Bagging 流水线的关键差异：步骤 2（`feature_names`）是 Boosting 系列独有的——GBDT 和 LightGBM 都有特征重要性评估。
- 所有集成分类流水线的数据准备部分完全一致——`copy()` -> 分离 X/y -> 分层切分 -> 训练集拟合标准化器。

## 2. 训练细节：`model.fit(X_train_s, y_train)`

LightGBM 的 `fit()` 在概念上与 GBDT 一致（加法模型 + 负梯度），但实现了多项工程优化。

### 训练过程（300 棵树串行）

1. **第 1 棵树**：在原始标签上训练——粗糙的初始分界面
2. **第 $m$ 棵树**（$m = 2, \dots, 300$）：拟合前 $m-1$ 棵树的负梯度（多类对数损失梯度），Leaf-wise 生长到最多 `num_leaves=31` 个叶子
3. **直方图分桶**：将 20 维连续特征离散化为 255 个 bins——分裂点搜索从 $O(N)$ 降到 $O(\#\text{bins})$
4. **行采样**：`subsample=0.9`——每轮随机保留 90% 训练样本
5. **列采样**：`colsample_bytree=0.9`——每棵树随机选择 18/20 个特征，增强树间多样性
6. **学习率收缩**：每棵树的输出乘以 `learning_rate=0.05`

### 参数速览

| 参数名 | 当前取值 | 训练中的作用 |
|---|---|---|
| `n_estimators` | `300` | 串行训练的弱学习器数量——步数更多但每步更小 |
| `learning_rate` | `0.05` | 每棵树输出的收缩乘数——防止单棵树修正过猛 |
| `num_leaves` | `31` | 每棵树的最大叶子数——Leaf-wise 生长的复杂度上限 |
| `max_depth` | `-1` | 不限制深度——Leaf-wise 生长由 `num_leaves` 控制复杂度 |
| `subsample` | `0.9` | 行采样比例——每轮迭代随机保留 90% 的训练样本 |
| `colsample_bytree` | `0.9` | 列采样比例——每棵树随机选择 90% 的特征 |
| `n_jobs` | `-1` | 直方图构建和特征扫描级并行——非基学习器级并行 |
| `random_state` | `42` | 保证采样和训练可复现 |

### 理解重点

- LightGBM 的训练**在概念上**仍是串行 Boosting——第 $m$ 棵树依赖前 $m-1$ 棵树的结果。
- **直方图加速**是 LightGBM 训练快于 sklearn GBDT 的核心原因——分割点搜索从排序复杂度降到桶查找。
- `n_jobs=-1` 在 LightGBM 中是安全的——它并行的是直方图构建和特征扫描，而非基学习器训练。

## 3. 预测细节

### `model.predict(X_test_s)` — 硬预测

```
300 棵树加权累加（每类一个分数）
    -> softmax（分数转为概率分布）
    -> argmax（概率最大的类）
    -> {0, 1, 2, 3}
```

### `model.predict_proba(X_test_s)` — 软预测

```
300 棵树加权累加（每类一个分数）
    -> softmax（分数转为概率分布）
    -> [p_0, p_1, p_2, p_3]，∑p = 1.0
```

### 参数速览

| 方法 | 输入形状 | 输出形状 | 输出含义 |
|---|---|---|---|
| `predict(X)` | `(n, 20)` | `(n,)` | 预测类别标号——$\{0, 1, 2, 3\}$ |
| `predict_proba(X)` | `(n, 20)` | `(n, 4)` | 每类 softmax 概率——4 列和恒为 1.0 |

### 理解重点

- LightGBM 的预测接口与 sklearn `GradientBoostingClassifier` 完全兼容——`predict` 返回类别，`predict_proba` 返回概率。
- 与 Bagging 的投票聚合不同——Bagging 用等权投票（每棵树一票），LightGBM 用加权累加（每棵树 x 学习率 + softmax）。
- `predict_proba` 在 LightGBM 中始终可用——流水线未使用 `hasattr` 条件检查。

## 4. 标准化：训练/测试分离

### 参数速览

| 操作 | 方法 | 数据 | 目的 |
|---|---|---|---|
| 训练集 | `scaler.fit_transform(X_train)` | 训练集 `(800, 20)` | 计算 $\mu_j$ 和 $\sigma_j$，同时变换 |
| 测试集 | `scaler.transform(X_test)` | 测试集 `(200, 20)` | 使用训练集的 $\mu_j$ 和 $\sigma_j$ 变换 |

### 理解重点

- `fit_transform` 在训练集上**同时**计算统计量和变换——一步完成。
- `transform` 在测试集上**只**应用变换——使用训练集的统计量，防止信息泄露。
- LightGBM 基于直方图的树本身对特征缩放不敏感——但标准化保持与 GBDT 和 Bagging 流水线的一致性。

## 常见坑

1. 在缺少 `lightgbm` 的环境中直接运行流水线——会抛出 `ImportError`，需先 `pip install lightgbm`。
2. 误以为 `max_depth=-1` 表示树可以无限深——`num_leaves=31` 已限制最大叶子数，但单叶可能很深。
3. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
4. 忽略 `stratify=y` 的重要性——四分类数据中某个类别可能在测试集中缺失。

## 小结

- LightGBM 流水线的 7 步流程与 Bagging/GBDT 共享相同的数据准备段（切分 + 标准化），差异在训练和评估阶段。
- `fit()` 的核心流程：串行训练 300 棵 Leaf-wise 直方图树，每棵树拟合前序负梯度，由 `num_leaves=31` 控制复杂度，行/列双采样增强多样性，`learning_rate=0.05` 收缩修正幅度。
- `predict()` 和 `predict_proba()` 的接口与 sklearn 完全兼容——加权累加 -> softmax -> 类别/概率。

# 评估与诊断

## 本章目标

1. 理解当前 LightGBM 流水线的三项评估输出——混淆矩阵、ROC 曲线、特征重要性。
2. 理解每项评估背后的诊断意图和观察重点。
3. 明确当前流水线**已实现**和**未实现**的评估项——以及未实现的原因。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 混淆矩阵 | 图表 | 4x4 热力图——显示每个真实类别被预测为各个类别的样本数，对角线为正确预测 |
| ROC 曲线 | 图表 | one-vs-rest 多分类 ROC——4 个类别各一条曲线 + macro/micro 平均 |
| 特征重要性 | 图表 | 20 个特征按分裂增益排序——区分有效特征（x1~x8）与冗余/噪声特征 |
| 终端日志 | 文本 | 训练完成时打印超参数和训练耗时——无准确率等汇总指标 |

## 1. 混淆矩阵

`plot_confusion_matrix(y_test, y_pred, title="LightGBM 混淆矩阵", ...)` 绘制测试集混淆矩阵。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(200,)` | 测试集真实标签 $\{0, 1, 2, 3\}$ | 来自 `train_test_split` |
| `y_pred` | `ndarray`，形状 `(200,)` | 模型预测标签——300 棵树加权累加后 argmax | `model.predict(X_test_s)` |
| `title` | `str` | 图表标题 | `"LightGBM 混淆矩阵"` |
| `dataset_name` | `str` | 数据集名称——决定输出路径 | `"lightgbm"` |
| `model_name` | `str` | 模型名称——决定输出路径 | `"lightgbm"` |

### 示例代码

```python
y_pred = model.predict(X_test_s)
plot_confusion_matrix(
    y_test, y_pred,
    title="LightGBM 混淆矩阵",
    dataset_name="lightgbm",
    model_name="lightgbm",
)
```

### 输出

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

### 理解重点

- 这是一个 4x4 的矩阵（四分类）——对角线越亮、非对角越暗，模型越好。
- 对于 `class_sep=0.6` 的数据，对角线通常有明显集中趋势——但非对角会有一定分散，因为类别间隔较小。
- 与 Bagging（2x2）和 GBDT（3x3）对比——LightGBM 的 4x4 矩阵反映更高维度的分类复杂度。

## 2. ROC 曲线

`plot_roc_curve(y_test, y_scores, ...)` 绘制多分类 ROC 曲线。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(200,)` | 测试集真实标签 $\{0, 1, 2, 3\}$ | 来自 `train_test_split` |
| `y_scores` | `ndarray`，形状 `(200, 4)` | 4 列概率输出——每列对应一个类别的 softmax 概率 | `model.predict_proba(X_test_s)` |
| `title` | `str` | 图表标题 | `"LightGBM ROC 曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"lightgbm"` |
| `model_name` | `str` | 模型名称 | `"lightgbm"` |

### 示例代码

```python
y_scores = model.predict_proba(X_test_s)
plot_roc_curve(
    y_test, y_scores,
    title="LightGBM ROC 曲线",
    dataset_name="lightgbm",
    model_name="lightgbm",
)
```

### 输出

![ROC 曲线](https://img.yumeko.site/file/blog/articles/1780736321096.png)

### 理解重点

- 多分类 ROC 使用 one-vs-rest 策略——每个类别作为"正类"，其余三类作为"负类"，分别画一条 ROC 曲线。
- 同时绘制 macro-average（各曲线等权平均）和 micro-average（全局 TP/FP 累加）。
- LightGBM 的 `predict_proba()` 与 sklearn 接口完全兼容——无需 `hasattr` 条件检查。

## 3. 特征重要性

`plot_feature_importance(model, feature_names=feature_names, ...)` 绘制特征重要性柱状图。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `LGBMClassifier` | 已训练的 LightGBM 模型——含 `feature_importances_` 属性 | `model` |
| `feature_names` | `list[str]` | 特征名列表 | `["x1", "x2", ..., "x20"]` |
| `title` | `str` | 图表标题 | `"LightGBM 特征重要性"` |
| `top_n` | `int` | 显示前 N 个重要特征。默认全部 | `10`、`20` |

### 示例代码

```python
feature_names = list(X.columns)
plot_feature_importance(
    model,
    feature_names=feature_names,
    title="LightGBM 特征重要性",
    dataset_name="lightgbm",
    model_name="lightgbm",
)
```

### 输出

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

### 理解重点

- LightGBM 的特征重要性基于**分裂增益**（`gain`）——每次树分裂时该特征带来的损失下降量累加。
- 预期观察：有效特征（`x1`~`x8`）的重要性显著高于冗余特征（`x9`~`x13`）和噪声特征（`x14`~`x20`）。
- 与 GBDT 的对比意义——在更高维度（20 vs 8）下，LightGBM 的特征重要性排序更稳定，因为列采样减少了单特征的过拟合倾向。

## 4. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 混淆矩阵 | 已实现 | 分类评估的基础指标——4x4 多分类热力图 |
| ROC 曲线 | 已实现 | LightGBM 始终支持 `predict_proba`——无需条件检查 |
| 特征重要性 | 已实现 | LightGBM 的 `feature_importances_` 属性——训练后自动可用 |
| 准确率/精确率/召回率/F1 打印 | **未实现** | 可从混淆矩阵直接计算——图表更直观 |
| 学习曲线 | **未实现** | GBDT 分册已展示该评估——LightGBM 不做重复诊断 |
| 决策边界可视化 | **未实现** | 数据为 20 维——无法在二维平面上直接绘制 |
| LightGBM vs GBDT 训练耗时对比 | **未实现** | 可在练习中手动对比——流水线保持简洁 |
| 交叉验证 | **未实现** | 当前专注于单次 split 的评估——留出法在教学场景下足够 |

### 理解重点

- LightGBM 的评估集比 GBDT 少一项（学习曲线），比 Bagging 多一项（特征重要性）。
- 决策边界无法绘制（20 维）——与 Bagging 的 2 维双月牙形成有趣的对比。
- 评估设计遵循"够用且不冗余"原则——已通过 GBDT 展示过的诊断手段不再重复。

## 5. LightGBM vs GBDT vs Bagging 评估对比

| 评估维度 | Bagging | GBDT | LightGBM | 差异原因 |
|---|---|---|---|---|
| 混淆矩阵 | 2x2（二分类） | 3x3（三分类） | 4x4（四分类） | 数据类别数递增 |
| ROC 曲线 | 条件可用 | 始终可用 | 始终可用 | `predict_proba` 支持情况 |
| 特征重要性 | 无 | 8 特征排序 | 20 特征排序 | LightGBM 维度最高 |
| 学习曲线 | 无 | 有 | 无 | LightGBM 教学精简 |
| OOB 得分 | 有 | 无 | 无 | 仅 Bagging 有 Bootstrap |
| 训练耗时日志 | 有 | 有 | 有 | 三者一致 |

## 常见坑

1. 期待 4x4 混淆矩阵像二分类一样简洁——多分类混淆矩阵有 16 个单元格，需逐类对比。
2. 把特征重要性排序当成"因果关系的证明"——重要性仅反映特征在模型中被使用的程度，不等于因果影响。
3. 忽略 4 个类别之间的结构性混淆——某些类别间的混淆可能是数据固有的。
4. 认为评估项越多越好——教学场景下冗余评估反而分散注意力。

## 小结

- LightGBM 当前有三项评估输出：混淆矩阵（4x4 多分类热力图）、ROC 曲线（one-vs-rest 4 类）、特征重要性（20 特征按增益排序）。
- 与 GBDT 对比：少一项学习曲线（教学精简），多 4 个分类维度和 12 个特征维度的评估挑战。
- 未实现的评估项（学习曲线、准确率打印、决策边界、交叉验证）有明确的教学设计考量——保持流水线聚焦 LightGBM 独有价值的诊断。

# 工程实现

## 本章目标

1. 理解 LightGBM 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径。
3. 理解 LightGBM 与 GBDT 在工程实现上的关键差异——可选依赖处理、`num_leaves` 替代 `max_depth`、无学习曲线、训练层有装饰器。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.lightgbm()` | 方法 | 生成高维多类别分类数据——`make_classification(n_features=20, n_classes=4)` |
| `train_model(...)` | 函数 | 构建并训练 `LGBMClassifier`——含可选依赖检查和装饰器（`@print_func_info`、`@timeit`） |
| `run()` | 函数 | 端到端流水线编排——串联数据准备、标准化、训练、预测和三项可视化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制测试集混淆矩阵（4x4 多分类热力图） |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 ROC 曲线（one-vs-rest） |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图（20 个特征排序） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` -> `data_generation/__init__.py` | 生成四分类高维数据并导出 `lightgbm_data` | 全局 `DataFrame`（1000 行 x 21 列） |
| 模型训练层 | `model_training/ensemble/lightgbm.py` | 封装 `LGBMClassifier` 训练——含 `ImportError` 处理 + 装饰器 | `LGBMClassifier` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/lightgbm.py` | 串联数据准备、标准化、训练、预测和三项评估——端到端入口 | 终端日志 + 调用三个可视化函数 |
| 可视化层 | `result_visualization/confusion_matrix.py`、`roc_curve.py`、`feature_importance.py` | 生成三项评估图表 | 3 个 PNG 文件 |

### 理解重点

- LightGBM 的可视化层与 GBDT 共享三个模块——但**没有** `learning_curve.py`，评估项比 GBDT 少一项。
- 训练层有三重保护：`try/except ImportError`（可选依赖）+ `@print_func_info`（调用日志）+ `@timeit`（耗时日志）。
- 与 GBDT 的核心工程差异：（1）可选依赖处理；（2）`num_leaves` 替代 `max_depth` 控制复杂度；（3）内部直方图加速 + `n_jobs=-1` 特征级并行。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `lightgbm_data.copy()` | — | `DataFrame`，形状 `(1000, 21)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["label"])` | `DataFrame` | `DataFrame`，形状 `(1000, 20)` | 分离 20 维特征 X |
| 3 | `data["label"]` | `DataFrame` | `Series`，形状 `(1000,)` | 分离四分类标签 y |
| 4 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 20 | 提取特征名——供特征重要性图表使用 |
| 5 | `train_test_split(X, y, test_size=0.2, stratify=y)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 分层训练/测试切分 |
| 6 | `scaler.fit_transform(X_train)` | `DataFrame`，形状 `(800, 20)` | `ndarray`，形状 `(800, 20)` | 训练集标准化 |
| 7 | `scaler.transform(X_test)` | `DataFrame`，形状 `(200, 20)` | `ndarray`，形状 `(200, 20)` | 测试集标准化 |
| 8 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `LGBMClassifier` | 训练 300 棵 Leaf-wise 直方图树 |
| 9 | `model.predict(X_test_s)` | `ndarray`，形状 `(200, 20)` | `ndarray`，形状 `(200,)` | 硬预测（加权累加 + softmax + argmax） |
| 10 | `plot_confusion_matrix(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 4x4 混淆矩阵可视化 |
| 11 | `model.predict_proba(X_test_s)` | `ndarray`，形状 `(200, 20)` | `ndarray`，形状 `(200, 4)` | 软概率输出（softmax） |
| 12 | `plot_roc_curve(y_test, y_scores, ...)` | `(Series, ndarray)` | PNG 文件 | 多分类 ROC 曲线 |
| 13 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 20 个特征重要性排序柱状图 |

### 理解重点

- 步骤 8 内部触发 `lightgbm` 可选依赖检查——如果未安装会抛出 `ImportError`。
- 步骤 9-12 与 GBDT 完全一致——说明 LightGBM 的 scikit-learn 兼容接口与 `GradientBoostingClassifier` 的方法签名一致（`predict`/`predict_proba`）。
- 与 GBDT 对比：LightGBM 流水线少了一步（无学习曲线），但数据规模更大（1000 样本 vs 500 样本）。

## 3. 数据依赖关系

```
lightgbm_data (全局 DataFrame)
  - -> X = data.drop(columns=["label"])  ──-> feature_names = list(X.columns) ──┐
  - -> y = data["label"]                                                        │
  - -> train_test_split(X, y, test_size=0.2, stratify=y)                        │
    - -> X_train (800, 20) ──-> scaler.fit_transform() ──-> X_train_s ──┐      │
    - -> y_train (800,) ─────────────────────────────────────────────┤      │
    - -> X_test (200, 20) ──-> scaler.transform() ──-> X_test_s ──┐    │      │
    - -> y_test (200,) ─────────────────────────────────┐       │    │      │
    - train_model(X_train_s, y_train) ──-> model               │    │      │
      - -> model.predict(X_test_s) ──-> y_pred ──┐          │    │      │
      - -> model.predict_proba(X_test_s) ──-> y_scores ──┐  │    │      │
      - -> model.feature_importances_ ──-> + feature_names ──┼──┼──────┘  │
      - plot_confusion_matrix(y_test, y_pred, ...) <-───────┘  │          │
      - plot_roc_curve(y_test, y_scores, ...) <-───────────────┘          │
      - plot_feature_importance(model, feature_names, ...) <-─────────────┘
```

### 理解重点

- 数据流比 GBDT 少一个 `plot_learning_curve` 分支——结构更简洁。
- `y_train` 仅参与训练——与 GBDT 不同（GBDT 的 `y_train` 还参与学习曲线）。
- 特征重要性依赖 `model.feature_importances_` 和 `feature_names`——两个数据来自不同阶段，在可视化层交汇。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 混淆矩阵 | `outputs/lightgbm/confusion_matrix.png` | PNG | 测试集 4x4 混淆矩阵热力图 |
| ROC 曲线 | `outputs/lightgbm/roc_curve.png` | PNG | 多分类 ROC（one-vs-rest，4 类） |
| 特征重要性 | `outputs/lightgbm/feature_importance.png` | PNG | 20 个特征的重要性排序柱状图 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.lightgbm
```

### 输出

```text
============================================================
LightGBM 分类流水线
============================================================
模型训练完成
n_estimators: 300
learning_rate: 0.05
num_leaves: 31
max_depth: -1
subsample: 0.9
colsample_bytree: 0.9
模型训练耗时: 0.43s

============================================================
LightGBM 流水线完成！
============================================================
```

### 理解重点

- LightGBM 输出 3 个 PNG 文件——比 GBDT 少 1 个（无学习曲线），比 Bagging 多 1 个（有特征重要性）。
- 训练耗时通常显著短于 GBDT（~0.4s vs ~2s）——直方图加速 + `n_jobs=-1` 特征级并行。
- 终端日志多了 `num_leaves`、`subsample`、`colsample_bytree`——反映 LightGBM 更细粒度的控制参数。

## 5. 训练层细节：与 GBDT 的对比

| 工程维度 | GBDT (sklearn) | LightGBM |
|---|---|---|
| 依赖 | sklearn 内置——无需额外安装 | 可选依赖——`try/except ImportError` |
| 基学习器 | 内部自动 `DecisionTreeRegressor` | 内部直方图回归树（Leaf-wise 生长） |
| 复杂度控制 | `max_depth=3`（深度限制） | `num_leaves=31` + `max_depth=-1`（叶子数限制） |
| 树数量 | 200 | 300 |
| 学习率 | 0.1 | 0.05 |
| 行采样 | `subsample=1.0`（不使用） | `subsample=0.9`（行采样 + GOSS 思想） |
| 列采样 | 无 | `colsample_bytree=0.9`（列采样） |
| 并行 | 无（纯串行） | `n_jobs=-1`（直方图构建和特征扫描级并行） |
| 日志 | `n_estimators, learning_rate, max_depth, subsample` | `n_estimators, learning_rate, num_leaves, max_depth, subsample, colsample_bytree` |
| 装饰器 | `timer` | `@print_func_info` + `@timeit` + `timer` |
| 训练耗时 | ~2s | ~0.4s |

### 理解重点

- LightGBM 的 `max_depth=-1` 不是"无限深度"——Leaf-wise 生长下，复杂度由 `num_leaves` 控制，`max_depth=-1` 表示不额外限制最大深度。
- `n_estimators=300` + `learning_rate=0.05` 的总修正量（15）小于 GBDT 的 `200 x 0.1 = 20`——但步子更多更稳。
- LightGBM 的列采样（`colsample_bytree=0.9`）是 sklearn GBDT 不支持的——这是微软实现的独有正则化手段。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解 `lightgbm()` 的数据生成逻辑（高维四分类）
2. `model_training/ensemble/lightgbm.py` — 理解 `LGBMClassifier` 的构建、可选依赖和 Leaf-wise 训练
3. `pipelines/ensemble/lightgbm.py` — 看清端到端流程和三项评估的串联
4. `result_visualization/confusion_matrix.py` — 了解混淆矩阵实现
5. `result_visualization/roc_curve.py` — 了解多分类 ROC 实现
6. `result_visualization/feature_importance.py` — 了解特征重要性图表实现

## 常见坑

1. 在不含 `lightgbm` 的环境中直接 `from model_training.ensemble.lightgbm import train_model`——会触发 `ImportError`，需先 `pip install lightgbm`。
2. 把 `num_leaves=31` 和 `max_depth=-1` 当成"树可以无限深"——叶子数限制实际上限定了树复杂度。
3. 直接修改 `lightgbm_data` 而不先 `copy()`——会污染其他模块引用的同一全局变量。
4. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。

## 小结

- LightGBM 工程实现遵循本仓库标准四层架构：数据生成层 -> 模型训练层 -> 流水线编排层 -> 可视化层（含 3 个模块）。
- `run()` 是薄编排函数——13 步调用串联数据准备、标准化、训练、预测和三项评估。
- 与 GBDT 的四个关键工程差异：（1）可选依赖 `try/except`；（2）`num_leaves` 替代 `max_depth`；（3）多了 `colsample_bytree` 列采样；（4）少一项评估（无学习曲线）。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 LightGBM 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 LightGBM 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Leaf-wise 生长、GOSS、EFB、直方图、LightGBM vs GBDT 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 LightGBM 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 LightGBM 原始论文、官方文档和扩展阅读 |

## 1. 自检问题

1. LightGBM 的 Leaf-wise 生长策略与 GBDT 的 Level-wise 生长策略有何本质区别？Leaf-wise 在同等叶子数下为何损失下降更快？

2. 直方图算法如何将连续特征离散化？离散化带来的速度收益和可能的精度损失分别是什么？

3. GOSS（Gradient-based One-Side Sampling）的采样策略与简单的随机子采样（`subsample`）有何本质不同？为什么保留大梯度样本比均匀随机采样更高效？

4. 为什么 LightGBM 使用 `num_leaves` 而非 `max_depth` 来控制树复杂度？在 `max_depth=-1` 的情况下，`num_leaves=31` 的树可能有多深？

5. EFB（Exclusive Feature Bundling）在什么场景下最有效？在当前稠密 20 维数据上，EFB 的收益如何？

6. LightGBM 与 sklearn GBDT 在训练速度、参数体系、默认配置上的核心差异有哪些？

7. 当前 LightGBM 流水线中 `n_jobs=-1` 的并行发生在哪个层面？为什么 Boosting 算法仍能利用多核并行？

## 2. 动手练习

### 练习 1：改变叶子数 `num_leaves`

将 `num_leaves` 分别设为 `7`、`15`、`31`、`63`、`127`，观察特征重要性和混淆矩阵的变化。

```python
model = train_model(X_train_s, y_train, num_leaves=15)
```

回答：`num_leaves` 增大后，模型复杂度如何变化？`num_leaves=127` 在 800 个训练样本上是否会过拟合？

### 练习 2：改变学习率 `learning_rate`

将 `learning_rate` 分别设为 `0.01`、`0.02`、`0.05`、`0.1`、`0.2`，同时保持 `n_estimators=300`，观察混淆矩阵。

```python
model = train_model(X_train_s, y_train, learning_rate=0.01)
```

回答：`learning_rate=0.01` 且 `n_estimators=300` 时，模型是否欠拟合？你需要增加多少棵树来匹配较小学习率？

### 练习 3：改变列采样 `colsample_bytree`

将 `colsample_bytree` 分别设为 `0.3`、`0.5`、`0.7`、`0.9`、`1.0`，观察特征重要性的变化。

```python
model = train_model(X_train_s, y_train, colsample_bytree=0.3)
```

回答：`colsample_bytree` 减小后，特征重要性排序是否发生明显变化？这种变化对理解"哪些特征重要"有什么影响？

### 练习 4：对比 GBDT 训练速度

使用相同数据，分别训练 sklearn GBDT 和 LightGBM，对比训练耗时。

```python
from model_training.ensemble.gbdt import train_model as train_gbdt
from model_training.ensemble.lightgbm import train_model as train_lgbm

# 使用相同数据（需要调整维度匹配）
model_gbdt = train_gbdt(X_train_s, y_train, n_estimators=200, learning_rate=0.1, max_depth=3)
model_lgbm = train_lgbm(X_train_s, y_train, n_estimators=300, learning_rate=0.05, num_leaves=31)
```

回答：在 1000 样本 x 20 特征的数据上，LightGBM 比 GBDT 快多少？随着数据规模增大，这个差距如何变化？

### 练习 5：改变数据规模观察速度优势

修改 `data_generation/ensemble.py` 中的 `n_samples` 参数（分别设为 `500`、`1000`、`2000`、`5000`），重新运行 LightGBM 和 GBDT 流水线。

```python
# 在 data_generation/ensemble.py 的 __init__ 中
class EnsembleData:
    n_samples: int = 5000  # 试试 500, 1000, 2000, 5000
```

回答：数据规模从 500 增加到 5000 时，LightGBM 相对于 GBDT 的速度倍数如何变化？这验证了直方图算法的什么性质？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017. | LightGBM 原始论文——GOSS、EFB 和 Leaf-wise 生长的完整推导和实验验证 |
| 2 | LightGBM 官方文档 — [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) | 全部参数、加速技巧和调参指南的权威参考 |
| 3 | scikit-learn 兼容接口 — [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) | scikit-learn 兼容接口的 API 参考——与 sklearn 无缝集成 |
| 4 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232. | GBDT 的理论基础——LightGBM 在此数学框架上进行工程优化 |

## 常见坑

1. 把 `num_leaves=31` 和 `max_depth=-1` 当成"不受控的完全生长"——`num_leaves` 是实际复杂度控制参数，`max_depth=-1` 只是不另设上限。
2. 在新环境中忘记 `lightgbm` 是可选依赖——导入前需 `try/except`，在未安装环境运行会抛出 `ImportError`。
3. 把 LightGBM 的 `n_jobs=-1` 与 Bagging 的并行等同——LightGBM 的并行在直方图构建和特征扫描层面，而非基学习器级。
4. 在极小数据（<100 样本）上使用 LightGBM——直方图离散化损失可能超过精度收益。

## 小结

- 7 个自检问题覆盖 LightGBM 的核心创新：Leaf-wise vs Level-wise、直方图算法、GOSS vs 随机采样、`num_leaves` vs `max_depth`、EFB、与 sklearn GBDT 对比、并行机制。
- 5 个动手练习从不同角度探索 LightGBM 的行为——改变叶子数、学习率、列采样、对比 GBDT 速度、改变数据规模。
- 4 篇参考文献从原始论文（Ke et al. 2017）-> 官方文档 -> API 参考 -> GBDT 理论基础构成完整的阅读路线。
