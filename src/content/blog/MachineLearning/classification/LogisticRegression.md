---
title: LogisticRegression 逻辑回归分类
date: 2026-04-13
category: MachineLearning/Classification
tags:
  - Scikit-learn
  - 基础
description: 逻辑回归分类的数学原理、参数解析与完整实现流程。
image: https://img.yumeko.site/file/blog/LogisticRegression.png
status: published
---


# 数学原理

## 本章目标

1. 理解逻辑回归为什么虽然叫"回归"，本质上却是通过 Sigmoid 做概率输出的分类模型。
2. 理解线性得分、Sigmoid、对数几率、交叉熵损失和梯度在当前实现中的数学角色。
3. 理解正则化机制与 `C` 参数（$\lambda = 1/C$）的关系——`C` 越大正则越弱。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性得分 $z = \mathbf{w}^T\mathbf{x} + b$ | 模型部分 | 先对输入特征做线性加权求和，是 Sigmoid 的输入 |
| Sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$ | 概率映射 | 把线性得分 $z \in (-\infty, +\infty)$ 压缩到 $(0, 1)$ 概率区间 |
| 对数几率 $\ln\frac{P}{1-P}$ | 解释方式 | 逻辑回归对对数几率建模为 $\mathbf{w}^T\mathbf{x} + b$，使得概率比取对数后呈线性 |
| 交叉熵损失 $\mathcal{L}$ | 优化目标 | 衡量预测概率与真实标签的差异，最小化等价于极大似然估计 |
| 梯度 $\nabla_{\mathbf{w}}\mathcal{L}$ | 优化信息 | 决定参数 $\mathbf{w}$ 沿哪个方向更新能最快降低损失 |
| 正则化与 `C` | 超参数机制 | $C$ 是正则化强度 $\lambda$ 的倒数——$C$ 越小正则越强，系数越收缩 |

## 1. 逻辑回归的核心思想

逻辑回归先计算一个线性得分，再把这个得分通过 Sigmoid 压缩成概率输出。因此它虽然名字里有"回归"，最终目标是做分类概率估计。

### 线性部分

$$
z = \mathbf{w}^T\mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b
$$

### Sigmoid 函数

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Sigmoid 的性质：
- $z \to +\infty$ 时 $\sigma(z) \to 1$（十分确信正类）
- $z = 0$ 时 $\sigma(z) = 0.5$（最不确定）
- $z \to -\infty$ 时 $\sigma(z) \to 0$（十分确信负类）

### 概率输出

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}
$$

### 理解重点

- 当前模型不是直接输出"正类/负类"，而是先输出正类概率。
- 这也是为什么当前流水线可以直接调用 `predict_proba(...)` 来绘制 ROC 曲线。
- 逻辑回归的核心优势：分类结果和概率解释天然结合——输出不只是标签，还有置信度。

## 2. 对数几率与线性决策边界

逻辑回归对对数几率（log-odds）建模为线性函数：

$$
\ln \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}^T\mathbf{x} + b = z
$$

当 $P(y=1) = P(y=0) = 0.5$ 时，对数几率为 0，由此得到决策边界：

$$
\mathbf{w}^T\mathbf{x} + b = 0
$$

这是一个 $d$ 维空间中的超平面，$\mathbf{w}$ 是法向量，$b$ 控制超平面的偏移。

### 理解重点

- 决策边界 $\mathbf{w}^T\mathbf{x} + b = 0$ 是一张平坦的超平面——这就是"线性"的来源。
- $w_j > 0$ 意味着特征 $x_j$ 增大时会推高正类概率；$w_j < 0$ 则压低正类概率。
- 这也是为什么 `coef_` 和 `intercept_` 在逻辑回归里很有解释价值——它们直接描述了边界的位置和方向。

## 3. 极大似然与交叉熵损失

对训练集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，假设样本独立，似然函数为：

$$
L(\mathbf{w}, b) = \prod_{i=1}^{N} \hat{p}_i^{\,y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

其中 $\hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$。取负对数并除以 $N$ 后，得到交叉熵损失（对数损失）：

$$
\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Big]
$$

### 理解重点

- 逻辑回归不是靠最小二乘（MSE），而是靠极大似然 / 交叉熵目标来训练——这使它更适合概率建模。
- 与 MSE 不同，交叉熵对概率接近 0 或 1 时的错误预测惩罚很大（$\ln \hat{p} \to -\infty$ 当 $\hat{p} \to 0$），迫使模型给出更可信的概率估计。
- 当前数学章节应明确这一点，避免和线性回归的损失函数混淆。

## 4. 梯度推导

交叉熵损失对权重 $w_j$ 的偏导数形式非常简洁：

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i) \, x_{ij}
$$

向量形式：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{p}} - \mathbf{y}), \quad
\nabla_b \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)
$$

### 理解重点

- 梯度形式非常直观：预测概率与真实标签的误差 $(\hat{p}_i - y_i)$，乘上对应特征值 $x_{ij}$，汇总起来更新参数。
- 可以把它理解为"当前模型在哪些方向上高估或低估了正类概率"——误差大的方向更新靠前。
- 当前源码虽然没有手写梯度下降（使用 `lbfgs` 优化器封装），但理解梯度形式有助于理解优化行为。

## 5. 正则化与 `C` 参数

当前训练代码默认使用 L2 正则化。加入 L2 正则化后的完整损失函数为：

$$
\mathcal{L}_{\text{reg}}(\mathbf{w}, b) = \mathcal{L}(\mathbf{w}, b) + \frac{1}{2C} \|\mathbf{w}\|_2^2
$$

其中 $\|\mathbf{w}\|_2^2 = \sum_{j=1}^{d} w_j^2$。

**关键关系**：$\lambda = 1/C$，其中 $\lambda$ 是传统正则化强度系数，$C$ 是 sklearn 使用的正则化强度倒数。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` 或 `None` | 正则化类型。`"l2"` 对 $\|\mathbf{w}\|_2^2$ 惩罚，系数趋于均匀收缩；`"l1"` 对 $\|\mathbf{w}\|_1$ 惩罚，产生稀疏解；`"elasticnet"` 为两者混合；`None` 不做正则化。默认为 `"l2"` | `"l2"`、`"l1"`、`"elasticnet"`、`None` |
| `C` | `float` | 正则化强度的倒数，数学上 $\lambda = 1/C$。$C$ 越大 → 正则越弱 → 模型越自由；$C$ 越小 → 正则越强 → 系数越收缩趋于 0。默认为 `1.0` | `0.01`、`1.0`、`100.0` |
| `l1_ratio` | `float` | L1 正则化在 elasticnet 中的混合比例。仅当 `penalty='elasticnet'` 时生效。$\text{penalty} = \rho \|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2$。默认为 `None` | `0.0`、`0.5`、`1.0` |

### 理解重点

- `C` 是正则化强度的倒数——这是当前逻辑回归分册最容易写反的地方，文档必须明确。
- $C \to 0$（$\lambda \to \infty$）：强正则，系数趋近于 0，模型趋近于常数预测（仅剩截距起作用）。
- $C \to \infty$（$\lambda \to 0$）：弱正则，系数自由增长，容易过拟合。
- 当前默认 `C=1.0`、`penalty='l2'`，是 sklearn 的保守默认值。

## 6. 为什么标准化会影响训练与解释

逻辑回归使用梯度优化器（`lbfgs`）最小化交叉熵损失，特征尺度差异会导致：

1. 不同维度的梯度量级差异巨大——优化器收敛困难
2. 正则化惩罚不均匀——大值特征的系数被过度惩罚
3. `coef_` 之间不可直接比较——无法判断哪个特征更重要

标准化 $x_i' = (x_i - \mu_i) / \sigma_i$ 后，所有特征均值为 0、标准差为 1，以上问题全部消除。

### 理解重点

- 标准化对逻辑回归是有实益的——不是可有可无的工程惯性。
- 标准化后 $w_j$ 的大小可以粗略反映特征 $j$ 的相对重要性（因为各特征尺度统一）。
- 这也是当前流水线必须在训练前执行 `StandardScaler` 的原因。

## 7. 多分类扩展：Softmax

逻辑回归可以自然扩展到多分类（Softmax 回归 / 多项逻辑回归）。对 $K$ 个类别，每个类别有自己的权重向量 $\mathbf{w}_k$：

$$
P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}
$$

二分类退化为 Sigmoid：当 $K=2$ 时，Softmax 等价于 Sigmoid。

### 理解重点

- 当前数学章节保留此扩展以建立完整视角。
- 当前工程实现使用的是二分类数据和二分类逻辑回归，因此数据、模型、训练、评估章节都应聚焦二分类场景。
- 如果将来需要多分类，sklearn 的 `LogisticRegression` 已原生支持（通过 `multi_class='multinomial'`）。

## 8. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 线性得分 | $z = \mathbf{w}^T\mathbf{x} + b$ | `model.decision_function(X)` |
| Sigmoid 概率 | $\sigma(z) = 1/(1+e^{-z})$ | `model.predict_proba(X)[:, 1]` |
| 决策边界 | $\mathbf{w}^T\mathbf{x} + b = 0$ | `model.coef_` × `X` + `model.intercept_` = 0 |
| 权重系数 | $\mathbf{w} \in \mathbb{R}^d$ | `model.coef_` |
| 截距 | $b \in \mathbb{R}$ | `model.intercept_` |
| 交叉熵损失 | $\mathcal{L} = -\frac{1}{N}\sum[y_i\ln\hat{p}_i + (1-y_i)\ln(1-\hat{p}_i)]$ | `solver='lbfgs'` 内部优化目标 |
| L2 正则化 | $\frac{1}{2C}\|\mathbf{w}\|_2^2$ | `penalty='l2'`，`C=1.0` |
| 正则化倒数 | $\lambda = 1/C$ | `C=1.0` → $\lambda = 1.0$ |
| 优化器 | — | `solver='lbfgs'` |

## 常见坑

1. 把逻辑回归误当成线性回归加阈值——本质上是线性得分 + Sigmoid 概率映射 + 交叉熵优化。
2. 忽略交叉熵与极大似然的等价关系——最小化交叉熵 = 最大化对数似然。
3. 把 `C` 的含义写反——`C` 是正则化强度的倒数，$C$ 越大正则越弱，不是"正则化系数"。
4. 忽略标准化对优化和系数解释的影响——梯度优化器对特征尺度敏感，不标准化会导致收敛困难和系数不可比。

## 小结

- 逻辑回归的核心数学链：线性得分 $z = \mathbf{w}^T\mathbf{x} + b$ → Sigmoid 概率 $\sigma(z)$ → 交叉熵损失 $\mathcal{L}$ → 梯度下降优化 → 正则化控制复杂度。
- `coef_` 与 `intercept_` 直接决定线性决策边界 $\mathbf{w}^T\mathbf{x} + b = 0$ 的位置——$w_j > 0$ 推高正类概率，$w_j < 0$ 压低正类概率。
- `C` 是 $\lambda$ 的倒数（$\lambda = 1/C$），$C$ 越大正则越弱——这个方向是当前文档必须反复强调的重点。
- 当前源码默认使用 L2 正则化 + `lbfgs` 优化器的二分类逻辑回归，与当前高维近线性可分数据高度匹配。

# 数据构成

## 本章目标

1. 明确本仓库 Logistic Regression 数据来自 `ClassificationData.logistic_regression()` 的生成逻辑。
2. 明确 `make_classification` 各参数的数据含义与当前取值。
3. 明确训练集/测试集切分与标准化的顺序和边界——逻辑回归基于梯度优化，标准化直接影响收敛和系数可比性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClassificationData.logistic_regression()` | 方法 | 生成逻辑回归使用的高维二分类数据 |
| `make_classification(...)` | 函数 | scikit-learn 提供的监督分类数据生成器，可控制特征类型与噪声 |
| `logistic_regression_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前流水线中的监督分类标签，取值 $\{0, 1\}$ |
| `StandardScaler` | 类 | 对特征做标准化，改善梯度优化收敛与系数可比性 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `logistic_regression_data`
- 生成来源：`data_generation/classification.py` 中的 `ClassificationData.logistic_regression()`
- 流水线使用：`pipelines/classification/logistic_regression.py` 中的 `data = logistic_regression_data.copy()`

### 理解重点

- `logistic_regression_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的是避免后续处理意外修改原始数据对象。
- 当前数据是为逻辑回归教学场景专门构造的高维二分类数据，与线性分类边界假设高度匹配。

## 2. 数据生成函数 `ClassificationData.logistic_regression()`

底层调用 `sklearn.datasets.make_classification`，生成包含有效特征、冗余特征和标签噪声的高维二分类数据。

### 参数速览

适用函数：`sklearn.datasets.make_classification`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数 $N$。当前取 `400` | `100`、`400`、`1000` |
| `n_features` | `int` | 总特征数 $d$。当前取 `6`，包含有效、冗余和重复特征 | `2`、`6`、`20` |
| `n_informative` | `int` | 有效特征数——对类别有真实区分力的特征数量。当前取 `3`，即 6 个特征中只有 3 个真正携带分类信息 | `2`、`3`、`5` |
| `n_redundant` | `int` | 冗余特征数——由有效特征线性组合生成的随机线性组合。当前取 `1`，用于观察模型在冗余信息上的表现 | `0`、`1`、`3` |
| `n_repeated` | `int` | 重复特征数——从有效和冗余特征中随机复制。当前取 `0` | `0`、`1` |
| `n_classes` | `int` | 类别数 $K$。当前取 `2`（二分类） | `2`、`3` |
| `n_clusters_per_class` | `int` | 每个类别的簇数。默认为 `2`，影响类内数据分布形态 | `1`、`2` |
| `class_sep` | `float` | 类别间分离程度。值越大类别越容易区分。当前取 `1.2`——近线性可分但不完美，适合展示逻辑回归在中等难度数据上的行为 | `0.5`、`1.0`、`1.5` |
| `flip_y` | `float` | 标签噪声比例。随机翻转该比例样本的标签。当前取 `0.03`（3%）——模拟少数误标样本 | `0.0`、`0.03`、`0.1` |
| `random_state` | `int` | 随机种子，保证每次生成相同数据。当前取 `42` | `42` |
| `shuffle` | `bool` | 是否打乱样本顺序。默认为 `True` | `True` |

### 示例代码

```python
from sklearn.datasets import make_classification
from pandas import DataFrame

X, y = make_classification(
    n_samples=400,
    n_features=6,
    n_informative=3,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    class_sep=1.2,
    flip_y=0.03,
    random_state=42,
)
columns = [f"x{i + 1}" for i in range(6)]
data = DataFrame(X, columns=columns)
data["label"] = y
```

### 理解重点

- 当前数据是高维二分类数据（$d=6$），不是二维玩具问题——无法直接可视化原始空间中的决策边界，需要 PCA 降维。
- `n_informative=3`、`n_redundant=1` 意味着：6 个特征中 3 个真正有用，1 个是冗余的线性组合——适合展示逻辑回归对冗余特征的容忍度。
- `class_sep=1.2` 使得数据近线性可分但不完美——逻辑回归能找到合理的边界，但无法达到完美精度。
- `flip_y=0.03` 模拟 3% 的标签噪声，展示逻辑回归在轻微噪声下的鲁棒性。

## 3. 特征列与标签列

当前数据表结构：

- 特征列：`x1` ~ `x6`（6 维实数特征）
- 标签列：`label`（二分类标签，取值为 $0$ 或 $1$）

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `label` 是监督训练标签，会真实参与 `model.fit(X_train, y_train)`。
- 当前任务为二分类——逻辑回归的 Sigmoid 输出天然适合二分类概率建模。
- 6 维特征意味着原始空间中的决策边界是 5 维超平面——无法直接可视化，需借助 PCA。

## 4. 切分与标准化的顺序

标准化必须在切分之后执行——`fit_transform` 在训练集上计算 $\mu_i, \sigma_i$，`transform` 将相同统计量应用于测试集。顺序错误会导致数据泄露。

### 参数速览

适用函数：`train_test_split`、`StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比。当前取 `0.2`，即 80 个测试样本（总样本 400 × 20%） | `0.2`、`0.3` |
| `random_state` | `int` | 随机种子，保证每次切分结果一致。当前取 `42` | `42` |
| `stratify` | `array_like` | 按 `y` 的类别比例分层抽样。数学上保证 $\frac{n_{0,\text{train}}}{n_{0,\text{test}}} \approx \frac{N_{\text{train}}}{N_{\text{test}}}$，尤其重要因为 `flip_y` 可能让类别比例略有偏移 | `y`、`None` |
| `scaler.fit_transform(X_train)` | 方法 | 在训练集上计算 $\mu_i, \sigma_i$ 并变换：$x_i' = (x_i - \mu_i) / \sigma_i$ | — |
| `scaler.transform(X_test)` | 方法 | 使用训练集的 $\mu_i, \sigma_i$ 变换测试集，不重新计算统计量 | — |

### 示例代码

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 先切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 再标准化（仅在训练集上 fit）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 对逻辑回归来说，标准化有三个好处：梯度优化器（`lbfgs`）收敛稳定、正则化惩罚均匀、`coef_` 之间可直接比较大小。
- `stratify=y` 确保训练集和测试集的类别比例一致——在标签噪声存在时这尤其重要。
- 标准化后 $w_j \approx 0$ 的特征基本没有贡献，$w_j$ 绝对值大的特征对正类倾向影响强。

## 数据可视化

![类别分布](https://img.yumeko.site/file/articles/ML/logistic_regression/data_class_distribution.png)

![相关性热力图](https://img.yumeko.site/file/articles/ML/logistic_regression/data_correlation.png)

![特征空间二维投影](https://img.yumeko.site/file/articles/ML/logistic_regression/data_feature_space_2d.png)

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来。
2. 在切分之前就对全量数据做标准化——造成数据泄露，验证结果不可信。
3. 忽略 `stratify=y`，导致训练集和测试集类别比例不稳定。
4. 只看到"逻辑回归是线性模型"，却忽略当前数据中仍有 3 个冗余特征和 3% 的标签噪声——不是完美线性可分。

## 小结

- 当前 Logistic Regression 数据来自 `ClassificationData.logistic_regression()`，底层使用 `make_classification(n_samples=400, n_features=6, n_informative=3, n_redundant=1, class_sep=1.2, flip_y=0.03)`。
- 数据表结构：`x1` ~ `x6` 是 6 维特征，`label` 是二分类监督标签。
- 数据特点：高维、含冗余特征、近线性可分但不完美——适合展示逻辑回归在真实场景下的线性概率分类能力。
- 标准化对逻辑回归影响深远——不仅关乎收敛速度，还直接影响系数解释和正则化效果。

# 思路与直觉

## 本章目标

1. 用直观方式理解逻辑回归到底在做什么——线性打分 → Sigmoid → 概率输出。
2. 理解为什么它在当前高维近线性可分数据上是合理的概率分类基线。
3. 理解它与 KNN（局部投票）、决策树（轴对齐切分）、SVC（最大间隔）在思路上的关键差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 线性打分 | 核心直觉 | 先给样本一个线性分值 $z = \mathbf{w}^T\mathbf{x} + b$，分值越高越倾向正类 |
| Sigmoid 概率 | 输出形式 | 把无界分值 $z \in (-\infty, +\infty)$ 压缩成 $(0, 1)$ 的概率 |
| 线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$ | 决策形状 | 在 $\sigma(z) = 0.5$ 处形成一张平坦超平面 |
| 系数解释 | 可解释性 | $\mathbf{w}$ 的正负和大小直接反映特征对正类倾向的方向和强度 |
| KNN | 对比算法 | 局部邻域投票，边界贴合数据局部结构 |
| 决策树 | 对比算法 | 递归轴对齐切分，形成分段常数区域 |
| SVC | 对比算法 | 最大间隔分类，追求边界到样本的最小距离最大化 |

## 1. 为什么需要逻辑回归

对于很多分类问题，我们不只想要"分成两类"，还希望模型能告诉我们：

它认为样本属于正类的概率大概有多大？

逻辑回归的思路：

- 先对特征做线性加权求和：$z = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$
- 再把结果通过 Sigmoid 映射成概率：$\sigma(z) = \frac{1}{1+e^{-z}}$
- 得分越高 → 概率越接近 1；得分越低 → 概率越接近 0；$z=0$ → 概率正好 0.5

### 理解重点

- 逻辑回归天然兼具"分类能力"和"概率解释"——输出不只是标签，还有置信度。
- 当前流水线的 `predict_proba(...)`、ROC 曲线，都建立在概率输出之上。
- 与 KNN 的"看周围邻居"不同，逻辑回归的预测由一个全局公式 $\sigma(\mathbf{w}^T\mathbf{x} + b)$ 给出。

## 2. 为什么当前仓库示例里它表现合理

当前逻辑回归数据来自 `make_classification(n_features=6, n_informative=3, class_sep=1.2, flip_y=0.03)`，特点是：

- 二分类任务
- 6 维特征（含有效和冗余）
- 近线性可分（`class_sep=1.2`）
- 含少量标签噪声（`flip_y=0.03`）

### 理解重点

- 数据整体呈近线性可分结构——一条超平面能大致分开两类，虽不能完美（因为有噪声和冗余）。
- 对逻辑回归来说，这是很合适的教学场景：既能展示线性边界的清晰性，又不会因数据过于理想而失去现实感。
- 与 KNN 的双月牙数据（非线性弧形）和决策树的 blob 数据（区域化分布）教学目的截然不同。

## 3. 用"线性打分 → 概率"理解算法

可以把逻辑回归理解成一个三步走的过程：

1. 先给样本算一个线性得分——特征加权求和
2. 得分越大，越倾向正类；得分越小，越倾向负类
3. 再用 Sigmoid 把这个得分变成 0 到 1 之间的概率

### 理解重点

- 这也是为什么 `coef_` 和 `intercept_` 在当前分册中值得专门观察——它们直接告诉你每个特征"推"正类还是"拉"负类。
- 线性得分的符号决定分类方向（正分 → 正类），绝对值大小反映分类置信度。
- 如果把逻辑回归仅理解成"一个会分类的黑盒"，就会错过它最有价值的可解释性部分——系数本身就是模型解释。

## 4. 为什么 coef_ 值得重点看

在标准化后的特征空间中，`coef_` 有清晰的解释：

- $w_j > 0$：特征 $x_j$ 增大 → 正类概率升高
- $w_j < 0$：特征 $x_j$ 增大 → 正类概率降低
- $\vert w_j \vert$ 大：特征 $x_j$ 对分类结果影响大
- $\vert w_j \vert \approx 0$：特征 $x_j$ 在当前决策中基本不参与

### 理解重点

- 当前训练日志直接打印系数和截距，就是为了让读者把模型输出和数据特征联系起来看。
- 与 KNN（没有显式系数）和决策树（特征重要性不是方向性的）不同，逻辑回归的系数天然带有"正负方向"的含义。

## 5. 与 KNN、决策树、SVC 的直觉差异

四者在当前仓库中的核心差异：

| 算法 | 决策边界 | 核心依据 | 是否需要标准化 | 概率输出方式 | 适用场景 |
|---|---|---|---|---|---|
| LogisticRegression | 全局线性超平面 $\mathbf{w}^T\mathbf{x} + b = 0$ | Sigmoid 概率映射 + 交叉熵优化 | 是（梯度优化 + 系数可比） | Sigmoid 连续映射 | 近线性可分数据 |
| KNN | 局部非参数复杂边界 | $k$ 近邻多数投票 | 是（距离度量必需） | 邻域频率（离散值） | 非线性局部结构数据 |
| DecisionTreeClassifier | 局部轴对齐分段边界 | 不纯度下降 + 递归划分 | 否（阈值切分不依赖距离） | 叶节点内类别占比 | 区域化分布数据 |
| SVC | 最大间隔 + 核变换非线性边界 | 支持向量 + hinge loss | 是（距离/核函数依赖） | 需额外概率校准 | 复杂分布数据 |

### 理解重点

- 逻辑回归在问：哪个方向最能区分正负类（全局线性 $z = \mathbf{w}^T\mathbf{x} + b$）。
- KNN 在问：当前点周围最近的 $k$ 个邻居大多是什么类别（局部投票）。
- 决策树在问：先切哪一刀最能让类别变纯（递归划分）。
- 当前高维近线性可分数据最适合把逻辑回归作为线性概率基线来讲解。

## 可视化

![决策边界](https://img.yumeko.site/file/articles/ML/logistic_regression/decision_boundary.png)

## 常见坑

1. 只知道逻辑回归会分类，却说不出它为什么还能输出概率——Sigmoid 把 $z \in (-\infty, +\infty)$ 映射到 $(0, 1)$。
2. 把所有线性模型都当成同一种东西——逻辑回归用交叉熵 + Sigmoid 做概率分类，线性回归用 MSE 做连续值回归。
3. 只关注预测结果，不关注系数和截距的解释价值——标准化后 $\mathbf{w}$ 的正负和大小直接反映特征影响。
4. 忽略标准化，让系数比较和梯度优化都变得不稳定——不同量纲的特征在优化中"抢梯度"。

## 小结

- 逻辑回归的直觉核心：线性打分 → Sigmoid 概率映射——先算分值再转概率。
- 与 KNN（局部邻域投票）、决策树（轴对齐递归切分）和 SVC（最大间隔）不同，逻辑回归学习的是全局线性概率边界。
- 标准化后 `coef_` 直接反映特征对正类概率的影响方向和强度——这是逻辑回归区别于其他分类模型的重要可解释性优势。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `LogisticRegression`。
2. 理解每个构造器参数的数学含义与调参方向——特别是 `C`（$\lambda = 1/C$）和 `penalty` 的关系。
3. 理解 `coef_`、`intercept_`、`classes_` 在当前源码中的作用。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练 `LogisticRegression`，返回已训练模型 |
| `LogisticRegression(...)` | 构造器 | 创建逻辑回归分类器，通过超参数控制正则化、优化器与收敛条件 |
| `model.fit(X_train, y_train)` | 方法 | 使用 `lbfgs` 优化器最小化 L2 正则化交叉熵损失 |
| `model.classes_` | 属性 | 返回模型识别到的类别标签数组，形状 `(n_classes,)` |
| `model.intercept_` | 属性 | 返回逻辑回归截距 $b$，决定边界偏移量 |
| `model.coef_` | 属性 | 返回各特征对应系数 $\mathbf{w}$，反映特征对正类倾向的影响方向与强弱 |

## 1. `train_model(...)` 的函数签名

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 `(n_samples, n_features)`。传入 `model.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 `(n_samples,)`。二分类标签取值为 $\{0, 1\}$ | `y_train` |
| `penalty` | `str` | 正则化类型。`"l2"` 对 $\|\mathbf{w}\|_2^2$ 惩罚；`"l1"` 对 $\|\mathbf{w}\|_1$ 惩罚（稀疏解）；`"elasticnet"` 混合；`None` 不惩罚。当前默认 `"l2"` | `"l2"`、`"l1"`、`None` |
| `C` | `float` | 正则化强度倒数，$\lambda = 1/C$。$C$ 越大 → 正则越弱 → 系数越自由。当前默认 `1.0` | `0.01`、`1.0`、`100.0` |
| `solver` | `str` | 优化器。`"lbfgs"` 拟牛顿法（默认，适合小中型数据）；`"liblinear"` 坐标下降（适合小数据）；`"saga"` 支持 L1 + 弹性网络 + 大数据。当前默认 `"lbfgs"` | `"lbfgs"`、`"liblinear"`、`"saga"` |
| `max_iter` | `int` | 优化器最大迭代次数。默认 `100`，当前取 `1000`——给的比较宽裕，防止未收敛就停止 | `100`、`1000` |
| `class_weight` | `dict`、`str` 或 `None` | 类别权重。`"balanced"` 自动按 $w_k = n / (K \cdot n_k)$ 加权；`None` 各类等权。当前默认 `None` | `None`、`"balanced"`、`{0:0.5, 1:2.0}` |
| `random_state` | `int` | 随机种子，保证数据打乱与优化器初始化的可复现性。当前取 `42` | `42` |
| 返回值 | `LogisticRegression` | 已训练完成的模型对象，含 `coef_`、`intercept_`、`classes_` 等属性 | — |

### 示例代码

```python
from model_training.classification.logistic_regression import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `LogisticRegression` 模型。
- 和部分实验型代码不同，这里没有参数搜索逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `LogisticRegression(...)` 的完整参数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` 或 `None` | 正则化类型。`"l2"` = $\frac{1}{2C}\|\mathbf{w}\|_2^2$（均匀收缩）；`"l1"` = $\frac{1}{C}\|\mathbf{w}\|_1$（稀疏解，部分系数压为 0）；`"elasticnet"` = $\frac{1}{C}(\rho\|\mathbf{w}\|_1 + (1-\rho)\|\mathbf{w}\|_2^2)$；`None` 不惩罚。默认为 `"l2"` | `"l2"`、`"l1"`、`"elasticnet"`、`None` |
| `dual` | `bool` | 对偶或原始形式。仅 `solver='liblinear'` 且 `penalty='l2'` 时对偶形式可用。$n_{\text{samples}} > n_{\text{features}}$ 时应设为 `False`。默认为 `False` | `False`、`True` |
| `tol` | `float` | 优化收敛容忍度。优化器在两次迭代损失变化小于此值时停止。默认为 `1e-4` | `1e-4`、`1e-6` |
| `C` | `float` | 正则化强度倒数，数学上 $\lambda = 1/C$。$\lambda$ 越大 → 系数收缩越强 → 过拟合风险越低。默认为 `1.0` | `0.01`、`1.0`、`100.0` |
| `fit_intercept` | `bool` | 是否计算截距 $b$。`False` 时强制 $b=0$，决策边界过原点。默认为 `True` | `True`、`False` |
| `intercept_scaling` | `float` | 截距缩放因子。仅 `solver='liblinear'` 且 `fit_intercept=True` 时有效。值越大截距的正则化越小。默认为 `1.0` | `1.0`、`10.0` |
| `class_weight` | `dict`、`str` 或 `None` | 类别权重。`None` 各类等权；`"balanced"` 各样本权重 $w_k = n / (K \cdot n_k)$，$n_k$ 为类别 $k$ 的样本数。不均衡数据应关注。默认为 `None` | `None`、`"balanced"`、`{0:0.5, 1:2.0}` |
| `random_state` | `int` | 随机种子，控制数据打乱与优化器初始化。保证相同数据下结果可复现。默认为 `None` | `42` |
| `solver` | `str` | 优化器选择。`"lbfgs"` 拟牛顿法（稳健默认）；`"liblinear"` 坐标下降法（小数据快）；`"newton-cg"` 牛顿法；`"sag"` 随机平均梯度（大数据快）；`"saga"` sag 的改进版（支持稀疏 + elasticnet）。默认为 `"lbfgs"` | `"lbfgs"`、`"liblinear"`、`"saga"` |
| `max_iter` | `int` | 优化器最大迭代次数。当前取 `1000`——比默认 `100` 高很多，因为高维 + 正则化场景可能收敛慢。未收敛时会有 `ConvergenceWarning` | `100`、`500`、`1000` |
| `multi_class` | `str` | 多分类策略。`"auto"` 根据数据自动选择（二分类选 `"ovr"`）；`"ovr"` 一对多；`"multinomial"` 多项逻辑回归（Softmax 交叉熵）。默认为 `"auto"` | `"auto"`、`"ovr"`、`"multinomial"` |
| `warm_start` | `bool` | 是否复用上一次 `fit()` 的解作为初始化。`True` 时适合连续调参。默认为 `False` | `False`、`True` |
| `n_jobs` | `int` 或 `None` | 并行作业数。仅 `multi_class='ovr'` 时有效。`-1` 用全部核心。默认为 `None` | `None`、`-1`、`4` |
| `l1_ratio` | `float` | L1 在 elasticnet 中的混合比例。仅 `penalty='elasticnet'` 时生效。默认为 `None` | `None`、`0.15`、`0.5` |

### 示例代码

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,
    random_state=42,
)
model.fit(X_train_s, y_train)
```

### 理解重点

- 仓库没有自己实现交叉熵优化过程，而是直接调用 scikit-learn 的成熟实现。
- **`C` 是正则化强度的倒数**（$\lambda = 1/C$）——$C$ 越大正则越弱，这是逻辑回归文档最容易出错的地方。
- 最值得关注的核心参数：`penalty`、`C`、`solver`、`max_iter`——它们决定"怎么限制模型"和"怎么优化"。
- 当前 `max_iter=1000` 是教科书的保守设置，确保 `lbfgs` 在高维 + 正则化场景下有足够迭代次数收敛。

## 3. 训练完成后最重要的模型属性

### 属性表

| 属性 | 类型 | 数学含义 |
|---|---|---|
| `classes_` | `ndarray` | 模型识别到的类别标签，形状 `(n_classes,)`。当前二分类为 `[0, 1]` |
| `coef_` | `ndarray`，形状 `(1, d)` | 权重系数 $\mathbf{w}$。$w_j > 0$ 表示特征 $j$ 增大推高正类概率，$w_j < 0$ 表示压低正类概率。标准化后各系数可比 |
| `intercept_` | `ndarray`，形状 `(1,)` | 截距 $b$。$b > 0$ 表示在没有特征信息时（$\mathbf{x} = \mathbf{0}$）模型倾向正类 |
| `n_features_in_` | `int` | 训练时的特征维度 $d$。当前为 `6` |
| `n_iter_` | `ndarray`，形状 `(n_classes,)` | 优化器实际迭代次数。如果接近 `max_iter=1000`，说明可能未收敛 |
| `C_` | `float` | 实际使用（而非用户传入）的 $C$ 值。当 `C` 传入 0 或负数时，sklearn 会修正为很小的正数 |
| `penalty_` | `str` | 实际使用的正则化类型。当 `penalty='elasticnet'` 但 `l1_ratio=0` 时会修正为 `'l2'` |

### 示例代码

```python
print(f"类别: {model.classes_.tolist()}")
print(f"截距: {model.intercept_.round(4)}")
print(f"系数: {model.coef_.round(4)}")
print(f"实际迭代次数: {model.n_iter_}")
```

### 理解重点

- `coef_` 和 `intercept_` 是逻辑回归最有价值的训练结果——它们把"线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$"映射成可直接观察的数值。
- 在标准化后的特征空间中，$w_j$ 的大小可以粗略反映特征 $j$ 的相对重要性。
- `n_iter_` 值得关注：如果它等于 `max_iter`，说明优化器在到达最大迭代次数时可能尚未收敛——此时应考虑增大 `max_iter` 或调整 `solver`。

## 4. 训练阶段的工程封装

除了 `LogisticRegression(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| 函数调用标题（`@print_func_info`） | 帮助在终端中定位训练入口 |
| 训练耗时（`@timeit`） | 观察 `lbfgs` 优化器的拟合时间 |
| 超参数日志（`penalty`、`C`、`solver`、`max_iter`） | 确认当前训练配置 |
| 类别、截距与系数日志 | 把线性边界参数映射为源码里可直接观察的输出 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 与 KNN 的 `fit()`（只建索引）和决策树的 `fit()`（递归划分）不同，逻辑回归的 `fit()` 本质上是迭代优化交叉熵损失——这是训练耗时的主要来源。
- 这一层封装把"构建模型""训练模型""打印结果"收在一个函数里，方便文档和流水线复用。

## 常见坑

1. 把 `C` 的含义写反——`C` 是正则化强度的倒数，$C$ 越大正则越弱，不是"正则化系数"。
2. 只知道可以 `predict(...)`，却忽略 `coef_` 和 `intercept_` 才是理解逻辑回归行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的训练特征——原始特征会让系数不可比，且优化收敛困难。
4. 忽略 `n_iter_` 的值——如果它等于 `max_iter`，模型可能未收敛，预测结果不可靠。

## 小结

- `train_model(...)` 是本仓库 Logistic Regression 的核心训练入口，本质上是对 `sklearn.linear_model.LogisticRegression` 的薄封装。
- `LogisticRegression` 的 14 个构造器参数中，`penalty`、`C`、`solver`、`max_iter` 是最核心的四个——它们决定正则化方式和优化行为。
- **核心公式记忆**：损失 = 交叉熵 + $\frac{1}{2C}\|\mathbf{w}\|_2^2$（当 `penalty='l2'`），$\lambda = 1/C$。
- 训练后属性 `coef_`、`intercept_`、`n_iter_` 是后续模型解释与调参的直接数据来源。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 Logistic Regression 流水线到底执行了哪些步骤。
2. 理解训练集/测试集拆分、标准化、训练、类别预测和概率预测之间的连接关系。
3. 理解主模型与二维可视化模型在当前实现中的职责差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `logistic_regression_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `train_test_split(...)` | 方法 | 划分训练集与测试集 |
| `StandardScaler` | 类 | 对训练/测试特征做一致的标准化处理 |
| `train_model(X_train_s, y_train)` | 函数 | 使用 `lbfgs` 优化器训练主逻辑回归模型 |
| `model.predict(X_test_s)` | 方法 | 生成测试集硬分类结果，判断 $\sigma(\mathbf{w}^T\mathbf{x}+b) \geq 0.5$ |
| `model.predict_proba(X_test_s)` | 方法 | 生成测试集各类别概率输出，正类概率为 $\sigma(\mathbf{w}^T\mathbf{x}+b)$ |
| `PCA(n_components=2)` | 类 | 为决策边界可视化构造二维表示 |
| `model_2d` | 模型 | 专门用于二维决策边界展示 |

## 1. 流水线从复制数据开始

当前流水线先复制 `logistic_regression_data`，再拆出 `X` 和 `y`。

### 示例代码

```python
data = logistic_regression_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- 原始数据只读、流程内部再处理——这是当前仓库多个分册的统一习惯。
- 当前任务是监督二分类，因此 `y` 会真实参与训练和预测评估。

## 2. 先切分训练集与测试集

使用 `train_test_split` 按 8:2 切分，`stratify=y` 保持类别分布一致。

### 参数速览

适用函数：`sklearn.model_selection.train_test_split`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*arrays` | `array_like` | 待切分的数据序列。传入 `(X, y)` 分别对应切分 | `X, y` |
| `test_size` | `float` | 测试集占比。当前取 `0.2`，即 80 个测试样本 | `0.2` |
| `random_state` | `int` | 随机种子。当前取 `42` | `42` |
| `stratify` | `array_like` | 按 `y` 类别比例分层抽样。在当前含 3% 标签噪声的数据上尤其重要 | `y`、`None` |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- 当前流水线明确区分了训练阶段和测试阶段。
- `stratify=y` 保证训练集和测试集类别比例一致——在 `flip_y=0.03` 的场景下这尤其重要。

## 3. 标准化只在训练集上拟合

标准化必须严格在切分后执行——`fit_transform` 在训练集上计算 $\mu_i, \sigma_i$，`transform` 将相同统计量应用于测试集。

### 参数速览

适用类：`sklearn.preprocessing.StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `scaler.fit_transform(X_train)` | 方法 | 计算训练集的 $\mu_i$ 和 $\sigma_i$，并执行 $x_i' = (x_i - \mu_i)/\sigma_i$。返回 `X_train_s` | — |
| `scaler.transform(X_test)` | 方法 | 使用训练集的 $\mu_i$ 和 $\sigma_i$ 变换测试集。返回 `X_test_s` | — |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 标准化对逻辑回归有三个好处：`lbfgs` 优化器收敛稳定、L2 正则化惩罚均匀、`coef_` 之间可直接比较。
- `fit_transform` vs `transform` 的区分模拟了真实部署场景——新数据只能用训练时的标准化参数。

## 4. 主模型训练与正式预测

逻辑回归的 `fit()` 使用 `lbfgs` 优化器最小化 L2 正则化交叉熵损失。训练完成后，`model.predict(...)` 按概率阈值 0.5 输出类别标签。

### 参数速览

适用方法：`LogisticRegression.predict(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的标准化特征矩阵。$d = 6$，必须与训练特征维度一致 | `X_test_s` |
| 返回值 | `ndarray`，形状 `(n_samples,)` | 预测类别标签，$\hat{y}_i \in \{0, 1\}$。$\hat{y} = 1$ 当 $\sigma(\mathbf{w}^T\mathbf{x} + b) \geq 0.5$，等价于 $\mathbf{w}^T\mathbf{x} + b \geq 0$ | — |

### 示例代码

```python
model = train_model(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

### 理解重点

- `model` 是当前分册的主模型，用于正式训练和测试集类别预测。
- 类别预测的阈值默认是 0.5——概率 ≥ 0.5 判为正类，等价于 $\mathbf{w}^T\mathbf{x} + b \geq 0$。
- `y_pred` 是后续混淆矩阵评估的直接输入。

## 5. 概率输出如何进入流水线

`sigmoid` 映射后的正类概率是 ROC 曲线可视化的直接输入。

### 参数速览

适用方法：`LogisticRegression.predict_proba(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的标准化特征矩阵 | `X_test_s` |
| 返回值 | `ndarray`，形状 `(n_samples, n_classes)` | 各类别概率估计。第 1 列 $P(y=0\vert\mathbf{x})$，第 2 列 $P(y=1\vert\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$。每行和为 1 | — |

### 示例代码

```python
y_scores = model.predict_proba(X_test_s)
```

### 理解重点

- `predict_proba(...)` 是逻辑回归的重要接口——概率输出来自 Sigmoid 的连续映射，不像 KNN 那样是离散的邻域频率。
- 连续概率意味着 ROC 曲线是平滑的（而非阶梯状），这是逻辑回归相对于 KNN 在概率输出上的优势。
- 在当前二分类实现中，ROC 曲线实际使用的是 `y_scores[:, 1]`——正类概率列。

## 6. 决策边界为什么要额外训练一个 model_2d

主模型在标准化后的 6 维特征空间中训练，但决策边界图需要能在二维平面上对任意网格点做预测。
当前实现采用 PCA 投影到二维，再单独训练一个逻辑回归模型用于可视化。

### 参数速览

适用类：`sklearn.decomposition.PCA`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的主成分数 $k$。$k=2$ 时将 $d=6$ 维特征投影到二维平面。PCA 通过 SVD 分解 $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，取前 $k$ 个奇异向量 | `2` |
| `random_state` | `int` | 随机种子。PCA 基于 SVD 是确定性的，但某些求解器用随机化算法时需要。当前取 `42` | `42` |

### 示例代码

```python
pca = PCA(n_components=2, random_state=42)
X_all_s = scaler.transform(X)
X_2d = pca.fit_transform(X_all_s)
model_2d = LogisticRegression(max_iter=1000, random_state=42)
model_2d.fit(pca.transform(X_train_s), y_train)
```

### 理解重点

- `model_2d` 不是主评估模型，而是专门为二维可视化服务的辅助模型——它在 PCA 空间训练，仅在决策边界图中使用。
- 主模型训练在标准化后的原 6 维特征空间中，两者职责不同，不可混淆。
- 逻辑回归的 PCA 决策边界通常呈现一条直线——因为逻辑回归本身是线性分类器，在二维空间中边界就是一条直线。

## 7. 学习曲线如何接入流水线

学习曲线用于诊断模型性能是否随训练样本量增加而持续改善。

### 参数速览

适用函数：`result_visualization.learning_curve.plot_learning_curve`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的模型实例。传入 `LogisticRegression(max_iter=1000, random_state=42)`，内部会克隆并逐段训练 | `LogisticRegression(max_iter=1000, random_state=42)` |
| `X` | `array_like` | 标准化后的训练特征矩阵。当前传入 `X_train_s` | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标。`"accuracy"` = $\frac{\sum \mathbb{1}[y_i=\hat{y}_i]}{n}$ | `"accuracy"` |
| `cv` | `int` | 交叉验证折数。默认 `5` | `5`、`10` |

### 示例代码

```python
plot_learning_curve(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_s,
    y_train,
    title="逻辑回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线使用新的 `LogisticRegression` 实例，不直接复用 `model`——因为内部会克隆后重新训练。
- 对逻辑回归而言，学习曲线能直观反映：在 $C=1.0$、`penalty='l2'` 固定时，更多训练数据能否提升泛化性能。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/articles/ML/logistic_regression/learning_curve.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 混为一谈——前者返回标签，后者返回概率。
2. 把 `model_2d` 误认为正式预测模型本体——它仅在 PCA 空间训练。
3. 忘记标准化必须在训练集上 `fit_transform`、在测试集上 `transform`——反过来会造成数据泄露。
4. 混淆主模型预测（6 维空间）、二维可视化模型（PCA 空间）和学习曲线模型（交叉验证循环）三者的职责。

## 小结

- 当前 Logistic Regression 流水线的训练过程：复制数据 → 切分 → 标准化 → `lbfgs` 优化 L2 正则化交叉熵 → 类别预测 → Sigmoid 概率输出 → 多种可视化诊断。
- 逻辑回归的独特之处：概率输出来自连续的 Sigmoid 映射（相对于 KNN 的离散邻域频率），ROC 曲线更平滑。
- 对本仓库而言，`model`（6 维标准化空间）、`model_2d`（PCA 2D 空间）和学习曲线实例分别承担不同职责。

# 评估与诊断

## 本章目标

1. 明确当前仓库 Logistic Regression 实现实际上是如何做结果诊断的。
2. 理解混淆矩阵、ROC 曲线、PCA 决策边界图和学习曲线分别能说明什么。
3. 理解二分类 ROC 与二维决策边界图的展示边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出，由 $\sigma(\mathbf{w}^T\mathbf{x}+b) \geq 0.5$ 决定 |
| `y_scores` | 预测概率 | 测试集正类概率输出，来自连续的 Sigmoid 映射 |
| `plot_confusion_matrix(...)` | 函数 | 绘制预测标签与真实标签的混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制二分类 ROC 曲线 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |

## 1. 当前仓库的评估入口

当前 Logistic Regression 流水线里的主要结果诊断手段有四个：

1. 混淆矩阵
2. ROC 曲线
3. PCA 2D 决策边界图
4. 学习曲线

### 示例代码

```python
y_pred = model.predict(X_test_s)
y_scores = model.predict_proba(X_test_s)

plot_confusion_matrix(...)
plot_roc_curve(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 当前实现同时提供结果矩阵、概率曲线、边界图和曲线图四类视角。
- 逻辑回归没有特征重要性图（这是决策树特有的），但 `coef_` 直接反映了特征的影响方向和强度——这比可视化更重要。
- 四种可视化分别回答：分对了吗（混淆矩阵）、区分力如何（ROC）、边界长什么样（决策边界）、更多数据有用吗（学习曲线）。

## 2. 混淆矩阵能观察什么

混淆矩阵 $\mathbf{C}$ 是一个 $2 \times 2$ 矩阵（二分类）：

$$
C = \begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}
$$

### 参数速览

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $y_i \in \{0, 1\}$ | `y_test` |
| `y_pred` | `array_like`，形状 `(n_samples,)` | 模型预测标签，来自 $\sigma(\mathbf{w}^T\mathbf{x}+b) \geq 0.5$ 的硬分类 | `y_pred` |
| `normalize` | `bool` 或 `str` | 归一化方式。`True`/`'true'` 按行（真实类别），`'pred'` 按列，`'all'` 按全体。默认为 `False` | `True`、`'true'` |

### 示例代码

```python
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    title="逻辑回归 混淆矩阵",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 混淆矩阵最适合回答：正负类分别分对了多少，误分类偏向哪个方向。
- 在当前 `class_sep=1.2`、`flip_y=0.03` 的数据上，逻辑回归通常能获得较高准确率，但受标签噪声影响会有少量误分类。

## 3. ROC 曲线能观察什么

ROC 曲线绘制 TPR 随 FPR 变化的轨迹，通过改变分类阈值（默认 0.5）得到：

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

### 参数速览

适用函数：`plot_roc_curve(y_test, y_scores, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签 | `y_test` |
| `y_scores` | `array_like`，形状 `(n_samples, n_classes)` | 各类别概率估计，来自 `model.predict_proba(X_test_s)`。二分类使用 `y_scores[:, 1]`（正类概率列） | `y_scores` |

### 示例代码

```python
plot_roc_curve(
    y_test,
    y_scores,
    title="逻辑回归 ROC 曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 逻辑回归的概率输出来自连续的 Sigmoid 映射 $\sigma(\mathbf{w}^T\mathbf{x}+b) \in [0, 1]$，因此 ROC 曲线是平滑的——这与 KNN（离散邻域频率）形成对比。
- AUC 越接近 1 表示概率区分能力越强。在当前近线性可分数据上，逻辑回归通常能获得较高的 AUC。
- 当前任务是二分类，ROC 只使用正类概率列就足够了。

## 4. PCA 2D 决策边界图能观察什么

### 参数速览

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model_2d` | `LogisticRegression` | 在 PCA 二维空间上单独训练的逻辑回归模型。`max_iter=1000`，与主模型共享相同的正则化配置 | `model_2d` |
| `X_2d` | `ndarray`，形状 `(n_samples, 2)` | 标准化后 PCA 投影到二维的特征，列分别为 PC1、PC2 | `X_2d` |
| `y` | `array_like`，形状 `(n_samples,)` | 全量标签数组，用于散点的真实类别着色 | `y.values` |

### 示例代码

```python
plot_decision_boundary(
    model_2d,
    X_2d,
    y.values,
    title="逻辑回归 决策边界 (PCA 2D)",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 逻辑回归的决策边界在二维 PCA 空间中呈现为一条直线——这是因为逻辑回归本身是线性分类器。
- 这与 KNN 的蜿蜒边界和决策树的轴对齐分段边界形成鲜明对比。
- 但这只是 PCA 投影空间中的近似展示，原始 6 维特征空间中的决策面是 5 维超平面。

## 5. 学习曲线能观察什么

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的逻辑回归实例。传入 `LogisticRegression(max_iter=1000, random_state=42)` | `LogisticRegression(max_iter=1000, random_state=42)` |
| `X` | `array_like` | 标准化后的训练特征矩阵 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标。当前取 `"accuracy"` | `"accuracy"` |
| `cv` | `int` | 交叉验证折数。默认 `5` | `5`、`10` |
| `train_sizes` | `array_like` | 训练样本量的递增序列。默认为 `np.linspace(0.1, 1.0, 5)` | `[0.1, 0.33, 0.55, 0.78, 1.0]` |

### 示例代码

```python
plot_learning_curve(
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_s,
    y_train,
    title="逻辑回归 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 逻辑回归是参数化模型（$d+1$ 个参数），即使样本量不大通常也能稳定学习。学习曲线可以用来验证是否需要更多数据。
- 如果训练得分和验证得分都很高且接近，说明模型在当前 $C=1.0$ 下没有明显过拟合——正则化起作用了。

## 6. 当前实现中尚未纳入但常见的分类指标

| 指标 | 公式 | 说明 |
|---|---|---|
| 准确率（Accuracy） | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | 整体正确率 |
| 精确率（Precision） | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | 预测为正类中有多少真实正类 |
| 召回率（Recall） | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | 真实正类中有多少被正确找出 |
| F1 分数 | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | 精确率与召回率的调和平均 |

### 理解重点

- 当前仓库没有在 Logistic Regression 流水线中显式打印这些指标。
- 文档可以提到它们是常见扩展方向，但不能写成"当前源码已经在单独计算"。
- 混淆矩阵已经隐式包含了计算这些指标所需的所有信息（TP、TN、FP、FN）。

## 评估图表

![混淆矩阵](https://img.yumeko.site/file/articles/ML/logistic_regression/confusion_matrix.png)

![ROC 曲线](https://img.yumeko.site/file/articles/ML/logistic_regression/roc_curve.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 的用途混为一谈——前者用于混淆矩阵，后者用于 ROC。
2. 把 ROC 曲线误解成对类别预测标签直接作图——需要概率输出（Sigmoid 映射后的连续值）来变化阈值。
3. 把 PCA 决策边界图误认为原始 6 维特征空间决策面的完整表达——它只是二维投影近似。
4. 把当前仓库未实现的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 Logistic Regression 的评估方式：混淆矩阵看错误分布，ROC 曲线看概率区分力，PCA 决策边界图看边界形状，学习曲线看训练行为。
- 逻辑回归没有特征重要性评估（这是决策树特有的），但标准化后 `coef_` 的绝对值大小可以粗略反映特征影响——这比可视化更重要。
- 四项评估组合起来，能全面解释当前 $C=1.0$、L2 正则化逻辑回归在高维近线性可分数据上的实际表现。

# 工程实现

## 本章目标

1. 从工程角度看清 Logistic Regression 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解为什么当前实现要把训练逻辑、标准化逻辑、概率输出逻辑和可视化逻辑拆开。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.logistic_regression()` 生成高维二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `logistic_regression_data` |
| 训练封装 | `model_training/classification/logistic_regression.py` | 构建并训练 `LogisticRegression`，打印训练日志含 `coef_` 和 `intercept_` |
| 流水线入口 | `pipelines/classification/logistic_regression.py` | 组织切分、标准化、训练、预测与可视化评估的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 保存混淆矩阵图 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 保存二分类 ROC 曲线图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 保存学习曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.logistic_regression
```

### 理解重点

- 这个命令是理解 Logistic Regression 工程实现的最佳入口。
- 它会依次完成数据读取、标准化、模型训练（`lbfgs` 优化）、预测和结果可视化。
- 如果只读一个文件，建议先读 `pipelines/classification/logistic_regression.py`——编排层。

## 2. run() 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = logistic_regression_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 标准化（仅训练集上 fit）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. lbfgs 优化 L2 正则化交叉熵 & 正式预测
    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_scores = model.predict_proba(X_test_s)

    # 5. 可视化诊断（混淆矩阵、ROC、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_roc_curve(y_test, y_scores, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(LogisticRegression(...), X_train_s, y_train, ...)
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的优化在 `LogisticRegression.fit()`（`lbfgs`）中。
- 数据流是单向的：数据 → 切分 → 标准化 → `lbfgs` 优化 → 预测 → 评估。
- 标准化后 `coef_` 可解释，这是逻辑回归流水线相对于其他算法的一个重要工程特性。

## 3. 训练模块负责什么

`model_training/classification/logistic_regression.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `LogisticRegression(...)` 实例（按给定超参数：`penalty='l2'`、`C=1.0` 等）
2. 调用 `model.fit(X_train, y_train)`——`lbfgs` 优化器最小化 L2 正则化交叉熵
3. 打印训练日志：训练耗时、`penalty`、`C`、`solver`、`max_iter`、`classes_`、`intercept_`、`coef_`
4. 返回训练完成的主模型对象

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵。传入 `LogisticRegression.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，取值 $\{0, 1\}$ | `y_train` |
| `penalty` | `str` | 正则化类型。默认 `"l2"` | `"l2"`、`"l1"` |
| `C` | `float` | 正则化强度倒数 $\lambda = 1/C$。默认 `1.0` | `0.01`、`1.0`、`100.0` |
| `solver` | `str` | 优化器。默认 `"lbfgs"` | `"lbfgs"`、`"liblinear"` |
| `max_iter` | `int` | 最大迭代次数。默认 `1000` | `500`、`1000` |
| `class_weight` | `dict`、`str` 或 `None` | 类别权重。默认 `None` | `None`、`"balanced"` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| 返回值 | `LogisticRegression` | 已完成 `fit()` 的模型对象，含 `coef_`、`intercept_` 等属性 | — |

### 理解重点

- 逻辑回归的 `fit()` 本质是迭代优化——与 KNN（只建索引）和决策树（递归划分）的计算特征不同。
- `coef_` 和 `intercept_` 的输出使得逻辑回归的训练日志比其他算法更有信息量——直接看到特征影响方向和强度。

## 4. 四类评估模块分别负责什么

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 混淆矩阵图片（PNG） |
| ROC 曲线 | `plot_roc_curve(...)` | `y_test`、`y_scores` | 二分类 ROC 曲线图片（PNG） |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | PCA 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `estimator`、`X_train_s`、`y_train` | 训练/验证得分曲线（PNG） |

### 理解重点

- 四类可视化都不是训练的一部分，而是训练完成后的诊断步骤。
- 决策边界图依赖 PCA 降维（$d=6 \to 2$）后的 `model_2d`——在 PCA 空间中是直线边界；ROC 曲线依赖 Sigmoid 概率输出——是平滑曲线。
- 逻辑回归没有特征重要性评估，但有 `coef_` 提供更直接的系数解释。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `logistic_regression_data` | `data_generation/classification.py` | `pipelines/classification/logistic_regression.py` |
| `model`（主模型） | `train_model(...)` | `predict`、`predict_proba` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `y_scores` | `model.predict_proba(...)` | `plot_roc_curve` |
| `model_2d` | `LogisticRegression.fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/logistic_regression/` 目录 |

### 理解重点

- 逻辑回归的流水线与 KNN 结构相似（都需要标准化，都没有特征重要性），但与决策树不同（决策树不需要标准化且有特征重要性）。
- 数据流向单向、无循环依赖，每个模块可以独立测试和替换。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `逻辑回归分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`penalty`、`C`、`solver`、`classes_`、`intercept_`、`coef_` | 查看优化耗时、正则配置和线性边界参数 |
| 混淆矩阵图 | `outputs/logistic_regression/confusion_matrix.png` | 观察正负类误分类方向 |
| ROC 曲线图 | `outputs/logistic_regression/roc_curve.png` | 评估 Sigmoid 概率区分能力 |
| 决策边界图 | `outputs/logistic_regression/decision_boundary.png` | 观察 PCA 2D 空间中的线性边界 |
| 学习曲线图 | `outputs/logistic_regression/learning_curve.png` | 诊断过拟合/欠拟合倾向 |

### 理解重点

- 逻辑回归的训练日志特别有价值——`coef_` 直接反映标准化后各特征对正类倾向的影响。
- 例如 `coef_ = [[1.2, -0.8, 0.3, -0.1, 0.05, 0.02]]`，说明 `x1` 推正类（$w_1 > 0$），`x2` 压正类（$w_2 < 0$），而 `x4`、`x5`、`x6` 几乎不参与（$w_j \approx 0$——可能对应冗余特征）。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/logistic_regression.py` — 入口，了解整体流程
2. 再看 `model_training/classification/logistic_regression.py` — 训练封装，理解超参数和 `coef_` 日志
3. 再看 `result_visualization/confusion_matrix.py` — 基础分类结果评估
4. 再看 `result_visualization/roc_curve.py` — Sigmoid 概率区分能力评估
5. 再看 `result_visualization/decision_boundary.py` — PCA 空间线性边界可视化
6. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
7. 最后回到 `data_generation/classification.py` — 理解数据生成参数

### 理解重点

- 从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 这个顺序对应数据流方向：数据 → 标准化 → `lbfgs` 优化 → 预测 → 评估。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/articles/ML/logistic_regression/result_display.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的优化在 `LogisticRegression.fit()`（`lbfgs`）中。
2. 不区分"主模型"（6 维空间）、"二维可视化模型"（PCA 空间）和"学习曲线模型实例"（CV 循环）的职责边界。
3. 忽略 `coef_` 和 `intercept_` 的日志输出——这是逻辑回归最重要的训练产出。
4. 只看单个文件，不顺着调用链理解整体执行流程。

## 小结

- 当前 Logistic Regression 工程实现采用清晰的模块分层：数据生成 → 训练封装 → 流水线编排 → 结果可视化。
- `run()` 负责串联流程，`train_model(...)` 负责 `lbfgs` 优化 L2 正则化交叉熵，各可视化函数负责结果展示与诊断。
- 逻辑回归在工程上最不同于 KNN/决策树的地方：`fit()` 是真正的迭代优化（需要考虑收敛）；`coef_` 提供了显式的系数解释；标准化同时影响优化收敛和系数可比性。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Logistic Regression 实现。
2. 给出继续深入阅读逻辑回归与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/logistic_regression.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_classification(n_features=6, n_informative=3, class_sep=1.2)` 数据适合逻辑回归的线性边界假设？`n_informative=3` 在 6 个特征中有什么教学意义？
3. 当前 `train_model(...)` 中的 `penalty`、`C`、`solver`、`max_iter` 分别控制什么？`C` 与正则化强度 $\lambda$ 的关系是什么？
4. 为什么 `model.coef_` 与 `model.intercept_` 对理解逻辑回归很重要？标准化后 $w_j$ 的正负和大小分别代表什么？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？逻辑回归的 Sigmoid 概率输出与 KNN 的邻域频率概率输出有什么不同？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？逻辑回归的 PCA 边界通常长什么样？
7. `n_iter_` 属性的含义是什么？如果它等于 `max_iter=1000`，意味着什么？

## 练习方向

### 1. 改动 C

- 把 `C=1.0` 改成 `0.01`、`0.1`、`10.0`、`100.0`、`1000.0`
- 观察变化：
  - `coef_` 的绝对值大小——$C$ 越小（正则越强），系数越收缩趋近 0
  - $\vert w_j\vert$ 的分布——强正则下只有最重要的特征保留较大系数
  - 混淆矩阵中各类别正确/错误分布
  - 学习曲线中训练得分与验证得分的差距——强正则时两者接近（欠拟合），弱正则时训练得分远高于验证得分（过拟合）
- 核心理解：$C$ 是正则化强度的倒数，$\lambda = 1/C$

### 2. 改动 penalty

- 尝试 `penalty='l2'`、`penalty='l1'`（需切换 solver 为 `'saga'` 或 `'liblinear'`）、`penalty=None`
- 对比变化：
  - L1 正则化的系数稀疏性——部分 $w_j$ 会被压缩为 0，自动做特征选择
  - L2 正则化的系数均匀收缩——所有 $w_j$ 变小但非零
  - 无正则化时系数的量级——通常最大，但也最容易过拟合
- 理解 L1 和 L2 的数学公式差异：$\|\mathbf{w}\|_1 = \sum \vert w_j\vert$（稀疏）vs $\|\mathbf{w}\|_2^2 = \sum w_j^2$（均匀）

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test`
- 对比变化：
  - `coef_` 的值——各特征的系数不可直接比较
  - 训练收敛情况——可能收到 `ConvergenceWarning`
- 体会：标准化不仅影响系数可比性，还影响梯度优化的收敛速度和稳定性

### 4. 观察 coef_ 与特征序号的关系

- 逻辑回归数据中 `n_informative=3`、`n_redundant=1`，即前 3 个特征真正有用，第 4 个是冗余线性组合，后 2 个是随机噪声
- 观察 `coef_` 的 6 个值——看看模型是否自动赋予前几个特征更大的系数绝对值
- 对比强正则（$C=0.01$）和弱正则（$C=100$）下系数分布的差异

### 5. 与 KNN、决策树、SVC 对比

- 对照阅读 `docs/classification/knn/`、`docs/classification/decision_tree/`、`docs/classification/svc/`
- 比较要点：
  - 决策边界的性质：逻辑回归是全局线性超平面 $\mathbf{w}^T\mathbf{x} + b = 0$，KNN 是局部非参数边界，决策树是轴对齐分段边界，SVC 是最大间隔 + 核变换边界
  - 概率输出的来源：逻辑回归是连续的 Sigmoid 映射，KNN 是离散的邻域频率
  - 是否需要标准化：逻辑回归需要（梯度优化 + 系数可比），KNN 需要（距离度量），决策树不需要（阈值切分）
  - 可解释性：逻辑回归有 `coef_`（方向+强度），决策树有特征重要性（贡献度），KNN 没有显式特征解释

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`LogisticRegression` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_classification` | 高维分类数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Linear Models | 逻辑回归的数学原理、优化器选择与正则化策略详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 4 章：Linear Methods for Classification，涵盖逻辑回归、LDA、线性可分性的完整数学推导 |

- scikit-learn `LogisticRegression`：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn `make_classification`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
- scikit-learn 用户指南 Linear Models：https://scikit-learn.org/stable/modules/linear_model.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Logistic Regression 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），且对逻辑回归有三大好处（收敛稳定、正则均匀、系数可比）
  - 线性打分 → Sigmoid 概率 → 交叉熵优化的完整数学链
  - `C` 是 $\lambda$ 的倒数（$C$ 越大正则越弱）——这是最容易写反的核心概念
  - `coef_` 的正负和绝对值反映特征对正类概率的影响方向和强度
  - 逻辑回归的 Sigmoid 概率输出是连续的，ROC 曲线平滑——与 KNN 的离散邻域频率本质不同
  - `model`（6 维空间）、`model_2d`（PCA 空间）和学习曲线实例的职责差异
