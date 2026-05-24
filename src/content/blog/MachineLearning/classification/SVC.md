---
title: SVC 支持向量分类
date: 2026-04-28
category: MachineLearning/Classification
tags:
  - Scikit-learn
description: 支持向量分类的数学原理、RBF核技巧与完整工程实现。
image: https://img.yumeko.site/file/blog/SVC.png
status: published
---

# 数学原理

## 本章目标

1. 理解 SVC 的核心优化目标——最大化分类间隔，以及为什么间隔越大泛化能力越强。
2. 理解软间隔中 $C$ 对间隔宽度与误分类惩罚的权衡机制。
3. 理解 RBF 核如何通过隐式高维映射使非线性数据变得线性可分。
4. 理解 $\gamma$ 参数（`gamma`）对 RBF 核局部影响范围的控制。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 最大间隔超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ | 决策面 | 寻找使两类样本到边界最小距离最大化的分离面 |
| 软间隔 $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ | 优化目标 | 允许少量样本违反间隔约束，$C$ 控制容错程度 |
| 支持向量 | 关键样本 | 落在间隔边界上或违反间隔约束的样本——唯一决定最终分类面 |
| 对偶问题 | 优化形式 | 将原问题转化为仅依赖样本内积的形式，为核技巧铺路 |
| RBF 核 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ | 核函数 | 当前源码默认核，通过"距离越近相似度越高"构造非线性决策能力 |
| `gamma` | 超参数 | 控制 RBF 核的局部影响半径——$\gamma$ 越大，单个支持向量的影响范围越小 |
| 决策函数 $f(\mathbf{x}) = \sum \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$ | 预测公式 | 预测时只需支持向量参与计算，非支持向量的 $\alpha_i = 0$ |

## 1. 线性可分情形：硬间隔 SVM

当数据线性可分时，SVM 寻找能将两类正确分开且间隔最大的超平面 $\mathbf{w}^T \mathbf{x} + b = 0$。

### 原问题（Primal）

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$

其中 $y_i \in \{-1, +1\}$。约束 $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ 要求所有样本正确分类且到边界的函数间隔 ≥ 1。最小化 $\frac{1}{2}\|\mathbf{w}\|^2$ 等价于最大化间隔（因为间隔 = $2/\|\mathbf{w}\|$）。

### 对偶问题（Dual）

引入拉格朗日乘子后，对偶形式为：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle
$$

$$
\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

### 理解重点

- 对偶形式只依赖样本间的内积 $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$——这正是核技巧的入口：将内积替换为核函数即可获得非线性能力。
- KKT 条件保证只有支持向量的 $\alpha_i > 0$，其余样本的 $\alpha_i = 0$——预测时只需存储和计算支持向量。

## 2. 软间隔与参数 $C$

真实数据常有噪声或不可完全线性分离，软间隔 SVM 引入松弛变量 $\xi_i \geq 0$：

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i
$$

$$
\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 正则化参数（误分类惩罚系数）。$C$ 越大，越不容忍误分类（间隔变窄、模型更复杂）；$C$ 越小，越强调宽间隔（允许更多违例、模型更简单）。默认 `1.0` | `0.1`、`1.0`、`10.0`、`100.0` |

### 理解重点

- $C$ 的角色与逻辑回归中的 $C$ 一致——都是正则化强度的倒数：$C \uparrow$ 等价于 $\lambda \downarrow$（正则越弱）。
- $C \to \infty$ 逼近硬间隔 SVM；$C \to 0$ 会使模型过于简单（间隔宽到几乎忽略分类准确性）。
- 当前源码默认 `C=1.0`，是一个兼顾稳定性和教学简洁度的起点。

## 3. 支持向量：谁决定边界

KKT 条件揭示了支持向量的三种角色：

- $\alpha_i = 0$：样本被正确分类且在间隔之外——不影响模型
- $0 < \alpha_i < C$：样本恰好落在间隔边界上——自由支持向量
- $\alpha_i = C$：样本在间隔内或被误分类——有界支持向量

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `n_support_` | `ndarray`，形状 `(n_classes,)` | 各类别的支持向量数量 | 当前二分类返回长度为 2 的数组 |
| `n_support_.sum()` | `int` | 支持向量总数 | 反映模型依赖的关键样本规模 |
| `support_vectors_` | `ndarray`，形状 `(n_sv, n_features)` | 支持向量的特征值 | 这些样本唯一决定分类面 |
| `dual_coef_` | `ndarray` | $\alpha_i y_i$ | 对偶系数与标签的乘积 |
| `intercept_` | `ndarray` | $b$ | 决策函数的偏置项 |

### 理解重点

- 数以百计的训练样本中，往往只有几十个是支持向量——这正是 SVM 稀疏性的体现。
- `n_support_` 的规模直接反映分类任务的难度：支持向量越多，说明两类越纠缠、边界越复杂。
- 这也是当前训练日志打印 `n_support_` 的教学意义所在。

## 4. 核函数：从线性到非线性

对偶形式中的内积 $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$ 可以替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$，实现隐式的高维特征映射：

$$
\phi: \mathbb{R}^d \to \mathcal{H}, \quad K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle
$$

### 参数速览

| 核函数 | 公式 $K(\mathbf{x}, \mathbf{z})$ | `kernel` 取值 | 适用场景 |
|---|---|---|---|
| 线性核 | $\mathbf{x}^T \mathbf{z}$ | `'linear'` | 高维文本、特征数远大于样本数 |
| RBF（高斯）核 | $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ | `'rbf'` | 通用非线性——当前默认核 |
| 多项式核 | $(\gamma \mathbf{x}^T \mathbf{z} + r)^d$ | `'poly'` | 图像等已归一化的数据 |
| Sigmoid 核 | $\tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$ | `'sigmoid'` | 近似两层神经网络 |

### 理解重点

- 核技巧的价值：无需显式构造高维特征空间 $\mathcal{H}$（可能是无穷维），只需计算低维空间中的核函数值。
- 当前源码默认 `kernel='rbf'`——这是对同心圆非线性数据的直接回应。
- RBF 核的 Mercer 条件保证了 $K(\mathbf{x}, \mathbf{z})$ 确实对应于某个高维空间的内积。

## 5. RBF 核与参数 $\gamma$

### 参数速览

适用 API：`SVC(kernel='rbf', gamma='scale')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `gamma` | `float` 或 `str` | RBF 核系数。`'scale'`（默认）时 $\gamma = 1/(n\_features \cdot X.var())$；`'auto'` 时 $\gamma = 1/n\_features$；传入 `float` 时直接使用。$\gamma$ 越大，单个支持向量的影响范围越小、边界越精细弯曲 | `'scale'`、`'auto'`、`0.1`、`1.0`、`10.0` |

### 理解重点

- $\gamma$ 控制 RBF 核的"局部性"——$\gamma$ 小意味着高斯核的宽度大，单个支持向量影响范围远，决策边界更平滑；$\gamma$ 大意味着每个支持向量只影响很近的区域，边界精细但容易过拟合。
- `gamma='scale'`（scikit-learn 0.22+ 默认）会根据特征方差自动缩放 $\gamma$——这使得标准化后的数据获得合理的默认核宽度。
- $\gamma$ 和 $C$ 共同决定模型的复杂度：高 $C$ + 高 $\gamma$ 容易过拟合，低 $C$ + 低 $\gamma$ 容易欠拟合。

## 6. 标准化为何对 SVC 至关重要

RBF 核计算样本间的欧氏距离：

$$
\|\mathbf{x} - \mathbf{z}\|^2 = \sum_{j=1}^{d} (x_j - z_j)^2
$$

如果某个特征的取值量纲远大于其他特征（如 $x_1 \in [0, 1000]$ 而 $x_2 \in [0, 1]$），则 $x_1$ 将主导距离计算，扭曲核函数的几何意义。

### 理解重点

- 标准化后每个特征的均值为 0、方差为 1，距离计算中各维度平等贡献——这是核方法的标准工程做法。
- 当前流水线统一使用 `StandardScaler` 不仅是为了工程一致性，更是 RBF 核对特征尺度的数学依赖所要求的。

## 7. 决策函数与预测

训练完成后，决策函数为：

$$
f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b
$$

预测时取 $\hat{y} = \text{sign}(f(\mathbf{x}))$。注意求和只遍历支持向量（$\alpha_i > 0$），而非全部训练样本。

### 理解重点

- 预测时只需存储支持向量及其 $\alpha_i y_i$——对于稀疏解（支持向量少），这比 KNN（需存储全部训练集）更节省内存。
- SVC 默认不直接输出概率——`predict_proba(...)` 需要额外启用 `probability=True` 并通过 Platt scaling 校准，耗时显著增加。当前流水线未使用概率输出。

## 8. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 最大间隔超平面 | $\mathbf{w}^T \mathbf{x} + b = 0$ | SVC 算法核心优化目标 |
| 软间隔原问题 | $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ | `SVC(C=1.0)` |
| 对偶问题 | $\max_{\alpha} \sum\alpha_i - \frac{1}{2}\sum\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i,\mathbf{x}_j)$ | SVC 内部 `libsvm` 求解 |
| RBF 核 | $\exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ | `kernel='rbf'` |
| 核系数 | $\gamma = 1/(d \cdot \text{Var}(X))$（`'scale'`） | `gamma='scale'` |
| 支持向量数量 | — | `model.n_support_` |
| 支持向量 | $SV = \{\mathbf{x}_i \mid \alpha_i > 0\}$ | `model.support_vectors_` |
| 对偶系数 × 标签 | $\alpha_i y_i$ | `model.dual_coef_` |
| 决策函数偏置 | $b$ | `model.intercept_` |
| 类别标签 | $\{-1, +1\}$（内部），$\{0, 1\}$（用户侧） | `model.classes_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler` |

## 常见坑

1. 忽略标准化的关键性——RBF 核对特征尺度极为敏感，不标准化等于让距离计算被量纲绑架。
2. 把 $C$ 当成"越大越强"的参数——$C$ 越大越容易过拟合，$\lambda = 1/C$ 的逻辑与逻辑回归一致。
3. 忽略 $\gamma$ 与 $C$ 的联合效应——两者共同决定模型复杂度，单一调参往往效果不佳。
4. 在不需要核方法的线性数据上默认使用 RBF 核——增加了计算开销而无收益。
5. 混淆 SVC 的决策函数输出（$f(\mathbf{x})$ 的符号）与概率输出——当前流水线不使用 `predict_proba`，因为 Platt scaling 会额外引入交叉验证开销。

## 小结

- SVC 的数学核心链：最大间隔 $\min\frac{1}{2}\|\mathbf{w}\|^2$ → 软间隔 $+C\sum\xi_i$ → 对偶形式 + 内积 → RBF 核 $K(\mathbf{x},\mathbf{z}) = \exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$ → 决策函数 $f(\mathbf{x}) = \sum\alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$。
- $C$ 控制软间隔容错，$\gamma$ 控制 RBF 核局部半径——两者联合决定模型复杂度。
- 支持向量（`n_support_`）是 SVC 独有的教学视角——理解它们就是理解 SVC 行为的关键。
- 当前源码 `SVC(C=1.0, kernel='rbf', gamma='scale')` 是最经典的非线性 SVM 配置——直接回应同心圆数据的线性不可分特性。

# 数据构成

## 本章目标

1. 明确本仓库 SVC 数据来自 `make_circles(...)` 构造的同心圆二分类数据。
2. 理解 `noise`、`factor` 参数对同心圆数据形态的控制。
3. 明确训练集/测试集切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClassificationData.svc()` | 方法 | 生成 SVC 使用的非线性二分类同心圆数据 |
| `make_circles(...)` | 函数 | scikit-learn 提供的同心圆数据生成器 |
| `svc_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `label` | 列名 | 当前流水线中的监督分类标签，取值 $\{0, 1\}$ |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——对 RBF 核的距离计算至关重要 |

## 1. 数据生成：`make_circles()`

当前 SVC 数据来自 `ClassificationData.svc()`，底层调用 `sklearn.datasets.make_circles()`。

### 参数速览

适用函数：`make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数（如果指定了 `factor`，会按比例分配到内外圈）。默认 `400` | `400`、`1000` |
| `noise` | `float` | 添加到 x 和 y 坐标上的高斯噪声标准差。`0` 表示完全无噪声的同心圆，`0.1` 使样本轻微偏离理想圆 | `0.1`、`0.05`、`0.2` |
| `factor` | `float` | 内圈半径与外圈半径之比，取值 $(0, 1)$。`0.5` 表示内圈半径是外圈的一半 | `0.5`、`0.3`、`0.8` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| `shuffle` | `bool` | 是否打乱样本顺序。默认 `True` | `True` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(400, 2)$，$y$ 取值 $\{0, 1\}$ | — |

### 示例代码

```python
X, y = make_circles(
    n_samples=400,
    noise=0.1,
    factor=0.5,
    random_state=42,
)
columns = [f"x{i + 1}" for i in range(2)]
data = DataFrame(X, columns=columns)
data["label"] = y
```

### 理解重点

- 同心圆数据是典型的数据集，刻意排除了直接用线性边界正确分类的可能。
- 外圈（label=0）和内圈（label=1）构成环形嵌套结构——这一几何特征恰好需要 RBF 核的非线性映射能力。
- `noise=0.1` 使样本偏离理想圆，增加了一定的分类难度但保留了环形结构的主体特征。
- 只包含 $x_1$、$x_2$ 两个特征，既适合 RBF 核训练，也适合通过二维决策边界图直接展示非线性分类效果。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["label"])` |
| `y` | `Series` | 监督二分类标签，取值 $y_i \in \{0, 1\}$，0 为外圈、1 为内圈 | `data["label"]` |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- 标签 $y=0$ 对应外圈（样本数较多），$y=1$ 对应内圈（样本数较少）。
- `label` 是监督标签，会真实参与 `model.fit(X_train, y_train)`。
- 将特征和标签明确拆分是后续切分、标准化和训练的前提。

## 3. 训练/测试集切分

### 参数速览

适用函数：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 特征矩阵，形状 $(400, 2)$ | `X` |
| `y` | `Series` | 标签向量，取值 $\{0, 1\}$ | `y` |
| `test_size` | `float` | 测试集占比。400 × 0.2 = 80 测试样本，320 训练样本 | `0.2` |
| `random_state` | `int` | 随机种子，保证切分可复现 | `42` |
| `stratify` | `array_like` | 传入 `y` 使训练/测试集类别比例与原始一致 | `y` |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- `stratify=y` 确保内外圈比例在训练/测试集中一致——对 `factor=0.5` 下内外圈面积不等引起的样本数差异尤其重要。
- 切分必须在标准化之前执行，否则测试集信息会通过 $\mu_j$、$\sigma_j$ 泄露到训练流程中。

## 4. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(320, 2)$ | 训练特征矩阵，用于 `fit_transform`——计算 $\mu_j, \sigma_j$ 并原地变换 | `X_train` |
| `X_test` | `array_like`，形状 $(80, 2)$ | 测试特征矩阵，用训练集统计量 `transform` | `X_test` |
| 返回值 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，均值为 0 标准差为 1 | `X_train_s`、`X_test_s` |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 对 SVC 而言，标准化不是可选的——RBF 核 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ 直接依赖欧氏距离，未标准化的特征会让距离计算被量纲绑架。
- `gamma='scale'` 会使用标准化后的特征方差计算 $\gamma$，使核宽度自动适配数据尺度。
- `fit_transform` 在训练集上同时计算统计量和变换，`transform` 在测试集上使用同一统计量——这是避免数据泄露的标准工程做法。

## 数据可视化

![类别分布](https://img.yumeko.site/file/articles/ML/svc/data_class_distribution.png)

![相关性热力图](https://img.yumeko.site/file/articles/ML/svc/data_correlation.png)

![散点图矩阵](https://img.yumeko.site/file/articles/ML/svc/data_scatter.png)

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来——特征矩阵不能包含标签列。
2. 在切分之前就对全量数据做标准化——这是数据泄露，测试信息混入了训练统计量。
3. 忽略标准化对 RBF 核的绝对必要性——不标准化的 SVC 等于让核函数基于失真的距离工作。
4. 看到二维数据就误以为线性分类器足够——同心圆的环形嵌套结构决定了线性不可分。

## 小结

- 当前 SVC 数据来自 `make_circles(n_samples=400, noise=0.1, factor=0.5)`：2 个连续特征、环形嵌套的二分类结构。
- 数据流为：`make_circles` → DataFrame（`x1`、`x2` + `label`）→ 切分（`stratify=y`）→ 标准化（仅在训练集 `fit`）。
- 同心圆数据与 RBF 核 SVC 的组合，是展示非线性核方法最经典的教学配置。

# 思路与直觉

## 本章目标

1. 用直观方式理解 SVC 的"最大间隔"直觉——不只追求分类正确，还追求边界稳健。
2. 理解为什么同心圆数据必须依赖核方法，以及 RBF 核的直觉。
3. 通过与其他分类算法的对比，建立 SVC 在整个分类算法图中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 最大间隔 | 核心直觉 | 让分类边界离两边样本都尽可能远，提升泛化能力 |
| 支持向量 | 关键样本 | 只有离边界最近的样本才决定边界位置——其余样本可丢弃 |
| 软间隔 | 容错机制 | 允许少量样本违反间隔约束，换取更宽的边距和更好的泛化 |
| 核技巧 | 非线性能力 | 不显式构造高维特征就能获得非线性边界——RBF 核是其最经典实现 |
| RBF 核 | 局部相似度 | 离得近的点影响大、离得远的点影响小——以此构造弯曲边界 |

## 1. 为什么 SVC 追求"最大间隔"

大多数分类器只要能分开就行。但 SVC 追问一个更深层的问题：**在所有能正确分类的边界中，哪一个最稳健？**

如果把边界想象成一道墙，SVC 的目标是让这道墙离两边最近的房屋都尽可能远——这样即使将来有略微偏移的新样本落入，也不会轻易跨过边界。

### 理解重点

- 这不是为了在训练集上拿更高分——而是为了在未见过的测试数据上表现更好。
- 直觉上：窄边界意味着稍微移动一点就可能翻转预测，宽边界意味着容错空间更大。
- 这就是"最大间隔"的根本直觉——它不是数学技巧，而是泛化能力的直观追求。

## 2. 为什么同心圆数据需要核方法

当前 SVC 数据来自 `make_circles(n_samples=400, noise=0.1, factor=0.5)`——外圈包着内圈，标签 0 在外、标签 1 在内。

如果是逻辑回归或线性 SVM：一条直线只能把平面切成两半，无论如何切都无法将内圈从外圈中完整挖出。

### 理解重点

- 线性分类器在这个数据上的天花板极低——无论怎么调参，一条直线画不出一个圆。
- 这正是核方法的核心应用场景：原空间中无法用简单形状分离的数据，在高维映射后可能变得线性可分。
- 当前选择 `SVC(kernel='rbf')` 不是追求更复杂，而是对数据形状的直接回应。

## 3. 用"支持向量卡住边界"理解 SVC

可以把 SVC 的训练过程想象成：

1. 先假设一条初始分类边界
2. 观察离边界最近的那些样本——它们是"争议区"
3. 调整边界，让这些关键样本把边界"卡"在最合理的位置上
4. 距离边界较远的样本不参与边界的最终确定

### 理解重点

- 训练集中可能只有 20%~40% 的样本真正参与了决策——其余样本的去除不改变边界。
- 这也是 `model.n_support_` 的教学价值所在：它直接量化了"模型到底依赖了哪些样本"。
- 这种稀疏性是 SVC 区别于 KNN（必须记住全部训练样本）和逻辑回归（所有权重由全量数据决定）的关键特征。

## 4. 用"高维空间更容易分离"理解 RBF 核

RBF 核的直觉可以这样理解：

- 在原始二维平面上，内圈和外圈混在一起无法用直线分开。
- RBF 核隐式地将每个点映射到一个更高维的空间——在这个空间中，不同圈层的点可能自动"散开"。
- 核技巧的妙处在于：虽然映射是高维甚至无穷维的，但我们从不需要显式构造它——只需计算两点在原空间的相似度 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$。

### 理解重点

- RBF 核本质上是"局部相似度度量"：$\|\mathbf{x} - \mathbf{z}\|$ 越小，$K(\mathbf{x}, \mathbf{z})$ 越接近 1；距离越远，相似度迅速衰减到 0。
- $\gamma$ 控制"多远算远"——$\gamma$ 大意味着只有非常近的点才被认为相似，边界精细弯曲；$\gamma$ 小意味着较远的点仍有影响，边界更平滑。
- 对于同心圆数据，RBF 核为内圈和外圈的点各自分配了不同的"高维特征签名"，使得 SVM 能在高维空间中找到分离超平面。

## 5. 与逻辑回归、决策树的直觉对比

| 算法 | 核心问题 | 决策边界形状 | 依赖哪些样本 |
|---|---|---|---|
| SVC (RBF) | 怎样画一条弯曲的边界，且让它离两边样本尽量远？ | 非线性曲线——由支持向量和 RBF 核决定 | 仅支持向量（$\alpha_i > 0$） |
| 逻辑回归 | 怎样画一条直线，让正负类概率差异最大化？ | 全局直线 $\mathbf{w}^T \mathbf{x} + b = 0$ | 所有训练样本（通过梯度） |
| KNN | 离当前样本最近的邻居投给谁？ | 局部蜿蜒——无全局参数形式 | 所有训练样本（查询时） |
| 决策树 | 先按哪个特征切一刀，再按哪个特征切？ | 轴对齐分段 | 所有训练样本（通过划分） |

### 理解重点

- 在同心圆数据上，逻辑回归和线性 SVM 必然失败——它们的直线边界无法包裹内圈。
- 决策树可以通过多次轴对齐切分近似圆形边界，但需要较深的树；SVC 的 RBF 核能自然地产生光滑曲线边界。
- SVC 和 KNN 虽然都能产生非线性边界，但原理完全不同：KNN 在预测时现场计算邻居，SVC 在训练时已经确定了支持向量和边界形状。

## 6. 标准化对 SVC 直觉的影响

RBF 核把"两点距离的平方"转化为相似度。如果 $x_1$ 取值在 $[-2, 2]$ 而 $x_2$ 取值在 $[-1000, 1000]$，那么距离计算几乎完全由 $x_2$ 主导——$x_1$ 的差异被淹没。

### 理解重点

- 标准化让所有特征站在同一起跑线上——每个维度对距离的贡献平等。
- 对 SVC 来说这不是锦上添花，而是核函数几何意义正确的前提。
- 这一点与逻辑回归高度一致（梯度优化 + 系数可比），与决策树根本不同（决策树不依赖距离计算）。

## 可视化

![决策边界](https://img.yumeko.site/file/articles/ML/svc/decision_boundary.png)

## 常见坑

1. 只知道 SVC 是"强大的分类器"，却说不出"最大间隔"和"支持向量"的直觉意义。
2. 把 RBF 核理解成"更复杂所以更好"，而不是结合数据形状来理解选择理由。
3. 把支持向量当成"模型拟合不完全的标记"——实际上它们是 SVC 设计的核心，不是缺陷。
4. 忽略标准化，让 RBF 核基于失真的距离计算工作。

## 小结

- SVC 的直觉核心有两个：最大间隔（追求稳健边界）和支持向量（只让关键样本决定边界）。
- 当前仓库使用同心圆数据 + RBF 核，恰恰是展示"为什么有些数据需要非线性边界"的最佳教学组合。
- 理解 SVC 与其他分类算法在直觉层面的差异，是理解"何时选 SVC"的前提——它不是万能的默认选项，而是非线性富边界场景的利器。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `SVC`。
2. 理解 `SVC` 的核心构造器参数（`C`、`kernel`、`gamma`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`n_support_`、`support_vectors_`、`dual_coef_`、`intercept_`。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.svm.SVC` 模型，打印训练日志 |
| `SVC(...)` | 类 | scikit-learn 提供的 C-Support Vector Classification——基于 `libsvm` 的成熟实现 |
| `model.fit(X_train, y_train)` | 方法 | 求解对偶优化问题，找出支持向量和决策函数参数 |
| `model.n_support_` | 属性 | 各类别的支持向量数量——量化模型依赖的关键样本规模 |
| `model.support_vectors_` | 属性 | 支持向量的特征矩阵 |
| `model.dual_coef_` | 属性 | 对偶系数与标签的乘积 $\alpha_i y_i$ |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 $(320, 2)$，传入 `SVC.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，二分类取值 $\{0, 1\}$ | `y_train` |
| `C` | `float` | 正则化参数（误分类惩罚系数）。$C$ 越大，间隔越窄、越不容忍误分类。默认 `1.0` | `0.1`、`1.0`、`10.0` |
| `kernel` | `str` | 核函数类型。默认 `'rbf'`，当前同心圆数据的最优选择 | `'linear'`、`'rbf'`、`'poly'` |
| `gamma` | `float` 或 `str` | RBF 核系数。`'scale'`（默认）时 $\gamma = 1/(d \cdot X.var())$；`'auto'` 时 $\gamma = 1/d$ | `'scale'`、`'auto'`、`0.1`、`1.0` |
| `random_state` | `int` | 随机种子，保证概率估计等随机过程可复现。默认 `42` | `42` |
| 返回值 | `SVC` | 已完成 `fit()` 的模型对象，含 `n_support_`、`support_vectors_` 等属性 | — |

### 示例代码

```python
from model_training.classification.svc import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前入口很直接：只负责构建一个 RBF 核 `SVC` 并 `fit`，没有多核并行对比或网格搜索。
- 所有默认超参数（`C=1.0`、`kernel='rbf'`、`gamma='scale'`）都写在函数签名里，阅读成本低。
- `train_model(...)` 是对 `sklearn.svm.SVC` 的薄封装——算法本体是 scikit-learn 基于 `libsvm` 的 C++ 实现。

## 2. `SVC` 构造器核心参数

### 参数速览

适用 API：`SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 软间隔惩罚系数，对应目标函数 $C\sum\xi_i$。$C \uparrow$ → 间隔变窄、更关注训练精度；$C \downarrow$ → 间隔变宽、更关注泛化。默认 `1.0` | `0.1`、`1.0`、`10.0`、`100.0` |
| `kernel` | `str` | 核函数类型。`'linear'`、`'poly'`、`'rbf'`、`'sigmoid'` 或 `'precomputed'`。默认 `'rbf'` | `'rbf'`、`'linear'`、`'poly'` |
| `degree` | `int` | 多项式核的次数 $d$，仅当 `kernel='poly'` 时生效。默认 `3` | `2`、`3`、`4` |
| `gamma` | `float` 或 `str` | 核系数，控制单个训练样本的影响半径。`'scale'`（默认）时 $\gamma = 1/(n\_features \cdot X.var())$；`'auto'` 时 $\gamma = 1/n\_features$；传入 `float` 直接使用。$\gamma \uparrow$ → 影响半径缩小、边界更精细弯曲 | `'scale'`、`'auto'`、`0.01`、`1.0`、`10.0` |
| `coef0` | `float` | 核函数中的独立项 $r$，仅对 `'poly'` 和 `'sigmoid'` 核生效。默认 `0.0` | `0.0`、`1.0` |
| `probability` | `bool` | 是否启用概率估计。`True` 时会在训练后额外做 5 折交叉验证 Platt scaling，显著增加训练耗时。默认 `False` | `False`、`True` |
| `shrinking` | `bool` | 是否使用收缩启发式加速优化。默认 `True` | `True` |
| `tol` | `float` | 优化停止容差。默认 `1e-3` | `1e-3`、`1e-4` |
| `cache_size` | `float` | 核矩阵缓存大小（MB）。默认 `200` | `200`、`500` |
| `max_iter` | `int` | 求解器最大迭代次数。`-1` 表示无限制。默认 `-1` | `-1`、`1000` |
| `decision_function_shape` | `str` | 多分类决策函数形状。`'ovr'`（One-vs-Rest）或 `'ovo'`（One-vs-One）。默认 `'ovr'` | `'ovr'`、`'ovo'` |
| `random_state` | `int` | 随机种子，控制概率估计等随机过程。当前设为 `42` | `42` |

### 示例代码

```python
model = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)
model.fit(X_train, y_train)
```

### 理解重点

- SVC 的参数主要集中在核函数配置上——`kernel`、`gamma`、`degree`、`coef0` 都与非线性映射相关。
- `C` 和 `gamma` 是最需要关注的超参数组合：$C$ 控制容错，$\gamma$ 控制核局部性——两者共同决定模型复杂度。
- `probability=False`（默认）意味着当前流水线不使用 `predict_proba` 也不画 ROC 曲线——这是 SVC 与其他分类算法分册在评估体系上的重要差异。
- SVC 的 `fit()` 是迭代优化——求解对偶二次规划问题（`libsvm` 的 SMO 算法），这与 GaussianNB（解析解）和 KNN（无训练）在计算特征上完全不同。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `support_vectors_` | `ndarray`，形状 `(n_sv, n_features)` | $SV = \{\mathbf{x}_i \mid \alpha_i > 0\}$ | 所有支持向量的特征矩阵 |
| `n_support_` | `ndarray`，形状 `(n_classes,)` | — | 每个类别的支持向量数量，二分类返回 `[n_sv_class0, n_sv_class1]` |
| `support_` | `ndarray`，形状 `(n_sv,)` | — | 支持向量在训练集中对应的索引 |
| `dual_coef_` | `ndarray`，形状 `(n_classes-1, n_sv)` | $\alpha_i y_i$ | 对偶系数与标签的乘积——非支持向量的项为 0 |
| `intercept_` | `ndarray`，形状 `(n_classes*(n_classes-1)/2,)` | $b$ | 决策函数的偏置项 |
| `classes_` | `ndarray`，形状 `(n_classes,)` | — | 模型识别到的类别标签列表 |
| `shape_fit_` | `tuple` | — | 训练数据特征维度 $d$，当前为 `(2,)` |

### 示例代码

```python
print(f"支持向量总数: {model.n_support_.sum()}")
print(f"各类别支持向量数: {model.n_support_.tolist()}")
print(f"截距: {model.intercept_}")
```

### 理解重点

- `n_support_` 是 SVC 最有教学意义的属性——它直接将"支持向量决定边界"这一理论概念量化为可观察的数字。
- `dual_coef_` 和 `support_vectors_` 组合起来完整定义了决策函数 $f(\mathbf{x}) = \sum \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$。
- 支持向量通常只占训练样本的 20%~40%——这是 SVC 稀疏性的直接体现，也是它在内存效率上优于 KNN 的原因之一。

## 4. 训练阶段的工程封装

除了 `SVC(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 帮助在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察当前模型拟合时间——SVC 的二次规划迭代比 GaussianNB 慢但比深度学习快 |
| `支持向量总数` 日志 | 快速查看模型依赖的关键样本规模 |
| `各类别支持向量数` 日志 | 观察两类样本对边界的贡献差异 |

### 理解重点

- 当前封装强调教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出 `n_support_`。
- 支持向量数量是最重要的日志输出——它直接反映了分类任务的难度和模型的稀疏程度。
- 这一层把"构建模型""训练模型""打印结果"收在一个函数里，方便流水线和文档复用。

## 常见坑

1. 误以为当前实现默认是线性核——源码明确使用 `kernel='rbf'`，是对同心圆数据的直接回应。
2. 只知道 `predict(...)`，却忽略 `n_support_` 和 `support_vectors_` 才是理解 SVC 行为的关键属性。
3. 把 `C` 当成"越大模型越强"的参数——$C \uparrow$ 容易过拟合，需要结合数据噪声水平调整。
4. 忘记 `probability=False` 的默认值——当前流水线不产生概率输出、不画 ROC 曲线，这是与逻辑回归等分册的评估差异。
5. 把训练函数和后续评估逻辑混在一起理解——`train_model` 只负责训练主模型，不负责混淆矩阵等诊断。

## 小结

- `train_model(...)` 是本仓库 SVC 的核心训练入口，是对 `sklearn.svm.SVC` 的薄封装。
- `SVC` 的关键参数是 `C`（软间隔容错）、`kernel`（核函数类型）和 `gamma`（RBF 核局部半径）。
- 训练完成后的核心属性：`n_support_`（支持向量数）、`support_vectors_`（支持向量特征）、`dual_coef_`（$\alpha_i y_i$）、`intercept_`（$b$）——四者共同定义决策函数。
- SVC 的 `fit()` 是真正的迭代优化（二次规划 SMO 算法），在训练效率上介于解析解模型和深度学习之间。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 SVC 流水线从数据复制到硬分类预测的完整步骤。
2. 理解主模型 (`model`)、二维可视化模型 (`model_2d`) 和学习曲线实例三者的职责边界。
3. 理解 SVC 的 `predict(...)` 基于决策函数 $\text{sign}(f(\mathbf{x}))$——而非概率阈值。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `svc_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `train_test_split(...)` | 函数 | 按 `stratify=y` 划分训练/测试集 |
| `StandardScaler` | 类 | 对特征做一致性标准化——对 RBF 核的距离计算至关重要 |
| `train_model(...)` | 函数 | 训练主 `SVC` 模型，求解二次规划并输出支持向量统计 |
| `model.predict(X_test_s)` | 方法 | 输出测试集类别预测——$\text{sign}(f(\mathbf{x}))$ 硬分类 |
| `PCA(n_components=2)` | 类 | 将标准化特征投影到 2 维，为决策边界可视化提供服务 |
| `model_2d` | 模型 | 在 PCA 2D 空间单独训练的 `SVC(kernel='rbf')`，专用于决策边界绘图 |

## 1. 流水线起点：复制数据并拆出特征/标签

### 示例代码

```python
data = svc_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `.copy()` 确保后续处理不修改在模块导入时已经加载的全局 `svc_data`。
- 当前任务是有监督二分类，$y$ 既参与训练 `fit(X_train_s, y_train)`，也参与评估（混淆矩阵）。
- 这一步只是数据准备，不涉及任何算法逻辑。

## 2. 训练/测试集切分

### 参数速览

适用函数：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 特征矩阵，形状 $(400, 2)$ | `X` |
| `y` | `Series` | 标签向量，二分类取值 $\{0, 1\}$ | `y` |
| `test_size` | `float` | 测试集占比。400 × 0.2 = 80 测试样本，320 训练样本 | `0.2` |
| `random_state` | `int` | 随机种子，保证切分可复现 | `42` |
| `stratify` | `array_like` | 传入 `y` 使训练/测试集内外圈比例与原始一致 | `y` |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- `stratify=y` 确保内外圈样本比例在训练/测试集中稳定——`factor=0.5` 下内外圈面积不等，分层采样尤为重要。
- 切分必须在标准化之前执行，否则测试集统计量会泄露到训练中。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(320, 2)$ | 训练特征矩阵，用于计算 $\mu_j, \sigma_j$ 并原地标准化 | `X_train` |
| `X_test` | `array_like`，形状 $(80, 2)$ | 测试特征矩阵，使用训练集统计量进行标准化变换 | `X_test` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，每个特征化为均值 0 标准差 1 | `X_train_s`、`X_test_s` |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 对 SVC 而言，标准化是硬性要求——RBF 核 $\exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ 直接依赖欧氏距离，不标准化会让距离计算被量纲主导。
- `gamma='scale'` 在标准化后自动计算 $\gamma = 1/(2 \cdot 1.0) = 0.5$，获得合理的默认核宽度。
- `fit_transform` 在训练集上同时计算统计量和变换，`transform` 在测试集上使用同一统计量。

## 4. 主模型训练与硬分类预测

### 参数速览

适用 API：`train_model(X_train_s, y_train)` → `model.predict(X_test_s)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 $(320, 2)$ | 标准化后的训练特征，传入 `SVC.fit()`——内部求解对偶二次规划 | `X_train_s` |
| `y_train` | `array_like` | 训练标签，内部转换为 $\{-1, +1\}$ 后参与优化 | `y_train` |
| `X_test_s` | `ndarray`，形状 $(80, 2)$ | 标准化后的测试特征，传入 `model.predict()` | `X_test_s` |
| 返回值 (`y_pred`) | `ndarray`，形状 $(80,)$ | 硬分类预测标签，来自 $\hat{y} = \text{sign}(f(\mathbf{x}))$ | `y_pred` |

### 示例代码

```python
model = train_model(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

### 理解重点

- `train_model(...)` 的 `fit()` 内部：求解对偶二次规划问题 → 确定支持向量集合 → 存储 $\alpha_i y_i$ 和 $b$。这是真正的迭代优化（SMO 算法），而非解析解。
- `predict(...)` 内部：对每个测试样本计算 $f(\mathbf{x}) = \sum_{i\in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$，取符号得到类别——仅支持向量参与计算。
- `y_pred` 是后续混淆矩阵的直接输入。与逻辑回归不同，当前 SVC 流水线不调用 `predict_proba(...)` 也不画 ROC 曲线。

## 5. 决策边界需要单独训练 `model_2d`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `pca` | `PCA(n_components=2, random_state=42)` | 将标准化特征投影到 2 维主成分空间 | `pca` |
| `X_all_s` | `ndarray`，形状 $(400, 2)$ | 全量标准化特征，用于 PCA 拟合 | `scaler.transform(X)` |
| `X_2d` | `ndarray`，形状 $(400, 2)$ | PCA 二维投影后的全量特征，用于画散点 | `pca.fit_transform(X_all_s)` |
| `model_2d` | `SVC(kernel='rbf', random_state=42)` | 在 PCA 二维空间单独训练的 SVC，专用于决策边界绘图 | `model_2d` |

### 示例代码

```python
pca = PCA(n_components=2, random_state=42)
X_all_s = scaler.transform(X)
X_2d = pca.fit_transform(X_all_s)
model_2d = SVC_Model(kernel="rbf", random_state=42)
model_2d.fit(pca.transform(X_train_s), y_train)
```

### 理解重点

- `model_2d` 不是主评估模型——它的唯一目的是在二维空间提供可绘制的决策边界。
- 主模型 `model` 训练在原始 2 维标准化空间（`X_train_s` 就是 2 维的），`model_2d` 训练在 PCA 降维后的 2 维空间——两者特征空间不同。
- 由于原始数据就是 2 维的，PCA 主要做旋转和缩放——决策边界图仍能较好反映原始空间的非线性边界形态。

## 6. 学习曲线使用新的模型实例

### 参数速览

适用函数：`plot_learning_curve(SVC_Model(kernel='rbf', random_state=42), X_train_s, y_train, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `SVC` | 新创建的 `SVC(kernel='rbf', random_state=42)` 实例，学习曲线内部会克隆和重复训练 | `SVC_Model(kernel='rbf', random_state=42)` |
| `X` | `ndarray`，形状 $(320, 2)$ | 标准化后的训练特征矩阵 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标，当前取 `"accuracy"` | `"accuracy"` |
| `cv` | `int` | 交叉验证折数，默认 `5` | `5` |

### 示例代码

```python
plot_learning_curve(
    SVC_Model(kernel="rbf", random_state=42),
    X_train_s,
    y_train,
    title="SVC 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 传入的是 `SVC_Model(kernel='rbf', random_state=42)` 新实例而非 `model`——学习曲线内部会通过 `learning_curve()` 函数多次克隆和训练模型。
- 学习曲线函数会按不同训练样本量（如 10%、33%、55%、78%、100%）做交叉验证，绘制训练得分和验证得分的变化趋势。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/articles/ML/svc/learning_curve.png)

## 常见坑

1. 忘记标准化是 SVC 的硬性要求——不标准化的 RBF 核等于让距离计算被特征量纲绑架。
2. 把 `model_2d` 误认为正式预测模型——它只在 PCA 空间训练，仅服务于决策边界可视化。
3. 混淆主模型（原始 2 维标准化空间）、二维可视化模型（PCA 空间）和学习曲线模型（CV 克隆）的三者职责。
4. 期望 `predict_proba(...)` 可用——当前流水线未启用 `probability=True`，SVC 默认只输出硬分类标签。

## 小结

- 当前 SVC 流水线的训练过程：复制数据 → 特征/标签拆分 → 切分（`stratify=y`）→ 标准化 → 求解对偶二次规划 → 硬分类预测。
- 三个模型实例各司其职：`model`（原始 2 维空间主评估）、`model_2d`（PCA 空间画边界）、`SVC_Model(...)`（学习曲线克隆）。
- SVC 的 `predict(...)` 基于 $\text{sign}(f(\mathbf{x}))$，不依赖概率阈值——这与逻辑回归的 Sigmoid → 0.5 阈值机制不同。
- 当前流水线不使用 ROC 曲线——SVC 默认 `probability=False`，启用概率输出需要额外 Platt scaling 开销。

# 评估与诊断

## 本章目标

1. 明确当前仓库 SVC 实现的三种评估手段及其分别回答的问题。
2. 理解 2×2 混淆矩阵和 PCA 决策边界图在同心圆二分类场景下的解读方式。
3. 理解当前 SVC 流水线为何不使用 ROC 曲线——与 `probability=False` 的默认配置直接相关。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出，由 $\text{sign}(f(\mathbf{x}))$ 硬分类产生 |
| `plot_confusion_matrix(...)` | 函数 | 绘制 2×2 二分类混淆矩阵 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界——对同心圆数据最能体现模型非线性能力 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |
| `model.n_support_` | 属性 | 各类别支持向量数量——SVC 独有的诊断信息 |

## 1. 当前仓库的评估入口

当前 SVC 流水线里的主要诊断手段有三个：

1. 混淆矩阵 —— 回答"分对了多少？两类各有多少被误分类？"
2. PCA 2D 决策边界图 —— 回答"RBF 核能否正确画出弯曲边界将内外圈分离？"
3. 学习曲线 —— 回答"更多训练样本还能提升表现吗？"

### 示例代码

```python
y_pred = model.predict(X_test_s)

plot_confusion_matrix(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 当前 SVC 流水线**不使用 ROC 曲线**——因为 SVC 默认 `probability=False`，不启用概率估计（启用 `probability=True` 需额外 5 折交叉验证 Platt scaling，显著增加训练耗时）。
- SVC 独有的诊断信息是 `n_support_`——支持向量数量直接反映分类任务的难度和模型的稀疏性。
- 三种可视化分别回答不同问题，不能互相替代。

## 2. 混淆矩阵能观察什么

对于二分类任务，混淆矩阵 $\mathbf{C}$ 是一个 $2 \times 2$ 矩阵：

$$
C = \begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}
$$

### 参数速览

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $\{0, 1\}$ | `y_test` |
| `y_pred` | `array_like`，形状 `(n_samples,)` | 模型硬分类预测标签，来自 $\text{sign}(f(\mathbf{x}))$ | `y_pred` |
| `normalize` | `bool` 或 `str` | 归一化方式。`True`/`'true'` 按行（真实类别），`'pred'` 按列，`'all'` 按全体。默认 `False` | `True`、`'true'` |

### 示例代码

```python
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    title="SVC 混淆矩阵",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 在同心圆二分类上，混淆矩阵最直观地反映 RBF 核 SVC 是否正确分离了内外圈。
- 对于 `noise=0.1` 的数据，少量样本可能跨越环形边界进入错误区域——这些误分类会出现在非对角线上。
- 混淆矩阵已经隐式包含计算 Accuracy、Precision、Recall、F1 所需的所有信息（TP、TN、FP、FN），但当前流水线未显式计算这些指标。

## 3. PCA 2D 决策边界图能观察什么

### 参数速览

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model_2d` | `SVC(kernel='rbf')` | 在 PCA 二维空间单独训练的 SVC，共享主模型的 RBF 核配置 | `model_2d` |
| `X_2d` | `ndarray`，形状 `(n_samples, 2)` | 标准化后 PCA 投影到二维的全量特征 | `X_2d` |
| `y` | `array_like`，形状 `(n_samples,)` | 全量标签数组，用于散点的真实类别着色 | `y.values` |

### 示例代码

```python
plot_decision_boundary(
    model_2d,
    X_2d,
    y.values,
    title="SVC 决策边界 (PCA 2D)",
    dataset_name=DATASET,
)
```

### 理解重点

- 这是 SVC 分册最重要的一张图——它直观展示了 RBF 核能否生成弯曲的环形分类边界。
- 如果 RBF 核工作正常，决策边界应呈现弯曲形态，将内圈区域与外圈区域正确分离——这是非线性核能力的视觉见证。
- 线性核（`kernel='linear'`）的决策边界将是一条直线，无法分离同心圆——这在实验对比中是最有说服力的教学画面。
- 由于原始数据本身就是二维的（$x_1$、$x_2$），PCA 主要做旋转和缩放——决策边界图基本反映原始空间的真实几何形态。

## 4. 学习曲线能观察什么

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `SVC` | 新创建的 `SVC(kernel='rbf', random_state=42)` 实例——内部会通过 CV 克隆并重复训练 | `SVC_Model(kernel='rbf', random_state=42)` |
| `X` | `ndarray`，形状 `(n_train, 2)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分指标，默认 `"accuracy"` | `"accuracy"` |
| `cv` | `int` | 交叉验证折数，默认 `5` | `5` |
| `train_sizes` | `array_like` | 训练样本量的递增序列，默认为 `np.linspace(0.1, 1.0, 5)` | `[0.1, 0.33, 0.55, 0.78, 1.0]` |

### 示例代码

```python
plot_learning_curve(
    SVC_Model(kernel="rbf", random_state=42),
    X_train_s,
    y_train,
    title="SVC 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- SVC 是参数化模型（参数由支持向量决定），样本量增加时支持向量集合逐渐稳定——学习曲线反映这一收敛过程。
- 在同心圆数据中，一定数量的样本是构造环形边界所必需的——样本太少时模型可能找不到正确的圆形边界形状。
- 训练得分与验证得分之间的差距反映 $C=1.0$ 和 $\gamma$ 配置下的过拟合/欠拟合倾向。

## 5. 当前实现中尚未纳入但常见的评估手段

| 手段 | 公式/说明 | 不在当前流水线中的原因 |
|---|---|---|
| ROC 曲线 / AUC | $\text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}}$，$\text{FPR} = \frac{\text{FP}}{\text{FP}+\text{TN}}$ | SVC 默认 `probability=False`——启用需额外 Platt scaling 交叉验证 |
| 准确率（Accuracy） | $\frac{\text{TP}+\text{TN}}{\text{TP}+\text{TN}+\text{FP}+\text{FN}}$ | 未显式计算，但混淆矩阵已包含所需全部信息 |
| 精确率（Precision） | $\frac{\text{TP}}{\text{TP}+\text{FP}}$ | 同上——可从混淆矩阵直接推导 |
| 召回率（Recall） | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ | 同上 |
| F1 分数 | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | 同上 |

### 理解重点

- 当前仓库未在 SVC 流水线中显式打印 accuracy、precision、recall、f1——文档可以提到它们作为扩展方向，但不可写成"当前源码已在计算"。
- SVC 在 `probability=False` 下的评估体系天然比逻辑回归少一个维度（无概率输出、无 ROC）——这不是缺陷，而是 SVC 设计哲学（关注决策函数符号 $f(\mathbf{x})$，而非概率校准）的体现。

## 评估图表

![混淆矩阵](https://img.yumeko.site/file/articles/ML/svc/confusion_matrix.png)

## 常见坑

1. 期望 ROC 曲线出现在 SVC 评估中——当前流水线不启用概率估计（`probability=False`），没有 ROC。
2. 把 PCA 决策边界图误认为原始特征空间决策面的完整表达——虽然当前数据本身是二维的，但 PCA 的旋转仍可能改变视角。
3. 只看混淆矩阵的绝对数值，忽略 RBF 核决策边界的弯曲形状——后者才是理解 SVC 能力的核心视觉证据。
4. 把当前仓库未显式计算的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 SVC 的评估：混淆矩阵看错误分布（2×2 二分类），PCA 决策边界图看 RBF 核弯曲边界的形状，学习曲线看样本量对支持向量收敛的影响。
- SVC 没有 `predict_proba` 驱动的 ROC 曲线评估——这是它与其他分类算法分册在评估体系上的关键差异。
- 对于同心圆数据，PCA 决策边界图中的环形边界是最有说服力的评估——它直接展示了 RBF 核将线性不可分问题转化为可解问题的能力。

# 工程实现

## 本章目标

1. 从工程角度看清 SVC 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解 SVC 工程实现与其他分类算法的关键差异——支持向量统计日志、无 ROC 评估、标准化是硬性要求。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.svc()` 生成同心圆二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `svc_data` |
| 训练封装 | `model_training/classification/svc.py` | 构建并训练 `SVC(kernel='rbf')`，打印支持向量统计 |
| 流水线入口 | `pipelines/classification/svc.py` | 组织切分、标准化、训练、预测与可视化的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制并保存 2×2 二分类混淆矩阵图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 绘制并保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制并保存训练/验证得分曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.svc
```

### 理解重点

- 这个命令串起当前 SVC 分册中最核心的工程流程。
- 依次完成：数据复制 → 特征/标签拆分 → 切分 → 标准化 → SVC `fit()`（求解对偶二次规划）→ 硬分类预测 → 三种可视化。
- 对大多数读者来说，`pipelines/classification/svc.py` 是理解工程实现的最佳起点。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = svc_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 标准化（仅训练集上 fit）—— RBF 核的硬性要求
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. 求解对偶二次规划 & 硬分类预测
    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # 5. 可视化诊断（混淆矩阵、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(SVC_Model(kernel='rbf', ...), X_train_s, y_train, ...)
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的优化在 `SVC.fit()`（`libsvm` 的 SMO 算法求解对偶二次规划）中。
- 数据流是单向的：数据 → 切分 → 标准化 → 二次规划求解 → 硬分类预测 → 评估。
- 与逻辑回归流水线的关键差异：无 `predict_proba` 调用、无 ROC 曲线——因为 SVC 默认 `probability=False`。

## 3. 训练模块负责什么

`model_training/classification/svc.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)` 实例
2. 调用 `model.fit(X_train, y_train)`——`libsvm` SMO 算法求解对偶二次规划
3. 打印训练日志：耗时、支持向量总数、各类别支持向量数
4. 返回训练完成的主模型对象

### 参数速览

适用函数：`train_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，传入 `SVC.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，内部转换为 $\{-1, +1\}$ | `y_train` |
| `C` | `float` | 软间隔惩罚系数。默认 `1.0` | `0.1`、`1.0`、`10.0` |
| `kernel` | `str` | 核函数类型。默认 `'rbf'` | `'rbf'`、`'linear'`、`'poly'` |
| `gamma` | `float` 或 `str` | RBF 核系数。默认 `'scale'` | `'scale'`、`'auto'`、`0.1` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| 返回值 | `SVC` | 已完成 `fit()` 的模型对象，含 `n_support_`、`support_vectors_` 等属性 | — |

### 理解重点

- SVC 的 `fit()` 本质是迭代优化——`libsvm` 的 SMO 算法求解对偶二次规划，在计算特征上既不同于 GaussianNB（解析解）也不同于 KNN（无训练）。
- 训练日志中的 `n_support_` 是 SVC 最有教学意义的输出——它直接将"支持向量决定边界"理论量化为可观察的数字。

## 4. 三类评估模块分别负责什么

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 2×2 二分类混淆矩阵图（PNG） |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | PCA 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `SVC_Model(kernel='rbf', ...)`、`X_train_s`、`y_train` | 训练/验证得分随样本量变化曲线（PNG） |

### 理解重点

- 三类可视化都不是训练的一部分，而是训练完成后的诊断步骤。
- 决策边界图依赖额外训练的 `model_2d`（PCA 空间中 `SVC(kernel='rbf')`）——与主模型配置一致但特征空间不同。
- 当前 SVC 流水线**不使用 ROC 曲线模块**（`result_visualization/roc_curve.py`）——这是 SVC 与其他分类分册在评估体系上的重要差异。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `svc_data` | `data_generation/classification.py` | `pipelines/classification/svc.py` |
| `model`（主模型） | `train_model(...)` | `predict` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `model_2d` | `SVC_Model(kernel='rbf').fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/svc/` 目录 |

### 理解重点

- 数据流向单向、无循环依赖，每个模块可以独立测试和替换。
- SVC 的流水线结构与逻辑回归、KNN 高度一致——但缺少 `predict_proba` → ROC 评估分支。
- `model_2d` 与主模型共享 `kernel='rbf'` 配置，确保决策边界图反映的是同类核函数的表现。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `SVC 分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、支持向量总数、各类别支持向量数 | 查看二次规划求解耗时和模型稀疏性 |
| 混淆矩阵图 | `outputs/svc/confusion_matrix.png` | 观察内外圈误分类方向 |
| 决策边界图 | `outputs/svc/decision_boundary.png` | 观察 RBF 核弯曲边界的形状——环形嵌套分离效果 |
| 学习曲线图 | `outputs/svc/learning_curve.png` | 诊断样本量对支持向量收敛的影响 |

### 理解重点

- 训练日志中的 `n_support_` 是 SVC 独有的信息——它直接揭示模型依赖了多少关键样本来确定环形边界。
- 与逻辑回归输出 `coef_` 不同，SVC 输出 `n_support_` 反映的是稀疏解的特征——支持向量越少，模型越简洁。
- 决策边界图是 SVC 分册最具教学说服力的输出——环形边界直观展示了 RBF 核的非线性能力。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/svc.py` — 入口，了解整体流程（注意无 ROC 分支）
2. 再看 `model_training/classification/svc.py` — 训练封装，理解超参数和支持向量日志
3. 再看 `result_visualization/confusion_matrix.py` — 基础分类结果评估（2×2 矩阵）
4. 再看 `result_visualization/decision_boundary.py` — PCA 空间 RBF 核边界可视化
5. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
6. 最后回到 `data_generation/classification.py` — 理解同心圆数据生成参数

### 理解重点

- 从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 这个顺序对应数据流方向：数据 → 标准化 → 二次规划求解 → 硬分类预测 → 评估。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/articles/ML/svc/result_display.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的优化在 `SVC.fit()`（`libsvm`）中。
2. 不区分"主模型"（原始 2 维标准化空间）、"二维可视化模型"（PCA 空间）和"学习曲线模型实例"（CV 克隆）的职责边界。
3. 忽略 `n_support_` 和 `各类别支持向量数` 的日志输出——这是理解 SVC 稀疏性的入口。
4. 期望 ROC 曲线评估——当前流水线未调用 `plot_roc_curve`，因为 SVC 默认不启用概率估计。
5. 只看单个文件，不顺着调用链理解整体执行流程。

## 小结

- 当前 SVC 工程实现采用清晰的模块分层：数据生成 → 训练封装 → 流水线编排 → 结果可视化（三种评估）。
- `run()` 负责串联，`train_model(...)` 负责二次规划求解（SMO 迭代优化），各可视化函数负责结果展示与诊断。
- SVC 在工程上最不同于其他分类算法的地方：标准化是硬性要求（RBF 核距离敏感）、训练日志输出 `n_support_`（稀疏性）、无 `predict_proba` / ROC 评估分支（`probability=False`）。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 SVC 实现。
2. 给出继续深入阅读支持向量机与核方法的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/svc.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_circles(noise=0.1, factor=0.5)` 同心圆数据必须依赖 RBF 核而非线性核？RBF 核 $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$ 与线性核 $\mathbf{x}^T \mathbf{z}$ 在几何上有何本质差异？
3. 当前 `train_model(...)` 中的 `C`、`kernel`、`gamma` 分别控制什么？$C$ 与软间隔目标函数 $C\sum\xi_i$ 的关系是什么？`gamma='scale'` 时 $\gamma = 1/(d \cdot X.var())$ 的实际意义是什么？
4. 为什么 `model.n_support_` 对理解 SVC 很重要？它与 KKT 条件中的 $\alpha_i > 0$ 有什么关系？支持向量多说明什么？少说明什么？
5. 为什么当前 SVC 流水线没有 ROC 曲线评估？启用 ROC 需要做什么额外配置？代价是什么？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？主模型与 `model_2d` 的核函数配置有何异同？
7. 为什么 SVC 的标准化不是可选的优化手段而是硬性要求？RBF 核中的 $\|\mathbf{x} - \mathbf{z}\|^2$ 如何被特征量纲影响？

## 练习方向

### 1. 改动 $C$

- 把 `C=1.0` 改成 `0.01`、`0.1`、`1.0`、`10.0`、`100.0`
- 观察变化：
  - `n_support_` 的数量——$C$ 越小（间隔越宽），支持向量通常越多
  - 混淆矩阵中内外圈的正确率变化
  - 决策边界的弯曲程度——$C$ 小时边界更平滑但可能过于简单，$C$ 大时边界更精细但可能过拟合噪声
  - 学习曲线中训练得分与验证得分的差距——$C$ 大时容易过拟合（两者差距大）
- 核心理解：$C$ 越大 $\approx$ 正则越弱——与逻辑回归中 $C = 1/\lambda$ 的关系一致

### 2. 改动 `gamma`

- 把 `gamma='scale'` 改成 `0.01`、`0.1`、`1.0`、`10.0`、`'auto'`
- 观察变化：
  - RBF 核边界的弯曲精细程度——$\gamma$ 越大边界越"崎岖"，单个支持向量影响范围越小
  - 模型的过拟合/欠拟合倾向——$\gamma$ 过大时每个支持向量只影响周围很小的区域
  - 支持向量数量的变化
- 核心理解：$C$ 和 $\gamma$ 的联合效应——两者共同决定模型复杂度：$C \uparrow + \gamma \uparrow$ 最容易过拟合

### 3. 改用线性核

- 把 `kernel='rbf'` 改为 `kernel='linear'`
- 对比变化：
  - 决策边界的形状——从环形曲线变为一条直线，无法分离内外圈
  - 混淆矩阵——大量误分类集中在边界线上
  - 准确性上限——线性核对同心圆数据的理论最佳准确率约为 50%（一条直线最多切中一半的点）
- 核心理解：核函数不是"更复杂就更强"，而是必须匹配数据形状——这是 SVC 最核心的教学启示

### 4. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test` 训练
- 对比变化：
  - 决策边界的形状——RBF 核的距离计算失真，边界可能完全偏离正确的环形结构
  - 混淆矩阵中的误分类大幅增加
  - `gamma='scale'` 计算出的 $\gamma$ 值因量纲而变化
- 体会：标准化不是锦上添花——对 RBF 核 SVC 而言，它是核函数几何意义正确的前提

### 5. 观察支持向量数量与数据噪声的关系

- 修改 `make_circles(noise=...)` 的 `noise` 参数（`0`、`0.05`、`0.1`、`0.2`）
- 观察 `n_support_` 的变化趋势——噪声越大，两类样本越纠缠，支持向量通常越多
- 核心理解：支持向量数量间接反映了数据的线性不可分程度——支持向量越多，说明两类越"纠结"

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`SVC` | 完整构造器参数列表（`C`、`kernel`、`gamma`、`degree`、`probability` 等）、属性（`support_vectors_`、`n_support_`、`dual_coef_`、`intercept_`）与方法说明 |
| 2 | scikit-learn 官方文档：`make_circles` | 同心圆数据生成器的 `n_samples`、`noise`、`factor` 等参数说明 |
| 3 | scikit-learn 用户指南：SVM | C-SVC 的完整数学推导、核函数对比、多分类策略（OvO/OvR）与实用调参指南 |
| 4 | Cortes, C. and Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273-297. | SVM 的原始论文——最大间隔思想、软间隔引入和核技巧的源头 |

- scikit-learn `SVC`：https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- scikit-learn `make_circles`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
- scikit-learn 用户指南 SVM：https://scikit-learn.org/stable/modules/svm.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 SVC 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），且对 RBF 核是硬性要求（距离计算不能被量纲绑架）
  - 最大间隔 $\min\frac{1}{2}\|\mathbf{w}\|^2$ → 软间隔 $+C\sum\xi_i$ → 对偶 + 内积 → RBF 核 → 支持向量 → 决策函数 $f(\mathbf{x})$ 的完整数学链
  - $C$ 越大正则越弱（与逻辑回归一致），$\gamma$ 越大核窗口越窄——两者联合决定复杂度
  - 线性核对同心圆数据必然失败——这是 SVC 分册最有教学启示的实验对比
  - `n_support_` 是 SVC 独有的教学窗口——通过它可以看到模型依赖了多少关键样本
  - SVC 默认 `probability=False`——当前流水线无概率输出、无 ROC 曲线，这是与其他分类分册的核心差异
  - `model`（原始 2 维标准化空间）、`model_2d`（PCA 空间）和学习曲线实例的职责边界
