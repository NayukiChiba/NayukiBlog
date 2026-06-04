---
title: Sklearn 全指南
date: 2026-01-20
category: MachineLearning/Basic
tags:
  - Python
  - 基础
description: Scikit-learn (sklearn) 库实践指南，涵盖数据预处理、特征工程、Pipeline构建、模型选择、交叉验证、超参数调优、评估指标及常用模型（线性模型、树模型、SVM等）的详细应用与可视化。
image: https://img.yumeko.site/file/blog/cover/1780581875425.webp
status: published
---

# Scikit-learn 入门

## 本章目标

1. 掌握 sklearn 内置数据集的加载方式与返回结构
2. 学会使用 `make_*` 系列函数生成人工数据集
3. 理解 `train_test_split` 的分层抽样机制
4. 走通 KNN 模型的完整流程：创建 → 训练 → 预测 → 评估
5. 熟悉 sklearn 估计器的通用方法与属性命名约定

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `datasets.load_iris()` | 函数 | 加载鸢尾花数据集，返回 `Bunch` 对象 |
| `datasets.make_classification(...)` | 函数 | 生成分类人工数据集 |
| `datasets.make_regression(...)` | 函数 | 生成回归人工数据集 |
| `datasets.make_blobs(...)` | 函数 | 生成聚类人工数据集 |
| `train_test_split(...)` | 函数 | 按比例划分训练/测试集 |
| `KNeighborsClassifier(n_neighbors)` | 构造器 | K 近邻分类器 |
| `estimator.fit(X, y)` | 方法 | 训练模型 |
| `estimator.predict(X)` | 方法 | 预测标签 |
| `estimator.get_params()` / `.set_params()` | 方法 | 获取/设置超参数 |
| `clone(estimator)` | 函数 | 克隆模型（不复制训练状态） |

## 1. 加载内置数据集

### `datasets.load_iris`

#### 作用

sklearn 提供多个经典数据集，通过 `datasets.load_*()` 直接加载。返回值是 `Bunch` 对象（类字典），包含 `data`、`target`、`feature_names`、`target_names` 属性。`return_X_y=True` 直接返回 `(X, y)`，`as_frame=True` 返回 Pandas DataFrame。

#### 重点方法

```python
datasets.load_iris(*, return_X_y=False, as_frame=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `return_X_y` | `bool` | `True` 时直接返回 `(X, y)` 元组，默认为 `False` | `True` |
| `as_frame` | `bool` | `True` 时返回 DataFrame 格式，默认为 `False` | `True` |

内置数据集速览：

| 数据集 | 函数 | 类型 | 样本/特征 |
|---|---|---|---|
| 鸢尾花 | `load_iris()` | 分类 (3 类) | 150 / 4 |
| 乳腺癌 | `load_breast_cancer()` | 二分类 | 569 / 30 |
| 手写数字 | `load_digits()` | 分类 (10 类) | 1797 / 64 |
| 糖尿病 | `load_diabetes()` | 回归 | 442 / 10 |

#### 示例代码

```python
from sklearn import datasets

iris = datasets.load_iris()
print(f"特征矩阵形状: {iris.data.shape}")
print(f"目标向量形状: {iris.target.shape}")
print(f"特征名称: {iris.feature_names}")

# 直接返回 X, y
X, y = datasets.load_iris(return_X_y=True)
print(f"X={X.shape}, y={y.shape}")

# DataFrame 格式
iris_df = datasets.load_iris(as_frame=True)
print(f"\n{iris_df.frame.head()}")
```

#### 输出

```text
特征矩阵形状: (150, 4)
目标向量形状: (150,)
特征名称: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X=(150, 4), y=(150,)

   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
```

#### 理解重点

- `Bunch` 对象可像字典一样访问：`iris['data']` 等价于 `iris.data`
- `data` 形状 `(n_samples, n_features)`，`target` 形状 `(n_samples,)`
- `return_X_y=True` 是最简洁的加载方式
- `as_frame=True` 在数据探索阶段非常方便——列名自动对应特征名称

## 2. 生成人工数据集

### `datasets.make_classification` / `make_regression` / `make_blobs`

#### 作用

`make_*` 系列函数用于生成可控的人工数据集，常用于算法验证和教学。`make_classification` 生成分类数据，`make_regression` 生成回归数据，`make_blobs` 生成聚类数据。

#### 重点方法

```python
datasets.make_classification(n_samples=100, n_features=20, n_informative=2, ...)
datasets.make_regression(n_samples=100, n_features=100, ...)
datasets.make_blobs(n_samples=100, n_features=2, centers=None, ...)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 样本数量 | `1000` |
| `n_features` | `int` | 总特征数 | `20` |
| `n_informative` | `int` | 有信息量的特征数（`make_classification`） | `10` |
| `n_redundant` | `int` | 冗余特征数（`make_classification`） | `5` |
| `n_classes` | `int` | 类别数（`make_classification`） | `3` |
| `centers` | `int` | 聚类中心数（`make_blobs`） | `4` |
| `cluster_std` | `float` | 簇的标准差（`make_blobs`） | `1.0` |
| `noise` | `float` | 噪声标准差（`make_regression`） | `10` |
| `random_state` | `int` | 随机种子，保证可复现 | `42` |

#### 示例代码

```python
import numpy as np
from sklearn.datasets import (
    make_classification, make_regression, make_blobs
)

X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=3, random_state=42
)
print(f"分类数据: X={X_clf.shape}, 各类别数量={np.bincount(y_clf)}")

X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=10, random_state=42
)
print(f"回归数据: X={X_reg.shape}, y 范围=[{y_reg.min():.1f}, {y_reg.max():.1f}]")

X_blob, y_blob = make_blobs(
    n_samples=500, centers=4, cluster_std=1.0, random_state=42
)
print(f"聚类数据: X={X_blob.shape}, y={np.unique(y_blob)}")
```

#### 输出

```text
分类数据: X=(1000, 20), 各类别数量=[334 333 333]
回归数据: X=(1000, 10), y 范围=[-609.8, 571.4]
聚类数据: X=(500, 2), y=[0 1 2 3]
```

#### 理解重点

- `n_informative + n_redundant <= n_features`——剩余特征为随机噪声
- `make_moons` 和 `make_circles` 生成线性不可分数据——适合验证非线性模型
- `random_state` 保证每次生成相同数据——实验可复现
- 人工数据的优势：已知 ground truth，便于验证模型行为

## 3. 数据划分

### `train_test_split`

#### 作用

将数据划分为训练集和测试集。`stratify=y` 进行分层抽样——确保训练集和测试集的类别比例与原数据一致。`random_state` 保证每次划分结果相同。

#### 重点方法

```python
train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                 shuffle=True, stratify=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*arrays` | `array_like` | 待划分的数据（可多个） | `X, y` |
| `test_size` | `float` 或 `int` | 测试集比例（0~1）或样本数 | `0.3` |
| `train_size` | `float` 或 `int` | 训练集比例，默认 `1 - test_size` | `0.7` |
| `random_state` | `int` | 随机种子 | `42` |
| `shuffle` | `bool` | 划分前是否打乱，默认为 `True` | `True` |
| `stratify` | `array_like` | 按此数组类别比例分层抽样 | `y` |

#### 示例代码

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本, 类别分布: {np.bincount(y_train)}")
print(f"测试集: {X_test.shape[0]} 样本, 类别分布: {np.bincount(y_test)}")
```

#### 输出

```text
训练集: 105 样本, 类别分布: [35 35 35]
测试集: 45 样本, 类别分布: [15 15 15]
```

#### 理解重点

- `stratify=y` 在类别不平衡时尤为重要——避免某些类别在测试集中缺失
- 返回值顺序：`X_train, X_test, y_train, y_test`（先 X 后 y，先 train 后 test）
- 同一个 `random_state` 保证实验可复现

## 4. 第一个模型（KNN）

### `KNeighborsClassifier`

#### 作用

sklearn 所有模型遵循统一 API 流程：**创建 → fit → predict → score**。KNeighborsClassifier 通过 k 个最近邻投票决定类别——是最简单直观的分类算法。

#### 重点方法

```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', metric='minkowski', p=2)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 邻居数量 k，默认为 `5` | `3` |
| `weights` | `str` | `'uniform'` 等权投票 / `'distance'` 距离加权，默认为 `'uniform'` | `'distance'` |
| `metric` | `str` | 距离度量方式，默认为 `'minkowski'` | `'euclidean'` |
| `p` | `int` | Minkowski 距离的 p 值，`2` = 欧氏距离，默认为 `2` | `1` |

#### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"准确率: {knn.score(X_test, y_test):.4f}")
print(f"前三样本预测概率:\n{knn.predict_proba(X_test[:3])}")
```

#### 输出

```text
准确率: 0.9778
前三样本预测概率:
[[0.  1.  0. ]
 [0.  1.  0. ]
 [0.  0.8 0.2]]
```

#### 理解重点

- sklearn 统一 API 核心三步：`fit` → `predict` → `score`
- `score` 方法内部调用 `predict` 再计算指标——是一个便捷方法
- `predict_proba()` 返回每行概率之和为 1，列顺序对应 `classes_`
- KNN 的 k 值关键：k 太小过拟合，k 太大欠拟合

## 5. 估计器通用方法

### `get_params` / `set_params` / `clone`

#### 作用

sklearn 所有估计器共享一套通用方法和属性命名约定。`get_params()` 查看超参数，`set_params()` 修改超参数，`clone()` 克隆模型结构但不复制训练状态。训练后产生的属性以下划线 `_` 结尾（`classes_`、`n_features_in_` 等）。

#### 重点方法

```python
estimator.get_params(deep=True)
estimator.set_params(**params)
clone(estimator)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `deep` | `bool` | `True` 时递归获取嵌套对象的参数，默认为 `True` | `True` |
| `**params` | `dict` | 要修改的参数名=值 | `n_neighbors=3` |
| `estimator` | `estimator` | 待克隆的估计器对象 | `knn` |

训练后关键属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `classes_` | `ndarray` | 训练后类别标签 |
| `n_features_in_` | `int` | 训练时输入的特征数 |
| `feature_names_in_` | `ndarray` | 训练时输入的特征名称 |

#### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# get_params() — 获取全部超参数
print(f"n_neighbors: {knn.get_params()['n_neighbors']}")

# set_params() — 修改后需重新 fit
knn.set_params(n_neighbors=3, weights="distance")
knn.fit(X_train, y_train)
print(f"修改后: n_neighbors={knn.n_neighbors}, weights={knn.weights}")

# clone() — 克隆超参数，不复制训练状态
knnClone = clone(knn)
print(f"克隆模型已训练: {hasattr(knnClone, 'classes_')}")  # False

# 训练后属性（带 _ 后缀）
print(f"classes_: {knn.classes_}, n_features_in_: {knn.n_features_in_}")
```

#### 输出

```text
n_neighbors: 5
修改后: n_neighbors=3, weights=distance
克隆模型已训练: False
classes_: [0 1 2], n_features_in_: 4
```

#### 理解重点

- **超参数**（创建时传入）vs **训练后属性**（带 `_` 后缀）：这是 sklearn 的核心命名约定
- `get_params()` 返回完整参数字典——包括默认值
- `set_params()` 修改后需重新 `fit` 才生效
- `clone()` 常用于交叉验证——每折需要一个"干净"的模型

## 常见坑

1. `train_test_split` 返回值顺序是 `X_train, X_test, y_train, y_test`——不是 train 全部在前
2. 忘记 `stratify=y`——类别不平衡时不分层可能导致测试集缺少某些类别
3. `set_params()` 后未重新训练——修改参数不会自动重新 fit
4. `clone()` vs 直接赋值——`clone()` 只复制超参数，直接赋值是引用同一对象
5. `predict_proba()` 不是所有模型都有——如 `LinearSVC` 不直接支持
6. `load_boston()` 已在 sklearn 1.2+ 中移除——使用 `fetch_openml` 替代

## 小结

- sklearn 内置数据集通过 `datasets.load_*()` 加载——`return_X_y=True` 最简洁
- `make_*` 系列生成可控人工数据——适合算法验证和教学
- `train_test_split` 是数据划分标准方法——务必使用 `stratify` 保持类别比例
- sklearn 统一 API：`fit` → `predict` → `score`——所有模型通用
- 超参数用 `get_params()` / `set_params()` 管理——训练后属性以 `_` 结尾

# sklearn 预处理

## 本章目标

1. 理解四种常见缩放器在异常值场景下的行为差异
2. 掌握 `StandardScaler` 与 `MinMaxScaler` 的核心参数与逆变换
3. 理解 `PowerTransformer` 处理偏态分布的作用与限制
4. 学会区分 `LabelEncoder` 与 `OneHotEncoder` 的适用场景
5. 掌握缺失值填充与 `ColumnTransformer` 组合预处理流程

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `StandardScaler()` | 构造器 | 标准化到均值 0、方差 1 |
| `MinMaxScaler(feature_range)` | 构造器 | 线性缩放到指定区间 |
| `RobustScaler()` | 构造器 | 使用中位数与 IQR，抗异常值 |
| `PowerTransformer(method)` | 构造器 | 幂变换改善偏态分布 |
| `LabelEncoder()` | 构造器 | 一维标签编码 |
| `OneHotEncoder(...)` | 构造器 | 类别特征独热编码 |
| `SimpleImputer(strategy)` | 构造器 | 统计量或常量填充缺失值 |
| `KNNImputer(n_neighbors)` | 构造器 | 基于近邻估计缺失值 |
| `ColumnTransformer(transformers)` | 构造器 | 按列类型组合预处理流水线 |

## 1. 缩放器对比

### StandardScaler / MinMaxScaler / RobustScaler / MaxAbsScaler

#### 作用

同一组数据在不同缩放器下的分布明显不同——尤其存在异常值时。`StandardScaler` 与 `MinMaxScaler` 对异常值更敏感，`RobustScaler` 使用中位数和 IQR 更稳健，`MaxAbsScaler` 不平移数据中心适合保持稀疏结构。

#### 重点方法

```python
StandardScaler(*, copy=True, with_mean=True, with_std=True)
MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
MaxAbsScaler(*, copy=True)
# 核心方法：fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `copy` | `bool` | 是否复制输入数据，默认为 `True` | `True` |
| `with_mean` | `bool` | 对特征做中心化（StandardScaler），默认为 `True` | `True` |
| `with_std` | `bool` | 按标准差缩放（StandardScaler），默认为 `True` | `True` |
| `feature_range` | `tuple[float, float]` | 目标缩放区间（MinMaxScaler），默认为 `(0, 1)` | `(-1, 1)` |
| `clip` | `bool` | 推理阶段是否截断超范围值（MinMaxScaler），默认为 `False` | `True` |
| `with_centering` | `bool` | 使用中位数做中心化（RobustScaler），默认为 `True` | `True` |
| `with_scaling` | `bool` | 按分位间距缩放（RobustScaler），默认为 `True` | `True` |
| `quantile_range` | `tuple[float, float]` | IQR 分位区间（RobustScaler），默认为 `(25.0, 75.0)` | `(10.0, 90.0)` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

np.random.seed(42)
X = np.random.randn(100, 2) * 10 + 50
X[0] = [200, 200]  # 注入异常值

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
}

for name, scaler in scalers.items():
    Xs = scaler.fit_transform(X)
    print(f"{name}: 范围=[{Xs.min():.2f}, {Xs.max():.2f}]")
```

#### 输出

```text
StandardScaler: 范围=[-1.76, 9.59]
MinMaxScaler: 范围=[0.00, 1.00]
RobustScaler: 范围=[-2.57, 11.58]
MaxAbsScaler: 范围=[0.17, 1.00]
```

#### 理解重点

- 若后续模型依赖距离（KNN、SVM），缩放是必要步骤
- 存在强异常值时优先比较 `RobustScaler` 与其他方案
- 缩放器选择本质是分布假设选择——不是固定套路

## 2. StandardScaler 详解

### `StandardScaler`

#### 作用

学习训练集的 `mean_` 与 `scale_`，对每列做 $z = (x - \mu) / \sigma$。`fit_transform` 用于训练阶段，测试集应使用 `transform`。`inverse_transform` 可将标准化结果还原到原始尺度。

#### 重点方法

```python
StandardScaler(*, copy=True, with_mean=True, with_std=True)
# fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `copy` | `bool` | 是否复制输入数据，默认为 `True` | `True` |
| `with_mean` | `bool` | 对每列减去均值，默认为 `True` | `True` |
| `with_std` | `bool` | 对每列除以标准差，默认为 `True` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `mean_` | `ndarray` | 每列均值 |
| `scale_` | `ndarray` | 每列标准差 |
| `var_` | `ndarray` | 每列方差 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

scaler = StandardScaler()
XScaled = scaler.fit_transform(X)
XBack = scaler.inverse_transform(XScaled)

print(f"mean_: {scaler.mean_}")
print(f"scale_: {scaler.scale_}")
print(f"逆变换还原:\n{XBack}")
```

#### 输出

```text
mean_: [ 3. 30.]
scale_: [ 1.4142 14.1421]
逆变换还原:
[[ 1. 10.]
 [ 2. 20.]
 [ 3. 30.]
 [ 4. 40.]
 [ 5. 50.]]
```

#### 理解重点

- 标准化参数必须来自训练集——避免数据泄露
- `mean_` 和 `scale_` 也是可解释信息——可用于排查异常列
- 稀疏矩阵通常谨慎使用中心化——会破坏稀疏结构

## 3. MinMaxScaler 详解

### `MinMaxScaler`

#### 作用

将每列线性映射到指定区间 $[min, max]$，保持原始排序关系。常见区间为 `(0, 1)` 或 `(-1, 1)`。对异常值敏感——极端值会压缩其余样本的有效分辨率。

#### 重点方法

```python
MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
# fit(X) → transform(X) / fit_transform(X) → inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `feature_range` | `tuple[float, float]` | 目标区间，默认为 `(0, 1)` | `(-1, 1)` |
| `copy` | `bool` | 是否复制输入数组，默认为 `True` | `True` |
| `clip` | `bool` | 推理阶段是否截断超范围值，默认为 `False` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

s1 = MinMaxScaler(feature_range=(0, 1))
s2 = MinMaxScaler(feature_range=(-1, 1))

print(f"feature_range=(0,1):\n{s1.fit_transform(X)}")
print(f"\nfeature_range=(-1,1):\n{s2.fit_transform(X)}")
```

#### 输出

```text
feature_range=(0,1):
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [0.75 0.75]
 [1.   1.  ]]

feature_range=(-1,1):
[[-1.  -1. ]
 [-0.5 -0.5]
 [ 0.   0. ]
 [ 0.5  0.5]
 [ 1.   1. ]]
```

#### 理解重点

- 区间缩放不会让分布接近正态——仅改变取值区间
- 对树模型通常不是必须——但对基于距离或梯度的模型常常有帮助
- 当特征天然有上下界时 MinMax 缩放更直观

## 4. PowerTransformer 幂变换

### `PowerTransformer`

#### 作用

缓解偏态分布，使数据更接近对称分布。`yeo-johnson` 可处理非正数，`box-cox` 仅支持严格正数。变换后通常更利于线性模型和基于方差假设的方法。

#### 重点方法

```python
PowerTransformer(method='yeo-johnson', *, standardize=True)
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `method` | `str` | `'yeo-johnson'` 支持非正数 / `'box-cox'` 仅正数，默认为 `'yeo-johnson'` | `'box-cox'` |
| `standardize` | `bool` | 变换后是否再做标准化，默认为 `True` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `lambdas_` | `ndarray` | 每列的变换参数，反映分布拉伸程度 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

np.random.seed(42)
XSkewed = np.random.exponential(scale=2, size=(500, 1))

ptYj = PowerTransformer(method="yeo-johnson")
ptBc = PowerTransformer(method="box-cox")

Xyj = ptYj.fit_transform(XSkewed)
Xbc = ptBc.fit_transform(XSkewed)

print(f"原始: 均值={XSkewed.mean():.2f}, 偏度={np.mean((XSkewed - XSkewed.mean())**3) / XSkewed.std()**3:.2f}")
print(f"Yeo-Johnson lambda: {ptYj.lambdas_[0]:.3f}")
print(f"Box-Cox lambda: {ptBc.lambdas_[0]:.3f}")
```

#### 输出

```text
原始: 均值=2.01, 偏度=1.92
Yeo-Johnson lambda: -0.412
Box-Cox lambda: 0.264
```

#### 理解重点

- 偏态修正不等于信息增强——目标是改善建模假设匹配度
- 数据含 0 或负值优先使用 Yeo-Johnson
- 配合可视化直方图判断变换效果

## 5. 类别编码

### `LabelEncoder` / `OneHotEncoder`

#### 作用

`LabelEncoder` 适合目标标签编码——将类别映射为整数。`OneHotEncoder` 将类别映射为哑变量——更适合线性模型与距离模型。处理未知类别时 `handle_unknown='ignore'` 更稳妥。

#### 重点方法

```python
LabelEncoder()
OneHotEncoder(*, categories='auto', drop=None, sparse_output=True,
              handle_unknown='error')
# fit(y) → transform(y) / fit_transform(y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `sparse_output` | `bool` | OneHotEncoder：`False` 返回稠密数组，默认为 `True` | `False` |
| `handle_unknown` | `str` | OneHotEncoder：`'error'` 报错 / `'ignore'` 忽略，默认为 `'error'` | `'ignore'` |
| `drop` | `str` 或 `None` | OneHotEncoder：`'first'` 丢弃第一类避免共线性，默认为 `None` | `'first'` |

编码后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `classes_` | `ndarray` | 编码前后的类别映射 |
| `categories_` | `list[ndarray]` | OneHotEncoder：每列的类别列表 |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

colors = np.array([["红"], ["绿"], ["蓝"], ["红"], ["绿"]])

le = LabelEncoder()
colorsLe = le.fit_transform(colors.ravel())

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
colorsOhe = ohe.fit_transform(colors)

print(f"LabelEncoder 类别: {le.classes_}")
print(f"编码结果: {colorsLe}")
print(f"OneHotEncoder 特征名: {ohe.get_feature_names_out()}")
print(f"编码结果:\n{colorsOhe}")
```

#### 输出

```text
LabelEncoder 类别: ['蓝' '绿' '红']
编码结果: [2 1 0 2 1]
OneHotEncoder 特征名: ['x0_红' 'x0_绿' 'x0_蓝']
编码结果:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
```

#### 理解重点

- `LabelEncoder` 产生整数顺序——可能引入虚假大小关系，仅适合目标变量
- `OneHotEncoder` 增加维度——需关注稀疏性与内存开销
- 生产环境必须提前设计未知类别处理策略（`handle_unknown='ignore'`）
- `get_feature_names_out()` 返回编码后列名——便于调试和特征分析

## 6. 缺失值处理

### `SimpleImputer` / `KNNImputer`

#### 作用

`SimpleImputer` 提供均值、中位数、众数、常量等规则化填充。`KNNImputer` 利用样本相似性推断缺失值——通常更平滑但更耗时。填充策略应与特征分布和业务语义一致。

#### 重点方法

```python
SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None)
KNNImputer(*, n_neighbors=5, weights='uniform')
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `strategy` | `str` | SimpleImputer 填充策略：`'mean'` / `'median'` / `'most_frequent'` / `'constant'`，默认为 `'mean'` | `'median'` |
| `fill_value` | `str` 或 `float` | `strategy='constant'` 时的填充值，默认为 `None`（数值列填 0，字符串列填 `'missing_value'`） | `0` |
| `missing_values` | `int`、`float`、`str` | 缺失值标记符，默认为 `np.nan` | `np.nan` |
| `n_neighbors` | `int` | KNNImputer 参考邻居数量，默认为 `5` | `3` |
| `weights` | `str` | KNNImputer 权重策略：`'uniform'` / `'distance'`，默认为 `'uniform'` | `'distance'` |

#### 示例代码

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

X = np.array([[1, 2, np.nan], [3, np.nan, 6], [7, 8, 9], [np.nan, 5, 3]])

print("mean 填充:\n", SimpleImputer(strategy="mean").fit_transform(X))
print("\nmedian 填充:\n", SimpleImputer(strategy="median").fit_transform(X))
print("\nconstant=0 填充:\n", SimpleImputer(strategy="constant", fill_value=0).fit_transform(X))
print("\nKNN 填充:\n", KNNImputer(n_neighbors=2).fit_transform(X))
```

#### 输出

```text
mean 填充:
[[1.   2.   6.  ]
 [3.   5.   6.  ]
 [7.   8.   9.  ]
 [3.67 5.   3.  ]]

median 填充:
[[1. 2. 6.]
 [3. 5. 6.]
 [7. 8. 9.]
 [3. 5. 3.]]

constant=0 填充:
[[1. 2. 0.]
 [3. 0. 6.]
 [7. 8. 9.]
 [0. 5. 3.]]

KNN 填充:
[[1.  2.  6. ]
 [3.  5.  6. ]
 [7.  8.  9. ]
 [4.  5.  3. ]]
```

#### 理解重点

- 均值/中位数填充简单稳定——多数任务的首选基线
- KNN 填充更依赖特征尺度——通常应先做合理缩放
- 缺失值机制（MCAR/MAR/MNAR）影响填充有效性——需结合业务判断

## 7. ColumnTransformer 组合预处理

### `ColumnTransformer`

#### 作用

混合类型数据应采用分列处理：数值列与类别列使用不同流水线。`ColumnTransformer` 将多条子流水线拼接为统一特征空间——这是生产级预处理的核心模式，后续可直接接入模型。

#### 重点方法

```python
ColumnTransformer(transformers, *, remainder='drop', n_jobs=None,
                 verbose_feature_names_out=True)
# fit(X) → transform(X) / fit_transform(X) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transformers` | `list[tuple[str, estimator, columns]]` | 列分组与对应处理器列表 | `[("num", numPipe, numCols), ("cat", catPipe, catCols)]` |
| `remainder` | `str` 或 `estimator` | 未指定列的处理方式：`'drop'` 丢弃 / `'passthrough'` 保留，默认为 `'drop'` | `'passthrough'` |
| `verbose_feature_names_out` | `bool` | 特征名是否加前缀，默认为 `True` | `False` |

快捷列选择器：

```python
from sklearn.compose import make_column_selector
make_column_selector(dtype_include='number')  # 按类型自动选择列
```

#### 示例代码

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "年龄": [25, 30, np.nan, 40, 35],
    "收入": [50000, 60000, 55000, np.nan, 70000],
    "性别": ["男", "女", "男", "女", "男"],
    "城市": ["北京", "上海", "北京", "广州", "上海"],
})

numPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

catPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numPipe, selector(dtype_include="number")),
    ("cat", catPipe, selector(dtype_include="object")),
])

XProc = preprocessor.fit_transform(df)
print(f"处理后形状: {XProc.shape}")
print(f"特征名称: {preprocessor.get_feature_names_out()}")
```

#### 输出

```text
处理后形状: (5, 7)
特征名称: ['num__年龄' 'num__收入' 'cat__性别_女' 'cat__性别_男' 'cat__城市_上海' 'cat__城市_北京' 'cat__城市_广州']
```

#### 理解重点

- 列级流水线能把预处理逻辑完全纳入模型训练过程——减少线上线下不一致
- 该模式可直接嵌入 `Pipeline` 做联合调参与部署
- `make_column_selector` 按 dtype 自动选列——避免手动列名硬编码
- 当类别空间较大时应关注 One-Hot 维度膨胀问题

## 常见坑

1. 在训练前先对全量数据 `fit` 缩放器或填充器——导致数据泄露
2. 直接把 `LabelEncoder` 用在普通类别特征上——引入错误顺序关系
3. 在 `ColumnTransformer` 中忘记统一缺失值策略——导致推理阶段报错
4. 测试集使用 `fit_transform` 而非 `transform`——破坏了训练/测试隔离

## 小结

- 预处理不是独立步骤——而是模型流程的一部分
- 推荐将缩放、编码、填充封装进流水线并与模型共同训练
- 先建立稳定可复现的预处理基线——再做策略替换和调优
- `ColumnTransformer` + `Pipeline` 是生产级预处理的标准模式

# sklearn 特征工程

## 本章目标

1. 掌握文本、字典、数值三类常见特征构造方式
2. 理解 `TfidfVectorizer`、`PolynomialFeatures` 的参数影响
3. 学会使用过滤法、包裹法、模型法进行特征选择
4. 明确高维特征场景下的稀疏表示与维度控制策略

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `CountVectorizer()` | 构造器 | 将文本转为词频特征（词袋模型） |
| `TfidfVectorizer(...)` | 构造器 | 将文本转为 TF-IDF 特征 |
| `DictVectorizer(sparse=False)` | 构造器 | 将字典样本展开为特征矩阵 |
| `PolynomialFeatures(degree)` | 构造器 | 生成多项式与交互项特征 |
| `VarianceThreshold(threshold)` | 构造器 | 过滤低方差特征（无监督） |
| `SelectKBest(score_func, k)` | 构造器 | 基于统计检验筛选特征（过滤法） |
| `RFE(estimator, n_features_to_select)` | 构造器 | 递归特征消除（包裹法） |
| `SelectFromModel(estimator)` | 构造器 | 基于模型重要性筛选（嵌入式） |

## 1. 文本特征提取

### `CountVectorizer` / `TfidfVectorizer`

#### 作用

`CountVectorizer` 将文本转为词频特征（词袋模型），返回稀疏矩阵。`TfidfVectorizer` 在词频基础上引入逆文档频率——削弱高频泛化词影响。`max_df`、`min_df`、`ngram_range` 控制特征空间大小。

#### 重点方法

```python
CountVectorizer(*, lowercase=True, ngram_range=(1, 1), stop_words=None)
TfidfVectorizer(*, max_features=None, ngram_range=(1, 1), stop_words=None,
                max_df=1.0, min_df=1)
# fit(corpus) → transform(corpus) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_features` | `int` 或 `None` | 限制最大词特征数，默认为 `None` | `1000` |
| `ngram_range` | `tuple[int, int]` | n-gram 范围，默认为 `(1, 1)` | `(1, 2)` |
| `stop_words` | `str`、`list` 或 `None` | 停用词过滤，`'english'` 使用英文停用词表 | `'english'` |
| `max_df` | `float` 或 `int` | TfidfVectorizer：忽略出现频率过高的词，默认为 `1.0` | `0.9` |
| `min_df` | `float` 或 `int` | TfidfVectorizer：忽略出现频率过低的词，默认为 `1` | `2` |
| `lowercase` | `bool` | 是否统一小写化，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]

cv = CountVectorizer()
Xc = cv.fit_transform(corpus)
print(f"CountVectorizer 词汇表: {cv.get_feature_names_out()}")
print(f"词频矩阵:\n{Xc.toarray()}")

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
Xt = tfidf.fit_transform(corpus)
print(f"\nTfidfVectorizer (stop_words+bigram): {Xt.shape}")
print(f"词汇表: {tfidf.get_feature_names_out()}")
```

#### 输出

```text
CountVectorizer 词汇表: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
词频矩阵:
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]]

TfidfVectorizer (stop_words+bigram): (3, 8)
词汇表: ['document' 'document second' 'second' 'third' 'third one' ...]
```

#### 理解重点

- Count 特征易解释——但无法表达词义接近关系，对文本分类基线非常有效
- TF-IDF 更关注区分度高的词——不等于语义建模
- 中文文本需自定义分词与停用词策略
- 返回稀疏矩阵（`csr_matrix`）——适合高维文本特征

## 2. 字典特征展开

### `DictVectorizer`

#### 作用

将结构化字典输入自动展开为特征矩阵——对类别键做独热展开，对数值键保留原值。通过 `inverse_transform` 可回看特征与原字典的映射关系。

#### 重点方法

```python
DictVectorizer(*, sparse=True)
# fit(data) → transform(data) / fit_transform(data)
# get_feature_names_out() / inverse_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `sparse` | `bool` | `True` 返回稀疏矩阵 / `False` 返回稠密数组，默认为 `True` | `False` |

#### 示例代码

```python
from sklearn.feature_extraction import DictVectorizer

data = [
    {"city": "北京", "temperature": 20},
    {"city": "上海", "temperature": 25},
    {"city": "北京", "temperature": 18},
]

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data)

print(f"特征名: {dv.get_feature_names_out()}")
print(f"特征矩阵:\n{X}")
print(f"逆变换: {dv.inverse_transform(X)[0]}")
```

#### 输出

```text
特征名: ['city=上海' 'city=北京' 'temperature']
特征矩阵:
[[0. 1. 20.]
 [1. 0. 25.]
 [0. 1. 18.]]
逆变换: {'city=北京': 1.0, 'temperature': 20.0}
```

#### 理解重点

- DictVectorizer 常用于日志特征、规则特征、浅层推荐特征工程
- 产物特征名可追踪——利于可解释性
- 类别字段较多时建议结合频次阈值做后续裁剪

## 3. 多项式特征扩展

### `PolynomialFeatures`

#### 作用

生成多项式与交互项特征——可让线性模型拟合非线性关系。维度增长很快（$O(d^{degree})$），`interaction_only=True` 只保留交互项降低膨胀。

#### 重点方法

```python
PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True)
# fit(X) → transform(X) / get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `degree` | `int` | 最高多项式阶数，默认为 `2` | `3` |
| `interaction_only` | `bool` | `True` 只保留交互项（不含平方项），默认为 `False` | `True` |
| `include_bias` | `bool` | `True` 包含常数项 1，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 2], [3, 4]])

poly2 = PolynomialFeatures(degree=2, include_bias=True)
print(f"degree=2 特征名: {poly2.fit_transform(X).shape}")

polyInter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(f"interaction_only 特征名: {polyInter.fit(X).get_feature_names_out()}")
```

#### 输出

```text
degree=2 特征: ['1' 'x0' 'x1' 'x0^2' 'x0 x1' 'x1^2'] → 6 维
interaction_only 特征: ['x0' 'x1' 'x0 x1'] → 3 维
```

#### 理解重点

- 多项式特征是经典有效方法——但非常依赖正则化
- 若模型已具备强非线性能力（树模型）——未必需要此扩展
- `degree` 不宜盲目增大——维度呈组合爆炸增长

## 4. 过滤法特征选择

### `VarianceThreshold` / `SelectKBest`

#### 作用

**过滤法**在模型训练前独立筛选特征。`VarianceThreshold` 去掉方差为 0 的常量特征（无监督）。`SelectKBest` 根据统计检验分数保留前 K 个特征（监督式，分类用 `f_classif`，回归用 `f_regression`）。

#### 重点方法

```python
VarianceThreshold(threshold=0.0)
SelectKBest(score_func=f_classif, *, k=10)
# fit(X[, y]) → transform(X) / get_support()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `threshold` | `float` | VarianceThreshold：方差阈值，低于此值的特征被移除 | `0.1` |
| `score_func` | `callable` | SelectKBest：评分函数，`f_classif`（分类）或 `f_regression`（回归） | `f_classif` |
| `k` | `int` | SelectKBest：保留的特征数量 | `2` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `variances_` | `ndarray` | VarianceThreshold：每列方差 |
| `scores_` | `ndarray` | SelectKBest：各特征评分 |
| `get_support()` | `ndarray[bool]` | 被选中特征的布尔掩码 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# VarianceThreshold：去除常量特征
X = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
vt = VarianceThreshold(threshold=0)
Xvt = vt.fit_transform(X)
print(f"方差过滤: {X.shape} → {Xvt.shape}, 方差={vt.variances_}")

# SelectKBest：保留得分最高的 K 个特征
iris = datasets.load_iris()
skb = SelectKBest(score_func=f_classif, k=2)
Xskb = skb.fit_transform(iris.data, iris.target)
print(f"SelectKBest: {iris.data.shape} → {Xskb.shape}")
print(f"得分: {skb.scores_.round(1)}")
print(f"选中特征: {np.array(iris.feature_names)[skb.get_support()]}")
```

#### 输出

```text
方差过滤: (4, 3) → (4, 2), 方差=[0.25 0.25 0.  ]
SelectKBest: (150, 4) → (150, 2)
得分: [ 119.3   49.2 1180.2  960. ]
选中特征: ['petal length (cm)' 'petal width (cm)']
```

#### 理解重点

- 方差过滤是特征筛选的"第一刀"——放在流程最前面
- 单变量统计分数高不代表对所有模型最优——忽略特征联合效应
- 建议把 `k` 作为超参数用交叉验证选择

## 5. 包裹法与嵌入式特征选择

### `RFE` / `SelectFromModel`

#### 作用

**RFE**（包裹法）反复训练模型并移除最弱特征——依赖基学习器的权重或重要性。**SelectFromModel**（嵌入式）基于模型内置重要性一次性筛选——比 RFE 更高效。两者都适用于分类和回归。

#### 重点方法

```python
RFE(estimator, *, n_features_to_select=None, step=1)
SelectFromModel(estimator, *, threshold=None)
# fit(X, y) → transform(X) → get_support()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 提供特征权重的基模型 | `LogisticRegression(max_iter=1000)` |
| `n_features_to_select` | `int` 或 `None` | RFE：最终保留特征数；`None` 保留一半 | `2` |
| `step` | `int` 或 `float` | RFE：每轮移除的特征数或比例，默认为 `1` | `1` |
| `threshold` | `str` 或 `float` | SelectFromModel：`'median'` / `'mean'` 或数值阈值 | `'median'` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `ranking_` | `ndarray` | RFE：特征排名，1 表示最终选中 |
| `support_` | `ndarray[bool]` | 选中掩码 |
| `estimator_.feature_importances_` | `ndarray` | SelectFromModel：基模型的特征重要性 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

iris = datasets.load_iris()
X, y = iris.data, iris.target

# RFE
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
rfe.fit(X, y)
print(f"RFE ranking: {rfe.ranking_}, 选中: {np.array(iris.feature_names)[rfe.support_]}")

# SelectFromModel
rf = RandomForestClassifier(n_estimators=100, random_state=42)
sfm = SelectFromModel(rf, threshold="median")
sfm.fit(X, y)
print(f"SelectFromModel 重要性: {sfm.estimator_.feature_importances_.round(3)}")
print(f"选中: {np.array(iris.feature_names)[sfm.get_support()]}")
```

#### 输出

```text
RFE ranking: [3 2 1 1], 选中: ['petal length (cm)' 'petal width (cm)']
SelectFromModel 重要性: [0.102 0.023 0.436 0.439], 选中: ['petal length (cm)' 'petal width (cm)']
```

#### 理解重点

- RFE 结果依赖基模型——换模型可能得到不同子集
- `SelectFromModel` 对树模型与线性模型都适用——但重要性定义不同
- 维度很高时可先过滤再 RFE/SFM——降低计算开销
- 阈值是可调旋钮——控制维度与性能的平衡

## 常见坑

1. 文本向量化后维度过大却不限制 `max_features`——导致训练和推理成本骤增
2. 在训练前先查看全量数据选择特征——造成数据泄露
3. 把某一模型的特征选择结果直接迁移到完全不同模型而不复验

## 小结

- 特征工程的本质是表达能力与泛化能力的平衡
- 先用稳健基线方法构造与筛选——再通过交叉验证量化收益
- 推荐将特征工程步骤放入 Pipeline——以便和模型调参一体化管理
- 文本用 `CountVectorizer`/`TfidfVectorizer`，类别用 `DictVectorizer`/`OneHotEncoder`，数值用 `PolynomialFeatures`

# sklearn Pipeline

## 本章目标

1. 掌握 `Pipeline` 与 `make_pipeline` 的构建方式与差异
2. 学会访问流水线步骤、读取与修改步骤参数
3. 理解双下划线参数命名规则在调参中的作用
4. 掌握 `ColumnTransformer` 处理混合类型特征的标准写法
5. 学会把目标变量变换纳入回归流程

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `Pipeline(steps)` | 构造器 | 显式命名步骤构建流水线 |
| `make_pipeline(*steps)` | 函数 | 自动命名步骤快速构建 |
| `pipe.named_steps` | 属性 | 按名称访问步骤对象 |
| `pipe.set_params(**params)` | 方法 | 修改子步骤参数（双下划线规则） |
| `ColumnTransformer(transformers)` | 构造器 | 按列类型组合预处理 |
| `TransformedTargetRegressor(regressor)` | 构造器 | 目标变量变换回归 |

## 1. Pipeline 基础

### `Pipeline` / `make_pipeline`

#### 作用

流水线把预处理和模型封装成一个可复用对象，避免训练与推理逻辑分叉。`Pipeline` 需要手动命名步骤；`make_pipeline` 自动命名，写法更短。统一对象后可整体调用 `fit`、`predict`、`score`。

#### 重点方法

```python
Pipeline(steps, *, memory=None, verbose=False)
make_pipeline(*steps, memory=None, verbose=False)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `steps` | `list[tuple[str, estimator]]` | 步骤名与变换器/估计器列表 | `[("scaler", StandardScaler()), ("svm", SVC())]` |
| `*steps` | `estimator` | make_pipeline：按顺序传入变换器/估计器 | `StandardScaler(), SVC()` |
| `memory` | `str` 或 `None` | 缓存路径，None 不缓存，默认为 `None` | `"./cache"` |
| `verbose` | `bool` | 是否输出步骤耗时，默认为 `False` | `True` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `steps` | `list[tuple[str, estimator]]` | 步骤名与已训练对象列表 |
| `named_steps` | `utils.Bunch` | 按名称访问步骤对象 |
| `classes_` | `ndarray` | 最终估计器的类别标签 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])
pipe.fit(X_train, y_train)
print(f"Pipeline 准确率: {pipe.score(X_test, y_test):.4f}")

pipeAuto = make_pipeline(StandardScaler(), PCA(n_components=2), SVC())
pipeAuto.fit(X_train, y_train)
print(f"make_pipeline 准确率: {pipeAuto.score(X_test, y_test):.4f}")
print(f"自动命名步骤: {[name for name, _ in pipeAuto.steps]}")
```

#### 输出

```text
Pipeline 准确率: 0.9556
make_pipeline 准确率: 0.9556
自动命名步骤: ['standardscaler', 'pca', 'svc']
```

#### 理解重点

- 只要对象实现 sklearn 接口，就能被纳入同一流水线
- 步骤命名不是装饰，而是后续调参与调试的锚点
- 训练完成的流水线可整体持久化，部署更稳定
- 最后一步可以是分类器、回归器或任何估计器

## 2. 访问 Pipeline 步骤

### `pipe.steps` / `named_steps`

#### 作用

可以用 `steps`、`named_steps`、整数索引多种方式访问组件。`named_steps` 适合生产代码，稳定且可读。步骤对象可直接拿出来检查参数或属性。

#### 重点方法

```python
pipe.steps          # → list[tuple[str, estimator]]
pipe.named_steps    # → Bunch（属性式访问）
pipe[index]         # → 第 index 个步骤对象
pipe[-1]            # → 最后一个步骤（通常是预估器）
```

#### 示例代码

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])

print(f"步骤列表: {pipe.steps}")
print(f"named_steps['pca']: {pipe.named_steps['pca']}")
print(f"pipe[0]: {pipe[0]}")
print(f"pipe[-1]: {pipe[-1]}")
```

#### 输出

```text
步骤列表: [('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('svm', SVC())]
named_steps['pca']: PCA(n_components=2)
pipe[0]: StandardScaler()
pipe[-1]: SVC()
```

#### 理解重点

- 大多数调试问题都能通过检查 `named_steps` 快速定位
- 步骤顺序直接影响输入输出维度和模型表现
- 推荐在文档和代码中保持统一步骤命名规范
- 索引访问适合循环遍历步骤做诊断

## 3. Pipeline 参数设置

### `set_params`

#### 作用

子步骤参数通过 `步骤名__参数名` 写法进行设置。该规则同样适用于网格搜索与随机搜索。`set_params` 返回对象自身，支持链式调用。

#### 重点方法

```python
pipe.set_params(**params)      # 修改步骤参数，返回 self
pipe.get_params(deep=True)     # 获取所有可调参数字典
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `**params` | `dict` | 子步骤参数，格式 `step__param=value` | `pca__n_components=3` |
| `deep` | `bool` | 是否递归获取嵌套参数，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC(C=1.0)),
])

print(f"修改前: PCA n_components={pipe.named_steps['pca'].n_components}, "
      f"SVM C={pipe.named_steps['svm'].C}")

pipe.set_params(pca__n_components=3, svm__C=10)
print(f"修改后: PCA n_components={pipe.named_steps['pca'].n_components}, "
      f"SVM C={pipe.named_steps['svm'].C}")

# 查看完整参数
print(f"pca__n_components: {pipe.get_params()['pca__n_components']}")
```

#### 输出

```text
修改前: PCA n_components=2, SVM C=1.0
修改后: PCA n_components=3, SVM C=10
pca__n_components: 3
```

#### 理解重点

- 双下划线规则是 sklearn 组合对象调参的核心约定
- 复杂流水线里，参数命名准确性直接决定调参是否生效
- `get_params()` 返回的键名即为 `set_params` 接受的参数名
- 在网格搜索中大规模使用该规则（见下节）

## 4. Pipeline + GridSearchCV

### `GridSearchCV` 联合调参

#### 作用

将预处理与模型打包后调参，可避免预处理阶段数据泄露。参数网格按 `步骤名__参数名` 书写。网格搜索会自动在交叉验证中重复完整流水线，保证评估无偏。

#### 重点方法

```python
GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, cv=None, verbose=0)
# fit(X, y) → best_params_ / best_score_ / predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参的流水线或模型 | `make_pipeline(StandardScaler(), SVC())` |
| `param_grid` | `dict` 或 `list[dict]` | 搜索参数网格，键用 `step__param` 格式 | `{"svc__C": [0.1, 1, 10]}` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None`（用估计器默认） | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心，默认为 `None` | `-1` |
| `verbose` | `int` | 日志详细度，默认为 `0` | `1` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `best_params_` | `dict` | 最优参数组合 |
| `best_score_` | `float` | 最优参数对应的交叉验证平均分 |
| `best_estimator_` | `estimator` | 用最优参数在全量数据训练的模型 |
| `cv_results_` | `dict` | 完整搜索结果（各参数组合的分值、时间） |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)

pipe = make_pipeline(StandardScaler(), SVC())
paramGrid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
}

grid = GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy")
grid.fit(X, y)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.4f}")
print(f"最佳模型: {grid.best_estimator_}")
```

#### 输出

```text
最佳参数: {'svc__C': 1, 'svc__kernel': 'linear'}
最佳得分: 0.9733
最佳模型: Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(C=1, kernel='linear'))])
```

#### 理解重点

- 将预处理写进流水线后，调参与训练流程天然一致——每折内部做 `fit_transform`，外部做 `transform`
- 网格搜索成本高，参数空间要先做工程化收敛
- 实战可先随机搜索粗定位，再网格精搜
- `cv_results_` 包含所有候选参数的分值，可用于进一步分析

## 5. 跳过步骤

### `set_params(step='passthrough')`

#### 作用

可用 `passthrough` 暂时禁用某步骤，便于做消融实验。消融结果可帮助判断该步骤是否真正贡献性能。对比实验要保证其他设置不变，避免混杂结论。

#### 重点方法

```python
pipe.set_params(step_name='passthrough')
# 设置后该步骤不执行任何变换，输出 = 输入
```

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("svm", SVC()),
])
pipe.fit(X_train, y_train)
print(f"含 PCA: {pipe.score(X_test, y_test):.4f}")

pipe.set_params(pca="passthrough")
pipe.fit(X_train, y_train)
print(f"跳过 PCA: {pipe.score(X_test, y_test):.4f}")
print(f"当前 pca 步骤: {pipe.named_steps['pca']}")
```

#### 输出

```text
含 PCA: 0.9556
跳过 PCA: 0.9778
当前 pca 步骤: passthrough
```

#### 理解重点

- `passthrough` 是快速做 A/B 对比的高效工具
- 某步骤"可删"不代表永远不需要，取决于任务与数据规模
- 该技巧也适用于特征选择、标准化等模块
- 消融实验应成为建模流程的标准步骤

## 6. ColumnTransformer 混合类型处理

### `ColumnTransformer`

#### 作用

数值列和类别列应分开处理，再统一拼接。`ColumnTransformer` 将多条子流水线拼接为统一特征空间——这是生产级预处理的核心模式，后续可直接接入模型。

#### 重点方法

```python
ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3,
                 n_jobs=None, verbose=False, verbose_feature_names_out=True)
# fit(X) → transform(X) / fit_transform(X) → get_feature_names_out()
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transformers` | `list[tuple[str, estimator, columns]]` | 列分组与对应处理器列表 | `[("num", numPipe, [0, 1])]` |
| `remainder` | `str` 或 `estimator` | 未指定列的处理：`"drop"` 丢弃 / `"passthrough"` 保留，默认为 `"drop"` | `"passthrough"` |
| `sparse_threshold` | `float` | 稀疏阈值，低于此比例输出稀疏矩阵，默认为 `0.3` | `0.3` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `None` |
| `verbose` | `bool` | 是否输出耗时信息，默认为 `False` | `True` |
| `verbose_feature_names_out` | `bool` | 特征名是否加前缀，默认为 `True` | `True` |

快捷列选择器：

```python
from sklearn.compose import make_column_selector
make_column_selector(dtype_include='number')   # 按类型自动选择列
make_column_selector(dtype_include='object')   # 选字符串列
make_column_selector(pattern='.*')             # 按列名正则匹配
```

#### 示例代码

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "年龄": [25, 30, np.nan, 40, 35],
    "收入": [50000, 60000, 55000, np.nan, 70000],
    "学历": ["本科", "硕士", "本科", "博士", "硕士"],
})
y = [0, 1, 0, 1, 0]

numPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

catPipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numPipe, selector(dtype_include="number")),
    ("cat", catPipe, selector(dtype_include="object")),
])

fullPipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

fullPipe.fit(df, y)
print(f"特征名: {fullPipe.named_steps['preprocessor'].get_feature_names_out()}")
print(f"预测: {fullPipe.predict(df)}")
```

#### 输出

```text
特征名: ['num__年龄' 'num__收入' 'cat__学历_博士' 'cat__学历_本科' 'cat__学历_硕士']
预测: [0 1 0 1 0]
```

#### 理解重点

- 列级流水线能把预处理逻辑完全纳入模型训练过程——减少线上线下不一致
- 该模式可直接嵌入 `Pipeline` 做联合调参与部署
- `make_column_selector` 按 dtype 自动选列——避免手动列名硬编码
- 当类别空间较大时应关注 One-Hot 维度膨胀问题
- 训练后要检查输出特征名，确保下游解释与监控一致

## 7. TransformedTargetRegressor

### `TransformedTargetRegressor`

#### 作用

对目标变量做变换（如对数）可缓解长尾分布问题。训练在变换空间进行，预测自动做逆变换返回原尺度。常见于金额、流量等偏态回归任务。

#### 重点方法

```python
TransformedTargetRegressor(regressor=None, *, transformer=None, func=None,
                           inverse_func=None, check_inverse=True)
# fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `regressor` | `estimator` | 基础回归器，默认为 `None`（需指定） | `LinearRegression()` |
| `transformer` | `estimator` | 变换器（需实现 `transform`/`inverse_transform`） | `PowerTransformer(method="box-cox")` |
| `func` | `callable` 或 `None` | 目标变量前向变换函数 | `np.log1p` |
| `inverse_func` | `callable` 或 `None` | 预测值逆变换函数 | `np.expm1` |
| `check_inverse` | `bool` | 是否检查逆变换一致性，默认为 `True` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

lr = LinearRegression().fit(X_train, y_train)
print(f"普通回归 R²: {lr.score(X_test, y_test):.4f}")

ttr = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1,
)
ttr.fit(X_train, y_train)
print(f"对数目标变换 R²: {ttr.score(X_test, y_test):.4f}")
```

#### 输出

```text
普通回归 R²: 0.4773
对数目标变换 R²: 0.4314
```

#### 理解重点

- 是否使用目标变换应通过验证集结果决定，不是固定加分项
- 变换后指标变化要结合业务误差定义解读
- 若目标存在 0 或负值，要确认变换函数可用性（`yeo-johnson` 可处理非正数）
- `func`/`inverse_func` 与 `transformer` 二选一即可

## 常见坑

1. 先全量标准化再切分训练测试，导致数据泄露——流水线也不能解决外部 fit 的问题
2. 参数名漏写步骤前缀，导致 `set_params` 或网格搜索未生效
3. `ColumnTransformer` 的 `remainder='drop'` 会静默丢弃未指定列——建议显式设置
4. 用 `make_pipeline` 时忘记步骤名由类名决定——调参时需查看 `steps` 确认名称
5. `TransformedTargetRegressor` 的 `check_inverse` 为 `True` 时可能触发逆变换验证失败

## 小结

- Pipeline 是 sklearn 工程化落地的基础组件——推荐把预处理、特征构造、模型训练统一封装
- 双下划线参数命名是贯通 Pipeline 调参的唯一规则——`get_params` 返回的键名即网格搜索键名
- `ColumnTransformer` + `Pipeline` 是生产级预处理的标准模式——按列类型分而治之
- `passthrough` 与 `TransformedTargetRegressor` 是流程实验和优化的重要辅助工具

# sklearn 模型选择

## 本章目标

1. 理解交叉验证分数的统计意义与波动范围
2. 掌握 `cross_val_score` 与 `cross_validate` 的使用边界
3. 学会选择合适的划分器（KFold、StratifiedKFold、TimeSeriesSplit）
4. 掌握网格搜索与随机搜索的参数设计思路
5. 能用学习曲线与验证曲线诊断欠拟合和过拟合

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `cross_val_score(estimator, X, y)` | 函数 | 单指标交叉验证，返回每折分数 |
| `cross_validate(estimator, X, y)` | 函数 | 多指标交叉验证，返回训练与测试分数 |
| `KFold(n_splits)` | 构造器 | 基础 K 折划分 |
| `StratifiedKFold(n_splits)` | 构造器 | 分层 K 折，保持类别比例 |
| `TimeSeriesSplit(n_splits)` | 构造器 | 时间序列逐步扩窗划分 |
| `GridSearchCV(estimator, param_grid)` | 构造器 | 穷举参数组合搜索 |
| `RandomizedSearchCV(estimator, param_distributions)` | 构造器 | 随机采样参数搜索 |
| `learning_curve(estimator, X, y)` | 函数 | 训练集规模-性能曲线 |
| `validation_curve(estimator, X, y, ...)` | 函数 | 单参数-性能曲线 |

## 1. 交叉验证

### `cross_val_score`

#### 作用

`cross_val_score` 返回每折分数数组，适合快速评估模型稳定性。分数均值反映整体性能，标准差反映稳定性。推荐将分数与模型复杂度一起解读。

#### 重点方法

```python
cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
                n_jobs=None, verbose=0, params=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None`（用估计器默认） | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"各折得分: {scores}")
print(f"平均: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

#### 输出

```text
各折得分: [0.96666667 0.96666667 0.93333333 0.96666667 1.        ]
平均: 0.9667 (+/- 0.0422)
```

#### 理解重点

- 单次 train/test 切分容易偶然偏高或偏低，交叉验证更稳健
- 平均值高但方差大时，模型泛化稳定性仍需警惕
- 不同指标会改变结论，应与任务目标对齐
- 返回分数与 `scoring` 参数绑定——"accuracy" 越高越好，"neg_mean_squared_error" 越低越好

### `cross_validate`

#### 作用

`cross_validate` 能同时返回多指标结果与训练分数。训练分数与测试分数差距可辅助判断过拟合。返回字典结构便于后续可视化和日志记录。

#### 重点方法

```python
cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None,
               n_jobs=None, verbose=0, params=None,
               return_train_score=False, return_estimator=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `scoring` | `str`、`list[str]` 或 `dict` | 评估指标（支持多个），默认为 `None` | `["accuracy", "f1_macro"]` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `return_train_score` | `bool` | 是否返回训练集分数，默认为 `False` | `True` |
| `return_estimator` | `bool` | 是否返回每折训练的估计器，默认为 `False` | `True` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `-1` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())

cvResults = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True,
)

print(f"返回的键: {list(cvResults.keys())}")
print(f"测试准确率: {cvResults['test_accuracy'].mean():.4f}")
print(f"训练准确率: {cvResults['train_accuracy'].mean():.4f}")
print(f"测试 F1: {cvResults['test_f1_macro'].mean():.4f}")
```

#### 输出

```text
返回的键: ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_f1_macro', 'train_f1_macro']
测试准确率: 0.9667
训练准确率: 0.9833
测试 F1: 0.9664
```

#### 理解重点

- 多指标结果能避免"单指标最优但业务不优"的问题
- 训练分数显著高于测试分数时，优先检查过拟合
- 这类结果适合沉淀到实验追踪系统
- `return_estimator=True` 可取出每折模型做进一步分析

## 2. 划分策略

### `KFold` / `StratifiedKFold` / `TimeSeriesSplit`

#### 作用

划分器选择必须匹配数据分布与任务类型。分类任务默认优先 `StratifiedKFold` 保持类别比例。时间序列不能随机打乱，需使用 `TimeSeriesSplit`。错误划分策略会比模型选择本身造成更大偏差。

#### 重点方法

```python
KFold(n_splits=5, *, shuffle=False, random_state=None)
StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
TimeSeriesSplit(n_splits=5, *, max_train_size=None)
# split(X, y=None) → 迭代器，产出 (train_indices, test_indices)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_splits` | `int` | 划分折数 | `5` |
| `shuffle` | `bool` | 划分前是否打乱样本（KFold/StratifiedKFold），默认为 `False` | `True` |
| `random_state` | `int` | 随机种子，保证可复现 | `42` |
| `max_train_size` | `int` 或 `None` | TimeSeriesSplit：训练集最大容量限制 | `100` |

#### 示例代码

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

kf = KFold(n_splits=3, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tscv = TimeSeriesSplit(n_splits=3)

print("KFold (shuffle=True):")
for i, (tr, te) in enumerate(kf.split(X)):
    print(f"  Fold {i}: train={tr}, test={te}")

print("\nStratifiedKFold 类别分布:")
for i, (tr, te) in enumerate(skf.split(X, y)):
    print(f"  Fold {i}: train={np.bincount(y[tr])}, test={np.bincount(y[te])}")

print("\nTimeSeriesSplit:")
for i, (tr, te) in enumerate(tscv.split(X)):
    print(f"  Fold {i}: train={tr}, test={te}")
```

#### 输出

```text
KFold (shuffle=True):
  Fold 0: train=[2 3 4 6 7 9], test=[0 1 5 8]
  Fold 1: train=[0 1 4 5 7 8 9], test=[2 3 6]
  Fold 2: train=[0 1 2 3 5 6 8], test=[4 7 9]

StratifiedKFold 类别分布:
  Fold 0: train=[3 3], test=[2 2]
  Fold 1: train=[3 3], test=[2 2]
  Fold 2: train=[4 4], test=[1 1]

TimeSeriesSplit:
  Fold 0: train=[0 1 2 3], test=[4 5]
  Fold 1: train=[0 1 2 3 4 5], test=[6 7]
  Fold 2: train=[0 1 2 3 4 5 6 7], test=[8 9]
```

#### 理解重点

- 分类任务默认优先 `StratifiedKFold`——保持每折类别比例一致
- 时间序列任务应严格遵守时间先后关系——不能用随机 K 折
- 类别不平衡时不分层会导致评估结果波动异常
- `TimeSeriesSplit` 训练集逐步扩大——模拟真实时间预测场景

## 3. GridSearchCV

### `GridSearchCV`

#### 作用

网格搜索会遍历所有参数组合，结果稳定但计算成本高。参数空间应先由经验收敛，否则会出现组合爆炸。常与 Pipeline 结合以统一预处理与调参。

#### 重点方法

```python
GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True,
             cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan)
# fit(X, y) → best_params_ / best_score_ / best_estimator_ / predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参模型或流水线 | `make_pipeline(StandardScaler(), SVC())` |
| `param_grid` | `dict` 或 `list[dict]` | 参数网格，键用 `step__param` 格式 | `{"svc__C": [0.1, 1, 10]}` |
| `scoring` | `str`、`callable`、`list` 或 `dict` | 评估指标，默认为 `None` | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |
| `refit` | `bool` | 是否用最优参数在全量数据重训，默认为 `True` | `True` |
| `verbose` | `int` | 日志详细度，默认为 `0` | `1` |
| `pre_dispatch` | `int` 或 `str` | 并行任务预分配数，默认为 `"2*n_jobs"` | `"2*n_jobs"` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `best_params_` | `dict` | 最优参数组合 |
| `best_score_` | `float` | 最优参数对应的交叉验证平均分 |
| `best_estimator_` | `estimator` | 用最优参数在全量数据训练的模型 |
| `cv_results_` | `dict` | 完整搜索结果 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())

paramGrid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
}

grid = GridSearchCV(pipe, paramGrid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.4f}")
print(f"候选数: {len(grid.cv_results_['params'])}")
```

#### 输出

```text
最佳参数: {'svc__C': 1, 'svc__kernel': 'linear'}
最佳得分: 0.9733
候选数: 6
```

#### 理解重点

- GridSearchCV 更像"精细扫描"，前提是搜索区间合理
- 参数边界选择不当会浪费大量算力且结果无效
- 大规模任务可先随机搜索粗定位，再网格精调
- `cv_results_` 包含所有候选参数的时间和分值——可用于绘制热力图

## 4. RandomizedSearchCV

### `RandomizedSearchCV`

#### 作用

随机搜索通过概率分布采样参数，成本可控。在高维参数空间里，常比小网格更高效。常与 `loguniform` 分布配合搜索正实数超参数。

#### 重点方法

```python
RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None,
                   n_jobs=None, refit=True, cv=None, verbose=0,
                   random_state=None)
# fit(X, y) → best_params_ / best_score_ / best_estimator_
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待调参模型或流水线 | `make_pipeline(StandardScaler(), SVC())` |
| `param_distributions` | `dict` | 采样空间，值可为分布对象或列表 | `{"svc__C": loguniform(0.01, 100)}` |
| `n_iter` | `int` | 采样次数，默认为 `10` | `20` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `n_jobs` | `int` 或 `None` | 并行数，`-1` 使用全部核心 | `-1` |
| `random_state` | `int` | 采样可复现种子 | `42` |
| `refit` | `bool` | 是否最优参数重训，默认为 `True` | `True` |

#### 示例代码

```python
from scipy.stats import loguniform
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC())

paramDist = {
    "svc__C": loguniform(0.01, 100),
    "svc__gamma": loguniform(0.001, 10),
    "svc__kernel": ["rbf", "linear"],
}

search = RandomizedSearchCV(pipe, paramDist, n_iter=20, cv=5,
                            scoring="accuracy", random_state=42, n_jobs=-1)
search.fit(X, y)
print(f"最佳参数: {search.best_params_}")
print(f"最佳得分: {search.best_score_:.4f}")
```

#### 输出

```text
最佳参数: {'svc__C': 2.7323713729500725, 'svc__gamma': 0.013895024894200025, 'svc__kernel': 'rbf'}
最佳得分: 0.9800
```

#### 理解重点

- 采样分布比搜索算法本身更关键，应根据参数尺度设计
- `n_iter` 不是越大越好，应与预算和收益平衡
- 随机搜索结果可作为网格搜索的初始范围参考
- 连续参数用 `loguniform`/`uniform`，离散参数用列表

## 5. 学习曲线与验证曲线

### `learning_curve`

#### 作用

学习曲线观察训练样本量变化对性能的影响。训练分数高、验证分数低通常提示过拟合。两条曲线都低且接近，通常提示欠拟合或特征不足。

#### 重点方法

```python
learning_curve(estimator, X, y, *, groups=None, train_sizes=None, cv=None,
               scoring=None, exploit_incremental_learning=False,
               n_jobs=None, shuffle=True, random_state=None)
# 返回：train_sizes_abs, train_scores, test_scores
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `train_sizes` | `array_like` | 训练样本比例或绝对数量 | `np.linspace(0.3, 1.0, 5)` |
| `cv` | `int` 或 `splitter` | 交叉验证划分器，默认为 `None`（5 折） | `StratifiedKFold(3, shuffle=True, random_state=42)` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |
| `shuffle` | `bool` | 训练子集是否打乱，默认为 `True` | `True` |
| `random_state` | `int` | 随机种子 | `42` |
| `n_jobs` | `int` 或 `None` | 并行数，默认为 `None` | `-1` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), SVC())
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

trainSizes, trainScores, testScores = learning_curve(
    model, X, y, cv=cv,
    train_sizes=np.linspace(0.3, 1.0, 5),
    scoring="accuracy", shuffle=True, random_state=42,
)

print(f"训练集大小: {trainSizes}")
print(f"训练得分: {trainScores.mean(axis=1).round(3)}")
print(f"测试得分: {testScores.mean(axis=1).round(3)}")
```

#### 输出

```text
训练集大小: [ 36  49  63  76  90]
训练得分: [1.    0.986 0.974 0.969 0.972]
测试得分: [0.919 0.953 0.967 0.967 0.967]
```

#### 理解重点

- 随样本量增加，训练分数略降、验证分数上升是常见健康趋势
- 若两条曲线始终有大间隙，优先考虑正则化与特征简化
- 学习曲线是判定"继续收集数据是否有价值"的重要依据
- 训练分数和测试分数都低——特征不足或模型太简单

### `validation_curve`

#### 作用

验证曲线用于观察单个超参数变化对性能的影响。常用于确定参数大致有效区间，再进入细化搜索。训练曲线和验证曲线同时看，能识别过拟合拐点。

#### 重点方法

```python
validation_curve(estimator, X, y, *, param_name, param_range, groups=None,
                 cv=None, scoring=None, n_jobs=None, verbose=0)
# 返回：train_scores, test_scores
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `make_pipeline(StandardScaler(), SVC())` |
| `X` | `array_like` | 特征矩阵 | `iris.data` |
| `y` | `array_like` | 标签向量 | `iris.target` |
| `param_name` | `str` | 要扫描的参数名 | `"svc__C"` |
| `param_range` | `array_like` | 参数候选序列 | `np.logspace(-3, 2, 5)` |
| `cv` | `int` 或 `splitter` | 交叉验证折数，默认为 `None`（5 折） | `5` |
| `scoring` | `str` 或 `callable` | 评估指标，默认为 `None` | `"accuracy"` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
paramRange = np.logspace(-3, 2, 5)

trainScores, testScores = validation_curve(
    make_pipeline(StandardScaler(), SVC()),
    X, y,
    param_name="svc__C",
    param_range=paramRange,
    cv=5, scoring="accuracy",
)

print(f"C 值: {paramRange}")
print(f"测试得分: {testScores.mean(axis=1).round(3)}")
```

#### 输出

```text
C 值: [1.000e-03 1.778e-02 3.162e-01 5.623e+00 1.000e+02]
测试得分: [0.347 0.840 0.953 0.973 0.967]
```

#### 理解重点

- 参数过小通常欠拟合，参数过大可能过拟合
- 验证曲线能帮你发现"性能平台区"，降低调参敏感性
- 与网格搜索相比，验证曲线更偏诊断与解释
- 训练和测试分差最大处即为过拟合起始点

## 常见坑

1. 把时间序列数据用随机 K 折，导致评估严重乐观
2. 在极大参数空间直接网格搜索，计算成本不可控
3. 只看平均分不看标准差，忽略模型稳定性风险
4. `cross_val_score` 默认不返回训练分数——无法判断过拟合

## 小结

- 模型选择的核心不是"找到最高分"，而是"找到稳定可部署的方案"
- 推荐流程：先交叉验证基线 → 随机搜索粗调 → 网格精调
- 划分策略需匹配数据特征：分类用分层，时间序列按序划分
- 学习曲线和验证曲线是诊断工具——在调参前先确认方向

# sklearn 评估指标

## 本章目标

1. 掌握分类任务的核心指标：准确率、精确率、召回率、F1
2. 理解混淆矩阵与分类报告的阅读方式
3. 学会使用 ROC/PR 曲线与 AUC 评价概率输出质量
4. 掌握多分类指标中的 `average` 与 `multi_class` 参数
5. 学会回归指标与自定义评分函数的实践写法

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `accuracy_score(y_true, y_pred)` | 函数 | 分类准确率 |
| `precision_score(y_true, y_pred)` | 函数 | 精确率 |
| `recall_score(y_true, y_pred)` | 函数 | 召回率 |
| `f1_score(y_true, y_pred)` | 函数 | F1 分数 |
| `confusion_matrix(y_true, y_pred)` | 函数 | 生成混淆矩阵 |
| `classification_report(y_true, y_pred)` | 函数 | 输出分类统计摘要 |
| `roc_auc_score(y_true, y_score)` | 函数 | ROC AUC 值 |
| `roc_curve(y_true, y_score)` | 函数 | ROC 曲线坐标 |
| `precision_recall_curve(y_true, y_score)` | 函数 | PR 曲线坐标 |
| `r2_score(y_true, y_pred)` | 函数 | 回归拟合优度 |
| `mean_squared_error(y_true, y_pred)` | 函数 | 均方误差 |
| `mean_absolute_error(y_true, y_pred)` | 函数 | 平均绝对误差 |
| `make_scorer(score_func)` | 函数 | 自定义评分器 |

## 1. 分类指标基础

### `accuracy_score` / `precision_score` / `recall_score` / `f1_score`

#### 作用

准确率是全局正确比例，但在类别不平衡下可能失真。精确率关注"预测为正"的可靠性，召回率关注"真实为正"的覆盖率。F1 综合平衡精确率与召回率，适合综合评估。

#### 重点方法

```python
accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
                sample_weight=None, zero_division='warn')
recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
             sample_weight=None, zero_division='warn')
f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary',
         sample_weight=None, zero_division='warn')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_pred` | `array_like` | 预测标签 | `model.predict(X_test)` |
| `average` | `str` | 多分类聚合方式：`"binary"` / `"micro"` / `"macro"` / `"weighted"`，默认为 `"binary"` | `"macro"` |
| `pos_label` | `int` 或 `str` | 正类标签，默认为 `1` | `1` |
| `zero_division` | `str` 或 `float` | 分母为 0 时的返回值，`"warn"` 加警告返回 0 | `0` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
yPred = clf.predict(X_test)

print(f"准确率 (accuracy): {accuracy_score(y_test, yPred):.4f}")
print(f"精确率 (precision): {precision_score(y_test, yPred):.4f}")
print(f"召回率 (recall): {recall_score(y_test, yPred):.4f}")
print(f"F1 分数: {f1_score(y_test, yPred):.4f}")
```

#### 输出

```text
准确率 (accuracy): 0.9708
精确率 (precision): 0.9630
召回率 (recall): 0.9811
F1 分数: 0.9720
```

#### 理解重点

- 高准确率不代表业务可用，需结合误判类型分析
- 医疗、风控等高风险场景通常优先保证召回率
- 指标冲突时要回到业务代价函数做决策
- `zero_division` 控制当 TP+FP=0 时的行为——设为 0 而非报错

## 2. 混淆矩阵与分类报告

### `confusion_matrix` / `classification_report`

#### 作用

混淆矩阵直接给出 TN、FP、FN、TP，最利于误差归因。分类报告整合 precision/recall/F1/support，便于类别级比较。对二分类任务，FP 与 FN 往往对应不同业务风险。

#### 重点方法

```python
confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None,
                 normalize=None)
classification_report(y_true, y_pred, *, labels=None, target_names=None,
                      sample_weight=None, digits=2, output_dict=False,
                      zero_division='warn')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_pred` | `array_like` | 预测标签 | `model.predict(X_test)` |
| `normalize` | `str` | 混淆矩阵归一化：`"true"` / `"pred"` / `"all"` | `"true"` |
| `target_names` | `list[str]` | 报告中类别显示名称 | `["恶性", "良性"]` |
| `output_dict` | `bool` | 是否返回字典而非字符串，默认为 `False` | `True` |
| `digits` | `int` | 小数位数，默认为 `2` | `3` |

#### 示例代码

```python
from sklearn.metrics import confusion_matrix, classification_report

# 使用上一节的 y_test 和 yPred
cm = confusion_matrix(y_test, yPred)
print(f"混淆矩阵:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print(f"\n{classification_report(y_test, yPred, target_names=['恶性', '良性'])}")
```

#### 输出

```text
混淆矩阵:
[[ 62   1]
 [  4 104]]
TN=62, FP=1, FN=4, TP=104

              precision    recall  f1-score   support

          恶性       0.94      0.98      0.96        63
          良性       0.99      0.96      0.98       108

    accuracy                           0.97       171
   macro avg       0.96      0.97      0.97       171
weighted avg       0.97      0.97      0.97       171
```

#### 理解重点

- 混淆矩阵是阈值调优与误判成本分析的起点
- 观察 support 列可避免被样本量差异误导
- 模型上线前应把业务关注类别单独做阈值评估
- `output_dict=True` 适合将结果以编程方式消费

## 3. ROC 与 PR 曲线

### `roc_curve` / `roc_auc_score` / `precision_recall_curve`

#### 作用

ROC 关注 TPR 与 FPR 的权衡，PR 更适合类别不平衡场景。AUC 是曲线面积摘要，便于模型快速对比。这类指标依赖概率输出，通常使用 `predict_proba` 获取正类概率。

#### 重点方法

```python
roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None,
              max_fpr=None, multi_class='raise', labels=None)
precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_score` | `array_like` | 正类概率 | `clf.predict_proba(X_test)[:, 1]` |
| `pos_label` | `int` 或 `str` | 正类标签，默认为 `None`（自动判断） | `1` |
| `drop_intermediate` | `bool` | 是否删除次优阈值点，默认为 `True` | `True` |
| `multi_class` | `str` | 多分类 AUC 策略：`"ovr"` / `"ovo"` / `"raise"` | `"ovr"` |
| `max_fpr` | `float` | ROC AUC 只计算到指定 FPR 上限 | `0.1` |

#### 示例代码

```python
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve, auc)

# 使用上一节的 clf 和测试数据
yProba = clf.predict_proba(X_test)[:, 1]

rocAuc = roc_auc_score(y_test, yProba)
fpr, tpr, _ = roc_curve(y_test, yProba)

precision, recall, _ = precision_recall_curve(y_test, yProba)
prAuc = auc(recall, precision)

print(f"ROC AUC: {rocAuc:.4f}")
print(f"FPR 前 5 个阈值: {fpr[:5]}")
print(f"TPR 前 5 个阈值: {tpr[:5]}")
print(f"PR AUC: {prAuc:.4f}")
```

#### 输出

```text
ROC AUC: 0.9978
FPR 前 5 个阈值: [0.         0.         0.         0.         0.01587302]
TPR 前 5 个阈值: [0.         0.00943396 0.11320755 0.95283019 0.95283019]
PR AUC: 0.9981
```

#### 理解重点

- ROC AUC 高通常说明排序能力强，但阈值仍需业务化设定
- 正负样本极不平衡时，PR 曲线更有参考价值
- 概率未校准时，曲线仍可用但阈值解释要谨慎
- `roc_curve` 返回的阈值按降序排列，首尾阈值可能为 inf

## 4. 多分类指标

### `f1_score` / `roc_auc_score` 的聚合方式

#### 作用

多分类 F1 的 `average` 方式会影响结论。`micro` 强调总体样本，`macro` 强调类别公平，`weighted` 兼顾样本量。多分类 AUC 常用 `ovr` 与 `ovo` 两种策略。

#### 重点方法

```python
f1_score(y_true, y_pred, *, average='micro')     # micro / macro / weighted
roc_auc_score(y_true, y_score, *, multi_class='ovr')  # ovr / ovo
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `average` | `str` | F1 聚合方式：`"micro"` / `"macro"` / `"weighted"` | `"macro"` |
| `multi_class` | `str` | AUC 策略：`"ovr"`（one-vs-rest）/ `"ovo"`（one-vs-one） | `"ovr"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
yPred = clf.predict(X_test)
yProba = clf.predict_proba(X_test)

print("F1 不同 average:")
for avg in ["micro", "macro", "weighted"]:
    print(f"  {avg}: {f1_score(y_test, yPred, average=avg):.4f}")

print("多分类 ROC AUC:")
for strategy in ["ovr", "ovo"]:
    print(f"  {strategy}: {roc_auc_score(y_test, yProba, multi_class=strategy):.4f}")
```

#### 输出

```text
F1 不同 average:
  micro: 1.0000
  macro: 1.0000
  weighted: 1.0000
多分类 ROC AUC:
  ovr: 1.0000
  ovo: 1.0000
```

#### 理解重点

- 类别不均衡时应优先关注 `macro` 或类别级报告
- `ovr` 与 `ovo` 的差异在类别增多时更明显
- `micro` F1 等价于 accuracy——将各类别的 TP/FP/FN 先求和再计算
- 多分类概率质量可继续用校准曲线补充验证

## 5. 回归指标

### `r2_score` / `mean_squared_error` / `mean_absolute_error`

#### 作用

回归指标关注误差大小与解释能力两个维度。R² 衡量解释比例，MSE/RMSE/MAE 衡量误差幅度。业务上常用 MAE（可解释）与 RMSE（对大误差更敏感）组合。

#### 重点方法

```python
r2_score(y_true, y_pred, *, sample_weight=None, force_finite=True)
mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实值 | `y_test` |
| `y_pred` | `array_like` | 预测值 | `reg.predict(X_test)` |
| `sample_weight` | `array_like` | 样本权重 | `None` |
| `multioutput` | `str` 或 `array_like` | 多输出聚合方式：`"uniform_average"` / `"raw_values"` | `"uniform_average"` |
| `force_finite` | `bool` | R² 是否强制返回有限值（避免负无穷），默认为 `True` | `True` |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)
yPred = reg.predict(X_test)

r2 = r2_score(y_test, yPred)
mse = mean_squared_error(y_test, yPred)
mae = mean_absolute_error(y_test, yPred)

print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE: {mae:.4f}")
```

#### 输出

```text
R²: 0.4773
MSE: 2821.7509
RMSE: 53.1202
MAE: 42.7941
```

#### 理解重点

- R² 可比较解释力，但不代表误差绝对可接受
- RMSE 对异常误差更敏感，适合强调大偏差风险的场景
- MAE 与目标同单位，业务人员更易理解
- 指标应与业务容忍阈值一起解释——RMSE 53 在目标均值 ~152 下约 35% 误差

## 6. 自定义评分函数

### `make_scorer`

#### 作用

`make_scorer` 可把业务目标映射为可优化的评分函数。自定义评分可直接接入交叉验证与搜索器。适合将多指标加权成单一决策目标。

#### 重点方法

```python
make_scorer(score_func, *, greater_is_better=True, needs_proba=False,
            needs_threshold=False, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `score_func` | `callable` | 评分函数，签名为 `(y_true, y_pred, **kwargs)` | `custom_score` |
| `greater_is_better` | `bool` | 分数越大是否越好，默认为 `True` | `True` |
| `needs_proba` | `bool` | 是否需要概率输出，默认为 `False` | `False` |
| `needs_threshold` | `bool` | 是否需要决策阈值，默认为 `False` | `False` |
| `**kwargs` | `dict` | 传递给 `score_func` 的额外固定参数 | `{"pos_label": 1}` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_score

X, y = datasets.load_breast_cancer(return_X_y=True)
clf = LogisticRegression(max_iter=10000)

def customScore(yTrue, yPred):
    p = precision_score(yTrue, yPred)
    r = recall_score(yTrue, yPred)
    return 0.7 * p + 0.3 * r

scorer = make_scorer(customScore)
scores = cross_val_score(clf, X, y, cv=5, scoring=scorer)
print(f"自定义评分各折: {scores}")
print(f"平均: {scores.mean():.4f}")
```

#### 输出

```text
自定义评分各折: [0.9762 0.9861 0.9732 0.9815 0.9780]
平均: 0.9790
```

#### 理解重点

- 自定义指标应先做单元测试，确保方向与数值正确
- 若指标不可导或不稳定，搜索过程会更噪声化
- 评分函数最好与线上 KPI 口径保持一致
- `greater_is_better=False` 时搜索器会自动取负方向

## 常见坑

1. 直接用类别预测做 ROC/PR，导致曲线信息失真——必须输入概率
2. 多分类任务只看总体准确率，忽略少数类别表现
3. 自定义评分函数方向写反，搜索结果被误导
4. 混淆矩阵索引易搞混——`cm[0, 0]` 是 TN，`cm[1, 1]` 是 TP

## 小结

- 指标体系应围绕业务目标构建，不应只追求单一高分
- 建议建立固定评估模板：分类指标 + 混淆矩阵 + 曲线 + 业务加权分
- 有概率输出时优先看 ROC/PR AUC，仅类别标签时看 F1 + 分类报告
- 回归任务建议同时报告 R² + RMSE + MAE 三个维度

# sklearn 常用模型

## 本章目标

1. 建立 sklearn 常见模型族的整体认知与使用边界
2. 掌握线性、树、集成、核方法等模型的核心参数
3. 理解模型效果对数据缩放、特征分布的依赖关系
4. 学会用统一方式对比不同模型表现
5. 明确分类、聚类、降维模型在流程中的角色差异

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `LinearRegression()` | 构造器 | 普通最小二乘线性回归 |
| `Ridge(alpha)` / `Lasso(alpha)` / `ElasticNet(alpha)` | 构造器 | 带正则化的线性回归 |
| `LogisticRegression()` | 构造器 | 逻辑回归分类 |
| `DecisionTreeClassifier()` | 构造器 | 决策树分类 |
| `RandomForestClassifier()` | 构造器 | 随机森林集成分类 |
| `GradientBoostingClassifier()` | 构造器 | 梯度提升集成分类 |
| `SVC()` / `LinearSVC()` | 构造器 | 支持向量机 |
| `GaussianNB()` | 构造器 | 高斯朴素贝叶斯 |
| `KNeighborsClassifier(n_neighbors)` | 构造器 | K 近邻分类 |
| `KMeans(n_clusters)` / `DBSCAN()` | 构造器 | 聚类模型 |
| `PCA(n_components)` / `TSNE()` | 构造器 | 降维模型 |

## 1. 线性回归模型

### `LinearRegression` / `Ridge` / `Lasso` / `ElasticNet`

#### 作用

线性回归族可作为回归任务的强基线与可解释基线。Ridge 通过 L2 惩罚抑制系数震荡，Lasso 通过 L1 惩罚提供稀疏特征选择能力。ElasticNet 兼顾 L1 与 L2，适合特征相关性较强场景。

#### 重点方法

```python
LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
Ridge(alpha=1.0, *, fit_intercept=True, solver='auto')
Lasso(alpha=1.0, *, fit_intercept=True, max_iter=1000)
ElasticNet(alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, max_iter=1000)
# 核心方法：fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fit_intercept` | `bool` | 是否拟合截距项，默认为 `True` | `True` |
| `alpha` | `float` | 正则化强度，默认为 `1.0` | `0.1` |
| `l1_ratio` | `float` | ElasticNet：L1 在混合正则中的占比，`0` = Ridge，`1` = Lasso，默认为 `0.5` | `0.7` |
| `max_iter` | `int` | Lasso/ElasticNet：最大迭代次数，默认为 `1000` | `5000` |
| `solver` | `str` | Ridge：求解器，`"auto"` / `"svd"` / `"cholesky"` 等 | `"auto"` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `coef_` | `ndarray` | 各特征系数 |
| `intercept_` | `float` | 截距项 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: R² = {model.score(X_test, y_test):.4f}")
```

#### 输出

```text
LinearRegression: R² = 0.4773
Ridge (L2): R² = 0.4791
Lasso (L1): R² = 0.4770
ElasticNet: R² = 0.4432
```

#### 理解重点

- 正则化不是必然提分，而是控制方差与可解释性的手段
- 线性模型对特征尺度与共线性较敏感——建议先标准化
- Lasso 可将部分系数压到 0——天然具备特征选择功能
- 系数分布可作为特征重要性的初步参考

## 2. 逻辑回归

### `LogisticRegression`

#### 作用

逻辑回归是分类任务最强基线之一，稳定、可解释、可校准。`class_weight='balanced'` 可缓解类别不平衡。多分类默认使用 one-vs-rest 或 multinomial 方案。

#### 重点方法

```python
LogisticRegression(penalty='l2', *, C=1.0, fit_intercept=True, max_iter=100,
                   multi_class='auto', class_weight=None, solver='lbfgs')
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `penalty` | `str` | 正则类型，`"l2"` / `"l1"` / `"elasticnet"` / `None`，默认为 `"l2"` | `"l2"` |
| `C` | `float` | 正则强度的倒数（越小正则越强），默认为 `1.0` | `0.5` |
| `max_iter` | `int` | 最大迭代次数，默认为 `100` | `1000` |
| `class_weight` | `str` 或 `dict` | `"balanced"` 自动按类别频率加权 | `"balanced"` |
| `multi_class` | `str` | 多分类策略：`"auto"` / `"ovr"` / `"multinomial"` | `"multinomial"` |
| `solver` | `str` | 优化器：`"lbfgs"` / `"liblinear"` / `"saga"` 等 | `"lbfgs"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lrBal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(f"基础: 准确率 = {lr.score(X_test, y_test):.4f}")
print(f"balanced: 准确率 = {lrBal.score(X_test, y_test):.4f}")
print(f"预测概率 (前 3):\n{lr.predict_proba(X_test[:3])}")
```

#### 输出

```text
基础: 准确率 = 1.0000
balanced: 准确率 = 1.0000
预测概率 (前 3):
[[2.72513336e-04 2.51077806e-01 7.48649681e-01]
 [7.80283689e-01 2.18481740e-01 1.23457120e-03]
 [8.10557788e-01 1.88417402e-01 1.02481011e-03]]
```

#### 理解重点

- 逻辑回归在中小规模任务上常作为上线首选模型
- 类别不平衡下，建议配合 F1 与 Recall 联合评估
- 若线性边界不足，再考虑核方法或树模型
- `predict_proba` 输出的概率可配合校准方法进一步精调

## 3. 决策树

### `DecisionTreeClassifier`

#### 作用

决策树可捕捉非线性与特征交互，且无需标准化。易过拟合，通常需控制树深和叶子样本量。可直接输出特征重要性与树结构深度。

#### 重点方法

```python
DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       max_features=None, random_state=None)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `criterion` | `str` | 划分纯度指标：`"gini"` / `"entropy"`，默认为 `"gini"` | `"gini"` |
| `max_depth` | `int` 或 `None` | 树深度上限，`None` 不限制，默认为 `None` | `5` |
| `min_samples_split` | `int` 或 `float` | 内部节点最小划分样本数，默认为 `2` | `10` |
| `min_samples_leaf` | `int` 或 `float` | 叶子节点最小样本数，默认为 `1` | `5` |
| `max_features` | `int`、`str` 或 `None` | 每次划分考虑的候选特征数，默认为 `None`（全用） | `"sqrt"` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `feature_importances_` | `ndarray` | 特征重要性（Gini importance） |
| `n_features_in_` | `int` | 训练特征数 |
| `get_depth()` | `int` | 树实际深度 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2,
                            min_samples_leaf=1, criterion="gini", random_state=42)
dt.fit(X_train, y_train)

print(f"准确率: {dt.score(X_test, y_test):.4f}")
print(f"特征重要性: {dt.feature_importances_}")
print(f"树深度: {dt.get_depth()}")
```

#### 输出

```text
准确率: 1.0000
特征重要性: [0.         0.         0.56624158 0.43375842]
树深度: 4
```

#### 理解重点

- 决策树解释性强，但单树稳定性较弱
- 小数据集表现常很好，大数据集更推荐集成方法
- 重要性排序可反哺特征工程步骤
- `min_samples_leaf` 是最有效的过拟合控制参数之一

## 4. 集成模型

### `RandomForestClassifier` / `GradientBoostingClassifier` / `AdaBoostClassifier` / `HistGradientBoostingClassifier`

#### 作用

集成模型通过组合弱学习器获得更稳定的泛化能力。随机森林偏并行 bagging，梯度提升偏串行 boosting。同类任务中常可获得比单模型更鲁棒的结果。

#### 重点方法

```python
RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       max_features='sqrt', bootstrap=True, random_state=None)
GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1,
                           n_estimators=100, max_depth=3, random_state=None)
AdaBoostClassifier(estimator=None, *, n_estimators=50, learning_rate=1.0,
                   random_state=None)
HistGradientBoostingClassifier(loss='log_loss', *, learning_rate=0.1,
                               max_iter=100, max_depth=None, random_state=None)
# 核心方法：fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 基学习器数量（Hist 用 `max_iter`），默认为 `100` | `100` |
| `learning_rate` | `float` | Boosting 学习率，控制每棵树贡献，默认为 `0.1` | `0.05` |
| `max_depth` | `int` 或 `None` | 树深度上限，`None` 不限制，默认为 `None`（RF）/ `3`（GB） | `5` |
| `max_features` | `str` | 每次划分候选特征数：`"sqrt"` / `"log2"` / `None` | `"sqrt"` |
| `bootstrap` | `bool` | RF：是否自助采样，默认为 `True` | `True` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `feature_importances_` | `ndarray` | 特征重要性 |
| `estimators_` | `list` | 所有基学习器列表 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.4f}")
```

#### 输出

```text
RandomForest: 1.0000
GradientBoosting: 1.0000
AdaBoost: 0.9778
HistGradientBoosting: 0.9778
```

#### 理解重点

- 集成模型通常性能更优，但训练和解释成本更高
- 小样本下 boosting 更易过拟合，需关注验证曲线
- 随机森林是大多数分类任务的强基线——参数不敏感
- `HistGradientBoostingClassifier` 比传统 GBDT 快一个数量级，支持缺失值

## 5. SVM

### `SVC` / `LinearSVC`

#### 作用

SVM 对特征尺度敏感，通常必须先标准化。核函数选择决定决策边界形状与复杂度。线性不可分问题可用 RBF 或多项式核处理。

#### 重点方法

```python
SVC(*, C=1.0, kernel='rbf', gamma='scale', degree=3, probability=False,
    class_weight=None, random_state=None)
LinearSVC(penalty='l2', loss='squared_hinge', *, C=1.0, max_iter=1000,
          random_state=None)
# fit(X, y) → predict(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 误分类惩罚系数，越小正则越强，默认为 `1.0` | `10` |
| `kernel` | `str` | 核函数：`"linear"` / `"rbf"` / `"poly"` / `"sigmoid"`，默认为 `"rbf"` | `"rbf"` |
| `gamma` | `str` 或 `float` | 核系数，`"scale"` = 1/(n_features*Var)，`"auto"` = 1/n_features | `"scale"` |
| `degree` | `int` | 多项式核阶数，默认为 `3` | `3` |
| `probability` | `bool` | 是否输出概率（需额外训练），默认为 `False` | `True` |
| `max_iter` | `int` | LinearSVC 最大迭代次数，默认为 `1000` | `10000` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf"))
lsvc = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))

svc.fit(X_train, y_train)
lsvc.fit(X_train, y_train)
print(f"SVC (rbf): {svc.score(X_test, y_test):.4f}")
print(f"LinearSVC: {lsvc.score(X_test, y_test):.4f}")
```

#### 输出

```text
SVC (rbf): 0.9778
LinearSVC: 0.9778
```

#### 理解重点

- 先做标准化再训练 SVM 几乎是默认最佳实践
- RBF 核在非线性任务中常见，但需调节 `C` 与 `gamma`
- 边界更复杂不一定更好，需用验证曲线判断
- `LinearSVC` 比 `SVC(kernel='linear')` 更快，但不支持概率输出

## 6. 朴素贝叶斯

### `GaussianNB`

#### 作用

`GaussianNB` 训练速度快、参数少，是高效分类基线。假设特征条件独立，在现实中常不严格成立。对小数据集和快速原型非常友好。

#### 重点方法

```python
GaussianNB(*, priors=None, var_smoothing=1e-09)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别先验概率，`None` 从数据估计 | `[0.3, 0.7]` |
| `var_smoothing` | `float` | 方差平滑项，防止除零，默认为 `1e-09` | `1e-08` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

gnb = GaussianNB().fit(X_train, y_train)
print(f"GaussianNB: {gnb.score(X_test, y_test):.4f}")
```

#### 输出

```text
GaussianNB: 0.9778
```

#### 理解重点

- 朴素贝叶斯常用于"先跑通流程"的第一版模型
- 若准确率不够，可逐步切换到更复杂模型
- 模型虽简单，但在文本分类等场景仍常有竞争力
- 特征条件独立假设在连续数据上常不成立——但当数据量小时偏差可接受

## 7. K 近邻

### `KNeighborsClassifier`

#### 作用

KNN 基于邻域投票，直观但推理成本随样本数上升。对特征尺度敏感，应先标准化。`k` 的选择影响偏差-方差平衡。

#### 重点方法

```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
                     leaf_size=30, p=2, metric='minkowski', n_jobs=None)
# fit(X, y) → predict(X) → predict_proba(X) → score(X, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻个数 k，默认为 `5` | `3` |
| `weights` | `str` | `"uniform"` 等权投票 / `"distance"` 距离加权，默认为 `"uniform"` | `"distance"` |
| `algorithm` | `str` | 近邻搜索算法：`"auto"` / `"ball_tree"` / `"kd_tree"` / `"brute"` | `"auto"` |
| `p` | `int` | Minkowski 距离的 p 值，`2` = 欧氏距离，默认为 `2` | `2` |
| `metric` | `str` | 距离度量方式，默认为 `"minkowski"` | `"euclidean"` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(f"KNN (k=5): {knn.score(X_test, y_test):.4f}")
```

#### 输出

```text
KNN (k=5): 0.9778
```

#### 理解重点

- KNN 是距离模型，标准化优先级高
- 大规模数据上推理慢，常需近似检索或改用其他模型
- `k` 可通过验证曲线快速定位合理范围
- `weights='distance'` 可让近邻贡献更大——对小样本更友好

## 8. 聚类模型

### `KMeans` / `DBSCAN`

#### 作用

KMeans 需要预设簇数，最小化簇内平方和。DBSCAN 基于密度自动识别簇并标记噪声点。不同算法适配不同数据分布和噪声水平。

#### 重点方法

```python
KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300,
       random_state=None, algorithm='lloyd')
DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', algorithm='auto',
       n_jobs=None)
# fit(X) → predict(X) / fit_predict(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_clusters` | `int` | KMeans：聚类数，默认为 `8` | `4` |
| `init` | `str` | KMeans：初始化方法，`"k-means++"` / `"random"` | `"k-means++"` |
| `n_init` | `int` 或 `str` | KMeans：随机初始化次数，`"auto"` 自动选择，默认为 `"auto"` | `10` |
| `max_iter` | `int` | KMeans：单次运行最大迭代，默认为 `300` | `300` |
| `eps` | `float` | DBSCAN：邻域半径，默认为 `0.5` | `0.5` |
| `min_samples` | `int` | DBSCAN：核心点最小邻域样本数，默认为 `5` | `10` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `labels_` | `ndarray` | 每个样本的聚类标签（-1 = 噪声） |
| `cluster_centers_` | `ndarray` | KMeans：簇中心坐标 |

#### 示例代码

```python
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, yTrue = datasets.make_blobs(n_samples=300, centers=4, random_state=42)

labelsKm = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
labelsDb = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

nClustersDb = len(set(labelsDb)) - (1 if -1 in labelsDb else 0)
print(f"KMeans 轮廓系数: {silhouette_score(X, labelsKm):.4f}")
print(f"DBSCAN 聚类数: {nClustersDb}")
print(f"DBSCAN 噪声点数: {np.sum(labelsDb == -1)}")
```

#### 输出

```text
KMeans 轮廓系数: 0.7916
DBSCAN 聚类数: 4
DBSCAN 噪声点数: 0
```

#### 理解重点

- KMeans 对球形簇更友好，DBSCAN 对噪声和任意形状更稳健
- DBSCAN 的 `eps` 和 `min_samples` 对结果非常敏感——建议用 k-distance 图确定 `eps`
- 聚类结果应结合业务可解释性检验，不只看内部指标
- `n_clusters` 可通过肘部法或轮廓系数辅助确定

## 9. 降维模型

### `PCA` / `TSNE`

#### 作用

PCA 是线性降维，强调最大方差方向，适合特征压缩与去噪。t-SNE 是非线性嵌入，更偏可视化探索而非特征工程。降维后应检查类别可分性和信息保留程度。

#### 重点方法

```python
PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto',
    random_state=None)
TSNE(n_components=2, *, perplexity=30.0, learning_rate='auto',
     n_iter=1000, random_state=None)
# fit(X) → transform(X) / fit_transform(X)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` 或 `float` | 目标维度，`None` 保留全量，float=解释方差比例 | `2` / `0.95` |
| `whiten` | `bool` | PCA：白化变换，输出各分量方差为 1，默认为 `False` | `False` |
| `svd_solver` | `str` | PCA：SVD 求解器，`"auto"` 自动选择 | `"auto"` |
| `perplexity` | `float` | TSNE：平衡局部与全局的困惑度，默认为 `30.0` | `30.0` |
| `learning_rate` | `float` 或 `str` | TSNE：学习率，`"auto"` 自动，默认为 `"auto"` | `"auto"` |
| `n_iter` | `int` | TSNE：优化迭代次数，默认为 `1000` | `1000` |
| `random_state` | `int` | 随机种子 | `42` |

训练后属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `explained_variance_ratio_` | `ndarray` | PCA：各主成分解释方差比 |

#### 示例代码

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = datasets.load_iris(return_X_y=True)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)
print(f"PCA 解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA 累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

tsne = TSNE(n_components=2, random_state=42)
Xtsne = tsne.fit_transform(X)
print(f"t-SNE 输出形状: {Xtsne.shape}")
```

#### 输出

```text
PCA 解释方差比: [0.92461872 0.05306648]
PCA 累计解释方差: 0.9777
t-SNE 输出形状: (150, 2)
```

#### 理解重点

- PCA 可用于降噪和压缩，t-SNE 更适合可视化探索
- t-SNE 的空间距离不宜直接做定量解释——只适合辅助观察
- 降维后建模时应验证性能是否受损
- PCA 的 `n_components=0.95` 可自动保留 95% 解释方差的维度数

## 常见坑

1. 忘记对 SVM、KNN 做标准化，导致性能异常波动
2. 只比较准确率，不结合训练成本与可解释性
3. 将 t-SNE 结果直接作为下游生产特征而不做稳定性验证
4. 决策树 `max_depth=None` 在小数据上几乎必然过拟合
5. KMeans 的 `n_clusters` 靠猜——应结合肘部法和业务分群

## 小结

- 模型选择不应只看分数，还要考虑成本、稳定性与解释性
- 推荐先建立线性和树模型基线，再逐步引入复杂模型
- 距离模型（SVM、KNN）必须标准化；树模型无需
- 集成模型是多数表格任务的最强通用方案
- 降维和聚类用于辅助分析，不直接用于预测任务

# sklearn 工程技巧

## 本章目标

1. 掌握模型克隆与参数管理的工程化写法
2. 学会处理类别不平衡相关的权重配置
3. 了解如何编写自定义 Transformer 以接入 Pipeline
4. 掌握模型持久化、配置管理与版本检查实践
5. 学会快速检索 sklearn 可用估计器以提升研发效率

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `clone(estimator)` | 函数 | 克隆模型参数，不复制训练状态 |
| `estimator.get_params()` | 方法 | 获取全部超参数字典 |
| `estimator.set_params(**p)` | 方法 | 动态修改超参数 |
| `class_weight='balanced'` | 参数 | 自动按类别频率加权 |
| `compute_class_weight('balanced', classes, y)` | 函数 | 显式计算类别权重 |
| `BaseEstimator` + `TransformerMixin` | 基类 | 构建自定义转换器 |
| `joblib.dump(model, path)` | 函数 | 模型持久化保存 |
| `joblib.load(path)` | 函数 | 模型反序列化加载 |
| `set_config(...)` / `get_config()` | 函数 | sklearn 全局配置 |
| `all_estimators(type_filter)` | 函数 | 检索可用估计器列表 |

## 1. 模型克隆

### `clone`

#### 作用

`clone` 复制超参数配置，但不会复制拟合状态。适合在交叉验证或实验分支中复用模型配置。避免在不同实验间复用同一已训练对象造成污染。

#### 重点方法

```python
clone(estimator, *, safe=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待克隆的估计器对象 | `RandomForestClassifier()` |
| `safe` | `bool` | 仅允许 sklearn 估计器对象，默认为 `True` | `True` |

#### 示例代码

```python
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rfClone = clone(rf)

print(f"原模型已训练: {hasattr(rf, 'estimators_')}")
print(f"克隆模型已训练: {hasattr(rfClone, 'estimators_')}")
print(f"参数相同: {rf.get_params()['n_estimators'] == rfClone.get_params()['n_estimators']}")
```

#### 输出

```text
原模型已训练: True
克隆模型已训练: False
参数相同: True
```

#### 理解重点

- clone 是"复制配置"，不是"复制权重"
- 用于多实验并行时，可降低对象共享导致的副作用
- 与交叉验证评估流程天然契合——每折需要干净的模型
- `clone` 内部调用 `get_params` 获取配置，因此依赖正确的 `__init__`

## 2. get_params 与 set_params

### `get_params` / `set_params`

#### 作用

`get_params` 提供统一参数字典，便于日志、配置化、追踪。`set_params` 可动态更新模型超参数。该接口也是网格搜索和随机搜索的底层依赖。

#### 重点方法

```python
estimator.get_params(deep=True)
estimator.set_params(**params)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `deep` | `bool` | 是否递归获取子估计器参数，默认为 `True` | `True` |
| `**params` | `dict` | 要修改的参数名与值 | `n_estimators=50, max_depth=5` |

#### 示例代码

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
params = rf.get_params()
print(f"n_estimators: {params['n_estimators']}")
print(f"max_depth: {params['max_depth']}")

rf.set_params(n_estimators=50, max_depth=5)
print(f"修改后 n_estimators: {rf.get_params()['n_estimators']}")
print(f"修改后 max_depth: {rf.get_params()['max_depth']}")
```

#### 输出

```text
n_estimators: 100
max_depth: None
修改后 n_estimators: 50
修改后 max_depth: 5
```

#### 理解重点

- 参数接口是自动化实验系统的关键入口
- 推荐将关键参数记录到实验日志，便于复现和回滚
- 对 Pipeline 对象也适用同样机制——配合双下划线语法
- 自定义估计器必须将 `__init__` 参数以同名属性存储才能被正确读取

## 3. 类别权重处理

### `class_weight` / `compute_class_weight` / `compute_sample_weight`

#### 作用

类别不平衡时，模型会偏向多数类。`class_weight='balanced'` 根据类别频率自动调整损失权重。手动计算权重可获得更透明的类别补偿机制。

#### 重点方法

```python
LogisticRegression(class_weight='balanced', max_iter=1000)
compute_class_weight(class_weight='balanced', *, classes=None, y)
compute_sample_weight(class_weight='balanced', *, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `class_weight` | `str` 或 `dict` | `"balanced"` 自动 / 或手动 `{0: 0.6, 1: 2.5}` | `"balanced"` |
| `classes` | `array_like` | `compute_class_weight`：类别集合 | `np.unique(y)` |
| `y` | `array_like` | 标签向量 | `y` |

#### 示例代码

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
clfBal = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)

print(f"类别分布: {np.bincount(y_train)}")
print(f"无权重 少数类F1: {classification_report(y_test, clf.predict(X_test), output_dict=True, zero_division=0)['1']['f1-score']:.3f}")
print(f"balanced 少数类F1: {classification_report(y_test, clfBal.predict(X_test), output_dict=True, zero_division=0)['1']['f1-score']:.3f}")

classWeights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
print(f"计算得类别权重: {classWeights}")
```

#### 输出

```text
类别分布: [630  70]
无权重 少数类F1: 0.606
balanced 少数类F1: 0.740
计算得类别权重: [0.55555556 3.36842105]
```

#### 理解重点

- 准确率可能上升但少数类表现下降，需警惕指标幻觉
- 权重策略能提升召回，但可能牺牲精确率
- 业务上应提前定义错判代价，再决定权重方案
- 权重过大可能导致训练不稳定，需配合验证集检查

## 4. 自定义 Transformer

### `BaseEstimator` + `TransformerMixin`

#### 作用

继承 `BaseEstimator` 与 `TransformerMixin` 可无缝接入 Pipeline。只要实现 `fit` 与 `transform`，即可构建可复用处理器。自定义转换器是把业务规则工程化的关键手段。

#### 重点方法

```python
class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ...):    # 初始化参数
    def fit(self, X, y=None):   # 返回 self
    def transform(self, X):     # 返回变换后 ndarray
```

#### 示例代码

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """对数变换器：对每个特征做 log(x + offset)"""
    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)

X = np.array([[1, 10], [100, 1000]])
lt = LogTransformer(offset=1)
print(f"原始:\n{X}")
print(f"\n变换后:\n{lt.fit_transform(X)}")
```

#### 输出

```text
原始:
[[   1   10]
 [ 100 1000]]

变换后:
[[0.69314718 2.39789527]
 [4.61512052 6.90875478]]
```

#### 理解重点

- 自定义转换器应保持无副作用和确定性输出
- 建议给转换器写单测，验证边界值与缺失值行为
- 复杂逻辑拆分为多个小转换器更易维护
- `TransformerMixin` 自动提供 `fit_transform` 方法

## 5. 模型持久化

### `joblib.dump` / `joblib.load`

#### 作用

`joblib` 是 sklearn 模型持久化的常用方案，对 numpy 数组序列化做了优化。压缩可减少体积，但会增加读写开销。加载后的预测一致性必须验证。

#### 重点方法

```python
joblib.dump(value, filename, compress=0, protocol=None)
joblib.load(filename, mmap_mode=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `value` | `any` | 待序列化对象（模型、Pipeline 等） | `rf` |
| `filename` | `str` | 文件路径 | `"./model.joblib"` |
| `compress` | `int` | 压缩等级：`0` / `3`，默认为 `0` | `3` |
| `mmap_mode` | `str` 或 `None` | 加载时内存映射模式 | `None` |

#### 示例代码

```python
import joblib
import os
import numpy as np
from tempfile import mkdtemp
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
tmpDir = mkdtemp()
path = os.path.join(tmpDir, "model.joblib")

joblib.dump(rf, path)
rfLoaded = joblib.load(path)

print(f"预测一致: {np.array_equal(rfLoaded.predict(X_test), rf.predict(X_test))}")
print(f"文件大小: {os.path.getsize(path) / 1024:.1f} KB")
```

#### 输出

```text
预测一致: True
文件大小: 182.4 KB
```

#### 理解重点

- 模型文件应与训练代码版本、依赖版本一起管理
- 线上加载前要做一致性回归测试
- 生产环境优先使用稳定路径与权限管理，不用临时目录
- `compress=3` 可将体积压缩到原始的 50%-60%

## 6. 全局配置

### `set_config` / `get_config`

#### 作用

`set_config` 可修改 sklearn 的全局行为配置。`transform_output='pandas'` 在数据分析阶段更友好。配置变更应及时恢复，避免影响其他流程。

#### 重点方法

```python
sklearn.get_config()
sklearn.set_config(transform_output='pandas')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `transform_output` | `str` | 变换器输出格式：`"default"` / `"pandas"`，默认为 `"default"` | `"pandas"` |
| `assume_finite` | `bool` | 是否假设数据已有限，默认为 `False` | `True` |
| `working_memory` | `int` | 算法内存上限（MB），默认为 `1024` | `2048` |
| `enable_metadata_routing` | `bool` | 启用元数据路由（1.4+），默认为 `False` | `True` |

#### 示例代码

```python
from sklearn import get_config, set_config

print(f"当前 transform_output: {get_config()['transform_output']}")

set_config(transform_output="pandas")
print(f"设置后: {get_config()['transform_output']}")

set_config(transform_output="default")
print(f"恢复后: {get_config()['transform_output']}")
```

#### 输出

```text
当前 transform_output: default
设置后: pandas
恢复后: default
```

#### 理解重点

- 全局配置适合实验和分析，不宜在库代码中隐式修改
- 多人协作时建议显式记录配置变更
- 配置差异可能导致同一代码输出格式不一致
- `transform_output='pandas'` 让所有 `transform` 输出 DataFrame——在 EDA 阶段非常实用

## 7. 版本检查

### `sklearn.__version__` + `packaging.version`

#### 作用

不同 sklearn 版本 API 可用性不同，需显式校验。使用 `packaging.version` 比字符串比较更可靠。版本门控能避免线上环境 API 不匹配。

#### 重点方法

```python
import sklearn; sklearn.__version__
from packaging import version
version.parse(sklearn.__version__) >= version.parse('1.2')
```

#### 示例代码

```python
import sklearn
from packaging import version

print(f"sklearn 版本: {sklearn.__version__}")
print(f">= 1.0: {version.parse(sklearn.__version__) >= version.parse('1.0')}")
print(f">= 1.2 (set_output API): {version.parse(sklearn.__version__) >= version.parse('1.2')}")
print(f">= 1.6: {version.parse(sklearn.__version__) >= version.parse('1.6')}")
```

#### 输出

```text
sklearn 版本: 1.6.1
>= 1.0: True
>= 1.2 (set_output API): True
>= 1.6: True
```

#### 理解重点

- 版本门控应成为工具脚本与部署脚本的标准步骤
- 当文档示例依赖新特性时，必须标注最低版本要求
- 建议将关键依赖版本固定在项目配置中
- `packaging.version.parse` 正确处理 `1.10 > 1.2` 的语义化比较

## 8. all_estimators 快速检索

### `all_estimators`

#### 作用

`all_estimators` 可快速查看当前环境可用估计器。适合做模型候选池构建与自动化实验初始化。`type_filter` 可按任务类型筛选。

#### 重点方法

```python
all_estimators(type_filter=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `type_filter` | `str` 或 `None` | 筛选类型：`"classifier"` / `"regressor"` / `"transformer"` / `"cluster"` | `"classifier"` |

#### 示例代码

```python
from sklearn.utils import all_estimators

classifiers = all_estimators(type_filter="classifier")
regressors = all_estimators(type_filter="regressor")
transformers = all_estimators(type_filter="transformer")

print(f"分类器: {len(classifiers)} 个")
print(f"回归器: {len(regressors)} 个")
print(f"转换器: {len(transformers)} 个")
print(f"\n分类器前 5: {[name for name, _ in classifiers[:5]]}")
```

#### 输出

```text
分类器: 49 个
回归器: 55 个
转换器: 93 个

分类器前 5: ['AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB', 'CalibratedClassifierCV', 'CategoricalNB']
```

#### 理解重点

- 该工具可用于快速探索，但不替代模型选择流程
- 不同版本中估计器数量会变化，应结合版本信息解读
- 可与自动化评估框架结合构建候选模型库
- 返回的是类对象（不是实例）——需实例化后才能使用

## 常见坑

1. 把 clone 误认为深拷贝训练状态，导致实验误判
2. 忽略版本差异直接调用新 API，引发环境兼容问题
3. 持久化后不做预测一致性校验，埋下线上风险
4. 自定义 Transformer 未将 `__init__` 参数存为同名属性——`get_params` 失效

## 小结

- 技巧章节的核心是把"能跑"升级为"可维护、可复现、可部署"
- 推荐将权重策略、版本校验、持久化检查纳入项目模板
- clone + get_params/set_params 是自动化实验的基础设施
- 自定义 Transformer + Pipeline 可将业务规则工程化
- 模型文件、版本、配置三者应统一管理
