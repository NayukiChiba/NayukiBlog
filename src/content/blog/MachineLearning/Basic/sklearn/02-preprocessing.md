---
title: Scikit-learn 数据预处理
date: 2026-01-18
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 掌握数据标准化、编码和缺失值处理
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

# 数据预处理

## 1. 缩放器对比

| 缩放器           | 公式                                         | 输出范围 | 对异常值 | 适用场景                   |
| ---------------- | -------------------------------------------- | -------- | :------: | -------------------------- |
| `StandardScaler` | $z = \frac{x - \mu}{\sigma}$                 | 无界     | ⚠️ 敏感  | 正态分布数据，SVM/逻辑回归 |
| `MinMaxScaler`   | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | $[0,1]$  | ⚠️ 敏感  | 神经网络，需要有界输出     |
| `RobustScaler`   | $x' = \frac{x - median}{IQR}$                | 无界     | ✅ 鲁棒  | 含异常值的数据             |
| `MaxAbsScaler`   | $x' = \frac{x}{\vert x_{max} \vert}$         | $[-1,1]$ | ⚠️ 敏感  | 稀疏数据                   |

### 缩放器可视化

下图展示了不同缩放器处理含异常值数据的效果：

![02_scalers](https://img.yumeko.site/file/articles/sklearn/02_scalers.png)

### 选择建议

```
数据有异常值？
├─ 是 → RobustScaler
└─ 否 → 需要固定范围？
         ├─ 是 → MinMaxScaler
         └─ 否 → StandardScaler
```

---

## 2. StandardScaler

**作用**: 标准化，转换为均值 $\mu = 0$、标准差 $\sigma = 1$

**公式**: $z = \frac{x - \mu}{\sigma}$

```python
StandardScaler(
    copy=True,        # 是否复制数据
    with_mean=True,   # 是否减去均值（中心化）
    with_std=True     # 是否除以标准差
)
```

| 参数        | 默认 | 作用     | ⚠️ 什么时候改              |
| ----------- | ---- | -------- | -------------------------- |
| `copy`      | True | 复制数据 | 数据量大节省内存设 False   |
| `with_mean` | True | 中心化   | **稀疏矩阵必须设 False！** |
| `with_std`  | True | 缩放     | 只想中心化不缩放时设 False |

### 训练后属性

```python
scaler.mean_    # 每个特征的均值
scaler.scale_   # 每个特征的标准差
scaler.var_     # 每个特征的方差
```

> ⚠️ **稀疏矩阵警告**: 稀疏数据用 StandardScaler 必须 `with_mean=False`，否则会破坏稀疏性导致内存爆炸！

---

## 3. MinMaxScaler

**作用**: 归一化，缩放到指定范围

**公式**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}} \times (max - min) + min$

```python
MinMaxScaler(
    feature_range=(0, 1),  # 目标范围
    copy=True,
    clip=False             # 是否裁剪超出范围的值
)
```

| 参数            | 默认   | 作用     | ⚠️ 什么时候改                     |
| --------------- | ------ | -------- | --------------------------------- |
| `feature_range` | (0, 1) | 目标范围 | 需要 [-1,1] 时改 `(-1, 1)`        |
| `clip`          | False  | 裁剪边界 | 测试数据可能超出训练范围时设 True |

### 训练后属性

```python
scaler.data_min_   # 每个特征的最小值
scaler.data_max_   # 每个特征的最大值
scaler.data_range_ # max - min
```

---

## 4. RobustScaler

**作用**: 使用中位数和 IQR 缩放，对异常值鲁棒

**公式**: $x' = \frac{x - median}{Q_3 - Q_1}$

```python
RobustScaler(
    with_centering=True,        # 是否减去中位数
    with_scaling=True,          # 是否除以 IQR
    quantile_range=(25.0, 75.0) # IQR 分位数范围
)
```

| 参数             | 默认     | 作用     | ⚠️ 什么时候改             |
| ---------------- | -------- | -------- | ------------------------- |
| `quantile_range` | (25, 75) | IQR 范围 | 异常值极端时改 `(10, 90)` |
| `with_centering` | True     | 减中位数 | 稀疏数据设 False          |

---

## 5. 类别编码

### 5.1 编码器对比

| 编码器           | 输出              | 用途              |
| ---------------- | ----------------- | ----------------- |
| `LabelEncoder`   | 整数 $(0,1,2...)$ | 目标变量 $y$ 编码 |
| `OrdinalEncoder` | 整数矩阵          | 有序类别特征      |
| `OneHotEncoder`  | 二进制矩阵        | 无序类别特征      |

### 编码方法可视化

下图展示了 LabelEncoder 和 OneHotEncoder 的区别：

![02_encoding](https://img.yumeko.site/file/articles/sklearn/02_encoding.png)

### 5.2 OneHotEncoder 参数详解

```python
OneHotEncoder(
    categories='auto',       # 类别列表
    drop=None,               # 丢弃策略
    sparse_output=True,      # 输出类型
    handle_unknown='error',  # 未知类别处理
    min_frequency=None,      # 最小频率
    max_categories=None      # 最大类别数
)
```

| 参数             | 默认    | 作用         | ⚠️ 什么时候改                   |
| ---------------- | ------- | ------------ | ------------------------------- |
| `sparse_output`  | True    | 输出稀疏矩阵 | 想要普通数组设 **False**        |
| `drop`           | None    | 丢弃类别     | 回归模型设 `'first'` 避免共线性 |
| `handle_unknown` | 'error' | 未知类别     | **生产环境必须设 `'ignore'`！** |
| `min_frequency`  | None    | 合并稀有类别 | 类别太多时设 `5` 或 `0.01`      |

> ⚠️ **超级重要**: `handle_unknown='error'` 默认会在遇到新类别时报错！生产环境必须改成 `'ignore'`

---

## 6. 缺失值处理

### 6.1 SimpleImputer

```python
SimpleImputer(
    missing_values=np.nan,  # 缺失值标记
    strategy='mean',        # 填充策略
    fill_value=None,        # constant 时的填充值
    add_indicator=False     # 是否添加缺失指示列
)
```

| strategy 值       | 说明   | 适用                |
| ----------------- | ------ | ------------------- |
| `'mean'`          | 均值   | 数值，正态分布      |
| `'median'`        | 中位数 | 数值，有异常值      |
| `'most_frequent'` | 众数   | 类别或数值          |
| `'constant'`      | 固定值 | 需要指定 fill_value |

### 6.2 KNNImputer

```python
KNNImputer(
    n_neighbors=5,    # 近邻数
    weights='uniform' # 权重：'uniform' 或 'distance'
)
```

用 K 近邻的值填充，效果通常比简单策略好，但更慢。

---

## 7. ColumnTransformer

**作用**: 对不同列应用不同预处理（**实际项目必用**）

```python
ColumnTransformer(
    transformers=[
        ('name1', transformer1, columns1),
        ('name2', transformer2, columns2),
    ],
    remainder='drop',      # 未指定的列
    n_jobs=None           # 并行数
)
```

| 参数        | 默认   | 作用       | ⚠️ 什么时候改            |
| ----------- | ------ | ---------- | ------------------------ |
| `remainder` | `drop` | 剩余列处理 | `'passthrough'` 保留原样 |
| `n_jobs`    | None   | 并行       | `-1` 用全部 CPU          |

### columns 指定方式

```python
# 方式1: 列名列表
['age', 'income']

# 方式2: 列索引
[0, 1, 2]

# 方式3: 列选择器（推荐）
from sklearn.compose import make_column_selector as selector
selector(dtype_include='number')   # 所有数值列
selector(dtype_include='object')   # 所有字符串列
```

### 完整示例

```python
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), selector(dtype_include='number')),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), selector(dtype_include='object'))
])
```
