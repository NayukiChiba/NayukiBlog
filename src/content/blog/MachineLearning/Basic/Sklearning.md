---
title: Sklearn 全指南
date: 2026-01-20
category: MachineLearning/Basic
tags:
  - Python
  - 基础
description: Scikit-learn (sklearn) 库实践指南，涵盖数据预处理、特征工程、Pipeline构建、模型选择、交叉验证、超参数调优、评估指标及常用模型（线性模型、树模型、SVM等）的详细应用与可视化。
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: published
---

# 基础入门

## 运行方式

```python
# 方式1: 直接运行
python code/01_basics.py

# 方式2: 导入单个函数
from code.01_basics import demo_load_datasets
demo_load_datasets()

# 方式3: 运行全部
from code.01_basics import demo_all
demo_all()
```

---

## 1. 数据集加载

### 1.1 内置数据集

| 函数                   | 数据集   | 类型   | 样本数 |
| ---------------------- | -------- | ------ | ------ |
| `load_iris()`          | 鸢尾花   | 分类   | 150    |
| `load_wine()`          | 葡萄酒   | 分类   | 178    |
| `load_breast_cancer()` | 乳腺癌   | 二分类 | 569    |
| `load_digits()`        | 手写数字 | 分类   | 1797   |
| `load_diabetes()`      | 糖尿病   | 回归   | 442    |

### 数据集可视化

下图展示了鸢尾花数据集的特征分布和类别分布：

![01_datasets](https://img.yumeko.site/file/articles/sklearn/01_datasets.png)

### 1.2 load_xxx() 参数

```python
datasets.load_iris(
    return_X_y=False,    # True: 直接返回 (X, y) 元组
    as_frame=False       # True: 返回 DataFrame 格式
)
```

| 参数         | 默认  | 说明                                   |
| ------------ | ----- | -------------------------------------- |
| `return_X_y` | False | True 时返回 `(X, y)` 而不是 Bunch 对象 |
| `as_frame`   | False | True 时特征和目标都是 DataFrame        |

### 1.3 生成人工数据

| 函数                    | 用途   | 关键参数                                |
| ----------------------- | ------ | --------------------------------------- |
| `make_classification()` | 分类   | `n_classes`, `n_informative`, `weights` |
| `make_regression()`     | 回归   | `noise`                                 |
| `make_blobs()`          | 聚类   | `centers`, `cluster_std`                |
| `make_moons()`          | 月牙形 | `noise`                                 |
| `make_circles()`        | 同心圆 | `noise`, `factor`                       |

### 人工数据集可视化

下图展示了各种人工生成数据集：

![01_generate_data](https://img.yumeko.site/file/articles/sklearn/01_generate_data.png)

---

## 2. 数据划分

### train_test_split 参数

```python
train_test_split(
    X, y,
    test_size=0.25,      # 测试集比例
    train_size=None,
    random_state=None,   # 随机种子
    shuffle=True,        # 是否打乱
    stratify=None        # 分层抽样
)
```

| 参数           | 默认 | ⚠️ 什么时候改                     |
| -------------- | ---- | --------------------------------- |
| `test_size`    | 0.25 | 数据少用 0.2，数据多用 0.1        |
| `random_state` | None | **必须设固定值**保证可复现！如 42 |
| `shuffle`      | True | 时间序列数据设 False              |
| `stratify`     | None | **分类问题必须设 `stratify=y`！** |

> ⚠️ 分类问题不设 `stratify=y` 可能导致某类别全在训练集或测试集！

### 数据划分可视化

下图展示了训练集/测试集划分和分层抽样效果：

![01_train_test_split](https://img.yumeko.site/file/articles/sklearn/01_train_test_split.png)

---

## 3. 估计器 API

### 3.1 统一接口

| 方法                   | 说明      | 适用对象         |
| ---------------------- | --------- | ---------------- |
| `fit(X, y)`            | 训练      | 所有             |
| `predict(X)`           | 预测      | 分类器、回归器   |
| `transform(X)`         | 转换数据  | 预处理器、降维器 |
| `fit_transform(X)`     | 训练+转换 | 预处理器         |
| `score(X, y)`          | 评估      | 分类器、回归器   |
| `predict_proba(X)`     | 预测概率  | 部分分类器       |
| `get_params()`         | 获取参数  | 所有             |
| `set_params(**params)` | 设置参数  | 所有             |

### 3.2 训练后属性 (带下划线后缀)

```python
model.classes_          # 类别列表
model.n_features_in_    # 输入特征数
model.feature_names_in_ # 特征名（DataFrame输入时）
model.coef_             # 线性模型系数
model.intercept_        # 线性模型截距
```

### 3.3 常见问题

**Q: fit() vs fit_transform() 区别？**

- `fit()`: 只训练，用于预测模型
- `fit_transform()`: 训练+转换，用于预处理器

**Q: 测试集为什么只能用 transform() 不能用 fit_transform()？**

```python
# ✅ 正确
scaler.fit_transform(X_train)  # 训练集
scaler.transform(X_test)       # 测试集

# ❌ 错误 - 数据泄露
scaler.fit_transform(X_test)
```

**Q: random_state 作用？**
设置随机种子保证每次结果一致。常用 `random_state=42`

### KNN 模型可视化

下图展示了 KNN 分类器的决策边界和不同 k 值的准确率：

![01_knn](https://img.yumeko.site/file/articles/sklearn/01_knn.png)

# 数据预处理

## 1. 缩放器对比

| 缩放器              | 公式                                                        | 输出范围     | 对异常值  | 适用场景            |
| ---------------- | --------------------------------------------------------- | -------- | :---: | --------------- |
| `StandardScaler` | $z = \displaystyle\frac{x - \mu}{\sigma}$                 | 无界       | ⚠️ 敏感 | 正态分布数据，SVM/逻辑回归 |
| `MinMaxScaler`   | $x' = \displaystyle\frac{x - x_{min}}{x_{max} - x_{min}}$ | $[0,1]$  | ⚠️ 敏感 | 神经网络，需要有界输出     |
| `RobustScaler`   | $x' = \displaystyle\frac{x - median}{IQR}$                | 无界       | ✅ 鲁棒  | 含异常值的数据         |
| `MaxAbsScaler`   | $x' = \displaystyle\frac{x}{\vert x_{max} \vert}$         | $[-1,1]$ | ⚠️ 敏感 | 稀疏数据            |

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

**公式**: $$z = \frac{x - \mu}{\sigma}$$

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

**公式**: $$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \times (max - min) + min$$

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

**公式**: 
$$x' = \frac{x - median}{Q_3 - Q_1}$$

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

> [!] `handle_unknown='error'` 默认会在遇到新类别时报错！生产环境必须改成 `'ignore'`

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

# 特征工程

## 1. 文本特征提取

### 1.1 向量化器对比

| 向量化器            | 输出         | 优点         | 缺点             | 适用         |
| ------------------- | ------------ | ------------ | ---------------- | ------------ |
| `CountVectorizer`   | 词频(整数)   | 简单         | 无法体现词重要性 | 朴素贝叶斯   |
| `TfidfVectorizer`   | TF-IDF(浮点) | 体现词重要性 | 需要全部数据     | 通用文本分类 |
| `HashingVectorizer` | 哈希值       | 内存高效     | 无法逆映射       | 大规模数据   |

### 1.2 TfidfVectorizer 参数详解

```python
TfidfVectorizer(
    max_features=None,      # 最多保留N个词
    min_df=1,               # 词至少出现在N个文档
    max_df=1.0,             # 词最多出现在多少比例文档
    stop_words=None,        # 停用词
    ngram_range=(1, 1),     # n-gram范围
    norm='l2',              # 归一化方式
    sublinear_tf=False      # 是否用 1+log(tf)
)
```

| 参数           | 默认   | 作用         | ⚠️ 什么时候改                  |
| -------------- | ------ | ------------ | ------------------------------ |
| `max_features` | None   | 限制词表大小 | 数据量大时设 5000-10000        |
| `min_df`       | 1      | 过滤低频词   | 设 `2` 或 `0.01` 过滤拼写错误  |
| `max_df`       | 1.0    | 过滤高频词   | 设 `0.9` 过滤无意义高频词      |
| `stop_words`   | None   | 停用词       | 英文 `'english'`，中文需自定义 |
| `ngram_range`  | (1, 1) | 只用单词     | 包含词组设 `(1, 2)`            |
| `sublinear_tf` | False  | 对数化词频   | 某词出现极多时设 True          |

### 1.3 DictVectorizer

将字典列表转为特征矩阵，自动独热编码字符串值：

```python
DictVectorizer(
    sparse=True,    # 输出稀疏矩阵
    sort=True       # 按特征名排序
)
```

---

## 2. 多项式特征

### 2.1 PolynomialFeatures

**作用**: 生成多项式和交互特征

```python
PolynomialFeatures(
    degree=2,               # 最高次数
    interaction_only=False, # 只保留交互项
    include_bias=True       # 包含常数项1
)
```

| 参数               | 默认  | 作用       | ⚠️ 什么时候改                |
| ------------------ | ----- | ---------- | ---------------------------- |
| `degree`           | 2     | 多项式阶数 | **阶数高易过拟合！** 一般2-3 |
| `interaction_only` | False | 只保留 a×b | True 不生成 a², b²           |
| `include_bias`     | True  | 包含常数1  | 与有截距模型一起用时设 False |

### 2.2 特征数量增长

| 原特征数 | degree=2 | degree=3 |
| -------- | -------- | -------- |
| 2        | 6        | 10       |
| 5        | 21       | 56       |
| 10       | 66       | 286      |

> ⚠️ **注意**: 高次多项式特征数暴增，容易过拟合且计算慢！

---

## 3. 特征选择

### 3.1 方法分类

| 类型       | 方法                | 说明            |
| ---------- | ------------------- | --------------- |
| **过滤法** | `VarianceThreshold` | 移除低方差特征  |
| **过滤法** | `SelectKBest`       | 按统计指标选K个 |
| **包装法** | `RFE`               | 递归消除        |
| **嵌入法** | `SelectFromModel`   | 基于模型重要性  |

### 3.2 VarianceThreshold

```python
VarianceThreshold(threshold=0.0)  # 方差阈值
```

移除方差低于阈值的特征。threshold=0 移除常量特征。

### 3.3 SelectKBest

```python
SelectKBest(
    score_func=f_classif,  # 评分函数
    k=10                   # 选择K个
)
```

| score_func               | 适用 | 说明                   |
| ------------------------ | ---- | ---------------------- |
| `f_classif`              | 分类 | ANOVA F值              |
| `chi2`                   | 分类 | 卡方检验（需非负特征） |
| `mutual_info_classif`    | 分类 | 互信息                 |
| `f_regression`           | 回归 | F值                    |
| `mutual_info_regression` | 回归 | 互信息                 |

### 3.4 RFE - 递归特征消除

```python
RFE(
    estimator,              # 基础模型（需有coef_或feature_importances_）
    n_features_to_select=None,  # 选择特征数
    step=1                  # 每次移除数量
)
```

逐步移除最不重要的特征，直到剩余指定数量。

### 3.5 SelectFromModel

```python
SelectFromModel(
    estimator,            # 基础模型
    threshold='mean',     # 阈值：'mean', 'median', 数值
    prefit=False         # 模型是否已训练
)
```

| threshold    | 说明               |
| ------------ | ------------------ |
| `'mean'`     | 重要性 > 均值      |
| `'median'`   | 重要性 > 中位数    |
| `'1.5*mean'` | 重要性 > 1.5倍均值 |
| 数值如 `0.1` | 重要性 > 0.1       |

### 3.6 通用方法

所有选择器都有：

```python
selector.fit(X, y)
selector.transform(X)
selector.fit_transform(X, y)
selector.get_support()           # 返回布尔数组
selector.get_support(indices=True)  # 返回索引
```

# Pipeline 流水线

## 1. 为什么用 Pipeline

| 问题         | Pipeline 解决方案      |
| ------------ | ---------------------- |
| 代码冗长     | 一行 fit、一行 predict |
| 数据泄露风险 | 自动在正确的数据上 fit |
| 交叉验证复杂 | 整体作为一个估计器     |
| 部署困难     | 保存一个对象即可       |

---

## 2. 创建 Pipeline

### 2.1 显式命名

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),    # (名称, 转换器)
    ('pca', PCA(n_components=2)),
    ('svm', SVC())                   # 最后一步通常是模型
])
```

### 2.2 自动命名

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    SVC()
)
# 自动命名: standardscaler, pca, svc
```

### 2.3 使用

```python
pipe.fit(X_train, y_train)
pipe.predict(X_test)
pipe.score(X_test, y_test)
```

---

## 3. 参数访问

### 3.1 访问步骤

```python
pipe.steps              # [(name, estimator), ...]
pipe.named_steps        # {'name': estimator, ...}
pipe.named_steps['pca'] # 通过名称
pipe[0]                 # 通过索引
pipe[-1]                # 最后一步
pipe[:2]                # 切片（返回新 Pipeline）
```

### 3.2 设置参数

格式: `步骤名__参数名`

```python
pipe.set_params(
    pca__n_components=3,
    svm__C=10,
    svm__kernel='rbf'
)

# 获取参数
pipe.get_params()
```

### 3.3 嵌套 Pipeline 参数

```python
# preprocessor 是 ColumnTransformer
# num 是其中一个转换器
# imputer 是 num Pipeline 中的步骤
pipe.set_params(preprocessor__num__imputer__strategy='mean')
```

---

## 4. 与 GridSearchCV 结合

```python
param_grid = {
    'pca__n_components': [2, 3, 4],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

### 动态跳过步骤

```python
param_grid = [
    # 不用 PCA
    {'pca': ['passthrough'], 'svm__C': [1, 10]},
    # 用 PCA
    {'pca__n_components': [2, 3], 'svm__C': [1, 10]}
]
```

---

## 5. ColumnTransformer

对不同列应用不同预处理：

```python
from sklearn.compose import ColumnTransformer, make_column_selector as selector

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, ['age', 'income']),
    ('cat', categorical_pipeline, ['gender', 'city'])
])
```

### 5.1 参数详解

```python
ColumnTransformer(
    transformers=[...],
    remainder='drop',      # 剩余列处理
    sparse_threshold=0.3,
    n_jobs=None
)
```

| 参数        | 默认   | 选项                                |
| ----------- | ------ | ----------------------------------- |
| `remainder` | 'drop' | `'drop'` 丢弃, `'passthrough'` 保留 |
| `n_jobs`    | None   | `-1` 并行加速                       |

### 5.2 列选择方式

```python
# 列名列表
['age', 'income']

# 列索引
[0, 1, 2]

# 自动选择器
selector(dtype_include='number')    # 数值列
selector(dtype_include='object')    # 字符串列
selector(dtype_exclude='datetime')  # 排除日期
```

### 5.3 完整示例

```python
full_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), selector(dtype_include='number')),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), selector(dtype_include='object'))
    ])),
    ('classifier', LogisticRegression())
])
```

---

## 6. TransformedTargetRegressor

对目标变量 y 进行变换：

```python
from sklearn.compose import TransformedTargetRegressor

ttr = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,           # y -> log(1+y)
    inverse_func=np.expm1    # 逆变换
)
```

| 参数           | 说明                    |
| -------------- | ----------------------- |
| `func`         | 变换函数                |
| `inverse_func` | 逆变换函数              |
| `transformer`  | 也可传入 sklearn 转换器 |

---

## 7. Pipeline 缓存

```python
from tempfile import mkdtemp

pipe = Pipeline([...], memory=mkdtemp())
```

缓存中间步骤结果，GridSearchCV 时避免重复计算。

# 模型选择与调参

---

## 1. 交叉验证

### 1.1 cross_val_score

```python
cross_val_score(
    estimator,
    X, y,
    cv=5,              # 折数
    scoring='accuracy' # 评分指标
)
```

返回每折的得分数组。

### 交叉验证可视化

下图展示了 5 折交叉验证的各折得分：

![05_cross_val](https://img.yumeko.site/file/articles/sklearn/05_cross_val.png)

### 1.2 cross_validate

```python
cross_validate(
    estimator, X, y,
    cv=5,
    scoring=['accuracy', 'f1'],  # 多个指标
    return_train_score=True,     # 返回训练分数
    return_estimator=True        # 返回训练好的模型
)
```

返回字典，包含 `test_accuracy`, `train_accuracy`, `fit_time` 等。

### 1.3 常用评分指标

| 类型 | scoring                    | 说明         |
| ---- | -------------------------- | ------------ |
| 分类 | `'accuracy'`               | 准确率       |
| 分类 | `'f1'`                     | F1（二分类） |
| 分类 | `'f1_macro'`               | F1 宏平均    |
| 分类 | `'roc_auc'`                | ROC AUC      |
| 回归 | `'r2'`                     | R²           |
| 回归 | `'neg_mean_squared_error'` | 负MSE        |

---

## 2. 划分策略

| 划分器            | 适用场景             |
| ----------------- | -------------------- |
| `KFold`           | 通用                 |
| `StratifiedKFold` | 分类（保持类别比例） |
| `ShuffleSplit`    | 大数据集             |
| `TimeSeriesSplit` | 时间序列             |
| `LeaveOneOut`     | 小数据集             |

### 2.1 参数详解

```python
KFold(
    n_splits=5,      # 折数
    shuffle=False,   # 是否打乱
    random_state=None
)

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

TimeSeriesSplit(n_splits=5)  # 无 shuffle
```

> [!WARNING] 注意
>  **分类问题必须用 StratifiedKFold**，否则某折可能缺少某类别！

---

## 3. 网格搜索

### 3.1 GridSearchCV 参数

```python
GridSearchCV(
    estimator,
    param_grid,           # 参数网格
    scoring=None,         # 评分指标
    n_jobs=None,          # 并行数
    refit=True,           # 用最佳参数重训练
    cv=5,                 # 交叉验证
    verbose=0,            # 输出详细度
    return_train_score=False
)
```

| 参数      | 默认 | ⚠️ 什么时候改                 |
| --------- | ---- | ----------------------------- |
| `scoring` | None | 必须指定！分类用 `'accuracy'` |
| `n_jobs`  | None | 设 `-1` 用全部CPU             |
| `cv`      | 5    | 小数据用 3 或 10              |
| `verbose` | 0    | 看进度设 1 或 2               |
| `refit`   | True | 只找参数不训练设 False        |

### 3.2 param_grid 格式

```python
# 方式1: 字典
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# 方式2: 字典列表（不同组合）
param_grid = [
    {'kernel': ['linear'], 'C': [1, 10]},
    {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.1, 1]}
]
```

### 3.3 结果访问

```python
grid.best_params_      # 最佳参数
grid.best_score_       # 最佳交叉验证分数
grid.best_estimator_   # 最佳模型（已训练）
grid.cv_results_       # 详细结果字典
```

---

## 4. 随机搜索

### 4.1 RandomizedSearchCV

```python
from scipy.stats import uniform, loguniform

RandomizedSearchCV(
    estimator,
    param_distributions,  # 参数分布
    n_iter=10,           # 采样次数
    scoring=None,
    n_jobs=None,
    cv=5,
    random_state=None
)
```

| 参数                  | 说明                             |
| --------------------- | -------------------------------- |
| `n_iter`              | 采样组合数，越大越可能找到好参数 |
| `param_distributions` | 可以是列表或分布对象             |

### 4.2 参数分布

```python
from scipy.stats import uniform, loguniform, randint

param_distributions = {
    'C': loguniform(0.01, 100),    # 对数均匀分布
    'gamma': loguniform(1e-4, 1),
    'kernel': ['rbf', 'linear'],   # 离散值用列表
    'n_estimators': randint(50, 200)  # 整数均匀分布
}
```

### 4.3 Grid vs Random

|              | GridSearchCV | RandomizedSearchCV |
| ------------ | ------------ | ------------------ |
| 搜索方式     | 遍历所有组合 | 随机采样           |
| 参数多时     | 很慢         | 快                 |
| 保证找到最优 | ✅           | ❌ 但通常够好      |
| 适用         | 参数少       | 参数多             |

---

## 5. 学习曲线

诊断过拟合/欠拟合：

```python
learning_curve(
    estimator, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)
```

返回: `train_sizes, train_scores, test_scores`

### 学习曲线可视化

下图展示了模型的学习曲线：

![05_learning_curve](https://img.yumeko.site/file/articles/sklearn/05_learning_curve.png)

### 解读

| 现象           | 诊断   | 解决                 |
| -------------- | ------ | -------------------- |
| 训练高、测试低 | 过拟合 | 更多数据、正则化     |
| 两者都低       | 欠拟合 | 更复杂模型、更多特征 |
| 两者都高且接近 | 理想   | 保持                 |

---

## 6. 验证曲线

分析单个参数的影响：

```python
validation_curve(
    estimator, X, y,
    param_name='C',
    param_range=np.logspace(-3, 3, 7),
    cv=5
)
```

返回: `train_scores, test_scores`

### 验证曲线可视化

下图展示了 SVC 参数 C 的验证曲线：

![05_validation_curve](https://img.yumeko.site/file/articles/sklearn/05_validation_curve.png)

# 评估指标与可视化

---

## 1. 分类指标

### 1.1 基础指标

| 指标      | 公式                            | 适用场景           |
| --------- | ------------------------------- | ------------------ |
| Accuracy  | $\frac{TP+TN}{TP+TN+FP+FN}$     | 类别平衡           |
| Precision | $\frac{TP}{TP+FP}$              | 关注假正例代价     |
| Recall    | $\frac{TP}{TP+FN}$              | 关注假负例代价     |
| F1        | $\frac{2 \cdot P \cdot R}{P+R}$ | 平衡精确率和召回率 |

### 分类指标可视化

下图展示了乾腘癌数据集上的分类指标：

![06_classification_metrics](https://img.yumeko.site/file/articles/sklearn/06_classification_metrics.png)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

### 1.2 多分类 average 参数

```python
f1_score(y_true, y_pred, average='macro')
```

| average      | 说明                         |
| ------------ | ---------------------------- |
| `'binary'`   | 二分类默认，只算正类         |
| `'micro'`    | 全局计算 TP/FP/FN            |
| `'macro'`    | 各类别平均（不考虑类别大小） |
| `'weighted'` | 按类别样本数加权平均         |

### 1.3 ROC AUC

**ROC 曲线**: 不同阈值下 TPR (召回率) 与 FPR (假正识率) 的曲线。

- **TPR (True Positive Rate)**: $TPR = \frac{TP}{TP+FN}$
- **FPR (False Positive Rate)**: $FPR = \frac{FP}{FP+TN}$
- **AUC**: 曲线下面积，$1$ 为完美，$0.5$ 为随机

### ROC 和 PR 曲线可视化

下图展示了 ROC 曲线和 Precision-Recall 曲线：

![06_roc_pr](https://img.yumeko.site/file/articles/sklearn/06_roc_pr.png)

```python
roc_auc_score(
    y_true,
    y_score,           # predict_proba 的结果
    multi_class='ovr'  # 多分类: 'ovr' 或 'ovo'
)
```

### 1.4 classification_report

```python
print(classification_report(y_true, y_pred, target_names=['负类', '正类']))
```

输出精确率、召回率、F1、支持度的完整报告。

---

## 2. 回归指标

| 指标  | 公式                             | 说明                       |
| ----- | -------------------------------- | -------------------------- |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$  | 决定系数，$1$ 最好         |
| MSE   | $\frac{1}{n}\sum(y - \hat{y})^2$ | 均方误差，对大误差敏感     |
| RMSE  | $\sqrt{MSE}$                     | 均方根误差                 |
| MAE   | $\frac{1}{n}\sum y - \hat{y}$    | 平均绝对误差，对异常值鲁棒 |

### 回归指标可视化

下图展示了回归模型的预测 vs 真实值和残差分布：

![06_regression_metrics](<https://img.yumeko.site/file/articles/sklearn/06_regression_metrics(1).png>)

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2_score(y_true, y_pred)
mean_squared_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
```

---

## 3. 可视化工具

### 混淆矩阵可视化

下图展示了混淆矩阵及其解读：

![06_confusion_matrix](<https://img.yumeko.site/file/articles/sklearn/06_confusion_matrix(1).png>)

### 3.1 ConfusionMatrixDisplay

```python
from sklearn.metrics import ConfusionMatrixDisplay

# 方式1: 从预测结果
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

# 方式2: 从估计器
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

# 参数
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=['负', '正'],  # 标签
    normalize='true',             # 归一化: 'true', 'pred', 'all'
    cmap='Blues'                  # 颜色
)
```

### 3.2 RocCurveDisplay

```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
RocCurveDisplay.from_predictions(y_true, y_proba)
```

### 3.3 PrecisionRecallDisplay

```python
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
PrecisionRecallDisplay.from_predictions(y_true, y_proba)
```

### 3.4 plot_tree

```python
from sklearn.tree import plot_tree

plot_tree(
    decision_tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True
)
```

### 3.5 DecisionBoundaryDisplay

```python
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
    model, X,  # X 必须是 2 维
    response_method='predict',  # 'predict' 或 'predict_proba'
    alpha=0.5
)
```

---

## 4. 自定义评分

### 4.1 make_scorer

```python
from sklearn.metrics import make_scorer

def my_score(y_true, y_pred):
    # 返回数值，越大越好
    return ...

my_scorer = make_scorer(my_score)

# 使用
cross_val_score(model, X, y, scoring=my_scorer)
GridSearchCV(model, params, scoring=my_scorer)
```

### 4.2 参数

```python
make_scorer(
    score_func,
    greater_is_better=True,  # False 表示越小越好
    **kwargs                 # 传给 score_func 的额外参数
)
```

### 4.3 示例

```python
from sklearn.metrics import fbeta_score

# F2 分数（更重视召回率）
f2_scorer = make_scorer(fbeta_score, beta=2)
```

# 常用模型速查

---

## 1. 线性模型

### 1.1 回归

| 模型               | 正则化               | 损失函数                                 |
| ------------------ | -------------------- | ---------------------------------------- |
| `LinearRegression` | 无                   | $\|y - X\beta\|^2$                       |
| `Ridge`            | L2 ($\|\beta\|_2^2$) | $\|y - X\beta\|^2 + \alpha\|\beta\|_2^2$ |
| `Lasso`            | L1 ($\|\beta\|_1$)   | $\|y - X\beta\|^2 + \alpha\|\beta\|_1$   |
| `ElasticNet`       | L1+L2                | 混合正则化                               |

### 线性模型可视化

下图展示了不同正则化方法的系数对比：

![07_linear_models](https://img.yumeko.site/file/articles/sklearn/07_linear_models.png)

```python
Ridge(alpha=1.0)  # alpha 越大，正则化越强
Lasso(alpha=0.1)  # 会产生稀疏系数（特征选择）
```

### 1.2 LogisticRegression

```python
LogisticRegression(
    penalty='l2',           # 正则化: 'l1', 'l2', 'elasticnet', None
    C=1.0,                  # 正则化强度的倒数（C大=弱正则化）
    solver='lbfgs',         # 优化器
    max_iter=100,           # 最大迭代
    class_weight=None,      # 类别权重
    multi_class='auto'      # 多分类策略
)
```

| 参数           | 值            | 说明                              |
| -------------- | ------------- | --------------------------------- |
| `C`            | 1.0           | **C越大正则化越弱**，过拟合时减小 |
| `class_weight` | `'balanced'`  | 类别不平衡时使用                  |
| `solver`       | `'liblinear'` | L1正则化必须用这个                |

---

## 2. 树模型与集成

### 2.1 DecisionTreeClassifier

```python
DecisionTreeClassifier(
    max_depth=None,         # 最大深度
    min_samples_split=2,    # 分裂最小样本
    min_samples_leaf=1,     # 叶节点最小样本
    criterion='gini',       # 'gini' 或 'entropy'
    class_weight=None
)
```

### 2.2 RandomForestClassifier

```python
RandomForestClassifier(
    n_estimators=100,       # 树的数量
    max_depth=None,         # 树的深度
    max_features='sqrt',    # 分裂时考虑的特征数
    bootstrap=True,         # 有放回采样
    oob_score=False,        # 袋外评估
    n_jobs=None,            # 并行
    class_weight=None
)
```

| 参数           | 建议值  | 说明               |
| -------------- | ------- | ------------------ |
| `n_estimators` | 100-500 | 越多越好但越慢     |
| `max_depth`    | 5-20    | 过拟合时设置       |
| `n_jobs`       | -1      | 并行加速           |
| `oob_score`    | True    | 免交叉验证快速评估 |

### 2.3 GradientBoostingClassifier

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,      # 学习率
    max_depth=3,            # 通常设小一些
    subsample=1.0           # 样本采样比例
)
```

> 💡 **learning_rate 和 n_estimators 要一起调**：learning_rate 小时需要更多 n_estimators

---

## 3. SVM

### 3.1 SVC

```python
SVC(
    C=1.0,              # 软间隔，C大=硬间隔
    kernel='rbf',       # 核函数
    gamma='scale',      # 核系数
    class_weight=None,
    probability=False   # 开启后可用 predict_proba
)
```

| kernel     | 适用               |
| ---------- | ------------------ |
| `'linear'` | 线性可分，高维数据 |
| `'rbf'`    | 通用，默认选择     |
| `'poly'`   | 多项式核           |

| 参数    | 说明                           |
| ------- | ------------------------------ |
| `C`     | 大=拟合训练数据，小=更平滑边界 |
| `gamma` | 大=更复杂边界，小=更平滑       |

### SVM 决策边界可视化

下图展示了不同核函数的决策边界：

![07_svm](https://img.yumeko.site/file/articles/sklearn/07_svm.png)

> ⚠️ **SVM 必须标准化数据！**

---

## 4. 朴素贝叶斯

| 模型            | 适用数据          |
| --------------- | ----------------- |
| `GaussianNB`    | 连续特征          |
| `MultinomialNB` | 计数/词频（非负） |
| `BernoulliNB`   | 二值特征          |

```python
GaussianNB()  # 无超参数

MultinomialNB(alpha=1.0)  # alpha = 拉普拉斯平滑

BernoulliNB(binarize=0.0)  # 二值化阈值
```

---

## 5. 聚类

### 5.1 KMeans

```python
KMeans(
    n_clusters=8,       # 聚类数
    init='k-means++',   # 初始化
    n_init=10,          # 运行次数
    max_iter=300
)
```

### 5.2 DBSCAN

```python
DBSCAN(
    eps=0.5,            # 邻域半径
    min_samples=5       # 核心点最小样本数
)
```

| 参数             | 效果                   |
| ---------------- | ---------------------- |
| `eps` 大         | 更大的簇               |
| `min_samples` 大 | 更少的噪声点，更小的簇 |

### 聚类算法可视化

下图展示了 KMeans 和 DBSCAN 的聚类结果对比：

![07_clustering](https://img.yumeko.site/file/articles/sklearn/07_clustering.png)

---

## 6. 降维

### 6.1 PCA

```python
PCA(
    n_components=2,     # 保留组件数，可以是整数或比例如 0.95
    svd_solver='auto'
)
```

### 6.2 t-SNE

```python
TSNE(
    n_components=2,
    perplexity=30,      # 困惑度，5-50
    learning_rate='auto'
)
```

> ⚠️ **t-SNE 只能 fit_transform，不能 transform 新数据！**

### 降维结果可视化

下图展示了 PCA 和 t-SNE 的降维结果对比：

![07_dimensionality_reduction](https://img.yumeko.site/file/articles/sklearn/07_dimensionality_reduction.png)

# 实用技巧
---

## 1. 模型克隆

```python
from sklearn.base import clone

rf_clone = clone(rf)  # 复制参数，不复制训练状态
```

用途：需要用相同配置训练多个独立模型时。

---

## 2. 类别不平衡

### 2.1 class_weight 参数

```python
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')
SVC(class_weight='balanced')
```

| 值              | 说明                   |
| --------------- | ---------------------- |
| `None`          | 默认，所有类别权重=1   |
| `'balanced'`    | 自动计算，少数类权重高 |
| `{0: 1, 1: 10}` | 手动指定各类别权重     |

### 2.2 计算权重

```python
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# 类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

# 样本权重
sample_weights = compute_sample_weight('balanced', y)
```

### 2.3 何时使用

| 比例  | 建议                    |
| ----- | ----------------------- |
| 2:1   | 可以尝试 balanced       |
| 10:1  | 建议使用 balanced       |
| 100:1 | 必须使用 + 考虑其他方法 |

---

## 3. 自定义估计器

### 3.1 自定义转换器

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=1):
        self.param1 = param1

    def fit(self, X, y=None):
        # 学习参数（可选）
        self.learned_param_ = X.mean()  # 以下划线结尾
        return self

    def transform(self, X):
        return X - self.learned_param_
```

### 3.2 自定义分类器

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # 训练逻辑
        return self

    def predict(self, X):
        # 预测逻辑
        return predictions
```

### 3.3 规则

1. `__init__` 只保存参数，不做计算
2. 学习到的属性以 `_` 结尾（如 `classes_`）
3. `fit` 必须返回 `self`

---

## 4. 模型持久化

### 4.1 joblib vs pickle

|        | joblib         | pickle |
| ------ | -------------- | ------ |
| 推荐度 | ✅ sklearn推荐 | 可用   |
| 大数组 | 更快           | 较慢   |
| 压缩   | 支持           | 不支持 |

### 4.2 使用 joblib

```python
import joblib

# 保存
joblib.dump(model, 'model.joblib')

# 加载
model = joblib.load('model.joblib')

# 压缩保存（1-9级）
joblib.dump(model, 'model.joblib', compress=3)
```

### 4.3 保存整个 Pipeline

```python
joblib.dump(pipeline, 'pipeline.joblib')
```

### 4.4 版本兼容

```python
# 保存时记录版本
import sklearn
model_info = {
    'model': model,
    'sklearn_version': sklearn.__version__
}
joblib.dump(model_info, 'model_with_version.joblib')
```

---

## 5. 常见错误

| 错误                           | 原因             | 解决                      |
| ------------------------------ | ---------------- | ------------------------- |
| `ConvergenceWarning`           | 未收敛           | 增大 `max_iter`           |
| `ValueError: unknown category` | 新类别           | `handle_unknown='ignore'` |
| 稀疏矩阵内存爆炸               | 标准化破坏稀疏性 | `with_mean=False`         |
| 分类效果差                     | 类别不平衡       | `class_weight='balanced'` |
| Pipeline参数无效               | 格式错误         | 用 `步骤名__参数名`       |
| 结果不可复现                   | 未设随机种子     | 设 `random_state=42`      |

---

## 6. 版本检查

```python
import sklearn
print(sklearn.__version__)

from packaging import version
if version.parse(sklearn.__version__) >= version.parse("1.0"):
    print("新版本功能可用")
```

---

## 7. 查看可用估计器

```python
from sklearn.utils import all_estimators

# 所有分类器
classifiers = all_estimators(type_filter='classifier')

# 所有回归器
regressors = all_estimators(type_filter='regressor')
```
