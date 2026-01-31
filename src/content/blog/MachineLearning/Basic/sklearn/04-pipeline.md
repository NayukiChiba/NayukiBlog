---
title: Scikit-learn Pipeline
date: 2026-01-19
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 掌握 Pipeline 构建可复用的工作流
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

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
