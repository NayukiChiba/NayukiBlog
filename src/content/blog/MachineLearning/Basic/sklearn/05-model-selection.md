---
title: Scikit-learn 模型选择
date: 2026-01-19
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 学习交叉验证和超参数调优
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

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

> ⚠️ **分类问题必须用 StratifiedKFold**，否则某折可能缺少某类别！

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
