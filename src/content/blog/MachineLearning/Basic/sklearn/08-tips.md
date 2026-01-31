---
title: Scikit-learn 实用技巧
date: 2026-01-20
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 掌握模型持久化和最佳实践
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

# 实用技巧

> 对应代码: [code/08_tips.py](../code/08_tips.py)

## 目录

- [1. 模型克隆](#1-模型克隆)
- [2. 类别不平衡](#2-类别不平衡)
- [3. 自定义估计器](#3-自定义估计器)
- [4. 模型持久化](#4-模型持久化)
- [5. 常见错误](#5-常见错误)

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
