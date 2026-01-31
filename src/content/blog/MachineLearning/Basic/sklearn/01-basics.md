---
title: Scikit-learn 基础入门
date: 2026-01-18
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: Scikit-learn 机器学习库基础入门
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
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
