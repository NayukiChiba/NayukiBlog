---
title: Scikit-learn 特征工程
date: 2026-01-18
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 学习特征选择和特征降维方法
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

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
