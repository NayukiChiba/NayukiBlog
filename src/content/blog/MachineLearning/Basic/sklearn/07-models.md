---
title: Scikit-learn 常用模型
date: 2026-01-20
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 学习常用的分类、回归和聚类模型
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

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
