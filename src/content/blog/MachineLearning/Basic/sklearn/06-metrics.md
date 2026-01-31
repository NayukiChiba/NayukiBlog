---
title: Scikit-learn 评估指标
date: 2026-01-19
category: MachineLearning/Basic/sklearn
tags:
  - Python
  - Scikit-learn
description: 掌握分类和回归模型的评估指标
image: https://img.yumeko.site/file/blog/Sklearning.jpg
status: public
---

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
