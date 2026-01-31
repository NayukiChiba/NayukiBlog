---
title: 模型评估可视化
date: 2026-01-23
category: MachineLearning/Basic/visualization
tags:
  - Python
  - 可视化
description: 可视化混淆矩阵和 ROC 曲线
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

# 模型性能评估可视化

## 混淆矩阵

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

![08_confusion](https://img.yumeko.site/file/articles/viz/08_confusion.png)

## ROC 曲线

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], 'r--')
```

![08_roc](https://img.yumeko.site/file/articles/viz/08_roc.png)

## 学习曲线

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(clf, X, y)

ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
ax.plot(train_sizes, test_scores.mean(axis=1), label='Validation')
```

![08_learning](https://img.yumeko.site/file/articles/viz/08_learning.png)

## 练习

```bash
python Basic/Visualization/08_model_evaluation.py
```
