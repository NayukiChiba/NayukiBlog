---
title: 模型决策可视化
date: 2026-01-23
category: MachineLearning/Basic/visualization
tags:
  - Python
  - 可视化
description: 可视化决策边界和学习曲线
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

# 模型决策过程可视化

## 决策边界

```python
# 创建网格
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 预测并绘制
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=y)
```

![07_boundary](https://img.yumeko.site/file/articles/viz/07_boundary.png)

## 决策树可视化

```python
from sklearn.tree import plot_tree

plot_tree(clf, filled=True, feature_names=names)
```

![07_tree](https://img.yumeko.site/file/articles/viz/07_tree.png)

## 特征重要性

```python
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
ax.barh(range(len(importances)), importances[indices])
```

![07_importance](https://img.yumeko.site/file/articles/viz/07_importance.png)

## 练习

```bash
python Basic/Visualization/07_model_decision.py
```
