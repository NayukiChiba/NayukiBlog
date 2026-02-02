---
title: 数据可视化指南
date: 2026-01-24
category: MachineLearning/Basic
tags:
  - Python
  - Matplotlib
  - 可视化
  - Seaborn
description: Python 数据可视化全指南，系统讲解 Matplotlib、Seaborn、Pandas 及 Plotly 的核心用法。内容涵盖基础图表、探索性数据分析（EDA）、数据预处理可视化、机器学习模型决策与评估可视化，以及制作专业报告与交互式图表的最佳实践。
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

# Matplotlib 基础入门

## 图表结构

```
Figure (画布)
└── Axes (绑图区域)
    ├── Axis (坐标轴)
    ├── Title
    └── Legend
```

## 基本绑图

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Title')
ax.legend()
plt.show()
```

## 线条样式

| 符号 | 样式   |
| ---- | ------ |
| `-`  | 实线   |
| `--` | 虚线   |
| `:`  | 点线   |
| `-.` | 点划线 |

## 标记符号

`o` (圆), `s` (方), `^` (三角), `*` (星), `x` (叉)

## 子图布局

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
```

## 练习

```bash
python Basic/Visualization/01_matplotlib_basics.py
```

![01_subplots](https://img.yumeko.site/file/articles/viz/01_subplots.png)

![01_basic](https://img.yumeko.site/file/articles/viz/01_basic.png)

![01_line_styles](https://img.yumeko.site/file/articles/viz/01_line_styles.png)

![01_markers](https://img.yumeko.site/file/articles/viz/01_markers.png)

# Matplotlib 常用图表类型

## 柱状图

```python
ax.bar(x, y)       # 垂直
ax.barh(x, y)      # 水平
```

![02_bar](https://img.yumeko.site/file/articles/viz/02_bar.png)

## 散点图

```python
ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
```

![02_scatter](https://img.yumeko.site/file/articles/viz/02_scatter.png)

## 直方图

```python
ax.hist(data, bins=30, edgecolor='black')
```

![02_histogram](https://img.yumeko.site/file/articles/viz/02_histogram.png)

## 饼图

```python
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
```

![02_pie](https://img.yumeko.site/file/articles/viz/02_pie.png)

## 箱线图

```python
ax.boxplot(data)
```

![02_boxplot](https://img.yumeko.site/file/articles/viz/02_boxplot.png)

## 练习

```bash
python Basic/Visualization/02_matplotlib_charts.py
```

# Seaborn 库入门

## 基本设置

```python
import seaborn as sns
sns.set_theme(style='whitegrid')
```

## 分类图

```python
sns.barplot(x='category', y='value', hue='group', data=df)
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)
```

![03_catplot](https://img.yumeko.site/file/articles/viz/03_catplot.png)

## 分布图

```python
sns.histplot(data, kde=True)
sns.kdeplot(data, fill=True)
```

![03_distplot](https://img.yumeko.site/file/articles/viz/03_distplot.png)

## 回归图

```python
sns.regplot(x='x', y='y', data=df)
```

![03_regplot](https://img.yumeko.site/file/articles/viz/03_regplot.png)

## 热力图

```python
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

![03_heatmap](https://img.yumeko.site/file/articles/viz/03_heatmap.png)

## 配对图

```python
sns.pairplot(df, hue='category')
```

![03_pairplot](https://img.yumeko.site/file/articles/viz/03_pairplot.png)

## 练习

```bash
python Basic/Visualization/03_seaborn.py
```

# Pandas 数据可视化

## DataFrame.plot()

```python
df.plot()                    # 折线图
df.plot(kind='bar')          # 柱状图
df.plot(kind='area')         # 面积图
df.plot(kind='box')          # 箱线图
```

![04_df_plot](https://img.yumeko.site/file/articles/viz/04_df_plot.png)

![04_series_plot](https://img.yumeko.site/file/articles/viz/04_series_plot.png)

## kind 参数

| 值        | 图表类型   |
| --------- | ---------- |
| `line`    | 折线图     |
| `bar`     | 柱状图     |
| `barh`    | 水平柱状图 |
| `hist`    | 直方图     |
| `box`     | 箱线图     |
| `area`    | 面积图     |
| `scatter` | 散点图     |
| `pie`     | 饼图       |

## 分组绘图

```python
df.groupby('category')['value'].mean().plot(kind='bar')
```

![04_groupby](https://img.yumeko.site/file/articles/viz/04_groupby.png)

## 练习

```bash
python Basic/Visualization/04_pandas_viz.py
```

# 探索性数据分析可视化

## 分布分析

```python
# 直方图 + KDE
sns.histplot(df['column'], kde=True)

# 标记均值和中位数
ax.axvline(df['column'].mean(), color='red', label='Mean')
ax.axvline(df['column'].median(), color='green', label='Median')
```

![05_distribution](https://img.yumeko.site/file/articles/viz/05_distribution.png)

## 相关性分析

```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
```

![05_correlation](https://img.yumeko.site/file/articles/viz/05_correlation.png)

## 分类变量分析

```python
# 频数统计
df['category'].value_counts().plot(kind='bar')

# 分类箱线图
sns.boxplot(x='category', y='value', data=df)
```

![05_categorical](https://img.yumeko.site/file/articles/viz/05_categorical.png)

## 练习

```bash
python Basic/Visualization/05_eda.py
```

# 数据预处理可视化

## 缺失值可视化

```python
# 缺失值热力图
sns.heatmap(df.isnull(), cbar=True, cmap='YlOrRd')

# 缺失比例
(df.isnull().mean() * 100).plot(kind='bar')
```

![06_missing](https://img.yumeko.site/file/articles/viz/06_missing.png)

## 异常值可视化

```python
# 箱线图检测
ax.boxplot(data)

# IQR 边界
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
```

![06_outlier](https://img.yumeko.site/file/articles/viz/06_outlier.png)

## 特征变换

```python
# 对数变换
np.log1p(data)

# 标准化
(data - data.mean()) / data.std()
```

![06_transform](https://img.yumeko.site/file/articles/viz/06_transform.png)

## 练习

```bash
python Basic/Visualization/06_preprocessing_viz.py
```

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

# 交互式可视化

## Plotly Express

```python
import plotly.express as px

# 散点图
fig = px.scatter(df, x='x', y='y', color='category')

# 折线图
fig = px.line(df, x='date', y='value')

# 柱状图
fig = px.bar(df, x='category', y='value')

# 3D 散点图
fig = px.scatter_3d(df, x='x', y='y', z='z')

fig.show()
```

## 保存图表

```python
fig.write_html('chart.html')
fig.write_image('chart.png')
```

## 自定义布局

```python
fig.update_layout(
    title='Title',
    xaxis_title='X',
    yaxis_title='Y',
    template='plotly_dark'
)
```

## 练习

```bash
python Basic/Visualization/09_interactive.py
```

# 专业数据可视化报告

## 专业样式

```python
plt.style.use('seaborn-v0_8-whitegrid')
```

## 多面板布局

```python
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :2])  # 跨两列
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
```

![10_multipanel](https://img.yumeko.site/file/articles/viz/10_multipanel.png)

## 导出选项

```python
plt.savefig('fig.png', dpi=300, bbox_inches='tight')
plt.savefig('fig.pdf')  # 矢量格式
plt.savefig('fig.svg')  # 矢量格式
```

## 配色方案

| 类型 | Colormap               |
| ---- | ---------------------- |
| 顺序 | viridis, plasma, magma |
| 发散 | coolwarm, RdBu         |
| 定性 | Set1, Set2, tab10      |

![10_professional](https://img.yumeko.site/file/articles/viz/10_professional.png)

## 练习

```bash
python Basic/Visualization/10_reporting.py
```
