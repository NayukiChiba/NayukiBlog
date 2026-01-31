---
title: 数据预处理可视化
date: 2026-01-22
category: MachineLearning/Basic/visualization
tags:
  - Python
  - 可视化
description: 可视化缺失值和异常值检测
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
