---
title: 探索性数据分析 EDA
date: 2026-01-22
category: MachineLearning/Basic/visualization
tags:
  - Python
  - Seaborn
  - 可视化
description: 学习探索性数据分析可视化方法
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
