---
title: Matplotlib 图表类型
date: 2026-01-21
category: MachineLearning/Basic/visualization
tags:
  - Python
  - Matplotlib
  - 可视化
description: 掌握柱状图、散点图、饼图等常用图表类型
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
