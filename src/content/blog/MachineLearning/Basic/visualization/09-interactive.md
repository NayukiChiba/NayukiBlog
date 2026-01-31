---
title: 交互式可视化
date: 2026-01-23
category: MachineLearning/Basic/visualization
tags:
  - Python
  - 可视化
description: 学习 Plotly 交互式可视化
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
