---
title: Pandas 可视化
date: 2026-01-22
category: MachineLearning/Basic/visualization
tags:
  - Python
  - Pandas
  - 可视化
description: 掌握 Pandas 内置可视化功能
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
