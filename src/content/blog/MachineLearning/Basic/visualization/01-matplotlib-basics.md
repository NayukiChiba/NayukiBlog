---
title: Matplotlib 基础入门
date: 2026-01-21
category: MachineLearning/Basic/visualization
tags:
  - Python
  - Matplotlib
  - 可视化
description: Matplotlib 基础入门，学习图表结构和基本绑图
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
