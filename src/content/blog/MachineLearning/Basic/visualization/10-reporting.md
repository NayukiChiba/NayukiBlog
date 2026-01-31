---
title: 报告与分享
date: 2026-01-24
category: MachineLearning/Basic/visualization
tags:
  - Python
  - 可视化
description: 掌握可视化报告生成和分享方法
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: public
---

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
