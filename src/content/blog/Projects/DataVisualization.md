---
title: 温度数据可视化系统
date: 2026-06-26
category: 项目
tags:
  - Python
  - Matplotlib
  - Flask
  - Docker
description: 基于 Matplotlib 和 Flask 的温度数据可视化系统，支持多图表类型、集中化参数配置和交互式网页预览。
image: https://img.yumeko.site/file/blog/cover/1782480797729_DataVisualization.webp
status: published
---

# 温度数据可视化系统

::github[repo=NayukiChiba/DataVisualization]

## 1. 项目定位

`DataVisualization` 是一个温度数据可视化系统。它的重点不是只画一张图，而是把 Matplotlib 的图表参数显式配置化，
再通过 Flask 提供一个可以交互调整参数的网页应用。

项目支持六类图表：

| 图表 | 用途 |
|:--|:--|
| 折线图 | 展示温度随时间变化趋势 |
| 面积图 | 展示温度变化范围和累计视觉面积 |
| 箱线图 | 展示月、日、小时维度的统计分布 |
| 热力图 | 展示二维时间粒度下的温度分布 |
| 日历热力图 | 以日历形式展示全年温度 |
| 3D 曲面图 | 展示日期和小时两个维度上的温度曲面 |

---

## 2. 项目结构

```text
DataVisualization/
├── main.py
├── generate_charts.py
├── run_web.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config/
│   ├── settings.py
│   └── visualization_config.py
├── data/
│   └── temperature.py
└── src/
    ├── plot_utils.py
    ├── line_plot.py
    ├── area_plot.py
    ├── box_plot.py
    ├── heatmap.py
    ├── calendar_heatmap.py
    ├── surface_3d.py
    └── web_app.py
```

| 模块 | 职责 |
|:--|:--|
| `data/temperature.py` | 生成模拟温度数据 |
| `config/visualization_config.py` | 集中管理 300+ 可视化参数 |
| `src/*_plot.py` | 各类 Matplotlib 图表实现 |
| `src/web_app.py` | Flask 交互式网页 |
| `generate_charts.py` | 独立生成静态图表 |
| `run_web.py` | 启动网页应用 |
| `Dockerfile` | 容器化部署 |

---

## 3. 温度数据生成

项目使用模拟数据生成器构造温度序列。一个合理的温度模型通常由年周期、日周期、噪声和异常项构成：

$$
\boxed{
T(t, h)
= T_0
+ A_y \sin\left(\frac{2\pi t}{365}\right)
+ A_d \sin\left(\frac{2\pi h}{24}\right)
+ \epsilon
}
$$

其中：

| 符号 | 含义 |
|:--|:--|
| $T_0$ | 基准温度 |
| $A_y$ | 年周期振幅 |
| $A_d$ | 日周期振幅 |
| $t$ | 一年中的日期序号 |
| $h$ | 一天中的小时 |
| $\epsilon$ | 随机噪声 |

这样生成的数据既有季节性，也有昼夜变化，适合演示多种可视化图表。

---

## 4. 配置中心化

项目把 Matplotlib 参数集中放在 `config/visualization_config.py` 中。

参数分组包括：

| 类别 | 说明 |
|:--|:--|
| Figure | 画布尺寸、DPI、背景色 |
| Line | 线条颜色、宽度、样式、标记 |
| Axes | 坐标轴范围、标签、刻度 |
| Text | 标题、字号、字体、颜色 |
| Grid | 网格线颜色、透明度、线型 |
| Legend | 图例位置、边框、背景 |
| Output | 文件名、格式、保存路径 |

这种做法的价值是可复现：图表样式不是散落在各个绘图函数里，而是由配置文件统一控制。

---

## 5. 图表类型

### 5.1 折线图

折线图用于观察时间趋势：

$$
(t_i, T_i), \quad i=1,2,\dots,n
$$

适合回答“温度是否随时间上升或下降”。

### 5.2 箱线图

箱线图突出分位数结构：

$$
IQR = Q_3 - Q_1
$$

异常值常用范围判断：

$$
x < Q_1 - 1.5IQR
\quad \text{或} \quad
x > Q_3 + 1.5IQR
$$

### 5.3 热力图

热力图把日期和小时两个维度压到一张二维色块图里。每个格子的颜色表示对应时刻的温度，
适合展示“哪几天、哪些小时温度更高”。

### 5.4 3D 曲面图

3D 曲面图把温度看作二元函数：

$$
z = T(d, h)
$$

适合展示日期和小时对温度的联合影响。

---

## 6. 使用方式

Docker Compose 启动：

```bash
docker-compose up -d
```

本地安装依赖：

```bash
pip install -r requirements.txt
```

生成所有静态图表：

```bash
python generate_charts.py
```

启动交互式网页：

```bash
python run_web.py
```

主入口会先生成数据和图表，再启动网页：

```bash
python main.py
```

---

## 7. Web 交互

网页应用支持：

| 控件 | 用途 |
|:--|:--|
| 图表类型按钮 | 切换折线图、面积图、热力图等 |
| 滑动条 | 调整线宽、透明度、字号等数值 |
| 颜色选择器 | 调整线条、填充、背景颜色 |
| 下拉菜单 | 选择线型、颜色映射、插值方式 |
| 复选框 | 控制网格、标记点等开关 |

交互式配置适合快速比较不同图表参数对可读性的影响。

---

## 8. 总结

`DataVisualization` 的重点是“可控的可视化”：

| 设计 | 价值 |
|:--|:--|
| 显式 Matplotlib 参数 | 避免样式隐藏在默认值里 |
| 集中配置 | 图表风格可复现、可维护 |
| 多图表类型 | 覆盖趋势、分布、热力和曲面 |
| Flask 网页 | 降低调参成本 |
| Docker 部署 | 方便环境迁移 |

它适合作为 Matplotlib 深入练习项目，也可以作为数据分析报告图表模板库继续扩展。

---

> **相关文章**：
> - [[MachineLearning/Basic/visualization|数据可视化基础]]
> - [[MachineLearning/Basic/pandas|Pandas 基础]]
> - [[MachineLearning/Basic/numpy|NumPy 基础]]
