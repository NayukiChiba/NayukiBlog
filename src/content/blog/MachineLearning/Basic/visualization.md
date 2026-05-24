---
title: 数据可视化指南
date: 2026-01-24
category: MachineLearning/Basic
tags:
  - Python
  - 可视化
  - 基础
description: Python 数据可视化全指南，系统讲解 Matplotlib、Seaborn、Pandas 及 Plotly 的核心用法。内容涵盖基础图表、探索性数据分析（EDA）、数据预处理可视化、机器学习模型决策与评估可视化，以及制作专业报告与交互式图表的最佳实践。
image: https://img.yumeko.site/file/blog/VisualizationLearning.jpg
status: published
---

# Matplotlib 基础

## 本章目标

1. 理解 Figure、Axes、Axis 三层对象结构以及创建方式
2. 掌握 `plot` 的线型、标记、颜色等高频可视化参数
3. 学会使用 `subplots` 快速构建多图布局并保存输出

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plt.subplots(nrows, ncols)` | 函数 | 创建 Figure 与 Axes 容器 |
| `ax.plot(x, y)` | 方法 | 绘制折线并配置线型、颜色、标记 |
| `ax.set_xlabel / set_ylabel / set_title` | 方法 | 设置轴标签、标题 |
| `ax.legend()` | 方法 | 管理图例展示 |
| `ax.grid(True)` | 方法 | 显示网格线 |
| `fig.savefig(path)` | 方法 | 将图表写入文件 |
| `plt.tight_layout()` | 函数 | 自动调整子图间距 |

## 1. Figure 和 Axes

### `plt.subplots` / `ax.plot`

#### 作用

`plt.subplots` 返回 `(fig, ax)`，其中 `fig` 是画布，`ax` 是绘图区域。多条曲线可在同一个 `Axes` 上叠加，配合图例便于对比。轴标签、标题、网格属于最基础的读图语义信息，应显式设置。

#### 重点方法

```python
plt.subplots(nrows=1, ncols=1, *, figsize=None)
ax.plot(*args, label=None, color=None, linestyle='-')
ax.set_xlabel(xlabel) / ax.set_ylabel(ylabel) / ax.set_title(label)
ax.legend() / ax.grid(visible=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `nrows` | `int` | 子图行数，默认为 `1` | `1` |
| `ncols` | `int` | 子图列数，默认为 `1` | `1` |
| `figsize` | `tuple[float, float]` | 画布尺寸（英寸） | `(8, 5)` |
| `x` / `y` | `array_like` | 横/纵轴数据序列 | `np.linspace(0, 10, 100)`, `np.sin(x)` |
| `label` | `str` | 图例名称 | `"sin(x)"` |
| `color` | `str` | 线条颜色 | `"red"` |
| `linestyle` | `str` | 线条样式：`"-"` / `"--"` / `":"` / `"-."` | `"--"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Basic Plot")
ax.legend()
ax.grid(True)
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/01_basic.png
```

![基础图表](https://img.yumeko.site/file/articles/ML/visualization/01_basic.png)

#### 理解重点

- 把 `Figure` 理解为"画布"，`Axes` 理解为"具体图表区域"
- 任何复杂布局都可以拆解成多个 `Axes` 的组合
- `ax.plot` 返回 `Line2D` 列表，可用于后续修改

## 2. 线条样式

### `ax.plot` 的线型参数

#### 作用

`plot` 支持通过格式字符串快速指定颜色和线型（如 `"r-"`、`"g--"`）。`linewidth` 可以显著改善可读性，建议在对比图中统一设置。线型差异是彩色和灰阶打印都可区分的重要编码方式。

#### 重点方法

```python
ax.plot(x, y, fmt, *, linewidth=1.5, label=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fmt` | `str` | 颜色与线型快捷写法：`"r-"` `"g--"` `"b:"` `"m-."` | `"r-"` |
| `linewidth` | `float` | 线宽 | `2` |
| `label` | `str` | 图例名称 | `"solid"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 50)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), "r-", linewidth=2, label="solid")
ax.plot(x, np.sin(x + 0.5), "g--", linewidth=2, label="dashed")
ax.plot(x, np.sin(x + 1.0), "b:", linewidth=2, label="dotted")
ax.plot(x, np.sin(x + 1.5), "m-.", linewidth=2, label="dashdot")
ax.legend()
ax.set_title("Line Styles")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/01_line_styles.png
常用线型: - / -- / : / -.
```

![线条样式](https://img.yumeko.site/file/articles/ML/visualization/01_line_styles.png)

#### 理解重点

- 线型应优先用于"系列区分"，颜色用于"语义强调"
- 同时设置 `label` 与 `legend` 是对比图最小闭环
- 格式字符串第一个字符为颜色，第二个字符为线型

## 3. 标记符号

### `ax.plot` 的 marker 参数

#### 作用

标记（marker）可以突出离散采样点，适合小样本展示。`markersize` 决定视觉密度，过大容易遮挡趋势。多序列情况下，图例列数可通过 `legend(ncol=...)` 控制紧凑布局。

#### 重点方法

```python
ax.plot(x, y, *, marker=None, markersize=None, label=None)
ax.legend(*, ncol=1)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `marker` | `str` | 标记符号：`"o"` `"s"` `"^"` `"D"` `"v"` `"p"` `"*"` `"x"` | `"o"` |
| `markersize` | `float` | 标记大小 | `8` |
| `ncol` | `int` | 图例列数，默认为 `1` | `4` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
markers = ["o", "s", "^", "D", "v", "p", "*", "x"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, m in enumerate(markers):
    ax.plot(x, np.sin(x) + i * 0.5, marker=m, label=f"'{m}'", markersize=8)
ax.legend(ncol=4)
ax.set_title("Marker Symbols")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/01_markers.png
```

![标记符号](https://img.yumeko.site/file/articles/ML/visualization/01_markers.png)

#### 理解重点

- 标记是离散信息编码，不应替代颜色和线型的主语义
- 数据点很多时建议降低 `alpha` 或减少 marker 使用
- 标记符号的可见性取决于 `markersize` 和数据密度

## 4. 颜色设置

### 颜色指定方式

#### 作用

Matplotlib 支持单字符、颜色名、十六进制、RGB 元组与 colormap 多种写法。在团队协作中建议固定调色板，避免每张图配色风格漂移。

#### 重点方法

```python
ax.plot(x, y, color=None)           # 单字符 / 颜色名 / hex / RGB
plt.cm.get_cmap(name)               # 获取 colormap 对象
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `color` | `str` 或 `tuple` | 颜色指定：单字符 / 名称 / `#RRGGBB` / `(r,g,b)` | `"red"` `"#FF5733"` `(0.1,0.2,0.5)` |
| `name` | `str` | colormap 名称：`"viridis"` `"coolwarm"` `"Set1"` 等 | `"viridis"` |

#### 示例代码

```python
import matplotlib.pyplot as plt

print("单字符: r, g, b, c, m, y, k, w")
print("颜色名: red, green, blue, steelblue, coral")
print("十六进制: #FF5733")
print("RGB 元组: (0.1, 0.2, 0.5)")
print(f"Colormap 示例: {plt.cm.viridis}")
```

#### 输出

```text
单字符: r, g, b, c, m, y, k, w
颜色名: red, green, blue, steelblue, coral
十六进制: #FF5733
RGB 元组: (0.1, 0.2, 0.5)
Colormap 示例: <matplotlib.colors.LinearSegmentedColormap object>
```

#### 理解重点

- 颜色不是装饰，而是编码变量和强调信息的工具
- 在深浅背景切换时，优先验证颜色对比度是否足够
- 连续数据用顺序色图（viridis），发散数据用 diverging（coolwarm），类别用定性色图（Set1）

## 5. 子图布局

### `plt.subplots` 多图网格

#### 作用

`subplots(2, 2)` 可一次性创建网格布局，适合多指标对照。`axes[i, j]` 访问单个子图，配置方式与普通 `ax` 完全一致。保存前调用 `tight_layout` 可以避免标题和坐标标签重叠。

#### 重点方法

```python
plt.subplots(nrows=1, ncols=1, *, figsize=None)
plt.tight_layout()
fig.savefig(fname, dpi=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `nrows` | `int` | 子图行数，默认为 `1` | `2` |
| `ncols` | `int` | 子图列数，默认为 `1` | `2` |
| `figsize` | `tuple[float, float]` | 画布尺寸 | `(10, 8)` |
| `fname` | `str` | 保存路径 | `"output.png"` |
| `dpi` | `int` | 分辨率，默认为 `None` | `150` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x)); axes[0, 0].set_title("sin(x)")
axes[0, 1].plot(x, np.cos(x)); axes[0, 1].set_title("cos(x)")
axes[1, 0].plot(x, np.exp(-x / 5) * np.sin(x)); axes[1, 0].set_title("Damped")
axes[1, 1].plot(x, x ** 2); axes[1, 1].set_title("x²")
plt.tight_layout()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/01_subplots.png
```

![子图布局](https://img.yumeko.site/file/articles/ML/visualization/01_subplots.png)

#### 理解重点

- 子图布局是"同一视图比较"最有效的表达方式
- 建议保持统一配色和字体，避免多图布局视觉噪音
- `tight_layout` 在标题/标签较长时尤为重要

## 常见坑

1. 忘记调用 `plt.tight_layout()` 导致标题被截断
2. 坐标轴刻度标签与数据精度不匹配
3. 中文显示为方块——需配置中文字体

## 小结

- Matplotlib 采用 Figure → Axes → Axis 三层结构——先理解对象层级再绘图
- `subplots` 是创建画布的统一入口——单图和多图都用它
- 线型、标记、颜色三要素各自承载不同信息维度——避免混用
- 子图布局 + 统一样式是专业报告图的基础


# Matplotlib 图表类型

## 本章目标

1. 掌握柱状图、散点图、直方图、饼图、箱线图的典型绘制流程
2. 学会针对不同图表类型配置关键参数以提升可读性
3. 理解统计分布与类别对比场景下的图表选型逻辑

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ax.bar(x, height)` / `ax.barh(y, width)` | 方法 | 展示类别间数值对比 |
| `ax.scatter(x, y)` | 方法 | 展示变量关系与离散分布 |
| `ax.hist(x)` | 方法 | 展示数据频率分布 |
| `ax.pie(x)` | 方法 | 展示整体组成比例 |
| `ax.boxplot(x)` | 方法 | 展示中位数、分位区间和异常值 |

## 1. 柱状图

### `ax.bar` / `ax.barh`

#### 作用

`bar` 适合"类别-数值"比较，`barh` 适合类别标签较长的场景。为柱子添加边框（`edgecolor`）可提升打印和投影场景可读性。

#### 重点方法

```python
ax.bar(x, height, *, color=None, edgecolor=None)
ax.barh(y, width, *, color=None, edgecolor=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` / `y` | `array_like` | 类别标签 | `["A", "B", "C", "D", "E"]` |
| `height` / `width` | `array_like` | 柱高/柱长度 | `[23, 45, 56, 78, 32]` |
| `color` | `str` | 填充颜色 | `"steelblue"` |
| `edgecolor` | `str` | 边框颜色 | `"black"` |

#### 示例代码

```python
import matplotlib.pyplot as plt

categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(categories, values, color="steelblue", edgecolor="black")
axes[0].set_title("Vertical Bar")
axes[1].barh(categories, values, color="coral", edgecolor="black")
axes[1].set_title("Horizontal Bar")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/02_bar.png
左图为垂直柱状图，右图为水平柱状图
```

![柱状图](https://img.yumeko.site/file/articles/ML/visualization/02_bar.png)

#### 理解重点

- 类别比较优先柱状图，趋势比较优先折线图
- 横向柱状图对长文本标签更友好
- 柱子数量超过 ~15 个时考虑改用水平排列

## 2. 散点图

### `ax.scatter`

#### 作用

散点图可同时编码位置、颜色、大小三个维度信息。`alpha` 可降低点重叠遮挡，适合高密度数据。配合 colorbar 能把颜色映射转化为可解释变量。

#### 重点方法

```python
ax.scatter(x, y, *, c=None, s=None, alpha=None, cmap=None)
plt.colorbar(mappable, *, ax=None, label=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` / `y` | `array_like` | 横/纵轴数据 | `np.random.randn(100)` |
| `c` | `array_like` | 颜色映射值 | `np.random.rand(100)` |
| `s` | `float` 或 `array_like` | 点大小 | `200` |
| `alpha` | `float` | 透明度，`0`~`1` | `0.6` |
| `cmap` | `str` | colormap 名称 | `"viridis"` |
| `label` | `str` | colorbar 标题 | `"Color Value"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
colors = np.random.rand(100)
sizes = np.abs(np.random.randn(100)) * 200

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")
plt.colorbar(sc, ax=ax, label="Color Value")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/02_scatter.png
点的颜色和大小分别编码额外变量
```

![散点图](https://img.yumeko.site/file/articles/ML/visualization/02_scatter.png)

#### 理解重点

- 当点重叠严重时，`alpha` 与采样策略要一起调整
- 不同变量的编码优先级建议固定，避免读图歧义
- `c` 和 `cmap` 配合 colorbar 是最常见的三元关系可视化模式

## 3. 直方图

### `ax.hist`

#### 作用

直方图用于观察分布形态、偏度和离散程度。`bins` 影响分辨率，过小会丢失细节，过大则噪声明显。叠加均值线可快速定位中心位置。

#### 重点方法

```python
ax.hist(x, *, bins=None, edgecolor=None, alpha=None)
ax.axvline(x, *, color=None, linestyle=None, label=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 输入样本 | `np.random.randn(1000)` |
| `bins` | `int` | 直方图箱数，默认为 `None`（自动选择） | `30` |
| `edgecolor` | `str` | 箱体边框颜色 | `"black"` |
| `alpha` | `float` | 透明度 | `0.7` |
| `x`（axvline） | `float` | 竖线 x 坐标 | `data.mean()` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
ax.axvline(data.mean(), color="red", linestyle="--",
           label=f"Mean: {data.mean():.2f}")
ax.axvline(np.median(data), color="green", linestyle="--",
           label=f"Median: {np.median(data):.2f}")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/02_histogram.png
正态近似分布并标注均值和中位数位置
```

![直方图](https://img.yumeko.site/file/articles/ML/visualization/02_histogram.png)

#### 理解重点

- 直方图不是概率密度，除非额外进行归一化（`density=True`）
- 结合均值、中位数线可更好判断偏态与异常值影响
- `bins` 的经验公式：`int(np.sqrt(len(data)))` 可作为起点

## 4. 饼图

### `ax.pie`

#### 作用

饼图适合少类别的占比表达，不适合精确比较接近比例。`explode` 可强调关键类别。`autopct` 能直接显示百分比，提升报告阅读效率。

#### 重点方法

```python
ax.pie(x, *, labels=None, explode=None, autopct=None, startangle=None,
       colors=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 各类别占比 | `[35, 30, 20, 15]` |
| `labels` | `list[str]` | 类别标签 | `["Product A", "Product B", "Product C", "Product D"]` |
| `explode` | `tuple[float]` | 扇区偏移距离 | `(0.05, 0, 0, 0)` |
| `autopct` | `str` 或 `callable` | 百分比格式字符串 | `"%1.1f%%"` |
| `startangle` | `float` | 起始角度，0=右侧3点方向 | `90` |
| `colors` | `list[str]` | 扇区颜色列表 | `["gold", "silver"]` |

#### 示例代码

```python
import matplotlib.pyplot as plt

labels = ["Product A", "Product B", "Product C", "Product D"]
sizes = [35, 30, 20, 15]
explode = (0.05, 0, 0, 0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, explode=explode, autopct="%1.1f%%",
       startangle=90)
ax.set_title("Market Share")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/02_pie.png
Product A 扇区被突出显示
```

![饼图](https://img.yumeko.site/file/articles/ML/visualization/02_pie.png)

#### 理解重点

- 类别超过 5~6 个时建议改用条形图
- 比例对比不明显时应避免仅依赖角度感知
- `startangle=90` 让最大扇区从顶部开始——视觉效果最佳

## 5. 箱线图

### `ax.boxplot`

#### 作用

箱线图直接展示中位数、四分位区间与异常值。多组箱线图适合比较组间波动差异。`patch_artist=True` 后可对箱体填色，增强分组辨识度。

#### 重点方法

```python
ax.boxplot(x, *, patch_artist=False, labels=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` 或 `list[array_like]` | 单组或多组样本 | `[np.random.normal(0, std, 100) for std in range(1, 5)]` |
| `patch_artist` | `bool` | 是否允许箱体填充颜色，默认为 `False` | `True` |
| `labels` | `list[str]` | 组标签 | `["σ=1", "σ=2", "σ=3", "σ=4"]` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(data, patch_artist=True, labels=["σ=1", "σ=2", "σ=3", "σ=4"])
colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_xlabel("Group")
ax.set_ylabel("Value")
ax.set_title("Boxplot: Different Standard Deviations")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/02_boxplot.png
四组不同标准差分布的中位数和离散度对比
```

![箱线图](https://img.yumeko.site/file/articles/ML/visualization/02_boxplot.png)

#### 理解重点

- 箱线图适合稳健比较，不依赖分布假设
- 与直方图结合使用可同时获得整体形态与统计摘要
- 箱体边为 Q1/Q3，中间线为中位数，须线为 1.5*IQR 范围

## 常见坑

1. 饼图类别过多导致扇区不可辨识
2. 散点图数据量大时不做透明度导致 overplotting
3. 直方图 bins 选择不当导致分布形态误判

## 小结

- 柱状图看比较，散点图看关系，直方图看分布——先确定问题的图类型
- 饼图只适合 ≤5 类别的占比展示
- 箱线图是统计摘要最可靠的方式——搭配直方图使用效果更好

# Seaborn 统计可视化

## 本章目标

1. 掌握 Seaborn 在分类、分布、回归和相关性分析中的高效绘图方式
2. 理解 Seaborn 与 Matplotlib 的关系，以及 `ax` 级 API 的组合方式
3. 学会使用内置数据集快速搭建分析原型图

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `sns.set_theme(style)` | 函数 | 统一绘图主题样式 |
| `sns.barplot(data, x, y)` | 函数 | 分类均值对比柱状图 |
| `sns.boxplot(data, x, y)` | 函数 | 分类分布箱线图 |
| `sns.histplot(data)` | 函数 | 直方图与 KDE 叠加 |
| `sns.kdeplot(data)` | 函数 | 核密度估计曲线 |
| `sns.regplot(data, x, y)` | 函数 | 回归散点图与拟合线 |
| `sns.heatmap(data)` | 函数 | 矩阵与相关性热力图 |
| `sns.pairplot(data)` | 函数 | 多变量成对关系探索 |

## 1. 分类图

### `sns.barplot` / `sns.boxplot`

#### 作用

`barplot` 更强调均值等聚合统计，`boxplot` 更强调分布与异常值。`hue` 维度可在同一类别下继续拆分比较。统一主题样式后，多图报告视觉会更一致。

#### 重点方法

```python
sns.barplot(data=None, *, x=None, y=None, hue=None, ax=None)
sns.boxplot(data=None, *, x=None, y=None, hue=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | 输入数据 | `sns.load_dataset("tips")` |
| `x` | `str` | 分类轴字段 | `"day"` |
| `y` | `str` | 数值轴字段 | `"total_bill"` |
| `hue` | `str` | 组内分组变量 | `"sex"` |
| `ax` | `Axes` | 目标坐标轴，默认为当前 | `axes[0]` |

#### 示例代码

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[0])
axes[0].set_title("Bar Plot: Mean Total Bill by Day")
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[1])
axes[1].set_title("Box Plot: Distribution by Day")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/03_catplot.png
左图展示各天平均账单对比，右图展示分布与离散程度
```

![分类图](https://img.yumeko.site/file/articles/ML/visualization/03_catplot.png)

#### 理解重点

- 均值对比和分布对比通常应配对展示，避免单一视角误读
- `hue` 分类过多时建议控制图例数量
- Seaborn 的 `hue` 参数自动处理色板——比手动 Matplotlib 更方便

## 2. 分布图

### `sns.histplot` / `sns.kdeplot`

#### 作用

`histplot` 强调频率分布，`kdeplot` 强调平滑密度曲线。样本量较小时，KDE 形状可能不稳定，需要谨慎解释。分布图是异常值检查和特征变换决策的前置步骤。

#### 重点方法

```python
sns.histplot(data=None, *, kde=False, bins='auto', ax=None)
sns.kdeplot(data=None, *, fill=False, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like` 或 `DataFrame` | 输入样本 | `np.random.normal(0, 1, 1000)` |
| `kde` | `bool` | histplot：是否叠加 KDE 曲线，默认为 `False` | `True` |
| `bins` | `int` 或 `str` | histplot：分箱数，`"auto"` 自动，默认为 `"auto"` | `30` |
| `fill` | `bool` | kdeplot：是否填充曲线下方，默认为 `False` | `True` |
| `ax` | `Axes` | 目标坐标轴 | `axes[0]` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data, kde=True, ax=axes[0])
axes[0].set_title("Histogram + KDE")
sns.kdeplot(data, fill=True, ax=axes[1])
axes[1].set_title("KDE Only")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/03_distplot.png
左图为直方图+KDE，右图为独立 KDE 曲线
```

![分布图](https://img.yumeko.site/file/articles/ML/visualization/03_distplot.png)

#### 理解重点

- `bins` 与平滑程度共同影响"分布形态"判断
- 使用 KDE 时应与原始频数图交叉验证
- KDE 的带宽（bandwidth）影响曲线平滑度——过小噪声大，过大丢失细节

## 3. 回归图

### `sns.regplot`

#### 作用

`regplot` 可同时展示散点与拟合趋势线。在探索阶段可以快速判断线性关系方向与强弱。对高噪声数据，拟合线应作为趋势参考而非因果结论。

#### 重点方法

```python
sns.regplot(data=None, *, x=None, y=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | 输入数据 | `sns.load_dataset("tips")` |
| `x` | `str` | 自变量字段 | `"total_bill"` |
| `y` | `str` | 因变量字段 | `"tip"` |
| `ax` | `Axes` | 目标坐标轴 | `ax` |

#### 示例代码

```python
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(data=tips, x="total_bill", y="tip", ax=ax)
ax.set_title("Regression: Total Bill vs Tip")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/03_regplot.png
消费总额与小费呈正相关趋势
```

![回归图](https://img.yumeko.site/file/articles/ML/visualization/03_regplot.png)

#### 理解重点

- 回归线是趋势摘要，不代表模型最终效果
- 观察残差和分组差异可进一步验证关系稳定性
- `regplot` 默认画散点 + 回归线 + 置信区间——适合单变量探索

## 4. 热力图

### `sns.heatmap`

#### 作用

热力图适合显示矩阵强度，常用于相关系数和注意力矩阵。`annot=True` 可直接写入数值，适合教学与报告。`center` 与 `cmap` 联动决定颜色语义，应统一标准。

#### 重点方法

```python
sns.heatmap(data, *, annot=None, fmt='.2g', cmap=None, center=None,
            vmin=None, vmax=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like` | 输入二维矩阵 | `np.random.rand(10, 10)` |
| `annot` | `bool` | 是否标注格子数值，默认为 `None` | `True` |
| `fmt` | `str` | 数值显示格式，默认为 `".2g"` | `".2f"` |
| `cmap` | `str` | 颜色映射 | `"YlOrRd"` |
| `center` | `float` | 颜色中心值——发散色图用 | `0` |
| `vmin` / `vmax` | `float` | 颜色范围上下限 | `0`, `1` |
| `ax` | `Axes` | 目标坐标轴 | `ax` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data = np.random.rand(10, 10)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
ax.set_title("Heatmap")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/03_heatmap.png
10x10 数值矩阵被映射为颜色强度
```

![热力图](https://img.yumeko.site/file/articles/ML/visualization/03_heatmap.png)

#### 理解重点

- 颜色深浅应与数值大小保持单调关系
- 强调比较时建议固定统一的 `vmin` / `vmax`
- 相关性矩阵用 `center=0` 的发散色图（如 `coolwarm`）更合适

## 5. 配对图

### `sns.pairplot`

#### 作用

`pairplot` 可以一次查看多变量两两关系与单变量分布。`hue` 可以在同一图中区分类别，有助于发现可分性。对高维数据应先选特征子集，避免图过于拥挤。

#### 重点方法

```python
sns.pairplot(data, *, hue=None, vars=None, height=2.5, diag_kind='auto')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | 输入数据 | `sns.load_dataset("iris")` |
| `hue` | `str` | 分类上色字段 | `"species"` |
| `vars` | `list[str]` | 指定要展示的列子集 | `["sepal_length", "petal_length"]` |
| `height` | `float` | 每个子图边长（英寸），默认为 `2.5` | `2` |
| `diag_kind` | `str` | 对角图类型：`"auto"` / `"hist"` / `"kde"` | `"auto"` |

#### 示例代码

```python
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
g = sns.pairplot(iris, hue="species", height=2)
g.fig.suptitle("Pair Plot of Iris", y=1.02)

# pairplot 返回 PairGrid，需特殊处理保存
import matplotlib.pyplot as plt
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/03_pairplot.png
多特征成对散点图与对角分布图
```

![配对图](https://img.yumeko.site/file/articles/ML/visualization/03_pairplot.png)

#### 理解重点

- `pairplot` 常用于建模前特征筛查与类别可分性判断
- 高维场景建议先做降维或特征筛选后再绘制
- 对角线上为单变量分布（直方图或 KDE），非对角为两两散点图

## 常见坑

1. 未设置 `sns.set_theme` 导致默认样式不一致
2. `pairplot` 在高维数据上生成过多子图——先用 `vars` 筛选
3. 热力图 `fmt` 与数据精度不匹配导致标注重叠

## 小结

- Seaborn 是 Matplotlib 的高级封装——先 seaborn 快速出图，再 matplotlib 精调
- 分类图用 `barplot`+`boxplot` 配对，分布图用 `histplot`+`kdeplot` 配对
- `pairplot` 是特征探索利器——能在一张图中看到所有两两关系
- 热力图最适合展示矩阵结构——尤其是相关矩阵和混淆矩阵

# Pandas 内置可视化

## 本章目标

1. 掌握 `DataFrame.plot` 与 `Series.plot` 的常见图形类型与参数
2. 理解 Pandas 绘图与 Matplotlib `Axes` 之间的协作关系
3. 学会通过分组聚合结果快速绘制业务对比图

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `DataFrame.plot(*, kind, ax)` | 方法 | 对多列数据快速绘图 |
| `Series.plot(*, kind, ax)` | 方法 | 对单变量序列快速绘图 |
| `DataFrame.groupby(by)` | 方法 | 分组聚合后绘图 |
| `plt.subplots(nrows, ncols)` | 函数 | 组织多图布局 |

## 1. DataFrame.plot()

### `DataFrame.plot`

#### 作用

Pandas 绘图默认基于 Matplotlib，适合快速原型分析。同一个 DataFrame 可用 `kind` 参数切换线图、面积图、箱线图等。复杂布局建议先用 `plt.subplots` 创建 `Axes`，再把图绑定到指定子图。

#### 重点方法

```python
DataFrame.plot(*, kind='line', ax=None, title=None, alpha=None, figsize=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kind` | `str` | 图类型：`"line"` / `"bar"` / `"barh"` / `"hist"` / `"box"` / `"kde"` / `"density"` / `"area"` / `"pie"` / `"scatter"`，默认为 `"line"` | `"box"` |
| `ax` | `Axes` 或 `None` | 目标坐标轴，`None` 创建新图 | `axes[0, 0]` |
| `title` | `str` | 图标题 | `"Line Plot"` |
| `alpha` | `float` | 透明度 | `0.7` |
| `figsize` | `tuple[float, float]` | 画布尺寸 | `(10, 6)` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=30, freq="D")
df = pd.DataFrame({
    "A": np.cumsum(np.random.randn(30)),
    "B": np.cumsum(np.random.randn(30)),
    "C": np.cumsum(np.random.randn(30)),
}, index=dates)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
df.plot(ax=axes[0, 0], title="Line Plot")
df.plot(kind="area", alpha=0.5, ax=axes[0, 1], title="Area Plot")
df.plot(kind="bar", ax=axes[1, 0], title="Bar Plot")
df.plot(kind="box", ax=axes[1, 1], title="Box Plot")
plt.tight_layout()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/04_df_plot.png
线图、面积图、条形图、箱线图四宫格对比
```

![DataFrame 绘图](https://img.yumeko.site/file/articles/ML/visualization/04_df_plot.png)

#### 理解重点

- `DataFrame.plot` 适合快速探索，不必每次手写 Matplotlib 底层语句
- 当图形语义复杂时，可以混合使用 Pandas 与 Matplotlib API
- 时间索引的 DataFrame 用 `kind='line'` 最直观

## 2. Series.plot()

### `Series.plot`

#### 作用

`Series.plot` 是单变量分析最便捷入口。通过 `kind='hist'` 可快速切换到分布视角。同一序列可并行展示趋势图与分布图，互相校验。

#### 重点方法

```python
Series.plot(*, kind='line', ax=None, title=None, bins=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kind` | `str` | 图类型：`"line"` / `"hist"` / `"bar"` / `"kde"` 等，默认为 `"line"` | `"hist"` |
| `ax` | `Axes` 或 `None` | 目标坐标轴 | `axes[0]` |
| `title` | `str` | 图标题 | `"Line Plot"` |
| `bins` | `int` | 直方图分箱数（kind='hist' 时） | `20` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
s = pd.Series(np.random.randn(100).cumsum())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
s.plot(ax=axes[0], title="Line Plot")
s.plot(kind="hist", bins=20, ax=axes[1], title="Histogram")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/04_series_plot.png
左图展示累计走势，右图展示取值分布
```

![Series 绘图](https://img.yumeko.site/file/articles/ML/visualization/04_series_plot.png)

#### 理解重点

- 趋势图回答"怎么变化"，直方图回答"分布在哪里"
- 单变量分析阶段建议两个视角同时保留
- `Series.plot` 返回 `Axes` 对象——可继续用 Matplotlib API 修改

## 3. GroupBy 绘图

### `groupby` + `plot`

#### 作用

分组聚合是业务分析中最常见的数据预处理步骤。先 `groupby` 再 `mean` 可压缩噪声，强调组间差异。聚合结果是 Series，可直接使用 `plot(kind='bar')` 绘制。

#### 重点方法

```python
DataFrame.groupby(by)
SeriesGroupBy.mean()
Series.plot(*, kind='bar', ax=None, color=None)
```

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "Category": np.repeat(["A", "B", "C"], 20),
    "Value": np.random.randn(60),
})

fig, ax = plt.subplots(figsize=(8, 6))
(df.groupby("Category")["Value"].mean()
   .plot(kind="bar", ax=ax, color=["red", "green", "blue"]))
ax.set_xlabel("Category")
ax.set_ylabel("Mean Value")
ax.set_title("Group Mean Comparison")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/04_groupby.png
A/B/C 三个类别的均值对比柱状图
```

![分组绘图](https://img.yumeko.site/file/articles/ML/visualization/04_groupby.png)

#### 理解重点

- 先聚合后绘图能显著降低噪音干扰
- 分组统计应同时配合样本量信息，避免均值误导
- `groupby` 后可链式调用 `.mean().plot()` ——代码紧凑但需注意可读性

## 常见坑

1. Pandas 绘图 x 轴标签自动旋转可能与数据不匹配
2. `kind='box'` 只适合单列或多列的整体分布——不区分分组
3. 时间序列绘图前未设置 datetime index 导致 x 轴刻度混乱

## 小结

- Pandas `plot()` 是探索阶段最快的出图方式——比 Matplotlib 少写大量样板代码
- 先 Pandas 快速出图 → 再用 Matplotlib API 精调细节
- DataFrame 多列绘图天然适合比较型图表
- GroupBy + plot 是业务分析的标准闭环——分而治之，一目了然

# EDA 探索性数据分析

## 本章目标

1. 掌握连续变量分布、相关关系和分类变量分析的可视化套路
2. 学会在 EDA 中组合 Seaborn 与 Pandas API 完成快速验证
3. 建立"先分布、再相关、后分组"的分析顺序意识

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `sns.histplot(data)` | 函数 | 查看连续变量频率分布 |
| `ax.axvline(x)` | 方法 | 在分布图中标注统计线 |
| `DataFrame.corr()` | 方法 | 计算变量相关系数矩阵 |
| `sns.heatmap(data)` | 函数 | 可视化相关矩阵 |
| `Series.value_counts()` | 方法 | 统计类别频次 |
| `sns.boxplot(data, x, y)` | 函数 | 比较分类变量下的数值分布 |

## 1. 分布分析

### `sns.histplot` + 统计线标注

#### 作用

分布图用于观察偏态、离群和集中趋势。同时标注均值和中位数有助于识别偏态分布——两者差距大通常意味着偏态或异常值影响。EDA 第一张图建议优先看分布，而不是直接建模。

#### 重点方法

```python
sns.histplot(data=None, *, kde=False, ax=None)
ax.axvline(x, *, color=None, linestyle='--', label=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like` | 某列的数值数据 | `df["age"]` |
| `kde` | `bool` | 是否叠加核密度曲线，默认为 `False` | `True` |
| `ax` | `Axes` | 目标坐标轴 | `axes[0]` |
| `x` | `float` | 竖线 x 坐标 | `df["age"].mean()` |
| `color` | `str` | 竖线颜色 | `"red"` |
| `linestyle` | `str` | 竖线样式：`"--"` / `":"` / `"-."` | `"--"` |
| `label` | `str` | 图例名称 | `"Mean"` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.normal(35, 10, 200).astype(int),
    "income": np.random.exponential(50000, 200),
    "score": np.random.beta(2, 5, 200) * 100,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, df.columns):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
    ax.axvline(df[col].median(), color="green", linestyle="--", label="Median")
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/05_distribution.png
age、income、score 三个变量的分布与均值/中位数标记
```

![分布分析](https://img.yumeko.site/file/articles/ML/visualization/05_distribution.png)

#### 理解重点

- 均值和中位数差距较大通常意味着偏态或异常值影响
- 先看分布形态（偏态、峰度、多峰），再考虑建模策略
- 偏态变量可能需要幂变换后才适合线性模型

## 2. 相关性分析

### `DataFrame.corr` + `sns.heatmap`

#### 作用

相关矩阵可快速定位强相关与弱相关变量。热力图适合表达相关强度与方向（正负相关）。相关不等于因果，仍需结合业务逻辑验证。

#### 重点方法

```python
DataFrame.corr(method='pearson', numeric_only=False)   # → DataFrame
sns.heatmap(data, *, annot=None, cmap=None, center=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `method` | `str` | 相关系数：`"pearson"` / `"spearman"` / `"kendall"`，默认为 `"pearson"` | `"pearson"` |
| `annot` | `bool` | 是否显示数值标签 | `True` |
| `cmap` | `str` | 颜色映射——建议发散色图 | `"coolwarm"` |
| `center` | `float` | 颜色中心值，相关矩阵用 `0` | `0` |
| `ax` | `Axes` | 目标坐标轴 | `ax` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 100
x = np.random.randn(n)
df = pd.DataFrame({
    "x": x,
    "y_strong": x + np.random.randn(n) * 0.3,
    "y_weak": x + np.random.randn(n) * 2,
    "y_none": np.random.randn(n),
})

corr = df.corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Matrix")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/05_correlation.png
x 与 y_strong 相关性最高（~0.96），x 与 y_none 接近无关
```

![相关性分析](https://img.yumeko.site/file/articles/ML/visualization/05_correlation.png)

#### 理解重点

- 强相关特征在建模前应考虑共线性处理策略
- 分析相关性时要同步关注样本规模与异常值敏感性
- `spearman` 适合非线性单调关系，`pearson` 只捕捉线性关系

## 3. 分类变量分析

### `value_counts` + `sns.boxplot`

#### 作用

类别频数图回答"每类有多少"，箱线图回答"每类分布如何"。两者组合可以兼顾规模与质量两个维度。分类变量分析是异常组识别和分层建模的重要入口。

#### 重点方法

```python
Series.value_counts(normalize=False, dropna=True)    # → Series
sns.boxplot(data=None, *, x=None, y=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `normalize` | `bool` | value_counts：是否返回比例，默认为 `False` | `False` |
| `dropna` | `bool` | value_counts：是否忽略缺失值，默认为 `True` | `True` |
| `data` | `DataFrame` | boxplot：输入数据 | `df` |
| `x` | `str` | 分类字段 | `"category"` |
| `y` | `str` | 数值字段 | `"value"` |
| `ax` | `Axes` | 目标坐标轴 | `axes[1]` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame({
    "category": np.random.choice(["A", "B", "C", "D"], 200),
    "value": np.random.randn(200),
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["category"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Category Frequencies")
sns.boxplot(data=df, x="category", y="value", ax=axes[1])
axes[1].set_title("Value Distribution by Category")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/05_categorical.png
左图展示类别频数，右图展示各类别数值分布与离群点
```

![分类变量分析](https://img.yumeko.site/file/articles/ML/visualization/05_categorical.png)

#### 理解重点

- 类别不平衡会直接影响模型评估，需在 EDA 阶段尽早识别
- 同一类别中离散度明显更大时，建议追查数据来源和采样口径
- 频数图 + 箱线图是分类变量的黄金组合——兼顾样本量与分布

## 常见坑

1. EDA 跳过分布分析直接建模——错过异常值和偏态等关键信号
2. 只看相关性数值不看散点图——可能被 Anscombe's quartet 类数据误导
3. 分类变量分析只关注频数不看分布——忽略组间质量差异

## 小结

- EDA 推荐"先分布、再相关、后分组"的标准分析顺序
- 分布图 + 统计线（均值/中位数）是判断偏态最快的方法
- 相关矩阵 + 热力图是定位多变量关系的标准工具
- 分类变量的频数 + 箱线图组合能兼顾规模和分布的完整视图

# 预处理可视化

## 本章目标

1. 学会可视化缺失值分布、比例与列间差异
2. 掌握异常值识别中箱线图与 IQR 边界的组合表达
3. 对比常见数值变换对分布形态的影响并形成直觉

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.isnull()` | 方法 | 生成缺失值布尔矩阵 |
| `sns.heatmap(data)` | 函数 | 可视化缺失值模式 |
| `ax.boxplot(x)` | 方法 | 发现异常值和四分位范围 |
| `np.percentile(a, q)` | 函数 | 计算 IQR 阈值 |
| `np.log1p(x)` / `np.sqrt(x)` | 函数 | 变换偏态分布 |

## 1. 缺失值可视化

### `isnull` + `sns.heatmap`

#### 作用

缺失值热力图可以快速定位"哪几列、哪些样本段"缺失集中。缺失比例柱状图可用于排序优先级，决定填补或删除策略。缺失分析应在建模前完成，避免隐式数据泄露和偏差。

#### 重点方法

```python
DataFrame.isnull()           # → DataFrame[bool]
sns.heatmap(data, *, cbar=True, cmap=None, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | sns.heatmap 输入——通常是 `df.isnull()` | `df.isnull()` |
| `cbar` | `bool` | 是否显示颜色条，默认为 `True` | `True` |
| `cmap` | `str` | 颜色映射 | `"YlOrRd"` |
| `ax` | `Axes` | 目标坐标轴 | `axes[0]` |

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])
for col in df.columns:
    mask = np.random.rand(len(df)) < 0.1
    df.loc[mask, col] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=True, ax=axes[0], cmap="YlOrRd")
axes[0].set_title("Missing Value Pattern")
(df.isnull().mean() * 100).plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Missing Percentage per Column")
axes[1].set_ylabel("%")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/06_missing.png
左图展示缺失位置，右图展示各列缺失百分比
```

![缺失值可视化](https://img.yumeko.site/file/articles/ML/visualization/06_missing.png)

#### 理解重点

- 缺失模式随机与否会决定后续填补方法选择
- 某一列缺失比例过高时应优先评估业务可用性
- 热力图能看到"哪些样本同时缺多个特征"——这是比例图看不到的

## 2. 异常值可视化

### `ax.boxplot` + IQR 边界

#### 作用

箱线图是异常值检测最常用的统计图形。IQR 阈值线可直观标注"正常区间"边界。异常值处理前建议先可视化再决策，避免误删有效样本。

#### 重点方法

```python
ax.boxplot(x, *, patch_artist=False)
np.percentile(a, q)       # 计算分位数，如 [25, 75]
ax.axvline(x, *, color=None, linestyle='--')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | boxplot 输入 | `data` |
| `patch_artist` | `bool` | 是否启用箱体填色，默认为 `False` | `False` |
| `a` | `array_like` | percentile 输入 | `data` |
| `q` | `array_like` | 目标分位点 | `[25, 75]` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.randn(100)
data = np.append(data, [5, -5, 6, -6])

q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(data)
axes[0].set_title("Box Plot with IQR")
axes[1].hist(data, bins=20, edgecolor="black", alpha=0.7)
axes[1].axvline(lower, color="red", linestyle="--", label=f"Lower={lower:.2f}")
axes[1].axvline(upper, color="red", linestyle="--", label=f"Upper={upper:.2f}")
axes[1].legend()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/06_outlier.png
箱线图显示离群点，直方图标出 IQR 上下界
```

![异常值可视化](https://img.yumeko.site/file/articles/ML/visualization/06_outlier.png)

#### 理解重点

- IQR 规则稳健但并非适用于所有分布——指数分布可能标记过多"异常"
- 异常值处理需结合业务含义，不宜机械截断
- 箱线图须线外的点未必是错误——它们是需要独立分析的样本

## 3. 特征变换可视化

### 变换前后分布对比

#### 作用

同一变量在不同变换下的分布形态可显著变化。对数变换适合右偏分布，平方根变换更温和，标准化适合尺度统一。变换选择应以模型需求与解释性目标共同决定。

#### 重点方法

```python
np.log1p(x)       # log(1 + x)，适合含 0 的正值数据
np.sqrt(x)        # 平方根变换
(x - x.mean()) / x.std()   # Z-score 标准化
```

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.exponential(5, 1000)
transforms = {
    "Original": data,
    "Log (log1p)": np.log1p(data),
    "Sqrt": np.sqrt(data),
    "Standardized": (data - data.mean()) / data.std(),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, arr) in zip(axes.flat, transforms.items()):
    ax.hist(arr, bins=30, edgecolor="black", alpha=0.7)
    ax.set_title(name)
plt.tight_layout()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/06_transform.png
四宫格对比原始偏态分布与三种变换后的分布形态
```

![特征变换可视化](https://img.yumeko.site/file/articles/ML/visualization/06_transform.png)

#### 理解重点

- 变换不是为了"好看"，而是为了改善建模稳定性
- 变换后应重新评估可解释性与业务阈值含义
- 对数变换不能处理 0 或负值——此时用 `yeo-johnson` 变换替代

## 常见坑

1. 缺失值热力图只看颜色不看比例——掩盖真实缺失严重程度
2. 对非正态分布盲目用 IQR 标记异常值——导致正常样本被误删
3. 变换后不检查分布就建模——改善可能有限甚至反向

## 小结

- 缺失值分析先看整体比例，再看行/列模式——热力图 + 柱状图组合
- 箱线图 + IQR 边界是异常值检测的稳健起点——但需结合业务
- 对数变换是右偏数据的首选——若效果不足再尝试 Box-Cox
- 所有预处理决策都应可视化验证后再进入建模阶段

# 模型决策可视化

## 本章目标

1. 理解二维分类任务中决策边界的可视化构建方法
2. 掌握决策树结构图与特征重要性图的解释方式
3. 学会用可视化连接"模型行为"与"特征贡献"

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `sklearn.tree.DecisionTreeClassifier()` | 构造器 | 训练树模型并生成边界/结构 |
| `ax.contourf(X, Y, Z)` | 方法 | 绘制二维决策区域填充 |
| `sklearn.tree.plot_tree(tree)` | 函数 | 绘制决策树节点结构 |
| `RandomForestClassifier.feature_importances_` | 属性 | 获取特征重要性得分 |

## 1. 决策边界

### `ax.contourf` + 网格预测

#### 作用

决策边界图把分类器在特征空间中的划分区域可视化。网格预测是边界绘制核心：先生成网格，再对每个网格点预测。真实样本散点叠加在边界图上，可直观看到误分类风险区域。

#### 重点方法

```python
np.meshgrid(x, y)        # 生成二维网格
clf.predict(grid)        # 对网格点预测类别
ax.contourf(X, Y, Z, *, alpha=None, cmap=None)
ax.scatter(x, y, *, c=None, cmap=None, edgecolors=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X, Y` | `ndarray` | meshgrid 生成的网格坐标矩阵 | `xx`, `yy` |
| `Z` | `ndarray` | 每个网格点预测类别（需 reshape） | `Z.reshape(xx.shape)` |
| `alpha` | `float` | 填充透明度 | `0.3` |
| `cmap` | `str` | 区域与散点色板 | `"RdYlBu"` |
| `edgecolors` | `str` | 散点边框颜色 | `"black"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=42)
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="black")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title(f"Decision Boundary (max_depth={clf.max_depth})")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/07_boundary.png
背景为模型分类区域，散点为真实样本标签
```

![决策边界](https://img.yumeko.site/file/articles/ML/visualization/07_boundary.png)

#### 理解重点

- 边界越曲折通常表示模型复杂度越高，过拟合风险也更高
- 该图仅适用于低维特征——真实高维问题需结合降维或局部解释
- `contourf` 的网格密度决定边界的视觉精度

## 2. 决策树可视化

### `sklearn.tree.plot_tree`

#### 作用

`plot_tree` 能直观展示每个节点的分裂规则和类别分布。`filled=True` 会用颜色突出节点主要类别，方便快速解读。树图适合教学与解释，但复杂树需限制深度保证可读性。

#### 重点方法

```python
sklearn.tree.plot_tree(decision_tree, *, max_depth=None, feature_names=None,
                       class_names=None, filled=False, rounded=False, ax=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `decision_tree` | `DecisionTreeClassifier` | 已训练树模型 | `clf` |
| `max_depth` | `int` 或 `None` | 显示的最大深度 | `3` |
| `feature_names` | `list[str]` | 特征名称 | `["F1", "F2", "F3", "F4"]` |
| `class_names` | `list[str]` | 类别名称 | `["Class 0", "Class 1"]` |
| `filled` | `bool` | 节点背景按类别着色，默认为 `False` | `True` |
| `rounded` | `bool` | 节点边框圆角，默认为 `False` | `True` |
| `ax` | `Axes` | 目标坐标轴 | `ax` |

#### 示例代码

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree

X, y = make_classification(n_samples=100, n_features=4, n_redundant=0,
                           random_state=42)
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(clf, ax=ax, filled=True, rounded=True,
          feature_names=["F1", "F2", "F3", "F4"],
          class_names=["Class 0", "Class 1"])
ax.set_title("Decision Tree Structure")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/07_tree.png
每个节点展示分裂条件、样本数和类别分布
```

![决策树可视化](https://img.yumeko.site/file/articles/ML/visualization/07_tree.png)

#### 理解重点

- 树模型的可解释性优势来自节点规则的可读表达
- 若节点过多，可通过调小 `max_depth` 或增大 `min_samples_leaf` 简化
- `gini` 值越小的节点纯度越高——颜色越深

## 3. 特征重要性

### `feature_importances_` + 水平柱状图

#### 作用

集成树模型可输出每个特征对整体预测的相对贡献。重要性排序有助于特征筛选与业务解释。重要性不代表因果关系，需要与领域知识结合。

#### 重点方法

```python
RandomForestClassifier.feature_importances_     # → ndarray，和为 1
ax.barh(y, width, *, color=None)
```

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=200, n_features=10, n_redundant=3,
                           n_informative=5, random_state=42)
featureNames = [f"Feature_{i}" for i in range(10)]

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(importances)), importances[indices], color="steelblue")
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([featureNames[i] for i in indices])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance (RandomForest)")
ax.invert_yaxis()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/07_importance.png
特征按重要性从高到低排序展示，顶部为最相关特征
```

![特征重要性](https://img.yumeko.site/file/articles/ML/visualization/07_importance.png)

#### 理解重点

- 重要性图可用于快速筛选，但不应替代交叉验证评估
- 不同模型的"重要性定义"不同——跨模型比较需谨慎
- 重要性之和为 1，既可用于排序也可用于设定累积阈值

## 常见坑

1. 决策边界图在高维数据上毫无意义——只能在 ≤3 维使用
2. `plot_tree` 对深树输出不可读——务必限制 max_depth
3. 特征重要性只反映"对模型预测的贡献"，不是因果关系

## 小结

- 决策边界图是理解模型行为的直观工具——仅限低维场景
- `plot_tree` 让树模型的白盒优势可视化——配以节点颜色和规则文本
- 特征重要性排序是特征筛选的快速起点——但需结合领域知识

# 模型评估可视化

## 本章目标

1. 掌握分类任务中混淆矩阵、ROC 曲线和学习曲线的可视化流程
2. 理解概率输出、阈值变化与分类性能之间的关系
3. 学会通过学习曲线判断欠拟合与过拟合趋势

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `confusion_matrix(y_true, y_pred)` | 函数 | 计算预测与真实标签的混淆矩阵 |
| `ConfusionMatrixDisplay(cm)` | 构造器 | 标准化展示混淆矩阵图 |
| `roc_curve(y_true, y_score)` | 函数 | 计算 ROC 曲线坐标 |
| `auc(x, y)` | 函数 | 计算曲线下面积 |
| `learning_curve(estimator, X, y)` | 函数 | 评估样本规模与泛化性能关系 |

## 1. 混淆矩阵

### `confusion_matrix` + `ConfusionMatrixDisplay`

#### 作用

混淆矩阵直接展示 TP、TN、FP、FN 组成，是分类诊断基础。`ConfusionMatrixDisplay` 能快速绘制规范图形并附带标签。在类别不平衡任务中，混淆矩阵比单一准确率更有解释力。

#### 重点方法

```python
confusion_matrix(y_true, y_pred, *, labels=None, normalize=None)
ConfusionMatrixDisplay(confusion_matrix, *, display_labels=None)
ConfusionMatrixDisplay.plot(*, ax=None, cmap=None, colorbar=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_pred` | `array_like` | 预测标签 | `clf.predict(X_test)` |
| `normalize` | `str` | 归一化：`"true"` / `"pred"` / `"all"` | `"true"` |
| `display_labels` | `list[str]` | 显示标签 | `["Class 0", "Class 1"]` |
| `cmap` | `str` | 颜色映射 | `"Blues"` |

#### 示例代码

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = LogisticRegression(random_state=42).fit(X_train, y_train)
cm = confusion_matrix(y_test, clf.predict(X_test))

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/08_confusion.png
2x2 混淆矩阵展示每类预测正确与错误数量
```

![混淆矩阵](https://img.yumeko.site/file/articles/ML/visualization/08_confusion.png)

#### 理解重点

- 误报和漏报的业务代价不同——混淆矩阵是阈值调优依据
- 报告时建议同时给出 precision、recall 与混淆矩阵
- `normalize='true'` 可将数值转为行百分比——适合类别不平衡场景

## 2. ROC 曲线

### `roc_curve` + `auc`

#### 作用

ROC 曲线反映不同阈值下的 TPR 与 FPR 权衡。AUC 越大通常表示排序能力越强。ROC 图可用于比较多个模型的判别性能。完美分类器 AUC=1，随机猜测 AUC=0.5。

#### 重点方法

```python
roc_curve(y_true, y_score, *, pos_label=None)
auc(x, y)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like` | 真实标签 | `y_test` |
| `y_score` | `array_like` | 正类概率得分 | `clf.predict_proba(X_test)[:, 1]` |
| `pos_label` | `int` 或 `str` | 正类标签，默认为 `None`（自动判断） | `1` |

#### 示例代码

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = LogisticRegression(random_state=42).fit(X_train, y_train)
yProba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, yProba)
rocAuc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {rocAuc:.3f})")
ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/08_roc.png
模型 ROC 曲线位于随机基线之上并给出 AUC
```

![ROC 曲线](https://img.yumeko.site/file/articles/ML/visualization/08_roc.png)

#### 理解重点

- ROC 关注排序能力，不直接反映阈值下的精确率
- 正负样本极不平衡时建议同时观察 PR 曲线
- 随机基线（对角线）是最低参照——任何模型应在此之上

## 3. 学习曲线

### `learning_curve`

#### 作用

学习曲线描述训练样本量变化对训练分数与验证分数的影响。训练分数高而验证分数低通常提示过拟合。两条曲线都偏低通常提示欠拟合或特征不足。

#### 重点方法

```python
learning_curve(estimator, X, y, *, train_sizes=None, cv=None, scoring=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 待评估模型 | `LogisticRegression(random_state=42)` |
| `X, y` | `array_like` | 全量特征与标签 | `X, y` |
| `train_sizes` | `array_like` | 训练集比例序列 | `np.linspace(0.1, 1.0, 10)` |
| `cv` | `int` | 交叉验证折数 | `5` |
| `scoring` | `str` | 评分方式，默认为 `None` | `"accuracy"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
clf = LogisticRegression(random_state=42)

trainSizes, trainScores, testScores = learning_curve(
    clf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
trainMean = trainScores.mean(axis=1)
testMean = testScores.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trainSizes, trainMean, "o-", label="Training Score")
ax.plot(trainSizes, testMean, "o-", label="Validation Score")
ax.fill_between(trainSizes,
                testMean - testScores.std(axis=1),
                testMean + testScores.std(axis=1), alpha=0.2)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend()
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/08_learning.png
训练曲线与验证曲线随样本增加逐步收敛
```

![学习曲线](https://img.yumeko.site/file/articles/ML/visualization/08_learning.png)

#### 理解重点

- 学习曲线是判断"继续加数据是否有收益"的核心依据
- 高方差（大间隙）= 过拟合，高偏差（低分平缓）= 欠拟合
- 曲线分析应与模型复杂度和特征工程一起综合判断

## 常见坑

1. 混淆矩阵只看数字不看比例——类别不平衡时归一化更清晰
2. ROC 曲线用类别预测而非概率——曲线退化为单点失去意义
3. 学习曲线的标准差带过宽——暗示数据划分或模型不稳定

## 小结

- 混淆矩阵是分类评估的基石——先看矩阵，再看指标
- ROC 曲线反映模型排序能力——AUC 是快速对比工具
- 学习曲线诊断过拟合/欠拟合——调参前先确认方向
- 评估可视化核心三件套：混淆矩阵 + ROC 曲线 + 学习曲线

# Plotly 交互式图表

## 本章目标

1. 理解 Plotly 交互式图表的核心 API 与工作流
2. 掌握常见交互图类型（折线、散点、柱状、3D）的构建方法
3. 学会将交互图导出为 HTML 或静态图片用于报告交付

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `px.scatter(df, x, y)` | 函数 | 快速构建交互散点图 |
| `px.line(df, x, y)` | 函数 | 构建交互折线图 |
| `px.bar(df, x, y)` | 函数 | 构建交互柱状图 |
| `px.scatter_3d(df, x, y, z)` | 函数 | 构建三维交互散点图 |
| `fig.update_layout(**kwargs)` | 方法 | 统一图表布局样式 |
| `fig.write_html(file)` | 方法 | 导出 HTML 文件（保留交互） |
| `fig.write_image(file)` | 方法 | 导出静态图片 |

## 1. Plotly 基础

### `plotly.express.scatter`

#### 作用

Plotly 图表默认支持缩放、平移、悬停提示等交互动作。Plotly Express 适合快速构建，Graph Objects 适合精细控制。在 Notebook 与 Web 报告中，Plotly 的交互优势明显——无需编写 JavaScript。

#### 重点方法

```python
px.scatter(data_frame=None, *, x=None, y=None, color=None, size=None,
           title=None)
fig.show()         # 在当前环境渲染交互图
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data_frame` | `DataFrame` | 输入数据 | `df` |
| `x` | `str` | 横轴字段 | `"x"` |
| `y` | `str` | 纵轴字段 | `"y"` |
| `color` | `str` | 颜色映射字段 | `"category"` |
| `size` | `str` | 点大小映射字段 | `"value"` |
| `title` | `str` | 图标题 | `"Plotly Scatter"` |

#### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [2, 3, 2, 5],
    "category": ["A", "A", "B", "B"],
})

fig = px.scatter(df, x="x", y="y", color="category", title="Plotly Scatter")
fig.show()
```

#### 输出

```text
交互能力: 鼠标悬停显示点信息，滚轮缩放坐标轴
运行结果: 浏览器或 Notebook 渲染可交互散点图
```

#### 理解重点

- Plotly 的"可交互默认值"降低了前端开发成本
- 先用 Express 快速验证，再按需下沉到 Graph Objects
- `fig.show()` 在脚本中会打开浏览器，在 Notebook 中内嵌渲染

## 2. 交互式图表实例

### `px.line` / `px.bar` / `px.scatter_3d`

#### 作用

折线、散点、柱状、3D 散点是最常见的业务展示组合。相同数据在不同图形中关注重点不同，应按问题选图。交互图允许读者自己探索局部细节，提升分析透明度。

#### 重点方法

```python
px.line(data_frame, *, x=None, y=None, color=None, title=None)
px.bar(data_frame, *, x=None, y=None, color=None, title=None)
px.scatter_3d(data_frame, *, x=None, y=None, z=None, color=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data_frame` | `DataFrame` | 输入数据 | `df` |
| `x` / `y` | `str` | 横/纵轴字段 | `"date"`, `"value"` |
| `color` | `str` | 分组上色字段 | `"group"` |
| `z` | `str` | 3D 图的第三轴字段 | `"z"` |
| `title` | `str` | 图标题 | `"Time Series"` |

#### 示例代码

```python
import pandas as pd
import plotly.express as px
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
    "value": np.cumsum(np.random.randn(30)),
    "category": np.random.choice(["A", "B"], 30),
})

figLine = px.line(df, x="date", y="value", title="Time Series")
figBar = px.bar(df, x="category", y="value", color="category",
                title="Category Value")
```

#### 输出

```text
折线图: 可缩放时间区间并查看局部波动
柱状图: 可点击图例切换类别显示
```

#### 理解重点

- 交互式图表适合面向业务方的自助探索场景
- 图形越多越要统一颜色和命名，降低认知负担
- 3D 散点图适合展示聚类或降维结果——可旋转视角

## 3. Plotly 实用技巧

### 布局与导出

#### 作用

导出 HTML 可保留完整交互能力，适合分享与归档。导出静态图片适合论文、报告与邮件场景。统一布局配置是构建图表风格系统的关键步骤。

#### 重点方法

```python
fig.update_layout(*, title=None, xaxis_title=None, yaxis_title=None,
                  template=None)
fig.write_html(file)           # 导出交互式 HTML
fig.write_image(file)          # 导出静态图片（需 kaleido）
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `title` | `str` | 图标题 | `"Chart Title"` |
| `xaxis_title` / `yaxis_title` | `str` | 轴标题 | `"X Axis"`, `"Y Axis"` |
| `template` | `str` | 全局主题：`"plotly"` / `"plotly_dark"` / `"ggplot2"` / `"seaborn"` 等 | `"plotly_dark"` |
| `file` | `str` | 导出文件路径 | `"chart.html"` |

#### 示例代码

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 4]})
fig = px.line(df, x="x", y="y", title="Demo")
fig.update_layout(
    title="Styled Chart",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    template="plotly_dark",
)
fig.write_html("chart.html")
print("已导出: chart.html")
```

#### 输出

```text
导出结果: chart.html 可在浏览器独立打开
可选导出: fig.write_image("chart.png") 适合静态文档嵌入
```

#### 理解重点

- 导出策略取决于读者是否需要交互能力
- 统一模板和标题规范能显著提升团队交付质量
- `write_image` 需要安装 kaleido：`pip install kaleido`
- Plotly Express 返回的 `Figure` 对象与 Graph Objects 完全兼容

## 常见坑

1. `fig.show()` 在纯脚本中可能打开空浏览器窗口——Notebook 中使用更佳
2. `write_image` 未安装 kaleido 导致导出失败
3. 3D 图默认视角可能遮挡关键数据——需手动设置 `camera` 参数

## 小结

- Plotly Express 是构建交互图的最快路径——一行代码出图且默认可交互
- 交互图适合探索和演示——静态 Matplotlib 图更适合印刷报告
- 导出 HTML 保留交互能力，导出 PNG/PDF 适合固定交付
- `update_layout` 统一设置标题和主题是专业交付的基本要求

# 图表报告与交付

## 本章目标

1. 掌握专业报告图的样式统一、布局设计和输出规范
2. 学会使用 GridSpec 构建复杂多面板可视化版式
3. 理解导出参数与配色体系对交付质量的影响

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plt.style.use(style)` | 函数 | 统一图表风格模板 |
| `fig.add_gridspec(nrows, ncols)` | 方法 | 创建复杂网格布局 |
| `fig.add_subplot(gs[...])` | 方法 | 在布局中添加子图 |
| `plt.savefig(fname, dpi)` | 函数 | 导出高分辨率图像 |
| `plt.cm.get_cmap(name)` | 函数 | 使用内置色图管理配色 |

## 1. 专业样式设置

### `plt.style.use` + 标题层级

#### 作用

报告图首要目标是可读性一致，而非单图视觉炫技。`style.use` 可统一网格线、字体、背景等全局风格。标题、轴标签、图例应形成固定层级规范。

#### 重点方法

```python
plt.style.use(style)                              # 设置全局样式
ax.set_title(label, *, fontsize=None, fontweight=None)
fig.savefig(fname, *, dpi=None, bbox_inches=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `style` | `str` | 全局样式模板：`"seaborn-v0_8-whitegrid"` / `"ggplot"` / `"fivethirtyeight"` 等 | `"seaborn-v0_8-whitegrid"` |
| `label` | `str` | 标题文本 | `"Professional Style Chart"` |
| `fontsize` | `int` | 标题字号 | `14` |
| `fontweight` | `str` | 标题字重：`"normal"` / `"bold"` | `"bold"` |
| `fname` | `str` | 输出文件路径 | `"output.png"` |
| `dpi` | `int` | 分辨率，默认为 `None` | `150` |
| `bbox_inches` | `str` | 边界控制：`"tight"` 紧凑裁切，默认为 `None` | `"tight"` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, np.sin(x), linewidth=2, label="sin(x)")
ax.plot(x, np.cos(x), linewidth=2, label="cos(x)")
ax.set_title("Professional Style Chart", fontsize=14, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=True, fancybox=True, shadow=True)
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/10_professional.png
统一网格风格、标题层级和图例外观
```

![专业样式](https://img.yumeko.site/file/articles/ML/visualization/10_professional.png)

#### 理解重点

- 风格一致性比单图复杂度更能提升报告专业感
- 建议把常用样式配置固化为团队模板
- `plt.style.available` 可查看所有内置样式名称

## 2. 多面板布局

### `GridSpec` 不规则布局

#### 作用

GridSpec 支持不规则布局，适合仪表盘和报告页组合图。一个主图配多个辅助图是最常见的讲故事结构。布局阶段就应确定主次关系与读图顺序。

#### 重点方法

```python
fig.add_gridspec(nrows, ncols, *, hspace=None, wspace=None)
fig.add_subplot(gs[row_slice, col_slice])
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `nrows` | `int` | GridSpec 行数 | `2` |
| `ncols` | `int` | GridSpec 列数 | `3` |
| `hspace` | `float` | 行间距（相对高度） | `0.3` |
| `wspace` | `float` | 列间距（相对宽度） | `0.3` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :2])    # 上方主图（跨2列）
ax2 = fig.add_subplot(gs[0, 2])     # 右上辅助图
ax3 = fig.add_subplot(gs[1, 0])     # 左下
ax4 = fig.add_subplot(gs[1, 1])     # 中下
ax5 = fig.add_subplot(gs[1, 2])     # 右下

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x)); ax1.set_title("Main: sin(x)")
ax2.hist(np.random.randn(500), bins=20); ax2.set_title("Distribution")
ax3.plot(x, np.cos(x)); ax3.set_title("cos(x)")
ax4.scatter(np.random.randn(50), np.random.randn(50)); ax4.set_title("Scatter")
ax5.plot(x, np.exp(-x/3) * np.sin(x)); ax5.set_title("Damped")
plt.close()
```

#### 输出

```text
控制台提示: 图表已保存到 outputs/visualization/10_multipanel.png
上方主图 + 右上分布图 + 下方三图组合布局
```

![多面板布局](https://img.yumeko.site/file/articles/ML/visualization/10_multipanel.png)

#### 理解重点

- 复杂布局先画草图再编码，效率更高
- 主图面积通常应大于辅助图，避免重点分散
- `gs[row, col]` 支持切片——`gs[0, :2]` 表示第0行占前两列

## 3. 导出选项

### `plt.savefig` 格式与参数

#### 作用

不同交付场景需要不同导出格式和分辨率策略。PNG 适合网页，PDF/SVG 适合矢量打印与论文。`bbox_inches='tight'` 能有效减少多余留白。

#### 重点方法

```python
plt.savefig(fname, *, dpi=None, bbox_inches=None, transparent=False,
            facecolor='auto')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fname` | `str` | 输出路径，扩展名决定格式 | `"fig.png"` / `"fig.pdf"` / `"fig.svg"` |
| `dpi` | `int` | 分辨率（位图格式有效），默认为 `None` | `300` |
| `bbox_inches` | `str` | 边界控制：`"tight"` 紧凑裁切 | `"tight"` |
| `transparent` | `bool` | 是否透明背景，默认为 `False` | `True` |
| `facecolor` | `str` | 背景色，默认为 `"auto"` | `"white"` |

#### 示例代码

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

plt.savefig("fig.png", dpi=300, bbox_inches="tight")
plt.savefig("fig.pdf", bbox_inches="tight")
plt.savefig("fig.svg", transparent=True)
print("已导出: fig.png / fig.pdf / fig.svg")
```

#### 输出

```text
已导出: fig.png / fig.pdf / fig.svg
推荐策略: 报告预览用 PNG，正式发布优先 PDF 或 SVG
```

#### 理解重点

- 导出前先确认下游使用场景，避免重复返工
- 位图格式（PNG）适合屏幕阅读，矢量格式（PDF/SVG）适合印刷
- `dpi=300` 适合打印，`dpi=150` 适合屏幕

## 4. 配色方案

### 色图类型与选择

#### 作用

配色应服务信息层级，而不是追求"颜色多"。连续变量、发散变量、类别变量应使用不同色图类别。团队报告建议固定主色板与强调色，保证视觉一致。

#### 重点方法

```python
plt.cm.get_cmap(name, lut=None)          # 获取色图对象
plt.cm.Set1(np.linspace(0, 1, n))        # 抽样 n 种离散色
```

#### 速查表

| 色图类型 | 用途 | 典型色图 |
|---|---|---|
| 顺序色图 | 数值大小编码 | `viridis` `plasma` `magma` `cividis` |
| 发散色图 | 正负/偏离编码 | `coolwarm` `RdBu` `seismic` |
| 定性色图 | 类别区分 | `Set1` `Set2` `tab10` `Pastel1` |

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

print("顺序色图: viridis, plasma, magma, cividis, inferno")
print("发散色图: coolwarm, RdBu, seismic, bwr")
print("定性色图: Set1, Set2, tab10, Pastel1")

colors = plt.cm.Set1(np.linspace(0, 1, 5))
print(f"Set1 抽样 5 色: {colors}")
```

#### 输出

```text
顺序色图: viridis, plasma, magma, cividis, inferno
发散色图: coolwarm, RdBu, seismic, bwr
定性色图: Set1, Set2, tab10, Pastel1
Set1 抽样 5 色: [[0.894 0.102 0.110 1.   ] ...]
```

#### 理解重点

- 颜色体系应与业务语义绑定——例如红色表示风险、绿色表示健康
- 建议做色盲友好检查，避免关键信息仅靠颜色传达
- `viridis` 是 matplotlib 默认色图——色盲友好且在灰度打印中可区分

## 常见坑

1. `plt.style.use` 会影响整个脚本后续所有图——应在文件开头调用
2. `bbox_inches='tight'` 可能裁掉部分图例——需检查最终输出
3. SVG 文件可能包含多余白边——配合 `fig.subplots_adjust` 微调
4. 顺序色图用于类别数据——导致无意义颜色梯度误导读者

## 小结

- 报告图的核心是统一风格 + 合理布局 + 精准配色
- `plt.style.use` 设置全局样式，GridSpec 构建复杂版面
- 导出前明确目标格式和分辨率——不同场景不同策略
- 配色遵循"连续用顺序、类别用定性、偏离用发散"的选图规则
