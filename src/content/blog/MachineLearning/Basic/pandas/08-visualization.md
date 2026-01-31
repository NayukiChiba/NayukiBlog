---
title: Pandas 数据可视化
date: 2026-01-13
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 使用 Pandas 内置绑图功能快速可视化数据
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

# Pandas 数据可视化

## 学习目标

- 掌握 df.plot() 基本绑图
- 了解不同图表类型
- 学会自定义图表样式

## 基本绑图

Pandas 基于 Matplotlib 提供了简单便捷的绘图接口，使用 `.plot()` 方法即可快速可视化数据。

### 1. 准备数据

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 创建测试数据
df = pd.DataFrame({
    'A': np.random.randn(50).cumsum(),
    'B': np.random.randn(50).cumsum(),
    'C': np.random.randn(50).cumsum(),
    'D': np.random.randn(50).cumsum()
})
```

### 2. 默认折线图

```python
# 默认绘制所有列的折线图
df.plot()
plt.title('折线图示例')
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.show()
```

**说明**：

- 默认 `kind='line'`，绘制折线图
- 自动为每列生成不同颜色
- 自动添加图例（legend）

### 3. 选择特定列绘图

```python
# 只绘制 A 和 B 列
df[['A', 'B']].plot()
plt.title('A 和 B 的趋势')
plt.show()
```

### 4. 调整图表大小

```python
# 设置图表大小
df.plot(figsize=(12, 6))  # 宽 12 英寸，高 6 英寸
plt.show()
```

**应用场景**：

- 时间序列趋势分析
- 多个指标对比
- 数据探索性分析

## 图表类型

Pandas 支持多种常用图表类型，通过 `kind` 参数指定。

### 1. 柱状图（Bar Chart）

```python
# 准备数据
data = pd.DataFrame({
    '销售额': [120, 150, 180, 90],
    '成本': [80, 100, 120, 60]
}, index=['第一季度', '第二季度', '第三季度', '第四季度'])

# 垂直柱状图
data.plot(kind='bar', figsize=(10, 6))
plt.title('季度销售对比')
plt.ylabel('金额（万元）')
plt.xticks(rotation=0)  # X轴标签不旋转
plt.show()
```

**输出**：并排的柱状图，显示每个季度的销售额和成本。

```python
# 水平柱状图
data.plot(kind='barh', figsize=(10, 6))
plt.title('季度销售对比（水平）')
plt.xlabel('金额（万元）')
plt.show()
```

```python
# 堆叠柱状图
data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('堆叠柱状图')
plt.show()
```

**效果**：

![pandas_bar_plot](https://img.yumeko.site/file/articles/pdlearn/pandas_bar_plot.png)

### 2. 散点图（Scatter Plot）

```python
# 创建数据
df = pd.DataFrame({
    '身高': np.random.normal(170, 10, 100),
    '体重': np.random.normal(65, 10, 100)
})

# 绘制散点图
df.plot(kind='scatter', x='身高', y='体重',
        figsize=(10, 6), alpha=0.6, s=50)  # alpha：透明度, s：点大小
plt.title('身高与体重关系')
plt.show()
```

**应用**：分析两个变量之间的相关性。

**效果**：

![pandas_scatter](https://img.yumeko.site/file/articles/pdlearn/pandas_scatter.png)

### 3. 直方图（Histogram）

```python
# 分析数据分布
data = pd.Series(np.random.normal(100, 15, 1000))
data.plot(kind='hist', bins=30, figsize=(10, 6),
          edgecolor='black', alpha=0.7)
plt.title('数据分布直方图')
plt.xlabel('值')
plt.ylabel('频数')
plt.show()
```

**参数说明**：

- `bins`：柱子数量（区间数）
- `edgecolor`：边框颜色
- `alpha`：透明度

**效果**：

![pandas_histogram](https://img.yumeko.site/file/articles/pdlearn/pandas_histogram.png)

### 4. 箱线图（Box Plot）

```python
# 创建数据
df = pd.DataFrame({
    'A 组': np.random.normal(100, 10, 100),
    'B 组': np.random.normal(110, 15, 100),
    'C 组': np.random.normal(95, 8, 100)
})

# 绘制箱线图
df.plot(kind='box', figsize=(10, 6))
plt.title('三组数据分布对比')
plt.ylabel('值')
plt.show()
```

**箱线图的含义**：

- 中间线：中位数
- 箱体：25% 到 75% 分位数（IQR）
- 须线：1.5 \* IQR 范围
- 圆点：异常值

**效果**：

![pandas_boxplot](https://img.yumeko.site/file/articles/pdlearn/pandas_boxplot.png)

### 5. 饼图（Pie Chart）

```python
# 市场份额数据
market_share = pd.Series([30, 25, 20, 15, 10],
                         index=['产品A', '产品B', '产品C', '产品D', '产品E'])

market_share.plot(kind='pie', figsize=(8, 8),
                  autopct='%1.1f%%',  # 显示百分比
                  startangle=90)      # 起始角度
plt.title('市场份额分布')
plt.ylabel('')  # 隐藏 y 轴标签
plt.show()
```

**效果**：

![pandas_pie](https://img.yumeko.site/file/articles/pdlearn/pandas_pie.png)

### 图表类型总结

| kind 参数 | 图表类型   | 适用场景             |
| --------- | ---------- | -------------------- |
| `line`    | 折线图     | 时间序列、趋势分析   |
| `bar`     | 柱状图     | 分类对比、排名       |
| `barh`    | 水平柱状图 | 长标签的分类比较     |
| `scatter` | 散点图     | 相关性分析           |
| `hist`    | 直方图     | 数据分布             |
| `box`     | 箱线图     | 分布对比、异常值检测 |
| `pie`     | 饼图       | 比例组成             |
| `area`    | 面积图     | 累积趋势             |
| `density` | 密度图     | 概率密度分布         |
| `hexbin`  | 六角箱图   | 大量数据点的分布密度 |

## 常用参数

掌握常用参数可以让图表更加美观和专业。

### 1. 基本参数

```python
df.plot(
    kind='line',           # 图表类型
    figsize=(12, 6),       # 图表大小 (宽, 高)
    title='标题',         # 标题
    xlabel='X轴标签',     # X轴标签
    ylabel='Y轴标签',     # Y轴标签
    legend=True,           # 显示图例
    grid=True,             # 显示网格
    style='.-',            # 线条样式
    color='blue',          # 颜色
    alpha=0.7,             # 透明度 (0-1)
    linewidth=2,           # 线宽
    rot=45                 # X轴标签旋转角度
)
plt.show()
```

### 2. 颜色和样式

```python
# 指定颜色
df.plot(color=['red', 'green', 'blue'])

# 使用颜色映射
df.plot(colormap='viridis')  # 'hot', 'cool', 'spring', 'rainbow'

# 线条样式
df.plot(style=['-', '--', '-.', ':'])  # 实线、虚线、点划线、点线
```

### 3. 图例位置

```python
df.plot(figsize=(10, 6))
plt.legend(loc='best')  # 'upper left', 'upper right', 'lower left', 'lower right', 'center'
plt.show()
```

### 4. 坐标轴设置

```python
df.plot(figsize=(10, 6))
plt.xlim(0, 50)      # X轴范围
plt.ylim(-10, 10)    # Y轴范围
plt.xticks(rotation=45)  # X轴标签旋转
plt.show()
```

## 多子图

多子图用于在一个图表中展示多个相关的可视化。

### 1. 自动分割子图

```python
# 创建数据
df = pd.DataFrame(np.random.randn(50, 4), columns=['A', 'B', 'C', 'D'])

# 每列一个子图
df.plot(subplots=True, layout=(2, 2), figsize=(12, 8))
plt.tight_layout()  # 自动调整子图间距
plt.show()
```

**说明**：

- `subplots=True`：每列单独绘制
- `layout=(2, 2)`：2行2列排列
- `tight_layout()`：避免子图重叠

### 2. 自定义子图布局

```python
# 创建图表和子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 在指定子图上绘制
df['A'].plot(ax=axes[0, 0], title='子图 A')
df['B'].plot(ax=axes[0, 1], title='子图 B', color='red')
df['C'].plot(ax=axes[1, 0], title='子图 C', kind='bar')
df['D'].plot(ax=axes[1, 1], title='子图 D', kind='hist')

plt.tight_layout()
plt.show()
```

**优点**：每个子图可以使用不同的图表类型和参数。

## 实际应用示例

### 示例 1：销售数据分析

```python
# 创建销售数据
dates = pd.date_range('2024-01-01', periods=365, freq='D')
sales_data = pd.DataFrame({
    '销售额': np.random.randint(1000, 5000, 365) +
              np.sin(np.arange(365) * 2 * np.pi / 365) * 500,
    '访问量': np.random.randint(100, 500, 365)
}, index=dates)

# 绘制多子图
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 销售额趋势
sales_data['销售额'].plot(ax=axes[0], title='日销售额趋势',
                           color='steelblue', linewidth=1)
axes[0].set_ylabel('销售额（元）')

# 30天移动平均
ma30 = sales_data['销售额'].rolling(30).mean()
ma30.plot(ax=axes[0], color='red', label='30天均线', linewidth=2)
axes[0].legend()

# 访问量分布
sales_data['访问量'].plot(kind='hist', ax=axes[1],
                           title='访问量分布',
                           bins=30, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('访问量')
axes[1].set_ylabel('频数')

plt.tight_layout()
plt.show()
```

### 示例 2：多维数据对比

```python
# 部门绩效对比
performance = pd.DataFrame({
    'Q1': [85, 92, 78, 88, 95],
    'Q2': [88, 95, 82, 90, 92],
    'Q3': [90, 93, 85, 92, 94],
    'Q4': [92, 96, 88, 95, 98]
}, index=['部门A', '部门B', '部门C', '部门D', '部门E'])

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 分组柱状图
performance.plot(kind='bar', ax=axes[0],
                title='各部门季度绩效对比')
axes[0].set_ylabel('绩效分数')
axes[0].set_xlabel('部门')
axes[0].legend(title='季度')
axes[0].set_xticklabels(performance.index, rotation=0)

# 雷达图
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

# 简化：使用折线图展示平均绩效
avg_performance = performance.mean(axis=1)
avg_performance.plot(kind='barh', ax=axes[1], color='skyblue',
                    title='平均绩效排名')
axes[1].set_xlabel('平均分数')

plt.tight_layout()
plt.show()
```

## 可视化最佳实践

### 1. 选择合适的图表类型

```python
# 时间序列 -> 折线图
ts_data.plot(kind='line')

# 分类比较 -> 柱状图
category_data.plot(kind='bar')

# 相关性 -> 散点图
df.plot(kind='scatter', x='var1', y='var2')

# 分布 -> 直方图/箱线图
df.plot(kind='hist')
df.plot(kind='box')
```

### 2. 确保图表可读性

```python
# 设置合适的图表大小
df.plot(figsize=(12, 6))

# 添加标题和标签
plt.title('明确的标题', fontsize=14)
plt.xlabel('X轴说明', fontsize=12)
plt.ylabel('Y轴说明', fontsize=12)

# 添加网格
plt.grid(True, alpha=0.3)

# 调整图例
plt.legend(loc='best', fontsize=10)
```

### 3. 颜色和样式

```python
# 使用色盲友好的颜色
df.plot(colormap='Set2')  # 或 'Paired', 'tab10'

# 设置透明度
df.plot(alpha=0.7)

# 使用不同线型
df.plot(style=['-', '--', '-.', ':'])
```

### 4. 保存图表

```python
df.plot(figsize=(10, 6))
plt.title('我的图表')
plt.savefig('my_chart.png', dpi=300, bbox_inches='tight')
plt.show()
```

**参数说明**：

- `dpi=300`：高分辨率
- `bbox_inches='tight'`：去除空白边缘
- 支持格式：png, jpg, pdf, svg

```bash
python Basic/Pandas/08_visualization.py
```
