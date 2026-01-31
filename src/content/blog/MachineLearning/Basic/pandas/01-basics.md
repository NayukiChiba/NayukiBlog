---
title: Pandas 基础入门
date: 2026-01-10
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: Pandas 基础教程，掌握 Series 和 DataFrame 核心数据结构
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

# Pandas 基础入门

## 学习目标

- 了解 Pandas 的核心数据结构
- 掌握 Series 和 DataFrame 的创建
- 学会基本的数据查看方法

## 核心数据结构

| 数据结构      | 维度 | 说明                 |
| ------------- | ---- | -------------------- |
| **Series**    | 一维 | 带索引的一维数组     |
| **DataFrame** | 二维 | 带行列索引的二维表格 |

## Series（一维数据结构）

### 什么是 Series？

Series 是 Pandas 的一维数据结构，类似于带标签的数组或字典。它由两部分组成：

- **索引（index）**：数据的标签
- **值（values）**：实际存储的数据

### 创建 Series

#### 1. 从列表创建

```python
import pandas as pd

# 从列表创建（使用默认的整数索引 0, 1, 2...）
s = pd.Series([1, 2, 3, 4, 5])
print(s)
```

**输出结果**：

```
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

**说明**：当不指定索引时，Pandas 自动创建从 0 开始的整数索引。

#### 2. 指定自定义索引

```python
# 指定索引（使用有意义的标签）
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
```

**输出结果**：

```
a    10
b    20
c    30
dtype: int64
```

**原理**：自定义索引使数据更有意义，可以通过标签访问数据，如 `s['a']` 返回 `10`。

#### 3. 从字典创建

```python
# 从字典创建（键自动成为索引）
s = pd.Series({'apple': 100, 'banana': 200, 'orange': 150})
print(s)
```

**输出结果**：

```
apple     100
banana    200
orange    150
dtype: int64
```

**优势**：字典创建方式最直观，键值对应关系清晰，适合存储配对数据。

### Series 的基本属性

```python
print(f"索引: {s.index}")        # 获取索引
print(f"值: {s.values}")         # 获取值数组
print(f"数据类型: {s.dtype}")    # 数据类型
print(f"形状: {s.shape}")        # 形状（元素个数）
```

## DataFrame（二维数据结构）

### 什么是 DataFrame？

DataFrame 是 Pandas 的核心数据结构，是一个二维表格，类似于：

- Excel 中的工作表
- SQL 数据库中的表
- R 语言中的 data.frame

它具有**行索引**和**列索引**，每一列可以是不同的数据类型。

### 创建 DataFrame

#### 1. 从字典创建（最常用）

```python
# 字典的键成为列名，值成为列数据
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
    'Salary': [8000, 12000, 15000, 10000]
}
df = pd.DataFrame(data)
print(df)
```

**输出结果**：

```
      Name  Age       City  Salary
0    Alice   25    Beijing    8000
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
3    David   28   Shenzhen   10000
```

**原理**：

- 每个键值对表示一列数据
- 行索引自动生成（0, 1, 2, 3...）
- 各列数据类型可以不同（Name 是字符串，Age 是整数）

#### 2. 指定自定义行索引

```python
df = pd.DataFrame(data, index=['emp1', 'emp2', 'emp3', 'emp4'])
print(df)
```

**输出结果**：

```
          Name  Age       City  Salary
emp1    Alice   25    Beijing    8000
emp2      Bob   30   Shanghai   12000
emp3  Charlie   35  Guangzhou   15000
emp4    David   28   Shenzhen   10000
```

**应用场景**：使用有意义的索引（如员工编号、日期等）方便数据检索。

#### 3. 从列表的列表创建

```python
data = [
    ['Alice', 25, 'Beijing'],
    ['Bob', 30, 'Shanghai'],
    ['Charlie', 35, 'Guangzhou']
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'City'])
print(df)
```

**说明**：需要手动指定列名，每个内部列表代表一行数据。

#### 4. 从 NumPy 数组创建

```python
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
```

**输出结果**：

```
   A  B  C
0  1  2  3
1  4  5  6
2  7  8  9
```

## 基本数据查看

掌握快速查看和了解数据是数据分析的第一步。

### 1. 查看前几行和后几行

```python
# 查看前 3 行（默认 5 行）
print(df.head(3))
```

**输出结果**：

```
      Name  Age      City  Salary
0    Alice   25   Beijing    8000
1      Bob   30  Shanghai   12000
2  Charlie   35 Guangzhou   15000
```

**应用**：快速了解数据的结构和内容，验证数据是否正确加载。

```python
# 查看后 2 行
print(df.tail(2))
```

**输出结果**：

```
      Name  Age      City  Salary
2  Charlie   35 Guangzhou   15000
3    David   28  Shenzhen   10000
```

### 2. 查看数据基本信息

```python
df.info()
```

**输出结果**：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Name    4 non-null      object
 1   Age     4 non-null      int64
 2   City    4 non-null      object
 3   Salary  4 non-null      int64
dtypes: int64(2), object(2)
memory usage: 256.0+ bytes
```

**说明**：

- **RangeIndex**: 索引范围
- **Non-Null Count**: 非空值数量（用于检测缺失值）
- **Dtype**: 每列的数据类型
- **memory usage**: 内存占用

**用途**：检查数据完整性、数据类型是否正确、是否有缺失值。

### 3. 统计描述

```python
print(df.describe())
```

**输出结果**：

```
             Age        Salary
count   4.000000      4.000000
mean   29.500000  11250.000000
std     4.203173   2986.066574
min    25.000000   8000.000000
25%    27.250000   9500.000000
50%    29.000000  11000.000000
75%    31.250000  12750.000000
max    35.000000  15000.000000
```

**说明**：

- **count**: 数据个数
- **mean**: 平均值
- **std**: 标准差
- **min/max**: 最小值/最大值
- **25%/50%/75%**: 四分位数

**注意**：`describe()` 默认只显示数值型列的统计信息。

```python
# 查看所有列（包括非数值列）
print(df.describe(include='all'))
```

### 4. 查看形状和数据类型

```python
# 形状：(行数, 列数)
print(f"数据形状: {df.shape}")  # 输出: (4, 4)

# 各列数据类型
print(f"数据类型:\n{df.dtypes}")
```

**输出结果**：

```
数据形状: (4, 4)
数据类型:
Name      object
Age        int64
City      object
Salary     int64
dtype: object
```

### 5. 查看索引和列名

```python
print(f"列名: {df.columns.tolist()}")  # ['Name', 'Age', 'City', 'Salary']
print(f"索引: {df.index.tolist()}")    # [0, 1, 2, 3]
```

### 6. 查看唯一值和计数

```python
# 查看某列的唯一值
print(df['City'].unique())  # ['Beijing' 'Shanghai' 'Guangzhou' 'Shenzhen']

# 查看唯一值数量
print(df['City'].nunique())  # 4

# 值计数（统计频率）
print(df['City'].value_counts())
```

**应用场景**：

- 检查数据中有多少个不同的城市
- 找出出现频率最高的类别
- 检测数据是否有异常值

### 常用查看方法速查表

| 方法            | 说明                         | 返回类型  |
| --------------- | ---------------------------- | --------- |
| `df.head(n)`    | 查看前 n 行（默认 5）        | DataFrame |
| `df.tail(n)`    | 查看后 n 行（默认 5）        | DataFrame |
| `df.info()`     | 数据基本信息（类型、缺失值） | None      |
| `df.describe()` | 数值列统计描述               | DataFrame |
| `df.shape`      | 形状 (行数, 列数)            | tuple     |
| `df.dtypes`     | 各列数据类型                 | Series    |
| `df.columns`    | 列名                         | Index     |
| `df.index`      | 行索引                       | Index     |
| `df.values`     | 转为 NumPy 数组              | ndarray   |

## 练习

```bash
python Basic/Pandas/01_basics.py
```
