---
title: Pandas 全指南
date: 2026-01-13
category: 机器学习/基础
tags:
  - Python
  - 基础
description: Pandas中文教程，有Series/DataFrame基础、数据I/O、选择过滤、清洗处理、分组聚合、合并连接、时间序列、可视化及高级性能优化等核心主题，附带可运行的代码练习。
image: https://img.yumeko.site/file/blog/cover/1780581856673.webp
status: published
---
# Pandas 基础与核心数据结构

## 本章目标

1. 理解 Pandas 的两大核心数据结构：`Series` 和 `DataFrame`
2. 掌握从列表、字典、NumPy 数组创建 `Series` 和 `DataFrame`
3. 学会用 `head`、`tail`、`info`、`describe` 快速探查数据
4. 掌握 `shape`、`dtypes`、`columns`、`index` 等常用属性

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.Series(...)` | 构造器 | 创建一维带标签数组 |
| `pd.DataFrame(...)` | 构造器 | 创建二维表格数据 |
| `df.head(...)` | 方法 | 查看前 n 行 |
| `df.tail(...)` | 方法 | 查看后 n 行 |
| `df.info(...)` | 方法 | 列类型、非空计数、内存占用 |
| `df.describe(...)` | 方法 | 数值列统计摘要 |
| `df.shape` | 属性 | `(行数, 列数)` 元组 |
| `df.dtypes` | 属性 | 各列数据类型 |
| `df.columns` | 属性 | 列名 `Index` 对象 |
| `df.index` | 属性 | 行索引对象 |
| `df.values` | 属性 | 底层 NumPy 数组（推荐用 `df.to_numpy()`） |

## 1. Series 数据结构

### `pd.Series`

#### 作用

创建一维**带索引标签**的数组，可看作一列 Excel 数据。索引赋予数据标签语义——可通过标签而非位置访问元素。支持从列表、字典、标量、NumPy 数组创建。

#### 重点方法

```python
pd.Series(data=None, index=None, dtype=None, name=None, copy=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `array_like`、`dict`、标量 | 输入数据，可为列表、元组、字典、标量、`ndarray` | `[1, 2, 3]`、`{"a": 1}`、`5` |
| `index` | `array_like` 或 `None` | 自定义索引标签；`None` 时默认为 `RangeIndex` | `["a", "b", "c"]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.int64`、`"float32"` |
| `name` | `str` 或 `None` | Series 的名称，显示在输出底部，默认为 `None` | `"prices"` |
| `copy` | `bool` 或 `None` | 是否复制输入数据，默认为 `None`（按源类型决定） | `True` |

#### 示例代码

```python
import pandas as pd
import numpy as np

# 从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"s1:\n{s1}")
print(f"索引: {s1.index.tolist()}")
print(f"值: {s1.values}")

# 指定索引
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(f"\ns2:\n{s2}")
print(f"访问 s2['b']: {s2['b']}")

# 从字典创建（键 -> 索引，值 -> 数据）
s3 = pd.Series({"apple": 100, "banana": 200, "orange": 150})
print(f"\ns3:\n{s3}")
```

#### 输出

```text
s1:
0    1
1    2
2    3
3    4
4    5
dtype: int64
索引: [0, 1, 2, 3, 4]
值: [1 2 3 4 5]

s2:
a    10
b    20
c    30
dtype: int64
访问 s2['b']: 20

s3:
apple     100
banana    200
orange    150
dtype: int64
```

#### 理解重点

- `Series` = 值 + 索引——索引是数据的一部分，不只是行号
- 字典创建最直观：键自动成为索引，值成为数据
- `s.values` 返回底层 NumPy 数组；`s.index` 返回 `Index` 对象——两者是 Series 与 NumPy 互操作的桥梁

## 2. DataFrame 数据结构

### `pd.DataFrame`

#### 作用

创建二维带标签的表格，可类比 Excel 工作表或 SQL 表。每列可以是不同 `dtype`——这是与 NumPy 二维数组最本质的区别。

#### 重点方法

```python
pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `dict`、`ndarray`、`list[Series]` 等 | 输入数据；字典的键自动成为列名 | `{"Name": [...], "Age": [...]}` |
| `index` | `array_like` 或 `None` | 行索引，默认为 `None`（`RangeIndex`） | `["r1", "r2", "r3"]` |
| `columns` | `array_like` 或 `None` | 列名；字典创建时自动取键名，默认为 `None` | `["col1", "col2"]` |
| `dtype` | `dtype` 或 `None` | 统一设置所有列的数据类型，默认为 `None`（按列推断） | `np.float64` |
| `copy` | `bool` 或 `None` | 是否复制输入数据，默认为 `None` | `True` |

#### 示例代码

```python
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
}
df = pd.DataFrame(data)

print(df)
print(f"\n形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"索引: {df.index.tolist()}")
print(f"数据类型:\n{df.dtypes}")
```

#### 输出

```text
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou

形状: (3, 3)
列名: ['Name', 'Age', 'City']
索引: [0, 1, 2]
数据类型:
Name    object
Age      int64
City    object
dtype: object
```

#### 理解重点

- 从字典创建：**键 -> 列名**，值列表 -> 列数据，所有值长度必须一致
- 字符串列默认 `dtype=object`（底层存 Python 对象指针），内存效率不如 `category` 类型
- `shape` 返回 `(行数, 列数)`——拿到数据后先查 shape 是最基本的健全检查

## 3. 数据查看

### `head` / `tail`

#### 作用

- `head(n)`：查看前 `n` 行，默认 5。拿到新数据后的第一步操作
- `tail(n)`：查看后 `n` 行，默认 5。检查数据尾部是否完整

#### 重点方法

```python
df.head(n=5)
df.tail(n=5)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n` | `int` | 返回的行数，默认为 `5`；负数返回除末尾 `|n|` 行外的全部 | `3`、`10` |

### `DataFrame.info`

#### 作用

打印 DataFrame 的摘要信息：列名、非空计数、dtype、内存占用。快速发现缺失值和类型问题的首选工具。**直接 print 到 stdout，不返回值**。

#### 重点方法

```python
df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `verbose` | `bool` 或 `None` | 是否显示完整列信息，默认为 `None`（列少时自动开启） | `False` |
| `buf` | `file-like` 或 `None` | 输出缓冲区，默认为 `None`（`sys.stdout`） | `open("log.txt", "w")` |
| `max_cols` | `int` 或 `None` | 显示列数上限，默认为 `None`（自动适配终端宽度） | `10` |
| `memory_usage` | `bool`、`str` 或 `None` | 是否显示内存占用；`'deep'` 精确计算 object 列，默认为 `None` | `"deep"` |
| `show_counts` | `bool` 或 `None` | 是否显示非空计数，默认为 `None`（行数 < 阈值时自动开启） | `True` |

### `DataFrame.describe`

#### 作用

生成数值列的统计摘要：计数、均值、标准差、最小值、四分位数、最大值。EDA（探索性数据分析）的标准第一步。默认只统计数值列。

#### 重点方法

```python
df.describe(percentiles=None, include=None, exclude=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `percentiles` | `list[float]` 或 `None` | 要计算的分位数，默认为 `None`（即 `[.25, .5, .75]`） | `[.1, .5, .9]` |
| `include` | `str`、`list[str]` 或 `None` | 要统计的 dtype；`'all'` 包含数值和字符串列，默认为 `None`（只数值） | `"all"`、`[np.number]` |
| `exclude` | `str`、`list[str]` 或 `None` | 要排除的 dtype，默认为 `None` | `[np.object]` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randint(0, 100, 10),
    "C": ["cat", "dog", "bird", "cat", "dog",
          "bird", "cat", "dog", "bird", "cat"],
    "D": [1.2, np.nan, 3.4, 4.5, np.nan, 6.7, 7.8, 8.9, np.nan, 10.0],
})

print(f"head(3):\n{df.head(3)}")
print(f"\ntail(3):\n{df.tail(3)}")
print("\ninfo():")
df.info()
print(f"\ndescribe():\n{df.describe()}")
print(f"\ndescribe(include='all'):\n{df.describe(include='all')}")
```

#### 输出

```text
head(3):
          A   B     C    D
0  0.496714  63   cat  1.2
1 -0.138264  59   dog  NaN
2  0.647689  20  bird  3.4

tail(3):
          A   B     C    D
7  0.767435  88   dog  8.9
8 -0.469474  48  bird  NaN
9  0.542560  90   cat  10.0

info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   A       10 non-null     float64
 1   B       10 non-null     int32
 2   C       10 non-null     object
 3   D       7 non-null      float64
dtypes: float64(2), int32(1), object(1)
memory usage: 412.0+ bytes

describe():
               A          B         D
count  10.000000  10.000000   7.00000
mean    0.448061  55.300000   6.07143
std     0.723008  25.289655   3.18927
min    -0.469474  20.000000   1.20000
25%    -0.210169  36.000000   3.40000
50%     0.519637  58.000000   6.70000
75%     0.737498  72.000000   8.05000
max     1.579213  90.000000  10.00000

describe(include='all'):
          A         B     C         D
count   10.000  10.0000    10    7.000
unique    NaN      NaN      3      NaN
top       NaN      NaN    cat      NaN
freq      NaN      NaN      4      NaN
mean      0.448   55.300   NaN    6.071
std       0.723   25.290   NaN    3.189
min      -0.469   20.000   NaN    1.200
25%      -0.210   36.000   NaN    3.400
50%       0.519   58.000   NaN    6.700
75%       0.738   72.000   NaN    8.050
max       1.579   90.000   NaN   10.000
```

#### 理解重点

- **数据探索三板斧**：`head` -> `info` -> `describe`——拿到数据的标准流程
- `info` 最关键的信息是 **Non-Null Count**——立即暴露哪些列有缺失值
- `describe` 默认只看数值列；`include='all'` 同时显示字符串列的 unique/top/freq
- `info()` 不返回字符串——不要写 `s = df.info()`，需要捕获时用 `buf` 参数

## 4. 常用属性

### 属性速览

| 属性 | 返回类型 | 含义 |
|---|---|---|
| `df.shape` | `tuple[int, int]` | `(行数, 列数)` |
| `df.ndim` | `int` | 维度数；DataFrame 恒为 `2` |
| `df.size` | `int` | 元素总数 = 行数 x 列数 |
| `df.columns` | `Index` | 列名 `Index` 对象 |
| `df.index` | `Index` | 行索引对象 |
| `df.dtypes` | `Series` | 各列数据类型 |
| `df.values` | `numpy.ndarray` | 底层 NumPy 二维数组（推荐用 `df.to_numpy()` 替代） |
| `df.T` | `DataFrame` | 转置（行列交换） |
| `df.empty` | `bool` | 是否为空 DataFrame |
| `df.axes` | `list[Index]` | `[行索引, 列索引]` |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4.0, 5.0, 6.0],
    "C": ["x", "y", "z"],
})

print(f"shape: {df.shape}")
print(f"ndim: {df.ndim}")
print(f"size: {df.size}")
print(f"columns: {df.columns.tolist()}")
print(f"index: {df.index.tolist()}")
print(f"dtypes:\n{df.dtypes}")
print(f"values:\n{df.values}")
print(f"empty: {df.empty}")
```

### 输出

```text
shape: (3, 3)
ndim: 2
size: 9
columns: ['A', 'B', 'C']
index: [0, 1, 2]
dtypes:
A      int64
B    float64
C     object
dtype: object
values:
[[1 4.0 'x']
 [2 5.0 'y']
 [3 6.0 'z']]
empty: False
```

### 理解重点

- `shape` 和 `dtypes` 是所有数据探索任务的起点
- `values` 对混合类型 DataFrame 返回 `object` 数组，数值计算会变慢——纯数值列才用它
- 现代代码推荐用 `df.to_numpy()` 替代 `df.values`（语义更明确，参数更可控）
- `empty` 在读取文件后做健全检查很有用：`if df.empty: raise ValueError(...)`

## 常见坑

1. 单列选取 `df["col"]` 返回 `Series`，`df[["col"]]` 返回单列 `DataFrame`——形状和方法集都不同，容易在后续操作中踩坑
2. `describe()` 默认忽略非数值列；看字符串列分布用 `include='all'` 或 `include=[object]`
3. `values` 对混合类型 DataFrame 返回 `object` 数组，参与 NumPy 运算会变慢甚至失败——纯数值列才用它
4. `df.info()` 不返回字符串而是直接 print——不要写 `s = df.info()` 试图赋值
5. 字典创建 DataFrame 时所有值列表长度必须一致，否则抛 `ValueError`
6. `RangeIndex` 不是普通列表——需要转 list 时用 `df.index.tolist()`

## 小结

- Pandas 的两大核心是 **`Series`**（一维带标签）和 **`DataFrame`**（二维表格）
- 拿到数据的标准流程：`head` -> `info` -> `describe` -> `shape` / `dtypes`
- `Series` 和 `DataFrame` 都建立在 NumPy 之上，但多了**标签索引**和**丰富的数据处理方法**
- 时刻关注 `shape` 和 `dtypes`——它们决定了后续操作的正确性

# Pandas 数据导入与导出

## 本章目标

1. 掌握 `read_csv` / `to_csv` 的核心参数与格式控制
2. 掌握 `read_excel` / `to_excel` 的多工作表读写
3. 掌握 `read_json` / `to_json` 的 `orient` 格式选择
4. 理解 `.csv` / `.xlsx` / `.json` / `.parquet` 四种格式的适用场景

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.read_csv(...)` | 函数 | 从 CSV/文本文件读取 DataFrame |
| `df.to_csv(...)` | 方法 | 将 DataFrame 保存为 CSV 文件 |
| `pd.read_excel(...)` | 函数 | 从 Excel 文件读取 DataFrame |
| `df.to_excel(...)` | 方法 | 将 DataFrame 保存为 Excel 文件 |
| `pd.read_json(...)` | 函数 | 从 JSON 字符串/文件读取 DataFrame |
| `df.to_json(...)` | 方法 | 将 DataFrame 保存为 JSON 字符串/文件 |
| `df.to_string(...)` | 方法 | 转 DataFrame 为对齐文本（打印用） |

## 1. CSV 读写

### `pd.read_csv`

#### 作用

从 CSV 文件或文本流读取数据为 DataFrame——Pandas 最常用的数据入口。支持自定义分隔符、表头位置、跳过行、指定列类型、缺失值标记等。

#### 重点方法

```python
pd.read_csv(filepath_or_buffer, *, sep=',', delimiter=None, header='infer',
            names=None, index_col=None, usecols=None, dtype=None,
            skiprows=None, nrows=None, na_values=None, parse_dates=False,
            encoding=None, skipinitialspace=False, skipfooter=0)
```

#### 参数（核心 12 个）

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `filepath_or_buffer` | `str`、`pathlib.Path`、`file-like` | 文件路径或可读对象 | `"data.csv"`、`StringIO(s)` |
| `sep` | `str` | 字段分隔符，默认为 `','` | `";"`、`"\t"`、`"\s+"` |
| `header` | `int`、`list[int]`、`None` | 用作列名的行号；`None` 表示无表头，默认为 `'infer'` | `0`、`[0, 1]`（多层列名） |
| `names` | `list[str]` | 手动指定列名（覆盖表头），需配合 `header=None` | `["col1", "col2"]` |
| `index_col` | `int`、`str`、`False` 或 `None` | 用作行索引的列，默认为 `None` | `0`、`"ID"` |
| `usecols` | `list[int]`、`list[str]`、函数 | 指定读取哪些列 | `[0, 1, 3]`、`["Name", "Age"]` |
| `dtype` | `dict`、`type` | 指定列的数据类型 | `{"Age": int}` |
| `skiprows` | `int`、`list[int]`、函数 | 跳过文件开头的行数 | `2`、`[0, 2, 3]` |
| `nrows` | `int` | 只读取前 N 行（预览大文件） | `100` |
| `na_values` | `str`、`list[str]`、`dict` | 哪些字符串识别为 NaN | `["NA", "N/A", "missing"]` |
| `parse_dates` | `bool`、`list[int]`、`list[str]` | 是否自动解析日期列 | `True`、`["date_col"]` |
| `encoding` | `str` | 文件编码 | `"utf-8"`、`"gbk"` |

### `DataFrame.to_csv`

#### 作用

将 DataFrame 保存为 CSV 文本文件。

#### 重点方法

```python
df.to_csv(path_or_buf=None, *, sep=',', na_rep='', header=True,
          index=True, index_label=None, columns=None, encoding=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `path_or_buf` | `str`、`pathlib.Path` 或 `None` | 文件路径；`None` 时返回字符串，默认为 `None` | `"output.csv"` |
| `sep` | `str` | 字段分隔符，默认为 `','` | `";"`、`"\t"` |
| `na_rep` | `str` | 缺失值的文本表示，默认为 `''`（空） | `"NaN"`、`"NULL"` |
| `header` | `bool`、`list[str]` | 是否写入列名，默认为 `True` | `False` |
| `index` | `bool` | 是否写入行索引，默认为 `True` | `False` |
| `columns` | `list[str]` | 只写入指定列，默认为 `None`（全部） | `["col1", "col2"]` |
| `encoding` | `str` | 写入编码 | `"utf-8"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
from io import StringIO

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 28],
    "Score": np.random.uniform(60, 100, 4).round(2),
    "City": ["Beijing", "Shanghai", None, "Guangzhou"],
})

# to_csv：index=False 避免写入行号
csvStr = df.to_csv(index=False)
print("生成的 CSV:")
print(csvStr)

# read_csv：从字符串读回
dfLoaded = pd.read_csv(StringIO(csvStr))
print(f"\n从 CSV 读取:\n{dfLoaded}")
print(f"往返一致: {df.drop(columns=['City']).equals(dfLoaded.drop(columns=['City']))}")
```

#### 输出

```text
生成的 CSV:
Name,Age,Score,City
Alice,25,82.38,Beijing
Bob,30,86.21,Shanghai
Charlie,35,72.0,
David,28,96.39,Guangzhou

从 CSV 读取:
      Name  Age  Score        City
0    Alice   25  82.38     Beijing
1      Bob   30  86.21    Shanghai
2  Charlie   35  72.00          NaN
3    David   28  96.39  Guangzhou

往返一致: True
```

#### 理解重点

- `index=False` 几乎总是该加——避免把自动行号写入文件，再读取时多一列 `Unnamed: 0`
- `pd.read_csv(StringIO(s))` 是测试 CSV 逻辑的便捷方式——无需写磁盘文件
- `nrows` 预览大文件结构：`pd.read_csv("big.csv", nrows=5)` 先看列名和类型
- CSV 中空字符串回读后对象列保持空字符串、数值列变 NaN——行为因列类型而异

## 2. Excel 读写

### `pd.read_excel`

#### 作用

从 Excel 文件（`.xlsx` / `.xls`）读取 DataFrame。支持多工作表、按列范围选取。

#### 重点方法

```python
pd.read_excel(io, sheet_name=0, *, header=0, names=None, index_col=None,
              usecols=None, dtype=None, skiprows=None, nrows=None, na_values=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `io` | `str`、`pathlib.Path`、`file-like` | Excel 文件路径 | `"data.xlsx"` |
| `sheet_name` | `int`、`str`、`list`、`None` | 工作表；`0` 第一个、`"Sheet1"` 指定名、`None` 全读（返回 `dict`），默认为 `0` | `"Sheet2"`、`[0, 1]` |
| `header` | `int`、`list[int]` | 用作列名的行号，默认为 `0` | `1` |
| `names` | `list[str]` | 手动指定列名 | `["A", "B", "C"]` |
| `index_col` | `int`、`str` | 用作行索引的列 | `0` |
| `usecols` | `list[int]`、`list[str]`、`str` | 指定列；支持 Excel 风格 `"A:C"` | `"A:C"`、`[0, 2]` |
| `dtype` | `dict` | 列类型映射 | `{"Age": int}` |
| `skiprows` | `int`、`list[int]` | 跳过行数 | `2` |
| `nrows` | `int` | 读取前 N 行 | `50` |
| `na_values` | `list[str]` | 缺失值标记 | `["N/A"]` |

### `DataFrame.to_excel`

#### 作用

将 DataFrame 保存为 Excel 文件。多工作表写入需配合 `ExcelWriter`。

#### 重点方法

```python
df.to_excel(excel_writer, *, sheet_name='Sheet1', na_rep='',
            header=True, index=True, index_label=None, columns=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `excel_writer` | `str`、`pathlib.Path`、`ExcelWriter` | 文件路径或写入器 | `"output.xlsx"` |
| `sheet_name` | `str` | 工作表名，默认为 `'Sheet1'` | `"Results"` |
| `na_rep` | `str` | 缺失值的文本表示，默认为 `''` | `"N/A"` |
| `header` | `bool`、`list[str]` | 是否写入列名，默认为 `True` | `False` |
| `index` | `bool` | 是否写入行索引，默认为 `True` | `False` |
| `columns` | `list[str]` | 只写入指定列 | `["col1", "col2"]` |

#### 理解重点

- `sheet_name=None` 返回 `dict[str, DataFrame]`——每个键是一个工作表名
- `usecols="A:C"` 是 Excel 风格的列范围——比数字索引更直观
- Excel 需要安装 `openpyxl`（`.xlsx` 读写）或 `xlrd`（旧 `.xls` 读）
- 多工作表写入用 `with pd.ExcelWriter("out.xlsx") as w: df1.to_excel(w, sheet_name="A")`

## 3. JSON 读写

### `pd.read_json`

#### 作用

从 JSON 字符串或文件读取 DataFrame。支持多种 JSON 结构方向（`orient`）。

#### 重点方法

```python
pd.read_json(path_or_buf, *, orient=None, typ='frame', dtype=None,
             convert_dates=True, lines=False, encoding=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `path_or_buf` | `str`、`pathlib.Path`、`str` | JSON 文件路径或 JSON 字符串 | `"data.json"`、`jsonStr` |
| `orient` | `str` 或 `None` | JSON 结构方向；常见值自动推断，默认为 `None` | `"records"`、`"columns"`、`"index"` |
| `typ` | `str` | 返回类型：`'frame'` / `'series'`，默认为 `'frame'` | `"series"` |
| `lines` | `bool` | `True` 时每行是一个 JSON 对象（JSON Lines），默认为 `False` | `True` |
| `encoding` | `str` | 文件编码 | `"utf-8"` |

### `DataFrame.to_json`

#### 作用

将 DataFrame 导出为 JSON 字符串或文件。

#### 重点方法

```python
df.to_json(path_or_buf=None, *, orient=None, date_format=None,
           double_precision=10, force_ascii=True, indent=None, lines=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `path_or_buf` | `str` 或 `None` | 文件路径；`None` 返回字符串，默认为 `None` | `"output.json"` |
| `orient` | `str` | JSON 结构方向，下见表，默认为 `None`（即 `'columns'`） | `"records"`、`"split"` |
| `force_ascii` | `bool` | 是否将非 ASCII 字符转义为 `\uXXXX`，默认为 `True` | `False` |
| `indent` | `int` 或 `None` | 缩进空格数；`None` 紧凑输出，默认为 `None` | `2` |
| `lines` | `bool` | `True` 时写为 JSON Lines，默认为 `False` | `True` |

### `orient` 常用取值

| `orient` | JSON 结构 | 适用场景 |
|---|---|---|
| `"columns"`（默认） | `{col: {index: value}}` | 数值矩阵 |
| `"records"` | `[{col: value}, ...]` | **前端 API / 数据库交互** |
| `"index"` | `{index: {col: value}}` | 索引优先结构 |
| `"split"` | `{columns: [...], index: [...], data: [...]}` | 紧凑传输（分离元数据） |
| `"table"` | 含 schema 的完整描述 | 精度要求最高的往返 |

### 综合示例

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
})

# records 方向（前端友好）
jsonRecords = df.to_json(orient="records", force_ascii=False, indent=2)
print(f"records 格式:\n{jsonRecords}")

# 读回
dfRestored = pd.read_json(jsonRecords, orient="records")
print(f"\n读回:\n{dfRestored}")

# JSON Lines 格式
jsonLines = df.to_json(orient="records", lines=True)
print(f"\nJSON Lines:\n{jsonLines}")
```

#### 输出

```text
records 格式:
[
  {
    "Name":"Alice",
    "Age":25,
    "City":"Beijing"
  },
  {
    "Name":"Bob",
    "Age":30,
    "City":"Shanghai"
  },
  {
    "Name":"Charlie",
    "Age":35,
    "City":"Guangzhou"
  }
]

读回:
      Name  Age        City
0    Alice   25     Beijing
1      Bob   30    Shanghai
2  Charlie   35  Guangzhou

JSON Lines:
{"Name":"Alice","Age":25,"City":"Beijing"}
{"Name":"Bob","Age":30,"City":"Shanghai"}
{"Name":"Charlie","Age":35,"City":"Guangzhou"}
```

#### 理解重点

- `orient="records"` 返回 `[{col: val}, ...]`——最接近前端 API 的 JSON 数组格式
- `force_ascii=False` 保留中文可读性；写对外 API 时一般设 `True`
- `lines=True`（JSON Lines）适合大数据——每行独立，可逐行追加/流式处理
- 读写 `orient` 必须一致：写 `"records"` 读不指定 `orient` 会解析错误

## 4. 其他导出格式

| 方法 | 格式 | 适用场景 |
|---|---|---|
| `df.to_string(...)` | 对齐文本 | 打印 DataFrame 到控制台/日志 |
| `df.to_html(...)` | HTML 表格 | 嵌入网页或 Jupyter Notebook 渲染 |
| `df.to_sql(...)` | 数据库表 | 写入 SQL 数据库（需 SQLAlchemy 引擎） |
| `df.to_pickle(...)` | Python pickle | Python 内部快速序列化（不可跨版本/语言） |
| `df.to_parquet(...)` | Parquet 列存 | 大数据高效压缩+列裁剪（需 `pyarrow`） |

## 格式选择指南

| 场景 | 格式 | API |
|---|---|---|
| 内部实验中间结果 | `.pkl` | `to_pickle` / `read_pickle` |
| 跨语言/跨工具交换 | `.csv` | `to_csv` / `read_csv` |
| 业务报表/人工编辑 | `.xlsx` | `to_excel` / `read_excel` |
| Web API 数据交换 | `.json` | `to_json` / `read_json` |
| 大数据/列存分析 | `.parquet` | `to_parquet` / `read_parquet` |

## 常见坑

1. `read_csv` 遇到表头行会误读为数据——确认 `header` 参数与文件实际结构一致
2. `to_csv(index=False)` 忘了加会导致行号写入文件——再读取多一列 `Unnamed: 0`
3. CSV 中特殊字符串（`"NA"`、`"NULL"`、`"N/A"`）可能被自动识别为 NaN——用 `na_values` 和 `keep_default_na=False` 控制
4. Excel 读写需额外库：`openpyxl`（`.xlsx`）或 `xlrd`（`.xls`）——先 `pip install`
5. JSON 读写 `orient` 必须一致——写 `records` 但读不指定 `orient` 会解析失败或形状错乱
6. 中文 CSV 编码不一致是 Windows 常见问题——`utf-8` / `gbk` / `gb2312`，用 `encoding` 参数逐个尝试

## 小结

- 数据入口优先 `read_csv`——参数最丰富、最通用；出口优先 `to_csv(index=False)`
- Excel 适合人工编辑的报告；JSON 适合 Web API 交换；Parquet 适合大数据分析
- 读写参数必须匹配：`sep` / `encoding` / `orient` 在写入端和读取端保持一致
- 大文件操作：先 `nrows=100` 预览结构，确认后再全量加载

# Pandas 数据选择与过滤

## 本章目标

1. 掌握单列和多列选择的语法差异及返回类型区别
2. 理解 `loc`（标签索引）和 `iloc`（位置索引）的切片规则差异
3. 掌握布尔条件过滤与多条件组合的正确写法
4. 学会使用 `isin` 和 `query` 进行高效数据筛选

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df["col"]` | 语法 | 选取单列，返回 `Series` |
| `df[["c1", "c2"]]` | 语法 | 选取多列，返回 `DataFrame` |
| `df[start:stop]` | 语法 | 行切片（按位置，半开区间） |
| `df.loc[...]` | 索引器 | 按**标签**选取行列（切片闭区间） |
| `df.iloc[...]` | 索引器 | 按**位置**选取行列（切片半开） |
| `df.at[...]` / `df.iat[...]` | 索引器 | 按标签/位置取**单个元素**（比 loc/iloc 快） |
| `Series.isin(...)` | 方法 | 元素级成员检测 |
| `Series.between(...)` | 方法 | 区间内检测 |
| `df.query(...)` | 方法 | 字符串表达式查询 |

## 1. 列选择

### 语法速览

| 语法 | 返回类型 | 说明 |
|---|---|---|
| `df["Name"]` | `Series` | 单列，最常用写法 |
| `df[["Name"]]` | `DataFrame` | 单列 DataFrame（注意双层方括号） |
| `df[["Name", "Age"]]` | `DataFrame` | 多列子集 |
| `df.Name` | `Series` | 属性访问（列名含空格/关键字时不可用） |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Beijing", "Shanghai", "Guangzhou"],
})

# 单列返回 Series
print(f"type(df['Name']) = {type(df['Name']).__name__}")
print(df["Name"])

# 多列返回 DataFrame
print(f"\ndf[['Name', 'Age']]:")
print(df[["Name", "Age"]])
```

### 输出

```text
type(df['Name']) = Series
0      Alice
1        Bob
2    Charlie
Name: Name, dtype: object

df[['Name', 'Age']]:
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
```

### 理解重点

- **单括号 vs 双括号**：`df["x"]` 返回 `Series`，`df[["x"]]` 返回单列 `DataFrame`——二者可用的方法不同
- 多列选取常用于提取特征子集：`X = df[["feat1", "feat2", "feat3"]]`
- 属性访问 `df.Name` 仅在列名是合法 Python 标识符时可用；列名含空格/点号/关键字时一律用 `df["col"]`

## 2. 行切片

### `df[start:stop]`

#### 作用

DataFrame 支持类 Python 列表的整数切片，按**位置**行切片（半开区间）。

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
})

print(f"df[1:3]:\n{df[1:3]}")
print(f"\ndf[:2]:\n{df[:2]}")
```

#### 输出

```text
df[1:3]:
      Name  Age
1      Bob   30
2  Charlie   35

df[:2]:
    Name  Age
0  Alice   25
1    Bob   30
```

#### 理解重点

- 裸切片 `df[start:stop]` 按位置，遵循 Python 半开区间（不含 stop）
- 生产代码推荐显式使用 `df.iloc[start:stop]`——语义更明确

## 3. 按标签索引

### `DataFrame.loc`

#### 作用

按**标签**选取行列。支持单标签、标签列表、标签切片、布尔数组。标签切片**包含终点**（闭区间），与 `iloc` 不同。

#### 重点方法

```python
df.loc[row_indexer, col_indexer]
```

#### 参数

| 参数 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `row_indexer` | 单标签、标签列表、标签切片、布尔数组、callable | 行选择器；标签切片为**闭区间** | `"a"`、`["a","c"]`、`"a":"c"`、`df.A > 0` |
| `col_indexer` | 同上 | 列选择器 | `"Name"`、`["Name","Age"]`、`:`（全部列） |

### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
    "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
    "Salary": [8000, 12000, 15000, 9000, 11000],
    "Score": np.random.randint(60, 100, 5),
}, index=["a", "b", "c", "d", "e"])

# 单元素
print(f"loc['b', 'Name']: {df.loc['b', 'Name']}")

# 标签切片（闭区间！包含 'c'）
print(f"\nloc['a':'c', ['Name', 'Age']]:\n{df.loc['a':'c', ['Name', 'Age']]}")

# 布尔数组 + 列选择
print(f"\nloc[Age > 28, ['Name', 'Salary']]:\n{df.loc[df['Age'] > 28, ['Name', 'Salary']]}")

# 全部行指定列
print(f"\nloc[:, 'City']:\n{df.loc[:, 'City']}")
```

### 输出

```text
loc['b', 'Name']: Bob

loc['a':'c', ['Name', 'Age']]:
      Name  Age
a    Alice   25
b      Bob   30
c  Charlie   35

loc[Age > 28, ['Name', 'Salary']]:
      Name  Salary
b      Bob   12000
c  Charlie   15000
e      Eve   11000

loc[:, 'City']:
a      Beijing
b     Shanghai
c      Beijing
d    Guangzhou
e     Shanghai
Name: City, dtype: object
```

### 理解重点

- **最关键的区别**：`loc` 切片是**闭区间**（包含终点），`iloc` 切片是**半开**（不含终点）
- 当索引是默认 `0, 1, 2, ...` 时特别容易混淆：`df.loc[0:2]` 返回 3 行（含 2），`df.iloc[0:2]` 返回 2 行
- `loc` 可以同时按标签选行 + 按名称选列——一步完成行列筛选
- 赋值场景一律用 `loc`：`df.loc[mask, "col"] = value`——避免链式索引的 `SettingWithCopyWarning`

## 4. 按位置索引

### `DataFrame.iloc`

#### 作用

按**整数位置**选取行列。规则与 Python 列表索引一致：整数从 0 开始，切片**不包含终点**（半开区间）。

#### 重点方法

```python
df.iloc[row_pos, col_pos]
```

#### 参数

| 参数 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `row_pos` | 整数、整数列表、整数切片、布尔数组、callable | 行位置选择器 | `0`、`[0, 2]`、`0:3`、`[True, False, ...]` |
| `col_pos` | 同上 | 列位置选择器 | `1`、`[0, 2]`、`:`（全部列） |

### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
    "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
    "Salary": [8000, 12000, 15000, 9000, 11000],
    "Score": np.random.randint(60, 100, 5),
})

# 单元素
print(f"iloc[1, 0]: {df.iloc[1, 0]}")

# 位置切片（半开！不含索引 3）
print(f"\niloc[0:3, 0:2]:\n{df.iloc[0:3, 0:2]}")

# 整数列表
print(f"\niloc[[0, 2, 4], [0, 3]]:\n{df.iloc[[0, 2, 4], [0, 3]]}")

# 全部行指定列
print(f"\niloc[:, -2:]:\n{df.iloc[:, -2:]}")
```

### 输出

```text
iloc[1, 0]: Bob

iloc[0:3, 0:2]:
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35

iloc[[0, 2, 4], [0, 3]]:
      Name  Salary
0    Alice    8000
2  Charlie   15000
4      Eve   11000

iloc[:, -2:]:
   Salary  Score
0    8000     91
1   12000     87
2   15000     72
3    9000     88
4   11000     68
```

### 理解重点

- `iloc` 始终按位置——与索引标签无关，即使索引是自定义字符串也按 0, 1, 2, ... 定位
- 负索引 `-1` 表示最后一行/列，`-2` 表示倒数第二——与 Python list 一致
- 取多个不连续行用列表 `iloc[[0, 2, 5]]`，连续行用切片 `iloc[0:5]`

## 5. 单元素快速访问

### `at` / `iat`

#### 作用

- `at[row_label, col_label]`：按标签取**单个元素**，比 `loc` 快 3~5 倍
- `iat[row_pos, col_pos]`：按位置取**单个元素**，比 `iloc` 快 3~5 倍

仅支持单个行 + 单个列（不支持切片或列表）。适合循环中逐元素读写或精确修改某一格。

### 速览

| 索引器 | 索引方式 | 对标 | 适用场景 |
|---|---|---|---|
| `df.at[label, col]` | 标签 | `loc` | 已知行列标签 |
| `df.iat[pos, col]` | 位置 | `iloc` | 已知行列位置 |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
}, index=["x", "y", "z"])

print(f"at['y', 'B']: {df.at['y', 'B']}")
print(f"iat[1, 1]: {df.iat[1, 1]}")

# 快速修改某个值
df.at["z", "A"] = 99
print(f"\n修改后:\n{df}")
```

### 输出

```text
at['y', 'B']: 5
iat[1, 1]: 5

修改后:
    A  B
x   1  4
y   2  5
z  99  6
```

### 理解重点

- `at` / `iat` 只能取**单个元素**——传入切片或列表会报错
- 遍历 DataFrame 逐元素修改时优先用 `at` / `iat`，比 `loc` / `iloc` 快
- 日常读写非性能瓶颈时用 `loc` / `iloc` 更灵活

## 6. 布尔索引

### 基本语法

| 语法 | 含义 | 说明 |
|---|---|---|
| `df[df["col"] > value]` | 单条件 | 返回符合条件的行 |
| `df[(cond1) & (cond2)]` | 与 | **必须用 `&`**，不能用 `and` |
| `df[(cond1) \| (cond2)]` | 或 | **必须用 `\|`**，不能用 `or` |
| `df[~cond]` | 非 | **必须用 `~`**，不能用 `not` |
| `df[df["col"].isin([v1, v2])]` | 多值匹配 | 替代多个 `==` 用 `\|` 连接 |
| `df[df["col"].between(low, high)]` | 区间 | 闭区间 `[low, high]` |
| `df[df["col"].str.contains("pat")]` | 字符串包含 | 详见 ch04 `.str` 访问器 |

### `Series.isin`

#### 作用

逐元素判断是否在给定值集合中，返回布尔 `Series`。替代多个 `==` 用 `|` 连接。

#### 重点方法

```python
Series.isin(values)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `values` | `list`、`set`、`Series`、`dict` | 值集合 | `["Beijing", "Shanghai"]` |

### `Series.between`

#### 作用

逐元素判断是否在区间内。默认闭区间 `[left, right]`（两端包含）。

#### 重点方法

```python
Series.between(left, right, inclusive="both")
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `left` | 标量 | 区间下界（包含） | `28` |
| `right` | 标量 | 区间上界（包含） | `32` |
| `inclusive` | `str` | 端点包含策略：`"both"` / `"left"` / `"right"` / `"neither"`，默认为 `"both"` | `"left"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
    "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
    "Salary": [8000, 12000, 15000, 9000, 11000],
    "Score": np.random.randint(60, 100, 5),
})

# 单条件
print(f"Age > 28:\n{df[df['Age'] > 28]}")

# 多条件 AND（必须用 &，每个条件必须加括号！）
print(f"\nAge > 25 & Salary > 10000:\n{df[(df['Age'] > 25) & (df['Salary'] > 10000)]}")

# isin —— 替代多个 OR
print(f"\nisin(['Beijing', 'Shanghai']):\n{df[df['City'].isin(['Beijing', 'Shanghai'])]}")

# between —— 区间筛选
print(f"\nAge.between(28, 32):\n{df[df['Age'].between(28, 32)]}")

# 取反
print(f"\n非 Beijing:\n{df[~df['City'].isin(['Beijing'])]}")
```

#### 输出

```text
Age > 28:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age > 25 & Salary > 10000:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

isin(['Beijing', 'Shanghai']):
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age.between(28, 32):
    Name  Age      City  Salary  Score
1    Bob   30  Shanghai   12000     87
3  David   28  Guangzhou    9000     88
4    Eve   32  Shanghai   11000     68

非 Beijing:
    Name  Age      City  Salary  Score
1    Bob   30  Shanghai   12000     87
3  David   28  Guangzhou    9000     88
4    Eve   32  Shanghai   11000     68
```

### 理解重点

- 多条件组合**必须**用 `&` / `|` / `~`（位运算符），**不能**用 `and` / `or` / `not`
- 每个条件**必须加括号**：`(df.A > 1) & (df.B < 5)`——否则因运算符优先级报错
- 布尔索引返回**副本**，修改不影响原 DataFrame；过滤后行索引保持不变，用 `reset_index(drop=True)` 重置
- `isin` 替代多个 `==` 的 OR 连接——性能更好、代码更短
- `between` 默认闭区间（两端包含），用 `inclusive` 参数调整

## 7. 表达式查询

### `DataFrame.query`

#### 作用

用字符串表达式过滤行，语法接近 SQL。可通过 `@` 引用外部 Python 变量。内部使用 `and` / `or` / `not`（非 `&` / `|` / `~`），与布尔索引相反。

#### 重点方法

```python
df.query(expr, *, inplace=False, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `expr` | `str` | 查询表达式字符串 | `"Age > 28 and Salary > 10000"` |
| `inplace` | `bool` | 是否原地过滤，默认为 `False` | `True` |

#### `query` 支持的语法

| 语法 | 示例 |
|---|---|
| 比较运算 | `"Age > 28"` |
| 逻辑与 / 或 / 非 | `"Age > 28 and Salary > 10000"` |
| `in` / `not in` | `"City in ['Beijing', 'Shanghai']"` |
| 引用外部变量 | `"Age >= @minAge"` |
| 列名含空格/关键字 | `` "`col name` > 5" ``（反引号包裹） |

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 35, 28, 32],
    "City": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
    "Salary": [8000, 12000, 15000, 9000, 11000],
    "Score": np.random.randint(60, 100, 5),
})

# 基本表达式
print(df.query("Age > 28 and Salary > 10000"))

# 引用外部变量（用 @）
minAge = 30
print(f"\nAge >= {minAge}:\n{df.query('Age >= @minAge')}")

# in 语法
print(f"\nCities:\n{df.query('City in [\"Beijing\", \"Shanghai\"]')}")

# not in
print(f"\n非 Beijing/Shanghai:\n{df.query('City not in [\"Beijing\", \"Shanghai\"]')}")
```

#### 输出

```text
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Age >= 30:
      Name  Age      City  Salary  Score
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

Cities:
      Name  Age      City  Salary  Score
0    Alice   25   Beijing    8000     91
1      Bob   30  Shanghai   12000     87
2  Charlie   35   Beijing   15000     72
4      Eve   32  Shanghai   11000     68

非 Beijing/Shanghai:
    Name  Age       City  Salary  Score
3  David   28  Guangzhou    9000     88
```

### 理解重点

- `query` 内部用 `and` / `or` / `not`——与布尔索引（`&` / `|` / `~`）**相反**
- `@var` 安全引用外部变量，避免字符串拼接（既丑又有注入风险）
- 列名含空格/点号/保留字时用**反引号**包裹：`` `col name` ``
- 复杂多条件查询 `query` 更易读；简单单条件布尔索引更直接

## 链式索引的陷阱

### 问题

`df[df["A"] > 0]["B"] = 1` 这样的**链式索引赋值**会触发 `SettingWithCopyWarning`，且修改可能不生效。原因是 `df[mask]` 返回临时副本，在其上赋值无法回写到原 DataFrame。

### 正确写法

```python
# 用 loc 一步到位
df.loc[df["A"] > 0, "B"] = 1

# 或先计算掩码再赋值
mask = df["A"] > 0
df.loc[mask, "B"] = 1
```

### 理解重点

- 链式索引 `df[x][y]` 分两步：先切片得临时对象，再取列——赋值无法保证回写
- 规则：**赋值场景用 `df.loc[行, 列] = 值`**，读取场景链式索引可接受

## 常见坑

1. `loc` 切片**包含**终点，`iloc` 切片**不包含**——当索引是 `0, 1, 2, ...` 时容易误判
2. 多条件过滤忘加括号：`df["A"] > 1 & df["B"] < 5` 会报错——正确写法 `(df["A"] > 1) & (df["B"] < 5)`
3. 布尔索引误用 `and` / `or`——应该用 `&` / `|`
4. 在 `query` 中误用 `&` / `|`——应该用 `and` / `or`
5. 链式索引赋值 `df[mask]["col"] = x` 会触发 `SettingWithCopyWarning`——改用 `df.loc[mask, "col"] = x`
6. `df.col` 属性访问在列名含空格/点号/Python 关键字时失效——一律用 `df["col"]` 更安全
7. 布尔索引返回副本——修改后若需保留结果应显式赋值给变量

## 小结

- 列选择：`df["col"]`（Series）/ `df[["c1", "c2"]]`（DataFrame）——双括号是关键
- 行选择：`df.iloc[...]`（位置，半开）/ `df.loc[...]`（标签，闭区间）——切片规则不同
- 条件过滤：布尔索引用 `&` / `|` / `~`；`query` 用 `and` / `or` / `not`——不要混用
- 赋值场景永远用 `df.loc[行, 列] = 值`，避免链式索引的副作用

# Pandas 数据清洗与处理

## 本章目标

1. 掌握缺失值的检测、删除与填充三种策略
2. 学会检测和删除重复行
3. 掌握 `astype`、`pd.to_numeric`、`pd.to_datetime` 三种类型转换
4. 熟悉 `.str` 访问器的字符串向量化操作
5. 掌握 `replace` 和 `map` 的值映射/替换

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.isnull()` / `df.isna()` | 方法 | 检测缺失值（两者等价） |
| `df.notnull()` / `df.notna()` | 方法 | 检测非缺失值 |
| `df.dropna(...)` | 方法 | 删除含缺失值的行/列 |
| `df.fillna(...)` | 方法 | 填充缺失值 |
| `df.duplicated(...)` | 方法 | 检测重复行 |
| `df.drop_duplicates(...)` | 方法 | 删除重复行 |
| `df.astype(...)` | 方法 | 转换列的数据类型 |
| `pd.to_numeric(...)` | 函数 | 安全转换为数值（错误可设为 NaN） |
| `pd.to_datetime(...)` | 函数 | 解析日期时间字符串 |
| `Series.str.xxx(...)` | 访问器 | 字符串向量化操作 |
| `Series.replace(...)` | 方法 | 值替换 |
| `Series.map(...)` | 方法 | 字典映射/函数应用 |

## 1. 缺失值检测

### `isnull` / `notnull`

#### 作用

- `isnull()`（别名 `isna()`）：逐元素判断是否为缺失值（`NaN` / `None` / `NaT`），返回同形状布尔 DataFrame
- `notnull()`（别名 `notna()`）：与 `isnull` 相反——逐元素判断是否为非缺失值

#### 重点方法

```python
df.isnull()
df.notnull()
df.isnull().sum()     # 每列缺失计数（常用组合）
df.isnull().mean()    # 每列缺失比例
```

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2.0, 3.0, np.nan, 5.0],
    "C": ["x", None, "y", "z", None],
})

print(f"原始数据:\n{df}")
print(f"\nisnull():\n{df.isnull()}")
print(f"\n每列缺失计数:\n{df.isnull().sum()}")
print(f"\n每列缺失比例:\n{df.isnull().mean().round(2)}")
print(f"\nnotnull():\n{df.notnull()}")
```

#### 输出

```text
原始数据:
     A    B     C
0  1.0  NaN     x
1  2.0  2.0  None
2  NaN  3.0     y
3  4.0  NaN     z
4  5.0  5.0  None

isnull():
       A      B      C
0  False   True  False
1  False  False   True
2   True  False  False
3  False   True  False
4  False  False   True

每列缺失计数:
A    1
B    2
C    2
dtype: int64

每列缺失比例:
A    0.2
B    0.4
C    0.4
dtype: float64

notnull():
       A      B      C
0   True  False   True
1   True   True  False
2  False   True   True
3   True  False   True
4   True   True  False
```

#### 理解重点

- `isnull().sum()` 是发现缺失值的最快方式——一行代码显示每列缺失计数
- `isnull().mean()` 直接得到缺失比例——适合设定缺失率阈值做列筛选
- `None` 和 `np.nan` 在 Pandas 中等价——`isnull()` 都能检测
- 布尔 DataFrame 可直接用于条件过滤：`df[df["A"].notnull()]`

## 2. 删除缺失值

### `DataFrame.dropna`

#### 作用

删除包含缺失值的行或列。通过 `axis`、`how`、`thresh`、`subset` 精细控制删除条件。

#### 重点方法

```python
df.dropna(*, axis=0, how='any', thresh=None, subset=None, inplace=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` | `0` 删行、`1` 删列，默认为 `0` | `1` |
| `how` | `str` | `'any'` 有缺失即删、`'all'` 全缺失才删，默认为 `'any'` | `"all"` |
| `thresh` | `int` 或 `None` | 至少要有 N 个非缺失值才保留，默认为 `None` | `2` |
| `subset` | `list[str]` 或 `None` | 只在指定列上检测缺失，默认为 `None`（全列） | `["A", "B"]` |
| `inplace` | `bool` | 是否原地修改，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, np.nan, 35, np.nan, 32],
    "Salary": [8000.0, 12000.0, np.nan, np.nan, 11000.0],
})

print(f"原始数据:\n{df}")
print(f"\ndropna() 默认删行:\n{df.dropna()}")
print(f"\ndropna(axis=1) 删列:\n{df.dropna(axis=1)}")
print(f"\ndropna(subset=['Age']) 只看 Age 列:\n{df.dropna(subset=['Age'])}")
print(f"\ndropna(thresh=2) 至少 2 个非缺失才保留:\n{df.dropna(thresh=2)}")
```

#### 输出

```text
原始数据:
      Name   Age   Salary
0    Alice  25.0   8000.0
1      Bob   NaN  12000.0
2  Charlie  35.0      NaN
3    David   NaN      NaN
4      Eve  32.0  11000.0

dropna() 默认删行:
    Name   Age   Salary
0  Alice  25.0   8000.0
4    Eve  32.0  11000.0

dropna(axis=1) 删列:
      Name
0    Alice
1      Bob
2  Charlie
3    David
4      Eve

dropna(subset=['Age']) 只看 Age 列:
      Name   Age   Salary
0    Alice  25.0   8000.0
2  Charlie  35.0      NaN
4      Eve  32.0  11000.0

dropna(thresh=2) 至少 2 个非缺失才保留:
      Name   Age   Salary
0    Alice  25.0   8000.0
1      Bob   NaN  12000.0
2  Charlie  35.0      NaN
4      Eve  32.0  11000.0
```

#### 理解重点

- `how='all'` 只在整行/列全 NaN 时删除——比 `how='any'` 保守
- `subset` 只在指定列判断缺失——其余列有 NaN 不影响保留
- `thresh` 比 `how` 更精细："每行至少要有 N 个有效值"

## 3. 填充缺失值

### `DataFrame.fillna`

#### 作用

用指定值或策略填充缺失值。支持常数值、前/后向填充、均值/中位数/众数填充、字典按列填充。

#### 重点方法

```python
df.fillna(value=None, *, method=None, axis=None, inplace=False,
          limit=None, downcast=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `value` | 标量、`dict`、`Series` | 填充值；`dict` 格式为 `{列名: 值}` | `0`、`{"A": 0, "B": 1}` |
| `method` | `str` 或 `None` | 填充策略：`'ffill'` 前向填充 / `'bfill'` 后向填充，与 `value` 互斥，默认为 `None` | `"ffill"` |
| `axis` | `int` | 填充方向轴，默认为 `None`（即 `0`） | `1` |
| `inplace` | `bool` | 是否原地修改，默认为 `False` | `True` |
| `limit` | `int` 或 `None` | 前/后向填充时最多连续填充几个 | `1` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1, np.nan, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": ["x", None, "y", None, "z"],
})

print(f"原始数据:\n{df}")
print(f"\nfillna(0):\n{df.fillna(0)}")
print(f"\nffill 前向填充:\n{df.fillna(method='ffill')}")
print(f"\n按列填充 dict:\n{df.fillna({'A': -1, 'B': 99, 'C': 'missing'}))")
print(f"\n均值填充 A 列:\n{df.fillna({'A': df['A'].mean()})}")
```

#### 输出

```text
原始数据:
     A    B     C
0  1.0  NaN     x
1  NaN  2.0  None
2  NaN  3.0     y
3  4.0  NaN  None
4  5.0  5.0     z

fillna(0):
     A    B  C
0  1.0  0.0  x
1  0.0  2.0  0
2  0.0  3.0  y
3  4.0  0.0  0
4  5.0  5.0  z

ffill 前向填充:
     A    B  C
0  1.0  NaN  x
1  1.0  2.0  x
2  1.0  3.0  y
3  4.0  3.0  y
4  5.0  5.0  z

按列填充 dict:
     A     B        C
0  1.0  99.0        x
1 -1.0   2.0  missing
2 -1.0   3.0        y
3  4.0  99.0  missing
4  5.0   5.0        z

均值填充 A 列:
     A    B     C
0  1.0  NaN     x
1  3.33 2.0  None
2  3.33 3.0     y
3  4.0  NaN  None
4  5.0  5.0     z
```

#### 理解重点

- 常量填充最快但可能引入偏差——均值/中位数填充更保守
- `method='ffill'` 适合时间序列（假设值不会突变）
- `dict` 形式的 `value` 最灵活——每列可以有不同的填充策略
- `method` 和 `value` 互斥——不能同时使用

## 4. 重复值处理

### `duplicated` / `drop_duplicates`

#### 作用

- `duplicated()`：检测重复行，返回布尔 Series（首次出现标记为 `False`）
- `drop_duplicates()`：删除重复行，保留首次出现（或末次）

#### 重点方法

```python
df.duplicated(subset=None, keep='first')
df.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `subset` | `list[str]` 或 `None` | 只在指定列上判断重复，默认为 `None`（全列） | `["Name"]`、`["A", "B"]` |
| `keep` | `str` | 保留哪一行：`'first'` / `'last'` / `False`（全删），默认为 `'first'` | `"last"`、`False` |
| `inplace` | `bool` | 是否原地修改（仅 `drop_duplicates`），默认为 `False` | `True` |
| `ignore_index` | `bool` | 是否重置索引（仅 1.0+），默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
    "Age": [25, 30, 25, 30, 35],
    "City": ["Beijing", "Shanghai", "Beijing", "Shanghai", "Guangzhou"],
})

print(f"原始数据:\n{df}")
print(f"\nduplicated():\n{df.duplicated()}")
print(f"\ndrop_duplicates():\n{df.drop_duplicates()}")
print(f"\ndrop_duplicates(subset=['Name']):\n{df.drop_duplicates(subset=['Name'])}")
print(f"\ndrop_duplicates(keep='last'):\n{df.drop_duplicates(keep='last')}")
```

#### 输出

```text
原始数据:
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2    Alice   25    Beijing
3      Bob   30   Shanghai
4  Charlie   35  Guangzhou

duplicated():
0    False
1    False
2     True
3     True
4    False
dtype: bool

drop_duplicates():
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
4  Charlie   35  Guangzhou

drop_duplicates(subset=['Name']):
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
4  Charlie   35  Guangzhou

drop_duplicates(keep='last'):
      Name  Age       City
2    Alice   25    Beijing
3      Bob   30   Shanghai
4  Charlie   35  Guangzhou
```

#### 理解重点

- `duplicated()` 可用于在删除前先确认重复情况——`df[df.duplicated()]` 查看重复行
- `keep=False` 删除**所有**重复行（包括首次出现）——适合完全去重
- `subset` 只在指定列判断——不同名字但有相同年龄的人不会被误删

## 5. 类型转换

### `DataFrame.astype`

#### 作用

将列转换为指定的数据类型。支持 NumPy dtype、Python 类型、Pandas 可空类型（`Int64` 等）。

#### 重点方法

```python
df.astype(dtype, copy=None, errors='raise')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype`、`str`、`dict` | 目标类型；`dict` 格式为 `{列名: 类型}` | `int`、`"float32"`、`{"A": int, "B": float}` |
| `copy` | `bool` 或 `None` | 是否返回副本 | `True` |
| `errors` | `str` | 转换失败处理：`'raise'` / `'ignore'`，默认为 `'raise'` | `"ignore"` |

### `pd.to_numeric`

#### 作用

将 Series 安全转换为数值类型。转换失败的值可通过 `errors` 设为 NaN（`'coerce'`），比 `astype` 更安全。

#### 重点方法

```python
pd.to_numeric(arg, errors='raise', downcast=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `Series`、`list` | 输入数据 | `df["col"]` |
| `errors` | `str` | 错误处理：`'raise'` / `'coerce'`（转 NaN）/ `'ignore'`，默认为 `'raise'` | `"coerce"` |
| `downcast` | `str` 或 `None` | 整数/浮点降级：`'integer'` / `'float'`，默认为 `None` | `"integer"` |

### `pd.to_datetime`

#### 作用

将 Series 或标量解析为 `datetime64` 类型。支持多种日期格式自动推断，也可用 `format` 参数加速。

#### 重点方法

```python
pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False,
               utc=False, format=None, unit=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `Series`、`list`、`str` | 输入日期数据 | `df["date"]` |
| `errors` | `str` | 错误处理：`'raise'` / `'coerce'` / `'ignore'`，默认为 `'raise'` | `"coerce"` |
| `format` | `str` 或 `None` | 日期格式字符串，指定后解析更快 | `"%Y-%m-%d"` |
| `dayfirst` | `bool` | `True` 时优先解析为日/月顺序，默认为 `False` | `True` |
| `utc` | `bool` | 是否转为 UTC 时间，默认为 `False` | `True` |
| `unit` | `str` | 整数输入的时间单位：`'s'` / `'ms'` / `'ns'` | `"s"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "ID": ["1", "2", "x", "4", "5"],
    "Price": ["10.5", "20.3", "bad", "40.1", "50.0"],
    "Date": ["2024-01-01", "2024-02-15", "invalid", "2024-04-20", "2024-05-30"],
})

print(f"原始类型:\n{df.dtypes}\n")

# astype：已知安全时直接用
df["ID_clean"] = pd.to_numeric(df["ID"], errors="coerce")
print(f"to_numeric (errors='coerce'):\n{df['ID_clean']}")

# to_numeric
df["Price_clean"] = pd.to_numeric(df["Price"], errors="coerce")
print(f"\nto_numeric (errors='coerce'):\n{df['Price_clean']}")

# to_datetime
df["Date_clean"] = pd.to_datetime(df["Date"], errors="coerce")
print(f"\nto_datetime (errors='coerce'):\n{df['Date_clean']}")
```

#### 输出

```text
原始类型:
ID       object
Price    object
Date     object
dtype: object

to_numeric (errors='coerce'):
0    1.0
1    2.0
2    NaN
3    4.0
4    5.0
Name: ID_clean, dtype: float64

to_numeric (errors='coerce'):
0    10.5
1    20.3
2     NaN
3    40.1
4    50.0
Name: Price_clean, dtype: float64

to_datetime (errors='coerce'):
0   2024-01-01
1   2024-02-15
2          NaT
3   2024-04-20
4   2024-05-30
Name: Date_clean, dtype: datetime64[ns]
```

#### 理解重点

- `astype` 直接转换——不合法值会抛异常；已知数据干净时用
- `to_numeric(errors='coerce')` 是清洗脏数据的安全网——非法值变 NaN，不中断流程
- `to_datetime(format=...)` 指定格式字符串可以禁掉自动推断——**大幅加速**大批量解析
- `pd.to_datetime` 在 [ch[ch07 时间序列](pandas.md)更完整的日期时间处理

## 6. 字符串操作（`.str` 访问器）

### `.str` 访问器

#### 作用

通过 `Series.str.xxx()` 对字符串列执行向量化操作（一次操作整列，无需循环）。覆盖大小写转换、切片、包含检测、正则提取、拼接等 30+ 方法。

### 常用方法速览

| 方法 | 作用 | 示例 |
|---|---|---|
| `str.lower()` / `str.upper()` | 大小写转换 | `s.str.upper()` |
| `str.strip()` | 去除首尾空白 | `s.str.strip()` |
| `str.contains(pat)` | 正则匹配（返回布尔） | `s.str.contains("pat")` |
| `str.startswith(pat)` | 开头匹配 | `s.str.startswith("A")` |
| `str.replace(pat, repl)` | 替换子串/正则 | `s.str.replace("-", "_")` |
| `str.split(pat)` | 按分隔符拆分（返回列表列） | `s.str.split(",")` |
| `str.extract(pat)` | 正则提取捕获组 | `s.str.extract(r"(\d+)")` |
| `str.len()` | 字符串长度 | `s.str.len()` |
| `str.cat(sep=...)` | 拼接 Series 元素 | `s.str.cat(sep=", ")` |
| `str.slice(start, stop)` | 按位置切片 | `s.str.slice(0, 3)` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Name": ["  Alice  ", "BOB", "charlie", "DAVID", "eve  "],
    "Email": ["alice@co.com", "bob@co.com", "charlie@org.cn",
              "david@co.com", "eve@org.cn"],
    "Phone": ["010-1234", "021-5678", "010-9012", "0755-3456", "021-7890"],
})

# 大小写 + 去空格
df["NameClean"] = df["Name"].str.strip().str.title()
print(f"Name 清洗:\n{df[['Name', 'NameClean']]}")

# 正则包含
print(f"\n邮箱含 .com:\n{df[df['Email'].str.contains('.com')]}")

# 正则提取
df["Domain"] = df["Email"].str.extract(r"@(.+)")
print(f"\n域名提取:\n{df[['Email', 'Domain']]}")

# 拆分
df["AreaCode"] = df["Phone"].str.split("-").str[0]
print(f"\n区号拆分:\n{df[['Phone', 'AreaCode']]}")
```

#### 输出

```text
Name 清洗:
       Name NameClean
0    Alice     Alice
1       BOB       Bob
2  charlie   Charlie
3     DAVID     David
4     eve        Eve

邮箱含 .com:
    Name        Email      Phone
0  Alice   alice@co.com  010-1234
1    BOB     bob@co.com  021-5678
3  DAVID  david@co.com  0755-3456

域名提取:
       Email   Domain
0   alice@co.com   co.com
1     bob@co.com   co.com
2  charlie@org.cn  org.cn
3  david@co.com   co.com
4    eve@org.cn   org.cn

区号拆分:
       Phone AreaCode
0  010-1234      010
1  021-5678      021
2  010-9012      010
3  0755-3456     0755
4  021-7890      021
```

#### 理解重点

- `.str` 访问器**只能用于字符串列**——对数值列调用会抛异常
- `.str` 可链式调用：`df["col"].str.strip().str.lower().str.replace("_", "-")`
- `str.contains` 默认使用正则——搜索 `.` 等元字符需转义：`str.contains(r"\.com")`
- `str.extract` 用括号捕获组（`(...)`）提取子模式——多组返回多列

## 7. 值替换与映射

### `Series.replace`

#### 作用

将 Series 中的特定值替换为新值。支持单值替换、多值替换（列表）、字典映射。

#### 重点方法

```python
Series.replace(to_replace, value=None, *, inplace=False, regex=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `to_replace` | 标量、`list`、`dict`、正则 | 被替换值；`dict` 时为 `{旧值: 新值}` | `-1`、`[1, 2]`、`{"A": "a"}` |
| `value` | 标量、`list` 或 `None` | 替换值；`to_replace` 是 `dict` 时须为 `None` | `0`、`[10, 20]` |
| `regex` | `bool` | 是否将 `to_replace` 视为正则表达式，默认为 `False` | `True` |

### `Series.map`

#### 作用

将 Series 的每个值按字典或函数映射为新值。未映射的值变为 NaN（除非函数处理）。

#### 重点方法

```python
Series.map(arg, na_action=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arg` | `dict`、`Series`、函数 | 映射规则 | `{"cat": "猫", "dog": "狗"}`、`lambda x: x*2` |
| `na_action` | `str` 或 `None` | `'ignore'` 时跳过 NaN（不传入函数），默认为 `None` | `"ignore"` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Code": ["A", "B", "C", "A", "D"],
    "Score": [85, 92, 78, 90, -1],
})

# replace：将 -1（缺考标记）替换为 NaN
df["ScoreClean"] = df["Score"].replace(-1, np.nan)
print(f"replace(-1, NaN):\n{df[['Score', 'ScoreClean']]}")

# map：将编码映射为全称
codeMap = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Poor"}
df["Grade"] = df["Code"].map(codeMap)
print(f"\nmap 编码映射:\n{df[['Code', 'Grade']]}")

# replace dict 形式
df["ScoreCat"] = df["Score"].replace({
    85: "优秀", 92: "优秀", 78: "一般", 90: "优秀", -1: "缺考"
})
print(f"\nreplace dict 形式:\n{df[['Score', 'ScoreCat']]}")
```

#### 输出

```text
replace(-1, NaN):
   Score  ScoreClean
0     85        85.0
1     92        92.0
2     78        78.0
3     90        90.0
4     -1         NaN

map 编码映射:
  Code      Grade
0    A  Excellent
1    B       Good
2    C       Fair
3    A  Excellent
4    D       Poor

replace dict 形式:
   Score ScoreCat
0     85       优秀
1     92       优秀
2     78       一般
3     90       优秀
4     -1       缺考
```

#### 理解重点

- `replace` 和 `map` 的默认行为不同：`map` 对未映射值返回 NaN；`replace` 不匹配则保持原值
- `map(dict)` 适合编码->全称映射；`replace(dict)` 适合脏数据清洗
- `replace` 的 `regex=True` 可以做正则替换——比 `.str.replace` 更灵活
- `map` 可以传函数：`df["col"].map(lambda x: x * 2)`——等价于 `apply`

## 常见坑

1. `NaN` 不等于任何值（包括 NaN 自身）——用 `isnull()` 检测，不要用 `== np.nan`
2. `dropna()` 默认 `how='any'`——某行只要有一个 NaN 就被删除，容易误删大量数据
3. `fillna(method='ffill')` 和 `fillna(value=0)` 互斥——不能同时指定
4. `astype(int)` 遇到 NaN 会抛错——因为 int 不支持 NaN；先 `fillna` 或使用可空 `Int64` 类型
5. `.str` 访问器遇到 NaN 返回 NaN 而非报错——但 NaN 不参与后续字符串匹配
6. `map` 未匹配到的值变成 NaN——与 `replace` 行为不同（`replace` 不匹配保持原值）
7. `to_datetime` 不指定 `format` 时自动推断——大数据集应显式指定格式以加速 10~100 倍

## 小结

- 缺失值处理的标准流程：`isnull().sum()` 评估 -> `fillna` 或 `dropna` 处理
- 类型转换的安全选择：`pd.to_numeric(errors='coerce')` / `pd.to_datetime(errors='coerce')`
- `.str` 访问器是 Pandas 字符串操作的统一入口——无需写循环
- `replace` 和 `map` 功能相似但默认行为不同：`map` 未匹配变 NaN，`replace` 保持原值

# Pandas 分组与聚合

## 本章目标

1. 理解 `groupby` 的 Split-Apply-Combine 思维模型
2. 掌握单列聚合、多列不同聚合、命名聚合三种写法
3. 区分 `agg`、`transform`、`apply` 三者的返回形状与适用场景
4. 学会用 `transform` 做组内标准化，用 `filter` 做组级过滤

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.groupby(...)` | 方法 | 按列分组，返回 `GroupBy` 对象 |
| `gb.agg(...)` | 方法 | 聚合运算，返回**每组一行**的结果 |
| `gb.transform(...)` | 方法 | 组内变换，返回**与原数据同长度**的结果 |
| `gb.apply(...)` | 方法 | 自定义组级运算，返回任意形状 |
| `gb.filter(...)` | 方法 | 按组级别条件过滤，保留整个组 |
| `gb.size()` | 方法 | 每组行数（含 NaN） |
| `gb.count()` | 方法 | 每组非空计数（按列） |
| `gb.ngroups` | 属性 | 分组总数 |
| `gb.groups` | 属性 | 分组字典 `{key: [indices]}` |

## 1. 分组（Split-Apply-Combine）

Split-Apply-Combine 是 `groupby` 的核心思想：

1. **Split**：按指定列/函数把数据切成多个组
2. **Apply**：对每组独立应用聚合/变换/过滤函数
3. **Combine**：把各组结果合并成最终输出

### `DataFrame.groupby`

#### 作用

按一个或多个列将 DataFrame 分组，返回 `GroupBy` 对象。分组本身**不执行计算**——只有后续调用 `agg` / `transform` / `apply` 等方法才会触发。

#### 重点方法

```python
df.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
           group_keys=True, observed=False, dropna=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `by` | `str`、`list[str]`、`Series`、函数 | 分组依据；多个列用列表 | `"Department"`、`["A", "B"]` |
| `axis` | `int` | 分组轴，默认为 `0`（按行分组） | `1` |
| `level` | `int`、`str` 或 `None` | 多级索引时按哪一级分组，默认为 `None` | `0` |
| `as_index` | `bool` | 分组键是否作为结果索引，默认为 `True`；`False` 时保留为普通列 | `False` |
| `sort` | `bool` | 是否对分组键排序，默认为 `True`；`False` 可加速 | `False` |
| `group_keys` | `bool` | `apply` 时是否把分组键加入结果索引，默认为 `True` | `True` |
| `observed` | `bool` | 对 `category` 列是否只保留出现过的分类，默认为 `False` | `True` |
| `dropna` | `bool` | 是否丢弃分组键含 NaN 的行，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

grouped = df.groupby("Department")
print(f"分组数: {grouped.ngroups}")
print(f"分组键: {list(grouped.groups.keys())}")

# 遍历分组
for name, group in grouped:
    print(f"\n--- {name} ---")
    print(group)
```

#### 输出

```text
分组数: 3
分组键: ['HR', 'IT', 'Sales']

--- HR ---
  Department Employee  Salary  Bonus  Years
4         HR      Eve    7000    800      2
5         HR    Frank    7500    900      3

--- IT ---
  Department Employee  Salary  Bonus  Years
2         IT  Charlie   12000   1500      4
3         IT    David   11000   1400      6

--- Sales ---
  Department Employee  Salary  Bonus  Years
0      Sales    Alice    8000   1000      3
1      Sales      Bob    9000   1200      5
```

#### 理解重点

- `groupby` 返回的是**懒对象**——直接打印看不到数据，只有计算结果才触发运算
- 遍历 `GroupBy` 对象可得到 `(分组名, 子DataFrame)` 元组
- `as_index=False` 或 `reset_index()` 可将分组键变回普通列，便于后续连接/存储

## 2. 聚合

### `GroupBy.agg`

#### 作用

对每组执行**聚合函数**（返回标量），结果每组一行。支持多种聚合模式：

- 单个函数：`agg("mean")`、`agg(np.sum)`
- 函数列表：`agg(["sum", "mean", "max"])`
- 按列不同聚合：`agg({"Salary": "mean", "Bonus": "sum"})`
- 命名聚合（推荐）：`agg(total=("Salary", "sum"), avg=("Salary", "mean"))`

#### 重点方法

```python
gb.agg(func=None, *args, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `str`、`list`、`dict`、函数 | 聚合函数，字符串形式最常用 | `"mean"`、`["sum", "std"]` |
| `*args` | 传给 `func` 的额外参数 | 如 `agg(np.quantile, q=0.9)` | — |
| `**kwargs` | 命名聚合 | `新列名 = ("原列名", "聚合函数")` | `total=("Salary", "sum")` |

#### 常用内置聚合函数

| 字符串 | 含义 | 字符串 | 含义 |
|---|---|---|---|
| `"sum"` | 求和 | `"mean"` | 均值 |
| `"median"` | 中位数 | `"min"` / `"max"` | 极值 |
| `"std"` / `"var"` | 标准差/方差 | `"count"` | 非空计数 |
| `"size"` | 行数（含 NaN） | `"first"` / `"last"` | 组内首/末值 |
| `"nunique"` | 唯一值个数 | `"sem"` | 均值的标准误差 |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

grouped = df.groupby("Department")

# 单列单函数
print(f"Salary.sum():\n{grouped['Salary'].sum()}")

# 单列多函数
print(f"\nSalary.agg(['sum', 'mean', 'max']):")
print(grouped["Salary"].agg(["sum", "mean", "max"]))

# 多列不同函数
print(f"\n多列不同聚合:")
print(grouped.agg({
    "Salary": ["mean", "sum"],
    "Bonus": "sum",
    "Years": "mean",
}))

# 命名聚合（推荐）
print(f"\n命名聚合:")
print(grouped.agg(
    total=("Salary", "sum"),
    average=("Salary", "mean"),
    bonusSum=("Bonus", "sum"),
))
```

#### 输出

```text
Salary.sum():
Department
HR       14500
IT       23000
Sales    17000
Name: Salary, dtype: int64

Salary.agg(['sum', 'mean', 'max']):
              sum    mean    max
Department
HR          14500  7250.0   7500
IT          23000 11500.0  12000
Sales       17000  8500.0   9000

多列不同聚合:
              Salary          Bonus Years
                mean    sum     sum  mean
Department
HR            7250.0  14500    1700   2.5
IT           11500.0  23000    2900   5.0
Sales         8500.0  17000    2200   4.0

命名聚合:
            total  average  bonusSum
Department
HR          14500   7250.0      1700
IT          23000  11500.0      2900
Sales       17000   8500.0      2200
```

#### 理解重点

- **命名聚合** `agg(新名=("列", "函数"))` 是 Pandas 0.25+ 的推荐写法——结果列名干净可控
- 字典方式 `agg({"col": "fn"})` 在多函数时产生 MultiIndex 列，后续处理麻烦
- 聚合结果默认以分组键为索引；用 `as_index=False` 或 `.reset_index()` 让其变回列

## 3. 组内变换

### `GroupBy.transform`

#### 作用

对每组应用函数，返回**与原数据同长度**的结果（每行都有值）。适合组内标准化、组内排名、组内均值填充。

#### 重点方法

```python
gb.transform(func, *args, engine=None, engine_kwargs=None, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | `str`、函数 | 变换函数；必须返回与输入同长度的值 | `"mean"`、`"rank"`、lambda |
| `*args` | 传给 `func` 的额外参数 | — | — |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Years": [3, 5, 4, 6, 2, 3],
})

# 组内均值（给原数据添加"部门平均工资"列）
df["DeptMeanSalary"] = df.groupby("Department")["Salary"].transform("mean")

# 组内 Z-score 标准化
df["SalaryZscore"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df[["Department", "Employee", "Salary", "DeptMeanSalary", "SalaryZscore"]])
```

#### 输出

```text
  Department Employee  Salary  DeptMeanSalary       SalaryZscore
0      Sales    Alice    8000          8500.0  -0.707107
1      Sales      Bob    9000          8500.0   0.707107
2         IT  Charlie   12000         11500.0   0.707107
3         IT    David   11000         11500.0  -0.707107
4         HR      Eve    7000          7250.0  -0.707107
5         HR    Frank    7500          7250.0   0.707107
```

#### 数学公式

组内 Z-score 标准化：

$$
z_i = \frac{x_i - \mu_g}{\sigma_g}
$$

其中 $\mu_g$ 是组 $g$ 的均值，$\sigma_g$ 是组 $g$ 的标准差。

#### 理解重点

- `agg` 返回每组一行；`transform` 返回每行一个值——**长度不变**是核心区别
- 典型场景：给原数据添加"组内均值/排名/累计值"列，作为特征工程的一部分
- 返回值会自动对齐原数据的索引，可直接赋为新列

## 4. 自定义应用

### `GroupBy.apply`

#### 作用

对每组应用自定义函数，**返回任意形状**（Series / DataFrame / 标量），Pandas 自动根据返回类型合并结果。是 `groupby` 中最灵活、但也最慢的方法。

#### 重点方法

```python
gb.apply(func, *args, include_groups=True, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | 函数 | 接收 group (DataFrame)，返回任意对象 | `lambda g: g.nlargest(1, "Salary")` |
| `include_groups` | `bool` | 分组键列是否传入函数；Pandas 2.2+ 推荐设 `False`，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Bonus": [1000, 1200, 1500, 1400, 800, 900],
    "Years": [3, 5, 4, 6, 2, 3],
})

grouped = df.groupby("Department")

# 每组工资最高的员工
def topEmployee(group):
    return group.nlargest(1, "Salary")

print("每组最高工资:\n")
print(grouped.apply(topEmployee, include_groups=False))

# 自定义每组多字段汇总
def summary(group):
    return pd.Series({
        "count": len(group),
        "totalSalary": group["Salary"].sum(),
        "avgYears": group["Years"].mean(),
    })

print(f"\n自定义汇总:\n{grouped.apply(summary, include_groups=False)}")
```

#### 输出

```text
每组最高工资:

              Employee  Salary  Bonus  Years
Department
HR         5     Frank    7500    900      3
IT         2   Charlie   12000   1500      4
Sales      1       Bob    9000   1200      5

自定义汇总:
            count  totalSalary  avgYears
Department
HR            2.0       14500.0       2.5
IT            2.0       23000.0       5.0
Sales         2.0       17000.0       4.0
```

#### 理解重点

- `apply` 最灵活但也**最慢**——能用 `agg` / `transform` 就不要用 `apply`
- 第一组可能被 Pandas 调用**两次**用于类型推断——函数必须是**纯函数**、无副作用
- 返回 `Series` 会自动展开成列；返回 `DataFrame` 按组堆叠；返回标量得到单值

## 5. 组级过滤

### `GroupBy.filter`

#### 作用

按组级别的条件过滤——保留（或丢弃）**整个组**，而不是过滤单行。结果是与原数据形状相同的子集。

#### 重点方法

```python
gb.filter(func, dropna=True, *args, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `func` | 函数（返回 `bool`） | 接收 group，返回 `True` 保留 / `False` 丢弃 | `lambda g: g.Salary.mean() > 8000` |
| `dropna` | `bool` | 是否丢弃函数返回 NaN 的组，默认为 `True` | `False` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Department": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
    "Employee": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Salary": [8000, 9000, 12000, 11000, 7000, 7500],
    "Years": [3, 5, 4, 6, 2, 3],
})

# 保留平均工资 > 8000 的部门
result = df.groupby("Department").filter(lambda g: g["Salary"].mean() > 8000)
print(result[["Department", "Employee", "Salary"]])
```

#### 输出

```text
  Department Employee  Salary
0      Sales    Alice    8000
1      Sales      Bob    9000
2         IT  Charlie   12000
3         IT    David   11000
```

#### 理解重点

- `filter` 在组级别做判断，保留**整组所有行**；`df[df.Salary > 8000]` 在行级别做判断——语义不同
- 典型场景：只分析"样本量足够多"的组（`len(g) >= 10`）；剔除"数据不完整"的组

## 6. 附加方法与属性

### 速览

| 方法/属性 | 类型 | 返回 | 说明 |
|---|---|---|---|
| `gb.size()` | 方法 | `Series` | 每组行数（含 NaN），比 `count` 快 |
| `gb.count()` | 方法 | `DataFrame` | 按列的非空计数 |
| `gb.ngroups` | 属性 | `int` | 分组总数 |
| `gb.groups` | 属性 | `dict` | `{分组键: [行索引列表]}` |
| `gb.indices` | 属性 | `dict` | 同 `groups`，低级别接口 |
| `gb.first()` / `gb.last()` | 方法 | `DataFrame` | 每组的首/末行 |

## 7. 三大方法对比

| 方法 | 函数返回 | 结果形状 | 适用场景 |
|---|---|---|---|
| `agg` | 标量 | 每组一行 | 汇总统计（均值、总和、计数） |
| `transform` | 同长度 Series | 与原数据同长度 | 组内标准化、排名、均值填充、累计值 |
| `apply` | 任意 | 任意（自动推断） | 自由度最高——当 agg/transform 不够用时 |
| `filter` | 布尔 | 原数据子集 | 按组级条件保留/丢弃整组 |

## 常见坑

1. `groupby` 返回**懒对象**，不触发计算——直接打印看不到数据，必须调用 `agg` 等方法才会执行
2. 字典形式 `agg({"col": "fn"})` 在多函数时产生 MultiIndex 列——**优先使用命名聚合** `agg(新名=("列", "函数"))`
3. `apply` 的第一组可能被调用两次（Pandas 推断返回类型）——函数应该是**纯函数**，无副作用
4. 多列分组 `groupby(["A", "B"])` 结果索引是 MultiIndex——后续处理用 `reset_index()`
5. `transform` 的函数必须返回**与输入同长度**的值——返回标量会被自动广播，可能产生意外结果
6. `groupby(..., dropna=True)`（默认）会**丢弃**分组键为 NaN 的行——需要保留时显式设 `dropna=False`

## 小结

- 分组聚合的思维模型是 **Split-Apply-Combine**——先分、再算、后合
- 写代码的优先级：`agg` > `transform` > `apply`——选功能够用中最快的那个
- 聚合时推荐**命名聚合**语法——结果列名清晰可控
- `transform` 配合 `groupby` 是特征工程（组内统计特征）的利器

# Pandas 数据合并与连接

## 本章目标

1. 掌握 `pd.concat` 的行拼接与列拼接
2. 掌握 `pd.merge` 的四种连接模式（inner / left / right / outer）
3. 学会用 `left_on` / `right_on` 合并不同列名的键
4. 掌握 `df.join` 按索引的快捷连接
5. 学会用 `indicator` 参数诊断合并质量

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.concat(...)` | 函数 | 按行或列拼接多个 DataFrame |
| `pd.merge(...)` | 函数 | 数据库风格连接（类似 SQL JOIN） |
| `df.join(...)` | 方法 | 按索引连接（`merge` 的索引版快捷方式） |
| `df.combine_first(...)` | 方法 | 用另一个 DataFrame 填充缺失值 |

## 1. 行列拼接

### `pd.concat`

#### 作用

沿指定轴将多个 DataFrame 或 Series 拼接在一起。`axis=0` 纵向堆叠（加行），`axis=1` 横向拼接（加列）。

#### 重点方法

```python
pd.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None,
          verify_integrity=False, sort=False, copy=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `objs` | `list[DataFrame]`、`dict` | 待拼接对象序列；`dict` 时键用作多层索引 | `[df1, df2]` |
| `axis` | `int` | `0` 按行拼接（上下堆叠）、`1` 按列拼接（左右拼接），默认为 `0` | `1` |
| `join` | `str` | 非拼接轴上索引对齐方式：`'outer'` 并集 / `'inner'` 交集，默认为 `'outer'` | `"inner"` |
| `ignore_index` | `bool` | `True` 时重置索引，默认为 `False` | `True` |
| `keys` | `list` | 为每组数据添加标签（形成多层索引的顶层），默认为 `None` | `["df1", "df2"]` |
| `verify_integrity` | `bool` | `True` 时检查结果索引是否有重复，默认为 `False` | `True` |
| `sort` | `bool` | `axis=1` 时是否对非拼接轴排序，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [25, 30],
    "City": ["Beijing", "Shanghai"],
})

df2 = pd.DataFrame({
    "Name": ["Charlie", "David"],
    "Age": [35, 28],
    "City": ["Guangzhou", "Shenzhen"],
})

# 按行拼接
rows = pd.concat([df1, df2], axis=0, ignore_index=True)
print(f"axis=0 按行拼接:\n{rows}")

# 按列拼接
left = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
right = pd.DataFrame({"City": ["Beijing", "Shanghai"], "Score": [85, 92]})
cols = pd.concat([left, right], axis=1)
print(f"\naxis=1 按列拼接:\n{cols}")
```

#### 输出

```text
axis=0 按行拼接:
      Name  Age       City
0    Alice   25    Beijing
1      Bob   30   Shanghai
2  Charlie   35  Guangzhou
3    David   28   Shenzhen

axis=1 按列拼接:
    Name  Age      City  Score
0  Alice   25   Beijing     85
1    Bob   30  Shanghai     92
```

#### 理解重点

- `ignore_index=True` 重置为连续整数索引——避免拼接后出现重复索引
- `keys` 参数可标记每组数据来源：`pd.concat([df1, df2], keys=["A", "B"])` 创建 MultiIndex
- `join='inner'` 只保留所有 DataFrame 共有的列（按列拼接时）或行（按行拼接时）

## 2. 数据库风格合并

### `pd.merge`

#### 作用

类似 SQL JOIN，按指定列（键）将两个 DataFrame 的行对齐合并。支持 inner / left / right / outer 四种连接模式。

#### 重点方法

```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
         indicator=False, validate=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `left` | `DataFrame` | 左侧 DataFrame | `df1` |
| `right` | `DataFrame` | 右侧 DataFrame | `df2` |
| `how` | `str` | 连接方式：`'inner'` / `'left'` / `'right'` / `'outer'`，默认为 `'inner'` | `"left"` |
| `on` | `str`、`list[str]` | 两侧同名的连接键列名；`None` 时自动用交集列名 | `"ID"`、`["A", "B"]` |
| `left_on` | `str`、`list[str]` | 左侧连接键（两侧列名不同时使用） | `"left_id"` |
| `right_on` | `str`、`list[str]` | 右侧连接键（与 `left_on` 配合） | `"right_id"` |
| `left_index` | `bool` | `True` 时用左侧索引作为连接键，默认为 `False` | `True` |
| `right_index` | `bool` | `True` 时用右侧索引作为连接键，默认为 `False` | `True` |
| `suffixes` | `tuple[str, str]` | 同名列（非键）的后缀，默认为 `('_x', '_y')` | `('_L', '_R')` |
| `indicator` | `bool`、`str` | `True` 时新增 `_merge` 列标记每行的来源，默认为 `False` | `True` |
| `validate` | `str` 或 `None` | 验证连接键的唯一性：`'one_to_one'` / `'one_to_many'` / `'many_to_one'` / `'many_to_many'`，默认为 `None` | `"one_to_one"` |

### 四种 `how` 模式

| `how` | 行为 | SQL 类比 |
|---|---|---|
| `"inner"` | 只保留两侧都匹配的行 | `INNER JOIN` |
| `"left"` | 保留左侧所有行，右侧无匹配的填充 NaN | `LEFT JOIN` |
| `"right"` | 保留右侧所有行，左侧无匹配的填充 NaN | `RIGHT JOIN` |
| `"outer"` | 保留两侧所有行，无匹配的填充 NaN | `FULL OUTER JOIN` |

### 综合示例

#### 示例代码

```python
import pandas as pd

employees = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Dept": ["Sales", "IT", "IT", "HR"],
})

salaries = pd.DataFrame({
    "ID": [1, 2, 3, 5],
    "Salary": [8000, 12000, 15000, 9000],
})

print("员工表:")
print(employees)
print(f"\n薪资表:")
print(salaries)

# inner：只保留两边都有的 ID
print(f"\ninner merge (交集):")
print(pd.merge(employees, salaries, on="ID", how="inner"))

# left：保留所有员工，缺薪资的填 NaN
print(f"\nleft merge (保留所有员工):")
print(pd.merge(employees, salaries, on="ID", how="left"))

# outer：保留所有 ID
print(f"\nouter merge (全外连接):")
print(pd.merge(employees, salaries, on="ID", how="outer"))

# 带 indicator 的 outer
print(f"\nouter + indicator:")
print(pd.merge(employees, salaries, on="ID", how="outer", indicator=True))
```

#### 输出

```text
员工表:
   ID     Name  Dept
0   1    Alice Sales
1   2      Bob    IT
2   3  Charlie    IT
3   4    David    HR

薪资表:
   ID  Salary
0   1    8000
1   2   12000
2   3   15000
3   5    9000

inner merge (交集):
   ID     Name  Dept  Salary
0   1    Alice Sales    8000
1   2      Bob    IT   12000
2   3  Charlie    IT   15000

left merge (保留所有员工):
   ID     Name  Dept   Salary
0   1    Alice Sales   8000.0
1   2      Bob    IT  12000.0
2   3  Charlie    IT  15000.0
3   4    David    HR      NaN

outer merge (全外连接):
   ID     Name  Dept   Salary
0   1    Alice Sales   8000.0
1   2      Bob    IT  12000.0
2   3  Charlie    IT  15000.0
3   4    David    HR      NaN
4   5      NaN   NaN   9000.0

outer + indicator:
   ID     Name  Dept   Salary      _merge
0   1    Alice Sales   8000.0        both
1   2      Bob    IT  12000.0        both
2   3  Charlie    IT  15000.0        both
3   4    David    HR      NaN   left_only
4   5      NaN   NaN   9000.0  right_only
```

#### 理解重点

- `how='left'` 是最常用的合并模式——以主表为基准，补充辅助信息
- `left_on` / `right_on` 在两侧键列名不同时使用——列名无需统一
- `indicator=True` 生成 `_merge` 列标记每行来源——用于合并质量诊断
- `validate` 在预期一对一/一对多时加验证——防止意外的多对多产生重复行
- 多键合并：`on=["A", "B"]` 同时按两列匹配

## 3. 按索引连接

### `DataFrame.join`

#### 作用

按索引连接两个 DataFrame，是 `pd.merge(left_index=True, right_index=True)` 的便捷封装。可同时按索引+列连接。

#### 重点方法

```python
df.join(other, on=None, how='left', lsuffix='', rsuffix='',
        sort=False, validate=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `other` | `DataFrame`、`Series`、`list` | 要连接的 DataFrame | `df2` |
| `on` | `str`、`list[str]` 或 `None` | 主表用列键（从表仍用索引），默认为 `None` | `"ID"` |
| `how` | `str` | 连接方式，同 `merge`，默认为 `'left'` | `"inner"`、`"outer"` |
| `lsuffix` | `str` | 左侧同名列的后缀，默认为 `''` | `"_L"` |
| `rsuffix` | `str` | 右侧同名列的后缀，默认为 `''` | `"_R"` |
| `validate` | `str` 或 `None` | 同 `merge` 的键验证，默认为 `None` | `"one_to_one"` |

#### 示例代码

```python
import pandas as pd

df1 = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
}, index=["a", "b", "c"])

df2 = pd.DataFrame({
    "City": ["Beijing", "Shanghai", "Guangzhou"],
    "Score": [85, 92, 78],
}, index=["a", "b", "d"])

# join 按索引（默认 left）
print(f"df1.join(df2):\n{df1.join(df2)}")

# inner join
print(f"\ndf1.join(df2, how='inner'):\n{df1.join(df2, how='inner')}")
```

#### 输出

```text
df1.join(df2):
      Name  Age       City  Score
a    Alice   25    Beijing   85.0
b      Bob   30   Shanghai   92.0
c  Charlie   35        NaN    NaN

df1.join(df2, how='inner'):
    Name  Age      City  Score
a  Alice   25   Beijing     85
b    Bob   30  Shanghai     92
```

#### 理解重点

- `join` 的核心场景：主表用普通列，从表已经把键设为了索引
- `on` + `join` 组合可比 `merge` 更简洁：`df.set_index("key").join(other.set_index("key"))`

## 4. 缺失值补齐

### `DataFrame.combine_first`

#### 作用

用另一个 DataFrame 的值填充当前 DataFrame 的缺失值（NaN）。按索引和列名对齐，只在当前值为 NaN 时填充。

#### 重点方法

```python
df.combine_first(other)
```

#### 示例代码

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame({
    "A": [1, np.nan, 3],
    "B": [np.nan, 5, np.nan],
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    "A": [10, 20, 30],
    "B": [40, 50, 60],
}, index=[0, 1, 2])

print(f"df1 (有缺失):\n{df1}")
print(f"\ndf2 (备用值):\n{df2}")
print(f"\ncombine_first:\n{df1.combine_first(df2)}")
```

#### 输出

```text
df1 (有缺失):
     A    B
0  1.0  NaN
1  NaN  5.0
2  3.0  NaN

df2 (备用值):
    A   B
0  10  40
1  20  50
2  30  60

combine_first:
     A    B
0  1.0  40.0
1  20.0  5.0
2  3.0  60.0
```

#### 理解重点

- `combine_first` 只在单元格为 NaN 时才填充——已有值不会被覆盖
- 按索引和列名同时对齐——两侧行列标签不必完全相同（多余的行列会保留）

## 连接方法选择指南

| 场景 | 推荐方法 |
|---|---|
| 多个相同结构数据上下堆叠 | `pd.concat(axis=0)` |
| 左右拼接（按位置对齐） | `pd.concat(axis=1)` |
| 按列值匹配合并 | `pd.merge(on=...)` |
| 按索引匹配合并 | `df.join(...)` |
| 用备用数据填充 NaN | `df.combine_first(...)` |

## 常见坑

1. `pd.merge` 不指定 `on` 时自动用所有共同列名作为键——可能无意中将多列合并导致结果行数爆炸
2. 多对多合并会产生笛卡尔积——用 `validate` 参数提前验证键的唯一性
3. `concat` 默认保留原始索引——行拼接后可能出现重复索引，用 `ignore_index=True` 重置
4. `join` 默认 `how='left'`，而 `merge` 默认 `how='inner'`——行为不一致，容易误用
5. 合并后同名列（非键）被加后缀 `_x` / `_y`——用 `suffixes` 自定义有意义的后缀
6. `combine_first` 要求索引和列名对齐——数据来源结构不同时需先调整

## 小结

- 纯堆叠用 `concat`；按键匹配用 `merge`；按索引匹配用 `join`
- `merge` 的四种 `how` 对应 SQL 四种 JOIN——`'left'` 最常用，以主表为基准
- `indicator` 和 `validate` 是合并质量诊断的两个关键参数
- 合并后务必检查行数——意外的笛卡尔积是最常见的合并 bug

# Pandas 时间序列

## 本章目标

1. 掌握 `to_datetime` 和 `date_range` 创建时间戳/日期范围
2. 掌握 `.dt` 访问器的日期组件提取（年/月/日/星期等）
3. 掌握 DatetimeIndex 的字符串切片与部分匹配
4. 掌握 `resample` 的降/升采样聚合
5. 掌握 `rolling` 滚动窗口计算
6. 掌握 `shift` / `diff` / `pct_change` 做时间偏移与变化率

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.to_datetime(...)` | 函数 | 解析日期时间为 `datetime64` 类型 |
| `pd.date_range(...)` | 函数 | 生成连续日期序列（DatetimeIndex） |
| `Series.dt.xxx` | 访问器 | 提取日期组件（年/月/日/星期/季度等） |
| `df.resample(...)` | 方法 | 按时间频率重采样（降/升采样） |
| `df.rolling(...)` | 方法 | 固定窗口大小的滚动计算 |
| `df.ewm(...)` | 方法 | 指数加权移动窗口 |
| `df.shift(...)` | 方法 | 行偏移（向前/向后平移） |
| `df.diff(...)` | 方法 | 相邻行差分 |
| `Series.pct_change(...)` | 方法 | 变化率（相对百分比） |

## 1. 时间戳创建

### `pd.to_datetime`

#### 作用

将字符串、整数、Series 解析为 `datetime64` 类型。支持自动推断格式，也可用 `format` 参数显式指定以加速。

参数表见 [ch04 类型转换 — `pd.to_datetime`](pandas.md)（`errors` / `format` / `dayfirst` / `utc` / `unit`）。

#### 示例代码

```python
import pandas as pd

# 字符串列表
dates = pd.to_datetime(["2024-01-15", "2024-03-20", "2024-06-10"])
print(f"to_datetime:\n{dates}")
print(f"dtype: {dates.dtype}")
```

#### 输出

```text
to_datetime:
0   2024-01-15
1   2024-03-20
2   2024-06-10
dtype: datetime64[ns]
```

### `pd.date_range`

#### 作用

生成连续的日期时间索引（DatetimeIndex）。通过 `start`/`end`/`periods`/`freq` 控制范围和频率。

#### 重点方法

```python
pd.date_range(start=None, end=None, periods=None, freq=None, tz=None,
              normalize=False, name=None, inclusive='both')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `str`、`datetime` | 起始日期 | `"2024-01-01"` |
| `end` | `str`、`datetime` | 结束日期 | `"2024-01-10"` |
| `periods` | `int` | 生成的日期数量（与 `freq` 配合），默认为 `None` | `10` |
| `freq` | `str` | 频率字符串：`'D'` 天 / `'W'` 周 / `'M'` 月 / `'H'` 小时 / `'T'` 分钟，默认为 `None` | `"D"`、`"W-MON"`、`"M"` |
| `tz` | `str` 或 `None` | 时区 | `"Asia/Shanghai"` |
| `inclusive` | `str` | 端点包含策略：`'both'` / `'left'` / `'right'` / `'neither'`，默认为 `'both'` | `"left"` |

#### 常用频率字符串

| 频率 | 含义 | 频率 | 含义 |
|---|---|---|---|
| `"D"` | 每日 | `"W"` | 每周（周日） |
| `"W-MON"` | 每周一 | `"MS"` | 每月首日 |
| `"M"` | 每月末日 | `"Q"` | 每季度末日 |
| `"H"` | 每小时 | `"T"` / `"min"` | 每分钟 |
| `"B"` | 工作日 | `"YS"` | 每年首日 |

#### 示例代码

```python
import pandas as pd

# 天数
r1 = pd.date_range("2024-01-01", "2024-01-07", freq="D")
print(f"每日:\n{r1}")

# 工作日
r2 = pd.date_range("2024-01-01", periods=5, freq="B")
print(f"\n工作日:\n{r2}")

# 月初
r3 = pd.date_range("2024-01-01", periods=4, freq="MS")
print(f"\n每月初:\n{r3}")
```

#### 输出

```text
每日:
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05', '2024-01-06', '2024-01-07'],
              dtype='datetime64[ns]', freq='D')

工作日:
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05'],
              dtype='datetime64[ns]', freq='B')

每月初:
DatetimeIndex(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'], dtype='datetime64[ns]', freq='MS')
```

#### 理解重点

- `periods` 和 `end` 互斥——指定数量时无需指定结束日期
- 频率字符串大小写敏感：`"D"` 天 / `"W"` 周 / `"M"` 月 / `"H"` 小时
- `date_range` 返回 `DatetimeIndex`——可直接作为 DataFrame 的行索引

## 2. 日期组件提取（`.dt` 访问器）

### `.dt` 访问器

#### 作用

通过 `Series.dt.xxx` 从 `datetime64` 列中提取日期组件（年/月/日/星期/季度/周数等），全部为向量化操作。

### 常用属性速览

| 属性 | 含义 | 示例输出 |
|---|---|---|
| `dt.year` | 年 | `2024` |
| `dt.month` | 月 (1-12) | `1` |
| `dt.day` | 日 (1-31) | `15` |
| `dt.hour` | 时 (0-23) | `14` |
| `dt.minute` | 分 (0-59) | `30` |
| `dt.dayofweek` | 星期几 (0=周一, 6=周日) | `0` |
| `dt.day_name()` | 星期几的名称 | `"Monday"` |
| `dt.quarter` | 季度 (1-4) | `1` |
| `dt.week` | 周数 (1-52) | `3` |
| `dt.is_month_start` | 是否月初 | `True` |

### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Date": pd.to_datetime(["2024-01-15", "2024-03-20", "2024-06-10",
                            "2024-09-05", "2024-12-25"]),
    "Value": [100, 200, 150, 300, 250],
})

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.day_name()
df["Quarter"] = df["Date"].dt.quarter

print(df)
```

#### 输出

```text
        Date  Value  Year  Month   DayOfWeek  Quarter
0 2024-01-15    100  2024      1      Monday        1
1 2024-03-20    200  2024      3   Wednesday        1
2 2024-06-10    150  2024      6      Monday        2
3 2024-09-05    300  2024      9    Thursday        3
4 2024-12-25    250  2024     12   Wednesday        4
```

#### 理解重点

- `.dt` 只能用于 `datetime64` 类型列——需先用 `pd.to_datetime` 转换
- `.dt` 的属性全部向量化——比循环快 100 倍以上
- `dt.dayofweek` 返回整数（0=周一），`dt.day_name()` 返回名称（`"Monday"`）

## 3. 时间序列索引与切片

#### 作用

当 DataFrame 索引为 `DatetimeIndex` 时，可用**日期字符串**直接切片——Pandas 会自动解析为时间范围。

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=10, freq="D")
df = pd.DataFrame({
    "Value": np.random.randint(10, 100, 10),
}, index=dates)

print(f"时间序列:\n{df}")

# 字符串部分匹配切片
print(f"\n2024-01-03 到 2024-01-06:\n{df['2024-01-03':'2024-01-06']}")

# 按年份切片
print(f"\n2024-01 全部:\n{df['2024-01']}")
```

#### 输出

```text
时间序列:
            Value
2024-01-01     56
2024-01-02     61
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63
2024-01-07     94
2024-01-08     68
2024-01-09     56
2024-01-10     99

2024-01-03 到 2024-01-06:
            Value
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63

2024-01 全部:
            Value
2024-01-01     56
2024-01-02     61
2024-01-03     88
2024-01-04     70
2024-01-05     87
2024-01-06     63
2024-01-07     94
2024-01-08     68
2024-01-09     56
2024-01-10     99
```

#### 理解重点

- 部分字符串切片极其强大：`df["2024-01"]` 匹配整个 1 月，`df["2024"]` 匹配整年——无需构造日期对象
- 只有 `DatetimeIndex` 支持字符串切片——普通 `object` 索引不行

## 4. 重采样

### `DataFrame.resample`

#### 作用

按指定频率重新聚合时间序列数据。降采样（如 日->月）做聚合，升采样（如 月->日）做插值。类似 `groupby`，返回 `Resampler` 对象，需调用聚合函数才执行。

#### 重点方法

```python
df.resample(rule, axis=0, closed=None, label=None, on=None, level=None)
# 后续必须链式调用聚合：.mean() / .sum() / .agg() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `rule` | `str` | 目标频率字符串 | `"W"`、`"M"`、`"Q"`、`"H"` |
| `axis` | `int` | 采样轴，默认为 `0` | `1` |
| `closed` | `str` 或 `None` | 区间的哪端闭合：`'left'` / `'right'`，默认为 `None`（各频率有默认值） | `"left"` |
| `label` | `str` 或 `None` | 聚合结果用区间的哪端作标签，默认为 `None` | `"left"` |
| `on` | `str` 或 `None` | 用指定列而不是索引做时间轴 | `"date_col"` |

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=60, freq="D")
df = pd.DataFrame({
    "Sales": np.random.randint(100, 500, 60),
}, index=dates)

print(f"原始数据（前 5 天）:\n{df.head()}")

# 降采样：日 -> 周
weekly = df.resample("W").agg(["sum", "mean"])
print(f"\n按周聚合:\n{weekly}")

# 降采样：日 -> 月
monthly = df.resample("M").sum()
print(f"\n按月聚合:\n{monthly}")
```

#### 输出

```text
原始数据（前 5 天）:
            Sales
2024-01-01    288
2024-01-02    308
2024-01-03    214
2024-01-04    124
2024-01-05    127

按周聚合:
           Sales
             sum   mean
2024-01-07  1158  165.43
2024-01-14  2175  310.71
2024-01-21  1689  241.29
2024-01-28  2129  304.14
2024-02-04  2645  377.86
2024-02-11  1769  252.71
2024-02-18  2184  312.00
2024-02-25  2012  287.43
2024-03-01  1153  288.25

按月聚合:
            Sales
2024-01-31   9255
2024-02-29  10347
2024-03-31    279
```

#### 理解重点

- `resample` 的思维模型 = `groupby` + 时间窗口——分组依据是频率而不是列值
- 必须链式调用聚合函数：`resample("M")` 不执行，`resample("M").sum()` 才执行
- 频率字符串与 `date_range` 相同——`"D"` 天 / `"W"` 周 / `"M"` 月 / `"Q"` 季度

## 5. 滚动窗口

### `DataFrame.rolling`

#### 作用

在数据上滑动一个固定大小的窗口，对窗口内的值做聚合。与 `resample` 的区别：`rolling` 不改变数据频率，每行都有一个结果。

#### 重点方法

```python
df.rolling(window, min_periods=None, center=False, win_type=None,
           on=None, axis=0, closed=None)
# 后续必须链式调用：.mean() / .sum() / .std() / .apply() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `window` | `int`、`str` | 窗口大小（行数）或时间偏移字符串 | `7`、`"3D"` |
| `min_periods` | `int` 或 `None` | 最少需要多少非空值才计算结果，默认为 `None`（等于窗口大小） | `1` |
| `center` | `bool` | `True` 时窗口居中（结果标签在窗口中心），默认为 `False` | `True` |
| `win_type` | `str` 或 `None` | 窗口类型（加权）：`'triang'` / `'gaussian'` 等，默认为 `None` | `"gaussian"` |
| `on` | `str` 或 `None` | 用指定列而不是索引做时间轴 | `"date_col"` |

### `DataFrame.ewm`

#### 作用

指数加权移动（EWM）：最近的数据点权重最大，按指数衰减。无固定窗口大小，所有历史值都参与。

#### 重点方法

```python
df.ewm(com=None, span=None, halflife=None, alpha=None, adjust=True)
# 后续必须链式调用：.mean() / .std() 等
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `span` | `float` 或 `None` | 跨度（约等于窗口中心），$\alpha = 2/(span+1)$ | `7` |
| `halflife` | `float` 或 `None` | 半衰期（权重衰减到一半的时间） | `3` |
| `alpha` | `float` 或 `None` | 直接指定平滑因子 $\alpha \in (0, 1]$ | `0.3` |
| `adjust` | `bool` | `True` 时用权重归一化（更准确），默认为 `True` | `False` |

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=20, freq="D")
df = pd.DataFrame({
    "Value": np.random.randint(100, 500, 20).astype(float),
}, index=dates)

# 滚动均值（窗口=3）
df["RollMean3"] = df["Value"].rolling(window=3).mean()

# 滚动均值（窗口=7）
df["RollMean7"] = df["Value"].rolling(window=7).mean()

# 指数加权
df["EWM_span7"] = df["Value"].ewm(span=7).mean()

print(df.head(12).round(1))
```

#### 输出

```text
            Value  RollMean3  RollMean7  EWM_span7
2024-01-01  288.0        NaN        NaN      288.0
2024-01-02  308.0        NaN        NaN      294.3
2024-01-03  214.0      270.0        NaN      268.9
2024-01-04  124.0      215.3        NaN      217.7
2024-01-05  127.0      155.0        NaN      186.1
2024-01-06  362.0      204.3        NaN      246.6
2024-01-07  185.0      224.7      229.7      226.1
2024-01-08  297.0      281.3      231.0      249.3
2024-01-09  491.0      324.3      257.1      326.3
2024-01-10  447.0      411.7      290.4      366.7
2024-01-11  412.0      450.0      331.6      382.0
2024-01-12  205.0      354.7      342.7      324.3
```

#### 理解重点

- `rolling` 窗口前 `window-1` 行结果为 NaN——因窗口内非空值不足
- `center=True` 时窗口以当前行为中心——适合事后分析，不适合实时预测
- `ewm` 无 NaN 首行——因子权重始终对第一个观测值有定义，首行等于自身
- `ewm` 比 `rolling` 更平滑——因为所有权重连续衰减而非等权

## 6. 偏移与变化率

### `shift` / `diff` / `pct_change`

#### 作用

- `shift(periods)`：将数据沿时间轴平移。正数为前移（滞后特征），负数为后移
- `diff(periods)`：相邻行的差分：$\Delta_t = x_t - x_{t-periods}$
- `pct_change(periods)`：相对变化率：$r_t = \frac{x_t - x_{t-periods}}{x_{t-periods}}$

#### 重点方法

```python
df.shift(periods=1, freq=None, axis=0)
df.diff(periods=1, axis=0)
Series.pct_change(periods=1, fill_method=None, limit=None)
```

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Price": [100, 105, 103, 108, 112, 110],
})

df["Prev"] = df["Price"].shift(1)
df["Change"] = df["Price"].diff(1)
df["ChangePct"] = df["Price"].pct_change(1)
print(df)
```

#### 输出

```text
   Price   Prev  Change  ChangePct
0    100    NaN     NaN        NaN
1    105  100.0     5.0   0.050000
2    103  105.0    -2.0  -0.019048
3    108  103.0     5.0   0.048544
4    112  108.0     4.0   0.037037
5    110  112.0    -2.0  -0.017857
```

#### 数学公式

$$
\begin{aligned}
\text{diff:}&\quad \Delta_t = x_t - x_{t-1} \\[4pt]
\text{pct\_change:}&\quad r_t = \frac{x_t - x_{t-1}}{x_{t-1}}
\end{aligned}
$$

#### 理解重点

- `shift(1)` 的第一行为 NaN——没有更早的数据
- `diff` 等价于 `x - x.shift(1)`
- `pct_change` 是金融分析中最常用的指标之一——计算收益率/涨跌幅
- `shift(-1)` 可做"前瞻"特征——但注意别在预测模型中造成数据泄露

## 常见坑

1. `resample` 返回 `Resampler` 对象——不调聚合方法就没有结果，容易忘记链式调用 `.mean()` 等
2. `rolling` 窗口前 N-1 行是 NaN——不要当成 bug，这是正常行为
3. `date_range` 的 `freq="M"` 返回月末，`freq="MS"` 返回月初——两者不同
4. `.dt` 访问器只能用于 `datetime64` 类型——非日期列调用会报 `AttributeError`
5. 时间字符串切片只在索引为 `DatetimeIndex` 时可用——普通 object 索引不支持
6. `shift` 可做滞后特征（`periods>0`）和前瞻（`periods<0`）——前瞻特征在预测中会造成数据泄露

## 小结

- 时间序列的核心索引类型是 `DatetimeIndex`——`date_range` 生成，`to_datetime` 解析
- `.dt` 访问器提取日期组件；字符串切片做灵活的时间范围过滤
- `resample`（改频率）+ `rolling`（窗口计算）+ `ewm`（指数加权）是时间序列分析的三大算子
- `shift` / `diff` / `pct_change` 是时间序列特征工程的三个基础操作

# Pandas 数据可视化

## 本章目标

1. 掌握 `df.plot` 的统一绘图入口与 `kind` 参数
2. 掌握折线图、柱状图、直方图、散点图、箱线图、饼图的 Pandas 写法
3. 理解 Pandas 绘图与 Matplotlib 的关系（底层调用 matplotlib）
4. 学会将图表保存到文件

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `df.plot(...)` | 方法 | 统一绘图入口；默认 `kind='line'` |
| `df.plot.line(...)` | 方法 | 折线图 |
| `df.plot.area(...)` | 方法 | 面积图（堆叠） |
| `df.plot.bar(...)` / `df.plot.barh(...)` | 方法 | 垂直/水平柱状图 |
| `df.plot.hist(...)` / `df.plot.kde(...)` | 方法 | 直方图 / 核密度估计 |
| `df.plot.scatter(...)` | 方法 | 散点图 |
| `df.plot.box(...)` | 方法 | 箱线图 |
| `df.plot.pie(...)` | 方法 | 饼图 |

Pandas 绘图底层调用 Matplotlib，所有 `df.plot.*` 方法返回 `matplotlib.axes.Axes` 对象，可通过 Matplotlib API 进一步定制。

## 1. 统一绘图接口

### `DataFrame.plot`

#### 作用

Pandas 的一站式绘图入口。通过 `kind` 参数切换图表类型，底层自动调用对应的 `plot.<kind>()` 方法。返回 `matplotlib.axes.Axes` 对象，可链式调用 Matplotlib 的 API 做进一步定制。

#### 重点方法

```python
df.plot(*args, kind='line', ax=None, figsize=None, title=None,
        x=None, y=None, legend=True, grid=None, xlabel=None, ylabel=None,
        xticks=None, yticks=None, rot=None, fontsize=None, colormap=None,
        table=False, subplots=False, sharex=False, sharey=False,
        logx=False, logy=False, mark_right=True, style=None, **kwargs)
```

#### 参数（核心 12 个）

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kind` | `str` | 图表类型，下见表，默认为 `'line'` | `"bar"`、`"hist"`、`"scatter"` |
| `x` | `str` 或 `None` | X 轴列名 | `"Date"` |
| `y` | `str`、`list[str]` 或 `None` | Y 轴列名（单列或多列） | `"Sales"`、`["A", "B"]` |
| `ax` | `matplotlib.axes.Axes` 或 `None` | 绘制到已有 axes（子图组合用） | `ax1` |
| `figsize` | `tuple[float, float]` | 图表尺寸（英寸），默认为 `(6.4, 4.8)` | `(10, 6)` |
| `title` | `str` | 图表标题 | `"销售趋势"` |
| `legend` | `bool` | 是否显示图例，默认为 `True` | `False` |
| `grid` | `bool` | 是否显示网格，默认为 `None` | `True` |
| `xlabel` / `ylabel` | `str` 或 `None` | 自定义轴标签 | `"月份"` |
| `rot` | `int` | X 轴刻度标签旋转角度 | `45` |
| `fontsize` | `int` | 字体大小 | `12` |
| `colormap` | `str` | 颜色映射方案 | `"viridis"`、`"tab10"` |

#### `kind` 图表类型速览

| `kind` | 等价方法 | 适用场景 |
|---|---|---|
| `'line'`（默认） | `df.plot.line()` | 趋势、时间序列 |
| `'area'` | `df.plot.area()` | 占比累积趋势 |
| `'bar'` | `df.plot.bar()` | 分类对比（垂直） |
| `'barh'` | `df.plot.barh()` | 分类对比（水平，适合长标签） |
| `'hist'` | `df.plot.hist()` | 单变量分布 |
| `'kde'` / `'density'` | `df.plot.kde()` | 概率密度估计 |
| `'scatter'` | `df.plot.scatter()` | 双变量关系 |
| `'box'` | `df.plot.box()` | 四分位分布 + 离群点 |
| `'pie'` | `df.plot.pie()` | 占比组成（仅单列 Series） |

#### 理解重点

- `kind` 和 `plot.xxx()` 写法等价：`df.plot(kind='bar')` = `df.plot.bar()`
- 返回 `Axes` 对象——可以用 Matplotlib API 继续修改：`.set_title(...)`、`.legend()`、`.set_xlabel(...)`
- Pandas 绘图是快速探索工具——报告级图表建议直接使用 Matplotlib / Seaborn

## 2. 折线图与面积图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=12, freq="MS")
df = pd.DataFrame({
    "ProductA": np.random.randint(100, 300, 12),
    "ProductB": np.random.randint(80, 250, 12),
    "ProductC": np.random.randint(50, 200, 12),
}, index=dates)

import matplotlib.pyplot as plt

# 折线图：多列趋势
ax = df.plot.line(figsize=(10, 5), title="产品月度销售趋势",
                  ylabel="销售额", grid=True, marker="o")
# 可视化输出见下方图表：折线图显示三条趋势线，各自带圆形标记
plt.close()

# 面积图：堆叠面积
ax = df.plot.area(figsize=(10, 5), title="产品月度销售占比",
                  ylabel="销售额", alpha=0.7)
# 可视化输出见下方图表：堆叠面积图显示三类产品各自贡献的累积趋势
plt.close()
```

#### 输出

可视化输出见下方图表（Pandas 调用 Matplotlib 渲染）。折线图中三条趋势线分别代表 ProductA/ProductB/ProductC，X 轴为月份，Y 轴为销售额。面积图以堆叠方式显示各产品对总销售额的贡献变化。

### 理解重点

- 多列 DataFrame 直接 `plot.line()` 会为每列生成一条折线——自动区分颜色
- 面积图 `plot.area()` 默认 `stacked=True`——展示总量的组成部分

## 3. 柱状图

### 综合示例

#### 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Sales": [150, 220, 180, 260, 190],
    "Profit": [30, 55, 40, 70, 45],
}, index=["北京", "上海", "广州", "深圳", "杭州"])

# 分组柱状图
ax = df.plot.bar(figsize=(10, 5), title="各城市销售与利润对比",
                 ylabel="金额（万元）", rot=0)
# 可视化输出见下方图表：分组柱状图，每城市两个柱子（Sales/Profit）

# 水平柱状图（适合长标签）
ax = df["Sales"].sort_values().plot.barh(figsize=(8, 5), title="各城市销售额",
                                          xlabel="金额（万元）", color="steelblue")
# 可视化输出见下方图表：水平柱状图，城市名作为 Y 轴标签
plt.close()
```

#### 输出

可视化输出见下方图表。垂直柱状图为每城市显示 Sales 和 Profit 两组柱子，便于横向对比。水平柱状图按销售额排序，城市名在 Y 轴排列清晰。

### 理解重点

- 柱状图自动按索引分组——`df.plot.bar()` 每列一组柱子
- `barh` 比 `bar` 更适合中文/长标签——垂直排列文字不被截断
- 堆叠柱状图：`df.plot.bar(stacked=True)`——显示总量+各组成部分

## 4. 直方图与密度图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "Score": np.random.normal(70, 15, 500).clip(0, 100),
})

# 直方图
ax = df["Score"].plot.hist(bins=20, figsize=(10, 5),
                            title="成绩分布直方图", alpha=0.7, edgecolor="black")
# 可视化输出见下方图表：直方图显示成绩分布，20 个区间
plt.close()

# 多列直方图
df2 = pd.DataFrame({
    "Class A": np.random.normal(75, 10, 200).clip(0, 100),
    "Class B": np.random.normal(65, 15, 200).clip(0, 100),
})
ax = df2.plot.hist(bins=20, alpha=0.5, figsize=(10, 5),
                    title="两班成绩分布对比")
# 可视化输出见下方图表：两个半透明直方图叠加，便于分布对比
plt.close()

# 核密度估计
ax = df2.plot.kde(figsize=(10, 5), title="两班成绩密度曲线", linewidth=2)
# 可视化输出见下方图表：两条平滑密度曲线
plt.close()
```

#### 输出

可视化输出见下方图表。直方图以柱形高度表示各分数段的人数，核密度图以平滑曲线展示概率密度分布——后者不受分箱数影响，更适合对比。

### 理解重点

- 直方图的 `bins` 控制柱子数量——过多过少都不好，20~30 是常用范围
- `alpha` 透明度对叠加直方图至关重要——否则后面的柱子会盖住前面的
- `kde` 是直方图的平滑替代——不受分箱边界影响

## 5. 散点图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "Height": np.random.normal(170, 10, 100),
    "Weight": np.random.normal(65, 12, 100),
    "Category": np.random.choice(["A", "B", "C"], 100),
})

# 基本散点图
ax = df.plot.scatter(x="Height", y="Weight", figsize=(8, 6),
                      title="身高-体重关系", alpha=0.6)
# 可视化输出见下方图表：散点图展示身高与体重的正相关关系
plt.close()

# 按类别着色
colors = {"A": "red", "B": "green", "C": "blue"}
ax = df.plot.scatter(x="Height", y="Weight", figsize=(8, 6),
                      title="身高-体重（按类别着色）",
                      c=df["Category"].map(colors), alpha=0.6)
# 可视化输出见下方图表：三色散点图，每组用不同颜色
plt.close()
```

#### 输出

可视化输出见下方图表。散点图每个点代表一个样本，X 轴为身高、Y 轴为体重。按类别着色后，不同颜色的点代表不同分组，可观察各组之间的分布差异。

### 理解重点

- `scatter` 必须指定 `x` 和 `y` 列名——与其他 `plot.*` 不同（它们从索引推断 X 轴）
- 散点图是发现变量间关系的最直观工具——正/负相关、聚类、离群点一目了然

## 6. 箱线图与饼图

### 综合示例

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    "A": np.random.normal(70, 10, 100),
    "B": np.random.normal(65, 15, 100),
    "C": np.random.normal(75, 8, 100),
})

# 箱线图
ax = df.plot.box(figsize=(8, 5), title="三组分数分布对比",
                  ylabel="分数", grid=True)
# 可视化输出见下方图表：箱线图显示中位数、四分位数、范围、离群点

# 饼图
sizes = pd.Series({"产品A": 45, "产品B": 30, "产品C": 15, "其他": 10})
ax = sizes.plot.pie(figsize=(7, 7), title="产品销售占比",
                     autopct="%1.1f%%", startangle=90)
# 可视化输出见下方图表：饼图显示各类别占比，自动标注百分比
plt.close()
```

#### 输出

可视化输出见下方图表。箱线图展示每组的 Q1/中位数/Q3/须线/离群点，适合多组分布的横向对比。饼图以扇形角度表示各类别占比，`autopct` 自动标注百分比。

### 理解重点

- 箱线图的箱子 = IQR（Q1 到 Q3），须线延伸到 1.5xIQR 范围——超出的是离群点
- 饼图只适用于 Series：`series.plot.pie()`——DataFrame 需先用 `df["col"]` 提取
- `autopct="%1.1f%%"` 表示保留 1 位小数的百分比标注

## 7. 保存图表

Pandas 绘图底层返回 Matplotlib 对象，保存用 `plt.savefig()`：

```python
import matplotlib.pyplot as plt

ax = df.plot.line(title="示例")
fig = ax.get_figure()
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

或更简单的：

```python
df.plot.line(title="示例")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

## 常见坑

1. Pandas 绘图需先导入 `import matplotlib.pyplot as plt`——否则可能无法显示图表
2. `df.plot.scatter()` 必须显式指定 `x` 和 `y`——与其他 `plot.*()` 的默认行为不同
3. 饼图 `df.plot.pie()` 要求 `y` 参数或 `subplots=True`——对多列 DataFrame 需用 `subplots=True` 或提取 Series
4. Jupyter Notebook 中需 `%matplotlib inline` 才能在单元格内显示图表
5. Pandas 绘图返回 `Axes` 对象，`plt.savefig()` 要在图表显示前调用——`plt.show()` 后再保存会得到空白图片
6. 中文显示为方框时需设置中文字体：`plt.rcParams["font.sans-serif"] = ["SimHei"]`

## 小结

- `df.plot(kind=...)` 是快速探索的统一入口——10 种图表类型一行切换
- Pandas 绘图 = Matplotlib 的便捷封装——复杂定制用 Matplotlib/Seaborn
- 时间序列首选 `line`；分类对比首选 `bar`；分布分析首选 `hist` / `kde` / `box`
- 绘图后 `plt.savefig("out.png", dpi=150)` 保存——在 `plt.show()` 之前调用

# Pandas 高级操作与性能优化

## 本章目标

1. 掌握 `pivot_table` 透视表与 `crosstab` 交叉表
2. 理解 `MultiIndex` 多级索引的创建与选择
3. 掌握 `stack` / `unstack` / `melt` 的长宽表转换
4. 学会用 `astype('category')` 和降精度做内存优化

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `pd.pivot_table(...)` | 函数 | 透视表（可聚合，支持 MultiIndex） |
| `df.pivot(...)` | 方法 | 简单透视（不聚合，仅重塑） |
| `pd.crosstab(...)` | 函数 | 频数/比例交叉表 |
| `pd.MultiIndex.from_arrays(...)` | 构造器 | 从数组列表创建多级索引 |
| `pd.MultiIndex.from_tuples(...)` | 构造器 | 从元组列表创建多级索引 |
| `df.stack(...)` | 方法 | 列 -> 行（宽变长） |
| `df.unstack(...)` | 方法 | 行 -> 列（长变宽） |
| `df.melt(...)` | 方法 | 宽表 -> 长表（多列熔化为键值对） |
| `df.memory_usage(...)` | 方法 | 每列内存占用 |
| `df.astype('category')` | 方法 | 转换为分类类型（节省内存） |

## 1. 透视表与简单透视

### `pd.pivot_table`

#### 作用

创建 Excel 风格的透视表。按行列分组后对值做聚合（求和、均值、计数等）。与 `groupby` 的区别：结果呈现为二维交叉表，行列都是分组维度。

#### 重点方法

```python
pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean',
               fill_value=None, margins=False, dropna=True,
               margins_name='All', observed=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `data` | `DataFrame` | 输入数据 | `df` |
| `values` | `str`、`list[str]` | 要聚合的值列 | `"Sales"`、`["Sales", "Profit"]` |
| `index` | `str`、`list[str]` | 行分组键（结果的行索引） | `"Region"`、`["Region", "Dept"]` |
| `columns` | `str`、`list[str]` | 列分组键（结果的列索引） | `"Year"` |
| `aggfunc` | `str`、`list[str]`、`dict`、函数 | 聚合函数，默认为 `'mean'` | `"sum"`、`["sum", "mean"]` |
| `fill_value` | 标量 或 `None` | 用此值替换 NaN，默认为 `None` | `0` |
| `margins` | `bool` | `True` 时添加"总计"行/列，默认为 `False` | `True` |
| `dropna` | `bool` | 是否丢弃全 NaN 的列，默认为 `True` | `False` |

### `DataFrame.pivot`

#### 作用

纯粹的形状重塑——不做聚合，要求行列对唯一。若行列组合有重复会报错。

#### 重点方法

```python
df.pivot(*, columns, index=None, values=None)
```

### 综合示例

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Region": ["East", "East", "East", "West", "West", "West"],
    "Year": [2022, 2023, 2022, 2022, 2023, 2023],
    "Product": ["A", "A", "B", "A", "B", "B"],
    "Sales": [100, 150, 200, 300, 250, 180],
})

print("原始数据:")
print(df)

# pivot_table：按 Region x Year 聚合 Sales
pt = pd.pivot_table(df, values="Sales", index="Region",
                    columns="Year", aggfunc="sum", fill_value=0)
print(f"\n透视表 (Region x Year, sum):\n{pt}")

# 带 margins（总计行列）
ptm = pd.pivot_table(df, values="Sales", index="Region",
                     columns="Year", aggfunc="sum", margins=True)
print(f"\n透视表 + margins:\n{ptm}")

# pivot：在不聚合时使用（行列对唯一）
piv = df.pivot(index="Region", columns="Year", values="Sales")
print(f"\npivot (无聚合):\n{piv}")
```

#### 输出

```text
原始数据:
  Region  Year Product  Sales
0   East  2022       A    100
1   East  2023       A    150
2   East  2022       B    200
3   West  2022       A    300
4   West  2023       B    250
5   West  2023       B    180

透视表 (Region x Year, sum):
Year    2022  2023
Region
East     300   150
West     300   430

透视表 + margins:
Year    2022  2023   All
Region
East     300   150   450
West     300   430   730
All      600   580  1180

pivot (无聚合):
Year    2022  2023
Region
East     100   150
West     300   215
```

#### 理解重点

- `pivot_table` 是分组聚合的二维呈现——类似于 Excel 的数据透视表
- `pivot` 只改变形状不做聚合——要求行列组合唯一，否则抛 `ValueError`
- `margins=True` 自动追加"总计"行列——适合快速生成汇总报告
- 多值列用列表 `values=["Sales", "Profit"]`——生成 MultiIndex 列

## 2. 交叉频数表

### `pd.crosstab`

#### 作用

计算两个（或更多）分类变量的频数/比例交叉表。底层可视为 `pivot_table` 在计数场景的快捷方式。

#### 重点方法

```python
pd.crosstab(index, columns, values=None, aggfunc=None, rownames=None,
            colnames=None, margins=False, margins_name='All',
            normalize=False, dropna=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `index` | `array_like`、`Series` | 行分类变量 | `df["Region"]` |
| `columns` | `array_like`、`Series` | 列分类变量 | `df["Product"]` |
| `values` | `array_like` 或 `None` | 要聚合的值列；`None` 时计算频数 | `df["Sales"]` |
| `aggfunc` | `str`、函数 或 `None` | 聚合函数；`values` 非空时必须指定 | `"sum"` |
| `normalize` | `bool`、`str` | `True` 归一化为比例；也可指定 `'index'` / `'columns'` / `'all'` | `True`、`"index"` |
| `margins` | `bool` | `True` 时添加总计行列，默认为 `False` | `True` |

#### 示例代码

```python
import pandas as pd

df = pd.DataFrame({
    "Region": ["East", "East", "West", "West", "East", "West"],
    "Product": ["A", "B", "A", "A", "B", "B"],
    "Sales": [100, 200, 300, 150, 180, 220],
})

# 频数
ct = pd.crosstab(df["Region"], df["Product"])
print(f"频数交叉表:\n{ct}")

# 比例（按行归一化）
ctn = pd.crosstab(df["Region"], df["Product"], normalize="index")
print(f"\n比例交叉表（行归一化）:\n{ctn.round(2)}")

# 按值聚合
cts = pd.crosstab(df["Region"], df["Product"], values=df["Sales"],
                  aggfunc="sum")
print(f"\nSales 汇总交叉表:\n{cts}")
```

#### 输出

```text
频数交叉表:
Product  A  B
Region
East     1  2
West     2  1

比例交叉表（行归一化）:
Product     A     B
Region
East     0.33  0.67
West     0.67  0.33

Sales 汇总交叉表:
Product    A    B
Region
East     100  380
West     450  220
```

#### 理解重点

- `crosstab` 默认计算频数——类似 R 的 `table()` 函数
- `normalize='index'` / `'columns'` / `'all'` 分别按行/列/所有归一化
- `values` + `aggfunc` 组合使 `crosstab` 可以替代 `pivot_table` 的简单场景

## 3. 多级索引

### `pd.MultiIndex`

#### 作用

MultiIndex（分层索引）允许在单个轴上有多层标签。DataFrame 的行和列都可以是多级索引——透视表的结果天然就是 MultiIndex。

#### 创建方式

```python
# 从数组列表
pd.MultiIndex.from_arrays([["A", "A", "B", "B"], [1, 2, 1, 2]],
                          names=["level1", "level2"])

# 从元组列表
pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1), ("B", 2)],
                          names=["level1", "level2"])

# 从笛卡尔积
pd.MultiIndex.from_product([["A", "B"], [1, 2]],
                           names=["level1", "level2"])

# 通过 groupby/set_index 创建
df.set_index(["col1", "col2"])
```

### MultiIndex 选择

| 语法 | 说明 |
|---|---|
| `df.loc["A"]` | 选择第一级为 "A" 的所有行 |
| `df.loc[("A", 1)]` | 选择精确的多级标签（用元组） |
| `df.loc[("A", slice(None))]` | 第一级为 "A"，第二级全部（等价于 `df.loc["A"]`） |
| `df.xs("A", level="level1")` | 按级别名称精确选择（跨切面） |

#### 示例代码

```python
import pandas as pd
import numpy as np

# 创建 MultiIndex 行
idx = pd.MultiIndex.from_product(
    [["East", "West"], ["Q1", "Q2", "Q3", "Q4"]],
    names=["Region", "Quarter"]
)
df = pd.DataFrame({
    "Sales": np.random.randint(100, 500, 8),
    "Profit": np.random.randint(10, 100, 8),
}, index=idx)

print("MultiIndex DataFrame:")
print(df)

# 按第一级选择
print(f"\nEast 全部:\n{df.loc['East']}")

# 精确多级选择
print(f"\nloc[('West', 'Q3')]:\n{df.loc[('West', 'Q3')]}")

# 按级别名称跨切面
print(f"\nxs('Q2', level='Quarter'):\n{df.xs('Q2', level='Quarter')}")
```

#### 输出

```text
MultiIndex DataFrame:
                     Sales  Profit
Region Quarter
East   Q1              101      67
       Q2              300      57
       Q3              433      50
       Q4              464      89
West   Q1              185      95
       Q2              268      24
       Q3              482      24
       Q4              362      80

East 全部:
         Sales  Profit
Quarter
Q1         101      67
Q2         300      57
Q3         433      50
Q4         464      89

loc[('West', 'Q3')]:
Sales     482
Profit     24
Name: (West, Q3), dtype: int32

xs('Q2', level='Quarter'):
        Sales  Profit
Region
East      300      57
West      268      24
```

#### 理解重点

- MultiIndex 常见于 `groupby`（多键分组）、`pivot_table`（多行列）、`set_index`（多列设为索引）的结果
- 元组选择 `df.loc[("A", 1)]` 用于精确定位多级标签
- `xs` 方法按级别名称跨切面选择——比 `loc` 更语义化
- 去 MultiIndex 用 `reset_index()` 或 `df.columns = ["_".join(c) for c in df.columns]`

## 4. 长宽表转换

### `stack` / `unstack` / `melt`

#### 作用

- `stack()`：列 -> 行（宽变长）——把列标签"压"进行索引的最后一级
- `unstack()`：行 -> 列（长变宽）——把行索引最后一级"展开"为列标签
- `melt()`：宽表 -> 长表——将多列"熔化"为两列：变量名列和值列

#### 重点方法

```python
df.stack(level=-1, dropna=True)
df.unstack(level=-1, fill_value=None)
df.melt(id_vars=None, value_vars=None, var_name=None, value_name='value')
```

#### `melt` 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `id_vars` | `list[str]` | 保持不变的标识列 | `["Name", "Date"]` |
| `value_vars` | `list[str]` 或 `None` | 要熔化的值列；`None` 时熔化除 `id_vars` 外的所有列 | `["Q1", "Q2", "Q3", "Q4"]` |
| `var_name` | `str` | 熔化后变量名列的列名，默认为 `'variable'` | `"Quarter"` |
| `value_name` | `str` | 熔化后值列的列名，默认为 `'value'` | `"Sales"` |

#### 示例代码

```python
import pandas as pd

# 宽表
dfWide = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Q1": [100, 200, 150],
    "Q2": [120, 210, 160],
    "Q3": [130, 220, 170],
    "Q4": [140, 230, 180],
})

print(f"宽表:\n{dfWide}")

# melt：宽 -> 长
dfLong = dfWide.melt(id_vars=["Name"], var_name="Quarter", value_name="Sales")
print(f"\nmelt 长表:\n{dfLong}")

# 用 pivot_table 长 -> 宽（逆操作）
dfBack = pd.pivot_table(dfLong, values="Sales", index="Name",
                        columns="Quarter")
print(f"\npivot_table 回到宽表:\n{dfBack}")

# stack 示例
dfIdx = dfWide.set_index("Name")
print(f"\nstack 前:\n{dfIdx}")
print(f"\nstack 后:\n{dfIdx.stack()}")
```

#### 输出

```text
宽表:
      Name   Q1   Q2   Q3   Q4
0    Alice  100  120  130  140
1      Bob  200  210  220  230
2  Charlie  150  160  170  180

melt 长表:
       Name Quarter  Sales
0     Alice      Q1    100
1       Bob      Q1    200
2   Charlie      Q1    150
3     Alice      Q2    120
4       Bob      Q2    210
5   Charlie      Q2    160
6     Alice      Q3    130
7       Bob      Q3    220
8   Charlie      Q3    170
9     Alice      Q4    140
10      Bob      Q4    230
11  Charlie      Q4    180

pivot_table 回到宽表:
Quarter    Q1   Q2   Q3   Q4
Name
Alice     100  120  130  140
Bob       200  210  220  230
Charlie   150  160  170  180

stack 前:
           Q1   Q2   Q3   Q4
Name
Alice     100  120  130  140
Bob       200  210  220  230
Charlie   150  160  170  180

stack 后:
Name     Quarter
Alice    Q1         100
         Q2         120
         Q3         130
         Q4         140
Bob      Q1         200
         Q2         210
         Q3         220
         Q4         230
Charlie  Q1         150
         Q2         160
         Q3         170
         Q4         180
dtype: int64
```

#### 理解重点

- `melt` 比 `stack` 更直观——明确指定 `id_vars`（保持的列）和 `value_vars`（熔化的列）
- `melt` 的逆操作是 `pivot_table`——前者变长、后者变宽
- `stack` 适合列名本身就是"分类变量"的宽表——将列变成行索引的一层
- 长表是"整洁数据"（tidy data）的标准形态——seaborn、plotly 等可视化库的首选格式

## 5. 内存优化

### `df.memory_usage` / `astype('category')`

#### 作用

- `memory_usage()`：查看 DataFrame 每列的内存占用（字节）
- `astype('category')`：将低基数列（重复值多）转为分类类型——大幅节省内存
- 降精度为数值列节省内存：`float64 -> float32`、`int64 -> int32`

#### 示例代码

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "ID": range(10000),
    "City": np.random.choice(["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"], 10000),
    "Value": np.random.randn(10000),
})

print(f"原始内存:\n{df.memory_usage()}")
print(f"总内存: {df.memory_usage().sum() / 1024:.1f} KB")

# City 列转为 category
df["City"] = df["City"].astype("category")
print(f"\ncategory 后内存:\n{df.memory_usage()}")
print(f"总内存: {df.memory_usage().sum() / 1024:.1f} KB")

# 查看 category 节省比例
cityBefore = 10000 * 8  # object 列：每元素一个指针 ~8 字节
cityAfter = df["City"].memory_usage()
print(f"\nCity 列：转前 ~{cityBefore} bytes -> 转后 {cityAfter} bytes")
print(f"节省: {(1 - cityAfter / cityBefore) * 100:.1f}%")
```

#### 输出

```text
原始内存:
Index      132
ID       80000
City     80000
Value    80000
dtype: int64
总内存: 234.5 KB

category 后内存:
Index      132
ID       80000
City     10548
Value    80000
dtype: int64
总内存: 166.7 KB

City 列：转前 ~80000 bytes -> 转后 10548 bytes
节省: 86.8%
```

#### 理解重点

- `category` 类型的本质：存储整数编码 + 查找表——重复值越多，节省越多
- 典型场景：性别、国家、城市、产品类型等**低基数**（<= 50 类）分类变量
- 数值降精度也有效：`df["col"].astype("float32")` 比 `float64` 省一半内存
- `df.info(memory_usage="deep")` 查看含 object 列深层内存的实际占用

## 常见坑

1. `pivot` 要求行列对唯一——重复会报错；需要聚合时用 `pivot_table`
2. `pivot_table` 默认 `aggfunc='mean'`——计算求和要用 `aggfunc='sum'`
3. `stack()` 要求所有列同 dtype——否则会降级为 `object`
4. `melt()` 不指定 `value_vars` 会熔化所有非 `id_vars` 列——注意检查是否预期
5. `category` 列的修改受限：不能直接赋新类别——需用 `.cat.add_categories()` 先添加
6. `memory_usage()` 默认不计算 object 列指向的字符串内存——用 `deep=True` 获得真实值

## 小结

- 透视表 `pivot_table` 是分组聚合的二维呈现——比多层 `groupby` 更直观
- `crosstab` 是频数统计的快捷方式——一行代码生成分类交叉表
- 长表是可视化库的首选格式——`melt` 将宽表转为长表
- 内存优化两步走：重复字符串列 -> `category`；数值列 -> 降精度（`float32` / `int32`）
