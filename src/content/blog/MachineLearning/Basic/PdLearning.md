---
title: Pandas 全指南
date: 2026-01-13
category: MachineLearning/Basic
tags:
  - Python
  - Pandas
description: Pandas中文教程，有Series/DataFrame基础、数据I/O、选择过滤、清洗处理、分组聚合、合并连接、时间序列、可视化及高级性能优化等核心主题，附带可运行的代码练习。
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

# 数据导入与导出

## 学习目标

- 掌握 CSV、Excel、JSON 等格式的读写
- 了解常用读取参数

## 读取数据

### CSV 文件读取

#### 基本读取

CSV（Comma-Separated Values）是最常用的数据交换格式，纯文本文件，每行代表一条记录。

```python
import pandas as pd

# 最简单的读取方式
df = pd.read_csv('file.csv')
```

**默认行为**：

- 第一行作为列名（header）
- 逗号 `,` 作为分隔符
- 自动推断数据类型
- UTF-8 编码

#### 指定参数读取

```python
# 完整参数示例
df = pd.read_csv(
    'file.csv',
    encoding='utf-8',     # 编码格式
    sep=',',              # 分隔符
    header=0,             # 第0行是列名
    index_col=0           # 第0列作为索引
)
```

**假设 file.csv 内容**：

```
ID,Name,Age,City,Salary
1,Alice,25,Beijing,8000
2,Bob,30,Shanghai,12000
3,Charlie,35,Guangzhou,15000
```

**读取结果**：

```
   ID     Name  Age       City  Salary
0   1    Alice   25    Beijing    8000
1   2      Bob   30   Shanghai   12000
2   3  Charlie   35  Guangzhou   15000
```

### 常用参数详解

| 参数          | 类型 | 说明                       | 示例                              |
| ------------- | ---- | -------------------------- | --------------------------------- |
| `sep`         | str  | 列分隔符                   | `sep=','` 或 `sep='\t'`（制表符） |
| `header`      | int  | 指定哪一行作为列名         | `header=0`（第一行，默认）        |
| `names`       | list | 自定义列名（覆盖原有列名） | `names=['A', 'B', 'C']`           |
| `usecols`     | list | 只读取指定的列             | `usecols=['Name', 'Age']`         |
| `nrows`       | int  | 只读取前 n 行              | `nrows=1000`（读取前1000行）      |
| `skiprows`    | int  | 跳过开头的 n 行            | `skiprows=2`（跳过前2行）         |
| `encoding`    | str  | 文件编码格式               | `encoding='utf-8'` 或 `'gbk'`     |
| `na_values`   | list | 指定哪些值视为缺失值       | `na_values=['NA', 'null', '']`    |
| `dtype`       | dict | 指定列的数据类型           | `dtype={'Age': int, 'Name': str}` |
| `parse_dates` | list | 将指定列解析为日期类型     | `parse_dates=['Date']`            |

#### 实用示例

**示例 1：读取大文件的一部分**

```python
# 只读取前 1000 行和指定的列
df = pd.read_csv(
    'large_file.csv',
    usecols=['Name', 'Age', 'Salary'],
    nrows=1000
)
```

**原理**：处理大文件时，限制行数和列数可以节省内存，加快加载速度。

**示例 2：处理中文编码问题**

```python
# 中文 CSV 常用 gbk 或 gb2312 编码
df = pd.read_csv('chinese_data.csv', encoding='gbk')
```

**说明**：Windows 系统生成的 CSV 文件通常使用 GBK 编码，如果出现乱码，尝试更改编码。

**示例 3：自定义列名**

```python
# 文件没有表头，手动指定列名
df = pd.read_csv(
    'no_header.csv',
    header=None,  # 没有表头
    names=['姓名', '年龄', '城市']
)
```

**示例 4：处理不同分隔符**

```python
# 读取制表符分隔的文件（TSV）
df = pd.read_csv('file.tsv', sep='\t')

# 读取分号分隔的文件
df = pd.read_csv('file.txt', sep=';')
```

### Excel 文件读取

#### 基本用法

```python
# 读取 Excel 文件（默认读取第一个工作表）
df = pd.read_excel('file.xlsx')

# 指定工作表名称
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# 指定工作表索引（0 表示第一个工作表）
df = pd.read_excel('file.xlsx', sheet_name=0)
```

**注意**：需要安装 `openpyxl` 或 `xlrd` 库：

```bash
pip install openpyxl  # 用于 .xlsx 文件
pip install xlrd      # 用于旧版 .xls 文件
```

#### 读取多个工作表

```python
# 读取所有工作表（返回字典）
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)

# 访问特定工作表
df1 = all_sheets['Sheet1']
df2 = all_sheets['Sheet2']

print(f"工作表名称: {all_sheets.keys()}")
```

**输出示例**：

```
工作表名称: dict_keys(['Sheet1', 'Sheet2', 'Sheet3'])
```

**应用场景**：一个 Excel 文件包含多个相关数据表时批量读取。

#### Excel 特有参数

```python
df = pd.read_excel(
    'file.xlsx',
    sheet_name='销售数据',
    header=0,           # 表头行
    index_col=0,        # 索引列
    usecols='A:D',      # 只读取 A 到 D 列
    skiprows=2,         # 跳过前两行
    nrows=100           # 只读取100行
)
```

**原理**：Excel 文件可能包含标题、说明等非数据行，使用 `skiprows` 可以跳过这些行。

### JSON 文件读取

#### 基本用法

JSON（JavaScript Object Notation）是一种轻量级的数据交换格式。

```python
# 读取 JSON 文件
df = pd.read_json('file.json')
```

#### JSON 格式类型

**1. 记录格式（orient='records'）**

```json
[
  { "Name": "Alice", "Age": 25, "City": "Beijing" },
  { "Name": "Bob", "Age": 30, "City": "Shanghai" }
]
```

```python
df = pd.read_json('file.json', orient='records')
```

**结果**：

```
    Name  Age      City
0  Alice   25   Beijing
1    Bob   30  Shanghai
```

**2. 列格式（orient='columns'，默认）**

```json
{
  "Name": { "0": "Alice", "1": "Bob" },
  "Age": { "0": 25, "1": 30 },
  "City": { "0": "Beijing", "1": "Shanghai" }
}
```

**3. 索引格式（orient='index'）**

```json
{
  "0": { "Name": "Alice", "Age": 25 },
  "1": { "Name": "Bob", "Age": 30 }
}
```

**说明**：不同的 JSON 结构需要指定正确的 `orient` 参数才能正确解析。

### 其他常见格式

#### SQL 数据库

```python
import sqlite3

# 从 SQLite 数据库读取
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
conn.close()
```

**原理**：执行 SQL 查询，将结果转换为 DataFrame。

#### HTML 表格

```python
# 从网页中提取表格
tables = pd.read_html('https://example.com/data.html')
df = tables[0]  # 获取第一个表格
```

**说明**：自动识别网页中的 `<table>` 标签并解析。

#### 剪贴板

```python
# 从剪贴板读取（复制 Excel 数据后）
df = pd.read_clipboard()
```

**应用**：快速测试，从 Excel 复制数据后直接粘贴到代码中。

## 导出数据

数据处理完成后，需要将结果保存到文件中供后续使用。

### CSV 文件导出

```python
# 基本导出
df.to_csv('output.csv')
```

**默认行为**：

- 包含行索引（第一列）
- 使用逗号分隔
- UTF-8 编码

**结果示例** `output.csv`：

```
,Name,Age,City,Salary
0,Alice,25,Beijing,8000
1,Bob,30,Shanghai,12000
2,Charlie,35,Guangzhou,15000
```

#### 常用参数

```python
# 不保存行索引（推荐）
df.to_csv('output.csv', index=False)

# 指定编码（中文数据）
df.to_csv('output.csv', encoding='utf-8-sig')  # Excel 可正确打开中文

# 指定分隔符
df.to_csv('output.tsv', sep='\t', index=False)  # 制表符分隔

# 只保存部分列
df.to_csv('output.csv', columns=['Name', 'Age'], index=False)

# 追加到现有文件
df.to_csv('output.csv', mode='a', header=False, index=False)
```

**参数说明**：

- `index=False`：不保存行索引（避免多余的一列）
- `encoding='utf-8-sig'`：带 BOM 的 UTF-8，Excel 能正确识别中文
- `mode='a'`：追加模式（append），不覆盖原文件
- `header=False`：不写入列名（追加时避免重复表头）

**最佳实践**：

```python
# 导出供 Excel 使用的中文 CSV
df.to_csv('中文数据.csv', index=False, encoding='utf-8-sig')
```

### Excel 文件导出

```python
# 导出到 Excel
df.to_excel('output.xlsx', sheet_name='数据', index=False)
```

**优点**：

- 保留格式（与 CSV 相比）
- 支持多个工作表
- Excel 可直接打开，无编码问题

#### 导出多个工作表

```python
# 使用 ExcelWriter 导出多个工作表
with pd.ExcelWriter('report.xlsx') as writer:
    df1.to_excel(writer, sheet_name='销售数据', index=False)
    df2.to_excel(writer, sheet_name='统计汇总', index=False)
    df3.to_excel(writer, sheet_name='图表数据', index=False)
```

**输出结果**：一个 Excel 文件包含 3 个工作表。

**应用场景**：生成数据报告，不同工作表存储不同类型的数据。

### JSON 文件导出

```python
# 导出为 JSON（记录格式）
df.to_json('output.json', orient='records', force_ascii=False, indent=2)
```

**参数说明**：

- `orient='records'`：每行数据是一个字典（最常用格式）
- `force_ascii=False`：保留中文字符（不转义）
- `indent=2`：缩进格式化（美化输出）

**输出结果** `output.json`：

```json
[
  {
    "Name": "Alice",
    "Age": 25,
    "City": "Beijing",
    "Salary": 8000
  },
  {
    "Name": "Bob",
    "Age": 30,
    "City": "Shanghai",
    "Salary": 12000
  }
]
```

**应用场景**：数据传输、Web API、配置文件。

### 其他格式导出

#### HTML 表格

```python
# 导出为 HTML 表格
df.to_html('table.html', index=False)
```

**应用**：嵌入网页展示数据。

#### SQL 数据库

```python
import sqlite3

# 写入 SQLite 数据库
conn = sqlite3.connect('database.db')
df.to_sql('table_name', conn, if_exists='replace', index=False)
conn.close()
```

**参数说明**：

- `if_exists='replace'`：替换已存在的表
- `if_exists='append'`：追加数据
- `if_exists='fail'`：如果表存在则报错（默认）

### 导出格式对比

| 格式  | 优点                     | 缺点                 | 适用场景         |
| ----- | ------------------------ | -------------------- | ---------------- |
| CSV   | 轻量、通用、纯文本       | 无格式、编码问题     | 数据交换、存储   |
| Excel | 保留格式、多工作表、易用 | 文件大、需要软件打开 | 报告、展示       |
| JSON  | 结构化、Web 友好         | 文件较大、不直观     | API、配置、传输  |
| SQL   | 高效查询、大数据量       | 需要数据库环境       | 持久化存储、共享 |

### 快速保存和加载（Pickle）

```python
# 保存为 Pickle（保留所有 Pandas 信息）
df.to_pickle('data.pkl')

# 加载 Pickle
df = pd.read_pickle('data.pkl')
```

**优点**：

- 速度快
- 完全保留数据类型、索引等所有信息
- 压缩效率高

**缺点**：

- 只能在 Python 中使用
- 不同 Pandas 版本可能不兼容

**应用场景**：Python 项目内部的数据传递和缓存。

# 数据选择与过滤

## 学习目标

- 掌握列选择和行选择
- 理解 loc 和 iloc 的区别
- 学会条件过滤

## 列选择

列选择是数据分析中最基本的操作，用于提取需要的字段。

### 预备数据

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou'],
    'Salary': [8000, 12000, 15000, 10000, 13000]
}
df = pd.DataFrame(data)
print(df)
```

**输出**：

```
      Name  Age       City  Salary
0    Alice   25    Beijing    8000
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
3    David   28   Shenzhen   10000
4      Eve   32   Hangzhou   13000
```

### 1. 选择单列

```python
# 方法 1：使用单个方括号（返回 Series）
name_series = df['Name']
print(type(name_series))  # <class 'pandas.core.series.Series'>
print(name_series)
```

**输出**：

```
0      Alice
1        Bob
2    Charlie
3      David
4        Eve
Name: Name, dtype: object
```

**说明**：

- 单个方括号返回 **Series** 类型（一维数据）
- Series 包含索引和值，有 `name` 属性

```python
# 方法 2：使用点符号（仅当列名符合 Python 变量命名规则）
name_series = df.Name
print(name_series)
```

**注意**：

- 点符号方式更简洁，但有限制：
  - 列名不能包含空格或特殊字符
  - 列名不能与 DataFrame 方法重名（如 `df.count` 会执行方法而非访问列）

### 2. 选择多列

```python
# 使用列表（返回 DataFrame）
subset = df[['Name', 'Age']]
print(type(subset))  # <class 'pandas.core.frame.DataFrame'>
print(subset)
```

**输出**：

```
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
3    David   28
4      Eve   32
```

**重要区别**：

- `df['Name']` → 返回 **Series**（一维）
- `df[['Name']]` → 返回 **DataFrame**（二维）

```python
# 证明区别
print(type(df['Name']))     # Series
print(type(df[['Name']]))   # DataFrame
```

### 3. 选择列的应用场景

```python
# 场景 1：只需要部分字段进行分析
analysis_cols = ['Age', 'Salary']
df_analysis = df[analysis_cols]
print(df_analysis.describe())

# 场景 2：列重排序
df_reordered = df[['Name', 'City', 'Age', 'Salary']]

# 场景 3：列重命名（通过复制）
df_renamed = df[['Name', 'Age']].copy()
df_renamed.columns = ['姓名', '年龄']
```

## 行选择

行选择用于提取数据的子集。

### 1. 切片选择

```python
# 使用切片符号 [:]（基于位置）
print(df[0:3])  # 选择第 0, 1, 2 行（不包括 3）
```

**输出**：

```
      Name  Age       City  Salary
0    Alice   25    Beijing    8000
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
```

**原理**：

- 切片语法：`[start:end:step]`
- `start` 包含，`end` 不包含（左闭右开）
- 类似 Python 列表的切片

```python
# 其他切片示例
print(df[1:4])    # 第 1, 2, 3 行
print(df[::2])    # 每隔2行取一次（0, 2, 4...）
print(df[-2:])    # 最后 2 行
```

### 2. 使用 head() 和 tail()

```python
# 查看前 3 行
print(df.head(3))
```

**输出**：

```
      Name  Age       City  Salary
0    Alice   25    Beijing    8000
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
```

```python
# 查看后 2 行
print(df.tail(2))
```

**输出**：

```
  Name  Age      City  Salary
3  David   28  Shenzhen   10000
4    Eve   32  Hangzhou   13000
```

**应用场景**：

- 快速预览数据
- 验证数据加载是否正确
- 检查数据的开头和结尾

### 3. 随机抽样

```python
# 随机选择 2 行
print(df.sample(n=2, random_state=42))
```

**输出**（示例）：

```
      Name  Age      City  Salary
3    David   28  Shenzhen   10000
0    Alice   25   Beijing    8000
```

**参数说明**：

- `n=2`：抽取 2 行
- `random_state=42`：设置随机种子，保证结果可复现
- `frac=0.5`：抽取 50% 的数据

**应用**：数据抽样、模型训练验证。

## loc 和 iloc：精确定位数据

`loc` 和 `iloc` 是 Pandas 中最强大的数据选择工具。

### 核心区别

| 方法   | 索引类型 | 包含 end | 示例                     |
| ------ | -------- | -------- | ------------------------ |
| `loc`  | **标签** | **是**   | `df.loc['row1', 'Name']` |
| `iloc` | **位置** | 否       | `df.iloc[0, 1]`          |

### loc：标签索引

#### 基本用法

```python
# 设置有意义的索引
df_indexed = df.set_index('Name')
print(df_indexed)
```

**输出**：

```
         Age       City  Salary
Name
Alice     25    Beijing    8000
Bob       30   Shanghai   12000
Charlie   35  Guangzhou   15000
David     28   Shenzhen   10000
Eve       32   Hangzhou   13000
```

#### 选择单个元素

```python
# 选择 Bob 的 Age
age = df_indexed.loc['Bob', 'Age']
print(age)  # 30
```

**原理**：`loc[行标签, 列标签]`

#### 选择多行多列

```python
# 选择多个人的多个字段
subset = df_indexed.loc[['Alice', 'Charlie'], ['Age', 'City']]
print(subset)
```

**输出**：

```
         Age       City
Name
Alice     25    Beijing
Charlie   35  Guangzhou
```

#### 切片选择（包含端点）

```python
# 注意：loc 的切片是包含结束标签的！
subset = df_indexed.loc['Alice':'Charlie', 'Age':'City']
print(subset)
```

**输出**：

```
         Age       City  Salary
Name
Alice     25    Beijing    8000
Bob       30   Shanghai   12000
Charlie   35  Guangzhou   15000
```

**重要**：

- `loc` 的切片是 **包含两端** 的（不同于 Python 列表）
- `'Alice':'Charlie'` 包含 Alice、Bob、Charlie

#### 选择所有行或列

```python
# 选择所有行的 Age 列
ages = df_indexed.loc[:, 'Age']
print(ages)

# 选择 Alice 的所有列
alice_data = df_indexed.loc['Alice', :]
print(alice_data)
```

**说明**：`:` 表示选择所有。

### iloc：位置索引

`iloc` 基于整数位置，类似 NumPy 数组的索引。

#### 选择单个元素

```python
# 选择第 1 行第 0 列（Bob 的 Name）
value = df.iloc[1, 0]
print(value)  # 'Bob'
```

**原理**：`iloc[行位置, 列位置]`（从 0 开始）

#### 选择多行多列

```python
# 选择第 0, 2 行的第 0, 1 列
subset = df.iloc[[0, 2], [0, 1]]
print(subset)
```

**输出**：

```
      Name  Age
0    Alice   25
2  Charlie   35
```

#### 切片选择（不包含端点）

```python
# 选择第 0-2 行（不包括 3）的第 0-1 列（不包括 2）
subset = df.iloc[0:3, 0:2]
print(subset)
```

**输出**：

```
      Name  Age
0    Alice   25
1      Bob   30
2  Charlie   35
```

**注意**：`iloc` 的切片遵循 Python 规则，**左闭右开**。

#### 选择整行或整列

```python
# 选择第 2 行的所有列
row = df.iloc[2, :]
print(row)

# 选择第 1 列的所有行
column = df.iloc[:, 1]
print(column)
```

### loc vs iloc 对比示例

```python
# 创建一个带自定义索引的 DataFrame
df_custom = pd.DataFrame(
    {'A': [1, 2, 3], 'B': [4, 5, 6]},
    index=['x', 'y', 'z']
)
print(df_custom)
```

**输出**：

```
   A  B
x  1  4
y  2  5
z  3  6
```

```python
# loc 使用标签
print(df_custom.loc['x', 'A'])  # 1
print(df_custom.loc['x':'y', :])  # 包含 x 和 y

# iloc 使用位置
print(df_custom.iloc[0, 0])  # 1
print(df_custom.iloc[0:2, :])  # 不包含位置 2
```

### 选择方法总结

| 场景                | 使用方法                  |
| ------------------- | ------------------------- |
| 通过标签名选择      | `df.loc[行标签, 列标签]`  |
| 通过整数位置选择    | `df.iloc[行位置, 列位置]` |
| 需要切片包含结束点  | 使用 `loc`                |
| 跟 NumPy 一致的索引 | 使用 `iloc`               |
| 不确定索引类型      | 尽量明确使用 loc/iloc     |

## 条件过滤

条件过滤是数据分析中最常用的操作，用于筛选满足特定条件的数据。

### 1. 单条件过滤

```python
# 筛选年龄大于 28 的人
filtered = df[df['Age'] > 28]
print(filtered)
```

**输出**：

```
      Name  Age       City  Salary
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
4      Eve   32   Hangzhou   13000
```

**原理解析**：

```python
# 步骤 1：创建布尔索引
boolean_mask = df['Age'] > 28
print(boolean_mask)
```

**输出**：

```
0    False
1     True
2     True
3    False
4     True
Name: Age, dtype: bool
```

```python
# 步骤 2：使用布尔索引选择数据
filtered = df[boolean_mask]
```

**说明**：True 的行会被选中，False 的行会被过滤掉。

### 2. 多条件过滤

#### AND 条件（&）

```python
# 筛选年龄 > 25 且薪水 > 10000 的人
filtered = df[(df['Age'] > 25) & (df['Salary'] > 10000)]
print(filtered)
```

**输出**：

```
      Name  Age       City  Salary
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
4      Eve   32   Hangzhou   13000
```

**注意**：

- 必须使用 `&` 而不是 `and`
- 每个条件必须用括号包裹：`(df['Age'] > 25)`

#### OR 条件（|）

```python
# 筛选年龄 < 26 或薪水 > 14000 的人
filtered = df[(df['Age'] < 26) | (df['Salary'] > 14000)]
print(filtered)
```

**输出**：

```
      Name  Age       City  Salary
0    Alice   25    Beijing    8000
2  Charlie   35  Guangzhou   15000
```

#### NOT 条件（~）

```python
# 筛选城市不是 Beijing 的人
filtered = df[~(df['City'] == 'Beijing')]
print(filtered)
```

**说明**：`~` 表示取反，将 True 变为 False，False 变为 True。

### 3. 使用 isin() 方法

```python
# 筛选城市在指定列表中的人
cities = ['Beijing', 'Shanghai', 'Shenzhen']
filtered = df[df['City'].isin(cities)]
print(filtered)
```

**输出**：

```
    Name  Age      City  Salary
0  Alice   25   Beijing    8000
1    Bob   30  Shanghai   12000
3  David   28  Shenzhen   10000
```

**原理**：`isin()` 检查每个值是否在给定列表中，等价于多个 OR 条件。

```python
# 取反：不在列表中
filtered = df[~df['City'].isin(cities)]
print(filtered)
```

### 4. 字符串过滤

```python
# 筛选城市名包含 'Sh' 的人
filtered = df[df['City'].str.contains('Sh')]
print(filtered)
```

**输出**：

```
    Name  Age      City  Salary
1    Bob   30  Shanghai   12000
3  David   28  Shenzhen   10000
```

**说明**：`str.contains()` 用于字符串模糊匹配。

```python
# 不区分大小写
filtered = df[df['City'].str.contains('sh', case=False)]

# 正则表达式
filtered = df[df['City'].str.contains(r'^S', regex=True)]  # 以 S 开头
```

### 5. 使用 query() 方法

`query()` 提供了更简洁的语法，类似 SQL WHERE 子句。

```python
# 基本用法
filtered = df.query('Age > 28 and Salary > 10000')
print(filtered)
```

**输出**：

```
      Name  Age       City  Salary
1      Bob   30   Shanghai   12000
2  Charlie   35  Guangzhou   15000
4      Eve   32   Hangzhou   13000
```

**优点**：

- 语法简洁，不需要 `df[]` 和多个括号
- 可以直接使用 `and`/`or`/`not`
- 更接近自然语言

```python
# 使用变量
min_age = 28
filtered = df.query('Age > @min_age')

# 复杂条件
filtered = df.query('(Age > 28 and Salary > 10000) or City == "Beijing"')

# 字符串匹配
filtered = df.query('City in ["Beijing", "Shanghai"]')
```

### 6. 结合 loc 进行过滤

```python
# 过滤后选择特定列
filtered = df.loc[df['Age'] > 28, ['Name', 'City']]
print(filtered)
```

**输出**：

```
      Name       City
1      Bob   Shanghai
2  Charlie  Guangzhou
4      Eve   Hangzhou
```

**说明**：在过滤的同时选择所需的列，一步完成。

### 过滤方法对比

| 方法                   | 优点             | 缺点           | 适用场景   |
| ---------------------- | ---------------- | -------------- | ---------- |
| `df[condition]`        | 简单直接         | 多条件复杂     | 单条件过滤 |
| `df.query()`           | 语法简洁         | 不支持所有操作 | 多条件过滤 |
| `df.loc[condition]`    | 可同时选择列     | 语法较长       | 精确选择   |
| `df[col].isin(list)`   | 处理多值匹配     | 仅适用于单列   | 值在列表中 |
| `df[col].str.method()` | 强大的字符串功能 | 仅适用于字符串 | 文本处理   |

### 常见错误

```python
# ✘ 错误：使用 and 而不是 &
df[df['Age'] > 25 and df['Salary'] > 10000]  # 报错！

# ✔ 正确
df[(df['Age'] > 25) & (df['Salary'] > 10000)]

# ✘ 错误：缺少括号
df[df['Age'] > 25 & df['Salary'] > 10000]  # 报错！

# ✔ 正确：每个条件都要括号
df[(df['Age'] > 25) & (df['Salary'] > 10000)]
```

# 数据清洗与处理

## 学习目标

- 掌握缺失值检测和处理
- 学会去除重复值
- 了解数据类型转换和字符串操作

## 缺失值处理

缺失值（Missing Values）是数据分析中最常见的问题之一，必须正确处理。

### 1. 检测缺失值

```python
import pandas as pd
import numpy as np

# 创建包含缺失值的数据
data = {
    'Name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'Age': [25, None, 35, 28, 32],
    'City': ['Beijing', 'Shanghai', 'Guangzhou', None, 'Hangzhou'],
    'Salary': [8000, 12000, None, 10000, 13000]
}
df = pd.DataFrame(data)
print(df)
```

**输出**：

```
    Name   Age       City   Salary
0  Alice  25.0    Beijing   8000.0
1    Bob   NaN   Shanghai  12000.0
2   None  35.0  Guangzhou      NaN
3  David  28.0       None  10000.0
4    Eve  32.0   Hangzhou  13000.0
```

**说明**：Pandas 使用 `NaN`（Not a Number）或 `None` 表示缺失值。

```python
# 检测每个元素是否为缺失值（返回布尔矩阵）
print(df.isnull())
# 或使用 df.isna()（同义词）
```

**输出**：

```
    Name    Age   City  Salary
0  False  False  False   False
1  False   True  False   False
2   True  False  False    True
3  False  False   True   False
4  False  False  False   False
```

```python
# 统计每列的缺失值数量
print(df.isnull().sum())
```

**输出**：

```
Name      1
Age       1
City      1
Salary    1
dtype: int64
```

**应用**：快速了解哪些列有缺失值，缺失多少。

```python
# 缺失值比例
print(df.isnull().sum() / len(df))

# 检查是否有任何缺失值
print(df.isnull().any())  # 每列是否有缺失值
print(df.isnull().any().any())  # 整个 DataFrame 是否有缺失值
```

### 2. 删除缺失值

```python
# 删除包含任何缺失值的行
df_dropped = df.dropna()
print(df_dropped)
```

**输出**：

```
    Name   Age      City   Salary
0  Alice  25.0   Beijing   8000.0
4    Eve  32.0  Hangzhou  13000.0
```

**说明**：只保留所有列都不为空的行。

```python
# 删除所有值都是缺失值的行
df_dropped = df.dropna(how='all')

# 删除特定列为空的行
df_dropped = df.dropna(subset=['Age', 'Salary'])

# 删除包含缺失值的列
df_dropped = df.dropna(axis=1)
```

**参数详解**：

- `how='any'`：有任何缺失值就删除（默认）
- `how='all'`：所有值都是缺失值才删除
- `subset=['col']`：只考虑指定列
- `axis=1`：删除列而非行

### 3. 填充缺失值

#### 固定值填充

```python
# 用 0 填充
df_filled = df.fillna(0)
print(df_filled)
```

**输出**：

```
    Name   Age       City   Salary
0  Alice  25.0    Beijing   8000.0
1    Bob   0.0   Shanghai  12000.0
2      0  35.0  Guangzhou      0.0
3  David  28.0          0  10000.0
4    Eve  32.0   Hangzhou  13000.0
```

#### 不同列使用不同值填充

```python
# 为不同列指定不同的填充值
fill_values = {
    'Name': 'Unknown',
    'Age': df['Age'].mean(),      # 用平均值填充
    'City': 'Unknown',
    'Salary': df['Salary'].median()  # 用中位数填充
}
df_filled = df.fillna(fill_values)
print(df_filled)
```

**原理**：

- 数值型列常用平均值、中位数填充
- 分类型列常用众数或默认值填充

#### 前向/后向填充

```python
# 前向填充（用上一个非空值填充）
df_ffill = df.fillna(method='ffill')
print(df_ffill)
```

**输出示例**：

```
    Name   Age       City   Salary
0  Alice  25.0    Beijing   8000.0
1    Bob  25.0   Shanghai  12000.0  # Age 用上一行的 25
2    Bob  35.0  Guangzhou  12000.0  # Name 和 Salary 被填充
...
```

```python
# 后向填充（用下一个非空值填充）
df_bfill = df.fillna(method='bfill')
```

**应用场景**：时间序列数据，前后值具有相关性。

### 4. 插值填充

```python
# 线性插值
df['Age'] = df['Age'].interpolate(method='linear')

# 多项式插值
df['Salary'] = df['Salary'].interpolate(method='polynomial', order=2)
```

**原理**：基于已知数据点，使用数学方法推算缺失值。

## 重复值处理

重复数据会影响统计分析结果，需要及时清除。

### 1. 检测重复值

```python
# 创建包含重复值的数据
data = {
    'Name': ['Alice', 'Bob', 'Alice', 'David', 'Bob'],
    'Age': [25, 30, 25, 28, 30],
    'City': ['Beijing', 'Shanghai', 'Beijing', 'Shenzhen', 'Shanghai']
}
df = pd.DataFrame(data)
print(df)
```

**输出**：

```
    Name  Age      City
0  Alice   25   Beijing
1    Bob   30  Shanghai
2  Alice   25   Beijing  # 完全重复
3  David   28  Shenzhen
4    Bob   30  Shanghai  # 完全重复
```

```python
# 检测重复行（返回布尔值）
print(df.duplicated())
```

**输出**：

```
0    False  # 第一次出现，不算重复
1    False
2     True  # 与第0行重复
3    False
4     True  # 与第1行重复
dtype: bool
```

**原理**：默认保留第一次出现的记录，后续重复标记为 True。

```python
# 统计重复行数
print(f"重复行数: {df.duplicated().sum()}")

# 查看重复的行
print(df[df.duplicated()])
```

### 2. 删除重复值

```python
# 删除完全重复的行
df_unique = df.drop_duplicates()
print(df_unique)
```

**输出**：

```
    Name  Age      City
0  Alice   25   Beijing
1    Bob   30  Shanghai
3  David   28  Shenzhen
```

**说明**：保留每组重复值的第一次出现。

```python
# 保留最后一次出现
df_unique = df.drop_duplicates(keep='last')

# 删除所有重复行（包括第一次出现）
df_unique = df.drop_duplicates(keep=False)
```

### 3. 基于特定列去重

```python
# 只基于 Name 列去重
df_unique = df.drop_duplicates(subset=['Name'])
print(df_unique)
```

**输出**：

```
    Name  Age      City
0  Alice   25   Beijing
1    Bob   30  Shanghai
3  David   28  Shenzhen
```

**应用场景**：根据主键去重，如身份证号、用户ID等。

```python
# 基于多列去重
df_unique = df.drop_duplicates(subset=['Name', 'City'])
```

## 数据类型转换

正确的数据类型对于性能和功能至关重要。

### 1. 查看数据类型

```python
print(df.dtypes)
```

**输出示例**：

```
Name      object   # 字符串
Age        int64   # 整数
City      object   # 字符串
Salary   float64   # 浮点数
dtype: object
```

### 2. 基本类型转换

```python
# 字符串转整数
df['Age'] = df['Age'].astype(int)

# 整数转字符串
df['ID'] = df['ID'].astype(str)

# 转为浮点数
df['Salary'] = df['Salary'].astype(float)

# 转为分类类型（节省内存）
df['City'] = df['City'].astype('category')
```

**分类类型的优势**：

- 节省内存（内部存储为整数）
- 提高性能
- 适用于有限集合的值（如性别、城市等）

### 3. 日期时间转换

```python
# 字符串转日期
df['Date'] = pd.to_datetime(df['Date'])

# 指定日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 处理错误日期
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 错误转为 NaT
```

**示例**：

```python
dates = pd.Series(['2024-01-01', '2024-02-15', 'invalid', '2024-03-20'])
dates_converted = pd.to_datetime(dates, errors='coerce')
print(dates_converted)
```

**输出**：

```
0   2024-01-01
1   2024-02-15
2          NaT  # Not a Time
3   2024-03-20
dtype: datetime64[ns]
```

### 4. 批量类型转换

```python
# 一次转换多列
df = df.astype({
    'Age': int,
    'Salary': float,
    'City': 'category',
    'Active': bool
})
```

## 字符串操作

字符串处理是数据清洗的重要部分，使用 `.str` 访问器。

### 1. 基本字符串操作

```python
# 创建测试数据
df = pd.DataFrame({
    'Name': ['  Alice  ', 'BOB', 'charlie', 'David '],
    'Email': ['alice@example.com', 'bob@TEST.com', 'charlie@', 'david@example.com']
})
```

```python
# 去除空格
df['Name'] = df['Name'].str.strip()  # 去除两端空格
print(df['Name'])
```

**输出**：

```
0      Alice
1        BOB
2    charlie
3      David
Name: Name, dtype: object
```

```python
# 大小写转换
df['Name_lower'] = df['Name'].str.lower()    # 转小写
df['Name_upper'] = df['Name'].str.upper()    # 转大写
df['Name_title'] = df['Name'].str.title()    # 首字母大写
df['Name_cap'] = df['Name'].str.capitalize() # 句首大写
```

### 2. 字符串检查

```python
# 包含匹配
has_example = df['Email'].str.contains('example')
print(df[has_example])
```

**输出**：

```
     Name                Email
0   Alice  alice@example.com
3   David  david@example.com
```

```python
# 以特定字符串开头/结尾
starts_with_a = df['Name'].str.startswith('A')
ends_with_com = df['Email'].str.endswith('.com')

# 长度
df['Name_len'] = df['Name'].str.len()
```

### 3. 字符串分割和提取

```python
# 分割字符串
email_parts = df['Email'].str.split('@', expand=True)
email_parts.columns = ['Username', 'Domain']
print(email_parts)
```

**输出**：

```
   Username           Domain
0     alice      example.com
1       bob         TEST.com
2   charlie             None
3     david      example.com
```

**原理**：`expand=True` 将分割结果展开为多列。

```python
# 提取特定位置的字符
first_char = df['Name'].str[0]        # 第一个字符
first_three = df['Name'].str[:3]      # 前三个字符

# 提取每个分割后的第一部分
usernames = df['Email'].str.split('@').str[0]
```

### 4. 字符串替换

```python
# 普通替换
df['Email'] = df['Email'].str.replace('TEST', 'example')

# 正则表达式替换
df['Email'] = df['Email'].str.replace(r'\s+', '', regex=True)  # 去除所有空格

# 多个替换
df['Name'] = df['Name'].str.replace('Bob', 'Robert').str.replace('charlie', 'Charlie')
```

### 5. 正则表达式提取

```python
# 提取邮箱域名
domains = df['Email'].str.extract(r'@([\w.]+)', expand=False)

# 提取数字
text = pd.Series(['abc123', 'def456', 'ghi789'])
numbers = text.str.extract(r'(\d+)', expand=False)
print(numbers)  # ['123', '456', '789']
```

**说明**：使用圆括号 `()` 标记要提取的部分。

## 值替换

值替换用于数据标准化和编码转换。

### 1. 单值替换

```python
# 替换单个值
df['City'] = df['City'].replace('BJ', 'Beijing')
```

### 2. 多值替换

```python
# 列表到列表
df['Status'] = df['Status'].replace(
    ['Y', 'N', 'Unknown'],
    ['Yes', 'No', 'Not Sure']
)
```

### 3. 字典替换

```python
# 使用字典映射
replace_dict = {
    'yes': 1,
    'no': 0,
    'maybe': 0.5
}
df['Response'] = df['Response'].replace(replace_dict)
```

**输出示例**：

```python
# 原数据: ['yes', 'no', 'yes', 'maybe']
# 替换后: [1, 0, 1, 0.5]
```

### 4. 多列替换

```python
# 同时替换多列
df.replace({
    'Gender': {'M': 'Male', 'F': 'Female'},
    'Status': {'A': 'Active', 'I': 'Inactive'}
})
```

### 5. 条件替换

```python
# 使用 where 进行条件替换
df['Age_Group'] = df['Age'].where(df['Age'] < 30, 'Senior')
df['Age_Group'] = df['Age_Group'].where(df['Age'] >= 30, 'Junior')

# 使用 np.where
import numpy as np
df['Age_Group'] = np.where(
    df['Age'] < 30, 'Junior',
    np.where(df['Age'] < 50, 'Mid', 'Senior')
)
```

**说明**：嵌套的 `np.where` 实现多条件分类。

## 数据清洗最佳实践

### 清洗流程

```python
def clean_dataframe(df):
    """ 数据清洗通用流程 """
    # 1. 复制数据避免修改原始数据
    df_clean = df.copy()

    # 2. 处理列名（去除空格、转小写）
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3. 删除完全重复的行
    df_clean = df_clean.drop_duplicates()

    # 4. 处理缺失值
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    # 5. 处理字符串列
    string_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[string_cols] = df_clean[string_cols].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

    # 6. 转换数据类型
    # 根据实际情况调整

    return df_clean
```

### 注意事项

1. **始终使用 `.copy()`**：避免修改原始数据
2. **记录清洗步骤**：便于追溯和复现
3. **验证结果**：清洗后检查数据质量
4. **备份原始数据**：清洗前保存副本

# 数据分组与聚合

## 学习目标

- 掌握 groupby 分组操作
- 学会使用聚合函数
- 理解 transform 和 apply 的区别

## GroupBy 基本操作

GroupBy 是 Pandas 中最强大的功能之一，用于分组聚合操作，遵循 **"split-apply-combine"（拆分-应用-合并）** 模式。

### 1. 创建测试数据

```python
import pandas as pd
import numpy as np

# 创建员工数据
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Department': ['IT', 'HR', 'IT', 'HR', 'IT', 'Sales'],
    'Salary': [8000, 6000, 9000, 6500, 8500, 7000],
    'Bonus': [1000, 800, 1200, 900, 1100, 1000],
    'Years': [2, 5, 3, 4, 2, 6]
}
df = pd.DataFrame(data)
print(df)
```

**输出**：

```
      Name Department  Salary  Bonus  Years
0    Alice         IT    8000   1000      2
1      Bob         HR    6000    800      5
2  Charlie         IT    9000   1200      3
3    David         HR    6500    900      4
4      Eve         IT    8500   1100      2
5    Frank      Sales    7000   1000      6
```

### 2. 基本分组

```python
# 按部门分组
grouped = df.groupby('Department')
print(type(grouped))  # <class 'pandas.core.groupby.generic.DataFrameGroupBy'>
```

**说明**：`groupby()` 返回一个 GroupBy 对象，并不立即计算结果（惰性计算）。

```python
# 查看分组数量
print(f"分组数: {grouped.ngroups}")  # 3 个部门

# 查看各组大小
print(grouped.size())
```

**输出**：

```
Department
HR       2
IT       3
Sales    1
dtype: int64
```

### 3. 遍历分组

```python
# 遍历每个组
for name, group in grouped:
    print(f"\n{name} 部门:")
    print(group)
```

**输出**：

```
HR 部门:
    Name Department  Salary  Bonus  Years
1    Bob         HR    6000    800      5
3  David         HR    6500    900      4

IT 部门:
      Name Department  Salary  Bonus  Years
0    Alice         IT    8000   1000      2
2  Charlie         IT    9000   1200      3
4      Eve         IT    8500   1100      2

Sales 部门:
    Name Department  Salary  Bonus  Years
5  Frank      Sales    7000   1000      6
```

**应用**：对每个组执行自定义操作。

### 4. 获取单个组

```python
# 获取 IT 部门的数据
it_group = grouped.get_group('IT')
print(it_group)
```

**输出**：

```
      Name Department  Salary  Bonus  Years
0    Alice         IT    8000   1000      2
2  Charlie         IT    9000   1200      3
4      Eve         IT    8500   1100      2
```

## 聚合函数

聚合函数用于将每个组的数据计算为单个值。

### 1. 单一聚合函数

```python
# 计算每个部门的平均薪水
avg_salary = grouped['Salary'].mean()
print(avg_salary)
```

**输出**：

```
Department
HR       6250.0
IT       8500.0
Sales    7000.0
Name: Salary, dtype: float64
```

**原理**：

1. 按 Department 分组
2. 对每组的 Salary 列计算平均值
3. 返回以部门为索引的 Series

```python
# 其他常用聚合函数
print(grouped['Salary'].sum())      # 总和
print(grouped['Salary'].max())      # 最大值
print(grouped['Salary'].min())      # 最小值
print(grouped['Salary'].count())    # 计数
print(grouped['Salary'].std())      # 标准差
```

### 2. 多个聚合函数

```python
# 同时应用多个聚合函数
salary_stats = grouped['Salary'].agg(['sum', 'mean', 'max', 'min', 'count'])
print(salary_stats)
```

**输出**：

```
              sum     mean   max   min  count
Department
HR          12500   6250.0  6500  6000      2
IT          25500   8500.0  9000  8000      3
Sales        7000   7000.0  7000  7000      1
```

**说明**：`agg()` 可以接受函数列表，一次性计算多个统计量。

### 3. 多列不同聚合

```python
# 为不同的列应用不同的聚合函数
result = grouped.agg({
    'Salary': ['mean', 'sum'],      # 薪水：平均值和总和
    'Bonus': 'sum',                  # 奖金：总和
    'Years': ['mean', 'max']         # 工龄：平均值和最大值
})
print(result)
```

**输出**：

```
             Salary          Bonus Years
               mean    sum   sum  mean  max
Department
HR           6250.0  12500  1700   4.5    5
IT           8500.0  25500  3300   2.3    3
Sales        7000.0   7000  1000   6.0    6
```

**应用场景**：生成综合性的统计报表。

### 4. 自定义聚合函数

```python
# 使用 lambda 函数
result = grouped['Salary'].agg(lambda x: x.max() - x.min())
print(result)
```

**输出**：

```
Department
HR        500   # 6500 - 6000
IT       1000   # 9000 - 8000
Sales       0   # 7000 - 7000
Name: Salary, dtype: int64
```

**说明**：计算每个部门的薪水差距。

```python
# 命名自定义函数
def salary_range(x):
    return x.max() - x.min()

result = grouped['Salary'].agg(
    平均薪水='mean',
    薪水差距=salary_range
)
print(result)
```

## Transform：保持原始形状

`transform()` 返回与原始数据**相同长度**的结果，常用于添加组统计量到原始数据。

### Transform vs Aggregation

```python
# 聚合：返回每组一个值
agg_result = grouped['Salary'].mean()
print(f"Aggregation 结果长度: {len(agg_result)}")  # 3

# Transform：返回与原数据相同长度
trans_result = grouped['Salary'].transform('mean')
print(f"Transform 结果长度: {len(trans_result)}")  # 6
print(trans_result)
```

**输出**：

```
Aggregation 结果长度: 3
Transform 结果长度: 6

0    8500.0  # Alice 所在 IT 部门的平均薪水
1    6250.0  # Bob 所在 HR 部门的平均薪水
2    8500.0  # Charlie 所在 IT 部门的平均薪水
3    6250.0  # David 所在 HR 部门的平均薪水
4    8500.0  # Eve 所在 IT 部门的平均薪水
5    7000.0  # Frank 所在 Sales 部门的平均薪水
Name: Salary, dtype: float64
```

### 实际应用

```python
# 添加部门平均薪水列
df['Dept_Mean_Salary'] = grouped['Salary'].transform('mean')
print(df)
```

**输出**：

```
      Name Department  Salary  Bonus  Years  Dept_Mean_Salary
0    Alice         IT    8000   1000      2            8500.0
1      Bob         HR    6000    800      5            6250.0
2  Charlie         IT    9000   1200      3            8500.0
3    David         HR    6500    900      4            6250.0
4      Eve         IT    8500   1100      2            8500.0
5    Frank      Sales    7000   1000      6            7000.0
```

```python
# 计算与部门平均值的偏离
df['Salary_Diff'] = df['Salary'] - df['Dept_Mean_Salary']
print(df[['Name', 'Salary', 'Dept_Mean_Salary', 'Salary_Diff']])
```

**输出**：

```
      Name  Salary  Dept_Mean_Salary  Salary_Diff
0    Alice    8000            8500.0         -500.0
1      Bob    6000            6250.0         -250.0
2  Charlie    9000            8500.0          500.0
3    David    6500            6250.0          250.0
4      Eve    8500            8500.0            0.0
5    Frank    7000            7000.0            0.0
```

**应用场景**：

- 标准化（每组单独标准化）
- 缺失值填充（用组平均值）
- 特征工程（添加组统计量）

## Apply：灵活应用自定义函数

`apply()` 可以应用任意函数，返回任意形状的结果。

### 1. 返回标量值

```python
# 查找每个部门薪水最高的员工
def top_employee(group):
    return group.nlargest(1, 'Salary')

top_employees = grouped.apply(top_employee)
print(top_employees)
```

**输出**：

```
                 Name Department  Salary  Bonus  Years
Department
HR         3    David         HR    6500    900      4
IT         2  Charlie         IT    9000   1200      3
Sales      5    Frank      Sales    7000   1000      6
```

**说明**：返回每个部门薪水最高的员工信息。

### 2. 返回 Series

```python
# 计算每个部门的多个指标
def dept_summary(group):
    return pd.Series({
        '人数': len(group),
        '总薪水': group['Salary'].sum(),
        '平均工龄': group['Years'].mean(),
        '最高薪水': group['Salary'].max()
    })

summary = grouped.apply(dept_summary)
print(summary)
```

**输出**：

```
             人数    总薪水  平均工龄  最高薪水
Department
HR           2.0  12500.0      4.5    6500.0
IT           3.0  25500.0      2.3    9000.0
Sales        1.0   7000.0      6.0    7000.0
```

### 3. Apply vs Transform vs Agg

| 方法          | 返回形状         | 灵活性 | 应用场景       |
| ------------- | ---------------- | ------ | -------------- |
| `agg()`       | 每组一个值       | 低     | 简单聚合统计   |
| `transform()` | 与原数据相同长度 | 中     | 添加组统计量   |
| `apply()`     | 任意形状         | 高     | 复杂自定义逻辑 |

### 4. 多列分组

```python
# 按多个列分组
df['Experience'] = df['Years'].apply(lambda x: 'Senior' if x >= 4 else 'Junior')
multi_grouped = df.groupby(['Department', 'Experience'])
print(multi_grouped['Salary'].mean())
```

**输出**：

```
Department  Experience
HR          Junior        6500.0
            Senior        6000.0
IT          Junior        8500.0
Sales       Senior        7000.0
Name: Salary, dtype: float64
```

# 数据合并与连接

## 学习目标

- 掌握 concat 纵向和横向合并
- 理解 merge 的各种连接方式
- 学会使用 join 操作

## Concat 合并

`concat()` 用于沿着轴方向拼接数据，类似于 NumPy 的 `concatenate`。

### 1. 准备测试数据

```python
import pandas as pd

# 创建两个 DataFrame
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=[0, 1, 2])

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
}, index=[3, 4, 5])

print("df1:")
print(df1)
print("\ndf2:")
print(df2)
```

**输出**：

```
df1:
    A   B
0  A0  B0
1  A1  B1
2  A2  B2

df2:
    A   B
3  A3  B3
4  A4  B4
5  A5  B5
```

### 2. 纵向合并（axis=0，默认）

```python
# 纵向堆叠（沿着行方向）
result = pd.concat([df1, df2], axis=0)
print(result)
```

**输出**：

```
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
5  A5  B5
```

**原理**：沿着行方向（axis=0）拼接，保留原始索引。

```python
# 重置索引
result = pd.concat([df1, df2], axis=0, ignore_index=True)
print(result)
```

**输出**：

```
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
5  A5  B5
```

**说明**：`ignore_index=True` 重新生成连续的索引。

### 3. 横向合并（axis=1）

```python
# 准备横向合并的数据
df3 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
}, index=[0, 1, 2])

# 横向拼接（沿着列方向）
result = pd.concat([df1, df3], axis=1)
print(result)
```

**输出**：

```
    A   B   C   D
0  A0  B0  C0  D0
1  A1  B1  C1  D1
2  A2  B2  C2  D2
```

**应用**：添加新的列到现有 DataFrame。

### 4. 处理不匹配的索引

```python
df4 = pd.DataFrame({
    'C': ['C3', 'C4'],
    'D': ['D3', 'D4']
}, index=[1, 3])  # 索引不完全匹配

result = pd.concat([df1, df4], axis=1)
print(result)
```

**输出**：

```
     A    B    C    D
0   A0   B0  NaN  NaN
1   A1   B1   C3   D3
2   A2   B2  NaN  NaN
3  NaN  NaN   C4   D4
```

**原理**：默认使用外连接（outer join），保留所有索引，缺失值填充 NaN。

```python
# 使用内连接（只保留共同索引）
result = pd.concat([df1, df4], axis=1, join='inner')
print(result)
```

**输出**：

```
    A   B   C   D
1  A1  B1  C3  D3
```

### 5. 添加分层索引

```python
# 为合并后的数据添加标签
result = pd.concat([df1, df2], keys=['batch1', 'batch2'])
print(result)
```

**输出**：

```
          A   B
batch1 0  A0  B0
       1  A1  B1
       2  A2  B2
batch2 3  A3  B3
       4  A4  B4
       5  A5  B5
```

**应用**：追踪数据来源，创建多级索引。

## Merge 连接

Merge 类似 SQL 的 JOIN 操作，用于根据共同列合并数据。

### 1. 准备测试数据

```python
# 员工基本信息
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [10, 20, 10, 30]
})

# 部门信息
departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['IT', 'HR', 'Sales']
})

print("employees:")
print(employees)
print("\ndepartments:")
print(departments)
```

**输出**：

```
employees:
   emp_id     name  dept_id
0       1    Alice       10
1       2      Bob       20
2       3  Charlie       10
3       4    David       30

departments:
   dept_id dept_name
0       10        IT
1       20        HR
2       40     Sales
```

### 2. 内连接（Inner Join）

```python
# 内连接：只保留共同的键
result = pd.merge(employees, departments, on='dept_id', how='inner')
print(result)
```

**输出**：

```
   emp_id     name  dept_id dept_name
0       1    Alice       10        IT
1       3  Charlie       10        IT
2       2      Bob       20        HR
```

**原理**：

- 只保留两个表中 dept_id 都存在的记录
- David（dept_id=30）和 Sales部门（dept_id=40）被过滤

### 3. 左连接（Left Join）

```python
# 左连接：保留左表所有记录
result = pd.merge(employees, departments, on='dept_id', how='left')
print(result)
```

**输出**：

```
   emp_id     name  dept_id dept_name
0       1    Alice       10        IT
1       2      Bob       20        HR
2       3  Charlie       10        IT
3       4    David       30       NaN
```

**说明**：

- 保留 employees 表的所有记录
- David 的 dept_id=30 在 departments 中不存在，dept_name 填充 NaN

### 4. 右连接（Right Join）

```python
# 右连接：保留右表所有记录
result = pd.merge(employees, departments, on='dept_id', how='right')
print(result)
```

**输出**：

```
   emp_id     name  dept_id dept_name
0     1.0    Alice       10        IT
1     3.0  Charlie       10        IT
2     2.0      Bob       20        HR
3     NaN      NaN       40     Sales
```

**说明**：保留 departments 表的所有记录，Sales 部门没有员工。

### 5. 外连接（Outer Join）

```python
# 外连接：保留两表所有记录
result = pd.merge(employees, departments, on='dept_id', how='outer')
print(result)
```

**输出**：

```
   emp_id     name  dept_id dept_name
0     1.0    Alice       10        IT
1     2.0      Bob       20        HR
2     3.0  Charlie       10        IT
3     4.0    David       30       NaN
4     NaN      NaN       40     Sales
```

**说明**：保留两个表的所有记录，缺失值填充 NaN。

### 连接方式对比

| 连接方式 | SQL 等价        | 说明             | 使用场景                 |
| -------- | --------------- | ---------------- | ------------------------ |
| `inner`  | INNER JOIN      | 只保留匹配的记录 | 需要两表都有的数据       |
| `left`   | LEFT JOIN       | 保留左表所有记录 | 主表为基准，添加关联信息 |
| `right`  | RIGHT JOIN      | 保留右表所有记录 | 参考表为基准             |
| `outer`  | FULL OUTER JOIN | 保留两表所有记录 | 不丢失任何数据           |

### 6. 不同列名的合并

```python
# 两表的键列名不同
orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'employee_id': [1, 2, 1],
    'amount': [1000, 1500, 2000]
})

result = pd.merge(
    employees,
    orders,
    left_on='emp_id',    # employees 表的键
    right_on='employee_id',  # orders 表的键
    how='inner'
)
print(result)
```

**输出**：

```
   emp_id   name  dept_id  order_id  employee_id  amount
0       1  Alice       10       101            1    1000
1       1  Alice       10       103            1    2000
2       2    Bob       20       102            2    1500
```

### 7. 多个键的合并

```python
# 根据多个列合并
result = pd.merge(df1, df2, on=['key1', 'key2'], how='inner')
```

### 8. 合并指示器

```python
# 添加标记列，显示数据来源
result = pd.merge(
    employees,
    departments,
    on='dept_id',
    how='outer',
    indicator=True
)
print(result)
```

**输出**：

```
   emp_id     name  dept_id dept_name      _merge
0     1.0    Alice       10        IT        both
1     2.0      Bob       20        HR        both
2     3.0  Charlie       10        IT        both
3     4.0    David       30       NaN   left_only
4     NaN      NaN       40     Sales  right_only
```

**\_merge 列值的含义**：

- `both`：在两个表中都存在
- `left_only`：只在左表中存在
- `right_only`：只在右表中存在

**应用**：识别数据质量问题，检查未匹配的记录。

## Join 操作

Join 是基于**索引**的合并，与 Merge 的区别是不需要指定列名。

### 1. 基本 Join

```python
# 准备数据（使用有意义的索引）
df_left = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
}, index=['K0', 'K1', 'K2'])

df_right = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
}, index=['K0', 'K2', 'K3'])

print("df_left:")
print(df_left)
print("\ndf_right:")
print(df_right)
```

**输出**：

```
df_left:
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2

df_right:
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3
```

### 2. 左连接（默认）

```python
result = df_left.join(df_right)
print(result)
```

**输出**：

```
     A   B    C    D
K0  A0  B0   C0   D0
K1  A1  B1  NaN  NaN
K2  A2  B2   C2   D2
```

**原理**：默认使用左连接，保留左表的所有索引。

### 3. 其他连接方式

```python
# 内连接
result = df_left.join(df_right, how='inner')
print(result)
```

**输出**：

```
     A   B   C   D
K0  A0  B0  C0  D0
K2  A2  B2  C2  D2
```

```python
# 外连接
result = df_left.join(df_right, how='outer')
print(result)
```

**输出**：

```
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3
```

### 4. Join vs Merge

| 特性     | Join            | Merge                |
| -------- | --------------- | -------------------- |
| 基于     | 索引（index）   | 列（column）         |
| 默认方式 | 左连接          | 内连接               |
| 语法     | `df1.join(df2)` | `pd.merge(df1, df2)` |
| 灵活性   | 较低            | 较高                 |
| 适用场景 | 索引对齐的数据  | 通过列关联的数据     |

**最佳实践**：

- 索引对齐时使用 `join()`，语法简洁
- 需要根据列值匹配时使用 `merge()`，更灵活

### 5. 实际应用示例

```python
# 场景：合并员工基本信息和薪资信息
basic_info = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}, index=[1, 2, 3])

salary_info = pd.DataFrame({
    'salary': [8000, 12000, 15000],
    'bonus': [1000, 1500, 2000]
}, index=[1, 2, 3])

# 使用 join 合并（索引已对齐）
employee_data = basic_info.join(salary_info)
print(employee_data)
```

**输出**：

```
      name  age  salary  bonus
1    Alice   25    8000   1000
2      Bob   30   12000   1500
3  Charlie   35   15000   2000
```

## 数据合并最佳实践

### 1. 选择合适的方法

```python
# 堆叠数据：使用 concat
pd.concat([df1, df2, df3])

# SQL 式关联：使用 merge
pd.merge(employees, departments, on='dept_id')

# 索引对齐：使用 join
df1.join(df2)
```

### 2. 注意事项

1. **检查重复键**：合并前确保键列无重复

```python
print(df['key'].duplicated().any())
```

2. **验证结果**：检查合并后的行数

```python
print(f"合并前: {len(df1)} + {len(df2)}")
print(f"合并后: {len(result)}")
```

3. **处理缺失值**：合并后可能产生 NaN

```python
result = result.fillna(0)  # 或其他适当值
```

4. **性能优化**：大数据集合并时

```python
# 确保键列类型一致
df1['key'] = df1['key'].astype(int)
df2['key'] = df2['key'].astype(int)

# 对大表进行预过滤
df_filtered = df[df['date'] > '2023-01-01']
```

```bash
python Basic/Pandas/06_merge.py
```

# 时间序列处理

## 学习目标

- 掌握时间序列的创建和索引
- 学会时间重采样
- 理解滚动窗口操作

## 创建时间序列

时间序列是按时间顺序排列的数据点集合，Pandas 提供了强大的时间序列处理能力。

### 1. 创建日期时间对象

```python
import pandas as pd
import numpy as np

# 字符串转换为日期
dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
print(dates)
print(type(dates[0]))  # <class 'pandas._libs.tslibs.timestamps.Timestamp'>
```

**输出**：

```
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'],
              dtype='datetime64[ns]', freq=None)
```

**说明**：`to_datetime()` 将字符串转换为 Pandas 的 Timestamp 对象。

### 2. 生成日期范围

```python
# 生成 10 天的日期序列
date_range = pd.date_range('2024-01-01', periods=10, freq='D')
print(date_range)
```

**输出**：

```
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
               '2024-01-09', '2024-01-10'],
              dtype='datetime64[ns]', freq='D')
```

```python
# 指定开始和结束日期
date_range = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print(f"总共 {len(date_range)} 天")
```

### 3. 常用频率参数

| 频率         | 说明                   | 示例                                                 |
| ------------ | ---------------------- | ---------------------------------------------------- |
| `D`          | 天（Day）              | `pd.date_range('2024-01-01', periods=7, freq='D')`   |
| `H`          | 小时（Hour）           | `pd.date_range('2024-01-01', periods=24, freq='H')`  |
| `T` 或 `min` | 分钟                   | `pd.date_range('2024-01-01', periods=60, freq='T')`  |
| `S`          | 秒（Second）           | `pd.date_range('2024-01-01', periods=60, freq='S')`  |
| `W`          | 周（Week）             | `pd.date_range('2024-01-01', periods=4, freq='W')`   |
| `ME`         | 月末（Month End）      | `pd.date_range('2024-01-01', periods=12, freq='ME')` |
| `MS`         | 月初（Month Start）    | `pd.date_range('2024-01-01', periods=12, freq='MS')` |
| `QE`         | 季度末                 | `pd.date_range('2024-01-01', periods=4, freq='QE')`  |
| `YE`         | 年末                   | `pd.date_range('2024-01-01', periods=5, freq='YE')`  |
| `B`          | 工作日（Business Day） | `pd.date_range('2024-01-01', periods=10, freq='B')`  |

```python
# 工作日示例（自动跳过周末）
business_days = pd.date_range('2024-01-01', periods=10, freq='B')
print(business_days)
```

**输出**：

```
DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
               '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10',
               '2024-01-11', '2024-01-12'],
              dtype='datetime64[ns]', freq='B')
```

**注意**：2024-01-06和07是周六日，被自动跳过。

### 4. 创建时间序列 DataFrame

```python
# 创建带时间索引的 DataFrame
dates = pd.date_range('2024-01-01', periods=10, freq='D')
ts = pd.DataFrame({
    'sales': np.random.randint(100, 200, 10),
    'visitors': np.random.randint(50, 100, 10)
}, index=dates)
print(ts.head())
```

**输出示例**：

```
            sales  visitors
2024-01-01    156        87
2024-01-02    123        65
2024-01-03    189        92
2024-01-04    145        78
2024-01-05    167        83
```

## 时间索引属性

时间索引提供了丰富的属性访问，方便提取日期信息。

### 1. 基本时间属性

```python
# 创建时间序列数据
dates = pd.date_range('2024-01-15', periods=5, freq='D')
ts = pd.Series(range(5), index=dates)
print(ts)
```

**输出**：

```
2024-01-15    0
2024-01-16    1
2024-01-17    2
2024-01-18    3
2024-01-19    4
Freq: D, dtype: int64
```

```python
# 访问时间属性
print(f"年份: {ts.index.year.tolist()}")       # [2024, 2024, ...]
print(f"月份: {ts.index.month.tolist()}")      # [1, 1, ...]
print(f"日期: {ts.index.day.tolist()}")        # [15, 16, 17, 18, 19]
print(f"星期几: {ts.index.dayofweek.tolist()}")  # 0=周一, 6=周日
print(f"年中第几天: {ts.index.dayofyear.tolist()}")
```

**输出**：

```
年份: [2024, 2024, 2024, 2024, 2024]
月份: [1, 1, 1, 1, 1]
日期: [15, 16, 17, 18, 19]
星期几: [0, 1, 2, 3, 4]  # 周一到周五
年中第几天: [15, 16, 17, 18, 19]
```

### 2. 常用属性汇总

```python
# 创建 DataFrame 添加时间特征
df = pd.DataFrame({'value': range(5)}, index=dates)
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['weekday'] = df.index.dayofweek
df['weekday_name'] = df.index.day_name()  # 星期名称
df['is_weekend'] = df.index.dayofweek >= 5  # 是否周末
print(df)
```

**输出**：

```
            value  year  month  day  weekday weekday_name  is_weekend
2024-01-15      0  2024      1   15        0       Monday       False
2024-01-16      1  2024      1   16        1      Tuesday       False
2024-01-17      2  2024      1   17        2    Wednesday       False
2024-01-18      3  2024      1   18        3     Thursday       False
2024-01-19      4  2024      1   19        4       Friday       False
```

**应用场景**：

- 按周几分组统计
- 筛选工作日/周末数据
- 生成时间特征用于机器学习

## 时间切片

Pandas 对时间序列数据提供了非常灵活的切片功能。

### 1. 按年月选择

```python
# 创建一年的数据
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
ts = pd.Series(np.random.randn(len(dates)), index=dates)

# 选择整个月份
jan_data = ts['2024-01']
print(f"1月数据点数: {len(jan_data)}")
```

**输出**：

```
1月数据点数: 31
```

**原理**：使用部分字符串匹配，自动选择该月所有数据。

```python
# 选择整年
year_data = ts['2024']
print(f"全年数据点数: {len(year_data)}")
```

### 2. 日期范围选择

```python
# 选择日期范围
range_data = ts['2024-01-15':'2024-01-20']
print(range_data)
```

**输出**：

```
2024-01-15   -0.234
2024-01-16    1.123
2024-01-17   -0.456
2024-01-18    0.789
2024-01-19   -0.321
2024-01-20    0.654
Freq: D, dtype: float64
```

### 3. 精确时间点选择

```python
# 选择单个日期
value = ts['2024-01-15']
print(f"2024-01-15 的值: {value}")

# 使用 loc
value = ts.loc['2024-01-15']
```

### 4. 条件筛选

```python
# 筛选工作日
weekday_data = ts[ts.index.dayofweek < 5]
print(f"工作日数据点数: {len(weekday_data)}")

# 筛选特定月份
march_data = ts[ts.index.month == 3]
print(f"3月数据点数: {len(march_data)}")
```

## 重采样（Resample）

重采样用于改变时间序列的频率，是时间序列分析的核心功能。

### 1. 降采样（Down-sampling）

```python
# 创建小时级数据
dates = pd.date_range('2024-01-01', periods=168, freq='h')  # 7天，每小时一条
ts = pd.Series(np.random.randn(168), index=dates)
print(f"原始数据点数: {len(ts)}")

# 按天聚合（求平均）
daily = ts.resample('D').mean()
print(f"按天聚合后: {len(daily)} 天")
print(daily.head())
```

**输出**：

```
原始数据点数: 168
按天聚合后: 7 天

2024-01-01    0.123
2024-01-02   -0.456
2024-01-03    0.789
2024-01-04   -0.234
2024-01-05    0.567
Freq: D, dtype: float64
```

**原理**：将每 24 个小时的数据聚合为一天的平均值。

### 2. 不同聚合函数

```python
# 按周聚合
weekly_sum = ts.resample('W').sum()      # 求和
weekly_mean = ts.resample('W').mean()    # 平均值
weekly_max = ts.resample('W').max()      # 最大值
weekly_count = ts.resample('W').count()  # 计数

print(f"每周数据点数: {weekly_count.tolist()}")
```

### 3. 多列数据重采样

```python
# 创建多列数据
dates = pd.date_range('2024-01-01', periods=24, freq='h')
df = pd.DataFrame({
    'temperature': np.random.uniform(15, 25, 24),
    'humidity': np.random.uniform(40, 80, 24),
    'sales': np.random.randint(100, 200, 24)
}, index=dates)

# 不同列使用不同聚合函数
daily = df.resample('D').agg({
    'temperature': 'mean',  # 温度取平均
    'humidity': 'mean',     # 湿度取平均
    'sales': 'sum'          # 销售额求和
})
print(daily)
```

**输出示例**：

```
            temperature  humidity  sales
2024-01-01        20.3      62.1   3456
```

### 4. 升采样（Up-sampling）

```python
# 创建每天一条的数据
dates = pd.date_range('2024-01-01', periods=7, freq='D')
ts = pd.Series(range(7), index=dates)
print("原始数据:")
print(ts)

# 升采样到每 6 小时
hourly = ts.resample('6h').ffill()  # 前向填充
print(f"\n升采样后数据点数: {len(hourly)}")
print(hourly.head(10))
```

**输出**：

```
原始数据:
2024-01-01    0
2024-01-02    1
...

升采样后数据点数: 28
2024-01-01 00:00:00    0
2024-01-01 06:00:00    0
2024-01-01 12:00:00    0
2024-01-01 18:00:00    0
2024-01-02 00:00:00    1
...
```

**填充方法**：

- `ffill()`：前向填充
- `bfill()`：后向填充
- `interpolate()`：插值

## 滚动窗口（Rolling Window）

滚动窗口用于计算移动统计量，常用于平滑数据和趋势分析。

### 1. 简单移动平均（SMA）

```python
# 创建数据
dates = pd.date_range('2024-01-01', periods=10, freq='D')
ts = pd.Series([10, 12, 15, 14, 16, 18, 17, 19, 21, 20], index=dates)
print("原始数据:")
print(ts)

# 3期移动平均
ma3 = ts.rolling(window=3).mean()
print("\n3期移动平均:")
print(ma3)
```

**输出**：

```
原始数据:
2024-01-01    10
2024-01-02    12
2024-01-03    15
...

3期移动平均:
2024-01-01         NaN  # 前两个值不足，无法计算
2024-01-02         NaN
2024-01-03   12.333333  # (10+12+15)/3
2024-01-04   13.666667  # (12+15+14)/3
2024-01-05   15.000000  # (15+14+16)/3
...
```

**原理**：每次计算最近 3 个数据点的平均值。

### 2. 其他滚动统计量

```python
ts.rolling(3).sum()     # 移动求和
ts.rolling(3).max()     # 移动最大值
ts.rolling(3).min()     # 移动最小值
ts.rolling(3).std()     # 移动标准差
ts.rolling(3).count()   # 移动计数
```

### 3. 指数加权移动平均（EWMA）

```python
# 指数加权移动平均（近期数据权重更大）
ewma = ts.ewm(span=3).mean()
print("EWMA:")
print(ewma)
```

**输出**：

```
2024-01-01    10.000
2024-01-02    11.000
2024-01-03    13.000
2024-01-04    13.500
...
```

**应用场景**：

- 股票价格平滑
- 销售额趋势分析
- 信号去噪

### 4. 滚动窗口应用

```python
# 计算滚动相关系数
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100)
}, index=pd.date_range('2024-01-01', periods=100))

corr = df['A'].rolling(window=20).corr(df['B'])
print(corr.tail())
```

## 时间偏移（Shift）

时间偏移用于计算同比、环比等指标。

### 1. 基本偏移

```python
# 创建数据
dates = pd.date_range('2024-01-01', periods=5, freq='D')
ts = pd.Series([100, 105, 110, 108, 112], index=dates)
print("原始数据:")
print(ts)

# 向后偏移 1 期
ts_shifted = ts.shift(1)
print("\n向后偏移 1 期:")
print(ts_shifted)
```

**输出**：

```
原始数据:
2024-01-01    100
2024-01-02    105
2024-01-03    110
2024-01-04    108
2024-01-05    112

向后偏移 1 期:
2024-01-01    NaN
2024-01-02    100.0
2024-01-03    105.0
2024-01-04    110.0
2024-01-05    108.0
```

### 2. 计算变化率

```python
# 计算百分比变化
pct_change = ts.pct_change()
print("百分比变化:")
print(pct_change)
```

**输出**：

```
2024-01-01         NaN
2024-01-02    0.050000  # (105-100)/100 = 5%
2024-01-03    0.047619  # (110-105)/105 = 4.76%
2024-01-04   -0.018182  # (108-110)/110 = -1.82%
2024-01-05    0.037037  # (112-108)/108 = 3.70%
```

### 3. 计算差分

```python
# 第一次差分（当前值 - 前一个值）
diff = ts.diff()
print("差分:")
print(diff)
```

**输出**：

```
2024-01-01    NaN
2024-01-02    5.0   # 105 - 100
2024-01-03    5.0   # 110 - 105
2024-01-04   -2.0   # 108 - 110
2024-01-05    4.0   # 112 - 108
```

### 4. 向前偏移

```python
# 向前偏移（负数）
ts_future = ts.shift(-1)
print("向前偏移 1 期:")
print(ts_future)
```

**输出**：

```
2024-01-01    105.0
2024-01-02    110.0
2024-01-03    108.0
2024-01-04    112.0
2024-01-05      NaN
```

**应用场景**：

- 同比增长率：`ts / ts.shift(12) - 1`（月度数据）
- 环比增长率：`ts.pct_change()`
- 滚动增长：`ts - ts.shift(1)`
- 预测特征生成：`df['next_value'] = df['value'].shift(-1)`

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

# 高级操作与性能优化

## 学习目标

- 掌握透视表和交叉表
- 理解多级索引操作
- 学会性能优化技巧

## 透视表(Pivot Table)

透视表是 Excel 中最强大的数据分析工具,Pandas 提供了同样强大的功能。

### 1. 准备数据

```python
import pandas as pd
import numpy as np

# 创建销售数据
data = {
    '日期': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02',
             '2024-01-01', '2024-01-02'],
    '地区': ['北京', '上海', '北京', '上海', '广州', '广州'],
    '产品': ['手机', '电脑', '手机', '电脑', '手机', '电脑'],
    '销售额': [5000, 8000, 5500, 8500, 5200, 8200],
    '数量': [5, 3, 6, 4, 5, 3]
}
df = pd.DataFrame(data)
print("原始数据:")
print(df)
```

**输出**:

```
         日期 地区 产品  销售额  数量
0  2024-01-01 北京 手机   5000   5
1  2024-01-01 上海 电脑   8000   3
2  2024-01-02 北京 手机   5500   6
3  2024-01-02 上海 电脑   8500   4
4  2024-01-01 广州 手机   5200   5
5  2024-01-02 广州 电脑   8200   3
```

### 2. 基本透视表

```python
# 创建透视表:地区为行,产品为列,销售额求和
pivot = pd.pivot_table(
    df,
    values='销售额',      # 要聚合的列
    index='地区',         # 行索引
    columns='产品',       # 列索引
    aggfunc='sum'          # 聚合函数
)
print("透视表:")
print(pivot)
```

**输出**:

```
产品    电脑    手机
地区
北京   NaN   10500
上海  16500    NaN
广州   8200   5200
```

**原理**:

- 将每个地区-产品组合的销售额求和
- 空值表示该组合没有数据

### 3. 多级索引透视表

```python
# 多个行索引
pivot = pd.pivot_table(
    df,
    values='销售额',
    index=['地区', '日期'],  # 多级行索引
    columns='产品',
    aggfunc='sum'
)
print(pivot)
```

**输出**:

```
产品             电脑    手机
地区 日期
北京 2024-01-01  NaN   5000
    2024-01-02  NaN   5500
上海 2024-01-01  8000   NaN
    2024-01-02  8500   NaN
广州 2024-01-01  NaN   5200
    2024-01-02  8200   NaN
```

### 4. 多个聚合函数

```python
# 同时计算总和、平均值、数量
pivot = pd.pivot_table(
    df,
    values='销售额',
    index='地区',
    columns='产品',
    aggfunc=['sum', 'mean', 'count'],  # 多个聚合函数
    fill_value=0  # 用 0 填充空值
)
print(pivot)
```

**输出**:

```
        sum           mean          count
产品     电脑    手机     电脑    手机   电脑  手机
地区
北京      0  10500    0.0  5250.0    0    2
上海  16500      0  8250.0    0.0    2    0
广州   8200   5200  8200.0  5200.0    1    1
```

### 5. 添加总计

```python
# 添加行和列的总计
pivot = pd.pivot_table(
    df,
    values='销售额',
    index='地区',
    columns='产品',
    aggfunc='sum',
    fill_value=0,
    margins=True,      # 添加总计
    margins_name='总计'  # 总计名称
)
print(pivot)
```

**输出**:

```
产品     电脑    手机     总计
地区
北京      0  10500  10500
上海  16500      0  16500
广州   8200   5200  13400
总计  24700  15700  40400
```

## 交叉表(Crosstab)

交叉表用于计算两个或多个因子之间的频数分布。

### 1. 基本交叉表

```python
# 创建数据
data = {
    '性别': ['男', '女', '男', '女', '男', '女', '男', '女'],
    '城市': ['北京', '北京', '上海', '上海', '北京', '上海', '上海', '北京'],
    '喜好': ['运动', '读书', '运动', '音乐', '读书', '运动', '音乐', '音乐']
}
df = pd.DataFrame(data)

# 创建交叉表
crosstab = pd.crosstab(df['性别'], df['城市'])
print("性别-城市交叉表:")
print(crosstab)
```

**输出**:

```
城市  上海  北京
性别
女      2    2
男      2    2
```

**说明**:统计每个性别在各城市的人数。

### 2. 多维交叉表

```python
# 三维交叉表
crosstab = pd.crosstab(
    index=[df['性别'], df['城市']],
    columns=df['喜好']
)
print("多维交叉表:")
print(crosstab)
```

**输出**:

```
喜好       读书  运动  音乐
性别 城市
女   上海    0    1    1
    北京    1    0    1
男   上海    0    1    1
    北京    1    1    0
```

### 3. 添加百分比

```python
# 计算行百分比
crosstab = pd.crosstab(df['性别'], df['城市'], normalize='index')
print("行百分比:")
print(crosstab)
```

**输出**:

```
城市  上海  北京
性别
女    0.5  0.5
男    0.5  0.5
```

**normalize 参数**:

- `'index'`:行百分比(每行和为 1)
- `'columns'`:列百分比(每列和为 1)
- `'all'` 或 `True`:总体百分比(所有值和为 1)

### 4. Pivot Table vs Crosstab

| 特性         | Pivot Table        | Crosstab        |
| ------------ | ------------------ | --------------- |
| 主要用途     | 数值聚合           | 频数统计        |
| 默认聚合函数 | mean               | count           |
| 语法         | 需要指定 DataFrame | 直接传入 Series |
| 灵活性       | 更灵活             | 更简单          |

## 多级索引(MultiIndex)

多级索引允许在一个轴上有多个索引级别,适用于高维数据。

### 1. 创建多级索引

#### 方法 1:使用数组

```python
# 从数组创建
arrays = [
    ['北京', '北京', '上海', '上海'],
    ['2023', '2024', '2023', '2024']
]
index = pd.MultiIndex.from_arrays(arrays, names=['城市', '年份'])
data = pd.Series([100, 120, 90, 110], index=index)
print(data)
```

**输出**:

```
城市  年份
北京  2023    100
     2024    120
上海  2023     90
     2024    110
dtype: int64
```

#### 方法 2:使用笛卡尔积

```python
# 创建所有可能的组合
index = pd.MultiIndex.from_product(
    [['北京', '上海'], ['Q1', 'Q2', 'Q3', 'Q4']],
    names=['城市', '季度']
)
data = pd.Series(np.random.randint(80, 120, 8), index=index)
print(data)
```

**输出**:

```
城市  季度
北京  Q1     105
     Q2      98
     Q3     112
     Q4      87
上海  Q1     115
     Q2     101
     Q3      94
     Q4     108
dtype: int64
```

#### 方法 3:从 DataFrame 设置

```python
# 从现有 DataFrame 创建
df = pd.DataFrame({
    '城市': ['北京', '北京', '上海', '上海'],
    '年份': [2023, 2024, 2023, 2024],
    '销售额': [100, 120, 90, 110]
})
df_multi = df.set_index(['城市', '年份'])
print(df_multi)
```

### 2. 访问多级索引

```python
# 创建示例数据
arrays = [
    ['北京', '北京', '上海', '上海'],
    ['Q1', 'Q2', 'Q1', 'Q2']
]
index = pd.MultiIndex.from_arrays(arrays, names=['城市', '季度'])
data = pd.Series([100, 120, 90, 110], index=index, name='销售额')

# 选择第一级索引
print("北京的数据:")
print(data['北京'])
```

**输出**:

```
季度
Q1    100
Q2    120
Name: 销售额, dtype: int64
```

```python
# 选择特定组合
print("北京 Q1:")
print(data[('北京', 'Q1')])
```

**输出**:`100`

```python
# 使用 loc
print(data.loc[('北京', 'Q1')])         # 单个值
print(data.loc['北京'])                 # 一级索引
print(data.loc[[('北京', 'Q1'), ('上海', 'Q2')]])  # 多个值
```

### 3. 切片操作

```python
# 使用 slice 切片
print(data.loc[('北京', slice(None))])  # 北京的所有季度
print(data.loc[(slice(None), 'Q1')])       # 所有城市的 Q1
```

### 4. 索引级别操作

```python
# 交换索引级别
swapped = data.swaplevel()
print(swapped)
```

**输出**:

```
季度  城市
Q1   北京    100
Q2   北京    120
Q1   上海     90
Q2   上海    110
```

```python
# 按指定级别排序
sorted_data = data.sort_index(level='季度')
print(sorted_data)
```

### 5. 聚合操作

```python
# 按级别聚合
print("按城市聚合:")
print(data.groupby(level='城市').sum())

print("\n按季度聚合:")
print(data.groupby(level='季度').mean())
```

**输出**:

```
按城市聚合:
城市
北京    220
上海    200
Name: 销售额, dtype: int64

按季度聚合:
季度
Q1     95.0
Q2    115.0
Name: 销售额, dtype: float64
```

### 6. 重置索引

```python
# 将多级索引重置为普通列
df_reset = data.reset_index()
print(df_reset)
```

**输出**:

```
   城市 季度  销售额
0  北京  Q1    100
1  北京  Q2    120
2  上海  Q1     90
3  上海  Q2    110
```

## 向量化操作

向量化是 Pandas 高效运行的关键,避免使用循环。

### 1. 为什么需要向量化

```python
import time

# 准备数据
df = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000)
})

# 方法 1:使用循环(慢)
start = time.time()
result = []
for i in range(len(df)):
    result.append(df['A'].iloc[i] + df['B'].iloc[i])
df['C_loop'] = result
loop_time = time.time() - start

# 方法 2:向量化(快)
start = time.time()
df['C_vectorized'] = df['A'] + df['B']
vec_time = time.time() - start

print(f"循环耗时: {loop_time:.4f}秒")
print(f"向量化耗时: {vec_time:.4f}秒")
print(f"速度提升: {loop_time/vec_time:.1f}倍")
```

**输出示例**:

```
循环耗时: 5.2341秒
向量化耗时: 0.0023秒
速度提升: 2276.1倍
```

### 2. 常见向量化操作

```python
# 算术运算
df['sum'] = df['A'] + df['B']
df['diff'] = df['A'] - df['B']
df['product'] = df['A'] * df['B']
df['ratio'] = df['A'] / df['B']

# 数学函数
df['sqrt'] = np.sqrt(df['A'])
df['log'] = np.log(df['A'])
df['exp'] = np.exp(df['A'])

# 条件运算
df['category'] = np.where(df['A'] > 0.5, 'High', 'Low')

# 多条件
df['grade'] = np.select(
    [df['A'] > 0.8, df['A'] > 0.6, df['A'] > 0.4],
    ['A', 'B', 'C'],
    default='D'
)
```

### 3. 应用函数

```python
# apply 用于复杂逻辑(但比循环快)
def complex_calc(row):
    return row['A'] * 2 + row['B'] ** 2

df['result'] = df.apply(complex_calc, axis=1)

# 尽量使用向量化替代
df['result'] = df['A'] * 2 + df['B'] ** 2  # 更快!
```

### 4. 向量化最佳实践

```python
# ✔ 好:使用向量化操作
df['total'] = df['price'] * df['quantity']

# ✘ 差:使用循环
for i in range(len(df)):
    df.loc[i, 'total'] = df.loc[i, 'price'] * df.loc[i, 'quantity']

# ✔ 好:使用 NumPy 函数
df['log_value'] = np.log(df['value'])

# ✘ 差:使用 apply + lambda
df['log_value'] = df['value'].apply(lambda x: np.log(x))
```

## 内存优化

大数据集处理时,内存优化至关重要。

### 1. 数据类型优化

```python
# 查看内存使用
df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 10000),
    'float_col': np.random.rand(10000),
    'str_col': ['category_' + str(i%10) for i in range(10000)]
})

print("优化前:")
print(df.memory_usage(deep=True))
print(f"总内存: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 优化数据类型
df['int_col'] = df['int_col'].astype('int8')        # int64 -> int8
df['float_col'] = df['float_col'].astype('float32')  # float64 -> float32
df['str_col'] = df['str_col'].astype('category')     # object -> category

print("\n优化后:")
print(df.memory_usage(deep=True))
print(f"总内存: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
```

**输出示例**:

```
优化前:
Index        128
int_col    80000
float_col  80000
str_col   640000
dtype: int64
总内存: 781.38 KB

优化后:
Index        128
int_col    10000   # 减少 87.5%
float_col  40000   # 减少 50%
str_col    10512   # 减少 98.4%
dtype: int64
总内存: 59.02 KB  # 总体减少 92.4%
```

### 2. 分类类型的优势

```python
# 适用于有限集合的字符串
df = pd.DataFrame({
    'city': np.random.choice(['北京', '上海', '广州', '深圳'], 100000)
})

print(f"Object 类型内存: {df.memory_usage(deep=True)['city'] / 1024:.2f} KB")

df['city'] = df['city'].astype('category')
print(f"Category 类型内存: {df.memory_usage(deep=True)['city'] / 1024:.2f} KB")
```

### 3. 分块处理大文件

```python
# 分块读取 CSV
chunk_size = 10000
total = 0

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # 处理每一块
    result = chunk[chunk['value'] > 0]
    total += len(result)

print(f"总计符合条件的行数: {total}")
```

**优势**:

- 不需要将整个文件载入内存
- 适用于内存不足的情况
- 可以边读边处理

### 4. 其他优化技巧

```python
# 使用 eval 进行复杂表达式计算
# 对于大型 DataFrame 性能更好
df.eval('total = price * quantity', inplace=True)

# 使用 query 进行筛选
result = df.query('price > 100 and quantity < 10')

# 避免链式索引,使用 loc/iloc
# ✘ 差
value = df[df['A'] > 0]['B'][0]
# ✔ 好
value = df.loc[df['A'] > 0, 'B'].iloc[0]
```

## 高级技巧总结

### 1. 数据类型选择指南

| 数据特点             | 推荐类型   | 原因            |
| -------------------- | ---------- | --------------- |
| 小范围整数 (0-127)   | `int8`     | 节省 87.5% 内存 |
| 中等范围整数         | `int16/32` | 平衡范围和内存  |
| 精度要求不高的浮点数 | `float32`  | 节省 50% 内存   |
| 有限集合字符串       | `category` | 大幅减少内存    |
| 布尔值               | `bool`     | 最小内存        |

### 2. 性能优化检查清单

- [ ] 使用向量化操作替代循环
- [ ] 选择合适的数据类型
- [ ] 将重复字符串转为 category
- [ ] 使用 eval/query 进行复杂计算
- [ ] 大文件使用分块读取
- [ ] 避免不必要的数据复制
- [ ] 使用 inplace 参数减少内存
- [ ] 定期检查内存使用情况

### 3. 常用调试命令

```python
# 查看 DataFrame 信息
df.info(memory_usage='deep')

# 查看内存使用
print(df.memory_usage(deep=True))

# 性能分析
import cProfile
cProfile.run('your_function()')

# 时间测量
import time
start = time.time()
# ... 代码 ...
print(f"耗时: {time.time() - start:.4f}秒")
```