---
title: Pandas 数据选择与过滤
date: 2026-01-11
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 掌握 loc、iloc 和条件过滤，精确选择数据
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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

## 练习

```bash
python Basic/Pandas/03_selection.py
```
