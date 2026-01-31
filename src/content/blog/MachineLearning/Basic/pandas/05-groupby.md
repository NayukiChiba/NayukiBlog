---
title: Pandas 数据分组与聚合
date: 2026-01-12
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 掌握 GroupBy 分组操作和聚合函数
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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

## 练习

```bash
python Basic/Pandas/05_groupby.py
```
