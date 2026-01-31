---
title: Pandas 数据合并与连接
date: 2026-01-12
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 学习 concat、merge 和 join 操作，灵活合并数据
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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
