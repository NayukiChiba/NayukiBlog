---
title: Pandas 数据清洗与处理
date: 2026-01-11
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 学习缺失值处理、重复值删除和数据类型转换
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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

## 练习

```bash
python Basic/Pandas/04_cleaning.py
```
