---
title: Pandas 高级操作与性能优化
date: 2026-01-13
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 掌握透视表、多级索引和性能优化技巧
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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

## 练习

```bash
python Basic/Pandas/09_advanced.py
```
