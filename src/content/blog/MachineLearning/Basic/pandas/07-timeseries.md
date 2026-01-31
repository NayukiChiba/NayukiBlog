---
title: Pandas 时间序列处理
date: 2026-01-13
category: MachineLearning/Basic/pandas
tags:
  - Python
  - Pandas
description: 掌握时间序列的创建、重采样和滚动窗口操作
image: https://img.yumeko.site/file/blog/PandasLearning.jpg
status: public
---

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

## 练习

```bash
python Basic/Pandas/07_timeseries.py
```
