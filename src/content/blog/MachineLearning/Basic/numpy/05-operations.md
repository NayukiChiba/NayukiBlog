---
title: NumPy 数组运算
date: 2026-01-05
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: 学习 NumPy 的数学运算、统计函数和聚合操作
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 数组运算

## 学习目标

- 掌握 NumPy 数组的基本数学运算
- 理解向量化操作的优势
- 学会使用比较运算和逻辑运算

## 为什么学习数组运算？

数据处理和机器学习中，90% 以上的时间都在进行数学运算：

- 数据标准化：减去均值，除以标准差
- 特征计算：求和、均值、方差
- 模型预测：矩阵乘法、激活函数

NumPy 的向量化运算让这些操作既简洁又高效。

## 运算类型

| 类型     | 运算符/函数                         | 说明         |
| -------- | ----------------------------------- | ------------ |
| 算术运算 | `+`, `-`, `*`, `/`, `**`, `//`, `%` | 元素级运算   |
| 比较运算 | `>`, `<`, `>=`, `<=`, `==`, `!=`    | 返回布尔数组 |
| 逻辑运算 | `&`, `\|`, `~`, `np.logical_and()`  | 布尔数组运算 |
| 统计运算 | `sum()`, `mean()`, `std()`, `var()` | 聚合运算     |

## 向量化的优势

### 传统 Python 循环方式（慢）

```python
a = [1, 2, 3, 4, 5]
b = [5, 4, 3, 2, 1]
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])
print(result)
```

**结果**：`[6, 6, 6, 6, 6]`

### NumPy 向量化方式（快）

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
result = a + b
print(result)
```

**结果**：`[6 6 6 6 6]`

**性能对比**：向量化操作比 Python 循环快 **10-100 倍**！

**原理**：

- Python 循环：每次迭代都有解释器开销、类型检查
- NumPy 向量化：底层用 C 语言批量处理，没有循环开销

## 算术运算

### 数组与数组运算

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
```

**加减乘除**：

```python
print(a + b)   # [6  8 10 12]  每个位置相加
print(a - b)   # [-4 -4 -4 -4]  每个位置相减
print(a * b)   # [5 12 21 32]  每个位置相乘
print(a / b)   # [0.2  0.33 0.43 0.5]  每个位置相除
```

**原理**：这是"元素级运算"，也叫"逐元素运算"（element-wise），对应位置的元素进行计算。

### 数组与标量运算

```python
a = np.array([1, 2, 3, 4])

print(a + 10)   # [11 12 13 14]  每个元素加 10
print(a * 2)    # [2 4 6 8]      每个元素乘 2
print(a ** 2)   # [1 4 9 16]     每个元素平方
```

**原理**：标量会自动"广播"到数组的每个元素。

### 更多运算符

```python
a = np.array([1, 2, 3, 4, 5])

# 幂运算
print(a ** 2)      # [1 4 9 16 25]  平方
print(a ** 0.5)    # [1.  1.41 1.73 2.  2.24]  平方根

# 整除
print(a // 2)      # [0 1 1 2 2]  整数除法，向下取整

# 取余
print(a % 2)       # [1 0 1 0 1]  余数，可用于判断奇偶
```

**应用场景**：

- `% 2 == 0`：判断偶数
- `** 0.5`：计算平方根（比 `np.sqrt()` 更直观）

## 比较运算

比较运算返回布尔数组，常用于条件筛选。

```python
a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

# 元素级比较
print(a > b)    # [False False  True  True]
print(a < b)    # [ True  True False False]
print(a == b)   # [False False False False]
print(a >= 2)   # [False  True  True  True]
```

**结果解读**：

- `a > b` 比较对应位置：1>4? False, 2>3? False, 3>2? True, 4>1? True

### 数组整体比较

```python
# 检查两个数组是否完全相同
np.array_equal(a, b)  # False

# 检查是否有任意元素相同
np.any(a == b)        # False

# 检查是否所有元素都满足条件
np.all(a != b)        # True
```

**使用场景**：

- `np.array_equal()`：验证两个数组完全一致
- `np.any()`：检查是否存在满足条件的元素
- `np.all()`：检查是否全部满足条件

## 统计运算

统计运算是数据分析的核心功能。

| 函数                   | 说明            | 数学符号 |
| ---------------------- | --------------- | -------- |
| `sum()`                | 求和            | Σ        |
| `mean()`               | 均值            | μ        |
| `std()`                | 标准差          | σ        |
| `var()`                | 方差            | σ²       |
| `min()`, `max()`       | 最小/最大值     | min, max |
| `argmin()`, `argmax()` | 最小/最大值索引 | —        |
| `cumsum()`             | 累积和          | —        |
| `cumprod()`            | 累积积          | —        |

### 基本统计

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr.sum())     # 15  = 1+2+3+4+5
print(arr.mean())    # 3.0 = 15/5
print(arr.std())     # 1.41 标准差
print(arr.var())     # 2.0 方差 = std²
print(arr.min())     # 1 最小值
print(arr.max())     # 5 最大值
```

**标准差和方差的含义**：

- 方差：每个数与均值差的平方的平均值，衡量数据的"分散程度"
- 标准差：方差的平方根，与原数据同单位

### 索引统计

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(arr.argmin())  # 1  最小值的索引（值为1的位置）
print(arr.argmax())  # 5  最大值的索引（值为9的位置）
```

**应用场景**：找到最高分学生的位置、找到最低价格的商品索引。

### 累积运算

```python
arr = np.array([1, 2, 3, 4, 5])

print(arr.cumsum())   # [1 3 6 10 15]
print(arr.cumprod())  # [1 2 6 24 120]
```

**cumsum 计算过程**：

- 第1个：1
- 第2个：1+2=3
- 第3个：1+2+3=6
- 第4个：1+2+3+4=10
- 第5个：1+2+3+4+5=15

**应用场景**：计算累计销售额、累计概率等。

## 沿轴运算 (axis)

对于多维数组，可以指定沿哪个轴进行运算。

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
```

### axis=None：所有元素

```python
print(arr.sum())  # 45
```

**含义**：对所有 9 个元素求和。

### axis=0：沿行方向（按列计算）

```python
print(arr.sum(axis=0))   # [12 15 18]
print(arr.mean(axis=0))  # [4. 5. 6.]
```

**计算过程**：

```
列0: 1+4+7=12
列1: 2+5+8=15
列2: 3+6+9=18
```

**记忆技巧**：`axis=0` 表示"压缩第0维（行）"，结果按列排列。

### axis=1：沿列方向（按行计算）

```python
print(arr.sum(axis=1))   # [6 15 24]
print(arr.mean(axis=1))  # [2. 5. 8.]
```

**计算过程**：

```
行0: 1+2+3=6
行1: 4+5+6=15
行2: 7+8+9=24
```

**记忆技巧**：`axis=1` 表示"压缩第1维（列）"，结果按行排列。

## 数学函数

### 三角函数

```python
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
# 对应角度：0°, 30°, 45°, 60°, 90°

print(np.sin(angles).round(3))  # [0.  0.5  0.707 0.866 1.]
print(np.cos(angles).round(3))  # [1.  0.866 0.707 0.5  0.]
```

**注意**：NumPy 使用弧度制，不是角度制。`np.pi` ≈ 3.14159。

### 指数和对数

```python
arr = np.array([1, 2, 3, 4, 5])

print(np.exp(arr))     # [2.72 7.39 20.09 54.60 148.41]  e^x
print(np.log(arr))     # [0. 0.69 1.10 1.39 1.61]  自然对数 ln(x)
print(np.log10(arr))   # [0. 0.30 0.48 0.60 0.70]  以10为底
print(np.log2(arr))    # [0. 1. 1.58 2. 2.32]  以2为底
```

**应用场景**：

- `exp`：机器学习中的 softmax 函数
- `log`：信息熵计算、对数变换

### 取整函数

```python
arr = np.array([1.2, 2.5, 3.7, -1.2, -2.5])

print(np.floor(arr))   # [1. 2. 3. -2. -3.]  向下取整（往小的方向）
print(np.ceil(arr))    # [2. 3. 4. -1. -2.]  向上取整（往大的方向）
print(np.round(arr))   # [1. 2. 4. -1. -2.]  四舍五入
print(np.abs(arr))     # [1.2 2.5 3.7 1.2 2.5]  绝对值
```

**注意**：`np.round(2.5)` 结果是 2（四舍五入到偶数），这是银行家舍入法。

## 逻辑运算

用于组合多个布尔数组。

```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(np.logical_and(a, b))  # [True False False False]  与
print(np.logical_or(a, b))   # [True True True False]   或
print(np.logical_not(a))     # [False False True True]  非
print(np.logical_xor(a, b))  # [False True True False]  异或
```

**真值表**：
| a | b | and | or | xor |
| ----- | ----- | ----- | ----- | ----- |
| True | True | True | True | False |
| True | False | False | True | True |
| False | True | False | True | True |
| False | False | False | False | False |

## 小结

1. **向量化运算**：比循环快 10-100 倍，是 NumPy 的核心优势
2. **元素级运算**：`+`, `-`, `*`, `/` 对应位置计算
3. **统计函数**：`sum`, `mean`, `std`, `min`, `max` 是数据分析的基础
4. **axis 参数**：控制沿哪个维度计算，`axis=0` 按列，`axis=1` 按行
5. **数学函数**：`sin`, `cos`, `exp`, `log` 等都支持数组输入

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/05_operations.py
```

**练习题**：

1. 创建数组 `[1,2,3,4,5]`，计算所有元素的平方和
2. 创建 3×4 矩阵，分别计算每行的均值和每列的均值
3. 创建 100 个 0-100 的随机整数，统计大于 50 的有多少个
