---
title: NumPy 数组属性
date: 2026-01-03
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: 掌握 NumPy 数组的核心属性，如 shape、dtype、ndim 等
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 数组属性

## 学习目标

- 掌握 NumPy 数组的重要属性
- 理解数组形状、维度、数据类型等概念
- 学会查看和修改数组属性

## 为什么要了解数组属性？

在数据处理中，你经常需要回答这些问题：

- 这个数组有多少行多少列？（`shape`）
- 这个数组是几维的？（`ndim`）
- 总共有多少个数据？（`size`）
- 数据是什么类型？（`dtype`）
- 占用多少内存？（`nbytes`）

这些属性帮助你快速了解数据的"规格"，是数据探索的第一步。

## 重要属性一览

| 属性       | 说明                     | 示例结果  |
| ---------- | ------------------------ | --------- |
| `shape`    | 数组的形状（各维度大小） | `(3, 4)`  |
| `ndim`     | 数组的维度数             | `2`       |
| `size`     | 数组的元素总数           | `12`      |
| `dtype`    | 数组的数据类型           | `float64` |
| `itemsize` | 每个元素的字节大小       | `8`       |
| `nbytes`   | 数组总字节数             | `96`      |

## 形状属性

### shape - 数组的形状

`shape` 是最常用的属性，返回一个元组，表示数组在每个维度上的大小。

```python
arr = np.random.random((3, 4))
print(arr.shape)
```

**结果**：`(3, 4)`

**含义**：这是一个 3 行 4 列的二维数组。

```python
# 获取行数和列数
print(arr.shape[0])   # 3 (行数)
print(arr.shape[1])   # 4 (列数)
```

**实用技巧**：在机器学习中，`shape[0]` 通常表示样本数，`shape[1]` 表示特征数。

### ndim - 维度数

```python
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2], [3, 4]])
arr_3d = np.zeros((2, 3, 4))

print(arr_1d.ndim)  # 1
print(arr_2d.ndim)  # 2
print(arr_3d.ndim)  # 3
```

**理解维度**：

- 1 维：一条线（向量）
- 2 维：一个面（矩阵/表格）
- 3 维：一个立体（如彩色图片：高度 × 宽度 × 颜色通道）

### size - 元素总数

```python
arr = np.random.random((3, 4))
print(arr.size)  # 12
```

**结果**：`12`

**计算方式**：`size = shape[0] × shape[1] = 3 × 4 = 12`

**与 len() 的区别**：

```python
print(len(arr))   # 3 (只返回第一维的长度)
print(arr.size)   # 12 (返回所有元素的总数)
```

## 内存属性

### dtype - 数据类型

```python
arr = np.random.random((3, 4))
print(arr.dtype)  # float64
```

**结果**：`float64`

**含义**：每个元素是 64 位（8 字节）的浮点数。

```python
arr_int = np.array([1, 2, 3])
print(arr_int.dtype)  # int32 或 int64（取决于系统）
```

### itemsize - 每个元素的字节数

```python
arr_float = np.array([1.0, 2.0, 3.0])
print(arr_float.itemsize)  # 8
```

**结果**：`8`

**含义**：float64 类型每个元素占用 8 字节（64 位 ÷ 8 = 8 字节）。

```python
arr_int32 = np.array([1, 2, 3], dtype=np.int32)
print(arr_int32.itemsize)  # 4
```

### nbytes - 总字节数

```python
arr = np.random.random((3, 4))
print(arr.nbytes)  # 96
```

**结果**：`96`

**计算方式**：`nbytes = size × itemsize = 12 × 8 = 96` 字节

**实用场景**：估算大数组的内存占用。

```python
# 创建一个大数组
big_arr = np.zeros((10000, 10000))
print(f"占用内存: {big_arr.nbytes / 1024 / 1024:.2f} MB")
```

**结果**：`占用内存: 762.94 MB`

## 常用数据类型

### 整数类型

| 类型    | 说明      | 范围                           | 字节数 |
| ------- | --------- | ------------------------------ | ------ |
| `int8`  | 8 位整数  | -128 ~ 127                     | 1      |
| `int16` | 16 位整数 | -32,768 ~ 32,767               | 2      |
| `int32` | 32 位整数 | -2,147,483,648 ~ 2,147,483,647 | 4      |
| `int64` | 64 位整数 | -9×10¹⁸ ~ 9×10¹⁸               | 8      |

**如何选择？**

- 一般用默认的 `int64`
- 处理大量数据且值范围小时，用 `int32` 或 `int16` 节省内存

### 浮点类型

| 类型      | 说明        | 精度                    | 字节数 |
| --------- | ----------- | ----------------------- | ------ |
| `float16` | 16 位浮点数 | 半精度                  | 2      |
| `float32` | 32 位浮点数 | 单精度（~7位有效数字）  | 4      |
| `float64` | 64 位浮点数 | 双精度（~15位有效数字） | 8      |

**如何选择？**

- 科学计算用 `float64`（默认）
- 深度学习训练常用 `float32` 节省显存
- `float16` 用于极端内存优化场景

### 其他类型

| 类型         | 说明                  |
| ------------ | --------------------- |
| `bool`       | 布尔类型 (True/False) |
| `complex64`  | 复数 (2 个 float32)   |
| `complex128` | 复数 (2 个 float64)   |

## 数据类型转换 (astype)

`astype()` 方法可以将数组转换为其他数据类型。

```python
# 创建浮点数组
arr_float = np.array([1.5, 2.7, 3.2, 4.8])
print(f"原数组: {arr_float}")
print(f"原类型: {arr_float.dtype}")
```

**结果**：

```
原数组: [1.5 2.7 3.2 4.8]
原类型: float64
```

```python
# 转换为整数（截断小数部分）
arr_int = arr_float.astype(np.int32)
print(f"转换后: {arr_int}")
print(f"新类型: {arr_int.dtype}")
```

**结果**：

```
转换后: [1 2 3 4]
新类型: int32
```

**注意**：浮点转整数是**截断**（直接丢弃小数），不是四舍五入！

- 1.5 → 1（不是 2）
- 2.7 → 2（不是 3）

```python
# 如果需要四舍五入
arr_rounded = np.round(arr_float).astype(np.int32)
print(arr_rounded)  # [2 3 3 5]
```

```python
# 转换为字符串
arr_str = arr_float.astype(str)
print(arr_str)  # ['1.5' '2.7' '3.2' '4.8']
```

> **重要**：`astype()` 返回新数组，不会修改原数组！

## 布尔数组

布尔数组是 NumPy 中非常强大的工具，用于条件筛选。

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 通过比较运算创建布尔数组
bool_arr = arr > 5
print(bool_arr)
```

**结果**：`[False False False False False  True  True  True  True  True]`

**原理**：每个元素与 5 比较，大于 5 的位置为 True，否则为 False。

```python
# 布尔数组的类型
print(bool_arr.dtype)  # bool
```

### 布尔索引

布尔数组可以直接用于索引，筛选出 True 对应的元素：

```python
# 筛选大于 5 的元素
print(arr[bool_arr])  # [6, 7, 8, 9, 10]

# 或者直接写
print(arr[arr > 5])   # [6, 7, 8, 9, 10]
```

### 统计 True 的数量

```python
# True 被当作 1，False 被当作 0
print(bool_arr.sum())  # 5
```

**结果**：`5`

**含义**：有 5 个元素大于 5。

```python
# 计算比例
print(bool_arr.mean())  # 0.5
```

**结果**：`0.5`

**含义**：50% 的元素大于 5。

## 查看数据类型信息

```python
# 整数类型信息
print(np.iinfo(np.int32))
```

**结果**：

```
Machine parameters for int32
---------------------------------------------------------------
min = -2147483648
max = 2147483647
---------------------------------------------------------------
```

**用途**：了解某个整数类型能表示的最大最小值。

```python
# 浮点类型信息
print(np.finfo(np.float64))
```

**结果**显示精度、最小正数、最大值等信息。

## 小结

1. **形状相关**：`shape`（形状）、`ndim`（维度数）、`size`（元素总数）
2. **内存相关**：`dtype`（数据类型）、`itemsize`（单元素字节数）、`nbytes`（总字节数）
3. **类型转换**：使用 `astype()` 转换数据类型，返回新数组
4. **布尔数组**：用于条件筛选，`sum()` 统计 True 数量

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/03_attributes.py
```

**练习题**：

1. 创建一个 100×100 的随机数组，计算它占用多少 KB 内存
2. 将浮点数组 `[1.1, 2.5, 3.9, 4.4]` 转换为整数，对比 `astype(int)` 和 `np.round().astype(int)` 的区别
3. 创建数组 `[1,2,3,...,100]`，统计其中有多少个偶数
