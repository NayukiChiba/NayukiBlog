---
title: NumPy 数组拼接与分割
date: 2026-01-08
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: 学习 NumPy 数组的拼接、堆叠和分割操作
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 连接与分割

## 学习目标

- 掌握 NumPy 数组的连接方法
- 学会分割数组
- 理解不同轴的连接和分割

## 为什么需要连接和分割？

在数据处理中，经常需要：

- 合并多个数据集（如训练集和验证集合并）
- 拼接特征列（如把不同来源的特征合并）
- 分割数据集（如划分训练集、验证集、测试集）
- 批量处理大型数组

## 连接函数

| 函数               | 说明                  | 常用场景     |
| ------------------ | --------------------- | ------------ |
| `np.concatenate()` | 沿指定轴连接          | 通用连接     |
| `np.vstack()`      | 垂直堆叠（沿 axis=0） | 合并表格行   |
| `np.hstack()`      | 水平堆叠（沿 axis=1） | 合并表格列   |
| `np.stack()`       | 沿新轴堆叠            | 创建批次维度 |
| `np.dstack()`      | 沿深度方向堆叠        | 合并通道     |

## 分割函数

| 函数               | 说明           | 常用场景         |
| ------------------ | -------------- | ---------------- |
| `np.split()`       | 沿指定轴分割   | 通用分割         |
| `np.vsplit()`      | 垂直分割       | 分割表格行       |
| `np.hsplit()`      | 水平分割       | 分割表格列       |
| `np.array_split()` | 允许不均匀分割 | 数据不能整除时用 |

## 连接 vs 堆叠

| 对比     | concatenate/vstack/hstack | stack/dstack         |
| -------- | ------------------------- | -------------------- |
| 维度变化 | 不增加维度                | 增加一个维度         |
| 形状要求 | 除连接轴外维度相同        | 所有数组形状完全相同 |

**形象理解**：

- **连接**：把两张纸并排放在一起，还是一张（更大的）纸
- **堆叠**：把两张纸叠在一起，变成一叠纸

## concatenate 连接

`concatenate` 是最通用的连接函数，沿指定轴拼接数组。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:\n", A)
print("B:\n", B)
```

**结果**：

```
A:
 [[1 2]
  [3 4]]
B:
 [[5 6]
  [7 8]]
```

### 沿 axis=0 连接（垂直拼接）

```python
result = np.concatenate([A, B], axis=0)
print(result)
print("形状:", result.shape)
```

**结果**：

```
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
形状: (4, 2)
```

**过程**：把 B 接在 A 的下面，行数相加 (2+2=4)，列数不变。

### 沿 axis=1 连接（水平拼接）

```python
result = np.concatenate([A, B], axis=1)
print(result)
print("形状:", result.shape)
```

**结果**：

```
[[1 2 5 6]
 [3 4 7 8]]
形状: (2, 4)
```

**过程**：把 B 接在 A 的右边，行数不变，列数相加 (2+2=4)。

## vstack 和 hstack

这是 `concatenate` 的便捷版本，更直观。

### vstack：垂直堆叠

**记忆**：v = vertical（垂直）

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
B = np.array([[7, 8, 9]])              # (1, 3)

result = np.vstack([A, B])
print(result)
print("形状:", result.shape)
```

**结果**：

```
[[1 2 3]
 [4 5 6]
 [7 8 9]]
形状: (3, 3)
```

**要求**：列数必须相同。

### hstack：水平堆叠

**记忆**：h = horizontal（水平）

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
C = np.array([[10], [20]])             # (2, 1)

result = np.hstack([A, C])
print(result)
print("形状:", result.shape)
```

**结果**：

```
[[ 1  2  3 10]
 [ 4  5  6 20]]
形状: (2, 4)
```

**要求**：行数必须相同。

## stack 沿新轴堆叠

`stack` 会创建一个**新的维度**，把数组沿这个新维度堆叠。

```python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)
```

### 沿 axis=0 堆叠

```python
result = np.stack([A, B], axis=0)
print("形状:", result.shape)  # (2, 2, 2)
print(result)
```

**结果**：

```
形状: (2, 2, 2)
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
```

**理解**：创建了一个"批次"维度，result[0] 是 A，result[1] 是 B。

**应用场景**：深度学习中把多个样本堆叠成一个 batch。

### 沿 axis=2 堆叠

```python
result = np.stack([A, B], axis=2)
print("形状:", result.shape)  # (2, 2, 2)
```

**理解**：在最后一个维度堆叠，result[:,:,0] 是 A，result[:,:,1] 是 B。

> **注意**：`stack` 要求所有数组形状完全相同！

## dstack 深度堆叠

沿第三个轴（深度方向）堆叠。

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

result = np.dstack([A, B])
print("形状:", result.shape)  # (2, 2, 2)
```

**应用场景**：把多个灰度图合并成彩色图（RGB 三通道）。

## split 分割

`split` 是连接的逆操作，把一个数组分割成多个部分。

```python
arr = np.arange(12).reshape(4, 3)
print(arr)
```

**结果**：

```
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
```

### 均匀分割

```python
# 沿 axis=0 分成 2 份
parts = np.split(arr, 2, axis=0)

print("第 1 部分:\n", parts[0])
print("第 2 部分:\n", parts[1])
```

**结果**：

```
第 1 部分:
 [[0 1 2]
  [3 4 5]]
第 2 部分:
 [[ 6  7  8]
  [ 9 10 11]]
```

**过程**：4 行平均分成 2 份，每份 2 行。

```python
# 沿 axis=1 分成 3 份
parts = np.split(arr, 3, axis=1)
# 返回 3 个 (4, 1) 数组
```

### 指定分割位置

```python
# 在索引 1 和 3 处分割
parts = np.split(arr, [1, 3], axis=0)

print("第 1 部分 ([:1]):\n", parts[0])
print("第 2 部分 ([1:3]):\n", parts[1])
print("第 3 部分 ([3:]):\n", parts[2])
```

**结果**：

```
第 1 部分 ([:1]):
 [[0 1 2]]
第 2 部分 ([1:3]):
 [[3 4 5]
  [6 7 8]]
第 3 部分 ([3:]):
 [[ 9 10 11]]
```

**理解**：`[1, 3]` 表示在索引 1 和 3 处切两刀，分成 3 份。

## vsplit 和 hsplit

```python
arr = np.arange(12).reshape(4, 3)

# vsplit: 垂直分割（按行分）
parts = np.vsplit(arr, 2)  # 分成 2 个 (2, 3) 数组

# hsplit: 水平分割（按列分）
parts = np.hsplit(arr, 3)  # 分成 3 个 (4, 1) 数组
```

## array_split 不均匀分割

当数组不能均匀分割时，用 `array_split`。

```python
arr = np.arange(10)  # [0, 1, 2, ..., 9] 共 10 个元素

# 分成 3 份（不能均分：10÷3 = 3...1）
parts = np.array_split(arr, 3)

print("第 1 部分:", parts[0])
print("第 2 部分:", parts[1])
print("第 3 部分:", parts[2])
```

**结果**：

```
第 1 部分: [0 1 2 3]    # 4 个元素
第 2 部分: [4 5 6]      # 3 个元素
第 3 部分: [7 8 9]      # 3 个元素
```

**规则**：前面的份会多分一些元素。

**对比**：

- `split(arr, 3)` 会报错，因为 10 不能被 3 整除
- `array_split(arr, 3)` 可以工作

## 常用场景

| 场景                   | 方法                              |
| ---------------------- | --------------------------------- |
| 垂直拼接表格（增加行） | `vstack` 或 `concatenate(axis=0)` |
| 水平拼接特征（增加列） | `hstack` 或 `concatenate(axis=1)` |
| 创建批次维度           | `stack(axis=0)`                   |
| 分割训练/测试集        | `split` 或 `array_split`          |
| 合并 RGB 通道          | `dstack` 或 `stack(axis=2)`       |

## 小结

### 连接方法

1. **concatenate**：通用连接，需指定 axis
2. **vstack**：垂直堆叠（axis=0 的简写）
3. **hstack**：水平堆叠（axis=1 的简写）
4. **stack**：创建新维度并堆叠

### 分割方法

1. **split**：均匀分割
2. **vsplit/hsplit**：垂直/水平分割
3. **array_split**：允许不均匀分割

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/09_concat_split.py
```

**练习题**：

1. 创建两个 (3, 4) 矩阵，垂直拼接成 (6, 4) 矩阵
2. 创建两个 (3, 4) 矩阵，水平拼接成 (3, 8) 矩阵
3. 把长度为 17 的数组分成 5 份（使用 array_split）
