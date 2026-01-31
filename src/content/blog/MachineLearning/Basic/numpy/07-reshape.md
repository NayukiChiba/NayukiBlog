---
title: NumPy 数组变形
date: 2026-01-06
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: 掌握 NumPy 数组的 reshape、flatten、transpose 等变形操作
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 数组变形

## 学习目标

- 掌握 NumPy 数组的变形方法
- 学会改变数组的形状和维度
- 理解数组转置和轴变换

## 为什么需要数组变形？

在数据处理中，经常需要改变数据的形状：

- 把一维数据变成二维表格
- 神经网络输入需要特定的维度格式
- 矩阵运算前需要调整形状
- 把多维数据展平用于可视化

## 核心方法

| 方法              | 说明             | 返回类型         |
| ----------------- | ---------------- | ---------------- |
| `reshape(shape)`  | 改变数组形状     | 视图（如果可能） |
| `flatten()`       | 展平为一维       | 副本             |
| `ravel()`         | 展平为一维       | 视图（如果可能） |
| `T`               | 转置             | 视图             |
| `transpose(axes)` | 按指定轴顺序转置 | 视图             |

## 视图 vs 副本

这是 NumPy 中非常重要的概念！

| 类型            | 说明             | 修改是否影响原数组 |
| --------------- | ---------------- | ------------------ |
| **视图 (View)** | 与原数组共享数据 | 是                 |
| **副本 (Copy)** | 独立的数据拷贝   | 否                 |

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# flatten() 返回副本
flat = arr.flatten()
flat[0] = 999
print(arr[0, 0])  # 1 (原数组不变)

# ravel() 返回视图
rav = arr.ravel()
rav[0] = 999
print(arr[0, 0])  # 999 (原数组被修改！)
```

**选择建议**：

- 需要独立副本：用 `flatten()`
- 只需要临时访问：用 `ravel()`（更快、更省内存）

## reshape 变形

`reshape` 是最常用的变形方法，可以把数组转换成任意兼容的形状。

```python
arr = np.arange(1, 13)  # [1, 2, 3, ..., 12]
print(arr.shape)  # (12,)
```

### 变形为二维

```python
# 变形为 3 行 4 列
result = arr.reshape(3, 4)
print(result)
```

**结果**：

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

**原理**：元素按行填充（C 顺序），先填满第一行，再填第二行...

```python
# 变形为 4 行 3 列
print(arr.reshape(4, 3))
```

**结果**：

```
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```

### 使用 -1 自动计算

当你知道一个维度，让 NumPy 自动计算另一个维度。

```python
# 知道要 2 行，自动计算列数
print(arr.reshape(2, -1))   # (2, 6)

# 知道要 6 列，自动计算行数
print(arr.reshape(-1, 6))   # (2, 6)

# 知道要 3 行，自动计算列数
print(arr.reshape(3, -1))   # (3, 4)
```

**注意**：只能有一个 `-1`，不能 `reshape(-1, -1)`。

### 变形限制

元素总数必须保持不变！

```python
arr = np.arange(12)  # 12 个元素

arr.reshape(3, 4)    # ✓ 3×4=12
arr.reshape(2, 6)    # ✓ 2×6=12
arr.reshape(3, 5)    # ✗ 错误！3×5=15≠12
```

## flatten 和 ravel 展平

把多维数组变成一维。

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
```

### 默认按行展平（C 顺序）

```python
print(arr.flatten())   # [1 2 3 4 5 6]
print(arr.ravel())     # [1 2 3 4 5 6]
```

**顺序**：先遍历第一行，再遍历第二行。

### 按列展平（Fortran 顺序）

```python
print(arr.flatten('F'))  # [1 4 2 5 3 6]
print(arr.ravel('F'))    # [1 4 2 5 3 6]
```

**顺序**：先遍历第一列，再遍历第二列。

## 转置

### 二维数组转置

转置就是把行变成列、列变成行。

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])  # (2, 3)

print(arr.T)
```

**结果**：

```
[[1 4]
 [2 5]
 [3 6]]
```

**形状变化**：(2, 3) → (3, 2)

**应用场景**：

- 矩阵运算前调整形状
- 行列数据互换

### 一维数组转置

```python
arr_1d = np.array([1, 2, 3])
print(arr_1d.T)  # [1 2 3] 形状不变！
```

**注意**：一维数组转置后形状不变！如果需要变成列向量：

```python
arr_1d[:, np.newaxis]  # 变成 (3, 1)
arr_1d.reshape(-1, 1)  # 变成 (3, 1)
```

### 三维数组转置

```python
arr_3d = np.arange(24).reshape(2, 3, 4)  # (2, 3, 4)

# 默认转置：完全反转维度顺序
print(arr_3d.T.shape)  # (4, 3, 2)

# 指定轴顺序
print(np.transpose(arr_3d, axes=(1, 0, 2)).shape)  # (3, 2, 4)
```

**原理**：`axes=(1, 0, 2)` 表示新数组的第 0 维来自原数组的第 1 维，以此类推。

## squeeze 和 expand_dims

### squeeze：移除长度为 1 的维度

当数组有多余的维度时，用 `squeeze` 压缩。

```python
arr = np.array([[[1, 2, 3]]])  # (1, 1, 3)
print(np.squeeze(arr).shape)   # (3,)
```

**应用场景**：

- 模型输出有多余的批次维度
- 从 (1, 10) 变成 (10,)

```python
arr = np.zeros((1, 5, 1, 3))
print(np.squeeze(arr).shape)   # (5, 3) 所有 1 都被移除
```

### expand_dims：增加维度

为数组增加一个新的维度（长度为 1）。

```python
arr = np.array([1, 2, 3])  # (3,)

# 在第 0 维增加（变成行向量）
print(np.expand_dims(arr, axis=0).shape)  # (1, 3)

# 在第 1 维增加（变成列向量）
print(np.expand_dims(arr, axis=1).shape)  # (3, 1)
```

**应用场景**：

- 神经网络输入需要批次维度 (batch_size, ...)
- 为广播做准备

## np.newaxis 增加维度

`np.newaxis` 是另一种增加维度的方式，更简洁。

```python
arr = np.array([1, 2, 3, 4, 5])  # (5,)

# 增加行维度
print(arr[np.newaxis, :].shape)  # (1, 5)
# 等价于 arr.reshape(1, -1)

# 增加列维度
print(arr[:, np.newaxis].shape)  # (5, 1)
# 等价于 arr.reshape(-1, 1)
```

**记忆方法**：`np.newaxis` 放在哪个位置，就在那里增加一个维度。

## resize 调整大小

`resize` 可以改变数组的总大小（元素数量可以变化）。

```python
arr = np.array([1, 2, 3, 4])

# 目标更大时，重复元素
result = np.resize(arr, (2, 4))
print(result)
```

**结果**：

```
[[1 2 3 4]
 [1 2 3 4]]
```

```python
# 不整除时循环填充
result = np.resize(arr, (3, 3))
print(result)
```

**结果**：

```
[[1 2 3]
 [4 1 2]
 [3 4 1]]
```

**与 reshape 的区别**：

- `reshape`：元素数量必须相同
- `resize`：元素数量可以变化

## 常用场景总结

| 场景         | 方法                                       |
| ------------ | ------------------------------------------ |
| 一维变二维   | `reshape(m, n)` 或 `reshape(m, -1)`        |
| 二维变一维   | `flatten()` 或 `ravel()`                   |
| 行列转换     | `.T` 或 `transpose()`                      |
| 添加批次维度 | `expand_dims(axis=0)` 或 `arr[np.newaxis]` |
| 移除多余维度 | `squeeze()`                                |
| 一维变列向量 | `arr[:, np.newaxis]` 或 `reshape(-1, 1)`   |

## 小结

1. **reshape**：最常用，改变形状但元素总数不变
2. **flatten/ravel**：多维变一维，flatten 返回副本，ravel 返回视图
3. **转置 T**：行列互换
4. **squeeze/expand_dims**：移除或增加维度
5. **-1 的妙用**：让 NumPy 自动计算某一维的大小

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/07_reshape.py
```

**练习题**：

1. 把 `np.arange(24)` 变形为 (4, 6) 的矩阵
2. 把一维数组 `[1,2,3,4,5]` 变成列向量 (5, 1)
3. 创建形状 (1, 3, 1, 4) 的数组，用 squeeze 移除长度为 1 的维度
