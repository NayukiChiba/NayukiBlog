---
title: NumPy 文件读写
date: 2026-01-09
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: 掌握 NumPy 的文件 IO 操作，包括 npy、npz 和文本文件读写
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 文件操作

## 学习目标

- 掌握 NumPy 数组的保存和加载方法
- 学会读写文本文件
- 理解不同文件格式的特点

## 为什么需要文件操作？

在实际项目中，你需要：

- **保存中间结果**：避免每次重新计算
- **共享数据**：把处理好的数据给其他人用
- **持久化存储**：程序结束后数据不丢失
- **与其他工具交换**：如 Excel、Pandas

## 文件格式对比

| 格式        | 函数                  | 特点                     | 适用场景          |
| ----------- | --------------------- | ------------------------ | ----------------- |
| `.npy`      | `save()/load()`       | 二进制，快速，保留 dtype | 临时存储/快速 I/O |
| `.npz`      | `savez()/load()`      | 压缩的多数组文件         | 存储多个数组      |
| `.txt/.csv` | `savetxt()/loadtxt()` | 文本格式，可读性好       | 与其他程序交换    |

## 二进制文件 (.npy)

### 保存单个数组

```python
arr = np.random.random((3, 4))
print("原始数组:\n", arr)
```

**结果**：

```
原始数组:
 [[0.37454012 0.95071431 0.73199394 0.59865848]
  [0.15601864 0.15599452 0.05808361 0.86617615]
  [0.60111501 0.70807258 0.02058449 0.96990985]]
```

```python
# 保存到文件
np.save('data.npy', arr)
print("已保存到 data.npy")
```

**说明**：

- 文件名自动添加 `.npy` 后缀（如果没有写）
- 保存后会创建一个二进制文件

### 加载数组

```python
# 从文件加载
loaded = np.load('data.npy')
print("加载的数组:\n", loaded)
```

### 验证数据完整性

```python
# 验证数据是否完全一致
print("数据相等:", np.array_equal(arr, loaded))  # True
print("类型保留:", loaded.dtype)  # float64
```

### .npy 格式的优点

1. **速度快**：二进制格式，读写速度最快
2. **文件小**：比文本格式占用空间小
3. **完整保留数据类型**：int32 存进去，读出来还是 int32
4. **支持任意维度**：无论多少维都可以保存

## 多数组文件 (.npz)

当需要保存多个数组时，使用 `.npz` 格式。

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2], [3, 4]])
arr3 = np.random.random((2, 3))

# 保存多个数组（使用关键字参数命名）
np.savez('data.npz', x=arr1, matrix=arr2, random=arr3)
print("已保存到 data.npz")
```

**说明**：`x=arr1` 表示把 arr1 命名为 'x' 存储。

### 加载多数组文件

```python
# 加载文件
data = np.load('data.npz')

# 查看包含哪些数组
print("包含的数组:", data.files)  # ['x', 'matrix', 'random']

# 按名称访问
print("x:", data['x'])
print("matrix:\n", data['matrix'])
print("random:\n", data['random'])
```

**结果**：

```
包含的数组: ['x', 'matrix', 'random']
x: [1 2 3 4 5]
matrix:
 [[1 2]
  [3 4]]
random:
 [[0.78227816 0.58985146 0.01292458]
  [0.02946096 0.31286622 0.28756624]]
```

### 压缩版本

```python
# 使用 gzip 压缩（文件更小，但速度稍慢）
np.savez_compressed('data_compressed.npz', x=arr1, matrix=arr2)
```

**使用场景**：数组很大、磁盘空间有限时使用压缩版本。

## 文本文件 (.txt/.csv)

文本格式的优点是人类可读，可以用记事本打开查看。

### 保存为文本

```python
arr = np.array([[1.234, 2.345, 3.456],
                [4.567, 5.678, 6.789],
                [7.890, 8.901, 9.012]])

# 默认格式（科学计数法）
np.savetxt('data.txt', arr)
```

**data.txt 内容**：

```
1.234000000000000e+00 2.345000000000000e+00 3.456000000000000e+00
4.567000000000000e+00 5.678000000000000e+00 6.789000000000000e+00
7.890000000000000e+00 8.901000000000000e+00 9.012000000000000e+00
```

### 自定义格式

```python
np.savetxt('data.csv', arr,
           delimiter=',',     # 分隔符改为逗号
           fmt='%.2f',        # 保留 2 位小数
           header='A,B,C',    # 添加表头
           comments='')       # 不添加 # 注释符
```

**data.csv 内容**：

```
A,B,C
1.23,2.35,3.46
4.57,5.68,6.79
7.89,8.90,9.01
```

**这样的 CSV 文件可以直接用 Excel 打开！**

### 格式说明符

| 格式     | 说明              | 示例         |
| -------- | ----------------- | ------------ |
| `%.2f`   | 2 位小数          | `3.14`       |
| `%.4f`   | 4 位小数          | `3.1416`     |
| `%d`     | 整数              | `3`          |
| `%.2e`   | 科学计数法        | `3.14e+00`   |
| `%10.4f` | 宽度 10，4 位小数 | `    3.1416` |

### 加载文本文件

```python
# 基本加载
arr = np.loadtxt('data.txt')
print(arr)
```

```python
# 加载 CSV 文件
arr = np.loadtxt('data.csv', delimiter=',', skiprows=1)  # 跳过表头
print(arr)
```

**结果**：

```
[[1.23 2.35 3.46]
 [4.57 5.68 6.79]
 [7.89 8.9  9.01]]
```

### loadtxt 常用参数

```python
arr = np.loadtxt(
    'data.csv',
    delimiter=',',     # 分隔符
    skiprows=1,        # 跳过开头 1 行（表头）
    usecols=(0, 2),    # 只读取第 0 和第 2 列
    dtype=np.float32,  # 指定数据类型
)
```

## savetxt 参数详解

```python
np.savetxt(
    fname,           # 文件路径
    X,               # 要保存的数组
    fmt='%.18e',     # 格式字符串
    delimiter=' ',   # 分隔符（默认空格）
    newline='\n',    # 行分隔符
    header='',       # 文件头（第一行注释）
    footer='',       # 文件尾（最后一行注释）
    comments='# '    # 注释前缀
)
```

**常见用法**：

```python
# 保存为整数格式的 CSV
np.savetxt('integers.csv', arr.astype(int), fmt='%d', delimiter=',')

# 保存为科学计数法
np.savetxt('scientific.txt', arr, fmt='%.4e')
```

## loadtxt 参数详解

```python
np.loadtxt(
    fname,           # 文件路径
    dtype=float,     # 数据类型（默认 float）
    delimiter=None,  # 分隔符（None 表示任意空白）
    skiprows=0,      # 跳过开头行数
    usecols=None,    # 读取的列（None 表示全部）
    unpack=False     # 是否转置
)
```

### unpack 参数

```python
# 假设文件有 3 列
# 不使用 unpack
data = np.loadtxt('data.csv', delimiter=',')  # shape: (n, 3)

# 使用 unpack=True
col1, col2, col3 = np.loadtxt('data.csv', delimiter=',', unpack=True)
# 每个变量是一列数据
```

## 使用场景总结

| 场景                   | 推荐方法      | 原因                 |
| ---------------------- | ------------- | -------------------- |
| 临时存储/快速 I/O      | `.npy`        | 速度最快             |
| 存储多个相关数组       | `.npz`        | 一个文件管理多个数组 |
| 与 Excel/其他程序交换  | `.csv`/`.txt` | 通用格式             |
| 需要人工查看/编辑      | `.csv`/`.txt` | 文本可读             |
| 大型数组，磁盘空间紧张 | `.npz` 压缩版 | 节省空间             |

> **性能提示**：对于大型数组，二进制格式 (`.npy`) 比文本格式快 **10-100 倍**。

## 完整示例：数据处理工作流

```python
# 1. 生成或加载原始数据
np.random.seed(42)
raw_data = np.random.random((1000, 10))

# 2. 处理数据
processed_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0)

# 3. 保存处理结果
np.save('processed_data.npy', processed_data)
np.savez('all_data.npz', raw=raw_data, processed=processed_data)

# 4. 之后可以直接加载处理后的数据
data = np.load('processed_data.npy')
```

## 小结

1. **二进制格式 (.npy)**：最快，完整保留数据类型，推荐用于 Python 内部
2. **多数组格式 (.npz)**：一个文件存多个数组，支持压缩
3. **文本格式 (.txt/.csv)**：人类可读，适合与其他程序交换
4. **性能考虑**：大数据用二进制，小数据或需要检查时用文本

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/10_file_io.py
```

**练习题**：

1. 创建一个 100×100 的随机矩阵，保存为 `.npy` 格式，再加载回来验证数据一致
2. 创建两个数组，保存到一个 `.npz` 文件，然后分别加载出来
3. 创建一个 5×3 的整数矩阵，保存为 CSV 格式（带表头 A,B,C），然后用记事本打开查看
