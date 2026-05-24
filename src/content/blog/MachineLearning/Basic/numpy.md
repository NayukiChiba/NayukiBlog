---
title: NumPy 全指南
date: 2026-01-11
category: MachineLearning/Basic
tags:
  - Python
  - 基础
description: 一个全面的 NumPy 学习教程，涵盖从基础到高级的所有核心概念，包括数组创建、索引切片、数学运算、线性代数、广播机制、文件操作等，包含大量代码示例和实战项目。
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: published
---

# NumPy 基础与数组概念

## 本章目标

1. 理解 NumPy 的核心对象 `ndarray` 及其与 Python 列表的本质差异
2. 掌握 `np.array` 创建数组、`shape` 与 `ndim` 等基础属性
3. 理解广播机制的运算语义
4. 理解向量化带来的性能优势

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.__version__` | 属性 | 查看 NumPy 版本号 |
| `np.get_printoptions()` | 函数 | 读取当前数组打印配置 |
| `np.array(...)` | 函数 | 从 Python 序列创建 `ndarray` |
| `arr * scalar` / `arr + scalar` | 运算符 | 广播：逐元素算术运算（非列表拼接） |
| `arr.shape` | 属性 | 数组各维度长度组成的元组 |
| `arr.ndim` | 属性 | 数组的维度数量 |

## 1. 环境与打印配置

### `np.get_printoptions`

#### 作用

读取当前 NumPy 的数组打印配置。只负责查询，不修改配置。

#### 重点方法

```python
np.get_printoptions()
```

#### 返回内容

| 键名 | 类型 | 含义 |
|---|---|---|
| `precision` | `int` | 小数显示精度，默认为 `8` |
| `threshold` | `int` | 元素总数超过此值时省略中间内容，默认为 `1000` |
| `edgeitems` | `int` | 省略时首尾各保留的元素数，默认为 `3` |
| `linewidth` | `int` | 每行最大字符宽度，默认为 `75` |
| `suppress` | `bool` | 是否尽量抑制科学计数法，默认为 `False` |
| `nanstr` | `str` | NaN 的显示文本，默认为 `'nan'` |
| `infstr` | `str` | 无穷大的显示文本，默认为 `'inf'` |
| `sign` | `str` | 正负号显示策略，默认为 `'-'` |
| `floatmode` | `str` | 浮点数格式模式，默认为 `'maxprec'` |
| `formatter` | `dict` 或 `None` | 自定义格式化器，默认为 `None` |
| `legacy` | `bool` | 兼容旧版打印行为，默认为 `False` |

可使用 `np.set_printoptions(**kwargs)` 修改这些配置。

#### 示例代码

```python
import numpy as np

print(np.get_printoptions())
```

#### 输出

```python
{'edgeitems': 3,
 'threshold': 1000,
 'floatmode': 'maxprec',
 'precision': 8,
 'suppress': False,
 'linewidth': 75,
 'nanstr': 'nan',
 'infstr': 'inf',
 'sign': '-',
 'formatter': None,
 'legacy': False,
 'override_repr': None}
```

### `np.__version__`

#### 作用

查看当前安装的 NumPy 版本号。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `str` | NumPy 版本号字符串 |

#### 示例代码

```python
import numpy as np

print(np.__version__)
```

#### 输出

```text
2.1.3
```

## 2. ndarray 与 Python 列表的差异

### `np.array`

#### 作用

从 Python 列表或嵌套序列创建 `ndarray`，是 NumPy 最基础的数组构造入口。

#### 重点方法

```python
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `object` | `array_like` | 输入数据，可为列表、元组、嵌套序列、其他数组 | `[1, 2, 3, 4, 5]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.float64` |
| `copy` | `bool` | 是否复制数据，默认为 `True` | `False` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `subok` | `bool` | `True` 时保留子类类型，默认为 `False`（强制基础 `ndarray`） | `True` |
| `ndmin` | `int` | 返回数组的最小维度，不足时在前方补 1，默认为 `0` | `2` |
| `like` | `array_like` 或 `None` | 参考数组原型，用于兼容第三方数组库（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

pyList = [1, 2, 3, 4, 5]
npArray = np.array([1, 2, 3, 4, 5])

print(f"Python列表: {pyList}")
print(f"NumPy数组: {npArray}")
```

#### 输出

```text
Python列表: [1, 2, 3, 4, 5]
NumPy数组: [1 2 3 4 5]
```

#### 理解重点

- 列表有逗号分隔，数组没有——这是最直观的视觉差异
- 显式指定 `dtype` 可避免后续隐式类型转换

### 广播机制

#### 作用

NumPy 的 `*` 和 `+` 对数组执行**逐元素**运算，并通过广播自动扩展标量或低维数组。这与 Python 列表的拼接/重复语义完全不同。

#### 乘法对比

```python
import numpy as np

pyList = [1, 2, 3, 4, 5]
npArray = np.array([1, 2, 3, 4, 5])

print(pyList * 2)    # 列表：重复拼接
print(npArray * 2)   # 数组：逐元素乘法
```

```text
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
[ 2  4  6  8 10]
```

#### 加法对比

```python
print(pyList + [6])     # 列表：拼接
print(npArray + 6)      # 数组：逐元素加法（标量广播）
```

```text
[1, 2, 3, 4, 5, 6]
[ 7  8  9 10 11]
```

#### 理解重点

- 列表 `* n` 是重复拼接，数组 `* n` 是逐元素乘法
- 列表 `+` 是拼接，数组 `+` 是逐元素加法
- 标量与数组运算时，标量自动广播到每个元素——这是向量化的基础

## 3. 向量化性能优势

#### 作用

NumPy 的向量化运算将循环下沉到 C 层执行，避免了 Python 解释器的逐元素开销。数据规模越大，优势越明显。

#### 示例代码

```python
import time
import numpy as np

size = 10_000_000

# Python 列表推导
pyList = list(range(size))
start = time.time()
result = [x * 2 for x in pyList]
pyTime = time.time() - start

# NumPy 向量化
npArray = np.arange(size)
start = time.time()
result = npArray * 2
npTime = time.time() - start

print(f"数据规模: {size:,}")
print(f"Python列表耗时: {pyTime:.4f}秒")
print(f"NumPy数组耗时: {npTime:.4f}秒")
print(f"NumPy快了约 {pyTime / npTime:.1f} 倍")
```

#### 输出

```text
数据规模: 10,000,000
Python列表耗时: 0.4064秒
NumPy数组耗时: 0.0191秒
NumPy快了约 21.3 倍
```

#### 理解重点

- 向量化把循环下沉到 C 层，解释器开销更低
- 性能差距随数据量增大而拉大——小数据（<1000）差异不明显
- 实际倍数受硬件、BLAS 实现、数据类型影响

## 4. ndarray 基础属性

### `arr.shape`

#### 作用

返回数组各维度长度组成的元组。一维向量的 `shape` 为 `(n,)`，注意尾部的逗号表示它是元组。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `tuple[int, ...]` | 各维度长度，如 `(2, 3)` 表示 2 行 3 列 |

### `arr.ndim`

#### 作用

返回数组的维度数量（轴的数量），等于 `len(arr.shape)`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 维度数量，标量为 `0`，向量为 `1`，矩阵为 `2` |

### 常用属性速览

| 属性 | 类型 | 含义 |
|---|---|---|
| `shape` | `tuple[int, ...]` | 各维度长度 |
| `ndim` | `int` | 维度数量 |
| `size` | `int` | 元素总数，等于各维度长度的乘积 |
| `dtype` | `numpy.dtype` | 元素的数据类型 |
| `itemsize` | `int` | 每个元素占用的字节数 |
| `nbytes` | `int` | 数组总字节数，等于 `size × itemsize` |

### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float32)

print(arr.shape, type(arr.shape))
print(arr.ndim, type(arr.ndim))
print(arr.dtype, type(arr.dtype))
print(arr.itemsize, arr.nbytes)
```

### 输出

```text
(3,) tuple
1 int
float32 numpy.dtype
4 12
```

### 理解重点

- 一维向量的 `shape` 是 `(3, )`，注意尾部逗号——它是单元素元组，去掉逗号就成了整数 `3`
- 二维理解为"行 × 列"，三维及以上建议用"轴"来思考
- 大数组先用 `nbytes / 1024**2` 估算内存占用量

## 常见坑

1. 把列表的 `+` / `*` 语义误用到数组上——列表是拼接/重复，数组是逐元素运算
2. 忽略 `shape` 导致后续广播或矩阵运算维度不匹配
3. 性能对比时数据规模太小（<1000），倍数差异不明显，结论不可靠

## 小结

- NumPy 的核心不是"更简短的语法"，而是"统一的数组对象 + 向量化计算"
- 从本章开始建立 `shape`、`dtype`、广播的思维模型，后续所有章节都建立在这之上
- `np.array` 是数据的入口，`shape` 和 `ndim` 是理解数据形状的第一手信息

# NumPy 创建数组

## 本章目标

1. 掌握 `np.array`、`zeros`、`ones`、`eye`、`full` 创建指定值数组
2. 理解 `arange`（按步长）与 `linspace`（按点数）的使用差异
3. 学会生成可复现的随机数组（`seed`），掌握常用随机分布 API

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.array(...)` | 函数 | 从列表或嵌套列表创建 `ndarray` |
| `np.zeros(...)` | 函数 | 创建全 0 数组 |
| `np.ones(...)` | 函数 | 创建全 1 数组 |
| `np.eye(...)` | 函数 | 创建单位矩阵或对角线偏移矩阵 |
| `np.full(...)` | 函数 | 用固定值填充数组 |
| `np.arange(...)` | 函数 | 按步长生成等差序列（半开区间 `[start, stop)`） |
| `np.linspace(...)` | 函数 | 按点数生成等间距序列（默认包含终点） |
| `np.random.seed(...)` | 函数 | 固定随机种子，保证可复现 |
| `np.random.rand(...)` | 函数 | 生成 `[0, 1)` 均匀分布随机数 |
| `np.random.random(...)` | 函数 | 与 `rand` 等价，使用 `size` 关键字参数 |
| `np.random.randint(...)` | 函数 | 生成离散整数随机样本 |
| `np.random.randn(...)` | 函数 | 生成标准正态分布 $\mathcal{N}(0, 1)$ 随机样本 |
| `np.random.normal(...)` | 函数 | 生成指定均值与标准差的正态分布样本 |
| `np.array_equal(...)` | 函数 | 判断两个数组是否完全相等 |

## 1. 从列表创建数组

### `np.array`

#### 作用

从 Python 列表或嵌套列表创建 `ndarray`。嵌套列表自动推断为多维数组，可通过 `dtype` 指定元素类型。

#### 重点方法

```python
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `object` | `array_like` | 输入数据，可为列表、元组、嵌套序列、其他数组 | `[1, 2, 3]`、`[[1, 2], [3, 4]]` |
| `dtype` | `dtype` 或 `None` | 元素数据类型，默认为 `None`（自动推断） | `np.float64` |
| `copy` | `bool` | `True` 总是复制数据，`False` 在可行时共享内存，默认为 `True` | `False` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `subok` | `bool` | `True` 保留子类类型，`False` 强制基础 `ndarray`，默认为 `False` | `True` |
| `ndmin` | `int` | 返回数组的最小维度，不足时在前方补 1，默认为 `0` | `2` |
| `like` | `array_like` 或 `None` | 参考数组原型，用于兼容第三方数组库（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arrFloat = np.array([1, 2, 3], dtype=np.float64)

print(arr1d, arr1d.shape, arr1d.ndim)
print(arr2d)
print(arr2d.shape, arr2d.ndim)
print(arrFloat, arrFloat.dtype)
```

#### 输出

```text
[1 2 3 4 5] (5,) 1
[[1 2 3]
 [4 5 6]]
(2, 3) 2
[1. 2. 3.] float64
```

#### 理解重点

- 显式指定 `dtype` 可避免后续隐式类型转换——不确定类型时先定好
- `shape` 与 `ndim` 是后续所有变换的基础：拿到数组先看形状

## 2. 特殊数组

### `np.zeros`

#### 作用

创建指定形状的全 0 数组，常用于初始化缓存、mask、累加容器。

#### 重点方法

```python
np.zeros(shape, dtype=float, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(3, 4)` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `float`（即 `float64`） | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.zeros((3, 4)))
```

#### 输出

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

### `np.ones`

#### 作用

创建指定形状的全 1 数组，常用于基线向量、偏置初始化。

#### 重点方法

```python
np.ones(shape, dtype=None, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(2, 3)` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（等价 `float64`） | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.ones((2, 3)))
```

#### 输出

```text
[[1. 1. 1.]
 [1. 1. 1.]]
```

### `np.eye`

#### 作用

创建单位矩阵或带对角线偏移的矩阵。$k=0$ 为主对角线，$k>0$ 向右上偏移，$k<0$ 向左下偏移。

#### 重点方法

```python
np.eye(N, M=None, k=0, dtype=float, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `N` | `int` | 行数 | `3` |
| `M` | `int` 或 `None` | 列数，默认为 `None`（等于 `N`，生成方阵） | `4` |
| `k` | `int` | 对角线偏移：`0` 主对角线 / `>0` 向右上 / `<0` 向左下，默认为 `0` | `1` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `float` | `np.int32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.eye(3))
print(np.eye(3, k=1))
```

#### 输出

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
```

### `np.full`

#### 作用

用指定常量值填充整个数组，快速构造常量矩阵。

#### 重点方法

```python
np.full(shape, fill_value, dtype=None, order='C', *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 输出数组形状 | `(2, 2)` |
| `fill_value` | `scalar` 或 `array_like` | 填充值，标量或可广播到 `shape` 的数组 | `7` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（由 `fill_value` 推断） | `np.float32` |
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先，默认为 `'C'` | `'F'` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.full((2, 2), 7))
```

#### 输出

```text
[[7 7]
 [7 7]]
```

## 3. 序列数组

### `np.arange`

#### 作用

生成半开区间 $[start, stop)$ 的等差序列，类似 Python 的 `range`，但返回 `ndarray`。支持浮点步长与负步长。

#### 重点方法

```python
np.arange([start,] stop, [step,] dtype=None, *, like=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `int` 或 `float` | 起始值（包含），省略时默认为 `0` | `0`、`10` |
| `stop` | `int` 或 `float` | 终止值（不包含） | `10`、`0` |
| `step` | `int` 或 `float` | 步长，省略时默认为 `1`，支持负数和浮点数 | `2`、`-1` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（自动推断） | `np.float32` |
| `like` | `array_like` 或 `None` | 参考数组原型（NumPy 1.20+） | —— |

#### 示例代码

```python
import numpy as np

print(np.arange(0, 10, 2))
print(np.arange(10, 0, -1))
```

#### 输出

```text
[0 2 4 6 8]
[10  9  8  7  6  5  4  3  2  1]
```

### `np.linspace`

#### 作用

在指定区间 $[start, stop]$ 内按点数均匀切分，默认包含终点。浮点场景下比 `arange` 更稳定。

#### 重点方法

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `array_like` | 起始值，可为标量或数组 | `0` |
| `stop` | `array_like` | 终止值，可为标量或数组 | `2 * np.pi` |
| `num` | `int` | 采样点个数，默认为 `50` | `5`、`10` |
| `endpoint` | `bool` | 结果是否包含 `stop`，默认为 `True` | `False` |
| `retstep` | `bool` | `True` 时额外返回步长 `(array, step)`，默认为 `False` | `True` |
| `dtype` | `dtype` 或 `None` | 元素类型，默认为 `None`（自动推断） | `np.float32` |
| `axis` | `int` | `start` / `stop` 为数组时结果沿哪个轴排布，默认为 `0` | `1` |

#### 示例代码

```python
import numpy as np

print(np.linspace(0, 1, 5))
print(np.linspace(0, 2 * np.pi, 10))
```

#### 输出

```text
[0.   0.25 0.5  0.75 1.  ]
[0.         0.6981317  1.3962634  2.0943951  2.7925268  3.4906585
 4.1887902  4.88692191 5.58505361 6.28318531]
```

#### 理解重点

- `arange` 按**步长**切分，区间是 $[start, stop)$——适用于已知步长的场景
- `linspace` 按**点数**切分，默认包含终点——适用于已知采样点数的场景
- 浮点步长优先选 `linspace`，可读性和稳定性都更好

## 4. 随机数组

### `np.random.seed`

#### 作用

固定随机数生成器的起点，保证随机序列可复现。机器学习实验必须固定随机种子。

#### 重点方法

```python
np.random.seed(seed=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `seed` | `int` 或 `None` | 非负整数（$0$ 到 $2^{32}-1$）固定序列起点，`None` 使用系统随机源 | `42` |

### `np.random.rand`

#### 作用

生成 $[0, 1)$ 均匀分布随机数，以位置参数指定各维度长度。

#### 重点方法

```python
np.random.rand(d0, d1, ..., dn)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `d0, d1, ...` | `int` | 以位置参数逐维指定输出形状，省略所有参数返回单个标量 | `2, 3` |

### `np.random.random`

#### 作用

与 `rand` 等价，生成 $[0, 1)$ 均匀分布随机数。区别在于使用 `size` 关键字参数指定形状。

#### 重点方法

```python
np.random.random(size=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `(2, 3)` |

### `np.random.randint`

#### 作用

在 $[low, high)$ 区间内生成离散整数随机样本。

#### 重点方法

```python
np.random.randint(low, high=None, size=None, dtype=int)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `low` | `int` | 下界（包含）；若 `high=None`，则采样区间变为 $[0, low)$ | `0` |
| `high` | `int` 或 `None` | 上界（不包含） | `10` |
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `(3, 3)` |
| `dtype` | `dtype` | 整数元素类型，默认为 `int`（即 `int64`） | `np.int32` |

### `np.random.randn`

#### 作用

生成标准正态分布 $\mathcal{N}(0, 1)$ 的随机样本。概率密度函数：

$$
f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$

#### 重点方法

```python
np.random.randn(d0, d1, ..., dn)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `d0, d1, ...` | `int` | 以位置参数逐维指定输出形状，省略所有参数返回单个标量 | `5` |

### `np.random.normal`

#### 作用

生成指定均值 $\mu$ 和标准差 $\sigma$ 的正态分布样本。概率密度函数：

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

#### 重点方法

```python
np.random.normal(loc=0.0, scale=1.0, size=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `loc` | `float` | 正态分布的均值 $\mu$（分布中心），默认为 `0.0` | `10` |
| `scale` | `float` | 正态分布的标准差 $\sigma$，必须非负，默认为 `1.0` | `2` |
| `size` | `int` 或 `tuple[int, ...]` 或 `None` | 输出形状，默认为 `None`（返回单个标量） | `5` |

### 综合示例

> 随机数 API 共享全局状态，独立运行时结果可能不同。下例在 `seed(42)` 后按顺序调用，保证输出可复现。

#### 示例代码

```python
import numpy as np

np.random.seed(42)
print(np.random.rand(2, 3))
print(np.random.random(size=(2, 3)))
print(np.random.randint(0, 10, (3, 3)))

arr = np.random.randn(5)
print(arr)
print(arr.mean(), arr.std())

print(np.random.normal(loc=10, scale=2, size=5))
```

#### 输出

```text
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]]
[[0.05808361 0.86617615 0.60111501]
 [0.70807258 0.02058449 0.96990985]]
[[5 1 4]
 [0 9 5]
 [8 0 9]]
[-0.25104397 -0.16386712 -1.47632969  1.48698096 -0.02445518]
-0.0857 0.9428
[10.71110263 10.83402222 11.66492371  9.41320171  9.94032286]
```

#### 理解重点

- `rand` / `random` 用于均匀分布 $[0, 1)$
- `randint` 用于离散整数采样
- `randn` 是标准正态分布 $\mathcal{N}(0, 1)$；`normal` 可指定 $\mu$ 和 $\sigma$

## 5. 随机种子与可复现性

### `np.array_equal`

#### 作用

判断两个数组是否完全相等（形状相同且每个元素值相同）。常用于验证实验结果的可复现性。

#### 重点方法

```python
np.array_equal(a1, a2, equal_nan=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a1` | `array_like` | 待比较的第一个数组 | `arr1` |
| `a2` | `array_like` | 待比较的第二个数组 | `arr2` |
| `equal_nan` | `bool` | 是否将两个 `NaN` 视为相等（NumPy 1.19+），默认为 `False` | `True` |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr1 = np.random.random((2, 2))

np.random.seed(42)
arr2 = np.random.random((2, 2))

print(arr1)
print(arr2)
print(np.array_equal(arr1, arr2))
```

#### 输出

```text
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
[[0.37454012 0.95071431]
 [0.73199394 0.59865848]]
True
```

#### 理解重点

- 相同的 `seed` + 相同的调用顺序 ⇒ 相同的随机结果
- 中间插入新的随机调用会消耗随机状态，改变后续所有序列

## 常见坑

1. `arange(0, 1, 0.1)` 可能出现浮点累积误差——优先用 `linspace`
2. 只在一处设置 `seed` 后，中间插入新随机调用会改变后续序列——设置 `seed` 要对应具体的随机操作之前
3. 未指定 `dtype` 时 NumPy 自动推断，可能与预期不符——整数输入默认 `int64`，浮点输入默认 `float64`

## 小结

- 创建数组是后续索引、运算、线代和机器学习的入口
- 看到问题先想用哪种创建方式：固定值（`zeros`/`ones`/`full`/`eye`）、序列（`arange`/`linspace`）、随机（`rand`/`randn`/`randint`）
- 养成"固定随机种子"的习惯——`seed(42)` 一行，省去无数调试时间

# NumPy 属性与 dtype

## 本章目标

1. 掌握数组结构属性 `shape`、`ndim`、`size` 及其关系
2. 掌握类型与内存属性 `dtype`、`itemsize`、`nbytes`
3. 会用 `np.iinfo` / `np.finfo` 查询整数与浮点类型的取值范围和精度
4. 掌握 `astype` 的类型转换用法
5. 掌握布尔数组与条件筛选的基础模式

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.shape` | 属性 | 各维度长度组成的元组 |
| `arr.ndim` | 属性 | 维度数量，等于 `len(shape)` |
| `arr.size` | 属性 | 元素总数，等于各维度长度乘积 |
| `arr.dtype` | 属性 | 元素数据类型对象 |
| `arr.itemsize` | 属性 | 每个元素占用字节数 |
| `arr.nbytes` | 属性 | 数组总字节数，等于 `size × itemsize` |
| `arr.astype(...)` | 方法 | 返回类型转换后的新数组 |
| `np.iinfo(...)` | 函数 | 查询整数类型的取值范围和位数 |
| `np.finfo(...)` | 函数 | 查询浮点类型的精度参数和范围 |
| `arr > x` 等比较运算符 | 表达式 | 生成布尔数组 |
| `arr[mask]` | 表达式 | 用布尔数组进行索引筛选 |

## 1. 数组结构属性

### `arr.shape`

#### 作用

返回各维度长度组成的元组。例如 `(3, 4)` 表示 3 行 4 列，一维向量为 `(n, )`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `tuple[int, ...]` | 各维度长度，`ndim` 等于元组长度 |

### `arr.ndim`

#### 作用

返回数组的维度数量（轴数），等于 `len(arr.shape)`。标量为 `0`，向量为 `1`，矩阵为 `2`。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 维度数量 |

### `arr.size`

#### 作用

返回数组中所有元素的总数，等于各维度长度的乘积：$\prod \text{shape}[i]$。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 元素总数 |

### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(arr)
print(f"shape: {arr.shape}")
print(f"ndim: {arr.ndim}")
print(f"size: {arr.size}")
print(f"行数: {arr.shape[0]}, 列数: {arr.shape[1]}")
```

### 输出

```text
[[0.86395484 0.55333229 0.49186088 0.65651355]
 [0.65818868 0.01198379 0.0954384  0.54282681]
 [0.3904872  0.28345003 0.64304407 0.45011224]]
shape: (3, 4)
ndim: 2
size: 12
行数: 3, 列数: 4
```

### 理解重点

- `ndim == len(shape)`，始终成立
- `size == shape[0] × shape[1] × ...`，始终成立
- `shape` 是后续索引、广播、变形的基础——拿到数组先看 `shape`

## 2. 内存与数据类型属性

### `arr.dtype`

#### 作用

返回数组元素的数据类型对象，决定了每个元素的存储方式和取值范围。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `numpy.dtype` | 数据类型描述对象，如 `float64`、`int32` |

### `arr.itemsize`

#### 作用

返回每个元素占用的字节数。`float64` 为 8 字节，`float32` 为 4 字节。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 单个元素的字节数 |

### `arr.nbytes`

#### 作用

返回数组占用的总字节数，计算公式：$\text{nbytes} = \text{size} \times \text{itemsize}$。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `int` | 数组总字节数 |

### 示例代码

```python
import numpy as np

arr = np.random.random((3, 4))
print(f"dtype: {arr.dtype}")
print(f"itemsize: {arr.itemsize}")
print(f"nbytes: {arr.nbytes}")
print(f"验证 size × itemsize: {arr.size * arr.itemsize}")
```

### 输出

```text
dtype: float64
itemsize: 8
nbytes: 96
验证 size × itemsize: 96
```

### 理解重点

- `float64` 每元素 8 字节，`float32` 每元素 4 字节——内存翻倍
- 大数组先估算内存：`arr.nbytes / 1024**2` 得 MB 数
- 内存紧张时可考虑 `float32` 替换 `float64`

## 3. 常见数据类型

### 常见 dtype 一览

| dtype 名称 | 类别 | 每元素字节数 | 典型取值范围 / 精度 |
|---|---|---|---|
| `bool_` | 布尔 | 1 | `True` / `False` |
| `int8` | 有符号整数 | 1 | $[-128, 127]$ |
| `int16` | 有符号整数 | 2 | $[-32768, 32767]$ |
| `int32` | 有符号整数 | 4 | $[-2^{31}, 2^{31}-1]$ |
| `int64` | 有符号整数 | 8 | $[-2^{63}, 2^{63}-1]$ |
| `uint8` | 无符号整数 | 1 | $[0, 255]$ |
| `float16` | 半精度浮点 | 2 | 约 3 位有效数字 |
| `float32` | 单精度浮点 | 4 | 约 6~7 位有效数字 |
| `float64` | 双精度浮点 | 8 | 约 15 位有效数字 |
| `complex64` | 复数 | 8 | 实部 + 虚部各 `float32` |
| `complex128` | 复数 | 16 | 实部 + 虚部各 `float64` |

### `np.iinfo`

#### 作用

查询整数类型的元信息：最小值、最大值、位数。返回一个包含这些属性值的对象。

#### 重点方法

```python
np.iinfo(int_type)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `int_type` | `dtype` | 任意 NumPy 整数 dtype 或整数数组的类型 | `np.int32` |

#### 返回内容

| 属性 | 类型 | 含义 |
|---|---|---|
| `.min` | `int` | 该类型可表示的最小值 |
| `.max` | `int` | 该类型可表示的最大值 |
| `.bits` | `int` | 占用的二进制位数 |
| `.dtype` | `dtype` | 对应的 dtype 对象 |

#### 示例代码

```python
import numpy as np

for dtype in [np.int8, np.int16, np.int32, np.int64]:
    info = np.iinfo(dtype)
    print(f"{dtype.__name__}: [{info.min}, {info.max}]")
```

#### 输出

```text
int8: [-128, 127]
int16: [-32768, 32767]
int32: [-2147483648, 2147483647]
int64: [-9223372036854775808, 9223372036854775807]
```

### `np.finfo`

#### 作用

查询浮点类型的精度参数：机器精度、有效位数、最小/最大值等。

#### 重点方法

```python
np.finfo(dtype)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` | 任意 NumPy 浮点 dtype 或浮点数组的类型 | `np.float32` |

#### 返回内容

| 属性 | 类型 | 含义 |
|---|---|---|
| `.eps` | `float` | 机器精度，使 $1 + \varepsilon > 1$ 的最小正数 |
| `.min` | `float` | 该类型可表示的最小值（最负） |
| `.max` | `float` | 该类型可表示的最大值（最正） |
| `.precision` | `int` | 十进制有效位数 |
| `.bits` | `int` | 占用位数 |
| `.resolution` | `float` | 该类型的近似分辨率 |

#### 示例代码

```python
import numpy as np

for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__}: 精度 {info.precision} 位, eps={info.eps}")
```

#### 输出

```text
float16: 精度 3 位, eps=0.000977
float32: 精度 6 位, eps=1.1920929e-07
float64: 精度 15 位, eps=2.220446049250313e-16
```

## 4. 类型转换

### `ndarray.astype`

#### 作用

将数组元素转换为目标 `dtype`，返回新数组。浮点转整数是**截断**（向零取整），不是四舍五入。

#### 重点方法

```python
arr.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` 或 `str` | 目标数据类型 | `np.int32`、`str` |
| `order` | `str` | 内存布局：`'K'` 保持原样 / `'C'` 行优先 / `'F'` 列优先 / `'A'` 任意，默认为 `'K'` | `'C'` |
| `casting` | `str` | 类型转换策略：`'no'` / `'equiv'` / `'safe'` / `'same_kind'` / `'unsafe'`，从严格到宽松，默认为 `'unsafe'` | `'safe'` |
| `subok` | `bool` | `True` 保留子类类型，默认为 `True` | `False` |
| `copy` | `bool` | `True` 总是复制，默认为 `True` | `False` |

#### 示例代码

```python
import numpy as np

arrFloat = np.array([1.5, 2.7, 3.2, 4.8])
arrInt = arrFloat.astype(np.int32)
arrStr = arrFloat.astype(str)

print(f"原数组: {arrFloat}, dtype={arrFloat.dtype}")
print(f"转 int32: {arrInt}, dtype={arrInt.dtype}")
print(f"转 str: {arrStr}, dtype={arrStr.dtype}")
```

#### 输出

```text
原数组: [1.5 2.7 3.2 4.8], dtype=float64
转 int32: [1 2 3 4], dtype=int32
转 str: ['1.5' '2.7' '3.2' '4.8'], dtype=<U32
```

#### 理解重点

- 浮点转整数是**截断**（向零取整），`1.9` → `1`，`-1.9` → `-1`——如需四舍五入先用 `np.round`
- `astype` 返回新数组，不修改原数组——每次调用都有拷贝开销
- 大数组频繁类型转换是性能瓶颈，应在创建时指定正确的 `dtype`

## 5. 布尔数组与条件筛选

#### 作用

通过比较运算符（`>`、`<`、`==`、`!=` 等）生成与原数组同形状的布尔数组，再用布尔数组作为索引完成条件过滤。这是 NumPy 最常用、最高效的筛选模式。

### 运算符一览

| 运算符 | 含义 | 返回 |
|---|---|---|
| `>` | 逐元素大于 | 布尔数组 |
| `<` | 逐元素小于 | 布尔数组 |
| `>=` | 逐元素大于等于 | 布尔数组 |
| `<=` | 逐元素小于等于 | 布尔数组 |
| `==` | 逐元素等于 | 布尔数组 |
| `!=` | 逐元素不等 | 布尔数组 |

组合条件使用 `&`（与）、`|`（或）、`~`（非），**不能**使用 Python 的 `and`/`or`/`not`，每个条件必须加括号。

### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5

print(f"原数组: {arr}")
print(f"mask (arr > 5): {mask}")
print(f"mask.dtype: {mask.dtype}")
print(f"筛选结果 arr[mask]: {arr[mask]}")
print(f"大于 5 的元素个数: {mask.sum()}")
```

### 输出

```text
原数组: [ 1  2  3  4  5  6  7  8  9 10]
mask (arr > 5): [False False False False False  True  True  True  True  True]
mask.dtype: bool
筛选结果 arr[mask]: [ 6  7  8  9 10]
大于 5 的元素个数: 5
```

### 理解重点

- 比较表达式返回同形状的布尔数组
- 布尔数组可直接作为索引完成无显式循环的过滤
- `mask.sum()` 利用 $True=1$、$False=0$ 统计命中个数
- 多条件必须用 `&` / `|` / `~`，每个条件必须加括号——`(arr >= 3) & (arr <= 7)` 而非 `arr >= 3 & arr <= 7`

## 常见坑

1. 整数与浮点混合运算时 NumPy 自动提升为浮点，结果类型可能与预期不同
2. `astype` 每次都拷贝，大数组频繁调用是性能瓶颈
3. 大数组先用 `nbytes` 估算内存占用量，再决定用哪种 `dtype`
4. 布尔组合用 `&` / `|` 忘记加括号，会因运算符优先级报 `TypeError` 或得出错误结果
5. `arr.astype(int)` 是截断不是四舍五入——`1.9` → `1`

## 小结

- `shape` 与 `dtype` 是理解数组的两个核心维度：形状决定了数据组织，类型决定了数值能力
- 后续所有运算、索引、广播本质上都建立在属性之上
- 布尔数组是 NumPy 过滤的第一选择，优先于显式 `for` 循环
- 养成拿到数组先查 `shape` + `dtype` 的习惯

# NumPy 索引与切片

## 本章目标

1. 掌握一维与二维数组的基础整数索引
2. 掌握切片语法 `[start:stop:step]` 及其在多维数组上的组合用法
3. 熟练使用布尔索引完成条件过滤
4. 掌握花式索引（整数数组索引）的批量取值及行列配对语义
5. 掌握 `np.where` 的"取索引 / 三元替换"两种模式

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr[i]`、`arr[i, j]` | 语法 | 基础整数索引，支持负索引 |
| `arr[start:stop:step]` | 语法 | 切片索引，支持负步长 |
| `arr[row_slice, col_slice]` | 语法 | 多维切片：逗号分隔各轴 |
| `arr[mask]` | 语法 | 布尔索引：保留 `True` 位置元素 |
| `arr[[i1, i2, ...]]` | 语法 | 花式索引：整数数组批量取值 |
| `np.where(...)` | 函数 | 条件取索引或三元条件替换 |

## 1. 基本索引

#### 作用

按整数位置从数组中取出单个元素。一维用单下标，二维用逗号分隔各维度下标。负数从末尾倒数（`-1` 为最后一个元素）。

#### 语法

```python
arr[i]               # 一维
arr[i, j]            # 二维
arr[i, j, k]         # 三维
arr[-1]              # 最后一个
```

#### 示例代码

```python
import numpy as np

arr1d = np.array([10, 20, 30, 40, 50])
print(f"arr_1d[0] = {arr1d[0]}")
print(f"arr_1d[-1] = {arr1d[-1]}")

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"arr_2d[0, 0] = {arr2d[0, 0]}")
print(f"arr_2d[1, 2] = {arr2d[1, 2]}")
print(f"arr_2d[-1, -1] = {arr2d[-1, -1]}")
```

#### 输出

```text
arr_1d[0] = 10
arr_1d[-1] = 50
arr_2d[0, 0] = 1
arr_2d[1, 2] = 6
arr_2d[-1, -1] = 9
```

#### 理解重点

- 推荐 `arr[i, j]` 而非 `arr[i][j]`：前者一次性定位，后者先切行再取列，多一次视图创建
- 越界索引抛 `IndexError`

## 2. 切片索引

### 一维切片

#### 作用

按 `[start:stop:step]` 格式取连续或间隔的子序列。`stop` 不包含在结果内。步长可为负数实现反向遍历。

#### 语法

```python
arr[start:stop:step]
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `start` | `int` 或省略 | 起始位置（包含），省略时默认为 `0`（正步长）或末尾（负步长） | `2` |
| `stop` | `int` 或省略 | 终止位置（不包含），省略时默认为末尾（正步长）或 `0`（负步长） | `7` |
| `step` | `int` | 步长，默认为 `1`，可为负数表示反向 | `2`、`-1` |

#### 示例代码

```python
import numpy as np

arr = np.arange(10)
print(f"arr       = {arr}")
print(f"arr[2:7]  = {arr[2:7]}")
print(f"arr[:5]   = {arr[:5]}")
print(f"arr[5:]   = {arr[5:]}")
print(f"arr[::2]  = {arr[::2]}")
print(f"arr[1::2] = {arr[1::2]}")
print(f"arr[::-1] = {arr[::-1]}")
print(f"arr[::-2] = {arr[::-2]}")
```

#### 输出

```text
arr       = [0 1 2 3 4 5 6 7 8 9]
arr[2:7]  = [2 3 4 5 6]
arr[:5]   = [0 1 2 3 4]
arr[5:]   = [5 6 7 8 9]
arr[::2]  = [0 2 4 6 8]
arr[1::2] = [1 3 5 7 9]
arr[::-1] = [9 8 7 6 5 4 3 2 1 0]
arr[::-2] = [9 7 5 3 1]
```

### 二维切片

#### 作用

用逗号分隔各轴的切片规则，每个轴独立遵循 `[start:stop:step]`。可同时控制行和列的范围。

#### 示例代码

```python
import numpy as np

arr = np.arange(20).reshape(4, 5)
print(f"原 4x5 数组:\n{arr}")
print(f"\n前 2 行 arr[:2, :]:\n{arr[:2, :]}")
print(f"\n第 2~4 列 arr[:, 1:4]:\n{arr[:, 1:4]}")
print(f"\n子矩阵 arr[1:3, 1:3]:\n{arr[1:3, 1:3]}")
print(f"\n隔行隔列 arr[::2, ::2]:\n{arr[::2, ::2]}")
```

#### 输出

```text
原 4x5 数组:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]

前 2 行 arr[:2, :]:
[[0 1 2 3 4]
 [5 6 7 8 9]]

第 2~4 列 arr[:, 1:4]:
[[ 1  2  3]
 [ 6  7  8]
 [11 12 13]
 [16 17 18]]

子矩阵 arr[1:3, 1:3]:
[[ 6  7]
 [11 12]]

隔行隔列 arr[::2, ::2]:
[[ 0  2  4]
 [10 12 14]]
```

#### 理解重点

- 切片返回的是**视图**，修改切片会影响原数组——需要独立数据用 `.copy()`
- `stop` 不包含在结果内；反向切片时 `start` 应大于 `stop`（如 `arr[5:1:-1]`）

## 3. 布尔索引

#### 作用

用同形状的布尔数组作为索引，保留 `True` 位置的元素，返回新数组（拷贝）。是 NumPy 条件过滤的最高效方式。

#### 多条件组合

| 运算符 | 含义 | 示例 |
|---|---|---|
| `&` | 逻辑与 | `(arr >= 3) & (arr <= 7)` |
| `\|` | 逻辑或 | `(arr < 3) \| (arr > 8)` |
| `~` | 逻辑非 | `~(arr > 5)` |

不能使用 Python 的 `and`/`or`/`not`。每个条件必须加括号防止运算符优先级问题。

#### 示例代码

```python
import numpy as np

arr = np.arange(1, 11)
print(f"原数组: {arr}")
print(f"arr > 5: {arr[arr > 5]}")
print(f"arr % 2 == 0: {arr[arr % 2 == 0]}")
print(f"(arr >= 3) & (arr <= 7): {arr[(arr >= 3) & (arr <= 7)]}")
print(f"(arr < 3) | (arr > 8): {arr[(arr < 3) | (arr > 8)]}")
print(f"~(arr > 5): {arr[~(arr > 5)]}")
```

#### 输出

```text
原数组: [ 1  2  3  4  5  6  7  8  9 10]
arr > 5: [ 6  7  8  9 10]
arr % 2 == 0: [ 2  4  6  8 10]
(arr >= 3) & (arr <= 7): [3 4 5 6 7]
(arr < 3) | (arr > 8): [ 1  2  9 10]
~(arr > 5): [1 2 3 4 5]
```

#### 理解重点

- 布尔索引返回**拷贝**，修改不影响原数组——与切片（视图）相反
- `&` / `|` 的优先级高于比较运算符，不加括号会导致 `arr >= 3 & arr <= 7` 被解析为 `arr >= (3 & arr) <= 7`，报错

## 4. 花式索引

#### 作用

用整数列表或数组作为索引，按指定位置批量取值。传入两个索引数组时按"行列配对"取元素，而非笛卡儿积。

#### 语法

```python
arr[[i1, i2, i3, ...]]                 # 一维批量取
arr2d[[r1, r2, ...]]                   # 取多行
arr2d[[r1, r2, ...], [c1, c2, ...]]    # 按行列对逐元素取
```

#### 示例代码

```python
import numpy as np

arr = np.arange(10, 20)
print(f"原数组: {arr}")
print(f"arr[[0, 2, 5, 8]]: {arr[[0, 2, 5, 8]]}")

arr2d = np.arange(12).reshape(3, 4)
print(f"\n二维数组:\n{arr2d}")
print(f"\n选第 0、2 行:\n{arr2d[[0, 2]]}")
print(f"\n对角线 (0,0)(1,1)(2,2): {arr2d[[0, 1, 2], [0, 1, 2]]}")
```

#### 输出

```text
原数组: [10 11 12 13 14 15 16 17 18 19]
arr[[0, 2, 5, 8]]: [10 12 15 18]

二维数组:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

选第 0、2 行:
[[ 0  1  2  3]
 [ 8  9 10 11]]

对角线 (0,0)(1,1)(2,2): [ 0  5 10]
```

#### 理解重点

- 花式索引返回**拷贝**，不是视图——与布尔索引一致
- 传入两个索引数组时，结果是"行列逐对组合"而非笛卡儿积——需要笛卡儿积用 `np.ix_`

## 5. 条件函数

### `np.where`

#### 作用

有两种使用模式：
1. `np.where(condition)` ——返回满足条件的元素**索引**（元组）
2. `np.where(condition, x, y)` ——条件三元替换，$True$ 取 $x$，$False$ 取 $y$

#### 重点方法

```python
np.where(condition, [x, y])
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `condition` | `array_like` | 布尔数组或可广播到目标形状的条件表达式 | `arr > 0` |
| `x` | `scalar` 或 `array_like` | 条件为 `True` 时取的值，与 `y` 必须同时提供 | `arr`、`1` |
| `y` | `scalar` 或 `array_like` | 条件为 `False` 时取的值 | `0`、`-1` |

#### 示例代码

```python
import numpy as np

arr = np.array([1, -2, 3, -4, 5, -6])

# 模式 1：取索引
idx = np.where(arr > 0)
print(f"正数的索引: {idx[0]}")
print(f"正数的值: {arr[idx]}")

# 模式 2：条件替换
print(f"负数替换为 0: {np.where(arr > 0, arr, 0)}")
print(f"正数标 1 其他 -1: {np.where(arr > 0, 1, -1)}")
```

#### 输出

```text
正数的索引: [0 2 4]
正数的值: [1 3 5]
负数替换为 0: [1 0 3 0 5 0]
正数标 1 其他 -1: [ 1 -1  1 -1  1 -1]
```

#### 理解重点

- `np.where(cond)` 返回**元组**，一维场景取 `[0]` 得到索引数组
- `np.where(cond, x, y)` 中 `x`、`y` 可以是标量或数组，自动广播到 `condition` 的形状

## 常见坑

1. 切片返回**视图**，花式索引和布尔索引返回**拷贝**——修改行为不同
2. 二维索引推荐 `arr[i, j]` 而非 `arr[i][j]`——后者多一次视图创建
3. 布尔条件组合必须加括号且用 `&`/`|`：`(arr >= 3) & (arr <= 7)` 而非 `arr >= 3 & arr <= 7`
4. `np.where(cond, x, y)` 中 `x`/`y` 必须同时提供；只传 `cond` 是取索引模式
5. 传两个索引数组取二维元素时是"行列配对"而非笛卡儿积——需要子矩阵用 `np.ix_`

## 小结

- 索引决定了对数组的表达力，是一切后续操作的基础
- 布尔索引和 `np.where` 是数据清洗、特征工程的高频工具
- **切片 = 视图**，**花式/布尔索引 = 拷贝**——这个区别影响内存、性能和是否修改原数组

# NumPy 运算与统计

## 本章目标

1. 掌握 NumPy 的逐元素算术运算与比较运算
2. 掌握常用统计函数：求和、均值、方差、极值、累积
3. 理解 `axis` 参数在多维聚合中的语义
4. 熟练使用三角、指对、取整等常见数学函数
5. 掌握布尔数组的逻辑运算

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `+` `-` `*` `/` `**` `//` `%` | 运算符 | 逐元素算术运算 |
| `==` `!=` `>` `<` `>=` `<=` | 运算符 | 逐元素比较，返回布尔数组 |
| `np.array_equal(...)` | 函数 | 判断两个数组是否完全相等 |
| `np.any(...)` / `np.all(...)` | 函数 | 至少一个为真 / 全部为真 |
| `arr.sum(...)` / `arr.mean(...)` | 方法 | 求和 / 均值，支持按轴聚合 |
| `arr.std(...)` / `arr.var(...)` | 方法 | 标准差 / 方差 |
| `arr.min(...)` / `arr.max(...)` | 方法 | 最小值 / 最大值 |
| `arr.argmin(...)` / `arr.argmax(...)` | 方法 | 极值所在索引 |
| `arr.cumsum(...)` / `arr.cumprod(...)` | 方法 | 累积和 / 累积积 |
| `np.sin(...)` / `np.cos(...)` | 函数 | 三角函数（弧度） |
| `np.exp(...)` / `np.log(...)` | 函数 | 指数 / 自然对数 |
| `np.floor(...)` / `np.ceil(...)` / `np.round(...)` | 函数 | 向下 / 向上取整 / 四舍五入 |
| `np.abs(...)` | 函数 | 绝对值 |
| `np.logical_and/or/not/xor(...)` | 函数 | 布尔数组的逻辑运算 |

## 1. 算术运算

#### 作用

NumPy 的算术运算符全部为逐元素操作（element-wise）。标量与数组运算时标量自动广播到每个位置。`*` 是逐元素乘而非矩阵乘法。

### 运算符一览

| 运算符 | 含义 | 示例 |
|---|---|---|
| `+` | 逐元素加 | `a + b` |
| `-` | 逐元素减 | `a - b` |
| `*` | 逐元素乘（非矩阵乘法） | `a * b` |
| `/` | 逐元素除，返回浮点 | `a / b` |
| `**` | 逐元素幂 | `a ** 2` |
| `//` | 逐元素整除（地板除） | `a // 2` |
| `%` | 逐元素取模 | `a % 2` |

### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
print(f"a // 2 = {a // 2}")
print(f"a % 2 = {a % 2}")
```

### 输出

```text
a + b = [ 6  8 10 12]
a - b = [-4 -4 -4 -4]
a * b = [ 5 12 21 32]
a / b = [0.2        0.33333333 0.42857143 0.5       ]
a ** 2 = [ 1  4  9 16]
a // 2 = [0 1 1 2]
a % 2 = [1 0 1 0]
```

### 理解重点

- `*` 是逐元素乘，**不是**矩阵乘法——矩阵乘法用 `@` 或 `np.dot`
- 整数数组做 `/` 返回浮点数组；做 `//` 保持整数

## 2. 比较与聚合

### 比较运算符

| 运算符 | 含义 | 返回 |
|---|---|---|
| `==` | 逐元素等于 | 布尔数组 |
| `!=` | 逐元素不等 | 布尔数组 |
| `>` | 逐元素大于 | 布尔数组 |
| `<` | 逐元素小于 | 布尔数组 |
| `>=` | 逐元素大于等于 | 布尔数组 |
| `<=` | 逐元素小于等于 | 布尔数组 |

### `np.array_equal`

#### 作用

判断两个数组是否完全相等（形状一致且每个元素值相等）。这是整体判断，不同于逐元素的 `a == b`。

#### 重点方法

```python
np.array_equal(a1, a2, equal_nan=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a1` | `array_like` | 第一个数组 | `a` |
| `a2` | `array_like` | 第二个数组 | `b` |
| `equal_nan` | `bool` | 是否将两个 `NaN` 视为相等（NumPy 1.19+），默认为 `False` | `True` |

### `np.any` / `np.all`

#### 作用

- `np.any`：至少一个元素为真（逻辑或）
- `np.all`：所有元素均为真（逻辑与）

两者均支持 `axis` 按轴聚合。

#### 重点方法

```python
np.any(a, axis=None, out=None, keepdims=False, *, where=True)
np.all(a, axis=None, out=None, keepdims=False, *, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组，非布尔值会按 `bool(x)` 转换 | `a == b` |
| `axis` | `int` 或 `None` | 聚合轴，默认为 `None`（对全部元素） | `0` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `keepdims` | `bool` | 是否保留被聚合的轴（长度为 1），默认为 `False` | `True` |
| `where` | `array_like` | 只对 `True` 位置参与聚合，默认为 `True` | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([4, 3, 2, 1])

print(f"a == b: {a == b}")
print(f"a > b: {a > b}")
print(f"np.array_equal(a, b): {np.array_equal(a, b)}")
print(f"np.any(a == b): {np.any(a == b)}")
print(f"np.all(a != b): {np.all(a != b)}")
```

#### 输出

```text
a == b: [False False False False]
a > b: [False False  True  True]
np.array_equal(a, b): False
np.any(a == b): False
np.all(a != b): True
```

## 3. 统计函数

以下 API 是 `ndarray` 的方法，同时存在同名的 `np.xxx` 函数，两者等价。

### `ndarray.sum`

#### 作用

对数组元素求和，支持按 `axis` 聚合。全局求和公式：

$$
\text{sum} = \sum_{i} a_i
$$

#### 重点方法

```python
arr.sum(axis=None, dtype=None, out=None, keepdims=False, initial=<no value>, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `tuple[int, ...]` 或 `None` | 聚合轴，默认为 `None`（全部元素） | `0`、`1` |
| `dtype` | `dtype` 或 `None` | 累加中间类型，整数数组可指定 `np.int64` 避免溢出 | `np.int64` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `keepdims` | `bool` | 保留被聚合的轴（长度为 1），便于后续广播，默认为 `False` | `True` |
| `initial` | `scalar` | 聚合初始值 | `0` |
| `where` | `array_like` | 只对 `True` 位置参与聚合，默认为 `True` | —— |

### `ndarray.mean`

#### 作用

计算算术平均值，签名与 `sum` 高度一致。公式：

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

#### 重点方法

```python
arr.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)
```

参数含义同 `sum`，不再重复。

### `ndarray.std` / `ndarray.var`

#### 作用

计算标准差 / 方差。`ddof` 控制自由度修正：

- $ddof=0$（默认）：总体方差 $\sigma^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$
- $ddof=1$：样本方差 $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$

#### 重点方法

```python
arr.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)
arr.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `tuple` 或 `None` | 聚合轴 | `0`、`1` |
| `ddof` | `int` | 自由度修正，默认为 `0`（总体）；`1` 为样本（无偏） | `1` |
| `keepdims` | `bool` | 是否保留聚合轴，默认为 `False` | `True` |

### `ndarray.min` / `ndarray.max`

#### 作用

返回数组的最小值 / 最大值，支持按轴聚合。

#### 重点方法

```python
arr.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
arr.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
```

### `ndarray.argmin` / `ndarray.argmax`

#### 作用

返回最小值 / 最大值所在的**索引**。多维时 `axis=None` 返回扁平后的整数索引。

#### 重点方法

```python
arr.argmin(axis=None, out=None, keepdims=False)
arr.argmax(axis=None, out=None, keepdims=False)
```

### `ndarray.cumsum` / `ndarray.cumprod`

#### 作用

沿指定轴计算累积和 / 累积积。累积和公式：

$$
\text{cumsum}_k = \sum_{i=1}^{k} x_i
$$

#### 重点方法

```python
arr.cumsum(axis=None, dtype=None, out=None)
arr.cumprod(axis=None, dtype=None, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `axis` | `int` 或 `None` | `None` 先展平再累积，默认为 `None` | `0` |
| `dtype` | `dtype` 或 `None` | 中间类型，整数乘积易溢出可升高精度 | `np.int64` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.randint(1, 100, size=10)

print(f"数组: {arr}")
print(f"sum={arr.sum()}, mean={arr.mean():.2f}")
print(f"std={arr.std():.2f}, var={arr.var():.2f}")
print(f"min={arr.min()}, max={arr.max()}")
print(f"argmin={arr.argmin()}, argmax={arr.argmax()}")
print(f"cumsum: {arr.cumsum()}")
print(f"前 5 个 cumprod: {arr[:5].cumprod()}")
```

#### 输出

```text
数组: [52 93 15 72 61 21 83 87 75 75]
sum=634, mean=63.40
std=25.37, var=643.64
min=15, max=93
argmin=2, argmax=1
cumsum: [ 52 145 160 232 293 314 397 484 559 634]
前 5 个 cumprod: [       52      4836     72540   5222880 318595680]
```

## 4. 沿轴聚合（axis）

#### 作用

对多维数组聚合时，`axis` 指定沿哪个轴方向进行。核心直觉（二维）：

- `axis=0`：沿**行方向**走 → 对每一列聚合，结果形状 = 列数
- `axis=1`：沿**列方向**走 → 对每一行聚合，结果形状 = 行数
- `axis=None`（默认）：对所有元素聚合为标量

**口诀**："聚合轴就是被吃掉的轴"——`axis=0` 聚合后行维度消失，剩下列。

#### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原数组:\n{arr}")
print(f"全体 sum(): {arr.sum()}")
print(f"按列 sum(axis=0): {arr.sum(axis=0)}")
print(f"按行 sum(axis=1): {arr.sum(axis=1)}")
print(f"按列 mean(axis=0): {arr.mean(axis=0)}")
print(f"按行 mean(axis=1): {arr.mean(axis=1)}")
```

#### 输出

```text
原数组:
[[1 2 3]
 [4 5 6]
 [7 8 9]]
全体 sum(): 45
按列 sum(axis=0): [12 15 18]
按行 sum(axis=1): [ 6 15 24]
按列 mean(axis=0): [4. 5. 6.]
按行 mean(axis=1): [2. 5. 8.]
```

#### 理解重点

- 三维以上建议始终用 `keepdims=True` 调试，结果形状比不带 `keepdims` 直观
- `axis=0` 聚合结果取第一行看：`[12, 15, 18]` = `[1+4+7, 2+5+8, 3+6+9]`

## 5. 数学函数

### `np.sin` / `np.cos`

#### 作用

计算三角函数，输入按**弧度**计算（非角度）。转换公式：$\text{rad} = \theta° \times \frac{\pi}{180}$。

#### 重点方法

```python
np.sin(x, /, out=None, *, where=True, dtype=None)
np.cos(x, /, out=None, *, where=True, dtype=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 输入弧度值 | `[0, np.pi/2, np.pi]` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `where` | `array_like` | 只对 `True` 位置执行运算 | —— |
| `dtype` | `dtype` 或 `None` | 指定输出类型 | —— |

### `np.exp`

#### 作用

计算自然指数 $e^x$，其中 $e \approx 2.71828$。

#### 重点方法

```python
np.exp(x, /, out=None, *, where=True, dtype=None)
```

### `np.log` / `np.log10`

#### 作用

`np.log` 计算自然对数 $\ln x$，`np.log10` 计算常用对数 $\log_{10} x$。输入必须 $>0$，否则返回 `-inf` 或 `nan`。

#### 重点方法

```python
np.log(x, /, out=None, *, where=True, dtype=None)
np.log10(x, /, out=None, *, where=True, dtype=None)
```

### `np.floor` / `np.ceil`

#### 作用

向下取整 / 向上取整。注意"向下"对负数而言是更负方向：$floor(-1.2) = -2$。

#### 重点方法

```python
np.floor(x, /, out=None, *, where=True, dtype=None)
np.ceil(x, /, out=None, *, where=True, dtype=None)
```

### `np.round`

#### 作用

四舍五入到指定小数位。NumPy 采用**银行家舍入**（`.5` 向偶数舍入），结果可能与"严格四舍五入"不同。

#### 重点方法

```python
np.round(a, decimals=0, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1.5, 2.5, 3.5]` |
| `decimals` | `int` | 保留小数位数，可为负数（舍入到 10/100 位），默认为 `0` | `2` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### `np.abs`

#### 作用

计算绝对值。复数输入时返回模长：$|a + bi| = \sqrt{a^2 + b^2}$。

#### 重点方法

```python
np.abs(x, /, out=None, *, where=True, dtype=None)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"sin: {np.sin(arr).round(3)}")
print(f"cos: {np.cos(arr).round(3)}")

arr2 = np.array([1, 2, 3, 4, 5])
print(f"exp: {np.exp(arr2).round(3)}")
print(f"log: {np.log(arr2).round(3)}")
print(f"log10: {np.log10(arr2).round(3)}")

arr3 = np.array([1.2, 2.5, 3.7, -1.2, -2.5])
print(f"floor: {np.floor(arr3)}")
print(f"ceil:  {np.ceil(arr3)}")
print(f"round: {np.round(arr3)}")
print(f"abs:   {np.abs(arr3)}")
```

#### 输出

```text
sin: [0.    0.5   0.707 0.866 1.   ]
cos: [1.    0.866 0.707 0.5   0.   ]
exp: [  2.718   7.389  20.086  54.598 148.413]
log: [0.    0.693 1.099 1.386 1.609]
log10: [0.    0.301 0.477 0.602 0.699]
floor: [ 1.  2.  3. -2. -3.]
ceil:  [ 2.  3.  4. -1. -2.]
round: [ 1.  2.  4. -1. -2.]
abs:   [1.2 2.5 3.7 1.2 2.5]
```

## 6. 逻辑函数

专门用于布尔数组的逻辑运算。输入非布尔时先按 `bool(x)` 转换。

### `np.logical_and`

#### 作用

按位逻辑与，等价于 `a & b`。

#### 重点方法

```python
np.logical_and(x1, x2, /, out=None, *, where=True, dtype=None)
```

### `np.logical_or`

#### 作用

按位逻辑或，等价于 `a | b`。

#### 重点方法

```python
np.logical_or(x1, x2, /, out=None, *, where=True, dtype=None)
```

### `np.logical_not`

#### 作用

按位逻辑非，等价于 `~a`。

#### 重点方法

```python
np.logical_not(x, /, out=None, *, where=True, dtype=None)
```

### `np.logical_xor`

#### 作用

按位异或：相同为 `False`，不同为 `True`。

#### 重点方法

```python
np.logical_xor(x1, x2, /, out=None, *, where=True, dtype=None)
```

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(f"logical_and: {np.logical_and(a, b)}")
print(f"logical_or:  {np.logical_or(a, b)}")
print(f"logical_not(a): {np.logical_not(a)}")
print(f"logical_xor: {np.logical_xor(a, b)}")
```

#### 输出

```text
logical_and: [ True False False False]
logical_or:  [ True  True  True False]
logical_not(a): [False False  True  True]
logical_xor: [False  True  True False]
```

## 常见坑

1. `*` 是逐元素乘，**不是**矩阵乘法——矩阵乘法用 `@` 或 `np.dot`
2. `axis` 方向容易理解反：`axis=0` 是对列聚合（行被吃掉），不是"对第 0 行操作"
3. `np.round` 遵循**银行家舍入**（`.5` 向偶数舍入），`2.5` → `2`，`3.5` → `4`
4. `ddof=0`（默认）是**有偏**方差 / 标准差；推断统计应用 `ddof=1` 做无偏估计
5. `np.log(0)` 返回 `-inf` 并发出 `RuntimeWarning`；负数返回 `nan`
6. 大整数数组做 `cumprod` 极易溢出，应先 `.astype(np.int64)` 或转浮点

## 小结

- 运算与统计是 NumPy 在数据分析中的核心价值：无循环处理任意维度数据
- 熟练使用 `axis` 是 NumPy 进阶的分水岭
- 统计函数的 `ddof`、`keepdims` 两个参数对结果影响大，每次调用时明确意图
- ufunc 的通用参数（`out`、`where`、`dtype`）是内存和性能优化的入口

# NumPy 线性代数

## 本章目标

1. 掌握向量点积与矩阵乘法（`dot` / `matmul` / `@`）的区别与使用场景
2. 掌握转置、行列式、逆矩阵的基本用法
3. 会求方阵的特征值与特征向量
4. 会用 `np.linalg.solve` 解线性方程组 $Ax = b$
5. 掌握 `np.linalg.norm` 计算常见向量与矩阵范数

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.dot(...)` | 函数 | 点积 / 矩阵乘法（兼容多维） |
| `np.matmul(...)` / `@` | 函数 / 运算符 | 矩阵乘法（高维批量） |
| `arr.T` | 属性 | 转置视图（反转所有维度） |
| `np.transpose(...)` | 函数 | 转置，可指定轴顺序 |
| `np.linalg.det(...)` | 函数 | 方阵行列式 |
| `np.linalg.inv(...)` | 函数 | 方阵逆矩阵 |
| `np.linalg.eig(...)` | 函数 | 特征值与特征向量 |
| `np.linalg.solve(...)` | 函数 | 解线性方程组 $Ax = b$ |
| `np.linalg.norm(...)` | 函数 | 向量 / 矩阵范数 |
| `np.allclose(...)` | 函数 | 浮点容差比较 |

## 1. 点积与矩阵乘法

### `np.dot`

#### 作用

向量点积或矩阵乘法，行为随输入维度变化：
- 两个一维向量 → 点积 $a \cdot b = \sum a_i b_i$，返回标量
- 两个二维矩阵 → 矩阵乘法 $C_{ik} = \sum_j A_{ij} B_{jk}$
- 高维数组 → 沿最后两轴做矩阵乘法

#### 重点方法

```python
np.dot(a, b, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 第一个输入 | `[1, 2, 3]`、`A(2x2)` |
| `b` | `array_like` | 第二个输入，形状需与 `a` 在相应轴上匹配 | `[4, 5, 6]`、`B(2x2)` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |

### `np.matmul` / `@`

#### 作用

专用的矩阵乘法，`@` 是运算符写法。与 `dot` 的差异：
- **不支持**标量乘法
- 高维时视为"批量矩阵乘法"：最后两轴是矩阵维度，前轴做广播

#### 重点方法

```python
np.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None)
# 等价运算符
A @ B
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x1` | `array_like` | 左矩阵 | `A(2x2)` |
| `x2` | `array_like` | 右矩阵，列数需与 `x1` 行数匹配 | `B(2x2)` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |
| `casting` | `str` | 类型转换策略，默认为 `'same_kind'` | `'safe'` |
| `dtype` | `dtype` 或 `None` | 结果类型 | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"向量点积 np.dot(a, b) = {np.dot(a, b)}")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A @ B:\n{A @ B}")
print(f"np.dot(A, B):\n{np.dot(A, B)}")
```

#### 输出

```text
向量点积 np.dot(a, b) = 32
A @ B:
[[19 22]
 [43 50]]
np.dot(A, B):
[[19 22]
 [43 50]]
```

#### 理解重点

- 现代代码优先用 `@`（可读性高、仅限矩阵乘法）
- 需要向量点积或对标量操作时用 `np.dot`
- 三维以上批量矩阵乘用 `matmul` / `@`，不要用 `dot`

## 2. 转置

### `arr.T`

#### 作用

返回数组的转置**视图**，所有维度顺序反转。对二维数组即常规"行列交换"。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `ndarray` | 转置视图，修改会影响原数组 |

### `np.transpose`

#### 作用

比 `arr.T` 更灵活，可通过 `axes` 参数指定任意的轴排列顺序。

#### 重点方法

```python
np.transpose(a, axes=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `A(2x3)` |
| `axes` | `tuple[int, ...]` 或 `None` | 新轴顺序，默认为 `None`（反转所有维度） | `(1, 0)`、`(1, 0, 2)` |

### 示例代码

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A (2x3):\n{A}")
print(f"A.T (3x2):\n{A.T}")
print(f"np.transpose(A):\n{np.transpose(A)}")
```

### 输出

```text
A (2x3):
[[1 2 3]
 [4 5 6]]
A.T (3x2):
[[1 4]
 [2 5]
 [3 6]]
np.transpose(A):
[[1 4]
 [2 5]
 [3 6]]
```

## 3. 行列式与逆矩阵

### `np.linalg.det`

#### 作用

计算方阵的行列式。行列式为 $0$ 表示矩阵奇异（不可逆）。公式（以 $2 \times 2$ 为例）：

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

#### 重点方法

```python
np.linalg.det(a)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 方阵，至少二维；高维时对最后两轴批量计算 | `[[4, 7], [2, 6]]` |

### `np.linalg.inv`

#### 作用

计算方阵的逆矩阵 $A^{-1}$，满足 $A A^{-1} = I$。对奇异或接近奇异的矩阵数值不稳定，应改用 `solve`。

#### 重点方法

```python
np.linalg.inv(a)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 方阵；行列式为 $0$ 时抛 `LinAlgError` | `[[4, 7], [2, 6]]` |

### `np.allclose`

#### 作用

判断两个数组在给定容差内是否近似相等。浮点数不应直接 `==` 比较，应用 `allclose`。判断条件：

$$
|a - b| \le \text{atol} + \text{rtol} \cdot |b|
$$

#### 重点方法

```python
np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 待比较的数组 1 | `A @ A_inv` |
| `b` | `array_like` | 待比较的数组 2 | `np.eye(2)` |
| `rtol` | `float` | 相对容差，默认为 `1e-05` | `1e-10` |
| `atol` | `float` | 绝对容差，默认为 `1e-08` | `1e-12` |
| `equal_nan` | `bool` | `True` 时将两个 `NaN` 视为相等，默认为 `False` | `True` |

### 综合示例

#### 示例代码

```python
import numpy as np

A = np.array([[4, 7], [2, 6]])
det = np.linalg.det(A)
AInv = np.linalg.inv(A)

print(f"det(A) = {det:.4f}")
print(f"A^-1:\n{AInv}")
print(f"A @ A^-1:\n{(A @ AInv).round(10)}")
print(f"是否单位矩阵: {np.allclose(A @ AInv, np.eye(2))}")
```

#### 输出

```text
det(A) = 10.0000
A^-1:
[[ 0.6 -0.7]
 [-0.2  0.4]]
A @ A^-1:
[[ 1.  0.]
 [-0.  1.]]
是否单位矩阵: True
```

## 4. 特征值与特征向量

### `np.linalg.eig`

#### 作用

计算方阵 $A$ 的特征值与特征向量，满足：

$$
A v = \lambda v
$$

返回二元组 `(eigenvalues, eigenvectors)`。

#### 重点方法

```python
np.linalg.eig(a)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入方阵 | `[[4, 2], [1, 3]]` |

#### 返回值

| 返回值 | 类型 | 含义 |
|---|---|---|
| `eigenvalues` | `ndarray` | 一维数组，长度等于方阵阶数 |
| `eigenvectors` | `ndarray` | 二维数组，**第 `i` 列**是 `eigenvalues[i]` 对应的特征向量 |

#### 示例代码

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证 A @ v = λ @ v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"λ={lam:.2f}: A@v = {A @ v}, λ*v = {lam * v}, "
          f"相等={np.allclose(A @ v, lam * v)}")
```

#### 输出

```text
特征值: [5. 2.]
特征向量:
[[ 0.89442719 -0.70710678]
 [ 0.4472136   0.70710678]]
λ=5.00: A@v = [4.47213595 2.23606798], λ*v = [4.47213595 2.23606798], 相等=True
λ=2.00: A@v = [-1.41421356  1.41421356], λ*v = [-1.41421356  1.41421356], 相等=True
```

#### 理解重点

- 特征向量按**列**存放，切片用 `eigvecs[:, i]`，不是 `eigvecs[i]`
- 特征值可能为复数（即使输入全为实数），返回类型可能为 `complex128`
- 对称矩阵用 `np.linalg.eigh` 得到更稳定的实值结果

## 5. 解线性方程组

### `np.linalg.solve`

#### 作用

解线性方程组 $Ax = b$。比 $\text{inv}(A) @ b$ 更快更稳定，应始终作为首选。本质是通过 LU 分解等数值方法直接求解，避免显式计算逆矩阵。

#### 重点方法

```python
np.linalg.solve(a, b)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 系数方阵，必须可逆 | `[[2, 1], [1, 3]]` |
| `b` | `array_like` | 右端向量（或多个右端组成的矩阵） | `[5, 7]` |

#### 示例代码

```python
import numpy as np

# 方程组:
#   2x + y = 5
#   x + 3y = 7
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

x = np.linalg.solve(A, b)
print(f"解 x = {x}")
print(f"验证 A @ x = {A @ x}")
print(f"是否等于 b: {np.allclose(A @ x, b)}")
```

#### 输出

```text
解 x = [1.6 1.8]
验证 A @ x = [5. 7.]
是否等于 b: True
```

#### 理解重点

- **永远优先 `solve`**，而不是 `inv(A) @ b`——前者更快且数值更稳定
- 非方阵或超定 / 欠定系统用最小二乘 `np.linalg.lstsq`

## 6. 范数

### `np.linalg.norm`

#### 作用

计算向量或矩阵的范数。通过 `ord` 参数选择不同范数类型。

- 向量 L2 范数（欧几里得距离）：$\|v\|_2 = \sqrt{\sum |v_i|^2}$
- 向量 L1 范数（曼哈顿距离）：$\|v\|_1 = \sum |v_i|$
- 矩阵 Frobenius 范数：$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}$

#### 重点方法

```python
np.linalg.norm(x, ord=None, axis=None, keepdims=False)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `x` | `array_like` | 向量或矩阵输入 | `[3, 4]`、`A(2x2)` |
| `ord` | `int` 或 `str` 或 `None` | 范数阶数，默认为 `None`（见下表） | `1`、`2`、`np.inf` |
| `axis` | `int` 或 `None` | 指定按轴计算范数，默认为 `None`（整体计算） | `0` |
| `keepdims` | `bool` | 是否保留被聚合的轴，默认为 `False` | `True` |

#### `ord` 常见取值

| `ord` | 向量含义 | 矩阵含义（二维输入） |
|---|---|---|
| `None`（默认） | L2 范数 $\sqrt{\sum x_i^2}$ | Frobenius 范数 |
| `1` | L1 范数 $\sum \|x_i\|$ | 列绝对值和的最大值 |
| `2` | L2 范数（同 `None`） | 最大奇异值（谱范数） |
| `np.inf` | 最大绝对值 $\max \|x_i\|$ | 行绝对值和的最大值 |
| `-np.inf` | 最小绝对值 $\min \|x_i\|$ | 行绝对值和的最小值 |
| `'fro'` | — | Frobenius 范数 |

#### 示例代码

```python
import numpy as np

v = np.array([3, 4])
print(f"L1 范数 (曼哈顿距离): {np.linalg.norm(v, ord=1)}")
print(f"L2 范数 (欧几里得距离): {np.linalg.norm(v, ord=2)}")
print(f"无穷范数: {np.linalg.norm(v, ord=np.inf)}")

A = np.array([[1, 2], [3, 4]])
print(f"Frobenius 范数: {np.linalg.norm(A):.4f}")
```

#### 输出

```text
L1 范数 (曼哈顿距离): 7.0
L2 范数 (欧几里得距离): 5.0
无穷范数: 4.0
Frobenius 范数: 5.4772
```

#### 理解重点

- 向量 $[3, 4]$ 的 L2 范数为 $\sqrt{3^2 + 4^2} = 5$
- 矩阵默认范数是 Frobenius，等于把矩阵展平为向量后的 L2 范数

## 常见坑

1. 只有方阵才能直接 `inv` / `det` / `eig`——非方阵考虑 SVD 或伪逆 `np.linalg.pinv`
2. 奇异或接近奇异的矩阵求逆数值不稳定，条件数大时结果误差大
3. 解线性方程组**永远**用 `np.linalg.solve`，不要写 `np.linalg.inv(A) @ b`
4. 比较浮点结果用 `np.allclose`，不要直接用 `==`
5. `@` 和 `np.dot` 在高维数组上的广播语义不同：批量矩阵乘用 `@` / `matmul`
6. `np.linalg.eig` 的特征向量按**列**存放，切片写成 `eigvecs[i]` 是错误

## 小结

- 本章是机器学习线代工具箱的基础：`solve`、`eig`、`norm` 出现频率最高
- 理解维度匹配规则（$m \times k$ @ $k \times n \to m \times n$）是避免维度错误的关键
- 浮点比较用 `allclose`，求逆用 `solve`——这两点能避开 90% 的初学者陷阱

# NumPy 变形

## 本章目标

1. 掌握 `reshape` 的规则与 `-1` 自动推导维度
2. 区分 `flatten`（副本）与 `ravel`（视图）的语义差异
3. 掌握 `transpose` 对多维数组的轴重排
4. 掌握 `squeeze` / `expand_dims` / `np.newaxis` 对维度长度的增删
5. 理解 `np.resize` 与 `reshape` 的差别

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr.reshape(...)` | 方法 | 改变形状，元素总数不变 |
| `arr.flatten(...)` | 方法 | 展平为一维，**总是返回副本** |
| `arr.ravel(...)` | 方法 | 展平为一维，**尽量返回视图** |
| `arr.T` | 属性 | 转置视图（反转所有维度） |
| `np.transpose(...)` | 函数 | 转置，可指定轴顺序 |
| `np.squeeze(...)` | 函数 | 删除长度为 1 的维��� |
| `np.expand_dims(...)` | 函数 | 在指定轴插入长度为 1 的维度 |
| `np.newaxis` | 常量 | 等价于 `None`，在切片中插入新轴 |
| `np.resize(...)` | 函数 | 调整大小，可改变元素总数（不足时循环填充） |

## 1. 形状变换

### `ndarray.reshape`

#### 作用

改变数组形状而不改变元素总数。返回视图（内存连续时）或副本。一个维度可写 `-1` 让 NumPy 自动推导。

#### 重点方法

```python
arr.reshape(*shape, order='C')
# 或传元组:
arr.reshape(shape, order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `shape` | `int` 或 `tuple[int, ...]` | 目标形状，可用 `-1` 自动推导一个维度；元素总数必须一致 | `(3, 4)`、`(2, -1)`、`(-1, 6)` |
| `order` | `str` | 读写顺序：`'C'` 行优先 / `'F'` 列优先 / `'A'` 任意 / `'K'` 保持内存顺序，默认为 `'C'` | `'F'` |

#### 示例代码

```python
import numpy as np

arr = np.arange(1, 13)
print(f"原数组: {arr}, shape={arr.shape}")
print(f"reshape(3, 4):\n{arr.reshape(3, 4)}")
print(f"reshape(4, 3):\n{arr.reshape(4, 3)}")
print(f"reshape(2, -1):\n{arr.reshape(2, -1)}")
print(f"reshape(-1, 6):\n{arr.reshape(-1, 6)}")
```

#### 输出

```text
原数组: [ 1  2  3  4  5  6  7  8  9 10 11 12], shape=(12,)
reshape(3, 4):
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
reshape(4, 3):
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
reshape(2, -1):
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
reshape(-1, 6):
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]
```

#### 理解重点

- `-1` 最多出现一次，由 NumPy 根据元素总数与其他维度反推
- 变形前后元素总数必须相同，否则抛 `ValueError`
- `reshape` 通常返回视图，修改结果会影响原数组

## 2. 展平

### `ndarray.flatten`

#### 作用

将数组展平为一维，**总是返回副本**。修改返回结果不影响原数组。

#### 重点方法

```python
arr.flatten(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 读取顺序：`'C'` 行优先 / `'F'` 列优先 / `'A'` / `'K'`，默认为 `'C'` | `'F'` |

### `ndarray.ravel`

#### 作用

将数组展平为一维，**尽量返回视图**（若内存连续）。修改结果可能影响原数组。比 `flatten` 高效但须注意副作用。

#### 重点方法

```python
arr.ravel(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 读取顺序，同 `flatten`，默认为 `'C'` | `'F'` |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

flat = arr.flatten()
flat[0] = 999  # 不影响原数组

rav = arr.ravel()
rav[1] = 888   # 影响原数组（视图）

print(f"flatten 副本修改不影响原数组:\n{arr}")
```

### 输出

```text
flatten 副本修改不影响原数组:
[[  1 888   3]
 [  4   5   6]]
```

### 理解重点

- `flatten` = "一定是副本"，安全但有拷贝开销——适合传出去的数据
- `ravel` = "能视图就视图"，高效但可能联动修改——适合临时读取
- 不确定时用 `flatten` 或显式 `.copy()`，安全性优先

## 3. 转置与轴重排

### `ndarray.T`

#### 作用

返回数组转置视图，等价于反转所有轴。对二维即行列交换。

#### 返回内容

| 类型 | 含义 |
|---|---|
| `ndarray` | 转置视图，修改会影响原数组 |

### `np.transpose`

#### 作用

比 `.T` 更灵活，可通过 `axes` 参数指定任意轴的排列顺序，适合多维张量的轴重排。

#### 重点方法

```python
np.transpose(a, axes=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `arr3d` |
| `axes` | `tuple[int, ...]` 或 `None` | 新的轴顺序，默认为 `None`（反转所有轴），元组长度需等于 `a.ndim` | `(1, 0, 2)` |

### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"arr.T:\n{arr.T}")

arr3d = np.arange(24).reshape(2, 3, 4)
print(f"arr_3d.shape: {arr3d.shape}")
print(f"arr_3d.T.shape: {arr3d.T.shape}")
print(f"transpose((1,0,2)).shape: "
      f"{np.transpose(arr3d, axes=(1, 0, 2)).shape}")
```

### 输出

```text
arr.T:
[[1 4]
 [2 5]
 [3 6]]
arr_3d.shape: (2, 3, 4)
arr_3d.T.shape: (4, 3, 2)
transpose((1,0,2)).shape: (3, 2, 4)
```

## 4. 维度增删

### `np.squeeze`

#### 作用

删除数组中长度为 1 的维度。常用于从 `(1, n)` 或 `(1, 1, n)` 恢复到 `(n,)`。

#### 重点方法

```python
np.squeeze(a, axis=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `(1, 1, 3)` |
| `axis` | `int` 或 `tuple[int, ...]` 或 `None` | 只删除指定轴（必须长度为 1），默认为 `None`（删除所有长度为 1 的轴） | `0` |

### `np.expand_dims`

#### 作用

在指定位置插入一个长度为 1 的维度。常用于给一维向量增加 batch 或 channel 维度。

#### 重点方法

```python
np.expand_dims(a, axis)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 2, 3]` |
| `axis` | `int` | 插入新维度的位置，支持负索引 | `0`、`1`、`-1` |

### `np.newaxis`

#### 作用

一个常量（实际是 `None`），在切片中使用可在指定位置插入长度为 1 的维度，是 `expand_dims` 的语法糖。

### 示例代码

```python
import numpy as np

# squeeze
arr = np.array([[[1, 2, 3]]])
print(f"原形状: {arr.shape}")
print(f"squeeze 后: {np.squeeze(arr).shape}")

# expand_dims
v = np.array([1, 2, 3])
print(f"原 v.shape: {v.shape}")
print(f"expand_dims(axis=0).shape: "
      f"{np.expand_dims(v, axis=0).shape}")
print(f"expand_dims(axis=1).shape: "
      f"{np.expand_dims(v, axis=1).shape}")

# newaxis
print(f"v[np.newaxis, :].shape: {v[np.newaxis, :].shape}")
print(f"v[:, np.newaxis].shape: {v[:, np.newaxis].shape}")
```

### 输出

```text
原形状: (1, 1, 3)
squeeze 后: (3,)
原 v.shape: (3,)
expand_dims(axis=0).shape: (1, 3)
expand_dims(axis=1).shape: (3, 1)
v[np.newaxis, :].shape: (1, 3)
v[:, np.newaxis].shape: (3, 1)
```

### 理解重点

- `(n,)`（一维向量）与 `(1, n)`（行向量）与 `(n, 1)`（列向量）在广播中语义不同——它们不可互换
- 模型输入加 batch 维度：`x[np.newaxis, ...]` 是常用写法
- `squeeze` 删尺寸为 1 的轴比 `reshape` 更语义化

## 5. 改变元素总数

### `np.resize`

#### 作用

与 `reshape` 不同，`np.resize` **可以改变元素总数**。空间不足时循环重复原数组元素填充；空间多余时截断。

#### 重点方法

```python
np.resize(a, new_shape)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 2, 3, 4]` |
| `new_shape` | `int` 或 `tuple[int, ...]` | 目标形状，可大于或小于原元素总数 | `(2, 4)`、`(3, 3)` |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(f"resize((2, 4)):\n{np.resize(arr, (2, 4))}")
print(f"resize((3, 3)):\n{np.resize(arr, (3, 3))}")
```

#### 输出

```text
resize((2, 4)):
[[1 2 3 4]
 [1 2 3 4]]
resize((3, 3)):
[[1 2 3]
 [4 1 2]
 [3 4 1]]
```

#### 理解重点

- `np.resize(arr, shape)`（函数形式）循环填充；`arr.resize(shape)`（方法形式）原地修改且不循环填充——两者行为不同
- `reshape` 元素总数不变；`resize` 总数可变——功能不同，不可互换

## 常见坑

1. `reshape` 失败通常因为元素总数不匹配——先查 `arr.size` 再写目标形状
2. `ravel` 返回视图还是副本取决于内存布局，**不要假设**——确定要独立数据时用 `flatten` 或 `.copy()`
3. 列向量 `(n, 1)` 与行向量 `(1, n)` 在广播和矩阵乘法中语义完全不同——不要省 `np.newaxis`
4. `np.resize` 和 `arr.resize` 行为不同：函数形式循环填充，方法形式原地填 0 扩展
5. `transpose(axes=...)` 中 `axes` 长度必须等于数组维度

## 小结

- **总数不变**用 `reshape`；**总数可变**用 `resize`
- **只插入/删除长度为 1 的维度**优先 `np.newaxis` / `squeeze`，比 `reshape` 更语义化
- 牢记"副本 vs 视图"：`flatten` = 副本，`ravel` / `reshape` = 通常视图

# NumPy 广播

## 本章目标

1. 掌握广播（broadcasting）的三条核心规则
2. 掌握标量、行向量、列向量与矩阵的广播计算
3. 学会用广播实现外积
4. 掌握广播在数据标准化中的典型应用

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `arr + scalar` | 表达式 | 标量与数组广播（逐元素运算） |
| `arr2d + arr1d` | 表达式 | 一维向量沿行方向广播 |
| `arr2d + col` | 表达式 | 列向量 `(n, 1)` 沿列方向广播 |
| `a[:, np.newaxis] * b` | 表达式 | 用广播手动构造外积 |
| `np.outer(...)` | 函数 | 外积（一维向量专用 API） |
| `arr.mean(axis=0)` | 方法 | 按列均值，配合广播用于标准化 |
| `arr.std(axis=0)` | 方法 | 按列标准差，配合广播用于标准化 |

## 1. 广播规则

### 三条核心规则

1. **从后往前**（末尾对齐）逐维比较两个数组的 `shape`
2. 每一维满足**相等**或**其中一个为 1**即可广播
3. **缺失的维度**按 `1` 处理，自动在前方补上

### 形状兼容速查

| 左形状 | 右形状 | 是否兼容 | 广播后形状 |
|---|---|---|---|
| `(3, 4)` | `(4,)` | 是 | `(3, 4)` |
| `(3, 4)` | `(3, 1)` | 是 | `(3, 4)` |
| `(3, 1)` | `(1, 4)` | 是 | `(3, 4)` |
| `(2, 3, 4)` | `(3, 4)` | 是 | `(2, 3, 4)` |
| `(3, 4)` | `(3,)` | 否 | — |
| `(3, 4)` | `(2, 4)` | 否 | — |

### 示例代码

```python
import numpy as np

A = np.ones((3, 4))
B1 = np.array([1, 2, 3, 4])         # (4,)
B2 = np.array([[1], [2], [3]])      # (3, 1)

print(f"(3,4) + (4,) = {(A + B1).shape}")
print(f"(3,4) + (3,1) = {(A + B2).shape}")
print(f"(3,1) + (1,4) = "
      f"{(np.ones((3, 1)) + np.ones((1, 4))).shape}")
```

### 输出

```text
(3,4) + (4,) = (3, 4)
(3,4) + (3,1) = (3, 4)
(3,1) + (1,4) = (3, 4)
```

## 2. 标量与数组广播

#### 作用

标量（0 维）与任意形状的数组均可广播，标量自动扩展为同形状后逐元素运算。

#### 示例代码

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"arr + 10:\n{arr + 10}")
print(f"arr * 2:\n{arr * 2}")
```

#### 输出

```text
arr + 10:
[[11 12 13]
 [14 15 16]
 [17 18 19]]
arr * 2:
[[ 2  4  6]
 [ 8 10 12]
 [14 16 18]]
```

## 3. 一维数组沿行广播

#### 作用

`arr2d (m, n)` 与 `arr1d (n,)` 相加时，一维数组被视作 `(1, n)` 后在行方向复制 `m` 份，相当于"每一行加同一个向量"。

#### 示例代码

```python
import numpy as np

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr1d = np.array([10, 20, 30])
print(f"arr2d + arr1d:\n{arr2d + arr1d}")
```

#### 输出

```text
arr2d + arr1d:
[[11 22 33]
 [14 25 36]
 [17 28 39]]
```

#### 理解重点

- `(m, n)` 与 `(n,)` 对齐后，第 0 轴长度自动补为 `1`，再广播到 `m`
- 若要"每一列加同一个向量"，需要形状 `(m, 1)` 而不是 `(m,)`

## 4. 列向量沿列广播

#### 作用

`arr2d (m, n)` 与 `col (m, 1)` 相加时，列向量在列方向复制 `n` 份，相当于"每一列加同一个列向量"。

#### 示例代码

```python
import numpy as np

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col = np.array([[100], [200], [300]])  # (3, 1)
print(f"arr2d + col:\n{arr2d + col}")
```

#### 输出

```text
arr2d + col:
[[101 102 103]
 [204 205 206]
 [307 308 309]]
```

#### 理解重点

- 想要"列向量"必须显式做成 `(m, 1)`：可用 `arr.reshape(-1, 1)`、`arr[:, np.newaxis]` 或 `np.expand_dims(arr, axis=1)`
- `(m,)` 与 `(m, 1)` 在广播中方向不同：前者沿行方向广播，后者沿列方向广播

## 5. 广播应用：外积

外积 $\mathbf{a} \otimes \mathbf{b}$ 的结果形状为 `(len(a), len(b))`，元素公式：

$$
(\mathbf{a} \otimes \mathbf{b})_{ij} = a_i \cdot b_j
$$

用广播的视角：把 $a$ 变成列向量 `(m, 1)`，$b$ 保持 `(n,)`，相乘即得。

### `np.outer`

#### 作用

直接计算两个一维向量的外积，等价于 `a[:, np.newaxis] * b`。

#### 重点方法

```python
np.outer(a, b, out=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 第一个输入，会被展平为一维 | `[1, 2, 3]` |
| `b` | `array_like` | 第二个输入，会被展平为一维 | `[10, 20, 30, 40]` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])

# 方式 1：广播
print(f"广播外积:\n{a[:, np.newaxis] * b}")

# 方式 2：np.outer
print(f"np.outer:\n{np.outer(a, b)}")
```

#### 输出

```text
广播外积:
[[ 10  20  30  40]
 [ 20  40  60  80]
 [ 30  60  90 120]]
np.outer:
[[ 10  20  30  40]
 [ 20  40  60  80]
 [ 30  60  90 120]]
```

## 6. 广播应用：特征标准化

Z-score 标准化公式：

$$
z = \frac{x - \mu}{\sigma}
$$

其中 $\mu$（均值）和 $\sigma$（标准差）沿样本维（`axis=0`）计算。它们形状为 `(n_features,)`，与数据 `(n_samples, n_features)` 相减/相除时自动沿行广播。

#### 示例代码

```python
import numpy as np

np.random.seed(42)
data = np.random.randint(0, 100, size=(5, 3))
print(f"原始数据:\n{data}")

mean = data.mean(axis=0)  # (3,)
std = data.std(axis=0)    # (3,)
normalized = (data - mean) / std

print(f"每列均值: {mean.round(2)}")
print(f"每列标准差: {std.round(2)}")
print(f"标准化后:\n{normalized.round(2)}")
print(f"标准化后均值: {normalized.mean(axis=0).round(10)}")
print(f"标准化后标准差: {normalized.std(axis=0).round(2)}")
```

#### 输出

```text
原始数据:
[[51 92 14]
 [71 60 20]
 [82 86 74]
 [74 87 99]
 [23  2 21]]
每列均值: [60.2 65.4 45.6]
每列标准差: [21.22 33.61 34.4 ]
标准化后:
[[-0.43  0.79 -0.92]
 [ 0.51 -0.16 -0.74]
 [ 1.03  0.61  0.83]
 [ 0.65  0.64  1.55]
 [-1.75 -1.89 -0.72]]
标准化后均值: [-0. -0. -0.]
标准化后标准差: [1. 1. 1.]
```

#### 理解重点

- `mean` / `std` 形状 `(3,)`，与 `data (5, 3)` 运算时自动沿行广播到 `(5, 3)`
- 标准化后每列均值为 0、标准差为 1——这正是 `StandardScaler` 的底层实现
- 用 `keepdims=True` 得 `(1, 3)` 形状，语义更明确，对高维也通用

## 常见坑

1. 维度不兼容时 NumPy 抛 `ValueError: operands could not be broadcast together`——先打印各自 `shape` 定位
2. `(n,)`、`(n, 1)`、`(1, n)` 在广播中语义完全不同，不要凭直觉写
3. 广播本身不复制数据（内部用 stride trick），但**运算结果**会分配新内存
4. 标准化时分母 `std` 可能为 0（常数列），结果出现 `inf` / `nan`，生产代码应加 `epsilon`
5. 高维张量建议始终用 `keepdims=True` 聚合，结果形状更可控

## 小结

- 广播是 NumPy 高性能表达力的核心，消除了大量显式循环
- 记住三条规则（后向对齐、为 1 可扩、缺失补 1），绝大多数场景都能自洽
- 主动使用 `np.newaxis` / `expand_dims` 控制广播方向，比依赖隐式 `(n,)` 更清晰

# NumPy 拼接与拆分

## 本章目标

1. 掌握数组拼接：`concatenate`、`vstack`、`hstack`、`stack`、`dstack`
2. 掌握数组拆分：`split`、`vsplit`、`hsplit`、`array_split`
3. 理解"沿现有轴拼接"与"沿新轴堆叠"的本质区别
4. 掌握等分与不等分拆分的正确 API 选择

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.concatenate(...)` | 函数 | 沿现有轴拼接多个数组 |
| `np.vstack(...)` | 函数 | 垂直拼接（相当于 `axis=0`） |
| `np.hstack(...)` | 函数 | 水平拼接（二维时相当于 `axis=1`） |
| `np.stack(...)` | 函数 | 沿新轴堆叠，结果维度 +1 |
| `np.dstack(...)` | 函数 | 沿第三轴（depth）堆叠 |
| `np.split(...)` | 函数 | 等分拆分，必须整除 |
| `np.vsplit(...)` / `np.hsplit(...)` | 函数 | 垂直 / 水平拆分 |
| `np.array_split(...)` | 函数 | 不均匀拆分，允许不能整除 |

## 1. 沿现有轴拼接

### `np.concatenate`

#### 作用

沿已有的某个轴将多个数组拼接到一起。非拼接轴的长度必须一致，拼接后维度不变。

#### 重点方法

```python
np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting='same_kind')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arrays` | `sequence of array_like` | 待拼接的数组序列，元素维度必须相同 | `[A, B]` |
| `axis` | `int` | 沿哪个现有轴拼接，默认为 `0`；`None` 时先展平再拼接 | `1` |
| `out` | `ndarray` 或 `None` | 写入结果的目标数组 | —— |
| `dtype` | `dtype` 或 `None` | 结果 dtype | —— |
| `casting` | `str` | 类型转换策略，默认为 `'same_kind'` | `'safe'` |

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"axis=0 垂直拼接:\n{np.concatenate([A, B], axis=0)}")
print(f"axis=1 水平拼接:\n{np.concatenate([A, B], axis=1)}")
```

#### 输出

```text
axis=0 垂直拼接:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
axis=1 水平拼接:
[[1 2 5 6]
 [3 4 7 8]]
```

#### 理解重点

- `axis=0`：行方向增长（叠行），**列数必须一致**
- `axis=1`：列方向增长（拼列），**行数必须一致**
- 一维数组拼接 `axis` 只能是 `0`

### `np.vstack`

#### 作用

垂直（纵向）堆叠。对二维数组等价于 `concatenate(axis=0)`；对一维数组会先视作行向量再堆叠。

#### 重点方法

```python
np.vstack(tup, *, dtype=None, casting='same_kind')
```

### `np.hstack`

#### 作用

水平（横向）堆叠。对二维数组等价于 `concatenate(axis=1)`；对一维数组等价于 `concatenate(axis=0)`。

#### 重点方法

```python
np.hstack(tup, *, dtype=None, casting='same_kind')
```

### 综合示例

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9]])           # (1, 3)
C = np.array([[10], [20]])          # (2, 1)

print(f"vstack([A, B]):\n{np.vstack([A, B])}")
print(f"hstack([A, C]):\n{np.hstack([A, C])}")
```

#### 输出

```text
vstack([A, B]):
[[1 2 3]
 [4 5 6]
 [7 8 9]]
hstack([A, C]):
[[ 1  2  3 10]
 [ 4  5  6 20]]
```

## 2. 沿新轴堆叠

### `np.stack`

#### 作用

在新插入的轴上堆叠多个形状完全一致的数组，结果维度比输入多 1。经典场景：把 $N$ 张 $(H, W)$ 图像堆成 $(N, H, W)$ 的 batch。

#### 重点方法

```python
np.stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `arrays` | `sequence of array_like` | 待堆叠的数组序列，**形状必须完全一致** | `[A, B]` |
| `axis` | `int` | 新轴插入位置，范围 $[0, ndim]$（可为负），默认为 `0` | `2` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |
| `dtype` | `dtype` 或 `None` | 结果 dtype | —— |
| `casting` | `str` | 类型转换策略，默认为 `'same_kind'` | —— |

#### 示例代码

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

s0 = np.stack([A, B], axis=0)
s2 = np.stack([A, B], axis=2)

print(f"stack axis=0 形状: {s0.shape}")
print(s0)
print(f"\nstack axis=2 形状: {s2.shape}")
print(s2)
```

#### 输出

```text
stack axis=0 形状: (2, 2, 2)
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

stack axis=2 形状: (2, 2, 2)
[[[1 5]
  [2 6]]

 [[3 7]
  [4 8]]]
```

#### 理解重点

- `concatenate` 不改变维度；`stack` 维度 +1——这是两者最核心的区别
- `axis=2` 时相当于把两张矩阵按像素位置"摞"在一起——类似图像通道堆叠

### `np.dstack`

#### 作用

沿第三轴（depth）堆叠。对一维数组视作 `(1, N, 1)`，对二维数组视作 `(M, N, 1)` 后拼接。常用于图像 RGB 三通道堆叠。

#### 重点方法

```python
np.dstack(tup)
```

## 3. 拆分

### `np.split`

#### 作用

将数组沿指定轴等分拆分。传整数 `n` 时要求该轴长度能被 `n` 整除；也可传索引列表精确控制分割点。

#### 重点方法

```python
np.split(ary, indices_or_sections, axis=0)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `ary` | `ndarray` | 待拆分数组 | `arr(4x3)` |
| `indices_or_sections` | `int` 或 `list[int]` | 整数为等分块数（必须整除）；列表为分割点索引 | `2`、`[2, 5]` |
| `axis` | `int` | 沿哪个轴拆分，默认为 `0` | `1` |

### `np.vsplit` / `np.hsplit`

#### 作用

- `vsplit` 等价于 `split(axis=0)`，沿行方向拆分
- `hsplit` 等价于 `split(axis=1)`，沿列方向拆分

#### 重点方法

```python
np.vsplit(ary, indices_or_sections)
np.hsplit(ary, indices_or_sections)
```

### `np.array_split`

#### 作用

与 `np.split` 类似，但**允许整除不均**。无法整除时，`size % n` 块多一个元素。机器学习交叉验证、batch 划分常用。

#### 重点方法

```python
np.array_split(ary, indices_or_sections, axis=0)
```

### 综合示例

#### 示例代码

```python
import numpy as np

# 等分拆分
arr = np.arange(12).reshape(4, 3)
print(f"原数组:\n{arr}")

partsRow = np.split(arr, 2, axis=0)
print(f"\nsplit(axis=0) 分 2 块:")
for i, p in enumerate(partsRow):
    print(f"第 {i+1} 块:\n{p}")

# 不等分拆分
arr1d = np.arange(10)
parts = np.array_split(arr1d, 3)
for i, p in enumerate(parts):
    print(f"第 {i+1} 块(大小 {len(p)}): {p}")
```

#### 输出

```text
原数组:
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

split(axis=0) 分 2 块:
第 1 块:
[[0 1 2]
 [3 4 5]]
第 2 块:
[[ 6  7  8]
 [ 9 10 11]]
第 1 块(大小 4): [0 1 2 3]
第 2 块(大小 3): [4 5 6]
第 3 块(大小 3): [7 8 9]
```

#### 理解重点

- 10 个元素分 3 份 → `4, 3, 3`：前面 `size % n` 块各多一个元素
- 等分优先 `split`（明确意图），不整除用 `array_split`（容忍不齐）

## 常见坑

1. `np.split(arr, n)` 不能整除会抛 `ValueError`——不确定整除时用 `np.array_split`
2. `np.stack` 要求所有输入形状完全相同；不同形状应先 `reshape` / `pad` 对齐
3. `concatenate` 的 `axis` 容易写反——拼接前先 `print(a.shape, b.shape)` 排查
4. `hstack` 对一维与二维行为不同：一维等价 `axis=0`，二维等价 `axis=1`
5. `stack` 和 `concatenate` 不可互换：前者加维度，后者不加维度

## 小结

- 拼接与拆分是数据批处理、窗口构造、特征组合的核心操作
- 选 API 的思路：**沿现有轴** → `concatenate` / `vstack` / `hstack`；**沿新轴** → `stack` / `dstack`
- 拆分首选 `split`（严格），不整除退化到 `array_split`
- 先想清楚"要沿哪个轴变化"，再选具体 API

# NumPy 文件 IO

## 本章目标

1. 掌握二进制保存/加载：`save`、`load`、`savez`、`savez_compressed`
2. 掌握文本保存/加载：`savetxt`、`loadtxt`
3. 掌握 `fmt` / `delimiter` / `header` / `skiprows` 等格式控制参数
4. 理解 `.npy` / `.npz` / `.csv` 三种格式的选择场景

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.save(...)` | 函数 | 保存单个数组到 `.npy` 二进制文件 |
| `np.savez(...)` | 函数 | 保存多个数组到 `.npz` 容器 |
| `np.savez_compressed(...)` | 函数 | 同 `savez`，额外压缩存储 |
| `np.load(...)` | 函数 | 加载 `.npy` / `.npz` 文件 |
| `np.savetxt(...)` | 函数 | 保存数组为可读文本（`.txt` / `.csv`） |
| `np.loadtxt(...)` | 函数 | 从文本文件读取数组 |

## 1. 二进制：`.npy` 单数组

### `np.save`

#### 作用

将单个 NumPy 数组保存为 `.npy` 二进制文件，保真度高、速度快，是 NumPy 内部数据的首选格式。文件自动追加 `.npy` 后缀。

#### 重点方法

```python
np.save(file, arr, allow_pickle=True, fix_imports=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径；不带 `.npy` 后缀会自动追加 | `"array.npy"` |
| `arr` | `array_like` | 要保存的数组 | —— |
| `allow_pickle` | `bool` | 是否允许保存 object 数组（依赖 pickle，存在安全风险），默认为 `True` | `False` |
| `fix_imports` | `bool` | Python 2/3 兼容选项，默认为 `True` | —— |

### `np.load`

#### 作用

加载 `.npy` 或 `.npz` 文件。`.npy` 返回单个数组；`.npz` 返回可按键访问的容器对象。大文件可配合 `mmap_mode` 内存映射加载。

#### 重点方法

```python
np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径 | `"array.npy"` |
| `mmap_mode` | `str` 或 `None` | 内存映射模式：`'r'` / `'r+'` / `'c'` / `'w+'`；大文件避免全量加载 | `'r'` |
| `allow_pickle` | `bool` | 是否允许反序列化 object 数组，默认为 `False`（安全） | `True` |
| `encoding` | `str` | object 数组读取时的字符编码，默认为 `'ASCII'` | `'latin1'` |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

np.save("array.npy", arr)
loaded = np.load("array.npy")

print(f"原数组:\n{arr}")
print(f"加载后:\n{loaded}")
print(f"相等: {np.array_equal(arr, loaded)}")
```

#### 输出

```text
原数组:
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
加载后:
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]]
相等: True
```

#### 理解重点

- `.npy` 保存完整的 `dtype` / `shape` / 内存布局，读写往返完全无损
- 大文件用 `mmap_mode='r'` 实现按需读取，不占内存

## 2. 二进制：`.npz` 多数组

### `np.savez`

#### 作用

将多个数组保存到一个 `.npz` 容器中。用关键字参数命名，后续按键读取。

#### 重点方法

```python
np.savez(file, *args, **kwds)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `file` | `str` 或 `file-like` | 文件路径，自动追加 `.npz` 后缀 | `"arrays.npz"` |
| `*args` | `array_like` | 位置参数保存的数组，自动命名为 `arr_0`、`arr_1`... | —— |
| `**kwds` | `array_like` | 关键字参数保存的数组，按给定名称存取 | `a=arr1, b=arr2` |

### `np.savez_compressed`

#### 作用

语义与 `savez` 完全相同，但内部压缩存储，文件体积更小，适合分发或归档。

#### 重点方法

```python
np.savez_compressed(file, *args, **kwds)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2], [3, 4]])
arr3 = np.arange(10)

np.savez("arrays.npz", a=arr1, b=arr2, c=arr3)
data = np.load("arrays.npz")

print(f"包含的数组: {list(data.keys())}")
print(f"data['a']: {data['a']}")
print(f"data['b']:\n{data['b']}")
print(f"data['c']: {data['c']}")
```

#### 输出

```text
包含的数组: ['a', 'b', 'c']
data['a']: [1 2 3 4 5]
data['b']:
[[1 2]
 [3 4]]
data['c']: [0 1 2 3 4 5 6 7 8 9]
```

## 3. 文本：`savetxt` / `loadtxt`

### `np.savetxt`

#### 作用

将一维或二维数组保存为可读文本（`.txt` / `.csv`）。支持自定义分隔符、格式化、表头。

#### 重点方法

```python
np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n',
           header='', footer='', comments='# ', encoding=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fname` | `str` 或 `file-like` | 文件路径 | `"array.csv"` |
| `X` | `array_like` | 一维或二维数组 | `arr(3, 4)` |
| `fmt` | `str` | 格式字符串，遵循 C 的 `printf` 语法，默认为 `'%.18e'` | `'%.4f'`、`'%d'` |
| `delimiter` | `str` | 列分隔符，默认为 `' '`（空格） | `','` |
| `newline` | `str` | 行间分隔符，默认为 `'\n'` | —— |
| `header` | `str` | 文件头部附加字符串，默认为 `''` | `"col1,col2,col3"` |
| `footer` | `str` | 文件末尾附加字符串，默认为 `''` | —— |
| `comments` | `str` | `header` / `footer` 前的注释前缀，默认为 `'# '`；写纯 CSV 头时必须设为 `''` | `''` |
| `encoding` | `str` 或 `None` | 写入编码，默认为 `None` | `'utf-8'` |

#### `fmt` 常见写法

| `fmt` | 含义 | 示例：`1.23456` |
|---|---|---|
| `'%.2f'` | 保留 2 位小数 | `1.23` |
| `'%.4f'` | 保留 4 位小数 | `1.2346` |
| `'%d'` | 整数格式 | `1` |
| `'%.2e'` | 科学计数法 2 位小数 | `1.23e+00` |
| `'%10.4f'` | 宽度 10，4 位小数 | <code>&nbsp;&nbsp;&nbsp;1.2346</code> |
| `'%s'` | 字符串格式 | — |

### `np.loadtxt`

#### 作用

从文本文件读取数值数组。支持自定义分隔符、跳行、选取列等。

#### 重点方法

```python
np.loadtxt(fname, dtype=float, comments='#', delimiter=None,
           skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `fname` | `str` 或 `file-like` | 文件路径 | `"array.csv"` |
| `dtype` | `dtype` | 结果数据类型，默认为 `float` | `int` |
| `comments` | `str` | 以此字符开头的行视为注释跳过，默认为 `'#'` | —— |
| `delimiter` | `str` 或 `None` | 列分隔符，默认为 `None`（任意空白字符） | `','` |
| `skiprows` | `int` | 跳过文件头部行数，常用于跳过 CSV 表头，默认为 `0` | `1` |
| `usecols` | `tuple[int, ...]` | 指定读取哪些列（从 0 开始），默认为 `None`（全部） | `(0, 2)` |
| `unpack` | `bool` | `True` 时返回按列解包的结果，便于 `x, y = loadtxt(...)` | `True` |
| `ndmin` | `int` | 结果最小维度，避免单行/单列被降维为一维，默认为 `0` | `2` |
| `encoding` | `str` | 读取编码，默认为 `'bytes'` | `'utf-8'` |

### 综合示例

#### 示例代码

```python
import numpy as np

np.random.seed(42)
arr = np.random.random((3, 4))

# CSV 格式，带表头
np.savetxt(
    "array.csv", arr,
    delimiter=",", fmt="%.4f",
    header="col1,col2,col3,col4",
    comments="",
)

# 加载（跳过表头）
loaded = np.loadtxt("array.csv", delimiter=",", skiprows=1)
print(f"加载一致: {np.allclose(arr, loaded)}")
```

#### `array.csv` 文件内容

```text
col1,col2,col3,col4
0.3745,0.9507,0.7320,0.5987
0.1560,0.1560,0.0581,0.8662
0.6011,0.7081,0.0206,0.9699
```

#### 理解重点

- `comments=''` 不可省略：默认 `'# '` 会让表头变成 `# col1,col2,...`，`loadtxt` 将其识别为注释
- 文本读写有精度损失（由 `fmt` 决定）；需无损保存用 `.npy`
- 验证读写往返用 `np.allclose`，不用 `==`（浮点精度可能微差）

## 4. 格式选择指南

| 场景 | 格式 | API |
|---|---|---|
| 训练/实验中间结果（单数组） | `.npy` | `save` / `load` |
| 多数组（权重、配置、数据） | `.npz` | `savez` / `load` |
| 分发或归档（多数组 + 压缩） | `.npz` | `savez_compressed` / `load` |
| 与 Excel/pandas/其他语言交换 | `.csv` | `savetxt` / `loadtxt` |
| 大文件（超出内存） | `.npy` + mmap | `load(mmap_mode='r')` |

## 常见坑

1. `np.loadtxt` 遇到表头行会报 `ValueError`——加 `skiprows=1` 跳过
2. 写 CSV 时忘记 `comments=''`，表头被加 `#` 前缀变成注释
3. `fmt` 精度过低会丢失有效数字——科学计算默认用 `'%.18e'`
4. `np.load` 的 `allow_pickle=False` 是默认值——加载旧版 object 数组文件时须手动开 `True`（仅在信任来源时）
5. CSV 分隔符可能是 `,` / `;` / `\t`——读写必须前后一致
6. 内存紧张场景用 `mmap_mode='r'` 做内存映射，避免全量加载

## 小结

- 训练过程优先 `.npy`（单数组）或 `.npz`（多数组）——无损且快速
- 跨语言/跨工具交换数据用 `.csv`，明确指定 `delimiter` 和 `fmt`
- 写入端明确格式、读取端明确解析规则——避免隐式错误
- 大文件或内存紧张时用 `mmap_mode='r'` 按需读取

# NumPy 实用函数

## 本章目标

1. 掌握排序 `sort` 与索引排序 `argsort` 的区别
2. 掌握去重 `unique` 及其附加输出（索引、计数、逆映射）
3. 掌握集合运算：`intersect1d` / `union1d` / `setdiff1d` / `setxor1d` / `isin`
4. 掌握搜索：`argmax` / `argmin` / `where` / `nonzero`
5. 掌握裁剪 `clip` 与取整函数
6. 理解引用 / 视图 / 副本三种语义的区别

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.sort(...)` | 函数 | 返回排序后的新数组 |
| `np.argsort(...)` | 函数 | 返回排序后的索引 |
| `np.unique(...)` | 函数 | 返回唯一值，可选索引/计数/逆映射 |
| `np.intersect1d(...)` | 函数 | 两数组交集 |
| `np.union1d(...)` | 函数 | 两数组并集 |
| `np.setdiff1d(...)` | 函数 | 差集 $a \setminus b$ |
| `np.setxor1d(...)` | 函数 | 对称差集 |
| `np.isin(...)` | 函数 | 元素级成员检测（替代已弃用的 `in1d`） |
| `np.argmax(...)` / `np.argmin(...)` | 函数 | 最大 / 最小值索引 |
| `np.where(...)` | 函数 | 条件索引或三元替换 |
| `np.nonzero(...)` | 函数 | 非零元素索引 |
| `np.clip(...)` | 函数 | 截断到 $[a_{min}, a_{max}]$ 区间 |
| `np.floor/ceil/round/trunc(...)` | 函数 | 向下/向上/四舍五入/截断取整 |
| `arr.view(...)` | 方法 | 返回共享数据的视图 |
| `arr.copy(...)` | 方法 | 返回独立数据的副本 |

## 1. 排序

### `np.sort`

#### 作用

返回排序后的新数组，不修改原数组。支持多种排序算法和多维沿轴排序。

#### 重点方法

```python
np.sort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[3, 1, 4, 1, 5]` |
| `axis` | `int` 或 `None` | 沿哪个轴排序，默认为 `-1`（最后一轴）；`None` 先展平 | `0` |
| `kind` | `str` 或 `None` | 算法：`'quicksort'` / `'mergesort'` / `'heapsort'` / `'stable'`，默认为 `'quicksort'` | `'stable'` |
| `stable` | `bool` 或 `None` | `True` 保持相等元素原序（等价 `kind='stable'`） | `True` |

### `np.argsort`

#### 作用

返回将数组排序所需的索引。`arr[np.argsort(arr)]` 等价于 `np.sort(arr)`。常用于跟随排序多个数组。

#### 重点方法

```python
np.argsort(a, axis=-1, kind=None, order=None, stable=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[3, 1, 4, 1, 5]` |
| `axis` | `int` 或 `None` | 沿哪个轴排序，默认为 `-1` | `0` |
| `kind` | `str` 或 `None` | 排序算法 | `'stable'` |
| `stable` | `bool` 或 `None` | 是否稳定排序 | `True` |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"sort: {np.sort(arr)}")
print(f"argsort: {np.argsort(arr)}")
print(f"arr[argsort]: {arr[np.argsort(arr)]}")

arr2d = np.array([[3, 1, 2], [6, 4, 5]])
print(f"按行排序 axis=1:\n{np.sort(arr2d, axis=1)}")
print(f"按列排序 axis=0:\n{np.sort(arr2d, axis=0)}")
```

#### 输出

```text
sort: [1 1 2 3 4 5 6 9]
argsort: [1 3 6 0 2 4 7 5]
arr[argsort]: [1 1 2 3 4 5 6 9]
按行排序 axis=1:
[[1 2 3]
 [4 5 6]]
按列排序 axis=0:
[[3 1 2]
 [6 4 5]]
```

#### 理解重点

- `argsort` 返回"排序后位置对应的原索引"——可用于按 `scores` 排对应的 `names`
- 降序排序用 `np.sort(arr)[::-1]` 或 `arr[np.argsort(arr)[::-1]]`

## 2. 去重与计数

### `np.unique`

#### 作用

返回数组中的唯一值（默认已升序排列）。可选返回首次出现位置、逆映射、出现次数。

#### 重点方法

```python
np.unique(ar, return_index=False, return_inverse=False,
          return_counts=False, axis=None, *, equal_nan=True)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `ar` | `array_like` | 输入数组 | `[1, 2, 2, 3, 3, 3]` |
| `return_index` | `bool` | `True` 返回每个唯一值首次出现的索引，默认为 `False` | `True` |
| `return_inverse` | `bool` | `True` 返回把原数组映射回唯一值的逆索引，默认为 `False` | `True` |
| `return_counts` | `bool` | `True` 返回每个唯一值的出现次数，默认为 `False` | `True` |
| `axis` | `int` 或 `None` | `None` 先展平；指定轴时按行/列去重，默认为 `None` | `0` |
| `equal_nan` | `bool` | 是否将多个 `NaN` 视为同一值，默认为 `True` | —— |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

print(f"唯一值: {np.unique(arr)}")
print(f"含首次索引: {np.unique(arr, return_index=True)}")
print(f"含计数: {np.unique(arr, return_counts=True)}")
```

#### 输出

```text
唯一值: [1 2 3 4]
含首次索引: (array([1, 2, 3, 4]), array([0, 1, 3, 6]))
含计数: (array([1, 2, 3, 4]), array([1, 2, 3, 4]))
```

#### 理解重点

- `return_counts=True` 是一行代码做频数统计的最快方式
- 值-计数对：`dict(zip(*np.unique(arr, return_counts=True)))`

## 3. 集合运算

所有集合运算都先展平再去重，结果一维且升序排列。

### 接口速览

| API | 数学符号 | 含义 |
|---|---|---|
| `np.intersect1d(a, b)` | $A \cap B$ | 交集 |
| `np.union1d(a, b)` | $A \cup B$ | 并集 |
| `np.setdiff1d(a, b)` | $A \setminus B$ | 差集（顺序敏感） |
| `np.setxor1d(a, b)` | $A \triangle B$ | 对称差集（只在一边出现） |
| `np.isin(a, b)` | $a_i \in B$ | 成员检测，返回布尔数组 |

### `np.isin`

#### 作用

对数组 `element` 的每个元素判断是否出现在 `test_elements` 中，返回同形状布尔数组。替代已弃用的 `np.in1d`。

#### 重点方法

```python
np.isin(element, test_elements, assume_unique=False, invert=False, *, kind=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `element` | `array_like` | 被检测的数组 | `a` |
| `test_elements` | `array_like` | 参考集合 | `[2, 4]` |
| `assume_unique` | `bool` | 假设两边已去重以加速，默认为 `False` | `True` |
| `invert` | `bool` | `True` 时返回"不在集合中"的掩码，默认为 `False` | `True` |
| `kind` | `str` 或 `None` | 算法提示：`'sort'` 或 `'table'`（NumPy 1.23+） | —— |

### 综合示例

#### 示例代码

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])

print(f"交集: {np.intersect1d(a, b)}")
print(f"并集: {np.union1d(a, b)}")
print(f"差集 a-b: {np.setdiff1d(a, b)}")
print(f"差集 b-a: {np.setdiff1d(b, a)}")
print(f"对称差: {np.setxor1d(a, b)}")
print(f"a 是否在 [2, 4]: {np.isin(a, [2, 4])}")
```

#### 输出

```text
交集: [3 4 5]
并集: [1 2 3 4 5 6 7]
差集 a-b: [1 2]
差集 b-a: [6 7]
对称差: [1 2 6 7]
a 是否在 [2, 4]: [False  True False  True False]
```

## 4. 搜索

### `np.argmax` / `np.argmin`

#### 作用

返回最大值 / 最小值的索引。多维时 `axis=None` 返回扁平索引。

#### 重点方法

```python
np.argmax(a, axis=None, out=None, keepdims=False)
np.argmin(a, axis=None, out=None, keepdims=False)
```

### `np.nonzero`

#### 作用

返回非零元素的索引元组，每个轴对应一个索引数组。等价于 `np.where(arr != 0)`。

#### 重点方法

```python
np.nonzero(a)
```

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
print(f"argmax: {np.argmax(arr)}, argmin: {np.argmin(arr)}")

idx = np.where(arr > 5)
print(f"大于 5 的索引: {idx[0]}, 值: {arr[idx]}")

arr2 = np.array([0, 1, 0, 2, 0, 3])
print(f"非零索引: {np.nonzero(arr2)[0]}")
```

#### 输出

```text
argmax: 5, argmin: 0
大于 5 的索引: [3 5 7], 值: [8 9 7]
非零索引: [1 3 5]
```

## 5. 裁剪与取整

### `np.clip`

#### 作用

将数组元素截断到 $[a_{min}, a_{max}]$ 区间：小于 $a_{min}$ 取 $a_{min}$，大于 $a_{max}$ 取 $a_{max}$。

#### 重点方法

```python
np.clip(a, a_min, a_max, out=None, **kwargs)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `a` | `array_like` | 输入数组 | `[1, 5, 10, 15, 20]` |
| `a_min` | `scalar` 或 `None` | 下界，`None` 表示不设下界 | `5` |
| `a_max` | `scalar` 或 `None` | 上界，`None` 表示不设上界 | `15` |
| `out` | `ndarray` 或 `None` | 目标数组 | —— |

### 取整函数对比

| 函数 | 对 `-1.2` 结果 | 对 `2.5` 结果 | 规则 |
|---|---|---|---|
| `np.floor` | `-2.0` | `2.0` | 向下（更负方向） |
| `np.ceil` | `-1.0` | `3.0` | 向上（更正方向） |
| `np.round` | `-1.0` | `2.0` | 银行家舍入（`.5` 向偶数） |
| `np.trunc` | `-1.0` | `2.0` | 截断（向零方向） |

#### 示例代码

```python
import numpy as np

arr = np.array([1, 5, 10, 15, 20])
print(f"clip(5, 15): {np.clip(arr, 5, 15)}")

x = np.array([1.2, 2.5, 3.7, -1.2, -2.5, -3.7])
print(f"floor: {np.floor(x)}")
print(f"ceil:  {np.ceil(x)}")
print(f"round: {np.round(x)}")
print(f"trunc: {np.trunc(x)}")
```

#### 输出

```text
clip(5, 15): [ 5  5 10 15 15]
floor: [ 1.  2.  3. -2. -3. -4.]
ceil:  [ 2.  3.  4. -1. -2. -3.]
round: [ 1.  2.  4. -1. -2. -4.]
trunc: [ 1.  2.  3. -1. -2. -3.]
```

## 6. 引用、视图、副本

#### 三种语义对比

| 操作 | 共享数据 | 修改影响原数组 | 场景 |
|---|---|---|---|
| `ref = arr` | 是（同一对象） | 是 | Python 赋值 |
| `arr.view()` | 是 | 是 | 重读 dtype / 不同视角 |
| `arr.copy()` | 否 | 否 | 真正独立副本 |

### `arr.view`

#### 作用

返回共享底层数据的新数组对象。可重解读为不同 `dtype`（只要字节数匹配）。修改视图影响原数组。

#### 重点方法

```python
arr.view(dtype=None, type=None)
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `dtype` | `dtype` 或 `None` | 新的数据类型，默认为 `None`（保持原类型） | `np.int32` |
| `type` | `type` 或 `None` | 新的数组子类 | —— |

### `arr.copy`

#### 作用

返回完全独立的数据副本，修改不影响原数组。

#### 重点方法

```python
arr.copy(order='C')
```

#### 参数

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `order` | `str` | 内存布局：`'C'` 行优先 / `'F'` 列优先 / `'A'` 任意 / `'K'` 保持原样，默认为 `'C'` | `'F'` |

### 综合示例

#### 示例代码

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 引用
ref = arr
ref[0] = 100
print(f"改引用后，原数组: {arr}")

arr[0] = 1  # 复位

# 视图
v = arr.view()
v[1] = 200
print(f"改视图后，原数组: {arr}")

arr[1] = 2  # 复位

# 副本
c = arr.copy()
c[2] = 300
print(f"改副本后，原数组不变: {arr}")
```

#### 输出

```text
改引用后，原数组: [100   2   3   4   5]
改视图后，原数组: [  1 200   3   4   5]
改副本后，原数组不变: [1 2 3 4 5]
```

#### 理解重点

- 切片通常返回视图、花式索引/布尔索引返回副本——容易混淆
- 不确定时用 `arr.base` 判断：`None` 表示独立数据，非 `None` 表示视图
- 需要传出去且不希望被改的数据，显式 `.copy()` 最安全

## 常见坑

1. `argsort` 返回的是索引不是排序值——别直接当值用
2. `np.unique` 默认返回排序后的唯一值——非原序
3. `np.in1d` 已弃用——新代码用 `np.isin`
4. `np.round` 是银行家舍入（`.5` 向偶数）——需严格四舍五入时用 `np.floor(x + 0.5)`
5. 视图 vs 副本：误把视图当副本会出现"神秘联动修改"——大型项目优先显式 `.copy()`
6. `arr.view(np.int32)` 时 `float64` 的 8 字节被重读为两个 `int32`，形状也会变——这不是类型转换

## 小结

- 本章集合了日常数据处理中最常用的"工具抽屉"
- 集合运算、`unique`、`clip` 是数据清洗/EDA 的高频工具
- 搞清"引用 vs 视图 vs 副本"能避开 NumPy 最常见的一类 bug

# NumPy 综合实战

## 本章目标

1. 把前 11 章的知识串起来，解决完整的数据处理任务
2. 学会在真实场景中组合使用索引、统计、线代、广播
3. 建立"问题 → NumPy 算子"的映射能力

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `np.random.randint(...)` | 函数 | 生成整数随机样本 |
| `np.random.normal(...)` | 函数 | 生成正态分布样本 |
| `arr.sum(...)` / `arr.mean(...)` | 方法 | 聚合统计 |
| `arr.std(...)` / `arr.var(...)` | 方法 | 标准差 / 方差 |
| `np.argmax(...)` / `np.argmin(...)` | 函数 | 极值索引 |
| `np.argsort(...)` | 函数 | 排序索引，配合 `[::-1]` 得降序 |
| `np.column_stack(...)` | 函数 | 按列拼接多个一维数组成设计矩阵 |
| `np.linalg.solve(...)` | 函数 | 解线性方程组 $Ax = b$ |
| `np.rot90(...)` | 函数 | 二维数组逆时针旋转 90° |
| `np.percentile(...)` | 函数 | 分位数统计 |
| `np.histogram(...)` | 函数 | 区间频数统计 |

## 1. 案例：学生成绩分析

#### 问题描述

给定 5 名学生在 3 门课程上的成绩（$5 \times 3$ 矩阵），计算每个学生的总分与平均分、每门课程的统计指标，并给出总分排名。

#### 涉及方法

- `np.sum(axis=1)` / `np.mean(axis=1)`：学生维度聚合
- `np.mean/std/max/min(axis=0)`：课程维度聚合
- `np.argmax` / `np.argmin`：找最高/最低分学生
- `np.argsort(...)[::-1]`：降序排名

#### 示例代码

```python
import numpy as np

np.random.seed(42)
grades = np.random.randint(60, 101, size=(5, 3))
students = ["学生A", "学生B", "学生C", "学生D", "学生E"]
courses = ["数学", "英语", "物理"]

# 每个学生统计
totalScores = np.sum(grades, axis=1)
avgScores = np.mean(grades, axis=1)

# 每门课程统计
courseMean = np.mean(grades, axis=0)
courseStd = np.std(grades, axis=0)
courseMax = np.max(grades, axis=0)
courseMin = np.min(grades, axis=0)

# 排名
bestIdx = np.argmax(totalScores)
worstIdx = np.argmin(totalScores)
rankIndices = np.argsort(totalScores)[::-1]

print(f"成绩矩阵:\n{grades}")
print(f"总分: {totalScores}")
print(f"平均分: {avgScores.round(2)}")
print(f"课程均值: {courseMean.round(1)}")
print(f"课程标准差: {courseStd.round(1)}")
print(f"总分最高: {students[bestIdx]} ({totalScores[bestIdx]})")
print(f"总分最低: {students[worstIdx]} ({totalScores[worstIdx]})")
print(f"排名: {[students[i] for i in rankIndices]}")
```

#### 输出

```text
成绩矩阵:
[[98 88 74]
 [67 80 98]
 [78 82 70]
 [70 83 95]
 [99 83 62]]
总分: [260 245 230 248 244]
平均分: [86.67 81.67 76.67 82.67 81.33]
课程均值: [82.4 83.2 79.8]
课程标准差: [13.6  2.6 14.2]
总分最高: 学生A (260)
总分最低: 学生C (230)
排名: ['学生A', '学生D', '学生B', '学生E', '学生C']
```

#### 理解重点

- `axis=1` 沿列方向走 → 每行得一个聚合值（学生维度）
- `axis=0` 沿行方向走 → 每列得一个聚合值（课程维度）
- `argsort(...)[::-1]` 是降序索引的标准写法

## 2. 案例：线性回归

#### 问题描述

给定带噪声的一维特征 $x$ 和目标 $y$，用最小二乘法拟合 $y = w_1 x + w_0$。

#### 数学推导

最小二乘解：

$$
w = (X^T X)^{-1} X^T y
$$

为保证数值稳定性，不直接求逆，而是解正规方程：

$$
(X^T X)\, w = X^T y \quad\Rightarrow\quad w = \text{solve}(X^T X,\ X^T y)
$$

评估指标：

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}, \quad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

#### 涉及方法

- `np.column_stack([np.ones(n), x])`：构造设计矩阵 $X$（含偏置列）
- `X.T @ X`、`X.T @ y`：矩阵乘法构造正规方程
- `np.linalg.solve(A, b)`：解正规方程 $Aw = b$

#### 示例代码

```python
import numpy as np

np.random.seed(42)
n = 50
x = np.linspace(0, 10, n)
trueSlope, trueIntercept = 2, 1
noise = np.random.normal(0, 1, n)
y = trueSlope * x + trueIntercept + noise

# 正规方程
X = np.column_stack([np.ones(n), x])
w = np.linalg.solve(X.T @ X, X.T @ y)
intercept, slope = w

# 预测与指标
yPred = slope * x + intercept
ssRes = np.sum((y - yPred) ** 2)
ssTot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ssRes / ssTot
rmse = np.sqrt(np.mean((y - yPred) ** 2))

print(f"估计参数: y = {slope:.4f}x + {intercept:.4f}")
print(f"斜率误差: {abs(slope - trueSlope):.4f}")
print(f"截距误差: {abs(intercept - trueIntercept):.4f}")
print(f"R² = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")

# 验证：用 allclose 检查正规方程解的残差
residual = X.T @ X @ w - X.T @ y
print(f"残差接近零: {np.allclose(residual, 0)}")
```

#### 输出

```text
估计参数: y = 1.9420x + 1.0644
斜率误差: 0.0580
截距误差: 0.0644
R² = 0.9754
RMSE = 0.9084
残差接近零: True
```

#### 理解重点

- 永远不要写 `inv(X.T @ X) @ X.T @ y`——用 `solve`，更快更稳定
- 设计矩阵第一列是 `np.ones(n)`，对应截距项 $w_0$
- $R^2$ 越接近 1 拟合越好；RMSE 以目标变量的单位度量误差

## 3. 案例：图像矩阵操作

#### 问题描述

用 `uint8` 数组模拟一张 $8 \times 8$ 灰度图像，演示翻转、旋转、裁剪、归一化等常见操作。

#### 涉及方法

- `np.random.randint(0, 256, dtype=np.uint8)`：生成灰度图
- 切片 `[:, ::-1]` / `[::-1, :]`：水平/垂直翻转
- `np.rot90(m, k=1)`：逆时针旋转 90°
- `arr[2:6, 2:6]`：中心裁剪
- `arr.astype(np.float64) / 255.0`：归一化到 $[0, 1]$

#### 示例代码

```python
import numpy as np

np.random.seed(42)
image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

# 统计
print(f"min={image.min()}, max={image.max()}, "
      f"mean={image.mean():.1f}, std={image.std():.1f}")

# 变换
flippedH = image[:, ::-1]
flippedV = image[::-1, :]
rotated = np.rot90(image)
cropped = image[2:6, 2:6]
normalized = image.astype(np.float64) / 255.0

print(f"水平翻转前三行:\n{flippedH[:3]}")
print(f"旋转 90°:\n{rotated}")
print(f"裁剪 [2:6, 2:6]:\n{cropped}")
print(f"归一化后前两行:\n{normalized[:2].round(2)}")
```

#### 输出

```text
min=3, max=245, mean=137.3, std=73.7
水平翻转前三行:
[[255 242 191 106 182  19 188 102]
 [229 123  61  55   5 218  46  93]
 [ 24  31 214 147 144 198  20 248]]
旋转 90°:
[[102  93 248 117 192 141 202 255]
 [188  46  20 148  85 241 215 189]
 ...
 [219 203  51  95  12 191 190 212]
 [ 84 133 231 212 231 168 183  52]]
裁剪 [2:6, 2:6]:
[[ 99 187  71 212]
 [ 65 153  20  44]
 [240  39 121  24]
 [239  39 214 244]]
归一化后前两行:
[[0.4  0.86 0.88 0.37 0.7  0.24 0.92 0.8 ]
 [0.36 0.01 0.38 0.95 0.05 0.58 0.96 0.18]]
```

#### 理解重点

- `uint8` 溢出会回绕（$255 + 1 \to 0$）——做运算前先 `astype(np.float64)`
- 归一化时务必除以 `255.0`（浮点）而非 `255`（整数），避免整数除法截断
- 切片（翻转、裁剪）返回视图，需独立数据用 `.copy()`

## 4. 案例：统计分析与直方图

#### 问题描述

从 $\mathcal{N}(100, 15^2)$ 采样 1000 个点，计算常用统计量、分位数，并做 10 个区间的直方图统计。

#### 涉及方法

- `np.random.normal(loc, scale, size)`：生成正态样本
- `arr.mean()` / `arr.std()`：均值/标准差的点估计
- `np.percentile(a, q)`：分位数
- `np.histogram(a, bins)`：区间频数统计

#### 示例代码

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

print(f"均值: {data.mean():.2f}, 标准差: {data.std():.2f}")
print(f"范围: [{data.min():.2f}, {data.max():.2f}]")

# 分位数
for p in [25, 50, 75, 90, 95, 99]:
    print(f"P{p}: {np.percentile(data, p):.2f}")

# 直方图
hist, edges = np.histogram(data, bins=10)
for i in range(len(hist)):
    print(f"[{edges[i]:.1f}, {edges[i+1]:.1f}): {hist[i]}")
```

#### 输出

```text
均值: 100.29, 标准差: 14.68
范围: [51.38, 157.79]
P25: 90.29
P50: 100.38
P75: 109.72
P90: 119.58
P95: 125.15
P99: 134.74
[51.4, 62.0): 4
[62.0, 72.7): 22
[72.7, 83.3): 96
[83.3, 93.9): 228
[93.9, 104.6): 272
[104.6, 115.2): 226
[115.2, 125.9): 104
[125.9, 136.5): 38
[136.5, 147.1): 9
[147.1, 157.8): 1
```

#### 理解重点

- 样本均值/标准差围绕真实参数波动——样本量越大越接近真值
- `np.histogram` 返回 `(counts, edges)`，`edges` 长度比 `counts` 多 1
- 配合 matplotlib 的 `plt.hist(data, bins=10)` 可生成可视化图表

## 学习建议

1. 把四个案例分别拆成独立函数，自己重写一遍
2. 改变随机种子后观察指标的稳定性变化
3. 将线性回归案例扩展到多特征版本（$X$ 变为 $(n, p)$ 形状）
4. 给图像案例加上阈值分割（`np.where(image > 128, 255, 0)`）
5. 统计案例试试偏态分布（如 `np.random.lognormal`），观察分位数的变化

## 小结

- 本章的重点不是某个具体 API，而是组合能力——把基础算子拼成完整的数据处理流水线
- 成绩分析靠聚合 + 排序；线性回归靠矩阵运算 + `solve`；图像操作靠切片 + 类型转换；统计分析靠采样 + `histogram`
- 当你能用 NumPy 实现"数据加载 → 计算 → 统计 → 输出"的完整流水线，NumPy 就真正入门了
- 后续学习 Pandas、SciPy、scikit-learn 时，本章建立的计算思维会一直受用——它们底层都建立在 NumPy 之上
