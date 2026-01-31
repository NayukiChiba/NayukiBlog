---
title: NumPy 实用工具
date: 2026-01-10
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: NumPy 常用工具函数，包括排序、搜索、去重等
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 实用函数

## 学习目标

- 掌握 NumPy 的实用函数
- 学会排序和搜索操作
- 理解集合操作

## 为什么需要实用函数？

在数据处理中，除了基本的数值运算，还经常需要：

- **排序数据**：找出最大最小值，生成排行榜
- **去重统计**：统计不同类别的出现次数
- **集合运算**：找出两个数据集的交集、差集
- **搜索定位**：快速找到满足条件的元素位置

NumPy 提供了丰富的工具函数来高效完成这些任务。

## 排序函数

| 函数              | 说明             | 返回值   |
| ----------------- | ---------------- | -------- |
| `np.sort(arr)`    | 返回排序后的副本 | 新数组   |
| `arr.sort()`      | 原地排序         | None     |
| `np.argsort(arr)` | 返回排序后的索引 | 索引数组 |

### sort 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("原数组:", arr)
```

**结果**：

```
原数组: [3 1 4 1 5 9 2 6]
```

```python
# 方法1: np.sort() - 返回副本，原数组不变
sorted_arr = np.sort(arr)
print("排序后:", sorted_arr)
print("原数组:", arr)  # 原数组不变
```

**结果**：

```
排序后: [1 1 2 3 4 5 6 9]
原数组: [3 1 4 1 5 9 2 6]
```

**说明**：`np.sort()` 是非破坏性操作，返回新数组，原数组保持不变。

```python
# 方法2: arr.sort() - 原地排序，修改原数组
arr.sort()
print("原地排序后:", arr)
```

**结果**：

```
原地排序后: [1 1 2 3 4 5 6 9]
```

**说明**：`arr.sort()` 会直接修改原数组，不返回值。适合不需要保留原数组的场景。

### argsort 排序索引

`argsort` 不返回排序后的值，而是返回**排序后的索引**。这在很多场景下非常有用。

```python
arr = np.array([3, 1, 4, 1, 5])
print("原数组:", arr)

indices = np.argsort(arr)
print("排序索引:", indices)
```

**结果**：

```
原数组: [3 1 4 1 5]
排序索引: [1 3 0 2 4]
```

**理解排序索引**：

- 索引 `1` 对应值 `1`（最小）
- 索引 `3` 对应值 `1`（第二小）
- 索引 `0` 对应值 `3`（第三小）
- 索引 `2` 对应值 `4`（第四小）
- 索引 `4` 对应值 `5`（最大）

```python
# 使用索引重建排序后的数组
print("重建排序:", arr[indices])
```

**结果**：`[1 1 3 4 5]`

**应用场景**：当你需要根据一个数组的顺序来排列另一个数组时。

```python
# 示例：学生成绩排名
names = np.array(['Alice', 'Bob', 'Charlie', 'David'])
scores = np.array([85, 92, 78, 96])

# 按成绩从高到低排名
rank_indices = np.argsort(scores)[::-1]  # [::-1] 反转为降序
print("排名:")
for i, idx in enumerate(rank_indices, 1):
    print(f"  第{i}名: {names[idx]} - {scores[idx]}分")
```

**结果**：

```
排名:
  第1名: David - 96分
  第2名: Bob - 92分
  第3名: Alice - 85分
  第4名: Charlie - 78分
```

### 二维数组排序

```python
arr = np.array([[3, 1, 2],
                [6, 4, 5]])
print("原数组:\n", arr)
```

**结果**：

```
原数组:
 [[3 1 2]
  [6 4 5]]
```

```python
# 每行排序 (axis=1)
print("每行排序:\n", np.sort(arr, axis=1))
```

**结果**：

```
每行排序:
 [[1 2 3]
  [4 5 6]]
```

**过程**：每一行内部独立排序。

```python
# 每列排序 (axis=0)
print("每列排序:\n", np.sort(arr, axis=0))
```

**结果**：

```
每列排序:
 [[3 1 2]
  [6 4 5]]
```

**过程**：每一列内部独立排序（第一行是小值，第二行是大值）。

## 唯一值

`np.unique()` 用于找出数组中的唯一值，同时可以返回各种附加信息。

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print("原数组:", arr)
```

**结果**：`原数组: [1 2 2 3 3 3 4 4 4 4]`

### 基本用法

```python
# 获取唯一值
unique = np.unique(arr)
print("唯一值:", unique)
```

**结果**：`唯一值: [1 2 3 4]`

### 获取首次出现的索引

```python
unique, indices = np.unique(arr, return_index=True)
print("唯一值:", unique)
print("首次出现索引:", indices)
```

**结果**：

```
唯一值: [1 2 3 4]
首次出现索引: [0 1 3 6]
```

**含义**：1 首次出现在索引 0，2 首次出现在索引 1，3 首次出现在索引 3，4 首次出现在索引 6。

### 获取计数

```python
unique, counts = np.unique(arr, return_counts=True)
print("唯一值:", unique)
print("出现次数:", counts)
```

**结果**：

```
唯一值: [1 2 3 4]
出现次数: [1 2 3 4]
```

**含义**：1 出现 1 次，2 出现 2 次，3 出现 3 次，4 出现 4 次。

**应用场景**：统计分类数据的分布，如统计不同类别的样本数量。

```python
# 示例：统计用户评分分布
ratings = np.array([5, 4, 5, 3, 4, 5, 5, 4, 3, 2, 5, 4])
values, counts = np.unique(ratings, return_counts=True)
print("评分分布:")
for v, c in zip(values, counts):
    print(f"  {v}星: {c}人 ({c/len(ratings)*100:.1f}%)")
```

**结果**：

```
评分分布:
  2星: 1人 (8.3%)
  3星: 2人 (16.7%)
  4星: 4人 (33.3%)
  5星: 5人 (41.7%)
```

## 集合操作

NumPy 提供了类似数学集合的操作，用于处理两个数组之间的关系。

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
print("数组 a:", a)
print("数组 b:", b)
```

### 交集

```python
# 交集：同时在 a 和 b 中的元素
result = np.intersect1d(a, b)
print("交集:", result)
```

**结果**：`交集: [3 4 5]`

**图示**：

```
a: {1, 2, 3, 4, 5}
b:       {3, 4, 5, 6, 7}
交集:     {3, 4, 5}
```

### 并集

```python
# 并集：a 或 b 中的所有元素（不重复）
result = np.union1d(a, b)
print("并集:", result)
```

**结果**：`并集: [1 2 3 4 5 6 7]`

### 差集

```python
# 差集：在 a 中但不在 b 中的元素
result = np.setdiff1d(a, b)
print("差集 (a - b):", result)
```

**结果**：`差集 (a - b): [1 2]`

```python
# 反向差集
result = np.setdiff1d(b, a)
print("差集 (b - a):", result)
```

**结果**：`差集 (b - a): [6 7]`

### 对称差集

```python
# 对称差集：只在 a 或只在 b 中的元素（不包括交集）
result = np.setxor1d(a, b)
print("对称差集:", result)
```

**结果**：`对称差集: [1 2 6 7]`

**图示**：

```
a: {1, 2, 3, 4, 5}
b:       {3, 4, 5, 6, 7}
对称差集: {1, 2}     +     {6, 7} = {1, 2, 6, 7}
```

### 成员检测

```python
# 检测 a 中的元素是否在给定列表中
result = np.in1d(a, [2, 4])
print("a 中元素是否在 [2, 4] 中:", result)
```

**结果**：`a 中元素是否在 [2, 4] 中: [False  True False  True False]`

**含义**：a 中的 2 和 4 在列表 `[2, 4]` 中，其他元素不在。

**应用场景**：数据过滤、用户分组、A/B 测试中的用户分配等。

## 搜索函数

| 函数                  | 说明               | 返回值   |
| --------------------- | ------------------ | -------- |
| `np.where(condition)` | 返回满足条件的索引 | 索引元组 |
| `np.argmax(arr)`      | 最大值的索引       | 整数     |
| `np.argmin(arr)`      | 最小值的索引       | 整数     |
| `np.nonzero(arr)`     | 非零元素的索引     | 索引元组 |

```python
arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])
print("数组:", arr)
```

**结果**：`数组: [1 5 2 8 3 9 4 7]`

### argmax 和 argmin

```python
# 找最大值的索引
max_idx = np.argmax(arr)
print(f"最大值索引: {max_idx}, 最大值: {arr[max_idx]}")
```

**结果**：`最大值索引: 5, 最大值: 9`

```python
# 找最小值的索引
min_idx = np.argmin(arr)
print(f"最小值索引: {min_idx}, 最小值: {arr[min_idx]}")
```

**结果**：`最小值索引: 0, 最小值: 1`

**二维数组的 argmax**：

```python
arr_2d = np.array([[1, 5, 3],
                   [4, 2, 6]])
print("二维数组:\n", arr_2d)

# 全局最大值索引（展平后的索引）
print("全局最大值索引:", np.argmax(arr_2d))  # 5 (展平后第6个元素)

# 每列最大值索引
print("每列最大值索引:", np.argmax(arr_2d, axis=0))  # [1 0 1]

# 每行最大值索引
print("每行最大值索引:", np.argmax(arr_2d, axis=1))  # [1 2]
```

### where 条件搜索

```python
arr = np.array([1, 5, 2, 8, 3, 9, 4, 7])

# 找出大于 5 的元素索引
indices = np.where(arr > 5)
print("大于 5 的索引:", indices[0])
print("对应的值:", arr[indices])
```

**结果**：

```
大于 5 的索引: [3 5 7]
对应的值: [8 9 7]
```

**说明**：索引 3、5、7 位置的元素（8、9、7）大于 5。

### nonzero 非零元素

```python
arr = np.array([0, 1, 0, 2, 0, 3])
indices = np.nonzero(arr)
print("非零元素索引:", indices[0])
print("非零元素值:", arr[indices])
```

**结果**：

```
非零元素索引: [1 3 5]
非零元素值: [1 2 3]
```

**应用场景**：稀疏矩阵处理、找出有效数据的位置。

## 裁剪和取整

### clip 裁剪

`clip` 将数组中的值限制在指定范围内，超出范围的值会被截断到边界值。

```python
arr = np.array([1, 5, 10, 15, 20])
print("原数组:", arr)

# 裁剪到 [5, 15] 范围
clipped = np.clip(arr, 5, 15)
print("裁剪后:", clipped)
```

**结果**：

```
原数组: [ 1  5 10 15 20]
裁剪后: [ 5  5 10 15 15]
```

**过程**：

- 1 < 5，被裁剪为 5
- 5 在范围内，保持不变
- 10 在范围内，保持不变
- 15 在范围内，保持不变
- 20 > 15，被裁剪为 15

**应用场景**：

- 图像处理：像素值限制在 [0, 255]
- 机器学习：梯度裁剪防止梯度爆炸
- 数据预处理：异常值处理

```python
# 实际应用：像素值裁剪
pixels = np.array([-10, 50, 128, 200, 300])
valid_pixels = np.clip(pixels, 0, 255)
print("有效像素:", valid_pixels)  # [  0  50 128 200 255]
```

### 取整函数

```python
arr = np.array([1.2, 2.5, 3.7, -1.2, -2.5])
print("原数组:", arr)
```

**结果**：`原数组: [ 1.2  2.5  3.7 -1.2 -2.5]`

```python
# floor: 向下取整（向负无穷方向）
print("向下取整:", np.floor(arr))
```

**结果**：`向下取整: [ 1.  2.  3. -2. -3.]`

**理解**：每个数取不大于它的最大整数。-2.5 向下是 -3。

```python
# ceil: 向上取整（向正无穷方向）
print("向上取整:", np.ceil(arr))
```

**结果**：`向上取整: [ 2.  3.  4. -1. -2.]`

**理解**：每个数取不小于它的最小整数。-2.5 向上是 -2。

```python
# round: 四舍五入
print("四舍五入:", np.round(arr))
```

**结果**：`四舍五入: [ 1.  2.  4. -1. -2.]`

**注意**：NumPy 使用"银行家舍入"，2.5 会舍入到 2（最近的偶数）。

```python
# trunc: 截断取整（向零方向）
print("截断取整:", np.trunc(arr))
```

**结果**：`截断取整: [ 1.  2.  3. -1. -2.]`

**理解**：直接去掉小数部分。-2.5 截断是 -2。

**取整函数对比**：

| 函数    | 1.2 | 2.5 | 3.7 | -1.2 | -2.5 | 方向     |
| ------- | --- | --- | --- | ---- | ---- | -------- |
| `floor` | 1   | 2   | 3   | -2   | -3   | 向负无穷 |
| `ceil`  | 2   | 3   | 4   | -1   | -2   | 向正无穷 |
| `round` | 1   | 2   | 4   | -1   | -2   | 最近整数 |
| `trunc` | 1   | 2   | 3   | -1   | -2   | 向零     |

## 复制操作

理解 NumPy 的复制机制对于避免 bug 非常重要。

| 方式            | 类型   | 修改是否影响原数组 | 内存占用 |
| --------------- | ------ | ------------------ | -------- |
| `arr_ref = arr` | 引用   | 是                 | 无新开销 |
| `arr.view()`    | 视图   | 是                 | 无新开销 |
| `arr.copy()`    | 深拷贝 | 否                 | 双倍内存 |

### 引用

```python
arr = np.array([1, 2, 3, 4, 5])
print("原数组:", arr)

# 引用：只是给同一个数组起了别名
arr_ref = arr
arr_ref[0] = 100

print("修改 arr_ref[0] = 100 后:")
print("  arr_ref:", arr_ref)
print("  arr:    ", arr)
```

**结果**：

```
原数组: [1 2 3 4 5]
修改 arr_ref[0] = 100 后:
  arr_ref: [100   2   3   4   5]
  arr:     [100   2   3   4   5]
```

**原理**：`arr_ref` 和 `arr` 指向同一块内存，修改一个会影响另一个。

### 视图

```python
arr = np.array([1, 2, 3, 4, 5])

# 视图：共享数据，但可以有不同的形状
arr_view = arr.view()
arr_view[1] = 200

print("修改 arr_view[1] = 200 后:")
print("  arr_view:", arr_view)
print("  arr:     ", arr)
```

**结果**：

```
修改 arr_view[1] = 200 后:
  arr_view: [  1 200   3   4   5]
  arr:      [  1 200   3   4   5]
```

**说明**：视图和原数组共享同一块数据，切片操作返回的也是视图。

### 副本（深拷贝）

```python
arr = np.array([1, 2, 3, 4, 5])

# 副本：完全独立的新数组
arr_copy = arr.copy()
arr_copy[2] = 300

print("修改 arr_copy[2] = 300 后:")
print("  arr_copy:", arr_copy)
print("  arr:     ", arr)
```

**结果**：

```
修改 arr_copy[2] = 300 后:
  arr_copy: [  1   2 300   4   5]
  arr:      [1 2 3 4 5]
```

**说明**：副本是完全独立的，修改副本不会影响原数组。

**何时使用 copy()？**

- 需要保留原始数据时
- 将切片结果传递给其他函数前
- 不确定后续操作是否会修改数组时

## 其他实用函数

### 数组比较

| 函数                   | 说明                       | 返回值 |
| ---------------------- | -------------------------- | ------ |
| `np.array_equal(a, b)` | 判断两数组是否完全相同     | bool   |
| `np.allclose(a, b)`    | 判断两数组是否接近（浮点） | bool   |

```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.array([1, 2, 4])

print("a == b:", np.array_equal(a, b))  # True
print("a == c:", np.array_equal(a, c))  # False
```

**浮点数比较**：

```python
# 浮点数直接比较可能有问题
a = np.array([0.1 + 0.2])
b = np.array([0.3])

print("直接比较:", np.array_equal(a, b))  # False（浮点精度问题）
print("近似比较:", np.allclose(a, b))     # True
```

**说明**：由于浮点数精度问题，`0.1 + 0.2` 不完全等于 `0.3`，但 `allclose` 允许微小误差。

### 逻辑判断

```python
arr = np.array([True, False, True, False])

# any: 是否有任意 True
print("有任意 True:", np.any(arr))  # True

# all: 是否全为 True
print("是否全 True:", np.all(arr))  # False
```

**常见用法**：

```python
scores = np.array([85, 92, 78, 65, 88])

# 检查是否有不及格的
print("有人不及格:", np.any(scores < 60))  # False

# 检查是否全部及格
print("全部及格:", np.all(scores >= 60))  # True

# 检查是否全部优秀
print("全部优秀:", np.all(scores >= 90))  # False
```

## 小结

### 排序与搜索

1. **np.sort()**：返回排序副本
2. **arr.sort()**：原地排序
3. **np.argsort()**：返回排序索引（非常有用！）
4. **np.argmax()/argmin()**：找极值的索引

### 唯一值与集合

1. **np.unique()**：去重，可返回计数
2. **np.intersect1d()**：交集
3. **np.union1d()**：并集
4. **np.setdiff1d()**：差集

### 数值处理

1. **np.clip()**：值裁剪到指定范围
2. **np.floor/ceil/round/trunc()**：不同的取整方式

### 复制操作

1. **引用和视图**：共享内存，修改会相互影响
2. **copy()**：创建独立副本，安全但占用更多内存

## 练习

运行代码文件查看完整演示：

```bash
python Basic/Numpy/11_utilities.py
```

**练习题**：

1. 给定数组 `[3, 1, 4, 1, 5, 9, 2, 6]`，找出前 3 大的元素及其索引
2. 统计数组 `[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]` 中每个元素的出现次数
3. 给定两个用户 ID 列表，找出两个列表的共同用户和只在第一个列表中的用户
