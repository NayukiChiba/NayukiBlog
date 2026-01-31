---
title: NumPy 综合练习
date: 2026-01-11
category: MachineLearning/Basic/numpy
tags:
  - Python
  - NumPy
description: NumPy 综合练习题，巩固所学知识
image: https://img.yumeko.site/file/blog/NumpyLearning.jpg
status: public
---

# 综合实战

## 学习目标

- 综合运用所学的 NumPy 知识
- 解决实际问题
- 提高编程能力

## 实战项目

| 项目         | 涉及知识点                     | 难度   |
| ------------ | ------------------------------ | ------ |
| 学生成绩分析 | 数组创建、统计运算、排序、索引 | ⭐     |
| 线性回归实现 | 线性代数、矩阵运算、统计分析   | ⭐⭐   |
| 图像操作模拟 | 数组变形、切片、翻转           | ⭐     |
| 统计分析     | 随机数、百分位数、直方图       | ⭐     |
| 移动平均     | 滑动窗口、卷积运算             | ⭐⭐   |
| 矩阵分解     | 线性代数、特征值、SVD          | ⭐⭐⭐ |

---

## 项目 1: 学生成绩分析

### 任务目标

分析 5 名学生 3 门课程的成绩：

- 计算每个学生的总分和平均分
- 计算每门课程的统计信息
- 找出最高分和最低分的学生
- 按总分排名

### 示例代码

```python
np.random.seed(42)

# 创建成绩数据 (5 学生 x 3 课程)
grades = np.random.randint(60, 101, size=(5, 3))
students = ['学生A', '学生B', '学生C', '学生D', '学生E']
courses = ['数学', '英语', '物理']

# 计算每个学生的总分和平均分
total_scores = np.sum(grades, axis=1)
avg_scores = np.mean(grades, axis=1)

# 计算每门课程的统计信息
course_mean = np.mean(grades, axis=0)
course_std = np.std(grades, axis=0)

# 找出总分最高的学生
best_idx = np.argmax(total_scores)
print(f"总分最高: {students[best_idx]}")

# 按总分排名
rank_indices = np.argsort(total_scores)[::-1]
for rank, idx in enumerate(rank_indices, 1):
    print(f"第{rank}名: {students[idx]}")
```

---

## 项目 2: 线性回归实现

### 任务目标

使用正规方程实现最小二乘法线性回归：

- 生成带噪声的线性数据
- 计算回归系数
- 评估模型性能 (R², RMSE)

### 正规方程

$$
\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 示例代码

```python
np.random.seed(42)

# 生成数据: y = 2x + 1 + 噪声
n = 50
x = np.linspace(0, 10, n)
y = 2 * x + 1 + np.random.normal(0, 1, n)

# 构建设计矩阵
X = np.column_stack([np.ones(n), x])

# 正规方程求解
XTX = X.T @ X
XTy = X.T @ y
w = np.linalg.solve(XTX, XTy)

intercept, slope = w[0], w[1]
print(f"估计: y = {slope:.4f}x + {intercept:.4f}")

# 计算 R²
y_pred = slope * x + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R²: {r_squared:.4f}")
```

---

## 项目 3: 图像操作模拟

### 任务目标

模拟基本图像操作：

- 图像翻转（水平、垂直）
- 图像旋转
- 图像裁剪
- 图像归一化

### 示例代码

```python
# 创建模拟图像 (8x8 灰度图)
image = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

# 水平翻转
flipped_h = image[:, ::-1]

# 垂直翻转
flipped_v = image[::-1, :]

# 旋转 90 度
rotated = np.rot90(image)

# 裁剪
cropped = image[2:6, 2:6]

# 归一化到 [0, 1]
normalized = image.astype(np.float64) / 255.0
```

---

## 项目 4: 统计分析

### 任务目标

对正态分布数据进行统计分析：

- 计算基本统计量
- 计算百分位数
- 生成直方图

### 示例代码

```python
np.random.seed(42)

# 生成正态分布数据
data = np.random.normal(loc=100, scale=15, size=1000)

# 基本统计量
print(f"均值: {data.mean():.2f}")
print(f"标准差: {data.std():.2f}")
print(f"最小值: {data.min():.2f}")
print(f"最大值: {data.max():.2f}")

# 百分位数
for p in [25, 50, 75, 90, 95]:
    print(f"第{p}百分位: {np.percentile(data, p):.2f}")

# 直方图
hist, bin_edges = np.histogram(data, bins=10)
```

---

## 项目 5: 移动平均实现

### 任务目标

实现时间序列数据的移动平均：

- 简单移动平均（SMA）
- 加权移动平均（WMA）
- 使用卷积实现

### 示例代码

```python
np.random.seed(42)

# 生成时间序列数据
data = np.cumsum(np.random.randn(100)) + 100

# 简单移动平均 (窗口大小=5)
window = 5
sma = np.convolve(data, np.ones(window)/window, mode='valid')

# 加权移动平均 (近期权重更大)
weights = np.arange(1, window+1)
weights = weights / weights.sum()
wma = np.convolve(data, weights[::-1], mode='valid')

print(f"原始数据前10个: {data[:10]}")
print(f"SMA前5个: {sma[:5]}")
print(f"WMA前5个: {wma[:5]}")
```

---

## 项目 6: 矩阵分解应用

### 任务目标

实现矩阵分解技术：

- 奇异值分解（SVD）
- 主成分分析（PCA）
- 数据降维

### SVD 公式

$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

### 示例代码

```python
np.random.seed(42)

# 创建数据矩阵 (100 样本 x 10 特征)
data = np.random.randn(100, 10)

# 中心化数据
data_centered = data - data.mean(axis=0)

# SVD 分解
U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)

# 计算方差解释比例
variance_explained = (S ** 2) / (S ** 2).sum()
cumsum_variance = np.cumsum(variance_explained)

print("前5个成分的方差解释率:")
for i in range(5):
    print(f"  PC{i+1}: {variance_explained[i]:.4f} (累积: {cumsum_variance[i]:.4f})")

# 降维到2维
n_components = 2
data_reduced = U[:, :n_components] @ np.diag(S[:n_components])
print(f"\n降维后形状: {data_reduced.shape}")
```

---

## 性能优化技巧

### 1. 向量化运算

```python
# ❌ 慢速：使用循环
result = np.zeros(1000000)
for i in range(len(result)):
    result[i] = i ** 2

# ✅ 快速：向量化
arr = np.arange(1000000)
result = arr ** 2
```

### 2. 就地操作

```python
# ❌ 创建新数组
arr = arr + 5

# ✅ 就地修改
arr += 5
```

### 3. 选择合适的数据类型

```python
# ❌ 使用默认 float64
arr = np.random.rand(1000, 1000)

# ✅ 根据需求选择
arr = np.random.rand(1000, 1000).astype(np.float32)
```

### 4. 使用视图而非副本

```python
# ✅ 创建视图（共享内存）
view = arr[::2]

# ❌ 创建副本（新内存）
copy = arr[::2].copy()
```

---

## 常见错误与解决方案

### 错误 1: 形状不匹配

```python
# ❌ 错误
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3], [4]])
result = a + b  # 无法广播

# ✅ 正确
a = np.array([[1, 2, 3]])  # 变为 (1, 3)
b = np.array([[1], [2], [3], [4]])  # (4, 1)
result = a + b  # 广播为 (4, 3)
```

### 错误 2: 整数除法

```python
# ❌ 整数除法返回整数
arr = np.array([1, 2, 3])
result = arr / 2  # [0, 1, 1]

# ✅ 转换为浮点数
result = arr / 2.0  # [0.5, 1.0, 1.5]
```

### 错误 3: 视图 vs 副本

```python
# ❌ 意外修改原数组
arr = np.array([1, 2, 3, 4])
view = arr[:2]
view[0] = 999
print(arr)  # [999, 2, 3, 4]

# ✅ 使用副本
copy = arr[:2].copy()
copy[0] = 999
print(arr)  # [1, 2, 3, 4]
```

### 错误 4: 数组比较

```python
# ❌ 错误的比较方式
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
if arr1 == arr2:  # 报错！
    pass

# ✅ 正确的比较
if np.array_equal(arr1, arr2):
    print("数组相等")

# ✅ 元素级比较
comparison = arr1 == arr2  # 返回布尔数组
if np.all(comparison):
    print("所有元素相等")
```

---

## 实用技巧速查

### 数组创建技巧

```python
# 创建对角矩阵
np.diag([1, 2, 3])

# 创建单位矩阵
np.eye(3)

# 从函数创建
np.fromfunction(lambda i, j: i + j, (3, 3))

# 重复数组
np.repeat([1, 2, 3], 3)  # [1, 1, 1, 2, 2, 2, 3, 3, 3]
np.tile([1, 2, 3], 3)    # [1, 2, 3, 1, 2, 3, 1, 2, 3]
```

### 高级索引技巧

```python
# 多条件筛选
arr[(arr > 5) & (arr < 10)]

# 花式索引
arr[[0, 2, 4]]  # 取第0、2、4个元素

# 网格索引
rows = np.array([0, 1])
cols = np.array([0, 2])
arr[np.ix_(rows, cols)]
```

### 统计函数技巧

```python
# 去除异常值
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
filtered = data[(data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)]

# 标准化
standardized = (data - data.mean()) / data.std()

# 归一化到 [0, 1]
normalized = (data - data.min()) / (data.max() - data.min())
```

### 数组操作技巧

```python
# 交换轴
np.swapaxes(arr, 0, 1)

# 扩展维度
arr[np.newaxis, :]  # 在前面添加维度
arr[:, np.newaxis]  # 在后面添加维度

# 压缩维度
np.squeeze(arr)  # 移除长度为1的维度

# 数组拼接
np.concatenate([arr1, arr2], axis=0)
np.vstack([arr1, arr2])  # 垂直拼接
np.hstack([arr1, arr2])  # 水平拼接
```

---

## 总结：常用技巧速查

### 数组创建与初始化

```python
# 快速创建
np.zeros((3, 4))          # 全零
np.ones((3, 4))           # 全一
np.full((3, 4), 7)        # 填充指定值
np.empty((3, 4))          # 空数组（未初始化）
np.arange(0, 10, 2)       # 等差数列 [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)      # 等分数列 [0, 0.25, 0.5, 0.75, 1]
np.logspace(0, 2, 5)      # 对数空间 [1, 10, 100]

# 特殊矩阵
np.eye(3)                 # 单位矩阵
np.diag([1, 2, 3])        # 对角矩阵
np.tri(3)                 # 下三角矩阵
```

### 形状操作

```python
arr.reshape(3, 4)         # 变形
arr.resize((3, 4))        # 就地变形
arr.flatten()             # 展平（副本）
arr.ravel()               # 展平（尽可能返回视图）
arr.T                     # 转置
arr.transpose(1, 0)       # 指定轴转置
arr.squeeze()             # 移除长度为1的维度
np.expand_dims(arr, 0)    # 增加维度
```

### 统计运算

```python
arr.sum(axis=0)           # 按列求和
arr.mean(axis=1)          # 按行求均值
arr.std(ddof=1)           # 样本标准差
arr.var()                 # 方差
arr.max(), arr.argmax()   # 最大值及索引
arr.min(), arr.argmin()   # 最小值及索引
arr.cumsum()              # 累积和
arr.cumprod()             # 累积积
np.percentile(arr, 50)    # 中位数
np.median(arr)            # 中位数（另一种方法）
np.corrcoef(x, y)         # 相关系数
```

### 条件操作

```python
arr[arr > 5]              # 布尔索引
np.where(arr > 5)         # 返回索引
np.where(arr > 5, 1, 0)   # 条件替换
np.any(arr > 5)           # 任意满足
np.all(arr > 5)           # 全部满足
np.isnan(arr)             # 检查 NaN
np.isinf(arr)             # 检查无穷
np.nonzero(arr)           # 非零元素索引
```

### 数组合并与分割

```python
np.concatenate([a, b])    # 拼接
np.vstack([a, b])         # 垂直拼接
np.hstack([a, b])         # 水平拼接
np.dstack([a, b])         # 深度拼接
np.split(arr, 3)          # 等分
np.array_split(arr, 3)    # 不等分（允许）
np.hsplit(arr, 3)         # 水平分割
np.vsplit(arr, 3)         # 垂直分割
```

### 线性代数

```python
np.dot(a, b)              # 矩阵乘法
a @ b                     # 矩阵乘法（Python 3.5+）
np.linalg.inv(a)          # 矩阵求逆
np.linalg.det(a)          # 行列式
np.linalg.eig(a)          # 特征值和特征向量
np.linalg.svd(a)          # 奇异值分解
np.linalg.solve(A, b)     # 求解线性方程组
np.linalg.norm(a)         # 范数
np.trace(a)               # 迹
```

---

## 调试技巧

### 查看数组信息

```python
arr.shape                 # 形状
arr.dtype                 # 数据类型
arr.ndim                  # 维度数
arr.size                  # 元素总数
arr.itemsize              # 每个元素字节数
arr.nbytes                # 总字节数
arr.flags                 # 数组标志
```

### 设置打印选项

```python
# 控制打印格式
np.set_printoptions(
    precision=4,          # 小数精度
    suppress=True,        # 不使用科学计数法
    threshold=10,         # 超过10个元素时省略
    edgeitems=2,          # 省略时显示边缘元素数
    linewidth=100         # 每行字符数
)

# 查看完整数组
with np.printoptions(threshold=np.inf):
    print(arr)
```

### 内存管理

```python
# 检查是否共享内存
arr2.base is arr          # 如果是视图，返回 True

# 强制创建副本
arr2 = arr.copy()

# 查看内存地址
arr.__array_interface__['data'][0]
```

---

## 进阶学习建议

### 1. 深入学习资源

- **官方文档**：
  - NumPy 用户指南：https://numpy.org/doc/stable/user/
  - NumPy API 参考：https://numpy.org/doc/stable/reference/
- **进阶主题**：
  - 结构化数组（Structured Arrays）：处理表格数据
  - 记录数组（Record Arrays）：命名字段访问
  - 内存布局：C-order vs Fortran-order 的性能影响
  - NumPy C API：与 C/C++ 代码交互
  - 掩码数组（Masked Arrays）：处理缺失值

- **书籍推荐**：
  - 《Python for Data Analysis》by Wes McKinney
  - 《NumPy Essentials》by Leo (Liang-Huan) Chin

### 2. 相关库学习路径

```
NumPy (数值计算基础)
    ↓
├─→ Pandas (数据分析)
│       ↓
│   ├─→ Scikit-learn (机器学习)
│   └─→ Statsmodels (统计建模)
│
├─→ Matplotlib/Seaborn (数据可视化)
│
├─→ SciPy (科学计算)
│       ↓
│   └─→ SymPy (符号计算)
│
└─→ TensorFlow/PyTorch (深度学习)
```

**推荐学习顺序**：

1. **Pandas**：处理表格数据、时间序列
2. **Matplotlib**：数据可视化
3. **Scikit-learn**：机器学习算法
4. **SciPy**：高级科学计算

### 3. 实践项目建议

#### 入门级项目

- 图像处理：图像滤波、边缘检测
- 数据清洗：缺失值处理、异常值检测
- 简单统计分析：描述性统计、假设检验

#### 中级项目

- 实现机器学习算法：KNN、决策树
- 时间序列分析：ARIMA 模型
- 信号处理：傅里叶变换、滤波器设计

#### 高级项目

- 神经网络：从零实现反向传播
- 计算机视觉：目标检测算法
- 数值优化：梯度下降变体实现

### 4. 性能优化进阶

```python
# 使用 numexpr 加速表达式计算
import numexpr as ne
result = ne.evaluate('a + b * c')

# 使用 numba 进行 JIT 编译
from numba import jit

@jit(nopython=True)
def fast_function(arr):
    total = 0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# 使用 Cython 优化关键代码
# 查看 Cython 文档了解更多
```

### 5. 常见面试题

1. **NumPy 数组与 Python 列表的区别**
   - 类型一致性、内存连续、向量化运算

2. **什么是广播机制**
   - 不同形状数组的运算规则

3. **视图 vs 副本的区别**
   - 内存共享、修改影响

4. **如何优化 NumPy 代码性能**
   - 向量化、数据类型、避免循环

5. **NumPy 中的轴（axis）如何理解**
   - axis=0 沿着行、axis=1 沿着列

---

## 学习检查清单

完成以下检查项，确保掌握 NumPy 核心知识：

- [ ] 能熟练创建各种类型的数组
- [ ] 理解数组的形状、维度、数据类型
- [ ] 掌握切片、索引、花式索引
- [ ] 理解广播机制并能应用
- [ ] 会使用向量化运算替代循环
- [ ] 能进行数组变形和转置
- [ ] 掌握常用统计函数
- [ ] 理解视图和副本的区别
- [ ] 会使用布尔索引进行条件筛选
- [ ] 能进行基本的线性代数运算
- [ ] 了解随机数生成和统计分布
- [ ] 能读写 NumPy 数组文件
- [ ] 理解内存布局和性能优化

---

## 快速参考卡片

### 最常用的 20 个函数

| 函数               | 用途       | 示例                        |
| ------------------ | ---------- | --------------------------- |
| `np.array()`       | 创建数组   | `np.array([1, 2, 3])`       |
| `np.zeros()`       | 全零数组   | `np.zeros((3, 4))`          |
| `np.ones()`        | 全一数组   | `np.ones((2, 3))`           |
| `np.arange()`      | 等差序列   | `np.arange(0, 10, 2)`       |
| `np.linspace()`    | 等分序列   | `np.linspace(0, 1, 5)`      |
| `np.random.rand()` | 随机数组   | `np.random.rand(3, 4)`      |
| `arr.reshape()`    | 变形       | `arr.reshape(2, -1)`        |
| `arr.T`            | 转置       | `arr.T`                     |
| `np.concatenate()` | 拼接       | `np.concatenate([a, b])`    |
| `arr.sum()`        | 求和       | `arr.sum(axis=0)`           |
| `arr.mean()`       | 均值       | `arr.mean()`                |
| `arr.max()`        | 最大值     | `arr.max(axis=1)`           |
| `arr.argmax()`     | 最大值索引 | `arr.argmax()`              |
| `np.where()`       | 条件选择   | `np.where(arr > 0, 1, -1)`  |
| `np.dot()`         | 矩阵乘法   | `np.dot(a, b)`              |
| `np.linalg.inv()`  | 矩阵求逆   | `np.linalg.inv(A)`          |
| `np.save()`        | 保存数组   | `np.save('data.npy', arr)`  |
| `np.load()`        | 加载数组   | `arr = np.load('data.npy')` |
| `arr[arr > 0]`     | 布尔索引   | `arr[arr > 0]`              |
| `arr[[0, 2]]`      | 花式索引   | `arr[[0, 2, 4]]`            |

---

🎉 **恭喜你完成了 NumPy 学习教程！**

### 下一步行动

1. **立即实践**：运行完整演示代码

   ```bash
   python Basic/Numpy/12_practice.py
   ```

2. **巩固练习**：尝试以下挑战
   - 实现 K-means 聚类算法
   - 编写图像卷积函数
   - 实现梯度下降优化器

3. **深入学习**：
   - 阅读 NumPy 源码中感兴趣的部分
   - 研究 NumPy 底层 C API
   - 学习 BLAS/LAPACK 线性代数库

4. **分享交流**：
   - 在 GitHub 上分享你的实践项目
   - 参与 Stack Overflow 问答
   - 为开源项目贡献代码

### 学习资源汇总

- **官方资源**：
  - NumPy 官网：https://numpy.org/
  - 用户指南：https://numpy.org/doc/stable/user/
  - API 参考：https://numpy.org/doc/stable/reference/

- **教程与课程**：
  - NumPy 快速入门：https://numpy.org/doc/stable/user/quickstart.html
  - NumPy 101：https://numpy.org/numpy-tutorials/

- **社区资源**：
  - Stack Overflow：标签 [numpy]
  - Reddit：r/Python, r/learnpython
  - GitHub：https://github.com/numpy/numpy

### 反馈与改进

如果你发现文档中的错误或有改进建议，欢迎：

- 提交 Issue
- 发起 Pull Request
- 联系文档维护者

---

**持续学习，不断进步！** 🚀
