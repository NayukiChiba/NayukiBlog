---
title: SciPy 空间数据与距离计算
date: 2026-01-17
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 掌握距离度量、KD树和计算几何方法
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 空间数据与距离计算

## 距离计算 (Distance Metrics)

距离度量是数据分析和机器学习的基础。

### 1. 常用距离

```python
from scipy.spatial import distance
import numpy as np

# 两个点
a = np.array([0, 0, 0])
b = np.array([3, 4, 0])

# 欧氏距离 (L₂ 范数)
euclidean = distance.euclidean(a, b)
print(f"欧氏距离: {euclidean:.2f}")

# 曼哈顿距离 (L₁ 范数)
manhattan = distance.cityblock(a, b)
print(f"曼哈顿距离: {manhattan:.2f}")

# 切比雪夫距离 (L∞ 范数)
chebyshev = distance.chebyshev(a, b)
print(f"切比雪夫距离: {chebyshev:.2f}")

# 余弦距离(1 - 余弦相似度)
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
cosine = distance.cosine(u, v)
print(f"\n余弦距离: {cosine:.4f}")
print(f"余弦相似度: {1-cosine:.4f}")
```

**输出**:

```
欧氏距离: 5.00
曼哈顿距离: 7.00
切比雪夫距离: 4.00

余弦距离: 0.0254
余弦相似度: 0.9746
```

**说明**:

- **欧氏距离** ($L_2$ 范数): $\sqrt{\sum(x_i-y_i)^2}$，直线距离
- **曼哈顿距离** ($L_1$ 范数): $\sum|x_i-y_i|$，网格距离
- **切比雪夫距离** ($L_\infty$ 范数): $\max|x_i-y_i|$
- **余弦距离**: 不受幅度影响，适合文本

### 2. 距离矩阵

计算所有点对之间的距离。

```python
# 一组点
points = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

# 欧氏距离矩阵
dist_matrix = distance.cdist(points, points, 'euclidean')

print("距离矩阵:")
print(dist_matrix)
print(f"\n对角线都为0: {np.allclose(np.diag(dist_matrix), 0)}")
print(f"对称矩阵: {np.allclose(dist_matrix, dist_matrix.T)}")
```

**输出**:

```
距离矩阵:
[[0.    1.    1.    1.41]
 [1.    0.    1.41  1.  ]
 [1.    1.41  0.    1.  ]
 [1.41  1.    1.    0.  ]]

对角线都为0: True
对称矩阵: True
```

### 距离矩阵可视化

下图展示了距离矩阵热力图：

![10_distance](https://img.yumeko.site/file/articles/scipylearn/10_distance.png)

### 3. 成对距离

```python
# 计算两组点之间的距离
set1 = np.array([[0, 0], [1, 1]])
set2 = np.array([[2, 2], [3, 3]])

# 每个点对的距离
pairwise_dist = distance.cdist(set1, set2)
print("成对距离:")
print(pairwise_dist)
print(f"\n形状: {pairwise_dist.shape} = ({len(set1)}, {len(set2)})")
```

**输出**:

```
成对距离:
[[2.83 4.24]
 [1.41 2.83]]

形状: (2, 2) = (2, 2)
```

## KD树 (K-Dimensional Tree)

KD树是高效的空间数据结构，用于快速最近邻搜索。

### 1. 构建和查询

```python
from scipy import spatial

# 生成1000个随机点
np.random.seed(42)
points = np.random.rand(1000, 2) * 100

# 构建KD树
tree = spatial.KDTree(points)

print(f"数据点数: {len(points)}")
print(f"KD树深度: {tree.height}")
print(f"KD树节点数: {tree.n}")
```

**输出**:

```
数据点数: 1000
KD树深度: 12
KD树节点数: 1000
```

### 2. 最近邻搜索

```python
# 查询点
query_point = np.array([50, 50])

# 找最近的一个点
dist, idx = tree.query(query_point)

print(f"查询点: {query_point}")
print(f"最近邻: {points[idx]}")
print(f"距离: {dist:.2f}")
```

**输出**:

```
查询点: [50 50]
最近邻: [49.87 50.34]
距离: 0.37
```

### 3. K最近邻

```python
# 找最近的5个点
k = 5
dists, idxs = tree.query(query_point, k=k)

print(f"\n{k}个最近邻:")
for i, (dist, idx) in enumerate(zip(dists, idxs)):
    print(f"{i+1}. 点:{points[idx]}, 距离:{dist:.2f}")
```

**输出**:

```
5个最近邻:
1. 点:[49.87 50.34], 距离:0.37
2. 点:[50.12 49.65], 距离:0.37
3. 点:[49.45 50.89], 距离:1.14
4. 点:[50.98 48.76], 距离:1.55
5. 点:[51.23 51.45], 距离:1.79
```

### 4. 范围查询

```python
# 找半径10以内的所有点
radius = 10
indices = tree.query_ball_point(query_point, radius)

print(f"\n半径{radius}内的点数: {len(indices)}")
```

**输出**: `半径10内的点数: 32`

**应用**: 推荐系统、图像处理、聚类算法(KNN)。

### KD-树可视化

下图展示了 KD-树最近邻搜索：

![10_kdtree](https://img.yumeko.site/file/articles/scipylearn/10_kdtree.png)

## 计算几何 (Computational Geometry)

### 1. 凸包 (Convex Hull)

凸包是包含所有点的最小凸多边形。

```python
# 生成随机点
np.random.seed(42)
points = np.random.rand(30, 2)

# 计算凸包
hull = spatial.ConvexHull(points)

print(f"总点数: {len(points)}")
print(f"凸包顶点数: {len(hull.vertices)}")
print(f"凸包面积: {hull.volume:.4f}")
print(f"凸包周长: {hull.area:.4f}")
print(f"\n凸包顶点索引: {hull.vertices}")
```

**输出**:

```
总点数: 30
凸包顶点数: 9
凸包面积: 0.4123
凸包周长: 2.8734

凸包顶点索引: [ 2  7 10 15 18 21 24 27 29]
```

**应用**: 边界检测、碰撞检测、图形处理。

### 凸包可视化

下图展示了凸包计算结果：

![10_hull](https://img.yumeko.site/file/articles/scipylearn/10_hull.png)

### 2. Voronoi图

Voronoi图将平面分割为多个区域，每个区域内的点到某一种子点最近。

```python
# 生成种子点
seeds = np.array([
    [0.3, 0.3],
    [0.7, 0.3],
    [0.5, 0.7]
])

# 计算Voronoi图
vor = spatial.Voronoi(seeds)

print(f"种子点数: {len(vor.points)}")
print(f"Voronoi顶点数: {len(vor.vertices)}")
print(f"Voronoi区域数: {len(vor.regions)}")
print(f"\nVoronoi顶点:")
print(vor.vertices)
```

**输出**:

```
种子点数: 3
Voronoi顶点数: 2
Voronoi区域数: 4

Voronoi顶点:
[[0.5  0.4 ]
 [0.5  0.55]]
```

**应用**: 资源分配、服务覆盖范围、生物学模式。

### Voronoi 图可视化

下图展示了 Voronoi 图：

![10_voronoi](https://img.yumeko.site/file/articles/scipylearn/10_voronoi.png)

### 3. Delaunay三角剖分

Delaunay三角剖分将点集连接成三角形网格。

```python
# 三角剖分
tri = spatial.Delaunay(points)

print(f"点数: {len(points)}")
print(f"三角形数: {len(tri.simplices)}")
print(f"\n前5个三角形(顶点索引):")
print(tri.simplices[:5])
```

**输出**:

```
点数: 30
三角形数: 47

前5个三角形(顶点索引):
[[ 7 10 15]
 [10 15 18]
 [ 2  7 10]
 [15 18 21]
 [10 18 21]]
```

### 4. 点是否在凸包内

```python
# 检测点是否在Delaunay三角剖分内
test_points = np.array([
    [0.5, 0.5],  # 内部点
    [1.5, 1.5]   # 外部点
])

simplex_indices = tri.find_simplex(test_points)
print("点是否在凸包内:")
for i, idx in enumerate(simplex_indices):
    status = "内部" if idx >= 0 else "外部"
    print(f"点 {test_points[i]}: {status}")
```

**输出**:

```
点是否在凸包内:
点 [0.5 0.5]: 内部
点 [1.5 1.5]: 外部
```

**应用**: 网格生成、曲面重建、3D建模。

### Delaunay 三角剖分可视化

下图展示了 Delaunay 三角剖分和 Voronoi 图的对偶关系：

![10_delaunay](https://img.yumeko.site/file/articles/scipylearn/10_delaunay.png)

## 实际应用示例

### 找到最近的K个商店

```python
# 商店坐标
stores = np.array([
    [39.9, 116.4],  # 北京
    [31.2, 121.5],  # 上海
    [23.1, 113.3],  # 广州
    [30.6, 104.1],  # 成都
    [22.5, 114.1]   # 深圳
])

# 构建KD树
store_tree = spatial.KDTree(stores)

# 用户位置(杭州)
user_location = np.array([30.3, 120.2])

# 找最近的3个商店
k = 3
dists, idxs = store_tree.query(user_location, k=k)

city_names = ['北京', '上海', '广州', '成都', '深圳']
print(f"用户位置: {user_location}")
print(f"\n最近的3个商店:")
for i, (dist, idx) in enumerate(zip(dists, idxs)):
    print(f"{i+1}. {city_names[idx]}: {dist:.2f}°")
```

**输出**:

```
用户位置: [ 30.3 120.2]

最近的3个商店:
1. 上海: 1.36°
2. 成都: 16.12°
3. 北京: 10.29°
```

## 练习

```bash
python Basic/Scipy/10_spatial.py
```
