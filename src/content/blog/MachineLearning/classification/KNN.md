---
title: KNN K 近邻分类
date: 2026-04-11
category: 机器学习/分类
tags:
  - Scikit-learn
  - 基础
description: KNN分类的数学原理、距离度量、标准化必要性及完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581780169.webp
status: published
---


# 数学原理

## 本章目标

1. 理解 KNN 为什么不通过显式参数优化边界，而是通过邻域关系完成分类。
2. 理解闵可夫斯基距离、多数投票、加权投票和 $k$ 值在当前实现中的数学角色。
3. 理解为什么标准化会直接影响 KNN 的预测结果——距离型模型对特征尺度敏感。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 闵可夫斯基距离 $d_p(\mathbf{x}, \mathbf{y})$ | 距离度量 | 当前实现 `metric='minkowski'` 的底层框架，$p=2$ 时退化为欧几里得距离 |
| 多数投票 | 决策规则 | 当前默认 `weights='uniform'` 对应的预测方式，$k$ 个邻居等权投票 |
| 加权投票 | 决策规则 | `weights='distance'` 对应的加权方式，邻居越近权重越大 |
| $k$ 值 | 超参数 | 决定投票邻域范围，直接控制偏差-方差权衡 |
| 标准化 $x_i' = (x_i - \mu_i)/\sigma_i$ | 预处理 | KNN 的距离计算依赖特征尺度，不标准化会导致量纲大的特征主导近邻判断 |
| KD-Tree | 加速结构 | 通过空间划分在低维场景中加速近邻查询，$O(\log n)$ 平均复杂度 |

## 1. KNN 的核心思想

KNN（K-Nearest Neighbors）是一种基于实例的懒惰学习算法。它不通过最小化损失函数来学习一组显式参数，而是在预测时直接在训练集中寻找距离最近的 $k$ 个样本，由它们的类别分布决定输出。

### 理解重点

- KNN 的核心不是先学一条全局边界，而是"看待预测点周围有哪些样本"。
- 这使 KNN 对局部结构非常敏感，天然能适应非线性边界。
- 同时，这也意味着它对距离定义和数据尺度特别敏感——距离变了，近邻关系就变了。

## 2. 闵可夫斯基距离：定义"近"的数学框架

闵可夫斯基距离是当前源码 `metric='minkowski'` 对应的底层框架，通过参数 $p$ 控制距离类型。

### 参数速览

适用参数：`metric`、`p`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `metric` | `str` | 距离度量方式。默认为 `"minkowski"`，是一条通用距离框架：$d_p(\mathbf{x}, \mathbf{y}) = (\sum_{i=1}^{d} \vert x_i - y_i\vert^p)^{1/p}$。$p$ 通过独立参数 `p` 控制 | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| `p` | `int` | 闵可夫斯基距离的幂参数。$p=1$ 为曼哈顿距离，$p=2$ 为欧几里得距离。默认为 `2` | `1`、`2` |

闵可夫斯基距离的一般形式：

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{d} |x_i - y_i|^p \right)^{1/p}, \quad p \geq 1
$$

三种常见特例：

| $p$ 值 | 名称 | 公式 | 几何直觉 |
|---|---|---|---|
| $p=1$ | 曼哈顿距离 | $d_1 = \sum_{i=1}^{d} \vert x_i - y_i \vert$ | 只能沿坐标轴移动 |
| $p=2$ | 欧几里得距离 | $d_2 = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$ | 直线距离（默认） |
| $p \to \infty$ | 切比雪夫距离 | $d_\infty = \max_i \vert x_i - y_i \vert$ | 只考虑最大分量差 |

### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier

# 默认使用闵可夫斯基距离，p=2（等同欧几里得）
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# 显式使用曼哈顿距离
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)

# 等价写法
model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
```

### 理解重点

- 当前源码没有显式设置 `p`，因此使用默认值 $p=2$（欧几里得距离）。
- 一旦距离定义改变，近邻集合和最终分类结果也会随之变化。
- `metric='minkowski'` 是 sklearn 的默认值，它不独立指定距离类型，而是和 `p` 参数配合使用。

## 3. 为什么必须标准化

当不同特征的量纲差异悬殊时，大值特征会主导距离计算（在闵可夫斯基公式中，量纲大的分量对和的贡献更大），因此标准化对 KNN 是必需的：

$$
x_i' = \frac{x_i - \mu_i}{\sigma_i}
$$

其中 $\mu_i$ 和 $\sigma_i$ 是特征 $i$ 在训练集上的均值和标准差。标准化后每个特征均值为 0、标准差为 1，所有特征在距离计算中得到平等对待。

### 参数速览

适用类：`sklearn.preprocessing.StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `with_mean` | `bool` | 是否中心化（减去均值）。默认为 `True` | `True` |
| `with_std` | `bool` | 是否缩放（除以标准差）。默认为 `True` | `True` |
| `copy` | `bool` | 是否复制输入数据。默认为 `True` | `True` |

### 示例代码

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # 训练集上拟合统计量并变换
X_test_s = scaler.transform(X_test)        # 测试集只变换，用训练集的统计量
```

### 理解重点

- 对 KNN 来说，标准化不是锦上添花，而是距离型模型几乎必备的预处理。
- 如果不标准化，量纲大的特征会主导 $\vert x_i - y_i \vert^p$ 的计算，使得远近关系完全失真。
- 这也是 KNN 与决策树在当前仓库中最关键的工程差异之一——决策树基于阈值切分，不依赖距离尺度。

## 4. 分类决策规则

### 多数投票法（当前默认）

对待预测点 $\mathbf{x}$，定义其 $k$ 个最近邻集合为 $\mathcal{N}_k(\mathbf{x})$，则预测类别为：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

$k$ 个邻居每人一票，得票最多的类别胜出。平票时按 sklearn 内部规则处理（默认选择类别标签最小的那个）。

### 加权投票法

考虑距离越近权重越大的方案（`weights='distance'`）：

$$
\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \frac{\mathbb{1}(y_i = c)}{d(\mathbf{x}, \mathbf{x}_i)}
$$

权重与距离成反比——邻居越近，投票权重越大。

### 参数速览

适用参数：`weights`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `weights` | `str` | 投票权重方式。`"uniform"` 为等权投票 $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i=c)$；`"distance"` 为距离倒数加权 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$。默认为 `"uniform"` | `"uniform"`、`"distance"` |

### 示例代码

```python
# 多数投票（当前默认）
model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 距离加权投票
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

### 理解重点

- 当前源码默认 `weights='uniform'`，对应多数投票直觉。
- 如果改成 `'distance'`，邻居越近投票影响越大，边界通常更精细，但对噪声也更敏感。
- 这是 KNN 分册里应该重点解释的两个投票策略之一。

## 5. $k$ 值的偏差-方差权衡

$k$ 是 KNN 最核心的超参数，它直接控制决策的局部性程度。

| $k$ 值 | 偏差 | 方差 | 决策行为 |
|---|---|---|---|
| 小 $k$（如 $k=1$） | 低偏差 | 高方差 | 边界紧密贴合训练样本，对噪声异常敏感，容易过拟合 |
| 大 $k$（如 $k=50$） | 高偏差 | 低方差 | 边界过度平滑，丢失局部结构信息，容易欠拟合 |

### 参数速览

适用参数：`n_neighbors`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻数量 $k$。$k$ 越小 -> 偏差低、方差高、边界精细、易过拟合；$k$ 越大 -> 偏差高、方差低、边界平滑、易欠拟合。默认 `5` | `1`、`5`、`15`、`50` |

### 示例代码

```python
model = KNeighborsClassifier(n_neighbors=5)
```

### 理解重点

- $k=1$ 时每个训练样本自身就是一个 Voronoi 区域中心，训练误差为 0 但泛化差。
- $k=5$ 是 sklearn 默认值，也是教学上最常见的起点，兼顾了局部性和稳定性。
- $k$ 通常设为奇数以避免二分类平票，但多分类场景中平票仍可能发生。

## 6. 概率估计

KNN 的概率输出基于邻域内各类别占比：

$$
P(\hat{y} = c \mid \mathbf{x}) = \frac{1}{k} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}(y_i = c)
$$

对于 `weights='distance'` 的情况，概率为加权占比：

$$
P(\hat{y} = c \mid \mathbf{x}) = \frac{\sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{1}(y_i = c)}{\sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}
$$

### 理解重点

- KNN 的概率输出本质上是邻域内的类别频率，这不同于逻辑回归通过 sigmoid 映射得分到概率。
- 由于 $k$ 较小，概率值只取离散值（如 $k=5$ 时概率只能是 $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$），看起来不如其他模型的概率"平滑"。
- 这些概率是 ROC 曲线的直接输入——需要连续变化的阈值才能画出 TPR/FPR 轨迹。

## 7. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 闵可夫斯基距离 | $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i \vert^p)^{1/p}$ | `metric='minkowski'`，$p=2$（默认） |
| 欧几里得距离 | $d_2 = \sqrt{\sum (x_i - y_i)^2}$ | `p=2`（默认，未显式写出） |
| 多数投票 | $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i = c)$ | `weights='uniform'`（默认） |
| 加权投票 | $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$ | `weights='distance'` |
| 邻域大小 | $k$ | `n_neighbors=5` |
| 概率估计 | $P(c\vert\mathbf{x}) = \frac{1}{k} \sum \mathbb{1}(y_i=c)$ | `model.predict_proba(X)` |
| 标准化 | $x_i' = (x_i - \mu_i) / \sigma_i$ | `StandardScaler().fit_transform(X_train)` |
| KD-Tree 查询 | — | `algorithm='auto'`（默认，自动选择） |

## 常见坑

1. 把 KNN 当成"会自动学出参数边界"的模型——它是懒惰学习，`fit()` 只存储数据，不做优化。
2. 忽略标准化，让距离关系完全失真——量纲大的特征会主导 $\vert x_i - y_i \vert^p$。
3. 只会机械调 `k`，却不理解它对应偏差-方差权衡——小 $k$ 低偏差高方差，大 $k$ 高偏差低方差。
4. 把加权投票写成当前默认行为——源码默认 `weights='uniform'`，加权投票需要显式设置。
5. 混淆概率估计的来源——KNN 概率来自邻域频率，不是连续函数映射，取值是离散的（分母为 $k$）。

## 小结

- KNN 的核心数学：用闵可夫斯基距离 $d_p(\mathbf{x}, \mathbf{y})$ 定义近邻，再用投票规则 $\arg\max_c \sum \mathbb{1}(y_i=c)$ 完成分类。
- $k$（`n_neighbors`）、`weights`、`metric`/`p` 和标准化共同决定模型行为——哪一个变了，近邻关系和分类结果都会变。
- KNN 不通过最小化损失函数学习参数，`fit()` 只是存储训练数据，所有计算发生在 `predict()` 阶段。

# 数据构成

## 本章目标

1. 明确本仓库 KNN 数据来自 `ClassificationData.knn()` 的双月牙生成逻辑。
2. 明确 `make_moons` 各参数的数据含义与当前取值。
3. 明确训练集/测试集切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClassificationData.knn()` | 方法 | 生成 KNN 使用的非线性二分类双月牙数据 |
| `make_moons(...)` | 函数 | scikit-learn 提供的双月牙数据生成器，两个半月形各为一个类别 |
| `knn_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前流水线中的监督分类标签，取值 $\{0, 1\}$ |
| `StandardScaler` | 类 | 对特征做标准化，保证距离度量中的各维度贡献均衡 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `knn_data`
- 生成来源：`data_generation/classification.py` 中的 `ClassificationData.knn()`
- 流水线使用：`pipelines/classification/knn.py` 中的 `data = knn_data.copy()`

### 理解重点

- `knn_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的是避免后续处理意外修改原始数据对象。
- 当前数据是为 KNN 教学场景专门构造的双月牙二分类数据，与局部邻域分类思路高度匹配。

## 2. 数据生成函数 `ClassificationData.knn()`

底层调用 `sklearn.datasets.make_moons`，生成两个交错半月形的二分类数据。

### 参数速览

适用函数：`sklearn.datasets.make_moons`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` 或 `tuple[int, int]` | 总样本数 $N$。当前取 `400`，每个类别约 200 个样本（默认各半）。若传入元组 `(n0, n1)` 可分别指定两个类别的样本数。默认为 `100` | `100`、`400`、`(150, 250)` |
| `shuffle` | `bool` | 是否打乱样本顺序。默认为 `True` | `True` |
| `noise` | `float` | 添加到数据中的高斯噪声标准差 $\sigma_{\text{noise}}$。$\sigma_{\text{noise}} = 0$ 为完美半月形，值越大两类边界越模糊。当前取 `0.1` | `0.0`、`0.1`、`0.3` |
| `random_state` | `int` | 随机种子，保证每次生成相同数据。当前取 `42` | `42` |

### 示例代码

```python
from sklearn.datasets import make_moons
from pandas import DataFrame

X, y = make_moons(n_samples=400, noise=0.1, random_state=42)
# X.shape = (400, 2), y 取值 {0, 1}
columns = [f"x{i + 1}" for i in range(2)]
data = DataFrame(X, columns=columns)
data["label"] = y
```

### 理解重点

- `make_moons` 生成的两个半月形天然带有非线性边界——全局一条直线无法干净分开两个类别。
- 这种数据很适合展示 KNN 的局部感知能力，因为"周围邻居是谁"比"全局边界怎么切"更重要。
- `noise=0.1` 在半月形边界上添加少量高斯噪声，使数据更接近真实场景，同时不会完全破坏半月形结构。
- 这也是当前分册和逻辑回归分册数据选择明显不同的原因——逻辑回归用 blob 数据（近线性可分），KNN 用双月牙数据（非线性）。

## 3. 特征列与标签列

当前数据表结构：

- 特征列：`x1`、`x2`（二维实数特征，来自半月形的 $x$、$y$ 坐标）
- 标签列：`label`（二分类标签，取值为 $0$ 或 $1$）

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `label` 是监督训练标签，会真实参与 `model.fit(X_train, y_train)`。
- 当前任务为二分类，因此标签只有 0 和 1 两个取值。
- 与无监督聚类分册不同，这里的标签不是只用于对照，而是训练过程的一部分。

## 4. 切分与标准化的顺序

KNN 流水线中的标准化必须在切分之后执行，否则会造成数据泄露——测试集的信息会通过标准化统计量泄露到训练过程中。

### 参数速览

适用函数：`train_test_split`、`StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比。当前取 `0.2`，即 80 个测试样本（总样本 400 x 20%） | `0.2`、`0.3` |
| `random_state` | `int` | 随机种子，保证每次切分结果一致。当前取 `42` | `42` |
| `stratify` | `array_like` | 按 `y` 的类别比例分层抽样。数学上保证 $\frac{n_{0,\text{train}}}{n_{0,\text{test}}} \approx \frac{N_{\text{train}}}{N_{\text{test}}}$，避免某一类别在测试集中意外过多或过少 | `y`、`None` |
| `scaler.fit_transform(X_train)` | 方法 | 在训练集上计算 $\mu_i, \sigma_i$ 并变换：$x_i' = (x_i - \mu_i) / \sigma_i$ | — |
| `scaler.transform(X_test)` | 方法 | 使用训练集的 $\mu_i, \sigma_i$ 变换测试集，不重新计算统计量 | — |

### 示例代码

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 先切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 再标准化（仅在训练集上 fit）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 对 KNN 来说，标准化尤其关键——距离关系直接决定邻居集合和最终投票结果。
- `fit_transform` 在训练集上同时完成统计量学习（$\mu_i, \sigma_i$）和数据变换。
- `transform` 在测试集上只用训练集学到的统计量，模拟真实部署场景（新数据来了只能用训练时的标准化参数）。

## 数据可视化

![类别分布](https://img.yumeko.site/file/blog/articles/1780737778485.png)

![相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

![散点图矩阵](https://img.yumeko.site/file/blog/articles/1780736139527.png)

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来。
2. 在切分之前就对全量数据做标准化——造成数据泄露，验证结果不可信。
3. 忽略 `stratify=y`，导致训练集和测试集类别比例不稳定——尤其在小样本或类别不均衡时影响更明显。
4. 只看到 KNN 是"简单模型"，却忽略双月牙数据正好需要局部非线性判别能力——如果用逻辑回归的 blob 数据来评估 KNN，就错过了它的核心优势。

## 小结

- 当前 KNN 数据来自 `ClassificationData.knn()`，底层使用 `make_moons(n_samples=400, noise=0.1)`。
- 数据表结构清晰：`x1`、`x2` 是二维特征，`label` 是二分类监督标签。
- KNN 完全依赖距离关系，因此标准化是必需的预处理步骤——且必须严格在切分后、训练前执行。
- 读懂数据来源、切分方式和标准化顺序，是理解后续训练与评估章节的前提。

# 思路与直觉

## 本章目标

1. 用直观方式理解 KNN 到底在做什么——不看数学公式，先看"邻居"。
2. 理解为什么它在当前双月牙数据上更能体现局部分类优势。
3. 理解它与逻辑回归、决策树在思路上的关键差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 邻域投票 | 核心直觉 | 用最近样本的类别分布决定预测结果 |
| 局部结构 | 决策依据 | 当前点周围的几何关系比全局边界更重要——KNN 只关心"附近" |
| 非线性边界 | 决策形状 | 天然适应双月牙等弯曲边界，无需显式学习曲线参数 |
| $k$ 值 | 超参数 | 控制"看多大范围的邻居"——$k$ 小看近处，$k$ 大看远方 |
| 懒惰学习 | 算法范式 | 训练时只存数据，不做任何计算，预测时才"开工" |
| 逻辑回归 | 对比算法 | 学习全局线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$ |
| 决策树 | 对比算法 | 递归轴对齐切分，形成分段常数区域 |

## 1. 为什么需要 KNN

对于一些分类问题，我们不一定非要先学一条全局边界。有时更自然的问题是：

一个新样本周围都是什么类别的点？

KNN 给出的思路是：

- 不急着先学参数——甚至不需要"训练"这个阶段。
- 先找到离当前样本最近的 $k$ 个训练样本。
- 再根据这些邻居的类别来投票决定结果。

### 理解重点

- KNN 是一种非常典型的局部方法——它的预测不依赖一个全局固定公式，而依赖当前点周围的数据分布。
- 这意味着同一个模型在不同区域可以表现出完全不同的决策行为，天然适应非线性边界。
- 这也是为什么它常被称为"懒惰学习"——把所有工作推迟到预测阶段才做。

## 2. 为什么当前仓库示例里它表现合理

当前 KNN 数据来自 `make_moons(n_samples=400, noise=0.1)`，特点是：

- 两个半月形交错排列
- 类别边界呈明显弧线弯曲
- 不存在一条直线能干净分开两类

### 理解重点

- 对这类数据，"局部邻域关系"比"全局一条直线怎么切"更重要。
- 如果附近大多数点都属于同一类，那么当前点也很可能属于这一类——KNN 直接利用了这个朴素直觉。
- 这也是为什么 KNN 在此类数据上通常优于逻辑回归。

## 3. 用"看周围邻居是谁"理解算法

可以把 KNN 理解成一个三步走的过程：

1. 找到待预测点附近最近的 $k$ 个训练样本
2. 统计这 $k$ 个样本分别属于哪一类
3. 投票：得票最多的类别就是预测结果

### 理解重点

- 这也是为什么 `n_neighbors` 在当前分册里最值得专门观察——$k$ 直接决定了"周围"的范围。
- 如果 $k=1$，只看最近的一个邻居，边界会非常曲折——每个训练点周围都形成一个独立区域。
- 如果 $k$ 很大，相当于看很远，边界会变得更平滑，但可能忽略重要的局部结构。
- 如果把 KNN 仅理解成"一个会分类的黑盒"，就会错过它最有辨识度的局部投票思想。

## 4. 为什么标准化会显著影响结果

KNN 通过距离来定义"谁是谁的邻居"。如果两个特征的量纲差很多——比如 `x1` 的范围是 $[0, 1000]$，`x2` 的范围是 $[0, 1]$——那么：

- $x_1$ 对距离 $d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$ 的贡献是 $x_2$ 的 $10^6$ 倍
- 近邻关系几乎完全由 $x_1$ 决定，$x_2$ 形同虚设

### 理解重点

- 标准化 $x_i' = (x_i - \mu_i) / \sigma_i$ 让所有特征处于同一尺度，距离计算才公平。
- 这也是 KNN 比决策树更需要标准化的原因——决策树基于阈值切分（$x_1 \leq 3.2$），只关心相对顺序，不关心绝对量纲。

## 5. 与逻辑回归、决策树的直觉差异

三者的核心差异可以这样理解：

| 算法 | 决策边界 | 核心依据 | 是否需要标准化 | 适用场景 |
|---|---|---|---|---|
| LogisticRegression | 全局线性超平面 $\mathbf{w}^T\mathbf{x} + b = 0$ | 线性打分 + Sigmoid 概率映射 | 是（梯度优化敏感） | 近线性可分数据 |
| DecisionTreeClassifier | 局部轴对齐分段边界 | 不纯度下降 + 递归划分 | 否（阈值切分不依赖距离） | 区域化分布数据 |
| KNN | 局部非参数复杂边界 | 最近邻投票 | 是（距离度量必须） | 非线性局部结构数据 |

### 理解重点

- 逻辑回归在问：整张图上最好的一条全局边界是什么。
- 决策树在问：先切哪一刀最能让类别变纯。
- KNN 在问：当前点周围最近的 $k$ 个邻居大多是什么类别。
- 当前双月牙数据最适合把 KNN 作为局部分类基线来讲解——它的边界自然贴合数据的弧形结构。

## 可视化

![决策边界](https://img.yumeko.site/file/blog/articles/1780736145017.png)

## 常见坑

1. 只知道 KNN 很简单，却说不出它为什么适合当前双月牙数据——关键在"局部 > 全局"。
2. 把所有样本都当成同等重要，而忽略 KNN 只关心局部邻域——远离待预测点的样本对结果没有直接影响。
3. 只会机械调 $k$，却不理解它对应的是局部范围大小——$k$ 小看近处，$k$ 大看远方。
4. 忽略标准化，让近邻关系本身失真——量纲差异导致距离度量失去意义。

## 小结

- KNN 的直觉核心：局部邻域投票——先看附近是谁，再决定自己是谁。
- 当前仓库使用双月牙数据，正好体现了它在非线性局部结构上的优势——边界不需要是直线。
- 与逻辑回归（全局线性）和决策树（轴对齐分段）不同，KNN 的边界形状完全由数据局部密度决定，不需要显式学习任何参数。
- 标准化不是可选项，而是距离型模型的前提——不标准化就意味着"邻居"这个概念被量纲绑架。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `KNeighborsClassifier`。
2. 理解每个构造器参数的数学含义与调参方向。
3. 理解 KNN 的 `fit()` 与其他参数化模型的本质区别。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练 `KNeighborsClassifier`，返回已训练模型 |
| `KNeighborsClassifier(...)` | 构造器 | 创建 K 近邻分类器，通过超参数控制邻域定义与投票规则 |
| `model.fit(X_train, y_train)` | 方法 | 保存训练样本并建立近邻查询结构（KD-Tree 或 Ball-Tree） |
| `n_neighbors` | 超参数 | 控制近邻数量 $k$ |
| `weights` | 超参数 | 控制投票权重策略 |
| `metric` | 超参数 | 控制距离度量方式 |

## 1. `train_model(...)` 的函数签名

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 `(n_samples, n_features)`。传入 `model.fit()`。每行为一个样本，每列为一个特征 | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 `(n_samples,)`。二分类标签取值为 $\{0, 1\}$ | `y_train` |
| `n_neighbors` | `int` | 近邻数量 $k$。$k$ 越小偏差越低方差越高；$k$ 越大偏差越高方差越低。当前默认 `5` | `3`、`5`、`15` |
| `weights` | `str` | 投票权重策略。`"uniform"` 等权投票 $\hat{y} = \arg\max_c \sum \mathbb{1}(y_i=c)$；`"distance"` 距离倒数加权 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$。当前默认 `"uniform"` | `"uniform"`、`"distance"` |
| `metric` | `str` | 距离度量方式。`"minkowski"` 对应 $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i\vert^p)^{1/p}$，配合 `p` 参数使用。当前默认 `"minkowski"` | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| 返回值 | `KNeighborsClassifier` | 已完成 `fit()` 的模型对象，含 `_fit_X`、`_fit_y` 等内部属性，可立即调用 `predict()` 和 `predict_proba()` | — |

### 示例代码

```python
from model_training.classification.knn import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `KNeighborsClassifier` 模型。
- 和部分实验型代码不同，这里没有参数搜索逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `KNeighborsClassifier(...)` 的完整参数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_neighbors` | `int` | 近邻数量 $k$。决定了投票邻域的大小，是 KNN 最核心的超参数。默认为 `5` | `1`、`5`、`15`、`50` |
| `weights` | `str` 或 `callable` | 投票权重函数。`"uniform"` 等权，$w_i = 1$；`"distance"` 按距离倒数加权，$w_i = 1/d(\mathbf{x}, \mathbf{x}_i)$。默认为 `"uniform"` | `"uniform"`、`"distance"` |
| `algorithm` | `str` | 近邻搜索算法。`"auto"` 自动选择（当前默认）；`"ball_tree"` 球树；`"kd_tree"` KD 树；`"brute"` 暴力搜索。低维数据 KD-Tree 通常最快。默认为 `"auto"` | `"auto"`、`"kd_tree"`、`"ball_tree"`、`"brute"` |
| `leaf_size` | `int` | Ball-Tree 或 KD-Tree 的叶节点大小。影响构建速度和查询速度，小值 -> 构建慢查询快，大值 -> 构建快查询慢。仅当 `algorithm` 为 `"ball_tree"` 或 `"kd_tree"` 时生效。默认为 `30` | `20`、`30`、`50` |
| `p` | `int` | 闵可夫斯基距离的幂参数。$p=1$ 曼哈顿距离 $d_1 = \sum \vert x_i - y_i \vert$；$p=2$ 欧几里得距离 $d_2 = \sqrt{\sum (x_i - y_i)^2}$。仅当 `metric='minkowski'` 时生效。默认为 `2` | `1`、`2` |
| `metric` | `str` 或 `callable` | 距离度量方式。默认为 `"minkowski"`（配合 `p` 参数），也可直接设为 `"euclidean"`、`"manhattan"`、`"chebyshev"` 等 | `"minkowski"`、`"euclidean"`、`"manhattan"` |
| `metric_params` | `dict` 或 `None` | 距离度量的额外关键字参数。如对某些度量传入额外配置。默认为 `None` | `None`、`{}` |
| `n_jobs` | `int` 或 `None` | 并行作业数。`-1` 用全部核心，`None` 为单核。加速近邻搜索的并行计算。默认为 `None` | `None`、`-1`、`4` |

### 示例代码

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    metric="minkowski",
    p=2,
)
model.fit(X_train_s, y_train)
```

### 理解重点

- KNN 的"训练"与逻辑回归、SVC 很不同——`fit()` 不是优化损失函数，而是存储训练样本并建立近邻查询所需的数据结构（如 KD-Tree）。
- 因此 KNN 的 `fit()` 非常快（几乎无计算），但 `predict()` 较重（需要扫描训练集找近邻）。
- 当前封装的重点不是重写算法，而是把超参数、训练耗时和关键结果日志组织清楚。
- 最值得关注的三参数：`n_neighbors`、`weights`、`metric`/`p`——它们共同定义了"邻居是谁"和"怎么投票"。

## 3. 训练完成后最重要的模型属性

### 属性表

| 属性 | 类型 | 数学含义 |
|---|---|---|
| `classes_` | `ndarray` | 模型学到的类别标签数组，形状 `(n_classes,)`。当前二分类为 `[0, 1]` |
| `n_features_in_` | `int` | 训练时的特征维度 $d$。当前为 `2` |
| `effective_metric_` | `str` | 实际使用的距离度量名称。例如 `metric='minkowski'` 且 `p=2` 时返回 `'euclidean'` |
| `effective_metric_params_` | `dict` | 实际使用的距离度量参数。例如 `{'p': 2}` |
| `n_samples_fit_` | `int` | 训练样本数 $n_{\text{train}}$。当前为 $400 \times 0.8 = 320$ |
| `outputs_2d_` | `bool` | 输出是否为二维。用于内部判断 `predict_proba` 行为 |

### 示例代码

```python
print(f"实际度量: {model.effective_metric_}")
print(f"训练样本数: {model.n_samples_fit_}")
print(f"类别: {model.classes_}")
```

### 理解重点

- KNN 没有显式的"参数矩阵"（如逻辑回归的 $\mathbf{w}$），因此属性集中在配置信息和数据统计上。
- `effective_metric_` 和 `effective_metric_params_` 反映了 sklearn 内部的度量解析结果——你传 `'minkowski'` + `p=2`，它内部解析为 `'euclidean'` + `{'p': 2}`。
- `n_samples_fit_` 是 KNN 特有的属性，因为 KNN 的"知识"就是全体训练样本。

## 4. 训练阶段的工程封装

除了 `KNeighborsClassifier(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| 函数调用标题（`@print_func_info`） | 帮助在终端中定位训练入口 |
| 训练耗时（`@timeit`） | 观察 KNN `fit()` 的执行时间——通常非常快 |
| 超参数日志（`K`、`weights`、`metric`） | 确认当前训练使用的配置 |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把"构建模型""训练模型""打印结果"收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/knn.py` 保持简洁。

## 常见坑

1. 把 KNN 的 `fit(...)` 理解成和参数化模型一样的"求最优参数"过程——它是存储数据，不是优化。
2. 只知道可以 `predict(...)`，却忽略 `n_neighbors`、`weights`、`metric`/`p` 才是理解 KNN 行为的重要线索。
3. 忘记当前 `X_train` 应该是标准化后的特征——原始特征会让距离关系失真。
4. 忽略 `algorithm` 参数对大规模数据的影响——数据量大时暴力搜索会很慢。

## 小结

- `train_model(...)` 是本仓库 KNN 的核心训练入口，本质上是对 `sklearn.neighbors.KNeighborsClassifier` 的薄封装。
- `KNeighborsClassifier` 的 8 个构造器参数中，`n_neighbors`、`weights`、`metric`/`p` 是最核心的四个——它们决定"谁是你的邻居"和"怎么请邻居投票"。
- KNN 的 `fit()` 不学习参数，只存储数据 + 建立索引结构，这是它与所有参数化模型最根本的区别。
- 训练后属性 `effective_metric_`、`n_samples_fit_` 等反映了模型的实际底层配置。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 KNN 流水线到底执行了哪些步骤。
2. 理解训练集/测试集拆分、标准化、训练、类别预测和概率预测之间的连接关系。
3. 理解主模型与二维可视化模型在当前实现中的职责差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `knn_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `train_test_split(...)` | 方法 | 划分训练集与测试集 |
| `StandardScaler` | 类 | 对训练/测试特征做一致的标准化处理 |
| `train_model(X_train_s, y_train)` | 函数 | 训练主 KNN 模型 |
| `model.predict(X_test_s)` | 方法 | 生成测试集类别预测结果 |
| `model.predict_proba(X_test_s)` | 方法 | 生成测试集类别概率输出 |
| `PCA(n_components=2)` | 类 | 为决策边界可视化构造二维表示 |
| `model_2d` | 模型 | 专门用于二维决策边界展示 |

## 1. 流水线从复制数据开始

当前流水线先复制 `knn_data`，再拆出 `X` 和 `y`。

### 示例代码

```python
data = knn_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- 这和回归、决策树分册保持一致，体现了"原始数据只读、流程内部再处理"的习惯。
- 当前任务是监督二分类，因此 `y` 会真实参与训练和预测评估。

## 2. 先切分训练集与测试集

使用 `train_test_split` 将数据按 8:2 比例切分，并通过 `stratify=y` 保持类别分布一致。

### 参数速览

适用函数：`sklearn.model_selection.train_test_split`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*arrays` | `array_like` | 待切分的数据序列。传入 `(X, y)` 分别对应切分。长度必须一致 | `X, y` |
| `test_size` | `float` 或 `int` | 测试集占比（`0.0`~`1.0`）或绝对样本数。当前取 `0.2` | `0.2`、`0.3` |
| `random_state` | `int` | 随机种子，保证每次切分结果一致。当前取 `42` | `42` |
| `shuffle` | `bool` | 切分前是否打乱数据。默认为 `True` | `True` |
| `stratify` | `array_like` | 按此数组类别比例分层抽样。传入 `y` 确保每类在训练集和测试集中的比例一致，$\frac{n_{k,\text{train}}}{n_{k,\text{test}}} \approx \frac{N_{\text{train}}}{N_{\text{test}}}$ | `y`、`None` |

### 示例代码

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- 当前流水线明确区分了训练阶段和测试阶段。
- `stratify=y` 保证训练集和测试集保持相近的类别比例——这对二分类任务尤其重要。

## 3. 标准化只在训练集上拟合

标准化必须严格在切分后执行——`fit_transform` 在训练集上计算 $\mu_i, \sigma_i$，`transform` 将相同统计量应用于测试集。

### 参数速览

适用类：`sklearn.preprocessing.StandardScaler`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `scaler.fit_transform(X_train)` | 方法 | 计算训练集的均值 $\mu_i$ 和标准差 $\sigma_i$，并执行 $x_i' = (x_i - \mu_i)/\sigma_i$。返回 `X_train_s` | — |
| `scaler.transform(X_test)` | 方法 | 使用训练集的 $\mu_i$ 和 $\sigma_i$ 变换测试集，不重新计算统计量。返回 `X_test_s` | — |

### 示例代码

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 对 KNN 来说，标准化不是可选的——距离 $d_p(\mathbf{x}, \mathbf{y})$ 直接由各维度差异求和得到，量纲差异会让大值特征主导近邻关系。
- `fit_transform` vs `transform` 的区分模拟了真实部署场景：新数据只能用训练时的标准化参数。

## 4. 主模型训练与正式预测

KNN 的 `fit()` 并不做优化，而是存储训练样本并建立近邻查询索引。训练完成后，
`model.predict(...)` 为每个测试样本找到 $k$ 个最近邻居并投票决定输出类别。

### 参数速览

适用方法：`KNeighborsClassifier.predict(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的标准化特征矩阵。特征维度必须与训练时一致，即 $d = 2$ | `X_test_s` |
| 返回值 | `ndarray`，形状 `(n_samples,)` | 预测类别标签。对于二分类，$\hat{y}_i \in \{0, 1\}$。预测结果由 $k$ 个最近邻的多数投票决定：$\hat{y} = \arg\max_c \sum \mathbb{1}(y_i = c)$ | — |

### 示例代码

```python
model = train_model(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

### 理解重点

- `model` 是当前分册的主模型，用于正式训练和测试集类别预测。
- `fit()` 很快（只建索引），但 `predict()` 需要扫描全部训练样本计算距离——样本量越大预测越慢。
- `y_pred` 是后续混淆矩阵评估的直接输入。

## 5. 条件式概率输出如何进入流水线

当前流水线不是无条件调用概率输出，而是先做接口存在性检查：

### 参数速览

适用方法：`KNeighborsClassifier.predict_proba(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的标准化特征矩阵 | `X_test_s` |
| 返回值 | `ndarray`，形状 `(n_samples, n_classes)` | 各类别概率估计。对于 `weights='uniform'`，$P(\hat{y} = c \mid \mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k} \mathbb{1}(y_i = c)$。由于 $k=5$，概率值只能取 $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ | — |

### 示例代码

```python
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
```

### 理解重点

- `KNeighborsClassifier` 支持 `predict_proba(...)`，因此这段逻辑在当前实现中会生效。
- 显式加 `hasattr(...)` 是为了让流水线结构更稳健——方便复用到其他可能没有概率接口的分类器。
- KNN 的概率输出基于邻域内各类别频率，是离散值而非连续函数映射。
- 这些概率是 ROC 曲线可视化的直接输入。

## 6. 决策边界为什么要额外训练一个 model_2d

主模型在标准化后的原始特征空间中训练，但决策边界图需要能够在二维平面上对任意网格点做预测。
当前实现采用 PCA 投影到二维，再单独训练一个 KNN 模型用于可视化。

### 参数速览

适用类：`sklearn.decomposition.PCA`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的主成分数 $k$。$k=2$ 时将 $d$ 维特征投影到二维平面。PCA 通过 SVD 分解 $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$，取前 $k$ 个奇异向量构成投影矩阵 $\mathbf{V}_k$ | `2` |
| `random_state` | `int` | 随机种子。PCA 本身基于 SVD 是确定性的，但某些求解器使用随机化算法时需要。当前取 `42` | `42` |

### 示例代码

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_all_s = scaler.transform(X)  # 先标准化全量数据
X_2d = pca.fit_transform(X_all_s)
model_2d = KNeighborsClassifier(n_neighbors=5)
model_2d.fit(pca.transform(X_train_s), y_train)
```

### 理解重点

- 这里的 `model_2d` 不是主评估模型，而是专门为二维可视化服务的辅助模型。
- 主模型训练在标准化后的原特征空间中，而决策边界图需要二维输入来对每个网格点做预测。
- KNN 的边界可视化尤其有价值——可以直观看到局部投票产生的非线性、贴合数据的弧形分界。

## 7. 学习曲线如何接入流水线

学习曲线用于诊断模型性能是否随训练样本量增加而持续改善。

### 参数速览

适用函数：`result_visualization.learning_curve.plot_learning_curve`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的模型实例。传入 `KNeighborsClassifier(n_neighbors=5)`，内部会克隆并逐段训练，不修改传入实例 | `KNeighborsClassifier(n_neighbors=5)` |
| `X` | `array_like` | 标准化后的训练特征矩阵。当前传入 `X_train_s`，学习曲线内部按不同比例采样 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标。`"accuracy"` 即 $\frac{\sum \mathbb{1}[y_i = \hat{y}_i]}{n}$。默认为 `None`（使用 estimator 默认 score） | `"accuracy"`、`"f1"` |
| `cv` | `int` | 交叉验证折数。默认 `5`，每次对当前采样量做 5 折 CV 计算验证得分误差带 | `5`、`10` |

### 示例代码

```python
plot_learning_curve(
    KNeighborsClassifier(n_neighbors=5),
    X_train_s,
    y_train,
    title="KNN 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线使用新的 `KNeighborsClassifier` 实例，不直接复用 `model`——因为内部会克隆后重新训练。
- 对 KNN 而言，学习曲线尤其有助于观察：当 $k=5$ 固定、训练样本逐渐增加时，验证得分是否收敛。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/blog/articles/1780736299374.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 混为一谈——前者返回标签，后者返回概率。
2. 忽略当前流水线对 `predict_proba(...)` 做了接口存在性判断——这不是多余代码，而是结构稳健性的体现。
3. 把 `model_2d` 误认为正式预测模型本体——它仅在 PCA 空间训练，仅用于可视化。
4. 混淆主模型预测、二维可视化模型和学习曲线模型三者的职责——三者共享 `n_neighbors=5`，但在不同特征空间或数据子集上运行。

## 小结

- 当前 KNN 流水线的训练过程：复制数据 -> 切分 -> 标准化 -> 训练主模型（建索引）-> 类别预测 -> 概率预测 -> 多种可视化诊断。
- KNN 的独特之处：`fit()` 只建索引不优化；`predict()` 才真正做计算；标准化不是可选步骤。
- 对本仓库而言，`model`（标准化空间）、`model_2d`（PCA 空间）和学习曲线实例分别承担不同职责。

# 评估与诊断

## 本章目标

1. 明确当前仓库 KNN 实现实际上是如何做结果诊断的。
2. 理解混淆矩阵、ROC 曲线、PCA 决策边界图和学习曲线分别能说明什么。
3. 理解条件式概率输出与二维决策边界图的展示边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出，由 $k$ 近邻多数投票得到 |
| `y_scores` | 预测概率 | 测试集类别概率输出，由邻域内各类别频率得到 |
| `plot_confusion_matrix(...)` | 函数 | 绘制预测标签与真实标签的混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制二分类 ROC 曲线 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |

## 1. 当前仓库的评估入口

当前 KNN 流水线里的主要结果诊断手段有四个：

1. 混淆矩阵
2. ROC 曲线
3. PCA 2D 决策边界图
4. 学习曲线

### 示例代码

```python
y_pred = model.predict(X_test_s)

if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)

plot_confusion_matrix(...)
plot_roc_curve(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 当前实现同时提供结果矩阵、概率曲线、边界图和曲线图四类视角。
- KNN 没有特征重要性图（这是决策树特有的），因此评估方式比决策树少一项。
- 四种可视化分别回答不同问题：分对了吗（混淆矩阵）、概率区分力如何（ROC）、边界长什么样（决策边界）、更多数据有用吗（学习曲线）。

## 2. 混淆矩阵能观察什么

混淆矩阵 $\mathbf{C}$ 是一个 $2 \times 2$ 矩阵（二分类），$C_{ij}$ 表示真实类别为 $i$、预测类别为 $j$ 的样本数：

$$
C = \begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}
$$

### 参数速览

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $y_i \in \{0, 1\}$ | `y_test` |
| `y_pred` | `array_like`，形状 `(n_samples,)` | 模型预测标签，来自 `model.predict(X_test_s)` | `y_pred` |
| `normalize` | `bool` 或 `str` | 归一化方式：`True`/`'true'` 按行（真实类别），`'pred'` 按列（预测类别），`'all'` 按全体。默认为 `False`（绝对数量） | `True`、`'true'` |

### 示例代码

```python
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    title="KNN 混淆矩阵",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 混淆矩阵最适合回答：模型把正负类分别分对了多少，误分类倾向哪个方向。
- 对当前二分类双月牙任务，对角线元素即正确分类数，非对角线反映两个半月交界处的混淆情况。

## 3. ROC 曲线能观察什么

ROC 曲线绘制真正例率（TPR）随假正例率（FPR）变化的轨迹：

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

AUC（曲线下面积）$\in [0.5, 1.0]$，越接近 1 表示区分能力越强。

### 参数速览

适用函数：`plot_roc_curve(y_test, y_scores, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $y_i \in \{0, 1\}$ | `y_test` |
| `y_scores` | `array_like`，形状 `(n_samples, n_classes)` | 各类别概率估计，来自 `model.predict_proba(X_test_s)`。二分类时使用正类（类别 1）的概率列作为得分 | `y_scores` |
| `pos_label` | `int` | 正类标签。二分类默认取 `1`，即取 `y_scores[:, 1]` 作为阳性得分 | `1` |

### 示例代码

```python
plot_roc_curve(
    y_test,
    y_scores,
    title="KNN ROC 曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- KNN 的概率输出来自邻域频率，取值离散（分母为 $k=5$），因此 ROC 曲线的阶梯形状会比逻辑回归更明显。
- 当前代码在调用前显式检查了 `predict_proba(...)` 是否存在，体现了对不同分类器接口差异的兼容。

## 4. PCA 2D 决策边界图能观察什么

决策边界图在二维平面上以不同颜色填充不同预测区域，直观展示 KNN 的局部邻域分类产生的边界形状。

### 参数速览

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model_2d` | `KNeighborsClassifier` | 在 PCA 二维空间上单独训练的 KNN 模型。`n_neighbors=5`，与主模型共享相同的 $k$ | `model_2d` |
| `X_2d` | `ndarray`，形状 `(n_samples, 2)` | 标准化后 PCA 投影到二维的特征，用于生成网格点预测和散点着色。列分别为 PC1、PC2 | `X_2d` |
| `y` | `array_like`，形状 `(n_samples,)` | 全量标签数组，用于散点的真实类别着色 | `y.values` |

### 示例代码

```python
plot_decision_boundary(
    model_2d,
    X_2d,
    y.values,
    title="KNN 决策边界 (PCA 2D)",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- KNN 的边界通常会呈现蜿蜒曲线，贴近数据的局部结构，与决策树的轴对齐分段边界和逻辑回归的直线边界截然不同。
- 当 $k$ 较小时，边界锯齿状明显（紧贴训练样本）；$k$ 较大时，边界更平滑。
- 但这只是 PCA 投影空间中的近似展示，不是原始邻域关系的完整表达。

## 5. 学习曲线能观察什么

学习曲线绘制训练得分和交叉验证得分随训练样本数增加的变化：

- 训练得分高、验证得分低且差距大 -> 过拟合倾向（对 KNN，通常 $k$ 太小时出现）
- 验证得分持续上升且未收敛 -> 更多数据可能有帮助

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的 KNN 模型实例。传入 `KNeighborsClassifier(n_neighbors=5)`，内部克隆后逐段训练 | `KNeighborsClassifier(n_neighbors=5)` |
| `X` | `array_like` | 标准化后的训练特征矩阵。学习曲线内部按不同比例逐步增加样本量 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标。当前取 `"accuracy"`，即 $\frac{\sum \mathbb{1}[y_i = \hat{y}_i]}{n}$ | `"accuracy"`、`"f1"` |
| `cv` | `int` | 交叉验证折数。默认 `5`，每次对当前采样量做 5 折 CV 计算验证得分误差带 | `5`、`10` |
| `train_sizes` | `array_like` | 训练样本量的递增序列。默认为 `np.linspace(0.1, 1.0, 5)` | `[0.1, 0.33, 0.55, 0.78, 1.0]` |

### 示例代码

```python
plot_learning_curve(
    KNeighborsClassifier(n_neighbors=5),
    X_train_s,
    y_train,
    title="KNN 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- KNN 的学习曲线能反映：在 $k=5$ 固定的情况下，更多训练数据能否提升邻域投票的可靠性。
- 如果验证得分在小样本量时已经较高且接近训练得分，说明现有数据量已足够——邻域信息已经充分。
- 验证得分误差带（CV 标准差）能提示模型在不同数据划分下的稳定性。

## 6. 当前实现中尚未纳入但常见的分类指标

在一般分类任务中，还常见以下指标：

| 指标 | 公式 | 说明 |
|---|---|---|
| 准确率（Accuracy） | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | 整体正确率 |
| 精确率（Precision） | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | 预测为正类的样本中有多少确实是正类 |
| 召回率（Recall） | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | 真实正类样本中有多少被正确找出 |
| F1 分数 | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Precision 和 Recall 的调和平均 |

### 理解重点

- 当前仓库没有在 KNN 流水线中显式打印这些指标，而是通过混淆矩阵隐式呈现。
- 文档可以提到它们是常见扩展方向，但不能写成"当前源码已经在单独计算"。

## 评估图表

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

![ROC 曲线](https://img.yumeko.site/file/blog/articles/1780736321096.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 的用途混为一谈——前者用于混淆矩阵，后者用于 ROC。
2. 忽略当前流水线对 `predict_proba(...)` 做了条件检查——不是所有分类器都有概率接口。
3. 把 PCA 决策边界图误认为原始特征空间邻域结构的完整表达——它是投影近似。
4. 把当前仓库未实现的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 KNN 的评估方式：混淆矩阵看错误分布，ROC 曲线看概率区分力，PCA 决策边界图看边界形状，学习曲线看训练行为。
- KNN 没有特征重要性评估（它是基于实例的懒惰学习，不是基于特征分裂的模型），这与决策树分册不同。
- 四项评估组合起来，能全面解释当前 $k=5$ 的 KNN 在双月牙数据上的实际表现——特别是其非线性边界的优势。

# 工程实现

## 本章目标

1. 从工程角度看清 KNN 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解为什么当前实现要把训练逻辑、标准化逻辑、预测逻辑和可视化逻辑拆开。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.knn()` 生成双月牙二分类数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `knn_data` |
| 训练封装 | `model_training/classification/knn.py` | 构建并训练 `KNeighborsClassifier`，打印训练日志 |
| 流水线入口 | `pipelines/classification/knn.py` | 组织切分、标准化、训练、预测与可视化评估的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 保存混淆矩阵图 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 保存二分类 ROC 曲线图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 保存学习曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.knn
```

### 理解重点

- 这个命令是理解 KNN 工程实现的最佳入口。
- 它会依次完成数据读取、标准化、模型训练、预测和结果可视化。
- 如果只读一个文件，建议先读 `pipelines/classification/knn.py`——编排层。

## 2. run() 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = knn_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 标准化（仅训练集上 fit）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. 训练主模型（建索引）& 正式预测
    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_s)

    # 5. 可视化诊断（混淆矩阵、ROC、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_roc_curve(y_test, y_scores, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(KNeighborsClassifier(...), X_train_s, y_train, ...)
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——每一步的输入明确来自上一步的输出。
- 数据流是单向的：数据 -> 切分 -> 标准化 -> 训练 -> 预测 -> 评估。

## 3. 训练模块负责什么

`model_training/classification/knn.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `KNeighborsClassifier(...)` 实例（按给定超参数）
2. 调用 `model.fit(X_train, y_train)`——存储样本、建立近邻查询结构
3. 打印训练日志：训练耗时、$k$、`weights`、`metric`
4. 返回训练完成的主模型对象

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵。传入 `KNeighborsClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，取值 $\{0, 1\}$ | `y_train` |
| `n_neighbors` | `int` | 近邻数量 $k$。默认 `5` | `3`、`5`、`15` |
| `weights` | `str` | 投票权重方式。默认 `"uniform"` | `"uniform"`、`"distance"` |
| `metric` | `str` | 距离度量方式。默认 `"minkowski"` | `"minkowski"`、`"euclidean"` |
| 返回值 | `KNeighborsClassifier` | 已完成 `fit()` 的模型对象，可直接调用 `predict()` 和 `predict_proba()` | — |

### 理解重点

- 这层抽离让"模型构建逻辑"和"流程编排逻辑"分开——`train_model()` 可被流水线调用，也可单独运行做局部验证。
- 这也是当前仓库多个算法分册共享的组织方式。

## 4. 四类评估模块分别负责什么

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 混淆矩阵图片（PNG） |
| ROC 曲线 | `plot_roc_curve(...)` | `y_test`、`y_scores` | 二分类 ROC 曲线图片（PNG） |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `estimator`、`X_train_s`、`y_train` | 训练/验证得分曲线（PNG） |

### 理解重点

- 四类可视化都不是训练的一部分，而是训练完成后的诊断步骤——它们不修改模型。
- 决策边界图依赖单独训练的 `model_2d`（在 PCA 空间），ROC 曲线依赖 `predict_proba()` 的输出。
- KNN 没有特征重要性评估，因为它是基于实例的懒惰学习——没有"特征贡献"这个概念。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `knn_data` | `data_generation/classification.py` | `pipelines/classification/knn.py` |
| `model`（主模型） | `train_model(...)` | `predict`、`predict_proba` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `y_scores` | `model.predict_proba(...)` | `plot_roc_curve` |
| `model_2d` | `KNeighborsClassifier.fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/knn/` 目录 |

### 理解重点

- KNN 的流水线与决策树的关键差异在于：先标准化再训练；没有特征重要性评估模块。
- 数据流向单向、无循环依赖，每个模块可以独立测试和替换。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `KNN 分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、$k$、`weights`、`metric` | 确认 KNN 配置和训练效率 |
| 混淆矩阵图 | `outputs/knn/confusion_matrix.png` | 观察正负类误分类方向 |
| ROC 曲线图 | `outputs/knn/roc_curve.png` | 评估二分类概率区分能力 |
| 决策边界图 | `outputs/knn/decision_boundary.png` | 观察 KNN 的非线性弧形边界 |
| 学习曲线图 | `outputs/knn/learning_curve.png` | 诊断过拟合/欠拟合倾向 |

### 理解重点

- 运行结果不只是模型对象，还包括日志和多种图像产物。
- 对教学仓库而言，"代码 + 日志 + 图像"的组合比单纯返回分类结果更能帮助理解 KNN 的局部分类行为。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/knn.py` — 入口，了解整体流程
2. 再看 `model_training/classification/knn.py` — 训练封装，理解超参数和日志
3. 再看 `result_visualization/confusion_matrix.py` — 基础分类结果评估
4. 再看 `result_visualization/roc_curve.py` — 概率区分能力评估
5. 再看 `result_visualization/decision_boundary.py` — 空间划分可视化
6. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
7. 最后回到 `data_generation/classification.py` — 理解数据生成参数

### 理解重点

- 从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 这个顺序也对应了数据流方向：数据 -> 标准化 -> 训练 -> 预测 -> 评估。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/blog/articles/1780736366408.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的训练在 `model_training/` 中。
2. 不区分"主模型""二维可视化模型"和"学习曲线模型实例"的职责边界——三者训练在不同空间或不同数据子集。
3. 忽略 KNN 的 `fit()` 和参数化模型的 `fit()` 本质不同——前者建索引，后者做梯度优化。
4. 只看单个文件，不顺着调用链理解整体执行流程。

## 小结

- 当前 KNN 工程实现采用清晰的模块分层：数据生成 -> 训练封装 -> 流水线编排 -> 结果可视化。
- `run()` 负责串联流程，`train_model(...)` 负责训练主模型（实际是建索引），各可视化函数负责结果展示与诊断。
- 数据流单向，各模块职责单一，便于教学讲解和后续扩展。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 KNN 实现。
2. 给出继续深入阅读 KNN 与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/knn.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 `make_moons(n_samples=400, noise=0.1)` 数据适合 KNN 的局部邻域思路？如果用 `make_blobs` 会怎样？
3. 当前 `train_model(...)` 中的 `n_neighbors`、`weights`、`metric`/`p` 分别控制什么？各自的数学含义是什么？
4. 为什么标准化对 KNN 特别重要？如果不标准化，距离 $d_p(\mathbf{x}, \mathbf{y}) = (\sum \vert x_i - y_i \vert^p)^{1/p}$ 会出现什么问题？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？KNN 的概率值为什么只能是离散的？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它和主模型 `model` 在什么特征空间上训练？
7. KNN 的 `fit()` 和逻辑回归的 `fit()` 有什么本质区别？为什么 KNN 被称为懒惰学习？

## 练习方向

### 1. 改动 n_neighbors

- 把 `n_neighbors=5` 改成 `1`、`3`、`15`、`50`、`100`
- 观察变化：
  - 混淆矩阵中各类别正确/错误分布
  - ROC 曲线 AUC 值的变化
  - 决策边界图的弯曲程度（$k$ 小 -> 锯齿状边界，$k$ 大 -> 平滑边界）
  - 学习曲线中训练得分与验证得分的差距
- 思考：$k$ 值与偏差-方差权衡的关系——小 $k$ 低偏差高方差，大 $k$ 高偏差低方差

### 2. 改动 weights

- 把 `weights='uniform'` 改成 `weights='distance'`
- 对比变化：
  - 决策边界形状——加权投票通常产生更精细的边界
  - ROC 曲线 AUC 值——
  - 对噪声的敏感性——加权投票对噪声更敏感
- 理解：加权投票的数学公式 $\hat{y} = \arg\max_c \sum \frac{\mathbb{1}(y_i=c)}{d(\mathbf{x}, \mathbf{x}_i)}$，邻居越近权重越大

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test`
- 对比模型训练结果和可视化输出
- 体会：距离计算中量纲大的特征如何主导 $d_p(\mathbf{x}, \mathbf{y})$，量纲小的特征几乎形同虚设

### 4. 改动 metric 与 p

- 尝试 `metric='manhattan'`（$p=1$）与 `metric='minkowski'`（$p=2$，默认）
- 对比决策边界的变化——曼哈顿距离倾向于产生菱形边界，欧几里得距离倾向于产生圆弧边界
- 尝试调整 `noise` 参数（如 `0.0`、`0.3`），观察边界复杂度随噪声的变化

### 5. 与 Logistic Regression 和 Decision Tree 对比

- 对照阅读 `docs/classification/logistic_regression/` 和 `docs/classification/decision_tree/`
- 比较要点：
  - KNN 的局部投票 vs 逻辑回归的全局线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$ vs 决策树的轴对齐切分
  - 是否需要标准化：KNN 需要，逻辑回归需要，决策树不需要
  - 评估方式差异：KNN 和逻辑回归都没有特征重要性图，决策树有
- 分别在同一数据（`make_moons`）上运行三个流水线，对比混淆矩阵和 ROC 曲线

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`KNeighborsClassifier` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_moons` | 双月牙数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Nearest Neighbors | KNN 算法原理、距离度量选择与搜索算法的详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 13 章：Prototype Methods and Nearest-Neighbors，涵盖 KNN、偏差-方差分析、距离度量选择 |

- scikit-learn `KNeighborsClassifier`：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- scikit-learn `make_moons`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
- scikit-learn 用户指南 Nearest Neighbors：https://scikit-learn.org/stable/modules/neighbors.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 KNN 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），KNN 比决策树更需要标准化（距离度量依赖特征尺度）
  - 局部邻域投票思路——近邻是谁就投谁，$k$ 控制局部范围大小
  - $k$ 值与偏差-方差权衡——小 $k$ 过拟合、大 $k$ 欠拟合
  - `predict_proba(...)` 的概率来自邻域频率，取值离散（分母为 $k$）
  - `model`、`model_2d` 和学习曲线实例分别在标准化空间、PCA 空间和交叉验证循环中运行
