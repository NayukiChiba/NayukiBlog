---
title: DecisionTreeClassifier 决策树分类
date: 2026-04-09
category: 机器学习/分类
tags:
  - Scikit-learn
  - 基础
description: 决策树分类核心数学原理、划分标准及超参数解析。
image: https://img.yumeko.site/file/blog/cover/1780581733465.webp
status: published
---


# 数学原理


## 本章目标

1. 理解决策树如何通过递归划分特征空间完成分类。
2. 理解熵、信息增益、基尼不纯度在当前实现中的角色与数学定义。
3. 理解树深、叶节点数和剪枝思路为什么与过拟合直接相关。
4. 掌握每个超参数对应的数学含义与控制对象。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 信息熵 $H(Y)$ | 不纯度度量 | 衡量类别分布混乱程度，$H(Y) = -\sum p_k \log_2 p_k$ |
| 信息增益 $\text{Gain}(D, A)$ | 划分标准 | 衡量某个特征带来的不确定性下降，$\text{Gain} = H(D) - H(D \mid A)$ |
| 增益率 $\text{Gain\_ratio}$ | 划分标准 | 修正信息增益偏好多值特征的问题 |
| 基尼不纯度 $\text{Gini}(D)$ | 划分标准 | 当前源码默认 `criterion='gini'` 对应的核心度量，$\text{Gini}(D) = 1 - \sum p_k^2$ |
| 树深 | 复杂度指标 | 反映树的层级深度，与 `max_depth` 参数直接对应 |
| 叶节点数 | 复杂度指标 | 反映划分后的终端区域数量，与 `min_samples_leaf` 参数相关 |

## 1. 决策树的核心思想

决策树通过递归地选择最优特征并按其取值将数据集分割为子集，构建一棵树状判别结构。每个内部节点对应一个特征判断，每个叶节点对应一个类别输出。

### 理解重点

- 决策树不像逻辑回归那样先学习一条全局边界，而是不断把数据切成更纯净的小区域。
- 每一次划分，本质上都是在问：用哪个特征、按什么阈值切，能让子节点更"纯"。
- 这也是为什么它常常被理解为一连串 if-else 规则的组合。

## 2. 熵与条件熵

### 信息熵

对随机变量 $Y$ 取 $K$ 个类别，若分布为 $p_k = P(Y=k)$，信息熵定义为：

$$
H(Y) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

- 当所有类别等概率时，熵最大。
- 当所有样本属于同一类别时，熵为 0。

### 条件熵

给定特征 $A$ 将数据划分为 $V$ 个子集 $D_1, D_2, \dots, D_V$：

$$
H(Y \mid A) = \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(Y_{D_v})
$$

### 理解重点

- 熵衡量的是节点的类别混乱程度。
- 条件熵衡量的是：在用某个特征切分之后，子节点整体还剩多少混乱度。
- 因此一个好的划分，应该让条件熵尽量小。

## 3. 信息增益与增益率

### 信息增益

$$
\text{Gain}(D, A) = H(D) - H(D \mid A)
$$

ID3 算法选择信息增益最大的特征进行分裂。

### 增益率

为修正信息增益偏好多值特征的问题，C4.5 引入特征固有值：

$$
\text{IV}(A) = -\sum_{v=1}^{V} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}
$$

$$
\text{Gain\_ratio}(D, A) = \frac{\text{Gain}(D, A)}{\text{IV}(A)}
$$

### 理解重点

- 信息增益本质上是在比较划分前后的不确定性减少了多少。
- 增益率则试图抑制"取值很多的特征看起来天然更有用"这一偏差。
- 当前源码虽然没有直接使用这些术语作为参数，但理解这部分有助于读懂决策树分裂的基本思想。

## 4. 基尼不纯度与 `criterion` 参数

当前源码使用基尼不纯度作为默认划分标准。基尼不纯度的数学定义为：

$$
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
$$

对某个特征划分后的总体基尼不纯度：

$$
\text{Gini}(D, A) = \sum_{v=1}^{V} \frac{|D_v|}{|D|} \text{Gini}(D_v)
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `criterion` | `str` | 划分不纯度度量函数。`"gini"` 对应 $\text{Gini}(D) = 1 - \sum p_k^2$，计算更快；`"entropy"` 对应 $H(D) = -\sum p_k \log_2 p_k$，对概率变化更敏感。两者在大多数任务中表现接近。默认为 `"gini"` | `"gini"`、`"entropy"`、`"log_loss"` |

### 理解重点

- 当前工程实现的理论核心应该落在基尼不纯度，因为源码默认 `criterion='gini'`。
- 基尼和熵都能衡量节点纯度，但基尼计算更快，也是 CART 风格实现的默认选择。
- 因此文档不应平均展开所有树算法，而应优先解释当前真实实现对应的 CART 风格分类树。

## 5. 树深与叶节点——复杂度控制参数

### `max_depth`

树的最大深度 $d_{\max}$。深度每增加一层，模型就能多切一次特征空间。数学上，一棵深度为 $d$ 的完全二叉树最多有 $2^d$ 个叶节点。

### `min_samples_split`

内部节点继续分裂所需的最小样本数。设当前节点包含 $n$ 个样本，若 $n <$ `min_samples_split`，则该节点不再分裂。

### `min_samples_leaf`

叶节点必须包含的最小样本数。若一次分裂会导致任一子节点样本数少于该值，分裂被拒绝。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_depth` | `int` 或 `None` | 树的最大深度 $d_{\max}$。限制划分轮数——深度越大，模型越复杂，越容易过拟合。`None` 表示不限制，节点持续分裂直到纯净或触及其他停止条件。默认为 `None` | `3`、`6`、`None` |
| `min_samples_split` | `int` 或 `float` | 内部节点再划分所需最小样本数。若为 `float`（如 `0.1`），表示比例 × 总样本数。增大可抑制过拟合。默认为 `2` | `2`、`4`、`10` |
| `min_samples_leaf` | `int` 或 `float` | 叶节点最少样本数。若为 `float`，表示比例。增大使树更保守——叶节点不会包含极少数样本。默认为 `1` | `1`、`2`、`5` |

### 理解重点

- 树越深，越容易把训练集切得很细，拟合得很"完美"。
- 但切得太细通常意味着泛化能力下降，更容易记住噪声。
- 当前源码把 `max_depth=6`、`min_samples_split=4`、`min_samples_leaf=2` 显式写出来，就是在限制树的复杂度。

## 6. 剪枝与复杂度控制

### 预剪枝（当前源码采用）

在构建树时提前终止分裂，典型方式包括限制树深、限制叶节点最小样本数、限制继续分裂所需最小样本数。当前源码通过 `max_depth`、`min_samples_split`、`min_samples_leaf` 实现预剪枝。

### 后剪枝（理论补充）

在树构建完成后再回头删掉一部分分支，如代价复杂度剪枝（Cost Complexity Pruning）：

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

其中 $R(T)$ 是树的误分类损失，$|T|$ 是叶节点数，$\alpha$ 是复杂度惩罚系数。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `ccp_alpha` | `float` | 代价复杂度剪枝参数 $\alpha$。越大则剪枝越激进，叶节点越少。数学上最小化 $R(T) + \alpha \cdot |T|$。`0.0` 表示不剪枝。默认为 `0.0` | `0.0`、`0.01`、`0.1` |

### 理解重点

- 当前源码最直接体现的是预剪枝思想，因为它通过 `max_depth`、`min_samples_split`、`min_samples_leaf` 控制复杂度。
- 文档可以提到后剪枝是经典思路和 `ccp_alpha` 参数，但不能写成当前流水线里已经显式执行了某个后剪枝步骤。

## 7. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 基尼不纯度 | $\text{Gini}(D) = 1 - \sum p_k^2$ | `criterion='gini'`（`DecisionTreeClassifier` 默认） |
| 信息熵 | $H(D) = -\sum p_k \log_2 p_k$ | `criterion='entropy'` |
| 最大树深 | $d_{\max}$ | `max_depth=6` |
| 最小分裂样本数 | — | `min_samples_split=4` |
| 最小叶节点样本数 | — | `min_samples_leaf=2` |
| 代价复杂度剪枝 | $R_\alpha(T) = R(T) + \alpha \cdot \vert T\vert$ | `ccp_alpha=0.0`（当前未启用） |
| 特征重要性 | 基于不纯度下降加权 | `model.feature_importances_` |
| 树深度 | 实际 $d$ | `model.get_depth()` |
| 叶节点数 | 实际 $\vert T\vert$ | `model.get_n_leaves()` |

## 常见坑

1. 只知道决策树会"分裂"，却说不清它分裂的依据是不纯度下降。
2. 把信息增益、增益率、基尼不纯度混为一谈。
3. 忽略树深与叶节点数本质上是在控制模型复杂度。
4. 把 `criterion='entropy'` 与交叉熵损失函数混淆——这里的 entropy 是节点不纯度，不是损失函数。
5. 把剪枝写成当前源码里已经显式实现的完整后处理流程。

## 小结

- 决策树的核心，是不断选择能最大程度降低不纯度的划分。
- 当前源码默认使用 `criterion='gini'`，数学上对应 $\text{Gini}(D) = 1 - \sum p_k^2$。
- `max_depth`、`min_samples_split`、`min_samples_leaf` 分别从深度、分裂门槛、叶节点样本数三个角度控制复杂度。
- 读懂不纯度、树深和叶节点的关系之后，再看训练日志和特征重要性会更顺畅。

# 数据构成


## 本章目标

1. 明确本仓库 Decision Tree 数据来自 `ClassificationData.decision_tree()` 的 blob 生成逻辑。
2. 明确 `make_blobs` 各参数的数据含义与当前取值。
3. 明确训练集/测试集切分方式，以及为什么当前主模型流程里没有显式标准化步骤。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `ClassificationData.decision_tree()` | 方法 | 生成决策树使用的二维多分类数据 |
| `make_blobs(...)` | 函数 | scikit-learn 提供的多簇分类数据生成器 |
| `decision_tree_classification_data` | 变量 | 在 `data_generation/__init__.py` 中导出的数据对象 |
| `label` | 列名 | 当前流水线中的监督分类标签 |
| `feature_names` | 变量 | 用于特征重要性图显示的特征名列表 |

## 1. 本仓库数据入口

- 数据变量：`data_generation/__init__.py` 中导出的 `decision_tree_classification_data`
- 生成来源：`data_generation/classification.py` 中的 `ClassificationData.decision_tree()`
- 流水线使用：`pipelines/classification/decision_tree.py` 中的 `data = decision_tree_classification_data.copy()`

### 理解重点

- `decision_tree_classification_data` 在导入时就已经生成完成，因此流水线里直接 `.copy()` 使用即可。
- 用 `.copy()` 的目的，是避免后续处理意外修改原始数据对象。
- 当前数据是为决策树教学场景专门构造的，因此与区域切分直觉比较匹配。

## 2. 数据生成函数 `ClassificationData.decision_tree()`

底层调用 `sklearn.datasets.make_blobs`，生成多个各向同性高斯簇的数据。

### 参数速览

`make_blobs` 核心参数：

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数 $N$。当前取 `400`，每个类别约 100 个样本。默认为 `100` | `100`、`400`、`1000` |
| `centers` | `int` 或 `ndarray` | 簇中心数 $K$（即类别数）。当前取 `4`，生成 4 个各向同性高斯簇。若传入数组则为各中心坐标。默认为 `None`（需指定） | `3`、`4`、`[(0,0),(5,5)]` |
| `cluster_std` | `float` 或 `array_like` | 各簇的标准差 $\sigma$。控制类内离散程度——$\sigma$ 越大各类重叠越多，分类越难。当前取 `1.0`。默认为 `1.0` | `0.5`、`1.0`、`2.0` |
| `random_state` | `int` | 随机种子，保证每次生成相同数据。默认为 `None` | `42` |
| `n_features` | `int` | 特征维度 $d$。当前取 `2`，便于可视化。默认为 `2` | `2`、`10`、`100` |
| `return_centers` | `bool` | 是否同时返回簇中心坐标。默认为 `False` | `True` |

### 示例代码

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=1.0,
    random_state=42,
)
# X.shape = (400, 2), y 取值 {0, 1, 2, 3}
```

### 理解重点

- 当前数据是二维 4 分类 blob 数据，类别分布在不同区域。
- 这种数据很适合展示决策树如何通过一系列轴对齐切分把样本空间分成若干块。
- `cluster_std` 越大，各类之间的重叠越多，决策树边界越复杂——这是实验调参的好入口。

## 3. 特征列与标签列

当前数据表结构：

- 特征列：`x1`、`x2`（二维实数特征）
- 标签列：`label`（取值为 0、1、2、3 的多分类标签）

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
feature_names = list(X.columns)  # ['x1', 'x2']
```

### 理解重点

- `label` 是监督训练标签，会真实参与 `model.fit(X_train, y_train)`。
- `feature_names` 会被后续特征重要性图复用，因此当前流水线在早期就把它提取出来。
- 这说明决策树分册除了分类预测，还特别强调"树如何利用特征"的解释层。

## 4. 切分与当前预处理特点

### 参数速览

`train_test_split` 参数：

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` 或 `int` | 测试集占比（`0.0`~`1.0`）或绝对样本数。当前取 `0.2`（20%）。默认为 `None`（需指定其一） | `0.2`、`0.3`、`100` |
| `random_state` | `int` | 随机种子，保证每次切分结果一致。默认为 `None` | `42` |
| `stratify` | `array_like` | 按此数组类别比例分层抽样。传入 `y` 确保训练集和测试集各类别比例与原始数据一致。默认为 `None` | `y`、`None` |
| `shuffle` | `bool` | 切分前是否打乱数据。默认为 `True` | `True` |

### 示例代码

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- 当前主模型训练流程没有像 KNN、SVC、Logistic Regression 那样显式做标准化。
- 这是因为树模型基于阈值切分（如 $x_1 \leq 3.2$），不依赖欧氏距离或梯度优化，因此对特征尺度不敏感。
- 这也是当前决策树分册在工程流程上与距离型模型显著不同的地方。

## 数据可视化

![数据分布散点图](https://img.yumeko.site/file/blog/articles/1780736139527.png)

![类别分布](https://img.yumeko.site/file/blog/articles/1780737700996.png)

![特征相关性](https://img.yumeko.site/file/blog/articles/1780736130799.png)

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来。
2. 忽略 `feature_names` 在特征重要性图中的作用。
3. 误以为所有分类模型都必须先标准化，忽略树模型的阈值划分机制不同。
4. 只看到 blob 数据简单，却忽略它和决策树区域切分直觉高度匹配。

## 小结

- 当前 Decision Tree 数据来自 `ClassificationData.decision_tree()`，底层使用 `make_blobs(n_samples=400, centers=4)`。
- 数据表结构清晰：`x1`、`x2` 是二维特征，`label` 是 4 分类监督标签。
- 树模型基于阈值切分，不依赖距离尺度——因此不需要标准化，这是与 KNN/SVM/逻辑回归的核心工程差异。
- 读懂数据来源、切分方式和预处理选择，是理解后续训练与评估章节的前提。

# 思路与直觉


## 本章目标

1. 用直观方式理解决策树到底在做什么。
2. 理解为什么它在当前二维多分类 blob 数据上更能体现区域切分优势。
3. 理解它与更偏全局线性边界的方法在思路上的关键差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 规则切分 | 核心直觉 | 通过连续提问把数据逐步分开 |
| 区域划分 | 决策形状 | 形成一块块轴对齐的分类区域 |
| 树深 $d_{\max}$ | 复杂度指标 | 控制规则切分有多细，对应 `max_depth` 参数 |
| 特征重要性 | 可解释性 | 帮助理解哪些特征最常被用来切分 |
| 逻辑回归 | 对比算法 | 更偏全局线性边界 $\mathbf{w}^T\mathbf{x} + b = 0$ |

## 1. 为什么需要决策树

对于很多分类问题，我们不一定想先拟合一个整体公式。有时更自然的问题是：

能不能通过一连串规则判断，把样本一步步分到正确类别里？

决策树给出的思路是：

- 先问一个最有区分度的问题。
- 再在每个子区域里继续问下一个问题。
- 重复下去，直到区域足够纯净。

### 理解重点

- 这使决策树成为一种非常直观的规则模型。
- 它的预测更像执行一串 if-else 判断，而不是代入一个全局公式。
- 这也是为什么决策树天然具有较强可解释性。

## 2. 为什么当前仓库示例里它表现合理

当前决策树数据来自 `make_blobs(n_samples=400, centers=4)`，特点是：

- 4 个类别
- 二维空间
- 各类分布在不同区域

### 理解重点

- 对这类数据来说，把平面切成一块块区域是很自然的分类方式。
- 决策树通过轴对齐切分，能够较直观地把不同类别样本分到不同区域里。
- 这正是当前数据能体现决策树优势的原因。

## 3. 用"不断提问"理解算法

可以把决策树理解成：

1. 先问一个能最好分开样本的问题。
2. 根据回答把样本分到不同分支。
3. 对每个分支继续重复这一过程。

### 理解重点

- 这也是为什么 `max_depth` 在当前分册里最值得专门观察——它限制最多能问多少轮问题。
- 如果把决策树仅理解成"一个会分类的黑盒"，就会错过它最有辨识度的规则切分思想。

## 4. 为什么 blob 数据适合树的区域切分

### 理解重点

- 当前数据在二维空间中形成多个类别区域，和决策树"不断划分空间"的方式比较匹配。
- 决策树的边界通常是轴对齐的（如 $x_1 \leq 3.2$），因此在这类区域化分布中容易形成清晰解释。
- 这也是为什么当前分册里特征重要性图也很有意义：树确实是在用具体特征做一层层切分。

## 5. 与逻辑回归的直觉差异

两者的核心差异可以这样理解：

| 算法 | 决策边界 | 核心依据 | 适用场景 |
|---|---|---|---|
| LogisticRegression | 全局线性超平面 $\mathbf{w}^T\mathbf{x} + b = 0$ | 线性打分 + Sigmoid 概率映射 | 近线性可分数据 |
| DecisionTreeClassifier | 局部轴对齐分段边界 | 不纯度下降 + 递归划分 | 区域化分布数据 |

### 理解重点

- 逻辑回归更像是在问：整张图上最好的一条全局边界是什么。
- 决策树更像是在问：先切哪一刀最能让类别变纯，再切下一刀。
- 当前 blob 数据更适合把决策树作为规则切分基线来讲解。

## 可视化

![决策边界](https://img.yumeko.site/file/blog/articles/1780736145017.png)

## 常见坑

1. 只知道决策树很直观，却说不出它为什么适合当前 blob 数据。
2. 把所有树切分都理解成"随机切"，而忽略它其实在优化不纯度下降。
3. 只关注预测结果，不关注树深、叶节点和特征重要性的解释价值。
4. 忽略树越深越容易过拟合这一现实风险。

## 小结

- 决策树的直觉重点有两个：规则切分和区域划分。
- 当前仓库使用二维多分类 blob 数据，正好体现了它通过连续提问划分空间的能力。
- 与逻辑回归的全局线性边界不同，决策树形成的是局部、轴对齐的分段决策边界。
- 如果已经理解"为什么它不断提问并把空间切成越来越纯净的区域"，就已经抓住了本分册最核心的直觉。

# 模型构建


## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `DecisionTreeClassifier`。
2. 理解每个构造器参数的数学含义与调参方向。
3. 理解 `get_depth()`、`get_n_leaves()`、`feature_importances_` 在当前源码中的作用。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(X_train, y_train, ...)` | 函数 | 构建并训练 `DecisionTreeClassifier`，返回已训练模型 |
| `DecisionTreeClassifier(...)` | 构造器 | 创建分类决策树，通过超参数控制树的生长与复杂度 |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上递归学习划分规则 |
| `model.get_depth()` | 方法 | 返回树的实际深度 $d$ |
| `model.get_n_leaves()` | 方法 | 返回叶子节点数量 $\vert T\vert$ |
| `model.feature_importances_` | 属性 | 返回特征重要性分数，基于不纯度下降加权求和 |

## 1. `train_model(...)` 的函数签名

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 训练特征矩阵，形状 `(n_samples, n_features)`。传入 `model.fit()`。每行为一个样本，每列为一个特征 | `X_train.values` |
| `y_train` | `array_like` | 训练标签向量，形状 `(n_samples,)`。多分类标签取值为 `{0, 1, ..., K-1}` | `y_train.values` |
| `max_depth` | `int` | 树的最大深度 $d_{\max}$。当前默认 `6`——限制划分轮数防止过拟合 | `3`、`6`、`None` |
| `min_samples_split` | `int` | 内部节点继续分裂所需最小样本数。当前默认 `4` | `2`、`4`、`10` |
| `min_samples_leaf` | `int` | 叶节点最少样本数。当前默认 `2`——保证每个叶节点至少含 2 个样本 | `1`、`2`、`5` |
| `criterion` | `str` | 不纯度度量：`"gini"` = $1 - \sum p_k^2$，`"entropy"` = $-\sum p_k \log_2 p_k$。当前默认 `"gini"` | `"gini"`、`"entropy"` |
| `random_state` | `int` | 随机种子，保证分裂中的随机性可复现。默认为 `None` | `42` |
| 返回值 | `DecisionTreeClassifier` | 已训练完成的模型对象，含 `classes_`、`feature_importances_` 等属性 | — |

### 示例代码

```python
from model_training.classification.decision_tree import train_model

model = train_model(X_train.values, y_train.values)
```

### 理解重点

- 当前训练入口很直接，只负责训练一个 `DecisionTreeClassifier` 模型。
- 和部分实验型代码不同，这里没有剪枝调参逻辑，也没有多模型对比。
- 所有默认超参数都写在函数签名里，阅读成本较低，适合作为源码入口。

## 2. `DecisionTreeClassifier(...)` 的完整参数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `criterion` | `str` | 划分不纯度度量函数。`"gini"` = $\text{Gini}(D) = 1 - \sum p_k^2$；`"entropy"` = $H(D) = -\sum p_k \log_2 p_k$；`"log_loss"` 等同于 `"entropy"`。默认为 `"gini"` | `"gini"`、`"entropy"` |
| `splitter` | `str` | 划分策略。`"best"` 在所有特征中选最优划分点；`"random"` 随机选最优划分点。默认为 `"best"` | `"best"`、`"random"` |
| `max_depth` | `int` 或 `None` | 树的最大深度 $d_{\max}$。`None` 表示不限制，节点持续分裂直到纯净。增大 → 更复杂，过拟合风险更高。默认为 `None` | `3`、`6`、`None` |
| `min_samples_split` | `int` 或 `float` | 内部节点再划分所需最小样本数。`float`（如 `0.1`）表示比例 × `n_samples`。增大可抑制过拟合。默认为 `2` | `2`、`4`、`10` |
| `min_samples_leaf` | `int` 或 `float` | 叶节点最少样本数。`float` 表示比例。增大使树更保守，防止叶节点包含极少数样本。默认为 `1` | `1`、`2`、`5` |
| `min_weight_fraction_leaf` | `float` | 叶节点最小加权样本比例。仅在 `sample_weight` 非 `None` 时有效。默认为 `0.0` | `0.0`、`0.01` |
| `max_features` | `int`、`float`、`str` 或 `None` | 每次分裂考虑的候选特征数。`None` 用全部；`"sqrt"` = $\sqrt{d}$；`"log2"` = $\log_2 d$；`int` = 固定数量。默认为 `None` | `None`、`"sqrt"`、`2` |
| `random_state` | `int` | 随机种子，保证分裂和特征选择中的随机性可复现。默认为 `None` | `42` |
| `max_leaf_nodes` | `int` 或 `None` | 最大叶节点数限制。以最佳优先方式生长，达到限制后停止。`None` 不限制。默认为 `None` | `None`、`10`、`50` |
| `min_impurity_decrease` | `float` | 最小不纯度下降阈值。分裂必须使不纯度下降 ≥ 该值才被接受。默认为 `0.0` | `0.0`、`0.01` |
| `class_weight` | `str`、`dict` 或 `None` | 类别权重。`"balanced"` 自动按 $w_k = n / (K \cdot n_k)$ 加权；`dict` 手动指定 `{class: weight}`。默认为 `None` | `None`、`"balanced"`、`{0:1, 1:2}` |
| `ccp_alpha` | `float` | 代价复杂度剪枝参数 $\alpha$。最小化 $R(T) + \alpha \cdot \vert T\vert$。$\alpha > 0$ 时进行后剪枝。默认为 `0.0` | `0.0`、`0.01` |

### 示例代码

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    criterion="gini",
    random_state=42,
)
model.fit(X_train, y_train)
```

### 理解重点

- 仓库没有自己实现树分裂算法，而是直接调用 scikit-learn 的成熟 CART 实现。
- 当前封装的重点不是重写算法，而是把超参数、训练耗时和关键结果日志组织清楚。
- 最值得关注的是复杂度控制三参数：`max_depth`、`min_samples_split`、`min_samples_leaf`。

## 3. 训练完成后最重要的模型属性

### 属性表

| 属性 | 类型 | 数学含义 |
|---|---|---|
| `classes_` | `ndarray` | 模型学到的类别标签数组，形状 `(n_classes,)` |
| `n_classes_` | `int` | 类别数量 $K$ |
| `n_features_in_` | `int` | 训练时的特征维度 $d$ |
| `feature_importances_` | `ndarray` | 各特征重要性分数，基于不纯度下降加权求和，和为 1 |
| `tree_` | `Tree` | 底层 Cython Tree 对象，包含分裂阈值、子节点索引等内部结构 |
| `max_depth_` | `int` | 通过 `get_depth()` 获取的实际树深 |
| `n_leaves_` | `int` | 通过 `get_n_leaves()` 获取的实际叶节点数 |

### 示例代码

```python
print(f"实际深度: {model.get_depth()}")
print(f"叶节点数: {model.get_n_leaves()}")
print(f"特征重要性: {model.feature_importances_}")
```

### 理解重点

- `get_depth()` 和 `get_n_leaves()` 是当前决策树分册最值得关注的训练结果——它们把"树复杂度"映射成可直接观察的输出。
- `feature_importances_` 是后续特征重要性图的直接数据来源。
- `get_depth()` 返回的实际深度 ≤ `max_depth`。

## 4. 训练阶段的工程封装

除了 `DecisionTreeClassifier(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

| 输出项 | 作用 |
|---|---|
| 函数调用标题（`@print_func_info`） | 帮助在终端中定位训练入口 |
| 训练耗时（`@timeit`） | 观察当前模型拟合时间 |
| 深度与叶节点日志 | 帮助理解树的复杂度 |
| 划分标准日志 | 确认当前树使用的 criterion |

### 理解重点

- 当前封装强调的是教学型可读性，而不是复杂训练框架。
- 这一层封装把"构建模型""训练模型""打印结果"收在一个函数里，方便文档和流水线复用。
- 从工程角度看，这样的拆分也让 `pipelines/classification/decision_tree.py` 保持简洁。

## 模型可视化

![树结构](https://img.yumeko.site/file/blog/articles/1780736277845.png)

## 常见坑

1. 把决策树的 `fit(...)` 理解成和线性模型一样的参数优化过程——树是递归划分，不是梯度下降。
2. 只知道可以 `predict(...)`，却忽略 `get_depth()`、`get_n_leaves()`、`feature_importances_` 才是理解树行为的重要线索。
3. 忘记当前 `X_train` 直接使用原始特征值，而不是标准化后的特征。
4. 把训练函数和后续 ROC、特征重要性、学习曲线等评估逻辑混在一起理解。

## 小结

- `train_model(...)` 是本仓库 Decision Tree 的核心训练入口，本质是对 `sklearn.tree.DecisionTreeClassifier` 的薄封装。
- `DecisionTreeClassifier` 的 12 个构造器参数中，`criterion`、`max_depth`、`min_samples_split`、`min_samples_leaf` 是最核心的四个。
- 训练后属性 `feature_importances_`、`get_depth()`、`get_n_leaves()` 是后续评估与解释的直接数据来源。
- 读懂这一层之后，再看流水线中的概率输出、特征重要性和学习曲线会更顺畅。

# 训练与预测


## 本章目标

1. 按源码顺序看清当前 Decision Tree 流水线到底执行了哪些步骤。
2. 理解训练集/测试集拆分、训练、类别预测和概率预测之间的连接关系。
3. 理解主模型与二维可视化模型在当前实现中的职责差异。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `decision_tree_classification_data.copy()` | 方法 | 复制原始数据，避免修改源对象 |
| `train_test_split(...)` | 方法 | 划分训练集与测试集 |
| `train_model(X_train.values, y_train.values)` | 函数 | 训练主分类树模型 |
| `model.predict(X_test.values)` | 方法 | 生成测试集类别预测结果 |
| `model.predict_proba(X_test.values)` | 方法 | 生成测试集类别概率输出 |
| `PCA(n_components=2)` | 类 | 为决策边界可视化构造二维表示 |
| `model_2d` | 模型 | 专门用于二维决策边界展示 |

## 1. 流水线从复制数据开始

当前流水线先复制 `decision_tree_classification_data`，再拆出 `X`、`y` 和 `feature_names`。

### 示例代码

```python
data = decision_tree_classification_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
feature_names = list(X.columns)
```

### 理解重点

- `feature_names` 会在后续特征重要性图中使用，因此流水线较早就把它保存下来。
- 当前任务是监督多分类，因此 `y` 会真实参与训练和预测评估。

## 2. 先切分训练集与测试集

当前流水线使用 `train_test_split` 将数据按 8:2 比例切分，并通过 `stratify=y` 保持类别分布一致。

### 参数速览

适用函数：`sklearn.model_selection.train_test_split`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `*arrays` | `array_like` | 待切分的数据序列。传入 `(X, y)` 则分别对应切分。长度必须一致 | `X, y` |
| `test_size` | `float` 或 `int` | 测试集占比（`0.0`~`1.0`）或绝对样本数。当前取 `0.2`，即 80 个测试样本（总样本 400 × 20%） | `0.2`、`0.3`、`100` |
| `train_size` | `float` 或 `int` | 训练集占比或绝对样本数。默认 `1 - test_size`，通常不显式指定 | `0.8`、`None` |
| `random_state` | `int` | 随机种子，保证每次切分结果一致。当前取 `42` | `42` |
| `shuffle` | `bool` | 切分前是否打乱数据。默认为 `True`，确保样本顺序不引入偏差 | `True` |
| `stratify` | `array_like` | 按此数组类别比例分层抽样。传入 `y` 确保训练集和测试集各类别比例与原始数据一致，数学上保证 $\frac{n_{k,\text{train}}}{n_{k,\text{test}}} \approx \frac{N_{\text{train}}}{N_{\text{test}}}$，避免小类别在某一集合中意外缺失 | `y`、`None` |

### 示例代码

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- `stratify=y` 的作用，是让训练集和测试集保持相近的类别比例。
- 对当前 4 分类任务（每类约 100 样本）来说，stratify 保证每类约有 80 个训练样本、20 个测试样本。

## 3. 主模型训练与正式预测

当前决策树主流程没有显式标准化步骤，而是直接把原始数值特征传入模型。训练完成后，
`model.predict(...)` 为每个测试样本输出一个类别标签。

### 参数速览

适用方法：`DecisionTreeClassifier.predict(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的特征矩阵。特征维度必须与训练时一致，即 `n_features = d = 2` | `X_test.values` |
| 返回值 | `ndarray`，形状 `(n_samples,)` | 预测类别标签，取值 $\hat{y}_i \in \{0, 1, \dots, K-1\}$，其中 $K = 4$。每个样本被分配到对应叶节点中样本数最多的类别 | — |

### 示例代码

```python
model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
```

### 理解重点

- `model` 是当前分册的主模型，用于正式训练和测试集类别预测。
- 决策树通过阈值切分特征空间（如 $x_1 \leq 3.2$），不依赖欧氏距离或梯度优化，因此不像 KNN、SVC 那样强依赖标准化。
- `y_pred` 是后续混淆矩阵评估的直接输入。

## 4. 概率输出如何进入流水线

`predict_proba(...)` 给出每个测试样本在各个类别上的概率估计，是 ROC 曲线可视化的直接输入。

### 参数速览

适用方法：`DecisionTreeClassifier.predict_proba(X)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like`，形状 `(n_samples, n_features)` | 待预测的特征矩阵。特征维度必须与训练时一致 | `X_test.values` |
| 返回值 | `ndarray`，形状 `(n_samples, n_classes)` | 各类别概率估计，数学上 $P(\hat{y}_i = k \mid \mathbf{x}_i) = \frac{n_k^{\text{leaf}}}{n^{\text{leaf}}}$，即样本落入的叶节点中各类别样本占比。每行和为 1 | — |

### 示例代码

```python
y_scores = model.predict_proba(X_test.values)
```

### 理解重点

- 树模型的概率输出基于叶节点内各类别样本占比，这不同于逻辑回归通过 sigmoid/softmax 映射得分到概率。
- 当前任务是多分类（$K=4$），因此 `y_scores.shape = (80, 4)`，后续 ROC 模块按 One-vs-Rest 方式处理这些概率。
- 如果 `max_depth` 设得过深导致叶节点样本极少，概率估计会变得极不稳定（接近 0 或 1）。

## 5. 特征重要性如何进入流水线

树模型在分裂过程中天然累积特征重要性分数——每次分裂带来的不纯度下降按样本量加权后归到对应特征上。

### 参数速览

适用函数：`result_visualization.feature_importance.plot_feature_importance`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `DecisionTreeClassifier` | 已训练的主决策树模型，提供 `feature_importances_` 属性。其数学含义为 $\text{imp}_j = \frac{\sum_{t \in T_j} \Delta I(t) \cdot n_t}{\sum_{t \in T} \Delta I(t) \cdot n_t}$，其中 $\Delta I(t)$ 是节点 $t$ 的不纯度下降量，$n_t$ 是该节点样本数，$T_j$ 是使用特征 $j$ 分裂的所有节点集合 | `model` |
| `feature_names` | `list[str]` | 特征名列表，长度 = `n_features_in_`。用于图中标注横轴标签 | `['x1', 'x2']` |
| `title` | `str` | 图表标题 | `"决策树 特征重要性"` |
| `dataset_name` | `str` | 数据集名称，用于输出文件名 | `DATASET` |
| `model_name` | `str` | 模型名称，用于输出文件名 | `MODEL` |

### 示例代码

```python
plot_feature_importance(
    model,
    feature_names=feature_names,
    title="决策树 特征重要性",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- `feature_names` 与 `feature_importances_` 的组合，可以把抽象的树分裂信息转成直观的解释图。
- 这是当前分册区别于很多其他分类分册（如逻辑回归）的重要评估入口——树模型天然具备特征重要性解释能力。

## 6. 决策边界为什么要额外训练一个 model_2d

主模型在原始二维特征空间训练，但决策边界图需要支持任意网格点的预测。
当前实现采用 PCA 将特征投影到二维空间，再单独训练一个二维决策树模型用于可视化。

### 参数速览

适用类：`sklearn.decomposition.PCA`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_components` | `int` | 保留的主成分数 $k$。$k=2$ 时将 $d$ 维特征投影到二维平面，便于可视化。数学上 PCA 通过 SVD 分解 $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$，取前 $k$ 个奇异向量构成投影矩阵 $\mathbf{V}_k$ | `2`、`3`、`None` |
| `random_state` | `int` | 随机种子。PCA 本身是确定性的（基于 SVD），但某些求解器使用随机化算法时需要。当前取 `42` | `42` |

### 示例代码

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.values)
model_2d = DecisionTreeClassifier(max_depth=6, random_state=42)
model_2d.fit(pca.transform(X_train.values), y_train.values)
```

### 理解重点

- 这里的 `model_2d` 不是主评估模型，而是专门为二维可视化服务的辅助模型。
- 主模型训练在原始特征空间中，而决策边界图需要二维输入来对每个网格点做预测。
- 这是整个决策树分册里需要重点讲清的工程细节——`model` 和 `model_2d` 的职责不同，不能混淆。

## 7. 学习曲线如何接入流水线

学习曲线用于诊断模型性能是否随训练样本量增加而持续改善。

### 参数速览

适用函数：`result_visualization.learning_curve.plot_learning_curve`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的模型实例。传入 `DecisionTreeClassifier(max_depth=6, random_state=42)`，内部会克隆并逐段训练，不修改传入实例 | `DecisionTreeClassifier(...)` |
| `X` | `array_like` | 训练特征矩阵。当前传入 `X_train.values`，学习曲线内部会按不同比例采样 | `X_train.values` |
| `y` | `array_like` | 训练标签向量 | `y_train.values` |
| `scoring` | `str` | 评分类指标。`"accuracy"` 即 $\frac{\text{正确预测样本数}}{\text{总样本数}}$。默认为 `None`（使用 estimator 默认 score） | `"accuracy"`、`"f1_macro"` |
| `cv` | `int` | 交叉验证折数。默认 `5`，即 5 折交叉验证计算验证得分 | `5`、`10` |

### 示例代码

```python
plot_learning_curve(
    DecisionTreeClassifier(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    title="决策树 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 学习曲线使用的是一个新的 `DecisionTreeClassifier(...)` 实例，而不是直接复用 `model`。
- 这是因为学习曲线函数内部会自行克隆和重复训练模型（`sklearn.model_selection.learning_curve` 的默认行为）。
- 文档需要把"主模型用于正式预测"和"新模型实例用于曲线诊断"区分清楚。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/blog/articles/1780736299374.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 混为一谈——前者返回标签，后者返回概率。
2. 把特征重要性图看成与训练主流程无关的附加内容——它直接来自 `model.feature_importances_`。
3. 把 `model_2d` 误认为正式预测模型本体——它仅是二维可视化辅助模型。
4. 混淆主模型预测、二维可视化模型和学习曲线模型三者的职责——三者共享相同的 `max_depth=6` 超参数，但分别在原始特征空间（`model`）、PCA 空间（`model_2d`）和交叉验证循环（学习曲线）中运行。

## 小结

- 当前 Decision Tree 流水线的训练过程：复制数据 → 切分 → 训练主模型 → 类别预测 → 概率预测 → 特征重要性分析 → 多种可视化诊断。
- 对本仓库而言，`model`、`model_2d` 和学习曲线中的新模型实例分别承担不同职责。
- 关键数学关系：`predict_proba` 输出基于叶节点内类别占比 $n_k^{\text{leaf}} / n^{\text{leaf}}$；`feature_importances_` 基于不纯度下降加权求和 $\sum \Delta I(t) \cdot n_t$。

# 评估与诊断


## 本章目标

1. 明确当前仓库 Decision Tree 实现实际上是如何做结果诊断的。
2. 理解混淆矩阵、ROC 曲线、特征重要性图、PCA 决策边界图和学习曲线分别能说明什么。
3. 理解多分类 ROC、特征重要性和二维决策边界图的展示边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出 |
| `y_scores` | 预测概率 | 测试集各类别概率输出 |
| `plot_confusion_matrix(...)` | 函数 | 绘制预测标签与真实标签的混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 One-vs-Rest ROC 曲线 |
| `plot_feature_importance(...)` | 函数 | 绘制树模型特征重要性图 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |

## 1. 当前仓库的评估入口

当前 Decision Tree 流水线里的主要结果诊断手段有五个：

1. 混淆矩阵
2. ROC 曲线
3. 特征重要性图
4. PCA 2D 决策边界图
5. 学习曲线

### 示例代码

```python
y_pred = model.predict(X_test.values)
y_scores = model.predict_proba(X_test.values)

plot_confusion_matrix(...)
plot_roc_curve(...)
plot_feature_importance(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 当前实现没有把所有诊断都压缩成一个数字，而是同时提供结果矩阵、概率曲线、重要性图、边界图和曲线图五类视角。
- 五种可视化分别回答的是不同问题：
  - 混淆矩阵 → 分对了吗？分错了哪几类？
  - ROC 曲线 → 概率区分能力如何？
  - 特征重要性 → 树更依赖哪些特征？
  - 决策边界图 → 树的划分区域长什么样？
  - 学习曲线 → 更多训练数据有无帮助？

## 2. 混淆矩阵能观察什么

混淆矩阵 $\mathbf{C}$ 是一个 $K \times K$ 矩阵，其中 $C_{ij}$ 表示真实类别为 $i$、预测类别为 $j$ 的样本数。数学上：

$$
C_{ij} = \sum_{k=1}^{n_{\text{test}}} \mathbb{1}[y_k = i \land \hat{y}_k = j]
$$

从混淆矩阵可导出常用指标：准确率 $\text{Accuracy} = \frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}$，各类别精确率 $\text{Precision}_i = \frac{C_{ii}}{\sum_j C_{ji}}$，召回率 $\text{Recall}_i = \frac{C_{ii}}{\sum_j C_{ij}}$。

### 参数速览

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $y_i \in \{0, 1, \dots, K-1\}$，$K=4$ | `y_test` |
| `y_pred` | `array_like`，形状 `(n_samples,)` | 模型预测标签，取值 $\hat{y}_i \in \{0, 1, \dots, K-1\}$，来自 `model.predict(X_test.values)` | `y_pred` |
| `class_names` | `list[str]` | 类别显示名列表，长度 = $K$。默认为 `None`，使用类别标签自身 | `['0', '1', '2', '3']` |
| `normalize` | `bool` 或 `str` | 归一化方式：`True`/`'true'` 按行（真实类别），`'pred'` 按列（预测类别），`'all'` 按全体。默认为 `False`（绝对数量） | `True`、`'true'` |

### 示例代码

```python
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    title="决策树 混淆矩阵",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 混淆矩阵最适合回答：模型把哪些类别分对了，哪些类别更容易互相混淆。
- 对当前 4 分类任务来说，对角线元素 $C_{ii}$ 就是各类别的正确预测数。
- 当前流水线没有显式打印 accuracy，但混淆矩阵已能给出误差结构信息。

## 3. ROC 曲线能观察什么

ROC（Receiver Operating Characteristic）曲线绘制真正例率（TPR）随假正例率（FPR）变化的轨迹。多分类场景下按 One-vs-Rest 方式为每个类别分别计算：

$$
\text{TPR}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad
\text{FPR}_k = \frac{\text{FP}_k}{\text{FP}_k + \text{TN}_k}
$$

AUC（Area Under Curve）是 ROC 曲线下面积，$\text{AUC} \in [0.5, 1.0]$，越接近 1 表示区分能力越强。

### 参数速览

适用函数：`plot_roc_curve(y_test, y_scores, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，取值 $y_i \in \{0, 1, \dots, K-1\}$ | `y_test` |
| `y_scores` | `array_like`，形状 `(n_samples, n_classes)` | 各类别概率估计，来自 `model.predict_proba(X_test.values)`。第 $k$ 列作为类别 $k$ 的 One-vs-Rest 得分 | `y_scores` |
| `class_names` | `list[str]` | 类别显示名列表，长度 = $K$。用于图例标注每条 ROC 曲线对应的类别 | `['0', '1', '2', '3']` |
| `multi_class` | `str` | 多分类策略：`"ovr"`（One-vs-Rest，当前使用），`"ovo"`（One-vs-One）。默认为 `"ovr"` | `"ovr"`、`"ovo"` |

### 示例代码

```python
plot_roc_curve(
    y_test,
    y_scores,
    class_names=["0", "1", "2", "3"],
    title="决策树 ROC 曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 当前任务是多分类，因此 ROC 曲线按 One-vs-Rest 方式分别计算每个类别的 TPR/FPR。
- 这也是为什么需要强调 `predict_proba(...)`——没有概率输出就无法生成连续变化阈值的 ROC 曲线。
- 文档要明确：这里不是一条全局 ROC 曲线，而是每个类别各有一条对其余类别的区分曲线（共 4 条）。

## 4. 特征重要性能观察什么

特征重要性 $\text{imp}_j$ 反映特征 $j$ 在树分裂过程中带来的不纯度下降贡献：

$$
\text{imp}_j = \frac{\sum_{t \in T_j} \Delta I(t) \cdot n_t}{\sum_{t \in T} \Delta I(t) \cdot n_t}
$$

其中 $\Delta I(t) = I(t) - \sum_{c \in \text{children}(t)} \frac{n_c}{n_t} I(c)$ 是节点 $t$ 的不纯度下降量（基于 `criterion` 度量），$n_t$ 是该节点样本数，$T_j$ 是使用特征 $j$ 分裂的所有节点集合。所有特征重要性之和为 1。

### 参数速览

适用函数：`plot_feature_importance(model, feature_names, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `DecisionTreeClassifier` | 已训练的主决策树模型，提供 `feature_importances_` 属性。值来自训练时累积的不纯度下降加权和 | `model` |
| `feature_names` | `list[str]` | 特征名列表，长度 = `n_features_in_`。用于图中标注横轴标签 | `['x1', 'x2']` |
| `title` | `str` | 图表标题 | `"决策树 特征重要性"` |
| `dataset_name` | `str` | 数据集名称，用于输出文件命名 | `DATASET` |
| `model_name` | `str` | 模型名称，用于输出文件命名 | `MODEL` |

### 示例代码

```python
plot_feature_importance(
    model,
    feature_names=feature_names,
    title="决策树 特征重要性",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 特征重要性图回答：当前树在划分过程中，更依赖哪些特征。
- 但需要注意：特征重要性表示的是"当前树分裂时的贡献"，不等于严格因果关系。
- 对当前二维 blob 数据（`x1`、`x2` 各向同性生成），两个特征的重要性通常接近，差异不大。

## 5. PCA 2D 决策边界图能观察什么

决策边界图在二维平面上以不同颜色填充不同预测区域，直观展示树模型的区域切分形状。

### 参数速览

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model_2d` | `DecisionTreeClassifier` | 在 PCA 二维空间上单独训练的决策树模型。`max_depth=6`，与主模型共享相同超参数，仅在特征空间维度上不同 | `model_2d` |
| `X_2d` | `ndarray`，形状 `(n_samples, 2)` | PCA 投影后的二维特征，用于生成网格点预测和散点着色。列分别为 PC1、PC2 | `X_2d` |
| `y` | `array_like`，形状 `(n_samples,)` | 全量标签数组，用于散点的真实类别着色 | `y.values` |
| `title` | `str` | 图表标题 | `"决策树 决策边界 (PCA 2D)"` |

### 示例代码

```python
plot_decision_boundary(
    model_2d,
    X_2d,
    y.values,
    title="决策树 决策边界 (PCA 2D)",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 这张图最适合直观感受决策树的轴对齐切分——边界通常呈现垂直线段和平行线段的组合（如 $x_1 \leq 3.2$），而不是平滑曲线或对角线。
- 但它只是 PCA 投影空间中的近似展示，不是原始高维划分结构的完整真相。
- 当原始特征维度 $d > 2$ 时，PCA 投影会丢失部分划分信息。

## 6. 学习曲线能观察什么

学习曲线绘制训练得分和交叉验证得分随训练样本数增加的变化：

- 训练得分高、验证得分低且差距大 → 过拟合倾向
- 训练得分和验证得分都低且接近 → 欠拟合倾向
- 验证得分持续上升且未收敛 → 更多数据可能有帮助

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `estimator` | 新创建的模型实例。传入 `DecisionTreeClassifier(max_depth=6, random_state=42)`，内部克隆后逐段训练。使用新实例是为了避免干扰主模型 | `DecisionTreeClassifier(max_depth=6, random_state=42)` |
| `X` | `array_like`，形状 `(n_samples, n_features)` | 训练特征矩阵。学习曲线内部按不同比例（如 10%, 20%, ..., 100%）逐步增加样本量 | `X_train.values` |
| `y` | `array_like`，形状 `(n_samples,)` | 训练标签向量 | `y_train.values` |
| `scoring` | `str` | 评分类指标。当前取 `"accuracy"`，即 $\frac{\sum \mathbb{1}[y_i = \hat{y}_i]}{n}$。`"f1_macro"` 为各类别 F1 的算术平均 | `"accuracy"`、`"f1_macro"` |
| `cv` | `int` | 交叉验证折数。默认 `5`，每次对当前采样量做 5 折 CV 计算验证得分误差带 | `5`、`10` |
| `train_sizes` | `array_like` | 训练样本量的递增序列。默认为 `np.linspace(0.1, 1.0, 5)` | `[0.1, 0.33, 0.55, 0.78, 1.0]` |
| `n_jobs` | `int` | 并行作业数。`-1` 使用全部核心。默认为 `None`（单核） | `-1`、`1` |

### 示例代码

```python
plot_learning_curve(
    DecisionTreeClassifier(max_depth=6, random_state=42),
    X_train.values,
    y_train.values,
    title="决策树 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 对决策树而言，学习曲线尤其有助于观察 `max_depth=6` 受限时模型的泛化行为。
- 如果训练得分很高（接近 1.0）但验证得分明显偏低，说明当前树深仍可能导致过拟合。
- 验证得分误差带（CV 标准差）也能提示模型在不同数据划分下的稳定性。

## 7. 当前实现中尚未纳入但常见的分类指标

在一般分类任务中，还常见以下指标，数学定义如下：

| 指标 | 公式 | 说明 |
|---|---|---|
| 准确率（Accuracy） | $\frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}$ | 整体正确率，各类别样本数不均衡时可能误导 |
| 精确率（Precision） | $\frac{C_{ii}}{\sum_j C_{ji}}$ | 预测为类别 $i$ 的样本中有多少确实是 $i$ |
| 召回率（Recall） | $\frac{C_{ii}}{\sum_j C_{ij}}$ | 真实类别 $i$ 的样本中有多少被正确找出 |
| F1 分数 | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Precision 和 Recall 的调和平均 |

### 理解重点

- 当前仓库没有在 Decision Tree 流水线中显式打印这些指标，而是通过混淆矩阵隐式呈现。
- 文档可以提到它们是常见扩展方向，但不能写成"当前源码已经在单独计算"。

## 评估图表

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

![ROC 曲线](https://img.yumeko.site/file/blog/articles/1780736321096.png)

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 的用途混为一谈——前者用于混淆矩阵，后者用于 ROC。
2. 把特征重要性图误解为严格因果解释——它反映的是树的分裂选择，不是特征的真实因果贡献。
3. 把 PCA 决策边界图误认为原始特征空间划分结构的完整表达——它只是二维投影近似。
4. 把当前仓库未实现的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 Decision Tree 的评估方式：混淆矩阵看错误分布，ROC 曲线看概率区分能力，特征重要性图看解释性，PCA 决策边界图看边界形状，学习曲线看训练行为。
- 核心数学：混淆矩阵 $C_{ij} = \sum \mathbb{1}[y_k=i \land \hat{y}_k=j]$，ROC/TPR/FPR 通过阈值扫描得到，特征重要性基于 $\sum \Delta I(t) \cdot n_t$。
- 五者组合起来，比单一指标更能解释当前分类树模型的实际表现。

# 工程实现


## 本章目标

1. 从工程角度看清 Decision Tree 在本仓库中的完整调用链。
2. 理解数据生成、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解为什么当前实现要把训练逻辑、正式预测逻辑、概率输出逻辑和可视化逻辑拆开。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.decision_tree()` 生成二维多分类 blob 数据 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `decision_tree_classification_data` |
| 训练封装 | `model_training/classification/decision_tree.py` | 构建并训练 `DecisionTreeClassifier`，打印训练日志 |
| 流水线入口 | `pipelines/classification/decision_tree.py` | 组织切分、训练、预测与可视化评估的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 保存混淆矩阵图 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 保存多分类 One-vs-Rest ROC 曲线图 |
| 特征重要性可视化 | `result_visualization/feature_importance.py` | 保存特征重要性图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 保存学习曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.decision_tree
```

### 理解重点

- 对大多数读者来说，这个命令是理解当前 Decision Tree 工程实现的最佳入口。
- 它会依次完成数据读取、特征准备、模型训练、测试集预测、概率输出和结果绘图。
- 如果只读一个文件，建议先读 `pipelines/classification/decision_tree.py`——它是整个决策树流程的编排层。

## 2. run() 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格，每一步的输入明确来自上一步的输出。

### 核心逻辑

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = decision_tree_classification_data.copy()
    X = data.drop(columns=["label"])
    y = data["label"]
    feature_names = list(X.columns)

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 训练主模型 & 正式预测
    model = train_model(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    y_scores = model.predict_proba(X_test.values)

    # 4. 特征重要性
    plot_feature_importance(model, feature_names=feature_names, ...)

    # 5. 可视化诊断（混淆矩阵、ROC、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_roc_curve(y_test, y_scores, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(DecisionTreeClassifier(...), X_train.values, y_train.values, ...)
```

### 理解重点

- `run()` 本身没有复杂算法，它的职责是把不同模块串起来。
- 这类文件更像"编排层"（orchestrator），重点是流程顺序正确、调用关系清楚。
- 每一步的数据流向都是单向的：上一步的输出直接作为下一步的输入，没有循环依赖。

## 3. 训练模块负责什么

`model_training/classification/decision_tree.py` 里的 `train_model(...)` 是训练逻辑的封装层。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(n_samples, n_features)` | 训练特征矩阵。将原样传入 `DecisionTreeClassifier.fit()` | `X_train.values` |
| `y_train` | `array_like`，形状 `(n_samples,)` | 训练标签向量，取值 $\{0, 1, \dots, K-1\}$ | `y_train.values` |
| `max_depth` | `int` | 树的最大深度 $d_{\max}$，限制划分轮数。默认 `6` | `3`、`6`、`None` |
| `min_samples_split` | `int` | 内部节点继续分裂所需最小样本数。默认 `4` | `2`、`4`、`10` |
| `min_samples_leaf` | `int` | 叶节点最少样本数。默认 `2` | `1`、`2`、`5` |
| `criterion` | `str` | 不纯度度量：`"gini"` = $1 - \sum p_k^2$，`"entropy"` = $-\sum p_k \log_2 p_k$。默认 `"gini"` | `"gini"`、`"entropy"` |
| `random_state` | `int` | 随机种子，保证分裂中的随机性可复现。默认 `None` | `42` |
| 返回值 | `DecisionTreeClassifier` | 已训练完成的主模型对象，含 `classes_`、`feature_importances_`、`tree_` 等属性 | — |

### 职责清单

`train_model(...)` 主要负责四件事：

1. 创建 `DecisionTreeClassifier(...)` 实例（按给定超参数）
2. 调用 `model.fit(X_train, y_train)` 执行 CART 递归划分
3. 打印训练日志：训练耗时、实际深度、叶节点数、criterion
4. 返回训练完成的主模型对象

### 理解重点

- 这层抽离让"模型训练逻辑"和"业务流程编排逻辑"分开。
- 训练函数既可以被流水线调用，也可以单独运行做局部验证（模块级 `if __name__ == "__main__"` 自测）。
- 这也是当前仓库多个算法分册共享的组织方式：`model_training/classification/*.py` 各封装一个算法的训练逻辑。

## 4. 五类评估模块分别负责什么

当前 Decision Tree 的五种可视化各由一个独立模块负责，每个模块职责单一。

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 混淆矩阵图片（PNG） |
| ROC 曲线 | `plot_roc_curve(...)` | `y_test`、`y_scores` | 多分类 ROC 曲线图片（PNG） |
| 特征重要性 | `plot_feature_importance(...)` | `model`、`feature_names` | 特征重要性条状图（PNG） |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `estimator`、`X_train.values`、`y_train.values` | 训练/验证得分曲线（PNG） |

### 理解重点

- 五类可视化都不是训练的一部分，而是训练完成后的诊断步骤——它们不修改模型参数。
- 特征重要性图是当前分册区别于很多其他分类分册的重要评估入口，因为它直接读取树模型内部结构。
- 决策边界图依赖单独训练的 `model_2d`（在 PCA 空间），ROC 曲线依赖 `predict_proba(...)` 输出的概率矩阵。

## 5. 模块间的数据依赖关系

整个流水线的数据流向可以概括为以下依赖关系：

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `decision_tree_classification_data` | `data_generation/classification.py` | `pipelines/classification/decision_tree.py` |
| `model`（主模型） | `train_model(...)` | `predict`、`predict_proba`、`plot_feature_importance` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `y_scores` | `model.predict_proba(...)` | `plot_roc_curve` |
| `model_2d` | `DecisionTreeClassifier.fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/decision_tree/` 目录 |

### 理解重点

- 数据流向是单向的：数据生成 → 训练 → 预测 → 评估，各环节之间没有循环依赖。
- 这种清晰的数据流使得每个模块可以独立测试和替换。
- `model_2d` 和主模型 `model` 共享相同的 `max_depth=6` 超参数，但在不同特征空间中训练。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `决策树分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、树深度 $d$、叶节点数 $\vert T\vert$、`criterion` | 理解树复杂度和训练效率 |
| 混淆矩阵图 | `outputs/decision_tree/confusion_matrix.png` | 观察各类别误分类方向 |
| ROC 曲线图 | `outputs/decision_tree/roc_curve.png` | 评估多分类概率区分能力 |
| 特征重要性图 | `outputs/decision_tree/feature_importance.png` | 理解特征在树分裂中的贡献 |
| 决策边界图 | `outputs/decision_tree/decision_boundary.png` | 观察 PCA 2D 空间下的轴对齐切分 |
| 学习曲线图 | `outputs/decision_tree/learning_curve.png` | 诊断过拟合/欠拟合倾向 |

### 理解重点

- 运行结果并不只是一个模型对象，还包括面向阅读者的日志和多种图像产物。
- 对教学仓库而言，这种"代码 + 日志 + 图像"的组合比单纯返回分类结果更易理解。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/decision_tree.py` — 入口，了解整体流程
2. 再看 `model_training/classification/decision_tree.py` — 训练封装，理解超参数和日志
3. 再看 `result_visualization/confusion_matrix.py` — 最基础的分类结果评估
4. 再看 `result_visualization/roc_curve.py` — 概率区分能力评估
5. 再看 `result_visualization/feature_importance.py` — 树模型特有的解释性评估
6. 再看 `result_visualization/decision_boundary.py` — 空间划分可视化
7. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
8. 最后回到 `data_generation/classification.py` — 理解数据生成参数

### 理解重点

- 先从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 如果一开始就只看某一个可视化模块，容易看见局部却看不见完整链路。
- 这个阅读顺序也对应了数据流的方向：数据 → 训练 → 预测 → 评估。

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的训练在 `model_training/` 中。
2. 不区分"主模型""二维可视化模型"和"学习曲线模型实例"的职责边界——三者训练在不同空间或用不同数据子集。
3. 不区分类别预测输出（`predict`）、概率输出（`predict_proba`）和特征重要性（`feature_importances_`）的用途——它们分别服务于不同的评估方式。
4. 只看单个文件，不顺着调用链理解整体执行流程——缺少全局视角容易产生误解。

## 小结

- 当前 Decision Tree 工程实现采用清晰的模块分层：数据生成 → 训练封装 → 流水线编排 → 结果可视化。
- `run()` 负责串联流程，`train_model(...)` 负责训练主模型，各可视化函数负责结果展示与诊断。
- 数据流是单向的，各模块职责单一，便于教学讲解和后续扩展其他算法的同类结构。

# 练习与参考文献


## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Decision Tree 实现。
2. 给出继续深入阅读决策树与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/decision_tree.py` 的主流程里没有显式标准化步骤？
2. 为什么当前 `make_blobs(n_samples=400, centers=4, cluster_std=1.0)` 数据适合决策树的区域切分方式？
3. 当前 `train_model(...)` 中的 `max_depth`、`min_samples_split`、`min_samples_leaf`、`criterion` 分别控制什么？各自的数学含义是什么？
4. 为什么 `model.get_depth()` 与 `model.get_n_leaves()` 对理解树复杂度很重要？两者的关系是什么（理论最大叶节点数 $2^d$）？
5. 为什么特征重要性图对理解树模型有帮助？特征重要性的数学公式是什么？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它和主模型 `model` 在什么特征空间上训练？
7. `predict(...)` 和 `predict_proba(...)` 的返回值有什么区别？各自服务于哪种评估方式？

## 练习方向

### 1. 改动 max_depth

- 把 `max_depth=6` 改成 `None`、`2`、`3`、`10`、`20`
- 观察变化：
  - `model.get_depth()` 的实际深度
  - `model.get_n_leaves()` 的叶节点数（注意与 $2^d$ 的关系）
  - 混淆矩阵中各类别正确/错误分布
  - 学习曲线中训练得分与验证得分的差距
- 思考：树深与泛化误差之间的关系

### 2. 改动 criterion

- 尝试 `criterion="gini"` 与 `criterion="entropy"`
- 对比变化：
  - 树结构（深度、叶节点数）是否不同
  - ROC 曲线 AUC 值的变化
  - 特征重要性的分布变化
- 理解：两种不纯度度量公式不同（$\text{Gini}(D) = 1 - \sum p_k^2$ vs $H(D) = -\sum p_k \log_2 p_k$），但在当前小规模数据上表现通常接近

### 3. 观察 feature_importances_

- 同时查看训练日志（`get_depth()`、`get_n_leaves()`）与特征重要性图
- 对比 `x1` 和 `x2` 在当前树中的贡献差异
- 修改 `make_blobs` 的 `cluster_std`（如从 `1.0` 改为 `2.0`），观察特征重要性的变化

### 4. 与 Logistic Regression 对比

- 对照阅读 `docs/classification/logistic_regression/`
- 比较要点：
  - 决策树的局部规则切分（轴对齐 $x_j \leq \text{threshold}$）vs 逻辑回归的全局线性边界（$\mathbf{w}^T\mathbf{x} + b = 0$）
  - 是否需要标准化：树不需要，逻辑回归需要
  - 评估方式差异：树有特征重要性图，逻辑回归有权重系数解释
- 分别在同一数据上运行两个流水线，对比混淆矩阵和 ROC 曲线

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`DecisionTreeClassifier` | 完整构造器参数列表、属性与方法说明 |
| 2 | scikit-learn 官方文档：`make_blobs` | 数据生成器的参数与使用说明 |
| 3 | scikit-learn 用户指南：Decision Trees | CART 算法原理、复杂度控制与剪枝策略的详细讲解 |
| 4 | Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning*. | 第 9 章：Tree-Based Methods，涵盖 CART、信息增益、代价复杂度剪枝的数学推导 |

- scikit-learn `DecisionTreeClassifier`：https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- scikit-learn `make_blobs`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
- scikit-learn 用户指南 Decision Trees：https://scikit-learn.org/stable/modules/tree.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Decision Tree 分册的核心内容：
  - 为什么主流程不强调标准化（树基于阈值切分，不依赖距离尺度）
  - 树深和叶节点数的意义与数学关系（$d$、$\vert T\vert$、$2^d$）
  - 特征重要性的解释边界（基于不纯度下降加权，不等于因果关系）
  - `model`、`model_2d` 和学习曲线实例的职责差异
