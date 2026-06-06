---
title: GaussianNB 高斯朴素贝叶斯
date: 2026-04-25
category: 机器学习/分类
tags:
  - Scikit-learn
  - 高级教程
description: 高斯朴素贝叶斯的数学原理、条件独立假设与完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581811409.webp
status: published
---

# 数学原理

## 本章目标

1. 理解朴素贝叶斯如何用贝叶斯公式从先验和似然推算后验概率——这是一种生成式分类思路。
2. 理解"朴素"条件独立假设 $P(\mathbf{x}\vert Y) = \prod P(x_j\vert Y)$ 的数学含义、简化效果和现实代价。
3. 理解 `GaussianNB` 对连续特征的高斯建模方式——每个类别每个特征各拟合一个 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 贝叶斯公式 $P(Y\vert\mathbf{x}) \propto P(\mathbf{x}\vert Y)P(Y)$ | 基础公式 | 从先验 $P(Y)$ 和似然 $P(\mathbf{x}\vert Y)$ 推到后验概率 |
| 条件独立假设 $\prod P(x_j\vert Y)$ | 核心简化 | 把高维联合分布拆成单特征条件概率乘积，大幅减少参数 |
| MAP 决策 $\arg\max_c P(c)\prod P(x_j\vert c)$ | 分类规则 | 选择后验概率最大的类别作为预测输出 |
| 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ | 概率模型 | 当前实现中对每类每个连续特征的单高斯建模 |
| `var_smoothing` | 超参数 | 向方差中加入极小平滑项 $\epsilon$，防止 $\sigma_{kj}^2 \to 0$ 时数值崩溃 |
| 类别先验 $P(Y=c_k) = n_k/N$ | 概率项 | 各类别在训练集中的基础比例 |

## 1. 朴素贝叶斯的核心思想

朴素贝叶斯是一种生成式分类器：它不直接学习分类边界，而是先对"每个类别下数据长什么样"建模（$P(\mathbf{x}\vert Y)$），再结合各类别的先验概率（$P(Y)$），通过贝叶斯公式推算后验概率。

### 贝叶斯公式

$$
P(Y = c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid Y = c_k) \, P(Y = c_k)}{P(\mathbf{x})}
$$

其中：
- $P(Y = c_k)$：先验概率——在不知道特征值之前，样本属于类别 $c_k$ 的概率
- $P(\mathbf{x} \mid Y = c_k)$：似然——如果样本属于类别 $c_k$，看到特征 $\mathbf{x}$ 的概率有多大
- $P(Y = c_k \mid \mathbf{x})$：后验概率——在看到特征 $\mathbf{x}$ 后，样本属于类别 $c_k$ 的最终概率
- $P(\mathbf{x})$：证据项——对所有类别相同，分类时通常忽略

### 理解重点

- 朴素贝叶斯是生成式模型：先对 $P(\mathbf{x}, Y)$ 建模，再由贝叶斯公式推出 $P(Y\vert\mathbf{x})$。
- 这与逻辑回归（直接对 $P(Y\vert\mathbf{x})$ 用 Sigmoid 建模）的判别式思路有本质区别——逻辑回归不关心数据是怎么"生成"的。
- 证据项 $P(\mathbf{x})$ 在分类决策时可忽略，因为它对所有类别相同。

## 2. 为什么叫"朴素"：条件独立假设

如果不对似然 $P(\mathbf{x} \mid Y = c_k)$ 做任何简化，需要对 $d$ 维连续特征估计一个完整的 $d$ 维联合分布——这在样本有限时极难做到。

朴素贝叶斯的关键假设是：**在给定类别后，各个特征之间条件独立**。

$$
P(\mathbf{x} \mid Y = c_k) = P(x_1, x_2, \dots, x_d \mid Y = c_k) = \prod_{j=1}^{d} P(x_j \mid Y = c_k)
$$

这个假设让参数估计从 $\mathcal{O}(\text{指数级})$ 降到 $\mathcal{O}(d)$。代价是：如果特征之间存在强依赖关系（如 $x_2 \approx 2x_1$），模型会双倍计算同一条信息，导致概率估计偏差。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别的先验概率 $P(Y=c_k)$。`None` 时从训练数据估计：$P(Y=c_k) = n_k / N$。可手动传入数组来覆盖从数据估计的先验。默认为 `None` | `None`、`[0.3, 0.3, 0.4]` |

### 理解重点

- "朴素"不是指算法粗糙，而是对特征关系做了强简化——假设条件独立。
- 这个假设在真实数据里往往不严格成立，但好处是：联合分布建模的难度大幅下降，参数估计直接、训练极快。
- 即使特征不完全独立，朴素贝叶斯在实际应用中仍常给出不错的效果——特别是在高维文本分类中。

## 3. 分类决策：最大后验概率（MAP）

联合贝叶斯公式和条件独立假设，得到分类规则（对数形式，避免连乘下溢）：

$$
\hat{y} = \arg\max_{c_k} \left[ \ln P(Y = c_k) + \sum_{j=1}^{d} \ln P(x_j \mid Y = c_k) \right]
$$

直观理解：
- 第一项 $\ln P(Y=c_k)$：类别本身有多常见（先验）
- 第二项 $\sum \ln P(x_j\vert Y=c_k)$：当前特征值在类别 $c_k$ 下有多"自然"（似然之和）

两项相加，得分最高的类别就是预测输出。

### 理解重点

- 这是纯粹的代数计算，不涉及梯度下降或迭代优化——因此训练极快。
- 两项之间没有权重超参数调节——模型对先验和似然的信任程度完全由数据决定。
- 对数形式是工程实现的必须——$d$ 个 $(0,1)$ 区间概率连乘会迅速下溢到浮点数零。

## 4. GaussianNB：连续特征的高斯建模

`GaussianNB` 假设：在每个类别 $c_k$ 内，每个连续特征 $x_j$ 服从高斯（正态）分布。

$$
P(x_j \mid Y = c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
$$

其中 $\mu_{kj}$ 和 $\sigma_{kj}^2$ 是从训练数据中该类该特征的最大似然估计：

$$
\mu_{kj} = \frac{1}{n_k} \sum_{i: y_i = c_k} x_{ij}, \quad
\sigma_{kj}^2 = \frac{1}{n_k} \sum_{i: y_i = c_k} (x_{ij} - \mu_{kj})^2
$$

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。实际计算使用 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，其中 $\sigma_{\max}^2$ 是所有特征所有类别中最大的方差。防止方差异常小时 $\frac{1}{\sqrt{2\pi\sigma^2}} \to \infty$ 导致数值问题。默认为 `1e-9` | `1e-9`、`1e-8`、`1e-7` |

### 示例代码

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_s, y_train)
# model.theta_   -> 各类别各特征的均值 mu_kj，形状 (n_classes, n_features)
# model.var_     -> 各类别各特征的方差 sigma_kj^2（应用平滑后的），形状 (n_classes, n_features)
```

### 理解重点

- iris 数据的 4 个特征都是连续值（萼片长宽、花瓣长宽），因此用高斯分布建模是自然选择。
- 高斯似然的参数（$\mu_{kj}$、$\sigma_{kj}^2$）通过简单的统计公式一步得到——不涉及迭代优化。
- 如果某些类别下某些特征的方差极小（所有样本取值几乎相同），$\sigma_{kj}^2 \approx 0$ 会让高斯概率密度值趋于无穷——`var_smoothing` 就是为此设置的数值保险。

## 5. 为什么 GaussianNB 不需要像逻辑回归那样迭代优化

GaussianNB 的参数估计全部是解析解（闭式解）：

- 先验：$P(Y=c_k) = n_k / N$（计数除总数）
- 均值：$\mu_{kj} = \frac{1}{n_k} \sum x_{ij}$（样本均值）
- 方差：$\sigma_{kj}^2 = \frac{1}{n_k} \sum (x_{ij} - \mu_{kj})^2$（样本方差）

没有需要迭代优化的损失函数，没有梯度计算，没有收敛判断。这使得 GaussianNB 的 `fit()` 是所有分类模型中最快的之一。

### 理解重点

- GaussianNB 的训练等价于"统计各类别下各特征的均值和方差"——纯粹的数据扫描。
- 这与逻辑回归（`lbfgs` 迭代优化交叉熵）、KNN（虽然不优化但需建索引）、决策树（递归贪心搜分裂点）在计算特征上有本质区别。
- 代价是：高斯假设 + 条件独立假设在复杂真实数据上可能偏离较大，需要权衡模型假设的合理性与计算效率。

## 6. 数学原理如何映射到当前源码

以下表格将本章涉及的数学概念与当前仓库的代码实现一一对应：

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 贝叶斯公式 | $P(Y\vert\mathbf{x}) = \frac{P(\mathbf{x}\vert Y)P(Y)}{P(\mathbf{x})}$ | `model.predict_proba(X)` 内部 |
| 条件独立假设 | $P(\mathbf{x}\vert Y) = \prod_j P(x_j\vert Y)$ | `GaussianNB` 算法核心假设 |
| MAP 决策 | $\hat{y} = \arg\max_c [\ln P(c) + \sum \ln P(x_j\vert c)]$ | `model.predict(X)` |
| 类别先验 | $P(Y=c_k) = n_k / N$ | `model.class_prior_` |
| 类别样本数 | $n_k$ | `model.class_count_` |
| 高斯均值 | $\mu_{kj}$ | `model.theta_`，形状 `(n_classes, n_features)` |
| 高斯方差 | $\sigma_{kj}^2$ | `model.var_`（平滑后），形状 `(n_classes, n_features)` |
| 方差平滑 | $\sigma^2 + \epsilon \cdot \sigma_{\max}^2$ | `var_smoothing=1e-9` |
| 平滑绝对值 | $\epsilon \cdot \sigma_{\max}^2$ | `model.epsilon_` |
| 类别标签 | $\{c_1, \dots, c_K\}$ | `model.classes_` |

## 常见坑

1. 把朴素贝叶斯理解成"简单版分类器"，而忽略它是生成式概率模型——对 $P(\mathbf{x}\vert Y)$ 建模，而非直接拟合边界。
2. 把"条件独立"误解成特征在原始数据中必须完全独立——这个假设是为计算可行性做的简化，现实中很少严格成立，但模型常常仍然有效。
3. 混淆 GaussianNB（连续特征，高斯似然）与其他朴素贝叶斯变体——MultinomialNB（离散计数）、BernoulliNB（二元特征）、ComplementNB（不平衡文本）。
4. 忽略 `var_smoothing` 的数值稳定作用——特别是当某些类别样本极少、特征方差接近 0 时。

## 小结

- 朴素贝叶斯的核心数学链：贝叶斯公式 $P(Y\vert\mathbf{x}) \propto P(\mathbf{x}\vert Y)P(Y)$ -> 条件独立 $\prod P(x_j\vert Y)$ -> 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ -> MAP 决策（对数形式）-> `var_smoothing` 数值保护。
- 所有参数（先验、均值、方差）都是解析解——不需要迭代优化，训练极快。
- 当前源码使用 `GaussianNB(var_smoothing=1e-9)`，对应连续特征的高斯建模——与 iris 数据的 4 个连续特征天然匹配。

# 数据构成

## 本章目标

1. 明确本仓库 Naive Bayes 数据来自 `load_iris()` 真实数据集。
2. 理解 iris 的 4 个连续特征与 `GaussianNB` 高斯假设之间的天然适配关系。
3. 明确训练集/测试集切分与标准化的顺序和边界。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `load_iris()` | 函数 | 加载 iris 经典多分类真实数据集 |
| `naive_bayes_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `label` | 列名 | 当前流水线中的监督分类标签，取值 $\{0, 1, 2\}$ |
| `train_test_split` | 函数 | 按 `stratify=y` 保持类别比例划分训练/测试集 |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化，统一量纲并利于 PCA 可视化 |

## 1. 数据加载：`load_iris()`

当前 Naive Bayes 数据来自 `ClassificationData.naive_bayes()`，底层调用 `sklearn.datasets.load_iris()`。

### 参数速览

适用函数：`load_iris()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `return_X_y` | `bool` | 是否仅返回 `(X, y)` 元组。默认 `False` 返回 Bunch 对象 | `False` |
| `as_frame` | `bool` | 是否以 DataFrame 形式返回。默认 `False` 返回 ndarray | `False` |
| 样本数 | `int` | iris 数据集固定为 150 个样本，三类鸢尾花各 50 个 | `150` |
| 特征数 | `int` | 4 个连续特征，来自 `iris.feature_names` | `4` |
| 类别数 | `int` | 3 个类别（Setosa / Versicolour / Virginica），标签为 $0, 1, 2$ | `3` |

### 示例代码

```python
from sklearn.datasets import load_iris
from pandas import DataFrame

iris = load_iris()
data = DataFrame(iris.data, columns=iris.feature_names)
data["label"] = iris.target
```

### 理解重点

- iris 是真实经典基准数据集，不是人工合成数据——四个连续特征分别对应萼片长宽和花瓣长宽。
- 4 个特征全为连续值，与 `GaussianNB` 对连续特征的高斯建模假设天然匹配。
- 三类各 50 样本的均衡设计使得类别先验 $P(Y=c_k) \approx 1/3$，无需处理类别不平衡。
- 三分类结构让 ROC 曲线部分需要使用 One-vs-Rest 方式分别绘制。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 含 4 个连续特征的特征矩阵，列名来自 `iris.feature_names` | `data.drop(columns=["label"])` |
| `y` | `Series` | 监督分类标签，取值 $y_i \in \{0, 1, 2\}$ | `data["label"]` |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- 特征列名与 iris 原始特征名称一致，具有明确的物理解释（萼片长宽、花瓣长宽）。
- `label` 是监督训练标签，会真实参与 `model.fit(X_train, y_train)`——与聚类分册不同，标签在这里是训练过程的一部分。
- 将特征和标签明确拆分是后续切分、标准化和训练的前提。

## 3. 训练/测试集切分

### 参数速览

适用函数：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `array_like` | 特征矩阵，形状 $(150, 4)$ | `X` |
| `y` | `array_like` | 标签向量，形状 $(150,)$ | `y` |
| `test_size` | `float` | 测试集占比，默认 `0.2`。150 样本下训练 120 / 测试 30 | `0.2` |
| `random_state` | `int` | 随机种子，保证每次运行切分结果一致。默认 `42` | `42` |
| `stratify` | `array_like` | 分层变量，传入 `y` 时保持训练/测试集类别比例与原始数据一致 | `y` |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- `stratify=y` 确保三类各 50 样本在三分类中训练/测试比例稳定——这对小样本数据集（150 条）尤为重要。
- 标准化必须在切分**之后**执行，否则测试集信息会通过标准化统计量泄露到训练过程中。

## 4. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 训练特征矩阵，形状 $(120, 4)$，用于 `fit_transform`——即计算 $\mu_j, \sigma_j$ 并原地变换 | `X_train` |
| `X_test` | `array_like` | 测试特征矩阵，形状 $(30, 4)$，用训练集统计量 `transform` | `X_test` |
| 返回值 | `ndarray` | 标准化后的特征矩阵 $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$ | `X_train_s`、`X_test_s` |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 虽然 GaussianNB 不依赖梯度优化，但标准化使各特征方差估计更稳定（不受量纲影响），且利于后续 PCA 可视化。
- `fit_transform` 在训练集上同时计算统计量和变换，`transform` 在测试集上使用同一统计量——这是标准工程做法。
- 当前仓库在所有分类流水线中统一保留标准化步骤，便于跨算法分册对比和风格一致。

## 数据可视化

![类别分布](https://img.yumeko.site/file/blog/articles/1780737805661.png)

![相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

![特征空间二维投影](https://img.yumeko.site/file/blog/articles/1780737808100.png)

## 常见坑

1. 忘记把 `label` 从特征表中剥离出来——特征矩阵不能包含标签列。
2. 在切分之前就对全量数据做标准化——这是数据泄露，测试信息混入了训练统计量。
3. 忽略 `stratify=y`——小样本多分类任务中类别比例偏差会显著影响评估结论。
4. 误以为 iris 是人工合成的玩具数据——它是真实经典基准集，四个特征有明确物理含义。

## 小结

- 当前 Naive Bayes 数据来自 `load_iris()`：150 样本、4 个连续特征、3 个均衡类别。
- 数据流为：加载 -> 特征/标签拆分 -> 切分（`stratify=y`）-> 标准化（仅在训练集 `fit`）。
- iris 的连续特征与高斯朴素贝叶斯的 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ 假设天然匹配，是教学场景中的自然选择。

# 思路与直觉

## 本章目标

1. 用直观方式理解 GaussianNB 的生成式分类思路——从"数据怎么生成"出发，而非从"边界画在哪"出发。
2. 理解为什么 iris 的连续特征天然适合高斯假设。
3. 通过与其他分类算法的对比，建立 GaussianNB 在整个分类算法图中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 生成式分类 | 核心直觉 | 先建模 $P(\mathbf{x} \vert Y)$，再通过贝叶斯公式反推 $P(Y \vert \mathbf{x})$ |
| 类别先验 | 基础概率 | 反映各类别的基础出现比例 $P(Y=c_k) = n_k/N$ |
| 高斯似然 | 概率建模 | 在每个类别内用 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ 描述每个连续特征的分布 |
| 条件独立 | 简化假设 | 假设特征间在给定类别后相互独立，使高维概率可计算 |
| 后验概率 | 决策依据 | 综合先验和似然，选出概率最大的类别 |

## 1. 生成式分类：从"数据怎么来"思考

大多数分类算法直接学习分类边界——即给定特征后直接输出类别。朴素贝叶斯走的是另一条路：

**先回答"如果样本属于类别 $c_k$，那它的特征值长什么样的概率最大"，再反推向"看到这组特征值，样本最可能属于哪一类"。**

这就是生成式模型的直觉核心：对 $P(\mathbf{x} \vert Y)$ 建模，而非直接建模 $P(Y \vert \mathbf{x})$。

### 理解重点

- 生成式视角的本质是：比较不同类别对当前样本的"解释能力"。
- 如果当前样本的特征值在类别 A 的高斯分布下很常见，在类别 B 下很罕见，那就倾向于选 A。
- 这与逻辑回归（直接拟合 $\sigma(\mathbf{w}^T \mathbf{x} + b)$）的判别式思路有本质区别——一个关心数据怎么生成，一个关心边界画在哪。

## 2. 为什么 iris 适合 GaussianNB

iris 数据集的四个特征（萼片长宽、花瓣长宽）都是连续测量值：

- 同类鸢尾花的花瓣长度通常在某个均值附近波动——这恰好是高斯分布能描述的模式。
- 三类鸢尾花的特征分布有明显的均值差异——这为基于似然的分类提供了区分力。

### 理解重点

- GaussianNB 在每个类别内为每个特征单独拟合一个高斯分布——iris 的连续特征与这一假设天然匹配。
- 如果换成文本词频数据（离散计数），应该用 MultinomialNB；如果换成二元特征，应该用 BernoulliNB。
- 当前分册选择 iris + GaussianNB 的组合，是朴素贝叶斯家族中最适合连续特征教学的配置。

## 3. 用"先验 + 似然"理解整个预测过程

可以把 GaussianNB 的预测想象成三步：

1. **先验判断**：在不看任何特征之前，先问"这个类别本身有多常见"——$P(Y=c_k)$
2. **似然评估**：对每个特征，计算"这个特征值在类别 $c_k$ 的高斯分布下有多自然"——$P(x_j \vert Y=c_k)$
3. **综合决策**：把先验和所有特征的似然乘起来（取对数相加），选得分最高的类别

$$
\hat{y} = \arg\max_{c_k} \left[ \ln P(Y=c_k) + \sum_{j=1}^{d} \ln P(x_j \vert Y=c_k) \right]
$$

### 理解重点

- 两个信息源：类别本身有多常见（先验）+ 当前特征值在这个类别下有多典型（似然）。
- 两者之间没有人为赋予的权重超参数——模型对先验和似然的信任完全由数据决定。
- 对数形式不是数学花招，而是工程必须——连乘 $d$ 个 $(0,1)$ 区间的概率值会迅速下溢到浮点数零。

## 4. 与其他分类算法的直觉对比

| 算法 | 核心问题 | 回答方式 | 是否需要迭代优化 |
|---|---|---|---|
| GaussianNB | 这个样本更像由哪个类别生成的？ | 比较各类别下的 $P(\mathbf{x} \vert Y) \cdot P(Y)$ | 否——统计量一步到位 |
| 逻辑回归 | 正类和负类之间的分界线长什么样？ | 拟合 $\sigma(\mathbf{w}^T \mathbf{x} + b)$ 的权重 $\mathbf{w}$ | 是——`lbfgs` 迭代优化交叉熵 |
| KNN | 周围最近的邻居投给哪个类别？ | 在训练集中找到最近邻，多数表决 | 否——但需要遍历全部训练集 |
| 决策树 | 按哪个特征在哪个阈值切分最好？ | 递归选择使 Gini/Entropy 下降最快的分裂 | 是——递归贪心搜索分裂点 |

### 理解重点

- GaussianNB 的独特之处在于：它既不像逻辑回归那样迭代优化，也不像 KNN 那样依赖全部训练集做预测——参数估计极快，预测也极快。
- 但代价是两项强假设：1 每类内特征服从高斯分布；2 给定类别后特征条件独立。在真实数据上这两者都可能被违反。
- 理解 GaussianNB 与其他算法的直觉差异，是理解"何时选它、何时不选它"的基础。

## 5. "朴素"假设的直觉

条件独立假设 $P(\mathbf{x} \vert Y) = \prod_j P(x_j \vert Y)$ 在直觉上意味着：

**在已知类别的条件下，每个特征独立地提供自己的证据，各说各的，互不商量。**

- 对于 iris：模型分别看"萼片长度在类别 0 下多典型""萼片宽度在类别 0 下多典型"……然后把四条证据简单相乘。
- 它不会捕捉"萼片长度和萼片宽度通常一起变化"这样的相关性。

### 理解重点

- 这个假设在真实数据里几乎从不严格成立，但朴素贝叶斯在实践中仍然经常表现良好。
- 原因在于：即使概率估计有偏差，只要各类别的得分排序正确，分类决策就不受影响。
- 这也是朴素贝叶斯的魅力所在——用极其简单的假设换取计算上的极大便利。

## 可视化

![决策边界](https://img.yumeko.site/file/blog/articles/1780736145017.png)

## 常见坑

1. 把朴素贝叶斯理解成"简化版分类器"而忽略它本质上是概率生成模型——它对 $P(\mathbf{x} \vert Y)$ 的建模方式与其他算法完全不同。
2. 把"条件独立"假设当成必须严格成立的现实前提，而不是计算可行性上的实用简化。
3. 不区分 GaussianNB（连续高斯似然）和其他朴素贝叶斯变体（MultinomialNB、BernoulliNB、ComplementNB）。
4. 只关注最终类别预测，忽略 `predict_proba(...)` 给出的后验概率信息。

## 小结

- GaussianNB 的直觉核心是生成式分类：先理解每类数据长什么样，再看当前样本更像由哪类生成。
- iris 连续特征与高斯建模的天然适配，使这个组合成为教学场景中的理想选择。
- 条件独立假设是朴素贝叶斯最大的简化——它带来了极致的计算效率，代价是概率估计可能不精确，但排序往往仍是正确的。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GaussianNB`。
2. 理解 `GaussianNB` 的构造器参数 `priors` 和 `var_smoothing` 的数学含义。
3. 看清训练完成后最重要的模型属性及其与数学公式的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `GaussianNB` 模型，打印训练日志 |
| `GaussianNB(...)` | 类 | scikit-learn 提供的高斯朴素贝叶斯分类器——对连续特征的每个类别每个特征拟合 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ |
| `model.fit(X_train, y_train)` | 方法 | 在训练数据上统计类别先验和特征高斯参数——纯统计计算，无迭代优化 |
| `model.classes_` | 属性 | 模型识别到的类别标签数组 |
| `model.class_prior_` | 属性 | 各类别先验概率 $P(Y=c_k)$ |
| `model.theta_` | 属性 | 各类别各特征的均值 $\mu_{kj}$ |
| `model.var_` | 属性 | 各类别各特征的方差 $\sigma_{kj}^2$（平滑后） |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, var_smoothing=1e-9)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，形状 $(120, 4)$，传入 `GaussianNB.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，形状 $(120,)$，取值 $y_i \in \{0, 1, 2\}$ | `y_train` |
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。实际计算 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，防止 $\sigma_{kj}^2 \to 0$ 数值崩溃。默认 `1e-9` | `1e-9`、`1e-8` |
| 返回值 | `GaussianNB` | 已完成 `fit()` 的模型对象，含 `classes_`、`class_prior_`、`theta_`、`var_` 等属性 | — |

### 示例代码

```python
from model_training.classification.naive_bayes import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- 当前入口很直接：只负责构建一个 `GaussianNB` 并 `fit`，没有变体对比或超参数搜索。
- 所有默认超参数都写在函数签名里，阅读成本低，适合作为源码入门。
- `train_model(...)` 是对 `sklearn.naive_bayes.GaussianNB` 的薄封装——算法本体在 sklearn，本仓库负责组织日志和工程流程。

## 2. `GaussianNB` 构造器参数

### 参数速览

适用 API：`GaussianNB(priors=None, var_smoothing=1e-9)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `priors` | `array_like` 或 `None` | 类别的先验概率 $P(Y=c_k)$。`None` 时从训练数据估计：$P(Y=c_k) = n_k / N$。可手动传入数组覆盖数据估计 | `None`、`[0.3, 0.3, 0.4]` |
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$。最终方差为 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$，其中 $\sigma_{\max}^2$ 是所有特征所有类别中最大的方差。防止 $\sigma_{kj}^2 \approx 0$ 时 $\frac{1}{\sqrt{2\pi \sigma^2}} \to \infty$ 导数炸 | `1e-9`、`1e-8`、`1e-7` |

### 示例代码

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_s, y_train)
```

### 理解重点

- GaussianNB 的参数极简——一共两个：`priors`（先验）和 `var_smoothing`（数值保护）。这反映了朴素贝叶斯"参数少、假设强"的特点。
- `priors` 默认为 `None` 时从训练数据按频率估计，对 iris 均衡数据来说三个类别的先验约为 $[0.33, 0.33, 0.33]$。
- `var_smoothing` 是当前分册最重要的超参数——它直接关联到方差为零时的数值稳定性问题。
- GaussianNB 的 `fit()` 不涉及迭代优化——它只是扫描数据统计均值和方差。这与逻辑回归的 `lbfgs` 迭代和决策树的递归分裂形成鲜明对比。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `classes_` | `ndarray`，形状 `(n_classes,)` | $\{c_1, c_2, c_3\}$ | 模型识别到的类别标签列表，iris 中为 `[0, 1, 2]` |
| `class_prior_` | `ndarray`，形状 `(n_classes,)` | $P(Y=c_k) = n_k / N$ | 各类别的先验概率 |
| `class_count_` | `ndarray`，形状 `(n_classes,)` | $n_k$ | 训练集中各类别的样本数 |
| `theta_` | `ndarray`，形状 `(n_classes, n_features)` | $\mu_{kj}$ | 各类别各特征的均值，对应高斯分布的位置参数 |
| `var_` | `ndarray`，形状 `(n_classes, n_features)` | $\sigma_{kj}^2$（平滑后） | 各类别各特征的方差，对应高斯分布的尺度参数——已应用 `var_smoothing` |
| `epsilon_` | `float` | $\epsilon \cdot \sigma_{\max}^2$ | `var_smoothing` 对应的实际平滑绝对值 |

### 示例代码

```python
print(f"类别: {model.classes_.tolist()}")
print(f"类别先验: {model.class_prior_.round(4)}")
print(f"各类别样本数: {model.class_count_}")
print(f"均值(theta_):\n{model.theta_}")
print(f"方差(var_):\n{model.var_}")
```

### 理解重点

- `theta_` 和 `var_` 是 GaussianNB 最核心的两个训练产出——它们就是各类别下各特征高斯分布的参数。
- `theta_` 形状 $(3, 4)$ 意味着 3 个类别 x 4 个特征 = 12 个均值；`var_` 同样有 12 个方差——模型一共只估计 24 个数字，训练极快。
- `class_prior_` 把"先验概率"这一理论概念直接映射为可观察的数值，是理解生成式分类思路的入口。
- `epsilon_` 提供了方差平滑的实际量级，对于理解 `var_smoothing` 是否真正生效有参考价值。

## 4. 训练阶段的工程封装

除了 `GaussianNB(...).fit(...)` 之外，`train_model(...)` 还做了几层工程包装：

### 参数速览

| 输出项 | 作用 |
|---|---|
| `@print_func_info` 标题 | 在终端中定位训练入口 |
| `@timeit` 训练耗时 | 观察 `fit()` 的执行时间——对 GaussianNB 通常是毫秒级 |
| `var_smoothing` 日志 | 确认当前平滑参数配置 |
| `类别` 日志 | 确认多分类类别集合 |
| `类别先验` 日志 | 观察各类别基础比例，对应 $P(Y=c_k)$ |

### 理解重点

- 当前封装强调的是教学型可读性——通过装饰器打印函数信息和耗时，通过 `print` 输出关键属性。
- 这一层把"构建模型""训练模型""打印结果"收在一个函数里，方便流水线和文档复用。
- 从工程角度看，这样的拆分让 `pipelines/classification/naive_bayes.py` 保持简洁——编排层不需要关心日志打印细节。

## 常见坑

1. 误以为当前实现使用的是所有朴素贝叶斯的通用封装——`train_model` 明确构建 `GaussianNB`，不是 `MultinomialNB` 或 `BernoulliNB`。
2. 只知道 `predict(...)`，却忽略 `theta_`、`var_`、`class_prior_` 才是理解概率分类本质的关键属性。
3. 忘记当前 `X_train` 应该是标准化后的特征——虽然 GaussianNB 不像逻辑回归那样对尺度敏感，但标准化影响方差估计的稳定性和 PCA 可视化。
4. 把训练函数和后续评估逻辑混在一起理解——`train_model` 只负责训练主模型，不负责混淆矩阵、ROC 等诊断。

## 小结

- `train_model(...)` 是本仓库 Naive Bayes 的核心训练入口，是对 `GaussianNB` 的薄封装。
- `GaussianNB` 只有两个构造器参数（`priors` 和 `var_smoothing`），属于参数最少的分类模型之一。
- 训练完成后的关键属性：`theta_`（均值 $\mu_{kj}$）、`var_`（方差 $\sigma_{kj}^2$）、`class_prior_`（先验概率）——全部是解析计算，无迭代优化。

# 训练与预测

## 本章目标

1. 按源码顺序看清当前 Naive Bayes 流水线从数据复制到概率输出的完整步骤。
2. 理解主模型 (`model`)、二维可视化模型 (`model_2d`) 和学习曲线实例三者的职责边界。
3. 理解 `predict(...)` 与 `predict_proba(...)` 在 Naive Bayes 中分别对应什么数学计算。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `naive_bayes_data.copy()` | 方法 | 复制原始数据，避免后续处理修改源对象 |
| `train_test_split(...)` | 函数 | 按 `stratify=y` 划分训练/测试集 |
| `StandardScaler` | 类 | 对特征做一致性标准化——训练集 `fit_transform`，测试集 `transform` |
| `train_model(...)` | 函数 | 训练主 `GaussianNB` 模型，返回含 `theta_`、`var_` 的模型对象 |
| `model.predict(X_test_s)` | 方法 | 输出测试集类别预测——选择后验概率最大的类别 |
| `model.predict_proba(X_test_s)` | 方法 | 输出测试集各类别的后验概率 $P(Y=c_k \vert \mathbf{x})$ |
| `PCA(n_components=2)` | 类 | 将 4 维特征投影到 2 维，为决策边界可视化提供服务 |
| `model_2d` | 模型 | 在 PCA 2D 空间单独训练的 `GaussianNB`，专用于决策边界绘图 |

## 1. 流水线起点：复制数据并拆出特征/标签

### 示例代码

```python
data = naive_bayes_data.copy()
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `.copy()` 确保后续处理不修改在模块导入时已经加载的全局 `naive_bayes_data`。
- 当前任务是有监督多分类，因此 `y` 既参与训练 `fit(X_train_s, y_train)`，也参与评估（混淆矩阵、ROC）。
- 这一步只是数据准备，不涉及任何算法逻辑。

## 2. 训练/测试集切分

### 参数速览

适用函数：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame` | 特征矩阵，形状 $(150, 4)$ | `X` |
| `y` | `Series` | 标签向量，取值 $y_i \in \{0, 1, 2\}$ | `y` |
| `test_size` | `float` | 测试集占比。150 x 0.2 = 30 测试样本，120 训练样本 | `0.2` |
| `random_state` | `int` | 随机种子，保证切分可复现 | `42` |
| `stratify` | `array_like` | 传入 `y` 使训练/测试集类别比例与原始数据一致 | `y` |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 理解重点

- `stratify=y` 在小样本（150 条）多分类（3 类）场景下尤其重要——确保训练集和测试集都包含三类样本。
- 切分必须在标准化之前执行，否则测试集信息会通过均值和标准差泄露到训练流程中。

## 3. 标准化

### 参数速览

适用 API：`StandardScaler().fit_transform(X_train)` / `StandardScaler().transform(X_test)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 $(120, 4)$ | 训练特征矩阵，用于计算 $\mu_j, \sigma_j$ 并原地标准化 | `X_train` |
| `X_test` | `array_like`，形状 $(30, 4)$ | 测试特征矩阵，使用训练集统计量进行标准化变换 | `X_test` |
| 输出 | `ndarray` | $z_{ij} = (x_{ij} - \mu_j) / \sigma_j$，每个特征化为均值 0 标准差 1 | `X_train_s`、`X_test_s` |

### 示例代码

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 标准化后每个特征的尺度统一，使得方差估计不受原始量纲（如萼片长度以 cm 为单位）的影响。
- 虽然 GaussianNB 不依赖梯度优化，但标准化有利于 PCA 可视化和跨特征方差比较。
- 当前仓库在所有分类流水线中统一保留标准化步骤——这是工程一致性设计，而非 GaussianNB 的硬性要求。

## 4. 主模型训练与硬分类预测

### 参数速览

适用 API：`train_model(X_train_s, y_train)` -> `model.predict(X_test_s)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 $(120, 4)$ | 标准化后的训练特征，传入 `GaussianNB.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签，用于统计各类别样本数及各类别下各特征的 $\mu_{kj}$、$\sigma_{kj}^2$ | `y_train` |
| `X_test_s` | `ndarray`，形状 $(30, 4)$ | 标准化后的测试特征，传入 `model.predict()` | `X_test_s` |
| 返回值 (`y_pred`) | `ndarray`，形状 $(30,)$ | 硬分类预测标签，来自 MAP 决策 $\hat{y} = \arg\max_c [\ln P(c) + \sum \ln P(x_j \vert c)]$ | `y_pred` |

### 示例代码

```python
model = train_model(X_train_s, y_train)
y_pred = model.predict(X_test_s)
```

### 理解重点

- `train_model(...)` 的 `fit()` 内部：扫描数据 -> 统计 $n_k$ -> 估计 $P(Y=c_k)$ -> 每类每特征计算 $\mu_{kj}$ 和 $\sigma_{kj}^2$ -> 应用 `var_smoothing`。不涉及任何迭代。
- `predict(...)` 内部：对每个测试样本计算所有类别的后验概率（对数形式），选最大值——这是纯粹的代数运算。
- `y_pred` 是后续混淆矩阵的直接输入。

## 5. 概率输出

### 参数速览

适用 API：`model.predict_proba(X_test_s)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_test_s` | `ndarray`，形状 $(30, 4)$ | 标准化后的测试特征 | `X_test_s` |
| 返回值 (`y_scores`) | `ndarray`，形状 $(30, 3)$ | 每个测试样本属于各类别的后验概率 $P(Y=c_k \vert \mathbf{x})$，每行和为 1 | `y_scores` |

### 示例代码

```python
y_scores = model.predict_proba(X_test_s)
```

### 理解重点

- GaussianNB 的概率输出来自贝叶斯公式：$P(c_k \vert \mathbf{x}) \propto P(c_k) \prod_j \mathcal{N}(x_j \vert \mu_{kj}, \sigma_{kj}^2)$。
- 这些概率是连续的，因为高斯似然是连续分布——这与 KNN 的离散邻域频率概率输出本质不同。
- `y_scores` 直接支撑多分类 One-vs-Rest ROC 曲线：三分类任务会对每个类别各画一条 ROC。

## 6. 决策边界需要单独训练 `model_2d`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `pca` | `PCA(n_components=2, random_state=42)` | 将 4 维标准化特征投影到 2 维主成分空间 | `pca` |
| `X_all_s` | `ndarray`，形状 $(150, 4)$ | 全量标准化特征，用于 PCA 拟合 | `scaler.transform(X)` |
| `X_2d` | `ndarray`，形状 $(150, 2)$ | PCA 二维投影后的全量特征，用于画散点 | `pca.fit_transform(X_all_s)` |
| `model_2d` | `GaussianNB()` | 在 PCA 二维空间单独训练的高斯朴素贝叶斯，专用于决策边界绘图 | `model_2d` |

### 示例代码

```python
pca = PCA(n_components=2, random_state=42)
X_all_s = scaler.transform(X)
X_2d = pca.fit_transform(X_all_s)
model_2d = GaussianNB()
model_2d.fit(pca.transform(X_train_s), y_train)
```

### 理解重点

- `model_2d` 不是主评估模型——它的唯一目的是在二维空间提供可绘制的决策边界。
- 主模型 `model` 训练在原始 4 维标准化空间，`model_2d` 训练在 PCA 2 维空间——两者是独立的对象，职责完全不同。
- PCA 降维会损失信息，因此 `model_2d` 的边界只是原始高维分类面的近似投影展示。

## 7. 学习曲线使用新的模型实例

### 参数速览

适用函数：`plot_learning_curve(GaussianNB(), X_train_s, y_train, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `GaussianNB` | 新创建的 `GaussianNB()` 实例，学习曲线内部会克隆和重复训练 | `GaussianNB()` |
| `X` | `ndarray`，形状 $(120, 4)$ | 标准化后的训练特征矩阵 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分类指标，当前取 `"accuracy"` | `"accuracy"` |
| `cv` | `int` | 交叉验证折数，默认 `5` | `5` |

### 示例代码

```python
plot_learning_curve(
    GaussianNB(),
    X_train_s,
    y_train,
    title="朴素贝叶斯 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 传入的是 `GaussianNB()` 新实例而非 `model`——因为 `plot_learning_curve` 内部会通过 `learning_curve()` 函数多次克隆和训练模型。
- 学习曲线函数会按不同训练样本量（如 10%、33%、55%、78%、100%）做交叉验证，绘制训练得分和验证得分的变化趋势。

## 训练诊断可视化

![学习曲线](https://img.yumeko.site/file/blog/articles/1780736299374.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 混为一谈——前者用于混淆矩阵（硬分类标签），后者用于 ROC 曲线（概率输出）。
2. 把 `model_2d` 误认为正式预测模型——它只在 PCA 2D 空间训练，仅服务于决策边界可视化。
3. 忘记标准化必须在训练集上 `fit_transform`、测试集上 `transform`——在切分之前标准化是数据泄露。
4. 混淆主模型（4 维空间正式预测）、二维可视化模型（PCA 空间画边界）和学习曲线模型（CV 循环克隆）的三者职责。

## 小结

- 当前 Naive Bayes 流水线的训练过程：复制数据 -> 特征/标签拆分 -> 切分（`stratify=y`）-> 标准化 -> 训练主模型 -> 硬分类预测 -> 概率输出。
- 三个模型实例各司其职：`model`（4 维主评估）、`model_2d`（PCA 2D 可视化）、`GaussianNB()`（学习曲线克隆）。
- GaussianNB 的训练（`fit`）和预测（`predict`/`predict_proba`）都是纯代数运算，不涉及迭代——这是它在工程上区别于逻辑回归的最显著特征。

# 评估与诊断

## 本章目标

1. 明确当前仓库 Naive Bayes 实现的四种评估手段及其分别回答的问题。
2. 理解 3x3 混淆矩阵、One-vs-Rest ROC 曲线和 PCA 决策边界图在多分类场景下的解读方式。
3. 理解 GaussianNB 的独有可解释性来源——`theta_`、`var_`、`class_prior_`——而非决策树式的特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `y_pred` | 预测结果 | 测试集类别输出，由 MAP 决策 $\arg\max_c [\ln P(c) + \sum \ln P(x_j \vert c)]$ 产生 |
| `y_scores` | 预测概率 | 测试集各类别后验概率，来自贝叶斯公式后验归一化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制 3x3 多分类混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 One-vs-Rest ROC 曲线——每类别一条 |
| `plot_decision_boundary(...)` | 函数 | 绘制 PCA 2D 空间下的分类边界 |
| `plot_learning_curve(...)` | 函数 | 绘制训练/验证得分随样本量变化的曲线 |

## 1. 当前仓库的评估入口

当前 Naive Bayes 流水线里的主要诊断手段有四个：

1. 混淆矩阵 —— 回答"分对了多少？哪两类最容易混淆？"
2. ROC 曲线（One-vs-Rest）—— 回答"每个类别的概率区分能力如何？"
3. PCA 2D 决策边界图 —— 回答"在二维投影视角下，边界长什么样？"
4. 学习曲线 —— 回答"更多训练样本还能提升表现吗？"

### 示例代码

```python
y_pred = model.predict(X_test_s)
y_scores = model.predict_proba(X_test_s)

plot_confusion_matrix(...)
plot_roc_curve(...)
plot_decision_boundary(...)
plot_learning_curve(...)
```

### 理解重点

- 四种可视化分别回答不同问题，不能互相替代。
- GaussianNB 没有决策树式的特征重要性评估（`feature_importances_`），也没有逻辑回归式的 `coef_` 系数解释——但它有 `theta_`（各类别下特征均值差异）和 `class_prior_`（先验概率）提供概率视角的可解释性。
- 对教学型仓库来说，这种多视角诊断设计比只打印一个准确率数字更利于理解模型行为。

## 2. 混淆矩阵能观察什么

对于 iris 三分类任务，混淆矩阵 $\mathbf{C}$ 是一个 $3 \times 3$ 矩阵：

$$
C_{ij} = \text{真实类别 } i \text{ 被预测为类别 } j \text{ 的样本数}
$$

### 参数速览

适用函数：`plot_confusion_matrix(y_true, y_pred, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，$y_i \in \{0, 1, 2\}$ | `y_test` |
| `y_pred` | `array_like`，形状 `(n_samples,)` | 模型预测标签，来自 MAP 硬分类 | `y_pred` |
| `normalize` | `bool` 或 `str` | 归一化方式。`True`/`'true'` 按行（真实类别），`'pred'` 按列，`'all'` 按全体。默认 `False` | `True`、`'true'` |

### 示例代码

```python
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    title="朴素贝叶斯 混淆矩阵",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 在 iris 三分类上，混淆矩阵最能揭示哪些类别之间容易混淆——例如 Versicolour 和 Virginica 在特征空间中有重叠，误分类通常集中在这两类之间。
- Setosa 通常与另两类完全分离，对应对角线上的高值。
- 矩阵已经隐式包含计算 Accuracy、Precision、Recall、F1 所需的所有信息（各类的 TP、FP、FN），但当前仓库未显式计算这些指标。

## 3. ROC 曲线能观察什么（多分类 One-vs-Rest）

多分类 ROC 采用 One-vs-Rest 策略：对每个类别 $c_k$，将其视为"正类"，其余两类合并为"负类"，分别画一条 ROC。

$$
\text{TPR}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad
\text{FPR}_k = \frac{\text{FP}_k}{\text{FP}_k + \text{TN}_k}
$$

### 参数速览

适用函数：`plot_roc_curve(y_test, y_scores, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_true` | `array_like`，形状 `(n_samples,)` | 测试集真实标签，$y_i \in \{0, 1, 2\}$ | `y_test` |
| `y_scores` | `array_like`，形状 `(n_samples, 3)` | 各类别后验概率，来自 `model.predict_proba(X_test_s)` | `y_scores` |

### 示例代码

```python
plot_roc_curve(
    y_test,
    y_scores,
    title="朴素贝叶斯 ROC 曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- 与逻辑回归（二分类，一条 ROC）不同，iris 的三分类任务会生成三条 ROC 曲线——每条对应一个类别 vs 其余类别。
- GaussianNB 的概率输出来自贝叶斯后验概率的归一化，是连续值——因此 ROC 曲线是平滑的。这与 KNN 的离散邻域频率形成对比。
- 三条 ROC 曲线的 AUC 可以直观比较哪些类别的概率区分力强、哪些弱。

## 4. PCA 2D 决策边界图能观察什么

### 参数速览

适用函数：`plot_decision_boundary(model_2d, X_2d, y.values, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model_2d` | `GaussianNB` | 在 PCA 二维空间单独训练的朴素贝叶斯模型 | `model_2d` |
| `X_2d` | `ndarray`，形状 `(150, 2)` | 标准化后 PCA 投影到二维的全量特征 | `X_2d` |
| `y` | `array_like`，形状 `(150,)` | 全量标签数组，用于散点的真实类别着色 | `y.values` |

### 示例代码

```python
plot_decision_boundary(
    model_2d,
    X_2d,
    y.values,
    title="朴素贝叶斯 决策边界 (PCA 2D)",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- GaussianNB 的决策边界由高斯似然等高线相交形成——在二维空间中是二次曲线（椭圆相交），而非逻辑回归的直线或决策树的轴对齐分段。
- 但这是 PCA 投影空间中的近似展示，原始 4 维特征空间中的决策面是三区域的高斯密度比较。
- 三个类别的区域大小和形状可以直观反映各类别高斯分布的方差估计差异。

## 5. 学习曲线能观察什么

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `GaussianNB` | 新创建的 `GaussianNB()` 实例——内部会通过 CV 克隆并重复训练 | `GaussianNB()` |
| `X` | `ndarray`，形状 `(120, 4)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `y` | `array_like` | 训练标签向量 | `y_train` |
| `scoring` | `str` | 评分指标，默认 `"accuracy"` | `"accuracy"` |
| `cv` | `int` | 交叉验证折数，默认 `5` | `5` |
| `train_sizes` | `array_like` | 训练样本量的递增序列，默认为 `np.linspace(0.1, 1.0, 5)` | `[0.1, 0.33, 0.55, 0.78, 1.0]` |

### 示例代码

```python
plot_learning_curve(
    GaussianNB(),
    X_train_s,
    y_train,
    title="朴素贝叶斯 学习曲线",
    dataset_name=DATASET,
    model_name=MODEL,
)
```

### 理解重点

- GaussianNB 的参数极少（每类每特征只需估计 $\mu$ 和 $\sigma^2$），因此即使训练样本量小也能获得稳定估计——学习曲线通常在较早阶段就趋于平稳。
- 训练得分和验证得分通常很接近，反映了简单模型的低方差特性。
- 如果验证得分远低于训练得分，说明高斯假设或条件独立假设在当前数据上偏离较大。

## 6. 当前实现中尚未纳入但常见的分类指标

| 指标 | 公式 | 说明 |
|---|---|---|
| 准确率（Accuracy） | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | 整体正确率，多分类中即对角线之和除以总和 |
| 精确率（Precision） | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | 预测为正类中有多少真实正类——多分类可取宏平均/微平均 |
| 召回率（Recall） | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | 真实正类中有多少被正确找出 |
| F1 分数 | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | 精确率与召回率的调和平均 |

### 理解重点

- 当前仓库未在 Naive Bayes 流水线中显式打印这些指标——文档可以提到它们作为扩展方向，但不能写成"当前源码已在单独计算"。
- 混淆矩阵已经隐式包含了计算这些指标所需的所有信息。
- 对于多分类任务，宏平均（每类指标求均值）和微平均（全局统计再计算）是两个常见选择。

## 评估图表

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

![ROC 曲线](https://img.yumeko.site/file/blog/articles/1780736321096.png)

## 常见坑

1. 把 `predict(...)` 和 `predict_proba(...)` 的用途混为一谈——前者用于混淆矩阵，后者用于 ROC。
2. 把多分类 ROC（One-vs-Rest，三条曲线）误解为只有一条全局曲线。
3. 把 PCA 决策边界图误认为原始 4 维特征空间决策面的完整表达——它只是二维投影近似。
4. 把当前仓库未显式计算的 accuracy、precision、recall、f1 写成现有流程的一部分。

## 小结

- 当前仓库对 Naive Bayes 的评估：混淆矩阵看错误分布（3x3 多分类），ROC 曲线看各类别概率区分力（One-vs-Rest 三条），PCA 决策边界图看二维投影边界形状，学习曲线看样本量对表现的影响趋势。
- GaussianNB 不产生 `feature_importances_`（决策树）或 `coef_`（逻辑回归），但其 `theta_`（各类别下各特征均值）和 `var_`（各类别下各特征方差）提供了生成式视角下的可解释性。
- 四项评估组合起来，能全面解释 `GaussianNB(var_smoothing=1e-9)` 在 iris 三分类任务上的实际表现。

# 工程实现

## 本章目标

1. 从工程角度看清 Naive Bayes 在本仓库中的完整调用链。
2. 理解数据加载、模型训练、流水线编排和结果可视化分别负责什么。
3. 理解 GaussianNB 工程实现与其他分类算法的关键差异——无需迭代、极简封装。

## 对应代码速览

| 组件 | 路径 | 说明 |
|---|---|---|
| 数据生成 | `data_generation/classification.py` | `ClassificationData.naive_bayes()` 加载 iris 真实数据集 |
| 数据导出 | `data_generation/__init__.py` | 向外暴露 `naive_bayes_data` |
| 训练封装 | `model_training/classification/naive_bayes.py` | 构建并训练 `GaussianNB`，打印 `classes_`、`class_prior_` |
| 流水线入口 | `pipelines/classification/naive_bayes.py` | 组织切分、标准化、训练、预测与可视化的完整编排 |
| 混淆矩阵可视化 | `result_visualization/confusion_matrix.py` | 绘制并保存 3x3 多分类混淆矩阵图 |
| ROC 曲线可视化 | `result_visualization/roc_curve.py` | 绘制并保存多分类 One-vs-Rest ROC 曲线图 |
| 决策边界可视化 | `result_visualization/decision_boundary.py` | 绘制并保存 PCA 二维决策边界图 |
| 学习曲线可视化 | `result_visualization/learning_curve.py` | 绘制并保存训练/验证得分曲线图 |

## 1. 端到端运行入口

### 示例代码

```bash
python -m pipelines.classification.naive_bayes
```

### 理解重点

- 这个命令串起当前 Naive Bayes 分册中最核心的工程流程。
- 依次完成：数据复制 -> 特征/标签拆分 -> 切分 -> 标准化 -> GaussianNB `fit()`（统计 $\mu_{kj}$、$\sigma_{kj}^2$）-> 预测 -> 概率输出 -> 四种可视化。
- 对大多数读者来说，`pipelines/classification/naive_bayes.py` 是理解工程实现的最佳起点。

## 2. `run()` 串起了整个流程

当前流水线的核心函数 `run()` 采用线性编排风格：

```python
def run():
    # 1. 复制数据 & 拆出特征/标签
    data = naive_bayes_data.copy()
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

    # 4. 统计各类别的先验、均值和方差 & 正式预测
    model = train_model(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_scores = model.predict_proba(X_test_s)

    # 5. 可视化诊断（混淆矩阵、ROC、决策边界、学习曲线）
    plot_confusion_matrix(y_test, y_pred, ...)
    plot_roc_curve(y_test, y_scores, ...)
    plot_decision_boundary(model_2d, X_2d, y.values, ...)
    plot_learning_curve(GaussianNB(), X_train_s, y_train, ...)
```

### 理解重点

- `run()` 的职责是编排，不是算法实现——真正的训练在 `GaussianNB.fit()`（统计 $\mu_{kj}$ 和 $\sigma_{kj}^2$）中。
- 数据流是单向的：数据 -> 切分 -> 标准化 -> 参数估计 -> 预测 -> 评估。
- 与其他分类流水线（逻辑回归、KNN、决策树）结构高度一致——统一采用"数据准备 -> 训练 -> 预测 -> 四类可视化"的模式。

## 3. 训练模块负责什么

`model_training/classification/naive_bayes.py` 里的 `train_model(...)` 主要负责四件事：

1. 创建 `GaussianNB(var_smoothing=1e-9)` 实例
2. 调用 `model.fit(X_train, y_train)`——纯统计计算：计数 $n_k$，估计 $P(Y=c_k)$、$\mu_{kj}$、$\sigma_{kj}^2$
3. 打印训练日志：耗时、`var_smoothing`、`classes_`、`class_prior_`
4. 返回训练完成的主模型对象

### 参数速览

适用函数：`train_model(X_train, y_train, var_smoothing=1e-9)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like` | 标准化后的训练特征矩阵，传入 `GaussianNB.fit()` | `X_train_s` |
| `y_train` | `array_like` | 训练标签向量，$y_i \in \{0, 1, 2\}$ | `y_train` |
| `var_smoothing` | `float` | 方差平滑项 $\epsilon$，防止 $\sigma_{kj}^2 \to 0$ 数值崩溃。默认 `1e-9` | `1e-9`、`1e-8` |
| 返回值 | `GaussianNB` | 已完成 `fit()` 的模型对象，含 `classes_`、`class_prior_`、`theta_`、`var_` 等 | — |

### 理解重点

- GaussianNB 的 `fit()` 本质是扫描数据算统计量——不需要迭代、不需要梯度、不需要收敛判断。这是它工程上最大的亮点。
- 训练日志中的 `class_prior_` 直接对应 $P(Y=c_k)$，是连接数学理论与工程实现的桥梁。

## 4. 四类评估模块分别负责什么

### 模块职责速览

| 模块 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | `plot_confusion_matrix(...)` | `y_test`、`y_pred` | 3x3 多分类混淆矩阵图（PNG） |
| ROC 曲线 | `plot_roc_curve(...)` | `y_test`、`y_scores` | 多分类 One-vs-Rest ROC 曲线图（PNG）——3 条曲线 |
| 决策边界 | `plot_decision_boundary(...)` | `model_2d`、`X_2d`、`y.values` | PCA 二维分类边界图（PNG） |
| 学习曲线 | `plot_learning_curve(...)` | `GaussianNB()`、`X_train_s`、`y_train` | 训练/验证得分随样本量变化曲线（PNG） |

### 理解重点

- 四类可视化都不是训练的一部分，而是训练完成后的诊断步骤。
- 决策边界图依赖额外训练的 `model_2d`（PCA 空间中），ROC 曲线依赖 `predict_proba(...)` 的连续后验概率——两者各有特殊依赖。
- GaussianNB 没有特征重要性评估（决策树特有），也没有 `coef_`（逻辑回归特有）——但 `theta_` 的各类别均值差异在评估阶段同样有参考价值。

## 5. 模块间的数据依赖关系

| 数据 | 生产者 | 消费者 |
|---|---|---|
| `naive_bayes_data` | `data_generation/classification.py` | `pipelines/classification/naive_bayes.py` |
| `model`（主模型） | `train_model(...)` | `predict`、`predict_proba` |
| `y_pred` | `model.predict(...)` | `plot_confusion_matrix` |
| `y_scores` | `model.predict_proba(...)` | `plot_roc_curve` |
| `model_2d` | `GaussianNB().fit(...)`（PCA 空间） | `plot_decision_boundary` |
| 图片产物 | 各可视化函数 | `outputs/naive_bayes/` 目录 |

### 理解重点

- 数据流向单向、无循环依赖，每个模块可以独立测试和替换。
- Naive Bayes 的流水线结构与逻辑回归、KNN 高度一致——这体现了当前仓库工程架构的统一性。

## 6. 运行后能得到什么

### 输出项

| 输出类型 | 当前结果 | 用途 |
|---|---|---|
| 终端标题 | `朴素贝叶斯分类流水线` | 在终端中定位当前运行入口 |
| 训练日志 | 训练耗时、`var_smoothing`、`classes_`、`class_prior_` | 查看参数估计耗时和各类别先验概率 |
| 混淆矩阵图 | `outputs/naive_bayes/confusion_matrix.png` | 观察三分类误分类方向——哪两类最易混淆 |
| ROC 曲线图 | `outputs/naive_bayes/roc_curve.png` | 评估各类别贝叶斯后验概率的区分能力（3 条曲线） |
| 决策边界图 | `outputs/naive_bayes/decision_boundary.png` | 观察 PCA 2D 空间中的高斯似然区域划分 |
| 学习曲线图 | `outputs/naive_bayes/learning_curve.png` | 诊断样本量对高斯参数估计的影响趋势 |

### 理解重点

- 训练日志中的 `class_prior_` 是 GaussianNB 独有的信息——它直接揭示三类鸢尾花的先验分布。
- 与逻辑回归输出 `coef_` 不同，GaussianNB 输出 `class_prior_` 反映的是生成式建模的视角。
- 输出不仅是图片，还有终端日志中可观察的关键统计量。

## 7. 推荐的源码阅读顺序

1. 先看 `pipelines/classification/naive_bayes.py` — 入口，了解整体流程
2. 再看 `model_training/classification/naive_bayes.py` — 训练封装，理解参数估计和日志输出
3. 再看 `result_visualization/confusion_matrix.py` — 基础分类结果评估（3x3 矩阵）
4. 再看 `result_visualization/roc_curve.py` — One-vs-Rest 概率区分能力评估
5. 再看 `result_visualization/decision_boundary.py` — PCA 空间边界可视化
6. 再看 `result_visualization/learning_curve.py` — 训练行为诊断
7. 最后回到 `data_generation/classification.py` — 理解 iris 数据加载逻辑

### 理解重点

- 从入口看整体流程，再下钻到训练与可视化细节，阅读成本最低。
- 这个顺序对应数据流方向：数据 -> 标准化 -> 参数估计 -> 预测 -> 评估。

## 运行结果

![运行结果展示](https://img.yumeko.site/file/blog/articles/1780736366408.png)

## 常见坑

1. 把 `pipeline` 文件误认为训练算法实现本体——它只是编排层，真正的算法在 `GaussianNB.fit()` 中。
2. 不区分"主模型"（4 维空间）、"二维可视化模型"（PCA 空间）和"学习曲线模型实例"（CV 循环克隆）的职责边界。
3. 忽略 `train_model(...)` 中打印的 `class_prior_` 日志——这是理解生成式模型先验概率的入口。
4. 只看单个文件，不顺着调用链理解整体执行流程。

## 小结

- 当前 Naive Bayes 工程实现采用清晰的模块分层：数据生成 -> 训练封装 -> 流水线编排 -> 结果可视化。
- `run()` 负责串联，`train_model(...)` 负责参数估计（纯统计计算），各可视化函数负责结果展示与诊断。
- GaussianNB 在工程上最不同于逻辑回归/决策树的地方：`fit()` 不涉及迭代优化——纯粹的统计量扫描；训练日志输出 `class_prior_` 而非 `coef_`；封装极简（仅 2 个构造器参数）。

# 练习与参考文献

## 本章目标

1. 用练习题帮助读者检查自己是否真正理解当前 Naive Bayes 实现。
2. 给出继续深入阅读高斯朴素贝叶斯与相关数据集工具的可靠入口。

## 自检题

1. 为什么 `pipelines/classification/naive_bayes.py` 要先做训练/测试切分，再做标准化？如果在切分前标准化会有什么问题？
2. 为什么当前 iris 连续特征数据适合 `GaussianNB` 而非文本分类常见的 `MultinomialNB`？`GaussianNB` 的高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ 与另外两种朴素贝叶斯变体的似然建模有什么本质区别？
3. 当前 `train_model(...)` 中的 `var_smoothing` 控制什么？实际计算中 $\sigma_{kj}^2 + \epsilon \cdot \sigma_{\max}^2$ 里的 $\sigma_{\max}^2$ 是什么？为什么方差接近 0 时会引发数值问题？
4. 为什么 `model.class_prior_`、`model.theta_` 和 `model.var_` 对理解 GaussianNB 很重要？它们分别对应贝叶斯公式中的哪些项？
5. 为什么 ROC 曲线这里使用 `predict_proba(...)` 而不是 `predict(...)`？GaussianNB 的连续后验概率与 KNN 的离散邻域频率概率输出有什么不同？
6. 为什么决策边界图里需要额外训练一个 `model_2d`？它在什么特征空间上训练？GaussianNB 的 PCA 二维边界为什么可能呈现曲线而非直线？
7. GaussianNB 的 `fit()` 为什么是所有分类模型中最快的之一？它不依赖迭代优化的数学原因是什么？

## 练习方向

### 1. 改动 `var_smoothing`

- 把 `var_smoothing=1e-9` 改成 `1e-12`、`1e-9`、`1e-6`、`1e-3`
- 观察变化：
  - `model.epsilon_` 的实际值——即 $\epsilon \cdot \sigma_{\max}^2$ 的量级
  - `model.var_` 中各类别各特征方差的变化——平滑越小，方差越接近原始样本方差
  - 混淆矩阵和 ROC 曲线的变化——极端平滑值可能导致概率估计失准
- 核心理解：`var_smoothing` 在数值稳定性和模型精度之间的权衡——$\epsilon$ 太大过度平滑，$\epsilon$ 太小可能数值崩溃

### 2. 观察 `theta_` 与 `var_` 的类别间差异

- 在训练完成后打印 `model.theta_`（形状 $(3, 4)$）和 `model.var_`（形状 $(3, 4)$）
- 对比三类鸢尾花在 4 个特征上的均值差异——例如 Setosa 与另外两类的花瓣长度均值差距最大
- 对比各类别各特征的方差——方差较小的特征（如 Setosa 的花瓣长度）表示该类在该特征上更集中
- 核心理解：$\mu_{kj}$ 的类间差异越大、$\sigma_{kj}^2$ 越小，该特征对该类别的区分力越强

### 3. 去掉标准化

- 暂时去掉 `StandardScaler()`，直接用 `X_train`、`X_test` 训练和预测
- 对比变化：
  - `theta_` 和 `var_` 的值——特征量纲差异直接反映在方差数值上
  - PCA 决策边界图和混淆矩阵的变化
- 体会：虽然 GaussianNB 不像逻辑回归那样依赖标准化做梯度优化，但标准化使方差估计更稳定，并影响 PCA 可视化的主导方向

### 4. 观察 `predict_proba` 与 `predict` 的关系

- 对同一测试样本同时输出 `y_pred`（硬分类）和 `y_scores`（各类别概率）
- 验证：`y_pred[i]` 是否总是等于 `np.argmax(y_scores[i])`
- 观察三类概率的分布——正确预测样本的概率是否接近 1？错误预测样本的概率是否较均匀（如 $[0.1, 0.45, 0.45]$）？
- 核心理解：MAP 决策 $\hat{y} = \arg\max_c P(c \vert \mathbf{x})$ 等价于在 `predict_proba` 的输出行上取 `argmax`

### 5. 与逻辑回归、KNN、决策树对比

- 对照阅读 `docs/classification/logistic_regression/`、`docs/classification/knn/`、`docs/classification/decision_tree/`
- 比较要点：
  - 建模方式：GaussianNB 是生成式（对 $P(\mathbf{x} \vert Y)$ 建模），逻辑回归是判别式（对 $P(Y \vert \mathbf{x})$ 直接建模），KNN 是非参数（无显式 $P(\mathbf{x} \vert Y)$ 建模），决策树是判别式（递归划分）
  - 训练方式：GaussianNB 是统计量扫描（$\mu_{kj}$、$\sigma_{kj}^2$ 一步到位），逻辑回归是迭代优化（`lbfgs`），KNN 无训练（仅建索引），决策树是递归贪心搜索
  - 是否需要标准化：GaussianNB 不强制（但利于可视化和方差比较），逻辑回归必须（梯度收敛），KNN 必须（距离度量），决策树不需要（阈值切分）
  - 可解释性：GaussianNB 有 `theta_`/`var_`（各类别特征分布）、`class_prior_`（先验），逻辑回归有 `coef_`（线性权重），决策树有 `feature_importances_`（贡献度），KNN 没有显式特征重要性
  - 概率输出性质：GaussianNB 是连续的高斯后验（平滑 ROC），KNN 是离散的邻域频率（阶梯状 ROC）

## 参考文献

| # | 文献 | 说明 |
|---|---|---|
| 1 | scikit-learn 官方文档：`GaussianNB` | 完整构造器参数（`priors`、`var_smoothing`）、属性（`class_prior_`、`theta_`、`var_`、`epsilon_`）与方法说明 |
| 2 | scikit-learn 官方文档：`load_iris` | iris 数据集的来源、特征含义与类别说明 |
| 3 | scikit-learn 用户指南：Naive Bayes | GaussianNB、MultinomialNB、BernoulliNB、ComplementNB 的完整数学推导与使用场景对比 |
| 4 | Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. | 第 3 章：生成式分类模型；第 4 章：高斯判别分析（GDA）与朴素贝叶斯的数学关系 |

- scikit-learn `GaussianNB`：https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- scikit-learn `load_iris`：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- scikit-learn 用户指南 Naive Bayes：https://scikit-learn.org/stable/modules/naive_bayes.html

## 小结

- 这一章的重点不是新增概念，而是把前面章节学到的内容重新落到源码和实验现象上。
- 如果能独立解释以下问题，说明已经掌握了当前 Naive Bayes 分册的核心内容：
  - 标准化必须在切分后执行（防止数据泄露），GaussianNB 保留标准化主要是为了方差稳定性和 PCA 可视化
  - 贝叶斯公式 $P(Y \vert \mathbf{x}) \propto P(\mathbf{x} \vert Y) P(Y)$ -> 条件独立 $\prod P(x_j \vert Y)$ -> 高斯似然 $\mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$ -> MAP 决策（对数形式）的完整数学链
  - `var_smoothing` 是 $\sigma_{kj}^2$ 的数值保护——方差近零时 $\frac{1}{\sqrt{2\pi \sigma^2}} \to \infty$
  - `theta_` 和 `var_` 反映各类别各特征的分布特征，类间均值差异大且方差小的特征是区分力最强的特征
  - GaussianNB 的概率输出是连续的贝叶斯后验概率，ROC 曲线平滑——与 KNN 的离散邻域频率本质不同
  - `model`（4 维空间）、`model_2d`（PCA 空间）和学习曲线实例的职责差异
  - GaussianNB 的所有参数（先验、均值、方差）都是解析解——不涉及迭代优化，这是它训练极快的根本原因
