---
title: Bagging 集成学习
date: 2026-05-10
category: MachineLearning/Ensemble
tags:
  - Scikit-learn
description: Bagging集成学习的数学原理、Bootstrap采样与方差缩减及完整实现流程。
image: https://img.yumeko.site/file/blog/cover/1780581699942.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 Bootstrap 抽样的概率基础——每个样本被选入训练子集的概率约为 63.2%。
2. 理解 Bagging 为何能降低方差——通过并行训练多个不相关（或弱相关）模型并投票平均。
3. 理解 OOB（Out-of-Bag）误差估计的数学原理——利用未参与训练的样本做无偏估计。
4. 理解为什么 Bagging 选择完全生长的决策树（高方差低偏差）作为基学习器。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Bootstrap 采样 | 抽样方法 | 从 $N$ 个样本中有放回地抽取 $m$ 个样本——每个子训练集约含 63.2% 的原始样本 |
| 方差缩减 | 核心原理 | 对 $n$ 个方差均为 $\sigma^2$、两两相关系数为 $\rho$ 的模型取平均，集成方差为 $\rho\sigma^2 + (1-\rho)\sigma^2/n$ |
| 投票聚合 | 输出方式 | 分类任务：$n$ 个基学习器投票，多数票决定最终预测 |
| OOB 误差 | 评估指标 | 用未参与训练的约 36.8% 样本评估每个基学习器——等价于交叉验证 |
| `n_estimators` | 源码参数 | 基学习器数量——越多方差越低，但边际收益递减 |
| `max_samples` | 源码参数 | 每个 Bootstrap 子集的样本比例——控制训练子集与原始数据的差异度 |

## 1. Bootstrap 抽样

给定 $N$ 个样本的数据集 $\mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_N, y_N)\}$，Bootstrap 抽样从中**有放回**地抽取 $m$ 个样本，构成一个训练子集 $\mathcal{D}_b$。

### 单个样本未被抽中的概率

每个样本在一次抽取中被选中的概率为 $1/N$，未被选中的概率为 $1 - 1/N$。$m$ 次独立抽取后：

$$
P(\text{样本 } i \text{ 未被抽中}) = \left(1 - \frac{1}{N}\right)^m
$$

当 $m = N$ 时（即子集大小等于原始数据大小），取极限：

$$
\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368
$$

### 理解重点

- 每个 Bootstrap 子集约含原始数据中约 **63.2%** 的样本——剩余的约 **36.8%** 就是 OOB 样本。
- 当前源码 `max_samples=0.8` 表示 $m = 0.8N$——子集比原始数据稍小，进一步增加了子集间的差异性。
- Bootstrap 采样的随机性使每个基学习器看到的数据分布略有不同——这是"模型多样性"的来源。

## 2. 方差缩减原理

### 独立模型的方差

设 $n$ 个基学习器 $f_1, \dots, f_n$，每个的预测方差均为 $\sigma^2$，两两之间的相关系数为 $\rho$。Bagging 通过投票（分类）或平均（回归）聚合：

$$
f_{\text{bag}}(\mathbf{x}) = \frac{1}{n} \sum_{b=1}^{n} f_b(\mathbf{x})
$$

集成模型的方差为：

$$
\text{Var}[f_{\text{bag}}] = \frac{1}{n^2} \left( \sum_{b=1}^{n} \text{Var}[f_b] + \sum_{b \neq c} \text{Cov}[f_b, f_c] \right) = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2
$$

### 两种极端情况

- $\rho = 1$（完全相关——所有基学习器完全相同）：$\text{Var}[f_{\text{bag}}] = \sigma^2$——Bagging 无帮助
- $\rho = 0$（完全不相关——基学习器完全独立）：$\text{Var}[f_{\text{bag}}] = \sigma^2 / n$——方差随 $n$ 线性下降

实际情况下 $0 < \rho < 1$，Bagging 在方差缩减和基学习器多样性之间取得平衡。

### 理解重点

- Bagging **降低方差，不降低偏差**——集成模型的偏差约等于单个基学习器的偏差。
- 这就是为什么 Bagging 选择**完全生长的决策树**（`max_depth=None`）——它们偏差极低但方差极高，正是 Bagging 最受益的对象。
- `n_estimators=80` 意味着理论上方差约缩减为 $\rho\sigma^2 + (1-\rho)\sigma^2/80$——当 $\rho$ 较小时，方差大幅下降。
- 若基学习器本身偏差就很高（如浅层决策树），Bagging 无法纠正——低偏差是基学习器的必要前提。

## 3. OOB（Out-of-Bag）误差估计

对每个样本 $(\mathbf{x}_i, y_i)$，找到所有未使用该样本训练的基学习器 $\{b : (\mathbf{x}_i, y_i) \notin \mathcal{D}_b\}$，仅用这些基学习器预测 $\hat{y}_i^{\text{OOB}}$，计算：

$$
\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[y_i \neq \hat{y}_i^{\text{OOB}}]
$$

等价地，OOB 得分：

$$
\text{OOB Score} = 1 - \text{OOB Error}
$$

### 理解重点

- OOB 误差等价于**对每个样本做一次留出验证**——无需额外划分验证集。
- 与交叉验证不同，OOB 误差在训练过程中"免费"获得——不需要额外训练。
- 当前源码 `oob_score=True` 启用此功能——`model.oob_score_` 打印到 4 位小数。
- OOB 得分可以直接作为模型泛化能力的参考——当它与测试集准确率接近时，说明模型泛化良好。

## 4. 为什么选择完全生长的决策树

Bagging 的方差缩减依赖于基学习器满足两个条件：

1. **低偏差**——基学习器必须能拟合训练数据（偏差小）
2. **高方差**——不同的训练子集应导致明显不同的模型（方差大）

完全生长的决策树（`max_depth=None`）完美满足这两个条件：
- 能完美拟合训练数据（偏差 $\approx 0$）
- 对训练数据的微小变化极其敏感（方差极大）

### 理解重点

- 如果使用浅层决策树或线性模型（低方差），Bagging 的方差缩减效果非常有限——因为没有方差可缩减。
- 当前源码中的基学习器参数 `max_depth=None, min_samples_split=2, min_samples_leaf=1`——刻意让每棵树完全生长，最大化方差。
- 这与 Boosting 形成对比——Boosting 通常使用浅层决策树（弱学习器），因为它的目标是**降低偏差**而非方差。

## 5. Bagging 与 Boosting 的数学对比

| 维度 | Bagging | Boosting（如 GBDT） |
|---|---|---|
| 训练方式 | 并行——$n$ 个模型独立训练 | 串行——每个模型拟合前一个模型的残差 |
| 核心目标 | 降低方差 | 降低偏差 |
| 基学习器 | 强学习器（低偏差高方差，如完全生长树） | 弱学习器（高偏差低方差，如浅层树） |
| 样本权重 | 等权重 Bootstrap 采样 | 自适应加权——错分样本权重增大 |
| 模型权重 | 等权重投票 | 按模型性能加权 |
| 过拟合风险 | 低——并行平均天然正则化 | 较高——串行拟合可能过度追逐残差 |
| 标志性参数 | `n_estimators`、`max_samples` | `n_estimators`、`learning_rate`、`max_depth` |

### 理解重点

- Bagging 和 Boosting 不是"谁更好"——Bagging 在基学习器过拟合时帮它"冷静下来"（降方差），Boosting 在基学习器欠拟合时帮它"变得更准"（降偏差）。
- 当前高噪声双月牙数据 + 完全生长树是一个经典场景——单棵树严重过拟合（方差极大），Bagging 通过并行投票大幅改善。

## 6. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| Bootstrap 采样 | $P(\text{未抽中}) \approx e^{-m/N}$ | `bootstrap=True`，`max_samples=0.8` |
| 基学习器数 | $n$ | `n_estimators=80` |
| 集成预测（分类） | $\hat{y} = \text{majority}\{f_b(\mathbf{x})\}_{b=1}^{n}$ | `model.predict(X)` |
| 集成概率（分类） | $\hat{p} = \frac{1}{n}\sum_b p_b$ | `model.predict_proba(X)` |
| OOB 得分 | $1 - \frac{1}{N}\sum_i \mathbb{I}[y_i \neq \hat{y}_i^{\text{OOB}}]$ | `model.oob_score_`（`oob_score=True`） |
| 基学习器 | $f_b$：完全生长决策树 | `DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)` |
| 方差缩减 | $\text{Var}[f_{\text{bag}}] = \rho\sigma^2 + (1-\rho)\sigma^2/n$ | Bagging 核心机制 |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler`（训练集拟合/测试集变换） |
| 分层抽样 | 保持类别比例 | `train_test_split(stratify=y)` |

## 常见坑

1. 混淆 Bagging 与 Boosting 的目标——Bagging 降方差（并行投票），Boosting 降偏差（串行纠错）。
2. 把 Bagging 的基学习器设为低方差模型——方差缩减需要高方差基学习器。
3. 忽略 OOB 得分的存在——它提供了"免费的"泛化能力估计，无需额外划分验证集。
4. 把 `n_estimators` 设得过小——$n < 10$ 时方差缩减效果有限。

## 小结

- Bagging 的数学核心链：Bootstrap 采样（约 63.2% 样本被抽中）→ 并行训练 $n$ 个高方差基学习器 → 投票/平均聚合 → 方差从 $\sigma^2$ 缩减到 $\rho\sigma^2 + (1-\rho)\sigma^2/n$ → OOB 误差提供无偏估计。
- Bagging 降低方差而不降低偏差——因此需要低偏差高方差的基学习器（完全生长的决策树）。
- 当前源码 `BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None), n_estimators=80, max_samples=0.8, bootstrap=True, oob_score=True)` 是针对高噪声双月牙数据最经典的 Bagging 配置。

# 数据构成

## 本章目标

1. 明确本仓库 Bagging 数据来自 `EnsembleData.bagging()` 构造的高噪声双月牙二分类数据。
2. 理解为什么选择高噪声数据——`noise=0.35` 使单棵完全生长树严重过拟合，从而最大程度体现 Bagging 的方差缩减价值。
3. 明确当前流程中的训练/测试切分（分层抽样）和标准化顺序。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.bagging()` | 方法 | 生成 Bagging 使用的高噪声双月牙二分类数据 |
| `make_moons(...)` | 函数 | scikit-learn 提供的双月牙数据生成器 |
| `bagging_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `bagging_noise` | 参数 | 噪声水平 `0.35`——刻意高于其他算法，体现 Bagging 降方差优势 |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——训练集拟合、测试集变换 |

## 1. 数据生成：`EnsembleData.bagging()`

当前 Bagging 数据来自 `EnsembleData.bagging()`，底层调用 `sklearn.datasets.make_moons()`。

### 参数速览

适用函数：`make_moons(n_samples=500, noise=0.35, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。默认 `500`——适中规模，80 个 Bagging 子学习器可在秒级完成训练 | `500`、`1000` |
| `noise` | `float` | 标签噪声的标准差。`0.35` 刻意偏高——使单棵树严重过拟合，凸显 Bagging 的方差缩减效果 | `0.1`、`0.35` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(500, 2)$，$y$ 取值 $\{0, 1\}$ | — |

### 示例代码

```python
X, y = make_moons(
    n_samples=500,
    noise=0.35,
    random_state=42,
)
data = DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
```

### 理解重点

- `make_moons` 生成两个交错弯月形的类别——边界弯曲且非线性，单棵完全生长树容易在噪声区域过拟合出极其复杂的锯齿状边界。
- `noise=0.35` 是设计选择——高于 DBSCAN 分册中的 `noise=0.08`。高噪声使单棵树的决策边界充满"噪声驱动的伪结构"，这正是 Bagging 擅长处理的场景。
- 与 DBSCAN 使用同一数据生成器但不同噪声水平——DBSCAN 需要干净的密度结构（低噪声），Bagging 需要高噪声来展示方差缩减。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(500, 2)$ | 含 2 个连续特征的特征矩阵，列名 `x1`、`x2` | `data.drop(columns=["label"])` |
| `y` | `Series`，形状 $(500,)$ | 二分类标签 $\{0, 1\}$——参与 Bagging 训练和评估 | `data["label"]` |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
```

### 理解重点

- `label` 是真正的二分类监督标签——参与 `model.fit()`、`model.predict()` 和混淆矩阵/ROC 评估。
- 这是监督分类（与降维和聚类分册不同）——`label` 既是训练目标，也是评估基准。

## 3. 训练/测试切分与标准化

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(500, 2)$ | 全量特征矩阵 | `X` |
| `y` | `Series`，形状 $(500,)$ | 全量标签 | `y` |
| `test_size` | `float` | 测试集比例。默认 `0.2` | `0.2`、`0.3` |
| `stratify` | `array_like` | 分层抽样依据——确保训练/测试集中类别比例一致 | `y` |
| `random_state` | `int` | 随机种子。默认 `42` | `42` |
| 返回值 | `(DataFrame, DataFrame, Series, Series)` | `X_train`（400 样本）、`X_test`（100 样本）及对应标签 | — |

### 示例代码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- 当前流水线**有**训练/测试切分——这与降维和聚类分册不同，与分类分册（SVC、Naive Bayes）一致。
- `stratify=y` 确保两个集合中类别比例与原始数据一致——对于二分类月牙数据，这避免了某个集合中一类完全缺失。
- 标准化采用监督学习的标准做法：`fit_transform` 在训练集上计算 $\mu$ 和 $\sigma$，`transform` 在测试集上使用相同统计量——防止测试集信息泄露。

## 数据可视化

![类别分布图](https://img.yumeko.site/file/articles/ML/bagging/data_class_distribution.png)

![标注散点图](https://img.yumeko.site/file/articles/ML/bagging/data_scatter.png)

![特征相关性热力图](https://img.yumeko.site/file/articles/ML/bagging/data_correlation.png)

## 常见坑

1. 把高噪声当成数据缺陷——`noise=0.35` 是有意设计，噪声太低无法体现 Bagging 相对于单棵树的优势。
2. 忽略 `stratify=y` 的重要性——不平衡数据上不设分层抽样可能导致测试集中某类别缺失。
3. 在测试集上 `fit_transform` 而非 `transform`——这是数据泄露的典型错误，测试集标准化必须使用训练集的统计量。

## 小结

- 当前 Bagging 数据来自 `make_moons(n_samples=500, noise=0.35)`：2 个连续特征、二分类、高噪声双月牙。
- 数据流为：`make_moons` → DataFrame（`x1`、`x2` + `label`）→ 分层训练/测试切分 → 训练集拟合标准化器 / 测试集变换。
- `noise=0.35` 的设计意图是让单棵完全生长树过拟合到锯齿状边界——从而最大程度体现 Bagging 通过并行投票平滑边界的价值。

# 思路与直觉

## 本章目标

1. 用直观方式理解 Bagging 的核心思路——"找一群各执己见的专家，让他们投票决定"。
2. 理解为什么 Bagging 选择完全生长的决策树作为基学习器——高方差才有缩减空间。
3. 通过与单棵决策树和 Boosting 的对比，建立 Bagging 在集成学习谱系中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| Bootstrap 采样 | 基础操作 | 从原始数据中有放回抽样——每个基学习器看到的数据略有不同 |
| 并行训练 | 训练方式 | $n$ 个基学习器完全独立训练——互不依赖，可并行加速 |
| 投票聚合 | 输出方式 | 分类时多数投票——每个基学习器一票，少数服从多数 |
| 方差缩减 | 核心收益 | 多个"意见不同但各有道理"的模型取平均——抵消各自的随机波动 |
| OOB 估计 | 免费诊断 | 用未参与训练的样本评估——不需要额外划分验证集 |
| 高噪声数据 | 场景设计 | `noise=0.35` 使单棵树严重过拟合——充分展示 Bagging 的平滑作用 |

## 1. 为什么需要 Bagging

单棵完全生长的决策树有一个致命弱点：**对训练数据的微小变化极其敏感**。稍微换一批训练数据，树的形状可能完全不同——这就是高方差的表现。

Bagging 的思路很直接：

> 既然一棵树太"神经质"，那就种一片森林。每棵树的训练数据略有不同，过拟合的模式也各不相同。让它们投票——各自的过拟合噪声互相抵消，剩下的就是真正的信号。

### 理解重点

- 这就像做一个重要决定前咨询多个朋友——每个人都有自己的偏见（方差），但多数人的共识往往更可靠。
- Bagging 不改变基学习器的偏差（它没有纠正错误），它只降低方差（让不同模型的错误互相抵消）。
- 因此 Bagging 的两个前提是：（1）基学习器偏差足够低（能拟合数据）；（2）基学习器之间足够不同（采样子集差异足够大）。

## 2. 用"不同老师教出的不同学生"理解 Bagging

Bagging 的工作方式可以想象成：

1. **从同一份教材中每人随机抽取一部分章节**（Bootstrap 采样——每棵树看到约 63.2% 的数据）
2. **每个人独立学习，完全不受他人影响**（并行训练——80 棵树各自 `fit`）
3. **每人学到的东西略有不同**（不同的采样子集 → 不同的决策边界）
4. **面对新问题时投票决定**（分类：80 棵树投票，多数票为最终答案）

### 理解重点

- 第一个学生可能在噪声点上纠结出极其复杂的规则——这是过拟合。
- 但 80 个学生在不同子集上纠结出的噪声规则各不相同——投票时这些噪声规则互相矛盾，自动抵消。
- 而真正的信号（两个月牙的整体形状）在所有子集中都存在——投票时信号会被放大。
- 这就是方差缩减的直觉：**噪声随机、信号一致 → 平均后噪声减弱、信号增强**。

## 3. 为什么选择完全生长的决策树

决策树的 `max_depth` 决定了它的偏差-方差特征：

- `max_depth=1`（决策树桩）：偏差极高、方差极低——Bagging 70 个树桩仍然是高偏差
- `max_depth=5`：中等偏差、中等方差——Bagging 有改善但有限
- `max_depth=None`（完全生长）：**偏差极低、方差极高**——Bagging 最受益

### 理解重点

- Bagging 只能降方差，不能降偏差——因此必须从低偏差的基学习器出发。
- 完全生长的决策树对数据极其敏感——两个 Bootstrap 子集产出的树结构可能完全不同，这正是方差缩减的前提。
- 当前源码 `max_depth=None, min_samples_split=2, min_samples_leaf=1`——每一项参数都在鼓励树"充分生长、高度敏感"。

## 4. 为什么高噪声数据最适合展示 Bagging

当前数据 `make_moons(noise=0.35)` 的噪声水平远高于常规设置：

- **低噪声**（`noise=0.1`）：单棵树的边界已经比较平滑——Bagging 的改善空间有限
- **高噪声**（`noise=0.35`）：单棵树的边界是锯齿状迷宫——严重过拟合噪声点
- Bagging 80 棵树投票后：锯齿被平滑为接近真实弯月形状的边界

### 理解重点

- Bagging 的优势在**高噪声场景下最明显**——这正是当前数据设计的意图。
- 如果数据本身就很干净，单棵树已经表现良好——Bagging 的边际收益就很小。
- 教学型数据设计刻意放大了场景差异——让读者一眼看出 Bagging 在做什么。

## 5. OOB 得分的直觉：免费的考试

每次 Bootstrap 采样后，约 36.8% 的样本没有被抽中——它们天然构成了该树的"测试集"。

- 对每个样本，找出所有"没见过它"的树，只用这些树预测它
- 这些预测的准确率就是 OOB 得分——**不需要额外划分验证集**

### 理解重点

- 这就像每个学生只学了教材的一部分，考全班时只看他没学过的那部分的得分——对泛化能力的无偏估计。
- 当前源码 `oob_score=True` 启用此功能——训练完成后 `model.oob_score_` 直接给出估计。
- OOB 得分与测试集准确率通常接近——如果差距很大，说明数据分布可能有问题。

## 6. 与 Boosting 的直觉对比

| 维度 | Bagging | Boosting |
|---|---|---|
| 核心问题 | 如何让一群"太敏感"的模型冷静下来？ | 如何让一个"太迟钝"的模型变得更聪明？ |
| 训练方式 | 平行——各学各的，最后投票 | 串行——后面的人补前面的人犯的错 |
| 基学习器 | 强学习器（完全生长树——偏差低方差高） | 弱学习器（浅层树——偏差高方差低） |
| 核心收益 | 降方差——投票平滑了过拟合的锯齿 | 降偏差——接力修正了欠拟合的粗糙 |
| 过拟合风险 | 低——并行平均天然正则化 | 较高——串行纠错可能过度追逐噪声 |
| 并行能力 | 天然可并行——各树独立训练 | 必须串行——每棵树依赖前序结果 |
| 理想场景 | 高噪声数据 + 复杂基学习器 | 中等噪声数据 + 简单基学习器 |

### 理解重点

- Bagging 和 Boosting 不是"谁更强"——Bagging 是个"过于敏感"的解决方案（降方差），Boosting 是个"过于迟钝"的解决方案（降偏差）。
- 当前高噪声双月牙数据 + 完全生长树，恰好是 Bagging 最理想、Boosting 需要小心控制的场景。

## 可视化

![混淆矩阵](https://img.yumeko.site/file/articles/ML/bagging/confusion_matrix.png)

## 常见坑

1. 把 Bagging 当成"万能的模型增强器"——它只对高方差基学习器有效，对已低方差的模型收效甚微。
2. 以为 `n_estimators` 越多越好——边际收益递减，`n_estimators=80` 后继续增加带来的方差缩减微乎其微。
3. 忽略 Bagging 与 Boosting 在基学习器选择上的本质差异——Bagging 用强学习器，Boosting 用弱学习器。
4. 不启用 `oob_score`——放弃了免费的泛化能力估计。

## 小结

- Bagging 的直觉核心是并行集成降方差：Bootstrap 采样产生差异化的训练子集 → 并行训练多个高方差基学习器 → 投票平均使随机噪声互相抵消 → 决策边界更平滑。
- 当前高噪声双月牙数据 + 完全生长决策树 + `n_estimators=80` 是展示 Bagging 方差缩减能力的最佳教学组合。
- Bagging 与 Boosting 在直觉上截然相反：一个靠"人多势众"抵消个体偏见（降方差），一个靠"接力纠错"逐步逼近真相（降偏差）——选哪个取决于基学习器是"过于敏感"还是"过于迟钝"。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `BaggingClassifier`（含基学习器的创建）。
2. 理解 `BaggingClassifier` 的核心构造器参数（`n_estimators`、`max_samples`、`bootstrap`、`oob_score`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`oob_score_`（OOB 得分）、`estimators_`（基学习器列表）、`predict_proba`（概率输出）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.BaggingClassifier` 模型，打印 OOB 得分日志 |
| `BaggingClassifier(...)` | 类 | scikit-learn 提供的 Bagging 集成分类器——通过 Bootstrap 采样 + 并行投票实现方差缩减 |
| `DecisionTreeClassifier(...)` | 类 | 基学习器——完全生长的决策树（`max_depth=None`），高方差低偏差 |
| `model.fit(X_train, y_train)` | 方法 | 并行训练 $n$ 个基学习器——每个在各自的 Bootstrap 子集上独立 `fit` |
| `model.oob_score_` | 属性 | OOB 得分——用未参与训练的样本估计泛化能力 |
| `model.predict(X)` | 方法 | 投票聚合——$n$ 个基学习器多数投票决定最终预测 |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(400, 2)` | 标准化后的训练特征矩阵，传入 `BaggingClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(400,)` | 训练标签 $\{0, 1\}$——二分类监督信息 | `y_train` |
| `n_estimators` | `int` | 基学习器数量。当前设为 `80`——在方差缩减与计算成本间取得平衡 | `10`、`50`、`80`、`200` |
| `max_samples` | `float` | 每个 Bootstrap 子集的样本比例。`0.8` 表示子集大小为 $0.8 \times N$ | `0.5`、`0.8`、`1.0` |
| `max_features` | `float` | 每个 Bootstrap 子集的随机特征比例。`1.0` 表示使用全部特征 | `0.5`、`1.0` |
| `bootstrap` | `bool` | 是否使用有放回 Bootstrap 采样。`True`——Bagging 的核心操作 | `True`、`False` |
| `oob_score` | `bool` | 是否启用 OOB 得分估计。`True`——训练后可用 `model.oob_score_` | `True`、`False` |
| `random_state` | `int` | 随机种子，保证 Bootstrap 采样和基学习器可复现。默认 `42` | `42` |
| 返回值 | `BaggingClassifier` | 已完成 `fit()` 的模型对象，含 `oob_score_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.bagging import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。这是分类算法与降维/聚类算法最根本的差异。
- `n_estimators=80` 和 `max_samples=0.8` 是教学平衡选择——80 棵树在秒级完成训练，0.8 的采样比例使子集间有足够差异。
- `train_model(...)` 内部还会创建 `DecisionTreeClassifier` 作为基学习器——它是 Bagging 的核心组件。

## 2. 基学习器：`DecisionTreeClassifier`

Bagging 的方差缩减效果高度依赖基学习器的特性。

### 参数速览

适用 API：`DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `max_depth` | `int` 或 `None` | 树的最大深度。`None`——完全生长，不限制深度（最低偏差、最高方差） | `None`、`5`、`10` |
| `min_samples_split` | `int` | 内部节点再分裂的最小样本数。`2`——允许分裂到极致 | `2`、`5`、`10` |
| `min_samples_leaf` | `int` | 叶节点的最小样本数。`1`——允许叶节点只含一个样本 | `1`、`5` |
| `random_state` | `int` | 随机种子。传入 `BaggingClassifier` 的 `random_state` | `42` |

### 理解重点

- 这三项参数（`None`、`2`、`1`）的组合刻意让每棵树**充分生长、高度敏感**——对 Bootstrap 子集的微小差异产生截然不同的树结构。
- 这正是 Bagging 方差缩减的前提——基学习器方差越大（但偏差保持极低），Bagging 的改善越显著。
- 如果改为 `max_depth=3`（浅层树），Bagging 的改善将非常有限——因为浅层树本身的方差就不高。

## 3. `BaggingClassifier` 构造器参数

### 参数速览

适用 API：`BaggingClassifier(estimator=..., n_estimators=80, max_samples=0.8, max_features=1.0, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `BaseEstimator` | 基学习器对象。当前为完全生长的 `DecisionTreeClassifier`。scikit-learn 1.2+ 使用此参数名，旧版本使用 `base_estimator` | `DecisionTreeClassifier(max_depth=None)` |
| `n_estimators` | `int` | 基学习器数量。默认 `80`——Bagging 的核心参数，越大方差越低但边际递减 | `10`、`80`、`200` |
| `max_samples` | `int` 或 `float` | 每个 Bootstrap 子集的样本数/比例。`0.8` 表示 80% 的样本量 | `0.5`、`0.8`、`1.0` |
| `max_features` | `int` 或 `float` | 每个 Bootstrap 子集的随机特征数/比例。`1.0` 表示使用全部特征 | `0.5`、`1.0` |
| `bootstrap` | `bool` | 是否 Bootstrap 采样。`True`——Bagging 的核心；`False` 时使用全部样本（退化为简单投票） | `True`、`False` |
| `bootstrap_features` | `bool` | 是否对特征也做 Bootstrap 采样。默认 `False` | `False`、`True` |
| `oob_score` | `bool` | 是否计算 OOB 得分。`True`——训练后 `model.oob_score_` 可用 | `True`、`False` |
| `n_jobs` | `int` | 并行作业数。`-1` 使用所有 CPU 核心——Bagging 天然并行 | `-1`、`1`、`4` |
| `random_state` | `int` | 随机种子，保证 Bootstrap 采样可复现。默认 `42` | `42` |
| `verbose` | `int` | 日志详细程度。默认 `0` | `0`、`1` |

### 示例代码

```python
base = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state,
)

# sklearn 版本兼容
try:
    model = BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1,
    )
except TypeError:
    model = BaggingClassifier(
        base_estimator=base,  # 旧版本参数名
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        random_state=random_state,
        n_jobs=-1,
    )
```

### 理解重点

- `n_estimators` 是 Bagging 最重要的参数——它直接控制方差缩减的程度。80 棵树通常已经足够。
- `max_samples=0.8`（而非 `1.0`）使每个子集更小、差异更大——进一步增强了模型多样性。
- `n_jobs=-1` 利用 Bagging 的天然并行性——80 棵树可以同时训练，大幅缩短训练时间。
- 源码中的 `try/except TypeError` 是 sklearn 版本兼容处理——`estimator` 参数名在 1.2 版本从 `base_estimator` 改为 `estimator`。

## 4. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `oob_score_` | `float` | OOB 得分 $\in [0, 1]$ | 用未参与训练的样本估计的泛化准确率——$1 - \text{OOB Error}$。仅 `oob_score=True` 时可用 |
| `estimators_` | `list`，长度 `n_estimators` | 基学习器集合 $\{f_b\}_{b=1}^{n}$ | 80 个已完成 `fit()` 的 `DecisionTreeClassifier` 对象 |
| `estimators_features_` | `list` | 每个基学习器使用的特征子集 | `max_features < 1.0` 时有意义——当前为全特征 |
| `classes_` | `ndarray`，形状 `(2,)` | 类别标签 | 二分类——`[0, 1]` |
| `n_features_in_` | `int` | 特征维度 $d$ | 训练时输入的特征维数，当前为 `2` |
| `oob_decision_function_` | `ndarray` | 各样本在各类别上的 OOB 投票概率 | `oob_score=True` 时可用——提供比 `oob_score_` 更细粒度的信息 |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"max_samples: {max_samples}")
print(f"max_features: {max_features}")
print(f"bootstrap: {bootstrap}")
if oob_score:
    print(f"OOB 得分: {model.oob_score_:.4f}")
```

### 理解重点

- `oob_score_` 是 Bagging 独有的输出——它提供了一个无需额外划分验证集的泛化能力估计。当前源码打印到 4 位小数。
- `estimators_` 是集成学习的标志性属性——它存储了所有 80 个基学习器，可用于单独检查或分析。
- 与单模型分类器（如 SVC）的关键对比：Bagging 有 `estimators_`（基学习器集合）和 `oob_score_`（免费泛化估计），单模型分类器没有。

## 5. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1\}$ | 80 棵树投票——多数获胜 |
| `predict_proba(X)` | `array_like`，形状 `(n, 2)` | `ndarray`，形状 `(n, 2)` | 80 棵树预测概率的平均——用于 ROC 曲线 |

### 理解重点

- `predict()` 是硬投票——每棵树投一票，取多数。
- `predict_proba()` 是软投票——取 80 棵树预测概率的平均值。对 ROC 曲线而言，概率输出比硬分类标签更有信息量。
- 当前流水线用 `hasattr(model, "predict_proba")` 做条件判断——`BaggingClassifier` 始终支持 `predict_proba`（只要基学习器支持），但条件判断是防御性工程习惯。

## 常见坑

1. 把基学习器设为低方差模型——Bagging 的方差缩减效果完全依赖基学习器的高方差特性。
2. 忽略 `n_estimators` 的边际递减效应——80→200 的改善远小于 10→80 的改善。
3. 忘记 `oob_score=True`——放弃了 Bagging 独有的免费泛化估计。
4. 混淆 `max_samples` 和 `max_features`——前者控制样本采样比例（每棵树看到的数据量），后者控制特征采样比例（每棵树看到的特征子集）。

## 小结

- `train_model(...)` 是本仓库 Bagging 的核心训练入口，是对 `sklearn.ensemble.BaggingClassifier` 的薄封装。
- `BaggingClassifier` 的核心参数是 `n_estimators`（基学习器数量）、`max_samples`（采样比例）、`bootstrap`（采样方式）、`oob_score`（OOB 估计）——四者共同决定方差缩减的程度和泛化估计的可用性。
- 基学习器 `DecisionTreeClassifier(max_depth=None)` 的配置（完全生长）是刻意选择——高方差是 Bagging 受益的前提。
- 训练完成后的核心属性：`oob_score_`（免费泛化估计）、`estimators_`（基学习器集合）——前者是 Bagging 独有的诊断工具，后者是集成学习的标志性结构。

# 训练与预测

## 本章目标

1. 理解 Bagging 流水线的完整执行流程——从数据拆解到模型预测。
2. 认清 Bagging 作为监督分类算法的流程特征——有训练/测试切分、有 `y_train` 参与训练、有 `predict` 和 `predict_proba`。
3. 理解流水线中每一步的意图和与算法原理的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | Bagging 分类端到端流水线入口——串联数据准备、标准化、训练、预测、评估 |
| `train_test_split(..., stratify=y)` | 函数 | 分层训练/测试切分——保证两集合类别比例一致 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集、`transform` 在测试集 |
| `model.predict(X_test)` | 方法 | 80 棵树多数投票——输出硬分类标签 |
| `model.predict_proba(X_test)` | 方法 | 80 棵树概率平均——输出软分类概率 |
| `hasattr(model, "predict_proba")` | 函数 | 防御性检查——`BaggingClassifier` 始终支持概率输出 |

## 1. `run()` 流水线总览

Bagging 流水线是一个典型的监督分类流程——与 DBSCAN、KMeans 等无监督算法不同，它包含训练/测试切分步骤。

### 参数速览

适用函数：`run()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 无参数 | — | `run()` 无参数——所有配置硬编码在函数体内 | — |
| 返回值 | `None` | 触发完整的 Bagging 训练+预测+可视化流程 | — |

### 示例代码

```python
from pipelines.ensemble.bagging import run

run()
```

或命令行：

```bash
python -m pipelines.ensemble.bagging
```

### 理解重点

- `run()` 是薄流程编排层——每个步骤调用现有模块，本身不包含算法逻辑。
- 流水线中的每一步都是独立可替换的——换数据、换模型、换评估方式只需替换对应组件。
- 与 DBSCAN/KMeans 流水线的关键差异在于**有监督**——必须传入 `y_train` 给 `train_model`。

## 2. 数据准备：复制、拆解、切分

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 复制数据 | `data = bagging_data.copy()` | 避免修改模块级全局变量 |
| 拆解 X/y | `X = data.drop(columns=["label"])`、`y = data["label"]` | 分离特征和标签——监督学习的标准操作 |
| 分层切分 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | 80% 训练 + 20% 测试，按类别比例分层 |

### 理解重点

- `stratify=y` 确保训练集和测试集中类别 0 和类别 1 的比例一致——对于 `noise=0.35` 的高噪声数据，虽然两个弯月类别数量本就大致均衡，但分层抽样仍是监督分类的标准做法。
- 训练集 400 样本、测试集 100 样本——规模适中，80 个 Bagging 子学习器在秒级完成训练。
- `bagging_data` 来自 `data_generation/__init__.py` 的模块级变量——`copy()` 是防御性工程习惯。

## 3. 标准化：训练集拟合、测试集变换

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 训练集拟合 | `scaler.fit_transform(X_train)` → `X_train_s` | 在训练集上计算 $\mu$ 和 $\sigma$，同时变换 |
| 测试集变换 | `scaler.transform(X_test)` → `X_test_s` | 使用训练集的 $\mu$ 和 $\sigma$ 变换——防止数据泄露 |

### 理解重点

- 标准化对 Bagging 不是必需的——决策树天然不受特征尺度影响。但当前代码保留标准化是为了一致性（其他算法如 SVC、逻辑回归需要标准化）。
- 正确做法是 `fit_transform` 训练集、`transform` 测试集——如果在测试集上 `fit_transform`，会导致信息泄露（测试集的统计量不应该出现在训练阶段）。
- 标准化后的数据 `X_train_s` 和 `X_test_s` 是 `ndarray`（不是 `DataFrame`）——这是 `StandardScaler` 的标准行为。

## 4. 模型训练：`train_model(X_train_s, y_train)`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 `(400, 2)` | 标准化后的训练特征 | `scaler.fit_transform(X_train)` |
| `y_train` | `Series`，形状 `(400,)` | 训练标签 $\{0, 1\}$ | `y_train` |
| 返回值 | `BaggingClassifier` | 已完成 `fit()` 的模型——含 80 棵完全生长的基学习器 | — |

### 理解重点

- `train_model(...)` **必须有 `y_train`**——这是 Bagging 作为监督分类算法的根本标志。与 PCA、KMeans 等无监督算法不同。
- 训练过程中内部并行执行：80 棵树独立 `fit`，利用 `n_jobs=-1` 使用全部 CPU 核心。
- 训练完成后终端打印 `n_estimators`、`max_samples`、`max_features`、`bootstrap` 以及 OOB 得分（4 位小数）。

## 5. 预测：`predict()` 和 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 机制 |
|---|---|---|---|
| `model.predict(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100,)`，取值 $\{0, 1\}$ | 80 棵树硬投票——多数获胜 |
| `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 80 棵树软投票——概率取平均 |

### 示例代码

```python
y_pred = model.predict(X_test_s)
# y_pred 形状 (100,)，取值 {0, 1}

if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
    # y_scores 形状 (100, 2)，每行两个类别的预测概率
```

### 输出

```text
# predict 输出示例（前 10 个预测）
[1 0 1 0 0 1 1 0 1 0]

# predict_proba 输出示例（前 5 行）
[[0.1  0.9 ]
 [0.7  0.3 ]
 [0.05 0.95]
 [0.85 0.15]
 [0.6  0.4 ]]
```

### 理解重点

- `predict()` 输出硬标签——直接用于混淆矩阵。
- `predict_proba()` 输出概率——用于 ROC 曲线。概率输出比硬标签更有信息量（不仅知道预测类别，还知道置信度）。
- `hasattr(model, "predict_proba")` 是防御性检查——虽然 `BaggingClassifier` 始终支持概率输出（只要基学习器支持），但检查避免了因 API 变化导致的崩溃。
- 测试集 100 个样本——规模足够让混淆矩阵和 ROC 曲线展示有意义的统计信息。

## 6. 评估触发：混淆矩阵 + ROC 曲线

### 参数速览

| 步骤 | 触发条件 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | **始终** | `y_test` + `y_pred` | `outputs/bagging/confusion_matrix.png` |
| ROC 曲线 | **条件**：`predict_proba` 可用 | `y_test` + `y_scores` | `outputs/bagging/roc_curve.png` |

### 理解重点

- 混淆矩阵是 Bagging 分类评估的必选项——无论 `predict_proba` 是否可用，都能用硬标签生成。
- ROC 曲线是条件可选项——需要概率输出。对 Bagging 而言这个条件始终满足，但条件判断是防御性工程习惯。
- 与 DBSCAN 流水线的对比：DBSCAN 的 `plot_clusters` 只做可视化，没有硬指标——Bagging 同时有混淆矩阵（硬标签）和 ROC 曲线（软概率），评估更全面。

## 完整流程总结

```
bagging_data.copy()
    │
    ├─ X = data.drop(columns=["label"])
    ├─ y = data["label"]
    │
    ├─ train_test_split(test_size=0.2, stratify=y)
    │   ├─ X_train (400, 2)、y_train (400,)
    │   └─ X_test (100, 2)、y_test (100,)
    │
    ├─ StandardScaler
    │   ├─ X_train_s = scaler.fit_transform(X_train)
    │   └─ X_test_s = scaler.transform(X_test)
    │
    ├─ model = train_model(X_train_s, y_train)
    │   └─ 终端打印: n_estimators, max_samples, max_features, bootstrap, OOB 得分
    │
    ├─ y_pred = model.predict(X_test_s)           → 混淆矩阵
    └─ y_scores = model.predict_proba(X_test_s)   → ROC 曲线（条件）
```

## 常见坑

1. 混淆 Bagging 的流程与无监督算法——Bagging 需要 `y_train`，有训练/测试切分，有 `predict` 和 `predict_proba`。
2. 在测试集上 `fit_transform` 而非 `transform`——这是数据泄露的经典错误。
3. 忘记 `stratify=y`——不平衡数据上可能导致测试集中某类别缺失。
4. 忽略 `hasattr(model, "predict_proba")` 防御性检查的作用——虽然 Bagging 始终支持，但换用其他模型时这个检查可能是必要的。

## 小结

- Bagging 流水线是一个标准的监督分类流程：数据拆解 → 分层切分 → 训练集拟合标准化/测试集变换 → 训练（含 OOB 估计） → 硬预测/软概率 → 混淆矩阵/ROC 曲线。
- 与 DBSCAN/KMeans 等无监督流水线的核心差异：有 `y_train`、有 `train_test_split`、有混淆矩阵和 ROC 曲线（而非聚类散点图）。
- `run()` 是薄编排层——每步调用现有模块，自身不含算法逻辑。

# 评估与诊断

## 本章目标

1. 理解 Bagging 的三项评估输出——混淆矩阵（硬标签）、ROC 曲线（软概率）、OOB 得分（训练内）。
2. 理解 OOB 得分作为泛化估计的内在价值——无需额外划分验证集。
3. 明确当前代码中评估的实现范围——什么叫"已实现"、什么叫"未实现"、以及为什么。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_confusion_matrix(...)` | 函数 | 在测试集上绘制混淆矩阵——评估 80 棵树投票后的硬分类准确率 |
| `plot_roc_curve(...)` | 函数 | 绘制 ROC 曲线——评估概率输出的排序能力 |
| `model.oob_score_` | 属性 | OOB 得分——训练过程中"免费"获得的泛化能力估计 |
| `accuracy` | 概念 | 混淆矩阵对角元素之和 / 总样本数——当前通过混淆矩阵可视化间接呈现 |
| AUC | 概念 | 面积 Under ROC Curve——当前通过 ROC 曲线可视化间接呈现 |

## 1. 混淆矩阵：硬标签评估

混淆矩阵评估的是 `model.predict()` 的输出——80 棵树硬投票的最终分类结果。

### 参数速览

适用 API：`plot_confusion_matrix(y_test, y_pred, title="Bagging 混淆矩阵", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_pred` | `ndarray`，形状 `(100,)` | 模型硬投票预测结果 | `model.predict(X_test_s)` |
| `title` | `str` | 图表标题 | `"Bagging 混淆矩阵"` |
| `dataset_name` | `str` | 数据集名称——用于输出路径和标识 | `"bagging"` |
| `model_name` | `str` | 模型名称——用于输出路径区分 | `"bagging"` |
| 输出 | PNG 文件 | 混淆矩阵热力图 | `outputs/bagging/confusion_matrix.png` |

### 示例代码

```python
y_pred = model.predict(X_test_s)

plot_confusion_matrix(
    y_test, y_pred,
    title="Bagging 混淆矩阵",
    dataset_name="bagging",
    model_name="bagging",
)
```

### 理解重点

- 混淆矩阵的四个格子分别对应：
  - 真阳性（TP）——正确预测为类别 1
  - 真阴性（TN）——正确预测为类别 0
  - 假阳性（FP）——错误预测为类别 1
  - 假阴性（FN）——错误预测为类别 0
- 当前高噪声双月牙数据（`noise=0.35`）下——Bagging 的混淆矩阵相比单棵决策树应有明显改善：对角元素更"粗"、非对角元素更"细"。
- 混淆矩阵始终生成——不依赖 `predict_proba`，这是硬投票评估的底线。

## 2. ROC 曲线：软概率评估

ROC 曲线评估的是 `model.predict_proba()` 的输出——80 棵树概率平均后，在不同阈值下的 TPR/FPR 分布。

### 参数速览

适用 API：`plot_roc_curve(y_test, y_scores, title="Bagging ROC 曲线", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_scores` | `ndarray`，形状 `(100, 2)` | 模型概率输出——每行两个类别的预测概率 | `model.predict_proba(X_test_s)` |
| `title` | `str` | 图表标题 | `"Bagging ROC 曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"bagging"` |
| `model_name` | `str` | 模型名称 | `"bagging"` |
| 输出 | PNG 文件 | ROC 曲线图 | `outputs/bagging/roc_curve.png` |

### 示例代码

```python
if hasattr(model, "predict_proba"):
    y_scores = model.predict_proba(X_test_s)
    plot_roc_curve(
        y_test, y_scores,
        title="Bagging ROC 曲线",
        dataset_name="bagging",
        model_name="bagging",
    )
```

### 理解重点

- ROC 曲线是条件输出——只有 `predict_proba` 可用时才生成。`BaggingClassifier` 始终支持概率输出（只要基学习器支持），条件判断是防御性工程习惯。
- AUC（Area Under Curve）越接近 1.0，模型的排序能力越强——当前高噪声场景下，Bagging 的 AUC 通常显著高于单棵决策树。
- 概率输出比硬标签更有信息量——混淆矩阵只看最终分类是否正确，ROC 曲线看模型对每个预测的置信度。

## 3. OOB 得分：训练内的泛化估计

OOB 得分是 Bagging 独有的诊断工具——不需要额外划分验证集，在训练过程中"免费"获得。

### 参数速览

| 属性名 | 类型 | 取值范围 | 获取条件 | 含义 |
|---|---|---|---|---|
| `model.oob_score_` | `float` | $[0, 1]$ | `oob_score=True` | 用未参与训练的样本估计的泛化准确率——OOB Score = 1 - OOB Error |

### 示例代码

```python
# 训练时打印（4 位小数）
if oob_score:
    print(f"OOB 得分: {model.oob_score_:.4f}")

# 典型输出
# OOB 得分: 0.8975
```

### 理解重点

- OOB 得分与测试集准确率通常接近——如果差距很大（如 OOB 得分远高于测试集准确率），说明数据分布可能有问题或测试集存在偏差。
- OOB 估计等价于对每个样本做一次留出验证——对每个样本，只用没"见过"它的基学习器做预测。
- 当前源码打印到 4 位小数——提供足够精度用于模型对比。
- OOB 得分只存在于终端日志中——不会生成单独的图表或文件。

## 4. 当前代码已实现 vs 未实现的评估内容

### 已实现

| 评估项 | 输出形式 | 触发条件 |
|---|---|---|
| 混淆矩阵 | PNG 热力图（`outputs/bagging/confusion_matrix.png`） | 始终 |
| ROC 曲线 | PNG 曲线图（`outputs/bagging/roc_curve.png`） | `hasattr(model, "predict_proba")` |
| OOB 得分 | 终端打印（4 位小数） | `oob_score=True` |
| 训练超参数日志 | 终端打印（n_estimators、max_samples 等） | `train_model(...)` 调用 |

### 未实现（以及原因）

| 未实现的评估项 | 原因 |
|---|---|
| 测试集准确率（硬数字打印） | 当前通过混淆矩阵可视化间接呈现——可直接读出对角元素占比 |
| 精确率/召回率/F1 分数 | 教学型代码保持最小范围——混淆矩阵 + ROC 已覆盖二分类评估的核心 |
| 学习曲线（n_estimators 递增 vs 性能） | 需要额外训练开销——教学型流水线保持轻量 |
| 与单棵决策树的性能对比 | 流水线中未训练单棵决策树——但在直觉章节已做定性对比 |
| 基学习器多样性分析 | 属于深度分析——超出了教学型流水线的范围 |
| 决策边界可视化 | 虽然 `predict` 可支持，但当前流水线选择混淆矩阵 + ROC 作为评估重点 |

### 理解重点

- 当前评估策略是"可视化优先 + OOB 补足"——两项图表（混淆矩阵 + ROC）覆盖二分类的核心诊断维度，OOB 得分提供训练内的泛化参考。
- 未实现并非"做不到"——而是教学型流水线有意保持轻量，聚焦于最核心的评估维度。
- 如果需要精确率/召回率/F1，可以在此基础上添加 `sklearn.metrics.classification_report` 一行代码。

## 5. Bagging vs 单模型分类器评估对比

| 评估维度 | Bagging | 单模型分类器（如 SVC、逻辑回归） |
|---|---|---|
| 硬分类评估 | 混淆矩阵——80 棵树投票结果 | 混淆矩阵——单个模型决策边界结果 |
| 概率评估 | ROC 曲线——80 棵树概率取平均 | ROC 曲线——单个模型的概率输出 |
| 训练内诊断 | **OOB 得分**——免费且独有 | 无——单模型没有 Bootstrap 采样 |
| 基学习器诊断 | `estimators_`——可逐个检查 80 棵树 | 无——只有一个模型 |
| 方差诊断 | OOB 得分与测试集准确率的差距反映方差大小 | 无法直接诊断方差 |

### 理解重点

- OOB 得分和 `estimators_` 是 Bagging 独有的诊断优势——单模型分类器没有。
- 混淆矩阵和 ROC 曲线是所有二分类器的共性评估——但 Bagging 的输出来自 80 棵树的聚合，本质上是"委员会"的集体表现。

## 常见坑

1. 只看 OOB 得分不看测试集表现——OOB 得分是泛化能力的参考，但最终评估应以测试集为准。
2. 忽略 OOB 得分与测试集准确率的差距——差距过大说明数据分布或切分方式有问题。
3. 认为混淆矩阵够好了就不需要 ROC 曲线——混淆矩阵只看一个阈值点，ROC 曲线看所有阈值下的表现，信息更全面。
4. 把"未实现"当成"做不到"——大部分"未实现"的评估指标都是一行代码的事。

## 小结

- Bagging 有三项评估输出：混淆矩阵（硬标签）→ ROC 曲线（软概率）→ OOB 得分（训练内免费估计）——三者从不同角度描述模型质量。
- OOB 得分是 Bagging 独有的诊断工具——无需额外划分验证集，利用 Bootstrap 采样天然产生的"没被抽中"样本做无偏估计。
- 当前评估策略是"可视化优先 + OOB 补足"——聚焦于最核心的诊断维度，保持教学型代码的轻量性。

# 工程实现

## 本章目标

1. 理解 Bagging 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径。
3. 了解输出文件结构和阅读顺序。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.bagging()` | 方法 | 生成高噪声双月牙二分类数据——`make_moons(noise=0.35)` |
| `train_model(...)` | 函数 | 构建并训练 `BaggingClassifier`——含 sklearn 版本兼容处理 |
| `run()` | 函数 | 端到端流水线编排——串联数据准备、标准化、训练、预测、可视化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制测试集混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制 ROC 曲线（条件：`predict_proba` 可用） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` → `data_generation/__init__.py` | 生成高噪声双月牙数据并导出为模块变量 `bagging_data` | 全局 `DataFrame`（500 行 × 3 列） |
| 模型训练层 | `model_training/ensemble/bagging.py` | 封装 `BaggingClassifier` 训练——含基学习器创建、sklearn 版本兼容、OOB 日志 | `BaggingClassifier` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/bagging.py` | 串联数据准备、标准化、训练、预测、评估——端到端入口 | 终端日志 + 调用可视化函数 |
| 可视化层 | `result_visualization/confusion_matrix.py`、`result_visualization/roc_curve.py` | 生成混淆矩阵热力图和 ROC 曲线图 | PNG 文件 |

### 理解重点

- 四层架构是本仓库所有算法的通用模式——数据生成、模型训练、流水线编排、结果可视化各司其职。
- 数据生成层使用 `@dataclass` 管理参数——可灵活调整噪声水平而不影响其他模块。
- 训练层使用 `@print_func_info` 和 `@timeit` 装饰器——自动打印函数调用信息和耗时。
- 流水线层是薄编排层——不包含算法逻辑，只负责串起各步骤。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `bagging_data.copy()` | — | `DataFrame`，形状 `(500, 3)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["label"])` | `DataFrame` | `DataFrame`，形状 `(500, 2)` | 分离特征 X |
| 3 | `data["label"]` | `DataFrame` | `Series`，形状 `(500,)` | 分离标签 y |
| 4 | `train_test_split(X, y, test_size=0.2, stratify=y)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 分层训练/测试切分 |
| 5 | `scaler.fit_transform(X_train)` | `DataFrame`，形状 `(400, 2)` | `ndarray`，形状 `(400, 2)` | 训练集标准化 |
| 6 | `scaler.transform(X_test)` | `DataFrame`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 测试集标准化 |
| 7 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `BaggingClassifier` | 训练含 80 棵树的 Bagging 模型 |
| 8 | `model.predict(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100,)` | 硬投票预测 |
| 9 | `plot_confusion_matrix(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 混淆矩阵可视化 |
| 10 | `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 2)` | `ndarray`，形状 `(100, 2)` | 软概率输出 |
| 11 | `plot_roc_curve(y_test, y_scores, ...)` | `(Series, ndarray)` | PNG 文件（条件） | ROC 曲线可视化 |

### 理解重点

- 步骤 5 和 6 的差异至关重要——`fit_transform` 在训练集上同时计算统计量和变换，`transform` 在测试集上仅使用训练集的统计量。
- 步骤 7 的内部并行训练 80 棵决策树——`n_jobs=-1` 利用全部 CPU 核心。
- 步骤 10-11 是条件执行的——`hasattr(model, "predict_proba")` 检查通过后才执行。对 Bagging 而言始终满足条件。

## 3. 数据依赖关系

```
bagging_data (全局 DataFrame)
    │
    ├─→ X = data.drop(columns=["label"])  ──┐
    ├─→ y = data["label"]                   │
    │                                        │
    ├─→ train_test_split(X, y, test_size=0.2, stratify=y)
    │   ├─→ X_train (400, 2) ──→ scaler.fit_transform() ──→ X_train_s ──┐
    │   ├─→ y_train (400,) ─────────────────────────────────────────────┤
    │   │                                                                  │
    │   ├─→ X_test (100, 2) ──→ scaler.transform() ──→ X_test_s ──┐      │
    │   └─→ y_test (100,) ─────────────────────────────────┐       │      │
    │                                                       │       │      │
    │   ┌───────────────────────────────────────────────────┘       │      │
    │   │                                                           │      │
    │   │  train_model(X_train_s, y_train) ──→ model               │      │
    │   │      │                                                     │      │
    │   │      ├─→ model.predict(X_test_s) ──→ y_pred ──┐          │      │
    │   │      │                                         │          │      │
    │   │      └─→ model.predict_proba(X_test_s) ──→ y_scores ──┐  │      │
    │   │                                                        │  │      │
    │   │  plot_confusion_matrix(y_test, y_pred, ...) ←─────────┘  │      │
    │   │  plot_roc_curve(y_test, y_scores, ...) ←─────────────────┘      │
    │   │                                                                   │
    │   └───────────────────────────────────────────────────────────────────┘
```

### 理解重点

- `y` 是最"忙碌"的变量——同时参与了训练（`y_train` → `train_model`）和两项评估（`y_test` → 混淆矩阵 + ROC 曲线）。
- 与 PCA/LDA 的数据依赖图对比——Bagging 的 `y` 承担了"训练目标"和"评估基准"双重角色，PCA 的标签仅用于可视化着色。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 混淆矩阵 | `outputs/bagging/confusion_matrix.png` | PNG | 测试集 100 个样本的硬投票分类结果热力图 |
| ROC 曲线 | `outputs/bagging/roc_curve.png` | PNG（条件） | 测试集概率输出的 ROC 曲线——含 AUC 标注 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + OOB 得分 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.bagging
```

### 输出

```text
============================================================
Bagging 分类流水线
============================================================
模型训练完成
n_estimators: 80
max_samples: 0.8
max_features: 1.0
bootstrap: True
OOB 得分: 0.8975
模型训练耗时: 0.32s

============================================================
Bagging 流水线完成！
============================================================
```

### 理解重点

- 终端是唯一能看到 OOB 得分的地方——不会单独存为文件。
- `outputs/bagging/` 目录在流水线运行时自动创建——无需手动创建。
- 混淆矩阵始终生成——ROC 曲线在 `BaggingClassifier` 下也始终生成（因为 `predict_proba` 始终可用）。

## 5. 训练层细节：sklearn 版本兼容

### 示例代码

```python
# 兼容不同 sklearn 版本
try:
    model = BaggingClassifier(
        estimator=base,          # sklearn ≥ 1.2
        n_estimators=n_estimators,
        ...
    )
except TypeError:
    model = BaggingClassifier(
        base_estimator=base,     # sklearn < 1.2
        n_estimators=n_estimators,
        ...
    )
```

### 理解重点

- sklearn 1.2 版本将 `base_estimator` 参数名改为 `estimator`——`try/except TypeError` 处理了这个 API 变化。
- 首次使用 `estimator` 参数名——如果当前 sklearn 版本 < 1.2，会抛出 `TypeError`，随后用旧参数名重试。
- 这两个分支创建的是完全相同的模型对象——只是参数名不同。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解数据来源和噪声设计意图
2. `model_training/ensemble/bagging.py` — 理解 Bagging 模型的构建和训练
3. `pipelines/ensemble/bagging.py` — 看清端到端流程
4. `result_visualization/confusion_matrix.py` — 了解混淆矩阵的实现
5. `result_visualization/roc_curve.py` — 了解 ROC 曲线的实现

## 常见坑

1. 直接修改 `bagging_data` 而不先 `copy()`——会污染其他模块引用的同一变量。
2. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
3. 混淆 `estimator` 和 `base_estimator` 参数名——当前源码已兼容，但写作评估代码时需要注意 sklearn 版本。
4. 忘记检查 `hasattr(model, "predict_proba")`——虽然 Bagging 始终支持，但换用其他模型时可能崩溃。

## 小结

- Bagging 工程实现遵循本仓库标准四层架构：数据生成层 → 模型训练层 → 流水线编排层 → 可视化层。
- `run()` 是薄编排函数——10 步调用串联数据准备、标准化、训练、预测和评估。
- sklearn 版本兼容（`estimator` vs `base_estimator`）和防御性 `hasattr` 检查是本实现的两个工程细节亮点。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 Bagging 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 Bagging 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 Bootstrap、方差缩减、OOB、Bagging vs Boosting 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 Bagging 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 Bagging 原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. Bootstrap 采样中，当 $m = N$ 时，单个样本未被抽中的概率约等于多少？这个概率对应的常数名称是什么？

2. Bagging 降低的是偏差还是方差？为什么必须选择完全生长的决策树（`max_depth=None`）作为基学习器？

3. OOB 得分的数学含义是什么？它为什么能作为泛化能力的无偏估计——无需额外划分验证集？

4. `n_estimators` 从 10 增加到 80 时，方差缩减的效果如何变化？从 80 增加到 200 呢？（提示：边际递减效应）

5. `max_samples=0.8` 和 `max_samples=1.0` 的区别是什么？哪个配置下子集间的多样性更大？为什么？

6. Bagging 和 Boosting 在训练方式（并行 vs 串行）、基学习器选择（强学习器 vs 弱学习器）、核心目标（降方差 vs 降偏差）上有哪些本质区别？

7. 当前 Bagging 流水线中，`hasattr(model, "predict_proba")` 条件检查的目的是什么？对 `BaggingClassifier` 而言，这个条件是否始终为 `True`？

## 2. 动手练习

### 练习 1：改变基学习器数量 `n_estimators`

将 `n_estimators` 分别设为 `1`、`5`、`10`、`50`、`80`、`200`，观察 OOB 得分和混淆矩阵的变化。

```python
# 修改 train_model 调用
model = train_model(X_train_s, y_train, n_estimators=5)   # 试试不同值
```

回答：从多少棵树开始，OOB 得分趋于稳定？单棵树（`n_estimators=1`）和 80 棵树的混淆矩阵有什么肉眼可见的差异？

### 练习 2：改变采样比例 `max_samples`

将 `max_samples` 分别设为 `0.3`、`0.5`、`0.8`、`1.0`，观察 OOB 得分的变化。

```python
model = train_model(X_train_s, y_train, max_samples=0.3)
```

回答：`max_samples` 越小，每个基学习器看到的样本越少——这对模型多样性和单个基学习器的偏差分别有什么影响？是否存在一个"最佳"采样比例？

### 练习 3：改变特征采样比例 `max_features`

将 `max_features` 从 `1.0` 改为 `0.5`（即每个子集只随机使用 1 个特征），观察效果变化。

```python
model = train_model(X_train_s, y_train, max_features=0.5)
```

回答：在仅有 2 个特征的情况下，`max_features=0.5`（随机使用 1 个特征）对模型性能有什么影响？如果在高维数据（如 100 个特征）上，`max_features < 1.0` 的意义是什么？

### 练习 4：关闭 Bootstrap 和 OOB

分别设置 `bootstrap=False` 和 `oob_score=False`，观察效果变化。

```python
# 关闭 Bootstrap——每棵树使用全部训练数据
model = train_model(X_train_s, y_train, bootstrap=False)

# 关闭 OOB 得分——训练后无法获取 oob_score_
model = train_model(X_train_s, y_train, oob_score=False)
```

回答：`bootstrap=False`（每棵树看到完全相同的训练数据）时，Bagging 还能降低方差吗？为什么？OOB 得分关闭后是否还能通过其他方式估计泛化能力？

### 练习 5：改变数据噪声水平

修改 `data_generation/ensemble.py` 中的 `bagging_noise` 参数（分别设为 `0.05`、`0.2`、`0.35`、`0.5`），重新运行流水线。

```python
# 在 data_generation/ensemble.py 中
class EnsembleData:
    bagging_noise: float = 0.05  # 试试 0.05, 0.2, 0.35, 0.5
```

回答：噪声为 `0.05`（极低）时，单棵决策树大约已经表现良好——此时 Bagging 的改善有多大？噪声为 `0.5`（极高）时，Bagging 是否仍然能有效平滑边界？这说明了什么？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24(2), 123-140. | Bagging 的原始论文——Bootstrap 聚合的理论基础和实验验证 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 8.7, 15. | ESL 教材——Bagging 和随机森林的数学推导与偏差-方差分解 |
| 3 | scikit-learn 官方文档 — [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. Chapter 7. | 实战教材——Bagging 和随机森林的实现与调参指南 |

## 常见坑

1. 把 `n_estimators=1` 当成"Bagging 的简化版"——此时没有集成、没有投票、没有方差缩减，就是一棵单决策树。
2. 认为 `max_samples` 越小越好——太小会导致每个基学习器偏差增大（样本太少无法学到完整模式），抵消方差缩减收益。
3. 忘记在修改 `bagging_noise` 后重新导入数据——`bagging_data` 是模块级变量，修改 `EnsembleData` 类的默认参数后需要重新实例化并调用 `bagging()`。
4. 混淆 `max_samples`（样本采样比例）和 `max_features`（特征采样比例）——两者共同决定子集与原始数据的差异度。

## 小结

- 7 个自检问题覆盖 Bagging 的核心概念：Bootstrap 概率、方差缩减、OOB 估计、参数选择、与 Boosting 对比。
- 5 个动手练习从不同角度探索 Bagging 的行为——改变基学习器数量、采样比例、特征比例、Bootstrap 开关、数据噪声。
- 4 篇参考文献从原始论文 → 教材 → 官方文档 → 实战指南构成完整的阅读路线。
