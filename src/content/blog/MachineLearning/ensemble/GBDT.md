---
title: GBDT 梯度提升树
date: 2026-05-19
category: 机器学习/集成学习
tags:
  - Scikit-learn
description: GBDT梯度提升树的数学原理、串行降偏差机制与完整工程实现。
image: https://img.yumeko.site/file/blog/cover/1780581751602.webp
status: published
---

# 数学原理

## 本章目标

1. 理解 GBDT 的加法模型结构——$M$ 棵树按学习率加权累加，逐步逼近真实函数。
2. 理解梯度提升的核心思想——每棵新树拟合前 $M-1$ 棵树的负梯度（残差方向）。
3. 理解学习率（shrinkage）的数学作用——控制每步更新幅度，防止过拟合。
4. 理解为什么 GBDT 使用浅层决策树（`max_depth=3`）——弱学习器是偏差缩减的前提。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 加法模型 | 模型结构 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x})$——$M$ 棵树按学习率加权累加 |
| 梯度提升 | 训练策略 | 第 $m$ 棵树拟合前 $m-1$ 棵树集成在损失函数上的负梯度方向 |
| 学习率收缩 | 正则化 | $\nu \in (0, 1]$ 控制每棵树的贡献——$\nu$ 越小越保守，泛化越好但需要更多树 |
| 对数损失 | 损失函数 | 多分类的交叉熵损失——$L = -\sum_{k=1}^{K} y_k \log p_k(\mathbf{x})$ |
| 随机梯度提升 | 采样策略 | `subsample < 1.0` 时每棵树只使用部分样本——引入随机性增强泛化 |

## 1. 加法模型

GBDT 的核心是一个**加法模型**——$M$ 棵决策树按学习率加权后累加：

$$
F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x}; \Theta_m)
$$

其中：
- $F_M(\mathbf{x})$ 是 $M$ 轮迭代后的集成模型输出
- $h_m(\mathbf{x}; \Theta_m)$ 是第 $m$ 棵决策树（当前为浅层树，`max_depth=3`）
- $\nu$ 是学习率（`learning_rate=0.1`）
- $\Theta_m$ 是第 $m$ 棵树的参数（分裂点、叶节点值等）

### 理解重点

- 加法模型意味着每棵树**直接与前序所有树的输出相加**——不是投票，不是平均，是累加。
- 学习率 $\nu$ 控制每棵树的贡献幅度——$\nu=0.1$ 意味着每棵树只贡献其完整输出的 10%。
- 与 Bagging 的对比：Bagging 是 $f_{\text{bag}} = \frac{1}{n}\sum f_b$（等权平均），GBDT 是 $F_M = \sum \nu h_m$（学习率加权累加）。

## 2. 梯度提升——在函数空间做梯度下降

GBDT 的训练策略可以理解为**在函数空间中执行梯度下降**。

### 前向分步算法

GBDT 以贪心方式逐棵添加树。第 $m$ 步，已知前 $m-1$ 棵树的集成 $F_{m-1}(\mathbf{x})$，寻找最优的新树 $h_m$ 使损失最小：

$$
h_m = \underset{h}{\arg\min} \sum_{i=1}^{N} L\big(y_i, F_{m-1}(\mathbf{x}_i) + \nu \cdot h(\mathbf{x}_i)\big)
$$

### 负梯度——"残差"的方向

直接求解上述优化问题很困难。GBDT 的巧妙之处在于——将损失函数 $L$ 对当前预测值 $F_{m-1}(\mathbf{x}_i)$ 求负梯度，作为新树的拟合目标：

$$
r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}
$$

这 $N$ 个负梯度值 $\{(\mathbf{x}_i, r_{im})\}_{i=1}^{N}$ 构成了第 $m$ 棵树的训练目标——树 $h_m$ 的任务是**逼近负梯度方向**。

### 理解重点

- 负梯度指向损失函数下降最快的方向——GBDT 在函数空间中向这个方向迈出步长 $\nu$。
- 对回归任务（平方损失），负梯度恰好等于残差 $y_i - F_{m-1}(\mathbf{x}_i)$——这也是"拟合残差"这一直觉说法的来源。
- 对分类任务（对数损失），负梯度是"伪残差"——不是简单的 $y_i - p$，而是损失对 log-odds 的导数。
- 这就是为什么 GBDT 的核心是**降偏差**——每棵新树专门修正前序集成犯的错误。

## 3. 对数损失（多分类）

当前 GBDT 处理的是 3 分类问题（$K = 3$），使用多分类对数损失（交叉熵）：

$$
L = -\sum_{k=1}^{K} y_k \log p_k(\mathbf{x})
$$

其中 $p_k(\mathbf{x})$ 是模型对类别 $k$ 的预测概率，由 softmax 函数从 $F_M$ 的原始输出转换而来。

### 理解重点

- 多分类 GBDT 内部实际上训练了 $K$ 组树——每组对应一个类别（one-vs-rest 风格，但共享梯度结构）。
- `GradientBoostingClassifier` 使用 `loss='log_loss'`（默认）——即多分类对数损失。
- 负梯度的形式取决于损失函数的选择——对数损失的负梯度是 $y_k - p_k$，即"真实概率 - 预测概率"。

## 4. 学习率收缩（Shrinkage）

学习率 $\nu$（`learning_rate`）是 GBDT 最重要的正则化参数：

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})
$$

### 理解重点

- $\nu$ 越小，每棵树的影响越小——需要更多的树（更大的 $M$）才能达到相同的拟合程度。
- 经验上，小 $\nu$ + 大 $M$ 的组合泛化效果更好——这就是为什么当前源码 `n_estimators=200` 搭配 `learning_rate=0.1`。
- $\nu$ 与 $M$ 存在权衡：$\nu=0.01$ 可能需要 $M=2000$ 棵树，$\nu=1.0$ 可能 $M=50$ 就过拟合。
- 与 Bagging 的对比：Bagging 没有学习率——每棵树等权投票，不需要缩放。

## 5. 随机梯度提升（Stochastic GBDT）

当 `subsample < 1.0` 时，每棵树只在随机抽取的部分训练样本上拟合——这被称为随机梯度提升：

$$
\mathcal{D}_m \subset \mathcal{D}, \quad |\mathcal{D}_m| = \text{subsample} \times N
$$

### 理解重点

- `subsample < 1.0` 同时在两个方面起作用：降低计算量、增加模型多样性（类似 Bagging 的 Bootstrap 思路）。
- 当前源码 `subsample=1.0`（默认值）——不使用随机梯度提升。设置为 `0.8` 可获得额外的方差缩减效果。
- 与 Bagging 的 `max_samples` 对比：Bagging 的每个子集完全独立且并行，GBDT 的子集是串行的——第 $m$ 棵树看到的数据子集不影响第 $m+1$ 棵树所见。

## 6. GBDT 与 Bagging 的数学对比

| 维度 | Bagging | GBDT |
|---|---|---|
| 模型结构 | $f_{\text{bag}} = \frac{1}{n}\sum f_b$（等权平均） | $F_M = \sum_{m=1}^{M} \nu h_m$（学习率加权累加） |
| 训练方式 | 并行——$n$ 棵树独立训练 | 串行——第 $m$ 棵树拟合前 $m-1$ 棵的负梯度 |
| 核心目标 | 降方差——$\text{Var}[f_{\text{bag}}] = \rho\sigma^2 + (1-\rho)\sigma^2/n$ | 降偏差——$F_M$ 逐步逼近 $F^*$ |
| 基学习器 | 强学习器（完全生长树，低偏差高方差） | 弱学习器（浅层树 `max_depth=3`，高偏差低方差） |
| 核心参数 | `n_estimators`、`max_samples` | `n_estimators`、`learning_rate`、`max_depth` |
| 正则化 | 并行平均天然正则化 | 学习率收缩 + 树深度限制 + subsample |
| 过拟合风险 | 低——投票平均天然平滑 | 较高——串行拟合可能过度追逐训练噪声 |
| 并行能力 | 天然可并行——各树独立 | 必须串行——每棵树依赖前序结果 |
| 独有诊断 | OOB 得分 | 特征重要性（`feature_importances_`） |

## 7. 数学原理如何映射到当前源码

| 数学概念 | 数学符号/公式 | 代码实现 |
|---|---|---|
| 加法模型 | $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ | `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)` |
| 基学习器（浅层树） | $h_m$：`max_depth=3` | `max_depth=3`（对比 Bagging 的 `max_depth=None`） |
| 学习率收缩 | $\nu$ | `learning_rate=0.1` |
| 负梯度（伪残差） | $r_{im} = -\partial L / \partial F$ | GBDT 内部自动计算——用户不可见 |
| 对数损失（多分类） | $L = -\sum_k y_k \log p_k$ | `loss='log_loss'`（默认值） |
| 随机梯度提升 | $\mathcal{D}_m \subset \mathcal{D}$ | `subsample=1.0`（当前未启用） |
| 特征重要性 | 基于分裂增益的加权平均 | `model.feature_importances_` |
| 标准化 | $z_j = (x_j - \mu_j)/\sigma_j$ | `StandardScaler`（训练集拟合/测试集变换） |
| 分层抽样 | 保持类别比例 | `train_test_split(stratify=y)` |

## 常见坑

1. 混淆 GBDT 与 Bagging 的基学习器选择——GBDT 用弱学习器（浅层树）降偏差，Bagging 用强学习器（深层树）降方差。
2. 把 `learning_rate` 设得过大——$\nu=1.0$ 时 GBDT 退化为无收缩的简单加法模型，极易过拟合。
3. 把 `n_estimators` 设得过小——$\nu=0.1$ 时 200 棵树是合理起点，10 棵树远不足以收敛。
4. 忽略学习率与树数量的耦合关系——$\nu$ 越小需要越大的 $M$，两者必须协同调整。

## 小结

- GBDT 的数学核心链：加法模型 $\to$ 函数空间负梯度 $\to$ 每棵树拟合伪残差 $\to$ 学习率收缩控制步长 $\to$ $M$ 轮后得到低偏差集成 $\to$ 特征重要性基于分裂增益。
- GBDT 降偏差而不降方差——因此需要高偏差低方差的基学习器（浅层决策树 `max_depth=3`）。
- 当前源码 `GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)` 是针对多类别分类数据最经典的 GBDT 配置。

# 数据构成

## 本章目标

1. 明确本仓库 GBDT 数据来自 `EnsembleData.gbdt()` 构造的多类别分类数据。
2. 理解为什么选择 8 特征 x 3 类别的数据——中等复杂度，充分展示 GBDT 串行纠错的偏差缩减能力。
3. 明确当前流程中的训练/测试切分（分层抽样）和标准化顺序。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.gbdt()` | 方法 | 生成 GBDT 使用的多类别分类数据 |
| `make_classification(...)` | 函数 | scikit-learn 提供的合成分类数据生成器 |
| `gbdt_data` | 变量 | 在 `data_generation/__init__.py` 中导出的 DataFrame |
| `gbdt_n_informative` | 参数 | 有效特征数 `4`——8 个特征中 4 个携带分类信号 |
| `gbdt_n_redundant` | 参数 | 冗余特征数 `2`——通过线性组合从有效特征生成 |
| `StandardScaler` | 类 | 对特征做 Z-score 标准化——训练集拟合、测试集变换 |

## 1. 数据生成：`EnsembleData.gbdt()`

当前 GBDT 数据来自 `EnsembleData.gbdt()`，底层调用 `sklearn.datasets.make_classification()`。

### 参数速览

适用函数：`make_classification(n_samples=500, n_features=8, n_informative=4, n_redundant=2, n_classes=3, class_sep=0.7, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_samples` | `int` | 总样本数。默认 `500`——适中规模，200 个 GBDT 弱学习器可在秒级完成串行训练 | `500`、`1000` |
| `n_features` | `int` | 总特征数。`8`——中等维度，提供足够的特征空间让 GBDT 展示特征选择能力 | `8`、`20` |
| `n_informative` | `int` | 有效特征数。`4`——只有一半特征携带真正的分类信号 | `4`、`8` |
| `n_redundant` | `int` | 冗余特征数。`2`——通过有效特征的线性组合生成 | `2`、`5` |
| `n_classes` | `int` | 类别数。`3`——多分类场景，比二分类更能展示 GBDT 的拟合能力 | `2`、`3`、`4` |
| `class_sep` | `float` | 类别间隔。`0.7`——中等难度，不是完全分离也不是严重重叠 | `0.5`、`0.7`、`1.0` |
| `random_state` | `int` | 随机种子，保证数据可复现。默认 `42` | `42` |
| 返回值 | `(ndarray, ndarray)` | `(X, y)` 元组，$X$ 形状 $(500, 8)$，$y$ 取值 $\{0, 1, 2\}$ | — |

### 示例代码

```python
X, y = make_classification(
    n_samples=500,
    n_features=8,
    n_informative=4,
    n_redundant=2,
    n_classes=3,
    class_sep=0.7,
    random_state=42,
)
data = DataFrame(X, columns=[f"x{i+1}" for i in range(8)])
data["label"] = y
```

### 理解重点

- 8 个特征中只有 4 个（`x1` - `x4`）携带真正的分类信号——剩余 4 个（`x5` - `x8`）是冗余或噪声特征。这为 GBDT 的特征重要性评估提供了有意义的测试场景。
- `class_sep=0.7` 是中等难度——类别有一定重叠但并非不可分。GBDT 的串行纠错能力在这种"中等混沌"中优势明显。
- 与 Bagging 的 `make_moons(noise=0.35)` 对比：Bagging 用 2 特征二分类高噪声数据展示降方差，GBDT 用 8 特征三分类中等难度数据展示降偏差。

## 2. 特征列与标签列

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(500, 8)$ | 含 8 个连续特征的特征矩阵，列名 `x1` - `x8` | `data.drop(columns=["label"])` |
| `y` | `Series`，形状 $(500,)$ | 三分类标签 $\{0, 1, 2\}$——参与 GBDT 训练和评估 | `data["label"]` |

### 特征一览

| 列名 | 特征类型 | 说明 |
|---|---|---|
| `x1` - `x4` | 有效特征（informative） | 携带分类信号——GBDT 特征重要性应对这 4 列给出较高值 |
| `x5` - `x6` | 冗余特征（redundant） | 由有效特征线性组合生成——重要性应低于 `x1` - `x4` |
| `x7` - `x8` | 噪声特征（noise） | 纯随机噪声——重要性应接近零 |
| `label` | 标签列 | 取值 $\{0, 1, 2\}$，三分类监督信号 |

### 示例代码

```python
X = data.drop(columns=["label"])
y = data["label"]
feature_names = list(X.columns)  # ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
```

### 理解重点

- `label` 是三分类监督标签——取值 $\{0, 1, 2\}$，参与 `model.fit()`、`model.predict()` 和混淆矩阵/ROC 评估。
- `feature_names` 在 GBDT 流水线中被显式提取——用于后续特征重要性图表的 x 轴标注。这是 Bagging 流水线中没有的步骤。
- 有效/冗余/噪声的三层特征结构是教学设计的亮点——它让特征重要性图表的解读有了"正确答案"做参照。

## 3. 训练/测试切分与标准化

### 参数速览

适用 API：`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X` | `DataFrame`，形状 $(500, 8)$ | 全量特征矩阵 | `X` |
| `y` | `Series`，形状 $(500,)$ | 全量标签 $\{0, 1, 2\}$ | `y` |
| `test_size` | `float` | 测试集比例。默认 `0.2` | `0.2`、`0.3` |
| `stratify` | `array_like` | 分层抽样依据——确保训练/测试集中三个类别的比例一致 | `y` |
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

- `stratify=y` 确保三个类别在训练/测试集中比例一致——对于三分类数据，这避免了某个类别在测试集中完全缺失。
- 标准化采用监督学习的标准做法：`fit_transform` 在训练集上计算 $\mu$ 和 $\sigma$，`transform` 在测试集上使用相同统计量——防止测试集信息泄露。
- 与 Bagging 的差异：GBDT 有 8 个特征（而非 2 个），标准化对 GBDT 来说也非必需（决策树不受尺度影响），但保留是为了流水线一致性。

## 数据可视化

![类别分布图](https://img.yumeko.site/file/blog/articles/1780737748247.png)

![特征相关性热力图](https://img.yumeko.site/file/blog/articles/1780736130799.png)

## 常见坑

1. 把 GBDT 的多分类数据当成"越复杂越好"——3 分类 + 中等间隔是教学平衡选择，过高的复杂度会掩盖算法特性。
2. 忽略特征的三层结构（有效/冗余/噪声）——这是理解特征重要性图表的"标准答案"。
3. 在测试集上 `fit_transform` 而非 `transform`——这是数据泄露的典型错误。
4. 认为 GBDT 不需要 `stratify=y`——多分类比二分类更容易出现类别不平衡问题，分层抽样更重要。

## 小结

- 当前 GBDT 数据来自 `make_classification(n_samples=500, n_features=8, n_informative=4, n_classes=3)`：8 个连续特征（4 有效 + 2 冗余 + 2 噪声）、三分类、中等难度。
- 数据流为：`make_classification` -> DataFrame（`x1` - `x8` + `label`）-> 分层训练/测试切分 -> 训练集拟合标准化器 / 测试集变换。
- 特征的三层结构（有效/冗余/噪声）为 GBDT 独有的特征重要性评估提供了"有标准答案"的验证场景。

# 思路与直觉

## 本章目标

1. 用直观方式理解 GBDT 的核心思路——"每个新成员专注于修正前人犯的错误"。
2. 理解为什么 GBDT 选择浅层决策树作为基学习器——弱学习器才有偏差可降。
3. 通过与 Bagging 和单棵决策树的对比，建立 GBDT 在集成学习谱系中的定位。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 串行纠错 | 训练方式 | 第 $m$ 棵树不看原始标签——它看的是前 $m-1$ 棵树"还剩多少没学好" |
| 梯度下降在函数空间 | 核心直觉 | 每棵树朝着损失下降最快的方向迈一小步——学习率控制步长 |
| 弱学习器 | 基学习器选择 | `max_depth=3` 的浅层树——单棵很弱，但串行累加后变强 |
| 偏差缩减 | 核心收益 | 每棵新树修正前序集成犯的错——步步逼近真实函数 |
| 学习率收缩 | 正则化直觉 | 每次只修正一点点——宁可慢，不可过 |
| 特征重要性 | 附加收益 | 基于分裂增益自动排序——知道哪些特征"说了算" |

## 1. 为什么需要 GBDT

单棵浅层决策树（`max_depth=3`）有一个天生缺陷：**太粗糙**。只分裂 3 次，最多 8 个叶子节点——复杂边界根本拟合不出来。这是高偏差。

直接加深树（`max_depth=None`）可以降低偏差——但方差会急剧上升（对数据微小变化极其敏感）。

GBDT 的思路很优雅：

> 既然一棵浅树太笨，那就让一群浅树接力。第一棵树粗糙拟合，第二棵树专修第一棵树的错误，第三棵树再修前两棵的错误——每棵树只学一点点，200 棵树累加起来，粗糙变精细。

### 理解重点

- 这就像用粗砂纸 -> 中砂纸 -> 细砂纸逐步打磨——每一遍只磨掉一点，最终表面极其光滑。
- GBDT 不降低方差（浅层树本身方差就不高），它只降低偏差——让粗糙的浅层树串行累加成精细的强模型。
- 因此 GBDT 的前提是：（1）基学习器偏差高但方差低（浅层树）；（2）串行训练每步只修正一点点（学习率控制步长）。

## 2. 用"接力打磨"理解 GBDT

GBDT 的工作方式可以想象成：

1. **第一个人粗略画轮廓**——第一棵树 `fit(X, y)`，只分裂 3 次，得到一个粗糙的分界面
2. **第二个人看残差**——第二棵树不看原始标签，而是看"第一个人哪里画得不够好"（残差 = 真实标签 - 当前预测概率）
3. **只修正一点点**——学习率 `0.1` 意味着只修正残差的 10%，避免修正过头
4. **重复 200 次**——200 棵树串行累加，最初的粗糙轮廓逐渐精确

### 理解重点

- 第一个人（第一棵树）可能只对了 60%——非常粗糙。
- 第二个人针对第一人犯的错修正——可能提高 5%。
- 第三个人再修正前两人遗留的错——可能提高 3%。
- 边际收益递减——第 200 棵树的贡献远小于第 2 棵。但累加起来，精度显著提升。
- 这就是偏差缩减的直觉：**粗糙轮廓 + 逐步精细化 = 最终精确边界**。

## 3. 为什么选择浅层决策树

GBDT 对基学习器的要求与 Bagging 相反：

- `max_depth=None`（完全生长）：**低偏差高方差**——单棵已经很好，串行累加无偏差可降，还可能过拟合
- `max_depth=3`（浅层树）：**高偏差低方差**——单棵很粗糙，但每次只修正一点点，200 次累加后极为精细
- `max_depth=1`（决策树桩）：偏差更高——需要更多树才能达到同等精度

### 理解重点

- GBDT 只能降偏差，不能降方差——因此必须从高偏差的基学习器出发。
- 浅层树对数据不敏感——两个不同子集产出的树结构相似，单个树方差很低。这正是 GBDT 需要的——方差不需要再降。
- 当前源码 `max_depth=3`——每一项参数都在鼓励树"浅尝辄止"，以高偏差换取低方差。

## 4. 学习率收缩的直觉：小步快跑

学习率 $\nu = 0.1$ 意味着每棵树只贡献其完整输出的 10%。

- $\nu = 1.0$：每棵树全力修正——容易修正过头（过拟合）
- $\nu = 0.1$：每棵树只修正 10%——需要更多树，但泛化更好
- $\nu = 0.01$：每棵树只修正 1%——需要大量树，训练更慢但可能泛化更好

### 理解重点

- 学习率是 GBDT 最重要的正则化手段——它控制"每一步迈多大"。
- `n_estimators=200` + `learning_rate=0.1` 是经典组合——总共 200 步，每步迈 0.1，总修正量可控。
- 这就像下坡——大步可能迈过头摔跤（过拟合），小步虽然慢但稳（泛化好）。

## 5. 与 Bagging 的直觉对比

| 维度 | Bagging | GBDT |
|---|---|---|
| 核心问题 | 如何让一群"太敏感"的专家冷静下来？ | 如何让一群"太迟钝"的新手变得聪明？ |
| 训练方式 | 平行——各学各的，最后投票 | 串行——后面的人补前面的人犯的错 |
| 基学习器 | 强学习器（完全生长树——偏差低方差高） | 弱学习器（浅层树 `max_depth=3`——偏差高方差低） |
| 核心收益 | 降方差——投票平滑了过拟合的锯齿 | 降偏差——接力修正了欠拟合的粗糙 |
| 过拟合风险 | 低——并行平均天然正则化 | 较高——串行纠错可能过度追逐噪声 |
| 并行能力 | 天然可并行——各树独立训练 | 必须串行——每棵树依赖前序结果 |
| 诊断工具 | OOB 得分（免费泛化估计） | 特征重要性 + 学习曲线 |
| 理想场景 | 高噪声数据 + 复杂基学习器 | 中等复杂度数据 + 简单基学习器 |

### 理解重点

- Bagging 和 GBDT 不是"谁更强"——Bagging 是针对"过敏感"的方差药方，GBDT 是针对"过迟钝"的偏差药方。
- 当前多类别中等难度数据（8 特征 x 3 类）+ 浅层树 `max_depth=3`，恰好是 GBDT 最理想、Bagging 难以发挥的场景（Bagging 需要完全生长树做基学习器）。

## 6. 特征重要性的直觉：谁在真正做决策

GBDT 训练完成后，`feature_importances_` 自动计算每个特征的重要性。

- 重要性的计算基于**分裂增益**——一个特征在 200 棵树中被用作分裂点的次数越多、每次分裂带来的损失降低越大，特征越重要。
- 当前数据有 4 个有效特征（`x1` - `x4`）、2 个冗余特征（`x5` - `x6`）、2 个噪声特征（`x7` - `x8`）。

### 理解重点

- 一个好的 GBDT 模型，`x1` - `x4` 的特征重要性应该显著高于 `x5` - `x8`。
- 特征重要性是 GBDT 的"自动特征选择"能力——不需要手动筛选特征，直接看重要性柱状图。
- Bagging 也有 `feature_importances_`（如果基学习器支持），但通常不如 GBDT 的稳定——因为 GBDT 的特征重要性来自 200 次有序选择，而非 80 次随机采样。

## 可视化

![混淆矩阵](https://img.yumeko.site/file/blog/articles/1780736298280.png)

![特征重要性](https://img.yumeko.site/file/blog/articles/1780736291957.png)

## 常见坑

1. 把 GBDT 当成"万能增强器"——它只对高偏差基学习器有效，对已低偏差的模型收效甚微。
2. 以为 `n_estimators` 越多越好——学习率固定时过多树会过拟合，应配合学习曲线判断。
3. 忽略 GBDT 与 Bagging 在基学习器选择上的本质差异——GBDT 用弱学习器（`max_depth=3`），Bagging 用强学习器（`max_depth=None`）。
4. 把学习率设得过大（$\nu = 1.0$）——失去了收缩正则化效果。

## 小结

- GBDT 的直觉核心是串行集成降偏差：浅层树粗糙拟合 -> 计算残差（负梯度） -> 下一棵树拟合残差 -> 学习率控制修正幅度 -> 200 次接力后粗糙变精细。
- 当前 8 特征三分类数据 + 浅层树 `max_depth=3` + `n_estimators=200` + `learning_rate=0.1` 是展示 GBDT 偏差缩减能力的最佳教学组合。
- GBDT 与 Bagging 在直觉上截然相反：一个靠串行接力把粗糙变精细（降偏差），一个靠并行投票把敏感变稳定（降方差）——选哪个取决于基学习器是"过于迟钝"还是"过于敏感"。

# 模型构建

## 本章目标

1. 明确 `train_model(...)` 如何构建并训练 `GradientBoostingClassifier`。
2. 理解 `GradientBoostingClassifier` 的核心构造器参数（`n_estimators`、`learning_rate`、`max_depth`、`subsample`）及其数学对应关系。
3. 看清训练完成后最重要的模型属性——`feature_importances_`（特征重要性）、`estimators_`（弱学习器列表）、`n_estimators_`（实际迭代数）。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `train_model(...)` | 函数 | 构建并训练一个 `sklearn.ensemble.GradientBoostingClassifier` 模型，打印超参数日志 |
| `GradientBoostingClassifier(...)` | 类 | scikit-learn 提供的 GBDT 分类器——通过串行梯度提升实现偏差缩减 |
| `model.fit(X_train, y_train)` | 方法 | 串行训练 $M$ 个弱学习器——每棵新树拟合前序集成的负梯度 |
| `model.feature_importances_` | 属性 | 特征重要性——基于分裂增益的加权平均 |
| `model.estimators_` | 属性 | 弱学习器集合——$M$ 个已完成 `fit()` 的浅层 `DecisionTreeRegressor` 对象（注意是回归树） |

## 1. `train_model(...)` 的函数签名

### 参数速览

适用函数：`train_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `array_like`，形状 `(400, 8)` | 标准化后的训练特征矩阵，传入 `GradientBoostingClassifier.fit()` | `X_train_s` |
| `y_train` | `array_like`，形状 `(400,)` | 训练标签 $\{0, 1, 2\}$——三分类监督信息 | `y_train` |
| `n_estimators` | `int` | 弱学习器（提升阶段）数量。`200`——配合 `learning_rate=0.1` 的经典配置 | `50`、`100`、`200`、`500` |
| `learning_rate` | `float` | 学习率——每棵树的贡献缩放因子。`0.1` 是经验默认值 | `0.01`、`0.05`、`0.1`、`1.0` |
| `max_depth` | `int` | 基学习器的最大深度。`3`——浅层树，高偏差低方差 | `1`、`3`、`5` |
| `subsample` | `float` | 随机梯度提升的采样比例。`1.0` 表示使用全部样本 | `0.5`、`0.8`、`1.0` |
| `random_state` | `int` | 随机种子，保证训练可复现。默认 `42` | `42` |
| 返回值 | `GradientBoostingClassifier` | 已完成 `fit()` 的模型对象，含 `feature_importances_`、`estimators_` 等 | — |

### 示例代码

```python
from model_training.ensemble.gbdt import train_model

model = train_model(X_train_s, y_train)
```

### 理解重点

- `train_model(...)` 是有监督训练——**必须有 `y_train` 参数**。GBDT 比 Bagging 更依赖标签——每棵树的拟合目标（负梯度）由标签和前序预测共同决定。
- `n_estimators=200` 和 `learning_rate=0.1` 是经典组合——200 次迭代、每次修正 10%，总修正量充足但不过量。
- 与 Bagging 的 `train_model` 对比：GBDT 没有 `bootstrap` 和 `oob_score` 参数（它们属于 Bagging 独有），但有 `learning_rate`（GBDT 独有）。

## 2. `GradientBoostingClassifier` 构造器参数

### 参数速览

适用 API：`GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=42)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `n_estimators` | `int` | 提升阶段数（弱学习器数量）。`200`——GBDT 的核心参数，配合学习率决定总修正量 | `50`、`100`、`200` |
| `learning_rate` | `float` | 学习率——每棵树的贡献缩放因子。$\nu \in (0, 1]$，越小越保守 | `0.01`、`0.1`、`1.0` |
| `max_depth` | `int` | 基学习器最大深度。`3`——浅层树，GBDT 标注配置 | `1`、`3`、`5` |
| `subsample` | `float` | 每棵树使用的样本比例。`1.0`——不启用随机梯度提升 | `0.5`、`0.8`、`1.0` |
| `loss` | `str` | 损失函数。默认 `'log_loss'`——多分类对数损失（交叉熵） | `'log_loss'`、`'exponential'` |
| `min_samples_split` | `int` | 内部节点再分裂的最小样本数。默认 `2` | `2`、`5`、`10` |
| `min_samples_leaf` | `int` | 叶节点的最小样本数。默认 `1` | `1`、`5` |
| `random_state` | `int` | 随机种子——保证训练可复现 | `42` |
| `verbose` | `int` | 日志详细程度。默认 `0` | `0`、`1` |

### 示例代码

```python
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    random_state=42,
)
model.fit(X_train, y_train)
```

### 理解重点

- `n_estimators` 和 `learning_rate` 是 GBDT 最重要的两个参数——它们共同决定总修正量：$M \times \nu$ 约等于"有效迭代次数"。`200 x 0.1 = 20` 等效步长。
- `max_depth=3` 是 GBDT 的标志性配置——与 Bagging 的 `max_depth=None` 形成鲜明对比。浅层树的高偏差是 GBDT 降偏差的前提。
- `subsample=1.0`（默认）——当前未启用随机梯度提升。若设为 `0.8`，每棵树随机使用 80% 样本，可同时获得方差缩减的额外收益。
- 与 Bagging 的参数对比：Bagging 有 `bootstrap`、`oob_score`、`n_jobs`（并行），GBDT 有 `learning_rate`、`loss`（学习率和损失函数）。

## 3. 训练完成后的关键属性

### 参数速览

| 属性名 | 类型 | 数学含义 | 说明 |
|---|---|---|---|
| `feature_importances_` | `ndarray`，形状 `(8,)` | 基于分裂增益加权的特征重要性 | 值越大越重要——可排序绘制柱状图 |
| `estimators_` | `list`，长度 `n_estimators` x `n_classes` | 弱学习器集合（注意：每类一组树） | 三分类下有 $200 \times 3 = 600$ 个回归树对象 |
| `n_estimators_` | `int` | 实际使用的提升阶段数 | 通常等于 `n_estimators`，除非触发早停 |
| `train_score_` | `ndarray`，形状 `(n_estimators_,)` | 每轮迭代后的训练集得分 | 用于诊断是否过拟合 |
| `n_classes_` | `int` | 类别数 | 当前为 `3` |
| `classes_` | `ndarray`，形状 `(3,)` | 类别标签 | `[0, 1, 2]` |
| `n_features_in_` | `int` | 特征维度 $d$ | 当前为 `8` |

### 示例代码

```python
print(f"n_estimators: {n_estimators}")
print(f"learning_rate: {learning_rate}")
print(f"max_depth: {max_depth}")
print(f"subsample: {subsample}")

# 特征重要性（管道外部使用）
importances = model.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")
```

### 理解重点

- `feature_importances_` 是 GBDT 独有的诊断优势——不需要额外的排列重要性或 SHAP 计算，训练完成后直接可用。
- `estimators_` 的结构与 Bagging 不同——多分类 GBDT 内部为每个类别维护一组树，三分类下 `estimators_` 含 $200 \times 3 = 600$ 棵回归树（注意是回归树，不是分类树）。
- `train_score_` 记录了每轮迭代后的训练集得分——可用来绘制训练曲线，判断是否需要更多树或更少树。

## 4. `predict()` 与 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `predict(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n,)`，取值 $\{0, 1, 2\}$ | 200 棵树加权累加后取最大概率类别 |
| `predict_proba(X)` | `array_like`，形状 `(n, 8)` | `ndarray`，形状 `(n, 3)` | softmax 概率输出——用于多分类 ROC 曲线 |

### 理解重点

- `predict()` 不是投票——是 200 棵树加权累加后取 softmax 最大值。
- `predict_proba()` 是 softmax 输出——每行 3 个概率值，和为 1。
- GBDT 始终支持 `predict_proba`——当前流水线直接调用，没有条件检查（不像 Bagging 的 `hasattr` 防御）。

## 常见坑

1. 把 GBDT 的 `n_estimators` 当成 Bagging 的 `n_estimators`——GBDT 需要更多树（200+），因为每棵树只修正一点点。
2. 忽略 `learning_rate` 与 `n_estimators` 的耦合——调整学习率必须同步调整树数量。
3. 把 GBDT 的基学习器设为深层树——`max_depth=None` 会导致串行过拟合极快。
4. 忘记 GBDT 的 `estimators_` 内部是回归树而非分类树——GBDT 拟合的是连续负梯度值，不是离散标签。

## 小结

- `train_model(...)` 是本仓库 GBDT 的核心训练入口，是对 `sklearn.ensemble.GradientBoostingClassifier` 的薄封装。
- `GradientBoostingClassifier` 的核心参数是 `n_estimators`（提升阶段数）、`learning_rate`（学习率收缩）、`max_depth`（基学习器深度）——三者共同决定偏差缩减的程度和泛化能力。
- 基学习器 `max_depth=3`（浅层树）是刻意选择——高偏差是 GBDT 降偏差的前提，与 Bagging 的完全生长树形成对比。
- 训练完成后的核心属性：`feature_importances_`（特征重要性诊断）、`estimators_`（600 棵回归树的集合）、`train_score_`（训练过程得分）——第一个是 GBDT 独有的特征选择工具。

# 训练与预测

## 本章目标

1. 理解 GBDT 流水线的完整执行流程——从数据拆解到模型预测。
2. 认清 GBDT 作为监督多分类算法的流程特征——有训练/测试切分、有 `y_train` 参与训练、有 `predict` 和 `predict_proba`。
3. 理解流水线中每一步的意图和与算法原理的对应关系。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `run()` | 函数 | GBDT 多分类端到端流水线入口——串联数据准备、标准化、训练、预测和四项评估 |
| `train_test_split(..., stratify=y)` | 函数 | 分层训练/测试切分——保证三个类别的比例一致 |
| `StandardScaler` | 类 | Z-score 标准化——`fit_transform` 在训练集、`transform` 在测试集 |
| `model.predict(X_test)` | 方法 | 200 棵树加权累加后 softmax 取最大——输出硬分类标签 |
| `model.predict_proba(X_test)` | 方法 | softmax 概率输出——直接用于多分类 ROC 曲线 |

## 1. `run()` 流水线总览

GBDT 流水线是一个典型的有监督多分类流程——与 Bagging 结构相似，但 GBDT 多了特征重要性（`feature_importances_`）和学习曲线两项评估。

### 参数速览

适用函数：`run()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 无参数 | — | `run()` 无参数——所有配置硬编码在函数体内 | — |
| 返回值 | `None` | 触发完整的 GBDT 训练+预测+四项评估+可视化流程 | — |

### 示例代码

```python
from pipelines.ensemble.gbdt import run

run()
```

或命令行：

```bash
python -m pipelines.ensemble.gbdt
```

### 理解重点

- `run()` 是薄流程编排层——每个步骤调用现有模块，本身不包含算法逻辑。
- 与 Bagging 流水线的关键差异在于评估环节：GBDT 有 4 项评估输出（混淆矩阵 + ROC + 特征重要性 + 学习曲线），Bagging 有 2 项（混淆矩阵 + ROC + OOB 日志）。
- `feature_names = list(X.columns)` 是 GBDT 流水线特有的步骤——为特征重要性图表提供 x 轴标注。

## 2. 数据准备：复制、拆解、切分

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 复制数据 | `data = gbdt_data.copy()` | 避免修改模块级全局变量 |
| 拆解 X/y | `X = data.drop(columns=["label"])`、`y = data["label"]` | 分离 8 个特征和 3 分类标签 |
| 提取特征名 | `feature_names = list(X.columns)` | 供特征重要性图表使用——`['x1', ..., 'x8']` |
| 分层切分 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | 80% 训练 + 20% 测试，按三类别比例分层 |

### 理解重点

- `feature_names` 是 GBDT 流水线独有的中间变量——Bagging 没有特征重要性图表，所以不需要这个步骤。
- `stratify=y` 对三分类场景尤其重要——避免某个类别在训练集或测试集中比例失衡。
- 8 个特征中只有 4 个有效——训练完成后特征重要性图表会揭示哪些特征"说了算"。

## 3. 标准化：训练集拟合、测试集变换

### 参数速览

| 步骤 | 代码 | 意图 |
|---|---|---|
| 训练集拟合 | `scaler.fit_transform(X_train)` -> `X_train_s` | 在训练集上计算 8 维 $\mu$ 和 $\sigma$，同时变换 |
| 测试集变换 | `scaler.transform(X_test)` -> `X_test_s` | 使用训练集的 $\mu$ 和 $\sigma$ 变换——防止数据泄露 |

### 理解重点

- 标准化对 GBDT 不是必需的——决策树天然不受特征尺度影响。但保留标准化是为了流水线一致性。
- 正确做法是 `fit_transform` 训练集、`transform` 测试集——如果在测试集上 `fit_transform`，会导致信息泄露。

## 4. 模型训练：`train_model(X_train_s, y_train)`

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train_s` | `ndarray`，形状 `(400, 8)` | 标准化后的训练特征 | `scaler.fit_transform(X_train)` |
| `y_train` | `Series`，形状 `(400,)` | 训练标签 $\{0, 1, 2\}$ | `y_train` |
| 返回值 | `GradientBoostingClassifier` | 已完成 `fit()` 的模型——含 200 x 3 = 600 棵回归树 | — |

### 理解重点

- `train_model(...)` **必须有 `y_train`**——GBDT 比 Bagging 更依赖标签，因为每个阶段的拟合目标（负梯度）由标签和当前预测共同决定。
- 训练过程是严格串行的——第 $m$ 棵树的训练依赖前 $m-1$ 棵的输出，无法并行化（与 Bagging 的 `n_jobs=-1` 形成对比）。
- 训练完成后终端打印 `n_estimators`、`learning_rate`、`max_depth`、`subsample`。

## 5. 预测：`predict()` 和 `predict_proba()`

### 参数速览

| 方法 | 输入 | 输出 | 机制 |
|---|---|---|---|
| `model.predict(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100,)`，取值 $\{0, 1, 2\}$ | 200 棵树加权累加 -> softmax -> argmax |
| `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100, 3)` | 200 棵树加权累加 -> softmax -> 三类概率 |

### 示例代码

```python
y_pred = model.predict(X_test_s)
# y_pred 形状 (100,)，取值 {0, 1, 2}

y_scores = model.predict_proba(X_test_s)
# y_scores 形状 (100, 3)，每行三个类别的 softmax 概率
```

### 输出

```text
# predict 输出示例（前 10 个预测）
[1 0 2 1 0 1 2 1 0 0]

# predict_proba 输出示例（前 3 行）
[[0.15 0.72 0.13]
 [0.81 0.10 0.09]
 [0.05 0.20 0.75]]
```

### 理解重点

- `predict()` 输出硬标签——200 棵树加权累加后取 softmax 最大概率对应的类别。
- `predict_proba()` 输出 softmax 概率——每行 3 个值之和为 1。直接用于多分类 ROC 曲线（one-vs-rest）。
- GBDT 流水线没有 `hasattr(model, "predict_proba")` 检查——直接调用，因为 `GradientBoostingClassifier` 始终支持概率输出。

## 6. 评估触发：四项输出

### 参数速览

| 步骤 | 触发条件 | 输入 | 输出 |
|---|---|---|---|
| 混淆矩阵 | **始终** | `y_test` + `y_pred` | `outputs/gbdt/confusion_matrix.png` |
| ROC 曲线 | **始终** | `y_test` + `y_scores` | `outputs/gbdt/roc_curve.png` |
| 特征重要性 | **始终** | `model` + `feature_names` | `outputs/gbdt/feature_importance.png` |
| 学习曲线 | **始终** | `GradientBoostingClassifier(...)` + `X_train_s` + `y_train` | `outputs/gbdt/learning_curve.png` |

### 理解重点

- 四项评估全部始终触发——没有条件判断（不像 Bagging 的 `hasattr` 检查）。
- 特征重要性是 GBDT 独有的评估——Bagging 没有此项。
- 学习曲线通过额外实例化一个 `GradientBoostingClassifier` 生成——不是用训练好的 `model`，而是用 `sklearn.model_selection.learning_curve` 进行交叉验证。

## 完整流程总结

```
gbdt_data.copy()
  - X = data.drop(columns=["label"])
  - y = data["label"]
  - feature_names = list(X.columns)
  - train_test_split(test_size=0.2, stratify=y)
    - X_train (400, 8)、y_train (400,)
    - X_test (100, 8)、y_test (100,)
  - StandardScaler
    - X_train_s = scaler.fit_transform(X_train)
    - X_test_s = scaler.transform(X_test)
  - model = train_model(X_train_s, y_train)
    - 终端打印: n_estimators, learning_rate, max_depth, subsample
  - y_pred = model.predict(X_test_s)               -> 混淆矩阵
  - y_scores = model.predict_proba(X_test_s)       -> ROC 曲线
  - plot_feature_importance(model, feature_names)  -> 特征重要性
  - plot_learning_curve(..., X_train_s, y_train)   -> 学习曲线
```

## 常见坑

1. 混淆 GBDT 的串行训练与 Bagging 的并行训练——GBDT 不能用 `n_jobs=-1` 并行训练多棵树（但 scikit-learn 在某些版本中支持并行化每棵树内部的分裂搜索）。
2. 在测试集上 `fit_transform` 而非 `transform`——数据泄露的经典错误。
3. 忘记 `feature_names` 的提取——没有特征名，特征重要性图表只有位置索引，可读性大打折扣。
4. 混淆 `predict` 的机制——GBDT 是加权累加后 softmax，不是投票。

## 小结

- GBDT 流水线是一个有监督多分类流程：数据拆解 -> 分层切分 -> 训练集拟合标准化/测试集变换 -> 串行梯度提升训练 -> 硬预测/软概率 -> 四项评估（混淆矩阵 + ROC + 特征重要性 + 学习曲线）。
- 与 Bagging 流水线的核心差异：（1）训练是串行而非并行；（2）有 `feature_names` 提取步骤；（3）有特征重要性和学习曲线两项额外评估；（4）没有 `hasattr` 条件判断。
- `run()` 是薄编排层——每步调用现有模块，自身不含算法逻辑。

# 评估与诊断

## 本章目标

1. 理解 GBDT 的四项评估输出——混淆矩阵、ROC 曲线、特征重要性、学习曲线，各自回答什么问题。
2. 理解特征重要性的计算原理和解读方法——它是 GBDT 独有的"自动特征选择"工具。
3. 理解学习曲线的诊断价值——判断模型是欠拟合、恰好还是过拟合。
4. 明确当前代码中评估的实现范围。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_confusion_matrix(...)` | 函数 | 在测试集上绘制混淆矩阵——评估多分类硬标签准确率 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 ROC 曲线（one-vs-rest）——评估概率输出的排序能力 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图——展示 8 个特征对分类的贡献排序 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线——训练集/测试集准确率随训练样本数变化 |
| `model.feature_importances_` | 属性 | 8 个特征的重要性值——基于 200 棵树中的分裂增益 |
| `model.train_score_` | 属性 | 每轮迭代的训练得分——可观察损失下降趋势 |

## 1. 混淆矩阵：多分类硬标签评估

混淆矩阵评估的是 `model.predict()` 的输出——200 棵树加权累加后 softmax 取最大的分类结果。

### 参数速览

适用 API：`plot_confusion_matrix(y_test, y_pred, title="GBDT 混淆矩阵", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 $\{0, 1, 2\}$ | `y_test` |
| `y_pred` | `ndarray`，形状 `(100,)` | 模型预测结果 | `model.predict(X_test_s)` |
| `title` | `str` | 图表标题 | `"GBDT 混淆矩阵"` |
| `dataset_name` | `str` | 数据集名称——用于输出路径 | `"gbdt"` |
| `model_name` | `str` | 模型名称——用于输出路径 | `"gbdt"` |
| 输出 | PNG 文件 | 3x3 混淆矩阵热力图 | `outputs/gbdt/confusion_matrix.png` |

### 理解重点

- 三分类混淆矩阵是 3x3 的表格——对角线是正确分类的样本，非对角线是错误分类。
- 与 Bagging 的二分类混淆矩阵（2x2）不同——三分类有更多错分组合，诊断信息更丰富。
- 对于 `class_sep=0.7` 的中等难度数据——对角线元素应该明显亮于非对角线元素，但可能不如 Bagging 的高噪声双月牙那样极端。

## 2. ROC 曲线：多分类概率评估

ROC 曲线评估的是 `model.predict_proba()` 的输出——多分类使用 one-vs-rest 策略，为每个类别单独绘制一条 ROC 曲线。

### 参数速览

适用 API：`plot_roc_curve(y_test, y_scores, title="GBDT ROC 曲线", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `Series`，形状 `(100,)` | 测试集真实标签 | `y_test` |
| `y_scores` | `ndarray`，形状 `(100, 3)` | softmax 概率输出——每行三类概率 | `model.predict_proba(X_test_s)` |
| `title` | `str` | 图表标题 | `"GBDT ROC 曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 含 3 条曲线的 ROC 图（每个类别一条 + 微平均/宏平均） | `outputs/gbdt/roc_curve.png` |

### 理解重点

- 多分类 ROC 为每个类别绘制一条曲线——可以看到模型在哪个类别上区分能力最强/最弱。
- 当前流水线直接调用 `predict_proba`，没有条件检查——`GradientBoostingClassifier` 始终支持概率输出。
- 与 Bagging 的二分类 ROC（只有一条曲线）不同——GBDT 的 ROC 图展示三条曲线的对比。

## 3. 特征重要性：谁在真正做决策

特征重要性是 GBDT 独有的诊断工具——基于 200 棵树中每个特征的分裂增益，自动排序特征贡献。

### 参数速览

适用 API：`plot_feature_importance(model, feature_names=feature_names, title="GBDT 特征重要性", dataset_name=DATASET, model_name=MODEL)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `model` | `GradientBoostingClassifier` | 已训练的 GBDT 模型——从中提取 `feature_importances_` | `model` |
| `feature_names` | `list[str]` | 8 个特征的名称列表 | `['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']` |
| `title` | `str` | 图表标题 | `"GBDT 特征重要性"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 水平或垂直柱状图——特征按重要性降序排列 | `outputs/gbdt/feature_importance.png` |

### 理解重点

- 好的模型应让 `x1` - `x4`（有效特征）的重要性显著高于 `x5` - `x8`（冗余和噪声特征）——这是数据设计的"标准答案"。
- 特征重要性来自 200 棵树的累积——不是单棵树的随机判断，比 Bagging 的特征重要性更稳定。
- 特征重要性是 GBDT 的"自动特征选择"——如果某些噪声特征重要性异常高，说明模型可能过拟合。
- Bagging 没有此项评估——因为 Bagging 的基学习器是并行随机采样，特征重要性不如 GBDT 稳定。

## 4. 学习曲线：诊断偏差-方差状态

学习曲线展示模型性能随训练样本数增加的变化——是诊断欠拟合/过拟合的标准工具。

### 参数速览

适用 API：`plot_learning_curve(GradientBoostingClassifier(n_estimators=100, random_state=42), X_train_s, y_train, ...)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `GradientBoostingClassifier` | 新实例化的 GBDT 模型（`n_estimators=100`，注意不同于主模型的 200） | `GradientBoostingClassifier(n_estimators=100, random_state=42)` |
| `X` | `ndarray`，形状 `(400, 8)` | 训练特征 | `X_train_s` |
| `y` | `Series`，形状 `(400,)` | 训练标签 | `y_train` |
| `title` | `str` | 图表标题 | `"GBDT 学习曲线"` |
| `dataset_name` | `str` | 数据集名称 | `"gbdt"` |
| `model_name` | `str` | 模型名称 | `"gbdt"` |
| 输出 | PNG 文件 | 训练集和交叉验证集准确率 vs 训练样本数 | `outputs/gbdt/learning_curve.png` |

### 理解重点

- 学习曲线使用交叉验证——在不同大小的训练子集上评估模型性能。
- 训练集准确率高 + 交叉验证准确率低 = **过拟合**（高方差）——两条曲线之间有大的间隙。
- 训练集准确率低 + 交叉验证准确率也低 = **欠拟合**（高偏差）——两条曲线都低且接近。
- 两条曲线收敛且都高 = **拟合良好**——GBDT 期望达到的状态。
- 注意学习曲线使用的 `n_estimators=100`（不同于主模型的 200）——以减少计算开销。

## 5. 当前代码已实现 vs 未实现的评估内容

### 已实现

| 评估项 | 输出形式 | 触发条件 |
|---|---|---|
| 混淆矩阵 | PNG 热力图（`outputs/gbdt/confusion_matrix.png`） | 始终 |
| ROC 曲线 | PNG 曲线图（`outputs/gbdt/roc_curve.png`） | 始终 |
| 特征重要性 | PNG 柱状图（`outputs/gbdt/feature_importance.png`） | 始终 |
| 学习曲线 | PNG 曲线图（`outputs/gbdt/learning_curve.png`） | 始终 |
| 训练超参数日志 | 终端打印（n_estimators、learning_rate、max_depth、subsample） | `train_model(...)` 调用 |

### 未实现（以及原因）

| 未实现的评估项 | 原因 |
|---|---|
| 准确率/精确率/召回率/F1 硬数字打印 | 教学型代码通过混淆矩阵可视化间接呈现 |
| 每轮迭代的训练损失曲线 | 需要从 `train_score_` 提取——但当前用学习曲线覆盖了类似需求 |
| 早停（early stopping） | 需要额外划分验证集——增加教学复杂度 |
| 与 Bagging 的性能定量对比 | 两个算法的数据和任务不同，直接对比不公平 |
| 特征交互效应分析 | 属于深度分析——超出教学型流水线范围 |
| SHAP 值 / 部分依赖图 | 需要额外依赖（shap 库）——当前保持最小依赖原则 |

### 理解重点

- GBDT 的评估体系比 Bagging 更丰富——四项图表输出覆盖了分类性能（混淆矩阵 + ROC）、特征诊断（特征重要性）和模型健康度（学习曲线）三个维度。
- "未实现"并非"做不到"——教学型流水线选择最具代表性的评估项，保持轻量。

## 6. GBDT vs Bagging 评估对比

| 评估维度 | Bagging | GBDT |
|---|---|---|
| 硬分类评估 | 混淆矩阵（2x2 二分类） | 混淆矩阵（3x3 三分类） |
| 概率评估 | ROC 曲线（1 条，二分类） | ROC 曲线（每类 1 条 + 平均，三分类） |
| 特征诊断 | 无 | **特征重要性**——基于分裂增益 |
| 模型健康度 | 无 | **学习曲线**——训练/验证准确率变化 |
| 训练内诊断 | **OOB 得分**——Bagging 独有 | `train_score_`——每轮迭代训练得分 |
| 基学习器诊断 | `estimators_`——80 棵分类树 | `estimators_`——600 棵回归树 |
| 评估项数量 | 3（混淆矩阵 + ROC + OOB） | 4（混淆矩阵 + ROC + 特征重要性 + 学习曲线） |

### 理解重点

- Bagging 有 OOB（免费泛化估计）但无特征重要性——因为并行随机采样的特征重要性不够稳定。
- GBDT 有特征重要性和学习曲线但无 OOB——因为串行训练没有"天然留出"的样本。
- 两者是互补的评估体系——各有各的优势诊断工具。

## 常见坑

1. 只看特征重要性柱状图的高度就下结论——高度受特征尺度影响，标准化后可比性更好。
2. 忽略学习曲线中两条曲线之间的间隙——间隙越大，过拟合越严重。
3. 把特征重要性当成因果关系——高重要性只说明"该特征被用于分裂很多次"，不说明因果关系。
4. 忽略 `n_estimators=100`（学习曲线）与 `n_estimators=200`（主模型）的差异——学习曲线的 GBDT 配置与主模型不同。

## 小结

- GBDT 有四项评估输出：混淆矩阵（多分类硬标签）-> ROC 曲线（one-vs-rest 软概率）-> 特征重要性（自动特征选择）-> 学习曲线（过拟合/欠拟合诊断）——四者从分类性能、特征贡献、模型健康度三个维度完整描述模型质量。
- 特征重要性是 GBDT 独有的诊断优势——基于 200 棵树的分裂增益累积，比 Bagging 的并行随机采样更稳定。
- 学习曲线是偏差-方差状态的标准诊断工具——两条曲线（训练集/验证集）的位置和间隙直接反映欠拟合或过拟合。

# 工程实现

## 本章目标

1. 理解 GBDT 流水线的模块分层——数据生成层、模型训练层、流水线编排层、可视化层。
2. 理清 `run()` 内部的函数调用链和数据流动路径。
3. 理解 GBDT 与 Bagging 在工程实现上的关键差异——串行训练、无版本兼容、更多评估项。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `EnsembleData.gbdt()` | 方法 | 生成多类别分类数据——`make_classification(n_classes=3, n_features=8)` |
| `train_model(...)` | 函数 | 构建并训练 `GradientBoostingClassifier`——无 sklearn 版本兼容处理 |
| `run()` | 函数 | 端到端流水线编排——串联数据准备、标准化、训练、预测和四项可视化 |
| `plot_confusion_matrix(...)` | 函数 | 绘制测试集混淆矩阵 |
| `plot_roc_curve(...)` | 函数 | 绘制多分类 ROC 曲线 |
| `plot_feature_importance(...)` | 函数 | 绘制特征重要性柱状图 |
| `plot_learning_curve(...)` | 函数 | 绘制学习曲线（训练集/验证集准确率变化） |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据生成层 | `data_generation/ensemble.py` -> `data_generation/__init__.py` | 生成多类别分类数据并导出为模块变量 `gbdt_data` | 全局 `DataFrame`（500 行 x 9 列） |
| 模型训练层 | `model_training/ensemble/gbdt.py` | 封装 `GradientBoostingClassifier` 训练——含超参数日志 | `GradientBoostingClassifier` 模型对象 |
| 流水线编排层 | `pipelines/ensemble/gbdt.py` | 串联数据准备、标准化、训练、预测和四项评估——端到端入口 | 终端日志 + 调用四个可视化函数 |
| 可视化层 | `result_visualization/confusion_matrix.py`、`roc_curve.py`、`feature_importance.py`、`learning_curve.py` | 生成四项评估图表 | 4 个 PNG 文件 |

### 理解重点

- GBDT 的可视化层比 Bagging 多了两个模块——`feature_importance.py` 和 `learning_curve.py`，体现 GBDT 更丰富的诊断能力。
- 训练层使用 `@print_func_info` 和 `@timeit` 装饰器——自动打印函数调用信息和耗时。
- 与 Bagging 对比：GBDT 的训练层没有 `try/except TypeError` 版本兼容处理——因为 `GradientBoostingClassifier` 的 API 更稳定。

## 2. `run()` 内部的函数调用链

### 参数速览

| 序号 | 调用 | 输入 | 输出 | 目的 |
|---|---|---|---|---|
| 1 | `gbdt_data.copy()` | — | `DataFrame`，形状 `(500, 9)` | 避免修改全局变量 |
| 2 | `data.drop(columns=["label"])` | `DataFrame` | `DataFrame`，形状 `(500, 8)` | 分离 8 维特征 X |
| 3 | `data["label"]` | `DataFrame` | `Series`，形状 `(500,)` | 分离三分类标签 y |
| 4 | `list(X.columns)` | `DataFrame` | `list[str]`，长度 8 | 提取特征名——供特征重要性图表使用 |
| 5 | `train_test_split(X, y, test_size=0.2, stratify=y)` | `(DataFrame, Series)` | `(X_train, X_test, y_train, y_test)` | 分层训练/测试切分 |
| 6 | `scaler.fit_transform(X_train)` | `DataFrame`，形状 `(400, 8)` | `ndarray`，形状 `(400, 8)` | 训练集标准化 |
| 7 | `scaler.transform(X_test)` | `DataFrame`，形状 `(100, 8)` | `ndarray`，形状 `(100, 8)` | 测试集标准化 |
| 8 | `train_model(X_train_s, y_train)` | `(ndarray, Series)` | `GradientBoostingClassifier` | 串行训练 200 棵浅层回归树 |
| 9 | `model.predict(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100,)` | 硬预测（加权累加 + softmax + argmax） |
| 10 | `plot_confusion_matrix(y_test, y_pred, ...)` | `(Series, ndarray)` | PNG 文件 | 混淆矩阵可视化 |
| 11 | `model.predict_proba(X_test_s)` | `ndarray`，形状 `(100, 8)` | `ndarray`，形状 `(100, 3)` | 软概率输出（softmax） |
| 12 | `plot_roc_curve(y_test, y_scores, ...)` | `(Series, ndarray)` | PNG 文件 | 多分类 ROC 曲线 |
| 13 | `plot_feature_importance(model, feature_names, ...)` | `(model, list)` | PNG 文件 | 特征重要性柱状图 |
| 14 | `plot_learning_curve(GradientBoostingClassifier(...), X_train_s, y_train, ...)` | `(estimator, ndarray, Series)` | PNG 文件 | 学习曲线 |

### 理解重点

- 步骤 4（`feature_names`）是 GBDT 独有的——Bagging 没有特征重要性评估，不需要这个步骤。
- 步骤 13 直接使用训练好的 `model` 提取 `feature_importances_`。
- 步骤 14 实例化了一个**新的** `GradientBoostingClassifier(n_estimators=100)`——不同于主模型的 200 棵树，以减少计算开销。
- 与 Bagging 的 11 步调用链对比，GBDT 多了 3 步（feature_names 提取 + 特征重要性 + 学习曲线）。

## 3. 数据依赖关系

```
gbdt_data (全局 DataFrame)
  - -> X = data.drop(columns=["label"])  ──-> feature_names = list(X.columns) ──┐
  - -> y = data["label"]                                                        │
  - -> train_test_split(X, y, test_size=0.2, stratify=y)                        │
    - -> X_train (400, 8) ──-> scaler.fit_transform() ──-> X_train_s ──┐       │
    - -> y_train (400,) ─────────────────────────────────────────────┤       │
    - -> X_test (100, 8) ──-> scaler.transform() ──-> X_test_s ──┐     │       │
    - -> y_test (100,) ─────────────────────────────────┐       │     │       │
    - train_model(X_train_s, y_train) ──-> model               │     │       │
      - -> model.predict(X_test_s) ──-> y_pred ──┐          │     │       │
      - -> model.predict_proba(X_test_s) ──-> y_scores ──┐  │     │       │
      - -> model.feature_importances_ ──-> + feature_names ──┼──┼─────┼───┐  │
      - plot_confusion_matrix(y_test, y_pred, ...) <-───────┘  │     │   │  │
      - plot_roc_curve(y_test, y_scores, ...) <-───────────────┘     │   │  │
      - plot_feature_importance(model, feature_names, ...) <-────────┘   │  │
      - plot_learning_curve(new_GBDT, X_train_s, y_train, ...) <-────────┘  │
```

### 理解重点

- `feature_names` 是一个独立的横向数据流——从数据准备阶段流向特征重要性可视化，不参与训练和预测。
- `y_train` 同时参与训练（`train_model`）和学习曲线（`plot_learning_curve`）——后者会再次进行交叉验证切分。
- 与 Bagging 的数据依赖图对比——GBDT 多了 `feature_names` 分支和 `feature_importances_` 分支，以及学习曲线的额外 `GradientBoostingClassifier` 实例。

## 4. 输出文件一览

### 参数速览

| 输出项 | 路径 | 格式 | 说明 |
|---|---|---|---|
| 混淆矩阵 | `outputs/gbdt/confusion_matrix.png` | PNG | 测试集 3x3 混淆矩阵热力图 |
| ROC 曲线 | `outputs/gbdt/roc_curve.png` | PNG | 多分类 ROC（one-vs-rest，每类一条 + 平均） |
| 特征重要性 | `outputs/gbdt/feature_importance.png` | PNG | 8 个特征的重要性排序柱状图 |
| 学习曲线 | `outputs/gbdt/learning_curve.png` | PNG | 训练/验证准确率 vs 训练样本数 |
| 终端日志 | 标准输出 | 文本 | 训练超参数 + 运行耗时 |

### 示例代码

```bash
python -m pipelines.ensemble.gbdt
```

### 输出

```text
============================================================
GBDT 分类流水线
============================================================
模型训练完成
n_estimators: 200
learning_rate: 0.1
max_depth: 3
subsample: 1.0
模型训练耗时: 2.15s

============================================================
GBDT 流水线完成！
============================================================
```

### 理解重点

- GBDT 输出 4 个 PNG 文件——比 Bagging 多 2 个（特征重要性 + 学习曲线）。
- 训练耗时通常比 Bagging 长——因为 200 棵树必须串行训练（Bagging 的 80 棵树可以并行）。
- 终端日志没有 OOB 得分——GBDT 没有 Bootstrap 采样，无 OOB 概念。

## 5. 训练层细节：与 Bagging 的对比

| 工程维度 | Bagging | GBDT |
|---|---|---|
| sklearn 版本兼容 | 有 `try/except TypeError`（`estimator` vs `base_estimator`） | 无——`GradientBoostingClassifier` 参数名稳定 |
| 训练并行 | `n_jobs=-1`——80 棵树并行 | 串行——每棵树依赖前序结果 |
| 基学习器创建 | 显式 `DecisionTreeClassifier(max_depth=None)` | 由 `GradientBoostingClassifier` 内部创建（用户只指定 `max_depth=3`） |
| OOB 得分 | 有 `oob_score_` | 无 |
| 日志内容 | `n_estimators, max_samples, max_features, bootstrap, OOB 得分` | `n_estimators, learning_rate, max_depth, subsample` |
| 返回模型 | `BaggingClassifier`（分类树基学习器） | `GradientBoostingClassifier`（回归树基学习器） |

### 理解重点

- GBDT 不需要显式创建基学习器——`GradientBoostingClassifier` 内部自动使用 `DecisionTreeRegressor`（注意是回归树，不是分类树）。
- GBDT 的训练是纯串行的——没办法像 Bagging 那样用 `n_jobs=-1` 并行训练多棵树。
- GBDT 的训练日志没有 OOB 得分——因为没有 Bootstrap 采样。

## 阅读顺序

1. `data_generation/ensemble.py` — 了解 `gbdt()` 的数据生成逻辑和参数设计
2. `model_training/ensemble/gbdt.py` — 理解 GBDT 模型的构建和串行训练
3. `pipelines/ensemble/gbdt.py` — 看清端到端流程和四项评估的串联
4. `result_visualization/confusion_matrix.py` — 了解混淆矩阵实现
5. `result_visualization/roc_curve.py` — 了解多分类 ROC 实现
6. `result_visualization/feature_importance.py` — 了解特征重要性图表实现
7. `result_visualization/learning_curve.py` — 了解学习曲线实现

## 常见坑

1. 直接修改 `gbdt_data` 而不先 `copy()`——会污染其他模块引用的同一变量。
2. 在测试集上使用 `fit_transform` 而非 `transform`——标准信息泄露。
3. 忘记提取 `feature_names`——没有特征名时特征重要性图表只有索引号。
4. 把 GBDT 的基学习器误解为分类树——GBDT 内部使用的是回归树（拟合连续负梯度值）。

## 小结

- GBDT 工程实现遵循本仓库标准四层架构：数据生成层 -> 模型训练层 -> 流水线编排层 -> 可视化层（含 4 个模块）。
- `run()` 是薄编排函数——14 步调用串联数据准备、标准化、串行训练、预测和四项评估。
- 与 Bagging 的三个关键工程差异：（1）训练串行不可并行化；（2）无需 sklearn 版本兼容处理；（3）多 2 项评估输出（特征重要性 + 学习曲线）。

# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 GBDT 核心概念的理解程度。
2. 通过动手练习在代码层面验证和探索 GBDT 的行为。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对加法模型、负梯度、学习率收缩、GBDT vs Bagging 等核心概念的理解 |
| 动手练习 | 实践 | 修改超参数观察 GBDT 行为变化——建立参数-效果的直觉 |
| 参考文献 | 入口 | 提供 GBDT 原始论文、教材章节和 scikit-learn 官方文档 |

## 1. 自检问题

1. GBDT 的加法模型 $F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu h_m(\mathbf{x})$ 与 Bagging 的投票平均 $f_{\text{bag}} = \frac{1}{n}\sum f_b$ 在结构上有何本质区别？

2. 为什么 GBDT 的每棵树拟合的是"负梯度"而不是原始标签 $y$？负梯度的直观含义是什么？

3. GBDT 降低的是偏差还是方差？为什么必须选择浅层决策树（`max_depth=3`）作为基学习器？

4. 学习率 $\nu=0.1$ 与 $\nu=1.0$ 的区别是什么？$\nu=0.1$ 时为什么需要更多的树（`n_estimators=200`）？

5. `subsample=0.8` 和 `subsample=1.0` 的区别是什么？引入随机子采样能带来什么额外收益？

6. GBDT 和 Bagging 在训练方式（串行 vs 并行）、基学习器选择（弱学习器 vs 强学习器）、核心目标（降偏差 vs 降方差）、独有诊断工具（特征重要性 + 学习曲线 vs OOB 得分）上有哪些本质区别？

7. 为什么 GBDT 的 `estimators_` 内部是回归树而非分类树？三分类场景下 `estimators_` 包含多少棵树？

## 2. 动手练习

### 练习 1：改变学习率 `learning_rate`

将 `learning_rate` 分别设为 `0.01`、`0.05`、`0.1`、`0.5`、`1.0`，同时调整 `n_estimators` 使总修正量（$\nu \times M$）大致相同，观察效果。

```python
# 保持总修正量约为 20
# learning_rate=0.01 -> n_estimators=2000
# learning_rate=0.05 -> n_estimators=400
# learning_rate=0.1  -> n_estimators=200
# learning_rate=0.5  -> n_estimators=40
# learning_rate=1.0  -> n_estimators=20

model = train_model(X_train_s, y_train, n_estimators=40, learning_rate=0.5)
```

回答：相同的总修正量下，大学习率 + 少树 vs 小学习率 + 多树，哪种组合的泛化效果更好？为什么？

### 练习 2：改变基学习器深度 `max_depth`

将 `max_depth` 分别设为 `1`、`3`、`5`、`10`、`None`，观察特征重要性和混淆矩阵的变化。

```python
model = train_model(X_train_s, y_train, max_depth=5)
```

回答：`max_depth` 增大后，模型是更倾向于 Bagging 还是保持了 GBDT 的特性？过深的基学习器对 GBDT 有什么危害？

### 练习 3：改变弱学习器数量 `n_estimators`

保持 `learning_rate=0.1`，将 `n_estimators` 分别设为 `10`、`50`、`100`、`200`、`500`，观察学习曲线的变化。

```python
model = train_model(X_train_s, y_train, n_estimators=10)
```

回答：从多少棵树开始测试集准确率趋于平稳？继续增加树数量是否会过拟合？

### 练习 4：启用随机梯度提升

将 `subsample` 从 `1.0` 改为 `0.8`，观察混淆矩阵和特征重要性的变化。

```python
model = train_model(X_train_s, y_train, subsample=0.8)
```

回答：`subsample=0.8` 对训练耗时和泛化能力分别有什么影响？这种影响与 Bagging 的 `max_samples=0.8` 有何异同？

### 练习 5：改变数据难度

修改 `data_generation/ensemble.py` 中的 `gbdt_class_sep` 参数（分别设为 `0.3`、`0.7`、`1.5`），重新运行流水线。

```python
# 在 data_generation/ensemble.py 中
class EnsembleData:
    gbdt_class_sep: float = 0.3  # 试试 0.3, 0.7, 1.5
```

回答：类别间隔越小，GBDT 相对于单棵决策树的优势是增大还是减小？为什么？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232. | GBDT 的原始论文——梯度提升框架的数学推导和泛化分析 |
| 2 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 10. | ESL 教材——Boosting 和加法模型的完整数学推导 |
| 3 | scikit-learn 官方文档 — [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) | API 参考——全部参数、属性和方法的详细说明 |
| 4 | Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. Chapter 7. | 实战教材——GBDT 的实现、调参与与 XGBoost 的对比 |

## 常见坑

1. 把 `n_estimators=10` 当成合理的 GBDT 配置——`learning_rate=0.1` 时 10 棵树的总修正量仅为 1.0 倍步长，远不足以收敛。
2. 认为 `max_depth` 越大越好——GBDT 的基学习器应是弱学习器，深度过大会破坏偏差缩减的串行机制。
3. 忘记 `learning_rate` 与 `n_estimators` 的耦合——调整一个必须同步调整另一个。
4. 把特征重要性当成因果关系的证明——高重要性只是"使用频率高"的统计描述。

## 小结

- 7 个自检问题覆盖 GBDT 的核心概念：加法模型、负梯度、偏差缩减、学习率收缩、随机梯度提升、与 Bagging 对比、回归树基学习器。
- 5 个动手练习从不同角度探索 GBDT 的行为——改变学习率、基学习器深度、树数量、子采样、数据难度。
- 4 篇参考文献从原始论文（Friedman 2001）-> 教材 -> 官方文档 -> 实战指南构成完整的阅读路线。
