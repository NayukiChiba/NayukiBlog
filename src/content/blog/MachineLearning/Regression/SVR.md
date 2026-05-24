---
title: SVR 支持向量回归
date: 2026-04-14
category: MachineLearning/Regression
tags:
  - Scikit-learn
description: SVR支持向量回归的数学原理、ε-管道损失与完整工程实现。
image: https://img.yumeko.site/file/blog/SVR.png
status: published
---

# 数学原理

## 本章目标

1. 理解 SVR 的 ε-不敏感损失函数——管道内误差不计，管道外线性惩罚。
2. 理解原始优化问题中的正则项 $\frac{1}{2}\|\mathbf{w}\|^2$ 与松弛变量 $\xi_i, \xi_i^*$ 的分工。
3. 理解对偶问题中核技巧的引入——内积替换为 $K(\mathbf{x}_i, \mathbf{x}_j)$ 实现非线性映射。
4. 将数学符号中的 $C$、$\epsilon$、$\gamma$ 与源码中的 `C=10.0`、`epsilon=0.1`、`gamma='scale'` 精确对应。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| ε-不敏感损失 | 损失函数 | $L_\epsilon(y, f(\mathbf{x})) = \max(0, \|y - f(\mathbf{x})\| - \epsilon)$——管道内误差为零 |
| $\frac{1}{2}\|\mathbf{w}\|^2$ | 正则项 | 控制模型平滑度——$\|\mathbf{w}\|$ 越小，函数越平坦 |
| $C$ | 超参数 | 正则化强度的倒数——越大越强调拟合，越小越强调平滑 |
| RBF 核 | 核函数 | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$——隐式高维映射 |
| 支持向量 | 概念 | 拉格朗日乘子 $\alpha_i - \alpha_i^* \neq 0$ 的样本——仅这些样本参与预测 |

## 1. ε-不敏感损失函数

SVR 的核心创新：不惩罚管道内的误差。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\epsilon$ | 标量 | 管道半宽——$2\epsilon$ 范围内误差不计损失 | `epsilon=0.1` |
| $L_\epsilon$ | 函数 | $\max(0, \|y - f(\mathbf{x})\| - \epsilon)$——超出管道才线性惩罚 | — |

$$
L_\epsilon(y, f(\mathbf{x})) = \max(0, |y - f(\mathbf{x})| - \epsilon)
$$

### 理解重点

- 当预测误差在 $[-\epsilon, +\epsilon]$ 内时损失为零——SVR 主动"忽视"小误差。
- ε 越大管道越宽，更多样本落入管道内——模型更平滑，支持向量更少。
- 这与 OLS（所有误差都平方惩罚）和 Lasso/Ridge（所有误差都平方惩罚 + 系数惩罚）根本不同。

## 2. 原始优化问题

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\mathbf{w}$ | 向量 | 线性模型的系数——$\frac{1}{2}\|\mathbf{w}\|^2$ 控制平滑度 | — |
| $C$ | 标量 | 管道外误差的惩罚权重——正则化强度的倒数 | `C=10.0` |
| $\xi_i$ | 标量 | 管道上方的松弛变量——样本超出管道上界的量 | — |
| $\xi_i^*$ | 标量 | 管道下方的松弛变量——样本超出管道下界的量 | — |

$$
\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}(\xi_i + \xi_i^*)
$$

$$
\text{s.t.} \quad
\begin{cases}
y_i - \mathbf{w}^T\mathbf{x}_i - b \leq \epsilon + \xi_i \\
\mathbf{w}^T\mathbf{x}_i + b - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

### 理解重点

- $\frac{1}{2}\|\mathbf{w}\|^2$ 的意义：使拟合函数尽可能平坦（flatness）——这是 SVM/SVR 区别于其他回归方法的核心。
- $C\sum(\xi_i + \xi_i^*)$：管道外样本的惩罚——$C$ 越大，偏离管道的代价越高，模型越倾向于缩小管道覆盖所有样本。
- $C$ 是正则化强度的**倒数**——$C \to \infty$ 退化为硬间隔（无正则化），$C \to 0$ 趋向完全平坦。
- 样本在管道内 → $\xi_i = 0$ 且 $\xi_i^* = 0$ → 不计损失，不是支持向量。

## 3. 对偶问题与核技巧

引入拉格朗日乘子 $\alpha_i, \alpha_i^*$ 后，对偶形式将 $\mathbf{w}$ 表示为训练样本的线性组合。

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\alpha_i, \alpha_i^*$ | 标量 | 拉格朗日乘子——约束在 $[0, C]$ 之间 | — |
| $K(\mathbf{x}_i, \mathbf{x}_j)$ | 函数 | 核函数——隐式计算高维特征空间的内积 | `kernel='rbf'` |
| $\gamma$ | 标量 | RBF 核宽度——$\gamma$ 越大影响范围越局部 | `gamma='scale'` |

对偶问题：

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)K(\mathbf{x}_i, \mathbf{x}_j)
- \epsilon\sum_i(\alpha_i + \alpha_i^*) + \sum_i y_i(\alpha_i - \alpha_i^*)
$$

$$
\text{s.t.} \quad \sum_i(\alpha_i - \alpha_i^*) = 0, \quad 0 \leq \alpha_i, \alpha_i^* \leq C
$$

RBF 核：

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$

### 理解重点

- 核技巧的本质：不显式计算 $\phi(\mathbf{x})$，而是直接计算内积 $K(\mathbf{x}_i, \mathbf{x}_j) = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$。
- RBF 核将数据映射到**无限维**空间——任何连续函数在理论上都可以被 RBF 核的 SVR 逼近。
- $\gamma$ 控制每个样本的影响半径：$\gamma$ 大 → 影响局部 → 可能过拟合；$\gamma$ 小 → 影响全局 → 可能欠拟合。
- `gamma='scale'` = $1/(d \cdot \text{Var}(X))$——scikit-learn 根据特征方差自动缩放。

## 4. 预测函数

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $f(\mathbf{x})$ | 函数 | 对新样本的预测——仅支持向量的加权核函数求和 | `model.predict(X_test_s)` |

$$
f(\mathbf{x}) = \sum_{i=1}^{N}(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b
$$

### 理解重点

- 预测不是 $\mathbf{w}^T\mathbf{x} + b$ 的矩阵乘法——而是支持向量与测试点的核函数加权和。
- $\alpha_i - \alpha_i^* = 0$ 的样本对预测**零贡献**——只有支持向量参与计算。
- 预测复杂度 $O(N_{\text{SV}} \cdot N_{\text{test}} \cdot d)$——支持向量越多预测越慢。

## 5. SMO 类优化算法

SVR 的对偶问题是一个带约束的二次规划。scikit-learn 使用 SMO（Sequential Minimal Optimization）类算法求解：

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 优化变量 | — | $\alpha_i, \alpha_i^*$——共 $2N$ 个变量 | — |
| 约束 | — | $\sum(\alpha_i - \alpha_i^*) = 0$，$0 \leq \alpha_i, \alpha_i^* \leq C$ | — |
| `tol` | `float` | 停止条件的对偶间隙容忍度 | `1e-3`（scikit-learn 默认） |
| `max_iter` | `int` | 最大迭代次数 | `-1`（无限制，scikit-learn 默认） |

### 理解重点

- SMO 每次只优化两个拉格朗日乘子——其余固定，子问题有解析解。
- 训练复杂度约 $O(N^2 \cdot d)$ 到 $O(N^3)$——样本量超过万级时训练显著变慢。
- 当前 200 样本的 SMO 求解几乎瞬时完成。

## 6. 数学概念与代码实现的映射

| 数学概念 | 数学符号 | 代码实现 |
|---|---|---|
| 特征矩阵 | $\mathbf{X} \in \mathbb{R}^{N \times d}$ | `X_train_s`——标准化后形状 `(160, 10)` |
| 目标向量 | $\mathbf{y} \in \mathbb{R}^N$ | `y_train`——形状 `(160,)` |
| 管道半宽 | $\epsilon$ | `SVR(epsilon=0.1)` |
| 惩罚系数（正则化倒数） | $C$ | `SVR(C=10.0)` |
| 核函数 | $K(\cdot, \cdot)$ | `SVR(kernel='rbf')` |
| RBF 核宽度 | $\gamma$ | `SVR(gamma='scale')` — 即 $1/(d \cdot \text{Var}(X))$ |
| 拉格朗日乘子差值 | $\alpha_i - \alpha_i^*$ | `model.dual_coef_`——形状 `(1, nSV)` |
| 支持向量索引 | $\{i : \alpha_i - \alpha_i^* \neq 0\}$ | `model.support_` |
| 支持向量数量 | $\|\{i : \alpha_i - \alpha_i^* \neq 0\}\|$ | `model.support_.shape[0]` |
| 截距 | $b$ | `model.intercept_`——标量 |

## 7. SVR vs 线性回归 vs 正则化回归 数学对比

| 数学维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 损失函数 | 平方损失 $\|y - \hat{y}\|^2$ | 平方损失 + $\lambda R(\mathbf{w})$ | **ε-不敏感损失——管道内不计** |
| 正则化 | 无 | L1/L2 系数惩罚 | **$\frac{1}{2}\|\mathbf{w}\|^2$ 平坦性惩罚** |
| 对偶形式 | 不需要——闭式解 | 不需要（Ridge 有闭式解，Lasso 用原问题） | **需要——引入核技巧的途径** |
| 核函数 | 无 | 无 | **RBF 核——映射到无限维空间** |
| 求解方法 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| 稀疏性 | 无 | Lasso: 系数稀疏 | **支持向量稀疏——仅部分样本参与预测** |
| 预测公式 | $\mathbf{X}\mathbf{w} + b$ | $\mathbf{X}\mathbf{w} + b$ | **$\sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 参数可解释性 | coef_ 直接对照 | coef_ 观察稀疏化 | **无法直接解释各特征贡献（RBF 核）** |

## 常见坑

1. 将 $C$ 理解为正则化强度——$C$ 是正则化的**倒数**，$C$ 越大正则化越弱，越容易过拟合。
2. 忘记 SVR 需要标准化——RBF 核基于欧氏距离，特征量纲不一致会导致某些维度主导核计算。
3. 将支持向量稀疏与 Lasso 系数稀疏混为一谈——前者的零是"不参与预测的样本"，后者的零是"不参与决策的特征"。
4. 期待 SVR 输出 `coef_` 做特征重要性——RBF 核的 SVR 没有 `coef_`，权重存在于对偶空间。

## 小结

- SVR 的数学核心是三层结构：ε-管道损失 + 平坦性正则化 + 核技巧非线性映射。
- ε-管道使管道内样本不计损失——它们不是支持向量，不参与预测，实现样本层面的稀疏性。
- RBF 核将数据隐式映射到无限维空间——使 SVR 能拟合任意非线性关系，但代价是可解释性下降。
- 数学公式中的 $C$、$\epsilon$、$\gamma$ 直接映射到源码中的 `C=10.0`、`epsilon=0.1`、`gamma='scale'`。


# 数据构成

## 本章目标

1. 明确 SVR 数据的来源——`loadSvrDataset()` 使用 `make_friedman1` 生成非线性合成数据。
2. 理解 Friedman1 的数据特性——前 5 维有效特征 + 后 5 维噪声特征 + 非线性目标函数。
3. 理解标准化在 SVR 中的必要性——RBF 核基于欧氏距离，对特征尺度敏感。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadSvrDataset()` | 方法 | 调用 `make_friedman1` 生成 200 样本 × 10 特征的非线性回归数据 |
| `make_friedman1(...)` | 函数 | scikit-learn 提供的 Friedman1 合成数据——目标函数高度非线性 |
| `x1`~`x10` | 列名 | 特征列——`x1`~`x5` 为有效特征，`x6`~`x10` 为噪声特征 |
| `price` | 列名 | 回归目标列——由 Friedman1 目标函数 + 噪声生成 |
| `StandardScaler` | 预处理 | Z-score 标准化——SVR（RBF 核）的强制前置步骤 |

## 1. 数据入口：`loadSvrDataset()`

### 参数速览

适用函数：`loadSvrDataset()`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `nSamples` | `int` | 样本总数——来自 `RegressionDatasetFactory` 默认属性 | `200` |
| `n_features` | `int` | 特征数——源码中固定为 10 | `10` |
| `noise` | `float` | 标签噪声标准差 | `1.0` |
| `randomState` | `int` | 随机种子——保证数据可复现 | `42` |
| 返回值 | `DataFrame` | 形状 `(200, 11)`——10 特征列 + 1 标签列 | — |

### 示例代码

```python
def loadSvrDataset(self) -> DataFrame:
    X, y = make_friedman1(
        n_samples=self.nSamples,
        n_features=10,
        noise=1.0,
        random_state=self.randomState,
    )
    columns = [f"x{i + 1}" for i in range(X.shape[1])]
    data = DataFrame(X, columns=columns)
    data["price"] = y
    return data
```

### 理解重点

- Friedman1 的目标函数是 $y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \varepsilon$——高度非线性，包含乘积项、平方项和正弦项。
- 前 5 个特征（`x1`~`x5`）参与目标函数的生成——是真正的信号特征。
- 后 5 个特征（`x6`~`x10`）是独立均匀随机变量——不参与目标函数，属于噪声特征。
- 这种"有效特征 + 噪声特征"的数据结构，使得 SVR 的 RBF 核价值得以体现——线性模型无法捕捉非线性信号。

## 2. Friedman1 数据特性

### 参数速览

| 特性 | 值 | 说明 |
|---|---|---|
| 样本数 | 200 | 小规模——适合 SVR 训练（SMO 复杂度 $O(N^2)$~$O(N^3)$） |
| 特征数 | 10 | 中等维度——5 个有效 + 5 个噪声 |
| 有效特征 | `x1`~`x5` | 参与目标函数的非线性组合 |
| 噪声特征 | `x6`~`x10` | 独立均匀分布——不含信号 |
| 目标函数 | $10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5$ | 含乘积、平方、正弦——高度非线性 |
| 标签噪声 | $\mathcal{N}(0, 1^2)$ | 叠加在目标函数上的高斯噪声 |

### 理解重点

- 目标函数中 $x_1$ 和 $x_2$ 以乘积形式出现——特征交互效应，线性模型无法捕捉。
- $\sin(\pi x_1 x_2)$ 是非线性的周期性分量——RBF 核的局部性天然适合拟合这种结构。
- 200 样本是刻意的小规模设计——SVR 的 SMO 训练复杂度随样本量平方增长，小样本保证训练瞬时完成。
- 与线性回归的合成数据（3 特征 × 纯线性公式）形成鲜明对比——SVR 的数据是故意非线性的。

## 3. 特征切分与标准化

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `test_size` | `float` | 测试集占比 | `0.2` |
| `random_state` | `int` | 切分随机种子 | `42` |
| 训练集形状 | — | `X_train`: `(160, 10)`, `y_train`: `(160,)` | — |
| 测试集形状 | — | `X_test`: `(40, 10)`, `y_test`: `(40,)` | — |
| `StandardScaler` | 预处理 | 将每列特征标准化为均值 0、标准差 1 | — |

### 示例代码

```python
X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 理解重点

- SVR 的 RBF 核基于欧氏距离 $\|\mathbf{x}_i - \mathbf{x}_j\|^2$——如果某特征量级远大于其他，会主导距离计算。
- 标准化使所有特征在核计算中拥有平等权重——这是 SVR 强制 `"standardScaler"` 的根本原因。
- 标准化在切分之后执行——`fit_transform` 仅用于训练集，`transform` 用于测试集。
- 与线性回归和决策树回归不同——它们不需要标准化（`PipelineSpec` 中预处理为 `None`）。

## 4. 数据特征总览

### 参数速览

| 属性 | 值 |
|---|---|
| 样本总数 | 200 |
| 特征总数 | 10（5 有效 + 5 噪声） |
| 训练样本数 | 160（80%） |
| 测试样本数 | 40（20%） |
| 标签列名 | `price` |
| 是否有缺失值 | 否——合成数据自动完整 |
| 数据来源 | `sklearn.datasets.make_friedman1` |
| 目标函数非线性程度 | 高——乘积 + 平方 + 正弦 |

### 理解重点

- 200 样本 × 10 特征——与线性回归的数据规模相同（200 × 3），但非线性结构完全不同。
- 5 有效 + 5 噪声的特征结构——与正则化回归的三层结构（原始 + 共线 + 噪声）思路相近，但 SVR 不追求观察特征选择而是观察核函数的非线性拟合。
- Friedman1 的目标函数已知——可以像线性回归一样通过对比真实公式评估模型（但非线性公式无法直接提取系数）。

## 5. 数据设计意图：与线性回归/决策树回归/正则化回归的对比

| 数据维度 | 线性回归 | 决策树回归 | 正则化回归 | SVR |
|---|---|---|---|---|
| 数据来源 | 手工合成 | 真实数据（California Housing） | 真实 + 人工干扰（diabetes + 共线 + 噪声） | **合成非线性（Friedman1）** |
| 样本量 | 200 | 20640 | 442 | **200** |
| 特征数 | 3 | 8 | 21（10+3+8） | **10（5 有效 + 5 噪声）** |
| 特征关系 | 完全独立 | 自然相关 | 刻意构造共线 | **前 5 维以乘积/平方/正弦耦合** |
| 标签生成 | 纯线性 + 高斯噪声 | 真实加州房价 | 真实糖尿病 + 噪声 | **非线性函数 + 高斯噪声** |
| 标准化 | 否 | 否 | **是** | **是** |
| 设计意图 | 透明验证 OLS | 大数据量树结构演示 | 观察稀疏化与收缩 | **展示 RBF 核的非线性拟合能力** |

## 常见坑

1. 忽略 Friedman1 的后 5 维是噪声——期待 SVR 对所有 10 个特征都学到"有意义"的权重。
2. 忘记 SVR 必须标准化——直接将原始 `X_train` 传入 `SVR.fit()`，RBF 核距离计算被量级大的特征主导。
3. 在切分之前标准化——`StandardScaler` 的均值和标准差包含了测试集信息，造成数据泄露。
4. 期待从 SVR 的 `coef_` 中读出特征重要性——RBF 核的 SVR 没有 `coef_` 属性。

## 小结

- SVR 数据来自 `make_friedman1`——200 样本 × 10 特征，目标函数高度非线性（乘积 + 平方 + 正弦）。
- 前 5 维为有效信号特征（`x1`~`x5`），后 5 维为噪声特征（`x6`~`x10`）——数据结构为展示 RBF 核的价值而设计。
- 标准化是 SVR 的强制前置步骤——RBF 核的欧氏距离对特征尺度敏感。
- 与线性回归数据的关键差异：SVR 的数据是非线性的——透明公式存在但无法从模型参数中直接提取。


# 思路与直觉

## 本章目标

1. 理解 SVR 的 ε-管道直觉——为什么管道内的样本不参与决策。
2. 理解 RBF 核的非线性映射直觉——如何在高维空间中找到线性划分。
3. 理解支持向量的稀疏性直觉——为什么模型复杂度由支持向量数量决定。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| ε-管道直觉 | 概念 | 管道宽度 $2\epsilon$——管道内的样本被"忽视"，管道外才受惩罚 |
| 平坦性直觉 | 概念 | $\frac{1}{2}\|\mathbf{w}\|^2$ 最小化——让拟合函数尽量平滑，不随样本剧烈弯曲 |
| RBF 核直觉 | 概念 | 每个支持向量是一个"影响中心"——RBF 核在支持向量周围形成局部响应 |
| 支持向量直觉 | 概念 | 只有管道边界上和管道外的样本才影响模型——稀疏性的来源 |
| $C$ 的权衡直觉 | 概念 | $C$ 小 → 宽管道/多容忍 → 平滑但可能欠拟合；$C$ 大 → 窄管道/少容忍 → 贴合但可能过拟合 |

## 1. 为什么当前数据适合讲 SVR

Friedman1 的目标函数是 $y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \varepsilon$——包含三种非线性结构。

### 参数速览

| 非线性类型 | Friedman1 中的体现 | 线性模型能否捕捉 | SVR（RBF）能否捕捉 |
|---|---|---|---|
| 特征交互 | $\sin(\pi x_1 x_2)$——$x_1$ 和 $x_2$ 以乘积耦合 | 否——需要手动构造交互项 | **是——RBF 核自动捕获** |
| 平方项 | $20(x_3 - 0.5)^2$——抛物线弯曲 | 否——需手动构造 $x_3^2$ | **是——核映射隐式构造** |
| 正弦周期 | $\sin(\pi x_1 x_2)$——周期性震荡 | 否 | **是——RBF 局部性适合拟合** |

### 理解重点

- 如果数据是线性的——SVR 与线性回归无异，核函数的非线性价值无从展示。
- Friedman1 刻意包含乘积、平方、正弦——三种典型的非线性结构，是展示 RBF 核能力的理想数据。
- 这份数据就是为"观察 SVR 如何处理非线性"而选择的。

## 2. ε-管道的直觉：忽视小误差，关注大偏差

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\epsilon$ | 管道半宽 | 误差在 $\pm\epsilon$ 内的样本不计损失 | `epsilon=0.1` |
| 管道内样本 | 概念 | $\|y - f(\mathbf{x})\| \leq \epsilon$——不计损失，不是支持向量 | — |
| 管道边界上/外样本 | 概念 | $\|y - f(\mathbf{x})\| \geq \epsilon$——成为支持向量 | — |

### 示例代码

```python
# 可视化 ε-管道概念
# 管道中心线: f(x)
# 管道上界: f(x) + epsilon
# 管道下界: f(x) - epsilon
# 管道内点: 无惩罚，非支持向量
# 管道外点: 线性惩罚，成为支持向量
```

### 理解重点

- 想象拟合曲线周围有一个宽度为 $2\epsilon$ 的"管道"——管道内的样本"不受惩罚"，管道外的样本才被惩罚。
- ε 是一种"宽容度"——告诉模型："误差在 ±0.1 以内就不算错"。
- 这与 OLS 形成鲜明对比——OLS 对所有误差（无论多小）都给予平方惩罚，SVR 对小误差零容忍零惩罚。
- ε 越大 → 管道越宽 → 更多样本被"放过" → 支持向量越少 → 模型越简单。

## 3. 平坦性的直觉：不仅要拟合，还要光滑

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| $\frac{1}{2}\|\mathbf{w}\|^2$ | 正则项 | 控制拟合函数的"弯曲程度"——$\|\mathbf{w}\|$ 越小越平坦 | — |
| $C$ | 权衡参数 | 拟合精度 vs 平坦性的权重——$C$ 大 = 更重视精度 | `C=10.0` |

### 理解重点

- SVR 与其他回归方法的根本哲学差异：SVR 不仅关心"预测是否准确"，还关心"拟合函数是否光滑"。
- 想象两张纸——一张平整，一张褶皱——SVR 倾向于平整的那张，即使它不能穿过所有数据点。
- $\|\mathbf{w}\|^2$ 越小，拟合函数越"不弯曲"——这在高维核空间中表现为更少的支持向量。
- $C$ 控制"拟合精度"与"平坦性"的权衡——$C=10.0$ 表示当前配置偏向拟合精度（较大 $C$ = 较少的正则化）。

## 4. RBF 核的直觉：每个支持向量是一个影响中心

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| RBF 核 | 函数 | $K(\mathbf{x}_i, \mathbf{x}) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}\|^2)$ | `kernel='rbf'` |
| $\gamma$ | 影响半径的倒数 | $\gamma$ 大 → 影响范围小（局部）；$\gamma$ 小 → 影响范围大（全局） | `gamma='scale'` |

### 理解重点

- RBF 核的本质：每个支持向量 $\mathbf{x}_i$ 是一个"影响源"——它对附近点的预测贡献大（$K \approx 1$），对远处点的贡献小（$K \approx 0$）。
- $\gamma$ 控制"附近"的定义——$\gamma$ 大 = 影响范围小，每个支持向量只影响极近邻的点 → 函数可以非常弯曲。
- $\gamma$ 小 = 影响范围大，每个支持向量影响远处 → 函数更平滑。
- 多个支持向量的核函数加权叠加，形成了最终的拟合曲面——这就像用一堆高斯钟形曲线拼接出目标函数。

## 5. 支持向量的三分类直觉

训练后，每个样本根据其在管道中的位置分为三类：

### 参数速览

| 样本类型 | 位置 | $\alpha_i - \alpha_i^*$ | 是否参与预测 |
|---|---|---|---|
| 管道内样本 | $\|y - f(\mathbf{x})\| < \epsilon$ | $= 0$ | **否——被模型忽略** |
| 管道边界上样本 | $\|y - f(\mathbf{x})\| = \epsilon$ | $\in (0, C)$ | **是——定义了管道位置** |
| 管道外样本 | $\|y - f(\mathbf{x})\| > \epsilon$ | $= C$ | **是——被惩罚的偏离点** |

### 理解重点

- 管道内的样本对模型**零影响**——这是 SVR 稀疏性的根源：不重要的样本被彻底忽略。
- 管道边界上的样本是"关键少数"——它们决定了拟合曲线的走向。
- 管道外的样本是"被惩罚的偏离点"——C 限制了它们的影响力上限。
- 支持向量数量 = 边界上样本数 + 管道外样本数——通常远小于总样本数。

## 6. C 和 ε 的联合调参直觉

### 参数速览

| 场景 | C | ε | 效果 |
|---|---|---|---|
| 追求高精度 | 大（如 100） | 小（如 0.01） | 窄管道 + 强惩罚 → 大量支持向量，可能过拟合 |
| 追求平滑 | 小（如 1） | 大（如 1.0） | 宽管道 + 弱惩罚 → 少量支持向量，可能欠拟合 |
| 当前默认 | 10.0 | 0.1 | 中等——适合 Friedman1 的教学展示 |

### 理解重点

- C 和 ε 不是独立的——调大 C 会让模型更努力覆盖管道外的点，可能需要调小 ε 来获得更细致的拟合。
- 先定 ε（根据目标变量的量级和能容忍的误差），再调 C（根据过拟合/欠拟合状态）。
- 支持向量数量是调参最直接的反馈——支持向量突然增多往往意味着 C 过大或 ε 过小。

## 7. 与线性回归和决策树回归的直觉对比

| 直觉维度 | 线性回归 | 决策树回归 | SVR |
|---|---|---|---|
| 拟合方式 | 全局直线/平面 | 局部 if-else 分段 | **全局核函数——支持向量加权叠加** |
| 误差态度 | 所有误差都被惩罚（平方） | 叶子均值——隐含惩罚 | **管道内不计——选择性惩罚** |
| 复杂度控制 | 固定（特征数） | max_depth 硬约束 | **C + ε + γ 软约束** |
| "看见"什么 | 全部样本平等 | 按分裂规则分区 | **只看支持向量——管道内样本被忽略** |
| 模型结构 | $\mathbf{w}$ 向量 | 二叉树 | **支持向量集 + RBF 核** |

## 常见坑

1. 以为 SVR 的"管道"会随训练数据自动选择最优宽度——ε 是超参数，需人工设定。
2. 忽略 $C$ 是正则化倒数——$C$ 越大 = 正则化越弱 = 越容易过拟合，直觉与其他模型的 α 相反。
3. 将 RBF 核的"无限维映射"理解成"一定能完美拟合"——无限维是理论能力，实际受 C、ε、γ 和样本量限制。
4. 认为支持向量越少越好——过少的支持向量可能意味着欠拟合（管道太宽）。

## 小结

- SVR 的核心直觉是三层：ε-管道提供宽容度，平坦性正则化提供光滑性，RBF 核提供非线性。
- ε-管道使大量样本被"忽略"——这是 SVR 稀疏性的根源，也是它与所有"所有样本都参与"的回归方法最根本的区别。
- RBF 核将每个支持向量变成一个"影响中心"——最终拟合是多个高斯钟形曲线的加权叠加。
- 支持向量数量是模型复杂度的直接反映——比 R² 或 MSE 更直观地告诉你模型"有多复杂"。


# 模型构建

## 本章目标

1. 理解 `trainSvrRegressionModel(...)` 如何构建并训练 `SVR`——本仓库最简训练函数之一（2 行）。
2. 理解 SVR 四个超参数——`C`、`epsilon`、`kernel`、`gamma`——的默认值及其选取理由。
3. 理解 `model.support_` 和 `model.dual_coef_` 的含义——SVR 的"参数"存在于对偶空间。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `trainSvrRegressionModel(...)` | 函数 | 构建并训练一个 `SVR` 模型——仅 2 行，比线性回归（3 行）更简 |
| `SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')` | 类 | scikit-learn 提供的 ε-支持向量回归器 |
| `model.fit(X_train_s, y_train)` | 方法 | SMO 类算法求解对偶问题——返回支持向量集 |
| `model.support_` | 属性 | 支持向量的训练集索引——模型复杂度的直接度量 |
| `model.dual_coef_` | 属性 | $\alpha_i - \alpha_i^*$ 的值——仅支持向量有非零值 |

## 1. `trainSvrRegressionModel(...)` 的函数签名

### 参数速览

适用函数：`trainSvrRegressionModel(XTrain, yTrain)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `XTrain` | `ndarray`，形状 `(160, 10)` | 标准化后的训练特征矩阵 | `X_train_s` |
| `yTrain` | `ndarray`，形状 `(160,)` | 训练目标值——Friedman1 生成 | `y_train` |
| 返回值 | `SVR` | 已完成 `fit()` 的模型对象——含 `support_` 和 `dual_coef_` | — |

### 示例代码

```python
from sklearn.svm import SVR

def trainSvrRegressionModel(XTrain, yTrain):
    model = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    model.fit(XTrain, yTrain)
    return model
```

### 理解重点

- 这是本仓库**最短的训练函数**——仅 2 行（线性回归 3 行），因为 `SVR` 的超参数在构造器中一次性给定。
- 与 `trainLinearRegressionModel` 对比：线性回归用无参 `LinearRegression()`，SVR 用含 4 个超参数的 `SVR(...)`。
- 函数签名没有 `randomState` 参数——SVR 的 SMO 求解是确定性的（对偶问题是凸优化）。

## 2. C：惩罚系数（正则化强度的倒数）

### 参数速览

适用 API：`SVR(C=10.0)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `C` | `float` | 管道外误差的惩罚权重——正则化强度 $\|\mathbf{w}\|^2$ 的倒数 | `10.0` |
| $C \to 0$ | — | 强正则化 → 极平坦函数 → 大量样本被容忍 → 少量支持向量 → 可能欠拟合 | — |
| $C \to \infty$ | — | 弱正则化 → 极力贴合数据 → 几乎所有管道外样本都被惩罚 → 大量支持向量 → 可能过拟合 | — |

### 理解重点

- $C$ 的作用方向与其他模型的 $\alpha$（正则化强度）**相反**——$C$ 是正则化的倒数，$C$ 越大正则化越弱。
- $C=10.0$ 在 Friedman1 数据上是中等偏大的值——偏向拟合精度，在 200 样本上通常不会严重过拟合。
- 调 $C$ 时最直接的反馈是支持向量数量——$C$ 增大会使更多边界样本成为支持向量。

## 3. epsilon：不敏感管道半宽

### 参数速览

适用 API：`SVR(epsilon=0.1)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `epsilon` | `float` | ε-不敏感管道半宽——误差在 ±ε 内不计损失 | `0.1` |
| ε 增大 | — | 管道更宽 → 更多样本被容忍 → 支持向量减少 → 模型更简单 | — |
| ε 减小 | — | 管道更窄 → 更少样本被容忍 → 支持向量增多 → 模型更复杂 | — |

### 理解重点

- `epsilon=0.1` 的取值需要结合目标变量的量级——Friedman1 的 $y$ 通常在 $[0, 25]$ 范围内，ε=0.1 是相对较小的管道。
- ε 和 $C$ 有交互效应——ε 决定"什么是误差"，$C$ 决定"误差有多大代价"。
- 与 OLS 的零容忍形成对比——OLS 相当于 ε=0 且用平方惩罚所有偏离。

## 4. kernel：核函数类型

### 参数速览

适用 API：`SVR(kernel='rbf')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `kernel` | `str` | 核函数类型——决定映射到什么样的特征空间 | `'rbf'` |
| `'rbf'` | — | 高斯径向基函数——$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | 默认——当前使用 |
| `'linear'` | — | 线性核——$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ | 退化为线性 SVR |
| `'poly'` | — | 多项式核——配合 `degree` 和 `coef0` 使用 | 需要额外调参 |
| `'sigmoid'` | — | Sigmoid 核——类似神经网络激活 | 较少使用 |

### 理解重点

- `kernel='rbf'` 是 SVR 的默认核——也是当前仓库的唯一配置。RBF 核的"无限维"特性使其能拟合任意连续函数。
- 线性核 SVR 等价于带 ε-管道损失的线性回归——失去了非线性能力，但保留了可解释性（有 `coef_`）。
- 当前仓库没有切换核函数的配置——`kernel='rbf'` 是硬编码的默认值。

## 5. gamma：RBF 核的宽度参数

### 参数速览

适用 API：`SVR(gamma='scale')`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `gamma` | `float` 或 `str` | RBF 核的 $\gamma$——控制单个样本的影响半径 | `'scale'` |
| `'scale'` | — | $\gamma = 1/(d \cdot \text{Var}(X))$——scikit-learn 自动按特征方差缩放 | 默认——当前使用 |
| `'auto'` | — | $\gamma = 1/d$——仅按特征维度缩放 | — |
| 数值（如 `0.1`） | — | 手动指定——越小影响范围越大 | — |

### 理解重点

- `gamma='scale'` 是 scikit-learn 0.22+ 的默认值——比旧版 `'auto'` 更稳定，考虑了特征方差。
- γ 是 RBF 核最敏感的超参数——小变化可能导致从欠拟合到过拟合的剧烈转变。
- γ 与 $C$ 有交互效应——γ 大（局部性强）+ $C$ 大（强拟合）极易导致过拟合。

## 6. 训练后的关键属性

### 参数速览

| 属性 | 类型 | 含义 | 诊断价值 |
|---|---|---|---|
| `support_` | `ndarray(nSV,)` | 支持向量在训练集中的索引 | **直接反映模型复杂度——nSV 越多越复杂** |
| `dual_coef_` | `ndarray(1, nSV)` | $\alpha_i - \alpha_i^*$ 的值 | 支持向量的权重——正值为上界支持向量，负值为下界 |
| `intercept_` | `float` | 截距 $b$ | 核函数加权和的偏置项 |
| `n_support_` | `ndarray` | 每类支持向量数量（回归中无意义） | — |
| `shape_fit_` | `tuple` | 训练数据形状 | — |

### 示例代码

```python
n_sv = model.support_.shape[0]
print(f"支持向量数量: {n_sv}")

# 支持向量数量占训练样本的比例
sv_ratio = n_sv / len(y_train)
print(f"支持向量占比: {sv_ratio:.1%}")
```

### 理解重点

- `support_` 是 SVR 最重要的属性——不亚于线性回归的 `coef_`。它告诉你模型"用了多少个样本做决策"。
- `dual_coef_` 中的值在 $[-C, +C]$ 之间——边界上的支持向量值在 $(-C, +C)$ 内，管道外的支持向量值 = $\pm C$。
- SVR（RBF 核）没有 `coef_` 属性——因为权重存在于对偶空间的 $\alpha_i - \alpha_i^*$ 中，无法直接映射回原始特征空间。

## 7. SVR vs 线性回归 vs 正则化回归 模型构建对比

| 模型维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 模型类 | `LinearRegression` | `Lasso` / `Ridge` / `ElasticNet` | **`SVR`** |
| 训练函数 | `trainLinearRegressionModel` | `trainRegularizationModels` | **`trainSvrRegressionModel`** |
| 函数行数 | 3 行 | ~10 行 | **2 行——最简** |
| 返回值 | 单个模型 | `dict`——三个模型 | **单个模型** |
| 超参数数 | 0 | 1~2 | **4（C, ε, kernel, γ）** |
| `random_state` | 不需要 | 需要（Lasso/EN） | **不需要——凸优化确定性** |
| 核心属性 | `coef_`, `intercept_` | `coef_`, `intercept_` + 近零计数 | **`support_`, `dual_coef_`, `intercept_`** |
| 是否有 `coef_` | **是** | **是** | **否（RBF 核）——权重在对偶空间** |
| 训练方式 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| 标准化 | 否 | **是** | **是** |

## 常见坑

1. 在 SVR（RBF 核）模型上访问 `model.coef_`——不存在此属性，应使用 `model.support_` 和 `model.dual_coef_`。
2. 认为 `C` 和其他模型的 `alpha` 方向一致——$C$ 是正则化倒数，增大 $C$ 意味着减弱正则化。
3. 忽略 `gamma='scale'` 不是固定值——它依赖输入数据的方差，不同数据集的 `'scale'` 值不同。
4. 将 `n_support_` 用于回归诊断——该属性为二分类 SVC 设计，回归中无参考价值。

## 小结

- `trainSvrRegressionModel(...)` 是本仓库最简训练函数——仅 2 行，但 `SVR(...)` 包含 4 个关键超参数。
- $C$（惩罚强度）、ε（管道宽度）、kernel（映射方式）、γ（核宽度）构成 SVR 的超参数体系——四者联合决定了模型行为。
- SVR（RBF 核）的模型参数不在 `coef_` 中——核心属性是 `support_`（哪些样本参与）和 `dual_coef_`（参与权重）。
- 支持向量数量是模型复杂度的最直观指标——不亚于决策树的深度和 Lasso 的近零系数数量。


# 训练与预测

## 本章目标

1. 理解 SVR 流水线的完整执行顺序——从数据加载到残差图和学习曲线输出。
2. 理解 SVR 的训练过程——SMO 迭代求解对偶问题，不可见但关键。
3. 理解 SVR 的预测方式——支持向量与测试点的核函数加权求和，而非矩阵乘法。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `loadSvrDataset()` | 方法 | 生成 Friedman1 非线性数据——返回 `(200, 11)` DataFrame |
| `StandardScaler` | 预处理 | Z-score 标准化——RBF 核距离计算的前置条件 |
| `trainSvrRegressionModel(...)` | 函数 | 构建并 `fit` SVR 模型——SMO 迭代求解 |
| `model.predict(X_test_s)` | 方法 | 支持向量与测试点的核函数加权求和预测 |
| `plot_residuals(...)` | 函数 | 残差诊断图 |
| `plot_learning_curve(...)` | 函数 | 学习曲线——使用新 `SVR(...)` 实例 |

## 1. 完整流水线流程

### 流程概述

```
loadSvrDataset()
    │
    ├─ ① X = data.drop(columns=["price"]), y = data["price"]
    ├─ ② X_train, X_test, y_train, y_test = train_test_split(test_size=0.2)
    ├─ ③ scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train)
    ├─ ④ X_test_s = scaler.transform(X_test)
    ├─ ⑤ model = trainSvrRegressionModel(X_train_s, y_train)
    ├─ ⑥ y_pred = model.predict(X_test_s)
    ├─ ⑦ plot_residuals(y_test, y_pred)
    └─ ⑧ plot_learning_curve(SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale'), X_train_s, y_train, scoring='r2')
```

### 参数速览

| 步骤 | 操作 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 加载数据 | `loadSvrDataset` | — | `DataFrame`，`(200, 11)` | Friedman1 非线性数据 |
| 特征标签拆分 | `drop` + 列选择 | `DataFrame` | `X(200,10)`, `y(200,)` | 标签列 `price` |
| 数据切分 | `train_test_split` | `X`, `y` | `X_train(160,10)`, `X_test(40,10)` | `test_size=0.2` |
| 标准化 | `StandardScaler` | `X_train`, `X_test` | `X_train_s`, `X_test_s` | **SVR（RBF 核）必需** |
| 训练 | `trainSvrRegressionModel` | `X_train_s`, `y_train` | `SVR` 模型 | SMO 迭代求解 |
| 预测 | `model.predict` | `X_test_s` | `y_pred(40,)` | 核函数加权求和 |
| 残差图 | `plot_residuals` | `y_test`, `y_pred` | PNG 图像 | 误差分布诊断 |
| 学习曲线 | `plot_learning_curve` | 新 `SVR(...)`, `X_train_s`, `y_train` | PNG 图像 | 样本量-得分趋势 |

### 理解重点

- SVR 流水线为 8 步——与线性回归（6 步）相比多了标准化（③④），与正则化回归（8 步）步骤数相同但无多模型循环。
- 标准化必须在切分之后——与正则化回归一致。
- SVR 没有特征重要性可视化——`PipelineSpec` 中训练后诊断列表为 `[]`。RBF 核的权重在对偶空间无法映射回原始特征。

## 2. 标准化：RBF 核训练的关键前置

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_train` | `ndarray(160, 10)` | 未标准化的训练特征 | 原始 Friedman1 各列 |
| `X_train_s` | `ndarray(160, 10)` | 标准化后——每列 μ=0, σ=1 | — |
| `X_test_s` | `ndarray(40, 10)` | 使用训练集统计量标准化 | — |

### 理解重点

- RBF 核计算 $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$——欧氏距离对尺度极敏感。若某特征量级为 100 而其他为 0.1，该特征将完全主导核计算。
- 标准化使所有特征在核距离中平等——这是 RBF 核 SVR 必须标准化的数学原因。
- `fit_transform` 仅用于训练集，`transform` 用于测试集——这是数据泄露的基本防护。

## 3. 训练细节：SMO 迭代求解

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| 优化算法 | — | SMO（Sequential Minimal Optimization） | scikit-learn 内部实现 |
| 优化变量 | — | $\alpha_i, \alpha_i^*$——共 $2 \times 160 = 320$ 个变量 | — |
| 目标 | — | 最大化对偶问题——凸二次规划 | — |
| 终止条件 | — | 对偶间隙 < `tol`（默认 1e-3） | `tol=1e-3` |

### 理解重点

- SVR 的训练是**不可见的迭代过程**——不像决策树那样可以观察分裂步骤，也不像线性回归的一步到位。
- 训练复杂度约 $O(N^2 \cdot d)$ 到 $O(N^3)$——在 160 样本上毫秒级完成。
- 训练过程是**确定性**的——对偶问题为凸优化，给定相同数据必然收敛到相同解。因此 `SVR` 无需 `random_state` 参数。
- 训练完成后，大部分样本的 $\alpha_i - \alpha_i^* = 0$——它们被 ε-管道"忽视"，不参与预测。

## 4. 预测细节：核函数加权求和

### 参数速览

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `X_test_s` | `ndarray(40, 10)` | 标准化后的测试特征 | — |
| `model.support_` | `ndarray(nSV,)` | 支持向量索引——仅这些样本参与预测 | — |
| `model.dual_coef_` | `ndarray(1, nSV)` | $\alpha_i - \alpha_i^*$ 的值——支持向量的权重 | — |
| `y_pred` | `ndarray(40,)` | 预测值 | — |

$$
f(\mathbf{x}) = \sum_{i \in \text{SV}} (\alpha_i - \alpha_i^*) K(\mathbf{x}_i, \mathbf{x}) + b
$$

### 理解重点

- 预测**不是** $\mathbf{X}\mathbf{w} + b$ 的矩阵乘法——而是与支持向量逐一计算核函数再加权求和。
- 预测复杂度 $O(nSV \cdot N_{\text{test}} \cdot d)$——支持向量越多预测越慢。
- 支持向量占比（nSV / N_train）直接决定了预测成本——通常为 30%~60%。
- 这与线性回归形成鲜明对比——线性回归的预测是固定 $O(N_{\text{test}} \cdot d)$，与训练集大小无关。

## 5. SVR 预测 vs 线性回归预测 对比

| 预测维度 | 线性回归 | SVR（RBF 核） |
|---|---|---|
| 公式 | $\hat{y} = \mathbf{X}\mathbf{w} + b$ | **$\hat{y} = \sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 复杂度 | $O(N_{\text{test}} \cdot d)$ | **$O(nSV \cdot N_{\text{test}} \cdot d)$** |
| 依赖训练集 | 否——参数已固化为 $\mathbf{w}$ | **是——需要存储支持向量集** |
| 参与计算的样本 | 无——仅用参数 | **仅支持向量——管道内样本被忽略** |
| 内存占用 | $O(d)$——仅存储系数 | **$O(nSV \cdot d)$——存储支持向量** |

## 6. SVR vs 线性回归 vs 正则化回归 训练对比

| 训练维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 数据 | 合成 `(200, 3)` | 真实+构造 `(442, 21)` | **合成非线性 `(200, 10)`** |
| 标准化 | 无 | **`StandardScaler`** | **`StandardScaler`** |
| 训练算法 | SVD 闭式解 | 坐标下降（Lasso/EN）+ 闭式解（Ridge） | **SMO——序列最小优化** |
| 训练模型数 | 1 | 3（并行） | **1** |
| 收敛判断 | 不需要 | `max_iter`（Lasso/EN） | **对偶间隙 < tol** |
| 预测 | $\mathbf{X}\mathbf{w} + b$ | $\mathbf{X}\mathbf{w} + b$ | **$\sum(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$** |
| 评估可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 | **残差图 + 学习曲线** |
| 独有诊断 | coef_ 对照真实公式 | 近零系数计数 | **支持向量数量 + 占比** |

## 常见坑

1. 期待 SVR（RBF 核）输出 `coef_`——线性核才有 `coef_`，RBF 核的权重在 `dual_coef_` 中，不可直接解释。
2. 预测时传未标准化的 `X_test`——RBF 核的欧氏距离对尺度敏感，未标准化会导致预测结果严重偏离。
3. 忽略支持向量数量——如果 nSV 接近 N_train（如 > 90%），说明模型近乎"记住"了所有训练样本，可能严重过拟合。
4. 认为 SVR 也是一步到位——训练过程是 SMO 迭代，虽然对 160 样本几乎瞬时，但对大规模数据会显著变慢。

## 小结

- SVR 流水线为 8 步：加载 → 拆分 → 切分 → 标准化 → 训练 → 预测 → 残差图 → 学习曲线。无特征重要性。
- 标准化是 SVR（RBF 核）的硬性要求——欧氏距离对特征尺度敏感。
- 训练使用 SMO 迭代求解凸二次规划——确定性过程，220 个变量（2N）在小样本上瞬时完成。
- 预测是支持向量与测试点的核函数加权求和——与线性回归的矩阵乘法本质不同，复杂度依赖支持向量数量。


# 评估与诊断

## 本章目标

1. 理解当前 SVR 流水线的两类评估输出——残差图和学习曲线。
2. 理解支持向量数量作为 SVR 独有的诊断指标的价值。
3. 明确当前流水线**已实现**和**未实现**的评估项——SVR（RBF 核）没有特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `plot_residuals(...)` | 函数 | 生成预测-真实散点图 + 残差分布图——诊断非线性拟合质量 |
| `plot_learning_curve(...)` | 函数 | 生成训练/验证 R² 曲线——诊断泛化趋势 |
| `residuals = y_true - y_pred` | 派生量 | 衡量每个样本的预测误差 |
| `model.support_.shape[0]` | 派生量 | 支持向量数量——SVR 独有的模型复杂度诊断 |

## 1. 残差图

### 参数速览

适用函数：`plot_residuals(y_test, y_pred, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `y_test` | `ndarray`，形状 `(40,)` | 测试集真实值——Friedman1 目标 | — |
| `y_pred` | `ndarray`，形状 `(40,)` | SVR 模型预测值 | `model.predict(X_test_s)` |
| `title` | `str` | 图标题 | `"SVR 残差分析"` |
| `dataset_name` | `str` | 输出目录名 | `"svr"` |
| `model_name` | `str` | 输出文件名前缀 | `"svr"` |

### 示例代码

```python
y_pred = model.predict(X_test_s)
residuals = y_test - y_pred

# 左图: 预测值 vs 真实值散点图（对角线 = 完美预测）
ax1.scatter(y_test, y_pred, alpha=0.6)

# 右图: 残差 vs 预测值散点图（红线 = 零残差）
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color="r", linestyle="--")
```

### 理解重点

- 对于 SVR 在 Friedman1 数据上，残差图应显示较好的非线性拟合——点应大致沿对角线分布。
- 右图（残差 vs 预测）重点观察残差是否有系统性弯曲——若残差呈现 U 形或正弦形模式，说明 RBF 核的 γ 或 ε 设置不当。
- 残差图回答的是"预测误差分布如何"——与支持向量数量回答的"模型多复杂"是互补视角。
- 对于非线性数据，残差图的波动通常比线性回归稍大——非线性拟合本身就比纯线性公式更难覆盖所有变化。

## 2. 学习曲线

### 参数速览

适用函数：`plot_learning_curve(estimator, X, y, scoring, cv, title, dataset_name, model_name)`

| 参数名 | 类型 | 说明 | 示例取值 |
|---|---|---|---|
| `estimator` | `SVR` | **未训练的**新 SVR 实例——学习曲线内部多次 fit | `SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')` |
| `X` | `ndarray(160, 10)` | 训练集特征——标准化后 | `X_train_s` |
| `y` | `ndarray(160,)` | 训练集目标 | `y_train` |
| `scoring` | `str` | 评分指标 | `"r2"` |
| `cv` | `int` | 交叉验证折数 | `5`（默认） |

### 示例代码

```python
plot_learning_curve(
    SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale"),
    X_train_s,
    y_train,
    scoring="r2",
    title="SVR 学习曲线",
    dataset_name="svr",
    model_name="svr",
)
```

### 理解重点

- 学习曲线传入的是**新 `SVR(...)` 实例**——而非已训练的 `model`。`_buildLearningCurveFactory("regression.svr")` 返回的工厂函数确保每次 CV 都从头训练。
- 对于 SVR + Friedman1（160 训练样本 + 10 特征）：训练和验证曲线间通常有一定的间隙——因为 SVR（RBF 核）在 160 样本上开始需要更多数据来稳定泛化。
- 如果训练 R² 远高于验证 R²——过拟合信号，考虑增大 ε 或减小 C。
- 如果训练和验证 R² 都低——欠拟合信号，考虑减小 ε 或增大 C，或检查 γ 设置。

## 3. 支持向量数量：SVR 独有的诊断指标

### 参数速览

| 指标 | 计算方式 | 诊断含义 |
|---|---|---|
| nSV | `model.support_.shape[0]` | 支持向量的绝对数量 |
| nSV 占比 | `nSV / len(y_train)` | 支持向量占训练样本的比例 |
| nSV 偏低（< 30%） | — | ε 管道宽——模型简单，可能欠拟合 |
| nSV 适中（30%~60%） | — | 模型复杂度合理——多数样本被管道容忍 |
| nSV 偏高（> 75%） | — | ε 管道窄或 C 大——模型复杂，可能过拟合 |

### 示例代码

```python
n_sv = model.support_.shape[0]
sv_ratio = n_sv / len(y_train)
print(f"支持向量数量: {n_sv}/{len(y_train)} ({sv_ratio:.1%})")
```

### 理解重点

- 支持向量数量比 R² 更直接地反映 SVR 的复杂度——R² 告诉你"预测多准"，nSV 告诉你"模型多复杂"。
- 如果 nSV 接近 N_train（如 150/160），模型几乎"记住"了每个样本——过拟合风险极高。
- 如果 nSV 极低（如 5/160），ε 管道覆盖了绝大多数样本——模型可能过于简单。
- nSV 是 SVR 独有的诊断——线性回归和决策树没有等效指标。

## 4. SVR 没有特征重要性：原因与替代

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 特征重要性（RBF 核） | **无法输出** | RBF 核将数据映射到无限维空间——权重存在于对偶空间，不可映射回原始特征 |
| 特征重要性（线性核） | 可输出（`coef_`） | 线性核 SVR 有 `coef_` 属性——相当于带 ε-管道的线性回归 |

### 理解重点

- `PipelineSpec` 中 SVR 的训练后诊断列表为 `[]`——不是遗漏，而是 RBF 核确实无法输出特征重要性。
- 这与线性回归（有 `coef_`）、决策树（有 `feature_importances_`）、正则化回归（有 `coef_` + 近零计数）形成鲜明对比。
- RBF 核的可解释性代价：获得了非线性拟合能力，但失去了"哪个特征更重要"的判断。
- 如果需要特征重要性，可以考虑切换为线性核（`kernel='linear'`）——但会丧失非线性拟合能力。

## 5. 残差图 + 学习曲线 + 支持向量数量：联合诊断

### 参数速览

| 残差图 | 学习曲线 | nSV | 联合诊断 |
|---|---|---|---|
| 好 | 训练 ≈ 验证 | 适中 | **模型健康——拟合与泛化平衡** |
| 好 | 训练 ≫ 验证 | 高 | **过拟合——模型复杂但泛化差，需增大 ε 或减小 C** |
| 差 | 训练 ≈ 验证（两者都低） | 低 | **欠拟合——模型太简单，需减小 ε 或增大 C** |
| 系统性弯曲（如 U 形） | 训练 ≈ 验证 | — | **非线性捕捉不足——尝试调整 γ** |

### 理解重点

- 单独看任何一个指标都可能给出片面判断——三者联合解读才有完整图景。
- nSV 是最快的诊断入口——如果 nSV 在 50~80（30%~50%），通常说明模型复杂度合理。
- 残差图的系统性弯曲是最明确的"核函数/参数不匹配"信号——应优先于 R² 数值被关注。

## 6. 已实现 vs 未实现的评估

### 参数速览

| 评估项 | 状态 | 原因 |
|---|---|---|
| 残差图 | 已实现 | 回归模型的核心诊断工具 |
| 学习曲线 | 已实现 | `_buildLearningCurveFactory("regression.svr")` |
| 支持向量数量打印 | 已实现 | 训练日志中打印——`model.support_.shape[0]` |
| 特征重要性 | **不适用** | RBF 核 SVR 无法输出特征重要性 |
| MSE / MAE / R² 数值打印 | **未实现** | 当前流水线侧重图形化诊断 |
| 交叉验证 R² 均值 | **未实现** | 学习曲线内部使用 CV，但未单独输出均值 |

## 7. SVR vs 线性回归 vs 正则化回归 评估对比

| 评估维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 核心可视化 | 残差图 + 学习曲线 | 残差图 + 特征重要性 | **残差图 + 学习曲线** |
| 模型解释 | coef_ + intercept_ | coef_ + intercept_ + 近零计数 | **support_ + dual_coef_——核空间不可解释** |
| 独有诊断 | 系数与真实公式对照 | 近零系数计数——L1 稀疏化 | **支持向量数量 + 占比——模型复杂度** |
| 特征重要性 | coef_ 直接输出 | coef_ 直接输出 | **无（RBF 核）——需切换为线性核** |
| 定量指标 | 无显式打印 | 无显式打印 | 无显式打印 |

## 常见坑

1. 期待 SVR（RBF 核）输出特征重要性——这是回归分册中唯一不适用特征重要性的模型。
2. 只关注残差图不关注 nSV——nSV 偏低或偏高会直接揭示模型是否"偷懒"或"过劳"。
3. 用训练好的 `model` 传入学习曲线——学习曲线需要新 `SVR(...)` 实例做内部 CV。
4. 在 Friedman1 这类非线性数据上要求残差与线性回归一样规整——非线性拟合天然有更大波动。

## 小结

- SVR 有两类评估输出（残差图 + 学习曲线）+ 一项日志输出（支持向量数量）——三者构成完整的诊断体系。
- 支持向量数量是 SVR 独有的诊断指标——比 R² 更直接地反映模型复杂度。
- SVR（RBF 核）没有特征重要性——这是非线性核的可解释性代价，也是 `PipelineSpec` 训练后诊断列表为 `[]` 的原因。
- 残差图 + 学习曲线 + nSV 的联合解读比单独依赖任何一个指标都更可靠。


# 工程实现

## 本章目标

1. 理解 SVR 流水线的模块分层——数据层、训练层、流水线注册层、运行器层和可视化层。
2. 理清从命令行入口到结果图落盘的完整调用链。
3. 理解 SVR 与线性回归、正则化回归在工程实现上的关键差异——标准化 + 学习曲线 + 无特征重要性。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| `RegressionDatasetFactory` | 类 | 数据工厂——`loadSvrDataset()` 生成 Friedman1 非线性数据 |
| `trainSvrRegressionModel(...)` | 函数 | 构建并训练 `SVR`——本仓库最短训练函数（2 行） |
| `PipelineSpec` | 数据类 | 声明式流水线配置——训练后诊断列表为 `[]` |
| `RegressionRunner` | 类 | 回归流水线运行器——读取 `PipelineSpec`，依次执行各阶段 |
| `plot_residuals(...)` | 函数 | 残差图绘制 |
| `plot_learning_curve(...)` | 函数 | 学习曲线绘制——使用 `learningCurveEstimatorFactory` 工厂 |

## 1. 模块分层总览

### 参数速览

| 层 | 文件 | 职责 | 输出 |
|---|---|---|---|
| 数据层 | `src/mlAlgorithms/datasets/tabular/regressionDatasets.py` | `loadSvrDataset()`——调用 `make_friedman1` 生成非线性数据 | `DataFrame`，形状 `(200, 11)` |
| 数据目录层 | `src/mlAlgorithms/datasets/datasetCatalog.py` | `DatasetSpec("regression.svr", ...)`——注册数据集描述与加载器 | 数据集元信息 |
| 训练层 | `src/mlAlgorithms/training/regression/regressionModels.py` | `trainSvrRegressionModel(...)`——构建 `SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')` 并 fit | `SVR` 模型对象 |
| 流水线注册层 | `src/mlAlgorithms/catalog/pipelines.py` | `PipelineSpec("regression.svr", ...)`——关联所有组件 | 流水线配置 |
| 运行器层 | `src/mlAlgorithms/workflows/regressionRunner.py` | 读取 PipelineSpec → 加载 → 标准化 → 训练 → 评估 → 可视化 | 终端日志 + 图像文件 |
| 可视化层 | `src/mlAlgorithms/visualization/` | 绘制残差图、学习曲线 | PNG 图像文件 |

### 理解重点

- SVR 的工程结构与线性回归几乎相同——都是单模型、单次训练、两组可视化。差异在于 SVR 多了标准化步骤。
- SVR 与正则化回归的工程结构差异明显——正则化回归是多模型（`multiModel=True`）+ 无学习曲线，SVR 是单模型 + 有学习曲线。
- 训练函数仅 2 行——比线性回归（3 行）更短，因为所有超参数在构造器中一次性全部给出。

## 2. `PipelineSpec` 配置详情

```python
PipelineSpec(
    "regression.svr",                      # pipeline ID
    TaskType.REGRESSION,                   # 任务类型
    "regression.svr",                      # dataset ID
    RunnerType.REGRESSION,                 # 运行器类型
    trainSvrRegressionModel,               # 训练函数——2 行封装
    "standardScaler",                      # 预处理——RBF 核强制标准化
    "randomSplit",                         # 切分策略
    "default",                             # 后处理
    "regression",                          # 输出目录前缀
    "regression",                          # 可视化目录前缀
    ["correlationHeatmap", "featureTargetScatter"],  # 训练前可视化
    [],                                    # 训练后诊断可视化——空列表！（RBF 核无特征重要性）
    ["learningCurve"],                     # 学习可视化
    "svr",                                 # 结果存储子目录
    metadata={
        "learningCurveEstimatorFactory": _buildLearningCurveFactory(
            "regression.svr"               # → SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')
        )
    },
)
```

### 理解重点

- `[]` 训练后诊断列表为空——这是 SVR（RBF 核）与线性回归/正则化回归在工程层面最显著的区别。RBF 核的 SVR 无法输出 `coef_` 或 `feature_importances_`。
- `"learningCurve"` 在学可视化列表中——SVR 有学习曲线（与正则化回归不同）。
- `learningCurveEstimatorFactory` 确保学习曲线使用与训练一致的超参数——`SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')`。
- `"standardScaler"` 预处理——SVR 和正则化回归都需要标准化，线性回归和决策树不需要。

## 3. 数据依赖关系

```
loadSvrDataset()
    │
    ├─→ X = data.drop(columns=["price"])
    ├─→ y = data["price"]
    ├─→ feature_names = list(X.columns)
    │
    ├─→ train_test_split(test_size=0.2)
    │       │
    │       ├─→ StandardScaler().fit_transform(X_train) ──→ X_train_s
    │       ├─→ StandardScaler().transform(X_test) ──→ X_test_s
    │       │
    │       ├─→ trainSvrRegressionModel(X_train_s, y_train)
    │       │       │
    │       │       └─→ model (support_, dual_coef_, intercept_)
    │       │
    │       ├─→ y_pred = model.predict(X_test_s)
    │       │       │
    │       │       └─→ plot_residuals(y_test, y_pred)
    │       │
    │       └─→ plot_learning_curve(SVR(...), X_train_s, y_train, scoring="r2")
```

### 理解重点

- 数据依赖图与正则化回归的差异：SVR 没有多模型循环，没有 `featureImportance` 分支。
- 标准化分支在切分之后——`fit_transform` 用于训练集，`transform` 用于测试集。
- 学习曲线分支使用工厂函数创建的新 `SVR(...)` 实例——而非已训练的 `model`。
- `model.support_` 是训练的核心产物——终端日志直接打印其数量。

## 4. 运行器层的执行链

| 序号 | 步骤 | 说明 |
|---|---|---|
| 1 | 根据 `datasetId` 查找 `DatasetSpec` | 获取数据加载器和描述信息 |
| 2 | 调用 `loadSvrDataset()` | 加载 `(200, 11)` DataFrame |
| 3 | 拆分 X / y + 保存 `feature_names` | 为后续日志和可视化作准备 |
| 4 | `train_test_split(test_size=0.2)` | 随机切分 |
| 5 | `StandardScaler().fit_transform(X_train)` | 标准化训练集——RBF 核必需 |
| 6 | `StandardScaler().transform(X_test)` | 标准化测试集 |
| 7 | 调用 `trainSvrRegressionModel(X_train_s, y_train)` | SMO 求解——打印支持向量数量 |
| 8 | `model.predict(X_test_s)` | 核函数加权求和预测 |
| 9 | `plot_residuals(y_test, y_pred)` | 残差诊断图 |
| 10 | `plot_learning_curve(SVR(...), X_train_s, y_train, scoring="r2")` | 学习曲线——使用工厂创建的新实例 |

### 理解重点

- 步骤 7 的训练耗时取决于 SMO 收敛速度——200 样本上几乎瞬时，但数据量大时是瓶颈。
- 步骤 8 的预测复杂度与支持向量数成正比——`nSV` 越多预测越慢。
- 与线性回归对比：多了标准化（5-6），少了 `coef_` 打印（RBF 核没有 `coef_`）。
- 与正则化回归对比：少了多模型循环，多了学习曲线（10）。

## 5. SVR vs 线性回归 vs 正则化回归 工程对比

| 工程维度 | 线性回归 | 正则化回归 | SVR |
|---|---|---|---|
| 训练函数 | `trainLinearRegressionModel` | `trainRegularizationModels` | **`trainSvrRegressionModel`** |
| 模型类 | `LinearRegression` | `Lasso`, `Ridge`, `ElasticNet` | **`SVR`** |
| 训练函数行数 | 3 行 | ~10 行 | **2 行——最简** |
| 预处理 | `None` | **`standardScaler`** | **`standardScaler`** |
| 超参数数 | 0 | 1~2 | **4** |
| 训练后诊断 | `["featureImportance"]` | `["featureImportance"]` | **`[]`——RBF 核无特征重要性** |
| 学习可视化 | `["learningCurve"]` | `[]`——无学习曲线 | **`["learningCurve"]`** |
| 数据量 | 200（手工合成） | 442（真实 + 构造） | **200（合成非线性）** |
| 训练方式 | SVD 闭式解 | 坐标下降 / 闭式解 | **SMO——序列最小优化** |
| PipelineSpec 元数据 | 无特殊 metadata | **`{"multiModel": True}`** | **`{"learningCurveEstimatorFactory": ...}`** |

## 常见坑

1. 误以为 SVR 也有特征重要性可视化——`PipelineSpec` 中训练后诊断列表明确为 `[]`，RBF 核不可输出。
2. 将 `learningCurveEstimatorFactory` 的作用与 `multiModel` 混淆——前者是"学习曲线用什么模型实例"，后者是"训练函数返回多个模型"。
3. 忽略标准化在运行器层而非数据层——数据层返回原始 Friedman1 数据，运行器负责 `StandardScaler` 调用。
4. 将 `PipelineSpec` 中的 `"svr"`（输出子目录名）与其他模型混淆——输出文件在 `outputs/svr/` 下。

## 小结

- SVR 工程实现采用声明式流水线架构——`PipelineSpec` 配置所有组件，`RegressionRunner` 按序编排执行。
- SVR 在回归分册中的工程定位是"标准化 + 学习曲线 + 无特征重要性"——填补了正则化回归（标准化 + 无学习曲线）和线性回归（无标准化 + 有学习曲线）之间的配置空白。
- 训练函数仅 2 行——`SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale').fit(XTrain, yTrain)`，是本仓库最简训练封装。
- 支持向量数量是终端日志中的关键诊断信息——比任何数值指标都更直观地反映模型复杂度。


# 练习与参考文献

## 本章目标

1. 通过自检问题确认对 SVR 核心概念的理解程度。
2. 通过动手练习在代码层面观察 C、ε、γ 和支持向量数量的联动关系。
3. 提供扩展阅读的参考文献入口。

## 重点方法与概念速览

| 名称 | 类型 | 作用 |
|---|---|---|
| 自检问题 | 诊断 | 确认对 ε-管道、RBF 核、支持向量稀疏性、C/ε/γ 角色等核心概念的理解 |
| 动手练习 | 实践 | 修改 C、ε、kernel、γ 并观察 nSV、残差图、学习曲线的变化 |
| 参考文献 | 入口 | 提供 SVR 经典论文和 scikit-learn 官方文档 |

## 1. 自检问题

1. SVR 的 ε-不敏感损失函数与 OLS 的平方损失有什么根本区别？为什么管道内的样本不参与预测？

2. C 在 SVR 中扮演什么角色？为什么说 C 是正则化强度的**倒数**？C=0.1 和 C=100 时模型行为有何差异？

3. RBF 核的作用是什么？当 `kernel='linear'` 时，SVR 退化为什么？gamma 控制什么？

4. 为什么 SVR（RBF 核）没有 `coef_` 属性？支持向量的权重存储在哪里？

5. SVR 为什么必须标准化？这与 RBF 核的数学公式有何关系？

6. 支持向量数量反映了什么？如果 nSV/160 > 90% 或 < 10%，分别意味着什么？

7. 当前 SVR 的 `PipelineSpec` 中训练后诊断列表为什么是 `[]`？这对 SVR 的评估策略有什么影响？

## 2. 动手练习

### 练习 1：改变 C 的值

修改 `trainSvrRegressionModel` 中的 `C`，分别设为 `0.1`、`10.0`（默认）、`100`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=0.1, epsilon=0.1, kernel="rbf", gamma="scale")
# 试试 0.1, 1.0, 10.0, 100
```

回答：C=0.1 时支持向量数量是否显著减少？C=100 时训练 R² 是否接近 1.0 但验证 R² 下降？残差图在三个 C 值下有何不同？

### 练习 2：改变 epsilon 的值

将 `epsilon` 分别设为 `0.01`、`0.1`（默认）、`1.0`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.01, kernel="rbf", gamma="scale")
# 试试 0.01, 0.1, 0.5, 1.0
```

回答：ε=0.01 时支持向量数量是否急剧增加？ε=1.0 时是否几乎所有样本都在管道内（nSV 极小）？残差图的散点分布随 ε 如何变化？

### 练习 3：对比线性核与 RBF 核

将 `kernel` 从 `'rbf'` 改为 `'linear'`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.1, kernel="linear")
```

回答：线性核在 Friedman1 上的残差图是否显著劣于 RBF 核？线性核的 SVR 现在是否有 `coef_` 属性？支持向量数量有何变化？

### 练习 4：改变 gamma 的值

将 `gamma` 从 `'scale'` 改为具体数值——`0.01`、`0.1`、`1.0`。

```python
# 在 trainSvrRegressionModel 中修改
model = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma=0.01)
# 试试 'scale', 0.01, 0.1, 1.0
```

回答：γ=0.01 时学习曲线的训练/验证间隙是否缩小（更平滑）？γ=1.0 时训练 R² 是否极高但验证 R² 暴跌（严重过拟合）？nSV 随 γ 如何变化？

### 练习 5：手动加入 R² 和 MSE 计算

在流水线预测后手动计算并打印 R² 和 MSE：

```python
from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test_s)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"测试集 R²: {r2:.4f}")
print(f"测试集 MSE: {mse:.4f}")
print(f"支持向量数量: {model.support_.shape[0]}/{len(y_train)}")
```

回答：R² 是否与学习曲线中的验证得分一致？"R² 更高"和"nSV 更少"是否同时发生？数值指标与残差图的视觉判断是否吻合？

## 3. 参考文献

| 序号 | 文献 | 说明 |
|---|---|---|
| 1 | Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199-222. | SVR 经典教程——从 ε-不敏感损失到对偶问题的完整推导 |
| 2 | Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1997). Support vector regression machines. *Advances in Neural Information Processing Systems*, 9. | SVR 原始论文——ε-SVR 的提出与算法实现 |
| 3 | Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 12. | 经典教材——支持向量机和核方法的完整数学推导 |
| 4 | scikit-learn 官方文档 — [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) | scikit-learn 的 SVR API 参考——所有构造器参数、属性和方法 |
| 5 | scikit-learn 官方文档 — [SVM 回归用户指南](https://scikit-learn.org/stable/modules/svm.html#svm-regression) | scikit-learn 的 SVR 使用指南——核函数选择和调参建议 |

## 常见坑

1. 修改 C 时忘记 C 是正则化倒数——增大 C = 减弱正则化，与 Lasso 的 α 方向相反。
2. 改 gamma 只看残差图不关注 nSV——gamma 的局部性变化最直接体现为 nSV 的剧烈变化。
3. 对比线性核和 RBF 核时忘记线性核不需要标准化（但当前流水线仍会标准化）——统一预处理保持对比公平。
4. 只改训练函数的参数忘记学习曲线中的 `SVR(...)` 也要同步——两者参数不一致会导致学习曲线的诊断无效。
5. 在 Friedman1 上期待线性核与 RBF 核表现相近——Friedman1 的目标函数高度非线性，线性核必然显著劣于 RBF 核。

## 小结

- 7 个自检问题覆盖 SVR 的核心概念：ε-管道损失、C 的角色、RBF 核、参数空间、标准化必要性、支持向量数量、PipelineSpec 配置。
- 5 个动手练习从不同角度探索 SVR 行为——调 C、调 ε、切换核函数、调 γ、加入数值指标。
- 5 篇参考文献覆盖 SVR 经典教程、原始论文、ESL 教材和 scikit-learn 文档——构成完整的 SVR 学习路线。
