---
title: 线性回归实现
date: 2026-01-21
category: MachineLearning/Regression
tags:
  - Python
  - Scikit-learn
  - 回归
  - 分类
description: 从数学原理、损失函数、正规方程推导，到使用Python和scikit-learn实现数据生成、探索、可视化、预处理、训练、评估及结果可视化的完整线性回归项目指南。
image: https://img.yumeko.site/file/blog/LinearRegression.png
status: published
---

# 线性回归数学基础与推导

这一章只讲原理，不涉及代码。你读完之后，再看后面的代码会更顺畅。

---
## 1. 什么是线性回归

线性回归是一种**监督学习**的回归算法，用来预测连续值。它假设：

- 输出变量 `y` 可以用输入特征的**线性组合**表示。
- 通过最小化误差来估计参数。

直观理解：
- 一元线性回归就是“拟合一条直线”。
- 多元线性回归就是“在高维空间拟合一个超平面”。

---
## 2. 模型形式

### 2.1 一元线性回归

$$
\hat{y} = \beta_0 + \beta_1 x
$$

- $\beta_0$：截距（常数项）
- $\beta_1$：斜率（特征权重）

### 2.2 多元线性回归

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

---

## 3. 矩阵表示

为了推导方便，我们把偏置项 $\beta_0$ 合并进参数向量：

$$
\mathbf{X} = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
$$

$$
\boldsymbol{\beta} = \begin{bmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p\end{bmatrix}
$$

预测写成：

$$
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}
$$

---

## 4. 损失函数（最小二乘）

我们希望预测值 $\hat{y}$ 与真实值 $y$ 的差越小越好。

### 4.1 SSE / MSE

**SSE（误差平方和）**：
$$
\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**MSE（均方误差）**：
$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

向量形式：
$$
J(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

---

## 5. 正规方程推导（闭式解）

### 5.1 目标函数

$$
J(\boldsymbol{\beta}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

### 5.2 对参数求导

$$
\frac{\partial J}{\partial \boldsymbol{\beta}} = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

### 5.3 令导数为 0

$$
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
$$

### 5.4 解出参数

$$
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

> 这就是正规方程（Normal Equation）。

**注意**：如果 $\mathbf{X}^T\mathbf{X}$ 不可逆，可以用伪逆或 SVD 求解。

---

## 6. 梯度下降（数值解）

当数据规模大或矩阵不可逆时，用梯度下降更稳：

### 6.1 更新公式

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \cdot \nabla J(\boldsymbol{\beta})
$$

$$
\nabla J(\boldsymbol{\beta}) = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

### 6.2 学习率 $\alpha$

- 太小：收敛慢
- 太大：可能发散

---

## 7. 线性回归的概率视角

如果假设误差 $\epsilon$ 服从高斯分布：

$$
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

则最大似然估计会导出最小二乘解。也就是说：

> 最小二乘 = 高斯噪声下的最大似然估计。

---

## 8. 评价指标（用于代码评估部分）

**MSE**：
$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

**RMSE**：
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

**MAE**：
$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
$$

**R²**：
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

---

## 9. 线性回归常见假设

- **线性关系**：特征与目标之间线性可表示
- **误差独立**：样本之间不相关
- **同方差**：误差的方差稳定
- **误差正态**：误差近似高斯分布
- **无严重多重共线性**：特征之间不能高度相关

> 这些假设不满足时，模型可信度会下降。

---

## 10. 特征缩放的意义

线性回归的闭式解对缩放不敏感，但以下情况推荐标准化：

- 梯度下降优化
- 需要比较系数大小时
- 特征量纲差异过大时

标准化公式：

$$
 x' = \frac{x - \mu}{\sigma}
$$

---

## 11. 过拟合与欠拟合

- **欠拟合**：模型太简单，训练集表现差
- **过拟合**：训练集很好，测试集差

检查方式：训练集 R² 与测试集 R² 差距过大。

---

## 12. 正则化（延伸阅读）

### 12.1 Ridge（L2）

$$
J(\boldsymbol{\beta}) = \text{MSE} + \lambda \sum \beta_i^2
$$

### 12.2 Lasso（L1）

$$
J(\boldsymbol{\beta}) = \text{MSE} + \lambda \sum |\beta_i|
$$

- Ridge：更稳定，降低方差
- Lasso：可做特征选择

---

## 13. 你在这个项目里会用到什么

- 使用 `sklearn.linear_model.LinearRegression` 拟合参数
- 用 `MSE / RMSE / MAE / R²` 评估模型
- 可视化帮助诊断线性关系和残差结构

读完这一章后，继续看代码部分即可。

# 生成模拟数据（generate_data.py）

本模块负责**造数据**，让你在没有真实数据集时也能完整跑通线性回归流程。

---
## 1. 核心目标

- 构造 3 个特征：**面积、房间数、房龄**
- 设置一个“真实线性关系”
- 添加噪声，模拟真实世界数据的不确定性
- 返回 `pandas.DataFrame` 供后续步骤使用

---
## 2. 代码结构

```python
from utils.decorate import print_func_info

@print_func_info
def generate_data(n_samples=200, noise=10, random_state=42) -> DataFrame:
    ...
```

- 使用装饰器 `print_func_info`：调用函数时会打印提示，方便学习与调试。

---
## 3. 关键参数

- `n_samples`：样本数量
- `noise`：噪声强度（越大，数据越“乱”）
- `random_state`：随机种子，保证结果可复现

---
## 4. 生成特征

```python
area = np.random.uniform(low=20, high=80, size=n_samples)
num  = np.random.uniform(low=1,  high=5,  size=n_samples)
age  = np.random.uniform(low=1,  high=20, size=n_samples)
```

解释：
- `np.random.uniform` 生成均匀分布随机数
- 这里的范围模拟现实场景（比如房屋面积 20~80）

---
## 5. 生成目标变量（价格）

核心公式：

$$
\text{价格} = 2\cdot\text{面积} + 10\cdot\text{房间数} - 3\cdot\text{房龄} + \epsilon + 50
$$

```python
price = 2*area + 10*num - 3*age + np.random.normal(0, noise, n_samples) + 50
```

解释：
- **线性关系**：面积和房间数对价格正向影响，房龄负向影响
- **噪声项 $\epsilon$**：让数据更接近真实场景
- **+50**：让价格整体上移（类似“基准价”）

---
## 6. 输出 DataFrame

```python
data = DataFrame({
    "面积": area,
    "房间数": num,
    "房龄": age,
    "价格": price
})
```

输出表格结构：

| 面积 | 房间数 | 房龄 | 价格 |
|------|--------|------|------|
| 45.2 | 3.0    | 12.1 |  ... |

---

## 7. 你可以尝试的修改

1) **改变噪声**
- `noise` 变大，拟合会更困难

2) **改系数**
- 修改 `2*area + 10*num - 3*age`，理解权重的意义

3) **增加非线性项（练习）**
- 例如加入 `area**2` 或 `num*age`

---

## 8. 小结

- 这个模块为你提供了可控的“实验数据”。
- 你知道了“真实关系”，所以更容易判断模型是否学对了。
- 后续的探索、训练、评估全部基于这里的数据。

# 数据探索（explore_data.py）

这一模块帮助你快速了解数据的结构、范围、统计特性和相关性。

---
## 1. 为什么要做数据探索

在训练模型前，必须弄清楚：

- 数据有多少行、多少列
- 每个特征的取值范围和分布
- 是否存在缺失值
- 特征与目标变量是否存在线性关系

这些信息会影响你是否需要清洗、特征工程或更换模型。

---

## 2. 核心函数

```python
@print_func_info
def explore_data(data: DataFrame):
    ...
```

输入是 `DataFrame`，输出是目标变量与各特征的相关系数。

---

## 3. 打印样本结构

```python
print(f"样本数量: {len(data)}")
print(f"特征数量: {len(data.columns) - 1}")
print(f"特征名称: {list(data.columns[:-1])}")
```

**你能得到：**
- 数据量是否足够
- 是否少了某个特征或目标列

---

## 4. 描述统计

```python
print(data.describe().round(2))
```

`describe()` 会显示：
- 均值 / 标准差 / 最大值 / 最小值 / 分位数

用途：
- 判断是否存在异常值
- 判断尺度是否差异很大

---

## 5. 缺失值检查

```python
missing = data.isnull().sum()
```

- 如果缺失值存在，要先处理（删除或填充）
- 这一步在真实数据中非常关键

---

## 6. 相关性分析

```python
correlation = data.corr()["价格"].drop("价格").sort_values(ascending=False)
```

- `corr()` 计算皮尔逊相关系数
- 值域在 [-1, 1]：
  - 接近 1：强正相关
  - 接近 -1：强负相关
  - 接近 0：线性相关性弱

**注意：相关性只是线性关系的指标**，不代表因果。

---

## 7. 输出结果示例

可能看到类似：

```
面积    0.85
房间数  0.72
房龄   -0.60
```

解释：
- 面积和房间数与价格正相关
- 房龄与价格负相关

这与我们在 `generate_data.py` 设置的“真实关系”一致。

---

## 8. 小结

- 这一步是建模前的“体检”
- 它不会改变数据，但能让你更了解数据
- 相关性结果能帮助你判断线性回归是否合适

# 数据可视化（visualize_data.py）

这一模块把数据的分布和关系“画出来”，让你直观看到线性趋势。

---

## 1. 为什么要可视化

- 直方图看分布（是否集中 / 偏斜 / 是否异常）
- 散点图看关系（是否近似线性）
- 相关性热力图看整体相关强弱

---

## 2. 输出位置

代码中通过 `config.py` 获取输出目录：

```python
from config import OUTPUTS_ROOT
LR_OUTPUTS = os.path.join(OUTPUTS_ROOT, "LinearRegression")
```

所有图像都会保存到：

```
outputs/LinearRegression/
```

---

## 3. 图 1：特征分布直方图

```python
axes[row, col].hist(data[feature], bins=30, color='skyblue')
```

作用：
- 判断特征范围是否集中
- 观察是否存在极端值

![01_data_distribution](https://img.yumeko.site/file/articles/LinearRegression/01_data_distribution.png)

---

## 4. 图 2：相关性热力图

```python
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

作用：
- 查看特征与目标之间的线性相关
- 相关系数越接近 1 或 -1，线性关系越明显

![02_correlation_heatmap](https://img.yumeko.site/file/articles/LinearRegression/02_correlation_heatmap.png)

---

## 5. 图 3：特征 vs 目标散点图

```python
axes[i].scatter(data[feature], data["价格"], alpha=0.6)
```

作用：
- 直接观察线性趋势
- 如果散点大致呈一条“斜线”，线性回归效果通常不错

![03_feature_relationship](https://img.yumeko.site/file/articles/LinearRegression/03_Feature_Relationship.png)

---

## 6. 中文字体设置

为了避免中文乱码，代码设置了字体：

```python
plt.rcParams["font.sans-serif"] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

如果你的电脑没有这些字体，可以替换为其他中文字体。

---

## 7. 小结

- 可视化帮你判断：线性回归是否适合当前数据
- 这一步不改变数据，只提供理解与判断
- 输出图像会被后续报告或笔记使用

# 数据预处理（preprocess_data.py）

这一模块负责 **划分训练集/测试集** 并进行 **特征标准化**。

---

## 1. 为什么要预处理

- 防止“训练集和测试集混用”，保证评估公正
- 特征尺度差异较大时，标准化可以让优化更稳定

---

## 2. 分离特征与目标

```python
features = data.drop("价格", axis=1)
price = data["价格"]
```

- `features` 是输入 $X$
- `price` 是输出 $y$

---

## 3. 划分训练集 / 测试集

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, price, test_size=test_size, random_state=random_state)
```

- `test_size=0.2`：20% 作为测试集
- `random_state`：确保划分结果可复现

划分后打印：

- 训练集样本数
- 测试集样本数
- 训练/测试占比

---

## 4. 标准化（StandardScaler）

标准化公式：

$$
 x' = \frac{x - \mu}{\sigma}
$$

代码：

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

注意：
- **只能在训练集上 fit**
- 测试集只能 transform，否则会泄露信息

---

## 5. 返回值说明

函数返回：

- `X_train_scaled`：标准化后的训练集
- `X_test_scaled`：标准化后的测试集
- `y_train, y_test`：标签
- `scaler`：标准化器（后续可逆变换）
- `X_train, X_test`：原始未标准化数据

为什么要保留原始数据？
- 结果可视化时需要原始尺度
- 解释模型时更直观

---

## 6. 小结

- 本模块不改变标签，只处理特征
- 训练集/测试集分离是机器学习的基本原则
- 标准化让训练更稳定，但别忘了保留原始数据

# 模型训练（train_model.py）

这一模块负责训练线性回归模型，并输出模型参数（截距与系数）。

---

## 1. 训练目标

通过最小二乘法拟合参数：

$$
\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

在代码中，`sklearn` 会用数值稳定的方法（如 SVD）实现这个过程。

---

## 2. 核心函数

```python
@print_func_info
def train_model(X_train, y_train, feature_names=None):
    model = LinearRegression()
    model.fit(X_train, y_train)
    ...
```

- `X_train`：训练特征（通常是标准化后的数据）
- `y_train`：训练标签
- `feature_names`：可选，用来打印系数时显示真实特征名

---

## 3. 训练过程

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

- `.fit()` 会计算参数 $\beta$
- 训练结束后可用 `coef_` 和 `intercept_` 访问参数

---

## 4. 输出参数含义

```python
print(f"截距: {model.intercept_:.2f}")
print("斜率(coefficients):")
for name, coef in zip(features_names, model.coef_):
    print(f"{name}: {coef:.2f}")
```

解释：
- `intercept_` 是 $\beta_0$
- `coef_` 是 $\beta_1 ... \beta_p$

正负号含义：
- **正数**：特征值增大，预测值增大
- **负数**：特征值增大，预测值减小

---

## 5. 特征名处理逻辑

代码中有一段逻辑，用于处理特征名：

```python
if feature_names is not None:
    features_names = feature_names
elif hasattr(X_train, 'columns'):
    features_names = list(X_train.columns)
else:
    features_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
```

这样做的目的：
- 如果传入了特征名就用它
- 如果是 DataFrame 就自动取列名
- 如果是 NumPy 数组就生成默认名

---

## 6. 小结

- 训练模块输出模型参数，方便解释模型
- 线性回归是可解释性很强的模型
- 你可以通过系数判断特征的重要性和方向

# 模型评估（evaluate_model.py）

这一模块对训练好的模型进行性能评估，包含 **训练集** 和 **测试集** 两部分。

---

## 1. 为什么要评估

评估的目的：

- 看模型是否学到了规律
- 判断泛化能力（是否过拟合）
- 给出可量化的指标

---

## 2. 核心函数

```python
@print_func_info
def evaluate_model(model, X_train, X_test, y_train, y_test):
    ...
```

输入：
- 训练好的 `model`
- 训练集 / 测试集特征与标签

输出：
- 训练集预测值 `y_train_pred`
- 测试集预测值 `y_test_pred`

---

## 3. 预测

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

- `predict` 只是把特征代入公式得到 $\hat{y}$

---

## 4. 评估指标

### 4.1 MSE / RMSE

$$
\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2
$$
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

- RMSE 与原始单位一致，更直观

### 4.2 MAE

$$
\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|
$$

- 对异常值更“温和”

### 4.3 R²

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- R² 越接近 1 越好
- 负值表示模型很差（甚至不如直接预测均值）

---

## 5. 过拟合检测逻辑

代码里比较训练集与测试集 R² 的差值：

```python
r2_diff = train_r2 - test_r2
```

判断标准：
- `< 0.05`：泛化良好
- `< 0.1`：轻微过拟合
- `>= 0.1`：可能过拟合

---

## 6. 输出示例

```text
训练集性能:
  R^2 Score:  0.92
  RMSE:      3.12
  MAE:       2.35

测试集性能:
  R^2 Score:  0.89
  RMSE:      3.56
  MAE:       2.72
```

---

## 7. 小结

- 评估模块给出定量指标
- 你可以快速判断模型是否可靠
- 训练集和测试集都要看，不能只看训练集

# 结果可视化（visualize_results.py）

这一模块把模型预测效果直观画出来，是理解模型好坏的关键一步。

---

## 1. 输出位置

图像保存到：

```
outputs/LinearRegression/
```

对应文件名：

- `04_Prediction_effect.png`
- `05_Residual_analysis.png`
- `06_Single_feature_regression.png`

---

## 2. 图 1：预测值 vs 真实值

```python
axes[0].scatter(y_train, y_train_pred)
axes[1].scatter(y_test, y_test_pred)
```

解释：
- 横轴是真实值
- 纵轴是预测值
- 越接近对角线越好

![04_prediction_effect](https://img.yumeko.site/file/articles/LinearRegression/04_Prediction_effect.png)

---

## 3. 图 2：残差分析

残差定义：

\[
\text{残差} = y - \hat{y}
\]

### 3.1 残差分布直方图

- 理想情况下残差以 0 为中心
- 分布越对称越好

### 3.2 残差 vs 预测值

- 理想：散点均匀分布在 0 附近
- 若呈现结构化趋势，说明模型可能遗漏了非线性关系

![05_residual_analysis](https://img.yumeko.site/file/articles/LinearRegression/05_Residual_analysis.png)

---

## 4. 图 3：单特征回归效果

```python
axes[i].scatter(X_test_original.iloc[:, i], y_test)
axes[i].scatter(X_test_original.iloc[:, i], y_test_pred)
```

意义：
- 查看单个特征与目标的关系
- 对比预测值与真实值在该特征上的分布

![06_single_feature_regression](https://img.yumeko.site/file/articles/LinearRegression/06_Single_feature_regression.png)

---

## 5. 为什么一定要看残差

- 残差能暴露模型假设是否成立
- 若残差呈现曲线趋势，说明模型可能需要非线性特征

---

## 6. 小结

- 这一步帮助你从“视觉层面”判断模型是否合理
- 预测效果好 ≠ 模型正确，残差结构很关键
- 如果残差结构明显，考虑增加特征或改用非线性模型

# 一键运行完整流程（main.py）

`main.py` 是整个线性回归项目的“入口”，按照顺序执行所有步骤。

---

## 1. 执行顺序

```text
1. 生成数据
2. 数据探索
3. 数据可视化
4. 数据预处理
5. 模型训练
6. 模型评估
7. 结果可视化
```

---

## 2. 代码结构

```python
from generate_data import generate_data
from explore_data import explore_data
from visualize_data import visualize_data
from preprocess_data import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model
from visualize_results import visualize_results
```

这些模块刚好对应前面每一个文档章节。

---

## 3. 步骤讲解

### 3.1 生成数据

```python
df = generate_data(n_samples=200, noise=10, random_state=42)
```

你可以在这里控制样本数和噪声大小。

---

### 3.2 数据探索

```python
correlation = explore_data(df)
```

输出统计信息与相关系数，帮助确认数据结构。

---

### 3.3 数据可视化

```python
visualize_data(df)
```

生成分布图、相关性热力图、散点图。

---

### 3.4 数据预处理

```python
X_train, X_test, y_train, y_test, scaler, X_train_orig, X_test_orig = preprocess_data(df)
```

- `X_train, X_test` 是标准化后的数据
- `X_train_orig, X_test_orig` 是原始数据（用于可视化）

---

### 3.5 模型训练

```python
model = train_model(X_train, y_train, feature_names=X_train_orig.columns.tolist())
```

模型用标准化数据训练，但系数输出仍使用原始特征名。

---

### 3.6 模型评估

```python
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
```

得到训练集/测试集预测结果并打印指标。

---

### 3.7 结果可视化

```python
visualize_results(
    y_train, y_train_pred,
    y_test, y_test_pred,
    X_test_orig, X_train_orig.columns
)
```

生成预测效果图、残差分析图、单特征回归效果图。

---

## 4. 常用修改点

- **改变噪声**：`generate_data` 中 `noise`
- **改变训练/测试比例**：`preprocess_data` 中 `test_size`
- **改变训练特征**：修改 `generate_data` 和 `preprocess_data` 中特征列

---

## 5. 小结

`main.py` 的目的就是：

> 把所有模块连起来，让你一键跑完整个线性回归项目。

当你理解了每个模块的原理，这个脚本就是你验证理解的“总开关”。

# 常见问题与延伸（FAQ）

这一节汇总学习线性回归时最常见的问题，并给出清晰的解释。

---

## 1. 为什么要标准化？

- 如果使用梯度下降，标准化能让收敛更快
- 不同量纲的特征会导致某些特征“支配”模型
- 标准化后系数比较更直观

---

## 2. 线性回归一定要求线性关系吗？

严格来说：
- 线性回归拟合的是**线性组合**，而不是必须“直线”趋势。
- 你可以通过特征工程（如 $x^2$、$\log x$）把非线性关系变成线性关系。

---

## 3. $R^2$ 为负数说明什么？

- 模型效果比“直接预测均值”还差
- 可能原因：
  - 特征与目标无关
  - 数据噪声太大
  - 训练/测试划分不合理

---

## 4. 过拟合怎么办？

- 减少特征
- 增加样本量
- 使用正则化（Ridge / Lasso）
- 用交叉验证选择更稳定的模型

---

## 5. 如何解释系数？

- 系数表示“在其他特征不变时，该特征对目标的边际影响”
- 例如：`面积` 系数是 2，表示面积每增加 1 单位，价格平均增加 2

---

## 6. 为什么训练集和测试集都很差？

- 可能是欠拟合：模型太简单
- 可能是数据本身不可预测
- 也可能是目标变量噪声太大

---

## 7. 如何改进这个项目？

你可以尝试：

1. 增加特征
- 比如“楼层”“朝向”“距离地铁”

2. 加入非线性特征
- `面积^2`、`房龄^2`

3. 尝试正则化
- Ridge、Lasso

4. 用更复杂的模型
- 决策树、随机森林、梯度提升

---

## 8. 学习路线建议

1.  先理解数学公式
2. 再读每个模块的代码
3. 自己改参数、改特征、看输出变化
4. 对照公式理解输出

---

## 9. 如果你想深入

建议继续学习：
- 多元回归 + 交互项
- 正则化与偏差方差权衡
- 统计检验（t 检验、F 检验）
- 残差诊断（自相关、异方差）
