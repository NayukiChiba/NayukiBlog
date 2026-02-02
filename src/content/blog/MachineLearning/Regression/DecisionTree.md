---
title: 决策树回归数学基础与完整实践指南
date: 2026-01-26
category: MachineLearning/Regression
tags:
  - Python
  - 回归
  - Scikit-learn
description: 决策树回归的完整教程，从数学原理到代码实践，涵盖了数据探索、模型训练、评估、可视化及常见问题解答的全流程。
image: https://img.yumeko.site/file/blog/DecisionTree.png
status: public
---
# 决策树回归数学基础（CART）

这一章先讲**决策树回归**的原理与核心公式，便于后续理解代码。

---
## 1. 决策树回归是什么

- 决策树回归是一种**非参数**模型，能拟合**非线性关系**。
- 它通过一系列“如果…那么…”的规则，把数据分成多个区域。
- 每个叶子节点输出一个数值（通常是该区域样本的平均值）。

---
## 2. 树的结构

- **根节点**：所有样本的起点
- **内部节点**：对某个特征进行切分（如 `MedInc <= 3.2`）
- **叶子节点**：给出预测值（回归问题中一般是平均值）

---

## 3. 回归树的切分准则

回归树的目标：让切分后的“子节点更纯”。

最常用的指标是**均方误差（MSE）**或**方差**。

### 3.1 节点内的误差

假设一个节点包含样本集合 $R$，回归树在该节点的预测是均值：

$$
\hat{y}_R = \frac{1}{|R|} \sum_{i \in R} y_i
$$

节点误差（MSE）：

$$
\text{MSE}(R) = \frac{1}{|R|} \sum_{i \in R} (y_i - \hat{y}_R)^2
$$

### 3.2 切分增益

一个切分将节点分成左子集 $R_L$ 和右子集 $R_R$：

目标是最小化：

$$
\text{MSE}(R_L) + \text{MSE}(R_R)
$$

也可以理解为：

> 选择能让误差下降最多的切分。

---

## 4. 贪心策略（CART）

决策树在每一步都采用**贪心**策略：

1) 在所有特征上尝试所有可能切分
2) 选择让误差下降最多的切分
3) 递归继续

优点：简单高效
缺点：可能不是全局最优

---

## 5. 叶子节点的预测

回归树在叶子节点的预测值通常为：

$$
\hat{y} = \frac{1}{n_{leaf}} \sum y_i
$$

即：**叶子中样本的平均值**。

---

## 6. 过拟合与控制

决策树容易过拟合，因此需要控制复杂度：

- `max_depth`：最大深度
- `min_samples_split`：节点继续分裂的最少样本数
- `min_samples_leaf`：叶子节点最少样本数
- `max_features`：每次分裂考虑的特征数

这些参数越小，模型越复杂，越容易过拟合。

---

## 7. 特征重要性

sklearn 提供的 `feature_importances_` 基于 **不纯度下降**：

- 某特征在各节点带来的误差降低越大，重要性越高。
- 这是一种“启发式”度量，不代表因果关系。

---

## 8. 决策树 vs 线性回归

| 对比 | 线性回归 | 决策树回归 |
|------|----------|------------|
| 关系假设 | 线性 | 可拟合非线性 |
| 特征缩放 | 通常需要 | 不需要 |
| 解释性 | 高 | 规则型可解释 |
| 过拟合 | 相对少 | 容易过拟合 |

---

## 9. 小结

- 决策树回归用**分段规则**逼近复杂函数
- 切分标准是“让子节点更纯（误差更小）”
- 需要控制深度和叶子大小避免过拟合

理解这些原理后，再看代码会更清晰。

# 加载数据集（generate_data.py）

本模块负责加载 **California Housing** 经典房价数据集，并返回 `DataFrame`。

---

## 1. 数据集简介

California Housing 是经典的回归数据集，特点：

- 特征全部是数值型
- 目标变量是房价中位数
- 适合回归模型学习

常见特征包括：

- `MedInc`：居民收入中位数
- `HouseAge`：房屋年龄中位数
- `AveRooms`：平均房间数
- `AveBedrms`：平均卧室数
- `Population`：区域人口
- `AveOccup`：平均居住人数
- `Latitude` / `Longitude`：地理位置

---

## 2. 代码结构

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
```

- `as_frame=True` 会直接返回 `pandas.DataFrame`
- 第一次加载可能需要联网下载数据

---

## 3. 目标变量改名

原始数据集中，目标列名为 `MedHouseVal`（单位 10 万美元）。

为了统一项目风格，代码改名为：

```python
df = df.rename(columns={"MedHouseVal": "price"})
```

---

## 4. 返回结果

```python
return df
```

输出的 `DataFrame` 将包含：
- 8 个特征列
- 1 个目标列 `price`

---

## 5. 小结

- 本模块只负责“加载 + 统一列名”
- 数据的清洗和划分在后续模块完成

# 数据探索（explore_data.py）

这一模块帮助你快速理解数据的规模、分布和相关性。

---

## 1. 样本结构

```python
print(f"样本数量: {len(data)}")
print(f"特征数量: {len(data.columns) - 1}")
print(f"特征名称: {list(data.columns[:-1])}")
```

作用：
- 了解数据量
- 确认特征列和目标列是否完整

---

## 2. 描述统计

```python
print(data.describe().round(2))
```

输出包括：均值、标准差、最大最小值、分位数。

用途：
- 判断特征的范围
- 发现异常值

---

## 3. 缺失值检查

```python
missing = data.isnull().sum()
```

California Housing 数据集默认无缺失值，但这是标准流程。

---

## 4. 相关性分析

```python
correlation = data.corr()["price"].drop("price").sort_values(ascending=False)
```

- 用皮尔逊相关系数观察线性相关性
- 相关性只是参考，不代表因果

---

## 5. 小结

- 这一步是建模前的“体检”
- 帮助判断数据是否适合当前模型
- 相关性高的特征通常更重要

# 数据可视化（visualize_data.py）

这一模块绘制特征分布、相关性热力图和特征与价格的散点关系。

---

## 1. 输出目录

图像保存到：

```
outputs/DecisionTree/
```

对应文件名：
- `01_feature_distribution.png`
- `02_correlation_heatmap.png`
- `03_feature_relationship.png`

---

## 2. 特征分布图

```python
axes[row, col].hist(data[feature], bins=30, color='skyblue')
```

作用：
- 查看每个特征的取值范围和分布
- 判断是否存在偏态或异常值

![01_feature_distribution](https://img.yumeko.site/file/articles/DecisionTree/01_feature_distribution.png)

---

## 3. 相关性热力图

```python
corr = data.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
```

作用：
- 观察特征之间以及与 `price` 的线性相关程度

![02_correlation_heatmap](https://img.yumeko.site/file/articles/DecisionTree/02_correlation_heatmap.png)

---

## 4. 特征 vs 价格散点图

```python
selected = ["MedInc", "AveRooms", "HouseAge", "Latitude"]
```

只绘制几个代表性特征，避免图像过于拥挤。

![03_feature_relationship](https://img.yumeko.site/file/articles/DecisionTree/03_feature_relationship.png)

---

## 5. 中文字体设置

```python
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
```

如果你的电脑没有这些字体，可以换成其他中文字体。

---

## 6. 小结

- 可视化帮助判断数据分布是否合理
- 决策树可以处理非线性关系，但可视化仍然很有价值

# 数据预处理（preprocess_data.py）

这一模块负责划分训练集/测试集。决策树不需要标准化，但必须保证评估公平。

---

## 1. 分离特征与目标

```python
features = data.drop("price", axis=1)
target = data["price"]
```

- `features` 是输入 X
- `target` 是输出 y

---

## 2. 划分训练集/测试集

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=test_size, random_state=random_state
)
```

- `test_size=0.2`：20% 用作测试集
- `random_state`：保证结果可复现

---

## 3. 为什么不做标准化

决策树只根据“大小比较”做切分，不依赖距离计算：

- 特征缩放不会改变排序关系
- 因此标准化不是必须的

---

## 4. 返回值说明

```text
X_train, X_test, y_train, y_test, features, target
```

- `features` 和 `target` 方便后续可视化或分析

---

## 5. 小结

- 决策树模型训练不需要标准化
- 训练/测试划分是必须的基本步骤

# 模型训练（train_model.py）

这一模块训练决策树回归模型，并输出树的深度和叶子节点数量。

---

## 1. 核心模型

```python
from sklearn.tree import DecisionTreeRegressor
```

这是 sklearn 提供的 CART 回归树实现。

---

## 2. 关键参数

```python
DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=3,
    random_state=42
)
```

参数解释：
- `max_depth`：限制树的最大深度，防止过拟合
- `min_samples_split`：节点继续分裂需要的最小样本数
- `min_samples_leaf`：叶子节点最少样本数
- `random_state`：结果可复现

---

## 3. 训练过程

```python
model.fit(X_train, y_train)
```

训练后可以输出：
- 树深度：`model.get_depth()`
- 叶子节点数：`model.get_n_leaves()`

---

## 4. 装饰器与计时

代码中用了两个工具：

- `@print_func_info`：打印函数调用
- `@timeit` + `timer`：记录训练耗时

这能帮助你观察模型训练速度。

---

## 5. 小结

- 决策树回归训练非常快
- 参数越大，树越复杂，越容易过拟合
- 你可以调节参数观察指标变化

# 模型评估（evaluate_model.py）

这一模块评估决策树回归模型的效果，包含训练集与测试集指标。

---

## 1. 评估指标

### MSE
$$
\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2
$$

### RMSE
$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

### MAE
$$
\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|
$$

### R²
$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

---

## 2. 预测

```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

---

## 3. 训练集与测试集指标

两者都需要看：

- 训练集好、测试集差 → 过拟合
- 训练集差、测试集差 → 欠拟合

---

## 4. 过拟合检查

代码比较训练集与测试集 R² 差值：

```python
r2_diff = train_r2 - test_r2
```

- `< 0.05`：泛化良好
- `< 0.1`：轻微过拟合
- `>= 0.1`：可能过拟合

---

## 5. 小结

- 指标越好越不一定可信，要结合训练/测试表现
- 过拟合是决策树常见问题

# 结果可视化（visualize_results.py）

这一模块绘制预测效果、残差分析、特征重要性和树结构。

---

## 1. 输出文件

图像保存到：

```
outputs/DecisionTree/
```

对应文件名：
- `04_prediction_effect.png`
- `05_residual_analysis.png`
- `06_feature_importance.png`
- `07_tree_structure.png`

---

## 2. 预测值 vs 真实值

- 越接近对角线越好
- 可以直观看到整体拟合效果

![04_prediction_effect](https://img.yumeko.site/file/articles/DecisionTree/04_prediction_effect.png)

---

## 3. 残差分析

残差定义：

$$
\text{残差} = y - \hat{y}
$$

理想残差应该：
- 分布在 0 附近
- 无明显结构

![05_residual_analysis](https://img.yumeko.site/file/articles/DecisionTree/05_residual_analysis.png)

---

## 4. 特征重要性

```python
importances = model.feature_importances_
```

- 数值越大，说明该特征在分裂时贡献越多
- 仅代表“模型使用频率”，不等价因果关系

![06_feature_importance](https://img.yumeko.site/file/articles/DecisionTree/06_feature_importance.png)

---

## 5. 决策树结构可视化

```python
plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=4
)
```

建议：
- 树太深时，图会挤在一起
- 可以通过 `max_depth` 只画前几层
- 增大 `figsize` 和 `fontsize` 提高清晰度

![07_tree_structure](https://img.yumeko.site/file/articles/DecisionTree/07_tree_structure.png)

---

## 6. 小结

- 可视化能帮助判断“预测是否合理”
- 特征重要性和树结构更利于解释模型

# 一键运行完整流程（main.py）

`main.py` 是决策树项目的入口文件，顺序调用所有模块。

---

## 1. 执行顺序

```text
1. 加载数据
2. 数据探索
3. 数据可视化
4. 数据预处理
5. 模型训练
6. 模型评估
7. 结果可视化
```

---

## 2. 关键代码

```python
df = generate_data()
correlation = explore_data(df)
visualize_data(df)
X_train, X_test, y_train, y_test, X, y = preprocess_data(df)
model = train_model(X_train, y_train, max_depth=6, min_samples_split=6, min_samples_leaf=3)
y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
visualize_results(y_train, y_train_pred, y_test, y_test_pred, model, X.columns.tolist())
```

---

## 3. 可调参数

你可以通过以下参数控制模型复杂度：

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

修改这些参数后，观察评估指标和树结构的变化。

---

## 4. 小结

这个脚本的作用就是：

> 用最少的代码，一键跑完整个决策树回归流程。

# 常见问题与延伸（FAQ）

---

## 1. 为什么不标准化？

决策树只比较大小，不依赖距离或梯度：
- 缩放不会改变“谁大谁小”的顺序
- 因此标准化不是必须的

---

## 2. 为什么容易过拟合？

树可以不断切分，直到叶子节点只剩很少样本，模型会“记住”训练集。

解决方法：
- 限制 `max_depth`
- 增大 `min_samples_split` / `min_samples_leaf`
- 使用剪枝或集成方法（随机森林）

---

## 3. 训练集好、测试集差怎么办？

这是过拟合：
- 降低树深度
- 增大叶子样本数
- 减少特征

---

## 4. 训练集和测试集都差？

可能是欠拟合：
- 增大 `max_depth`
- 减小 `min_samples_leaf`
- 增加特征

---

## 5. 特征重要性可靠吗？

- 它反映“在树中被使用的程度”
- 不代表真实因果关系
- 对于高度相关的特征，重要性可能被分摊

---

## 6. 决策树适合什么场景？

- 需要解释规则的场景
- 特征关系非线性
- 数据量中小规模时效果不错

---

## 7. 如何进一步提升效果？

- 使用 **随机森林**（Bagging）
- 使用 **梯度提升树**（XGBoost / LightGBM）
- 做更好的特征工程

---

## 8. 如何理解树结构？

树结构实际上是一系列规则：

> 如果 `MedInc <= 3.1` 且 `HouseAge > 25`，预测值为 2.1

你可以逐层读取规则，理解模型决策。
