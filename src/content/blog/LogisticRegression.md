---
title: LogisticRegression
date: 2025-12-29
tags: [机器学习, 逻辑回归]
description: asdfasd
---

### Logistic回归理论推导

#### 基本模型设定

设因变量 $y \in \lbrace 0,1 \rbrace$，自变量 $x = (1, x_1, x_2, ..., x_p)^T$，参数 $\theta = (\theta_0, \theta_1, ..., \theta_p)^T$
线性组合：
$$
z = \theta^T x = \theta_0 + \theta_1 x_1 + ... + \theta_p x_p
$$

#### Sigmoid函数引入

为将$z$映射到$[0,1]$区间，采用Sigmoid函数：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


定义条件概率：  
$$
P(y=1 | x; \theta) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

$$
P(y=0 | x; \theta) = 1 - \sigma(\theta^T x) = \frac{e^{-\theta^T x}}{1 + e^{-\theta^T x}}
$$

#### 统一概率表达式

合并两式（伯努利分布）：  
$$
P(y | x; \theta) = \left[ \sigma(\theta^T x) \right]^y \left[ 1 - \sigma(\theta^T x) \right]^{1-y}
$$


#### 似然函数构建

对于 $n$ 个独立样本 $\lbrace (x^{(i)}, y^{(i)}) \rbrace_{i=1}^n$，似然函数为：  
$$
L(\theta) = \prod_{i=1}^n P(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^n \left[ \sigma(\theta^T x^{(i)}) \right]^{y^{(i)}} \left[ 1 - \sigma(\theta^T x^{(i)}) \right]^{1-y^{(i)}} 
$$


#### 对数似然函数

取对数简化计算：  
$$
 \ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \left[ y^{(i)} \ln \sigma(\theta^T x^{(i)}) + (1-y^{(i)}) \ln \left(1 - \sigma(\theta^T x^{(i)})\right) \right] 
$$


#### 梯度计算

令$\sigma_i = \sigma(\theta^T x^{(i)})$，利用导数性质$\sigma'(z) = \sigma(z)(1-\sigma(z))$：  
$$
 \frac{\partial \ell}{\partial \theta_j} = \sum_{i=1}^n \left[ y^{(i)} \frac{1}{\sigma_i} \sigma_i (1-\sigma_i) x_j^{(i)} - (1-y^{(i)}) \frac{1}{1-\sigma_i} \sigma_i (1-\sigma_i) x_j^{(i)} \right] 
$$
化简得：  
$$
 \frac{\partial \ell}{\partial \theta_j} = \sum_{i=1}^n \left[ y^{(i)} (1-\sigma_i) x_j^{(i)} - (1-y^{(i)}) \sigma_i x_j^{(i)} \right] = \sum_{i=1}^n (y^{(i)} - \sigma_i) x_j^{(i)} 
$$
梯度向量形式：  
$$
 \nabla_{\theta} \ell = \sum_{i=1}^n (y^{(i)} - \sigma(\theta^T x^{(i)})) x^{(i)} 
$$


####  最大化似然函数

采用梯度上升法迭代更新参数（$\alpha$为学习率）：  
$$
\theta^{new} = \theta^{old} + \alpha \nabla_{\theta} \ell
$$
或定义损失函数$J(\theta) = -\ell(\theta)$后使用梯度下降法。

#### 最终算法形式

每次迭代时（向量形式）：  
$$
\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - \sigma(\theta^T x^{(i)})) x^{(i)}
$$

通常使用随机梯度下降（SGD）或批量梯度下降进行优化。

---

推导完毕。核心步骤：通过Sigmoid函数建立概率模型→构建似然函数→求导得梯度→迭代优化参数。