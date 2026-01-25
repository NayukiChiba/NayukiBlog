---
title: fghjkfghjfghj
date: 2025-12-29
tags: [测试]
description: dfghjfghjfghj
---

# 关于数学公式无法渲染！！！！

因为github作为一个前端，使用的是MathJax渲染数学公式，但是markdown使用的是Latex渲染，有一些公式无法正常显�?
如果想看公式，可以clone下来之后用typora或者vscode查看�?


# Machine-Learning-Algorithms

这是一个学习机器学�?*算法**和机器学�?*实战**的repository

| 知识�?  | 内容                                               |
| -------- | -------------------------------------------------- |
| 分类算法 | 逻辑回归，决策树，支持向量机，集成算法，贝叶斯算�?|
| 回归算法 | 线性回归，决策树，集成算法                         |
| 聚类算法 | k-means，dbscan�?                                 |
| 降维算法 | 主成分分析，线性判别分析等                         |
| 进阶算法 | GBDT提升算法，lightgbm，，EM算法，隐马尔科夫模型   |

## 算法

### 分类

#### LogisticRegression

##### 问题

一组特征变�?\mathbb{x} = (x_1, x_2, \dots, x_m) \in \mathbb{R}^{n \times m}$，一组标�?\mathbb{y} = (y_1, y_2, \dots, y_m), y_i \in \{-1, +1 \}$

目标是估计条件概�?P(Y = 1 | X)$，即给定特征$X = x$的情况下�?y = 1$的概率�?

##### 问题分析

如何用LogisticRegression做一个分类算�?

1. 逻辑回归模型可以表示�?

$$
z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n = \beta^T\mathbf{(1;x)}
$$

其中$\beta = (\beta_0, \beta_1, \dots\beta_n)^T, (1;x) = (1, x_1, x_2, \dots, x_n)^T$ 

2. 利用Sigmoid函数，将$z$变成概率

$$
h(x) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

这里$h(x)$表示：输入一�?x$，就可以判断这个$x$�?1$还是$0$



3. 训练目标：极大似然估�?

* 对于每一个输�?x^{(i)}$，我们希�?\mathbb{P}(h(x^{(i)}) = y^{(i)})$越大越好，这样才比较�?
* 因为$h(x^{(i)})$�?y^{(i)}$只有**1**�?*0**两个取值，所以可以得�?\mathbb{P}(h(x^{(i)}) = y^{(i)}) = h(x^{(i)})^{y^{(i)}}(1 - h(x^{(i)}))^{y^{(i)}}$，对于这个式子怎么推导出来的，请自行带�?h(x^{(i)})$�?y^{(i)}$�?�?的四种情�?
* 得到极大似然估计�?对于n个数据样�?

$$
L(\beta) = \displaystyle\prod_{i=1}^n h(x^{(i)})^{y^{(i)}}(1 - h(x^{(i)}))^{y^{(i)}}
$$

* 负对数似�?

$$
l(\beta) = -\sum_{i=1}^n \left[y^{(i)}log(h(x^{(i)})) + (1-y^{(i)})log(1-h(x^{(i)}))     \right]
$$



4. 优化方法

* 梯度下降�?

$$
\beta_j := \beta_j + \alpha\frac{\partial l}{\partial \beta_j}
$$

$$
\frac{\partial l}{\partial \beta_j} = -\sum_{i=1}^n (h(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

5. 加上正则化参数，然后整合

* 负对数似然，加上正则化参�?

$$
l(\beta) = -\sum_{i=1}^n \left[y^{(i)}log(h(x^{(i)})) + (1-y^{(i)})log(1-h(x^{(i)}))     \right] + \frac{\lambda}{2}\sum_{i=1}^n \beta_i^2
$$

* 梯度下降的时候，加上正则化参�?

$$
\frac{\partial l}{\partial \beta_j} = -\sum_{i=1}^n (h(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda\sum_{i=1}^n \beta_i
$$

* 参数修正后，这里的修正蕴含在上面的求导中

$$
\beta_j := \beta_j + \alpha\frac{\partial l}{\partial \beta_j}
$$

#### SVM

##### 问题分析

只考虑**二分�?*问题�?假设�?n$个训练点${x_i \in \mathbb{R}^N}, i = 1, 2 ,\dots, n$，每一个训练点有一个指�?y_i \in \{-1, +1\}$

问题：给一个输�?x$，如何判断他的标�?y$

方案：找到一个函�?g:\mathbb{R}^N \rightarrow \mathbb{R}$，然后定义决策函数为
$$
f(x) = \sigma(g(x))
$$
其中$\sigma(\cdot)$为激活函数，这里我们�?\sigma(\cdot) = sgn(\cdot)$ �?

和LogisticRegression都是同样的输入，也都是想要结果预测正确，那两者有什么区别呢�?
**区别�?*

* LogisticRegression目的是输出概率值，概率的阈值定�?.5，大�?.5预测�?，小�?.5预测�?。原数据的数据部分和标签部分对回归的影响非常大！一个标签的错误，可能会让这个回归结果偏移，导致预测错误�?
* SVM并不是想预测概率，单纯是想将两种数据分开，通过各种手段将数据分开，是软间隔分开还是硬间隔分开并不重要，重要的只有所有的向量支撑起来�?*超平�?*�?

<!-- ![SVM类型](./assets/SVM_1.png) -->

##### 硬间�?

<!-- ![SVM_2](./assets/SVM_2.png) -->

1. 假设划分超平面的线性方�?

$$
w^T x + b = 0 \\
w = (w_1, w_2, \dots, w_n)
$$



样本到超平面$w^Tx+b=0$的距离为$\gamma = \displaystyle\frac{|w^Tx+b|}{||w||}$�?当分类为硬间隔时
$$
w^T x_i + b \geq +1, y_i = +1 \\
w^T x_i + b \leq -1, y_i = -1
$$
所以所有的样本中，找到**离超平面最近的样本�?*，这几个样本点会让上面的不等式的等号**成立**，被称为**支持向量**，这些在超平�?*两端**的样本点，到超平面的距离之和�?
$$
\gamma = \frac{2}{||w||}
$$
想要找到**最大间�?*划分的超平面，就是要赵大鹏可以满足上式的参数$w$�?b$，即
$$
\max_{w,b} \frac{2}{||w||} \\
s.t.\,\, y_i(w^Tx_i+b) \geq 1, i = 1, 2, \dots, m
$$
显然这个问题可以转化成一个最小化问题
$$
\min_{w,b} \frac{1}{2}||w||^2 \\
s.t. \,\, y_i(w^Tx_i+b) \geq 1, i = 1, 2, \dots, m
$$
这就是SVM的基本形�?

##### 对偶问题

关于这个最小化问题 
$$
\min_{w,b} \frac{1}{2}||w||^2 \\
s.t. \,\, y_i(w^Tx_i+b) \geq 1, i = 1, 2, \dots, m
$$
我们可以得到**Lagrange函数**
$$
L(w,b,\alpha) = \frac{1}{2}||w||^2 + \sum_{i=1}^m \alpha_i(1-y_i(w^Tx_i+b))
$$
其中$\alpha \in \mathbb{R}^m$，令$L(w,b,\alpha)$�?w$�?b$求导�?得到
$$
\begin{align*}
\frac{\partial L}{\partial w} &= w - \sum_{i=1}^m \alpha_iy_ix_i = 0 \\
\frac{\partial L}{\partial b} &= \sum_{i=1}^m \alpha_iy_i = 0
\end{align*}
$$
将上�?*回代到Lagrange函数消去$w$�?b$**得到原问题的对偶问题
$$
\begin{align*}
\max_\alpha &  \,\, \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j \\
s.t.& \,\,\, \sum_{i=1}^{m}\alpha_i y_j = 0, \\
& \,\, \alpha_i \geq 0, i = 1, 2, \dots, m
\end{align*}
$$
解出$\alpha$后，求出$w$�?b$即可得到模型
$$
f(x) = w^T x + b = \sum_{i=1}^m \alpha_i y_i x_i^T x + b
$$
上述过程要求满足**KKT条件**
$$
\begin{cases}
\alpha \geq 0 \\
y_if(x_i) -1 \geq 0 \\
\alpha_i(y_i f(x_i) - 1) = 0
\end{cases}
$$

##### 核函�?

在对偶问题中
$$
\begin{align*}
\max_\alpha &  \,\, \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j \\
s.t.& \,\,\, \sum_{i=1}^{m}\alpha_i y_j = 0, \\
& \,\, \alpha_i \geq 0, i = 1, 2, \dots, m
\end{align*}
$$
这里出现了内�?<x_i, x_j>$，我们将$x$映射�?\phi(x)$上，得到特征空间$\Phi$上的超平面，模型表示�?
$$
f(x) = w^T\phi(x) + b
$$
将前面所有的$x_i^Tx_j$全部改成$\phi^T(x_i)\phi(x_j)$重新计算即可，记核函�?\kappa(x_i,x_j) = \phi^T(x_i)\phi(x_j)$

常用的核函数

<!-- ![SVM_3](./assets/SVM_3.png) -->

##### 软间�?

软间隔就是在硬间隔基础上，可以允许一些样本的出错，即允许某些样本出现
$$
y_i(w^Tx_i+b) \leq 1
$$


<!-- ![SVM_4](./assets/SVM_4.png) -->

  引入**惩罚�?*$C$，优化目标改写为
$$
\min_{w,b}\frac{1}{2}||w||^2 + C\sum_{i=1}^m \mathcal{l}_{0/1}(y_i(w^Tx_i+b) - 1)
$$
其中$C > 0$是一个常数， $\mathcal{l}_{0/1}$是一�?*0/1损失函数**，以下是常用�?*hinge损失函数**
$$
\mathcal{l}_{0/1}(x) = max(0, 1 - x)
$$
问题变成
$$
\min_{w,b}\frac{1}{2}||w||^2 + C\sum_{i=1}^m \max(0, 1 - y_i (w^T x_i +b))
$$
引入**松弛变量**$\xi_i = \max(0, 1 - y_i (w^T x_i +b))\geq 0$�?问题变成
$$
\begin{align*}
\max_{w,b,\xi_i}  \,\,\, &\frac{1}{2}||w||^2 + C\sum_{i=1}^{m}\xi_i \\
\text{s.t.}  \,\,\, & y_i (w^T x_i +b) \geq 1 - \xi_i \\
& \xi_i \geq 0, i = 1, 2, \dots , m
\end{align*}
$$
然后构�?*Lagrange函数**
$$
L(w,b,\alpha, \xi, \mu) = \frac{1}{2}||w||^2 + C\sum_{i=1}^m\xi_i + \sum_{i=1}^m \alpha_i(1 - \xi_i - y_i(w^Tx_i+b)) - \sum_{i=1}^{m}\mu_i\xi_i
$$
其中$\alpha_i \geq 0, \mu_i \geq 0$为Lagrange乘子，令$L(w,b,\alpha,\xi,\mu)$�?2,b,\xi_i$求偏导为0可以得到
$$
\begin{align*}
\frac{\partial L}{\partial w} &= w - \sum_{i=1}^m \alpha_iy_ix_i  = 0 \\
\frac{\partial L}{\partial b} &= \sum_{i=1}^m \alpha_iy_i = 0\\
\frac{\partial L}{\partial \xi_i} &= C - \alpha_i - \mu_i , i = 1, 2, \dots, m
\end{align*}
$$
**回代到Lagrange函数**，消�?w,b,\xi_i$得到**对偶问题**
$$
\begin{align*}
\max_\alpha \,\,\, & \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{s.t.} \,\,\,& \sum_{i=1}^m \alpha_iy_i = 0, \\
& 0 \leq \alpha_i \leq C, i = 1, 2, \dots, m
\end{align*}
$$
同样要满�?*KKT条件**
$$
\begin{cases}
\alpha_i \geq 0, \mu_i \geq 0 \\
y_i f(x_i) - 1 + \xi_i \geq 0 \\
\alpha_i(y_i f(x_i) - 1 + \xi_i) = 0\\
\xi_i \geq 0, \mu_i\xi_i =0
\end{cases}
$$




#### DecisionTree

#### 贝叶斯分类器

#### 集成学习Boosting

### 回归

#### 线性回归LinearRegression

#### 决策树DecisionTree

### 聚类

#### K-Means

#### DBscan

### 降维

#### PCA

#### LDA

#### 集成学习Boosting

### 其他

## 实战
