---
title: 分类模型评估指标完全指南：Accuracy, Precision, Recall, ROC, AUC
date: 2026-05-07
category: 神经网络/训练
tags:
  - 基础
  - 深度学习
  - 评估
description: 从混淆矩阵出发，系统讲解 Accuracy、Precision、Recall、F1、ROC 曲线与 AUC 的数学定义、直观含义、适用场景及 Python 实现。理解为什么 AUC 对类别不平衡最不敏感，以及 PR 曲线与 ROC 曲线的选择策略。
image: https://img.yumeko.site/file/blog/cover/1780581748468.webp
status: published
---

## 1. 问题的起点：模型好不好，谁说了算？

训练完一个分类模型后，第一个问题永远是：**这个模型到底有多好？**

假设你在做垃圾邮件检测——模型说某封邮件是垃圾邮件，你信还是不信？如果模型把 $99\%$ 的邮件都判为"正常"，准确率 $99\%$，它就是个好模型吗？

答案取决于：**垃圾邮件在全部邮件中占多少？** 如果只占 $1\%$，那么一个"永远说正常"的模型也能拿到 $99\%$ 的准确率——但它一封垃圾邮件也识别不出来。

这就是为什么我们需要多种评估指标，从不同角度审视模型的表现。

## 2. 混淆矩阵：一切指标的源头

所有分类指标都诞生自一张 $2 \times 2$ 的表格——**混淆矩阵**（Confusion Matrix）。

### 2.1 二分类混淆矩阵

设 $y_i \in \{0, 1\}$ 为真实标签（$1$ = 正类），$\hat{y}_i \in \{0, 1\}$ 为模型预测。定义四个计数：

$$
\begin{aligned}
TP &= \sum_i \mathbf{1}[y_i = 1 \land \hat{y}_i = 1] \quad\text{（真正例）} \\
TN &= \sum_i \mathbf{1}[y_i = 0 \land \hat{y}_i = 0] \quad\text{（真负例）} \\
FP &= \sum_i \mathbf{1}[y_i = 0 \land \hat{y}_i = 1] \quad\text{（假正例 · 误报）} \\
FN &= \sum_i \mathbf{1}[y_i = 1 \land \hat{y}_i = 0] \quad\text{（假负例 · 漏报）}
\end{aligned}
$$

|  | 预测为正（$\hat{y}=1$） | 预测为负（$\hat{y}=0$） |
|:--|:--:|:--:|
| **实际为正**（$y=1$） | $TP$ | $FN$ |
| **实际为负**（$y=0$） | $FP$ | $TN$ |

定义两个派生量：
$$
P = TP + FN \quad\text{（真实正样本总数）},\qquad N = FP + TN \quad\text{（真实负样本总数）}
$$

**记忆口诀**：第二个字母是模型的判断（Positive/Negative），第一个字母是这个判断对不对（True/False）。

### 2.2 多分类混淆矩阵

对于 $K$ 个类别，混淆矩阵 $\mathbf{C} \in \mathbb{N}^{K \times K}$，其中 $C_{ij}$ = 真实类别为 $i$、被预测为 $j$ 的样本数：

$$
\mathbf{C} = \begin{bmatrix}
C_{11} & C_{12} & \cdots & C_{1K} \\
C_{21} & C_{22} & \cdots & C_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
C_{K1} & C_{K2} & \cdots & C_{KK}
\end{bmatrix}
$$

- $\operatorname{tr}(\mathbf{C}) = \sum_{i} C_{ii}$（对角线之和）= 正确分类总数
- $C_{ij}\ (i \neq j)$ = 类别 $i$ 被误判为类别 $j$ 的数量

```python
from sklearn.metrics import confusion_matrix
import numpy as np

# allLabels: 真实标签列表, allPreds: 预测标签列表
cm = confusion_matrix(allLabels, allPreds)
print(cm)  # (num_classes, num_classes) 的矩阵

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.show()
```

## 3. 准确率（Accuracy）及其陷阱

### 3.1 定义

$$
\boxed{\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{P + N}}
$$

最直观的指标——"模型猜对的比例"。

```python
def computeAccuracy(outputs, labels):
    """计算准确率"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct / labels.size(0)
```

### 3.2 准确率的致命陷阱：类别不平衡

当 $P \ll N$（正样本极少）时，Accuracy 会撒谎。

**极端例子**：设 $P = 10$（10 封垃圾邮件），$N = 990$（990 封正常邮件），正样本占比仅为 $\frac{P}{P+N} = 1\%$。

一个**什么都不学的模型**——恒预测负类（$\hat{y}_i \equiv 0$）：

| | $\hat{y}=1$ | $\hat{y}=0$ |
|:--|:--:|:--:|
| $y=1$ | $TP = 0$ | $FN = 10$ |
| $y=0$ | $FP = 0$ | $TN = 990$ |

$$
\text{Accuracy} = \frac{0 + 990}{0 + 990 + 0 + 10} = \frac{990}{1000} = 99\%
$$

**$99\%$ 的准确率，却一封垃圾邮件都识别不出来（$TP = 0$）。**

**数学分析**：将 Accuracy 按 $TN$ 展开：

$$
\text{Accuracy} = \frac{TP + TN}{P + N} = \frac{TN}{P + N} + \frac{TP}{P + N}
$$

当 $N \gg P$ 时，$\frac{TN}{P+N} \approx \frac{N}{P+N} \to 1$，即 $TN$ 项主导了整个分数。模型只需让 $TN \approx N$（把多数类判对），就能拿高分——无需学习任何关于正类的知识。

### 3.3 Accuracy 何时可用？

- ✅ 正负样本大致均衡：$P \approx N$
- ✅ 各类错误的代价相同时
- ❌ 类别严重不均衡（$P \ll N$ 或 $N \ll P$）：如欺诈检测、疾病筛查、罕见事件预测

## 4. 精确率（Precision）与召回率（Recall）

当 Accuracy 不可靠时，需要分别考察模型在**正类**上的表现。

### 4.1 精确率（Precision）—— 不乱报

$$
\boxed{\text{Precision} = \frac{TP}{TP + FP}}
$$

**含义**：在所有被模型判为正类的样本中，真正是正类的比例。

**数值示例**：设模型判正 $100$ 封邮件（$TP + FP = 100$），其中 $TP = 80$，$FP = 20$：

$$
\text{Precision} = \frac{80}{100} = 0.80
$$

即模型喊"垃圾邮件"时，有 $80\%$ 的概率确实猜对了。

**适用场景**：误报代价高——你宁愿漏掉一些垃圾邮件（$FN$ 增大），也不愿把重要邮件扔进垃圾箱（$FP$ 减小）。

### 4.2 召回率（Recall）—— 不漏报

$$
\boxed{\text{Recall} = \frac{TP}{TP + FN} = \frac{TP}{P}}
$$

**含义**：在所有真正的正类样本中，模型找出了多少。

**数值示例**：设共有 $P = 100$ 封垃圾邮件，模型找出了 $TP = 60$ 封，漏掉了 $FN = 40$ 封：

$$
\text{Recall} = \frac{60}{100} = 0.60
$$

**适用场景**：漏报代价高——疾病筛查中，你宁愿让健康人多做一次检查（$FP$ 增大），也绝不能放过一个病人（$FN \to 0$）。

### 4.3 定义辨析

![Precision vs Recall](https://img.yumeko.site/file/blog/articles/1780581570404.webp)

$$
\begin{aligned}
\text{Precision} &= \frac{TP}{TP + FP} &&\text{—— 分母是"预测为正"的集合} \\[4pt]
\text{Recall} &= \frac{TP}{TP + FN} = \frac{TP}{P} &&\text{—— 分母是"真实为正"的集合}
\end{aligned}
$$

两者的分子相同（都是 $TP$），区别仅在分母——Precision 的分母是 $\hat{P} = TP + FP$（模型判正的集合），Recall 的分母是 $P = TP + FN$（真实正类的集合）。这是两者矛盾的根本来源。

### 4.4 代码实现

```python
from sklearn.metrics import precision_score, recall_score

# 二分类
precision = precision_score(labels, preds)      # 默认 pos_label=1
recall = recall_score(labels, preds)

# 多分类：指定平均方式
precisionMacro = precision_score(labels, preds, average='macro')  # 各类等权重
recallMicro = recall_score(labels, preds, average='micro')        # 各样本等权重
```

## 5. Precision 与 Recall 的权衡

### 5.1 为什么它们相互矛盾？

大多数分类模型输出一个概率分数 $s_i = P(y_i = 1 \mid x_i) \in [0, 1]$，而非硬标签。给定阈值 $\tau \in [0, 1]$：

$$
\hat{y}_i = \begin{cases}
1 & \text{if } s_i \ge \tau \\
0 & \text{if } s_i < \tau
\end{cases}
$$

$\tau$ 的变化重新分配了 $TP, FP, TN, FN$：

- **$\tau \uparrow$（阈值提高）**：模型更"保守"，只有非常确信时才判正 → $\hat{P} = TP + FP$ 缩小 → **$TP \downarrow$，$FP \downarrow$** → Precision 可能 $\uparrow$，Recall $\downarrow$
- **$\tau \downarrow$（阈值降低）**：模型更"激进"，稍有怀疑就判正 → $\hat{P}$ 膨胀 → **$TP \uparrow$，$FP \uparrow$** → Recall $\uparrow$，Precision 可能 $\downarrow$

极端情况的形式化：

| $\tau$ | $\hat{y}_i$ | $TP$ | $FP$ | Precision | Recall |
|:--|:--|:--|:--|:--|:--|
| $\tau = 1.0$ | 恒为 $0$ | $0$ | $0$ | 无定义（或 $=1$） | $0$ |
| $\tau = 0.0$ | 恒为 $1$ | $P$ | $N$ | $\frac{P}{P+N}$ | $1$ |

**核心矛盾**：Precision 的分母是 $\hat{P} = TP + FP$，Recall 的分母是 $P = TP + FN$。$\tau$ 变化时，$\hat{P}$ 和 $P$ 中各项此消彼长，但变化方向相反——Precision 依赖 $FP$（越小越好），Recall 依赖 $FN$（越小越好），而降低 $\tau$ 会同时减少 $FN$（好）但增加 $FP$（坏）。

### 5.2 PR 曲线的构造

PR 曲线以 $\tau$ 为隐参数，遍历所有 $\tau \in [0, 1]$，绘制 $(\text{Recall}(\tau), \text{Precision}(\tau))$ 的轨迹。形式化定义：

设模型对所有 $n$ 个样本的输出分数为 $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(n)}$（降序排列），每个分数对应真实标签 $y_{(k)}$。令阈值 $\tau_k = s_{(k)}$，计算对应 $(R_k, P_k)$，以阶梯函数连接。

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# probs: 模型输出的正类概率, labels: 真实标签
precisions, recalls, thresholds = precision_recall_curve(labels, probs)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
plt.xlabel('阈值 $\\tau$')
plt.ylabel('分数')
plt.title('Precision 与 Recall 随阈值的变化')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

你会看到两条曲线向相反方向延伸——Precision 随 $\tau$ 升高而上扬，Recall 随 $\tau$ 升高而下降。这就是权衡的视觉证据。

## 6. F1 Score：调和而非算术

### 6.1 为什么需要 F1？

需要一个单一数字综合 Precision 和 Recall。算术平均有致命缺陷：

$$
\text{Arithmetic Mean} = \frac{P + R}{2}
$$

当 $P = 1.0$，$R = 0.01$ 时：算术平均 $= 0.505$——看起来"还凑合"。但模型只找出了 $1\%$ 的正样本，是灾难。

**调和平均**对极端值更敏感——它等价于"倒数的算术平均的倒数"：

$$
\boxed{F_1 = \frac{2}{\frac{1}{P} + \frac{1}{R}} = 2 \times \frac{P \times R}{P + R}}
$$

当 $P = 1.0$，$R = 0.01$：
- $\text{Arithmetic Mean} = \frac{1.0 + 0.01}{2} = 0.505$（看起来不错 ❌）
- $F_1 = \frac{2}{\frac{1}{1.0} + \frac{1}{0.01}} \approx \frac{2}{1 + 100} \approx 0.02$（暴露了问题 ✅）

调和平均是"短板检测器"——只要 $P$ 和 $R$ 中有一个很小，其倒数就很大，支配了分母，$F_1$ 就无法拉高。这类似于电路中并联电阻的公式 $R_{\text{eq}} = \left(\frac{1}{R_1} + \frac{1}{R_2}\right)^{-1}$：总电阻由最小的那个主导。

**更一般的不等式**：对任意非负 $P, R$，有
$$
\min(P, R) \le F_1 \le \text{Arithmetic Mean}(P, R)
$$
等号成立当且仅当 $P = R$。即 $F_1$ 始终不超过算术平均，且越接近最小值——它是"悲观"的聚合。

### 6.2 $F_\beta$：加权 F Score

不是所有场景下 $P$ 和 $R$ 同等重要。$F_\beta$ 引入权重参数 $\beta > 0$：

$$
\boxed{F_\beta = (1 + \beta^2) \times \frac{P \times R}{\beta^2 P + R}}
$$

$\beta$ 的物理含义：**Recall 的重要性是 Precision 的 $\beta^2$ 倍**。

- $\beta = 1$：$F_1$，$P$ 和 $R$ 同等重要
- $\beta = 2$：$F_2$，Recall 权重 = Precision 的 $4$ 倍（漏报代价高，如疾病筛查）
- $\beta = 0.5$：$F_{0.5}$，Precision 权重 = Recall 的 $4$ 倍（误报代价高，如垃圾邮件过滤）

推导极限行为验证：$\beta \to \infty \implies F_\beta \to R$；$\beta \to 0 \implies F_\beta \to P$。

### 6.3 宏平均 vs 微平均（多分类）

对于 $K$ 个类别，设第 $k$ 类的 $TP_k, FP_k, FN_k$：

| 计算方式 | 数学定义 | 特点 |
|:--|:--|:--|
| **Micro（微平均）** | $P_{\text{micro}} = \frac{\sum_k TP_k}{\sum_k (TP_k + FP_k)}$，$R_{\text{micro}} = \frac{\sum_k TP_k}{\sum_k (TP_k + FN_k)}$ | 每个**样本**权重相同 |
| **Macro（宏平均）** | $P_{\text{macro}} = \frac{1}{K}\sum_k \frac{TP_k}{TP_k + FP_k}$，$R_{\text{macro}} = \frac{1}{K}\sum_k \frac{TP_k}{TP_k + FN_k}$ | 每个**类别**权重相同 |

当类别不平衡时，Micro 受大类主导（大类贡献了更多 $\sum TP_k$），Macro 给每个类同等话语权——小类的糟糕表现不会被大类的优秀掩盖。

```python
from sklearn.metrics import precision_recall_fscore_support

# Macro 平均 — 每个类别同等重要
precision, recall, f1, _ = precision_recall_fscore_support(
    allLabels, allPreds, average='macro'
)

# Micro 平均 — 每个样本同等重要
precision, recall, f1, _ = precision_recall_fscore_support(
    allLabels, allPreds, average='micro'
)

# 逐类查看 — 发现具体哪个类有问题
precision, recall, f1, support = precision_recall_fscore_support(
    allLabels, allPreds, average=None
)
for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
    print(f"类别 {i}: P={p:.3f}, R={r:.3f}, F1={f:.3f} (样本数={s})")
```

## 7. ROC 曲线：从排序视角看模型

### 7.1 换一个角度：把模型看作排序器

到目前为止，所有指标都依赖于一个**固定阈值 $\tau$**。但 $\tau$ 是人为选择的——与其让它掩盖模型的真实能力，不如遍历所有 $\tau$，看模型在不同切分点下的全部表现。

ROC 曲线的核心思想：**把分类模型看作一个排序器**——给正样本打高分、负样本打低分。ROC 衡量的是模型输出的**分数序列** $\{s_i\}$ 在所有可能阈值下的分类表现，而不绑定任何一个特定切分点。

### 7.2 ROC 的两个轴：类内比率

ROC（Receiver Operating Characteristic）的两个轴都是**各自类内的比值**——这是它区别于 Accuracy 的关键设计：

**横轴 — FPR（假正例率）**：

$$
\boxed{\text{FPR} = \frac{FP}{FP + TN} = \frac{FP}{N}}
$$

**含义**：在 $N$ 个真实负样本中，被模型误判为正的比例——"负样本内部的犯错率"。

**纵轴 — TPR（真正例率）**：

$$
\boxed{\text{TPR} = \frac{TP}{TP + FN} = \frac{TP}{P} = \text{Recall}}
$$

**含义**：在 $P$ 个真实正样本中，被模型正确找出的比例——"正样本内部的召回率"。TPR 就是 Recall。

### 7.3 为什么两个轴都是类内比率如此重要？

因为 $FPR$ 和 $TPR$ 分别只在自己类的内部计算：
- $FPR$ 仅依赖 $FP$ 和 $TN$（两类都是真实负样本），与正样本总数 $P$ 无关
- $TPR$ 仅依赖 $TP$ 和 $FN$（两类都是真实正样本），与负样本总数 $N$ 无关

**这意味着：** 如果把正样本数量从 $P=10$ 变为 $P=1000$，只要模型的分数分布相对位置不变，ROC 曲线的形状完全不变。这是 AUC 对类别不平衡不敏感的数学根源。

### 7.4 ROC 曲线的构造：以阈值为隐参数

设模型对所有 $n = P + N$ 个样本的输出分数降序排列为 $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(n)}$。令阈值从 $+\infty$ 逐步降至 $-\infty$（等价于从高分到低分扫描）：

1. 初始：$\tau = +\infty$，所有样本判负 → $FP = 0,\ TP = 0$ → 坐标 $(0, 0)$
2. 依次将每个样本从"负类"挪入"正类"：
   - 若该样本真实为正（$y=1$）→ $TP \mathrel{+}= 1$，点在 ROC 上**向上**移动 $\frac{1}{P}$
   - 若该样本真实为负（$y=0$）→ $FP \mathrel{+}= 1$，点在 ROC 上**向右**移动 $\frac{1}{N}$
3. 最终：$\tau = -\infty$，所有样本判正 → $TP = P,\ FP = N$ → 坐标 $(1, 1)$

ROC 曲线就是这条从 $(0,0)$ 到 $(1,1)$ 的阶梯路径。

**四个关键点**：

| $\tau$ | 坐标 $(FPR, TPR)$ | 含义 |
|:--|:--|:--|
| $\tau \to +\infty$ | $(0, 0)$ | 所有样本判负 |
| $\tau \to -\infty$ | $(1, 1)$ | 所有样本判正 |
| 理想分类器 | $(0, 1)$ | 左上角——$TP = P$ 且 $FP = 0$ |

### 7.5 ROC 的几何解读

- 曲线越靠近**左上角** $(0, 1)$ → 模型越好（高 $TPR$，低 $FPR$）
- 曲线沿**对角线** $TPR = FPR$ → 等价于随机猜测（正负样本的分数分布完全重叠）
- 曲线在**对角线下方** → 模型的排序反了（负样本分数系统性高于正样本）——取反即可修复
- **曲线形状不依赖阈值**——ROC 的每一点对应一个 $\tau$，整条曲线覆盖了全部 $\tau$，展示的是模型**内在的排序能力**

![ROC 曲线示意图](https://img.yumeko.site/file/blog/ClassificationMetrics/ROCCurve.png)

> **🖼️ AI 生图提示词：**
>
> ```
> A clean scientific diagram of a ROC curve. One main plot with FPR (False Positive Rate) on x-axis (0 to 1) and TPR (True Positive Rate) on y-axis (0 to 1). Three curves:
> 1. A perfect classifier: goes straight up from (0,0) to (0,1) then right to (1,1) — L-shaped in green, labeled "完美分类器"
> 2. A good classifier: smooth concave curve bowing toward top-left corner, in blue, labeled "实际模型 (AUC=0.85)"
> 3. A random classifier: diagonal dashed line from (0,0) to (1,1), in gray, labeled "随机猜测 (AUC=0.5)"
> The area under the blue curve is lightly shaded and labeled "AUC".
> Annotate: "左上角 = 理想" with arrow pointing to (0,1) corner.
> Clean academic style, white background, thin axis lines, LaTeX-style labels. Chinese text labels.
> ```

## 8. AUC：一个数字总结 ROC

### 8.1 定义

AUC（Area Under the Curve）是 ROC 曲线下的面积。以 $\text{FPR}$ 为积分变量：

$$
\boxed{\text{AUC} = \int_{0}^{1} \text{TPR}(t) \, dt}, \quad t = \text{FPR}
$$

取值范围：$\text{AUC} \in [0, 1]$。

| AUC 值 | 含义 |
|:--|:--|
| $\text{AUC} = 0.5$ | 随机猜测——正负样本分数分布完全重叠 |
| $\text{AUC} \approx 0.8$ | 较好的模型 |
| $\text{AUC} \ge 0.95$ | 优秀模型 |
| $\text{AUC} = 1.0$ | 完美分类器——存在某个 $\tau$ 使得 $TPR=1$ 且 $FPR=0$ |
| $\text{AUC} < 0.5$ | 模型学反了——对预测分数取负即可得到 $\text{AUC} > 0.5$ |

### 8.2 AUC 的概率等价定义（核心洞察）

AUC 有一个极其优雅的等价解释，这是理解它"为何对不平衡不敏感"的钥匙：

> 从正样本中随机抽取一个样本 $x_+$，从负样本中随机抽取一个样本 $x_-$。设模型对它们的输出分数分别为 $s_+ = f(x_+)$ 和 $s_- = f(x_-)$。则：

$$
\boxed{\text{AUC} = P(s_+ > s_-) + \frac{1}{2} P(s_+ = s_-)}
$$

如果模型输出连续分数（无平局），则简化为：

$$
\boxed{\text{AUC} = P(s_+ > s_-)}
$$

**即：AUC = 随机抽一个正样本和一个负样本，模型给正样本打更高分的概率。**

这正是 **Wilcoxon-Mann-Whitney 检验统计量** 的形式：

$$
\text{AUC} = \frac{1}{P \cdot N} \sum_{i: y_i = 1} \sum_{j: y_j = 0} \mathbf{1}[s_i > s_j]
$$

### 8.3 为什么 AUC 对类别不平衡不敏感？（数学论证）

AUC 衡量的是**排序质量**——正样本应排在负样本前面。这个性质与正负样本的**绝对数量无关**。

设正样本集合 $\mathcal{S}_+ = \{s_i \mid y_i = 1\}$，负样本集合 $\mathcal{S}_- = \{s_j \mid y_j = 0\}$。AUC 只依赖这两组分数的**相对排序**，而非 $|\mathcal{S}_+| = P$ 或 $|\mathcal{S}_-| = N$ 的大小。

如果将正样本从 $P=10$ 复制到 $P'=1000$（分数分布不变），则：
- $\text{FPR}(t) = \frac{|\{s \in \mathcal{S}_- : s \ge t\}|}{N}$ 不变（负样本集合未变）
- $\text{TPR}(t) = \frac{|\{s \in \mathcal{S}_+ : s \ge t\}|}{P}$ 不变（分子分母同比例缩放）
- ROC 曲线形状不变
- AUC 不变

对比 Accuracy：$\text{Accuracy} = \frac{TP + TN}{P + N}$，分母 $P + N$ 因 $P$ 的剧变而改变，直接导致数值剧烈波动。

**结论**：Accuracy 衡量的是"分类有多准"（依赖阈值 + 类别分布），AUC 衡量的是"排序有多好"（仅依赖分数的相对顺序）。两者衡量的是模型的不同方面。

### 8.4 代码实现

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# 计算 AUC
auc = roc_auc_score(labels, probs)
print(f"AUC: {auc:.4f}")

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(labels, probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测 (AUC = 0.5)')
plt.xlabel('FPR (假正例率)')
plt.ylabel('TPR (真正例率 / Recall)')
plt.title('ROC 曲线')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# 手动验证 AUC 的概率定义（小数据集）
def aucByDefinition(labels, probs):
    """AUC = P(s_+ > s_-)，使用 Wilcoxon-Mann-Whitney 统计量"""
    posScores = probs[labels == 1]
    negScores = probs[labels == 0]
    total = len(posScores) * len(negScores)
    wins = sum(1 for sp in posScores for sn in negScores if sp > sn)
    ties = sum(1 for sp in posScores for sn in negScores if sp == sn)
    return (wins + 0.5 * ties) / total

print(f"AUC (sklearn):     {roc_auc_score(labels, probs):.4f}")
print(f"AUC (WMW 定义):    {aucByDefinition(labels, probs):.4f}")
plt.show()
```

## 9. PR 曲线 vs ROC 曲线：何时用哪个？

### 9.1 ROC 的盲点

ROC 曲线在**极度不平衡**场景下可能过于乐观。

**数学分析**：考虑 $P \ll N$（正样本极稀少）。设模型在低 $FPR$ 区域表现差（$FP$ 不少），但 $FPR = \frac{FP}{N}$——因为分母 $N$ 巨大，$FPR$ 依然极低。ROC 曲线在低 $FPR$ 区看起来仍然"贴近左上角"，AUC 很高，但实际 Precision 可能已经崩溃。

**PR 曲线**（Precision-Recall Curve）绕过了这个问题：它完全不包含 $TN$。Precision 的分母是 $TP + FP$，直接暴露 $FP$ 的影响——$N$ 巨大时，即使 $FPR$ 很小，$FP$ 的绝对数量也可能远超 $TP$，导致 Precision 骤降。

### 9.2 数学对比

| 特性 | ROC $(FPR, TPR)$ | PR $(R, P)$ |
|:--|:--|:--|
| 横轴 | $FPR = \frac{FP}{N}$ | $R = \frac{TP}{P}$ |
| 纵轴 | $TPR = \frac{TP}{P}$ | $P = \frac{TP}{TP + FP}$ |
| 含 $TN$？ | 是（$FPR$ 分母） | **否**（完全不含 $TN$） |
| 类别不平衡 | 可能过于乐观 | 更能暴露问题 |
| 随机基线 | $TPR = FPR$（固定对角线） | $P = \frac{P}{P+N}$（随正样本占比变化） |

### 9.3 选择策略

- $\frac{P}{P+N} \approx 0.5$（正负样本大致平衡）→ ROC + AUC 足矣
- $\frac{P}{P+N} < 0.1$（正样本稀少）→ 追加 PR 曲线，关注 $P$ 在高 $R$ 区域的表现
- 不确定 → 两个都画

### 9.4 代码实现

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# PR 曲线
precisions, recalls, _ = precision_recall_curve(labels, probs)
ap = average_precision_score(labels, probs)  # PR 曲线下面积

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC 曲线'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PR
posRatio = labels.sum() / len(labels)  # 正样本占比 = 随机基线
axes[1].plot(recalls, precisions, linewidth=2, label=f'AP = {ap:.3f}')
axes[1].axhline(y=posRatio, color='k', linestyle='--', linewidth=1,
                label=f'随机基线 $P/(P+N)$ = {posRatio:.3f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR 曲线'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**注意**：PR 曲线的随机基线不是 $0.5$，而是 $\frac{P}{P+N}$——正样本占比。如果正样本仅占 $1\%$，随机模型的 Precision 上限就是 $0.01$。这是 PR 曲线比 ROC 更诚实的地方。

## 10. Top-K Accuracy 与 Per-Class Accuracy

### 10.1 Top-K Accuracy

模型输出 $K$ 个类别的概率分布 $\mathbf{p} \in [0,1]^K$。Top-K Accuracy 检查正确类别是否在概率最高的 $K$ 个类别中：

$$
\text{Top-}K\text{ Accuracy} = \frac{\sum_{i} \mathbf{1}[y_i \in \operatorname{top}_K(\mathbf{p}_i)]}{n}
$$

ImageNet 竞赛使用 $\text{Top-}5\text{ Accuracy}$（$K=1000$ 分类），因为图像中可能同时包含多个物体。

```python
def topKAccuracy(outputs, labels, k=5):
    """计算 Top-K 准确率"""
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1))
    return correct.any(dim=1).float().mean().item()
```

### 10.2 Per-Class Accuracy

逐类别准确率——模型对每个类别 $c$ 分别计算：

$$
\text{Accuracy}_c = \frac{C_{cc}}{\sum_{j} C_{cj}}
$$

其中 $C_{cc}$ 是类别 $c$ 被正确分类的样本数，$\sum_j C_{cj}$ 是类别 $c$ 的真实样本总数。

```python
def perClassAccuracy(outputs, labels, numClasses):
    """计算每个类别的准确率"""
    _, predicted = outputs.max(1)
    accuracies = []
    for c in range(numClasses):
        idx = labels == c
        if idx.sum() == 0:
            accuracies.append(float('nan'))
        else:
            correct = predicted[idx].eq(labels[idx]).sum().item()
            accuracies.append(correct / idx.sum().item())
    return accuracies
```

## 11. 完整的评估流程

```python
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


@torch.no_grad()
def evaluate(model, loader, criterion, device, numClasses):
    """
    完整评估函数

    Args:
        model: PyTorch 模型
        loader: DataLoader
        criterion: 损失函数
        device: 设备
        numClasses: 类别数

    Returns:
        dict: 包含所有评估指标的字典
    """
    model.eval()
    allPreds, allLabels, allProbs = [], [], []
    totalLoss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # 损失
        loss = criterion(outputs, labels)
        totalLoss += loss.item() * images.size(0)

        # 概率
        probs = torch.softmax(outputs, dim=1)

        # 预测
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        allPreds.extend(predicted.cpu().tolist())
        allLabels.extend(labels.cpu().tolist())
        allProbs.extend(probs.cpu().tolist())

    # 基础指标
    accuracy = correct / total
    avgLoss = totalLoss / total

    # 转换为 numpy
    allPreds = np.array(allPreds)
    allLabels = np.array(allLabels)
    allProbs = np.array(allProbs)

    # 混淆矩阵
    cm = confusion_matrix(allLabels, allPreds)

    # 分类报告（含 Precision/Recall/F1 per class）
    report = classification_report(allLabels, allPreds, digits=4)

    # AUC
    if numClasses == 2:
        auc = roc_auc_score(allLabels, allProbs[:, 1])
    else:
        auc = roc_auc_score(allLabels, allProbs, multi_class='ovr', average='macro')

    return {
        'loss': avgLoss,
        'accuracy': accuracy,
        'confusionMatrix': cm,
        'classificationReport': report,
        'auc': auc,
        'predictions': allPreds,
        'labels': allLabels,
        'probabilities': allProbs,
    }
```

## 12. 总结对比

### 12.1 各指标数学定义汇总

$$
\begin{aligned}
\text{Accuracy} &= \frac{TP + TN}{P + N} \\[6pt]
\text{Precision} &= \frac{TP}{TP + FP} \\[6pt]
\text{Recall} = \text{TPR} &= \frac{TP}{P} \\[6pt]
\text{FPR} &= \frac{FP}{N} \\[6pt]
F_\beta &= (1 + \beta^2) \frac{P \cdot R}{\beta^2 P + R} \\[6pt]
\text{AUC} &= P(s_+ > s_-) \quad\text{（连续分数、无平局时）} \\[6pt]
\text{AP} &= \int_0^1 P(R) \, dR \quad\text{（PR 曲线下面积）}
\end{aligned}
$$

### 12.2 各指标对类别不平衡的敏感度

| 指标 | 受不平衡影响？ | 原因 |
|:--|:--:|:--|
| **Accuracy** | ⚠️ 最严重 | 分母 $P+N$ 被多数类主导 |
| **Precision** | 部分 | $FP$ 涌入时被稀释，分母 $\hat{P}$ 随阈值变化 |
| **Recall** | 不直接 | 分母 $P$ 只在正样本内部，与 $N$ 无关 |
| **F1** | 继承 P 和 R | 调和平均传递了两者的特性 |
| **AUC** | ✅ 最不敏感 | $FPR$ 和 $TPR$ 都是类内比率；$P(s_+ > s_-)$ 只看排序 |
| **AP** | 敏感 | Precision 不含 $TN$，直接暴露不平衡的影响 |

### 12.3 指标选择决策树

```
你的数据均衡吗？
├── 均衡 (P ≈ N)  → Accuracy + ROC/AUC 为主，Confusion Matrix 辅助
└── 不均衡 (P ≪ N 或 P ≫ N)
    ├── 误报代价高 → 关注 Precision + PR 曲线
    ├── 漏报代价高 → 关注 Recall + PR 曲线
    └── 需要综合   → F1 + AUC 双指标
```

### 12.4 关键洞察

1. **没有万能指标**——每个指标只看到模型的一个侧面，完整评估必须多指标并行
2. **Accuracy 会撒谎**——当 $P \ll N$ 时，$TN$ 主导分母，一个恒判负类的模型也能拿高分
3. **Precision 和 Recall 是跷跷板**——由阈值 $\tau$ 耦合，$\tau \uparrow \implies P \uparrow, R \downarrow$；$\tau \downarrow \implies R \uparrow, P \downarrow$
4. **F1 是短板检测器**——调和平均对极端值远比算术平均敏感，只要 $P$ 或 $R$ 有一个很低，$F_1$ 就拉不起来
5. **AUC 衡量排序能力**——$\text{AUC} = P(s_+ > s_-)$，与阈值无关，与类别分布无关，是模型"内在判别力"的度量
6. **PR 曲线更诚实**——在正样本稀少时，它完全不受 $TN$ 的干扰，能暴露 ROC 隐藏的问题
7. **评估指标的选择本身就是领域知识**——不理解业务场景中 $FP$ 和 $FN$ 的代价比，就无法选对指标
