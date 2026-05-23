---
title: 评估指标详解：模型到底有多好
date: 2026-05-07
category: NeuralNetwork/Theory
tags:
  - 基础
  - 深度学习
description: 从 Accuracy 到混淆矩阵到 F1 分数，理解分类模型的完整评估体系。正确挑选loss函数是做好评估的前提。
image: https://img.yumeko.site/file/blog/EvaluationMetrics.png
status: published
---

## 1. 准确率（Accuracy）

$$
\text{Accuracy} = \frac{\text{正确预测数量}}{\text{总样本数量}} \times 100\%
$$

最直观的指标，也是最常用的。但仅靠准确率不够——当类别不均衡时（比如某类只有 1% 的样本），99% 的准确率也可能意味着模型完全忽略了少数类。

```python
def computeAccuracy(outputs, labels):
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct / labels.size(0)
```

## 2. 混淆矩阵（Confusion Matrix）

混淆矩阵是一个 $N \times N$ 的表格（$N$ = 类别数），行表示真实类别，列表示预测类别。

```
           预测
         0  1  2  3  4  5  6  7  8  9
真   0  [980  0  1  0  0  0  0  0  0  0]
实   1  [  0 1135 0  0  0  0  0  0  0  0]
     2  [  0  0 1032 0  0  0  0  0  0  0]
     3  [  0  0  0 1010 0  0  0  0  0  0]
     ...
```

- **对角线**上的数字 = 正确分类的样本数
- **非对角线**上的数字 = 错误分类的样本（能看出被误判成了什么）

对于 FashionMNIST，最常见的混淆发生在相似的服装类别之间，例如 "T-shirt" 和 "Shirt"，"Sneakers" 和 "Ankle Boots"。

### 从混淆矩阵发现问题

```
真实 3（猫）→ 100 被预测为 3（猫），20 被预测为 5（狗）
真实 5（狗）→ 110 被预测为 5（狗），15 被预测为 3（猫）
```

这说明模型容易在猫和狗之间混淆——可能需要更多的训练数据，或者这两个类别确实相似。

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(allLabels, allPreds)
print(cm)  # (num_classes, num_classes) 的矩阵
```

## 3. 精确率、召回率、F1 分数

对于类别不平衡问题，需要更细致的指标。

### 精确率（Precision）

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

"模型说是 A 类的样本中，真正是 A 类的比例"——衡量"不乱报"

### 召回率（Recall）

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

"所有 A 类样本中，模型找出了多少"——衡量"不漏报"
![PrecisionRecall.png](https://img.yumeko.site/file/articles/EvaluationMetrics/PrecisionRecall.png)

### F1 分数

精确率和召回率的调和平均：

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**为什么用调和平均而不用算术平均？** 如果 Precision=1.0 而 Recall=0.01，算术平均是 0.505（看起来还不错），但调和平均只有约 0.02——这个模型实际上很差（它只找出了 1% 的正样本）。调和平均对极端值更敏感，不会让一个极端指标掩盖另一个。

### 宏平均 vs 微平均

| 计算方式 | 公式 | 特点 |
| --- | --- | --- |
| Micro（微平均） | 先汇总所有类的 TP/FP/FN，再算指标 | 每个样本权重相同 |
| Macro（宏平均） | 先算每个类的指标，再取平均 | 每个类别权重相同 |

当类别不平衡时，Micro 受大类主导，Macro 给小类同等权重。

```python
from sklearn.metrics import precision_recall_fscore_support

# Macro 平均
precision, recall, f1, _ = precision_recall_fscore_support(
    allLabels, allPreds, average='macro'
)

# Micro 平均
precision, recall, f1, _ = precision_recall_fscore_support(
    allLabels, allPreds, average='micro'
)

# Per-class
precision, recall, f1, support = precision_recall_fscore_support(
    allLabels, allPreds, average=None
)
```

## 4. Top-K Accuracy

Top-K Accuracy 检查正确类别是否在预测概率最高的 K 个类别中。

$$
\text{Top-K Accuracy} = \frac{\text{正确类别在 Top-K 中的样本数}}{\text{总样本数}}
$$

ImageNet 竞赛标准使用 Top-5 Accuracy（因为图像中可能包含多个物体）。对于 1000 分类，Top-1 比 Top-5 严格得多。

```python
def topKAccuracy(outputs, labels, k=5):
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1))
    return correct.any(dim=1).float().mean().item()
```

## 5. Per-Class Accuracy

每个类别单独的准确率，对于发现类别特定问题非常有用：

```python
def perClassAccuracy(outputs, labels, numClasses):
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

## 6. 各指标的使用场景

| 指标 | 适用场景 | 不适用场景 |
| --- | --- | --- |
| Accuracy | 类别均衡 | 类别严重不均衡 |
| Precision | 误报代价高（垃圾邮件过滤） | 漏报代价高时不够用 |
| Recall | 漏报代价高（疾病检测） | 需要结合 Precision |
| F1 | 需要平衡 P 和 R | 不同场景 P 和 R 权重不同 |
| Top-K | 类别极多（ImageNet） | 类别少时意义不大 |
| Confusion Matrix | 分析具体混淆模式 | 类别极多时难以阅读 |

当训练准确率远高于测试准确率时，需排查 [[NeuralNetwork/Tips/Troubleshooting/Overfitting|过拟合]] 问题。

## 7. 完整的评估流程

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    allPreds, allLabels = [], []
    totalLoss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        totalLoss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        allPreds.extend(predicted.cpu().tolist())
        allLabels.extend(labels.cpu().tolist())

    accuracy = 100.0 * correct / total
    avgLoss = totalLoss / total

    # 计算完整指标
    from sklearn.metrics import classification_report
    report = classification_report(allLabels, allPreds, digits=4)

    return {
        'loss': avgLoss,
        'accuracy': accuracy,
        'predictions': allPreds,
        'labels': allLabels,
        'report': report
    }
```

评估指标的选择取决于任务场景，不同指标从不同角度衡量模型表现。
