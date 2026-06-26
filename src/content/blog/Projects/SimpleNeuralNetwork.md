---
title: NumPy 手写简单神经网络项目
date: 2026-06-26
category: 项目
tags:
  - NumPy
  - 神经网络
  - 反向传播
  - 教学项目
description: 一个用 NumPy 从零实现神经网络基础组件的项目，覆盖数据、层、激活函数、损失、优化器、训练器和模型保存。
image: https://img.yumeko.site/file/blog/cover/1782481187971_SimpleNeuralNetwork.webp
status: published
---

# NumPy 手写简单神经网络项目

> **前置阅读**：建议先阅读 [[NeuralNetwork/Training/ActivationFunctions|激活函数]]、[[NeuralNetwork/Training/LossFunctions|损失函数]] 与 [[NeuralNetwork/Training/Optimizers|优化器]]。

::github[repo=NayukiChiba/Simple-Neural-Network]

## 1. 项目定位

`Simple-Neural-Network` 是一个基于 `NumPy` 从零实现的简单神经网络项目。

它不依赖 PyTorch 或 TensorFlow，而是手写：

| 组件 | 实现内容 |
|:--|:--|
| 数据 | XOR、Spiral、Sine 三类示例数据 |
| 网络层 | Linear、ReLU、Sigmoid、Tanh |
| 损失函数 | CrossEntropyLoss、MSELoss |
| 模型容器 | SequentialModel |
| 优化器 | SGDOptimizer |
| 训练器 | mini-batch 训练、评估、日志 |
| 持久化 | `.npz` 参数保存与加载 |

项目目标是把神经网络训练主流程完整跑通：先准备数据，再完成前向计算和损失计算，
随后通过反向传播得到梯度，最后由优化器更新参数。

---

## 2. 项目结构

```text
Simple-Neural-Network/
├── config.py
├── main.py
├── datasets/
│   ├── xor/
│   ├── spiral/
│   └── sine/
├── src/
│   └── nn/
│       ├── data/
│       │   ├── dataGenerator.py
│       │   └── dataLoader.py
│       ├── layers/
│       │   ├── baseLayer.py
│       │   ├── linearLayer.py
│       │   └── activationLayer.py
│       ├── losses/
│       │   ├── crossEntropyLoss.py
│       │   └── mseLoss.py
│       ├── models/
│       │   └── sequentialModel.py
│       ├── optimizers/
│       │   └── sgdOptimizer.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── metrics.py
│       └── persistence/
│           └── checkpointIO.py
├── tests/
└── docs/
```

目录很适合教学：每个概念基本都有独立文件，抽象层次浅，阅读路径明确。

---

## 3. 三个任务

| 任务 | 类型 | 用途 |
|:--|:--|:--|
| `xor` | 二分类 | 验证网络学习非线性关系的能力 |
| `spiral` | 三分类 | 验证多分类和复杂决策边界 |
| `sine` | 回归 | 验证连续函数拟合能力 |

XOR 不能被线性模型解决。它需要至少一层非线性隐藏层：

$$
y = f_2(W_2 f_1(W_1x + b_1) + b_2)
$$

Spiral 数据用于测试模型能否学习弯曲的非线性边界。Sine 数据用于检查 MSE 回归训练是否正常。

---

## 4. 前向传播

线性层计算：

$$
\boxed{
Z = XW + b
}
$$

激活函数提供非线性：

$$
A = \sigma(Z)
$$

多层网络可以写成：

$$
h_1 = \phi_1(XW_1 + b_1)
$$

$$
h_2 = \phi_2(h_1W_2 + b_2)
$$

$$
\hat{y} = h_2W_3 + b_3
$$

`SequentialModel` 的作用就是按顺序调用每一层，把前一层输出作为后一层输入。

---

## 5. 反向传播

训练时需要计算损失对参数的梯度：

$$
\frac{\partial \mathcal{L}}{\partial W},
\quad
\frac{\partial \mathcal{L}}{\partial b}
$$

以线性层 $Z=XW+b$ 为例：

$$
\frac{\partial \mathcal{L}}{\partial W}
= X^\top \frac{\partial \mathcal{L}}{\partial Z}
$$

$$
\frac{\partial \mathcal{L}}{\partial b}
= \sum_{i=1}^{N}
\frac{\partial \mathcal{L}}{\partial Z_i}
$$

$$
\frac{\partial \mathcal{L}}{\partial X}
= \frac{\partial \mathcal{L}}{\partial Z}W^\top
$$

SGD 更新参数：

$$
\boxed{
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
}
$$

其中 $\eta$ 是学习率。

---

## 6. CLI 使用

安装依赖：

```bash
pip install -r requirements.txt
pre-commit install
```

运行 XOR：

```bash
python main.py --task xor
```

运行 Spiral：

```bash
python main.py --task spiral
```

运行 Sine：

```bash
python main.py --task sine
```

运行测试：

```bash
pytest tests -q
```

---

## 7. 默认训练效果

当前默认配置下，本地验证效果大致为：

| 任务 | 指标 | 表现 |
|:--|:--|:--|
| `xor` | Accuracy | 可达到 `1.0` |
| `spiral` | Accuracy | 可达到约 `0.9978` |
| `sine` | MSE | 可稳定下降 |

这说明项目已经跑通基础神经网络的完整闭环。

---

## 8. 可扩展方向

| 方向 | 说明 |
|:--|:--|
| Momentum | 在 SGD 基础上加入速度项 |
| Adam | 加入一阶和二阶矩估计 |
| Dropout | 增加正则化层 |
| BatchNorm | 增加归一化层 |
| 学习率调度 | 支持训练过程动态调整学习率 |
| 可视化 | 输出 loss 曲线和决策边界 |

---

## 9. 总结

这个项目的价值在于“拆开神经网络黑盒”：

| 问题 | 项目中的答案 |
|:--|:--|
| 模型如何组织 | `SequentialModel` |
| 层如何计算 | `LinearLayer` 与激活层 |
| loss 如何反传 | `CrossEntropyLoss` 与 `MSELoss` |
| 参数如何更新 | `SGDOptimizer` |
| 训练如何编排 | `Trainer` |
| 参数如何保存 | `CheckpointIO` |

它适合作为深度学习框架之前的底层理解项目：先把梯度、矩阵乘法和参数更新写明白，再去用 PyTorch 会更踏实。

---

> **相关文章**：
> - [[NeuralNetwork/Training/ActivationFunctions|激活函数]]
> - [[NeuralNetwork/Training/LossFunctions|损失函数]]
> - [[NeuralNetwork/Training/Optimizers|优化器]]
> - [[Projects/CharRNN|Char-RNN 字符级文本生成项目]]
