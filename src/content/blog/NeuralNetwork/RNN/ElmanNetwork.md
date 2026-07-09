---
title: Elman Network：循环神经网络的开端
date: 2026-05-21
category: 神经网络/RNN
tags:
  - 经典架构
  - 深度学习
description: 1990年 Jeff Elman 提出的 Simple Recurrent Network，奠定了所有 RNN 变体的基础范式。完整推导其数学原理、手算前向传播、从零 PyTorch 实现、BPTT 梯度分析及根本局限。
image: https://img.yumeko.site/file/blog/cover/1780581742225.webp
status: published
---

## 1. 历史背景

### 1.1 1990 年的神经网络研究

1990 年，反向传播（Rumelhart et al., 1986）刚刚证明了多层神经网络的训练可行性。但当时的研究几乎全部聚焦于**静态模式识别**——前馈网络处理固定尺寸的输入。

Jeff Elman 在认知科学领域提出了一个不同的问题：**人类如何处理随时间展开的信息？** 我们听一句话时，不会等到句子结束才开始理解——大脑在逐词处理的同时，维持着对前面内容的"工作记忆"。这个认知科学的洞察催生了 Elman Network。

### 1.2 核心动机

Elman 在 1990 年的论文《Finding Structure in Time》中提出了 RNN 的基本假设：

> "时间中展开的信息包含结构——这种结构可以通过一个在时间维度上具有内部状态的网络来发现。"

如果网络能保留之前时间步的信息，它就能发现序列中的模式。

---

## 2. 原始结构与数学定义

### 2.1 四组单元

![ElmanNetwork.png](https://img.yumeko.site/file/blog/articles/1780581391297.webp)

Elman Network 包含四组单元：

| 单元 | 角色 | 现代对应 |
|------|------|------|
| Input Layer | 接收外部输入 $x_t \in \mathbb{R}^{d_x}$ | 同现代的输入层 |
| Hidden Layer | 计算当前时刻的表示 $h_t \in \mathbb{R}^{d_h}$ | 同现代的隐藏层 |
| Context Units | 存储上一时刻的隐藏状态 $h_{t-1}$ | 现代的隐藏状态循环传递 |
| Output Layer | 产生输出 $y_t \in \mathbb{R}^{d_y}$ | 同现代的输出层 |

### 2.2 Context Units：显式的"复制-延迟"存储

Context Units 是 Elman Network 最关键的创新。在 Elman 的原始设计中，Context Units 是一个**物理上独立**的存储单元组——它不参与任何计算，只做一件事：保存上一时刻 Hidden Layer 输出的精确副本，并在下一时刻将其作为额外输入回传。

引入符号 $c_t$ 表示 Context Units 在时间步 $t$ 存储的内容。其工作机制分两步：

**第一步——复制（Copy）**：Hidden Layer 计算得到 $h_t$ 后，将其逐元素复制到 Context Units。这个复制是精确的、不可训练的——没有权重、没有非线性、没有变换：

$$
c_t = h_t
$$

**第二步——延迟回传（Delayed Feedback）**：在下一时间步 $t+1$，Context Units 中存储的 $c_t$（即 $h_t$）与新的外部输入 $x_{t+1}$ 一同送入 Hidden Layer。此时 Context Units 充当了"上一时刻的隐藏状态"的角色：

$$
h_{t+1} = \sigma(W_{hh} \cdot c_t + W_{xh} \cdot x_{t+1} + b_h)
$$

两步组合后，Context Units 实现了 $c_t = h_{t-1}$ 的效果——$c_t$ 始终是上一时刻隐藏状态的精确副本。在 Elman 的架构图中，Context Units 与 Hidden Layer 之间有明确的分隔线——它们是不同的物理单元，数据流向是单向的：Hidden Layer 写入 Context Units，Context Units 再被 Hidden Layer 读取。

这个"复制-延迟"机制是神经网络首次拥有**显式的时间记忆**。与现代 RNN 中将 $h_{t-1}$ 直接作为输入不同，Elman 的设计将"记忆存储"从"隐藏计算"中物理分离出来——Context Units 是专用的记忆槽，Hidden Layer 是专用的计算单元。后来的 LSTM 继承了这一思想，将记忆槽（$C_t$）和计算输出（$h_t$）显式分离。

### 2.3 数学形式（引入 Context Units 符号 $c_t$）

用现代符号重写 Elman Network，**保留 Context Units 的显式符号 $c_t$**。在时间步 $t$，整个计算分为三个阶段：

**阶段一——读取 Context Units（上一时刻的记忆）**：Context Units 存储着上一时刻 Hidden Layer 输出的精确副本。在 $t=1$ 时 Context Units 内容为零（初始没有记忆）：

$$
c_0 = \mathbf{0}, \quad c_t = h_{t-1} \;\; (t \geq 1)
$$

**阶段二——Hidden Layer 计算**：将外部输入 $x_t \in \mathbb{R}^{d_x}$ 与 Context Units 中的 $c_t \in \mathbb{R}^{d_h}$ 组合，经权重矩阵和 Sigmoid 激活产生新的隐藏状态：

$$
h_t = \sigma(W_{hh} \cdot c_t + W_{xh} \cdot x_t + b_h)
$$

这里 $W_{hh} \cdot c_t$ 是 Context Units 到 Hidden Layer 的连接——$c_t$ 存的是 $h_{t-1}$，所以此项功能上等价于现代的 $W_{hh} \cdot h_{t-1}$。

**阶段三——Context Units 更新与输出计算**：Hidden Layer 产出的 $h_t$ 同时走两条路：一条完整复制到 Context Units（覆盖旧值），另一条经 $W_{hy}$ 解码为输出：

$$
c_{t+1} = h_t \quad \text{（复制到 Context Units，供下一步使用）}
$$
$$
y_t = W_{hy} \cdot h_t + b_y
$$

注意时间索引：$c_{t+1}$ 存储的是 $h_t$，即在 $t+1$ 步使用的 Context Units 内容等于第 $t$ 步的隐藏状态。这造成了 $c_t = h_{t-1}$ 的"延迟一时刻"效果。

**参数矩阵的形状**：

- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$：Input Layer 到 Hidden Layer 的权重
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$：Context Units 到 Hidden Layer 的权重（循环连接）
- $W_{hy} \in \mathbb{R}^{d_y \times d_h}$：Hidden Layer 到 Output Layer 的权重
- $b_h \in \mathbb{R}^{d_h}$：Hidden Layer 偏置
- $b_y \in \mathbb{R}^{d_y}$：Output Layer 偏置

激活函数 $\sigma$ 在原始论文中是 Sigmoid：$\sigma(x) = \frac{1}{1 + e^{-x}}$。

**为什么 Elman 要显式区分 $c_t$ 和 $h_{t-1}$？** 在现代 RNN 中，$h_{t-1}$ 直接作为输入传给下一步——Context Units 被"折叠"进了循环连接，不再作为独立的物理单元。但在 Elman 的 1990 年认知科学视角中，Context Units 的显式存在对应着认知心理学中的"工作记忆"概念——大脑有一个专门的记忆缓冲区，计算的输出被写入缓冲区，下一步计算再从缓冲区读取。这种"计算-记忆"的物理分离是 Elman Network 的哲学基础，也是后来 LSTM 中 $C_t$（记忆细胞）与 $h_t$（输出）分离的思想源头。

### 2.4 完整前向传播流程（含 Context Units 状态追踪）

以序列长度 $T=4$ 为例，每一步同时追踪 $c_t$（Context Units 内容）和 $h_t$（Hidden Layer 输出），完整展示"复制-延迟"机制的实际运作。

**初始状态（$t=0$）**：
$$
c_0 = \mathbf{0} \quad \text{（Context Units 初始为空）}
$$
网络在开始时没有任何历史记忆。

**$t=1$（首步）**——Context Units 为零，Hidden Layer 纯由输入驱动：
$$
h_1 = \sigma(W_{hh} \cdot \underbrace{c_0}_{\mathbf{0}} + W_{xh} \cdot x_1 + b_h) = \sigma(W_{xh} \cdot x_1 + b_h)
$$
$$
y_1 = W_{hy} \cdot h_1 + b_y
$$
$$
c_1 = h_1 \quad \text{（将 } h_1 \text{ 复制到 Context Units，覆盖 } c_0 \text{）}
$$

由于 $c_0 = 0$，第一步的 $W_{hh} \cdot c_0$ 项为零，$h_1$ 完全由当前输入 $x_1$ 决定。步末，$h_1$ 被复制到 Context Units，网络拥有了第一份"记忆"。

**$t=2$（循环连接首次生效）**——Context Units 中的 $c_1$（即 $h_1$）参与计算：
$$
h_2 = \sigma(W_{hh} \cdot \underbrace{c_1}_{=h_1} + W_{xh} \cdot x_2 + b_h)
$$
$$
y_2 = W_{hy} \cdot h_2 + b_y
$$
$$
c_2 = h_2 \quad \text{（将 } h_2 \text{ 复制到 Context Units，覆盖 } c_1 \text{）}
$$

$h_2$ 同时融合了当前输入 $x_2$（经 $W_{xh}$）和 Context Units 中的 $c_1 = h_1$（经 $W_{hh}$）。$h_2$ 包含 $x_1$ 和 $x_2$ 的压缩信息。步末 $h_2$ 覆盖 Context Units。

**$t=3$**——Context Units 中的 $c_2$（即 $h_2$）包含了 $x_1, x_2$ 的混合信息：
$$
h_3 = \sigma(W_{hh} \cdot \underbrace{c_2}_{=h_2} + W_{xh} \cdot x_3 + b_h)
$$
$$
c_3 = h_3 \quad \text{（将 } h_3 \text{ 复制到 Context Units，覆盖 } c_2 \text{）}
$$

$h_3$ 包含 $x_1$（经两次循环和两次 Context Units 复制）、$x_2$（经一次循环）和 $x_3$（直接输入）的混合信息。

**$t=4=T$（末步）**——最后一步，$h_4$ 不再被复制（序列结束）：
$$
h_4 = \sigma(W_{hh} \cdot \underbrace{c_3}_{=h_3} + W_{xh} \cdot x_4 + b_h)
$$
$$
y_4 = W_{hy} \cdot h_4 + b_y
$$

**Context Units 状态演变总览**：

| 时刻 | Context Units $c_t$ | Hidden Layer $h_t$ | $W_{hh} \cdot c_t$ 中的有效信息 |
|:---:|------|------|------|
| $t=0$ | $c_0 = \mathbf{0}$ | — | 无 |
| $t=1$ | $c_1 = h_1$ | 仅含 $x_1$ | 零（$c_0 = 0$） |
| $t=2$ | $c_2 = h_2$ | 含 $x_1$（压缩一次）+ $x_2$ | $x_1$ 的信息 |
| $t=3$ | $c_3 = h_3$ | 含 $x_1$（压缩两次）+ $x_2$（压缩一次）+ $x_3$ | $x_1, x_2$ 的混合 |
| $t=4$ | $c_4 = h_4$ | 含 $x_1$（压缩三次）+ $x_2$（压缩两次）+ $x_3$（压缩一次）+ $x_4$ | $x_1, x_2, x_3$ 的混合 |

**信息衰减规律**：$x_1$ 的信息要到达 $h_4$，需经过以下路径：$x_1$ 写入 $h_1$，复制到 $c_1$，经 $W_{hh}$ 和 $\sigma$ 融入 $h_2$，复制到 $c_2$，再经 $W_{hh}$ 和 $\sigma$ 融入 $h_3$，复制到 $c_3$，最后经 $W_{hh}$ 和 $\sigma$ 融入 $h_4$。共 3 次 $\sigma$ 非线性压缩和 3 次 $W_{hh}$ 矩阵乘法——每步既压缩又旋转——这就是梯度消失的结构性根源。

---

## 3. 从零实现 Elman Network

### 3.1 手动实现 ElmanRNNCell（含 Context Units 显式变量）

以下代码精确实现了 2.3 节的三个阶段。关键设计——引入显式变量 `c` 表示 Context Units 的内容，展示"复制-延迟"机制在代码层的落地。

```python
import torch
import torch.nn as nn


class ElmanRNNCell(nn.Module):
    """Elman Network 单步单元（含 Context Units 显式变量 c）。

    三个阶段（对应 2.3 节）:
      阶段一: 读取 c_t (= h_{t-1})——从 Context Units 获取上一时刻记忆
      阶段二: h_t = sigmoid(W_hh @ c_t + W_xh @ x_t + b_h)
      阶段三: c_{t+1} = h_t（复制到 Context Units）
              y_t = W_hy @ h_t + b_y
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # W_xh: (hidden_dim, input_dim) —— Input Layer 到 Hidden Layer
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        # W_hh: (hidden_dim, hidden_dim) —— Context Units 到 Hidden Layer（循环连接）
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # b_h:  (hidden_dim,) —— Hidden Layer 偏置
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        # W_hy: (output_dim, hidden_dim) —— Hidden Layer 到 Output Layer
        self.W_hy = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x_t, c_t):
        """
        Args:
            x_t: 当前输入, (batch, input_dim)
            c_t: Context Units 内容 (= h_{t-1}), (batch, hidden_dim)
        Returns:
            y_t:    当前输出, (batch, output_dim)
            h_t:    当前隐藏状态, (batch, hidden_dim)
            c_next: 更新后的 Context Units 内容 (= h_t), (batch, hidden_dim)
        """
        # 阶段二: Hidden Layer 计算
        # W_hh(c_t)   —— Context Units 的贡献
        # W_xh(x_t)   —— 外部输入的贡献
        # 加 b_h, 过 sigmoid
        # (batch, input_dim)  -> Linear -> (batch, hidden_dim)
        # (batch, hidden_dim) -> Linear -> (batch, hidden_dim)
        h_t = torch.sigmoid(
            self.W_xh(x_t) + self.W_hh(c_t) + self.b_h
        )

        # 阶段三: 输出 + Context Units 更新
        # y_t = W_hy(h_t) + b_y
        # (batch, hidden_dim) -> Linear -> (batch, output_dim)
        y_t = self.W_hy(h_t)

        # c_{t+1} = h_t —— 将当前隐藏状态"复制"到 Context Units
        # 这是精确的逐元素赋值，不经过任何变换
        c_next = h_t

        return y_t, h_t, c_next
```

**参数命名与 Elman 原始结构的对应**：

| 代码变量 | Elman 原始术语 | 连接的物理单元 |
|------|------|------|
| `self.W_xh` | Input-to-Hidden 权重 | Input Layer 到 Hidden Layer |
| `self.W_hh` | Context-to-Hidden 权重 | Context Units 到 Hidden Layer |
| `self.W_hy` | Hidden-to-Output 权重 | Hidden Layer 到 Output Layer |
| `c_t` | Context Units 内容 | Context Units 中存储的向量 |
| `c_next` | 更新后的 Context Units | $h_t$ 的精确副本 |

### 3.2 手动展开：逐时间步循环（Context Units 状态追踪）

`ElmanRNNCell` 只处理单步。要用它处理完整序列，需要按 2.4 节的流程手动在时间维度上展开，并显式追踪 Context Units 的状态演变。

```python
def elman_forward(cell, input_sequence, c_0=None):
    """
    用 ElmanRNNCell 手动展开整个序列，显式追踪 Context Units。

    Args:
        cell:           ElmanRNNCell 实例
        input_sequence: (batch, seq_len, input_dim)
        c_0:            初始 Context Units 内容, None 则用零向量
    Returns:
        outputs:  (batch, seq_len, output_dim)  所有时间步的输出
        h_final:  (batch, hidden_dim)           最后一步的隐藏状态
        c_final:  (batch, hidden_dim)           最终的 Context Units 内容 (= h_final)
        c_list:   [T 个 (batch, hidden_dim)]    每一步开始时 c_t 的快照
    """
    batch_size, seq_len, _ = input_sequence.shape

    if c_0 is None:
        # Context Units 初始为空 —— 对应 c_0 = 0
        c = torch.zeros(
            batch_size, cell.W_hh.out_features,
            device=input_sequence.device
        )
    else:
        c = c_0

    outputs = []
    c_history = []  # 记录每一步的 Context Units 内容，用于分析

    for t in range(seq_len):
        # 第 t 步开始时的 Context Units 快照
        c_history.append(c.clone())

        # 取出第 t 步的输入
        x_t = input_sequence[:, t, :]  # (batch, input_dim)

        # 执行一步 Elman RNN（三个阶段）
        # y_t:     (batch, output_dim)
        # h_t:     (batch, hidden_dim)
        # c_next:  (batch, hidden_dim) —— 即 h_t 的副本
        y_t, h_t, c = cell(x_t, c)

        outputs.append(y_t)

    # 堆叠所有时间步的输出
    outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)
    return outputs, h_t, c, c_history
```

**关键观察**——这段代码展示了 Elman Network 的三个核心特性：

1. **参数时间共享**：同一个 `cell`（含同一套 $W_{xh}, W_{hh}, W_{hy}, b_h$）在每一步被重复调用，参数量与序列长度无关。
2. **Context Units 的复制机制**：`c = h_t`（通过 `c_next` 传递）是精确的逐元素赋值——Context Units 不做任何计算，仅存储副本。
3. **时间步间的唯一联系**：`c` 是跨时间步传递的唯一状态。如果切断 $c$ 的传递（即每步重置 $c = 0$），网络退化为独立的前馈网络，所有时间步信息完全隔离。

### 3.3 完整分类模型

```python
class ElmanClassifier(nn.Module):
    """基于 Elman Network 的序列分类器（Many-to-One 模式）。

    流程:
      输入序列 x -> ElmanRNNCell 逐步展开
        -> 取最后一步隐藏状态 h_T（或 Context Units c_T）
        -> 全连接分类
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.cell = ElmanRNNCell(input_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # 逐步前向传播，显式追踪 Context Units
        # outputs:  (batch, seq_len, hidden_dim)
        # h_final:  (batch, hidden_dim) —— 最后一步的 Hidden Layer 输出
        # c_final:  (batch, hidden_dim) —— 最后一步更新后的 Context Units (= h_final)
        # c_history: 每步开始时的 Context Units 快照
        outputs, h_final, c_final, c_history = elman_forward(self.cell, x)

        # Many-to-One: 取最后一步的隐藏状态做分类
        # h_final = c_final —— 两者相等，取哪个都一样
        logits = self.classifier(h_final)  # (batch, num_classes)
        return logits
```

### 3.4 维度变化追踪

以具体参数追踪前向传播：`batch=8`，`seq_len=20`，`input_dim=128`，`hidden_dim=256`，`output_dim=10`。

循环内部每一步的形状变化：

- `x_t`：`(8, 128)`，当前步输入
- `h_prev`：`(8, 256)`，上一步隐藏状态
- `W_xh(x_t)`：`(8, 256)`，输入编码
- `W_hh(h_prev)`：`(8, 256)`，循环连接的贡献
- `W_xh(x_t) + W_hh(h_prev) + b_h`：`(8, 256)`，逐元素加和
- `h_t = sigmoid(...)`：`(8, 256)`，新隐藏状态
- `y_t = W_hy(h_t)`：`(8, 10)`，当前步输出

最终 `outputs = stack([y_1, ..., y_20])` 形状 `(8, 20, 10)`，`h_final` 形状 `(8, 256)`。

---

## 4. 与 Jordan Network 的对比

![ElmanVSJordan.png](https://img.yumeko.site/file/blog/articles/1780581395302.webp)

### 4.1 Jordan Network (1986)

Michael Jordan 几乎与 Elman 同时期提出了另一种循环结构。两者的核心区别在于**什么信息被循环**：

| | Elman Network | Jordan Network |
|------|------|------|
| 循环的内容 | **隐藏状态** $h_{t-1}$ | **输出** $y_{t-1}$ |
| 循环连接 | Hidden 经 Context 回传 Hidden | Output 经 State 回传 Hidden |
| 信息类型 | 内部表示 | 外部输出 |
| 优势 | 更丰富的内部状态 | 更直接的监督信号 |

Jordan Network 的状态更新为：

$$
h_t = \sigma(W_{hh} \cdot y_{t-1} + W_{xh} \cdot x_t + b_h)
$$

注意这里 $W_{hh} \cdot y_{t-1}$ 中用的是**上一时刻的输出** $y_{t-1}$，而非 Elman 的 $h_{t-1}$。

### 4.2 为什么 Elman 的设计胜出？

- **隐藏状态比输出信息更丰富**：$h_t \in \mathbb{R}^{d_h}$ 可以编码任意内部表示（256 维或更多），$y_t$ 受限于输出空间（如 10 维分类）。循环隐藏状态的带宽远大于循环输出。
- **灵活性更高**：隐藏状态可以编码"对当前任务有用但不需要输出的信息"——例如在解析句子时维护一个不直接输出但有助于后续理解的语法状态。
- **解耦原则**：隐藏状态 = "记忆"，输出 = "行动"。两者应当解耦——记忆的内容不需要等于当前输出的内容。这个解耦被后来的 LSTM（$C_t$ 与 $h_t$ 分离）继承和强化。

---

## 5. BPTT 对 Elman Network 的梯度分析

### 5.1 展开形式与参数共享

Elman Network 沿时间展开后等价于一个深度为 $T$ 的前馈网络，同层参数在所有时间步共享。损失 $L$ 对 $W_{hh}$ 的总梯度是所有时间步贡献之和：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

对于单个时间步 $t$，$L_t$ 通过 $h_t$ 依赖 $W_{hh}$，而 $h_t$ 又通过 $W_{hh}$ 依赖 $h_{t-1}$，递归至 $h_0$：

$$
\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t} \cdot \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

### 5.2 梯度连乘项的具体形式

对于 Elman Network 使用 Sigmoid 激活函数的情形，中间导数项为：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}\big(\sigma'(W_{hh} \cdot h_{j-1} + W_{xh} \cdot x_j + b_h)\big) \cdot W_{hh}
$$

Sigmoid 的导数 $\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$ 的最大值为 $0.25$（在 $x=0$ 处），在远离 0 的区域迅速趋近于 0。

**结合 $W_{hh}$ 的范数后**，每一步梯度的衰减因子近似为：

$$
\left|\left| \frac{\partial h_j}{\partial h_{j-1}} \right|\right| \approx \max(\sigma') \cdot ||W_{hh}|| \leq 0.25 \cdot ||W_{hh}||
$$

当 $||W_{hh}|| < 4$ 时（大多数训练阶段满足），每步衰减因子小于 1，连乘导致指数衰减。

### 5.3 具体数值计算

以典型的小规模设置为例：$d_h=2$，$d_x=2$。设某批数据中 $W_{hh} = \begin{pmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{pmatrix}$，其谱范数约 $0.46$。

取序列中某个时间步的 $\sigma'$ 值构成的对角矩阵。假设 $\sigma'$ 的值为 $[0.20, 0.15]$（典型的饱和区偏导值），则：

$$
\text{diag}(\sigma') = \begin{pmatrix} 0.20 & 0 \\ 0 & 0.15 \end{pmatrix}
$$

单步梯度传递矩阵：

$$
\frac{\partial h_j}{\partial h_{j-1}} = \begin{pmatrix} 0.20 & 0 \\ 0 & 0.15 \end{pmatrix} \cdot \begin{pmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{pmatrix}
= \begin{pmatrix} 0.060 & -0.020 \\ 0.030 & 0.060 \end{pmatrix}
$$

矩阵元素的量级从 $0.1-0.4$ 降至 $0.02-0.06$，单步衰减约 5-10 倍。

$T$ 步连乘后：$(0.06)^{T} \approx$：

- $T=10$：$(0.06)^{10} \approx 6.0 \times 10^{-13}$
- $T=20$：$(0.06)^{20} \approx 3.7 \times 10^{-25}$
- $T=50$：$(0.06)^{50} \approx 8.1 \times 10^{-62}$

float32 的精度约 $10^{-7}$ 以下不可靠。这意味着 $T > 12$ 时，最早时间步的梯度就已经在 float32 下完全为零——这就是 Elman Network 的约 10 步有效学习窗口的数学根源。

---

## 6. 表征能力 vs 学习能力

![GradLimit.png](https://img.yumeko.site/file/blog/articles/1780581404785.webp)

### 6.1 理论上的图灵完备性

Siegelmann 和 Sontag 在 1995 年证明：只要有足够的隐藏单元和合理的权重，Elman Network **能够**表示任意序列到序列的映射——它是图灵完备的。这意味着理论上不存在 Elman Network"算不出来"的序列函数。

### 6.2 实践中的学习鸿沟

"能表示"不等于"能学会"。梯度消失意味着即使最优解存在于参数空间中，基于梯度的优化方法（SGD）也无法收敛到该解。在最需要梯度的长距离依赖位置上，梯度信号已经衰减为零——优化器在这些参数维度上得不到任何有效的更新方向。

这个表征能力与学习能力之间的鸿沟，直接推动了 Hochreiter 和 Schmidhuber 在 1997 年提出 LSTM。LSTM 的核心创新——CEC（Constant Error Carousel）——正是为了解决"梯度信号如何在时间轴上无损传递"这一 Elman Network 无法解决的问题。

---

## 7. 历史定位

![RNNTimeline.png](https://img.yumeko.site/file/blog/articles/1780736505354.png)

Elman Network 定义了 RNN 的基本架构模式，影响了后续所有变体：

1. **循环隐藏状态**：Elman 的 Context Units 思想定义了"隐藏状态在时间轴上传递"的范式，至今仍是所有 RNN 的核心。
2. **三层结构**：Input、Hidden、Output 的架构模式被 LSTM、GRU、BiRNN 完整继承。
3. **参数时间共享**：同一套权重在所有时间步复用——这一设计使 RNN 可以处理任意长度的序列。
4. **BPTT 训练**：虽然原始实现不完整，但建立了"沿时间轴反向传播梯度"的方向。

后续所有的 RNN 变体都保留了这套基本框架，只是在内部机制上做了增强：LSTM 增加了门控和细胞状态来缓解梯度消失，GRU 精简了门控机制，BiRNN 增加了反向信息流。而 Elman Network 作为这一切的起点，其简洁性使其至今仍是理解 RNN 原理的最佳入门模型。

---

> 现代 Vanilla RNN 的详细数学：[[NeuralNetwork/RNN/VanillaRNN|循环层详解]]
> LSTM 如何解决 Elman 的局限：[[NeuralNetwork/RNN/LSTM|LSTM 里程碑]]
> 回到主文档：[[NeuralNetwork/Overview/RNNOverview|RNN 详解主文档]]
