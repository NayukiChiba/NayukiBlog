---
name: write-blog
description: 编写 NayukiBlog 技术博客文章。触发词：写文章、写博客、write blog。覆盖数学理论、机器学习、笔试真题等类型的文章写作规范，包括 frontmatter、LaTeX、代码块、wiki 链接、fold 折叠块等约定。
---

# 博客写作规范

本 skill 用于在 NayukiBlog 项目中编写技术博客文章。所有文章存放在 `src/content/blog/` 下。

## 目录结构

```
src/content/blog/
├── Math/                              # 数学理论文章
│   ├── ProbabilityInequalities.md     # 概率不等式
│   ├── LawOfLargeNumbers.md           # 大数定律
│   ├── MultivariateStatistics.md      # 多元统计
│   └── CubicSplineInterpolation.md    # 数值分析
├── MachineLearning/                   # 传统机器学习
│   ├── Basic/                         # 基础工具（numpy, pandas, sklearn...）
│   ├── classification/                # 分类算法
│   ├── regression/                    # 回归算法
│   ├── clustering/                    # 聚类算法
│   ├── dimensionality/                # 降维
│   ├── ensemble/                      # 集成学习
│   ├── probabilistic/                 # 概率模型
│   └── FeatureEngineering/            # 特征工程
├── NeuralNetwork/                     # 神经网络 / 深度学习
│   ├── Training/                      # 训练相关（激活函数、损失、优化器...）
│   ├── CNN/                           # 卷积神经网络
│   ├── RNN/                           # 循环神经网络 / 注意力
│   ├── Transformer/                   # Transformer 架构与优化
│   ├── LLM/                           # 大语言模型（涌现、RLHF...）
│   ├── Diffusion/                     # 扩散模型
│   └── Troubleshooting/               # 训练问题排查
├── Internship/                        # 笔试面试真题
│   └── HUAWEI/                        # 按公司名
├── Engineering/                       # 工程工具
├── Programming/                       # 编程语言
├── Projects/                          # 项目
└── Life/                              # 生活类
```

**新建文章时的目录选择**：
- 纯数学推导 → `Math/`
- 传统机器学习算法 → `MachineLearning/<子领域>/`
- 深度学习/训练相关 → `NeuralNetwork/Training/`
- Transformer / 注意力机制 → `NeuralNetwork/Transformer/`
- 大语言模型 / RLHF → `NeuralNetwork/LLM/`
- 扩散模型 → `NeuralNetwork/Diffusion/`
- 笔试面试真题 → `Internship/<公司名>/`
- 其他技术文章 → 与已有目录保持一致

---

## Frontmatter 规范

每篇文章开头必须包含完整的 YAML frontmatter：

```yaml
---
title: 文章标题（中文，简洁明了）
date: YYYY-MM-DD          # 写作日期
category: 分类/子分类       # 如 数学、神经网络/训练、实习/华为
tags:
  - 标签1
  - 标签2
  - 标签3
description: 一句话描述文章内容，用于 SEO 和列表展示
image: https://img.yumeko.site/file/blog/ArticleName.png  # 封面图，可选
status: draft             # draft（草稿）或 published（已发布）
---
```

**注意事项**：
- `date` 使用当天日期
- `category` 与目录路径一致（`Math/` 对应 `数学`）
- `tags` 使用小分类标签，如 `概率论`、`深度学习`、`基础`
- `status` 新建时用 `draft`，发布时改为 `published`
- 已有文章如需修改，保持原有 `date` 不变

---

## 写作风格

### 语言

- **全部使用中文**，包括注释、解释、表格标题
- 技术术语保持英文原文（如 Accuracy、Precision、ROC），首次出现时附中文翻译
- 直接、技术化、简洁，避免废话

### 结构模式

技术理论文章推荐以下结构：

1. **问题的起点**：为什么需要这个知识？从实际场景引入
2. **定义与公式**：用 LaTeX 给出精确的数学定义，用 `\boxed{}` 高亮核心公式
3. **证明/推导**：逐步推导，关键步骤附直觉解释
4. **数值示例**：用小规模的具体数据走一遍完整流程
5. **代码实现**：Python + NumPy/SciPy，中文 docstring
6. **总结对比表**：将所有要点汇总到一张表格中

### 好的例子

```markdown
## 3. 切比雪夫不等式：引入方差

### 3.1 定理与推导

设随机变量 $Y$ 具有有限期望 $\mu = E[Y]$ 和有限方差 $\sigma^2 = \operatorname{Var}(Y)$。则：

$$
\boxed{P(|Y - \mu| \ge \varepsilon) \le \frac{\sigma^2}{\varepsilon^2}}
$$

**证明思路**：切比雪夫 = 马尔可夫 + 平方变换。

令 $X = (Y - \mu)^2 \ge 0$，则 $E[X] = \sigma^2$。对 $X$ 应用马尔可夫不等式...
```

---

## LaTeX 规范

### 基本要求

- **大量使用 LaTeX**——所有数学符号、公式、变量名都放入 `$...$`（行内）或 `$$...$$`（行间）
- 核心公式用 `\boxed{}` 包裹
- 使用 `\displaystyle` 仅在必要时，行间公式默认即为 displaystyle
- 矩阵用 `\begin{bmatrix} ... \end{bmatrix}`
- 对齐公式用 `\begin{aligned} ... \end{aligned}`

### 常用符号速查

| 用途 | LaTeX |
|:--|:--|
| 期望 | `E[X]` 或 `\mathbb{E}[X]` |
| 方差 | `\operatorname{Var}(X)` |
| 协方差 | `\operatorname{Cov}(X, Y)` |
| 概率 | `P(X \ge a)` |
| 指示函数 | `\mathbf{1}_{\{X \ge a\}}` |
| 依概率收敛 | `\xrightarrow{P}` |
| 几乎必然收敛 | `\xrightarrow{a.s.}` |
| 依分布收敛 | `\xrightarrow{D}` |
| 独立 | `\perp\!\!\!\perp` |
| 正态分布 | `\mathcal{N}(\mu, \sigma^2)` |
| 多元正态 | `\mathcal{N}_d(\boldsymbol{\mu}, \Sigma)` |
| 转置 | `^\top`（不用 `^T`） |
| 迹 | `\operatorname{tr}` |
| 对角矩阵 | `\operatorname{diag}` |
| 盒子公式 | `\boxed{...}` |

### 禁止事项

- **不要用 ASCII 艺术图**（`│├└┌` 等字符画的树形图、流程图）。用文字描述逻辑关系
- **不要用 Mermaid 图**
- 不要在代码块中使用非 ASCII 的箭头符号（代码注释中 `→` 可以保留）

---

## 代码块规范

### 语言标记

所有代码块必须指定语言：

```python
# Python 代码
import numpy as np
```

```bash
# Shell 命令
python script.py
```

### Python 代码规范

- 使用中文 docstring 和注释
- 函数用驼峰命名法（如 `computeAccuracy`、`sampleCovariance`）
- 变量用驼峰命名法（如 `allPreds`、`numClasses`）
- 类型提示放在 docstring 中
- 代码块只放代码，不放讲解文字

**正确示例**：

```python
def sampleCovariance(X: np.ndarray, bias: bool = False) -> np.ndarray:
    """
    计算样本协方差矩阵

    Args:
        X: 形状 (n, d)，每行一个样本
        bias: True 则除以 n（MLE），False 则除以 n-1（无偏）
    Returns:
        形状 (d, d) 的协方差矩阵
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    divisor = n if bias else n - 1
    return (Xc.T @ Xc) / divisor
```

**错误示例**（代码块中放了讲解）：

```python
# 这里我们先计算协方差矩阵
# 协方差矩阵是一个 d×d 的矩阵
# 对角元是方差，非对角元是协方差
cov = sampleCovariance(data)  # ← 这些讲解应该在代码块外面的正文里
```

---

## Wiki 链接规范

使用 Obsidian 风格的 `[[...]]` 链接关联其他文章：

```markdown
详见 [[Math/LawOfLargeNumbers|大数定律详解]]。
前置知识请参阅 [[Math/ProbabilityInequalities|概率不等式完全指南]]。
```

- 路径从 `src/content/blog/` 开始（不含此前缀）
- 文件名**不含** `.md` 扩展名
- 用 `|` 分隔路径和显示名称
- 每篇文章末尾应有"相关文章"小节，列出双链

---

## Fold 折叠块

用于隐藏答案、详细推导等"点击展开"的内容：

```markdown
:::fold[答案]

正确答案：A

解析：...

:::
```

**注意**：
- 使用 `:::fold[标题]` 和 `:::` 包裹
- 折叠块内的 `###` 标题不会出现在右侧 TOC 大纲中（已通过 JS 过滤 `<details>` 内部标题）
- 适合放答案解析、冗长推导、可选阅读材料

---

## Obsidian Callout 语法

项目支持 Obsidian 风格的 Callout 高亮块，用于在文章中突出提示、警告、注意事项等内容。渲染为带图标的彩色卡片。

### 语法

```markdown
> [!TYPE] 标题（可选）
> 内容行1
> 内容行2
```

### 支持的类型

| 类型 | 用途 | 图标颜色 |
|:--|:--|:--|
| `[!NOTE]` / `[!INFO]` | 信息、备注 | 蓝色 ℹ️ |
| `[!TIP]` | 技巧、建议 | 绿色 💡 |
| `[!WARNING]` | 警告、注意 | 黄色 ⚠️ |
| `[!DANGER]` / `[!ERROR]` | 危险、错误 | 红色 ⛔ |
| `[!SUCCESS]` / `[!CHECK]` / `[!DONE]` | 成功、完成 | 绿色 ✅ |
| `[!QUESTION]` / `[!FAQ]` / `[!HELP]` | 问题、FAQ | 蓝色 ❓ |
| `[!ABSTRACT]` / `[!SUMMARY]` / `[!TLDR]` | 摘要、总结 | 灰色 📋 |
| `[!BUG]` | Bug 报告 | 红色 🐛 |
| `[!EXAMPLE]` | 示例 | 紫色 📖 |
| `[!QUOTE]` / `[!CITE]` | 引用 | 灰色 💬 |
| `[!TODO]` | 待办 | 蓝色 📌 |

### 示例

```markdown
> [!NOTE] 关键前提
> 以下推导假设 $\{X_n\}$ 是独立同分布序列。

> [!WARNING] 注意
> 切比雪夫不等式仅给出上界，实际概率可能远小于 $1/k^2$。

> [!TIP] 记忆技巧
> 马尔可夫 → 切比雪夫 → 弱大数定律，这是一条"平方化"的推导链。
```

---

## 嵌入语法

### GitHub 仓库卡片

展示 GitHub 仓库的星级、Fork 数等信息，渲染为可点击的卡片。

```markdown
::github[repo=owner/repo]
```

示例：`::github[repo=torvalds/linux]`

### 视频嵌入

**YouTube**：

```markdown
::youtube[id=视频ID]
```

视频 ID 即 URL 中 `v=` 后面的部分。示例：`::youtube[id=dQw4w9WgXcQ]`

**Bilibili**：

```markdown
::bilibili[id=BV号]
```

示例：`::bilibili[id=BV1GJ411x7h7]`

### CodePen 嵌入

```markdown
::codepen[url=https://codepen.io/用户名/pen/代码ID]
```

---

## 图片与 AI 生图提示词

### 只加讲解图和封面

文章中的图片**严格限于以下两种**：

1. **封面 Banner**：文章唯一一张宽幅封面，放在 frontmatter 之后、正文之前
2. **讲解图**：直接服务于数学推导或概念解释的图，帮助读者理解公式和逻辑

**严禁添加**：
- 历史时间线图（如"XX 发展史"）
- 比喻/类比图（如"就像一座桥"）
- 人物肖像或场景图
- 与数学内容无直接解释关系的装饰图

每张图必须有明确的教学目的："没有这张图，读者理解某个公式/概念会更困难。"

### 图片占位符格式

```markdown
![图N: 图片中文描述](https://img.yumeko.site/file/blog/ArticleName/ImageName.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的科学示意图，[中文详细描述画面内容]。
> [构图要求]。[配色要求]。[标注要求]。
> ```
```

### 图片 URL 规范

- 封面图：`https://img.yumeko.site/file/blog/<ArticleName>.png`
- 内文讲解图：`https://img.yumeko.site/file/blog/<ArticleName>/<ImageDescription>.png`

### AI 生图提示词规范

- **使用中文**，详细描述画面内容
- 格式：先描述内容，再说明构图，然后配色，最后标注
- 风格统一要求：
  - 白色背景、学术/教科书风格
  - 细坐标轴线、清晰图例
  - 中英文标签均可，以可读性为准
  - 专业数据可视化风格

### 提示词模板

**封面 Banner**：

```
一张宽幅 Banner（宽高比 2.35:1），用于技术博客封面。
设计概念：[主题核心概念的可视化表达]。
左/中/右分区：[分别描述各区域内容]。
配色：深蓝到暖金渐变，现代数据科学美学风格。
简洁无衬线标签，淡色网格背景。顶部留白供标题叠加。
```

**讲解图（概念/对比/推导流程）**：

```
一张简洁的[科学示意图/对比图/教学流程图]，[描述要解释的数学内容]。
[构图说明：横轴/纵轴/分区/箭头关系]。
[关键标注：公式用 LaTeX 风格，数据点/曲线/区域的标注方式]。
白色背景，细坐标轴线，柔和蓝白配色，教科书插图风格。
```

### 文章中图片的摆放原则

- 每个核心概念（定理、重要公式、推导链）配一张讲解图
- 图片放在对应文字的**前面**（先看图，后看详解）
- 推导步骤超过 3 步时，配一张流程图
- 对比类内容（两种方法、两个概念）配对比图
- 封面 Banner 放在 `## 1.` 标题之前

---

## 文章模板

### 数学理论文章模板

```markdown
---
title: 文章标题
date: YYYY-MM-DD
category: 数学
tags:
  - 标签1
  - 标签2
description: 一句话描述
image: https://img.yumeko.site/file/blog/ArticleName.png
status: draft
---

> **前置阅读**：本文假定读者熟悉... 建议先阅读 [[Math/RelatedArticle|相关文章]]。

![图0: 文章主题 Banner](https://img.yumeko.site/file/blog/ArticleName.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张宽幅 Banner（宽高比 2.35:1），用于技术博客封面。
> 设计概念：[描述]。配色：[描述]。风格：[描述]。
> 顶部留白供标题叠加。
> ```

## 1. 问题的起点

从实际场景或直觉出发，引出问题。

![图1: 问题引入示意图](https://img.yumeko.site/file/blog/ArticleName/ProblemIntro.png)

> **🖼️ AI 生图提示词：**
>
> ```
> 一张简洁的科学示意图，[描述]。
> 白色背景，学术风格，中文标签。
> ```

## 2. 核心定义

用 LaTeX 给出精确的数学定义。

$$
\boxed{核心公式}
$$

![图2: 核心定义可视化](https://img.yumeko.site/file/blog/ArticleName/CoreDefinition.png)

> **🖼️ AI 生图提示词：**
>
> ```
> [描述]
> ```

## 3. 推导与性质

逐步推导，关键步骤附直觉解释。

> [!TIP] 记忆技巧
> [一句话总结推导链的关键 trick]

## 4. 数值示例

用小规模具体数据走一遍完整流程。

## 5. 代码实现

Python 代码，中文注释。

## 6. 总结

| 要点 | 说明 |
|:--|:--|
| ... | ... |

---

> **相关文章**：
> - [[Math/RelatedArticle1|文章1]]
> - [[Math/RelatedArticle2|文章2]]
```

### 笔试真题文章模板

```markdown
---
title: 公司名YYYY-MM-DD 机试真题（方向）
date: YYYY-MM-DD
category: 实习/公司名
tags:
  - 笔试
  - 公司名
  - 方向
description: 一句话描述
status: published
---

# 公司名 YYYY-MM-DD 机试真题（方向）

来源：[题库链接](https://...)

---

## 第1题-选择题（题目编号）共 N 道，单选/多选

### 1. 题目内容
- A. 选项A
- B. 选项B
- C. 选项C
- D. 选项D

:::fold[答案]

正确答案：X

链接: [[Related/Article|相关文章]]

解析：详细解释...
:::

### 2. 下一题...

---

## 第M题-编程题名称（题目编号）| 难度 | 时间限制

### 题目描述

...

:::fold[答案]

## 解题思路
...

## 参考代码
```python
...
```
:::

---

> 完整题目见 [题库链接](https://...)
```

---

## 检查清单

写完文章后，逐项检查：

- [ ] Frontmatter 完整（title/date/category/tags/description/status）
- [ ] 所有数学符号使用 LaTeX（`$...$` 或 `$$...$$`）
- [ ] 核心公式使用 `\boxed{}`
- [ ] 无 ASCII 艺术图（`│├└┌` 等）
- [ ] 无 Mermaid 图
- [ ] 代码块有语言标记且不包含讲解文字
- [ ] 变量/函数名使用驼峰命名法
- [ ] Wiki 链接格式正确（无 `.md` 扩展名）
- [ ] 文章末尾有"相关文章"双链
- [ ] Fold 块正确配对（`:::fold[...]` 和 `:::`）
- [ ] 图片 URL 使用 `https://img.yumeko.site/file/blog/` 前缀
- [ ] 插图仅限于封面 Banner + 讲解图，无历史时间线/比喻/装饰图
- [ ] AI 生图提示词使用中文
- [ ] Callout 语法使用正确（`> [!TYPE]` 格式）
- [ ] 嵌入语法使用 directives（`::github`、`::youtube`、`::bilibili`、`::codepen`）
- [ ] 注释和 docstring 使用中文
- [ ] 无 emoji 出现在代码或公式中
