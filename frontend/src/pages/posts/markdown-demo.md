---
layout: ../../layouts/MarkdownLayout.astro
title: Markdown 语法展示
date: 2025-12-17
desc: 这是一个展示 Markdown 各种语法的测试文档，包含代码块、数学公式、表格等。
tags: ["Demo", "Markdown"]
---

## 文本样式

**加粗文本**，*斜体文本*，~~删除线~~，`行内代码`。

## 列表

### 无序列表
- 项目 1
- 项目 2
  - 子项目 A
  - 子项目 B

### 有序列表
1. 第一步
2. 第二步
3. 第三步

## 代码块

```python
def hello_world():
    print("Hello, World!")
    return True
```

```javascript
const greeting = "Hello Astro";
console.log(greeting);
```

## 数学公式 (KaTeX)

行内公式：$E = mc^2$

块级公式：

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

## 表格

| 姓名 | 年龄 | 职业 |
|------|------|------|
| 张三 | 25 | 开发者 |
| 李四 | 30 | 设计师 |

## 引用

> 这是一个引用块。
> 
> 可以包含多行。

## 链接与图片

[访问 Astro 官网](https://astro.build)

![示例图片](https://via.placeholder.com/600x300?text=Markdown+Image)
