---
layout: ../../layouts/MarkdownLayout.astro
title: 关于 Astro 的使用心得
date: 2025-12-16
desc: 记录一下使用 Astro 框架重构个人博客的过程和体会。
tags: ["Astro", "Web"]
---

## 为什么选择 Astro？

Astro 是一个**以内容为中心**的 Web 框架。它的核心理念是：

1. **零 JS 默认**：默认情况下，Astro 发送给浏览器的 HTML 是纯静态的，没有 JavaScript 运行时开销。
2. **岛屿架构 (Islands Architecture)**：你可以在静态页面中嵌入动态组件（React, Vue, Svelte 等），只有这些组件会加载 JS。

## 目录结构

```
src/
  components/
  layouts/
  pages/
    index.astro
    posts/
      *.md
```

## 总结

Astro 非常适合用来构建博客、文档等内容型网站。速度快，开发体验好。
