---
title: NayukiBlog - 个人博客系统
description: 基于 Astro 的现代化静态博客，配套 Vue 3 可视化管理后台
date: 2025-01-30
category: Projects
tags:
  - Astro
  - 网站
image: https://img.yumeko.site/file/blog/NayukiBlog.png
status: published
link: https://github.com/NayukiChiba/NayukiBlog
---

# NayukiBlog

一个现代化的个人博客系统，采用**前后端分离**的双仓库架构。博客前端基于 Astro 构建，追求极致的访问速度；管理后台基于 Vue 3 开发，通过 GitHub API 直接管理博客内容，实现无服务器的内容管理体验。

## 项目架构

整个系统由两个独立的项目组成：

| 项目                 | 技术栈             | 用途                     |
| -------------------- | ------------------ | ------------------------ |
| **NayukiBlog**       | Astro              | 博客前端，静态网站生成   |
| **NayukiBlog-Admin** | Vue 3 + TypeScript | 管理后台，可视化内容管理 |

### 为什么选择这种架构？

传统的博客系统通常需要后端服务器和数据库来存储内容，这带来了服务器成本和维护负担。NayukiBlog 采用了一种更现代的方式：

- **内容即代码**：所有文章、日记等内容都以 Markdown/JSON 文件形式存储在 Git 仓库中
- **GitHub 作为 CMS**：通过 GitHub API 实现内容的增删改查，Git 提供版本控制
- **纯静态部署**：博客构建为纯静态 HTML，部署在 CDN 上，访问速度极快
- **零服务器成本**：无需维护服务器，只需 CDN 和 OAuth 代理

## 博客前端 (NayukiBlog)

博客前端使用 [Astro](https://astro.build/) 构建，这是一个专注于内容的现代化静态网站生成器。

### 核心特性

- **极速访问** - 纯静态 HTML，搭配 CDN 全球加速
- **Markdown 写作** - 支持 GFM 语法、LaTeX 数学公式、代码语法高亮
- **响应式设计** - 完美适配桌面端和移动端
- **自动部署** - 推送到 GitHub 后自动触发构建部署
- **版本控制** - 所有内容都在 Git 中，支持历史回溯和版本对比

### 内容模块

| 模块   | 说明                               |
| ------ | ---------------------------------- |
| 文章   | 技术博文、学习笔记，支持分类和标签 |
| 日记   | 日常记录，支持心情、天气标记和图片 |
| 书架   | 阅读记录与书籍管理                 |
| 项目   | 个人项目与作品集展示               |
| 待办   | 任务管理，分为短期、中期、长期目标 |
| 工具箱 | 常用工具和资源收藏                 |
| 图库   | 图片管理与展示                     |

### 技术栈

- **框架**: Astro 5
- **样式**: CSS + Tailwind CSS
- **Markdown**: MDX + remark/rehype 插件
- **数学公式**: KaTeX
- **代码高亮**: Expressive Code
- **部署**: 腾讯云 EdgeOne

## 管理后台 (NayukiBlog-Admin)

管理后台是一个独立的 Vue 3 单页应用，提供可视化的内容管理界面。

### 核心特性

- **GitHub OAuth 认证** - 安全的身份验证，支持用户白名单
- **Markdown 编辑器** - 实时预览，所见即所得
- **直接操作 GitHub** - 通过 GitHub API 读写仓库内容
- **无需后端服务器** - 所有逻辑在前端完成

### 认证流程

![Certification](https://img.yumeko.site/file/articles/NayukiBlog/Certification.png)

使用 Cloudflare Worker 作为 OAuth 代理是因为 GitHub OAuth 流程需要 Client Secret，而这个敏感信息不能暴露在前端代码中。Worker 安全地存储 Secret 并代理 Token 交换请求。

### 技术栈

- **框架**: Vue 3 + TypeScript
- **构建**: Vite
- **状态管理**: Pinia
- **路由**: Vue Router
- **样式**: Tailwind CSS
- **GitHub 集成**: Octokit
- **OAuth 代理**: Cloudflare Workers
- **部署**: 腾讯云 EdgeOne

## 部署架构
![Architecture](https://img.yumeko.site/file/articles/NayukiBlog/Architecture.png)
### 工作流程

1. **写作**: 在管理后台使用 Markdown 编辑器撰写内容
2. **保存**: 点击保存，通过 GitHub API 将内容提交到仓库
3. **触发构建**: GitHub Push 事件触发 EdgeOne 自动构建
4. **部署上线**: Astro 构建静态文件，自动部署到 CDN

整个流程完全自动化，从写作到发布只需要点击保存按钮。

## 本地开发

### 博客前端

```bash
git clone https://github.com/NayukiChiba/NayukiBlog.git
cd NayukiBlog
npm install
npm run dev
```

访问 `http://localhost:4321`

### 管理后台

管理后台需要配置 GitHub OAuth App 和 Cloudflare Worker，详细步骤请参考 [NayukiBlog-Admin](https://github.com/NayukiChiba/NayukiBlog-Admin) 仓库的 README。

```bash
git clone https://github.com/NayukiChiba/NayukiBlog-Admin.git
cd NayukiBlog-Admin
npm install
npm run dev
```

访问 `http://localhost:5173`

## 项目链接

- 博客前端: [NayukiBlog](https://github.com/NayukiChiba/NayukiBlog)
- 管理后台: [NayukiBlog-Admin](https://github.com/NayukiChiba/NayukiBlog-Admin)

## 未来计划

- 支持更多内容格式（视频、音频笔记）
- 评论系统集成
- 访问统计与数据分析
- 全文搜索功能
- RSS 订阅支持
