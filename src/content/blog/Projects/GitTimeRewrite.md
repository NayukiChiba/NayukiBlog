---
title: git时间重写工具
date: 2026-03-13
category: Projects
tags:
  - Rust
  - git
description: 使用Rust+Vue3+Tauri写一个轻量修改git历史的工具
image: https://img.yumeko.site/file/blog/git.jpeg
status: published
---
# GitTimeRewrite

GitTimeRewrite 是一个基于 Tauri + Vue3 + Rust 的桌面工具，用于可视化编辑 Git 提交信息与提交时间线。

## 功能概览

- 可视化浏览 Git 提交时间线（支持分支线轨道展示）
- 手动编辑单条提交：message、作者信息、作者时间、提交时间
- 批量改写提交时间：按日期区间与每日时间窗口平滑分布
- 设置 origin 远程地址（新增或更新）
- 一键执行 force push 到 origin（带确认流程）
- 操作过程实时日志、加载与执行状态弹窗

## 技术栈

- 前端：Vue 3 + TypeScript + Vite
- 桌面层：Tauri v2
- 后端：Rust

## 项目结构

- frontend：前端 UI、组件、类型与 API 调用
- backend：Tauri 配置、Rust 命令实现、打包配置
- .github/workflows：CI 与自动发布流程

## 本地开发

### 1. 安装依赖

```bash
npm --prefix frontend install
```

### 2. 启动开发模式

建议在仓库根目录运行：

```bash
npx tauri dev --config backend/tauri.conf.json
```

## 构建 exe

在仓库根目录执行：

```bash
npx tauri build --config backend/tauri.conf.json --no-bundle
```

生成文件：

- backend/target/release/gitTimeRewrite.exe

## 测试与检查

```bash
npm --prefix frontend run type-check
npm --prefix frontend run test:unit -- --run
cargo test --manifest-path backend/Cargo.toml
```

## 自动发布

仓库已提供 GitHub Actions 工作流：

- .github/workflows/release-exe.yml

触发方式：

- 手动触发 workflow_dispatch
- 推送 tag（格式 v\*，如 v1.0.1）

工作流会在 Windows 环境构建 exe，并在 tag 发布时自动上传到 Release。

## 版本发布示例

```bash
git add .
git commit -m "docs: 更新 README"
git tag -a v1.0.1 -m "release: v1.0.1"
git push
git push origin v1.0.1
```
