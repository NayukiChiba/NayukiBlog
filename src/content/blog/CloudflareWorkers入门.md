---
title: Cloudflare Workers 入门指南
date: 2026-01-25
category: 技术
tags:
  - Cloudflare
  - Serverless
  - JavaScript
description: 从零开始学习 Cloudflare Workers，打造属于自己的边缘计算应用
image: https://img.yumeko.site/file/wife/早坂爱.jpg
status: public
---

# Cloudflare Workers 入门指南

Cloudflare Workers 是一个强大的 Serverless 平台，让你可以在全球边缘节点运行 JavaScript 代码。本文将带你从零开始搭建第一个 Worker。

## 什么是 Cloudflare Workers？

Workers 运行在 Cloudflare 的全球 CDN 网络上，这意味着你的代码会在离用户最近的节点执行，响应速度极快。

### 优势

- **零冷启动**：不像传统 Serverless，Workers 几乎没有冷启动延迟
- **全球部署**：代码自动部署到 200+ 个边缘节点
- **免费额度高**：每天 10 万次请求免费
- **简单易用**：写 JavaScript 就行，不需要管理服务器

## 快速开始

### 1. 安装 Wrangler CLI

```bash
npm install -g wrangler
```

### 2. 登录 Cloudflare

```bash
wrangler login
```

### 3. 创建项目

```bash
wrangler init my-worker
cd my-worker
```

### 4. 编写代码

```javascript
// src/index.js
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // 简单的路由
    if (url.pathname === '/api/hello') {
      return new Response(JSON.stringify({ message: 'Hello, World!' }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new Response('Welcome to my Worker!');
  }
};
```

### 5. 本地测试

```bash
wrangler dev
```

### 6. 部署

```bash
wrangler deploy
```

## 实战：搭建 OAuth 代理

下面是一个 GitHub OAuth 代理的示例：

```javascript
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // 重定向到 GitHub 授权
    if (url.pathname === '/auth') {
      const authUrl = `https://github.com/login/oauth/authorize?client_id=${env.CLIENT_ID}&scope=repo`;
      return Response.redirect(authUrl, 302);
    }
    
    // 处理回调
    if (url.pathname === '/callback') {
      const code = url.searchParams.get('code');
      
      // 用 code 换取 access_token
      const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          client_id: env.CLIENT_ID,
          client_secret: env.CLIENT_SECRET,
          code
        })
      });
      
      const { access_token } = await tokenRes.json();
      
      // 返回 token 或重定向
      return new Response(JSON.stringify({ access_token }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new Response('Not Found', { status: 404 });
  }
};
```

## 配置环境变量

在 `wrangler.toml` 中配置：

```toml
name = "my-oauth-worker"
main = "src/index.js"

[vars]
CLIENT_ID = "your_client_id"

# 敏感信息用 secrets
# wrangler secret put CLIENT_SECRET
```

## 总结

Cloudflare Workers 是构建现代 Web 应用的利器，特别适合：

- API 代理和转发
- OAuth 认证服务
- 边缘计算和缓存
- A/B 测试和流量分发

免费额度足够个人项目使用，赶紧试试吧！

## 参考链接

- [Cloudflare Workers 官方文档](https://developers.cloudflare.com/workers/)
- [Wrangler CLI 文档](https://developers.cloudflare.com/workers/wrangler/)