---
title: Astro 静态博客搭建完全指南
date: 2026-01-26
category: 技术
tags: [Astro, 博客, 前端, SSG]
description: 手把手教你用 Astro 搭建一个高性能的静态博客，从零到部署
image: https://img.yumeko.site/file/wife/早坂爱.jpg
status: public
---

# Astro 静态博客搭建完全指南

Astro 是一个现代化的静态网站生成器，以其出色的性能和开发体验著称。本文将带你从零开始搭建一个功能完整的静态博客。

## 为什么选择 Astro？

### 性能优势

- **零 JavaScript**：默认不向客户端发送 JS，页面加载飞快
- **部分水合**：只在需要交互的组件加载 JS
- **构建时渲染**：HTML 在构建时生成，访问速度极快

### 开发体验

- **组件岛屿**：可以混用 React、Vue、Svelte 等框架
- **内置 Markdown**：原生支持 Markdown 和 MDX
- **Content Collections**：类型安全的内容管理

## 快速开始

### 1. 创建项目

```bash
npm create astro@latest my-blog
cd my-blog
npm install
```

### 2. 项目结构

```
my-blog/
├── public/          # 静态资源
├── src/
│   ├── components/  # 组件
│   ├── content/     # 内容集合
│   │   └── blog/    # 博客文章
│   ├── layouts/     # 布局
│   ├── pages/       # 页面
│   └── styles/      # 样式
├── astro.config.mjs # Astro 配置
└── package.json
```

### 3. 配置 Content Collections

创建 `src/content/config.ts`：

```typescript
import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    category: z.string().optional(),
    tags: z.array(z.string()).optional(),
    description: z.string().optional(),
    image: z.string().optional(),
    status: z.enum(['public', 'draft', 'private']).default('public'),
  }),
});

export const collections = { blog };
```

### 4. 创建博客文章

在 `src/content/blog/` 创建 Markdown 文件：

```markdown
---
title: 我的第一篇文章
date: 2026-01-26
category: 随笔
tags: [生活, 思考]
description: 这是我的第一篇博客文章
---

# 你好，世界！

这是我的第一篇文章内容...
```

### 5. 创建文章列表页

`src/pages/articles.astro`：

```astro
---
import { getCollection } from 'astro:content';
import Layout from '../layouts/Layout.astro';

const posts = await getCollection('blog', ({ data }) => {
  return data.status === 'public';
});

// 按日期排序
const sortedPosts = posts.sort((a, b) => 
  new Date(b.data.date).getTime() - new Date(a.data.date).getTime()
);
---

<Layout title="文章列表">
  <h1>文章列表</h1>
  <ul>
    {sortedPosts.map(post => (
      <li>
        <a href={`/posts/${post.slug}`}>
          <h2>{post.data.title}</h2>
          <p>{post.data.description}</p>
          <time>{post.data.date.toLocaleDateString()}</time>
        </a>
      </li>
    ))}
  </ul>
</Layout>
```

### 6. 创建文章详情页

`src/pages/posts/[...slug].astro`：

```astro
---
import { getCollection } from 'astro:content';
import Layout from '../../layouts/Layout.astro';

export async function getStaticPaths() {
  const posts = await getCollection('blog');
  return posts.map(post => ({
    params: { slug: post.slug },
    props: { post },
  }));
}

const { post } = Astro.props;
const { Content } = await post.render();
---

<Layout title={post.data.title}>
  <article>
    <h1>{post.data.title}</h1>
    <time>{post.data.date.toLocaleDateString()}</time>
    <Content />
  </article>
</Layout>
```

## 进阶配置

### 代码高亮

Astro 内置 Shiki 代码高亮，在 `astro.config.mjs` 配置：

```javascript
export default defineConfig({
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
```

### 数学公式支持

安装 KaTeX：

```bash
npm install remark-math rehype-katex
```

配置：

```javascript
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
```

### RSS 订阅

```bash
npm install @astrojs/rss
```

创建 `src/pages/rss.xml.js`：

```javascript
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = await getCollection('blog');
  return rss({
    title: 'My Blog',
    description: 'A blog about tech and life',
    site: context.site,
    items: posts.map(post => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description,
      link: `/posts/${post.slug}/`,
    })),
  });
}
```

## 部署

### 构建静态文件

```bash
npm run build
```

生成的文件在 `dist/` 目录。

### 部署到 Cloudflare Pages

1. 连接 GitHub 仓库
2. 构建命令：`npm run build`
3. 输出目录：`dist`

### 部署到 EdgeOne

使用 GitHub Actions 自动部署，配置腾讯云密钥即可。

## 总结

Astro 是搭建静态博客的绝佳选择：

- ✅ 性能极佳，几乎零 JS
- ✅ 开发体验好，支持多种框架
- ✅ 内置 Markdown，Content Collections 类型安全
- ✅ 部署简单，支持各种平台

快来试试吧！

## 参考资料

- [Astro 官方文档](https://docs.astro.build/)
- [Content Collections](https://docs.astro.build/en/guides/content-collections/)
- [Astro 主题市场](https://astro.build/themes/)