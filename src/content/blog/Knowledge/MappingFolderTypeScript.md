---
title: 在vitepress中映射图片文件夹
date: 2026-05-03
category: 技术
tags:
  - 设置
  - 网站
description: 在vitepress中，让markdown文件中可以使用根目录中的文件，不局限于docs文件夹中
image: https://img.yumeko.site/file/blog/cover/1780581804847.webp
status: published
---
## VitePress 引用项目外部目录图片的方法

### 核心原理

VitePress 把 markdown 中的 `![alt](/path/img.png)` 转成 Vue SFC 的 `import` 语句：

```js
import _imports_0 from '/path/img.png'
```

所以解决思路必须在 **Vite 模块解析层**，而不是 HTTP 中间件层。

### 配置方法

在 `docs/.vitepress/config.mts` 中：

```typescript
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  vite: {
    // 1. 放行上级目录，避免 Vite 出于安全拒绝访问
    server: {
      fs: {
	      // 这里写上级的文件夹路径
        allow: ['../..'],
      },
    },
    // 2. resolveId 把虚拟路径映射到实际文件系统路径
    plugins: [
      {
        name: 'resolve-external-images',
        resolveId(id) {
          // /visualizations/xxx.png → ../../visualizations/xxx.png（实际路径）
          if (id.startsWith('/{根目录的文件夹名字}/')) {
            return path.resolve(__dirname, '../..', id.slice(1))
          }
        },
      },
    ],
  },
})
```

### 三步走

| 步骤                | 做什么                          | 为什么                                |
| ----------------- | ---------------------------- | ---------------------------------- |
| `server.fs.allow` | 白名单放行项目根目录                   | Vite 默认禁止访问项目目录外的文件                |
| `resolveId`       | 虚拟路径 `/{外部文件夹的名字}/` → 实际文件路径 | VitePress 把图片当模块 import，必须从模块解析层介入 |
| markdown 路径       | 统一用 `/{外部文件夹的名字}/xxx.png`    | 绝对路径，不受 md 文件层级影响                  |
|                   |                              |                                    |

### markdown 中引用

```md
![描述](/{外部文件夹的名字}/xxx.png)
```

VitePress 会自动拼接 `base` 前缀（如 `/{project-name}/`），dev 和 build 均生效。构建时图片经 Vite 处理写入 dist，无需手动复制。