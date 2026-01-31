# Nayuki Blog

> 基于 Astro 的现代化静态博客系统，配套独立的 Vue 管理后台

## 特性

- **极速访问** - 纯静态网站，CDN 加速
- **Markdown 写作** - 支持 GFM 和 LaTeX 数学公式
- **独立管理后台** - Vue 3 构建的现代化管理界面
- **自动部署** - Push 到 GitHub 自动构建部署
- **版本控制** - 所有内容都在 Git 中，可随时回滚
- **EdgeOne CDN** - 腾讯云 EdgeOne 全球加速

## 技术栈

- **博客前端**: [Astro](https://astro.build/)
- **管理后台**: [Vue 3](https://vuejs.org/) + [Vite](https://vitejs.dev/) (独立仓库 [NayukiBlog-Admin](https://github.com/NayukiChiba/NayukiBlog-Admin))
- **部署平台**: [腾讯云 EdgeOne](https://cloud.tencent.com/product/eo)
- **CI/CD**: GitHub Actions
- **样式**: CSS + Tailwind CSS

## 项目结构

```
NayukiBlog/                 # 博客前端
├── public/
│   └── images/             # 图片资源
├── src/
│   ├── content/
│   │   └── blog/           # 文章 Markdown
│   ├── data/               # JSON 数据
│   ├── pages/              # 页面
│   ├── components/         # 组件
│   ├── layouts/            # 布局
│   └── lib/                # 工具函数
├── .github/
│   └── workflows/          # GitHub Actions
└── dist/                   # 构建输出

NayukiBlog-Admin/           # 管理后台 (独立仓库)
├── src/
│   ├── views/              # 页面视图
│   ├── components/         # Vue 组件
│   ├── api/                # API 调用
│   └── stores/             # Pinia 状态管理
└── workers/                # Cloudflare Workers (OAuth)
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/NayukiChiba/NayukiBlog.git
cd NayukiBlog
```

### 2. 安装依赖

```bash
npm install
```

### 3. 配置环境变量

项目根目录的 `.env` 文件用于配置 Admin 管理面板跳转地址：

```env
# Admin 管理面板 URL
# 本地开发: http://localhost:5173
# 生产环境: https://admin.yourdomain.com
PUBLIC_ADMIN_URL=http://localhost:5173
```

### 4. 启动开发服务器

```bash
npm run dev
```

访问 `http://localhost:4321` 查看网站。

### 5. 本地完整开发（博客 + 管理后台）

如需同时开发博客和管理后台，需要启动两个项目：

```bash
# 终端 1: 启动博客 (端口 4321)
cd NayukiBlog
npm run dev

# 终端 2: 启动管理后台 (端口 5173)
cd NayukiBlog-Admin
npm run dev
```

然后在博客页面点击右下角浮动按钮，即可跳转到管理后台。

### 6. 构建生产版本

```bash
npm run build
```

## 内容管理

### 方式 1：本地编辑

直接编辑 `src/content/blog/*.md` 和 `src/data/*.json` 文件，然后 Git commit & push。

### 方式 2：管理后台

访问 `admin.yourdomain.com` 使用可视化界面管理内容。

管理后台是独立项目，详见 [NayukiBlog-Admin](https://github.com/NayukiChiba/NayukiBlog-Admin)。

## 部署

本项目采用**双站点部署**架构，博客和管理后台分别独立部署：

| 站点     | 域名                   | 仓库             |
| -------- | ---------------------- | ---------------- |
| 博客前端 | `blog.yourdomain.com`  | NayukiBlog       |
| 管理后台 | `admin.yourdomain.com` | NayukiBlog-Admin |

### EdgeOne 部署步骤

1. **创建博客站点**
   - 在 EdgeOne 创建新站点，关联 `NayukiBlog` 仓库
   - 绑定域名 `blog.yourdomain.com`
   - 设置环境变量 `PUBLIC_ADMIN_URL=https://admin.yourdomain.com`

2. **创建管理后台站点**
   - 在 EdgeOne 创建新站点，关联 `NayukiBlog-Admin` 仓库
   - 绑定域名 `admin.yourdomain.com`

3. **自动部署**
   - 推送到对应仓库会自动触发构建部署
   - 两个项目独立管理，互不影响

```bash
# 更新博客
cd NayukiBlog
git add . && git commit -m "Update blog" && git push

# 更新管理后台
cd NayukiBlog-Admin
git add . && git commit -m "Update admin" && git push
```

## 文档

- [Astro 文档](https://docs.astro.build/)
- [Vue 3 文档](https://vuejs.org/)
- [EdgeOne 文档](https://cloud.tencent.com/document/product/1552)

## 开发命令

```bash
npm run dev          # 启动开发服务器
npm run build        # 构建静态网站
npm run preview      # 预览构建结果
npm run clean        # 清理构建输出
npm run check        # 检查配置和类型
```

## 许可证

MIT License

## 作者

Nayuki Chiba

- GitHub: [@NayukiChiba](https://github.com/NayukiChiba)

## 致谢

- [Astro](https://astro.build/) - 现代化的静态网站生成器
- [Vue 3](https://vuejs.org/) - 渐进式 JavaScript 框架
- [腾讯云 EdgeOne](https://cloud.tencent.com/product/eo) - 全球加速服务

---

如果这个项目对你有帮助，欢迎 Star!
