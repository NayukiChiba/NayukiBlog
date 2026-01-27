# Nayuki Blog - 静态博客

> 基于 Astro + Decap CMS 的现代化静态博客系统

## ✨ 特性

- 🚀 **极速访问** - 纯静态网站，CDN 加速
- 📝 **Markdown 写作** - 支持 GFM 和 LaTeX 数学公式
- 🎨 **可视化管理** - Decap CMS 提供友好的管理界面
- 🔄 **自动部署** - Push 到 GitHub 自动构建部署
- 📦 **版本控制** - 所有内容都在 Git 中，可随时回滚
- 🌐 **EdgeOne CDN** - 腾讯云 EdgeOne 全球加速

## 🏗️ 技术栈

- **前端框架**: [Astro](https://astro.build/)
- **内容管理**: [Decap CMS](https://decapcms.org/)
- **部署平台**: [腾讯云 EdgeOne](https://cloud.tencent.com/product/eo)
- **CI/CD**: GitHub Actions
- **样式**: CSS + Astro Components

## 📁 项目结构

```
NayukiBlog/
├── public/
│   ├── admin/              # Decap CMS 管理后台
│   └── images/             # 图片资源
├── src/
│   ├── content/
│   │   └── blog/           # 📝 文章 Markdown
│   ├── data/               # 📊 JSON 数据
│   ├── pages/              # 页面
│   ├── components/         # 组件
│   ├── layouts/            # 布局
│   └── lib/                # 工具函数
├── .github/
│   └── workflows/          # GitHub Actions
└── dist/                   # 构建输出
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/NayukiChiba/NayukiBlog.git
cd NayukiBlog
git checkout static-blog
```

### 2. 安装依赖

```bash
npm install
```

### 3. 启动开发服务器

```bash
npm run dev
```

访问 `http://localhost:4321` 查看网站。

### 4. 构建生产版本

```bash
npm run build
```

详细说明请查看 [QUICKSTART.md](./QUICKSTART.md)

## 📝 内容管理

### 方式 1：本地编辑（推荐）

直接编辑 `src/content/blog/*.md` 和 `src/data/*.json` 文件，然后 Git commit & push。

### 方式 2：管理后台

访问 `你的域名.com/admin` 使用可视化界面管理内容（需要配置 OAuth）。

## 🌐 部署

### 自动部署

推送到 `static-blog` 分支会自动触发 GitHub Actions 构建并部署到 EdgeOne：

```bash
git add .
git commit -m "Update content"
git push origin static-blog
```

### 手动部署

```bash
npm run build
# 将 dist/ 目录上传到你的服务器
```

## 📚 文档

- [快速开始指南](./QUICKSTART.md)
- [迁移指南](./MIGRATION_GUIDE.md)
- [Astro 文档](https://docs.astro.build/)
- [Decap CMS 文档](https://decapcms.org/docs/)

## 🛠️ 开发命令

```bash
npm run dev          # 启动开发服务器
npm run build        # 构建静态网站
npm run preview      # 预览构建结果
npm run clean        # 清理构建输出
npm run check        # 检查配置和类型
```

## 📄 许可证

MIT License

## 👤 作者

Nayuki Chiba

- GitHub: [@NayukiChiba](https://github.com/NayukiChiba)

## 🙏 致谢

- [Astro](https://astro.build/) - 现代化的静态网站生成器
- [Decap CMS](https://decapcms.org/) - 开源的内容管理系统
- [腾讯云 EdgeOne](https://cloud.tencent.com/product/eo) - 全球加速服务

---

⭐ 如果这个项目对你有帮助，欢迎 Star！
