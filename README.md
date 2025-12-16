# NayukiBlog

一个基于 Astro + FastAPI + SQLite 的现代化博客系统。

## 🚀 技术栈

- **前端**: [Astro](https://astro.build/) - 静态站点生成与动态组件
- **后端**: [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Python Web 框架
- **数据库**: SQLite - 轻量级关系型数据库
- **样式**: CSS Modules / Global CSS

## 📂 项目结构

```
blog-project
├── backend/          # FastAPI 后端
│   ├── app/          # 应用核心代码
│   ├── db/           # 数据库配置与迁移
│   └── requirements.txt
├── frontend/         # Astro 前端
│   ├── src/
│   │   ├── components/ # UI 组件
│   │   ├── layouts/    # 页面布局
│   │   └── pages/      # 路由页面
│   └── astro.config.mjs
└── README.md
```

## 🛠️ 快速开始

### 前端 (Frontend)

```bash
cd frontend
npm install
npm run dev
```

### 后端 (Backend)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## ✨ 特性

- ⚡️ 极速的页面加载 (Astro)
- 🔒 类型安全的开发体验 (TypeScript + Pydantic)
- 📝 Markdown 博客撰写支持
- 🎨 响应式设计

## 📄 许可证

MIT License
