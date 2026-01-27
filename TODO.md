# 📋 静态博客待办清单

## 🚀 部署上线（必需）

### 1. 推送代码到 GitHub
```bash
git push origin static-blog --force
```
> ⚠️ 使用 `--force` 因为修改了提交历史

### 2. 配置 GitHub Secrets
在 GitHub 仓库 → Settings → Secrets and variables → Actions 添加：

| Secret 名称 | 说明 |
|------------|------|
| `TENCENT_SECRET_ID` | 腾讯云 API 密钥 ID |
| `TENCENT_SECRET_KEY` | 腾讯云 API 密钥 Key |
| `EDGEONE_ZONE_ID` | EdgeOne 站点 ID |
| `EDGEONE_PROJECT_NAME` | EdgeOne 项目名称 |

### 3. 验证自动部署
- 检查 GitHub Actions 是否成功运行
- 访问网站确认部署成功

---

## 🎨 管理后台（可选）

如需使用 `/admin` 可视化管理，需配置 OAuth：

**方式一：使用 Netlify（推荐）**
- 在 Netlify 部署网站，自动获得 OAuth 支持

**方式二：本地编辑（最简单）**
- 直接编辑 `src/content/blog/*.md` 和 `src/data/*.json`
- 使用 Git 提交推送

---

## 🔧 已知小问题

- [ ] `/favicon.ico` 404（需要添加 favicon.ico 文件）
- [ ] 部分文章 LaTeX 公式有中文字符警告（不影响显示）

---

## 📝 日常更新内容

```bash
# 编辑文章或数据
# 然后提交推送
git add .
git commit -m "feat: 添加新文章"
git push origin static-blog
```

---

## ✅ 已完成

- [x] 静态博客架构重构
- [x] 数据迁移（文章、项目、书籍、日记等）
- [x] 页面组件开发
- [x] GitHub Actions 自动部署配置
- [x] 本地构建测试通过（16 个页面）
- [x] 删除旧的动态博客代码