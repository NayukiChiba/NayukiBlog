---
title: 用 1Panel 部署 RSSHub
date: 2026-06-13
category: 工程实践
tags:
  - 1Panel
  - RSS
  - 服务器
description: 从零开始，用 1Panel 面板一步步部署 RSSHub，包含 DNS 验证申请证书
image: https://img.yumeko.site/file/blog/cover/1781348760218_rsshub.webp
status: published
---

RSSHub 是一个开源 RSS 聚合器，能把各种网站内容转为 RSS 订阅源。

> **前置阅读**：本文假定你已安装 1Panel 并了解基础操作，建议先阅读 [[Engineering/WebsiteDeployment|网站部署指南]]。

# 部署

## 整体流程

部署顺序：**OpenResty → Redis → RSSHub → DNS 账户 → ACME 账户 → 申请证书 → 创建网站 → 验证**，跳步会报错。

### 第一步：安装 OpenResty

**应用商店** → 搜索 **OpenResty** → 点击安装：

| 参数 | 值 | 说明 |
|------|-----|------|
| 端口 | `80` / `443` | 保持默认，不要改 |

![openresty.png](https://img.yumeko.site/file/blog/articles/1781347012494_openresty.webp)

> [!WARNING] 务必放行端口
> 安装后去 **主机 → 防火墙**，确认 `80` 和 `443` 端口（TCP）已放行，否则后面网站无法访问。
> ![firewall_1.png](https://img.yumeko.site/file/blog/articles/1781346931056_firewall_1.webp)

### 第二步：安装 Redis

RSSHub 用 Redis 做缓存，必须先装。

**应用商店** → 搜索 **Redis** → 点击安装：

| 参数 | 值 | 说明 |
|------|-----|------|
| 名称 | `redis` | 保持默认 |
| 端口 | `6379` | 保持默认 |
| 允许外部访问 | ❌ 不勾选 | Redis 只给 RSSHub 内部用 |
| 密码 | 设置一个 | 务必设置，RSSHub 连接时需要 |

![redis.png](https://img.yumeko.site/file/blog/articles/1781347058087_redis.webp)

> [!DANGER] 安全提醒
> Redis 默认无密码，不设密码等于裸奔。开了外部访问更是会被扫端口+未授权访问，务必设密码、关外部访问。

### 第三步：安装 RSSHub

Redis 运行中之后，**应用商店** → 搜索 **RSSHub** → 点击安装：

| 参数 | 值 | 说明 |
|------|-----|------|
| 名称 | `rsshub` | 保持默认 |
| 端口 | `1200` | 保持默认 |
| 允许外部访问 | ❌ 不勾选 | 通过 Nginx 反代访问 |

![rsshub.png](https://img.yumeko.site/file/blog/articles/1781347097436_rsshub.webp)

安装时 1Panel 会自动检测到 Redis 并建立关联，无需手动配置。

### 第四步：添加 Cloudflare DNS 账户

**面板设置 → 证书 → DNS 账户 → 添加**：

| 字段 | 内容 |
|------|------|
| 类型 | `Cloudflare` |
| Email | Cloudflare 账号邮箱 |
| API Token | Cloudflare API Token |

![Token.png](https://img.yumeko.site/file/blog/articles/1781347300475_Token.webp)

> [!NOTE] API Token 获取
> Cloudflare 控制台 → 右上角头像 → **我的个人资料 → API 令牌 → 创建令牌** → 选 **编辑区域 DNS** 模板：
>
> | 字段 | 填写 |
> |------|------|
> | 区域资源 | `Include` → `All zones` → 选择你的邮箱账户 |
> | 客户端 IP 筛选 | 填写服务器 IP（可选） |
>
> 创建后复制 Token，只显示一次。

### 第五步：创建 ACME 账户

**面板设置 → 证书 → ACME 账户 → 添加**：

| 字段 | 内容 |
|------|------|
| 邮箱 | 你的邮箱 |
| 类型 | `Let's Encrypt` |
| 密钥算法 | `EC256`（一般够用，还有 EC384、RSA 可选） |
| 协议 | 勾选同意 |

![CFAPIToken.png](https://img.yumeko.site/file/blog/articles/1781347850907_CFAPIToken.webp)

### 第六步：申请 SSL 证书

先创建一个空网站作为证书容器：**网站 → 创建网站**，随便填个域名，点确定。

进入空网站的 **HTTPS 设置**：

| 字段 | 内容 |
|------|------|
| 主域名 | `rss.<你的域名>` |
| 申请方式 | `ACME` |
| ACME 账户 | 选择刚创建的 |
| 密钥算法 | `EC256` |
| 验证方式 | `DNS 账户` → 选择刚添加的 Cloudflare |
| 自动续签 | ✅ 勾选 |
| DNS 服务器 | 留空（Cloudflare 走 API，不需要） |
| 推送证书到本地目录 | 不勾 |
| 申请证书之后执行脚本 | 不勾 |

点击保存，1Panel 通过 Cloudflare DNS 验证自动签发证书。成功后刷新页面，证书列表就有数据了。

> Cloudflare 的 DNS 解析记得开 **小黄云**（代理），SSL/TLS 加密模式设为 **完全（严格）**。

### 第七步：创建网站

证书到手后，**网站 → 创建网站**。根据服务来源选不同方式：

#### 方式 A：一键部署（应用商店安装的服务）

RSSHub 是从应用商店装的，选 **一键部署**，1Panel 会自动关联容器：

| 字段 | 内容 |
|------|------|
| 应用 | 选择 `rsshub` |
| 域名 | `rss.<你的域名>` |
| 启用 HTTPS | ✅ 勾选（勾选后出现证书选项） |
| 监听 IPv6 | ✅ 勾选 |

勾选 HTTPS 后下方出现：

| 字段 | 内容 |
|------|------|
| ACME 账户 | 选择之前创建的 |
| 证书 | 选择刚为 `rss.<域名>` 申请的 |

![CreateWebsite.png](https://img.yumeko.site/file/blog/articles/1781347525178_CreateWebsite.webp)

一键部署自动读取容器端口，无需手动填代理地址。

#### 方式 B：反向代理（手动启动的服务）

如果是自己用 `screen`/`npm` 跑的 Node 或 Python 服务，选 **反向代理**：

| 字段 | 内容 |
|------|------|
| 代号 | 网站文件夹名，如 `myblog` |
| 域名 | `<你的域名>` |
| 代理地址 | `http://127.0.0.1:<端口号>` |
| 启用 HTTPS | ✅ 勾选 |
| 监听 IPv6 | ✅ 勾选 |

勾选 HTTPS 后同样会出现 ACME 账户和证书选择。

### 第八步：验证部署

浏览器访问 `https://rss.<你的域名>/bilibili/user/video/2267573`，能看到 XML 格式的 RSS 输出即部署成功。

![RSShub_1.png](https://img.yumeko.site/file/blog/articles/1781347568978_RSShub_1.webp)

---

# 使用

## 基本格式

RSSHub 的路由格式为：

```
https://rss.<你的域名>/<源>/<路由>?<参数>
```

例如你部署的 `https://rss.example.com/bilibili/user/video/2267573`，返回 B 站 UP 主的最新视频 RSS。

## 常用路由

### Bilibili

| 路由 | 说明 |
|------|------|
| `/bilibili/user/video/<UID>` | UP 主视频更新 |
| `/bilibili/user/dynamic/<UID>` | UP 主动态 |
| `/bilibili/bangumi/media/<番剧ID>` | 番剧更新 |
| `/bilibili/hot/search` | 热搜榜 |

### YouTube

| 路由 | 说明 |
|------|------|
| `/youtube/user/<用户名>` | 频道视频更新 |
| `/youtube/channel/<频道ID>` | 频道视频（ID 方式） |

### 社交媒体

| 路由 | 说明 |
|------|------|
| `/twitter/user/<用户名>` | 推特用户推文 |
| `/weibo/user/<UID>` | 微博用户动态 |

### 技术资讯

| 路由 | 说明 |
|------|------|
| `/github/trending/<语言>` | GitHub Trending |
| `/hackernews/<分类>` | Hacker News |
| `/zhihu/daily` | 知乎日报 |
| `/v2ex/topics/<节点>` | V2EX 节点帖子 |

## 通用参数

大部分路由支持以下参数，叠加使用：

| 参数 | 说明 | 示例 |
|------|------|------|
| `filter` | 标题过滤（正则，匹配保留） | `?filter=Python` |
| `filterout` | 标题反过滤（正则，匹配丢弃） | `?filterout=广告` |
| `limit` | 限制条数 | `?limit=10` |
| `sort` | 排序 | `?sort=hot` |

例如只看含 "AI" 的 B 站视频，最多 5 条：

```
https://rss.example.com/bilibili/user/video/2267573?filter=AI&limit=5
```

## 订阅方式

拿到 RSS 链接后，导入你常用的 RSS 阅读器：

- **浏览器**：直接打开链接查看 XML，或用 [Feedly](https://feedly.com)、[Inoreader](https://www.inoreader.com) 等在线服务
- **客户端**：Fluent Reader、NetNewsWire、Reeder 等
- **自建**：搭配 [[Engineering/WebsiteDeployment|网站部署指南]]，在博客或导航页汇总订阅源

> [!TIP]
> 查看 RSSHub 全部路由和参数，访问 [RSSHub 官方文档](https://docs.rsshub.app)。

---

> **相关文章**：
> - [[Engineering/WebsiteDeployment|网站部署指南]]：Nginx 手动配置和 1Panel 面板两种部署方式
