---
title: 网站部署指南：Nginx 手动配置与 1Panel 面板
date: 2026-06-13
category: 工程实践
tags:
  - 服务器
  - nginx
  - 1Panel
  - 网站
description: 两种网站部署方式：传统 Nginx 手动配置，以及使用 1Panel 面板简化操作
image: https://img.yumeko.site/file/blog/cover/1780581818314.webp
status: published
---

# 环境说明

- 服务器：Ubuntu 24.04 LTS
- 前端端口：4321，后端 API 端口：8000

# 前置步骤（两种方式通用）

## 连接服务器

```bash
ssh-keygen -R <IP>   # 删除旧密钥（如需要）
ssh ubuntu@<IP>
```

## 更新系统

```bash
sudo apt update
sudo apt upgrade
```

## 上传项目

```bash
cd /opt
git clone https://path/to/project.git
```

---

# 方式一：Nginx 手动配置

## 安装必要工具

```bash
sudo apt install curl wget vim git htop screen
```

## 安装 Node.js

```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## 安装 Python

```bash
sudo apt install python3 python3-pip python3-venv python-is-python3 pipx
```

## 安装 Nginx

```bash
sudo apt install nginx
```

## 准备 SSL 证书

将证书文件放到 `/etc/nginx/sites-available/` 目录下：

- 证书：`<域名>_bundle.crt`
- 密钥：`<域名>.key`

## 创建 Nginx 配置

在 `/etc/nginx/sites-available/` 下创建配置文件（文件名随意）：

```nginx
# HTTP 重定向到 HTTPS
server {
    listen 80;
    server_name <域名>;
    return 301 https://$server_name$request_uri;
}

# HTTPS 主配置
server {
    listen 443 ssl;
    server_name <域名>;

    # SSL 证书路径
    ssl_certificate /etc/nginx/sites-available/<域名>_bundle.crt;
    ssl_certificate_key /etc/nginx/sites-available/<域名>.key;

    ssl_session_timeout 5m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;

    # 日志
    access_log /var/log/nginx/<域名>_access.log;
    error_log /var/log/nginx/<域名>_error.log;

    # Gzip 压缩
    gzip on;
    gzip_proxied any;
    gzip_vary on;
    gzip_comp_level 6;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/x-javascript
        application/xml
        image/svg+xml
        font/woff
        font/woff2;

    # 前端
    location / {
        proxy_pass http://127.0.0.1:4321;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 后端 API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 屏蔽 WordPress 扫描
    location ~ ^/(wp-admin|wordpress|wp-content|wp-includes)/ {
        return 403;
    }
}
```

## 启用站点

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/<配置文件名> /etc/nginx/sites-enabled/
```

## 检查并重启

```bash
sudo nginx -t
sudo systemctl restart nginx
```

## 启动应用

```bash
# 后端
screen -S backend
cd /opt/project
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
# Ctrl+A, D 分离

# 前端
screen -S frontend
cd /opt/project/frontend
HOST=127.0.0.1 PORT=4321 node ./dist/server/entry.mjs
# Ctrl+A, D 分离
```

---

# 方式二：1Panel 面板

## 安装 1Panel

```bash
curl -sSL https://resource.fit2cloud.com/1panel/package/quick_start.sh -o quick_start.sh
sudo bash quick_start.sh
```

安装完成后会输出面板地址、端口、用户名和密码。

## 安装运行环境

面板中：**应用商店** → 搜索安装 **Node.js** 和 **Python**，一键完成。

## 安装 OpenResty

**应用商店** → 搜索 **OpenResty** → 点击安装：

| 参数 | 值 | 说明 |
|------|-----|------|
| 端口 | 保持默认 `80`/`443` | HTTP/HTTPS 端口 |

OpenResty 是 1Panel 内置的 Web 服务器（Nginx + Lua），安装后才能在面板中创建网站和配置反向代理。

> [!WARNING] 务必放行端口
> 安装 OpenResty 后需要手动放行防火墙端口：**主机 → 防火墙 → 创建规则**，分别放行 `80` 和 `443` 端口（TCP），否则网站无法从外网访问，Cloudflare 会报 522 错误。
> ![firewall_1.webp](https://img.yumeko.site/file/blog/articles/1781346931056_firewall_1.webp)

## 配置 SSL

域名托管在 Cloudflare，用 1Panel 的 ACME DNS 验证自动申请证书，步骤如下：

**第一步：添加 Cloudflare DNS 账户**

**面板设置** → **证书** → **DNS 账户** → 添加：

| 字段 | 内容 |
|------|------|
| 类型 | `Cloudflare` |
| Email | Cloudflare 账号邮箱 |
| API Token | Cloudflare API Token |

> Cloudflare API Token 获取方式：Cloudflare 控制台 → 右上角头像 → **我的个人资料** → **API 令牌** → 创建令牌 → 选择 **编辑区域 DNS** 模板：
>![Token.webp](https://img.yumeko.site/file/blog/articles/1781347300475_Token.webp)
> | 字段 | 填写 |
> |------|------|
> | 区域资源 | `Include` → `All zones` → 选择你的邮箱账户 |
> | 客户端 IP 筛选 | 填写服务器 IP（可选，提高安全性） |
>
> 创建后复制 Token，只显示一次，注意保存。
> ![CFAPIToken.png](https://img.yumeko.site/file/blog/articles/1781347850907_CFAPIToken.webp)

**第二步：创建 ACME 账户**

**面板设置** → **证书** → **ACME 账户** → 添加：

| 字段 | 内容 |
|------|------|
| 邮箱 | 你的邮箱 |
| 类型 | `Let's Encrypt` |
| 密钥算法 | `EC256`（性能好、兼容性高，一般够用） |
| 协议 | 勾选同意 |

**第三步：申请证书**

**网站 → 创建网站**，不要填任何东西，直接点确定——创建一个空网站作为证书容器。然后进入 **HTTPS 设置**：

| 字段 | 内容 |
|------|------|
| 主域名 | `<你的域名>` |
| 申请方式 | `ACME` |
| ACME 账户 | 刚创建的 |
| 密钥算法 | `EC256` |
| 验证方式 | `DNS 账户` → 选择 Cloudflare |
| 自动续签 | ✅ 勾选 |
| DNS 服务器 | 留空（Cloudflare 已通过 API 对接，不需要） |
| 推送证书到本地目录 | 不勾（1Panel 自己管理，无需导出） |
| 申请证书之后执行脚本 | 不勾（一般不触发其他操作） |

保存后 1Panel 通过 DNS 验证自动申请证书，到期自动续期。成功后刷新页面，证书列表就有数据了。

> Cloudflare 的 DNS 解析记得开 **小黄云**（代理）。DNS 验证方式不依赖 HTTP 访问，开着代理不影响申请证书，还能享受 CDN 加速和隐藏源站 IP。
>
> 另外 Cloudflare 的 **SSL/TLS 加密模式** 需要设为 **完全（严格）**，否则 Cloudflare 回源时可能不信任你的证书。

## 创建网站

证书就位后，回到空网站编辑。**网站 → 创建网站**，根据服务来源选不同方式：

### 方式 A：反向代理（手动启动的服务）

前后端是手动跑的，选 **反向代理**：

| 字段 | 内容 |
|------|------|
| 域名 | `<你的域名>` |
| 代号 | 网站文件夹名，如 `myblog` |
| 代理地址 | `http://127.0.0.1:4321` |
| 启用 HTTPS | ✅ 勾选（勾选后出现证书选项） |
| 监听 IPv6 | ✅ 勾选 |

勾选 HTTPS 后下方出现：

| 字段 | 内容 |
|------|------|
| ACME 账户 | 选择之前创建的 |
| 证书 | 选择刚申请的那个 |

### 方式 B：一键部署（应用商店安装的服务）

如果是应用商店装的服务（如 RSSHub），选 **一键部署**，选择对应应用即可，1Panel 自动读取容器端口，无需手动填代理地址。

## 添加后端 API 代理

网站建好后，点进网站 → **配置 → 反向代理 → 创建**：

| 字段 | 内容 |
|------|------|
| 名称 | `后端 API` |
| 匹配规则 | `^~`（前缀匹配） |
| 前端请求路径 | `/api` |
| 后端代理地址 | `http://127.0.0.1:8000` |
| 后端域名 | `$host`（默认，不用改） |

保存后，`/api` 路径的请求即转发到后端 8000 端口。

## 使用原有 Nginx 配置（可选）

1Panel 底层就是标准 Nginx。如果想直接用上面的手动配置：

- **面板粘贴**：网站 → 点击域名 → 配置文件，把方式一的 nginx.conf 贴进去
- **SSH 编辑**：配置文件在 `/etc/nginx/sites-available/` 下，可像手动部署一样编辑

## 启动应用

与方式一相同，使用 `screen` 或用 1Panel 的 **进程守护** 将前后端注册为系统服务。

---

> **相关文章**：[[Engineering/RSSHubDeployment|用 1Panel 部署 RSSHub]] — 从零开始的 RSSHub 实战部署教程

---

# 常用命令速查

| 命令 | 说明 |
|------|------|
| `sudo nginx -t` | 检查配置语法 |
| `sudo systemctl restart nginx` | 重启 Nginx |
| `sudo systemctl reload nginx` | 重载配置（不中断服务） |
| `sudo systemctl status nginx` | 查看 Nginx 状态 |
| `sudo tail -f /var/log/nginx/<域名>_error.log` | 查看错误日志 |

# 两种方式对比

| 步骤 | Nginx 手动 | 1Panel |
|------|-----------|--------|
| 安装环境 | 逐一手动 `apt install` | 应用商店一键 |
| 配置反向代理 | 手写 nginx.conf | 表单填写 |
| SSL 证书 | 手动上传 + 自己续期 | 自动申请 + 自动续期 |
| 添加 API 代理 | 编辑 location 块 | 表单添加 |
| 查看日志 | `tail -f` 命令行 | 面板可视化 |
| 灵活性 | 高，完全掌控 | 中，面板限制 |
| 上手难度 | 需要了解 Nginx 语法 | 低，有手就行 |
