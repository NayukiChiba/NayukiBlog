---
title: 如何完整部署一个网址
date: 2026-01-02
category: 技术
tags:
  - 服务器
  - 资源
  - 网站
description: 一般使用Nginx，但是可以用宝塔和1Panel简化流程
image: https://img.yumeko.site/file/blog/nginx.png
status: published
---

## 使用nginx

> 部署说明：
>
> 1. 使用fastapi+astro+sqlite
> 2. 服务器为ubuntu 24.04 LTS
> 3. 前端端口为4321，后端api端口为8000

### 连接服务器

```bash
# 删除该IP的旧密钥记录
ssh-keygen -R IP

# 然后重新连接
ssh ubuntu@IP
```
### 更新系统

```bash
sudo apt update
sudo apt upgrade
```
### 安装必要工具

#### 必须安装

```bash
sudo apt install curl wget vim git htop screen
```
#### Nodejs

```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```
#### Python

```bash
sudo apt install python3 python3-pip python3-venv python-is-python3 pipx
```
#### uv

1. 直接用官方下载

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

2. 在本地下载后，scp到服务器

```bash
chmod +x install.sh
./install.sh
```

3. 使用PyPI安装（推荐）

```bash
python -m venv venv
source ./venv/bin/activate
pip install uv
```
#### nginx

```bash
sudo apt install nginx
```
### 获取项目

1. git 拉取

```bash
git clone https://path/to/projects.git
```

2. scp上传

```
scp -r projects/ username@ip:/path
```
### 构建前端依赖

```bash
cd frontend
npm install 
// 自行处理依赖问题
```
### 构建后端

```bash
cd app
python -m venv venv
source venv/bin/activate
pip install uv
uv sync
```
### 设置nginx

1. 在nginx配置文件夹中创建配置文件

* 进入配置文件夹：`/etc/nginx/site-available/`
* 创建配置文件`{nginx_filename}`（随便什么名字）
* 把两个**ssl**证书和key放到文件夹下
  * 证书1：`{domain.name}_bundle.crt`
  * 证书2：`{domain.name}.key`
* 在`domain`文件中添加配置

```nginx
# HTTP 重定向到 HTTPS
server {
    listen 80;
    server_name {domain.name}; # 替换为你的域名
    return 301 https://$server_name$request_uri;
}

# HTTPS 主配置
server {
    listen 443 ssl;
    server_name {domain.name}; # 替换为你的域名

    # SSL 证书路径 (请替换为真实路径，不要带大括号)
    ssl_certificate /etc/nginx/sites-available/{domain.name}_bundle.crt;
    ssl_certificate_key /etc/nginx/sites-available/{domain.name}.key;

    ssl_session_timeout 5m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;

    # 日志
    access_log /var/log/nginx/{domain.name}_access.log;
    error_log /var/log/nginx/{domain.name}_error.log;
	
    # ========== Gzip 压缩 ==========
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
    # =====================================================
    # 1. 前端配置 (Astro SSR) - 关键修改
    # =====================================================
    location / {
        # 不要用 root 和 try_files
        # 转发给 Screen 中运行的 Node.js 端口 (默认 4321)
        proxy_pass http://127.0.0.1:4321;

        proxy_http_version 1.1;
        proxy_set_header Origin "http://127.0.0.1:4321";
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # =====================================================
    # 2. 后端 API 配置 (FastAPI)
    # =====================================================
    location /api {
        # 转发给 Screen 中运行的 Python 端口 (8000)
        # 注意：这里不要加最后的斜杠，让 Nginx 原样透传路径
        proxy_pass http://127.0.0.1:8000;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # =====================================================
    # 3. 其他配置
    # =====================================================

    # 屏蔽 WordPress 扫描
    location ~ ^/(wp-admin|wordpress|wp-content|wp-includes)/ {
    	return 403;
    }
}     
```
### 创建nginx配置软连接（重要！！）

* 删除原本的`default`配置，用自己的设置

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/{nginx_filename} /etc/nginx/sites-enabled/
```
### 重启nginx服务

```bash
sudo nginx -t
sudo systemctl restart nginx
```
### 使用screen启动

1. 启动后端

```bash
screen -S backend
source venv/bin/activate
uv run uvicorn app.main:app
```

2. 启动前端

```bash
screen -S frontend
cd frontend
npm run build
HOST=127.0.0.1 POST=4321 node ./dist/server/entry.mjs
```



