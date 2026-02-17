---
title: 服务器管理
date: 2026-02-14
category: 小巧思
tags:
  - 服务器
  - 工具
description: 安装一下1Panel
image: https://img.yumeko.site/file/blog/1Panel.png
status: published
---
# 下载1Panel
## 准备Linux服务器

1. 确保您有一台运行Linux系统的服务器，支持CentOS、Ubuntu、Debian等主流发行版，及麒麟、统信等国产操作系统

2. 支持各种服务器架构：x86_64、aarch64、armv7l、ppc64le、s390x
## 运行安装脚本

* 以root用户身份运行一键安装脚本，自动完成1Panel的下载和安装

```bash
bash -c "$(curl -sSL https://resource.fit2cloud.com/1panel/package/v2/quick_start.sh)"
```

## 访问管理面板

* 安装完成后，通过浏览器访问安装脚本提示的访问地址，开始使用`1Panel`
* 选择安装目录
```bash
设置 1Panel 安装目录 (默认为 /opt):  
[1Panel 2026-02-14 02:58:44 install Log]: 您选择的安装路径是 /opt  
```

* 安装`docker`
```bash
检测到未安装 Docker，是否安装 [y/n]: y  
[1Panel 2026-02-14 02:58:48 install Log]: ... 在线安装 Docker  
[1Panel 2026-02-14 02:58:48 install Log]: 无需更改源
==============================================================
+ sh -c docker version
Client: Docker Engine - Community
 Version:           29.2.1
 API version:       1.53
 Go version:        go1.25.6
 Git commit:        a5c7197
 Built:             Mon Feb  2 17:17:26 2026
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          29.2.1
  API version:      1.53 (minimum version 1.44)
  Go version:       go1.25.6
  Git commit:       6bc6209
  Built:            Mon Feb  2 17:17:26 2026
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v2.2.1
  GitCommit:        dea7da592f5d1d2b7755e3a161be07f43fad8f75
 runc:
  Version:          1.3.4
  GitCommit:        v1.3.4-0-gd6d73eb8
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
==============================================================
[1Panel 2026-02-14 02:59:58 install Log]: ... 启动 Docker  
Synchronizing state of docker.service with SysV service script with /usr/lib/systemd/systemd-sysv-install.  
Executing: /usr/lib/systemd/systemd-sysv-install enable docker  
[1Panel 2026-02-14 02:59:59 install Log]: Docker 安装成功  
```

* 设置`1Panel`端口，设置**用户名**和**密码**
```bash
设置 1Panel 端口 (默认是 28076):  
[1Panel 2026-02-14 03:00:04 install Log]: 您设置的端口是: 28076  
[1Panel 2026-02-14 03:00:04 install Log]: 正在打开防火墙端口 28076  
Rules updated  
Rules updated (v6)  
Firewall not enabled (skipping reload)  
设置 1Panel 安全入口 (默认是 {密码}):  
[1Panel 2026-02-14 03:00:18 install Log]: 您设置的面板安全入口是 {密码}  
设置 1Panel 面板用户 (默认是 {用户名}):  
[1Panel 2026-02-14 03:00:20 install Log]: 您设置的面板用户是 {用户名}  
[1Panel 2026-02-14 03:00:20 install Log]: 设置 1Panel 面板密码，设置后按回车键继续 (默认是 {密码}):  
```

* `1Panel`启动
```bash
[1Panel 2026-02-14 03:00:29 install Log]: 正在配置 1Panel 服务  
Created symlink /etc/systemd/system/multi-user.target.wants/1panel-agent.service → /etc/systemd/system/1panel-agent.service.  
Created symlink /etc/systemd/system/multi-user.target.wants/1panel-core.service → /etc/systemd/system/1panel-core.service.  
[1Panel 2026-02-14 03:00:30 install Log]: 正在启动 1Panel 服务  
[1Panel 2026-02-14 03:00:30 install Log]: 1Panel 服务已成功启动，正在继续执行后续配置，请稍候...  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: =================感谢您的耐心等待，安装已完成==================  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: 请使用您的浏览器访问面板:  
[1Panel 2026-02-14 03:00:43 install Log]: 外部地址: http://{ip}:28076/3afda6a3f7  
[1Panel 2026-02-14 03:00:43 install Log]: 内部地址: http://{ip}:28076/3afda6a3f7  
[1Panel 2026-02-14 03:00:43 install Log]: 面板用户: {用户名}  
[1Panel 2026-02-14 03:00:43 install Log]: 面板密码: {密码}  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: 官方网站: https://1panel.cn  
[1Panel 2026-02-14 03:00:43 install Log]: 项目文档: https://1panel.cn/docs  
[1Panel 2026-02-14 03:00:43 install Log]: 代码仓库: https://github.com/1Panel-dev/1Panel  
[1Panel 2026-02-14 03:00:43 install Log]: 前往 1Panel 官方论坛获取帮助: https://bbs.fit2cloud.com/c/1p/7  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: 如果您使用的是云服务器，请在安全组中打开端口 28076  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: 为了您的服务器安全，离开此屏幕后您将无法再次看到您的密码，请记住您的密码。  
[1Panel 2026-02-14 03:00:43 install Log]:  
[1Panel 2026-02-14 03:00:43 install Log]: ================================================================
```

