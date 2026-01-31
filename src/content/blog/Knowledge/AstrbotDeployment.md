---
title: 如何部署qq机器人Astrbot
date: 2026-01-20
category: 小巧思
tags:
  - 有趣
  - 服务器
description: 如何用服务器部署一个QQ机器人
image: https://img.yumeko.site/file/articles/AstrbotDeployment/AstrSeio.png
status: public
---

## 准备服务器

1. 购买一个服务器

![腾讯云服务器](https://img.yumeko.site/file/articles/AstrbotDeployment/Server.png)

2. 开放防火墙端口

![AstrbotPort](https://img.yumeko.site/file/articles/AstrbotDeployment/AstrbotPort.png)

3. 记住你的服务器ip和port

## 本地手动部署

### 使用screen隔离进程

启动Astrbot进程

```bash
screen -S Astrbot
```

### 从github获取源码

```bash
git clone https://github.com/AstrBotDevs/AstrBot.git
```

### Anaconda获取python环境

1. 创建Astrbot的环境

```bash
conda create -n Astrbot python==3.13
```

2. 进入Astrbot环境

```bash
conda activate Astrbot
```

3. 进入Astrbot文件夹

```bash
cd Astrbot
```

4. 使用requirements.txt下载依赖

```bash
pip install -r requirements.txt
```

### 运行Astrbot，启动WebUI

1. 使用python命令启动main.py

```bash
python main.py
```

2. 查看命令行信息

```bash
✨✨✨
  AstrBot v4.1.5 WebUI 已启动，可访问

   ➜  本地: http://localhost:6185
   ➜  网络: http://127.0.0.1:6185
   ➜  网络: http://10.0.4.15:6185
   ➜  默认用户名和密码: astrbot
 ✨✨✨
```

这个6185端口在服务器防火墙中开启

3. 进入WebUI

```bash
http://{ip}:6185
```

4. 登录WebUI

默认账号和密码都是astrbot

![AstrbotLogin.png](https://img.yumeko.site/file/articles/AstrbotDeployment/AstrbotLogin.png)

## 使用NapCatQQ实现LinuxQQ实例

### 下载NapCat

1. 命令下载

```bash
curl -o \
napcat.sh \
https://nclatest.znin.net/NapNeko/NapCat-Installer/main/script/install.sh \
&& sudo bash napcat.sh
```

2. 完成安装后出现下面信息

```bash
[2025-09-23 11:59:53]: 启动 Napcat (需要图形环境或 Xvfb):
[2025-09-23 11:59:53]:   sudo xvfb-run -a qq --no-sandbox
```

```bash
[2025-09-23 11:59:53]: Napcat 相关信息:
[2025-09-23 11:59:53]:   安装位置: /opt/QQ/resources/app/app_launcher/napcat
[2025-09-23 11:59:53]:   WebUI Token: 查看 /opt/QQ/resources/app/app_launcher/napcat/config/webui.json 文件获取
```

```bash
[2025-09-23 11:59:53]: 后台运行 Napcat (使用 screen)(请使用 root 账户):
[2025-09-23 11:59:53]:   启动: screen -dmS napcat bash -c "xvfb-run -a qq --no-sandbox"
[2025-09-23 11:59:53]:   带账号启动: screen -dmS napcat bash -c "xvfb-run -a qq --no-sandbox -q QQ号"
[2025-09-23 11:59:53]:   附加到会话: screen -r napcat (按 Ctrl+A 然后按 D 分离)
[2025-09-23 11:59:53]:   停止会话: screen -S napcat -X quit
```

```bash
[2025-09-23 11:59:53]: Shell 安装流程完成。
```

### 启动NapCat

1. 使用screen隔离napcat

```bash
screen -S napcat
```

2. 启动Napcat

```bash
sudo xvfb-run -a qq --no-sandbox
```

3. 扫码登录QQ

### 启动NapCatWebUI

1. 查询WebUI的token信息

访问webui的json文件

```bash
vim /opt/QQ/resources/app/app_launcher/napcat/config/webui.json
```

2. 查看json信息

```json
{
    "host": "0.0.0.0", // WebUI 监听地址
    "port": 6099, // WebUI 端口
    "token": "xxxx", // 登录密钥, 默认是自动生成的随机登录密码
    "loginRate": 3, // 每分钟登录次数限制
}
```

![webuiJson.png](https://img.yumeko.site/file/articles/AstrbotDeployment/webuiJson.png)

2. 访问WebUI

```bash
http://127.0.0.1:6099/webui?token=xxxx
```

### WebUI设置

1. 选择WebSocket客户端
2. 设置WebSocket客户端

```bash
URL：ws://{ip}:6199/ws
```

心跳间隔5000，重连间隔5000

![WebSocketClientConfig.png](https://img.yumeko.site/file/articles/AstrbotDeployment/WebSocketClientConfig.png)

保存之后，进入Astrbot仪表盘

### Astrbot完成适配器配置

1. 使用aiocqhttp是适配器

![Adapter.png](https://img.yumeko.site/file/articles/AstrbotDeployment/Adapter.png)

2. 反向Websocket使用端口为6199，token跟NapCat的保持一致

![aiochttp.png](https://img.yumeko.site/file/articles/AstrbotDeployment/aiochttp.png)