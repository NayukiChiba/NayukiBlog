---
title: 如何部署qq机器人Astrbot
date: 2026-02-16
category: 小巧思
tags:
  - 有趣
  - 服务器
description: 如何用服务器部署一个QQ机器人
image: https://img.yumeko.site/file/blog/articles/1780581690796.webp
status: published
---

# 准备服务器

1. 购买一个服务器

![腾讯云服务器](https://img.yumeko.site/file/blog/articles/1780580989264.webp)

2. 开放防火墙端口

![AstrbotPort](https://img.yumeko.site/file/blog/articles/1780580948345.webp)

3. 记住你的服务器ip和port

# 本地手动部署

## 使用screen隔离进程

启动Astrbot进程

```bash
screen -S Astrbot
```

## 从github获取源码

```bash
git clone https://github.com/AstrBotDevs/AstrBot.git
```

## 获取python环境
### 使用Anaconda
* 下载Anaconda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
```
* 自行安装Anaconda

* 创建Astrbot的环境

```bash
conda create -n Astrbot
```

* 进入Astrbot环境

```bash
conda activate Astrbot
```

* 进入Astrbot文件夹

```bash
cd Astrbot
```

* 使用requirements.txt下载依赖

```bash
pip install -r requirements.txt
```

### 使用uv(推荐)
* 下载python
```bash
sudo apt install python3 python3-pip python3-venv python-is-python3 pipx
```
* 进入Astrbot文件夹中创建虚拟环境
```bash
python3 -m venv venv
```
* 下载uv
```bash
source venv/bin/activate
pip install uv
```
* uv同步
```bash
uv sync
```
## 运行Astrbot，启动WebUI

* Anaconda安装使用python命令，uv安装用uv命令

```bash
python main.py
```

```bash
uv run -m main
```

* 查看命令行信息

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

* 进入WebUI

```bash
http://{ip}:6185
```

* 登录WebUI

默认账号和密码都是astrbot

![AstrbotLogin.png](https://img.yumeko.site/file/blog/articles/1780580946194.webp)

## 使用NapCatQQ实现LinuxQQ实例

### 下载NapCat

* 命令下载

```bash
curl -o \
napcat.sh \
https://nclatest.znin.net/NapNeko/NapCat-Installer/main/script/install.sh \
&& sudo bash napcat.sh
```
* 完成安装后出现下面信息
```bash
- Shell (Rootless) 安装完成 -   
安装位置:  /root/Napcat
```

```bash
启动 Napcat (无需 sudo):  
xvfb-run -a /root/Napcat/opt/QQ/qq --no-sandbox
```

```bash
后台运行 Napcat (使用 screen, 无需 sudo):  
启动: screen -dmS napcat bash -c "xvfb-run -a /root/Napcat/opt/QQ/qq --no-sandbox "  
带账号启动: screen -dmS napcat bash -c "xvfb-run -a /root/Napcat/opt/QQ/qq --no-sandbox -q QQ 号码"
```

```bash
后台运行 Napcat (使用 screen)(请使用 root 账户):
启动: screen -dmS napcat bash -c "xvfb-run -a qq --no-sandbox"
带账号启动: screen -dmS napcat bash -c "xvfb-run -a qq --no-sandbox -q QQ号"
附加到会话: screen -r napcat (按 Ctrl+A 然后按 D 分离)
停止会话: screen -S napcat -X quit
```

```bash
Napcat 相关信息:  
插件位置: /root/Napcat/opt/QQ/resources/app/app_launcher/napcat  
WebUI Token: 
查看sudo /root/Napcat/opt/QQ/resources/app/app_launcher/napcat/config/webui.json 
文件获取
```

```bash
TUI-CLI 工具用法 (napcat):  
启动: napcat
```

```bash
Shell (Rootless) 安装流程完成。
```

### 启动NapCat

* 使用screen隔离napcat

```bash
screen -S napcat
```

* 启动Napcat

```bash
sudo xvfb-run -a /root/Napcat/opt/QQ/qq --no-sandbox
```

* 扫码登录QQ

### 启动NapCatWebUI

* 查询WebUI的token信息

访问webui的json文件

```bash
sudo vim /root/Napcat/opt/QQ/resources/app/app_launcher/napcat/config/webui.json 
```

* 查看json信息

```json
{
    "host": "0.0.0.0", // WebUI 监听地址
    "port": 6099, // WebUI 端口
    "token": "xxxx", // 登录密钥, 默认是自动生成的随机登录密码
    "loginRate": 3, // 每分钟登录次数限制
}
```

![webuiJson.png](https://img.yumeko.site/file/blog/articles/1780581026897.webp)

* 访问WebUI

```bash
http://127.0.0.1:6099/webui?token=xxxx
```

### WebUI设置

* 选择WebSocket客户端
* 设置WebSocket客户端

```bash
URL：ws://{ip}:6199/ws
```

心跳间隔5000，重连间隔5000

![WebSocketClientConfig.png](https://img.yumeko.site/file/blog/articles/1780581024368.webp)

保存之后，进入Astrbot仪表盘

### Astrbot完成适配器配置

* 使用aiocqhttp是适配器

![Adapter.png](https://img.yumeko.site/file/blog/articles/1780580944965.webp)

* 反向Websocket使用端口为6199，token跟NapCat的保持一致

![aiochttp.png](https://img.yumeko.site/file/blog/articles/1780580948280.webp)

# 1Panel部署

## 安装Astrbot

* 在应用商店中搜索Astrbot
* 安装即可
![安装Astrbot](https://img.yumeko.site/file/blog/articles/1780580931960.webp)

## 设定容器规则

> [!INFO] 高级设置
> 可以把容器名字改一下，无所谓改不改

> [!NOTE] 重启规则
> 我建议选择不重启，因为出问题一直重启也会有问题

> [!TIP] 勾选端口外部访问
> 不勾选这个就无法访问webui了

![容器规则](https://img.yumeko.site/file/blog/articles/1780580951988.webp)

## 安装NapCat

### 获取NapCat的镜像

在源代码的README中，命令行安装为
```bash
docker run -d \
-e NAPCAT_GID=$(id -g) \
-e NAPCAT_UID=$(id -u) \
-p 3000:3000 \
-p 3001:3001 \
-p 6099:6099 \
--name napcat \
--restart=always \
mlikiowa/napcat-docker:latest
```

* 所以我们拉取`mlikiowa/napcat-docker:latest`镜像即可
* 设置端口映射
	* `3000:3000`
	* `3001:3001`
	* `6099:6099`
![拉取Napcat镜像](https://img.yumeko.site/file/blog/articles/1780580984870.webp)
### 启动容器

* 使用刚才拉取的镜像

![创建napcat容器](https://img.yumeko.site/file/blog/articles/1780580973908.webp)

> [!WARNING] napcat网络
> 需要napcat跟astrbot处于同一个网络

* 设置容器的网络
![NapcatNetwork.png](https://img.yumeko.site/file/blog/articles/1780580970457.webp)

### 确定astrbot和napcat处于同一网络

![Network.png](https://img.yumeko.site/file/blog/articles/1780580970273.webp)

## 启动NapCat

### 查看日志，进入WebUI

* 在napcat的容器右侧，有一个`日志`，点击查看webui的信息
![NapCatLog.png](https://img.yumeko.site/file/blog/articles/1780580966975.webp)

* 进入网页
```
http://{ip}:6099
```

* 复制上图中的token进入网页中
## 设置NapCat客户端

* 新建`WebSocket`**客户端**

> [!DANGER] 注意
> 是**客户端**，不是**服务器**

![NapCatSetting.png](https://img.yumeko.site/file/blog/articles/1780580967835.webp)

* 设置`WebSocket`客户端

![WebSocketClient.png](https://img.yumeko.site/file/blog/articles/1780581000311.webp)

## 设置Astrbot

### 进入WebUI

* Astrbot的WebUI地址为
```
http://{ip}:6185
```
* 进入之后修改账号密码
### 添加适配器

* 使用`OneBot v11`
* 复制上面的token到设置中，保持统一
![QQSetting.png](https://img.yumeko.site/file/blog/articles/1780580981811.webp)

## 验证

自行加入模型提供商的llm模型之后，在qq中发送消息即可
![Check.png](https://img.yumeko.site/file/blog/articles/1780580956943.webp)

# 使用更加简单的同时编排
## 放开 `ptrace` 限制

> [!DANGER] 注意
> **这一步必须先做，否则容器起来后 hook 注入会失败。**

SnowLuma 的 native addon 通过 `ptrace` 把 hook 注入到容器内的 QQ 进程。现代 Linux 发行版（Ubuntu / Debian / Arch 等）默认把 `kernel.yama.ptrace_scope` 设为 `1` 或更高，**即使容器启动加了 `--cap-add=SYS_PTRACE`，注入也会被宿主内核拦掉**。

在宿主机上**只需要执行一次**：

```
echo 'kernel.yama.ptrace_scope=0' | sudo tee /etc/sysctl.d/99-snowluma-ptrace.conf
```

验证是否生效：

```
sysctl kernel.yama.ptrace_scope

```
- 期望输出：kernel.yama.ptrace_scope = 0
写到 `/etc/sysctl.d/99-snowluma-ptrace.conf` 是为了让设置在重启后自动恢复。如果你的环境对宿主机安全有更严格要求（比如多租户、面向公网），请考虑使用受信任的隔离机制（独立物理机 / VM）替代这一步。

## 获取`compose.yml`
### Astrbot的`compose.yml`

```dockerfile
version: '3.8'

# When connecting to OneBot v11 Napcat, please use this compose file for one-click deployment: https://github.com/NapNeko/NapCat-Docker/blob/main/compose/astrbot.yml

services:
  astrbot:
    image: soulter/astrbot:latest
    container_name: astrbot
    restart: always
    security_opt:
      - no-new-privileges:true
    ports:
      - "6185:6185" # AstrBot WebUI
      - "6199:6199" # Optional. OneBot v11 Napcat Websocket Port
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - ./data:/AstrBot/data
      # - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
```

### Snowluma的`compose.yml`

```dockerfile

services:
  snowluma:
    image: ${SNOWLUMA_IMAGE:-motricseven7/snowluma:latest}
    container_name: ${SNOWLUMA_CONTAINER:-snowluma}
    restart: unless-stopped
    shm_size: 1gb
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    environment:
      VNC_PASSWD: ${VNC_PASSWD:-vncpasswd}
      SNOWLUMA_UID: ${SNOWLUMA_UID:-1000}
      SNOWLUMA_GID: ${SNOWLUMA_GID:-1000}
      SNOWLUMA_WEBUI_PORT: ${SNOWLUMA_WEBUI_PORT:-5099}
      SNOWLUMA_LOG_LEVEL: ${SNOWLUMA_LOG_LEVEL:-info}
      SNOWLUMA_SCREEN: ${SNOWLUMA_SCREEN:-1920x1080x16}
      SNOWLUMA_HOOK_AUTOLOAD: ${SNOWLUMA_HOOK_AUTOLOAD:-1}
    ports:
      - "${VNC_PORT:-5900}:5900"
      - "${NOVNC_PORT:-6081}:6081"
      - "${SNOWLUMA_WEBUI_HOST_PORT:-5099}:${SNOWLUMA_WEBUI_PORT:-5099}"
      - "${ONEBOT_HTTP_PORT:-3000}:3000"
      - "${ONEBOT_WS_PORT:-3001}:3001"
    volumes:
      - snowluma-data:/app/snowluma-data
      - snowluma-qq-config:/app/.config
      - snowluma-qq-data:/app/.local/share

volumes:
  snowluma-data:
  snowluma-qq-config:
  snowluma-qq-data:
```

### 把他们合并一下

```dockerfile
version: '3.8'

# When connecting to OneBot v11 Napcat, please use this compose file for one-click deployment: https://github.com/NapNeko/NapCat-Docker/blob/main/compose/astrbot.yml

services:
  astrbot:
    image: soulter/astrbot:latest
    container_name: astrbot
    restart: always
    security_opt:
      - "no-new-privileges:true"
    ports:
      - "6185:6185" # AstrBot WebUI
      - "6199:6199" # Optional. OneBot v11 Napcat Websocket Port
    environment:
      - TZ=Asia/Shanghai
    volumes:
      - astrbot-data:/AstrBot/data
      # - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
    networks:
      - astrbot-net

  snowluma:
    image: ${SNOWLUMA_IMAGE:-motricseven7/snowluma:latest}
    container_name: ${SNOWLUMA_CONTAINER:-snowluma}
    restart: unless-stopped
    shm_size: 1gb
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    environment:
      VNC_PASSWD: ${VNC_PASSWD:-vncpasswd}
      SNOWLUMA_UID: ${SNOWLUMA_UID:-1000}
      SNOWLUMA_GID: ${SNOWLUMA_GID:-1000}
      SNOWLUMA_WEBUI_PORT: ${SNOWLUMA_WEBUI_PORT:-5099}
      SNOWLUMA_LOG_LEVEL: ${SNOWLUMA_LOG_LEVEL:-info}
      SNOWLUMA_SCREEN: ${SNOWLUMA_SCREEN:-1920x1080x16}
      SNOWLUMA_HOOK_AUTOLOAD: ${SNOWLUMA_HOOK_AUTOLOAD:-1}
    ports:
      - "${VNC_PORT:-5900}:5900"
      - "${NOVNC_PORT:-6081}:6081"
      - "${SNOWLUMA_WEBUI_HOST_PORT:-5099}:${SNOWLUMA_WEBUI_PORT:-5099}"
      - "${ONEBOT_HTTP_PORT:-3000}:3000"
      - "${ONEBOT_WS_PORT:-3001}:3001"
    volumes:
      - snowluma-data:/app/snowluma-data
      - snowluma-qq-config:/app/.config
      - snowluma-qq-data:/app/.local/share
      - astrbot-data:/AstrBot/data
    networks:
      - astrbot-net

networks:
  astrbot-net:
    driver: bridge

volumes:
  snowluma-data:
  snowluma-qq-config:
  snowluma-qq-data:
  astrbot-data:
```

> [!INFO] 使用教程
> 直接把上面的合并`compose.yml`复制到 1Panel 的**编排**部分中

## 使用 `compose.yml`
在 1Panel 的**容器** -> **编排**中粘贴上面的合并compose即可，然后设置一下环境变量

![ComposeStep.png](https://img.yumeko.site/file/blog/articles/1780580956835.webp)

## 环境变量

![SnowlumaEnv.png](https://img.yumeko.site/file/blog/articles/1780580990985.webp)

> [!WARNING] 注意
> 要设置环境变量`VNC_PASSWD={密码}`，不然别人可以直接登录你的界面

## 打开防火墙

![Firewall.png](https://img.yumeko.site/file/blog/articles/1780580956580.webp)

- 打开 `6185`访问Astrbot的WebUI
- 打开`6081`是用来可视化登录qq的
- 打开`5099`是用来查看qq状态的

## 登录qq
访问 `http://{ip}:6081`进入VNC界面，这是一个可视化的虚拟机，可以直接扫码登录qq
![VNC.png](https://img.yumeko.site/file/blog/articles/1780580996573.webp)
凭证就是设置的**环境变量**`VNC_PASSWD`，然后登录进去扫码qq即可

## 设置OneBot的Websocket客户端

访问`http://{ip}:5099`进入Snowluma的Onebot控制台，如果qq正常登录，那么就是成功了，如果出现错误，请再来一遍
> [!TIP] 注意
> 优先排查是否放开`ptrace`限制
> 然后看登录状态是否正确

![SnowlumaControl.png](https://img.yumeko.site/file/blog/articles/1780580990796.webp)

开始设置**WebSocket客户端**

> [!WARNING] 注意
> 注意是客户端，不是服务端

![WSClient.png](https://img.yumeko.site/file/blog/articles/1780581025042.webp)

token就是密码，注意设置
![WSSettting.png](https://img.yumeko.site/file/blog/articles/1780581034519.webp)

> [!TIP] 设置完成之后
> 记得保存，不保存没有用

## 设置Astrbot的Onebot服务端


![Onebot.png](https://img.yumeko.site/file/blog/articles/1780580981083.webp)

注意输入Token，这样就可以设置成功了