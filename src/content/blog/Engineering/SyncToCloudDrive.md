---
title: 本地同步网盘
date: 2026-06-08
category: 工程实践
tags:
  - 同步
  - 本地
  - 网盘
description: 把本地文件夹备份到网盘的同时，排除一些必要的文件夹
image: https://img.yumeko.site/file/blog/cover/1780914638454.svg
status: published
---
# 配置OpenList
## 下载OpenList

> [!NOTE] OpenList文档
> 可以访问[OpenList](https://doc.oplist.org/)官网查看文档
## 推荐按使用Docker部署
1. 创建一个`openlist/`文件夹
2. 在`openlist/`下创建`data`文件夹
3. 在openlist/文件夹下写docker-compose.yml
```yaml
services:
  openlist:
    image: openlistteam/openlist:latest
    container_name: openlist
    volumes:
      - ./data:/opt/openlist/data
    ports:
      - "5244:5244"
    environment:
      - UMASK=022
    restart: unless-stopped
```
`data/`下面保存的是你的数据，不是网盘内容
![1780909400050.png](https://img.yumeko.site/file/blog/articles/1780909431081.webp)

4. 然后在`openlist/`文件夹下启动docker
```bash
docker compose up -d
```
5. 在docker的log中找到密码
6. 登录`127.0.0.1:5244`
![1780907572176.png](https://img.yumeko.site/file/blog/articles/1780909458738.webp)
账号默认是admin，密码在log中
## 挂载百度网盘到AList

![1780907712960.png](https://img.yumeko.site/file/blog/articles/1780909563840.webp)

> [!ATTENTION] 注意！！！
> 两种获取百度网盘token的方法都需要在[这里](https://api.oplist.org/)来获取
> ![1780909503749.png](https://img.yumeko.site/file/blog/articles/1780909532257.webp)
### 获取令牌
#### 使用百度网盘开发者权限


> [!NOTE] 提示
> 请注意如使用百度网盘的api，请务必确保OpenList存储设置中**使用在线API**选项为非勾选状态，否则会导致无法连接。

1. 前往百度开放平台
![dev_platform.png](https://img.yumeko.site/file/blog/articles/1780909603466.webp)
2. 登录百度账号
![dev_platform_login.jpg](https://img.yumeko.site/file/blog/articles/1780909621052.webp)
3. 按照步骤填写信息完成开发者认证。

4. 在完成后系统会自动跳转到百度开放控制台

5. 使用控制台的创建按钮创建一个应用

> [!ATTENTION] 注意！！！
> 请注意个人开发者只能创建一个应用，如您对应用有其他使用，请使用第二和三方法，此处默认您没有应用且愿意为openlist创建应用


![dev_app_create_button.jpg](https://img.yumeko.site/file/blog/articles/1780909632723.webp)

6. 选择应用类别为**软件**，并按自己的想法填写应用名称和描述，请注意这三个内容一旦设定完成无法更改，请深思熟虑后填写。
![dev_app_create_panel.jpg](https://img.yumeko.site/file/blog/articles/1780909658426.webp)

7. 创建完成后将会回到控制面板，点击应用名称进入应用详情页。
![dev_app_console.jpg](https://img.yumeko.site/file/blog/articles/1780909664360.webp)
> [!ATTENTION] 注意！
> 您不需要申请上线审核就可以正常使用

8. 选择安全设置，并填写应用回调地址为提供的回调地址并保存
```text
https://api.oplist.org/baiduyun/callback
```
![dev_app_settings_safety.jpg](https://img.yumeko.site/file/blog/articles/1780909693252.webp)

9. 回到应用详情页，复制必要的`Appkey`与`Secretkey`
![dev_app_info.jpg](https://img.yumeko.site/file/blog/articles/1780909692406.webp)

10. 使用[OpenList Token 获取工具](https://api.oplist.org/)通过百度网盘验证登录填写对应的Key，获取令牌
![dev_token.png](https://img.yumeko.site/file/blog/articles/1780909711439.webp)


#### 使用OpenList的刷新Token

内置API调用的实现方式，利用我们建立的[服务器中转](https://api.oplist.org/)刷新`access_token`，目前已经上线，勾选使用OpenList提供的参数，点击获取token即可获得刷新令牌。在OpenList的存储配置界面，打开使用online api选项，经刷新令牌填入即可使用。
![non_dev_token.png](https://img.yumeko.site/file/blog/articles/1780909729126.webp)

### 修改User-Agent

我是svip，不用改，嘻嘻
### 效果

![1780909777447.png](https://img.yumeko.site/file/blog/articles/1780909792356.webp)
# 开启WebDAV
## 设置用户权限

给一个用户开启**WebDAV读取**和**WebDAV管理**功能
1. **WebDAV 读取**
    
    - 必须开启此权限才能**查看和读取** WebDAV 中的文件和目录。
    - 如果用户**仅需查看或播放文件**，开启此权限即可。
2. **WebDAV 管理**
    
    - 必须开启此权限才能进行**写入操作**（创建、修改、删除等）。
    - **仅开启 `WebDAV 管理` 还不够！** 需要同时开启 `WebDAV 管理` **以及** 其计划执行操作所需的具体文件系统权限（如 `重命名`、`删除`、`复制`、`创建目录或上传` 等）。
## 基础连接配置

使用以下参数连接你的 WebDAV 客户端：

| 配置项           | 值 / 说明                                    |
| ------------- | ----------------------------------------- |
| **Url**       | `http[s]://你的域名:端口/dav/`                  |
| **Host / 主机** | 你的域名 (例如: `openlist.example.com`)         |
| **路径 / Path** | `dav`                                     |
| **协议**        | `http` 或 `https` (强烈建议使用 **https** 以保障安全) |
| **端口**        | 与访问 OpenList 网页端使用的端口**完全一致**             |
| **用户名**       | 你在 OpenList 网页端登录使用的**用户名**               |
| **密码**        | 你在 OpenList 网页端登录使用的**密码**                |
## 验证
在`powershell`中使用指令

```bash
curl.exe -X PROPFIND -u "用户名:密码" "http://127.0.0.1:5244/dav/" -H "Depth: 1"
```

返回内容
```bash
<?xml version="1.0" encoding="UTF-8"?><D:multistatus xmlns:D="DAV:"><D:response><D:href>/dav/</D:href><D:propstat><D:prop><D:resourcetype><D:collection xmlns:D="DAV:"/></D:resourcetype><D:displayname>root</D:displayname><D:creationdate>0001-01-01T00:00:00Z</D:creationdate><D:getlastmodified>Mon, 01 Jan 0001 00:00:00 GMT</D:getlastmodified><D:supportedlock><D:lockentry xmlns:D="DAV:"><D:lockscope><D:exclusive/></D:lockscope><D:locktype><D:write/></D:locktype></D:lockentry></D:supportedlock></D:prop><D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response><D:response><D:href>/dav/baidu/</D:href><D:propstat><D:prop><D:getlastmodified>Mon, 08 Jun 2026 09:16:00 GMT</D:getlastmodified><D:supportedlock><D:lockentry xmlns:D="DAV:"><D:lockscope><D:exclusive/></D:lockscope><D:locktype><D:write/></D:locktype></D:lockentry></D:supportedlock><D:resourcetype><D:collection xmlns:D="DAV:"/></D:resourcetype><D:displayname>baidu</D:displayname><D:creationdate>2026-06-08T09:16:00Z</D:creationdate></D:prop><D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response></D:multistatus>curl: (3) URL rejected: Bad hostname
```

返回了`200 OK`，证明没问题

# rclone使用
## 下载rclone
### 加入到环境变量中
在rclone的[github仓库](https://github.com/rclone/rclone)中下载rclone的最新版本，或者在[官网](https://rclone.org/downloads/)中下载
![1780910550682.png](https://img.yumeko.site/file/blog/articles/1780910593066.webp)
把rclone加入到环境变量中
![1780910644897.png](https://img.yumeko.site/file/blog/articles/1780910721003.webp)
![1780910685871.png](https://img.yumeko.site/file/blog/articles/1780910720983.webp)
### 验证
![1780910762069.png](https://img.yumeko.site/file/blog/articles/1780910773853.webp)

## 配置rclone
1. 输入`rclone config`
![1780912557466.png](https://img.yumeko.site/file/blog/articles/1780912618407.webp)
2. 输入`n`配置新的`remote`
3. 输入`name`
4. 在许多的值中输入`webdav
![1780912691229.png](https://img.yumeko.site/file/blog/articles/1780912709636.webp)
5. 输入`webdav`的地址`http://127.0.0.1:5244`
6. 供应商选择`8`
![1780912759235.png](https://img.yumeko.site/file/blog/articles/1780913215162.webp)
> [!DANGER] 不要选择7
> **第 7 项 `rclone WebDAV server`** 专指 **rclone 自己搭建的 WebDAV 服务器**（即运行 `rclone serve webdav` 的情况），而不是用来连接第三方 WebDAV。
7. 输入用户名，然后输入`y`，然后输入密码
8. `bearer_toekn`直接回车，不用填
9. 不用填`Advanced config`
回车后就是完成了

![1780913242109.png](https://img.yumeko.site/file/blog/articles/1780913251048.webp)



## 配置过滤功能
在自己希望同步的文件夹中创建一个`exclude.txt`文件，表示这些文件夹我不想上传
```txt
.venv/**
venv/**
node_modules/**
target/**
__pycache__/**
.pytest_cache/**
.ruff_cache/**
.git/**
dist/**
build/**
.idea/**
.vscode/**
*.log
*.tmp
dataset/**
models/**
checkpoints/**
```

![1780913492442.png](https://img.yumeko.site/file/blog/articles/1780913716365.webp)
## 测试同步

1. 我在`Book`文件夹中创建一个`.venv`文件夹，里面放了一个`test.txt`文件

![1780913505415.png](https://img.yumeko.site/file/blog/articles/1780913714486.webp)

2. 在同步文件夹中打开终端，输入`rclone`指令
```bash
rclone sync .\Book\ openlist:/baidu/YeYann/Book --exclude-from .\exclude.txt --transfers 2 --retries 999 --progress
```

说明：

- `copy` → 只上传新增/更新文件，不删除目标
- `--exclude-from exclude.txt` → 忽略不需要同步的目录
- `--transfers 2` → 同时上传两个文件，降低失败率
- `--retries 999` → 自动重试
- `--progress` → 显示上传进度
![1780913651535.png](https://img.yumeko.site/file/blog/articles/1780913711364.webp)

3. 在`openlist`中显示出了我们上传的文件夹，而且没有`venv`
![1780913670807.png](https://img.yumeko.site/file/blog/articles/1780913713679.webp)

# rclone指令
## 同步指令
### 空跑确认
```bash
rclone sync ".\" alist:/baidu/YeYann --exclude-from ".\exclude.txt" --transfers 2 --retries 999 --dry-run --progress
```

这个指令中有一个`--dry-run`，只显示要`sync`的内容，但是不同步

### 直接同步
确认之后，可以直接sync
> [!DANGER] 危险
> `sync` 会**删除**云端里比本地**多出来**的文件和目录，请先空跑确认：

```bash
rclone sync ".\" alist:/baidu/YeYann --exclude-from ".\exclude.txt" --transfers 2 --retries 999 --progress
```

## 上传指令
```bash
rclone copy ".\" alist:/baidu/YeYann --exclude-from ".\exclude.txt" --transfers 2 --retries 999 --progress
```
建议同样先 `--dry-run` 看一眼再正式运行。

## 其他指令和参数

| 参数                        | 说明                   |
| ------------------------- | -------------------- |
| `--dry-run` / `-n`        | 空跑，只显示操作不执行          |
| `--progress`              | 显示实时进度               |
| `--transfers N`           | 并行传输文件数              |
| `--retries N`             | 失败重试次数               |
| `--checksum`              | 通过哈希比较文件（更准确）        |
| `--exclude` / `--include` | 过滤文件（支持 `*`、`?` 通配符） |
| `--exclude-from 文件`       | 从文本文件读取排除规则          |
| `--ignore-existing`       | 跳过目标已存在的文件           |
| `--update`                | 仅源文件比目标新时覆盖          |
