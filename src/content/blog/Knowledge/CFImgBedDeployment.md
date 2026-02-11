---
title: 使用Cloudflare部署免费图床
date: 2025-12-31
category: 小巧思
tags:
  - 免费
  - Cloudflare
  - 资源
description: Cloudflare-imgbed 是一个基于 Cloudflare Workers + R2 的开源图床项目，具有 免费额度高、无需服务器、全球 CDN 加速等优点，非常适合个人博客、笔记、Markdown 图床使用
image: https://img.yumeko.site/file/blog/CFImgBedDeployment.png
status: published
---

## 使用开源项目部署CF的图床

本文将从 **0 到 1** 教你部署一个可用的 Cloudflare 图床。

---

## 部署前准备

* 一个**Cloudflare**账号
* 一个域名（最好是一级域名**domain.com**这种）
* 能访问Github

## 开源项目介绍

这是他们的github项目位置[Cloudflare-ImgBed](https://github.com/MarSeventh/CloudFlare-ImgBed.git)

![ImgBedProjects](https://img.yumeko.site/file/articles/CFImgBedDeployment/ImgBedProjects.png)

你也可以访问他们的网站[CloudFlare ImgBed](https://cfbed.sanyue.de/)

![ImgBedWebsite](https://img.yumeko.site/file/articles/CFImgBedDeployment/ImgBedWebsite.png)

利用CF的免费额度，可以创建存储桶，存放图片

## 获取域名

1. 在**腾讯云**上购买域名
2. 直接在**Cloudflare**上解析

![CFdomain](https://img.yumeko.site/file/articles/CFImgBedDeployment/CFdomain.png)

3. 选择**免费计划**就可以

![CFFreePlan](https://img.yumeko.site/file/articles/CFImgBedDeployment/CFFreePlan.png)

4. 我们在上一步添加站点中选择了**Quick Scan for DNS records**，Cloudflare会帮我们自动扫描域名现有的DNS记录。因为我们的域名刚注册时，默认 DNS 记录一般都是指向域名注册商的服务器，这些默认记录可以全部删除。

删掉之后我们再单独添加两条DNS记录：**@**和**www**

- @ 代表主域名，比如 @ 就表示 github.com。这条记录添加 A 记录，指向我们的服务器公网 IP。
- www 代表带有 www. 前缀的域名（如 `www.github.com`）。这条记录添加 CNAME 记录，指向主域名（也就是 `github.com`）。

![DNSRecords](https://img.yumeko.site/file/articles/CFImgBedDeployment/DNSRecords.png)

5. 添加DNS记录

| Type |    Name    |     IPv4     | Proxy |
| :--: | :--------: | :----------: | :---: |
|  A   | domain.com | 你的服务器IP | 开启  |

| Type  | Name |    IPv4    | Proxy |
| :---: | :--: | :--------: | :---: |
| CNAME | www  | domain.com | 开启  |

填写完成后应该是下面这种

| 类型  |    名称    |     内容     | 代理 |
| :---: | :--------: | :----------: | :--: |
| CNAME |    www     |  domain.com  | 开启 |
|   A   | domain.com | 你的服务器IP | 开启 |

6. 添加域名服务器

> 去你买域名的地方，修改DNS服务器

![CloudflareNameServers](https://img.yumeko.site/file/articles/CFImgBedDeployment/CloudflareNameServers.png)

例如：我是腾讯云购买的域名

![ChangeDNS](https://img.yumeko.site/file/articles/CFImgBedDeployment/ChangeDNS.png)

然后等待解析即可

![CFReady](https://img.yumeko.site/file/articles/CFImgBedDeployment/CFReady.png)

## 部署图床

### 准备R2存储桶

- 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
- 选择**R2 存储对象**
- 然后填写一下支付信息（我个人建议使用**Paypal**）

![R2](https://img.yumeko.site/file/articles/CFImgBedDeployment/R2.png)

- 点击**创建存储桶**（**Create bucket**）
- 输入存储桶名称（全局唯一，随便填）
- 选择存储区域（自动就可以了）
- 点击**创建存储桶**

### Fork项目

1. 访问 [CloudFlare ImgBed 项目](https://github.com/MarSeventh/CloudFlare-ImgBed)
2. 点击右上角的 "Fork" 按钮
3. 选择您的 GitHub 账户
4. 确认 Fork 完成

### 创建 Pages 项目

#### 访问 Cloudflare Dashboard

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. 选择左侧菜单的**计算和AI**（**Compute && AI**）
3. 选择 **Workers & Pages**
4. 点击**创建应用程序**（**Create application**）
5. 在最下方**Looking to deploy Pages?** 选择 **Get started**
6. 在**导入现有 Git 存储库**（**Import an existing Git repository**）处点击**开始使用**

![GetStarted](https://img.yumeko.site/file/articles/CFImgBedDeployment/GetStarted.png)

#### 连接Github仓库

1. 如果首次使用，需要授权 Cloudflare 访问 GitHub
2. 选择您 Fork 的**CloudFlare-ImgBed**仓库
3. 点击**开始设置**

![DeploymentPage](https://img.yumeko.site/file/articles/CFImgBedDeployment/DeploymentPage.png)

#### 配置项目设置

| 配置项       | 值                            | 说明                      |
| :----------- | :---------------------------- | :------------------------ |
| 项目名称     | cloudflare-imgbed（或自定义） | 项目标识符                |
| 生产分支     | main                          | 生产环境分支              |
| 构建命令     | npm install                   | **重要：v2.0 新构建命令** |
| 构建输出目录 | /                             | 保持默认                  |

![PageBuild](https://img.yumeko.site/file/articles/CFImgBedDeployment/PageBuild.png)

#### 部署项目

1. 点击 **保存并部署**
2. 等待首次部署完成（约 2-3 分钟）

### 配置数据库

数据库用于存储文件元数据，是必需的组件，可选数据库为 `KV` 数据库和 `D1` 数据库。两者对比如下表所示，根据自己使用场景**从其中选择一种配置即可**。这里我选择用KV，因为图床不耗费什么额度，读写性能高点更舒服

| 特点       | KV 数据库 | D1 数据库 |
| :--------- | :-------- | :-------- |
| 读写性能   | 高        | 较低      |
| 免费额度   | 少        | 多        |
| 大文件上传 | 支持      | 不支持    |

#### 配置KV数据库

1. 在 Cloudflare Dashboard 中选择**存储和数据库**（**Storage & databases**）
2. 点击 **Workers KV**
3. 点击**创建实例**（**Create Instances**）
4. 输入命名空间名称：**img_url**（建议使用此名称）
5. 点击**创建**

![CreateKV](https://img.yumeko.site/file/articles/CFImgBedDeployment/CreateKV.png)

#### 绑定 KV 到项目

1. 返回您的 Pages 项目
2. 选择 "设置" → "绑定"
3. 点击 "添加" → "KV 命名空间"
4. 填写绑定信息：
   - **变量名称**：`img_url`（必须是这个名称）
   - **KV 命名空间**：选择刚创建的命名空间
5. 点击 "保存"

![BindingKV](https://img.yumeko.site/file/articles/CFImgBedDeployment/BindingKV.png)

![KVnamespace](https://img.yumeko.site/file/articles/CFImgBedDeployment/KVnamespace.png)

#### 配置R2渠道

服务器部署时默认添加了 Cloudflare R2 存储方式，以下步骤仅针对 Cloudflare 部署方式：

在项目设置中绑定 R2 存储桶：

* 选择 "设置" → "绑定"

* 添加 "R2 存储桶"

- **变量名称**：`img_r2`

- **R2 存储桶**：选择已创建的存储桶

#### 第四步：重新部署

![BindingR2](https://img.yumeko.site/file/articles/CFImgBedDeployment/BindingR2.png)

绑定数据库后需要重新部署以生效：

1. 进入项目的**部署**（**Deployment**）页面
2. 找到最新的部署记录
3. 点击右侧的 "..." 菜单
4. 选择**重试部署**（**Retry Deployment**）
5. 等待部署完成

![RetryDeployment](https://img.yumeko.site/file/articles/CFImgBedDeployment/RetryDeployment.png)

### 配置自定义域名

我希望我访问的是**img.domain.com**，而不是**domain.com**

1. 进入**Pages**的设置界面
2. 选择**Custom Domains**
3. 输入自定义域名
4. 应用DNS

![CustomDomain](https://img.yumeko.site/file/articles/CFImgBedDeployment/CustomDomain.png)

## 访问图床

### 域名访问

访问你自己的域名+dashboard

```bash
https://img.domain.com/dashboard
```

### 安全设置

1. 左上角的菜单按钮
2. 进入系统设置界面
3. 进入安全设置界面
4. 设置admin账号密码
5. **重点：**要保存设置

<img src="https://img.yumeko.site/file/articles/CFImgBedDeployment/SafetySetting.png" alt="SafetySetting"  />

![SaveSettings](https://img.yumeko.site/file/articles/CFImgBedDeployment/SaveSettings.png)

### 默认渠道更改

在网页设置中，把默认渠道类型改成R2

![ChangeR2](https://img.yumeko.site/file/articles/CFImgBedDeployment/ChangeR2.png)

## 效果图

![UploadSuccess](https://img.yumeko.site/file/articles/CFImgBedDeployment/UploadSuccess.png)



