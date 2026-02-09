---
title: 关于Github的进阶用法
date: 2026-01-25
category: 技术
tags:
  - 高级教程
  - 资源
description: 除了代码托管之外，github还能做什么
image: https://img.yumeko.site/file/blog/GitHub.png
status: public
---
# Github Pages

使用github，部署一个网页，一般用于项目的**wiki**和**讲解**
## 在Settings中快速启动
在Settings中选择docs为部署文件夹，然后部署就可以得到网站的url，写在右边

![Projects.png](https://img.yumeko.site/file/articles/GithubUsage/Projects.png)
* 在Settings中设置**Source**，**分支**， **文件夹**

![Deployment.png](https://img.yumeko.site/file/articles/GithubUsage/Deployment.png)
* 等待一段时间就有url出现了，然后访问即可

* 在Actions中可以看见部署过程
# Github Actions
## 创建工作流文件
* 在项目的根文件夹创建`./github/workflow/ruff.yml`
## 创建项目限制文件
创建`pyproject.toml`

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```
## push到Github上
* 当代码进入github的时候，会触发工作流`workflow`，在github中使用`ruff`进行代码检查

```yml
name: ruff
on:
  push:
  pull_request:
permissions:
  contents: read
jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install ruff
        run: |
          python -m pip install --upgrade pip
          python -m pip install ruff
      - name: Ruff lint
        run: ruff check .
      - name: Ruff format (check)
        run: ruff format --check .
```
# 代码审查
如何使用ai来对项目进行代码审查
以`Codex`为例
* 在Codex的界面中，打开代码审查
![Codex.png](https://img.yumeko.site/file/articles/GithubUsage/Codex.png)
* 当`Pull Requests`从`draft`变成`Ready For Review`的时候，就会触发代码审查，`Codex`会回复你的`Pull Requests`，看看你的提交有什么问题
![CodeReview.png](https://img.yumeko.site/file/articles/GithubUsage/CodeReview.png)

# 代码静态检查CodeQL
`CodeQL`是`github`给的免费代码检查，看看你的代码有没有什么危险
* 打开**Security**界面，打开**Code Scanning Alerts**
![Security.png](https://img.yumeko.site/file/articles/GithubUsage/Security.png)
* 在Code Scanning部分点开CodeQL analysis，选择为Default就可以
![CodeQL.png](https://img.yumeko.site/file/articles/GithubUsage/CodeQL.png)
# Github OAuth
## Github 凭证申请
使用Github的账号，类似于Github登录的那种，可以授权给自己的网站登录
* 在个人的Developer Settings中，创建一个OAuth Apps
![OAuthApps.png](https://img.yumeko.site/file/articles/GithubUsage/OAuthApps.png)
* 在`OAuth APP`设置中，设置选择**主界面**的url和**login**界面的url（这里以`localhost`为例）
![Setting.png](https://img.yumeko.site/file/articles/GithubUsage/Setting.png)

> [!WARNING]
> 在生产环境要把Homepage改成你选择**登录之后**要到达的页面，callback URL就是**登录github账号，选择github账号**的页面

## Callback验证
![Web.png](https://img.yumeko.site/file/articles/GithubUsage/Web.png)