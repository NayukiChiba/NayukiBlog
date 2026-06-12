---
title: Git 与 GitHub CLI 完全指南——从入门到高阶工作流
date: 2026-01-17
category: 工程实践
tags:
  - git
  - github
  - 工具
description: 从安装配置到 Fork 工作流、stash 暂存、交互式 rebase、cherry-pick、bisect 排查的 Git 完全指南，含 GitHub CLI (gh) 认证机制详解
image: https://img.yumeko.site/file/blog/cover/1781272216423_gitUsage.webp
status: published
---

## 1. 安装与配置

### 1.1 安装 Git

**Windows:**

- 下载地址：https://git-scm.com/download/win
- 安装时建议勾选 "Git Bash Here" 选项

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install git
```

### 1.2 验证安装

```bash
git --version
# 输出示例：git version 2.42.0
```

### 1.3 初始配置（必须）

```bash
# 设置用户名和邮箱（每台电脑只需配置一次）
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"

# 查看当前配置
git config --list
```

### 1.4 可选但推荐的配置

```bash
# 设置默认分支名为 main
git config --global init.defaultBranch main

# 命令行输出颜色
git config --global color.ui auto

# 默认编辑器（VS Code 用户推荐）
git config --global core.editor "code --wait"

# 解决中文文件名显示问题
git config --global core.quotepath false

# Windows 换行符处理（Windows 用户必配）
git config --global core.autocrlf true
# Linux/Mac 用户改用：
# git config --global core.autocrlf input

# push 默认只推送当前分支（避免误推其他分支）
git config --global push.default simple

# rebase 时自动 stash + rebase + pop
git config --global rebase.autoStash true
```

> [!TIP] `push.default = simple`
> 不加此配置时 `git push` 会推送所有有同名远程分支的本地分支。设为 `simple` 后只推送当前分支，避免误推。

---

## 2. GitHub CLI (gh) —— 认证机制详解

GitHub CLI 不只是便利工具，它还解决了 Git 身份认证的根本问题。**理解它的认证机制，是理解整个工作流安全性的关键。**

### 2.1 背景：为什么需要 gh

2021 年 8 月起，GitHub 不再支持密码方式 `git push`。传统方案有两种：

| 方案 | 原理 | 缺点 |
|:--|:--|:--|
| **Personal Access Token** | 在 GitHub 网页生成 token，git push 时输入 token 作为密码 | token 需手动保管，每台机器要单独配置 credential.helper |
| **SSH Key** | 生成密钥对，公钥上传 GitHub，私钥本机保存 | 配置步骤多，新手容易出错 |

GitHub CLI 将以上两种方案**自动化**——登录 `gh auth login` 一步完成认证，后续 `git push` 无需再输密码。

### 2.2 安装 GitHub CLI

**Windows:**

```bash
winget install --id GitHub.cli
```

**Linux (Ubuntu/Debian):**

```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install gh -y
```

### 2.3 认证登录——两种方式

```bash
gh auth login
```

登录流程有四个选择步骤：

**步骤 1：选择平台**
```
? What account do you want to log into?
  GitHub.com        ← 绝大多数人选这个
  GitHub Enterprise Server
```

**步骤 2：选择认证协议**
```
? What is your preferred protocol for Git operations?
  HTTPS             ← 推荐：无需额外配置，gh 自动管理 token
  SSH               ← 需要提前配置 SSH Key 到 GitHub
```

> [!NOTE] HTTPS vs SSH
> 选 HTTPS：gh 将 token 存入系统凭据管理器（Windows Credential Manager / macOS Keychain / Linux secret-service），`git push` 时自动读取。
> 选 SSH：gh 使用本地 SSH 密钥认证，需先将公钥上传到 GitHub。

**步骤 3：选择登录方式**
```
? How would you like to authenticate GitHub CLI?
  Login with a web browser   ← 推荐：浏览器扫码/确认即可
  Paste an authentication token  ← 已有 token 时使用
```

**步骤 4（选 token 时）：粘贴 Personal Access Token**

从 GitHub Settings → Developer settings → Personal access tokens 生成 token，至少勾选 `repo`、`read:org`、`workflow` 权限。

![GhAuthFlow.png](https://img.yumeko.site/file/blog/articles/1781272427216_GhAuthFlow.webp)

### 2.4 认证存储在哪里

登录成功后，gh 做了两件事：

1. **OAuth token** 存入系统凭据管理器（`gh auth token` 可查看）
2. **Git credential helper** 自动配置为 `gh auth git-credential`，让 git 命令通过 gh 获取 token

```bash
# 查看 git 当前使用的凭据管理器
git config --global credential.helper
# 输出：!/path/to/gh auth git-credential

# 查看 gh 管理的 token（仅显示，不会泄露到日志）
gh auth token

# 查看登录状态
gh auth status
```

> [!WARNING] Token 安全
> `gh` 的 token 存储在系统级安全凭据管理器中，不会以明文形式出现在任何 git 配置文件中。**永远不要把 token 硬编码在脚本或 `.gitconfig` 里。**

### 2.5 账号切换与同步

```bash
# 查看登录状态
gh auth status

# 切换 GitHub 账号
gh auth switch --user NayukiChiba

# 登出
gh auth logout

# 将当前 gh 账号信息同步到 git 的 user.name / user.email
gh auth setup-git
```

执行 `gh auth setup-git` 后，你的 `~/.gitconfig` 会被写入：

```ini
[user]
    name = your-name
    email = your-github-email@example.com

[credential]
    helper = !gh auth git-credential
```

这意味着**只要 gh 登录了哪个账号，git push 就以那个账号的身份推送**——两个工具共享同一套认证凭据。

> [!TIP] 多账号场景
> 如果你有多个 GitHub 账号（公司 + 个人），可以用 `gh auth switch` 切换。每次切换后 git 的提交者身份也自动跟随。如果需要同时使用多个账号，建议按目录配置不同的 `user.name`，参见 [[Engineering/GitUsage#13.2-一台机器配置多个-github-账号|多账号配置]]。

### 2.6 仓库操作

```bash
# 创建新仓库
gh repo create my-project --public           # 公开仓库
gh repo create my-project --private          # 私有仓库
gh repo create my-project --public --clone   # 创建并克隆

# 克隆仓库
gh repo clone owner/repo
gh repo clone owner/repo my-folder           # 指定目录

# Fork 仓库
gh repo fork owner/repo                      # Fork 到自己账户
gh repo fork owner/repo --clone              # Fork 并克隆

# 查看仓库
gh repo view                                 # 当前仓库信息
gh repo view --web                           # 浏览器打开
gh repo list                                 # 列出自己的仓库
```

### 2.7 Pull Request 操作

```bash
gh pr create                                  # 交互式创建
gh pr create --title "标题" --body "描述"
gh pr create --base main --head feature       # 指定源和目标分支
gh pr create --draft                          # 草稿 PR
gh pr create --web                            # 浏览器中填写

gh pr list                                    # 列出 PR
gh pr list --author @me --state open
gh pr view 123                                # 查看 PR #123
gh pr checkout 123                            # 检出到本地

gh pr merge 123                               # 合并
gh pr merge 123 --squash                      # Squash 合并
gh pr merge 123 --rebase                      # Rebase 合并

gh pr review 123 --approve                    # 批准
gh pr review 123 --request-changes            # 请求修改
gh pr review 123 --comment -b "评论内容"
```

### 2.8 Issue 与 Actions

```bash
gh issue create --title "标题" --body "描述"
gh issue create --label "bug"
gh issue list --assignee @me
gh issue view 123
gh issue close 123

gh run list                                   # Actions 运行记录
gh run view 123456
gh run watch                                  # 实时监控

gh search repos "关键词"                      # 搜索仓库
gh search issues "关键词"                     # 搜索 Issue
```

> [!NOTE] 网页操作 vs CLI
> PR 的创建和合并推荐在网页完成（所见即所得、支持模板），`gh pr create --web` 可以直接打开浏览器。`gh` 的真正优势在于 `gh pr checkout`（检出任一 PR 到本地测试）、`gh auth` 认证管理、以及 CI/CD 脚本中的 API 调用。

---

## 3. 创建仓库

### 3.1 初始化新仓库

```bash
cd /path/to/your/project
git init
```

### 3.2 克隆已有仓库

```bash
# HTTPS 克隆（推荐：gh 自动管理认证）
git clone https://github.com/username/repository.git

# SSH 克隆（需要配置 SSH Key）
git clone git@github.com:username/repository.git

# 克隆到指定目录
git clone https://github.com/username/repository.git my-folder

# 浅克隆（只取最新版本，加快速度）
git clone --depth 1 https://github.com/username/repository.git

# 克隆指定分支
git clone -b develop https://github.com/username/repository.git
```

---

## 4. 基本工作流程

Git 的三个工作区域：

![ThreeLayers.png](https://img.yumeko.site/file/blog/articles/1781272766794_ThreeLayers.webp)

### 4.1 查看状态

```bash
git status                  # 完整状态
git status -s               # 简洁模式
```

**状态标识说明：**

|  标识  |   含义    |
| :--: | :-----: |
| `??` | 未跟踪的新文件 |
| `A`  | 新添加到暂存区 |
| `M`  |   已修改   |
| `D`  |   已删除   |
| `R`  |  已重命名   |

### 4.2 添加文件到暂存区

```bash
git add filename                    # 单个文件
git add file1 file2                 # 多个文件
git add .                           # 所有修改和新文件
git add -A                          # 所有修改（包括删除）
git add -p                          # 交互式选择（逐块暂存）
```

> [!TIP] `git add -p` 交互式暂存
> 当你在一个文件里同时改了多个不相关的功能时，`git add -p` 可以逐块（hunk）选择要暂存的内容，让你把一个大改动拆成多个语义清晰的提交。

### 4.3 提交更改

```bash
git commit -m "提交说明"
git commit -am "提交说明"            # 跳过 add（仅已跟踪文件）
git commit --amend -m "新信息"       # 修改上一次提交信息
git commit --amend --no-edit         # 追加文件到上一次提交
```

### 4.4 推送到远程

```bash
git push -u origin main             # 首次推送（设置上游追踪）
git push                            # 后续推送
git push --force-with-lease         # 安全强制推送
```

> [!DANGER] `--force` vs `--force-with-lease`
> `git push --force` 直接覆盖远程分支，如果别人在你上次 fetch 之后推送了新提交，会被你覆盖。
> `git push --force-with-lease` 会先检查远程分支是否被你之外的人改动过，改动过则拒绝推送。**永远优先用 `--force-with-lease`。**

### 4.5 拉取远程更新

```bash
git pull                            # fetch + merge
git pull --rebase                   # fetch + rebase（推荐）
git fetch                           # 仅下载，不合并
git fetch origin
git diff origin/main                # 查看远程差异后再决定
```

---

## 5. 分支管理

![MergeStrategies.png](https://img.yumeko.site/file/blog/articles/1781272428521_MergeStrategies.webp)

### 5.1 查看分支

```bash
git branch                          # 本地分支
git branch -a                       # 所有分支（含远程）
git branch -v                       # 分支详情（最后提交）
git branch -vv                      # 分支详情（含追踪关系）
```

### 5.2 创建与切换

```bash
git checkout -b feature-login       # 创建并切换（推荐）
git switch -c feature-login         # 新版命令（Git 2.23+）
git checkout feature-login          # 仅切换
git switch feature-login            # 新版切换命令
```

### 5.3 删除分支

```bash
git branch -d feature-login         # 删除已合并分支
git branch -D feature-login         # 强制删除（未合并）
git push origin --delete feature-login  # 删除远程分支
git remote prune origin             # 清理本地缓存中已删除的远程分支引用
```

### 5.4 分支重命名

```bash
git branch -m new-name              # 重命名当前分支
git branch -m old-name new-name     # 重命名指定分支
```

### 5.5 分支合并

```bash
git checkout main
git merge feature-login             # 快进合并（Fast-forward）
git merge --no-ff feature-login     # 保留分支历史
git merge --abort                   # 中止合并（冲突时）
```

### 5.6 Merge 的三种策略

**Fast-forward merge**：feature 分支直接快进到 main，不产生新 commit。适合简单的线性开发。

**`--no-ff` merge**：即使可以快进，也创建一个 merge commit，保留 feature 分支的独立历史。适合多人协作的功能分支。

**Rebase**：将 feature 分支的提交"搬到"main 最新提交之后，形成线性历史。适合个人分支整理。

| 策略 | 历史外观 | 适用场景 |
|:--|:--|:--|
| Fast-forward | 线性，看不出分支 | 个人小改动 |
| `--no-ff` | 保留分支痕迹 | 团队功能开发 |
| Rebase | 线性，干净 | 提交前整理历史 |

---

## 6. 远程仓库：Origin 与 Upstream

> [!ABSTRACT] 核心概念
> - **origin**：你自己的远程仓库（Fork 后的仓库）
> - **upstream**：原始仓库（你 Fork 的来源）

![ForkWorkflow.png](https://img.yumeko.site/file/blog/articles/1781272500366_ForkWorkflow.webp)

### 6.1 典型使用场景

| 操作 | 使用哪个远程 | 命令示例 |
|:--|:--|:--|
| 推送自己的修改 | origin | `git push origin main` |
| 获取原始仓库更新 | upstream | `git fetch upstream` |
| 提交 PR | origin → upstream | `gh pr create` |

### 6.2 设置 Origin 和 Upstream

```bash
# 克隆时自动设置 origin
git clone https://github.com/you/repo.git

# 查看当前远程配置
git remote -v

# 添加 upstream（指向原始仓库）
git remote add upstream https://github.com/original-owner/repo.git

# 验证
git remote -v
# origin    https://github.com/you/repo.git (fetch)
# origin    https://github.com/you/repo.git (push)
# upstream  https://github.com/original-owner/repo.git (fetch)
# upstream  https://github.com/original-owner/repo.git (push)
```

### 6.3 管理远程仓库

```bash
git remote set-url origin https://github.com/you/new-repo.git   # 修改 URL
git remote remove upstream                                       # 删除远程
git remote rename origin old-origin                              # 重命名
git remote show origin                                           # 查看详情
```

### 6.4 从 Upstream 同步更新

```bash
# 获取 upstream 最新内容
git fetch upstream

# 切换到本地 main
git checkout main

# 合并 upstream 的更改
git merge upstream/main
# 或使用 rebase 保持线性历史：
git rebase upstream/main

# 推送到自己的 origin
git push origin main
```

> [!WARNING] 强制覆盖（仅限紧急情况）
> ```bash
> git fetch upstream
> git reset --hard upstream/main    # 丢弃本地所有修改
> git push -f origin main           # 强制推送
> ```
> 这个操作会**丢失本地所有未推送的提交**。只在你的 Fork 严重偏离上游、手动合并太麻烦时使用。

### 6.5 处理合并冲突

```bash
# 冲突出现后
git status                          # 查看冲突文件

# 编辑冲突文件，删除标记：
# <<<<<<< HEAD
# 你的更改
# =======
# 远程的更改
# >>>>>>> upstream/main

git add conflicted_file             # 标记已解决
git commit -m "解决合并冲突"         # 完成合并

# 或者放弃合并
git merge --abort
```

---

## 7. Stash —— 暂存工作区（独立详解）

`git stash` 是日常开发中**使用频率最高的"紧急中断"工具**。当你在一个分支上写到一半，突然需要切到另一个分支处理紧急问题，stash 能让你快速保存当前进度并清空工作区。

![StashStack.png](https://img.yumeko.site/file/blog/articles/1781272528403_StashStack.webp)

### 7.1 核心概念

Stash 本质上是一个**临时的、基于栈的提交**。它不在任何分支上，存放在 `.git/refs/stash` 中，可以在任何时候恢复到任意分支。

### 7.2 基本操作

```bash
# 暂存当前所有修改（已跟踪的文件）
git stash

# 暂存并添加说明（强烈推荐：一个月后你还看得懂）
git stash push -m "登录页表单校验写到一半"

# 查看暂存列表
git stash list
# stash@{0}: On feature-login: 登录页表单校验写到一半
# stash@{1}: On main: 修复首页样式bug

# 恢复最近的暂存并从列表中删除
git stash pop

# 恢复最近的暂存但保留在列表中
git stash apply

# 恢复指定的暂存
git stash pop stash@{2}
git stash apply stash@{1}

# 删除指定暂存
git stash drop stash@{0}

# 清空所有暂存
git stash clear
```

### 7.3 进阶用法

**暂存未跟踪的文件（新建但未 git add 的文件）**

```bash
git stash -u
# 或
git stash --include-untracked
```

**暂存所有文件（包括被 .gitignore 忽略的）**

```bash
git stash -a
# 或
git stash --all
```

**只暂存特定文件**

```bash
git stash push -m "只暂存 API 文件" src/api/user.ts src/api/auth.ts
```

**交互式暂存（逐块选择）**

```bash
git stash push -p
# 类似 git add -p，逐块确认是否存入 stash
```

### 7.4 查看 Stash 内容

```bash
# 查看最近的 stash 改了哪些文件
git stash show

# 查看详细 diff
git stash show -p

# 查看指定 stash 的内容
git stash show -p stash@{1}
```

### 7.5 从 Stash 创建分支

当你 stash 了一堆修改，后来发现应该在一个新分支上开发：

```bash
git stash branch feature-new-feature
# 等价于：
# 1. git checkout -b feature-new-feature <stash 时的 commit>
# 2. git stash pop
# 一步完成：创建分支 + 恢复 stash
```

这是 stash 最被低估的功能——当你发现"我应该开个分支写这个"时，一条命令搞定。

### 7.6 Stash 冲突处理

当 `git stash pop` 遇到冲突时：

```bash
# pop 失败，提示冲突
git stash pop
# Auto-merging src/app.ts
# CONFLICT (content): Merge conflict in src/app.ts
# The stash entry is kept in case you need it again.

# 手动解决冲突后
git add .
git stash drop          # 手动删除 stash（因为 pop 失败时不会自动删除）
```

> [!TIP] pop 失败的 stash 不会被删除
> 如果 `git stash pop` 过程中发生冲突，stash **不会被自动删除**。解决冲突后需要手动 `git stash drop`。这实际上是一个安全机制——防止冲突导致修改丢失。

### 7.7 实际场景

**场景 1：紧急 bug 中断开发**

```bash
# 正在 feature 分支开发
git stash push -m "用户权限模块开发中"

# 切到 main 修 bug
git checkout main
git checkout -b hotfix-crash
# ... 修复 bug ...
git commit -am "修复启动崩溃"
git push -u origin hotfix-crash

# 回到 feature 继续
git checkout feature
git stash pop
```

**场景 2：改错分支了**

```bash
# 在 main 上写了一阵才发现应该在 feature 分支
git stash push -m "新功能的数据库模型"
git checkout -b feature-db-model
git stash pop
# 继续开发
```

**场景 3：合并不干净，先暂存再拉取**

```bash
# 本地有未提交的修改，pull 时冲突
git stash push -m "本地临时修改"
git pull --rebase
git stash pop
# 如有冲突，解决后继续
```

### 7.8 Stash 速查表

| 操作 | 命令 |
|:--|:--|
| 暂存所有修改 | `git stash` |
| 暂存 + 说明 | `git stash push -m "描述"` |
| 暂存含未跟踪文件 | `git stash -u` |
| 暂存特定文件 | `git stash push file1 file2` |
| 交互式暂存 | `git stash push -p` |
| 查看列表 | `git stash list` |
| 恢复并删除 | `git stash pop` |
| 恢复不删除 | `git stash apply` |
| 恢复指定 | `git stash pop stash@{2}` |
| 查看内容 | `git stash show -p stash@{0}` |
| 创建分支+恢复 | `git stash branch new-branch` |
| 删除 | `git stash drop stash@{0}` |
| 清空全部 | `git stash clear` |

---

## 8. 撤销与回退详解

Git 的撤销体系是最容易出错的部分。理解三个区域（工作区/暂存区/提交历史）后，每种撤销操作都有对应的命令。

![UndoDecisionTree.png](https://img.yumeko.site/file/blog/articles/1781272647959_UndoDecisionTree.webp)

### 8.1 命令对比总览

| 命令 | 作用范围 | 改变历史 | 安全性 | 典型场景 |
|:--|:--|:--|:--|:--|
| `git restore <file>` | 工作区 | 否 | 安全 | 丢弃文件修改 |
| `git restore --staged <file>` | 暂存区 | 否 | 安全 | 取消暂存 |
| `git reset --soft` | 提交历史 | 是 | 安全 | 重新组织提交 |
| `git reset --mixed` | 提交历史+暂存区 | 是 | 安全 | 取消提交和暂存 |
| `git reset --hard` | 全部 | 是 | 危险 | 完全回退 |
| `git revert` | 创建新提交 | 否 | 安全 | 撤销已推送的提交 |

### 8.2 restore：撤销工作区和暂存区

```bash
# 撤销工作区的修改（文件回到上次 add/commit 的状态）
git restore filename
git restore .                           # 所有文件
git restore src/                        # 指定目录

# 取消暂存（从暂存区移出，工作区修改保留）
git restore --staged filename
git restore --staged .

# 同时撤销工作区和暂存区
git restore --staged --worktree filename

# 恢复到指定提交的版本
git restore --source=HEAD~2 filename
git restore --source=abc1234 filename
```

### 8.3 reset：回退提交历史

> [!DANGER] 核心原则
> **已推送的提交不要用 reset，用 revert。** reset 改写历史，会影响其他协作者。

![ResetModes.png](https://img.yumeko.site/file/blog/articles/1781272806858_ResetModes.webp)

**`--soft`：只移动 HEAD，提交的修改回到暂存区。**

```bash
# 合并最近 3 个提交为一个
git reset --soft HEAD~3
git commit -m "合并后的提交信息"
```

**`--mixed`（默认）：移动 HEAD + 清空暂存区，修改回到工作区。**

```bash
# 取消最近一次提交和暂存
git reset HEAD~1
# 现在需要重新选择文件 add + commit
```

**`--hard`：完全回退，丢弃所有修改。**

```bash
# 完全放弃最近 2 次提交
git reset --hard HEAD~2

# 回退到远程分支的状态
git fetch origin
git reset --hard origin/main
```

| 模式 | 提交历史 | 暂存区 | 工作区 | 安全性 |
|:--|:--|:--|:--|:--|
| `--soft` | 回退 | 保留修改 | 保留修改 | 安全 |
| `--mixed` | 回退 | 清空 | 保留修改 | 安全 |
| `--hard` | 回退 | 清空 | 清空 | 危险 |

### 8.4 revert：安全撤销已推送的提交

`git revert` 不删除历史提交，而是创建一个**反向提交**来抵消目标提交的修改。

```bash
git revert HEAD                             # 撤销最近一次提交
git revert abc1234                          # 撤销指定提交
git revert -n abc1234                       # 撤销但不自动提交
git revert HEAD~2..HEAD                     # 撤销最近 3 次提交（每个生成一个 revert）
git revert --no-commit HEAD~2..HEAD         # 撤销多个提交合并为一个
git commit -m "Revert: 撤销最近3次提交"

# 解决冲突后继续
git add .
git revert --continue

# 放弃 revert
git revert --abort
```

### 8.5 reflog：终极后悔药

`git reflog` 记录所有 HEAD 变动，即使 `reset --hard` 误删的提交也能找回。

```bash
git reflog
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: 添加新功能（被误删的提交！）
# ghi9012 HEAD@{2}: commit: 修复 bug

# 恢复到被误删的提交
git reset --hard def5678

# 或创建分支恢复
git checkout -b recovery-branch def5678
```

> [!NOTE] reflog 保留期限
> reflog 只保存本地记录，默认保留 90 天（`gc.reflogExpire` 配置）。推送过的提交有更长的保留期。

### 8.6 实际场景速查表

| 我想要... | 使用命令 |
|:--|:--|
| 撤销工作区修改 | `git restore <file>` |
| 取消已暂存的文件 | `git restore --staged <file>` |
| 修改最近一次提交信息 | `git commit --amend -m "新信息"` |
| 追加文件到上一次提交 | `git add <file>` + `git commit --amend --no-edit` |
| 撤销未推送的 N 次提交（保留修改） | `git reset --soft HEAD~N` |
| 撤销未推送的 N 次提交（丢弃修改） | `git reset --hard HEAD~N` |
| 撤销已推送的某次提交 | `git revert <commit-id>` |
| 完全与远程同步 | `git fetch origin` + `git reset --hard origin/main` |
| 恢复被误删的提交 | `git reflog` + `git checkout <commit-id>` |

---

## 9. 查看历史与状态

### 9.1 提交历史

```bash
git log                                 # 完整历史
git log --oneline                       # 简洁模式
git log --oneline --graph --all         # 图形化分支历史
git log --stat                          # 文件变更统计
git log -p                              # 显示具体改了什么
git log -n 5                            # 最近 5 条
git log --author="用户名"               # 按作者筛选
git log --grep="关键词"                 # 搜索提交信息
git log --since="2026-01-01"            # 时间范围
git log --follow filename               # 跟踪文件重命名历史
git log main..feature                   # feature 有而 main 没有的提交
```

### 9.2 查看差异

```bash
git diff                                # 工作区 vs 暂存区
git diff --staged                       # 暂存区 vs 最新提交
git diff main..feature                  # 两个分支的差异
git diff abc1234 def5678                # 两次提交的差异
git diff --name-only                    # 只显示文件名
git diff --stat                         # 显示变更统计
```

### 9.3 查看特定提交

```bash
git show abc1234                        # 查看提交详情
git show abc1234:path/to/file           # 查看提交中的特定文件
git show --stat abc1234                 # 提交的变更统计
```

---

## 10. Fork 工作流完整流程

![ForkFullFlow.png](https://img.yumeko.site/file/blog/articles/1781272705787_ForkFullFlow.webp)

### 10.1 Fork 并设置本地环境

1. 在 GitHub 网页上 Fork 原仓库
![HowToFork.png](https://img.yumeko.site/file/blog/articles/1781273223900_HowToFork.webp)
2. 克隆自己的 Fork 到本地
![ForkedRepo.png](https://img.yumeko.site/file/blog/articles/1781273224593_ForkedRepo.webp)

```bash
git clone https://github.com/your-username/repo.git
cd repo

# 添加 upstream
git remote add upstream https://github.com/original-owner/repo.git
git remote -v
# origin    https://github.com/your-username/repo.git (fetch)
# origin    https://github.com/your-username/repo.git (push)
# upstream  https://github.com/original-owner/repo.git (fetch)
# upstream  https://github.com/original-owner/repo.git (push)
```

### 10.2 开发并提交 PR

```bash
# 创建功能分支
git checkout -b feature/new-feature

# 开发、提交
git add .
git commit -m "feat: 添加新功能"
git push -u origin feature/new-feature
```

### 10.3 提交 Pull Request

**PR 界面关键字段解读：**

- **base repository**（左侧）：要合并**进**的仓库，应该是 upstream（原仓库）
- **head repository**（右侧）：包含你修改的仓库，是你的 origin
- 如果左右都是你自己的仓库，那就和本地 merge 一样——不会到达上游
![PushToOrigin.png](https://img.yumeko.site/file/blog/articles/1781273221200_PushToOrigin.webp)
![CompareBranch.png](https://img.yumeko.site/file/blog/articles/1781273226998_CompareBranch.webp)

### 10.4 上游看到的 PR

![PRMergeTypes.png](https://img.yumeko.site/file/blog/articles/1781272736927_PRMergeTypes.webp)

三种合并方式：

| 方式                        | 效果                | 适用                  |
| :------------------------ | :---------------- | :------------------ |
| **Create a merge commit** | 保留所有原始 commit     | 提交记录需要完整追溯          |
| **Squash and merge**      | 所有 commit 压缩为一个   | 一个 PR = 一个干净 commit |
| **Rebase and merge**      | commit 线性附加到目标分支后 | 需要保持线性历史            |
![MergePR.png](https://img.yumeko.site/file/blog/articles/1781273221353_MergePR.webp)
### 10.5 PR 合并后的清理

```bash
# 同步 upstream 更新到本地 main
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# 删除已合并的功能分支
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

---

## 11. 高级技巧

### 11.1 交互式 Rebase（整理提交历史）

```bash
git rebase -i HEAD~3
```

执行后会打开编辑器，列出最近 3 次提交：

```
pick abc1234 feat: 添加登录
pick def5678 fix: 修改密码校验
pick ghi9012 chore: 调整格式
```

![InteractiveRebase.png](https://img.yumeko.site/file/blog/articles/1781272829231_InteractiveRebase.webp)

每行前面的命令可以修改：

| 命令 | 作用 |
|:--|:--|
| `pick` | 保留该提交 |
| `reword` | 保留但修改提交信息 |
| `edit` | 暂停 rebase 让你修改该提交的内容 |
| `squash` | 合并到上一个提交，保留提交信息 |
| `fixup` | 合并到上一个提交，丢弃提交信息 |
| `drop` | 删除该提交 |

```bash
# 示例：将后两个 fix 合并到第一个 feat
pick abc1234 feat: 添加登录
fixup def5678 fix: 修改密码校验
fixup ghi9012 chore: 调整格式
# 保存后，三个提交合并为一个
```

**解决 rebase 冲突：**

```bash
git rebase main                         # 将当前分支变基到 main

# 如有冲突：
git status                              # 查看冲突文件
# ... 手动解决冲突 ...
git add .
git rebase --continue                   # 继续

git rebase --abort                      # 放弃 rebase
git rebase --skip                       # 跳过当前提交
```

### 11.2 Cherry-pick —— 精确搬运提交

当你只需要另一个分支上的**某几个**提交（而不是整个分支）时：

```bash
# 将指定提交"复制"到当前分支
git cherry-pick abc1234

# 复制多个提交
git cherry-pick abc1234 def5678

# 复制一个范围的提交
git cherry-pick abc1234..def5678

# 复制但不自动提交（可以修改后再提交）
git cherry-pick -n abc1234

# 冲突时：
git cherry-pick --continue             # 解决冲突后继续
git cherry-pick --abort                # 放弃
git cherry-pick --skip                 # 跳过当前提交
```

> [!EXAMPLE] Cherry-pick 典型场景
> 1. 从实验分支搬一个工具函数到主分支
> 2. hotfix 分支修了一个 bug，需要同步到 release 分支
> 3. 错误的把 commit 提交到了 main，需要搬到 feature 分支

### 11.3 Bisect —— 二分法定位 Bug 引入点

当你知道某个版本有 bug、更早的版本没有，但不知道是哪个提交引入的：

```bash
# 启动二分查找
git bisect start

# 标记当前版本有问题
git bisect bad HEAD

# 标记某个旧版本没问题（比如 v1.0 标签）
git bisect good v1.0

# Git 会自动检出中间的某个提交
# 测试这个版本是否有 bug，然后告诉 Git：

git bisect good              # 这个版本没问题，bug 在更后面引入
# 或
git bisect bad               # 这个版本有 bug，bug 在更前面引入

# 重复直到 Git 找到第一个引入 bug 的提交
# bisect 结束后：
git bisect reset              # 回到原来的分支
```

> [!TIP] 自动化 bisect
> 如果你有测试脚本能自动判断是否存在 bug：
> ```bash
> git bisect start HEAD v1.0
> git bisect run python test_bug.py
> # Git 会自动二分查找，运行脚本判断每个版本
> ```

### 11.4 Worktree —— 同时操作多个分支

当不想频繁 stash/pop 时，用 worktree 在不同目录同时检出多个分支：

```bash
# 在隔壁目录检出另一个分支
git worktree add ../project-hotfix hotfix-urgent

# 现在你有两个工作目录：
# /project         → main 分支
# /project-hotfix  → hotfix-urgent 分支

# 查看所有 worktree
git worktree list

# 删除 worktree
git worktree remove ../project-hotfix
```

> [!NOTE] Worktree vs Stash
> Worktree 适合**长时间并行开发**（一边开发一边修 bug），Stash 适合**短时间切换**（几分钟内切回）。Worktree 省去了 stash/pop 的开销，但会占用额外的磁盘空间。

### 11.5 标签管理

```bash
git tag v1.0.0                          # 轻量标签
git tag -a v1.0.0 -m "正式发布 v1.0.0"  # 附注标签（推荐）
git tag -a v1.0.0 abc1234 -m "..."      # 给历史提交打标签
git push origin v1.0.0                  # 推送单个标签
git push origin --tags                  # 推送所有标签
git tag -d v1.0.0                       # 删除本地标签
git push origin --delete v1.0.0         # 删除远程标签
```

---

## 12. .gitignore 与 Git Hooks

### 12.1 .gitignore 最佳实践

```bash
touch .gitignore
```

推荐模板：

```
# ===== 操作系统 =====
.DS_Store
Thumbs.db
desktop.ini

# ===== IDE 与编辑器 =====
.idea/
.vscode/
*.swp
*.swo
*~

# ===== Python =====
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# ===== Node.js =====
node_modules/
.next/
dist/

# ===== LaTeX =====
build/
*.aux
*.log
*.out

# ===== 环境与密钥 =====
.env
.env.local
*.pem
*.key
secrets/

# ===== 日志与临时文件 =====
*.log
*.tmp
.cache/
```

> [!TIP] 已跟踪文件的忽略
> `.gitignore` 只对未跟踪的文件生效。如果文件已经被 `git add` 过，需要先：
> ```bash
> git rm --cached filename    # 停止跟踪但保留本地文件
> echo "filename" >> .gitignore
> git commit -m "chore: 从跟踪中移除 filename"
> ```

### 12.2 Git Hooks 简介

Git Hooks 是在特定事件触发时自动执行的脚本，存放在 `.git/hooks/` 目录下。

| Hook | 触发时机 | 典型用途 |
|:--|:--|:--|
| `pre-commit` | `git commit` 之前 | 代码检查、格式化 |
| `commit-msg` | 编辑完提交信息后 | 校验提交信息格式 |
| `pre-push` | `git push` 之前 | 运行测试 |
| `post-merge` | `git merge` 之后 | 依赖更新提醒 |

实际项目中推荐使用 [pre-commit](https://pre-commit.com/) 框架管理 hooks，详见 [[Engineering/PreCommitRuff|Pre-Commit Ruff 配置]]。

---

## 13. 常见问题解决

### 13.1 中文文件名显示乱码

```bash
git config --global core.quotepath false
```

### 13.2 一台机器配置多个 GitHub 账号

在 `~/.gitconfig` 中使用条件引入：

```ini
# ~/.gitconfig
[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work
[includeIf "gitdir:~/personal/"]
    path = ~/.gitconfig-personal
```

```ini
# ~/.gitconfig-work
[user]
    name = Work Name
    email = work@company.com
```

```ini
# ~/.gitconfig-personal
[user]
    name = Personal Name
    email = personal@gmail.com
```

### 13.3 修改历史提交的作者信息

```bash
# 修改最近一次提交的作者
git commit --amend --author="New Name <newemail@example.com>"

# 修改最近 N 次提交（交互式 rebase 后逐个改）
git rebase -i HEAD~5
# 将需要修改的提交从 pick 改为 edit
# 每个暂停时执行：
git commit --amend --author="..." --no-edit
git rebase --continue
```

### 13.4 恢复删除的分支

```bash
git reflog                              # 找到删除前的 commit ID
git checkout -b recovered-branch abc1234
```

### 13.5 误提交大文件后清理

```bash
# 从最近一次提交中移除
git rm --cached large-file.zip
git commit --amend

# 从整个历史中移除（使用 git-filter-repo）
pip install git-filter-repo
git filter-repo --path large-file.zip --invert-paths
```

### 13.6 清理未跟踪文件

```bash
git clean -n                            # 预览将被删除的文件
git clean -f                            # 删除未跟踪文件
git clean -fd                           # 删除未跟踪文件和目录
```

---

## 14. 快速参考卡片

### Git 基础命令

| 操作 | 命令 |
|:--|:--|
| 初始化仓库 | `git init` |
| 克隆仓库 | `git clone <url>` |
| 查看状态 | `git status` |
| 添加文件 | `git add .` |
| 交互式暂存 | `git add -p` |
| 提交更改 | `git commit -m "message"` |
| 推送代码 | `git push` |
| 安全强制推送 | `git push --force-with-lease` |
| 拉取更新 | `git pull --rebase` |
| 创建分支 | `git checkout -b <branch>` |
| 切换分支 | `git checkout <branch>` |
| 合并分支 | `git merge <branch>` |
| 查看历史 | `git log --oneline --graph --all` |

### 撤销与回退

| 场景 | 命令 |
|:--|:--|
| 撤销工作区修改 | `git restore <file>` |
| 取消暂存 | `git restore --staged <file>` |
| 回退未推送提交（保留修改） | `git reset --soft HEAD~1` |
| 回退未推送提交（丢弃修改） | `git reset --hard HEAD~1` |
| 撤销已推送提交 | `git revert <commit>` |
| 恢复误删提交 | `git reflog` + `git checkout <commit>` |

### Stash

| 操作 | 命令 |
|:--|:--|
| 暂存 + 说明 | `git stash push -m "描述"` |
| 暂存含未跟踪文件 | `git stash -u` |
| 查看列表 | `git stash list` |
| 恢复并删除 | `git stash pop` |
| 创建分支 + 恢复 | `git stash branch new-branch` |

### GitHub CLI

| 操作 | 命令 |
|:--|:--|
| 登录 | `gh auth login` |
| 查看状态 | `gh auth status` |
| 同步 git 配置 | `gh auth setup-git` |
| Fork 并克隆 | `gh repo fork owner/repo --clone` |
| 创建 PR | `gh pr create --web` |
| 检出 PR | `gh pr checkout 123` |
| 查看 PR 列表 | `gh pr list` |

### 高级技巧

| 操作 | 命令 |
|:--|:--|
| 交互式 Rebase | `git rebase -i HEAD~3` |
| 搬运提交 | `git cherry-pick abc1234` |
| 二分定位 Bug | `git bisect start` |
| 多分支并行 | `git worktree add ../path branch` |
| 打标签 | `git tag -a v1.0.0 -m "..."` |

---
