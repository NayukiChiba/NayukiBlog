---
title: Git 与 GitHub CLI 使用全流程指南
date: 2026-01-17
category: 技术
tags:
  - git
  - 工具
description: 本文档涵盖 Git 和 GitHub CLI (gh) 从安装到日常使用的完整流程，包含 Fork 工作流中 upstream 与 origin 的管理。
image: https://img.yumeko.site/file/blog/git.jpeg
status: public
---

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

### 1.4 可选配置

```bash
# 设置默认分支名为 main（推荐）
git config --global init.defaultBranch main

# 设置命令行输出颜色
git config --global color.ui auto

# 设置默认编辑器（可选 vim, nano, code 等）
git config --global core.editor "vim"

# 解决中文文件名显示问题
git config --global core.quotepath false
```

---

## 2. GitHub CLI (gh) 使用

> 🚀 GitHub CLI 是 GitHub 官方命令行工具，可以在终端中完成 GitHub 上的大部分操作。

### 2.1 安装 GitHub CLI

**Windows:**

```bash
# 使用 winget 安装
winget install --id GitHub.cli
```

![InstallGithubCLI](https://img.yumeko.site/file/articles/GitUsage/InstallGithubCLI.png)

**Linux (Ubuntu/Debian):**

```bash
# 添加官方源
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

# 安装
sudo apt update
sudo apt install gh
```

### 2.2 登录认证

```bash
# 交互式登录（推荐）
gh auth login

# 登录流程：
# 1. 选择 GitHub.com 或 GitHub Enterprise
# 2. 选择认证方式：浏览器 或 Token
# 3. 选择默认协议：HTTPS 或 SSH
# 4. 完成浏览器授权
```

![LoginGithub](https://img.yumeko.site/file/articles/GitUsage/LoginGithub.png)

### 2.3 查看github登录状态

```bash
# 查看登录状态
gh auth status

# 退出登录
gh auth logout

# 切换账号
gh auth switch --user NayukiChiba
```

![GithubStatus](https://img.yumeko.site/file/articles/GitUsage/GithubStatus.png)

### 2.4 设置与git同步账号

只要在github-cli中切换账号，当前的git就会自动切换账号，提交者就会变成github当前登录的账号

```bash
# 使用 gh 设置 Git 配置（会自动使用登录账户的信息）
gh auth setup-git
```

### 2.5 不常见的操作

> 虽然很方便，但是还是建议在github网页中操作

#### 仓库操作

```bash
# 创建新仓库
gh repo create my-project --public           # 创建公开仓库
gh repo create my-project --private          # 创建私有仓库
gh repo create my-project --public --clone   # 创建并克隆到本地

# 克隆仓库
gh repo clone owner/repo                     # 克隆指定仓库
gh repo clone owner/repo my-folder           # 克隆到指定文件夹

# Fork 仓库
gh repo fork owner/repo                      # Fork 到自己账户
gh repo fork owner/repo --clone              # Fork 并克隆到本地

# 查看仓库信息
gh repo view                                 # 查看当前仓库
gh repo view owner/repo                      # 查看指定仓库
gh repo view --web                           # 在浏览器中打开

# 列出仓库
gh repo list                                 # 列出自己的仓库
gh repo list owner                           # 列出指定用户的仓库
```

#### Pull Request 操作

```bash
# 创建 PR
gh pr create                                 # 交互式创建 PR
gh pr create --title "标题" --body "描述"     # 指定标题和描述
gh pr create --base main --head feature      # 指定基础分支和源分支
gh pr create --draft                         # 创建草稿 PR

# 查看 PR
gh pr list                                   # 列出所有 PR
gh pr list --state open                      # 列出打开的 PR
gh pr list --author @me                      # 列出自己创建的 PR
gh pr view 123                               # 查看指定 PR
gh pr view --web                             # 在浏览器中打开

# 检出 PR 到本地
gh pr checkout 123                           # 检出 PR #123 到本地分支

# 合并 PR
gh pr merge 123                              # 合并 PR
gh pr merge 123 --squash                     # Squash 合并
gh pr merge 123 --rebase                     # Rebase 合并
gh pr merge 123 --delete-branch              # 合并后删除分支

# 审核 PR
gh pr review 123 --approve                   # 批准 PR
gh pr review 123 --request-changes           # 请求更改
gh pr review 123 --comment -b "评论内容"     # 添加评论
```

#### Issue 操作

```bash
# 创建 Issue
gh issue create                              # 交互式创建
gh issue create --title "标题" --body "描述"  # 指定标题和描述
gh issue create --label "bug"                # 添加标签

# 查看 Issue
gh issue list                                # 列出所有 Issue
gh issue list --assignee @me                 # 列出分配给自己的
gh issue view 123                            # 查看指定 Issue
gh issue view 123 --web                      # 在浏览器中打开

# 关闭/重开 Issue
gh issue close 123                           # 关闭 Issue
gh issue reopen 123                          # 重新打开 Issue

# 编辑 Issue
gh issue edit 123 --add-label "enhancement"  # 添加标签
gh issue edit 123 --add-assignee @me         # 分配给自己
```

#### 其他命令

```bash
# 查看 GitHub Actions 工作流
gh run list                                  # 列出运行记录
gh run view 123456                           # 查看指定运行
gh run watch                                 # 实时监控运行状态

# Gist 操作
gh gist create file                      # 创建 Gist
gh gist list                                 # 列出 Gist

# 搜索
gh search repos "关键词"                     # 搜索仓库
gh search issues "关键词"                    # 搜索 Issue
gh search prs "关键词"                       # 搜索 PR

# SSH Key 管理
gh ssh-key list                              # 列出 SSH Key
gh ssh-key add ~/.ssh/id_rsa.pub             # 添加 SSH Key
```

---

## 3. 创建仓库

### 3.1 初始化新仓库

```bash
# 进入项目目录
cd /path/to/your/project

# 初始化 Git 仓库
git init

# 输出：Initialized empty Git repository in /path/to/your/project/.git/
```

![GitInit](https://img.yumeko.site/file/articles/GitUsage/GitInit.png)

### 3.2 克隆已有仓库

```bash
# 使用 HTTPS 克隆
git clone https://github.com/username/repository.git

# 使用 SSH 克隆（需要配置 SSH Key）
git clone git@github.com:username/repository.git

# 克隆到指定目录
git clone https://github.com/username/repository.git my-folder

# 只克隆最新版本（加快下载速度）
git clone --depth 1 https://github.com/username/repository.git
```

---

## 4. 基本工作流程

> 🔄 Git 工作流程：**工作区** → **暂存区** → **本地仓库** → **远程仓库**

### 4.1 查看状态

```bash
git status

# 简洁模式
git status -s
```

**状态标识说明：**
| 标识 | 含义 |
|:----:|:----:|
| ?? | 未跟踪的新文件 |
| A | 新添加到暂存区 |
| M | 已修改 |
| D | 已删除 |

![GithubStatus](https://img.yumeko.site/file/articles/GitUsage/GitStatus.png)

### 4.2 添加文件到暂存区

```bash
# 添加单个文件
git add filename

# 添加多个文件
git add file1 file2

# 添加所有修改和新文件
git add .

# 添加所有文件（包括删除）
git add -A
```

### 4.3 提交更改

```bash
# 提交并添加提交信息
git commit -m "提交说明"

# 添加并提交（仅限已跟踪文件）
git commit -am "提交说明"

# 修改上一次提交信息
git commit --amend -m "新的提交说明"

# 添加遗漏文件到上一次提交
git add forgotten_file
git commit --amend --no-edit
```

### 4.4 推送到远程仓库

```bash
# 首次推送（设置上游分支）
git push -u origin main

# 后续推送
git push

# 强制推送（谨慎使用！）
git push --force
```

### 4.5 拉取远程更新

```bash
# 拉取并合并
git pull

# 拉取但不自动合并
git fetch

# 拉取后查看差异
git fetch
git diff origin/main
```

---

## 5. 分支管理

### 5.1 查看分支

```bash
# 查看本地分支
git branch

# 查看所有分支（包括远程）
git branch -a

# 查看分支详情
git branch -v
```

### 5.2 创建与切换分支

```bash
# 创建并切换（推荐方式）
git checkout -b feature-login

# 切换分支
git checkout feature-login
git switch feature-login
```

### 5.3 删除分支

```bash
# 删除已合并的本地分支
git branch -d feature-login

# 强制删除未合并的分支
git branch -D feature-login

# 删除远程分支并且推送
git push origin --delete feature-login

# 清理本地缓存中已在远程删除的分支引用
git remote prune origin
git remote prune upstream
```

### 5.4 分支重命名

```bash
# 重命名当前分支
git branch -m new-name

# 重命名指定分支
git branch -m old-name new-name
```

### 5.5 分支合并

```bash
# 切换到目标分支
git checkout main

# 合并其他分支
git merge feature-login

# 合并时保留分支历史
git merge --no-ff feature-login

# 中止合并（遇到冲突时）
git merge --abort
```

### 关于merge和rebase

#### 一般的merge

合并就是合并了，直接把feature分支的历史全部删掉

![GitMergeFF](https://img.yumeko.site/file/articles/GitUsage/GitMergeFF.jpg)

#### merge --no-ff

虽然合并了，但是还是要保留feature分支的历史
一般用于，两个方案选一个

![GitMergeNoFF](https://img.yumeko.site/file/articles/GitUsage/GitMergeNoFF.jpg)

#### Rebase

两个人同时开发不同的功能，全部完成之后可以直接rebase，互不冲突

![GitRebase](https://img.yumeko.site/file/articles/GitUsage/GitRebase.jpg)

---

## 6. 远程仓库操作：Origin 与 Upstream

> 🔑 **核心概念区分：**
>
> - **origin**: 你自己的远程仓库（通常是 Fork 后的仓库）
> - **upstream**: 原始仓库（你 Fork 来源的仓库）

### 6.1 理解 Origin 和 Upstream

**Fork 工作流关系图：**

![Fork.png](https://img.yumeko.site/file/articles/GitUsage/Fork.png)

**典型使用场景：**

| 操作 | 使用哪个远程 | 说明 |
|------|-------------|------|
| 推送自己的修改 | origin | `git push origin main` |
| 获取原始仓库更新 | upstream | `git fetch upstream` |
| 提交 PR | origin → upstream | 从 origin 向 upstream 发起 PR |

### 6.2 设置origin和upstream

```bash
# 查看当前远程仓库配置
git remote -v
# 输出示例（只有 origin）：
# origin  https://github.com/you/repo.git (fetch)
# origin  https://github.com/you/repo.git (push)

# 添加 upstream（指向原始仓库）
git remote add upstream https://github.com/original-owner/repo.git

# 再次查看
git remote -v
# 输出示例（有 origin 和 upstream）：
# origin    https://github.com/you/repo.git (fetch)
# origin    https://github.com/you/repo.git (push)
# upstream  https://github.com/original-owner/repo.git (fetch)
# upstream  https://github.com/original-owner/repo.git (push)
```

### 6.3 管理origin和upstream

```bash
# 添加origin和upstream
git remote add <名称> <URL>
git remote add origin https://github.com/you/repo.git
git remote add upstream https://github.com/owner/repo.git

# 修改origin URL
git remote set-url origin https://github.com/you/new-repo.git
git remote set-url upstream https://github.com/owner/repo.git

# 删除upstream
git remote remove upstream
git remote remove origin

# 重命名origin
git remote rename origin old-origin
git remote remove upstream old-upstream

# 查看某个origin的详细信息
git remote show origin
git remote show upstream
```

### 6.4 从 origin同步更新

```bash
# 步骤 1: 获取 origin 的最新内容
git fetch origin

# 步骤 2: 切换到本地主分支
git checkout main

# 步骤 3: 合并 origin 的更改到本地
git merge origin/main

# 或者使用 rebase 保持提交历史整洁
git rebase origin/main
```

>  强制覆盖

```bash
git reset --hard origin/main
```

### 6.5 从 Upstream 同步更新

**场景：** 原始仓库有了新的提交，你需要把这些更新同步到自己的 Fork。

```bash
# 步骤 1: 获取 upstream 的最新内容
git fetch upstream

# 步骤 2: 切换到本地主分支
git checkout main

# 步骤 3: 合并 upstream 的更改到本地
git merge upstream/main

# 或者使用 rebase 保持提交历史整洁
git rebase upstream/main

# 步骤 4: 推送到自己的远程仓库 (origin)
git push origin main
```

> 强制覆盖

```bash
git reset --hard upstream/main
git push -f origin main
```

### 6.6 处理合并冲突

当 `git pull` 或 `git merge` 时出现冲突：

```bash
# 1. 查看冲突文件
git status

# 2. 打开冲突文件，手动编辑解决冲突
# 冲突标记示例：
# <<<<<<< HEAD
# 你的更改
# =======
# 远程的更改
# >>>>>>> upstream/main

# 3. 标记冲突已解决
git add conflicted_file

# 4. 完成合并
git commit -m "解决合并冲突"
```

## 7. 撤销与回退详解

> ⚠️ **这是最重要也最容易混淆的部分！** 请仔细理解不同命令的区别。

### 7.1 Git 的三个工作区域

**正向操作：**

![Forward.png](https://img.yumeko.site/file/articles/GitUsage/Forward.png)

**撤销操作：**

![revoked.png](https://img.yumeko.site/file/articles/GitUsage/revoked.png)

### 7.2 撤销命令全景图

![HowToRevoke.jpg](https://img.yumeko.site/file/articles/GitUsage/HowToRevoke.jpg)

### 7.3 命令对比总览

| 命令                          | 作用范围        | 是否改变历史 | 是否安全 | 典型使用场景     |
| ----------------------------- | --------------- | ------------ | -------- | ---------------- |
| `git restore <file>`          | 工作区          | ❌ 不改变    | ✅ 安全  | 撤销文件修改     |
| `git restore --staged <file>` | 暂存区          | ❌ 不改变    | ✅ 安全  | 取消暂存         |
| `git reset --soft`            | 提交历史        | ⚠️ 改变      | ✅ 安全  | 重新组织提交     |
| `git reset --mixed`           | 提交历史+暂存区 | ⚠️ 改变      | ✅ 安全  | 取消提交和暂存   |
| `git reset --hard`            | 全部            | ⚠️ 改变      | ❌ 危险  | 完全回退         |
| `git revert`                  | 创建新提交      | ❌ 不改变    | ✅ 安全  | 撤销已推送的提交 |

---

### 7.4 restore：撤销工作区和暂存区的更改（推荐）

> 📌 `git restore` 是 Git 2.23+ 引入的新命令，专门用于恢复文件，语义更清晰。

#### 撤销工作区的修改

**场景：** 你修改了文件，但还没有 `git add`，想放弃这些修改。

```bash
# 撤销单个文件的修改
git restore filename

# 撤销多个文件
git restore file1 file2

# 撤销所有文件的修改
git restore .

# 撤销指定目录下的所有修改
git restore src/
```

**效果：** 文件恢复到最后一次 `git add` 或 `git commit` 的状态。

#### 取消暂存（从暂存区移出）

**场景：** 你已经 `git add` 了文件，但想取消暂存。

```bash
# 取消暂存单个文件（文件内容保留在工作区）
git restore --staged filename

# 取消暂存所有文件
git restore --staged .
```

**效果：** 文件从暂存区移出，但**工作区的修改保留**。

#### 同时撤销工作区和暂存区

```bash
# 撤销暂存并丢弃工作区修改
git restore --staged --worktree filename

# 简写
git restore -SW filename
```

#### 恢复到指定提交的版本

```bash
# 将文件恢复到指定提交的状态
git restore --source=HEAD~2 filename

# 恢复到指定 commit ID 的状态
git restore --source=abc1234 filename
```

---

### 7.5 reset：回退提交历史

> ⚠️ `git reset` 会改变提交历史，已推送的提交不要使用 reset！

#### 三种模式对比

> 假设当前状态：`Commit: A -> B -> C (HEAD)`，执行 `git reset --xxx HEAD~1`

**初始状态：**

![init.png](https://img.yumeko.site/file/articles/GitUsage/init.png)

**`--soft` 模式（✅ 安全）：**

![soft.png](https://img.yumeko.site/file/articles/GitUsage/soft.png)

**`--mixed` 模式（默认，✅ 安全）：**

![mixed.png](https://img.yumeko.site/file/articles/GitUsage/mixed.png)

**`--hard` 模式（⚠️ 危险！）：**

![hard.png](https://img.yumeko.site/file/articles/GitUsage/hard.png)

**模式效果对比表：**

| 模式      | 提交历史     | 暂存区      | 工作区      | 安全性  |
| --------- | ------------ | ----------- | ----------- | ------- |
| `--soft`  | A → B (HEAD) | ✅ 保留更改 | ✅ 保留更改 | 安全    |
| `--mixed` | A → B (HEAD) | ❌ 清空     | ✅ 保留更改 | 安全    |
| `--hard`  | A → B (HEAD) | ❌ 清空     | ❌ 清空     | ⚠️ 危险 |

#### --soft：只移动 HEAD，保留所有更改

**场景：** 想要重新组织最近的提交，或者合并多个提交为一个。

```bash
# 回退最近 1 次提交，更改保留在暂存区
git reset --soft HEAD~1

# 回退最近 3 次提交
git reset --soft HEAD~3

# 现在可以重新提交
git commit -m "合并后的新提交信息"
```

**典型用途：**

- 修改提交信息
- 合并多个小提交为一个大提交
- 拆分一个大提交为多个小提交

#### --mixed（默认）：移动 HEAD，清空暂存区

**场景：** 想要取消提交和暂存，但保留工作区的修改。

```bash
# 回退最近 1 次提交（默认就是 --mixed）
git reset HEAD~1

# 等价于
git reset --mixed HEAD~1

# 现在需要重新 add 和 commit
git add .
git commit -m "新的提交信息"
```

**典型用途：**

- 重新选择要提交的文件
- 发现提交的文件不对，想重新组织

#### --hard：完全回退，丢弃所有更改

**场景：** 想要彻底放弃最近的提交和所有修改。

```bash
# ⚠️ 危险操作！会丢失所有未提交的更改！
git reset --hard HEAD~1

# 回退到指定提交
git reset --hard abc1234

# 回退到远程分支的状态
git reset --hard origin/main
```

**⚠️ 警告：** `--hard` 会永久删除工作区的修改，除非使用 `git reflog` 恢复！

#### reset 常用场景

```bash
# 场景 1: 取消最近的 git add（不影响工作区）
git reset HEAD
git reset HEAD filename

# 场景 2: 撤销最近 1 次提交，保留更改在暂存区
git reset --soft HEAD~1

# 场景 3: 撤销最近 1 次提交，保留更改在工作区
git reset HEAD~1

# 场景 4: 完全放弃最近 2 次提交
git reset --hard HEAD~2

# 场景 5: 回退到与远程一致的状态
git fetch origin
git reset --hard origin/main
```

---

### 7.6 revert：安全地撤销已推送的提交

> ✅ `git revert` 是撤销已推送提交的**唯一安全方式**，它不改变历史，而是创建新提交。

#### reset 与 revert 的根本区别

> 假设提交历史: `A -> B -> C (HEAD)`，想要撤销提交 C

**原始状态：**

![init.png](https://img.yumeko.site/file/articles/GitUsage/init.png)

**使用 reset（❌ 改写历史）：**

![reset.png](https://img.yumeko.site/file/articles/GitUsage/reset.png)

**使用 revert（✅ 保留历史）：**

![revert.png](https://img.yumeko.site/file/articles/GitUsage/revert.png)

**选择原则：**

![chooseRevoke.png](https://img.yumeko.site/file/articles/GitUsage/chooseRevoke.png)

#### revert 基本用法

```bash
# 撤销最近一次提交
git revert HEAD

# 撤销指定提交
git revert abc1234

# 撤销但不自动提交（可以修改后再提交）
git revert --no-commit abc1234
git revert -n abc1234
```

#### 撤销多个提交

```bash
# 撤销最近 3 次提交（每个都会生成一个 revert 提交）
git revert HEAD~2..HEAD

# 撤销多个提交并合并为一个 revert 提交
git revert --no-commit HEAD~2..HEAD
git commit -m "Revert: 撤销最近3次提交"

# 撤销多个不连续的提交
git revert abc1234 def5678 ghi9012
```

#### 处理 revert 冲突

```bash
# 如果 revert 过程中出现冲突
git status                    # 查看冲突文件
# 手动解决冲突后
git add .
git revert --continue

# 放弃 revert
git revert --abort
```

---

### 7.7 实际场景速查表

| 我想要...                           | 使用命令                                            |
| ----------------------------------- | --------------------------------------------------- |
| 撤销工作区的文件修改                | `git restore <file>`                                |
| 取消已暂存的文件                    | `git restore --staged <file>`                       |
| 修改最近一次提交的信息              | `git commit --amend -m "新信息"`                    |
| 往最近一次提交添加遗漏的文件        | `git add <file>` + `git commit --amend --no-edit`   |
| 撤销最近 N 次未推送的提交，保留更改 | `git reset --soft HEAD~N`                           |
| 撤销最近 N 次未推送的提交，丢弃更改 | `git reset --hard HEAD~N`                           |
| 撤销已推送的某次提交                | `git revert <commit-id>`                            |
| 完全放弃本地所有更改，与远程同步    | `git fetch origin` + `git reset --hard origin/main` |
| 恢复被 reset --hard 误删的提交      | `git reflog` + `git checkout <commit-id>`           |

---

### 7.8 reflog：操作历史记录（后悔药）

> 🆘 `git reflog` 记录了所有 HEAD 的变动，可以用来恢复误删的提交。

```bash
# 查看操作历史
git reflog

# 输出示例：
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: 添加新功能
# ghi9012 HEAD@{2}: commit: 修复 bug

# 恢复到之前的状态
git reset --hard HEAD@{1}

# 或者创建新分支来恢复
git checkout -b recovery-branch def5678
```

**注意：** reflog 只保存本地操作记录，一般保留 90 天。

---

## 8. 查看历史与状态

### 8.1 查看提交历史

```bash
# 查看提交历史
git log

# 简洁模式（一行显示）
git log --oneline

# 显示文件变更统计
git log --stat

# 图形化显示分支合并历史
git log --oneline --graph --all

# 限制显示数量
git log -n 5

# 搜索提交信息
git log --grep="关键词"

# 查看某个文件的修改历史
git log --follow filename
```

### 8.2 查看差异

```bash
# 查看工作区与暂存区差异
git diff

# 查看暂存区与最新提交差异
git diff --staged
git diff --cached

# 比较两个分支
git diff main..feature-login

# 比较两次提交
git diff abc1234 def5678

# 查看某个文件的差异
git diff filename
```

### 8.3 查看某次提交内容

```bash
# 查看提交详情
git show abc1234

# 查看某次提交的某个文件
git show abc1234:path/to/file
```

---

## 9. Fork 工作流完整流程

### 9.1 Fork原仓库

![HowToFork](https://img.yumeko.site/file/articles/GitUsage/HowToFork.png)

1. Fork原仓库之后，在自己的repo中就有一个fork之后的仓库了![ForkedRepo](https://img.yumeko.site/file/articles/GitUsage/ForkedRepo.png)
2. 然后在本地clone自己的repo，就可以开始开发了

### 9.2 进行开发

1. 查看原仓库

```bash
git remote -v

# 只有origin
# origin  https://github.com/you/repo-forked (fetch)
# origin  https://github.com/you/repo-forked (push)
```

2. 添加upstream，可以同步upstream

```bash
git remote add upstream https://github.com/original-owner/repo.git

git remote -v

# 现在origin和upstream都有了
# origin  https://github.com/you/repo-forked (fetch)
# origin  https://github.com/you/repo-forked (push)
# upstream        https://github.com/original-owner/repo.git (fetch)
# upstream        https://github.com/original-owner/repo.git (push)
```

3. 本地开发

```bash
# 4. 创建功能分支
git checkout -b feature/new-feature

# 5. 进行开发，提交更改
git add .
git commit -m "Add new feature"

# 6. 推送到自己的远程仓库
git push -u origin feature/new-feature
```

### 9.3 提pull request

1. 确认你提交到了自己的origin上![PushToOrigin](https://img.yumeko.site/file/articles/GitUsage/PushToOrigin.png)
2. 点击 **Compare & pull request**，提交给上游![CompareBranch](https://img.yumeko.site/file/articles/GitUsage/CompareBranch.png)

* 左侧的**base repo**指的是你想要**合并入**的repo
* 右侧的head repo指的是你**修改后**的repo
* 如果左右两边都是**你自己**的，那就跟本地开发**merge**没区别
* 左边一般就是**owner**，右边就是你自己：表示你想把自己的更改提交到**owner**的**某个分支**

### 9.4 上游审核一下pr

![MergePR](https://img.yumeko.site/file/articles/GitUsage/MergePR.png)

上游会看到这样的界面

1. 谁修改的：本地的github账号登陆的是**账号1**，所以修改的是**修改后**
2. 谁提交的pr：**origin**是谁，就是**谁**提交的pr，所以是**账号2**提交的pr
3. owner是谁：**upstream**才是真正的**owner**，决定你的修改是否可以进入**upstream**
4. 怎么合并

* Create A Merge Commit：保留**修改者**的所有commit，如果你交了3个修改，那就显示3个修改![CreateMergeCommit](https://img.yumeko.site/file/articles/GitUsage/CreateMergeCommit.jpg)
* Squash And Merge：只有**一个修改**，你交了n个commit，最后也只显示你改了一次![SquashAndMerge](https://img.yumeko.site/file/articles/GitUsage/SquashAndMerge.jpg)
* Rebase And Merge：修改者交了很多commit，跟rebase一样，被线性的加在当前历史的后方![RebaseAndMerge](https://img.yumeko.site/file/articles/GitUsage/RebaseAndMerge.jpg)

### 9.5 pr被合并之后

```bash
# 8. 如果 PR 合并后，同步更新
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# 9. 删除已合并的功能分支
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

---

## 10. 高级技巧

### 10.1 暂存工作区 (Stash)

> 📦 `git stash` 可以临时保存工作区的修改，让你可以切换分支处理其他事情。

**Stash 工作原理：**

![stash.png](https://img.yumeko.site/file/articles/GitUsage/stash.png)

**Stash 操作流程：**

![HowToStash.png](https://img.yumeko.site/file/articles/GitUsage/HowToStash.png)

**常用命令：**

```bash
# 暂存当前工作
git stash

# 暂存并添加说明
git stash save "正在开发的功能"

# 包含未跟踪的文件
git stash -u

# 查看暂存列表
git stash list

# 恢复最近的暂存（并从列表中删除）
git stash pop

# 恢复最近的暂存（保留在列表中）
git stash apply

# 恢复指定暂存
git stash apply stash@{1}

# 删除暂存
git stash drop stash@{0}

# 清空所有暂存
git stash clear

# 查看暂存的内容
git stash show -p stash@{0}
```

### 10.2 Rebase（变基）

```bash
# 将当前分支变基到 main
git rebase main

# 交互式变基（可重新排序、合并提交）
git rebase -i HEAD~3

# 中止变基
git rebase --abort

# 继续变基（解决冲突后）
git rebase --continue
```

### 10.3 .gitignore 文件

```bash
# 创建 .gitignore 文件
touch .gitignore
```

常用 `.gitignore` 模板：

```gitignore
# 操作系统文件
.DS_Store
Thumbs.db

# IDE 配置
.idea/
.vscode/
*.swp

# 依赖目录
node_modules/
vendor/
__pycache__/

# 构建输出
dist/
build/
*.log

# 环境配置
.env
.env.local
*.pem
```

---

## 11. 常见问题解决

### 11.1 中文文件名显示乱码

```bash
git config --global core.quotepath false
```

### 11.2 换行符问题（Windows/Linux 混用）

```bash
# Windows 推荐配置
git config --global core.autocrlf true

# Linux/Mac 推荐配置
git config --global core.autocrlf input
```

### 11.3 忽略已跟踪的文件

```bash
# 停止跟踪文件但保留本地文件
git rm --cached filename

# 添加到 .gitignore
echo "filename" >> .gitignore
```

### 11.4 修改错误的提交作者信息

```bash
# 修改最近一次提交的作者
git commit --amend --author="New Name <newemail@example.com>"
```

### 11.5 恢复删除的分支

```bash
# 查看操作记录
git reflog

# 找到删除前的提交 ID，创建新分支
git checkout -b recovered-branch abc1234
```

### 11.6 清理未跟踪文件

```bash
# 预览将被删除的文件
git clean -n

# 删除未跟踪的文件
git clean -f

# 删除未跟踪的文件和目录
git clean -fd
```

---

## 📌 快速参考卡片

### Git 基础命令

| 操作       | 命令                       |
| ---------- | -------------------------- |
| 初始化仓库 | `git init`                 |
| 克隆仓库   | `git clone <url>`          |
| 查看状态   | `git status`               |
| 添加文件   | `git add .`                |
| 提交更改   | `git commit -m "message"`  |
| 推送代码   | `git push`                 |
| 拉取更新   | `git pull`                 |
| 创建分支   | `git checkout -b <branch>` |
| 切换分支   | `git checkout <branch>`    |
| 合并分支   | `git merge <branch>`       |
| 查看历史   | `git log --oneline`        |

### 撤销与回退命令

| 场景               | 命令                          |
| ------------------ | ----------------------------- |
| 撤销工作区修改     | `git restore <file>`          |
| 取消暂存           | `git restore --staged <file>` |
| 回退提交(保留更改) | `git reset --soft HEAD~1`     |
| 回退提交(丢弃更改) | `git reset --hard HEAD~1`     |
| 撤销已推送的提交   | `git revert <commit>`         |

### GitHub CLI 命令

| 操作        | 命令                              |
| ----------- | --------------------------------- |
| 登录        | `gh auth login`                   |
| Fork 并克隆 | `gh repo fork owner/repo --clone` |
| 创建 PR     | `gh pr create`                    |
| 查看 PR     | `gh pr list`                      |
| 检出 PR     | `gh pr checkout 123`              |
| 创建 Issue  | `gh issue create`                 |

---

## 📚 推荐资源

- [Git 官方文档](https://git-scm.com/doc)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)
- [Learn Git Branching（交互式学习）](https://learngitbranching.js.org/?locale=zh_CN)
- [GitHub CLI 官方文档](https://cli.github.com/manual/)
- [GitHub Docs](https://docs.github.com/cn)
