---
title: 关于Github的进阶用法
date: 2026-01-25
category: 技术
tags:
  - 高级教程
  - 资源
  - git
description: 除了代码托管之外，github还能做什么
image: https://img.yumeko.site/file/blog/GitHub.png
status: published
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
## 使用第三方平台
### Codex
如何使用ai来对项目进行代码审查
以`Codex`为例
* 在Codex的界面中，打开代码审查

![Codex.png](https://img.yumeko.site/file/articles/GithubUsage/Codex.png)

* 当`Pull Requests`从`draft`变成`Ready For Review`的时候，就会触发代码审查，`Codex`会回复你的`Pull Requests`，看看你的提交有什么问题

![CodeReview.png](https://img.yumeko.site/file/articles/GithubUsage/CodeReview.png)

### Dosu

访问[Dosu](https://app.dosu.dev/)，Dosu是一个专门审查代码的第三方平台，但是要收费，我不建议
1. 登录github，给自己随便取一个名字
2. 每个月会免费赠送100的请求，个人项目是够用了
![DosuPay.png](https://img.yumeko.site/file/blog/DosuPay.png)

3. 点击**部署**，选择标准部署
4. 选择目标中，github下方的安装，转跳到github中安装Dosu
![DosuInstall.png](https://img.yumeko.site/file/blog/DosuInstall.png)

5. 选择自己的账户之后，选择你想要配置的仓库

![DosuRepoSelect.png](https://img.yumeko.site/file/blog/DosuRepoSelect.png)

6. 放回`Dosu`，在已经配置的仓库中，选择要配置`Dosu`的仓库，并且开启数据源
7. 设置仓库的配置
* 名称：就是你在Dosu中想要看到的名称
* 默认维护者：出了问题Dosu会通知谁
* Dosu注释：让Dosu忽略一些问题，注意某些问题，等等，类似prompt
* 代理回复行为：Mention就是被动回复，Draft就是只有pr为Draft才回复，Reply就是自动回复
* 自动标记问题：Dosu会根据问题自动给他加上标签
* 过时机器人：可以不加，用于大项目中问题不能放太长时间
* LGTM、尺寸、文档随PR更新：自己看着办
8. 审查完你的配置之后，你提交issue和pr就会有dosu自动回复了

### Sourcery

和`Dosu`类似，也是一个审查平台，可以访问[Sourcery](https://app.sourcery.ai/accounts/239236/github-app)，所有的步骤基本一致，但是他是按Issue和PR来区分，不是按仓库来区分，但是都是一样的

## 利用CICD，配置自己的机器人

既然我们是希望在issue和PR中有机器人回复，同样的，我们可以利用CICD的特点：
* 当我们提出PR的时候，调用LLM接口，把diff作为上下文发送给LLM，让LLM作为一个代码审查机器人审查我们的代码
### 配置ai调用的py文件
在代码的`.github/scripts`中加上一个`ai_review.py`文件，只要提出PR，就会调用这个py文件，请求LLM回复代码问题
```python
#!/usr/bin/env python3

"""
.github/scripts
AI Code Review Script
调用 LLM API 对 PR diff 进行代码审查
"""
import os
import httpx
SYSTEM_PROMPT = """你是一个资深的代码审查专家。请审查以下 Pull Request 的代码变更。
你需要关注以下方面：
1. **潜在的 Bug 和边缘情况** - 未处理的异常、空值检查、边界条件等
2. **安全漏洞** - SQL注入、XSS、敏感信息泄露、不安全的依赖等
3. **性能问题** - 不必要的循环、内存泄漏、N+1查询等
4. **代码质量** - 可读性、命名规范、重复代码、过于复杂的逻辑
5. **最佳实践** - 是否遵循该语言/框架的最佳实践
**请忽略以下文件的变更，不要对它们进行审查：**
- requirements.txt、requirements*.txt 等依赖声明文件
- 这些文件只是依赖版本声明，不需要代码审查
请用中文回复，格式如下：
- 如果发现问题，列出具体的问题和建议，引用具体的代码行
- 如果代码质量良好，简单说明即可
- 不要过度挑剔，只关注真正重要的问题
回复格式：
### 🤖 AI Code Review
**审查的提交:** `{commit_sha}`
#### 发现的问题
（如果有问题，按严重程度列出）
#### 总结
（简短总结代码质量）
---
<details>
<summary>ℹ️ 关于此审查</summary>
此审查由 AI 自动生成，仅供参考。如有误报请忽略。
</details>

"""
def getDiffContent() -> str:
    """读取 PR diff 内容"""
    diffFile = os.environ.get("diff_file", "pr_diff.txt")
    if os.path.exists(diffFile):
        with open(diffFile, encoding="utf-8", errors="ignore") as f:
            return f.read()
    # 备用：直接读取
    if os.path.exists("pr_diff.txt"):
        with open("pr_diff.txt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""
def truncateDiff(diff: str, maxChars: int = 60000) -> str:
    """截断过长的 diff，避免超出 token 限制"""
    if len(diff) <= maxChars:
        return diff
    return diff[:maxChars] + "\n\n... (diff 过长，已截断)"
def callChatApi(
    apiKey: str,
    baseUrl: str,
    model: str,
    systemPrompt: str,
    userMessage: str,
) -> str:
    """
    调用 OpenAI Chat Completions API
    Args:
        apiKey: API 密钥
        baseUrl: API 基础 URL (如 https://api.openai.com/v1)
        model: 模型名称
        systemPrompt: 系统提示
        userMessage: 用户消息
    Returns:
        模型响应内容
    """
    # 拼接完整 URL: baseUrl + /chat/completions
    url = baseUrl.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {apiKey}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userMessage},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
    # 校验响应结构
    choices = data.get("choices", [])
    if not choices:
        raise Exception(f"API 响应缺少 choices 字段: {data}")
    return choices[0].get("message", {}).get("content", "")
def callMessagesApi(
    apiKey: str,
    baseUrl: str,
    model: str,
    systemPrompt: str,
    userMessage: str,
) -> str:
    """
    调用 Anthropic Messages API
    Args:
        apiKey: API 密钥
        baseUrl: API 基础 URL (如 https://api.anthropic.com/v1)
        model: 模型名称
        systemPrompt: 系统提示
        userMessage: 用户消息
    Returns:
        模型响应内容
    """
    # 拼接完整 URL: baseUrl + /messages
    url = baseUrl.rstrip("/") + "/messages"
    headers = {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2000,
        "system": systemPrompt,
        "messages": [{"role": "user", "content": userMessage}],
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
    # 校验响应结构
    content = data.get("content", [])
    if not content:
        raise Exception(f"API 响应缺少 content 字段: {data}")
    return content[0].get("text", "")
def callResponseApi(
    apiKey: str,
    baseUrl: str,
    model: str,
    systemPrompt: str,
    userMessage: str,
) -> str:
    """
    调用 OpenAI Responses API
    Args:
        apiKey: API 密钥
        baseUrl: API 基础 URL (如 https://api.openai.com/v1)
        model: 模型名称
        systemPrompt: 系统提示
        userMessage: 用户消息
    Returns:
        模型响应内容
    """
    # 拼接完整 URL: baseUrl + /responses
    url = baseUrl.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {apiKey}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "instructions": systemPrompt,
        "input": userMessage,
    }
    with httpx.Client(timeout=120) as client:
        response = client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
    # Responses API 返回格式: output_text 或 output 数组
    if "output_text" in data:
        return data["output_text"]
    # 解析 output 数组结构
    output = data.get("output", [])
    if output and isinstance(output, list):
        for item in output:
            if item.get("type") == "message":
                content = item.get("content", [])
                # 遍历 content 找到 output_text 类型
                for block in content:
                    if block.get("type") == "output_text":
                        return block.get("text", "")
    raise Exception(f"无法解析 Responses API 响应: {data}")
VALID_API_TYPES = {"chat", "messages", "response"}
def callLlmApi(
    apiKey: str,
    baseUrl: str,
    model: str,
    systemPrompt: str,
    userMessage: str,
    apiType: str,
) -> str:
    """
    统一调用 LLM API
    Args:
        apiKey: API 密钥
        baseUrl: API 基础 URL (如 https://api.openai.com/v1)
        model: 模型名称
        systemPrompt: 系统提示
        userMessage: 用户消息
        apiType: API 类型 ('chat', 'messages', 'response')
    Returns:
        模型响应内容
    """
    if apiType not in VALID_API_TYPES:
        raise ValueError(f"无效的 API 类型: '{apiType}'，支持的类型: {VALID_API_TYPES}")
    print(f"Using API type: {apiType}")
    if apiType == "messages":
        return callMessagesApi(apiKey, baseUrl, model, systemPrompt, userMessage)
    elif apiType == "response":
        return callResponseApi(apiKey, baseUrl, model, systemPrompt, userMessage)
    else:
        return callChatApi(apiKey, baseUrl, model, systemPrompt, userMessage)
def main():
    apiKey = os.environ.get("LLM_API_KEY")
    if not apiKey:
        print("Error: LLM_API_KEY not set")
        return
    baseUrl = os.environ.get("LLM_BASE_URL")  # 必需，如 https://api.openai.com/v1
    if not baseUrl:
        print("Error: LLM_BASE_URL not set")
        return
    model = os.environ.get("LLM_MODEL")
    if not model:
        print("Error: LLM_MODEL not set")
        return
    apiType = os.environ.get("LLM_API_TYPE", "chat")  # chat, messages, response
    prTitle = os.environ.get("PR_TITLE", "")
    prBody = os.environ.get("PR_BODY", "")
    # 获取 commit SHA
    commitSha = os.environ.get("GITHUB_SHA", "unknown")[:10]
    diffContent = getDiffContent()
    if not diffContent:
        print("No diff content found, skipping review")
        return
    diffContent = truncateDiff(diffContent)
    # 构造用户消息
    userMessage = f"""## Pull Request 信息
**标题:** {prTitle}
**描述:**
{prBody or "无描述"}
## 代码变更 (diff)
\`\`\`diff
{diffContent}
\'\'\'
请审查以上代码变更。
"""

    try:
        systemPrompt = SYSTEM_PROMPT.format(commit_sha=commitSha)
        reviewContent = callLlmApi(
            apiKey=apiKey,
            baseUrl=baseUrl,
            model=model,
            systemPrompt=systemPrompt,
            userMessage=userMessage,
            apiType=apiType,
        )
        # 写入结果文件
        with open("review_result.md", "w", encoding="utf-8") as f:
            f.write(reviewContent)
        print("Review completed successfully!")
        print(reviewContent)
    except Exception as e:
        print(f"Error calling LLM API: {e}")
if __name__ == "__main__":

    main()
```

### 配置CI/CD流程
然后在工作流文件夹中创建调用py文件的流程
在`.github/workflow`文件夹中创建`ai-code-review.yml`文件

```yml
name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    runs-on: ubuntu-latest
    # 跳过 draft PR
    if: github.event.pull_request.draft == false
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install httpx
      - name: Get PR diff
        id: diff
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh pr diff ${{ github.event.pull_request.number }} > pr_diff.txt
          echo "diff_file=pr_diff.txt" >> $GITHUB_OUTPUT
      - name: AI Code Review
        env:
          # 在 repo settings -> secrets 中配置你的 API key
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          LLM_BASE_URL: ${{ secrets.LLM_BASE_URL }}  # 可选，如果用 OpenAI 兼容 API
          LLM_MODEL: ${{ secrets.LLM_MODEL }}        # 可选，默认 gpt-4o
          PR_TITLE: ${{ github.event.pull_request.title }}
          PR_BODY: ${{ github.event.pull_request.body }}
        run: |
          python .github/scripts/ai_review.py
      - name: Post Review Comment
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          if [ -f review_result.md ]; then
            gh pr review ${{ github.event.pull_request.number }} \
              --comment \
              --body-file review_result.md
          fi
```

### 在github中配置环境变量
在你想要配置的仓库中，配置`LLM_BASE_URL`，`LLM_API_TYPE`，`LLM_API_KEY`，`LLM_MODEL`
示例：

|变量名     |内容  |
| --- | --- |
|LLM_BASE_URL   |  `https://api.openai.com/v1`   |
|LLM_API_TYPE  |   `chat`/`message`/`response` 三选一  |
|LLM_API_KEY   |  API-KEY   |
|LLM_MODEL   |  模型名字   |


![SecretSetting.png](https://img.yumeko.site/file/blog/SecretSetting.png)



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