---
title: 自定义Copilot
date: 2026-01-20
category: 小巧思
tags:
  - ai
  - 资源
description: 把Copilot中接入第三方的api
image: https://img.yumeko.site/file/blog/github-copilot.png
status: private
---

## Github Copilot接入第三方api

### 修改配置文件

在 VS Code 的 `settings.json` 中添加一段配置。

```json
"github.copilot.chat.azureModels": {
    "模型名称": {
        "url": "中转站api/v1/chat/completions", 
        "name": "模型名称",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 1000000,
        "maxOutputTokens": 64000
    }
}
```

思考模型可以加上`thinking: true`

### 完整配置示例

以下以添加 `gemini-3-flash-preview` 和 `glm-4.7` 为例：

```json
"github.copilot.chat.azureModels": {
    "gemini-3-flash-preview": {
        "url": "https://new.123nhh.xyz/v1/chat/completions", 
        "name": "gemini-3-flash-preview",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 1000000,
        "maxOutputTokens": 64000
    },
    "glm-4.7": {
        "url": "https://new.123nhh.xyz/v1/chat/completions", 
        "name": "glm-4.7",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 1000000,
        "maxOutputTokens": 64000
    }
}
```

> **提示**：如果想要添加更多模型，按照上述格式继续往后面追加即可。

## 启用与配置 Key

1. 配置key：在切换模型界面中点击管理模型

![ManageModel](https://img.yumeko.site/file/articles/CopilotCustom/ManageModel.png)

2. 在正上方弹出的provider配置中选择Azure

![ChooseAzure](https://img.yumeko.site/file/articles/CopilotCustom/ChooseAzure.png)

3. 填入你的中转key，然后重启vscode。



补充：使用版本：https://vscode.download.prss.microsoft.com/dbazure/download/stable/7d842fb85a0275a4a8e4d7e040d2625abbf7f084/VSCodeUserSetup-x64-1.105.1.exe