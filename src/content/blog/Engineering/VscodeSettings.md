---
title: VS Code Settings 完整配置指南
date: 2026-06-12
category: 工程实践
tags:
  - 工具
  - 设置
  - vscode
description: VS Code settings.json 完整配置解析,涵盖 Copilot、LaTeX Workshop、GitLens、终端等各模块的配置说明
image: https://img.yumeko.site/file/blog/cover/1781270637967_VSCodeSettings.webp
status: published
---

## 1. 完整 settings.json

以下是我日常使用的完整 VS Code 用户配置:

```json
{
    // 使用github copilot 生成 git message 
    // 设置中文
    "github.copilot.chat.localeOverride": "zh-CN",
    // 生成 commit message 的指令
    "github.copilot.chat.commitMessageGeneration.instructions": [
        { "text": "始终使用简体中文生成 git commit message." },
        { "text": "严格按照以下格式输出: <type>(<scope>): <subject>" },
        { "text": "<body>" },
        { "text": "<footer>"},
        {"text": "type, scope使用英文, 说明功能改变方向"},
        { "text": "根据本次暂存改动的主要目的,选择最合适的一个 type,不要罗列多个." },
        {"text": "type和scope为必须填写的, 只能填写以下类型: feat, fix, docs, style, refactor, perf, test, chore, build, ci, revert"},
        {"text": "subject使用中文message, 简洁说明改动内容"},
        {"text": "body使用中文, 要写3条及以上, 可以选择'-m'或者'-'进行分隔不同行的内容"},
        {"text": "footer没有必要就不写, 但是关联issue, pr等就必须要写, 格式为'ref issue #123'或者'close issue #123'"},
        { "text": "不要输出英文,不要附加解释,不要输出列表,只输出最终的 commit message." },
    ],

    // 全角标点自动转半角(中文输入法误打的符号纠正)
    "punctuationConverter.enabled": true,
    // 自定义映射:全角～→半角~,全角￥→半角$
    "punctuationConverter.customMap": {
        "～": "~",
        "￥": "$"
    },
    // Copilot 预测下一个编辑位置并给出灰显建议
    "github.copilot.nextEditSuggestions.enabled": true,
    // 聊天面板中显示 MCP 服务器市场
    "chat.mcp.gallery.enabled": true,
    // 文件图标主题:Material Icon Theme
    "workbench.iconTheme": "material-icon-theme",
    // 终端启动时不显示"提示:使用 Ctrl+Shift+` 切换终端"
    "terminal.integrated.initialHint": false,
    // Python 文件差异对比时忽略行尾空格变化
    "[python]": {
        "diffEditor.ignoreTrimWhitespace": true
    },
    // Git 同步(push/pull)不弹确认对话框
    "git.confirmSync": false,
    //------------------------------LaTeX配置----------------------------------
    // 关闭 ChkTeX 语法风格检查
    "latex-workshop.linting.chktex.enabled": false,
    // 关闭 LaCheck 语法检查
    "latex-workshop.linting.lacheck.enabled": false,
    // 设置是否自动编译
    "latex-workshop.latex.autoBuild.run": "never",
    //右键菜单
    "latex-workshop.showContextMenu": true,
    //从使用的包中自动补全命令和环境
    "latex-workshop.intellisense.package.enabled": true,
    //编译出错时设置是否弹出气泡设置
    "latex-workshop.message.error.show": false,
    "latex-workshop.message.warning.show": false,
    //latexmk生成的pdf文件输出目录,%DIR%为.tex文件所在目录,%DOCFILE%为.tex文件名(不带扩展名)
    "latex-workshop.latex.outDir": "%DIR%/build",
    "latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-output-directory=%DIR%/build",
                "%DOCFILE%",
            ],
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-output-directory=%DIR%/build",
                "%DOCFILE%",
            ],
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "-outdir=%DIR%/build",
                "%DOCFILE%",
            ],
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
                "%DIR%/build/%DOCFILE%"
            ],
        },
    ],
    //用于配置编译链
    "latex-workshop.latex.recipes": [
        {
            "name": "XeLaTeX",
            "tools": [
                "xelatex"
            ],
        },
        {
            "name": "PDFLaTeX",
            "tools": [
                "pdflatex"
            ],
        },
        {
            "name": "BibTeX",
            "tools": [
                "bibtex"
            ],
        },
        {
            "name": "LaTeXmk",
            "tools": [
                "latexmk"
            ],
        },
        {
            "name": "xelatex->bibtex->xelatex*2",
            "tools": [
                "xelatex",
                "bibtex",
                "xelatex",
                "xelatex"
            ],
        },
        {
            "name": "pdflatex->bibtex->pdflatex*2",
            "tools": [
                "pdflatex",
                "bibtex",
                "pdflatex",
                "pdflatex"
            ],
        },
    ],
    //文件清理.此属性必须是字符串数组
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.ist",
        "*.fls",
        "*.log",
        "*.fdb_latexmk",
    ],
    //设置为onFaild在构建失败后清除辅助文件
    "latex-workshop.latex.autoClean.run": "onFailed",
    //使用上次的recipe编译组合
    "latex-workshop.latex.recipe.default": "lastUsed",
    //用于反向同步的内部查看器的键绑定.ctrl/cmd+点击(默认)或双击
    "latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
    //设置VScode内部查看生成的pdf文件
    "latex-workshop.view.pdf.viewer": "browser",
    // 以下命令不经过 Shell 执行,直接由 VS Code 扩展宿主处理
    "terminal.integrated.commandsToSkipShell": [
        "kilo-code.new.agentManagerOpen",
        "kilo-code.new.agentManager.showTerminal"
    ],
    // 次级侧栏仅显示图标,不显示标签文字
    "workbench.secondarySideBar.showLabels": false,
    // Claude Code 面板放在底部 Panel 区域(非侧边栏)
    "claudeCode.preferredLocation": "panel",
    // 文件浏览器中删除文件不弹确认对话框
    "explorer.confirmDelete": false,
    // GitLens AI 使用 VS Code 内置模型(复用 Copilot 订阅)
    "gitlens.ai.model": "vscode",
    // 具体模型:Copilot 提供的 GPT-4.1
    "gitlens.ai.vscode.model": "copilot:gpt-4.1",
    // 关闭自动创建 Python 虚拟环境的提示
    "python.createEnvironment.trigger": "off",
    // 终端不继承系统环境变量,使用 VS Code 自身管理
    "terminal.integrated.inheritEnv": false,
    // 编辑器字体:JetBrains Mono 等宽字体优先
    "editor.fontFamily": "Jetbrains Mono, Consolas, 'Courier New', monospace",
    // 颜色主题:One Dark Pro
    "workbench.colorTheme": "One Dark Pro",
    // 聊天会话列表纵向堆叠排列
    "chat.viewSessions.orientation": "stacked",
    // Roo Cline 关闭调试日志
    "roo-cline.debug": false,
    // Roo Cline 允许自动执行的命令白名单(仅限只读 git 命令)
    "roo-cline.allowedCommands": [
        "git log",
        "git diff",
        "git show"
    ],
    // Roo Cline 拒绝执行的命令黑名单(空=仅白名单生效)
    "roo-cline.deniedCommands": [],
    // 忽略 Rebase 进行中的警告提示
    "git.ignoreRebaseWarning": true,
    // Kilo Code 关闭自动触发代码补全
    "kilo-code.new.autocomplete.enableAutoTrigger": false,
    // Kilo Code 自动批准 AI 操作(无需逐一确认)
    "kilo-code.new.autoApprove.enabled": true,
    // 文件浏览器拖拽文件不弹确认对话框
    "explorer.confirmDragAndDrop": false,
    // Jupyter Notebook 内核重启不弹确认对话框
    "jupyter.askForKernelRestart": false,
    // 不将模糊 Unicode 字符(形似 ASCII 的字符)高亮警告
    "editor.unicodeHighlight.ambiguousCharacters": false,
    // VS Code 启动时不打开任何编辑器(不恢复上次会话)
    "workbench.startupEditor": "none",
    // 不将不可见 Unicode 字符(零宽空格等)高亮警告
    "editor.unicodeHighlight.invisibleCharacters": false,
    // Kilo Code 不显示任务时间线面板
    "kilo-code.new.showTaskTimeline": false,
    // 不受信任的文件直接打开(不进入受限模式)
    "security.workspace.trust.untrustedFiles": "open",
    // 按文件类型控制 Copilot 启用范围
    "github.copilot.enable": {
        "*": true,            // 默认全部文件类型启用
        "plaintext": false,   // 纯文本禁用
        "markdown": false,    // Markdown 禁用(避免博客写作被干扰)
        "scminput": false,    // 源代码管理输入框(commit message)禁用
        "python": true        // Python 文件显式启用
    },
}
```

## 2. 分段讲解

### 2.1 GitHub Copilot —— AI 辅助编码

```json
"github.copilot.chat.localeOverride": "zh-CN",
```

将 Copilot Chat 的界面语言设置为简体中文.对于习惯中文交互的开发者,这条配置让对话更自然.

```json
// 生成 commit message 的指令
"github.copilot.chat.commitMessageGeneration.instructions": [
	{ "text": "始终使用简体中文生成 git commit message." },
	{ "text": "严格按照以下格式输出: <type>(<scope>): <subject>" },
	{ "text": "<body>" },
	{ "text": "<footer>"},
	{"text": "type, scope使用英文, 说明功能改变方向"},
	{ "text": "根据本次暂存改动的主要目的,选择最合适的一个 type,不要罗列多个." },
	{"text": "type和scope为必须填写的, 只能填写以下类型: feat, fix, docs, style, refactor, perf, test, chore, build, ci, revert"},
	{"text": "subject使用中文message, 简洁说明改动内容"},
	{"text": "body使用中文, 要写3条及以上, 可以选择'-m'或者'-'进行分隔不同行的内容"},
	{"text": "footer没有必要就不写, 但是关联issue, pr等就必须要写, 格式为'ref issue #123'或者'close issue #123'"},
	{ "text": "不要输出英文,不要附加解释,不要输出列表,只输出最终的 commit message." },
],
```

自定义 Copilot 生成 Git 提交信息的规则.这里配置了一套完整的中文 commit message 规范:

- **格式强制**:`<type>(<scope>): <subject>` + body + footer
- **type 限定**:只能从 `feat, fix, docs, style, refactor, perf, test, chore, build, ci, revert` 中选择一个
- **body 要求**:至少 3 条,用 `-` 分隔
- **footer 按需**:关联 issue/PR 时使用 `ref issue #123` 或 `close issue #123`

这套规则与 [[Engineering/PreCommitRuff|Pre-Commit Ruff 配置]] 中提到的提交规范是一致的.

```json
"github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "markdown": false,
    "scminput": false,
    "python": true
},
```

按文件类型控制 Copilot 的启用范围.这里在 Markdown 和纯文本中关闭了 Copilot,避免写技术博客时被 AI 补全干扰思路.`scminput` 关闭是因为 commit message 输入框已有自定义指令控制,无需补全.

```json
"github.copilot.nextEditSuggestions.enabled": true,
```

启用 Next Edit Suggestions 功能.Copilot 会预测你下一步可能编辑的位置,在相关行显示灰显建议.适合批量修改相似代码时快速跳转.

### 2.2 外观与编辑器

```json
"workbench.colorTheme": "One Dark Pro",
"workbench.iconTheme": "material-icon-theme",
"editor.fontFamily": "Jetbrains Mono, Consolas, 'Courier New', monospace",
```

三件套配置:

- **颜色主题**:One Dark Pro,经典暗色主题,对比度适中
- **图标主题**:Material Icon Theme,文件图标按类型区分,直观易辨
- **字体**:JetBrains Mono 优先,这是 JetBrains 推出的开源等宽字体,支持连字(ligatures);回退到 Consolas 和 Courier New

```json
"workbench.startupEditor": "none",
```

启动 VS Code 时不恢复上次的编辑器标签页,直接显示空白欢迎页.每次打开都是干净的状态,避免被之前的上下文干扰.

```json
"workbench.secondarySideBar.showLabels": false,
"editor.unicodeHighlight.ambiguousCharacters": false,
"editor.unicodeHighlight.invisibleCharacters": false,
```

- **次级侧栏不显示标签**:仅显示图标,节省水平空间
- **关闭 Unicode 模糊字符警告**:不将"看上去像 ASCII 但实际是 Unicode"的字符高亮标记
- **关闭不可见字符警告**:不标记零宽空格等不可见 Unicode 字符

### 2.3 终端配置

```json
"terminal.integrated.initialHint": false,
"terminal.integrated.inheritEnv": false,
```

- **关闭初始提示**:不在终端启动时显示"提示:使用 Ctrl+Shift+\` 切换终端"之类的引导文字
- **不继承系统环境变量**:VS Code 使用自身管理的环境变量,而非从父进程继承.这样做的好处是终端行为在不同机器上保持一致,避免本地环境变量污染

```json
"terminal.integrated.commandsToSkipShell": [
    "kilo-code.new.agentManagerOpen",
    "kilo-code.new.agentManager.showTerminal"
],
```

这两个 Kilo Code 插件命令不通过 Shell 执行,而是直接交由 VS Code 扩展宿主处理,避免在终端中产生多余的命令行输出.

### 2.4 Git 相关配置

```json
"git.confirmSync": false,
"git.ignoreRebaseWarning": true,
```

- **静默同步**:点击同步按钮不弹确认对话框,直接执行 push/pull
- **忽略 Rebase 警告**:进行中的 rebase 操作不再弹出警告提示.适合频繁使用 `git rebase` 的工作流

> [!WARNING] 注意
> `git.confirmSync: false` 意味着你不会再次确认推送操作.如果你有多个远程仓库或使用 force push,建议保持谨慎.

### 2.5 GitLens AI

```json
"gitlens.ai.model": "vscode",
"gitlens.ai.vscode.model": "copilot:gpt-4.1",
```

GitLens 的 AI 功能使用 VS Code 内置的 Copilot 模型,具体为 GPT-4.1.这样无需额外配置 API Key,直接复用 Copilot 订阅.GitLens 的 AI 主要用于生成 PR 描述、解释提交历史、分析代码变更等场景.

### 2.6 Python 开发

```json
"[python]": {
    "diffEditor.ignoreTrimWhitespace": true
},
"python.createEnvironment.trigger": "off",
```

- **差异编辑器忽略尾部空格**:Python 文件在对比差异时,不将行尾空格的增删视为变更.这在团队协作中很重要——不同编辑器对行尾空格的处理不同,忽略它们可以减少无意义的 diff
- **关闭虚拟环境创建提示**:不再每次打开 Python 项目时弹出"是否创建虚拟环境"的提示.虚拟环境我习惯手动管理

### 2.7 LaTeX Workshop —— 重点模块

LaTeX Workshop 是整个配置中最大的一块,包含编译工具链、编译方案、辅助文件清理、PDF 预览等完整设置.

#### 2.7.1 基础开关

```json
"latex-workshop.linting.chktex.enabled": false,
"latex-workshop.linting.lacheck.enabled": false,
"latex-workshop.latex.autoBuild.run": "never",
"latex-workshop.showContextMenu": true,
"latex-workshop.intellisense.package.enabled": true,
"latex-workshop.message.error.show": false,
"latex-workshop.message.warning.show": false,
```

| 配置项 | 值 | 说明 |
|:--|:--|:--|
| `linting.chktex` | `false` | 关闭 ChkTeX 语法检查,避免严格的 LaTeX 风格警告 |
| `linting.lacheck` | `false` | 关闭 LaCheck 检查,同上 |
| `autoBuild.run` | `never` | 不自动编译,手动触发.大文档编译耗时长,自动编译会频繁触发 |
| `showContextMenu` | `true` | 右键菜单显示 LaTeX 相关操作 |
| `intellisense.package` | `true` | 根据 `\usepackage` 声明自动补全命令和环境 |
| `message.error.show` | `false` | 编译错误不弹气泡 |
| `message.warning.show` | `false` | 编译警告不弹气泡 |

> [!TIP] 为什么关闭自动编译
> 对于包含大量参考文献和交叉引用的论文,一次完整编译(xelatex + bibtex + xelatex × 2)可能需要十几秒到几十秒.如果每次保存都自动编译,会严重影响编辑体验.手动触发更可控.

#### 2.7.2 编译工具链

```json
"latex-workshop.latex.outDir": "%DIR%/build",
```

编译产物(`.pdf`、`.aux`、`.log` 等)统一输出到 `.tex` 文件同级的 `build/` 目录.这样源码目录保持整洁,`.gitignore` 中只需忽略 `build/` 即可.

编译工具定义了四个:

| 工具名 | 命令 | 用途 |
|:--|:--|:--|
| `xelatex` | `xelatex` | 支持 UTF-8 和系统字体,中文 LaTeX 首选 |
| `pdflatex` | `pdflatex` | 传统 LaTeX 编译器,不支持 UTF-8 直接输入 |
| `latexmk` | `latexmk` | 自动化编译工具,自动判断需要编译几次 |
| `bibtex` | `bibtex` | 处理参考文献数据库 |

所有工具都带以下通用参数:

- `-synctex=1`:启用 SyncTeX,支持 PDF 到源码的反向搜索
- `-interaction=nonstopmode`:遇错不停止,继续编译完(方便在问题面板中一次性查看所有错误)
- `-file-line-error`:错误信息使用 `文件:行号` 格式,点击可跳转

#### 2.7.3 编译方案

```json
"latex-workshop.latex.recipes": [
    { "name": "XeLaTeX", "tools": ["xelatex"] },
    { "name": "xelatex->bibtex->xelatex*2", "tools": ["xelatex", "bibtex", "xelatex", "xelatex"] },
    // ...
],
```

六种编译方案覆盖不同场景:

| 方案 | 适用场景 |
|:--|:--|
| `XeLaTeX` | 纯中文文档,无参考文献 |
| `PDFLaTeX` | 纯英文文档,无参考文献 |
| `BibTeX` | 单独处理参考文献 |
| `LaTeXmk` | 自动化编译(自动判断次数) |
| `xelatex → bibtex → xelatex × 2` | 中文文档含参考文献,完整编译链 |
| `pdflatex → bibtex → pdflatex × 2` | 英文文档含参考文献,完整编译链 |

> [!NOTE] xelatex × 2 的必要性
> 第一次 xelatex 生成 `.aux`(记录引用标签),bibtex 读取 `.aux` 生成 `.bbl`(参考文献格式),第二次 xelatex 将参考文献写入 PDF,第三次 xelatex 解析交叉引用的页码.所以含参考文献的文档至少需要两次 xelatex.

#### 2.7.4 辅助文件清理

```json
"latex-workshop.latex.clean.fileTypes": [
    "*.aux", "*.bbl", "*.blg", "*.idx", "*.ind",
    "*.lof", "*.lot", "*.out", "*.toc", "*.acn",
    "*.acr", "*.alg", "*.glg", "*.glo", "*.gls",
    "*.ist", "*.fls", "*.log", "*.fdb_latexmk",
],
"latex-workshop.latex.autoClean.run": "onFailed",
```

编译过程中会产生大量辅助文件.`clean.fileTypes` 列出了 18 种需要清理的扩展名.`autoClean.run: onFailed` 表示仅在编译失败时自动清理——编译成功则保留,方便调试.

#### 2.7.5 PDF 预览

```json
"latex-workshop.latex.recipe.default": "lastUsed",
"latex-workshop.view.pdf.internal.synctex.keybinding": "double-click",
"latex-workshop.view.pdf.viewer": "browser",
```

- **默认编译方案**:使用上次选择的 recipe,避免每次切换
- **反向搜索**:双击 PDF 中的文字跳转到 `.tex` 源码对应位置(依赖 SyncTeX)
- **PDF 查看器**:在外部浏览器打开,而非 VS Code 内置标签页.浏览器查看 PDF 体验更好(缩放、搜索、书签),且可以在另一块屏幕上独立显示

### 2.8 AI 编码助手(Roo Cline / Kilo Code / Claude Code)

```json
"roo-cline.debug": false,
"roo-cline.allowedCommands": [
    "git log",
    "git diff",
    "git show"
],
"roo-cline.deniedCommands": [],
```

Roo Cline 的命令白名单仅开放三个只读 Git 命令.AI 代理无法执行任何写入或删除操作,安全性最高.`debug: false` 关闭调试日志,减少终端噪音.

```json
"kilo-code.new.autocomplete.enableAutoTrigger": false,
"kilo-code.new.autoApprove.enabled": true,
"kilo-code.new.showTaskTimeline": false,
```

Kilo Code 的配置比较激进:关闭自动触发补全(手动触发更可控),但开启了自动批准(AI 操作无需逐一确认).任务时间线面板关闭,节省界面空间.

```json
"claudeCode.preferredLocation": "panel",
```

Claude Code 放在底部 Panel 区域,而非占用侧边栏.Panel 区域可以随时折叠,不影响编辑器布局.

> [!WARNING] 注意
> `kilo-code.new.autoApprove.enabled: true` 意味着 AI 代理的终端命令和文件修改无需逐一确认.请确保你信任 AI 代理的操作范围,或像 Roo Cline 那样配置白名单限制.

### 2.9 文件浏览器与安全

```json
"explorer.confirmDelete": false,
"explorer.confirmDragAndDrop": false,
"security.workspace.trust.untrustedFiles": "open",
```

- **删除免确认**:直接在文件浏览器中删除文件,不弹对话框
- **拖拽免确认**:拖拽移动文件不弹确认
- **不受信任文件直接打开**:不进入受限模式.个人项目环境下这个配置是安全的;如果经常打开外部代码,建议保持默认的受限模式

### 2.10 Jupyter

```json
"jupyter.askForKernelRestart": false,
```

重启 Jupyter Notebook 的 Python 内核时不再弹确认对话框.频繁调试模型时这条配置很实用——每次修改代码都要重启内核,省去一次点击.

### 2.11 扩展功能

```json
"punctuationConverter.enabled": true,
"punctuationConverter.customMap": {
    "～": "~",
    "￥": "$"
},
"chat.mcp.gallery.enabled": true,
```

- **标点转换**:自动将中文输入法误打的全角标点转为半角.`～` → `~`(路径分隔符)、`￥` → `$`(Shell 变量前缀)是两个最常见的场景
- **MCP Gallery**:在聊天面板中启用 MCP 服务器市场,方便浏览和安装 MCP 扩展

## 3. 配置哲学

回顾整套配置,可以看出几个一致的设计思路:

| 原则 | 体现 |
|:--|:--|
| **减少弹窗干扰** | 删除确认、同步确认、Python 环境提示、Jupyter 重启确认全部关闭 |
| **手动控制编译** | LaTeX 不自动编译,编译方案手动选择 |
| **安全优先** | Roo Cline 只开放只读命令;Copilot 在 Markdown 中禁用 |
| **中文友好** | Copilot 界面中文,commit message 中文规范,LaTeX 用 XeLaTeX |
| **界面简洁** | 启动空白页、侧栏仅图标、终端无提示、调试日志关闭 |

> [!SUMMARY] 总结
> VS Code 的 `settings.json` 是开发者日常使用频率最高的配置文件之一.一个好的配置应该**减少不必要的交互、保持编辑体验流畅、根据语言和工具特性做针对性优化**.本文的配置经过长期迭代打磨,可以作为个人配置的参考起点——但最合适的配置一定是你自己用着最顺手的那一套.

