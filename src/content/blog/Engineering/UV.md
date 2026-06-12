---
title: uv 管理 Python 项目依赖完整指南
date: 2026-02-09
category: 工程实践
tags:
  - 工具
  - python
  - 包管理
description: uv 管理 Python 项目依赖,virtual env 和 lock 文件
image: https://img.yumeko.site/file/blog/cover/1780581876594.webp
status: published
---

[uv](https://docs.astral.sh/uv/) 是 Astral 团队用 Rust 编写的 Python 包管理器，可以替代 `pip`、`pip-tools`、`pipreqs`、`virtualenv` 等多个工具。它的核心优势是速度——比 pip 快 10-100 倍，同时提供了现代化的依赖管理体验。

> [!NOTE] 为什么从 pipreqs 迁移到 uv
> `pipreqs` 只能生成 `requirements.txt`，但 uv 提供了完整的项目生命周期管理：虚拟环境创建、依赖添加/删除/同步、lock 文件、依赖树分析、工具安装。一个工具覆盖所有场景。

## 1. 安装 uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用 pip 安装（备选）
pip install uv
```

安装后验证：

```bash
uv --version
```

## 2. 项目初始化

### 创建新项目

```bash
uv init my-project        # 创建目录并初始化
uv init                   # 在当前目录初始化
```

`uv init` 生成的 `pyproject.toml` 比较简略，实际项目中推荐补全以下内容：

```toml
# --------------- 项目信息 ---------------
[project]
name = "my-project"         # 项目名（通常与包名相同）
version = "0.1.0"
description = "项目描述"     # 简要说明项目功能
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "YourName" },
]
keywords = ["关键词1", "关键词2"]  # 帮助 PyPI 搜索
dependencies = []            # 运行时依赖（uv add 写入此处）

# 可选依赖（兼容 pip install ".[dev]"）
[project.optional-dependencies]
dev = [
    "pytest>=9.0.3",
    "ruff>=0.15.12",
    "pre-commit>=4.6.0",
]

# --------------- PEP 735 依赖组 ---------------
# uv add --dev 写入此处，uv sync 自动安装
[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "ruff>=0.15.12",
    "pre-commit>=4.6.0",
    "ipykernel>=7.2.0",
    "nbstripout>=0.9.1",
]

# --------------- UV 镜像源配置 ---------------
# 默认镜像源
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# PyTorch 专用源（explicit = true 表示仅指定时才使用）
[[tool.uv.index]]
name = "pytorch-cu128-cn"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128-original"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

# 将特定包绑定到指定索引
[tool.uv.sources]
torch = { index = "pytorch-cu128-original" }
torchvision = { index = "pytorch-cu128-original" }
torchaudio = { index = "pytorch-cu128-original" }

# --------------- UV 配置 ---------------
[tool.uv]
# Python 安装策略：managed（uv 自动下载管理）/ system（使用系统 Python）
python-preference = "managed"
# 约束文件，用于限制特定包的版本上限
# constraint-dependencies = []

# --------------- Ruff 配置 ---------------
[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501", "F841"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

> [!NOTE] `optional-dependencies` vs `dependency-groups`
> `[project.optional-dependencies]` 是 PyPA 标准，兼容 `pip install ".[dev]"`。`[dependency-groups]` 是 PEP 735 标准，uv 原生支持，语法更清晰。**两者都写**即可兼容两种生态。

### 迁移已有项目

```bash
cd existing-project
uv init                    # 生成 pyproject.toml

# 从已有 requirements.txt 导入依赖
uv add -r requirements.txt
```

## 3. 依赖管理核心操作

### 添加依赖

```bash
uv add requests                          # 写入 dependencies
uv add "django>=4.2,<5.0"               # 指定版本范围
uv add --dev pytest pytest-cov           # 写入 dependency-groups.dev
uv add --group docs mkdocs-material      # 写入 dependency-groups.docs

# 从指定镜像源安装（需先在 pyproject.toml 中声明 [[tool.uv.index]]）
uv add torch --index pytorch-cu128-original
```

### 移除依赖

```bash
uv remove requests
uv remove --dev pytest
uv remove --group docs mkdocs
```

### 同步环境

```bash
uv sync                    # 安装所有依赖（含 dev）
uv sync --no-dev           # 仅安装生产依赖
uv sync --group docs       # 安装指定依赖组
```

`uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 同步虚拟环境，安装缺少的包、移除多余的包。

## 4. 生成依赖文件

### uv.lock —— 精确锁定版本

```bash
uv lock                    # 生成/更新 uv.lock
```

`uv.lock` 是跨平台的锁文件，记录所有依赖的精确版本和哈希值，确保不同环境下安装结果一致。**应提交到 git**。

### 导出为 requirements.txt

```bash
uv export --format requirements-txt > requirements.txt
uv export --no-dev --format requirements-txt > requirements.txt         # 仅生产依赖
uv export --no-hashes --format requirements-txt > requirements.txt      # 不含哈希
```

> [!TIP] 场景选择
> `uv.lock` 用于开发环境的确定性构建，`requirements.txt` 用于 CI/CD、Docker 镜像等需要传统格式的场景。

### 查看依赖树

```bash
uv tree                    # 完整依赖树
uv tree --depth 2          # 限制深度
uv tree --outdated         # 显示可更新的包
uv tree --no-dev           # 仅生产依赖
```

## 5. 虚拟环境管理

### 自动创建

uv 会在项目根目录自动创建 `.venv/`，无需手动操作：

```bash
uv sync                   # 自动创建 .venv 并安装依赖
uv run python main.py     # 在虚拟环境内运行
```

### 手动管理

```bash
uv venv                            # 创建虚拟环境
uv venv .venv --python 3.12        # 指定 Python 版本
uv venv --python pypy@3.10         # 使用 PyPy

# 删除重建
rm -rf .venv && uv sync
```

### 激活环境

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 或者直接用 uv run（无需手动激活）
uv run python script.py
uv run pytest
uv run pre-commit run --all-files
```

## 6. 全局工具管理

uv 可以像 `pipx` 一样安装和管理全局 CLI 工具：

```bash
uv tool install ruff             # 安装 ruff（独立环境）
uv tool install pre-commit
uv tool install cookiecutter

uv tool list                     # 查看已安装工具
uv tool upgrade ruff             # 升级工具
uv tool uninstall ruff           # 卸载工具

# 通过 uvx 直接运行（不安装）
uvx ruff check .
uvx pre-commit run --all-files
```

## 7. pip 兼容模式

需要和传统 pip 工作流兼容时使用 `uv pip` 子命令：

```bash
uv pip install numpy             # pip install 等效
uv pip install -r requirements.txt
uv pip list                      # 查看已安装包
uv pip freeze > requirements.txt # 冻结当前环境
uv pip uninstall numpy
```

> [!NOTE] 推荐用法
> 新项目优先使用 `uv add` / `uv sync` 管理依赖，`uv pip` 仅在迁移旧项目或调试时使用。

## 8. 实际工作流

### 完整项目流程

```bash
# 1. 创建项目
uv init my-project
cd my-project

# 2. 添加依赖
uv add numpy pandas scikit-learn
uv add --dev pytest ruff pre-commit

# 3. 同步环境
uv sync

# 4. 开发和运行
uv run python main.py
uv run pytest
uv run ruff check .

# 5. 导出 production 依赖（用于部署）
uv export --no-dev --format requirements-txt > requirements.txt
```

### 从 pipreqs 迁移

```bash
# pipreqs 工作流（旧）
pip install pipreqs
pipreqs ./ --encoding=utf-8 --force --ignore venv
pip install -r requirements.txt

# uv 工作流（新）
uv init
uv add -r requirements.txt    # 导入已有依赖
rm requirements.txt            # 不再需要
uv sync                       # 安装所有依赖
```

## 9. Pre-commit 集成

在 `pre-commit-config.yaml` 中添加 hook，在提交前自动更新 `requirements.txt`：

```yaml
repos:
  - repo: local
    hooks:
      - id: uv-export
        name: uv export (generate requirements.txt)
        entry: bash -c 'uv export --no-dev --no-hashes --format requirements-txt > requirements.txt && git add requirements.txt'
        language: system
        pass_filenames: false
        files: ^(pyproject\.toml|uv\.lock)$
```

对于想校验 lock 文件是否最新的场景：

```yaml
  - repo: local
    hooks:
      - id: uv-lock-check
        name: uv lock check
        entry: bash -c 'uv lock --locked'
        language: system
        pass_filenames: false
```

## 10. 最佳实践

| 做法 | 说明 |
|:--|:--|
| **提交 `uv.lock` 到 git** | 确保团队和 CI 使用相同的依赖版本 |
| **用 `uv add` 代替手动编辑 `pyproject.toml`** | 避免版本号冲突和格式错误 |
| **开发依赖用 `--dev`** | 生产环境 `uv sync --no-dev` 或 `uv export --no-dev` 可排除 |
| **`uv run` 代替手动激活 venv** | 一条命令搞定，不用记 activate 路径 |
| **CI 中优先使用 `uv`** | Dockerfile 中 `COPY pyproject.toml uv.lock . && uv sync --no-dev` 比 pip install 快得多 |

### Docker 示例

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY . .
CMD ["uv", "run", "python", "main.py"]
```

## 11. 镜像源与索引配置

### pyproject.toml 声明镜像源（推荐）

命令行指定镜像源是临时方案，团队协作时应写入 `pyproject.toml`，确保所有人使用相同的源：

```toml
# 默认源（未匹配到专用源时回退到此处）
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# 专用源：explicit = true 表示仅在显式指定时才使用
# 用法：uv add torch --index pytorch-cu128-original
[[tool.uv.index]]
name = "pytorch-cu128-original"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

# 将特定包绑定到指定索引（无需每次手动 --index）
[tool.uv.sources]
torch = { index = "pytorch-cu128-original" }
torchvision = { index = "pytorch-cu128-original" }
torchaudio = { index = "pytorch-cu128-original" }
```

| 字段 | 作用 |
|:--|:--|
| `default = true` | 设为默认源，所有包优先从这里下载 |
| `explicit = true` | 仅在 `--index <name>` 或 `sources` 指定时才使用，避免误伤 |
| `[tool.uv.sources]` | 将包永久绑定到某个索引，`uv add torch` 即可自动走对应源 |

### 命令行临时指定

```bash
uv add numpy --index-url https://mirrors.aliyun.com/pypi/simple/
uv add torch --index pytorch-cu128-original   # 需先在 toml 中声明
```

### 环境变量全局配置

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置优先级

`pyproject.toml` 中的 `[[tool.uv.index]]` > 环境变量 `UV_INDEX_URL` > 命令行 `--index-url` > 官方 PyPI。

### 解决依赖冲突

```bash
uv lock --resolution lowest          # 使用最低版本解析
uv lock --resolution lowest-direct   # 仅直接依赖使用最低版本
uv tree --invert -p <package>        # 查看哪些包依赖了某包
```

### 触发 lock 文件更新

```bash
uv lock                    # 手动触发
uv lock --upgrade          # 忽略 lock 文件重新解析
uv lock --upgrade-package numpy  # 仅升级指定包
```

### 离线使用

```bash
uv sync --frozen           # 严格按 uv.lock 安装，不联网
uv add <pkg> --frozen      # 添加包但不重新解析所有依赖
```

---