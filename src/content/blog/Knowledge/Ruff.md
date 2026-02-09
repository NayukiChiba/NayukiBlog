---
title: Ruff常用命令指南
date: 2026-02-01
category: 技术
tags:
  - Python
  - 工具
description: 详细介绍Ruff代码格式化工具的常用命令，包括代码格式化、代码检查、配置管理等功能的完整命令参考
image: https://img.yumeko.site/file/blog/Ruff.png
status: public
---
Ruff 的常用命令主要分为两大类：**代码格式化** 和 **代码检查**。

## 1. 代码格式化命令

### 基本格式化
- `ruff format <path>` - 格式化指定文件或目录
  ```bash
  ruff format .              # 格式化当前目录所有Python文件
  ruff format src/           # 格式化src目录
  ruff format main.py        # 格式化单个文件
  ```

### 检查模式
- `ruff format --check <path>` - 检查哪些文件需要格式化（不实际修改）
  ```bash
  ruff format --check .      # 检查当前目录
  ```

### 其他选项
- `--diff` - 显示格式化前后的差异
- `--config` - 指定配置文件路径

## 2. 代码检查命令

### 基本检查
- `ruff check <path>` - 检查代码问题
  ```bash
  ruff check .               # 检查当前目录
  ruff check --fix .         # 检查并自动修复可修复的问题
  ```

### 修复模式
- `ruff check --fix <path>` - 检查并自动修复
- `ruff check --fix-only <path>` - 只修复，不显示其他问题

### 特定规则
- `ruff check --select <rule>` - 只检查特定规则
  ```bash
  ruff check --select E501 .  # 只检查行长度问题
  ruff check --select F .     # 只检查pyflakes规则
  ```

## 3. 配置相关命令

### 生成配置文件
- `ruff init` - 在当前目录生成 `ruff.toml` 配置文件

### 列出规则
- `ruff rule` - 列出所有可用规则
- `ruff rule <rule-code>` - 查看特定规则的详细信息
  ```bash
  ruff rule E501            # 查看E501规则的详细信息
  ```

## 4. 常用组合命令

### 格式化并检查
```bash
ruff format . && ruff check .
```

### 检查并自动修复
```bash
ruff check --fix .
```

### 只检查特定目录
```bash
ruff check src/ tests/
```

## 5. 实用示例

```bash
# 1. 初始化配置
ruff init

# 2. 格式化整个项目
ruff format .

# 3. 检查并修复所有问题
ruff check --fix .

# 4. 只检查导入相关的问题
ruff check --select I .

# 5. 预览格式化差异
ruff format --diff .
```

## 常用规则类别
- `E` / `W` - PEP 8 风格问题
- `F` - Pyflakes 检查（未使用变量等）
- `I` - 导入排序
- `B` - Bugbear（潜在bug）
- `C` - 复杂度检查
- `UP` - pyupgrade（Python版本升级建议）

## CI/CD检查代码格式

### 项目规范
在`pyproject.toml`中
* 指定目标 Python 版本为 3.12，确保代码语法兼容性
* 设置代码行最大长度为 88 字符（这是 Black 格式化工具的默认值）
- `select = ["E", "F", "I", "UP"]`：启用以下规则集：
    
    - `"E"`：pycodestyle 的错误规则（PEP 8 规范检查）
        
    - `"F"`：pyflakes 规则（逻辑错误检查）
        
    - `"I"`：isort 规则（导入语句排序）
        
    - `"UP"`：pyupgrade 规则（自动升级到新 Python 语法）
* 忽略行长度限制的错误（E501）
* 使用双引号 `"` 而不是单引号
* 使用空格缩进（不是制表符）
* 自动检测并使用合适的行结束符（Windows 用 `\r\n`，Unix 用 `\n`）
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

### github流水线
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

你可以通过 `ruff rule` 查看完整的规则列表和说明。