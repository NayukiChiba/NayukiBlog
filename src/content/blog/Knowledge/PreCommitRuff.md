---
title: 在Commit前进行代码格式化
date: 2026-01-26
category: 技术
tags:
  - 工具
  - 高级教程
  - git
description: 在Commit的时候，自动进行ruff格式化代码
image: https://img.yumeko.site/file/blog/Ruff.png
status: public
---
# 安装pre-commit

```bash
pip install pre-commit
```

# 创建流程文件
在根目录中创建文件`pre-commit-config.yaml`

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format
        name: ruff (format)
        entry: ruff format
        language: python
        types: [python]
        additional_dependencies: [ruff]
      - id: ruff
        name: ruff (lint)
        entry: ruff check
        language: python
        types: [python]
        additional_dependencies: [ruff]
```

# pre-commit加载文件

```bash
pre-commit install
```

然后我们每次commit，就会自动看见ruff格式化了

# 直接使用 Ruff 命令行

你也可以在终端中直接运行 Ruff 命令来格式化代码。

**基本命令：**

- **格式化一个或多个文件**：
    
    ```bash
    ruff format path/to/your_file.py
    ```
    
- **格式化整个目录（如当前目录）**：
    
    ```bash
    ruff format .
    ```
    
- **检查哪些文件会被格式化（预览模式）**：
    
    ```bash
    ruff format --check .
    ```
    

# 总结

- **自动化流程**：使用 `pre-commit` 钩子，一劳永逸。
- **手动执行**：直接使用 `ruff format` 命令，灵活控制。

你的笔记已经包含了完整的 `pre-commit` 配置流程，按照它操作即可。