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