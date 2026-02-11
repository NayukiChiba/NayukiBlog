---
title: pipreqs 自动生成 Python 项目依赖文件完整指南
date: 2026-02-09
category: 技术
tags:
  - 工具
description: pipreqs 工具的完整使用指南，包括安装、基本用法、高级参数、最佳实践和常见问题解决方案
image: https://img.yumeko.site/file/blog/PipReqs.jpg
status: published
---
使用 `pipreqs` 自动生成 `requirements.txt` 的完整指南：

## 1. 安装 pipreqs

```bash
pip install pipreqs
```

## 2. 基本用法

### 为当前目录生成 requirements.txt
```bash
pipreqs ./
```

### 为指定路径生成
```bash
pipreqs /path/to/your/project
```

## 3. 常用参数选项

### 强制覆盖现有文件
```bash
pipreqs ./ --force
```

### 指定编码（解决中文路径/注释问题）
```bash
pipreqs ./ --encoding=utf-8
```

### 忽略特定目录
```bash
pipreqs ./ --ignore venv,test,tests
```

### 生成 requirements.txt 到指定路径
```bash
pipreqs ./ --savepath requirements/prod.txt
```

### 只生成当前目录（不递归子目录）
```bash
pipreqs ./ --no-pin  # 不固定版本号
```

### 使用本地模式（检查已安装的包版本）
```bash
pipreqs ./ --use-local
```

### 指定排除模式
```bash
pipreqs ./ --exclude venv,*.pyc
```

## 4. 实际使用示例

### 示例项目结构
```
my_project/
├── app/
│   ├── main.py
│   └── utils.py
├── tests/
│   └── test_app.py
└── venv/
```

### 生成项目依赖
```bash
cd my_project
pipreqs ./ --encoding=utf-8 --force --ignore venv,tests
```

## 5. 与 pip freeze 的对比

| 特性 | pipreqs | pip freeze |
|------|---------|------------|
| 依赖来源 | 分析 import 语句 | 已安装的包 |
| 环境隔离 | 好（只分析项目代码） | 差（包含所有包） |
| 精确性 | 高（只包含实际使用的） | 低（包含所有包） |
| 虚拟环境 | 推荐使用 | 必须在虚拟环境中使用 |

## 6. 高级用法

在`pre-commit-config.yaml`中，在commit之前自动生成`requirements.txt`

```yaml
repos:
  - repo: local
    hooks:
    - id: pipreqs
        name: pipreqs (generate requirements.txt)
        entry: bash -c 'pipreqs . --force --encoding=utf-8 --mode no-pin'
        language: system
        pass_filenames: false
        types: [python]
```

## 7. 最佳实践建议

1. **在虚拟环境中使用**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

pipreqs ./ --encoding=utf-8
```

2. **清理旧文件**
```bash
pipreqs ./ --force --ignore venv,__pycache__,*.pyc
```

3. **验证生成的依赖**
```bash
# 检查生成的 requirements.txt
cat requirements.txt

# 测试安装
pip install -r requirements.txt
```

4. **结合使用（完整工作流）**
```bash
# 1. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装开发依赖
pip install pipreqs

# 3. 生成 requirements.txt
pipreqs ./ --encoding=utf-8 --force --ignore venv

# 4. 在新环境中测试
deactivate
rm -rf venv
python -m venv new_venv
source new_venv/bin/activate
pip install -r requirements.txt
```

## 8. 常见问题解决

### 编码问题
```bash
# 如果遇到编码错误
pipreqs ./ --encoding=gbk  # Windows 中文系统
```

### 忽略隐藏文件
```bash
pipreqs ./ --exclude=".*"
```