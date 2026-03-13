---
title: Todo管理器
date: 2026-03-03
category: Projects
tags:
  - Python
description: Python的小项目，Todo管理器
image: https://img.yumeko.site/file/blog/Todo.png
status: published
---
# TODO 事务管理（CLI）

一个基于 Python 的命令行 TODO 事务管理项目，使用 JSON 文件持久化数据。

## 项目目标

- 使用命令行（CLI）进行 TODO 事项管理
- 使用 JSON 作为存储介质
- 数据文件统一放在 `data/` 目录
- 源代码统一放在 `src/` 目录

## 规划中的目录结构

```text
TodoManage/
├─ data/
│  └─ todos.json              # TODO 数据文件
├─ src/
│  ├─ main.py                 # CLI 入口
│  ├─ todo_service.py         # 事务逻辑（增删改查、状态流转）
│  ├─ storage.py              # JSON 读写封装
│  └─ models.py               # 数据模型定义
├─ README.md
├─ pyproject.toml
└─ requirements.txt
```

## 功能范围（CLI）

第一阶段先聚焦基础能力：

- 新增待办事项
- 查看全部事项
- 按状态筛选（待处理 / 进行中 / 已完成）
- 更新事项内容
- 标记完成
- 删除事项

可选扩展：

- 截止日期与优先级
- 标签分类
- 关键字搜索
- 统计报表（完成率、逾期数）

## 数据格式（JSON）

`data/todos.json` 预计示例：

```json
[
  {
    "id": "1",
    "title": "完成 README",
    "description": "补充项目说明",
    "status": "todo",
    "priority": "medium",
    "created_at": "2026-03-10T12:00:00+08:00",
    "updated_at": "2026-03-10T12:00:00+08:00",
    "due_date": null,
    "tags": ["docs"]
  }
]
```

状态建议使用：

- `todo`
- `in_progress`
- `done`

## CLI 交互草案

```bash
python -m src.main add "学习 Python" --desc "完成基础语法"
python -m src.main list
python -m src.main list --status todo
python -m src.main update 1 --title "学习 Python CLI"
python -m src.main done 1
python -m src.main delete 1
```

## 开发说明（当前阶段）

当前仅完成 README 规划文档，代码结构和实现将按上述目录推进。
