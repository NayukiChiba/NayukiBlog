---
title: 天气查询
date: 2026-03-01
category: Projects
tags:
  - Python
description: 访问OpenWeather的API，获取天气信息
image: https://img.yumeko.site/file/blog/WeatherSearch.webp
status: published
---
# 天气预报查询工具（Weather Search）

一个基于 Python 的天气查询项目模板：输入城市名后，返回**实时天气**与**未来预报**。

项目重点覆盖：网络请求、JSON 解析、API 密钥管理、命令行参数处理，并支持后续扩展为 Tkinter 图形界面。

---

## 功能简介

- 按城市名查询实时天气
- 查询未来天气预报（可配置天数）
- 友好的命令行输出
- API Key 通过环境变量管理（避免硬编码）

---

## 技术栈

- Python 3.12+
- `requests`（HTTP 请求）
- `argparse`（命令行参数）
- `json`（数据解析）
- `os` / 环境变量（密钥管理）

---

## 推荐项目结构

当前仓库是轻量模板，建议按以下结构逐步完善：

```text
weather_app/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ main.py
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ client.py
│  ├─ parser.py
│  ├─ formatter.py
│  ├─ cli.py
│  └─ errors.py
├─ tests/
│  ├─ test_parser.py
│  ├─ test_formatter.py
│  └─ test_client.py
└─ assets/
   └─ icons/
```

### 模块职责建议

- `main.py`：程序入口，串联整体流程
- `src/config.py`：配置读取（如 API Key）
- `src/client.py`：调用天气 API
- `src/parser.py`：解析并标准化 API JSON 数据
- `src/formatter.py`：输出格式化（命令行展示）
- `src/cli.py`：命令行参数定义与解析
- `src/errors.py`：自定义异常与错误分类

---

## API 选型（OpenWeatherMap）

可使用免费天气 API：OpenWeatherMap。

常用接口：
- Current Weather（实时天气）
- 5 day / 3 hour forecast（未来预报）

常见请求参数：
- `q=城市名`
- `appid=你的API_KEY`
- `units=metric`（摄氏度）
- `lang=zh_cn`（中文天气描述）

---

## 命令行参数设计（建议）

- `--city`：城市名（必填）
- `--days`：预报天数（默认 3）
- `--lang`：语言（默认 `zh_cn`）

示例：

```bash
python main.py --city Beijing --days 3 --lang zh_cn
```

---

## API Key 管理

请勿把 API Key 写死在源码中。

推荐方式：
1. 环境变量（推荐）
2. `.env` 文件（配合 `python-dotenv`）

`.env.example` 示例：

```env
OPENWEATHER_API_KEY=your_api_key_here
```

在代码中读取（示意）：

```python
import os

api_key = os.getenv("OPENWEATHER_API_KEY")
if not api_key:
    raise ValueError("未检测到 OPENWEATHER_API_KEY，请先配置环境变量")
```

---

## 输出示例

```text
城市：北京
当前：12°C，多云，湿度 45%，风速 3.2m/s

未来预报：
- 03-13：8~15°C，小雨
- 03-14：6~13°C，阴
- 03-15：7~16°C，多云
```

---

## 错误处理建议

- 网络错误：连接失败、超时
- API 错误：城市不存在、Key 无效、限流
- 解析错误：字段缺失、格式变化

可按错误类型输出清晰提示，提升可用性。

---

## 拓展方向

- 使用 Tkinter 构建图形界面
- 显示天气图标（根据 icon code 拉取图标）
- 支持历史查询记录
- 增加单元测试与 CI

---

## 学习目标对应

本项目可系统练习：

- 网络请求（`requests`）
- JSON 解析
- API 密钥管理
- 命令行参数处理
- 从 CLI 向 GUI 演进的工程化设计
