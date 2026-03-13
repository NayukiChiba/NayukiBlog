---
title: 爬虫豆瓣
date: 2026-03-07
category: Projects
tags:
  - Python
description: 使用requests获取静态网页信息，筛选出Top250电影信息
image: https://img.yumeko.site/file/blog/Douban.webp
status: published
---
# ReptileDouban（requests 抓取豆瓣 Top250）

这是一个**学习型项目**：用 [`requests.get()`](main.py:1) 抓取豆瓣电影 Top250，并完成解析与导出。

> 仅用于学习，请遵守网站规则与法律法规。

---

## 项目架构设计（重点）

推荐采用「入口层 + 业务层 + 基础设施层 + 配置层 + 测试层」：

1. **入口层**：负责启动与参数接收
2. **业务层**：负责抓取流程编排
3. **基础设施层**：负责 HTTP、解析、存储等可复用能力
4. **配置层**：统一管理常量、请求头、输出路径
5. **测试层**：保证解析与流程稳定

---

## 推荐文件命名与职责

```text
.
├─ main.py
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ crawler.py
│  ├─ parser.py
│  ├─ storage.py
│  └─ service.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_parser.py
│  └─ test_service.py
└─ output/
   └─ douban_top250.csv
```

### 文件说明

- [`main.py`](main.py)：程序入口，调用 [`service.py`](src/service.py)
- [`src/config.py`](src/config.py)：基础配置（BASE_URL、HEADERS、超时、间隔）
- [`src/models.py`](src/models.py)：数据模型（如 `Movie`）
- [`src/crawler.py`](src/crawler.py)：网络请求封装（重试、异常、状态码）
- [`src/parser.py`](src/parser.py)：HTML 解析与字段提取
- [`src/storage.py`](src/storage.py)：CSV/JSON 存储
- [`src/service.py`](src/service.py)：业务编排（分页循环、调用爬取+解析+保存）
- [`tests/test_parser.py`](tests/test_parser.py)：解析逻辑测试
- [`tests/test_service.py`](tests/test_service.py)：流程测试
- [`output/douban_top250.csv`](output/douban_top250.csv)：输出结果

---

## 最小运行说明

安装依赖（先补充 [`requirements.txt`](requirements.txt)）：

```bash
pip install requests beautifulsoup4 lxml
```

运行：

```bash
python main.py
```

---

## 你可以按这个顺序学习

1. 先写 [`src/crawler.py`](src/crawler.py)：单页请求成功
2. 再写 [`src/parser.py`](src/parser.py)：提取电影字段
3. 然后写 [`src/service.py`](src/service.py)：循环 10 页
4. 最后写 [`src/storage.py`](src/storage.py)：导出 CSV

这样结构清晰，后面扩展代理、重试、日志、数据库都会更容易。
