---
title: Python 异步编程
date: 2026-01-26
category: PythonAdvanceUsage
tags:
  - Python
  - 高级教程
description: Python 异步编程 (Asyncio) 的实战指南，涵盖协程、Task、异步上下文管理器、异步迭代器、HTTP/文件/数据库操作、队列、锁、信号量等核心概念，并包含流程图和代码示例。
image: https://img.yumeko.site/file/blog/Async.png
status: published
---
# 深入理解 Python 异步编程 (Asyncio) 实战指南

## 简介

在现代 Python 开发中，异步编程（Asynchronous Programming）是处理高并发、I/O 密集型任务（如网络请求、数据库操作、文件读写）的关键技术。

与传统的多线程相比，它使用单线程和事件循环（Event Loop）来管理任务，减少了上下文切换的开销，极大提升了程序效率。

### 核心机制：事件循环 (Event Loop) 工作流

理解异步的关键在于理解“事件循环”。它就像一个不知疲倦的调度员，不断地从任务队列中取出任务执行，遇到 I/O 等待时就切换到下一个任务。

![WorkFlow](https://img.yumeko.site/file/articles/PythonAdvanceUsage/Async/WorkFlow.svg)

---

## 1. 基础概念：定义与调用

在 Python 中，核心关键字是 `async def`（定义协程）和 `await`（挂起并等待结果）。

### 示例代码

```python
async def say_after(delay, what):
    await asyncio.sleep(delay) # 非阻塞等待
    print(what)

async def basic_async_example():
    # 顺序执行：总耗时约为 1s + 2s = 3s
    await say_after(1, "Hello")
    await say_after(2, "World")
```

**解析**：
`await` 就像一个“交出控制权”的标志。当程序遇到 `await asyncio.sleep(1)` 时，它不会傻等，而是把控制权交还给事件循环，去处理其他任务，直到 1 秒后回来继续执行。

---

## 2. 并发执行：让任务“同时”跑起来

如果顺序 `await`，效率并不会提升。要实现并发，我们需要创建 `Task`。

### 视觉化对比：顺序执行 vs 并发执行

![SyncVSAsync](https://img.yumeko.site/file/articles/PythonAdvanceUsage/Async/SyncVSAsync.svg)

### 示例代码

```python
async def concurrent_example():
    # create_task 会立即将协程调度进事件循环
    task1 = asyncio.create_task(say_after(1, "Hello"))
    task2 = asyncio.create_task(say_after(2, "World"))
    
    # 等待所有任务完成
    await task1
    await task2
```

**解析**：
通过 `asyncio.create_task`，`task1` 和 `task2` 几乎同时启动。总耗时取决于最慢的那个任务（约 2s），而不是两者之和。这是异步编程最大的优势。

---

## 3. 异步上下文管理器

我们熟悉 `with open(...)`，而在异步编程中，我们需要 `async with` 来处理涉及 I/O 的资源的分配与释放。

### 核心实现
需要实现 `__aenter__` 和 `__aexit__` 魔术方法。

```python
class AsyncContextManager:
    async def __aenter__(self):
        print("进入上下文")
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        print("退出上下文")

# 使用方式
async with AsyncContextManager() as manager:
    await manager.do_something()
```

---

## 4. 异步迭代器与生成器

处理大量数据流时，为了避免一次性加载或同步阻塞，我们可以使用异步迭代。

- **异步迭代器**：实现 `__aiter__` 和 `__anext__`。
- **异步生成器**：在 `async def` 函数中使用 `yield`。

```python
# 异步生成器示例
async def async_generator():
    for i in range(5):
        await asyncio.sleep(0.1)
        yield i

# 消费方式
async for value in async_generator():
    print(value)
```

---

## 5. 实战 I/O：HTTP、文件与数据库

这是异步编程最能发挥威力的领域。注意：标准库的 `requests`、`open()`、`sqlite3` 都是**同步阻塞**的，在异步函数中直接使用会卡死整个事件循环。我们需要特定的异步库。

### 5.1 异步 HTTP (aiohttp)
并发抓取多个 URL，效率极高。
```python
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async with aiohttp.ClientSession() as session:
    # asyncio.gather 接收多个 future/task 并发运行
    results = await asyncio.gather(*[fetch_url(session, url) for url in urls])
```

### 5.2 异步文件操作 (aiofiles)
```python
import aiofiles
async with aiofiles.open("example.txt", mode='w') as f:
    await f.write("异步写入内容")
```

### 5.3 异步数据库 (aiosqlite)
```python
import aiosqlite
async with aiosqlite.connect("example.db") as db:
    cursor = await db.execute("SELECT * FROM users")
    rows = await cursor.fetchall()
```

---

## 6. 通信与同步：队列、锁与信号量

当多个协程并发运行时，我们需要工具来协调它们。

### 生产者-消费者交互图

![ProducerAndConsumer](https://img.yumeko.site/file/articles/PythonAdvanceUsage/Async/ProducerAndConsumer.svg)

### 生产者-消费者模型 (asyncio.Queue)
这是解耦数据生产与处理的最佳实践。
```python
queue = asyncio.Queue()

# 生产者
await queue.put(item)

# 消费者
item = await queue.get()
# ...处理...
queue.task_done() # 告知队列该项已处理完毕
```

### 资源保护 (Lock & Semaphore)
- **Lock**：确保同一时间只有一个协程修改共享资源。
- **Semaphore**：限制并发数量（例如限制同时只有 2 个爬虫请求）。

```python
semaphore = asyncio.Semaphore(2)
async with semaphore:
    # 只有获得信号量的协程才能执行此处代码
    await network_request()
```

---

## 7. 高级控制：超时、事件循环与混合编程

### 异步与同步混合运行原理

当必须执行阻塞代码（如 `time.sleep` 或 CPU 密集型计算）时，直接运行会卡死 Event Loop。解决方案是将其扔到线程池。

![MixSyncAndAsync](https://img.yumeko.site/file/articles/PythonAdvanceUsage/Async/MixSyncAndAsync.svg)

### 超时处理
防止某个任务无限期卡住。
```python
try:
    result = await asyncio.wait_for(long_running_task(), timeout=3)
except asyncio.TimeoutError:
    print("任务超时！")
```

---

## 8. 异步 Web 服务器

使用 `aiohttp.web` 可以快速搭建高性能的异步 Web 服务。

```python
from aiohttp import web

async def handle(request):
    return web.Response(text="Hello, Async World!")

app = web.Application()
app.add_routes([web.get('/', handle)])
```



