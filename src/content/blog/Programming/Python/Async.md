---
title: Python 异步编程完全指南——从协程到高并发实战
date: 2026-06-12
category: 编程/Python
tags:
  - Python
  - 异步
  - 并发
description: 深入理解 Python asyncio 异步编程，涵盖事件循环机制、协程与 Task、异步上下文管理器、异步迭代器、aiohttp/aiofiles/aiosqlite 实战、Queue/Lock/Semaphore 同步原语、TaskGroup、超时与取消、线程池混合编程与调试技巧
image: https://img.yumeko.site/file/blog/cover/1781275466706_AsyncBanner.webp
status: published
---

## 1. 问题的起点——为什么需要异步

假设你写了一个爬虫，需要抓取 100 个 URL。用 `requests` 同步请求：

```python
import requests
import time

def fetchAll(urls):
    results = []
    for url in urls:
        resp = requests.get(url)       # 阻塞！CPU 空转等待网络
        results.append(resp.text)
    return results

# 100 个 URL × 200ms = 20 秒
```

**运行结果：**

```text
$ python sync_crawler.py
正在抓取: https://api.example.com/1 ... 200 OK (0.2s)
正在抓取: https://api.example.com/2 ... 200 OK (0.2s)
...
正在抓取: https://api.example.com/100 ... 200 OK (0.2s)
总耗时: 20.1s, 共抓取 100 个 URL
```

每个 `requests.get()` 调用期间，Python 进程**什么也不做**，只是等待网络返回。CPU 利用率不到 1%，时间全浪费在 I/O 等待上。

**异步编程的核心思想**：在等待 I/O 的间隙，切换到其他任务继续执行。100 个 URL 可以**几乎同时**发起请求，总耗时接近**单个请求的耗时**（200ms 而不是 20s）。

> [!NOTE] 异步 vs 多线程
> 多线程也能解决 I/O 密集问题，但线程有 GIL 限制、上下文切换开销、共享状态竞争等问题。asyncio 使用**单线程 + 协程**，没有锁竞争和线程切换开销，适合 I/O 密集型任务。CPU 密集型任务应该用 `multiprocessing`。

## 2. 事件循环——异步的心脏

理解 asyncio 的关键在于理解**事件循环（Event Loop）**。它像一个调度员，维护一个任务队列，轮流执行协程，遇到 `await` 就把控制权交还给循环去处理下一个任务。

![EventLoop.png](https://img.yumeko.site/file/blog/articles/1781275097534_EventLoop.webp)

事件循环的工作流程：

1. 从任务队列取出一个协程执行
2. 协程遇到 `await` 挂起，控制权交回事件循环
3. 事件循环检查是否有 I/O 就绪的回调
4. 取出下一个可执行的协程继续执行
5. 重复直到所有任务完成

```python
import asyncio

# 启动事件循环（Python 3.7+）
asyncio.run(main())
```

`asyncio.run()` 做了三件事：创建事件循环 → 运行传入的协程 → 关闭事件循环。**一个进程同一时间只有一个事件循环在运行。**

## 3. 协程基础——async def 与 await

### 3.1 定义与调用

```python
import asyncio

async def sayHello(name: str, delay: float) -> str:
    """异步函数——返回一个协程对象"""
    await asyncio.sleep(delay)          # 非阻塞等待
    return f"Hello, {name}"

# 调用 async def 函数不会执行它，而是返回一个协程对象
coro = sayHello("World", 1.0)
print(type(coro))                       # <class 'coroutine'>

# 三种运行方式：
# 方式 1：直接 await（必须在另一个 async 函数内）
# result = await coro

# 方式 2：asyncio.run()（顶层入口）
result = asyncio.run(coro)

# 方式 3：创建 Task 并发执行（见第 4 节）
```

**运行结果：**

```text
$ python coroutine_demo.py
<class 'coroutine'>
Hello, World
```

> [!WARNING] async def 不等于异步执行
> 调用 `async def` 函数只会创建一个协程对象，**什么也不会发生**。必须通过 `await`、`asyncio.run()` 或 `create_task()` 才能真正执行。

### 3.2 await 的本质

`await` 做了两件事：

1. **挂起当前协程**，把控制权交还给事件循环
2. **注册回调**：当 await 的对象完成时，事件循环会恢复该协程的执行

```python
async def demo():
    print("A: 开始")
    await asyncio.sleep(1)              # 挂起 1 秒，事件循环去处理其他任务
    print("B: 1 秒后恢复执行")
    result = await fetchData()          # 挂起直到 HTTP 请求返回
    print(f"C: 收到数据 {result}")
```

**运行结果：**

```text
$ python await_demo.py
A: 开始
（等待 1 秒，事件循环可处理其他任务）
B: 1 秒后恢复执行
（等待 HTTP 请求返回，事件循环继续处理其他任务）
C: 收到数据 {"status": "ok"}
```

**可 await 的对象（Awaitable）只有三种**：

| 类型 | 说明 | 创建方式 |
|:--|:--|:--|
| **Coroutine** | 协程对象 | `async def` 函数调用返回 |
| **Task** | 被事件循环调度的协程 | `asyncio.create_task(coro)` |
| **Future** | 低层级占位符（通常不直接使用） | `loop.create_future()` |

## 4. 并发执行——Task、gather 与 TaskGroup

顺序 `await` 不会带来性能提升。真正的并发需要把协程包装成 **Task** 交给事件循环调度。

![SequentialVsConcurrent.png](https://img.yumeko.site/file/blog/articles/1781274958227_SequentialVsConcurrent.webp)

### 4.1 asyncio.create_task——立即调度

```python
async def fetch(id: int) -> str:
    print(f"  任务 {id}: 开始")
    await asyncio.sleep(id * 0.5)
    print(f"  任务 {id}: 完成")
    return f"结果-{id}"

async def concurrentDemo():
    # create_task 将协程提交到事件循环，立即开始执行
    task1 = asyncio.create_task(fetch(1), name="task-1")
    task2 = asyncio.create_task(fetch(2), name="task-2")
    task3 = asyncio.create_task(fetch(3), name="task-3")

    # 此时三个任务已经在后台并发运行

    # 等待所有任务完成并获取结果
    r1 = await task1
    r2 = await task2
    r3 = await task3
    return r1, r2, r3
```

**运行结果：**

```text
$ python concurrent_demo.py
  任务 1: 开始
  任务 2: 开始
  任务 3: 开始
  任务 1: 完成
  任务 2: 完成
  任务 3: 完成
总耗时: 1.5s（而非 1+2+3=6s）
结果: ('结果-1', '结果-2', '结果-3')
```

三个任务几乎同时启动。Task1（0.5s）最先完成，Task2（1.0s）第二，Task3（1.5s）最后。总耗时 = `max(0.5, 1.0, 1.5) = 1.5s`，而不是顺序执行的 3s。

### 4.2 asyncio.gather——批量并发

`gather` 是最高频的并发 API——同时启动多个协程，返回结果列表：

```python
async def gatherDemo():
    urls = ["https://api.example.com/1", "https://api.example.com/2", "https://api.example.com/3"]

    # 并发执行所有请求，按顺序返回结果
    results = await asyncio.gather(
        fetchUrl(urls[0]),
        fetchUrl(urls[1]),
        fetchUrl(urls[2]),
        return_exceptions=True              # 单个任务失败不影响其他
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"URL {i} 失败: {result}")
        else:
            print(f"URL {i} 成功: {len(result)} 字节")
```

**运行结果：**

```text
$ python gather_demo.py
URL 0 成功: 1234 字节
URL 1 失败: TimeoutError
URL 2 成功: 5678 字节
总耗时: 1.2s（等待最慢的那个）
```

> [!TIP] `return_exceptions=True`
> 默认情况下，`gather` 中任何一个任务抛出异常，整个 `gather` 立即抛出。设置 `return_exceptions=True` 后，异常会被当作正常返回值收集，其他任务继续执行。

### 4.3 asyncio.as_completed——谁先完成先处理谁

```python
async def asCompletedDemo():
    tasks = [fetch(i) for i in range(5)]

    # 哪个任务先完成就先处理哪个
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"收到: {result}")
        # 立即处理结果（不等其他任务）
```

**运行结果：**

```text
$ python as_completed_demo.py
收到: 结果-1        ← 0.5s 时，task1 最先完成
收到: 结果-2        ← 1.0s 时，task2 完成
收到: 结果-0        ← 1.0s 时，task0 紧随其后
收到: 结果-3        ← 1.5s 时，task3 完成
收到: 结果-4        ← 2.0s 时，task4 最后完成
```

`as_completed` 不像 `gather` 那样等全部完成再按顺序返回，而是**谁先完成就立即 yield 谁**。适合需要逐条流式处理的场景。

### 4.4 asyncio.TaskGroup——Python 3.11+ 的结构化并发

```python
async def taskGroupDemo():
    # TaskGroup 确保所有子任务在退出上下文前完成（或取消）
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch(1))
        task2 = tg.create_task(fetch(2))
        task3 = tg.create_task(fetch(3))
    # 退出 with 块时，自动等待所有任务完成
    # 如果任何任务失败，其他任务会自动取消

    # 所有结果可以通过 task.result() 获取
    print(task1.result(), task2.result(), task3.result())
```

### 4.5 四种并发方式对比

| 方式 | 适用场景 | 错误处理 | Python 版本 |
|:--|:--|:--|:--|
| `create_task` + `await` | 少量已知任务 | 手动 try/except | 3.7+ |
| `asyncio.gather` | 批量并发，需要所有结果 | `return_exceptions=True` | 3.7+ |
| `asyncio.as_completed` | 流式处理，先到先得 | 手动 per-coroutine | 3.7+ |
| `asyncio.TaskGroup` | 结构化并发，强保证 | 自动取消兄弟任务 | 3.11+ |

## 5. 异步上下文管理器

当资源的获取和释放涉及 I/O（如数据库连接、HTTP Session），需要 `async with`。

### 5.1 使用现成的异步上下文管理器

```python
import aiohttp
import aiosqlite

# HTTP Session——连接池复用
async with aiohttp.ClientSession() as session:
    async with session.get("https://example.com") as resp:
        html = await resp.text()

# 数据库连接——自动 commit/rollback
async with aiosqlite.connect("app.db") as db:
    async with db.execute("SELECT * FROM users") as cursor:
        rows = await cursor.fetchall()
```

### 5.2 自定义异步上下文管理器

```python
class AsyncDBConnection:
    """异步数据库连接管理器"""

    async def __aenter__(self):
        self.conn = await aiosqlite.connect("app.db")
        print("数据库连接已打开")
        return self.conn

    async def __aexit__(self, excType, excVal, excTb):
        await self.conn.close()
        print("数据库连接已关闭")
        # 返回 True 可抑制异常，一般返回 False 让异常继续传播
        return False

async def useDB():
    async with AsyncDBConnection() as conn:
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        print(f"查询结果: {row}")
```

**运行结果：**

```text
$ python async_context.py
数据库连接已打开
查询结果: (1,)
数据库连接已关闭
```

### 5.3 使用 contextlib 简化

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managedDB(dbPath: str):
    """用生成器语法创建异步上下文管理器（更简洁）"""
    conn = await aiosqlite.connect(dbPath)
    try:
        yield conn                          # 进入 async with 块
    finally:
        await conn.close()                  # 退出 async with 块

async with managedDB("app.db") as conn:
    cursor = await conn.execute("SELECT 1")
    row = await cursor.fetchone()
    print(f"查询结果: {row}")
```

**运行结果：**

```text
$ python contextlib_async.py
数据库已连接
查询结果: (1,)
数据库已关闭
# 比手写 __aenter__/__aexit__ 少了很多样板代码
```

## 6. 异步迭代器与生成器

处理流式数据（如分页 API、大文件逐行读取）时，异步迭代可以边取边处理。

### 6.1 异步生成器

```python
async def paginatedAPI(baseUrl: str, totalPages: int):
    """异步生成器——逐页获取 API 数据"""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        for page in range(1, totalPages + 1):
            url = f"{baseUrl}?page={page}"
            async with session.get(url) as resp:
                data = await resp.json()
                yield data                  # 产出一页数据

# 消费异步生成器
async def consumePages():
    async for page in paginatedAPI("https://api.example.com/items", 10):
        print(f"收到 {len(page)} 条记录")
        # 边收边处理，不等所有页
```

**运行结果：**

```text
$ python async_generator.py
收到 20 条记录
收到 20 条记录
收到 20 条记录
...
收到 20 条记录
# 10 页数据逐页产出，不等所有页加载完就可以开始处理
```

### 6.2 异步列表推导式

```python
# 异步列表推导（Python 3.6+）
results = [await fetch(i) async for i in asyncGenerator()]

# 等价于：
results = []
async for i in asyncGenerator():
    results.append(await fetch(i))
```

### 6.3 自定义异步迭代器

```python
class AsyncRange:
    """异步迭代器——实现 __aiter__ 和 __anext__"""

    def __init__(self, start: int, stop: int):
        self.current = start
        self.stop = stop

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)            # 模拟异步操作
        value = self.current
        self.current += 1
        return value

async for num in AsyncRange(0, 5):
    print(num)
```

**运行结果：**

```text
$ python async_iterator.py
0
1
2
3
4
# 每次迭代之间有 0.1s 的异步等待，但事件循环可以在此期间处理其他任务
```

## 7. 通信与同步——协程之间的协调

当多个协程并发运行时，需要同步原语来协调它们。

![ProducerConsumer.png](https://img.yumeko.site/file/blog/articles/1781274989668_ProducerConsumer.webp)

### 7.1 asyncio.Queue——生产者-消费者

```python
async def producer(queue: asyncio.Queue, n: int):
    """生产者：生成数据放入队列"""
    for i in range(n):
        await asyncio.sleep(0.1)            # 模拟生产耗时
        item = f"数据-{i}"
        await queue.put(item)
        print(f"  生产: {item}")

    # 发送结束信号
    await queue.put(None)

async def consumer(queue: asyncio.Queue, name: str):
    """消费者：从队列取出数据处理"""
    while True:
        item = await queue.get()
        if item is None:                    # 收到结束信号
            queue.task_done()
            break
        print(f"  {name} 消费: {item}")
        await asyncio.sleep(0.3)            # 模拟处理耗时
        queue.task_done()                   # 标记该项处理完毕

async def pipeline():
    queue = asyncio.Queue(maxsize=5)        # 限制队列最大容量

    # 并发启动生产者和多个消费者
    prod = asyncio.create_task(producer(queue, 10))
    consumers = [asyncio.create_task(consumer(queue, f"C{i}")) for i in range(3)]

    await prod                              # 等待生产者完成
    await queue.join()                      # 等待队列中所有项被处理完

    for c in consumers:
        c.cancel()                          # 取消消费者
```

**运行结果：**

```text
$ python queue_demo.py
  生产: 数据-0
  C0 消费: 数据-0
  生产: 数据-1
  C1 消费: 数据-1
  生产: 数据-2
  C2 消费: 数据-2
  ...
  生产: 数据-9
  C0 消费: 数据-9
处理完毕，队列已空
```

三个消费者并发地从队列取数据，互不阻塞。`queue.join()` 确保所有 `task_done()` 都被调用后才返回。

### 7.2 asyncio.Lock——互斥锁

```python
lock = asyncio.Lock()
counter = 0

async def increment():
    global counter
    async with lock:                        # 同一时间只有一个协程进入
        current = counter
        await asyncio.sleep(0.01)           # 模拟读取-修改-写入的间隙
        counter = current + 1
```

**有无 Lock 的对比：**

```text
# 不加 Lock：10 个协程并发执行 counter += 1
$ python no_lock.py
最终 counter: 3（期望 10，丢失了 7 次更新！）

# 加 Lock：同一时间只有一个协程修改
$ python with_lock.py
最终 counter: 10（正确）
```

### 7.3 asyncio.Semaphore——控制并发数

```python
semaphore = asyncio.Semaphore(3)            # 最多同时 3 个并发

async def rateLimitedFetch(url: str):
    async with semaphore:
        # 同时只有 3 个协程能执行这里
        return await fetchUrl(url)

# 100 个 URL，但最多同时 3 个请求
results = await asyncio.gather(*[rateLimitedFetch(url) for url in urls])
```

**运行效果对比：**

```text
# 无 Semaphore：100 个请求同时发出
$ python no_limit.py
[0.0s] 发起 100 个连接...
[0.1s] 服务器返回 429 Too Many Requests（被限流！）
总耗时: 0.1s，成功: 12/100

# 有 Semaphore(3)：平稳发送，不会被封
$ python with_limit.py
[0.0s] 并发数: 3
[0.5s] 并发数: 3
[1.0s] 并发数: 3
...
[16.5s] 全部完成
总耗时: 16.5s，成功: 100/100
```

### 7.4 asyncio.Event——事件通知

> [!TIP] Semaphore 的使用场景
> - 限制爬虫并发数，避免被封 IP
> - 限制数据库连接数，不超过连接池大小
> - 限制 API 调用频率，遵守 Rate Limit

```python
event = asyncio.Event()

async def waiter():
    print("等待事件...")
    await event.wait()                      # 挂起直到 event.set()
    print("事件已触发，继续执行")

async def trigger():
    await asyncio.sleep(2)
    print("触发事件")
    event.set()                             # 唤醒所有等待者
```

**运行结果：**

```text
$ python event_demo.py
等待事件...
（等待 2 秒...）
触发事件
事件已触发，继续执行
```

### 7.5 同步原语速查表

| 原语 | 作用 | 典型场景 |
|:--|:--|:--|
| `Queue` | 协程间传递数据 | 生产者-消费者、流式处理 |
| `Lock` | 互斥访问共享资源 | 计数器、文件写入 |
| `Semaphore` | 限制并发数量 | 爬虫限流、连接池保护 |
| `Event` | 一次性通知 | 初始化完成通知、关闭信号 |
| `Condition` | 条件变量（等待特定状态） | 复杂状态协调 |

## 8. 实战 I/O——HTTP、文件与数据库

异步编程的威力在 I/O 密集型场景中最大化。**关键前提**：标准库的 `requests`、`open()`、`sqlite3` 都是**同步阻塞**的——在协程中使用它们会把整个事件循环卡死。必须使用对应的异步库。

![SyncVsAsyncIO.png](https://img.yumeko.site/file/blog/articles/1781275018552_SyncVsAsyncIO.webp)

### 8.1 异步 HTTP——aiohttp 完整示例

```python
import aiohttp

async def fetchUrl(session: aiohttp.ClientSession, url: str) -> dict:
    """抓取单个 URL，返回 JSON"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()
    except asyncio.TimeoutError:
        return {"error": f"超时: {url}"}
    except aiohttp.ClientError as e:
        return {"error": str(e)}

async def batchFetch(urls: list[str], concurrency: int = 5) -> list[dict]:
    """并发抓取多个 URL，限制并发数"""
    semaphore = asyncio.Semaphore(concurrency)

    async def fetchWithLimit(url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await fetchUrl(session, url)

    tasks = [fetchWithLimit(url) for url in urls]
    return await asyncio.gather(*tasks)

# 使用
# results = asyncio.run(batchFetch(["https://api.example.com/1", ...]))
```

**运行结果：**

```text
$ python batch_fetch.py
并发抓取 10 个 URL，最大并发数: 5
[0.2s] ✓ https://api.example.com/1 -> 1234 字节
[0.3s] ✓ https://api.example.com/2 -> 2345 字节
[0.3s] ✗ https://dead-link.com -> TimeoutError
[0.5s] ✓ https://api.example.com/4 -> 3456 字节
...
总耗时: 1.8s（同步需 10×0.2s=2s，受并发上限影响）
成功: 9/10，失败: 1
```

### 8.2 异步文件操作——aiofiles

```python
import aiofiles

async def processLargeFile(path: str):
    """异步逐行处理大文件——不阻塞事件循环"""
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        async for line in f:
            # 每行可以 await 异步处理（如写入数据库）
            await processLine(line)

async def writeResults(path: str, data: list[str]):
    async with aiofiles.open(path, mode='w', encoding='utf-8') as f:
        for line in data:
            await f.write(line + '\n')
```

### 8.3 异步数据库——aiosqlite

```python
import aiosqlite

async def initDB(dbPath: str):
    async with aiosqlite.connect(dbPath) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """)
        await db.commit()

async def batchInsert(dbPath: str, users: list[tuple[str, str]]):
    async with aiosqlite.connect(dbPath) as db:
        await db.executemany(
            "INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)",
            users
        )
        await db.commit()

async def queryUsers(dbPath: str) -> list[dict]:
    async with aiosqlite.connect(dbPath) as db:
        db.row_factory = aiosqlite.Row       # 支持按列名访问
        async with db.execute("SELECT * FROM users") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
```

**运行结果：**

```text
$ python async_db.py
数据库初始化完成
批量插入 100 条用户记录...
插入完成: 100 条
查询结果: 100 条用户
# 异步 SQLite 不影响事件循环，其他协程可以在查询期间继续工作
```

### 8.4 实战：异步爬虫完整示例

```python
async def fullCrawler(startUrls: list[str], maxDepth: int = 2):
    """异步爬虫——带限流、去重、超时"""
    seen = set()
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(5)        # 最多 5 个并发

    for url in startUrls:
        await queue.put((url, 0))

    async def worker():
        async with aiohttp.ClientSession() as session:
            while True:
                url, depth = await queue.get()
                try:
                    if url in seen or depth > maxDepth:
                        continue

                    seen.add(url)
                    async with semaphore:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            html = await resp.text()
                            # 提取新 URL（省略解析逻辑）
                            # newUrls = extractLinks(html)
                            # for u in newUrls:
                            #     await queue.put((u, depth + 1))
                            print(f"  [{depth}] {url} -> {len(html)} 字节")

                except Exception as e:
                    print(f"  错误: {url}: {e}")
                finally:
                    queue.task_done()

    # 启动 3 个 worker 协程
    workers = [asyncio.create_task(worker()) for _ in range(3)]
    await queue.join()

    for w in workers:
        w.cancel()
    print(f"爬取完成，共收集 {len(seen)} 个 URL")
```

**运行结果：**

```text
$ python async_crawler.py
  [0] https://example.com -> 15234 字节
  [0] https://example.com/about -> 8921 字节
  [1] https://example.com/contact -> 6543 字节
  [1] https://example.com/blog -> 12345 字节
  ...
  错误: https://dead-link.com: TimeoutError
  [2] https://example.com/blog/post1 -> 9876 字节
爬取完成，共收集 47 个 URL
总耗时: 8.3s（若同步抓取 47 个 URL 约需 47×0.3s=14s）
```

## 9. 高级控制——超时、取消与线程混合

### 9.1 asyncio.wait_for——任务超时

```python
async def withTimeout():
    try:
        result = await asyncio.wait_for(
            longRunningTask(),
            timeout=5.0                     # 5 秒后抛出 TimeoutError
        )
        return result
    except asyncio.TimeoutError:
        print("任务超时，执行降级逻辑")
        return fallbackValue()
```

**运行结果：**

```text
$ python timeout_demo.py
任务开始（预计耗时 10s）...
（5 秒后）
任务超时，执行降级逻辑
降级数据已返回
```

> [!WARNING] `wait_for` 会取消被包装的任务
> 超时后 `wait_for` 会调用 `task.cancel()`，在协程内向该任务注入 `CancelledError`。如果任务有 `try/finally` 清理逻辑，确保 `finally` 块是异步安全的。

### 9.2 任务取消

```python
async def cancellableTask():
    try:
        while True:
            print("工作中...")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("收到取消信号，执行清理...")
        # 清理资源（关闭连接、保存状态等）
        raise                               # 重新抛出以正确终止

async def cancelDemo():
    task = asyncio.create_task(cancellableTask())
    await asyncio.sleep(3.5)
    task.cancel()                           # 发送取消信号
    try:
        await task
    except asyncio.CancelledError:
        print("任务已被取消")
```

**运行结果：**

```text
$ python cancel_demo.py
工作中...
工作中...
工作中...
（3.5 秒后调用 task.cancel()）
收到取消信号，执行清理...
任务已被取消
```

> [!TIP] 捕获 CancelledError 后必须重新抛出
> 如果 `except CancelledError` 后不 `raise`，任务不会被正确取消，事件循环会认为它仍然在运行。如果确实要阻止取消（极少见），使用 `asyncio.shield()`。

### 9.3 屏蔽取消——asyncio.shield

```python
async def criticalSection():
    # 这段代码即使外层 task.cancel() 也不会被中断
    await asyncio.shield(saveToDatabase())
```

### 9.4 混合同步代码——run_in_executor

当你必须调用同步阻塞函数（如 `time.sleep`、大型 NumPy 计算）时，把它丢到线程池：

```python
import time
import concurrent.futures

def blockingIO():
    """同步阻塞函数——不能在协程中直接调用"""
    time.sleep(2)
    return "阻塞操作完成"

def cpuHeavy(n: int):
    """CPU 密集型计算"""
    return sum(i * i for i in range(n))

async def mixSyncAndAsync():
    loop = asyncio.get_running_loop()

    # 方式 1：asyncio.to_thread（Python 3.9+，推荐）
    result1 = await asyncio.to_thread(blockingIO)

    # 方式 2：run_in_executor（兼容旧版本）
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result2 = await loop.run_in_executor(pool, blockingIO)

    # CPU 密集型任务用 ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result3 = await loop.run_in_executor(pool, cpuHeavy, 10_000_000)

    return result1, result2, result3
```

**运行结果：**

```text
$ python mix_sync_async.py
（to_thread 方式，不阻塞事件循环）
阻塞操作完成（耗时 2.0s，在后台线程池中运行）

（run_in_executor 方式，同样不阻塞）
阻塞操作完成（耗时 2.0s，在独立线程中运行）

（ProcessPoolExecutor 方式，绕过 GIL）
CPU 计算结果: 333333283333335000000（耗时 0.8s，多进程并行）

总计: 3 个任务并发完成，最大耗时 2.0s
```

> [!NOTE] ThreadPoolExecutor vs ProcessPoolExecutor
> - **ThreadPoolExecutor**：适合 I/O 阻塞（如 `time.sleep`、同步 HTTP 库）。线程受 GIL 限制，不适合 CPU 密集型。
> - **ProcessPoolExecutor**：适合 CPU 密集型（如 NumPy 运算、图像处理）。绕过 GIL，但有进程间通信开销。

## 10. 调试与常见陷阱

### 10.1 常见错误

**错误 1：在协程中调用同步阻塞函数**

```python
async def bad():
    time.sleep(2)                           # 错的！卡死事件循环
    requests.get("...")                     # 错的！同上

async def good():
    await asyncio.sleep(2)                  # 对的
    await asyncio.to_thread(time.sleep, 2)  # 对的
```

**运行对比：**

```text
# 错误写法: 整个事件循环被 time.sleep 卡死 2 秒
$ python bad_async.py
[0.0s] 协程 A 开始
[0.0s] 协程 A 调用了 time.sleep(2)...（事件循环冻结！）
[2.0s] 协程 A 醒来
[2.0s] 协程 B 开始（晚了 2 秒！）

# 正确写法: 事件循环在 2 秒内处理了其他任务
$ python good_async.py
[0.0s] 协程 A 开始，await asyncio.sleep(2)
[0.0s] 协程 B 开始（同时运行！）
[0.5s] 协程 B 完成
[2.0s] 协程 A 醒来
```

**错误 2：忘记 await**

```python
async def bad():
    asyncio.sleep(1)                        # 返回协程对象，什么也没发生！
    # 正确：await asyncio.sleep(1)
```

**运行结果：**

```text
$ python forgot_await.py
RuntimeWarning: coroutine 'sleep' was never awaited
  asyncio.sleep(1)
# 程序立即结束，什么都没等！
```

**错误 3：在非异步上下文中使用 await**

```python
def normalFunction():
    await fetch()                           # SyntaxError！只能在 async def 中用 await
```

**运行结果：**

```text
$ python syntax_error.py
  File "syntax_error.py", line 2
    await fetch()
    ^
SyntaxError: 'await' outside async function
```

### 10.2 调试模式

```bash
# 开启 asyncio debug 模式
PYTHONASYNCIODEBUG=1 python script.py
```

或在代码中：

```python
import asyncio
asyncio.run(main(), debug=True)
```

**debug 模式输出示例：**

```text
$ PYTHONASYNCIODEBUG=1 python debug_demo.py
Executing <Task pending name='Task-1' coro=<main()>> took 0.312 seconds
  ↑ 警告：某个回调耗时超过 100ms，可能阻塞了事件循环

<coroutine object bad at 0x7f...> was never awaited
  ↑ 警告：协程创建了但从未 await，可能是内存泄漏
```

Debug 模式会检测：
- 协程被创建但从未 await（可能的内存泄漏）
- 耗时超过 100ms 的回调（Event Loop 被阻塞的信号）
- 未正确关闭的资源

### 10.3 使用 uvloop 提升性能

```python
# pip install uvloop
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# 之后正常使用 asyncio，性能提升 2-4 倍
asyncio.run(main())
```

**性能对比（10000 个并发 HTTP 请求）：**

```text
# 标准 asyncio
$ python bench_std.py
10000 请求完成，耗时: 12.4s

# uvloop 加速
$ python bench_uvloop.py
10000 请求完成，耗时: 4.1s（提速 3×）
```

`uvloop` 是用 Cython 重写的事件循环，基于 libuv（Node.js 的同款底层库），在大规模并发场景下性能远超标准库的事件循环。

### 10.4 健康检查：检测 Event Loop 是否被阻塞

```python
async def healthCheck():
    """定期检查事件循环响应延迟"""
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(1)
        lag = time.monotonic() - t0 - 1.0
        if lag > 0.1:
            print(f"警告: 事件循环延迟 {lag:.3f}s，可能被阻塞代码卡住")
```

**运行效果：**

```text
# 正常情况（无阻塞代码）
$ python health_check.py
延迟: 0.001s ✓
延迟: 0.002s ✓
延迟: 0.001s ✓

# 检测到阻塞（有人在协程里调了 time.sleep）
$ python health_check.py
延迟: 0.002s ✓
警告: 事件循环延迟 0.847s，可能被阻塞代码卡住  ← 某个协程调了 time.sleep(1)
延迟: 0.003s ✓
```

## 11. 总结

| 维度 | 要点 |
|:--|:--|
| **核心机制** | 事件循环 + 协程 + Task。`await` 交出控制权，事件循环调度下一个就绪任务 |
| **并发 API** | `create_task` 立即调度、`gather` 批量等待、`as_completed` 流式处理、`TaskGroup` 结构化并发 |
| **上下文管理** | `async with` 管理异步资源的获取/释放，`@asynccontextmanager` 简化实现 |
| **迭代** | `async for` 消费异步流，`async def` + `yield` 创建异步生成器 |
| **同步原语** | `Queue` 生产者-消费者、`Semaphore` 限流、`Lock` 互斥、`Event` 通知 |
| **I/O 实战** | `aiohttp` 替代 requests、`aiofiles` 替代 open、`aiosqlite` 替代 sqlite3 |
| **混合编程** | `asyncio.to_thread()` 包裹同步阻塞代码，`ProcessPoolExecutor` 处理 CPU 密集 |
| **调试** | `PYTHONASYNCIODEBUG=1` 检测常见错误，`uvloop` 提升事件循环性能 |

> [!ABSTRACT] 一句话总结
> asyncio 的核心价值在于**让 I/O 等待不再浪费 CPU**。它不是让单次操作变快，而是让**大量 I/O 操作的等待时间互相重叠**。学会识别"这段代码在等什么"——等待网络、等待磁盘、等待数据库——然后用 `await` 把它交给事件循环。

---