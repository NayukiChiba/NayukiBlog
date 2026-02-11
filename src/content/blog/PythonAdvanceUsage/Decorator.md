---
title: Python 装饰器
date: 2026-01-26
category: PythonAdvanceUsage
tags:
  - Python
  - 高级教程
description: Python 装饰器的详细技术指南，涵盖其定义、语法糖、基础包装器、带参数装饰器、元数据保持，以及权限校验、缓存、异常处理等常见应用场景。
image: https://img.yumeko.site/file/blog/Decorator.png
status: published
---

## 1. 定义与技术原理

**装饰器**是一种设计模式，在 Python 中通过**高阶函数 (Higher-order Function)** 实现。

*   **定义**：装饰器是一个接收函数作为参数，并返回一个新的可调用对象（通常是函数）的可调用对象。
*   **作用**：在不修改原函数定义、不侵入原代码逻辑的前提下，动态地扩展或增强函数的功能。

---

## 2. 语法糖 (Syntactic Sugar) 解析

Python 使用 `@` 符号作为应用装饰器的语法糖。

```python
@my_decorator
def my_func(): ...
```
完全等价于：
```python
my_func = my_decorator(my_func)
```

---

## 3. 装饰器结构模式

### 3.1 基础包装器 (Basic Wrapper)

最标准的实现方式，利用内部函数 `wrapper` 劫持调用。

**代码示例**：

```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"-> Calling {func.__name__} with {args}")
        result = func(*args, **kwargs)
        print(f"<- {func.__name__} returned {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

add(3, 5)
```

**运行结果**：

```text
-> Calling add with (3, 5)
<- add returned 8
```

### 3.2 带参数的装饰器 (Parametrized Decorator)

需要三层嵌套结构：`工厂 -> 装饰器 -> 包装器。`

**代码示例**：

```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Repeating {times} times:")
            for i in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(times=3)
def say_hi(name):
    print(f"  Hi, {name}!")

say_hi("Python")
```

**运行结果**：
```text
Repeating 3 times:
  Hi, Python!
  Hi, Python!
  Hi, Python!
```

### 3.3 保持元数据 (`functools.wraps`)

使用 `@wraps` 避免丢失原函数的文档和名字。

**代码示例**：

```python
from functools import wraps

def proper_decorator(func):
    @wraps(func)
    def wrapper():
        """Wrapper doc"""
        return func()
    return wrapper

@proper_decorator
def my_task():
    """Original docstring of my_task"""
    pass

print(f"Name: {my_task.__name__}")
print(f"Doc: {my_task.__doc__}")
```

**运行结果**：

```text
Name: my_task
Doc: Original docstring of my_task
```

---

## 4. 常见应用场景与源码分析

### 4.1 横切关注点分离 (AOP) - 权限校验

这是 Web 开发中的经典模式。业务逻辑无需包含 `if user.is_admin` 这样的判断，装饰器充当拦截器。

**代码示例**：

```python
current_user = {"role": "guest"}

def require_admin(func):
    def wrapper(*args, **kwargs):
        if current_user.get("role") != "admin":
            raise PermissionError("Access Denied: Admins only!")
        return func(*args, **kwargs)
    return wrapper

@require_admin
def delete_database():
    print("Database deleted!")

# 测试调用
try:
    delete_database()
except Exception as e:
    print(f"Error caught: {e}")
```

**运行结果**：
```text
Error caught: Access Denied: Admins only!
```

### 4.2 状态保持与缓存 (Memoization)

利用闭包的私有作用域实现简单缓存。

**代码示例**：

```python
def simple_cache(func):
    cache = {} # 闭包变量，随函数存在
    def wrapper(n):
        if n not in cache:
            print(f"Computing {n}...")
            cache[n] = func(n)
        else:
            print(f"Fetching {n} from cache...")
        return cache[n]
    return wrapper

@simple_cache
def square(n):
    return n * n

print(square(4))
print(square(4))
print(square(5))
```

**运行结果**：
```text
Computing 4...
16
Fetching 4 from cache...
16
Computing 5...
25
```

### 4.3 异常处理 (`error_handler`)

为函数提供统一的容错层。

**代码示例**：

```python
def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            print("Error: Div by zero detected!")
            return None
    return wrapper

@safe_run
def div(a, b):
    return a / b

print(f"Result 1: {div(10, 2)}")
print(f"Result 2: {div(10, 0)}")
```

**运行结果**：
```text
Result 1: 5.0
Error: Div by zero detected!
Result 2: None
```
