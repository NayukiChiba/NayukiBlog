---
title: Python的functools模块
date: 2026-01-26
category: PythonAdvanceUsage
tags:
  - Python
  - 高级教程
description: Python functools 模块的详细解析，涵盖偏函数(partial)、归约计算(reduce)、装饰器元数据保留(wraps)、LRU缓存(lru_cache)、单分派泛型函数(singledispatch)、类比较补全(total_ordering)和比较键转换(cmp_to_key)等核心功能。
image: https://img.yumeko.site/file/blog/Functools.png
status: published
---
# 深入解析 Python functools 模块

## 1. 模块概述

`functools` 是 Python 标准库中专门用于处理 **高阶函数 (Higher-order Functions)** 的模块。高阶函数是指那些作用于函数或返回函数的函数。该模块提供了用于缓存、重载、偏函数应用等场景的工具，是编写高效、通用代码的重要基础设施。

---

## 2. 核心功能组件详解

### 2.1 偏函数应用 (partial)

**技术原理**：
偏函数应用 (Partial Application) 是一种将函数的某些参数固定住，从而产生一个新的、参数更少的函数的技术。这在回调函数、适配器模式中非常有用。

**代码示例**：

```python
import functools

def greet(greeting, name, punctuation="!"):
    return f"{greeting}, {name}{punctuation}"

# 创建偏函数，固定 greeting 为 "Hello"
say_hello = functools.partial(greet, "Hello")

# 创建偏函数，固定 greeting 和 punctuation
say_hi_period = functools.partial(greet, "Hi", punctuation=".")

# 调用新函数
print(say_hello("Alice"))
print(say_hi_period("Bob"))
```

**运行结果**：
```text
Hello, Alice!
Hi, Bob.
```

### 2.2 归约计算 (reduce)

**技术原理**：
`reduce` 对序列中的元素进行累积操作。它从序列左侧开始，将二元函数应用到前两个元素，然后将结果与第三个元素应用，以此类推。

**计算流图**：
假设列表为 `[1, 2, 3, 4]`，函数为 `add`：

![Example](https://img.yumeko.site/file/articles/PythonAdvanceUsage/Functools/Example.svg)

**代码示例**：

```python
import functools

numbers = [1, 2, 3, 4, 5]

# 累积求和
sum_val = functools.reduce(lambda x, y: x + y, numbers)
# 累积求积
prod_val = functools.reduce(lambda x, y: x * y, numbers)
# 寻找最大值 (逻辑：如果 x>y 返回 x 否则 y)
max_val = functools.reduce(lambda x, y: x if x > y else y, numbers)

print(f"Sum: {sum_val}")
print(f"Product: {prod_val}")
print(f"Max: {max_val}")
```

**运行结果**：
```text
Sum: 15
Product: 120
Max: 5
```

### 2.3 装饰器元数据保留 (wraps)

**问题背景**：
手动编写装饰器时，返回的 `wrapper` 函数会覆盖原函数的 `__name__`、`__doc__` 等属性。这会导致反射机制失效，生成的文档不正确。

**代码示例**：

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # 关键行：保留元数据
    def wrapper(*args, **kwargs):
        """Wrapper docstring"""
        print("Calling decorated function")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def example():
    """Original docstring"""
    print("Called example")

print(f"Name: {example.__name__}")
print(f"Doc: {example.__doc__}")
```

**运行结果**：
```text
Name: example
Doc: Original docstring
```
*(如果不加 `@wraps`，输出将会是 `Name: wrapper` 和 `Doc: Wrapper docstring`)*

### 2.4 LRU 缓存策略 (lru_cache)

**概念与算法**：
LRU (Least Recently Used) 是一种缓存淘汰策略。
*   **Memoization (记忆化)**：将函数输入输出对存储在内存中，下次遇到相同输入直接返回结果，避免重复计算。
*   **淘汰机制**：当缓存达到 `maxsize` 时，优先丢弃最久未被访问的数据。

**代码示例**：

```python
import functools
import time

@functools.lru_cache(maxsize=32)
def heavy_compute(n): 
    # 模拟耗时操作
    time.sleep(0.1) 
    return n * n

print("第一次调用 (耗时):")
start = time.time()
print(f"Result: {heavy_compute(4)}")
print(f"Time: {time.time() - start:.4f}s")

print("\n第二次调用 (命中缓存，极快):")
start = time.time()
print(f"Result: {heavy_compute(4)}")
print(f"Time: {time.time() - start:.4f}s")

print(f"\nCache Info: {heavy_compute.cache_info()}")
```

**运行结果**：
```text
第一次调用 (耗时):
Result: 16
Time: 0.1005s

第二次调用 (命中缓存，极快):
Result: 16
Time: 0.0000s

Cache Info: CacheInfo(hits=1, misses=1, maxsize=32, currsize=1)
```

### 2.5 单分派泛型函数 (singledispatch)

**技术原理**：
Python 是动态类型语言，不支持像 C++/Java 那样的基于参数类型的静态函数重载。`singledispatch` 实现了一种基于第一个参数类型的动态分派机制。

**代码示例**：

```python
import functools

@functools.singledispatch
def process(arg):
    print(f"Default processing: {arg}")

@process.register(int)
def _(arg):
    print(f"Processing Integer: {arg * 2}")

@process.register(list)
def _(arg):
    print(f"Processing List: len={len(arg)}")

process(10)          # 匹配 int
process([1, 2, 3])   # 匹配 list
process("hello")     # 匹配 默认
```

**运行结果**：
```text
Processing Integer: 20
Processing List: len=3
Default processing: hello
```

### 2.6 类比较补全 (total_ordering)

**作用**：
在自定义类中，定义完整的比较方法（`__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`）通常很繁琐。
`total_ordering` 是一个类装饰器。你只需要定义 `__eq__` 和其余四个（`lt`, `le`, `gt`, `ge`）中的任意一个，它会自动补全剩余的比较方法。

**代码示例**：

```python
import functools

@functools.total_ordering
class Student:
    def __init__(self, grade):
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade

s1 = Student(80)
s2 = Student(90)

# 我们只定义了 eq 和 lt，但自动拥有了 gt (>) 和 le (<=)
print(f"s1 > s2? {s1 > s2}")
print(f"s1 <= s2? {s1 <= s2}")
```

**运行结果**：
```text
s1 > s2? False
s1 <= s2? True
```

### 2.7 比较键转换 (cmp_to_key)

**历史背景**：
Python 2 的排序函数支持 `cmp` 参数（接收两个元素返回 -1/0/1）。Python 3 移除了 `cmp`，全面转向 `key` 参数（接收一个元素返回一个用于排序的特征值）。
`cmp_to_key` 是一个适配器，将旧式的比较函数转换为新式的键函数对象。

**代码示例**：

```python
import functools

def compare_items(a, b):
    # 自定义排序逻辑：按长度的倒序排列
    if len(a) > len(b): return -1
    elif len(a) < len(b): return 1
    else: return 0

words = ["banana", "apple", "pear"]

# 转换为 key 函数
sorted_words = sorted(words, key=functools.cmp_to_key(compare_items))
print(sorted_words)
```

**运行结果**：
```text
['banana', 'apple', 'pear']
```

---

## 3. 最佳实践总结

*   使用 `partial` 替代简单的 lambda 表达式，可以保留更好的元数据和可读性。
*   在编写任何装饰器时，务必加上 `@wraps`。
*   对于计算密集且无副作用的纯函数，考虑加上 `lru_cache` 提升性能。
*   使用 `singledispatch` 替代复杂的 `if isinstance(...)` 链，符合开闭原则 (Open/Closed Principle)。

