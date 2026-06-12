---
title: Python collections 库完全指南——高性能数据结构实战
date: 2026-06-12
category: 编程/Python
tags:
  - Python
  - 数据结构
  - 高级教程
description: 深入掌握 Python collections 标准库，涵盖 namedtuple、deque、Counter、defaultdict、OrderedDict、ChainMap、UserDict 等容器类型的实战用法与性能对比
image: https://img.yumeko.site/file/blog/cover/1781275900618_CollectionsBanner.webp
status: published
---

## 1. 问题的起点——为什么需要 collections

Python 内置的 `list`、`dict`、`tuple` 已经很强了，但在特定场景下代码会变得臃肿。比如：

```python
# 需求：统计一段文本中每个单词的出现次数
text = "apple banana apple orange banana apple"

# 用原生 dict 写：
word_count = {}
for word in text.split():
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
print(word_count)  # {'apple': 3, 'banana': 2, 'orange': 1}
```

```python
# 用 Counter 写——一行搞定：
from collections import Counter
print(Counter(text.split()))  # Counter({'apple': 3, 'banana': 2, 'orange': 1})
```

`collections` 是 Python 标准库中专门提供**高性能、专用容器**的模块。它补齐了内置类型的缺口——不是为了替代 `list`/`dict`，而是**当你发现自己在写重复样板代码时，collections 里通常已经有一个更好的工具**。

> [!NOTE] 本文覆盖
> `namedtuple`、`deque`、`Counter`、`defaultdict`、`OrderedDict`、`ChainMap`、`UserDict`/`UserList`/`UserString`。`deque` 和 `Counter` 是**最常用**的。

## 2. namedtuple——有名字的元组

元组的痛点：只能用下标 `[0]`、`[1]` 访问，代码可读性差。`namedtuple` 给每个位置加上字段名，既保持了元组的轻量和不便修改，又有了面向对象的可读性。

![NamedTuple.png](https://img.yumeko.site/file/blog/articles/1781275966322_NamedTuple.webp)

### 2.1 基本用法

```python
from collections import namedtuple

# 定义类型：类名=Person, 字段=name age job
Person = namedtuple("Person", ["name", "age", "job"])

# 创建实例
alice = Person("Alice", 25, "Engineer")
bob = Person(name="Bob", age=30, job="Designer")

# 字段访问（跟属性一样）
print(alice.name)       # Alice
print(alice.age)        # 25
print(alice.job)        # Engineer

# 下标访问（兼容 tuple）
print(alice[0])         # Alice
print(alice[1])         # 25

# 解包
name, age, job = alice
print(name, age, job)   # Alice 25 Engineer
```

**运行结果：**

```text
Alice
25
Engineer
Alice
25
Alice 25 Engineer
```

### 2.2 常用方法

```python
# _asdict() —— 转为 OrderedDict（方便 JSON 序列化）
print(alice._asdict())
# {'name': 'Alice', 'age': 25, 'job': 'Engineer'}

# _replace() —— 创建副本并替换指定字段（原对象不变）
alice_older = alice._replace(age=26)
print(alice_older)       # Person(name='Alice', age=26, job='Engineer')

# _make() —— 从可迭代对象创建
data = ["Charlie", 28, "Manager"]
charlie = Person._make(data)
print(charlie)          # Person(name='Charlie', age=28, job='Manager')

# _fields —— 查看字段名
print(Person._fields)   # ('name', 'age', 'job')
```

### 2.3 带默认值——Python 3.7+ 的 defaults

```python
# 方式 1：namedtuple 的 defaults 参数（Python 3.7+）
Person = namedtuple("Person", ["name", "age", "job"], defaults=["Unknown", 0])
# defaults 从右往左匹配：job 默认 "Unknown"，age 默认 0
print(Person("Alice"))  # Person(name='Alice', age=0, job='Unknown')

# 方式 2：typing.NamedTuple（推荐——支持类型注解+默认值）
from typing import NamedTuple

class Person(NamedTuple):
    name: str
    age: int = 0
    job: str = "Unknown"

alice = Person("Alice", job="Engineer")
print(alice)  # Person(name='Alice', age=0, job='Engineer')
```

**运行结果：**

```text
Person(name='Alice', age=0, job='Unknown')
Person(name='Alice', age=0, job='Engineer')
```

### 2.4 实战：解析 CSV 数据

```python
from collections import namedtuple

# 模拟 CSV 数据
csv_data = [
    "Alice,25,Engineer",
    "Bob,30,Designer",
    "Charlie,28,Manager",
]

Employee = namedtuple("Employee", ["name", "age", "job"])

employees = []
for row in csv_data:
    name, age, job = row.split(",")
    employees.append(Employee(name, int(age), job))

# 筛选：找出 25 岁以上的员工
seniors = [e for e in employees if e.age > 25]
for e in seniors:
    print(f"{e.name}: {e.age} 岁, {e.job}")
```

**运行结果：**

```text
Bob: 30 岁, Designer
Charlie: 28 岁, Manager
```

> [!TIP] namedtuple vs dataclass
> `namedtuple` 是不可变的（immutable）、可哈希、内存占用小（和 tuple 一样）。
> `dataclass` 是可变的、功能更丰富（支持方法、继承）。选 `namedtuple` 当你需要轻量、不可变的数据载体；选 `dataclass` 当你需要可变对象或类方法。

## 3. deque——高速双端队列

Python 的 `list` 在头部插入/删除是 O(n)（需要移动所有元素）。`deque`（double-ended queue，读作 "deck"）的头尾操作都是 O(1)，适合实现队列、栈、滑动窗口。

![Deque.png](https://img.yumeko.site/file/blog/articles/1781275968048_Deque.webp)

### 3.1 基本用法

```python
from collections import deque

# 创建 deque
dq = deque(["a", "b", "c"])
print(dq)  # deque(['a', 'b', 'c'])

# 尾部操作（与 list 相同）
dq.append("d")          # 尾部追加
dq.pop()                # 尾部弹出 → 'd'

# 头部操作（list 没有的高效操作）
dq.appendleft("z")      # 头部追加 → O(1)！
dq.popleft()            # 头部弹出 → 'z'  O(1)！
print(dq)               # deque(['a', 'b', 'c'])
```

**运行结果：**

```text
deque(['a', 'b', 'c'])
deque(['a', 'b', 'c'])
```

### 3.2 maxlen——固定长度队列

```python
# 固定长度为 3：满了之后 append 会自动从另一端弹出
dq = deque(maxlen=3)
dq.append(1)
dq.append(2)
dq.append(3)
print(dq)               # deque([1, 2, 3], maxlen=3)

dq.append(4)            # 第 4 个加入，第 1 个自动从左侧移出
print(dq)               # deque([2, 3, 4], maxlen=3)
```

**运行结果：**

```text
deque([1, 2, 3], maxlen=3)
deque([2, 3, 4], maxlen=3)
```

这是实现**滑动窗口**的最简洁方式。

### 3.3 rotate——旋转队列

```python
dq = deque(["a", "b", "c", "d"])
dq.rotate(1)            # 右移 1 位：最后一个移到最前面
print(dq)               # deque(['d', 'a', 'b', 'c'])

dq.rotate(-2)           # 左移 2 位
print(dq)               # deque(['b', 'c', 'd', 'a'])
```

**运行结果：**

```text
deque(['d', 'a', 'b', 'c'])
deque(['b', 'c', 'd', 'a'])
```

### 3.4 实战：最近 N 条历史记录

```python
from collections import deque

history = deque(maxlen=5)

def add_action(action: str):
    history.append(action)
    print(f"操作: {action}")

# 模拟用户操作
for action in ["打开文件", "编辑内容", "保存", "关闭", "重新打开", "撤销"]:
    add_action(action)

print(f"\n最近操作: {list(history)}")
```

**运行结果：**

```text
操作: 打开文件
操作: 编辑内容
操作: 保存
操作: 关闭
操作: 重新打开
操作: 撤销

最近操作: ['编辑内容', '保存', '关闭', '重新打开', '撤销']
# "打开文件" 已自动从左侧移出
```

### 3.5 实战：广度优先搜索（BFS）

```python
from collections import deque

def bfs(graph: dict, start: str):
    """用 deque 实现 BFS"""
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()          # O(1) 从头部取
        if node not in visited:
            visited.add(node)
            print(f"访问: {node}")
            # 将邻居加入队列尾部
            queue.extend(graph.get(node, []))

graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [], "E": [], "F": []
}
bfs(graph, "A")
```

**运行结果：**

```text
访问: A
访问: B
访问: C
访问: D
访问: E
访问: F
```

## 4. Counter——统计计数

`Counter` 是 `dict` 的子类，专门用来**计数**。任何可哈希的对象都可以作为键。

### 4.1 基本用法

```python
from collections import Counter

# 从可迭代对象创建
cnt = Counter("abracadabra")
print(cnt)  # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# 从 dict 创建
cnt2 = Counter({"apple": 3, "banana": 2})
print(cnt2)  # Counter({'apple': 3, 'banana': 2})

# 访问计数（不存在的 key 返回 0，而不是 KeyError！）
print(cnt["a"])     # 5
print(cnt["z"])     # 0 ← 和普通 dict 不同！
```

**运行结果：**

```text
Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
Counter({'apple': 3, 'banana': 2})
5
0
```

> [!TIP] Counter 的 `__missing__`
> 普通 `dict` 访问不存在的 key 会抛出 `KeyError`，但 `Counter` 返回 `0`。这让统计代码简洁得多——不需要 `if key in counter` 的防御性检查。

### 4.2 most_common——Top N

```python
cnt = Counter("abracadabra")

# 出现频率最高的 3 个
print(cnt.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]

# 全部按频率排序
print(cnt.most_common())   # [('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1)]
```

**运行结果：**

```text
[('a', 5), ('b', 2), ('r', 2)]
[('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1)]
```

### 4.3 加减运算

```python
c1 = Counter(a=3, b=2, c=1)
c2 = Counter(a=1, b=2, c=3)

# 加法：计数相加
print(c1 + c2)          # Counter({'a': 4, 'b': 4, 'c': 4})

# 减法：计数相减（只保留正数）
print(c1 - c2)          # Counter({'a': 2})

# 交集：取最小值
print(c1 & c2)          # Counter({'b': 2, 'a': 1, 'c': 1})

# 并集：取最大值
print(c1 | c2)          # Counter({'a': 3, 'c': 3, 'b': 2})
```

**运行结果：**

```text
Counter({'a': 4, 'b': 4, 'c': 4})
Counter({'a': 2})
Counter({'b': 2, 'a': 1, 'c': 1})
Counter({'a': 3, 'c': 3, 'b': 2})
```

### 4.4 实战：分析日志文件

```python
from collections import Counter

log_lines = [
    "ERROR: disk full",
    "WARN: high memory usage",
    "ERROR: disk full",
    "INFO: server started",
    "ERROR: connection timeout",
    "WARN: high memory usage",
    "ERROR: disk full",
]

# 提取 ERROR 级别
error_types = Counter(
    line.split(": ", 1)[1]
    for line in log_lines
    if line.startswith("ERROR")
)

print(error_types)
print(f"最常见的错误: {error_types.most_common(1)[0]}")
```

**运行结果：**

```text
Counter({'disk full': 3, 'connection timeout': 1})
最常见的错误: ('disk full', 3)
```

## 5. defaultdict——带默认值的字典

处理"分组"类需求时，普通 `dict` 需要检查 key 是否存在。`defaultdict` 在访问不存在的 key 时自动调用工厂函数创建默认值。

### 5.1 基本用法

```python
from collections import defaultdict

# int 工厂：默认值 0（计数）
counter = defaultdict(int)
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
for w in words:
    counter[w] += 1    # 不需要 if w in counter！
print(dict(counter))   # {'apple': 3, 'banana': 2, 'orange': 1}

# list 工厂：默认值 []（分组）
groups = defaultdict(list)
students = [("A班", "Alice"), ("B班", "Bob"), ("A班", "Charlie")]
for cls, name in students:
    groups[cls].append(name)  # 不需要 if cls in groups！
print(dict(groups))     # {'A班': ['Alice', 'Charlie'], 'B班': ['Bob']}

# set 工厂：默认值 set()（去重收集）
tags = defaultdict(set)
items = [("水果", "苹果"), ("水果", "香蕉"), ("蔬菜", "白菜"), ("水果", "苹果")]
for cat, item in items:
    tags[cat].add(item)
print(dict(tags))       # {'水果': {'苹果', '香蕉'}, '蔬菜': {'白菜'}}
```

**运行结果：**

```text
{'apple': 3, 'banana': 2, 'orange': 1}
{'A班': ['Alice', 'Charlie'], 'B班': ['Bob']}
{'水果': {'香蕉', '苹果'}, '蔬菜': {'白菜'}}
```

### 5.2 自定义工厂函数

```python
from collections import defaultdict

# 默认值为今天的日期
from datetime import date
last_seen = defaultdict(lambda: date.today())
last_seen["Alice"] = date(2026, 1, 15)
print(last_seen["Alice"])  # 2026-01-15
print(last_seen["Bob"])    # 2026-06-12（自动生成的默认值）

# 嵌套 defaultdict：树形结构
def nested_dict():
    return defaultdict(nested_dict)

tree = nested_dict()
tree["a"]["b"]["c"] = 1
print(tree["a"]["b"]["c"])  # 1
# 不需要 tree.setdefault("a", {}).setdefault("b", {}).setdefault("c", 1)
```

**运行结果：**

```text
2026-01-15
2026-06-12
1
```

### 5.3 实战：构建倒排索引

```python
from collections import defaultdict

def build_index(docs: dict[str, str]):
    """构建倒排索引：单词 → 包含该单词的文档列表"""
    index = defaultdict(set)
    for doc_id, content in docs.items():
        for word in content.lower().split():
            index[word].add(doc_id)
    return dict(index)

docs = {
    "doc1": "Python is great for data science",
    "doc2": "Java is great for enterprise",
    "doc3": "Python and Java are popular",
}
inverted = build_index(docs)

# 查询：哪些文档包含 "python"
print(inverted.get("python", set()))   # {'doc1', 'doc3'}
# 查询：哪些文档同时包含 "python" 和 "java"
print(inverted.get("python", set()) & inverted.get("java", set()))  # {'doc3'}
```

**运行结果：**

```text
{'doc3', 'doc1'}
{'doc3'}
```

## 6. OrderedDict——记住插入顺序的字典

Python 3.7+ 的普通 `dict` 已经保证插入顺序。那么 `OrderedDict` 还有什么用？

### 6.1 独有的能力

```python
from collections import OrderedDict

od = OrderedDict()
od["a"] = 1
od["b"] = 2
od["c"] = 3

# 1. move_to_end —— 把 key 移到末尾或开头
od.move_to_end("a")         # 把 "a" 移到末尾
print(od)                   # OrderedDict({'b': 2, 'c': 3, 'a': 1})
od.move_to_end("c", last=False)  # 把 "c" 移到开头
print(od)                   # OrderedDict({'c': 3, 'b': 2, 'a': 1})

# 2. popitem(last=False) —— 从头部弹出（FIFO）
od2 = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
first = od2.popitem(last=False)  # 从头部弹出 → ('a', 1)
print(first)                # ('a', 1)
print(od2)                  # OrderedDict({'b': 2, 'c': 3})

# 3. 相等比较对顺序敏感
d1 = OrderedDict(a=1, b=2)
d2 = OrderedDict(b=2, a=1)
print(d1 == d2)             # False（顺序不同）
# 普通 dict: {"a":1,"b":2} == {"b":2,"a":1} → True
```

**运行结果：**

```text
OrderedDict({'b': 2, 'c': 3, 'a': 1})
OrderedDict({'c': 3, 'b': 2, 'a': 1})
('a', 1)
OrderedDict({'b': 2, 'c': 3})
False
```

### 6.2 实战：LRU 缓存淘汰

```python
from collections import OrderedDict

class LRUCache:
    """最近最少使用缓存"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)         # 标记为最近使用
        return self.cache[key]

    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 淘汰最久未使用的

cache = LRUCache(2)
cache.put("a", 1)
cache.put("b", 2)
cache.get("a")                              # a 被访问，移到末尾
cache.put("c", 3)                           # b 被淘汰（最久未用）
print(list(cache.cache.keys()))             # ['a', 'c']
```

**运行结果：**

```text
['a', 'c']
```

> [!NOTE] 生产环境
> 实际项目建议用 `functools.lru_cache` 或 `functools.cache`（Python 3.9+）做函数缓存，用 `cachetools` 做自定义缓存策略。手写 LRU 仅用于学习或特殊需求。

## 7. ChainMap——多层字典叠加

当你有多层配置（命令行参数 → 环境变量 → 配置文件 → 默认值），`ChainMap` 把它们串成一条查找链：先查第一层，找不到再查第二层，依此类推。

![ChainMap.png](https://img.yumeko.site/file/blog/articles/1781275970722_ChainMap.webp)

### 7.1 基本用法

```python
from collections import ChainMap

# 三层配置：命令行 > 环境变量 > 默认值
cli_args = {"host": "prod.example.com"}
env_vars = {"port": 8080, "debug": False}
defaults = {"host": "localhost", "port": 3000, "debug": True, "timeout": 30}

config = ChainMap(cli_args, env_vars, defaults)

# 查找：从第一层开始，逐层回退
print(config["host"])       # prod.example.com（来自 cli_args）
print(config["port"])       # 8080（来自 env_vars，覆盖了 defaults 的 3000）
print(config["debug"])      # False（来自 env_vars）
print(config["timeout"])    # 30（来自 defaults，前两层都没有）
```

**运行结果：**

```text
prod.example.com
8080
False
30
```

### 7.2 修改——只影响第一层

```python
# 修改总是作用在第一层
config["debug"] = True      # 修改的是 cli_args
config["timeout"] = 60      # timeout 原本在 defaults，修改后被写入 cli_args

print(cli_args)
# {'host': 'prod.example.com', 'debug': True, 'timeout': 60}
print(defaults)
# {'host': 'localhost', 'port': 3000, 'debug': True, 'timeout': 30} ← 未变
```

**运行结果：**

```text
{'host': 'prod.example.com', 'debug': True, 'timeout': 60}
{'host': 'localhost', 'port': 3000, 'debug': True, 'timeout': 30}
```

### 7.3 常用属性和方法

```python
# maps —— 查看所有层
print(config.maps)
# [{'host': 'prod.example.com', 'debug': True, 'timeout': 60},
#  {'port': 8080, 'debug': False},
#  {'host': 'localhost', 'port': 3000, 'debug': True, 'timeout': 30}]

# new_child —— 添加新的顶层
new_args = {"verbose": True}
config = config.new_child(new_args)  # new_args 成为第一层
print(config["verbose"])    # True

# parents —— 去掉第一层
config = config.parents
print("verbose" in config)  # False（已经找不到）
```

**运行结果：**

```text
True
False
```

## 8. UserDict / UserList / UserString——自定义容器

当你需要创建一个**行为像 dict/list/str 但又需要自定义逻辑**的类时，**不要继承 `dict`**——因为 CPython 对内置类型的某些方法做了优化，会绕过你重写的方法。正确的做法是继承 `collections.UserDict`。

### 8.1 为什么不能直接继承 dict

```python
# 反例：继承 dict，__setitem__ 在某些情况下被绕过
class FailDict(dict):
    def __setitem__(self, key, value):
        print(f"设置 {key} = {value}")
        super().__setitem__(key, value)

fd = FailDict()
fd["a"] = 1         # ✓ 打印 "设置 a = 1"
fd.update({"b": 2}) # ✗ 没打印！update 绕过了 __setitem__
```

**运行结果：**

```text
设置 a = 1
（update 什么都没打印！说明 __setitem__ 被绕过了）
```

### 8.2 正确做法：继承 UserDict

```python
from collections import UserDict

class LoggingDict(UserDict):
    """记录所有写操作的字典"""

    def __setitem__(self, key, value):
        print(f"[LOG] 设置 {key} = {value}")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        print(f"[LOG] 删除 {key}")
        super().__delitem__(key)

ld = LoggingDict()
ld["a"] = 1          # [LOG] 设置 a = 1
ld.update({"b": 2})  # [LOG] 设置 b = 2  ← 正确触发了！
ld["b"] = 3          # [LOG] 设置 b = 3
del ld["a"]          # [LOG] 删除 a
print(dict(ld))      # {'b': 3}
```

**运行结果：**

```text
[LOG] 设置 a = 1
[LOG] 设置 b = 2
[LOG] 设置 b = 3
[LOG] 删除 a
{'b': 3}
```

### 8.3 实战：自动类型转换的字典

```python
from collections import UserDict

class TypedDict(UserDict):
    """限制值的类型"""

    def __init__(self, value_type, *args, **kwargs):
        self.value_type = value_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, self.value_type):
            raise TypeError(f"期望 {self.value_type}, 实际 {type(value)}")
        super().__setitem__(key, value)

scores = TypedDict(int)
scores["Alice"] = 95
scores["Bob"] = 87
try:
    scores["Charlie"] = "高分"      # TypeError
except TypeError as e:
    print(f"类型错误: {e}")

print(dict(scores))          # {'Alice': 95, 'Bob': 87}
```

**运行结果：**

```text
类型错误: 期望 <class 'int'>, 实际 <class 'str'>
{'Alice': 95, 'Bob': 87}
```

## 9. 总结

| 类型 | 一句话 | 典型场景 | 复杂度亮点 |
|:--|:--|:--|:--|
| `namedtuple` | 有名字的 tuple | 数据载体、CSV 行、API 返回 | 内存与 tuple 相同 |
| `deque` | 高速双端队列 | 队列、滑动窗口、BFS | 头尾插入 O(1) |
| `Counter` | 自动统计计数 | 词频、日志分析、Top N | `most_common` O(n log k) |
| `defaultdict` | 自动默认值的 dict | 分组、嵌套结构、倒排索引 | 省去 if-key-in-dict |
| `OrderedDict` | 顺序敏感的 dict | LRU 缓存、FIFO 队列 | `move_to_end` / `popitem(last=False)` |
| `ChainMap` | 多层配置叠加 | 配置优先级、作用域链 | 查找 O(k·n) |
| `UserDict` | 可安全继承的 dict | 自定义容器、类型限制 | 所有方法都走 Python 层 |

> [!ABSTRACT] 核心原则
> `collections` 的精髓不是"提供新功能"，而是**把你在项目里反复手写的样板代码，浓缩为标准库的高性能单行调用**。下次写循环统计、手动检查 key、用 list 当队列时——停下来，查查 collections。

---