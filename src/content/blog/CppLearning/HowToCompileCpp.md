---
title: 如何编译Cpp
date: 2026-02-26
category: C++学习
tags:
  - Cpp
  - 基础
description: 使用g++和cmake编译c++项目
image: https://img.yumeko.site/file/blog/Cmake.jpg
status: published
---

# 使用 g++ (GNU C++ 编译器) 直接编译

`g++` 是 GNU 编译器集合 (GCC) 中的 C++ 编译器，适合编译简单的、文件数量不多的程序。

## 编译单个源文件
最基本的命令，将 `hello.cpp` 编译为名为 `hello` 的可执行文件。
```bash
g++ -o hello hello.cpp
```
- `-o hello`: 指定输出可执行文件的名称为 `hello`。
- `hello.cpp`: 源代码文件。

## 常用编译选项
- **指定C++标准**： `-std=c++11` / `-std=c++14` / `-std=c++17` / `-std=c++20`
- **显示所有警告**： `-Wall` （强烈建议启用）
- **生成调试信息**： `-g` （用于 GDB 调试器）
- **优化级别**： `-O0` (不优化), `-O1`, `-O2` (常用), `-O3` (激进优化)

**示例：使用C++17标准并显示警告**
```bash
g++ -std=c++17 -Wall -o myapp main.cpp
```

## 编译多个源文件的项目
假设项目有 `main.cpp`, `utils.cpp`, `helper.cpp` 和对应的头文件。
```bash
# 方法一：一步到位（适合小项目）
g++ -std=c++17 -Wall -o project main.cpp utils.cpp helper.cpp

# 方法二：分步编译（适合稍大项目，避免重复编译未修改的文件）
# 1. 将每个.cpp文件编译成.o（目标文件）
g++ -std=c++17 -Wall -c main.cpp -o main.o
g++ -std=c++17 -Wall -c utils.cpp -o utils.o
g++ -std=c++17 -Wall -c helper.cpp -o helper.o
# 2. 将所有的.o文件链接成可执行文件
g++ -std=c++17 -Wall -o project main.o utils.o helper.o
```

---

# 使用 CMake 管理并编译项目

CMake 是一个跨平台的**构建系统生成器**。它不直接编译代码，而是根据 `CMakeLists.txt` 配置文件，生成对应平台（如 Unix 的 `Makefile` 或 Windows 的 `Visual Studio` 项目文件）的构建脚本，然后再调用该平台的原生工具链进行编译。

这种方法适合管理结构复杂、跨平台的中大型项目。

**基本项目结构示例**
```
MyProject/
├── CMakeLists.txt    # CMake 配置文件
├── src/
│   ├── main.cpp
│   └── utils.cpp
├── include/
│   └── utils.h
└── build/            # 建议在此目录进行构建（保持源码目录清洁）
```

## 编写 CMakeLists.txt
在项目根目录创建此文件。
```cmake
# 指定CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 定义项目名称和使用的语言（CXX 代表 C++）
project(MyProject VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可执行目标
# 语法：add_executable(目标名 源文件1 源文件2 ...)
add_executable(my_app
    src/main.cpp
    src/utils.cpp
)

# 指定头文件搜索路径
target_include_directories(my_app PRIVATE include)

# 如果项目需要链接外部库，例如 pthread
# find_package(Threads REQUIRED)
# target_link_libraries(my_app PRIVATE Threads::Threads)
```

## 生成构建系统并编译
在命令行中操作：
```bash
# 1. 进入项目根目录
cd /path/to/MyProject

# 2. 创建一个独立的构建目录（通常是 `build`）并进入
mkdir build
cd build

# 3. 运行 cmake 命令生成构建系统（.. 表示 CMakeLists.txt 在上一级目录）
cmake ..

# 4. 运行生成的构建系统来实际编译项目
#    在 Linux/macOS 上，CMake 默认生成 Makefile，所以使用 make
make
#    在 Windows 上，如果生成的是 Visual Studio 解决方案，则用对应 IDE 打开，或使用 cmake --build .
```

编译完成后，可执行文件 `my_app` 将生成在 `build` 目录中。

**进阶特性（了解）**
- **添加子目录**： 对于更复杂的项目，可以使用 `add_subdirectory(subdir)` 来管理子模块。
- **创建库**： 使用 `add_library(my_lib STATIC/SHARED source.cpp)` 创建静态库或动态库，然后通过 `target_link_libraries(my_app PRIVATE my_lib)` 链接。
- **查找并使用系统库**： 使用 `find_package()` 命令。

---

## 总结与选择

| 方面 | g++ | CMake |
| :--- | :--- | :--- |
| **适用场景** | 快速编译单个或少量文件的小程序、脚本或测试。 | 管理结构化的、跨平台的中大型项目。 |
| **优点** | 命令直接，无需额外配置文件，灵活快速。 | 配置与构建分离，跨平台支持好，项目结构清晰，易于管理依赖和复杂构建逻辑。 |
| **缺点** | 命令随着文件增多变得冗长，手动管理依赖和编译顺序繁琐，跨平台性差。 | 需要学习 CMake 语法，对于极简单的项目显得“重”。 |

**建议**：
- **初学者/简单程序**： 从 `g++` 命令行开始，理解编译、链接的基本过程。
- **正式项目/多人协作/跨平台需求**： **务必使用 CMake**，它是现代 C/C++ 项目的业界标准构建工具。