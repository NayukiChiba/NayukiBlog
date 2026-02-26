---
title: C++安装
date: 2026-02-25
category: C++学习
tags:
  - Cpp
  - 基础
description: 如何安装C++的环境
image: https://img.yumeko.site/file/blog/Cpp.png
status: published
---

## Windows纯手动安装g++/gcc（不用任何包管理器）

### 第一步：下载MinGW-w64压缩包

1. **访问下载地址**：
   打开浏览器，访问：(https://www.mingw-w64.org/source/)[MinGW官网]
2. 点击最新版的`13.0.0`，进入下载页面
3. **选择版本下载**：
   - 在页面中找到最新版本（如 `13.2.0-rt_v11-rev1`）
   - 下载这个文件：`x86_64-13.2.0-release-posix-seh-msvcrt-rt_v11-rev1.7z`
   - 文件大小约 80MB

### 第二步：安装MinGW-w64

1. **解压文件**：
   - 你需要安装 7-Zip 来解压 `.7z` 文件（如果没有的话，去[7-zip官网](https://7-zip.org/)下载）
   - 右键点击下载的文件 → "7-Zip" → "提取到当前位置"
   
2. **移动文件夹**：
   - 将解压出的文件夹（如 `mingw64`）剪切或复制到：
     ```
     D:\
     ```
   - 最终路径应为：
     ```
     D:\mingw64
     ```
   
   **重要**：不要放在`C:\Program Files`或`C:\Program Files (x86)`，这些路径有空格，容易出问题。

3. **文件夹结构检查**：
   打开 `D:\mingw64\bin`，你应该看到这些文件：
   - `g++.exe`（C++编译器）
   - `gcc.exe`（C编译器）
   - `gdb.exe`（调试器）
   - `mingw32-make.exe`（Make工具）

### 第三步：配置环境变量（最重要！）

这是让Windows系统能找到g++的关键步骤。

#### 方法A：图形界面设置（推荐新手）

1. **打开系统属性**：
   - 按 `Win + R`，输入 `sysdm.cpl`，按回车
   - 或右键点击"此电脑" → "属性" → "高级系统设置"

2. **编辑环境变量**：
   - 点击"高级"选项卡 → "环境变量"
   - 在"系统变量"中找到 `Path`，双击它

3. **添加新路径**：
   - 点击"新建"
   - 输入：`D:\mingw64\bin`
   - 点击"确定"保存所有窗口

4. **验证是否生效**：
   - 按 `Win + R`，输入 `cmd`，打开命令提示符
   - 输入：`g++ --version`
   - 如果显示版本信息，说明成功！

### 第四步：VSCode配置和测试

#### 1. 安装VSCode扩展
在VSCode中安装这两个必须的扩展：
- **C/C++**（微软官方，提供代码高亮、智能提示）
- **C/C++ Extension Pack**（包含常用工具，可选但推荐）

#### 2. 创建测试项目

在VSCode中按顺序操作：

1. **新建文件夹**，比如 `C:\cpp_projects\test`
2. **用VSCode打开这个文件夹**
3. **新建文件** `hello.cpp`，输入：
   ```cpp
   #include <iostream>
   using namespace std;
   
   int main() {
       cout << "Hello, C++ World!" << endl;
       cout << "G++安装成功！" << endl;
       return 0;
   }
   ```

#### 3. 三种编译运行方式

##### 方式1：使用VSCode内置终端（最常用）
- 按 `` Ctrl + ` `` 打开终端
- 输入命令：
  ```bash
  g++ hello.cpp -o hello.exe
  .\hello.exe
  ```

##### 方式2：使用任务（Tasks）
1. 按 `Ctrl+Shift+P`，输入 `Tasks: Configure Task`
2. 选择 `C/C++: g++.exe build active file`
3. 这会生成 `.vscode/tasks.json` 文件
4. 按 `Ctrl+Shift+B` 编译

### 第五步：完整验证检查表

运行以下所有命令，确保都能成功：

```bash
# 1. 检查编译器
g++ --version
gcc --version

# 2. 检查调试器
gdb --version

# 3. 检查Make工具
mingw32-make --version
```

### 第六步：安装CMake（可选但推荐）
#### 手动安装CMake（推荐）

既然你不想用包管理器，手动安装CMake：

1. **下载CMake**：
   - 访问：`https://cmake.org/download/`
   - 下载 `Windows x64 Installer`

2. **安装时注意**：
   - 运行安装程序
   - **重要**：勾选`Add CMake to the system PATH for all users`
   - 完成安装

3. **验证**：
   ```bash
   cmake --version
   ```

#### 使用winget安装
```shell
winget install --id Kitware.CMake -e --silent --accept-source-agreements --accept-package-agreements
```

* 这种安装就是直接安装在`C:/ProgramFile/Cmake`文件夹下
### 常见问题解决

#### 问题1：`g++`命令不生效
**解决**：
1. 检查环境变量路径是否正确：`D:\mingw64\bin`
2. **重启VSCode**（重要！）
3. 重启电脑
4. 在VSCode终端输入：`echo %PATH%` 查看路径

#### 问题2：编译时出现"Permission denied"
**解决**：
1. 关闭可能正在运行的 `hello.exe` 进程
2. 或改名编译：`g++ hello.cpp -o hello2.exe`

#### 问题3：VSCode报红波浪线（找不到头文件）
**解决**：
1. 按 `Ctrl+Shift+P`
2. 输入 `C/C++: Edit Configurations (UI)`
3. 在"编译器路径"输入：`C:/mingw64/bin/g++.exe`
4. 在"IntelliSense 模式"选择：`windows-gcc-x64`
