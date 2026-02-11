---
title: Git 提交规范完整指南
date: 2026-02-06
category: 技术
tags:
  - git
  - 工具
description: Git提交消息的完整规范和示例
image: https://img.yumeko.site/file/blog/git.jpeg
status: published
---
# Git 提交规范完整指南

## 1. 提交消息结构

```bash
<type>(<scope>): <subject>
<空行>
<body>
<空行>
<footer>
```
## 2. Header 规范

### 2.1 Type (必需)

|Type|说明|示例|
|---|---|---|
|**feat**|新功能|feat: 添加SVM数据生成功能|
|**fix**|修复bug|fix: 修复数据生成索引越界问题|
|**docs**|文档变更|docs: 更新README使用说明|
|**style**|代码格式(不影响代码运行)|style: 格式化代码缩进|
|**refactor**|重构(既非新增功能,也非修复bug)|refactor: 重构数据处理模块|
|**perf**|性能优化|perf: 优化数据生成算法|
|**test**|测试相关|test: 添加SVM单元测试|
|**build**|构建系统或外部依赖变更|build: 升级scikit-learn版本|
|**ci**|CI配置变更|ci: 添加GitHub Actions配置|
|**chore**|其他不修改src或test的变更|chore: 更新gitignore|
|**revert**|回退之前的提交|revert: 回退feat: 添加功能A|

### 2.2 Scope (可选)
```bash
feat(svm): 添加SVM数据生成功能

fix(regression): 修复回归模块bug

docs(readme): 更新项目文档
```
### 2.3 Subject (必需)

- 使用中文简洁描述
- 不超过50个字符
- 动词开头,使用现在时
- 首字母小写
- 结尾不加句号

## 3. Body 规范 (可选)

详细描述改动内容:

- 使用中文
- 说明改动的原因和内容
- 每行不超过72个字符
- 可以分段
## 4. Footer 规范 (可选)

### 4.1 关闭Issue
```bash
close #123
closes #123, #456
fix #123
resolve #123
```
### 4.2 破坏性变更
```bash
BREAKING CHANGE: 说明破坏性变更的内容
```
## 5. 完整示例

### 5.1 简单提交
```bash
feat: 添加SVM数据生成功能
```

```bash
fix: 修复数据生成时的索引越界问题
```
### 5.2 带Scope的提交

```bash
feat(svm): 添加make_moons数据集生成器
```

```bash
fix(regression): 修复线性回归预测异常
```
### 5.3 完整提交
```bash
git commit -m "feat(svm): 添加SVM数据生成功能

- 实现基于make_moons的数据生成器
- 添加数据可视化功能
- 支持噪声参数配置
- 添加数据导出为CSV功能

本次更新为SVM模块提供了标准的测试数据集,
方便后续进行模型训练和验证。

close #123"
```
### 5.4 破坏性变更
```bash
git commit -m "refactor(api): 重构数据生成API接口

- 修改generate_data函数签名
- 移除deprecated参数
- 优化返回值结构

BREAKING CHANGE: generate_data函数不再支持old_param参数,
请使用new_param替代。详见迁移指南。

close #456"
```
### 5.5 Revert提交
```bash
git commit -m "revert: 回退feat(svm): 添加SVM数据生成功能

This reverts commit abc123def456.

回退原因: 发现该功能存在性能问题,需要重新设计"
```

## AI提示词
```
你是一个Git提交消息生成助手。请根据以下规范为我生成标准的Git提交消息：

## 提交格式
<type>(<scope>): <subject>

<body>

<footer>

## Type类型规范
- feat: 新功能
- fix: 修复bug
- docs: 文档变更
- style: 代码格式(不影响代码运行)
- refactor: 重构(既非新增功能，也非修复bug)
- perf: 性能优化
- test: 测试相关
- build: 构建系统或外部依赖变更
- ci: CI配置变更
- chore: 其他不修改src或test的变更
- revert: 回退之前的提交

## 编写要求
1. **Header**:
   - type: 必需，从上述类型中选择
   - scope: 可选，表示影响范围，使用英文小写
   - subject: 必需，使用中文简洁描述，不超过50字，动词开头，首字母小写，结尾不加句号

2. **Body** (可选):
   - 使用中文详细说明改动内容
   - 说明为什么做这个改动
   - 每行不超过72字符

3. **Footer** (可选):
   - 关闭Issue: close #123
   - 破坏性变更: BREAKING CHANGE: 说明内容

## 输出格式
请提供两个版本：
1. 简单版本（仅Header）
2. 完整版本（包含Body和Footer，如适用）

当前文件改动：[描述你的改动]
涉及的功能模块：[可选]
```
