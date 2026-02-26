---
title: Markdown样式完整测试文档
date: 2023-01-01
category: 技术
tags:
  - Markdown
description: 这是一个用于测试所有Markdown样式的综合测试文档，包含标题、列表、表格、代码块、数学公式、图片等所有常用元素。
image: https://img.yumeko.site/file/wife/早坂爱.jpg
status: draft
---

# 一级标题 - Markdown 样式完整测试

这是一个完整的 Markdown 样式测试文档，用于验证博客系统中所有样式元素的渲染效果。

## 二级标题 - 文本样式测试

### 三级标题 - 基础文本格式

这是一段**加粗文本**，这是一段*斜体文本*，这是一段***加粗斜体文本***。

这是一段包含`内联代码`的文本。这是一段~~删除线文本~~。

这是一个[链接示例](https://nayuki.blog)，点击可以跳转。

#### 四级标题 - 列表测试

##### 无序列表

- 第一项
  - 嵌套第一项
  - 嵌套第二项
    - 深层嵌套
- 第二项
- 第三项

##### 有序列表

1. 第一步
2. 第二步
   1. 子步骤一
   2. 子步骤二
3. 第三步

##### 任务列表

- [x] 已完成的任务
- [ ] 未完成的任务
- [x] 另一个已完成的任务
- [ ] 待办事项

## 引用块测试

> 这是一个普通的引用块。
>
> 引用块可以包含多个段落。

> **重要提示：**
>
> 引用块内也可以使用其他 Markdown 格式，比如**加粗**、*斜体*和`代码`。

## 代码块测试

### JavaScript 代码

```javascript
// 这是JavaScript代码示例
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log(fibonacci(10)); // 55
```

### Python 代码

```python
# 这是Python代码示例
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
```

### TypeScript 代码

```typescript
// TypeScript 接口和类型定义
interface User {
  id: number;
  name: string;
  email: string;
}

type UserResponse = {
  data: User[];
  total: number;
  page: number;
};

async function fetchUsers(): Promise<UserResponse> {
  const response = await fetch("/api/users");
  return response.json();
}
```

### SQL 代码

```sql
-- 创建用户表
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 查询活跃用户
SELECT u.username, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
WHERE u.is_active = TRUE
GROUP BY u.id
HAVING post_count > 5
ORDER BY post_count DESC;
```

## 表格测试

### 基础表格

|  类型   | 名称  |     内容     | 代理  |
| :---: | :-: | :--------: | :-: |
| CNAME | www | domain.com | 开启  |
|   A   |  @  | domain.com | 开启  |
|   A   | \*  |  你的服务器 IP  | 开启  |

### 复杂表格

|  功能  |        描述         | 状态  | 优先级 | 负责人 |
| :--: | :---------------: | :-: | :-: | :-: |
| 用户认证 |  实现 JWT token 认证  | 已完成 | P0  | 张三  |
| 数据导出 | 支持 Excel 和 CSV 格式 | 进行中 | P1  | 李四  |
| 邮件通知 |     异步发送邮件通知      | 待开始 | P2  | 王五  |
| 性能优化 |     优化数据库查询性能     | 已完成 | P0  | 赵六  |
| 暗黑模式 |     支持暗黑主题切换      | 进行中 | P3  | 张三  |

### 数据对比表格

|    方案    |        优点         |     缺点     |    性能    |   成本   |
| :--------: | :-----------------: | :----------: | :--------: | :------: |
|   MySQL    | 成熟稳定，生态丰富  | 水平扩展困难 |  ⭐⭐⭐⭐  |   💰💰   |
| PostgreSQL | 功能强大，支持 JSON | 学习曲线较陡 | ⭐⭐⭐⭐⭐ |   💰💰   |
|  MongoDB   |   灵活的文档模型    | 事务支持较弱 |  ⭐⭐⭐⭐  |  💰💰💰  |
|   Redis    |      极高性能       |  内存占用大  | ⭐⭐⭐⭐⭐ | 💰💰💰💰 |

## 数学公式测试

### 行内公式

这是一个行内公式示例：$E = mc^2$，这是质能方程。

圆的面积公式：$A = \pi r^2$

### 块级公式

二次方程求根公式：

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

矩阵表示：

$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

泰勒级数展开：

$$
f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
$$

欧拉公式：

$$
e^{i\pi} + 1 = 0
$$

## 图片测试

> **注意：** 图片部分需要实际的图片 URL 才能正常显示，这里仅展示 Markdown 语法。

![示例图片](https://img.yumeko.site/file/wife/早坂爱.jpg)

## Obsidian Callouts 测试

> [!NOTE] 这是一个 Note 提示
> 这里是提示的内容。可以包含 **加粗** 和 _斜体_。

> [!INFO] 这是一个 Info 信息
> 这是一个默认的信息块。

> [!TIP] 这是一个 Tip 技巧
> 这是一个非常有用的技巧提示。

> [!SUCCESS] 这是一个 Success 成功
> 操作已成功完成！

> [!WARNING] 这是一个 Warning 警告
> 请注意，这是一个警告信息。

> [!DANGER] 这是一个 Danger 危险
> 这是一个危险操作，请小心！

> [!BUG] 这是一个 Bug
> 已知存在问题，正在修复中。

> [!QUESTION] 这是一个 Question 问题
> 这是一个常见问题解答。

> [!ABSTRACT] 这是一个 Abstract 摘要
> 本文主要介绍了 Callout 的各种类型和使用方法。

> [!EXAMPLE] 这是一个 Example 示例
> 以下是一个具体的代码示例，展示如何使用该功能。

> [!QUOTE] 这是一个 Quote 引用
> 纸上得来终觉浅，绝知此事要躬行。—— 陆游

> [!TODO] 这是一个 Todo 待办
> - [ ] 完成文档编写
> - [ ] 添加单元测试
> - [x] 实现核心功能

## 表格样式测试

| ID  | 姓名 | 邮箱                 |  角色  |                  状态                  | 备注                                                     |
| :-: | :--: | :------------------- | :----: | :------------------------------------: | :------------------------------------------------------- |
|  1  | 张三 | zhangsan@example.com | 管理员 | <span style="color:green">活跃</span>  | 虽然是很长的备注但是应该能自动换行或者处理得很好不会炸开 |
|  2  | 李四 | lisi@example.com     |  用户  |  <span style="color:red">禁用</span>   | ---                                                      |
|  3  | 王五 | wangwu@example.com   |  编辑  | <span style="color:orange">离线</span> | 测试很长很长很长很长很长很长很长很长很长很长很长的文本   |

## 长代码测试（测试收缩）

```javascript
// 这是一个非常长的代码块，用于测试默认折叠功能
function potentiallyLongFunction() {
  console.log("Start");
  // 模拟很多行代码
  let a = 1;
  let b = 2;
  let c = a + b;
  // ...
  // ...
  // ... (Repeating lines to ensure length)
  // ...
  // ...
  console.log("Middle");
  // ...
  // ...
  // ...
  console.log("End");
  return c;
}
```

## 分隔线测试

下面是一条分隔线：

---

上面是一条分隔线。

## 嵌套复杂内容测试

### 列表中的代码块

1. 安装依赖

   ```bash
   npm install
   ```

2. 启动开发服务器

   ```bash
   npm run dev
   ```

### 引用中的列表

> **开发规范：**
>
> - 使用 ESLint 进行代码检查
> - 使用 Prettier 格式化代码

## HTML 标签测试

<div style="background-color: #f0f0f0; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
  这是一个使用HTML标签的自定义容器。
</div>

<details>
<summary>点击展开详细内容</summary>

这是隐藏的详细内容，点击上方可以展开或折叠。

- 支持 Markdown 格式
- 可以包含**加粗**和*斜体*
- 还可以包含`代码`

</details>

## 特殊字符测试

### Emoji 表情

😀 😃 😄 😁 😆 😅 😂 🤣 ❤️ 💯 🎉 🚀 ⭐ 🔥 ✅ ❌

### 特殊符号

© ® ™ § ¶ † ‡ • ‣ ⁃ ◦ ⁌ ⁍

### 数学符号

± × ÷ ≠ ≈ ≤ ≥ ∞ ∫ ∑ ∏ √ ∛ ∜

## Obsidian 双链测试

普通双链跳转：[[MarkdownTest]]

带别名的双链：[[MarkdownTest|点击查看 Markdown 测试文档]]

多个双链：[[MarkdownTest]] 和 [[Interest/AstrbotDeployment|AstrbotDeployment]]

双链与普通链接对比：这是[[MarkdownTest|双链]]，这是[普通链接](https://nayuki.blog)。

## 总结

这个测试文档涵盖了：

- ✅ 标题层级（H1-H5）
- ✅ 文本格式（加粗、斜体、删除线）
- ✅ 列表（有序、无序、任务列表）
- ✅ 引用块
- ✅ 代码块（多种语言）
- ✅ 表格（基础、复杂、数据对比）
- ✅ 数学公式（行内、块级）
- ✅ 图片
- ✅ 分隔线
- ✅ Mermaid 图表（流程图、时序图、甘特图）
- ✅ 嵌套内容
- ✅ HTML 标签
- ✅ 特殊字符和 Emoji
- ✅ Obsidian 双链（`[[文章名]]` / `[[文章名|别名]]`）

如果所有样式都能正确渲染，说明博客系统的 Markdown 渲染功能完整可用！
