---
title: Vue3 组合式 API 完全指南
date: 2026-01-24
category: 技术
tags:
  - Vue3
  - 前端
  - JavaScript
  - TypeScript
description: 深入理解 Vue3 组合式 API，从 Options API 到 Composition API 的优雅迁移
image: https://img.yumeko.site/file/wife/早坂爱.jpg
status: public
---

# Vue3 组合式 API 完全指南

Vue3 引入了组合式 API（Composition API），这是 Vue 生态系统中最重要的变革之一。本文将深入探讨组合式 API 的核心概念和最佳实践。

## 为什么需要组合式 API？

### Options API 的局限性

在 Vue2 中，我们使用 Options API 组织代码：

```javascript
export default {
  data() {
    return {
      count: 0,
      name: ''
    }
  },
  computed: {
    doubleCount() {
      return this.count * 2
    }
  },
  methods: {
    increment() {
      this.count++
    }
  },
  mounted() {
    console.log('mounted')
  }
}
```

**问题**：当组件变得复杂时，同一个功能的代码被分散在 `data`、`methods`、`computed` 等不同选项中，难以维护。

### 组合式 API 的优势

- **逻辑复用**：通过组合函数（Composables）轻松复用逻辑
- **更好的类型推断**：对 TypeScript 更友好
- **代码组织**：相关逻辑可以放在一起
- **更小的打包体积**：支持 Tree-shaking

## 核心概念

### 1. setup() 函数

`setup()` 是组合式 API 的入口：

```vue
<script>
import { ref, onMounted } from 'vue'

export default {
  setup() {
    const count = ref(0)
    
    function increment() {
      count.value++
    }
    
    onMounted(() => {
      console.log('组件已挂载')
    })
    
    // 返回的内容可以在模板中使用
    return {
      count,
      increment
    }
  }
}
</script>
```

### 2. `<script setup>` 语法糖

Vue3.2+ 推荐使用 `<script setup>`，更简洁：

```vue
<script setup>
import { ref, onMounted } from 'vue'

const count = ref(0)

function increment() {
  count.value++
}

onMounted(() => {
  console.log('组件已挂载')
})
</script>

<template>
  <button @click="increment">{{ count }}</button>
</template>
```

## 响应式 API

### ref() - 基本类型响应式

```javascript
import { ref } from 'vue'

const count = ref(0)
const name = ref('Nayuki')

// 在 JS 中需要 .value
console.log(count.value) // 0
count.value++

// 在模板中自动解包
// <span>{{ count }}</span>
```

### reactive() - 对象响应式

```javascript
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  user: {
    name: 'Nayuki',
    age: 20
  }
})

// 直接访问，无需 .value
state.count++
state.user.name = 'Yumeko'
```

### computed() - 计算属性

```javascript
import { ref, computed } from 'vue'

const count = ref(0)

// 只读计算属性
const doubleCount = computed(() => count.value * 2)

// 可写计算属性
const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`
  },
  set(newValue) {
    [firstName.value, lastName.value] = newValue.split(' ')
  }
})
```

### watch() - 侦听器

```javascript
import { ref, watch, watchEffect } from 'vue'

const count = ref(0)
const name = ref('Nayuki')

// 侦听单个 ref
watch(count, (newVal, oldVal) => {
  console.log(`count 从 ${oldVal} 变为 ${newVal}`)
})

// 侦听多个源
watch([count, name], ([newCount, newName], [oldCount, oldName]) => {
  console.log('count 或 name 变化了')
})

// watchEffect - 自动追踪依赖
watchEffect(() => {
  console.log(`count is: ${count.value}`)
})
```

## 生命周期钩子

```javascript
import {
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue'

onMounted(() => {
  console.log('组件挂载完成')
})

onUnmounted(() => {
  console.log('组件卸载，清理资源')
})
```

## 组合函数（Composables）

组合函数是复用有状态逻辑的利器：

### 示例：useMouse

```javascript
// composables/useMouse.js
import { ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(event) {
    x.value = event.pageX
    y.value = event.pageY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}
```

### 示例：useFetch

```javascript
// composables/useFetch.js
import { ref, watchEffect } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(true)

  watchEffect(async () => {
    loading.value = true
    try {
      const res = await fetch(url.value || url)
      data.value = await res.json()
    } catch (e) {
      error.value = e
    } finally {
      loading.value = false
    }
  })

  return { data, error, loading }
}
```

### 在组件中使用

```vue
<script setup>
import { useMouse } from '@/composables/useMouse'
import { useFetch } from '@/composables/useFetch'

const { x, y } = useMouse()
const { data, loading } = useFetch('/api/users')
</script>

<template>
  <div>鼠标位置: {{ x }}, {{ y }}</div>
  <div v-if="loading">加载中...</div>
  <div v-else>{{ data }}</div>
</template>
```

## 依赖注入

### provide / inject

```javascript
// 父组件
import { provide, ref } from 'vue'

const theme = ref('dark')
provide('theme', theme)

// 子组件（任意层级）
import { inject } from 'vue'

const theme = inject('theme', 'light') // 第二个参数是默认值
```

## TypeScript 支持

组合式 API 对 TypeScript 有极佳的支持：

```typescript
import { ref, computed } from 'vue'

interface User {
  id: number
  name: string
  email: string
}

const user = ref<User | null>(null)
const users = ref<User[]>([])

const userName = computed(() => user.value?.name ?? '未知')
```

### 定义 Props 和 Emits

```vue
<script setup lang="ts">
interface Props {
  title: string
  count?: number
}

const props = withDefaults(defineProps<Props>(), {
  count: 0
})

const emit = defineEmits<{
  (e: 'update', value: number): void
  (e: 'submit'): void
}>()

function handleClick() {
  emit('update', props.count + 1)
}
</script>
```

## 最佳实践

### 1. 命名规范

组合函数以 `use` 开头：

- `useMouse`
- `useFetch`
- `useLocalStorage`

### 2. 返回响应式对象

```javascript
// ✅ 好的做法
export function useCounter() {
  const count = ref(0)
  const increment = () => count.value++
  
  return { count, increment }
}

// ❌ 避免解构丢失响应性
const { count } = useCounter() // count 是响应式的 ✅
```

### 3. 使用 toRefs 保持响应性

```javascript
import { reactive, toRefs } from 'vue'

function useFeature() {
  const state = reactive({
    x: 0,
    y: 0
  })
  
  // 使用 toRefs 让解构后仍保持响应性
  return toRefs(state)
}

const { x, y } = useFeature() // x, y 都是响应式的
```

## 总结

组合式 API 是 Vue3 的核心特性，它提供了：

- ✅ 更灵活的代码组织方式
- ✅ 更好的逻辑复用（Composables）
- ✅ 更优秀的 TypeScript 支持
- ✅ 更小的打包体积

如果你还在使用 Options API，是时候拥抱组合式 API 了！

## 参考资料

- [Vue3 官方文档](https://vuejs.org/)
- [组合式 API FAQ](https://vuejs.org/guide/extras/composition-api-faq.html)
- [VueUse - 组合函数集合](https://vueuse.org/)