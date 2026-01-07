import { defineConfig, envField } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import node from '@astrojs/node';
import compress from '@playform/compress';

// https://astro.build/config
export default defineConfig({
  // 从根目录读取 .env 文件
  vite: {
    envDir: '..',
    build: {
      // 启用 CSS 代码分割
      cssCodeSplit: true,
      // 压缩配置
      minify: 'esbuild',
      // 分块策略，提升缓存效率
      rollupOptions: {
        output: {
          manualChunks: {
            // 将第三方库分离成独立 chunk
            vendor: ['react', 'react-dom'],
          },
        },
      },
    },
  },
  output: 'server',
  adapter: node({
    mode: 'standalone',
  }),

  // 预取链接，加速页面导航
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'viewport', // 当链接进入视口时预取
  },

  // === 添加这一段 ===
  // 强制关闭跨站请求检查 (解决 403 Forbidden 问题)
  security: {
    checkOrigin: false,
  },
  // 5. 集成 MDX 以支持内嵌 Astro 组件
  integrations: [
    mdx(),
    compress({
      CSS: true,
      HTML: true,
      Image: true,
      JavaScript: true,
      SVG: true,
    }),
  ],
  markdown: {
    // 3. & 4. remark-gfm 支持表格、HTML 元素 (kbd, b, i 等)
    remarkPlugins: [remarkMath, remarkGfm],
    rehypePlugins: [rehypeKatex],
    // 禁用内置语法高亮，由 [slug].astro 中的 Shiki 单例处理
    syntaxHighlight: false,
    extendDefaultPlugins: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
