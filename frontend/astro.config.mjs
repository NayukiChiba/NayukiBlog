import { defineConfig, envField } from "astro/config";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import node from "@astrojs/node";

// https://astro.build/config
export default defineConfig({
  // 从根目录读取 .env 文件
  vite: {
    envDir: "..",
    build: {
      // 启用 CSS 代码分割
      cssCodeSplit: true,
      // 压缩配置
      minify: "esbuild",
      // 分块策略，提升缓存效率
      rollupOptions: {
        output: {
          manualChunks: {
            // 将第三方库分离成独立 chunk
            vendor: ["react", "react-dom"],
          },
        },
      },
    },
  },
  // 🚀 Static 模式：用户端静态化，管理端通过 prerender = false 保持 SSR
  output: "static",
  adapter: node({
    mode: "standalone",
  }),

  // 预取链接，加速页面导航
  prefetch: {
    prefetchAll: true,
    defaultStrategy: "viewport", // 当链接进入视口时预取
  },

  // === 添加这一段 ===
  // 强制关闭跨站请求检查 (解决 403 Forbidden 问题)
  security: {
    checkOrigin: false,
  },
  // 集成插件（移除 compress 加速构建，生产环境可通过 nginx gzip 压缩）
  integrations: [],
  markdown: {
    // remark-math 解析数学公式，remark-gfm 支持表格等
    remarkPlugins: [remarkMath, remarkGfm],
    // rehype-katex 在构建时渲染数学公式为HTML
    rehypePlugins: [rehypeKatex],
    // 禁用构建时语法高亮，改用客户端 Prism.js CDN
    syntaxHighlight: false,
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
