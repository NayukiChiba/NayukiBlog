import { defineConfig } from "astro/config";
import node from "@astrojs/node";

// https://astro.build/config
export default defineConfig({
  // 基础路径设置为 /admin
  base: "/admin",

  // 从根目录读取 .env 文件
  vite: {
    envDir: "..",
    build: {
      cssCodeSplit: true,
      minify: "esbuild",
    },
  },
  // Admin 使用 SSR 模式
  output: "server",
  adapter: node({
    mode: "standalone",
  }),

  // 强制关闭跨站请求检查
  security: {
    checkOrigin: false,
  },

  integrations: [],

  server: {
    port: 4322,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
