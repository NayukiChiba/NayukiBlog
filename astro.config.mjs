import { defineConfig } from "astro/config";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkWikiLink from "remark-wiki-link";

// https://astro.build/config
export default defineConfig({
  // 纯静态输出
  output: "static",

  // 预取链接，加速页面导航
  prefetch: {
    prefetchAll: true,
    defaultStrategy: "viewport",
  },

  // 集成插件
  integrations: [],

  // Markdown 配置
  markdown: {
    remarkPlugins: [
      remarkMath,
      remarkGfm,
      // 将 Obsidian 双链 [[文章名]] 转换为 /posts/文章名 的 HTML 链接
      // 支持别名写法：[[文章名|显示文字]]
      [
        remarkWikiLink,
        {
          pageResolver: (name) => [name.replace(/\s+/g, "-").toLowerCase()],
          hrefTemplate: (permalink) => `/posts/${permalink}`,
          wikiLinkClassName: "wiki-link",
          aliasDivider: "|",
        },
      ],
    ],
    rehypePlugins: [rehypeKatex],
    syntaxHighlight: "shiki",
    shikiConfig: {
      theme: "github-light",
      langs: [],
      wrap: true,
    },
  },

  // Vite 配置
  vite: {
    // 忽略 Obsidian 配置文件和模板文件夹，避免热更新报错
    server: {
      watch: {
        ignored: ["**/.obsidian/**", "**/templates/**"],
      },
    },
    build: {
      cssCodeSplit: true,
      minify: "esbuild",
      // 确保输出文件使用 UTF-8 编码
      charset: "utf8",
    },
  },
});
