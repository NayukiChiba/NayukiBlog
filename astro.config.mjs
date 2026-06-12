import { defineConfig } from "astro/config";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkWikiLink from "remark-wiki-link";
import remarkDirective from "remark-directive";
import { remarkLeafDirectives } from "./remark-leaf-directives.mjs";
import { remarkContainerDirectives } from "./remark-container-directives.mjs";
import { rehypeLazyImages } from "./rehype-lazy-images.mjs";
import { rehypeListMarkers } from "./rehype-list-markers.mjs";

// https://astro.build/config
export default defineConfig({
  // 纯静态输出
  output: "static",

  // 预取链接：悬停时才后台预取，避免批量加载阻塞导航响应
  prefetch: {
    prefetchAll: true,
    defaultStrategy: "hover",
  },

  // 集成插件
  integrations: [],

  // Markdown 配置
  markdown: {
    remarkPlugins: [
      remarkDirective,
      remarkLeafDirectives,
      remarkContainerDirectives,
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
    rehypePlugins: [rehypeKatex, rehypeLazyImages, rehypeListMarkers],
    syntaxHighlight: "shiki",
    shikiConfig: {
      theme: "github-light",
      langs: [
        "json", "yaml", "toml", "xml", "ini",
        "bash", "shell", "powershell",
        "python", "javascript", "typescript", "html", "css", "scss",
        "markdown",
        "sql", "graphql",
        "dockerfile", "diff", "git-commit", "git-rebase",
        "c", "cpp", "rust", "go", "java", "kotlin",
        "r", "matlab",
        "latex", "tex",
      ],
      // 代码不自动换行，长行通过横向滚动查看
      wrap: false,
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
