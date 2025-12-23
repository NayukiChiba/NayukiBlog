import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import node from '@astrojs/node';

// https://astro.build/config
export default defineConfig({
  output: 'static',
  adapter: node({
    mode: 'standalone',
  }),
  // 5. 集成 MDX 以支持内嵌 Astro 组件
  integrations: [
    mdx(),
  ],
  markdown: {
    // 3. & 4. remark-gfm 支持表格、HTML 元素 (kbd, b, i 等)
    remarkPlugins: [remarkMath, remarkGfm],
    rehypePlugins: [rehypeKatex],
    // 使用 Shiki 进行代码高亮
    shikiConfig: {
      theme: 'github-light',
      wrap: true,
    },
    extendDefaultPlugins: true,
  },
});
