import { defineCollection, z } from 'astro:content';

// 定义博客文章集合
const blogCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.union([z.string(), z.date()]).transform((val) => {
      if (typeof val === 'string') return val;
      return val.toISOString().split('T')[0];
    }),
    category: z.string().optional(),
    tags: z.array(z.string()).optional(),
    description: z.string().optional(),
    image: z.string().optional(),
    status: z.enum(['published', 'draft', 'private']).default('published'),
    layout: z.string().optional(), // 兼容旧的 layout 字段
  }),
});

// 导出集合
export const collections = {
  blog: blogCollection,
};
