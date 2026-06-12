/**
 * 文章集合辅助函数
 *
 * 文章可见性规则：
 * - published：开发环境和生产环境均显示
 * - draft：仅开发环境（astro dev）显示，生产构建时不生成页面
 */
import { getCollection, type CollectionEntry } from "astro:content";

export type BlogPost = CollectionEntry<"blog">;

/** 判断文章在当前环境是否可见 */
export function isPostVisible(post: BlogPost): boolean {
  return import.meta.env.PROD ? post.data.status === "published" : true;
}

/** 获取当前环境可见的所有文章 */
export async function getVisiblePosts(): Promise<BlogPost[]> {
  return getCollection("blog", isPostVisible);
}
