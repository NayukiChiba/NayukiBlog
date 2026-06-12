import { visit } from 'unist-util-visit';

/**
 * rehype 插件：为正文图片注入懒加载属性
 *
 * 构建时给所有 <img> 加 loading="lazy" + decoding="async"，
 * 避免文章页打开时全量加载图片阻塞网络与主线程，
 * 保证加载过程中点击导航能迅速响应。
 */
export function rehypeLazyImages() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName !== 'img') return;
      node.properties ??= {};
      // 不覆盖已显式声明的属性
      node.properties.loading ??= 'lazy';
      node.properties.decoding ??= 'async';
    });
  };
}
