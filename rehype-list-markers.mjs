import { visit } from 'unist-util-visit';

/**
 * rehype 插件：为有序列表项注入真实序号
 *
 * Markdown 中被正文隔开的 "4. xxx" 会生成 <ol start="4">，
 * 但浏览器对 ::before 中 counter(list-item) 的 start 支持不一致，
 * 导致圆圈数字从 1 重新计数。
 * 构建时把每个 li 的实际序号写入 data-marker，CSS 用 attr() 直接渲染。
 */
export function rehypeListMarkers() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName !== 'ol') return;

      // 起始序号：尊重 start 属性，默认 1
      const start = parseInt(node.properties?.start ?? '1', 10);
      let index = Number.isNaN(start) ? 1 : start;

      for (const child of node.children) {
        if (child.type === 'element' && child.tagName === 'li') {
          child.properties ??= {};
          child.properties.dataMarker = String(index);
          index += 1;
        }
      }
    });
  };
}
