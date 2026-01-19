import { createHighlighter, type Highlighter } from 'shiki';

// 使用 globalThis 确保在 SSR 模式下跨请求持久化
const globalKey = '__shiki_highlighter__';

declare global {
  var __shiki_highlighter__: Highlighter | null;
}

globalThis[globalKey] = globalThis[globalKey] || null;

export async function getShikiHighlighter(): Promise<Highlighter> {
  if (!globalThis[globalKey]) {
    globalThis[globalKey] = await createHighlighter({
      themes: ['github-light'],
      langs: ['javascript', 'typescript', 'python', 'html', 'css', 'json', 'bash', 'shell', 'markdown', 'plaintext']
    });
  }
  return globalThis[globalKey]!;
}
