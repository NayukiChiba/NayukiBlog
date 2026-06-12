import { visit } from 'unist-util-visit';

// 格式化数字：>=1000 显示为 1.2k
function formatCount(num) {
  if (typeof num !== 'number') return '--';
  if (num >= 1000) return (num / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(num);
}

// 构建时抓取 GitHub 仓库信息（10s 超时，失败返回 null 由客户端兜底）
async function fetchRepoInfo(repo) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    const response = await fetch(`https://api.github.com/repos/${repo}`, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'NayukiBlog',
        Accept: 'application/vnd.github+json',
      },
    });
    clearTimeout(timer);
    if (!response.ok) return null;
    const data = await response.json();
    if (data.message) return null;
    return {
      description: data.description || '',
      stars: data.stargazers_count,
      forks: data.forks_count,
    };
  } catch {
    console.warn(`GitHub 卡片构建时抓取失败: ${repo}（将由客户端兜底）`);
    return null;
  }
}

// 生成 GitHub 卡片 HTML（info 为 null 时输出占位，客户端脚本兜底）
function renderGithubCard(repo, owner, name, info) {
  const loadedAttr = info ? ' data-loaded="true"' : '';
  const desc = info ? info.description : '';
  const stars = info ? formatCount(info.stars) : '--';
  const forks = info ? formatCount(info.forks) : '--';

  return `
    <a href="https://github.com/${repo}" class="gc-card" target="_blank" rel="noopener noreferrer" data-repo="${repo}"${loadedAttr}>
      <div class="gc-title-bar">
        <svg class="gc-icon" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1C5.9225 1 1 5.9225 1 12C1 16.8675 4.14875 20.9787 8.52125 22.4362C9.07125 22.5325 9.2775 22.2025 9.2775 21.9137C9.2775 21.6525 9.26375 20.7862 9.26375 19.865C6.5 20.3737 5.785 19.1912 5.565 18.5725C5.44125 18.2562 4.905 17.28 4.4375 17.0187C4.0525 16.8125 3.5025 16.3037 4.42375 16.29C5.29 16.2762 5.90875 17.0875 6.115 17.4175C7.105 19.0812 8.68625 18.6137 9.31875 18.325C9.415 17.61 9.70375 17.1287 10.02 16.8537C7.5725 16.5787 5.015 15.63 5.015 11.4225C5.015 10.2262 5.44125 9.23625 6.1425 8.46625C6.0325 8.19125 5.6475 7.06375 6.2525 5.55125C6.2525 5.55125 7.17375 5.2625 9.2775 6.67875C10.1575 6.43125 11.0925 6.3075 12.0275 6.3075C12.9625 6.3075 13.8975 6.43125 14.7775 6.67875C16.8813 5.24875 17.8025 5.55125 17.8025 5.55125C18.4075 7.06375 18.0225 8.19125 17.9125 8.46625C18.6138 9.23625 19.04 10.2125 19.04 11.4225C19.04 15.6437 16.4688 16.5787 14.0213 16.8537C14.42 17.1975 14.7638 17.8575 14.7638 18.8887C14.7638 20.36 14.75 21.5425 14.75 21.9137C14.75 22.2025 14.9563 22.5462 15.5063 22.4362C19.8513 20.9787 23 16.8537 23 12C23 5.9225 18.0775 1 12 1Z"/></svg>
        <span class="gc-repo"><span>${owner}</span><span class="gc-slash">/</span><strong>${name}</strong></span>
        <svg class="gc-icon gc-external" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
      </div>
      <p class="gc-desc" data-repo-desc>${desc}</p>
      <div class="gc-meta">
        <span class="gc-meta-item">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"/></svg>
          <span data-repo-stars>${stars}</span>
        </span>
        <span class="gc-meta-item">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z"/></svg>
          <span data-repo-forks>${forks}</span>
        </span>
      </div>
    </a>`;
}

// 同步类 directive 处理器
const embedHandlers = {
  youtube: (node) => {
    const id = node.attributes?.id ?? '';
    if (!id) return false;
    return `<figure class="embed-video"><lite-youtube videoid="${id}"></lite-youtube></figure>`;
  },

  bilibili: (node) => {
    const id = node.attributes?.id ?? '';
    if (!id) return false;
    return `<figure class="embed-video"><iframe class="bilibili-player" src="//player.bilibili.com/player.html?bvid=${id}&p=1&autoplay=0&muted=0" allowfullscreen loading="lazy"></iframe></figure>`;
  },

  codepen: (node) => {
    const url = node.attributes?.url ?? '';
    if (!url) return false;
    const match = url.match(/codepen\.io\/([^/]+)\/pen\/([^/?#]+)/);
    if (!match) return false;
    const [, user, slug] = match;
    return `<figure><iframe class="codepen-embed" src="https://codepen.io/${user}/embed/${slug}?default-tab=result" loading="lazy"></iframe></figure>`;
  },
};

// 将节点替换为 HTML
function applyHtml(node, htmlContent) {
  node.type = 'html';
  node.value = htmlContent;
  delete node.name;
  delete node.attributes;
  delete node.children;
}

export function remarkLeafDirectives() {
  return async (tree) => {
    const asyncTasks = [];

    visit(tree, 'leafDirective', (node) => {
      // github 卡片：构建时异步抓取仓库数据
      if (node.name === 'github') {
        const repo = node.attributes?.repo ?? '';
        if (!repo) return;
        const [owner, name] = repo.split('/');
        if (!owner || !name) return;

        asyncTasks.push(
          fetchRepoInfo(repo).then((info) => {
            applyHtml(node, renderGithubCard(repo, owner, name, info));
          }),
        );
        return;
      }

      // 其余同步 directive
      const handler = embedHandlers[node.name];
      if (!handler) return;

      const htmlContent = handler(node);
      if (!htmlContent) return;

      applyHtml(node, htmlContent);
    });

    await Promise.all(asyncTasks);
  };
}
