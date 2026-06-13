/**
 * RSS 增量抓取脚本（CI / 构建前运行）
 *
 * 功能：
 * 1. 读取已累积的文章缓存 src/data/rssCache.json
 * 2. 抓取所有 active 订阅源的最新文章（复用 src/lib/rss.ts 的解析逻辑）
 * 3. 以文章链接为唯一键去重,把新文章合并进缓存（增量,不丢历史）
 * 4. 抓取失败的源保留其历史文章不动,避免一次失败清空累积
 * 5. 写回 src/data/rssCache.json
 *
 * 运行方式：
 *     npm run fetch:rss
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { fetchAllFeeds, type FeedItem } from "./src/lib/rss";
import type { RssFeed } from "./src/lib/data";

// ==================== 路径常量 ====================

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.join(ROOT, "src", "data", "rss.json");
const CACHE_PATH = path.join(ROOT, "src", "data", "rssCache.json");

// 每个源在缓存中最多保留的文章数（防止 JSON 无限膨胀,按日期倒序截断）
const MAX_KEEP_PER_FEED = 300;

// ==================== 类型定义 ====================

// 缓存中单个订阅源的结构
interface CachedFeed {
  id: number;
  title: string;
  link: string;
  ok: boolean;
  items: FeedItem[];
}

// 缓存文件整体结构
interface RssCache {
  generatedAt: string; // 最近一次抓取的 ISO 时间
  feeds: CachedFeed[];
}

// ==================== 辅助函数 ====================

/**
 * 读取 JSON 文件,失败时返回 fallback
 *
 * @param filepath JSON 文件路径
 * @param fallback 读取/解析失败时的兜底值
 */
function readJson<T>(filepath: string, fallback: T): T {
  if (!existsSync(filepath)) return fallback;
  try {
    return JSON.parse(readFileSync(filepath, "utf-8")) as T;
  } catch (error) {
    console.warn(`读取 JSON 失败: ${filepath}`, error);
    return fallback;
  }
}

/**
 * 合并新旧文章列表
 *
 * 以文章链接为唯一键去重；已存在的旧文章保留不动（避免摘要反复变动）,
 * 仅追加新链接；最后按日期倒序并截断到每源上限。
 *
 * @param oldItems 缓存中的历史文章
 * @param newItems 本次抓取到的文章
 */
function mergeItems(oldItems: FeedItem[], newItems: FeedItem[]): FeedItem[] {
  const byLink = new Map<string, FeedItem>();

  // 先放历史文章,再用新文章补充未见过的链接
  for (const item of oldItems) {
    if (item.link) byLink.set(item.link, item);
  }
  for (const item of newItems) {
    if (item.link && !byLink.has(item.link)) {
      byLink.set(item.link, item);
    }
  }

  // 按日期倒序（无日期排最后）,截断到上限
  return Array.from(byLink.values())
    .sort((a, b) => (b.date || "").localeCompare(a.date || ""))
    .slice(0, MAX_KEEP_PER_FEED);
}

// ==================== 主流程 ====================

async function main(): Promise<void> {
  // 1. 读订阅源配置,只抓 active
  const config = readJson<{ feeds: RssFeed[] }>(CONFIG_PATH, { feeds: [] });
  const feeds = (config.feeds || []).filter((feed) => feed.status === "active");

  if (feeds.length === 0) {
    console.log("没有 active 订阅源,跳过抓取");
    return;
  }

  // 2. 读旧缓存,建立 feedId -> 历史源 的映射
  const cache = readJson<RssCache>(CACHE_PATH, { generatedAt: "", feeds: [] });
  const oldByFeedId = new Map<number, CachedFeed>();
  for (const feed of cache.feeds || []) {
    oldByFeedId.set(feed.id, feed);
  }

  // 3. 并发抓取所有源（单源失败不抛出）
  console.log(`开始抓取 ${feeds.length} 个订阅源...`);
  const results = await fetchAllFeeds(feeds);

  // 4. 逐源合并新旧文章
  const mergedFeeds: CachedFeed[] = results.map((result) => {
    const old = oldByFeedId.get(result.feed.id);
    const oldItems = old?.items || [];

    // 抓取失败：保留历史文章,标记 ok=false
    if (!result.ok) {
      console.warn(
        `抓取失败,保留历史: ${result.feed.feedUrl}（历史 ${oldItems.length} 篇）`,
      );
      return {
        id: result.feed.id,
        title: old?.title || result.title,
        link: old?.link || result.link,
        ok: false,
        items: oldItems,
      };
    }

    const merged = mergeItems(oldItems, result.items);
    const added = Math.max(0, merged.length - oldItems.length);
    console.log(`${result.title}: 共 ${merged.length} 篇（新增 ${added}）`);

    return {
      id: result.feed.id,
      title: result.title,
      link: result.link,
      ok: true,
      items: merged,
    };
  });

  // 5. 写回缓存
  const output: RssCache = {
    generatedAt: new Date().toISOString(),
    feeds: mergedFeeds,
  };
  writeFileSync(CACHE_PATH, JSON.stringify(output, null, 2) + "\n", "utf-8");

  const total = mergedFeeds.reduce((sum, feed) => sum + feed.items.length, 0);
  console.log(`完成,缓存共 ${total} 篇文章 -> ${CACHE_PATH}`);
}

main().catch((error) => {
  console.error("抓取脚本异常:", error);
  process.exit(1);
});
