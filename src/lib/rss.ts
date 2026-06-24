/**
 * RSS 订阅源抓取与解析工具（由 fetchRss.ts 在抓取时调用）
 *
 * 功能：
 * 1. 并发抓取所有订阅源的 feed（带超时，单源失败不阻塞）
 * 2. 自动探测格式：JSON Feed / Atom / RSS 2.0
 * 3. 输出统一的文章列表结构供 fetchRss.ts 增量合并写入缓存
 */

import type { RssFeed } from "./data";

// 单篇文章
export interface FeedItem {
  title: string;
  link: string;
  date: string; // ISO 字符串，无效日期为空
  summary: string;
  content?: string; // 用于页面搜索的较长纯文本内容
}

// 单个订阅源的抓取结果
export interface FeedResult {
  feed: RssFeed; // 原订阅源配置
  title: string; // feed 自带标题
  link: string; // feed 站点链接
  items: FeedItem[]; // 最新文章
  ok: boolean; // 抓取/解析是否成功
}

// 单次抓取每个源最多解析的文章数
// （feed 通常只输出最新 10~20 篇;历史累积由 fetchRss.ts 合并去重实现,
//   见 src/data/rssCache.json）
const MAX_ITEMS_PER_FEED = 10;
// 抓取超时（毫秒）
const FETCH_TIMEOUT = 30000;
// 摘要最大长度
const SUMMARY_MAX_LENGTH = 120;
// 搜索内容最大长度
const CONTENT_MAX_LENGTH = 3000;

// ==================== 文本处理 ====================

// 剥离 CDATA 包裹
function stripCdata(text: string): string {
  return text.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1");
}

// 解码常见 HTML 实体
function decodeEntities(text: string): string {
  return text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&amp;/g, "&");
}

// 清洗为纯文本摘要：去 CDATA、去 HTML 标签、解实体、压缩空白、截断
function toSummary(raw: string): string {
  const text = decodeEntities(stripCdata(raw))
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length <= SUMMARY_MAX_LENGTH) return text;
  return text.slice(0, SUMMARY_MAX_LENGTH) + "…";
}

// 清洗为用于搜索的较长纯文本内容
function toContent(raw: string): string {
  const text = decodeEntities(stripCdata(raw))
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return text.slice(0, CONTENT_MAX_LENGTH);
}

// 清洗标题/链接等单行字段
function toText(raw: string): string {
  return decodeEntities(stripCdata(raw)).replace(/\s+/g, " ").trim();
}

// 日期统一转 ISO 字符串，无效返回空
function toIsoDate(raw: string): string {
  const date = new Date(toText(raw));
  return isNaN(date.getTime()) ? "" : date.toISOString();
}

// ==================== XML 辅助提取 ====================

// 提取某段 XML 中第一个标签的内部文本（不含属性匹配）
function extractTag(xml: string, tag: string): string {
  const match = xml.match(
    new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`, "i"),
  );
  return match ? match[1] : "";
}

// 提取所有重复块（如 <item>...</item>）
function extractBlocks(xml: string, tag: string): string[] {
  const matches = xml.match(
    new RegExp(`<${tag}(?:\\s[^>]*)?>[\\s\\S]*?</${tag}>`, "gi"),
  );
  return matches || [];
}

// 提取 Atom <link> 的 href（优先 rel="alternate"，其次第一个 link）
function extractAtomLink(xml: string): string {
  const links = xml.match(/<link\b[^>]*>/gi) || [];
  let fallback = "";
  for (const linkTag of links) {
    const hrefMatch = linkTag.match(/href=["']([^"']*)["']/i);
    if (!hrefMatch) continue;
    const relMatch = linkTag.match(/rel=["']([^"']*)["']/i);
    if (!relMatch || relMatch[1] === "alternate") {
      return toText(hrefMatch[1]);
    }
    if (!fallback) fallback = toText(hrefMatch[1]);
  }
  return fallback;
}

// ==================== 三种格式的解析 ====================

// JSON Feed（https://jsonfeed.org/）
function parseJsonFeed(data: any): { title: string; link: string; items: FeedItem[] } {
  const items: FeedItem[] = (data.items || [])
    .slice(0, MAX_ITEMS_PER_FEED)
    .map((item: any) => {
      const rawContent = String(
        item.summary || item.content_text || item.content_html || "",
      );
      return {
        title: toText(String(item.title || "无标题")),
        link: String(item.url || item.external_url || ""),
        date: toIsoDate(String(item.date_published || item.date_modified || "")),
        summary: toSummary(rawContent),
        content: toContent(rawContent),
      };
    });

  return {
    title: toText(String(data.title || "")),
    link: String(data.home_page_url || ""),
    items,
  };
}

// Atom（<feed><entry>...）
function parseAtom(xml: string): { title: string; link: string; items: FeedItem[] } {
  const entries = extractBlocks(xml, "entry");
  // feed 头部 = 去掉所有 entry 后的部分
  const head = xml.replace(/<entry(?:\s[^>]*)?>[\s\S]*?<\/entry>/gi, "");

  const items: FeedItem[] = entries.slice(0, MAX_ITEMS_PER_FEED).map((entry) => {
    const rawContent = extractTag(entry, "summary") || extractTag(entry, "content");
    return {
      title: toText(extractTag(entry, "title")) || "无标题",
      link: extractAtomLink(entry),
      date: toIsoDate(
        extractTag(entry, "published") || extractTag(entry, "updated"),
      ),
      summary: toSummary(rawContent),
      content: toContent(rawContent),
    };
  });

  return {
    title: toText(extractTag(head, "title")),
    link: extractAtomLink(head),
    items,
  };
}

// RSS 2.0（<rss><channel><item>...）
function parseRss(xml: string): { title: string; link: string; items: FeedItem[] } {
  const itemBlocks = extractBlocks(xml, "item");
  // channel 头部 = 去掉所有 item 后的部分
  const head = xml.replace(/<item(?:\s[^>]*)?>[\s\S]*?<\/item>/gi, "");

  const items: FeedItem[] = itemBlocks.slice(0, MAX_ITEMS_PER_FEED).map((item) => {
    const rawContent =
      extractTag(item, "content:encoded") || extractTag(item, "description");
    return {
      title: toText(extractTag(item, "title")) || "无标题",
      link: toText(extractTag(item, "link")),
      date: toIsoDate(extractTag(item, "pubDate") || extractTag(item, "dc:date")),
      summary: toSummary(rawContent),
      content: toContent(rawContent),
    };
  });

  return {
    title: toText(extractTag(head, "title")),
    link: toText(extractTag(head, "link")),
    items,
  };
}

// ==================== 抓取入口 ====================

// 抓取并解析单个订阅源（格式自动探测：JSON → Atom → RSS 2.0）
async function fetchFeed(feed: RssFeed): Promise<FeedResult> {
  const failed: FeedResult = {
    feed,
    title: feed.name || feed.site,
    link: feed.site,
    items: [],
    ok: false,
  };

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
    const response = await fetch(feed.feedUrl, {
      signal: controller.signal,
      headers: { "User-Agent": "NayukiBlog RSS Reader" },
    });
    clearTimeout(timer);

    if (!response.ok) {
      console.warn(`RSS 抓取失败 (HTTP ${response.status}): ${feed.feedUrl}`);
      return failed;
    }

    const body = await response.text();

    // 1. 尝试 JSON Feed
    let parsed: { title: string; link: string; items: FeedItem[] } | null = null;
    try {
      const json = JSON.parse(body);
      if (json && Array.isArray(json.items)) {
        parsed = parseJsonFeed(json);
      }
    } catch {
      // 非 JSON，继续按 XML 解析
    }

    // 2. 按 XML 解析（Atom 优先于 RSS 2.0）
    if (!parsed) {
      if (/<feed[\s>]/i.test(body)) {
        parsed = parseAtom(body);
      } else if (/<rss[\s>]|<channel[\s>]/i.test(body)) {
        parsed = parseRss(body);
      }
    }

    if (!parsed) {
      console.warn(`RSS 格式无法识别: ${feed.feedUrl}`);
      return failed;
    }

    return {
      feed,
      // 自定义名称优先，其次 feed 自带标题，最后回退站点地址
      title: feed.name || parsed.title || feed.site,
      link: parsed.link || feed.site,
      items: parsed.items.filter((item) => item.link),
      ok: true,
    };
  } catch (error) {
    console.warn(`RSS 抓取异常: ${feed.feedUrl}`, error);
    return failed;
  }
}

// 并发抓取所有订阅源（任何单源失败都不会抛出）
export async function fetchAllFeeds(feeds: RssFeed[]): Promise<FeedResult[]> {
  const results = await Promise.allSettled(feeds.map((feed) => fetchFeed(feed)));
  return results.map((result, index) =>
    result.status === "fulfilled"
      ? result.value
      : {
          feed: feeds[index],
          title: feeds[index].name || feeds[index].site,
          link: feeds[index].site,
          items: [],
          ok: false,
        },
  );
}
