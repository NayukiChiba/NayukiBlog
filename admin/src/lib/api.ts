/**
 * API 配置工具
 * 用于获取 API 基础地址，支持本地开发和服务器部署
 */

// 服务端默认 API 地址（当环境变量未配置时使用）
// 服务端渲染时直接访问本地后端，避免 SSL 证书问题
const DEFAULT_SERVER_API_BASE = 'http://127.0.0.1:8000';

/**
 * 获取 API 基础地址
 * - 服务端渲染时：使用环境变量 PUBLIC_API_BASE，如未配置则使用默认地址
 * - 客户端：使用 window.API_BASE（由 Layout 注入），如未配置则使用相对路径
 * 
 * @returns API 基础地址
 */
export function getApiBase(): string {
  // 服务端渲染时
  if (typeof window === 'undefined') {
    // 服务端必须使用完整 URL，不能使用相对路径
    return import.meta.env.PUBLIC_API_BASE || DEFAULT_SERVER_API_BASE;
  }
  // 客户端可以使用相对路径（浏览器会自动补全）
  return (window as any).API_BASE || '';
}

/**
 * 构建完整的 API URL
 * @param path API 路径，例如 '/api/admin/articles'
 * @returns 完整的 API URL
 */
export function apiUrl(path: string): string {
  const base = getApiBase();
  // 确保 path 以 / 开头
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalizedPath}`;
}
