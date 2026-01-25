/**
 * 数据读取工具
 * 从本地 JSON 文件读取数据
 */

import projectsData from '../data/projects.json';
import booksData from '../data/books.json';
import diariesData from '../data/diaries.json';
import galleryData from '../data/gallery.json';
import todosData from '../data/todos.json';
import toolsData from '../data/tools.json';

// 类型定义
export interface Project {
  id: number;
  name: string;
  description?: string;
  link?: string;
  techStack?: string[];
  status?: string;
  visibility?: string;
}

export interface Book {
  id: number;
  title: string;
  cover?: string;
  url?: string;
  rating?: number;
  tags?: string[];
  status?: string;
}

export interface Diary {
  id: number;
  date: string;
  content?: string;
  mood?: string;
  weather?: string;
  images?: string[];
}

export interface GalleryItem {
  id: number;
  title?: string;
  url: string;
  date?: string;
  tags?: string[];
  status?: string;
}

export interface Todo {
  id: number;
  task: string;
  completed: boolean;
  priority?: string;
  type?: string;
  progress: number;
  icon?: string;
  status?: string;
}

export interface Tool {
  id: number;
  name: string;
  description?: string;
  url?: string;
  icon?: string;
  category?: string;
  status?: string;
}

// 获取项目列表
export function getProjects(options?: {
  visibility?: string;
  status?: string;
  limit?: number;
}): Project[] {
  let projects = projectsData.projects || [];
  
  // 过滤
  if (options?.visibility) {
    projects = projects.filter(p => p.visibility === options.visibility);
  }
  if (options?.status) {
    projects = projects.filter(p => p.status === options.status);
  }
  
  // 限制数量
  if (options?.limit) {
    projects = projects.slice(0, options.limit);
  }
  
  return projects;
}

// 获取书籍列表
export function getBooks(options?: {
  status?: string;
  tags?: string[];
  limit?: number;
}): Book[] {
  let books = booksData.books || [];
  
  // 过滤
  if (options?.status) {
    books = books.filter(b => b.status === options.status);
  }
  if (options?.tags && options.tags.length > 0) {
    books = books.filter(b => 
      b.tags && options.tags!.some(tag => b.tags!.includes(tag))
    );
  }
  
  // 限制数量
  if (options?.limit) {
    books = books.slice(0, options.limit);
  }
  
  return books;
}

// 获取日记列表
export function getDiaries(options?: {
  year?: string;
  month?: string;
  limit?: number;
}): Diary[] {
  let diaries = diariesData.diaries || [];
  
  // 按日期排序（降序）
  diaries = diaries.sort((a, b) => b.date.localeCompare(a.date));
  
  // 过滤
  if (options?.year && options?.month) {
    const prefix = `${options.year}-${options.month.padStart(2, '0')}`;
    diaries = diaries.filter(d => d.date.startsWith(prefix));
  } else if (options?.year) {
    diaries = diaries.filter(d => d.date.startsWith(options.year!));
  }
  
  // 限制数量
  if (options?.limit) {
    diaries = diaries.slice(0, options.limit);
  }
  
  return diaries;
}

// 获取图片列表
export function getGallery(options?: {
  status?: string;
  tags?: string[];
  sort?: 'asc' | 'desc';
  limit?: number;
}): GalleryItem[] {
  let gallery = galleryData.gallery || [];
  
  // 过滤
  if (options?.status) {
    gallery = gallery.filter(g => g.status === options.status);
  }
  if (options?.tags && options.tags.length > 0) {
    gallery = gallery.filter(g => 
      g.tags && options.tags!.some(tag => g.tags!.includes(tag))
    );
  }
  
  // 排序
  if (options?.sort === 'asc') {
    gallery = gallery.sort((a, b) => (a.date || '').localeCompare(b.date || ''));
  } else {
    gallery = gallery.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  }
  
  // 限制数量
  if (options?.limit) {
    gallery = gallery.slice(0, options.limit);
  }
  
  return gallery;
}

// 获取待办事项列表
export function getTodos(options?: {
  status?: string;
  priority?: string;
  type?: string;
  completed?: boolean;
  limit?: number;
}): Todo[] {
  let todos = todosData.todos || [];
  
  // 过滤
  if (options?.status) {
    todos = todos.filter(t => t.status === options.status);
  }
  if (options?.priority) {
    todos = todos.filter(t => t.priority === options.priority);
  }
  if (options?.type) {
    todos = todos.filter(t => t.type === options.type);
  }
  if (options?.completed !== undefined) {
    todos = todos.filter(t => t.completed === options.completed);
  }
  
  // 限制数量
  if (options?.limit) {
    todos = todos.slice(0, options.limit);
  }
  
  return todos;
}

// 获取工具列表
export function getTools(options?: {
  status?: string;
  category?: string;
  limit?: number;
}): Tool[] {
  let tools = toolsData.tools || [];
  
  // 过滤
  if (options?.status) {
    tools = tools.filter(t => t.status === options.status);
  }
  if (options?.category) {
    tools = tools.filter(t => t.category === options.category);
  }
  
  // 限制数量
  if (options?.limit) {
    tools = tools.slice(0, options.limit);
  }
  
  return tools;
}

// 获取所有分类
export function getCategories(tools: Tool[]): string[] {
  const categories = new Set<string>();
  tools.forEach(tool => {
    if (tool.category) {
      categories.add(tool.category);
    }
  });
  return Array.from(categories).sort();
}

// 获取所有标签
export function getTags(items: (Book | GalleryItem)[]): string[] {
  const tags = new Set<string>();
  items.forEach(item => {
    if (item.tags) {
      item.tags.forEach(tag => tags.add(tag));
    }
  });
  return Array.from(tags).sort();
}
