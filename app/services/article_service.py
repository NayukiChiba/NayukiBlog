import os
from fastapi import UploadFile

async def save_article_file(
    file: UploadFile,
    title: str,
    date: str,
    tags: str,
    desc: str,
    base_path: str = "frontend/blog"
) -> str:
    """
    Saves an uploaded article file with frontmatter and normalized line endings.
    Returns the relative URL path for the article.
    """
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, file.filename)
    
    content = await file.read()
    try:
        content_str = content.decode("utf-8")
    except UnicodeDecodeError:
        content_str = content.decode("gbk", errors="ignore")

    # Normalize line endings to prevent double newlines on Windows
    content_str = content_str.replace("\r\n", "\n").replace("\r", "\n")

    if not content_str.strip().startswith("---"):
        # Create frontmatter
        tags_list = tags.split(",") if tags else []
        tags_str = ", ".join([f"'{t.strip()}'" for t in tags_list])
        use_desc = desc if desc else ""
        
        frontmatter = f"""---
layout: ../../../layouts/MarkdownLayout.astro
title: {title}
date: {date}
tags: [{tags_str}]
description: {use_desc}
---

"""
        content_str = frontmatter + content_str
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_str)
        
    filename_no_ext = os.path.splitext(file.filename)[0]
    return f"/user/posts/{filename_no_ext}"

def delete_article_file(url: str, base_path: str = "frontend/blog") -> bool:
    """
    Deletes the physical file associated with an article URL.
    """
    if url and "/user/posts/" in url:
        filename_no_ext = url.split("/user/posts/")[-1]
        possible_extensions = [".md", ".mdx"]
        for ext in possible_extensions:
            file_path = os.path.join(base_path, filename_no_ext + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                    return True
                except Exception as e:
                    print(f"Error deleting old file {file_path}: {e}")
    return False
