from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.orm import Session
from typing import List
import json
import os
import shutil

from app.crud import blog as crud
from app.schemas import blog as schemas
from app.core.database import get_db

router = APIRouter()

@router.post("/login")
def login(login_data: schemas.LoginRequest, db: Session = Depends(get_db)):
    admin = crud.get_admin_by_username(db, username=login_data.username)

    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if login_data.password == admin.password:
        return {"message": "Login successful", "status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/articles", response_model=List[schemas.Post])
def read_admin_articles(
    skip: int = 0, 
    limit: int = 10, 
    category: str = None, 
    tags: List[str] = Query(None),
    sort: str = "desc",
    status: str = None,
    db: Session = Depends(get_db)
):
    posts = crud.get_posts(db, skip=skip, limit=limit, folder=category, tags=tags, sort=sort, status=status)
    return posts

@router.post("/articles/upload")
async def upload_article(
    title: str = Form(...),
    date: str = Form(...),
    category: str = Form(None),
    tags: str = Form(None),
    status: str = Form("draft"),
    desc: str = Form(None),
    image: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save file
    base_path = "frontend/src/pages/user/posts"
    # User requested flat structure for physical files, logical folder in DB only
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB entry
    # Construct URL: /user/posts/{filename_without_ext}
    filename_no_ext = os.path.splitext(file.filename)[0]
    url_path = f"/user/posts/{filename_no_ext}"

    # Normalize status
    if status == "published":
        status = "public"

    crud.create_post(
        db=db,
        title=title,
        date=date,
        folder=category,
        tags=tags,
        status=status,
        desc=desc,
        url=url_path,
        image=image
    )
    
    return {"status": "success", "message": "Article uploaded successfully"}

@router.put("/articles/{post_id}")
async def update_article(
    post_id: int,
    title: str = Form(None),
    date: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None),
    status: str = Form(None),
    desc: str = Form(None),
    image: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    post = crud.get_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Article not found")

    url_path = None
    if file and file.filename:
        # Save new file
        base_path = "frontend/src/pages/user/posts"
        os.makedirs(base_path, exist_ok=True)
        
        file_path = os.path.join(base_path, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        filename_no_ext = os.path.splitext(file.filename)[0]
        url_path = f"/user/posts/{filename_no_ext}"
        
        # Delete old file if filename is different
        if post.url and "/user/posts/" in post.url:
            old_filename_no_ext = post.url.split("/user/posts/")[-1]
            if old_filename_no_ext != filename_no_ext:
                possible_extensions = [".md", ".mdx"]
                for ext in possible_extensions:
                    old_file_path = os.path.join(base_path, old_filename_no_ext + ext)
                    if os.path.exists(old_file_path):
                        try:
                            os.remove(old_file_path)
                            print(f"Deleted old file: {old_file_path}")
                        except Exception as e:
                            print(f"Error deleting old file {old_file_path}: {e}")
                        break

    # Normalize status
    if status == "published":
        status = "public"

    updated_post = crud.update_post(
        db=db,
        post_id=post_id,
        title=title,
        date=date,
        folder=category,
        tags=tags,
        status=status,
        desc=desc,
        url=url_path,
        image=image
    )
    
    if updated_post:
        return {"status": "success", "message": "Article updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update article")

@router.delete("/articles/{post_id}")
def delete_article(post_id: int, db: Session = Depends(get_db)):
    post = crud.get_post(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Delete file
    if post.url and "/user/posts/" in post.url:
        filename_no_ext = post.url.split("/user/posts/")[-1]
        base_path = "frontend/src/pages/user/posts"
        
        # Try to find and delete the file (checking common extensions)
        possible_extensions = [".md", ".mdx"]
        for ext in possible_extensions:
            file_path = os.path.join(base_path, filename_no_ext + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
                break

    if crud.delete_post(db, post_id):
        return {"status": "success", "message": "Article deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete article from database")

@router.get("/articles/tags")
def read_admin_article_tags(db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=0, limit=10000)
    all_tags = set()
    for p in posts:
        if p.tags:
            try:
                tags_list = json.loads(p.tags)
                if isinstance(tags_list, list):
                    for tag in tags_list:
                        all_tags.add(tag)
            except:
                pass
    return {"tags": list(sorted(all_tags))}

@router.get("/books", response_model=List[schemas.Book])
def read_admin_books(
    skip: int = 0, 
    limit: int = 100, 
    tags: List[str] = Query(None),
    status: str = None,
    db: Session = Depends(get_db)
):
    books = crud.get_books(db, skip=skip, limit=limit, status=status, tags=tags)
    return books

@router.post("/books/upload")
async def upload_book(
    title: str = Form(...),
    cover: str = Form(None),
    url: str = Form(None),
    status: str = Form("published"),
    rating: int = Form(None),
    tags: str = Form(None), # JSON string
    db: Session = Depends(get_db)
):
    crud.create_book(
        db=db,
        title=title,
        cover=cover,
        url=url,
        status=status,
        rating=rating,
        tags=tags
    )
    return {"status": "success", "message": "Book created successfully"}

@router.put("/books/{book_id}")
async def update_book(
    book_id: int,
    title: str = Form(None),
    cover: str = Form(None),
    url: str = Form(None),
    status: str = Form(None),
    rating: int = Form(None),
    tags: str = Form(None),
    db: Session = Depends(get_db)
):
    book = crud.get_book(db, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    crud.update_book(
        db=db,
        book_id=book_id,
        title=title,
        cover=cover,
        url=url,
        status=status,
        rating=rating,
        tags=tags
    )
    return {"status": "success", "message": "Book updated successfully"}

@router.delete("/books/{book_id}")
def delete_book(book_id: int, db: Session = Depends(get_db)):
    if crud.delete_book(db, book_id):
        return {"status": "success", "message": "Book deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Book not found")

@router.get("/books/tags")
def read_admin_book_tags(db: Session = Depends(get_db)):
    books = crud.get_books(db, skip=0, limit=10000)
    all_tags = set()
    for b in books:
        if b.tags:
            try:
                tags_list = json.loads(b.tags)
                if isinstance(tags_list, list):
                    for tag in tags_list:
                        all_tags.add(tag)
            except:
                pass
    return {"tags": list(sorted(all_tags))}

@router.get("/projects", response_model=List[schemas.Project])
def read_admin_projects(
    skip: int = 0, 
    limit: int = 10, 
    tech_stack: List[str] = Query(None),
    status: str = None,
    visibility: str = None,
    db: Session = Depends(get_db)
):
    projects = crud.get_projects(db, skip=skip, limit=limit, visibility=visibility, tech_stack=tech_stack, project_status=status)
    return projects

@router.get("/projects/tech-stacks")
def read_admin_tech_stacks(db: Session = Depends(get_db)):
    projects = crud.get_projects(db, skip=0, limit=10000)
    all_stacks = set()
    for p in projects:
        if p.techStack:
            try:
                stack_list = json.loads(p.techStack)
                if isinstance(stack_list, list):
                    all_stacks.update(stack_list)
            except:
                pass
    return list(all_stacks)

@router.post("/projects/upload")
async def upload_project(
    name: str = Form(...),
    description: str = Form(None),
    link: str = Form(None),
    techStack: str = Form(None),
    status: str = Form(None),
    visibility: str = Form("published"),
    db: Session = Depends(get_db)
):
    crud.create_project(
        db=db,
        name=name,
        description=description,
        link=link,
        techStack=techStack,
        status=status,
        visibility=visibility
    )
    return {"status": "success", "message": "Project created successfully"}

@router.put("/projects/{project_id}")
async def update_project(
    project_id: int,
    name: str = Form(None),
    description: str = Form(None),
    link: str = Form(None),
    techStack: str = Form(None),
    status: str = Form(None),
    visibility: str = Form(None),
    db: Session = Depends(get_db)
):
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    crud.update_project(
        db=db,
        project_id=project_id,
        name=name,
        description=description,
        link=link,
        techStack=techStack,
        status=status,
        visibility=visibility
    )
    return {"status": "success", "message": "Project updated successfully"}

@router.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    if crud.delete_project(db, project_id):
        return {"status": "success", "message": "Project deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Project not found")

@router.get("/diaries", response_model=schemas.DiaryPagination)
def read_admin_diaries(
    skip: int = 0, 
    limit: int = 100, 
    year: str = Query(None),
    month: str = Query(None),
    db: Session = Depends(get_db)
):
    diaries = crud.get_diaries(db, skip=skip, limit=limit, year=year, month=month)
    total = crud.get_diaries_count(db, year=year, month=month)
    return {"total": total, "items": diaries}

@router.post("/diaries/upload")
async def upload_diary(
    date: str = Form(...),
    content: str = Form(None),
    mood: str = Form(None),
    weather: str = Form(None),
    images: str = Form(None), # JSON string
    db: Session = Depends(get_db)
):
    crud.create_diary(
        db=db,
        date=date,
        content=content,
        mood=mood,
        weather=weather,
        images=images
    )
    return {"status": "success", "message": "Diary created successfully"}

@router.put("/diaries/{diary_id}")
async def update_diary(
    diary_id: int,
    date: str = Form(None),
    content: str = Form(None),
    mood: str = Form(None),
    weather: str = Form(None),
    images: str = Form(None),
    db: Session = Depends(get_db)
):
    diary = crud.get_diary(db, diary_id)
    if not diary:
        raise HTTPException(status_code=404, detail="Diary not found")

    crud.update_diary(
        db=db,
        diary_id=diary_id,
        date=date,
        content=content,
        mood=mood,
        weather=weather,
        images=images
    )
    return {"status": "success", "message": "Diary updated successfully"}

@router.delete("/diaries/{diary_id}")
def delete_diary(diary_id: int, db: Session = Depends(get_db)):
    if crud.delete_diary(db, diary_id):
        return {"status": "success", "message": "Diary deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Diary not found")

@router.get("/gallery/tags")
def read_admin_gallery_tags(db: Session = Depends(get_db)):
    gallery = crud.get_gallery(db, skip=0, limit=10000)
    all_tags = set()
    for item in gallery:
        if item.tags:
            try:
                tags_list = json.loads(item.tags)
                if isinstance(tags_list, list):
                    for tag in tags_list:
                        all_tags.add(tag)
            except:
                pass
    return {"tags": list(sorted(all_tags))}

@router.get("/gallery", response_model=List[schemas.Gallery])
def read_admin_gallery(
    skip: int = 0, 
    limit: int = 100, 
    tags: List[str] = Query(None),
    sort: str = "desc",
    status: str = None,
    db: Session = Depends(get_db)
):
    gallery = crud.get_gallery(db, skip=skip, limit=limit, status=status, tags=tags, sort=sort)
    return gallery

@router.post("/gallery/upload")
async def upload_gallery(
    title: str = Form(None),
    url: str = Form(...),
    date: str = Form(None),
    tags: str = Form(None), # JSON string
    status: str = Form("published"),
    db: Session = Depends(get_db)
):
    crud.create_gallery(
        db=db,
        title=title,
        url=url,
        date=date,
        tags=tags,
        status=status
    )
    return {"status": "success", "message": "Image created successfully"}

@router.put("/gallery/{gallery_id}")
async def update_gallery(
    gallery_id: int,
    title: str = Form(None),
    url: str = Form(None),
    date: str = Form(None),
    tags: str = Form(None),
    status: str = Form(None),
    db: Session = Depends(get_db)
):
    gallery = crud.get_gallery_item(db, gallery_id)
    if not gallery:
        raise HTTPException(status_code=404, detail="Image not found")

    crud.update_gallery(
        db=db,
        gallery_id=gallery_id,
        title=title,
        url=url,
        date=date,
        tags=tags,
        status=status
    )
    return {"status": "success", "message": "Image updated successfully"}

@router.delete("/gallery/{gallery_id}")
def delete_gallery(gallery_id: int, db: Session = Depends(get_db)):
    if crud.delete_gallery(db, gallery_id):
        return {"status": "success", "message": "Image deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@router.get("/todos", response_model=List[schemas.Todo])
def read_admin_todos(
    skip: int = 0, 
    limit: int = 100, 
    priority: str = None,
    type: str = None,
    status: str = None,
    completed: bool = None,
    sort: str = "desc",
    db: Session = Depends(get_db)
):
    if priority:
        priority = priority.lower()
    todos = crud.get_todos(db, skip=skip, limit=limit, status=status, priority=priority, type=type, completed=completed, sort=sort)
    return todos

@router.post("/todos/upload")
async def upload_todo(
    task: str = Form(...),
    priority: str = Form("medium"),
    type: str = Form("short-term"),
    progress: int = Form(0),
    icon: str = Form(None),
    status: str = Form("published"),
    completed: bool = Form(False),
    db: Session = Depends(get_db)
):
    crud.create_todo(
        db=db,
        task=task,
        priority=priority,
        type=type,
        progress=progress,
        icon=icon,
        status=status,
        completed=completed
    )
    return {"status": "success", "message": "Task created successfully"}

@router.put("/todos/{todo_id}")
async def update_todo(
    todo_id: int,
    task: str = Form(None),
    priority: str = Form(None),
    type: str = Form(None),
    progress: int = Form(None),
    icon: str = Form(None),
    status: str = Form(None),
    completed: bool = Form(None),
    db: Session = Depends(get_db)
):
    todo = crud.get_todo(db, todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Task not found")

    crud.update_todo(
        db=db,
        todo_id=todo_id,
        task=task,
        priority=priority,
        type=type,
        progress=progress,
        icon=icon,
        status=status,
        completed=completed
    )
    return {"status": "success", "message": "Task updated successfully"}

@router.delete("/todos/{todo_id}")
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    if crud.delete_todo(db, todo_id):
        return {"status": "success", "message": "Task deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@router.get("/todos/types")
def read_admin_todo_types(db: Session = Depends(get_db)):
    todos = crud.get_todos(db, skip=0, limit=10000)
    all_types = set(t.type for t in todos if t.type)
    return {"types": list(sorted(all_types))}

@router.get("/tools", response_model=List[schemas.Tool])
def read_admin_tools(
    skip: int = 0, 
    limit: int = 100, 
    category: str = None,
    status: str = None,
    db: Session = Depends(get_db)
):
    tools = crud.get_tools(db, skip=skip, limit=limit, status=status, category=category)
    return tools

@router.get("/tools/categories")
def read_admin_tool_categories(db: Session = Depends(get_db)):
    tools = crud.get_tools(db, skip=0, limit=10000)
    all_categories = set(t.category for t in tools if t.category)
    return {"categories": list(sorted(all_categories))}

@router.post("/tools/upload")
async def upload_tool(
    name: str = Form(...),
    url: str = Form(...),
    description: str = Form(None),
    icon: str = Form(None),
    category: str = Form(None),
    status: str = Form("published"),
    db: Session = Depends(get_db)
):
    crud.create_tool(
        db=db,
        name=name,
        url=url,
        description=description,
        icon=icon,
        category=category,
        status=status
    )
    return {"status": "success", "message": "Tool created successfully"}

@router.put("/tools/{tool_id}")
async def update_tool(
    tool_id: int,
    name: str = Form(None),
    url: str = Form(None),
    description: str = Form(None),
    icon: str = Form(None),
    category: str = Form(None),
    status: str = Form(None),
    db: Session = Depends(get_db)
):
    tool = crud.get_tool(db, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    crud.update_tool(
        db=db,
        tool_id=tool_id,
        name=name,
        url=url,
        description=description,
        icon=icon,
        category=category,
        status=status
    )
    return {"status": "success", "message": "Tool updated successfully"}

@router.delete("/tools/{tool_id}")
def delete_tool(tool_id: int, db: Session = Depends(get_db)):
    if crud.delete_tool(db, tool_id):
        return {"status": "success", "message": "Tool deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Tool not found")
