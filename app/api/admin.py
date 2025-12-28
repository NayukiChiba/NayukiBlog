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
    summary: str = Form(None),
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
        desc=summary,
        url=url_path,
        image="/placeholder.jpg"
    )
    
    return {"status": "success", "message": "Article uploaded successfully"}

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
