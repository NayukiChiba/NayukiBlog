from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List
import os
import json
from dotenv import load_dotenv

load_dotenv()

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

@router.get("/books", response_model=List[schemas.Book])
def read_books(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    books = crud.get_books(db, skip=skip, limit=limit)
    return books

@router.get("/diaries", response_model=List[schemas.Diary])
def read_diaries(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    diaries = crud.get_diaries(db, skip=skip, limit=limit)
    return diaries

@router.get("/gallery", response_model=List[schemas.Gallery])
def read_gallery(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    gallery = crud.get_gallery(db, skip=skip, limit=limit)
    return gallery

@router.get("/posts", response_model=List[schemas.Post])
def read_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=skip, limit=limit)
    return posts

@router.get("/projects", response_model=List[schemas.Project])
def read_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = crud.get_projects(db, skip=skip, limit=limit)
    return projects

@router.get("/todos", response_model=List[schemas.Todo])
def read_todos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    todos = crud.get_todos(db, skip=skip, limit=limit)
    return todos

@router.get("/tools", response_model=List[schemas.Tool])
def read_tools(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tools = crud.get_tools(db, skip=skip, limit=limit)
    return tools

@router.get("/admin/articles", response_model=List[schemas.Post])
def read_admin_articles(skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=skip, limit=limit)
    return posts

@router.get("/articles/categories")
def read_categories(db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=0, limit=10000)
    folders = set(p.folder for p in posts if p.folder)
    return {"categories": [{"path": f} for f in folders]}

@router.get("/articles/tags")
def read_tags(db: Session = Depends(get_db)):
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

@router.post("/admin/articles/upload")
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
    # TODO: Implement actual file saving and DB creation
    print(f"Received upload: {title}, {category}")
    return {"status": "success", "message": "Article uploaded successfully"}
