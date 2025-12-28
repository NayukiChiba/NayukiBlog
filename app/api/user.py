from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json

from app.crud import blog as crud
from app.schemas import blog as schemas
from app.core.database import get_db

router = APIRouter()

@router.get("/books", response_model=List[schemas.Book])
def read_books(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    books = crud.get_books(db, skip=skip, limit=limit, status="published")
    return books

@router.get("/diaries", response_model=List[schemas.Diary])
def read_diaries(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    diaries = crud.get_diaries(db, skip=skip, limit=limit)
    return diaries

@router.get("/gallery", response_model=List[schemas.Gallery])
def read_gallery(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    gallery = crud.get_gallery(db, skip=skip, limit=limit, status="published")
    return gallery

@router.get("/posts", response_model=List[schemas.Post])
def read_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=skip, limit=limit, status="public")
    return posts

@router.get("/projects", response_model=List[schemas.Project])
def read_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = crud.get_projects(db, skip=skip, limit=limit, visibility="published")
    return projects

@router.get("/todos", response_model=List[schemas.Todo])
def read_todos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    todos = crud.get_todos(db, skip=skip, limit=limit, status="published")
    return todos

@router.get("/tools", response_model=List[schemas.Tool])
def read_tools(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tools = crud.get_tools(db, skip=skip, limit=limit, status="published")
    return tools

@router.get("/projects/tech-stacks")
def read_tech_stacks(db: Session = Depends(get_db)):
    projects = crud.get_projects(db, skip=0, limit=10000, visibility="published")
    all_stacks = set()
    for p in projects:
        if p.techStack:
            try:
                stack_list = json.loads(p.techStack)
                all_stacks.update(stack_list)
            except:
                pass
    return list(all_stacks)

@router.get("/articles/categories")
def read_categories(db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=0, limit=10000, status="public")
    folders = set(p.folder for p in posts if p.folder)
    return {"categories": [{"path": f} for f in folders]}

@router.get("/articles/tags")
def read_tags(db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=0, limit=10000, status="public")
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

@router.get("/books/tags")
def read_book_tags(db: Session = Depends(get_db)):
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

