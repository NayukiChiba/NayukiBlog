from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.orm import Session
from typing import List
import json

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
    # TODO: Implement actual file saving and DB creation
    print(f"Received upload: {title}, {category}")
    return {"status": "success", "message": "Article uploaded successfully"}

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
