from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import List

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
def read_admin_articles(skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)):
    posts = crud.get_posts(db, skip=skip, limit=limit)
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
