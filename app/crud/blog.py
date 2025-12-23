from sqlalchemy.orm import Session
from app.models import blog as models

def get_books(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Book).offset(skip).limit(limit).all()

def get_diaries(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Diary).offset(skip).limit(limit).all()

def get_gallery(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Gallery).offset(skip).limit(limit).all()

def get_posts(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Post).offset(skip).limit(limit).all()

def get_projects(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Project).offset(skip).limit(limit).all()

def get_todos(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Todo).offset(skip).limit(limit).all()

def get_tools(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Tool).offset(skip).limit(limit).all()
