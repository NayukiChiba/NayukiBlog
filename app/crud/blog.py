from sqlalchemy.orm import Session
from app.models import blog as models

def get_books(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Book)
    if status:
        query = query.filter(models.Book.visibility == status)
    return query.offset(skip).limit(limit).all()

def get_diaries(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Diary).offset(skip).limit(limit).all()

def get_gallery(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Gallery)
    if status:
        query = query.filter(models.Gallery.status == status)
    return query.offset(skip).limit(limit).all()

def get_posts(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Post)
    if status:
        print(f"Filtering posts by status: {status} (Type: {type(status)})")
        query = query.filter(models.Post.status == status)
    return query.offset(skip).limit(limit).all()

def get_projects(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Project)
    if status:
        query = query.filter(models.Project.visibility == status)
    return query.offset(skip).limit(limit).all()

def get_todos(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Todo)
    if status:
        query = query.filter(models.Todo.status == status)
    return query.offset(skip).limit(limit).all()

def get_tools(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Tool)
    if status:
        query = query.filter(models.Tool.status == status)
    return query.offset(skip).limit(limit).all()

def get_admin_by_username(db: Session, username: str):
    return db.query(models.Admin).filter(models.Admin.username == username).first()
