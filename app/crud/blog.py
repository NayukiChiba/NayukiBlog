from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models import blog as models

def get_books(db: Session, skip: int = 0, limit: int = 100, status: str = None, tags: list[str] = None):
    query = db.query(models.Book)
    if status:
        query = query.filter(models.Book.status == status)
    if tags:
        for tag in tags:
            query = query.filter(models.Book.tags.like(f'%"{tag}"%'))
    return query.offset(skip).limit(limit).all()

def get_diaries(db: Session, skip: int = 0, limit: int = 100, year: str = None, month: str = None):
    query = db.query(models.Diary)
    
    if year and month:
        query = query.filter(models.Diary.date.like(f"{year}-{month.zfill(2)}%"))
    elif year:
        query = query.filter(models.Diary.date.like(f"{year}%"))
    elif month:
        query = query.filter(models.Diary.date.like(f"%-{month.zfill(2)}-%"))
        
    return query.order_by(models.Diary.date.desc()).offset(skip).limit(limit).all()

def get_diaries_count(db: Session, year: str = None, month: str = None):
    query = db.query(models.Diary)
    
    if year and month:
        query = query.filter(models.Diary.date.like(f"{year}-{month.zfill(2)}%"))
    elif year:
        query = query.filter(models.Diary.date.like(f"{year}%"))
    elif month:
        query = query.filter(models.Diary.date.like(f"%-{month.zfill(2)}-%"))
        
    return query.count()

def get_gallery(db: Session, skip: int = 0, limit: int = 100, status: str = None, tags: list[str] = None, sort: str = "desc"):
    query = db.query(models.Gallery)
    if status:
        query = query.filter(models.Gallery.status == status)
    if tags:
        for tag in tags:
            query = query.filter(models.Gallery.tags.like(f'%"{tag}"%'))
            
    if sort == "asc":
        query = query.order_by(models.Gallery.date.asc())
    else:
        query = query.order_by(models.Gallery.date.desc())

    return query.offset(skip).limit(limit).all()

def get_posts(db: Session, skip: int = 0, limit: int = 100, status: str = None, folder: str = None, tags: list[str] = None, sort: str = "desc"):
    query = db.query(models.Post)
    if status:
        print(f"Filtering posts by status: {status} (Type: {type(status)})")
        query = query.filter(models.Post.status == status)
    if folder:
        # Filter by folder (exact match or subfolder)
        query = query.filter(or_(models.Post.folder == folder, models.Post.folder.like(f"{folder}/%")))
    if tags:
        for tag in tags:
            query = query.filter(models.Post.tags.like(f'%"{tag}"%'))
    
    if sort == "asc":
        query = query.order_by(models.Post.date.asc())
    else:
        query = query.order_by(models.Post.date.desc())

    return query.offset(skip).limit(limit).all()

def get_projects(db: Session, skip: int = 0, limit: int = 100, visibility: str = None, tech_stack: list[str] = None, project_status: str = None):
    query = db.query(models.Project)
    if visibility:
        query = query.filter(models.Project.visibility == visibility)
    if project_status:
        query = query.filter(models.Project.status == project_status)
    if tech_stack:
        for tech in tech_stack:
            query = query.filter(models.Project.techStack.like(f'%"{tech}"%'))
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
