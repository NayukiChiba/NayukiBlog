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

def create_book(db: Session, title: str, cover: str, url: str, status: str, rating: int, tags: str):
    db_book = models.Book(
        title=title,
        cover=cover,
        url=url,
        status=status,
        rating=rating,
        tags=tags
    )
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book

def update_book(db: Session, book_id: int, title: str = None, cover: str = None, url: str = None, status: str = None, rating: int = None, tags: str = None):
    db_book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if db_book:
        if title: db_book.title = title
        if cover: db_book.cover = cover
        if url: db_book.url = url
        if status: db_book.status = status
        if rating is not None: db_book.rating = rating
        if tags: db_book.tags = tags
        db.commit()
        db.refresh(db_book)
        return db_book
    return None

def get_book(db: Session, book_id: int):
    return db.query(models.Book).filter(models.Book.id == book_id).first()

def delete_book(db: Session, book_id: int):
    db_book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if db_book:
        db.delete(db_book)
        db.commit()
        return True
    return False

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

def create_diary(db: Session, date: str, content: str, mood: str, weather: str, images: str):
    db_diary = models.Diary(
        date=date,
        content=content,
        mood=mood,
        weather=weather,
        images=images
    )
    db.add(db_diary)
    db.commit()
    db.refresh(db_diary)
    return db_diary

def update_diary(db: Session, diary_id: int, date: str = None, content: str = None, mood: str = None, weather: str = None, images: str = None):
    db_diary = db.query(models.Diary).filter(models.Diary.id == diary_id).first()
    if db_diary:
        if date: db_diary.date = date
        if content: db_diary.content = content
        if mood: db_diary.mood = mood
        if weather: db_diary.weather = weather
        if images: db_diary.images = images
        db.commit()
        db.refresh(db_diary)
        return db_diary
    return None

def get_diary(db: Session, diary_id: int):
    return db.query(models.Diary).filter(models.Diary.id == diary_id).first()

def delete_diary(db: Session, diary_id: int):
    db_diary = db.query(models.Diary).filter(models.Diary.id == diary_id).first()
    if db_diary:
        db.delete(db_diary)
        db.commit()
        return True
    return False

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

def create_gallery(db: Session, title: str, url: str, date: str, tags: str, status: str):
    db_gallery = models.Gallery(
        title=title,
        url=url,
        date=date,
        tags=tags,
        status=status
    )
    db.add(db_gallery)
    db.commit()
    db.refresh(db_gallery)
    return db_gallery

def update_gallery(db: Session, gallery_id: int, title: str = None, url: str = None, date: str = None, tags: str = None, status: str = None):
    db_gallery = db.query(models.Gallery).filter(models.Gallery.id == gallery_id).first()
    if db_gallery:
        if title: db_gallery.title = title
        if url: db_gallery.url = url
        if date: db_gallery.date = date
        if tags: db_gallery.tags = tags
        if status: db_gallery.status = status
        db.commit()
        db.refresh(db_gallery)
        return db_gallery
    return None

def get_gallery_item(db: Session, gallery_id: int):
    return db.query(models.Gallery).filter(models.Gallery.id == gallery_id).first()

def delete_gallery(db: Session, gallery_id: int):
    db_gallery = db.query(models.Gallery).filter(models.Gallery.id == gallery_id).first()
    if db_gallery:
        db.delete(db_gallery)
        db.commit()
        return True
    return False

def create_post(db: Session, title: str, date: str, folder: str, tags: str, status: str, desc: str, url: str, image: str = None):
    db_post = models.Post(
        title=title,
        date=date,
        folder=folder,
        tags=tags,
        status=status,
        desc=desc,
        url=url,
        image=image
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

def update_post(db: Session, post_id: int, title: str = None, date: str = None, folder: str = None, tags: str = None, status: str = None, desc: str = None, url: str = None, image: str = None):
    db_post = db.query(models.Post).filter(models.Post.id == post_id).first()
    if db_post:
        if title: db_post.title = title
        if date: db_post.date = date
        if folder: db_post.folder = folder
        if tags: db_post.tags = tags
        if status: db_post.status = status
        if desc: db_post.desc = desc
        if url: db_post.url = url
        if image: db_post.image = image
        db.commit()
        db.refresh(db_post)
        return db_post
    return None

def get_post(db: Session, post_id: int):
    return db.query(models.Post).filter(models.Post.id == post_id).first()

def delete_post(db: Session, post_id: int):
    db_post = db.query(models.Post).filter(models.Post.id == post_id).first()
    if db_post:
        db.delete(db_post)
        db.commit()
        return True
    return False

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

def create_project(db: Session, name: str, description: str, link: str, techStack: str, status: str, visibility: str):
    db_project = models.Project(
        name=name,
        description=description,
        link=link,
        techStack=techStack,
        status=status,
        visibility=visibility
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def update_project(db: Session, project_id: int, name: str = None, description: str = None, link: str = None, techStack: str = None, status: str = None, visibility: str = None):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if db_project:
        if name: db_project.name = name
        if description: db_project.description = description
        if link: db_project.link = link
        if techStack: db_project.techStack = techStack
        if status: db_project.status = status
        if visibility: db_project.visibility = visibility
        db.commit()
        db.refresh(db_project)
        return db_project
    return None

def get_project(db: Session, project_id: int):
    return db.query(models.Project).filter(models.Project.id == project_id).first()

def delete_project(db: Session, project_id: int):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if db_project:
        db.delete(db_project)
        db.commit()
        return True
    return False

def get_todos(db: Session, skip: int = 0, limit: int = 100, status: str = None, priority: str = None, type: str = None, completed: bool = None, sort: str = "desc"):
    query = db.query(models.Todo)
    if status:
        query = query.filter(models.Todo.status == status)
    if priority:
        query = query.filter(models.Todo.priority == priority)
    if type:
        query = query.filter(models.Todo.type == type)
    if completed is not None:
        query = query.filter(models.Todo.completed == completed)
        
    if sort == "asc":
        query = query.order_by(models.Todo.id.asc())
    else:
        query = query.order_by(models.Todo.id.desc())
        
    return query.offset(skip).limit(limit).all()

def get_tools(db: Session, skip: int = 0, limit: int = 100, status: str = None, category: str = None):
    query = db.query(models.Tool)
    if status:
        query = query.filter(models.Tool.status == status)
    if category:
        query = query.filter(models.Tool.category == category)
    return query.offset(skip).limit(limit).all()

def get_admin_by_username(db: Session, username: str):
    return db.query(models.Admin).filter(models.Admin.username == username).first()
