from sqlalchemy import Column, Integer, String, Boolean, Text
from app.core.database import Base

class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    cover = Column(String)
    status = Column(String)
    rating = Column(Integer)
    tags = Column(Text) # JSON string

class Diary(Base):
    __tablename__ = "diaries"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, nullable=False)
    content = Column(Text)
    mood = Column(String)
    weather = Column(String)
    images = Column(Text) # JSON string

class Gallery(Base):
    __tablename__ = "gallery"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    url = Column(String, nullable=False)
    date = Column(String)
    tags = Column(Text) # JSON string

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    date = Column(String)
    desc = Column(Text)
    url = Column(String)
    tags = Column(Text) # JSON string
    image = Column(String)
    folder = Column(String)

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    link = Column(String)
    techStack = Column(Text) # JSON string
    image = Column(String)
    status = Column(String)

class Todo(Base):
    __tablename__ = "todos"
    id = Column(Integer, primary_key=True, index=True)
    task = Column(String, nullable=False)
    completed = Column(Boolean, default=False)
    priority = Column(String)
    type = Column(String)
    progress = Column(Integer, default=0)
    icon = Column(String)

class Tool(Base):
    __tablename__ = "tools"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    url = Column(String)
    icon = Column(String)
    category = Column(String)
