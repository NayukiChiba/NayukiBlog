from pydantic import BaseModel, field_validator
from typing import List, Optional
import json

# --- Shared Validator ---
def parse_json_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return v

# --- Books ---
class BookBase(BaseModel):
    title: str
    cover: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None
    rating: Optional[int] = None
    tags: Optional[List[str]] = []

    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        return parse_json_list(v)

class Book(BookBase):
    id: int
    class Config:
        from_attributes = True

# --- Diaries ---
class DiaryBase(BaseModel):
    date: str
    content: Optional[str] = None
    mood: Optional[str] = None
    weather: Optional[str] = None
    images: Optional[List[str]] = []

    @field_validator('images', mode='before')
    @classmethod
    def parse_images(cls, v):
        return parse_json_list(v)

class Diary(DiaryBase):
    id: int
    class Config:
        from_attributes = True

# --- Gallery ---
class GalleryBase(BaseModel):
    title: Optional[str] = None
    url: str
    date: Optional[str] = None
    tags: Optional[List[str]] = []

    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        return parse_json_list(v)

class Gallery(GalleryBase):
    id: int
    class Config:
        from_attributes = True

# --- Posts ---
class PostBase(BaseModel):
    title: str
    date: Optional[str] = None
    desc: Optional[str] = None
    url: Optional[str] = None
    tags: Optional[List[str]] = []
    image: Optional[str] = None
    folder: Optional[str] = None

    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        return parse_json_list(v)

class Post(PostBase):
    id: int
    class Config:
        from_attributes = True

# --- Projects ---
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    link: Optional[str] = None
    techStack: Optional[List[str]] = []
    image: Optional[str] = None
    status: Optional[str] = None

    @field_validator('techStack', mode='before')
    @classmethod
    def parse_techStack(cls, v):
        return parse_json_list(v)

class Project(ProjectBase):
    id: int
    class Config:
        from_attributes = True

# --- Todos ---
class TodoBase(BaseModel):
    task: str
    completed: bool = False
    priority: Optional[str] = None
    type: Optional[str] = None
    progress: int = 0
    icon: Optional[str] = None

class Todo(TodoBase):
    id: int
    class Config:
        from_attributes = True

# --- Tools ---
class ToolBase(BaseModel):
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None

class Tool(ToolBase):
    id: int
    class Config:
        from_attributes = True

# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str
