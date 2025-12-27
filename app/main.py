from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import user, admin
from app.core.database import Base, engine

# Create tables if they don't exist (though we already created them manually)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="NayukiBlog API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router, prefix="/api/user", tags=["user"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

@app.get("/")
def read_root():
    return {"message": "Welcome to NayukiBlog API"}
