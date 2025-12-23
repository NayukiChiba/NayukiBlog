from fastapi import FastAPI
from app.api import routes
from app.core.database import Base, engine

# Create tables if they don't exist (though we already created them manually)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="NayukiBlog API")

app.include_router(routes.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to NayukiBlog API"}
