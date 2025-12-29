import os
from dotenv import load_dotenv
from app.core.database import SessionLocal, engine
from app.models import blog as models
from app.utils.security import hash_password

# Load environment variables
load_dotenv()

def init_db():
    print("Initializing database...")
    
    # 1. Create Tables
    print("Creating tables...")
    models.Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    # 2. Import/Update Admin User
    print("Importing admin user...")
    username = os.getenv("ADMIN_NAME")
    password = os.getenv("ADMIN_PASSWORD")

    if not username or not password:
        print("Error: ADMIN_NAME or ADMIN_PASSWORD not found in .env file.")
        return

    db = SessionLocal()
    try:
        # Hash the password before storing
        hashed_password = hash_password(password)
        
        # Check if admin already exists
        existing_admin = db.query(models.Admin).filter(models.Admin.username == username).first()
        if existing_admin:
            print(f"Admin user '{username}' already exists. Updating password.")
            existing_admin.password = hashed_password
            db.commit()
        else:
            new_admin = models.Admin(username=username, password=hashed_password)
            db.add(new_admin)
            db.commit()
            print(f"Admin user '{username}' created successfully.")
            
    except Exception as e:
        print(f"An error occurred during admin import: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
