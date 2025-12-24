import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine
from app.models import blog as models

# Load environment variables
load_dotenv()

def import_admin():
    username = os.getenv("ADMIN_NAME")
    password = os.getenv("ADMIN_PASSWORD")

    if not username or not password:
        print("Error: ADMIN_NAME or ADMIN_PASSWORD not found in .env file.")
        return

    db = SessionLocal()
    try:
        # Check if admin already exists
        existing_admin = db.query(models.Admin).filter(models.Admin.username == username).first()
        if existing_admin:
            print(f"Admin user '{username}' already exists. Updating password.")
            existing_admin.password = password
            db.commit()
        else:
            new_admin = models.Admin(username=username, password=password)
            db.add(new_admin)
            db.commit()
            print(f"Admin user '{username}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Importing admin user...")
    # Ensure tables exist (in case they weren't created yet)
    models.Base.metadata.create_all(bind=engine)
    import_admin()
