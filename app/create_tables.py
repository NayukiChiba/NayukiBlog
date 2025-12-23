import sqlite3
import os

# Database file path
DB_FILE = "blog.db"

def create_tables():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Books Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        cover TEXT,
        status TEXT,
        rating INTEGER,
        tags TEXT -- Stored as JSON string or comma-separated values
    );
    """)

    # 2. Diaries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS diaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        content TEXT,
        mood TEXT,
        weather TEXT,
        images TEXT -- Stored as JSON string
    );
    """)

    # 3. Gallery Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gallery (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        url TEXT NOT NULL,
        date TEXT,
        tags TEXT -- Stored as JSON string
    );
    """)

    # 4. Posts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        date TEXT,
        desc TEXT,
        url TEXT,
        tags TEXT, -- Stored as JSON string
        image TEXT,
        folder TEXT
    );
    """)

    # 5. Projects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        link TEXT,
        techStack TEXT, -- Stored as JSON string
        image TEXT,
        status TEXT
    );
    """)

    # 6. Todos Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        completed BOOLEAN DEFAULT 0,
        priority TEXT,
        type TEXT,
        progress INTEGER DEFAULT 0,
        icon TEXT
    );
    """)

    # 7. Tools Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tools (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        url TEXT,
        icon TEXT,
        category TEXT
    );
    """)

    conn.commit()
    conn.close()
    print(f"Database {DB_FILE} created and tables initialized successfully.")

if __name__ == "__main__":
    create_tables()
