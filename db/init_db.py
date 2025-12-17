from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "support_bot.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = f.read()

cursor.executescript(schema)
conn.commit()
conn.close()

print("Database created successfully!")
