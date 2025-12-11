import sqlite3

conn = sqlite3.connect("../support_bot.db")
cursor = conn.cursor()

with open("schema.sql", "r", encoding="utf-8") as f:
    schema = f.read()

cursor.executescript(schema)
conn.commit()
conn.close()

print("Database created successfully!")
