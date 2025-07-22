import sqlite3

# Create a fake .coverage file for testing
conn = sqlite3.connect("tests/unit/fake.coverage")
cur = conn.cursor()

# Create tables
cur.execute(
    """
    CREATE TABLE coverage (
        id INTEGER PRIMARY KEY,
        data BLOB
    )
"""
)

cur.execute(
    """
    CREATE TABLE file (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE
    )
"""
)

cur.execute(
    """
    CREATE TABLE line_bits (
        file_id INTEGER,
        context_id INTEGER,
        line INTEGER,
        FOREIGN KEY (file_id) REFERENCES file(id)
    )
"""
)

# Insert data
cur.execute("INSERT INTO file (path) VALUES ('src/main.py')")
cur.execute("INSERT INTO line_bits (file_id, context_id, line) VALUES (1, 1, 1)")
cur.execute("INSERT INTO line_bits (file_id, context_id, line) VALUES (1, 1, 3)")

conn.commit()
conn.close()
