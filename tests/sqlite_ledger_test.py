import sqlite3

DB_PATH = "data/psh_ledger.db"

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    
    cursor.execute("""
                   SELECT * FROM ingested_log ORDER BY filename,course
                    """)
    for row in cursor.fetchall():
        print(row)
conn.close()