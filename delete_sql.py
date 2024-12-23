import os
import sqlite3

def delete_database():
    # Database filename
    db_name = 'documents_new.db'
    
    # Close any existing connections
    try:
        conn = sqlite3.connect(db_name)
        conn.close()
    except:
        pass
    
    # Delete the file if it exists
    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"Successfully deleted {db_name}")
    else:
        print(f"Database {db_name} does not exist")

if __name__ == "__main__":
    delete_database()