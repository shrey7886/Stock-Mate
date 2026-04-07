import sys
import os
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from backend_api.routes.auth import _hash_password
from backend_api.database.token_store import create_user
from uuid import uuid4

def reset_and_check():
    conn = sqlite3.connect('backend_api/database/backend.db')
    cursor = conn.cursor()
    
    email = "shreygolekar@gmail.com"
    cursor.execute("SELECT user_id, display_name FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    
    # We will force-inject or reset the password to 'Foleo@123' to guarantee entry
    hash_hex, salt = _hash_password("Foleo@123")
    stored_hash = f"{salt}:{hash_hex}"
    
    if user:
        cursor.execute("UPDATE users SET password_hash=? WHERE email=?", (stored_hash, email))
        conn.commit()
        print(f"[SUCCESS] Found existing account '{email}'. Forced password to 'Foleo@123'.")
    else:
        create_user(
            user_id=str(uuid4()),
            email=email,
            password_hash=stored_hash,
            display_name="Shrey Golekar"
        )
        print(f"[SUCCESS] Account '{email}' did not exist! It has now been securely created with password 'Foleo@123'.")
        
    conn.close()

if __name__ == "__main__":
    reset_and_check()
