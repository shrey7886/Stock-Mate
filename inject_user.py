import sys
import os

# Ensure backend_api is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from backend_api.routes.auth import _hash_password
from backend_api.database.token_store import create_user
from uuid import uuid4

def inject_foleo_user():
    email = "agentfoleo@gmail.com"
    password = "Foleo@123"
    
    hash_hex, salt = _hash_password(password)
    stored_hash = f"{salt}:{hash_hex}"

    success = create_user(
        user_id=str(uuid4()),
        email=email.lower().strip(),
        password_hash=stored_hash,
        display_name="Demo User Foleo"
    )
    
    if success:
        print(f"Successfully injected developer account: {email}")
    else:
        print(f"Account {email} already exists or failed to inject!")

if __name__ == "__main__":
    inject_foleo_user()
