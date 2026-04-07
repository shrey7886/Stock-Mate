import sys
import os
import sqlite3
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from backend_api.routes.auth import _hash_password
from backend_api.database.token_store import create_user
from uuid import uuid4

def fix_logo():
    print("Fixing logo...")
    img = Image.open('frontend/src/assets/logo.png').convert('RGB')
    pixels = img.load()
    width, height = img.size

    bg = pixels[0, 0]
    BgL = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]

    FgL = 0
    for y in range(height):
        for x in range(width):
            p = pixels[x, y]
            l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
            if l > FgL:
                FgL = l

    out = Image.new('RGBA', img.size)
    out_pixels = out.load()

    for y in range(height):
        for x in range(width):
            p = pixels[x, y]
            l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
            alpha = 0
            if FgL > BgL:
                ratio = (l - BgL) / (FgL - BgL)
                # boost contrast to drop background completely
                ratio = max(0.0, min(1.0, ratio * 1.8 - 0.15))
                alpha = int(ratio * 255)
            out_pixels[x, y] = (255, 255, 255, alpha)

    out.save('frontend/src/assets/logo_white.png')
    print("Saved frontend/src/assets/logo_white.png!")

def check_and_inject_user():
    print("Checking users in DB...")
    conn = sqlite3.connect('backend_api/database/app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT email, display_name FROM users")
    users = cursor.fetchall()
    print("Existing users:", users)
    conn.close()

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
        print(f"Successfully injected {email}!")
    else:
        print(f"User {email} already exists or injection failed!")

if __name__ == "__main__":
    try:
        fix_logo()
        check_and_inject_user()
    except Exception as e:
        print(f"Error: {e}")
