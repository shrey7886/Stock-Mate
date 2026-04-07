import requests
import json
import sqlite3

def run_test():
    # Attempt login as shreygolekar@gmail.com
    res = requests.post("http://127.0.0.1:8000/api/auth/login", json={
        "email": "shreygolekar@gmail.com",
        "password": "Foleo@123" 
    })
    
    if res.status_code != 200:
        print(f"Login failed: {res.status_code} - {res.text}")
        
        # Check DB to see if it even exists
        conn = sqlite3.connect('backend_api/database/backend.db')
        cursor = conn.cursor()
        cursor.execute("SELECT email, password_hash FROM users WHERE email='shreygolekar@gmail.com'")
        print("DB Record:", cursor.fetchone())
        conn.close()
        return
        
    token = res.json()["access_token"]
    print("Logged in successfully. Got Token.")
    
    # Hit /api/user/me
    res_me = requests.get("http://127.0.0.1:8000/api/user/me", headers={
        "Authorization": f"Bearer {token}"
    })
    print("User ME Status:", res_me.status_code)
    print("User ME Response:", res_me.text)

if __name__ == "__main__":
    run_test()
