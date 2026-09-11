import os
import json
import sqlite3
import bcrypt
from datetime import datetime

# Absolute path guarantees both registration and signin touch the exact same file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "surakshasetu.db")

def init_db():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            case_id TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS triage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            mode TEXT,
            distress_score REAL,
            triage_level TEXT,
            statement TEXT,
            features JSON
        )
    ''')
    conn.commit()
    conn.close()

def register_user(username, name, password, case_id):
    u = username.strip()
    n = name.strip()
    p = password.strip()
    c_id = case_id.strip()

    print(f"\n[REGISTER ATTEMPT] Username: '{u}'")

    if not u or not p or not n:
        print("[REGISTER FAILED] Missing required fields")
        return False

    # 1. Generate salt and hash with bcrypt
    salt = bcrypt.gensalt()
    pw_hash = bcrypt.hashpw(p.encode('utf-8'), salt).decode('utf-8')

    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, name, password_hash, case_id) VALUES (?, ?, ?, ?)', 
                  (u, n, pw_hash, c_id))
        conn.commit()
        print(f"[REGISTER SUCCESS] Registered '{u}' with hash prefix: {pw_hash[:10]}...")
        return True
    except sqlite3.IntegrityError:
        print(f"[REGISTER FAILED] Username '{u}' already exists in DB.")
        return False
    except Exception as e:
        print(f"[REGISTER ERROR] Database error: {e}")
        return False
    finally:
        conn.close()

def authenticate_user(identifier, password):
    ident = identifier.strip()
    p = password.strip()

    if not ident or not p:
        return None

    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    
    # Check BOTH handle (username) AND full name (name), case-insensitively
    c.execute('''
        SELECT username, name, password_hash, case_id 
        FROM users 
        WHERE LOWER(username) = LOWER(?) OR LOWER(name) = LOWER(?)
    ''', (ident, ident))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    stored_username, stored_name, stored_hash, stored_case_id = row

    p_bytes = p.encode('utf-8')
    h_bytes = stored_hash.encode('utf-8') if isinstance(stored_hash, str) else stored_hash

    try:
        if bcrypt.checkpw(p_bytes, h_bytes):
            return {
                "username": stored_username, 
                "name": stored_name, 
                "case_id": stored_case_id
            }
        else:
            return None
    except Exception as e:
        print(f"[LOGIN ERROR] {e}")
        return None
def log_session(username, mode, score, triage, statement, features):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute('''
        INSERT INTO triage_logs (username, timestamp, mode, distress_score, triage_level, statement, features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, score, triage, statement, json.dumps(features)))
    conn.commit()
    conn.close()

def get_history(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute('SELECT timestamp, mode, distress_score, triage_level FROM triage_logs WHERE username = ? ORDER BY id DESC', (username,))
    rows = c.fetchall()
    conn.close()
    return rows