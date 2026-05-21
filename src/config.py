import os

DB_DIR = "face_db"
SQLITE_DB_PATH = "faces_embeddings.db"

HOST = "0.0.0.0"
PORT = 8000
IP_WEBCAM_URL = "http://192.168.1.3:8080/video"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"  
ADMIN_COOKIE_NAME = "admin_session"
ADMIN_COOKIE_VALUE = "super-secret-session-token"

# ML Архитектура
MODEL_NAME = "Facenet512"
THRESHOLD = 23.5

latest_processed_frame = None
track_identities = {}
stats = {"total_persons": 0, "recognized": 0, "unknown": 0}
is_running = True
camera_changed = False