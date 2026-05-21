import os

# Пути к файлам
DB_DIR = "face_db"
SQLITE_DB_PATH = "faces_embeddings.db"  # Наша новая база данных

# Системные настройки
HOST = "0.0.0.0"
PORT = 8000
IP_WEBCAM_URL = "http://192.168.1.3:8080/video"

MODEL_NAME = "Facenet512"
THRESHOLD = 23.5

# Глобальный стейт
latest_processed_frame = None
track_identities = {}
stats = {"total_persons": 0, "recognized": 0, "unknown": 0}
is_running = True
camera_changed = False