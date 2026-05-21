import os
import cv2
import numpy as np
import time
import sqlite3
from ultralytics import YOLO
from deepface import DeepFace
from src import config

class MLEngine:
    def __init__(self):
        print("-> [ML] Loading YOLO models...")
        self.person_model = YOLO("yolov8n.pt")
        self.face_model = YOLO("yolov8n-face.pt")
        self.known_faces = {}
        
        # Инициализируем SQLite БД
        self._init_db()
        # Загружаем эмбеддинги в оперативную память для быстрого инференса
        self._load_embeddings()

    def _init_db(self):
        """Создает таблицу в БД, если её еще нет."""
        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def _load_embeddings(self):
        """Загружает эмбеддинги из БД. Если БД пуста, сканирует папки."""
        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, embedding FROM face_embeddings")
            rows = cursor.fetchall()

        if rows:
            print("-> [ML] Loading embeddings from SQLite database...")
            for name, emb_blob in rows:
                # Десериализуем вектор обратно в numpy array (FaceNet512 дает 512 значений float64)
                embedding = np.frombuffer(emb_blob, dtype=np.float64)
                if name not in self.known_faces:
                    self.known_faces[name] = []
                self.known_faces[name].append(embedding)
            print(f"-> [ML] Loaded {len(self.known_faces)} identities from DB.")
        else:
            print("-> [ML] Database is empty. Scanning 'face_db' folder for cold start...")
            self._cold_start_from_folders()

    def _cold_start_from_folders(self):
        """Первичный импорт картинок из папок в SQLite."""
        if not os.path.exists(config.DB_DIR):
            print(f"-> [ML] Warning: '{config.DB_DIR}' folder not found.")
            return

        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            
            for person_name in os.listdir(config.DB_DIR):
                person_folder = os.path.join(config.DB_DIR, person_name)
                if os.path.isdir(person_folder):
                    self.known_faces[person_name] = []
                    for img_name in os.listdir(person_folder):
                        img_path = os.path.join(person_folder, img_name)
                        try:
                            embedding_objs = DeepFace.represent(
                                img_path=img_path, 
                                model_name=config.MODEL_NAME, 
                                enforce_detection=True, 
                                detector_backend="opencv"
                            )
                            emb_vector = np.array(embedding_objs[0]["embedding"], dtype=np.float64)
                            
                            # Сохраняем в память
                            self.known_faces[person_name].append(emb_vector)
                            
                            # Сохраняем в SQLite в бинарном виде
                            cursor.execute(
                                "INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)",
                                (person_name, emb_vector.tobytes())
                            )
                        except Exception:
                            print(f"!!! [ML] Failed to process image: {img_path}")
            conn.commit()
        print(f"-> [ML] Cold start finished. Loaded {len(self.known_faces)} identities to DB.")

    def add_new_identity(self, name, frame_or_path):
        """Метод для добавления нового человека в базу данных 'на лету'."""
        try:
            embedding_objs = DeepFace.represent(
                img_path=frame_or_path, 
                model_name=config.MODEL_NAME, 
                enforce_detection=True, 
                detector_backend="opencv"
            )
            emb_vector = np.array(embedding_objs[0]["embedding"], dtype=np.float64)
            
            # Пишем в SQLite
            with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)",
                    (name, emb_vector.tobytes())
                )
                conn.commit()
            
            # Обновляем оперативную память движка
            if name not in self.known_faces:
                self.known_faces[name] = []
            self.known_faces[name].append(emb_vector)
            
            print(f"-> [ML] Successfully added new identity: {name}")
            return True
        except Exception as e:
            print(f"!!! [ML] Error adding new identity {name}: {e}")
            return False

    def process_frame(self, frame, frame_count):
        frame = cv2.resize(frame, (640, 480))
        track_results = self.person_model.track(frame, persist=True, classes=[0], conf=0.4, verbose=False)
        
        current_total = 0
        current_rec = 0
        current_unk = 0
        
        if track_results[0].boxes.id is not None:
            boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = track_results[0].boxes.id.cpu().numpy().astype(int)
            current_total = len(ids)
            
            for box, person_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                identity = config.track_identities.get(person_id, "Checking...")
                
                if identity in ["Checking...", "Unknown"] and frame_count % 25 == 0:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:
                        face_results = self.face_model(person_crop, verbose=False)
                        if face_results[0].boxes is not None and len(face_results[0].boxes) > 0:
                            fx1, fy1, fx2, fy2 = face_results[0].boxes.xyxy.cpu().numpy().astype(int)[0]
                            face_crop = person_crop[max(0, fy1):min(person_crop.shape[0], fy2), 
                                                    max(0, fx1):min(person_crop.shape[1], fx2)]
                            
                            if face_crop.size > 0:
                                try:
                                    current_emb_objs = DeepFace.represent(
                                        img_path=face_crop, model_name=config.MODEL_NAME, 
                                        enforce_detection=False, detector_backend="opencv"
                                    )
                                    current_emb = np.array(current_emb_objs[0]["embedding"])
                                    
                                    min_dist = float("inf")
                                    best_match = "Unknown"
                                    
                                    for name, embeddings in self.known_faces.items():
                                        for known_emb in embeddings:
                                            dist = np.linalg.norm(current_emb - known_emb)
                                            if dist < config.THRESHOLD and dist < min_dist:
                                                min_dist = dist
                                                best_match = name
                                    
                                    identity = best_match
                                    config.track_identities[person_id] = identity
                                except Exception:
                                    pass
                
                if identity not in ["Unknown", "Checking..."]:
                    current_rec += 1
                    color = (0, 255, 0)
                else:
                    current_unk += 1
                    color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {person_id} | {identity}", (x1, max(25, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        return frame, current_total, current_rec, current_unk


def video_capture_loop(ml_engine):
    # Код функции видеопотока оставляем прежним (с поддержкой config.camera_changed)
    print(f"-> [Capture] Connecting to initial source: {config.IP_WEBCAM_URL}")
    cap = cv2.VideoCapture(config.IP_WEBCAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_count = 0
    
    while config.is_running:
        if config.camera_changed:
            print(f"-> [Capture] Camera switch requested! Connecting to: {config.IP_WEBCAM_URL}")
            cap.release()
            config.track_identities.clear()
            cap = cv2.VideoCapture(config.IP_WEBCAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            config.camera_changed = False
            frame_count = 0
            
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"!!! [Capture] Error reading frame. Reconnecting to: {config.IP_WEBCAM_URL}")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(config.IP_WEBCAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
            
        frame_count += 1
        processed_frame, total, rec, unk = ml_engine.process_frame(frame, frame_count)
        
        config.stats["total_persons"] = total
        config.stats["recognized"] = rec
        config.stats["unknown"] = unk
        config.latest_processed_frame = processed_frame

    cap.release()