import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from src import config

class MLEngine:
    def __init__(self):
        print("-> [ML] Loading YOLO models...")
        self.person_model = YOLO("yolov8n.pt")
        self.face_model = YOLO("yolov8n-face.pt")
        self.known_faces = {}
        self._load_face_database()

    def _load_face_database(self):
        if os.path.exists(config.DB_DIR):
            print("-> [ML] Loading Face Database...")
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
                            self.known_faces[person_name].append(np.array(embedding_objs[0]["embedding"]))
                        except Exception:
                            pass
            print(f"-> [ML] Loaded {len(self.known_faces)} identities.")

    def process_frame(self, frame, frame_count):
        # Ресайз до 640х480 для стабильного FPS на YOLO
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
                
                # Запуск DeepFace раз в 25 кадров
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
    print(f"-> [Capture] Connecting to: {config.IP_WEBCAM_URL}")
    cap = cv2.VideoCapture(config.IP_WEBCAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    
    while config.is_running:
        ret, frame = cap.read()
        if not ret:
            print("!!! [Capture] Error reading frame. Reconnecting...")
            cap.release()
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(config.IP_WEBCAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
            
        frame_count += 1
        
        # Инференс
        processed_frame, total, rec, unk = ml_engine.process_frame(frame, frame_count)
        
        # Обновление глобального стейта
        config.stats["total_persons"] = total
        config.stats["recognized"] = rec
        config.stats["unknown"] = unk
        config.latest_processed_frame = processed_frame

    cap.release()