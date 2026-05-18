import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# Fix Qt/Wayland graphics platform issue on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"

DB_DIR = "face_db"
MODEL_NAME = "Facenet512"
THRESHOLD = 23.5

person_model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8n-face.pt")

print("-> Initializing face database...")
known_faces = {}

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)
    print(f"Directory '{DB_DIR}' created. Populate it and restart.")
    exit()

for person_name in os.listdir(DB_DIR):
    person_folder = os.path.join(DB_DIR, person_name)
    if os.path.isdir(person_folder):
        known_faces[person_name] = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                embedding_objs = DeepFace.represent(
                    img_path=img_path, 
                    model_name=MODEL_NAME, 
                    enforce_detection=True,
                    detector_backend="opencv"
                )
                embedding = np.array(embedding_objs[0]["embedding"])
                known_faces[person_name].append(embedding)
            except Exception:
                print(f"Skipping {img_path}: face not detected.")

print(f"-> Database loaded. Total people: {len(known_faces)}")

video_path = "data/14821899_2560_1440_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# Cache identities by tracker ID to prevent redundant face recognition on every frame
track_identities = {}
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame_count += 1
    track_results = person_model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False)
    
    if track_results[0].boxes.id is not None:
        boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = track_results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, person_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            identity = track_identities.get(person_id, "Checking...")
            
            # Run face recognition once every 15 frames per person to optimize CPU usage
            if identity in ["Checking...", "Unknown"] and frame_count % 15 == 0:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    face_results = face_model(person_crop, verbose=False)
                    
                    if face_results[0].boxes is not None and len(face_results[0].boxes) > 0:
                        fx1, fy1, fx2, fy2 = face_results[0].boxes.xyxy.cpu().numpy().astype(int)[0]
                        face_crop = person_crop[max(0, fy1):min(person_crop.shape[0], fy2), 
                                                max(0, fx1):min(person_crop.shape[1], fx2)]
                        
                        if face_crop.size > 0:
                            try:
                                current_emb_objs = DeepFace.represent(
                                    img_path=face_crop, model_name=MODEL_NAME, 
                                    enforce_detection=False, detector_backend="opencv"
                                )
                                current_emb = np.array(current_emb_objs[0]["embedding"])
                                
                                min_dist = float("inf")
                                best_match = "Unknown"
                                
                                # Compare current embedding with all known embeddings using Euclidean distance
                                for name, embeddings in known_faces.items():
                                    for known_emb in embeddings:
                                        dist = np.linalg.norm(current_emb - known_emb)
                                        if dist < THRESHOLD and dist < min_dist:
                                            min_dist = dist
                                            best_match = name
                                
                                identity = best_match
                                track_identities[person_id] = identity
                                
                            except Exception:
                                pass
            
            # Draw bounding boxes and text labels
            color = (0, 255, 0) if identity not in ["Unknown", "Checking..."] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {person_id} | {identity}", (x1, max(15, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    cv2.imshow("Face Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()