import threading
import uvicorn
from src.config import HOST, PORT
from src.ml_engine import MLEngine, video_capture_loop
from src.app import app

if __name__ == "__main__":
    # Инициализируем ML движок (загрузит веса и эмбеддинги базы данных)
    ml_engine = MLEngine()
    
    # Стартуем фоновый демон-поток для непрерывного захвата кадров из IP Webcam
    video_thread = threading.Thread(
        target=video_capture_loop, 
        args=(ml_engine,), 
        daemon=True
    )
    video_thread.start()
    
    # Запуск Uvicorn-сервера для веб-дашборда
    print(f"-> [Server] Starting web interface dashboard on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")