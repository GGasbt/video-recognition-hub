import threading
import uvicorn
from src.config import HOST, PORT
from src.ml_engine import MLEngine, video_capture_loop
from src.app import app

if __name__ == "__main__":

    ml_engine = MLEngine()

    video_thread = threading.Thread(
        target=video_capture_loop, 
        args=(ml_engine,), 
        daemon=True
    )
    video_thread.start()

    print(f"-> [Server] Starting web interface dashboard on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")