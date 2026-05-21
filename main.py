import threading
from src.app import app
from src.ml_engine import MLEngine, video_capture_loop
from src import config

if __name__ == "__main__":
    engine = MLEngine()
    
    app.state.ml_engine = engine

    capture_thread = threading.Thread(target=video_capture_loop, args=(engine,), daemon=True)
    capture_thread.start()

    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")