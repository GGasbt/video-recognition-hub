import cv2
import os
import numpy as np
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from src import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/")
async def pc_dashboard(request: Request):
    # Современный синтаксис FastAPI: контекст передается первым именованным аргументом (или просто через context=...)
    return templates.TemplateResponse(
        request=request, 
        name="dashboard.html"
    )

@app.get("/api/stats")
async def get_stats():
    return config.stats

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        while True:
            if config.latest_processed_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', config.latest_processed_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "CONNECTING TO STREAM THREAD...", (90, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', blank_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            await asyncio.sleep(0.03)
            
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    config.is_running = False