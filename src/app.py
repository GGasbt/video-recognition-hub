import cv2
import numpy as np
import asyncio
import os
import sqlite3
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def is_admin(request: Request) -> bool:
    return request.cookies.get(config.ADMIN_COOKIE_NAME) == config.ADMIN_COOKIE_VALUE


@app.get("/")
async def pc_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="dashboard.html", 
        context={"current_url": config.IP_WEBCAM_URL, "is_admin": is_admin(request)} 
    )

@app.get("/admin")
async def admin_panel(request: Request):
    if not is_admin(request):
        return HTMLResponse(content="""
        <div style="background:#0b0f19;color:#fff;height:100vh;display:flex;justify-content:center;align-items:center;font-family:sans-serif;">
            <form action="/login" method="post" style="background:#111827;padding:30px;border-radius:12px;border:1px solid #1f2937;display:flex;flex-direction:column;gap:12px;width:300px;">
                <h3 style="color:#38bdf8;margin-bottom:10px;">Admin Login</h3>
                <input type="text" name="username" placeholder="Username" required style="background:#1f2937;color:#fff;border:1px solid #374151;padding:10px;border-radius:6px;">
                <input type="password" name="password" placeholder="Password" required style="background:#1f2937;color:#fff;border:1px solid #374151;padding:10px;border-radius:6px;">
                <button type="submit" style="background:#3b82f6;color:#fff;border:none;padding:10px;border-radius:6px;cursor:pointer;font-weight:600;">Sign In</button>
                <a href="/" style="color:#9ca3af;text-align:center;font-size:12px;text-decoration:none;margin-top:5px;">Back to Dashboard</a>
            </form>
        </div>
        """, status_code=200)

    registered_users = []
    try:
        with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM face_embeddings")
            registered_users = [row[0] for row in cursor.fetchall()]
    except Exception:
        pass

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"registered_users": registered_users, "current_url": config.IP_WEBCAM_URL}
    )


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
        response = RedirectResponse(url="/admin", status_code=303)
        # Ставим куку авторизации (expires=None означает сессионную куку до закрытия браузера)
        response.set_cookie(key=config.ADMIN_COOKIE_NAME, value=config.ADMIN_COOKIE_VALUE, httponly=True)
        return response
    return HTMLResponse(content="<h2>Wrong credentials. <a href='/admin'>Try again</a></h2>", status_code=401)


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(config.ADMIN_COOKIE_NAME)
    return response


@app.post("/api/change_camera")
async def change_camera(camera_url: str = Form(...)):
    if camera_url.strip():
        config.IP_WEBCAM_URL = camera_url.strip()
        config.camera_changed = True
    return RedirectResponse(url="/", status_code=303)


@app.post("/api/add_person")
async def add_person(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required")
        
    if name.strip() and file.filename:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                engine = request.app.state.ml_engine
                success = engine.add_new_identity(name.strip(), frame)
                if success:
                    print(f"-> [Server] Registered face for: {name.strip()}")
        except Exception as e:
            print(f"!!! [Server] Error adding person: {e}")
            
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/api/delete_person")
async def delete_person(request: Request, name: str = Form(...)):
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required")
        
    if name.strip():
        engine = request.app.state.ml_engine
        engine.delete_identity(name.strip())
        
    return RedirectResponse(url="/admin", status_code=303)


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