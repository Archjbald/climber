import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from src.main import process_vid
from src.config import config as cfg
from src.utils import check_vid

cfg.DEBUG = False

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

tasks_db = {}

def submit_video(file_path: str, background_tasks: BackgroundTasks) -> str:
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "processing"}
    background_tasks.add_task(async_process_vid, task_id, file_path)
    return task_id

def async_process_vid(task_id: str, file_path: str):
    # Analyse video and update task DB
    print("** Starting process", task_id)
    try:
        results = process_vid(file_path)
        tasks_db[task_id] = {"status": "success", "analysis": results}
        print("** Finished process", task_id)
    except Exception as e:
        tasks_db[task_id] = {"status": "failed", "error": str(e)}
        print("** Aborted process", task_id)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    # Expect video file
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(
            status_code=400, detail="Only MP4, MOV, and AVI are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Write received vid
    try:
        with open(file_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):  # 1 Mo
                buffer.write(chunk)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    # Check video integrity
    if not (check_vid(file_path)):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400, detail="Uploaded file is corrupted or not a valid video."
        )

    # Launch video process in background
    task_id = submit_video(file_path, background_tasks)

    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Video received and verified. Analysis is running in background.",
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]
