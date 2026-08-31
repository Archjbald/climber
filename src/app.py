"""FastAPI service for uploading a climbing video and retrieving its analysis."""

import asyncio
import uuid
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import config as cfg
from src.main import process_vid
from src.utils import check_vid

app = FastAPI()

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_SUFFIXES = (".mp4", ".mov", ".avi")
CHUNK_SIZE = 1024 * 1024  # 1 MiB

tasks_db: dict[str, dict] = {}
_running_tasks: set[asyncio.Task] = set()
_busy = False


def _job_running() -> bool:
    return _busy or any(t["status"] == "processing" for t in tasks_db.values())


def _resolve_upload_path(filename: str) -> Path:
    """Map a client-supplied filename to a safe, unique path inside UPLOAD_DIR."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only MP4, MOV, and AVI are supported.")
    return UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream an upload to `dest`, enforcing the configured size cap."""
    max_bytes = cfg.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        async with await anyio.open_file(dest, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {cfg.MAX_UPLOAD_MB} MB limit.",
                    )
                await buffer.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e


async def _run_analysis(task_id: str, file_path: Path) -> None:
    """Run the pipeline off the event loop and store the outcome under `task_id`."""
    loop = asyncio.get_running_loop()
    try:
        analysis = await loop.run_in_executor(None, process_vid, str(file_path))
        tasks_db[task_id] = {"status": "success", "analysis": analysis}
    except Exception as e:  # noqa: BLE001 - background job must record any failure
        tasks_db[task_id] = {"status": "failed", "error": str(e)}
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/analyze", status_code=202)
async def analyze_video(file: UploadFile = File(...)) -> dict:
    """Validate an uploaded video and start its analysis (one job at a time)."""
    global _busy
    if _job_running():
        raise HTTPException(
            status_code=409, detail="An analysis is already running. Retry once it is done."
        )

    _busy = True
    try:
        dest = _resolve_upload_path(file.filename or "")
        await _save_upload(file, dest)

        if not check_vid(str(dest)):
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400, detail="Uploaded file is corrupted or not a valid video."
            )

        task_id = str(uuid.uuid4())
        tasks_db[task_id] = {"status": "processing"}
        task = asyncio.create_task(_run_analysis(task_id, dest))
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.discard)
    finally:
        _busy = False

    return {
        "status": "processing",
        "task_id": task_id,
        "message": "Video received and verified. Analysis is running in background.",
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """Return the stored status and result for `task_id`."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]
