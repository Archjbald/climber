"""Tests for the src.app FastAPI endpoints and background analysis handling."""

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.app
from src.app import app, tasks_db
from tests.helpers import make_video, make_process_results

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_tasks():
    tasks_db.clear()
    src.app._busy = False
    yield
    tasks_db.clear()
    src.app._busy = False


# --- POST /analyze ---------------------------------------------------------


@patch("src.app.check_vid")
def test_analyze_video_invalid_video(mock_check_vid):
    mock_check_vid.return_value = False

    response = client.post("/analyze", files=make_video())

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is corrupted or not a valid video."
    }
    mock_check_vid.assert_called_once()


@pytest.mark.parametrize("filename", ["video.txt", "video.png", "video.pdf"])
def test_analyze_video_invalid_extension(filename):
    response = client.post("/analyze", files=make_video(filename))

    assert response.status_code == 400
    assert response.json() == {"detail": "Only MP4, MOV, and AVI are supported."}


def test_analyze_video_empty_file_fails():
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is corrupted or not a valid video."
    }


def test_analyze_video_too_large():
    with patch.object(src.app.cfg, "MAX_UPLOAD_MB", 0):
        response = client.post("/analyze", files=make_video())

    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


def test_analyze_video_rejects_concurrent_job():
    tasks_db["already-running"] = {"status": "processing"}

    response = client.post("/analyze", files=make_video())

    assert response.status_code == 409


@patch("src.app.asyncio.create_task")
@patch("src.app._run_analysis")
@patch("src.app._save_upload")
@patch("src.app.check_vid", return_value=True)
def test_analyze_video_success(mock_check_vid, mock_save, mock_run, mock_create_task):
    response = client.post("/analyze", files=make_video())

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "processing"
    assert tasks_db[body["task_id"]] == {"status": "processing"}

    mock_check_vid.assert_called_once()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == body["task_id"]
    mock_create_task.assert_called_once()


@patch("builtins.open", side_effect=OSError("No space left on device"))
def test_analyze_video_upload_failure(mock_open):
    response = client.post("/analyze", files=make_video())

    assert response.status_code == 500
    assert "Failed to save upload" in response.json()["detail"]


# --- _run_analysis -------------------------------------------------------------


def test_run_analysis_success(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    tasks_db["t1"] = {"status": "processing"}

    results = make_process_results()
    with patch("src.app.process_vid", return_value=results) as mock_process:
        asyncio.run(src.app._run_analysis("t1", video))

    mock_process.assert_called_once_with(str(video))
    assert tasks_db["t1"] == {"status": "success", "analysis": results}
    assert not video.exists()


def test_run_analysis_failure(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    tasks_db["t1"] = {"status": "processing"}

    with patch("src.app.process_vid", side_effect=Exception("OpenCV: Frame corruption error")):
        asyncio.run(src.app._run_analysis("t1", video))

    assert tasks_db["t1"] == {
        "status": "failed",
        "error": "OpenCV: Frame corruption error",
    }
    assert not video.exists()


# --- GET /tasks/{task_id} ----------------------------------------------------


def test_get_task_existing():
    tasks_db["task-1"] = {"status": "success", "analysis": {"score": 42}}

    response = client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json() == {"status": "success", "analysis": {"score": 42}}


def test_get_task_not_found():
    response = client.get("/tasks/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}
