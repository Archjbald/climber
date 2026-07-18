from unittest.mock import patch, ANY
import pytest
from fastapi.testclient import TestClient

from fastapi import BackgroundTasks

from src.app import submit_video, tasks_db, app, async_process_vid
from tests.helpers import make_video, make_process_results

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_tasks():
    tasks_db.clear()
    yield
    tasks_db.clear()


"""
Test video upload
"""


# Test invalid video data
@patch("src.app.check_vid")
def test_analyze_video_invalid_video(mock_check_vid):
    mock_check_vid.return_value = False

    response = client.post("/analyze", files=make_video())

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is corrupted or not a valid video."
    }

    mock_check_vid.assert_called_once()


# Test invalid video extension
@pytest.mark.parametrize(
    "filename",
    [
        "video.txt",
        "video.png",
        "video.pdf",
    ],
)
def test_analyze_video_invalid_extension(filename):
    response = client.post("/analyze", files=make_video(filename))

    assert response.status_code == 400
    assert response.json() == {"detail": "Only MP4, MOV, and AVI are supported."}

# Test empty file
def test_analyze_video_empty_file_fails():
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/analyze", files=files)

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is corrupted or not a valid video."}

# Test successful upload
@patch("src.app.check_vid")
@patch("src.app.submit_video")
def test_analyze_video_success(
    mock_submit_video,
    mock_check_vid,
):
    mock_check_vid.return_value = True
    mock_submit_video.return_value = "12345678-1234-5678-1234-567812345678"

    response = client.post("/analyze", files=make_video())

    assert response.status_code == 200
    assert response.json() == {
        "status": "processing",
        "task_id": "12345678-1234-5678-1234-567812345678",
        "message": "Video received and verified. Analysis is running in background.",
    }

    mock_check_vid.assert_called_once()
    mock_submit_video.assert_called_once_with(
        "temp_uploads/video.mp4",
        ANY,
    )


# Test failed upload
@patch("builtins.open", side_effect=OSError("No space left on device"))
def test_analyze_video_upload_failure(mock_open):
    files = {"file": ("video.mp4", b"dummy bytes", "video/mp4")}

    response = client.post("/analyze", files=files)

    assert response.status_code == 500
    assert "Failed to save upload" in response.json()["detail"]


"""
Test submit video
"""


# Test submit video success
@patch("src.app.async_process_vid")
def test_submit_video(mock_async_process_vid):
    background_tasks = BackgroundTasks()

    with patch.object(background_tasks, "add_task") as mock_add_task:
        task_id = submit_video("video.mp4", background_tasks)

    assert task_id in tasks_db
    assert tasks_db[task_id] == {"status": "processing"}

    mock_add_task.assert_called_once_with(
        mock_async_process_vid,
        task_id,
        "video.mp4",
    )


"""
Test async process video
"""


@patch("src.app.os.remove")
@patch("src.app.process_vid")
def test_async_process_success(mock_process_vid, mock_remove):
    task_id = "12345678-1234-5678-1234-567812345678"
    # file_path = "TO_REMOVE.mp4"
    file_path = __file__  # Don't need a video but needs a file that exists

    process_result = make_process_results()
    mock_process_vid.return_value = process_result

    async_process_vid(task_id, file_path)

    mock_process_vid.assert_called_once_with(file_path)
    assert tasks_db[task_id]["status"] == "success"
    assert tasks_db[task_id]["analysis"] == process_result
    mock_remove.assert_called_once_with(file_path)


@patch("src.app.os.remove")
@patch("src.app.process_vid")
def test_async_process_fails(mock_process_vid, mock_remove):
    task_id = "12345678-1234-5678-1234-567812345678"
    # file_path = "TO_REMOVE.mp4"
    file_path = __file__  # Don't need a video but needs a file that exists
    mock_process_vid.side_effect = Exception("OpenCV: Frame corruption error")

    async_process_vid(task_id, file_path)

    mock_process_vid.assert_called_once_with(file_path)
    assert tasks_db[task_id]["status"] == "failed"
    assert tasks_db[task_id]["error"] == "OpenCV: Frame corruption error"
    mock_remove.assert_called_once_with(file_path)


"""
Test get task status
"""


def test_get_task_existing():
    tasks_db["task-1"] = {
        "status": "success",
        "analysis": {"score": 42},
    }

    response = client.get("/tasks/task-1")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "analysis": {"score": 42},
    }


def test_get_task_not_found():
    response = client.get("/tasks/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}
