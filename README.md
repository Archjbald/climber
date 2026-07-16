# Climber

A rock climbing analysis tool that tracks climbing metrics from video inputs using Human Pose Estimation (HPE), exposed via an asynchronous FastAPI backend.

## Status
Currently in active development. The codebase supports video upload verification, asynchronous background processing, and pose estimation.

## Tech Stack
* **Language:** Python 3.10+
* **HPE Engine:** Powered by [rtmlib](https://github.com/Tau-J/rtmlib) for fast and accurate real-time pose estimation.
* **API Framework:** FastAPI + Uvicorn for asynchronous task execution.

## Project Structure
```text
climber/
├── demo/           # Test scripts for rtmlib
├── src/        
│   ├── app.py  # FastAPI entry point & background tasks
│   ├── main.py     # Application entry point / pipeline runner
│   ├── config.py   # Global configuration settings
│   ├── pose.py     # HPE and tracking logic
│   └── utils.py    # Helper functions

```

## Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/yourusername/climber.git](https://github.com/yourusername/climber.git)
cd climber

```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the API

Start the local development server using:

```bash
uvicorn src.app:app --reload

```

Once running, access the interactive API documentation at:

👉 **http://127.0.0.1:8000/docs**
