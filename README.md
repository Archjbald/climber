# Climber

A rock climbing analysis tool that tracks climbing metrics from video inputs using Human Pose Estimation (HPE), exposed via an asynchronous FastAPI backend.

Built to propose a base framework for an end-to-end CV pipeline. Any training / finetuning / evaluation is out of scope here, mostly for lack of annotated data (and time).

Extracted metrics:
- Number of moves
- Not-moving time
- Trajectory

![demo](https://github.com/Archjbald/climber/blob/docs/documentation/demo.gif)

## Status
Currently in active development. The codebase supports video upload verification, asynchronous background processing, and pose estimation.

## Tech Stack
* **Language:** Python 3.10+
* **HPE Engine:** [rtmlib](https://github.com/Tau-J/rtmlib) for pose estimation
* **API Framework:** FastAPI + Uvicorn for asynchronous task execution
* **Containerization:** Docker

## Project Structure
```text
climber/
├── demo/           # Test scripts for rtmlib
├── src/            
│   ├── app.py      # FastAPI entry point & background tasks
│   ├── main.py     # Application entry point / pipeline runner
│   ├── config.py   # Global configuration settings
│   ├── pose.py     # HPE and tracking logic
│   └── utils.py    # Helper functions
├── test/           # Unitary tests
├── .dockerignore   # Build exclusion rules
├── Dockerfile      # Container definition
└── requirements.txt
```

## Getting Started
### Option A: Run with Docker (Recommended)

Make sure you have Docker installed.
1. Build the Docker image
```bash
docker build -t climber:latest .
```

2. Run the container
```bash
docker run -p 8000:8000 --name climber-app climber:latest
```

### Option B: Local Setup (Development)

1. Clone the repository
```bash
git clone https://github.com/yourusername/climber.git
cd climber
```

2. Set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the API

Start the local development server using:
```bash
uvicorn src.app:app --reload
```

## API Documentation

Once the application is running (via Docker or locally), access the interactive API documentation at:

👉 http://localhost:8000/docs

## Experiment Tracking (MLflow)

Due to lack of labeled climbing dataset, MLflow is used to track:
- rtmlib backend/mode selection
- tuning the post-processing heuristics (smoothing, velocity filtering)

Tuning decisions are made qualitatively by inspecting logged trajectory
output metrics per run.
- `reports/climbing_backend_sweep.csv` — FPS vs. detection confidence across modes
- `reports/climbing_heuristic_tuning.csv` — smoothing/threshold sweep results
- `reports/screenshots/` — MLflow UI run comparison + example artifacts

The full pipeline is packaged as a registered MLflow pyfunc model
(`src/mlflow_model.py`) and servable via `mlflow models serve`.