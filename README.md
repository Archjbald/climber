# Climber

A rock climbing analysis tool that reads video  and applies Human Pose Estimation (HPE) to track climbing metrics.

## Status
Currently in early development. The codebase includes a baseline framework for reading video inputs and running HPE to extract keypoints.

## Tech Stack
* **Language:** Python 3.10+
* **HPE Engine:** Powered by [rtmlib](https://github.com/Tau-J/rtmlib) for fast and accurate real-time pose estimation.

## Project Structure
```text
climber/
├── demo/           # test scripts for rtmlib
├── src/        
│   ├── main.py     # Application entry point
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

*(Note: Ensure you install `rtmlib` and your other required packages here)*

```bash
pip install -r requirements.txt

```

### 4. Run the application

Run the main script from the root directory using:

```bash
python src/main.py

```
