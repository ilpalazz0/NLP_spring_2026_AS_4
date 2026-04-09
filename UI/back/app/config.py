from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACK_DIR = APP_DIR.parent
UI_DIR = BACK_DIR.parent
PROJECT_ROOT = UI_DIR.parent
TASK2_DIR = PROJECT_ROOT / "code" / "task_2"
RUNS_DIR = TASK2_DIR / "runs"
CACHE_DIR = BACK_DIR / "cache"
SENTIMENT_MODEL_DIR = BACK_DIR / "sentiment_model"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
