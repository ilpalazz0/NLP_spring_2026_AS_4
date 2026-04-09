# Task 2 Showcase Backend

This backend reads saved Task 2 training runs from `./code/task_2/runs`, caches their metrics once at startup, and serves:
- model summaries
- per-epoch history
- inference for the selected run
- static files from the run folders

## Run

From project root:

```powershell
pip install -r .\UI\back\requirements.txt
uvicorn UI.back.app.main:app --reload
```

If your import path does not resolve from project root, use:

```powershell
cd .\UI\back
uvicorn app.main:app --reload
```

The API starts on `http://127.0.0.1:8000` by default.
