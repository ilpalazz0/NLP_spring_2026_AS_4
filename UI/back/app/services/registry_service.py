from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import CACHE_DIR, RUNS_DIR

CACHE_PATH = CACHE_DIR / "model_registry_cache.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_metrics_file(path: Path) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    if not path.exists():
        return history

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" in line:
                continue
            parts = line.split("	")
            if len(parts) not in {5, 8}:
                continue
            row = {
                "epoch": int(parts[0]),
                "train_loss": float(parts[1]),
                "val_loss": float(parts[2]),
                "val_em": float(parts[3]),
                "val_f1": float(parts[4]),
                "test_loss": float(parts[5]) if len(parts) == 8 else None,
                "test_em": float(parts[6]) if len(parts) == 8 else None,
                "test_f1": float(parts[7]) if len(parts) == 8 else None,
            }
            history.append(row)
    return history


def _runs_signature() -> List[Dict[str, Any]]:
    signature: List[Dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return signature

    for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        relevant = [run_dir / "config.json", run_dir / "summary.json", run_dir / "metrics_log.txt", run_dir / "best_model.pt"]
        latest_mtime = 0.0
        for path in relevant:
            if path.exists():
                latest_mtime = max(latest_mtime, path.stat().st_mtime)
        signature.append({"run_id": run_dir.name, "mtime": latest_mtime})
    return signature


def _build_registry() -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    if RUNS_DIR.exists():
        for run_dir in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            config_path = run_dir / "config.json"
            summary_path = run_dir / "summary.json"
            metrics_path = run_dir / "metrics_log.txt"
            best_model_path = run_dir / "best_model.pt"

            if not config_path.exists() or not best_model_path.exists():
                continue

            config = _load_json(config_path)
            summary = _load_json(summary_path) if summary_path.exists() else {}
            history = _parse_metrics_file(metrics_path)

            created_at = None
            try:
                created_at = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
            except Exception:
                created_at = None

            embedding_type = summary.get("embedding_type") or config.get("embedding_type", "unknown")
            name = f"{embedding_type.upper()} · {run_dir.name}"

            runs.append(
                {
                    "run_id": run_dir.name,
                    "name": name,
                    "embedding_type": embedding_type,
                    "best_epoch": summary.get("best_epoch"),
                    "best_val_f1": summary.get("best_val_f1"),
                    "test_em": summary.get("test_em"),
                    "test_f1": summary.get("test_f1"),
                    "train_file": config.get("train_file"),
                    "test_file": config.get("test_file"),
                    "created_at": created_at,
                    "cached": False,
                    "has_loss_plot": (run_dir / "loss_curve.png").exists(),
                    "has_score_plot": (run_dir / "score_curve.png").exists(),
                    "config_preview": {
                        "epochs": config.get("epochs"),
                        "batch_size": config.get("batch_size"),
                        "max_context_len": config.get("max_context_len"),
                        "max_question_len": config.get("max_question_len"),
                        "freeze_bert": config.get("freeze_bert"),
                        "bert_model_name": config.get("bert_model_name"),
                    },
                    "config": config,
                    "metrics_history": history,
                    "summary_json": summary,
                    "static_assets": {
                        "loss_curve_url": f"/runs/{run_dir.name}/loss_curve.png" if (run_dir / "loss_curve.png").exists() else None,
                        "score_curve_url": f"/runs/{run_dir.name}/score_curve.png" if (run_dir / "score_curve.png").exists() else None,
                        "metrics_log_url": f"/runs/{run_dir.name}/metrics_log.txt" if metrics_path.exists() else None,
                        "summary_url": f"/runs/{run_dir.name}/summary.json" if summary_path.exists() else None,
                    },
                }
            )

    runs.sort(key=lambda item: (item.get("test_f1") is not None, item.get("test_f1") or -1.0), reverse=True)
    return {"signature": _runs_signature(), "runs": runs}


def load_registry(force_refresh: bool = False) -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current_signature = _runs_signature()

    if not force_refresh and CACHE_PATH.exists():
        try:
            cached = _load_json(CACHE_PATH)
            if cached.get("signature") == current_signature:
                return cached
        except Exception:
            pass

    fresh = _build_registry()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(fresh, f, indent=2, ensure_ascii=False)
    return fresh
