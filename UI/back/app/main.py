from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import ALLOWED_ORIGINS, RUNS_DIR, SENTIMENT_MODEL_DIR
from .schemas import (
    ModelDetail,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    SentimentPredictRequest,
    SentimentPredictResponse,
)
from .services.model_loader import ModelManager
from .services.registry_service import load_registry
from .services.sentiment_service import SentimentManager

app = FastAPI(title="Task 2 QA Showcase API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry_payload = load_registry(force_refresh=False)
model_manager = ModelManager(registry_payload)
sentiment_manager = SentimentManager(SENTIMENT_MODEL_DIR)

if RUNS_DIR.exists():
    app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": model_manager.device.type,
        "runs_indexed": len(model_manager.registry_payload.get("runs", [])),
        "sentiment_model_ready": sentiment_manager.is_available(),
        "sentiment_model_cached": sentiment_manager.is_cached(),
    }


@app.get("/api/models", response_model=list[ModelSummary])
def list_models(refresh: bool = Query(default=False)):
    global registry_payload
    if refresh:
        registry_payload = load_registry(force_refresh=True)
        model_manager.refresh_registry(registry_payload)

    runs = []
    for item in model_manager.registry_payload.get("runs", []):
        row = dict(item)
        row["cached"] = model_manager.is_cached(item["run_id"])
        row.pop("config", None)
        row.pop("metrics_history", None)
        row.pop("summary_json", None)
        row.pop("static_assets", None)
        runs.append(row)
    return runs


@app.get("/api/models/{run_id}", response_model=ModelDetail)
def get_model_detail(run_id: str):
    item = model_manager.registry_map.get(run_id)
    if not item:
        raise HTTPException(status_code=404, detail="Model run not found")

    summary_payload = dict(item)
    summary_payload["cached"] = model_manager.is_cached(run_id)
    config = summary_payload.pop("config")
    metrics_history = summary_payload.pop("metrics_history")
    summary_json = summary_payload.pop("summary_json")
    static_assets = summary_payload.pop("static_assets")

    return ModelDetail(
        summary=ModelSummary(**summary_payload),
        config=config,
        metrics_history=metrics_history,
        summary_json=summary_json,
        static_assets=static_assets,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not payload.context.strip():
        raise HTTPException(status_code=400, detail="Context cannot be empty")

    try:
        result = model_manager.predict(
            run_id=payload.run_id,
            question=payload.question,
            context=payload.context,
            max_answer_len=payload.max_answer_len,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(run_id=payload.run_id, **result)


@app.post("/api/sentiment/predict", response_model=SentimentPredictResponse)
def predict_sentiment(payload: SentimentPredictRequest):
    if not sentiment_manager.is_available():
        raise HTTPException(
            status_code=503,
            detail="Sentiment model is not ready. Put config.json, model.safetensors, tokenizer.json, and tokenizer_config.json into UI/back/sentiment_model. If tokenizer loading fails, also add vocab.txt.",
        )

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        result = sentiment_manager.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SentimentPredictResponse(**result)
