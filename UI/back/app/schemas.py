from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    run_id: str
    name: str
    embedding_type: str
    best_epoch: Optional[int] = None
    best_val_f1: Optional[float] = None
    test_em: Optional[float] = None
    test_f1: Optional[float] = None
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    created_at: Optional[str] = None
    cached: bool = False
    has_loss_plot: bool = False
    has_score_plot: bool = False
    config_preview: Dict[str, Any] = Field(default_factory=dict)


class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_em: float
    val_f1: float
    test_loss: Optional[float] = None
    test_em: Optional[float] = None
    test_f1: Optional[float] = None


class ModelDetail(BaseModel):
    summary: ModelSummary
    config: Dict[str, Any]
    metrics_history: List[EpochMetrics]
    summary_json: Dict[str, Any]
    static_assets: Dict[str, Optional[str]]


class PredictRequest(BaseModel):
    run_id: str
    question: str
    context: str
    max_answer_len: int = 30


class PredictResponse(BaseModel):
    run_id: str
    answer: str
    confidence: float
    start_index: int
    end_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    cached_model: bool
    embedding_type: str


class SentimentPredictRequest(BaseModel):
    text: str


class SentimentScore(BaseModel):
    label: str
    score: float


class SentimentPredictResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    scores: List[SentimentScore]
    cached_model: bool
    model_name: str
    device: str
