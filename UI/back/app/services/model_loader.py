from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from ..config import TASK2_DIR

if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from dataset import Vocab, basic_tokenize_with_spans, load_glove_embeddings, load_squad
from models import BertBiDAF, GloveBiDAF
from utils import best_span_from_logits


def _safe_mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


@dataclass
class LoadedModelBundle:
    run_id: str
    embedding_type: str
    config: Dict[str, Any]
    model: torch.nn.Module
    device: torch.device
    tokenizer: Optional[Any] = None
    vocab: Optional[Vocab] = None


class ModelManager:
    def __init__(self, registry_payload: Dict[str, Any]):
        self.registry_payload = registry_payload
        self.registry_map = {item["run_id"]: item for item in registry_payload.get("runs", [])}
        self.loaded_models: Dict[str, LoadedModelBundle] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def refresh_registry(self, registry_payload: Dict[str, Any]) -> None:
        self.registry_payload = registry_payload
        self.registry_map = {item["run_id"]: item for item in registry_payload.get("runs", [])}

    def is_cached(self, run_id: str) -> bool:
        return run_id in self.loaded_models

    def _resolve_run_dir(self, run_id: str) -> Path:
        if run_id not in self.registry_map:
            raise KeyError(f"Run not found: {run_id}")
        return TASK2_DIR / "runs" / run_id

    def load_model_once(self, run_id: str) -> LoadedModelBundle:
        if run_id in self.loaded_models:
            return self.loaded_models[run_id]

        meta = self.registry_map.get(run_id)
        if not meta:
            raise KeyError(f"Unknown run_id: {run_id}")
        config = meta["config"]
        run_dir = self._resolve_run_dir(run_id)
        weights_path = run_dir / "best_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        embedding_type = config.get("embedding_type")
        if embedding_type == "bert":
            model = BertBiDAF(
                bert_model_name=config.get("bert_model_name", "bert-base-uncased"),
                hidden_size=int(config.get("hidden_size", 128)),
                dropout=float(config.get("dropout", 0.2)),
                freeze_bert=bool(config.get("freeze_bert", True)),
            )
            tokenizer = AutoTokenizer.from_pretrained(config.get("bert_model_name", "bert-base-uncased"), use_fast=True)
            vocab = None
        elif embedding_type == "glove":
            train_examples = load_squad(config["train_file"])
            sample_limit = int(config.get("train_sample_limit", 0) or 0)
            if sample_limit > 0:
                train_examples = train_examples[:sample_limit]
            vocab = Vocab.build(train_examples, min_freq=2)
            embedding_matrix = load_glove_embeddings(
                config["glove_path"],
                vocab,
                embedding_dim=100,
            )
            model = GloveBiDAF(
                embedding_matrix=embedding_matrix,
                hidden_size=int(config.get("hidden_size", 128)),
                dropout=float(config.get("dropout", 0.2)),
                train_embeddings=bool(config.get("train_embeddings", True)),
            )
            tokenizer = None
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        bundle = LoadedModelBundle(
            run_id=run_id,
            embedding_type=embedding_type,
            config=config,
            model=model,
            device=self.device,
            tokenizer=tokenizer,
            vocab=vocab,
        )
        self.loaded_models[run_id] = bundle
        return bundle

    def predict(self, run_id: str, question: str, context: str, max_answer_len: int = 30) -> Dict[str, Any]:
        was_cached = run_id in self.loaded_models
        bundle = self.load_model_once(run_id)
        if bundle.embedding_type == "bert":
            result = self._predict_bert(bundle, question, context, max_answer_len=max_answer_len)
        else:
            result = self._predict_glove(bundle, question, context, max_answer_len=max_answer_len)
        result["cached_model"] = was_cached
        result["embedding_type"] = bundle.embedding_type
        return result

    def _predict_bert(self, bundle: LoadedModelBundle, question: str, context: str, max_answer_len: int = 30) -> Dict[str, Any]:
        tokenizer = bundle.tokenizer
        config = bundle.config
        context_enc = tokenizer(
            context,
            truncation=True,
            max_length=int(config.get("max_context_len", 256)),
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        question_enc = tokenizer(
            question,
            truncation=True,
            max_length=int(config.get("max_question_len", 32)),
            add_special_tokens=True,
        )

        context_ids = torch.tensor([context_enc["input_ids"]], dtype=torch.long, device=bundle.device)
        context_mask = torch.tensor([context_enc["attention_mask"]], dtype=torch.long, device=bundle.device)
        question_ids = torch.tensor([question_enc["input_ids"]], dtype=torch.long, device=bundle.device)
        question_mask = torch.tensor([question_enc["attention_mask"]], dtype=torch.long, device=bundle.device)
        span_mask = torch.tensor([[0 if s == e else 1 for s, e in context_enc["offset_mapping"]]], dtype=torch.long, device=bundle.device)

        with torch.no_grad():
            start_logits, end_logits = bundle.model(
                context_ids=context_ids,
                question_ids=question_ids,
                context_mask=context_mask,
                question_mask=question_mask,
            )
            start_logits = _safe_mask_logits(start_logits, span_mask)
            end_logits = _safe_mask_logits(end_logits, span_mask)
            pred_start, pred_end = best_span_from_logits(start_logits, end_logits, max_answer_len=max_answer_len)
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

        start_idx = int(pred_start.item())
        end_idx = int(pred_end.item())
        offsets = context_enc["offset_mapping"]
        if start_idx >= len(offsets) or end_idx >= len(offsets):
            return {"answer": "", "confidence": 0.0, "start_index": start_idx, "end_index": end_idx, "start_char": None, "end_char": None}

        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]
        if end_char <= start_char:
            answer = ""
            start_char = None
            end_char = None
        else:
            answer = context[start_char:end_char]

        confidence = float(start_probs[0, start_idx].item() * end_probs[0, end_idx].item())
        return {
            "answer": answer,
            "confidence": confidence,
            "start_index": start_idx,
            "end_index": end_idx,
            "start_char": start_char,
            "end_char": end_char,
        }

    def _predict_glove(self, bundle: LoadedModelBundle, question: str, context: str, max_answer_len: int = 30) -> Dict[str, Any]:
        config = bundle.config
        max_context_len = int(config.get("max_context_len", 256))
        max_question_len = int(config.get("max_question_len", 32))
        q_tokens, _ = basic_tokenize_with_spans(question)
        c_tokens, c_spans = basic_tokenize_with_spans(context)

        q_tokens = q_tokens[:max_question_len]
        c_tokens = c_tokens[:max_context_len]
        c_spans = c_spans[:max_context_len]

        if not q_tokens or not c_tokens:
            return {"answer": "", "confidence": 0.0, "start_index": 0, "end_index": 0, "start_char": None, "end_char": None}

        question_ids = torch.tensor([[bundle.vocab.token_to_id(t) for t in q_tokens]], dtype=torch.long, device=bundle.device)
        context_ids = torch.tensor([[bundle.vocab.token_to_id(t) for t in c_tokens]], dtype=torch.long, device=bundle.device)
        question_mask = torch.ones_like(question_ids, device=bundle.device)
        context_mask = torch.ones_like(context_ids, device=bundle.device)

        with torch.no_grad():
            start_logits, end_logits = bundle.model(
                context_ids=context_ids,
                question_ids=question_ids,
                context_mask=context_mask,
                question_mask=question_mask,
            )
            pred_start, pred_end = best_span_from_logits(start_logits, end_logits, max_answer_len=max_answer_len)
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

        start_idx = int(pred_start.item())
        end_idx = int(pred_end.item())
        if start_idx >= len(c_spans) or end_idx >= len(c_spans):
            return {"answer": "", "confidence": 0.0, "start_index": start_idx, "end_index": end_idx, "start_char": None, "end_char": None}

        start_char = c_spans[start_idx][0]
        end_char = c_spans[end_idx][1]
        answer = context[start_char:end_char] if end_char > start_char else ""
        confidence = float(start_probs[0, start_idx].item() * end_probs[0, end_idx].item())
        return {
            "answer": answer,
            "confidence": confidence,
            "start_index": start_idx,
            "end_index": end_idx,
            "start_char": start_char,
            "end_char": end_char,
        }
