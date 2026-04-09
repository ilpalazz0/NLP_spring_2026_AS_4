from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast


@dataclass
class LoadedSentimentBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    model_name: str
    num_labels: int
    id2label: Dict[int, str]


class SentimentManager:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle: Optional[LoadedSentimentBundle] = None

    def is_available(self) -> bool:
        required = [
            self.model_dir / "config.json",
            self.model_dir / "model.safetensors",
            self.model_dir / "tokenizer.json",
        ]
        return all(path.exists() for path in required)

    def is_cached(self) -> bool:
        return self.bundle is not None

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer_file = self.model_dir / "tokenizer.json"
        tokenizer_config_file = self.model_dir / "tokenizer_config.json"

        special_tokens = {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }

        if tokenizer_config_file.exists():
            try:
                with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                for key in special_tokens.keys():
                    value = cfg.get(key)
                    if isinstance(value, str) and value.strip():
                        special_tokens[key] = value
            except Exception:
                pass

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            unk_token=special_tokens["unk_token"],
            sep_token=special_tokens["sep_token"],
            pad_token=special_tokens["pad_token"],
            cls_token=special_tokens["cls_token"],
            mask_token=special_tokens["mask_token"],
        )

        # Keep this bounded and practical for inference.
        tokenizer.model_max_length = min(int(getattr(tokenizer, "model_max_length", 512)), 512)

        return tokenizer

    def load_model_once(self) -> LoadedSentimentBundle:
        if self.bundle is not None:
            return self.bundle

        if not self.is_available():
            raise FileNotFoundError(
                "Sentiment model files were not found. Expected config.json, model.safetensors, and tokenizer.json in the sentiment_model folder."
            )

        tokenizer = self._load_tokenizer()

        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()

        model_name = getattr(model.config, "_name_or_path", None) or model.__class__.__name__
        num_labels = int(getattr(model.config, "num_labels", 2))

        raw_id2label = getattr(model.config, "id2label", None) or {}
        id2label: Dict[int, str] = {}
        for key, value in raw_id2label.items():
            try:
                id2label[int(key)] = str(value)
            except Exception:
                continue

        self.bundle = LoadedSentimentBundle(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            model_name=model_name,
            num_labels=num_labels,
            id2label=id2label,
        )
        return self.bundle

    def _normalize_label(self, raw_label: str, index: int, num_labels: int) -> str:
        label = (raw_label or "").strip()
        upper = label.upper()

        if upper in {"NEGATIVE", "POSITIVE", "NEUTRAL"}:
            return label.lower()

        if upper.startswith("LABEL_") and num_labels == 2:
            return "negative" if index == 0 else "positive"

        if upper.startswith("LABEL_") and num_labels == 3:
            mapping = {0: "negative", 1: "neutral", 2: "positive"}
            return mapping.get(index, label.lower())

        return label.lower() if label else f"label_{index}"

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        was_cached = self.bundle is not None
        bundle = self.load_model_once()

        encoded = bundle.tokenizer(
            text,
            truncation=True,
            max_length=min(int(getattr(bundle.tokenizer, "model_max_length", 512)), 512),
            return_tensors="pt",
        )
        encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = bundle.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        label_index = int(torch.argmax(probs).item())
        raw_label = bundle.id2label.get(label_index, f"LABEL_{label_index}")
        label = self._normalize_label(raw_label, label_index, bundle.num_labels)

        scores: List[Dict[str, Any]] = []
        for idx, score in enumerate(probs.tolist()):
            raw = bundle.id2label.get(idx, f"LABEL_{idx}")
            scores.append(
                {
                    "label": self._normalize_label(raw, idx, bundle.num_labels),
                    "score": float(score),
                }
            )

        return {
            "label": label,
            "label_index": label_index,
            "confidence": float(probs[label_index].item()),
            "scores": scores,
            "cached_model": was_cached,
            "model_name": bundle.model_name,
            "device": bundle.device.type,
        }