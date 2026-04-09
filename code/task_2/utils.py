import json
import os
import random
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_status() -> torch.device:
    device = get_device()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU found and will be used: {gpu_name}")
    else:
        print("[INFO] GPU not found. Training will run on CPU.")
    return device


@dataclass
class AverageMeter:
    value: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.total += value * n
        self.count += n
        self.avg = self.total / max(1, self.count)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(text: str) -> str:
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s):
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def em_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def compute_em_f1(prediction: str, gold_answers: List[str]) -> Tuple[float, float]:
    return (
        metric_max_over_ground_truths(em_score, prediction, gold_answers),
        metric_max_over_ground_truths(f1_score, prediction, gold_answers),
    )


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=logits.dtype)
    if mask.dtype != torch.bool:
        mask = mask != 0
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=dim)


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (arange < lengths.unsqueeze(1)).long()


def best_span_from_logits(start_logits: torch.Tensor, end_logits: torch.Tensor, max_answer_len: int = 30):
    """
    start_logits, end_logits: [B, T]
    Returns two tensors [B], [B]
    """
    batch_size, seq_len = start_logits.size()
    best_starts = []
    best_ends = []

    for b in range(batch_size):
        s = start_logits[b]
        e = end_logits[b]
        score_matrix = s.unsqueeze(1) + e.unsqueeze(0)  # [T, T]

        invalid = torch.triu(torch.ones(seq_len, seq_len, device=score_matrix.device), diagonal=0) == 0
        score_matrix = score_matrix.masked_fill(invalid, torch.finfo(score_matrix.dtype).min)

        if max_answer_len is not None:
            too_long = torch.triu(torch.ones(seq_len, seq_len, device=score_matrix.device), diagonal=max_answer_len)
            score_matrix = score_matrix.masked_fill(too_long.bool(), torch.finfo(score_matrix.dtype).min)

        flat_idx = torch.argmax(score_matrix).item()
        start_idx = flat_idx // seq_len
        end_idx = flat_idx % seq_len
        best_starts.append(start_idx)
        best_ends.append(end_idx)

    return torch.tensor(best_starts, device=start_logits.device), torch.tensor(best_ends, device=start_logits.device)


def save_metrics_txt(path: str, history: List[Dict], summary: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# epoch\ttrain_loss\tval_loss\tval_em\tval_f1\ttest_loss\ttest_em\ttest_f1\n")
        for row in history:
            test_loss = row.get("test_loss", float("nan"))
            test_em = row.get("test_em", float("nan"))
            test_f1 = row.get("test_f1", float("nan"))
            f.write(
                f"{row['epoch']}\t{row['train_loss']:.6f}\t{row['val_loss']:.6f}\t{row['val_em']:.4f}\t{row['val_f1']:.4f}\t{test_loss:.6f}\t{test_em:.4f}\t{test_f1:.4f}\n"
            )
        f.write("# summary\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
