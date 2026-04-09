import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def basic_tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens = []
    spans = []
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        tokens.append(match.group(0).lower())
        spans.append(match.span())
    return tokens, spans


def find_token_span(spans: List[Tuple[int, int]], answer_start: int, answer_end: int) -> Optional[Tuple[int, int]]:
    token_start = None
    token_end = None
    for i, (s, e) in enumerate(spans):
        if token_start is None and s <= answer_start < e:
            token_start = i
        if s < answer_end <= e:
            token_end = i
            break
        if answer_start <= s and e <= answer_end and token_start is None:
            token_start = i
        if answer_start <= s and e <= answer_end:
            token_end = i
    if token_start is None or token_end is None:
        return None
    if token_start > token_end:
        return None
    return token_start, token_end


def load_squad(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa["answers"]
                gold_texts = [a["text"] for a in answers]
                first_answer = answers[0]
                examples.append(
                    {
                        "id": qa["id"],
                        "context": context,
                        "question": qa["question"],
                        "answer_text": first_answer["text"],
                        "answer_start": int(first_answer["answer_start"]),
                        "answer_end": int(first_answer["answer_start"] + len(first_answer["text"])),
                        "gold_answers": gold_texts,
                    }
                )
    return examples


class Vocab:
    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, examples: List[Dict], min_freq: int = 2):
        counter = Counter()
        for ex in examples:
            q_tokens, _ = basic_tokenize_with_spans(ex["question"])
            c_tokens, _ = basic_tokenize_with_spans(ex["context"])
            counter.update(q_tokens)
            counter.update(c_tokens)

        itos = [PAD_TOKEN, UNK_TOKEN]
        for token, freq in counter.items():
            if freq >= min_freq:
                itos.append(token)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self):
        return len(self.itos)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK_TOKEN])


def load_glove_embeddings(glove_path: str, vocab: Vocab, embedding_dim: int = 100) -> np.ndarray:
    embeddings = np.random.normal(0.0, 0.02, size=(len(vocab), embedding_dim)).astype(np.float32)
    embeddings[vocab.stoi[PAD_TOKEN]] = 0.0

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab.stoi:
                vector = np.asarray(parts[1:], dtype=np.float32)
                if vector.shape[0] == embedding_dim:
                    embeddings[vocab.stoi[word]] = vector
                    found += 1
    print(f"[INFO] Loaded {found} GloVe vectors into the vocabulary.")
    return embeddings


class GloveQADataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        vocab: Vocab,
        max_context_len: int = 300,
        max_question_len: int = 30,
    ):
        self.vocab = vocab
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.samples = []

        skipped = 0
        for ex in examples:
            q_tokens, _ = basic_tokenize_with_spans(ex["question"])
            c_tokens, c_spans = basic_tokenize_with_spans(ex["context"])
            span = find_token_span(c_spans, ex["answer_start"], ex["answer_end"])
            if span is None:
                skipped += 1
                continue
            start_idx, end_idx = span
            if start_idx >= max_context_len or end_idx >= max_context_len:
                skipped += 1
                continue

            q_tokens = q_tokens[:max_question_len]
            c_tokens = c_tokens[:max_context_len]

            self.samples.append(
                {
                    "id": ex["id"],
                    "question_ids": [vocab.token_to_id(t) for t in q_tokens],
                    "context_ids": [vocab.token_to_id(t) for t in c_tokens],
                    "context_tokens": c_tokens,
                    "question_tokens": q_tokens,
                    "start_positions": start_idx,
                    "end_positions": end_idx,
                    "gold_answers": ex["gold_answers"],
                }
            )
        print(f"[INFO] GloveQADataset kept {len(self.samples)} examples and skipped {skipped}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class BertQADataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        model_name: str = "bert-base-uncased",
        max_context_len: int = 256,
        max_question_len: int = 32,
    ):
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.samples = []

        skipped = 0
        for ex in examples:
            context_enc = self.tokenizer(
                ex["context"],
                truncation=True,
                max_length=max_context_len,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            question_enc = self.tokenizer(
                ex["question"],
                truncation=True,
                max_length=max_question_len,
                return_offsets_mapping=False,
                add_special_tokens=True,
            )

            offsets = context_enc["offset_mapping"]
            answer_start = ex["answer_start"]
            answer_end = ex["answer_end"]

            token_start = None
            token_end = None
            for i, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if token_start is None and s <= answer_start < e:
                    token_start = i
                if s < answer_end <= e:
                    token_end = i
                    break
                if answer_start <= s and e <= answer_end and token_start is None:
                    token_start = i
                if answer_start <= s and e <= answer_end:
                    token_end = i

            if token_start is None or token_end is None:
                skipped += 1
                continue

            span_mask = [0 if s == e else 1 for s, e in offsets]
            self.samples.append(
                {
                    "id": ex["id"],
                    "context_input_ids": context_enc["input_ids"],
                    "context_attention_mask": context_enc["attention_mask"],
                    "context_span_mask": span_mask,
                    "context_offsets": offsets,
                    "question_input_ids": question_enc["input_ids"],
                    "question_attention_mask": question_enc["attention_mask"],
                    "context_text": ex["context"],
                    "start_positions": token_start,
                    "end_positions": token_end,
                    "gold_answers": ex["gold_answers"],
                }
            )
        print(f"[INFO] BertQADataset kept {len(self.samples)} examples and skipped {skipped}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def pad_1d(sequences: List[List[int]], pad_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = 1
    return out, mask


def glove_collate_fn(batch: List[Dict]) -> Dict:
    question_ids, question_mask = pad_1d([x["question_ids"] for x in batch], pad_value=0)
    context_ids, context_mask = pad_1d([x["context_ids"] for x in batch], pad_value=0)

    return {
        "question_ids": question_ids,
        "question_mask": question_mask,
        "context_ids": context_ids,
        "context_mask": context_mask,
        "start_positions": torch.tensor([x["start_positions"] for x in batch], dtype=torch.long),
        "end_positions": torch.tensor([x["end_positions"] for x in batch], dtype=torch.long),
        "context_tokens": [x["context_tokens"] for x in batch],
        "gold_answers": [x["gold_answers"] for x in batch],
        "ids": [x["id"] for x in batch],
    }


def bert_collate_fn(batch: List[Dict]) -> Dict:
    question_ids, question_mask = pad_1d([x["question_input_ids"] for x in batch], pad_value=0)
    context_ids, context_mask = pad_1d([x["context_input_ids"] for x in batch], pad_value=0)

    context_span_mask, _ = pad_1d([x["context_span_mask"] for x in batch], pad_value=0)
    return {
        "question_ids": question_ids,
        "question_mask": question_mask,
        "context_ids": context_ids,
        "context_mask": context_mask,
        "context_span_mask": context_span_mask,
        "start_positions": torch.tensor([x["start_positions"] for x in batch], dtype=torch.long),
        "end_positions": torch.tensor([x["end_positions"] for x in batch], dtype=torch.long),
        "context_offsets": [x["context_offsets"] for x in batch],
        "context_text": [x["context_text"] for x in batch],
        "gold_answers": [x["gold_answers"] for x in batch],
        "ids": [x["id"] for x in batch],
    }
