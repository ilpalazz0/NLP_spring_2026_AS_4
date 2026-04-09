from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, value: float | None = None) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0

    if value is None:
        value = torch.finfo(tensor.dtype).min
    else:
        if tensor.is_floating_point():
            value = max(value, torch.finfo(tensor.dtype).min)

    return tensor.masked_fill(~mask, value)


class HighwayEncoder(nn.Module):
    def __init__(self, input_size: int, num_layers: int = 2):
        super().__init__()
        self.transforms = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform, gate in zip(self.transforms, self.gates):
            gate_out = torch.sigmoid(gate(x))
            trans_out = torch.relu(transform(x))
            x = gate_out * trans_out + (1 - gate_out) * x
        return x


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return self.dropout(out)


class BiDAFAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.similarity = nn.Linear(hidden_size * 6, 1, bias=False)

    def forward(
        self,
        context: torch.Tensor,
        question: torch.Tensor,
        context_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_len = context.size(1)
        q_len = question.size(1)

        c_exp = context.unsqueeze(2).expand(-1, -1, q_len, -1)
        q_exp = question.unsqueeze(1).expand(-1, c_len, -1, -1)
        c_mul_q = c_exp * q_exp
        sim_input = torch.cat([c_exp, q_exp, c_mul_q], dim=-1)
        sim = self.similarity(sim_input).squeeze(-1)  # [B, C, Q]

        q_mask = question_mask.unsqueeze(1).expand_as(sim)
        sim_q = replace_masked_values(sim, q_mask)
        a = torch.softmax(sim_q, dim=2)
        c2q = torch.bmm(a, question)

        sim_max = torch.max(sim, dim=2).values
        sim_max = replace_masked_values(sim_max, context_mask)
        b = torch.softmax(sim_max, dim=1).unsqueeze(1)
        q2c = torch.bmm(b, context).repeat(1, c_len, 1)

        g = torch.cat([context, c2q, context * c2q, context * q2c], dim=-1)
        return g


class BiDAFCore(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.highway = HighwayEncoder(hidden_size)
        self.contextual = BiLSTMEncoder(hidden_size, hidden_size, dropout)
        self.attention = BiDAFAttention(hidden_size)
        self.modeling = BiLSTMEncoder(hidden_size * 8, hidden_size, dropout)
        self.output_rnn = BiLSTMEncoder(hidden_size * 2, hidden_size, dropout)
        self.start_proj = nn.Linear(hidden_size * 10, 1)
        self.end_proj = nn.Linear(hidden_size * 10, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context_emb: torch.Tensor,
        question_emb: torch.Tensor,
        context_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.highway(self.dropout(self.proj(context_emb)))
        question = self.highway(self.dropout(self.proj(question_emb)))

        context_enc = self.contextual(context, context_mask)
        question_enc = self.contextual(question, question_mask)

        g = self.attention(context_enc, question_enc, context_mask, question_mask)
        m = self.modeling(g, context_mask)
        m2 = self.output_rnn(m, context_mask)

        start_logits = self.start_proj(torch.cat([g, m], dim=-1)).squeeze(-1)
        end_logits = self.end_proj(torch.cat([g, m2], dim=-1)).squeeze(-1)

        start_logits = replace_masked_values(start_logits, context_mask)
        end_logits = replace_masked_values(end_logits, context_mask)
        return start_logits, end_logits


class GloveBiDAF(nn.Module):
    def __init__(self, embedding_matrix, hidden_size: int = 128, dropout: float = 0.2, train_embeddings: bool = True):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = train_embeddings
        self.bidaf = BiDAFCore(input_dim=emb_dim, hidden_size=hidden_size, dropout=dropout)

    def forward(self, context_ids, question_ids, context_mask, question_mask):
        context_emb = self.embedding(context_ids)
        question_emb = self.embedding(question_ids)
        return self.bidaf(context_emb, question_emb, context_mask, question_mask)


class BertBiDAF(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        hidden_size: int = 128,
        dropout: float = 0.2,
        freeze_bert: bool = True,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        self.freeze_bert = freeze_bert

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.bidaf = BiDAFCore(input_dim=bert_dim, hidden_size=hidden_size, dropout=dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(self, context_ids, question_ids, context_mask, question_mask):
        context_emb = self.encode(context_ids, context_mask)
        question_emb = self.encode(question_ids, question_mask)
        return self.bidaf(context_emb, question_emb, context_mask, question_mask)