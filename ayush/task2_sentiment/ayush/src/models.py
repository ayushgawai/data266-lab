"""Three classifiers. Embeddings are random and learned. No pretrained vectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab: int, emb: int = 128, n_filter: int = 100, kernels=(3, 4, 5), dropout: float = 0.5, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb, n_filter, k) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filter * len(kernels), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        h = self.emb(x).transpose(1, 2)
        pools = [F.relu(conv(h)).amax(dim=-1) for conv in self.convs]
        return self.fc(self.drop(torch.cat(pools, dim=-1))).squeeze(-1)


class BiLSTMPool(nn.Module):
    def __init__(self, vocab: int, emb: int = 128, hidden: int = 128, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def _lstm_out(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = self._lstm_out(x, lengths)
        mask = torch.arange(out.size(1), device=out.device)[None, :] >= lengths[:, None]
        out = out.masked_fill(mask[:, :, None], float("-inf"))
        pooled = out.amax(dim=1)
        return self.fc(self.drop(pooled)).squeeze(-1)


class BiLSTMAttn(BiLSTMPool):
    """Same trunk as BiLSTMPool. Max pool is replaced by additive attention."""

    def __init__(self, vocab: int, emb: int = 128, hidden: int = 128, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__(vocab, emb, hidden, dropout, pad_idx)
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_attn: bool = False):
        out = self._lstm_out(x, lengths)
        scores = self.v(torch.tanh(self.attn(out))).squeeze(-1)
        mask = torch.arange(out.size(1), device=out.device)[None, :] >= lengths[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logits = self.fc(self.drop(context)).squeeze(-1)
        if return_attn:
            return logits, weights
        return logits
