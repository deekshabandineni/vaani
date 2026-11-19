"""Compact Conformer-style CTC phoneme recognizer, sized for phones.

Defaults (d_model=192, 6 blocks) give ~5.3M parameters: 21.3 MB as FP32 ONNX
and 6.1 MB after INT8 dynamic quantization (measured). Attention and feed-forward layers are
plain Linear/MatMul ops so ONNX Runtime's dynamic quantizer can quantize them.

Input:  log-mel features [B, T, 80]
Output: logits [B, ceil(T/2), 40]  (20 ms per output frame)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phonemes import NUM_CLASSES


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.silu(self.fc1(self.norm(x))))))


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if key_padding_mask is not None:  # True = padding
            att = att.masked_fill(key_padding_mask[:, None, None, :], -1e4)
        att = self.drop(att.softmax(dim=-1))
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, T, D)
        return self.drop(self.out(y))


class ConvModule(nn.Module):
    """Depthwise conv gives the model relative position information."""

    def __init__(self, d: int, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        y = F.glu(self.pw1(self.norm(x)), dim=-1)
        if pad_mask is not None:
            y = y.masked_fill(pad_mask[..., None], 0.0)
        y = F.silu(self.bn(self.dw(y.transpose(1, 2)))).transpose(1, 2)
        return self.drop(self.pw2(y))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(d, dropout=dropout)
        self.att = SelfAttention(d, heads, dropout)
        self.conv = ConvModule(d, dropout=dropout)
        self.ff2 = FeedForward(d, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, pad_mask)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class PhonemeRecognizer(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 192, n_layers: int = 6,
                 heads: int = 4, dropout: float = 0.1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.config = dict(n_mels=n_mels, d_model=d_model, n_layers=n_layers,
                           heads=heads, dropout=dropout, num_classes=num_classes)
        self.subsample = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=1, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([ConformerBlock(d_model, heads, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return (lengths + 1) // 2

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None):
        x = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        pad_mask = None
        if lengths is not None:
            out_len = self.output_lengths(lengths)
            pad_mask = torch.arange(x.size(1), device=x.device)[None, :] >= out_len[:, None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
