"""Datasets for CTC phoneme training.

All real corpora are first converted to a JSONL manifest by
scripts/prepare_data.py, one utterance per line:
    {"id": "...", "audio": "/abs/path.wav", "text": "...", "phones": ["DH", "AH", ...]}
speechocean762 lines additionally carry human scores used in evaluate.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import SAMPLE_RATE, log_mel, resample
from .phonemes import PHONES, encode


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), sr)


def read_manifest(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, max_seconds: float = 16.0):
        self.items = [r for r in read_manifest(manifest) if r.get("phones")]
        self.max_samples = int(max_seconds * SAMPLE_RATE)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        audio = load_audio(r["audio"])[: self.max_samples]
        return torch.from_numpy(log_mel(audio)), torch.tensor(encode(r["phones"]), dtype=torch.long)


class SyntheticPhoneDataset(Dataset):
    """Toy 'speech' where every phone is a distinctive pair of tones.

    It is learnable in a few hundred steps on CPU, which lets the tests run the
    real training -> GOP -> ONNX export pipeline end-to-end without any corpus.
    """

    def __init__(self, n: int = 512, min_len: int = 3, max_len: int = 8, seed: int = 0,
                 phones: list[str] | None = None):
        self.n, self.min_len, self.max_len, self.seed = n, min_len, max_len, seed
        self.phones = phones or PHONES

    @staticmethod
    def tone_for(phone: str) -> tuple[float, float]:
        k = PHONES.index(phone)
        return 250.0 + 90.0 * k, 4200.0 - 55.0 * k

    @classmethod
    def synthesize(cls, phones: list[str], rng: random.Random) -> np.ndarray:
        chunks = [np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)]
        for p in phones:
            f1, f2 = cls.tone_for(p)
            dur = rng.uniform(0.07, 0.14)
            t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
            env = np.hanning(t.size)
            amp = rng.uniform(0.2, 0.6)
            chunks.append((amp * env * (np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t))).astype(np.float32))
            chunks.append(np.zeros(int(rng.uniform(0.0, 0.04) * SAMPLE_RATE), dtype=np.float32))
        chunks.append(np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32))
        audio = np.concatenate(chunks)
        return audio + np.float32(0.003) * np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal(audio.size).astype(np.float32)

    def sample(self, i: int) -> tuple[list[str], np.ndarray]:
        rng = random.Random(self.seed * 100003 + i)
        phones = [rng.choice(self.phones) for _ in range(rng.randint(self.min_len, self.max_len))]
        return phones, self.synthesize(phones, rng)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        phones, audio = self.sample(i)
        return torch.from_numpy(log_mel(audio)), torch.tensor(encode(phones), dtype=torch.long)


def spec_augment(feats: torch.Tensor, n_freq: int = 2, F: int = 10, n_time: int = 2, T: int = 40) -> torch.Tensor:
    feats = feats.clone()
    t_len, n_mels = feats.shape
    for _ in range(n_freq):
        w = random.randint(0, F)
        f0 = random.randint(0, max(0, n_mels - w))
        feats[:, f0:f0 + w] = 0.0
    for _ in range(n_time):
        w = random.randint(0, min(T, t_len // 5))
        t0 = random.randint(0, max(0, t_len - w))
        feats[t0:t0 + w, :] = 0.0
    return feats


def make_collate(augment: bool):
    def collate(batch):
        feats, targets = zip(*batch)
        if augment:
            feats = [spec_augment(f) for f in feats]
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        padded = torch.zeros(len(feats), int(lengths.max()), feats[0].shape[1])
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
        target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
        return padded, lengths, torch.cat(targets), target_lengths

    return collate
