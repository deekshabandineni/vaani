"""Synthetic tone 'speech' for tests (numpy only, no PyTorch).

Every phone is a distinctive pair of tones, so a small model can learn it in
minutes on CPU. Used by the training smoke test, the ML service tests and the
browser e2e test. Not real speech.
"""
from __future__ import annotations

import random

import numpy as np

from .features import SAMPLE_RATE
from .phonemes import PHONES


def tone_for(phone: str) -> tuple[float, float]:
    k = PHONES.index(phone)
    return 250.0 + 90.0 * k, 4200.0 - 55.0 * k


def synthesize(phones: list[str], rng: random.Random) -> np.ndarray:
    chunks = [np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)]
    for p in phones:
        f1, f2 = tone_for(p)
        dur = rng.uniform(0.07, 0.14)
        t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
        env = np.hanning(t.size)
        amp = rng.uniform(0.2, 0.6)
        chunks.append((amp * env * (np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t))).astype(np.float32))
        chunks.append(np.zeros(int(rng.uniform(0.0, 0.04) * SAMPLE_RATE), dtype=np.float32))
    chunks.append(np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(chunks)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal(audio.size).astype(np.float32)
    return audio + np.float32(0.003) * noise
