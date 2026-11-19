"""Log-mel feature extraction in plain numpy.

This is deliberately framework-free and simple so it can be re-implemented
line-for-line in JavaScript (client/src/core/features.js) for on-device
inference. tests/test_parity.py checks that both implementations agree.

Spec: 16 kHz mono, pre-emphasis 0.97, 25 ms Hann window (400 samples),
10 ms hop (160), 512-point FFT, power spectrum, 80 HTK-mel triangular filters
from 20 Hz to 7600 Hz, log(x + 1e-6), per-utterance mean/variance norm.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
F_MIN = 20.0
F_MAX = 7600.0
PREEMPH = 0.97
LOG_EPS = 1e-6


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """[N_MELS, N_FFT//2 + 1] triangular filters."""
    n_bins = N_FFT // 2 + 1
    mel_points = np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bin_freqs = np.arange(n_bins) * SAMPLE_RATE / N_FFT
    fb = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for m in range(N_MELS):
        lo, ce, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (bin_freqs - lo) / (ce - lo)
        down = (hi - bin_freqs) / (hi - ce)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


@lru_cache(maxsize=1)
def hann_window() -> np.ndarray:
    n = np.arange(WIN_LENGTH)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN_LENGTH)  # periodic Hann


def num_frames(num_samples: int) -> int:
    return 1 + max(0, num_samples - WIN_LENGTH) // HOP_LENGTH


def log_mel(audio: np.ndarray, normalize: bool = True) -> np.ndarray:
    """audio: float array in [-1, 1] at 16 kHz. Returns [T, 80] float32."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size < WIN_LENGTH:
        x = np.pad(x, (0, WIN_LENGTH - x.size))
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]])
    t = num_frames(x.size)
    idx = np.arange(WIN_LENGTH)[None, :] + HOP_LENGTH * np.arange(t)[:, None]
    frames = x[idx] * hann_window()[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = spec.real**2 + spec.imag**2
    mel = power @ mel_filterbank().T
    feats = np.log(mel + LOG_EPS)
    if normalize:
        feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-5)
    return feats.astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_db: float = -40.0, pad_ms: int = 150) -> np.ndarray:
    """Energy-based trim of leading/trailing silence (same rule as the JS VAD)."""
    frame = 320  # 20 ms
    if audio.size < frame:
        return audio
    n = audio.size // frame
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return audio
    pad = int(pad_ms / 20)
    start = max(0, (voiced[0] - pad) * frame)
    end = min(audio.size, (voiced[-1] + 1 + pad) * frame)
    return audio[start:end]
