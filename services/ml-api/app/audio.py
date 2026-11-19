from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from vaani_ml.features import SAMPLE_RATE, resample


class AudioError(ValueError):
    pass


def decode(data: bytes, max_seconds: float) -> np.ndarray:
    """Decode WAV/FLAC/OGG bytes to mono float32 at 16 kHz."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several types
        raise AudioError(f"could not decode audio: {exc}") from exc
    audio = resample(audio.mean(axis=1), sr)
    if audio.size < 0.3 * SAMPLE_RATE:
        raise AudioError("recording is shorter than 0.3 s")
    if audio.size > max_seconds * SAMPLE_RATE:
        raise AudioError(f"recording is longer than {max_seconds:.0f} s")
    return audio


def to_wav16k(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()
