"""ONNX Runtime inference: audio + reference text -> pronunciation report.

Used by the FastAPI service, the evaluation script and the benchmark. Needs
only numpy + onnxruntime (no PyTorch), which keeps the server image small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import log_mel, trim_silence
from .gop import Calibration, score_utterance
from .phonemes import text_to_phones


class OnnxScorer:
    def __init__(self, model_path: str | Path, calibration: str | Path | None = None, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.calibration = Calibration.load(calibration)
        self.model_path = str(model_path)

    def logits(self, audio: np.ndarray, trim: bool = True) -> np.ndarray:
        if trim:
            audio = trim_silence(audio)
        feats = log_mel(audio)[None]
        return self.session.run(["logits"], {"feats": feats})[0][0]

    def assess_words(self, audio: np.ndarray, words: list[dict], trim: bool = True) -> dict:
        return score_utterance(self.logits(audio, trim), words, self.calibration)

    def assess(self, audio: np.ndarray, reference_text: str, trim: bool = True) -> dict:
        words, oov = text_to_phones(reference_text)
        if not words:
            return {"ok": False, "error": "no scorable words in reference text", "oov": oov, "words": []}
        report = self.assess_words(audio, words, trim)
        report["oov"] = oov
        return report
