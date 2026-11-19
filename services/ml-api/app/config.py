from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    model_path: str = field(default_factory=lambda: os.getenv("VAANI_MODEL_PATH", "/models/vaani_fp32.onnx"))
    calibration_path: str = field(default_factory=lambda: os.getenv("VAANI_CALIBRATION", "/models/calibration.json"))
    azure_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", ""))
    azure_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", ""))
    azure_language: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_LANGUAGE", "en-US"))
    azure_weight: float = field(default_factory=lambda: float(os.getenv("FUSION_AZURE_WEIGHT", "0.6")))
    max_audio_seconds: float = field(default_factory=lambda: float(os.getenv("MAX_AUDIO_SECONDS", "20")))
    threads: int = field(default_factory=lambda: int(os.getenv("ORT_THREADS", "2")))

    @property
    def azure_enabled(self) -> bool:
        return bool(self.azure_key and self.azure_region)
