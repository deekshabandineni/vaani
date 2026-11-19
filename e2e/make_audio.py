"""Fake-microphone audio for the browser e2e test.

The e2e stack runs the small model trained on synthetic tone 'speech'
(ml/vaani_ml/synth.py), so the 'learner' here is synthesised the same way:
they say "wery well" (W instead of V) when the sentence is "very well".
"""
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from vaani_ml.synth import synthesize  # noqa: E402

said = ["W", "EH", "R", "IY", "W", "EH", "L"]
speech = synthesize(said, random.Random(11)) * 1.5
audio = np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(48000, np.float32)])
sf.write(sys.argv[1], audio, 16000, subtype="PCM_16")
print(f"wrote {sys.argv[1]} ({audio.size / 16000:.1f} s)")
