"""The browser must compute exactly what the server computes.

Runs client/src/core/{features,gop}.js under Node on the same audio/logits and
compares with the Python implementation.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vaani_ml.data import SyntheticPhoneDataset
from vaani_ml.features import log_mel, trim_silence
from vaani_ml.gop import score_utterance
from vaani_ml.phonemes import IPA, NUM_CLASSES, TOKENS, text_to_phones

CORE = Path(__file__).resolve().parents[2] / "client" / "src" / "core"

NODE_SCRIPT = """
import { readFileSync } from "node:fs";
import { logMel, trimSilence } from "%(core)s/features.js";
import { scoreUtterance } from "%(core)s/gop.js";
const inp = JSON.parse(readFileSync(process.argv[2], "utf8"));
const audio = Float32Array.from(inp.audio);
const f = logMel(audio);
const trimmed = trimSilence(audio);
const r = scoreUtterance(Float32Array.from(inp.logits), inp.T, inp.C, inp.words,
                         { tokens: inp.tokens, ipa: inp.ipa });
console.log(JSON.stringify({ frames: f.frames, feats: Array.from(f.data), trimmed: trimmed.length, report: r }));
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_node(tmp_path, payload):
    script = tmp_path / "parity.mjs"
    script.write_text(NODE_SCRIPT % {"core": CORE.as_uri()})
    data = tmp_path / "in.json"
    data.write_text(json.dumps(payload))
    out = subprocess.run(["node", str(script), str(data)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_features_and_gop_match_python(tmp_path):
    rng = random.Random(3)
    audio = SyntheticPhoneDataset.synthesize(["V", "EH", "R", "IY", "W", "EH", "L"], rng)
    audio = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(4000, np.float32)])
    words, _ = text_to_phones("very well")
    T = 60
    logits = np.random.default_rng(1).standard_normal((T, NUM_CLASSES)).astype(np.float32) * 3

    js = run_node(tmp_path, {"audio": audio.tolist(), "logits": logits.reshape(-1).tolist(), "T": T,
                             "C": NUM_CLASSES, "words": words, "tokens": TOKENS, "ipa": IPA})

    py_feats = log_mel(audio)
    assert js["frames"] == py_feats.shape[0]
    np.testing.assert_allclose(np.array(js["feats"]).reshape(py_feats.shape), py_feats, atol=2e-4)
    assert js["trimmed"] == trim_silence(audio).size

    py = score_utterance(logits, words)
    assert js["report"]["overall"]["accuracy"] == pytest.approx(py["overall"]["accuracy"], abs=0.11)
    for jw, pw in zip(js["report"]["words"], py["words"]):
        for jp, pp in zip(jw["phones"], pw["phones"]):
            assert (jp["phone"], jp["start"], jp["end"], jp["heard"]) == (pp["phone"], pp["start"], pp["end"], pp["heard"])
            assert jp["gop"] == pytest.approx(pp["gop"], abs=2e-4)
            assert jp["score"] == pytest.approx(pp["score"], abs=0.11)
