"""End-to-end: train on synthetic data -> export ONNX -> INT8 -> GOP. ~1-2 min on CPU."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vaani_ml import export, train  # noqa: E402
from vaani_ml.data import SyntheticPhoneDataset  # noqa: E402
from vaani_ml.infer import OnnxScorer  # noqa: E402

SUBSET = ["V", "W", "TH", "T", "Z", "JH", "IH", "IY"]


@pytest.mark.slow
def test_train_export_score(tmp_path, monkeypatch):
    # restrict the synthetic task to 8 phones so it learns fast
    orig_init = SyntheticPhoneDataset.__init__
    monkeypatch.setattr(SyntheticPhoneDataset, "__init__",
                        lambda self, **kw: orig_init(self, phones=SUBSET, **kw))
    res = train.train(train.parse_args([
        "--synthetic", "768", "--epochs", "14", "--batch-size", "16", "--d-model", "96",
        "--layers", "2", "--lr", "2e-3", "--out", str(tmp_path / "run")]))
    assert res["best_val_per"] < 0.35

    export.main([res["checkpoint"], "--out", str(tmp_path / "exp")])
    fp32, int8 = OnnxScorer(tmp_path / "exp" / "vaani_fp32.onnx"), OnnxScorer(tmp_path / "exp" / "vaani_int8.onnx")
    assert (tmp_path / "exp" / "vaani_int8.onnx").stat().st_size < (tmp_path / "exp" / "vaani_fp32.onnx").stat().st_size

    swap = {"V": "W", "W": "V", "TH": "T", "T": "TH", "Z": "JH", "JH": "Z", "IH": "IY", "IY": "IH"}
    for scorer in (fp32, int8):
        ok, bad = [], []
        for i in range(20):
            rng = random.Random(900 + i)
            phones = [rng.choice(SUBSET) for _ in range(4)]
            audio = SyntheticPhoneDataset.synthesize(phones, rng)
            ok += [p["score"] for p in scorer.assess_words(audio, [{"word": "x", "phones": phones}])["words"][0]["phones"]]
            wrong = list(phones)
            wrong[1] = swap[phones[1]]
            bad.append(scorer.assess_words(audio, [{"word": "x", "phones": wrong}])["words"][0]["phones"][1]["score"])
        assert np.mean(ok) - np.mean(bad) > 30
