import numpy as np
import pytest

from vaani_ml import features
from vaani_ml.gop import Calibration, ctc_force_align, score_utterance
from vaani_ml.l1_profiles import CONTRASTS, feedback_for, prior_for
from vaani_ml.phonemes import NUM_CLASSES, PHONE_TO_ID, PHONES, text_to_phones


def test_inventory():
    assert NUM_CLASSES == 40 and len(PHONES) == 39 and PHONE_TO_ID["<b>"] == 0
    assert all(p in PHONES for p in CONTRASTS)


def test_text_to_phones():
    words, oov = text_to_phones("Very well, thanks! Qwzxv")
    assert [w["word"] for w in words] == ["very", "well", "thanks"]
    assert words[0]["phones"] == ["V", "EH", "R", "IY"]
    assert oov == ["qwzxv"]


def test_features_shape_and_norm():
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    f = features.log_mel(audio)
    assert f.shape == (features.num_frames(16000), 80) == (98, 80)
    assert np.allclose(f.mean(0), 0, atol=1e-4)
    assert features.log_mel(np.zeros(10, dtype=np.float32)).shape == (1, 80)


def test_trim_silence():
    sil = np.zeros(16000, dtype=np.float32)
    tone = np.sin(np.linspace(0, 2000, 8000)).astype(np.float32)
    trimmed = features.trim_silence(np.concatenate([sil, tone, sil]))
    assert 8000 <= trimmed.size < 8000 + 2 * 0.2 * 16000


def _peaky_logits(seq, frames_per=4, gap=2):
    rows = []
    for p in seq:
        rows += [("<b>", 8.0)] * gap + [(p, 8.0)] * frames_per
    rows += [("<b>", 8.0)] * gap
    logits = np.zeros((len(rows), NUM_CLASSES))
    for t, (p, v) in enumerate(rows):
        logits[t, PHONE_TO_ID[p]] = v
    return logits


def test_alignment_assigns_every_phone():
    seq = ["V", "EH", "R", "IY"]
    lp = np.log(np.full((30, NUM_CLASSES), 1 / NUM_CLASSES))
    assign = ctc_force_align(lp, [PHONE_TO_ID[p] for p in seq])
    assert set(assign[assign >= 0]) == {0, 1, 2, 3}
    assert ctc_force_align(lp[:2], [1, 2, 3]) is None  # too short
    assert ctc_force_align(lp[:3], [5, 5, 5]) is None  # repeats need blanks between


def test_gop_scores_correct_vs_substituted():
    said = ["W", "EH", "R", "IY"]  # learner said "wery"
    logits = _peaky_logits(said)
    good = score_utterance(logits, [{"word": "wery", "phones": said}])
    bad = score_utterance(logits, [{"word": "very", "phones": ["V", "EH", "R", "IY"]}])
    assert good["ok"] and good["overall"]["accuracy"] > 90
    v = bad["words"][0]["phones"][0]
    assert v["score"] < 20 and v["heard"] == "W"
    assert all(p["score"] > 90 for p in bad["words"][0]["phones"][1:])


def test_calibration_monotonic():
    c = Calibration()
    assert c.score(0.0) > c.score(-1.0) > c.score(-3.0)


def test_l1_helpers():
    assert prior_for("V", "te") > prior_for("V", "other") > prior_for("K", "other")
    fb = feedback_for("V", "W")
    assert fb["known_confusion"] and "lower lip" in fb["tip"]
    assert feedback_for("K", None) is None
