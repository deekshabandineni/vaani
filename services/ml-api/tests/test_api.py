"""API tests. The fixture model was trained on synthetic tone 'speech' (see
ml/vaani_ml/synth.py), so we synthesise test audio the same way."""
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "services" / "ml-api")]

from fastapi.testclient import TestClient  # noqa: E402

from app import azure_client, main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.fusion import attach_feedback, fuse  # noqa: E402
from vaani_ml.synth import synthesize  # noqa: E402 (numpy only; this service has no PyTorch)

MODEL = Path(__file__).parent / "fixtures" / "synthetic_int8.onnx"


def wav_bytes(phones, seed=0):
    audio = synthesize(phones, random.Random(seed))
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client():
    main.load(Settings(model_path=str(MODEL), calibration_path="", azure_key="", azure_region=""))
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def azure_client_on(monkeypatch):
    main.load(Settings(model_path=str(MODEL), calibration_path="", azure_key="k", azure_region="eastus"))

    async def fake(wav, text, key, region, language="en-US", **_):
        assert wav[:4] == b"RIFF" and key == "k"
        return azure_client.normalize_response(AZURE_SAMPLE)

    monkeypatch.setattr(azure_client, "assess", fake)
    with TestClient(main.app) as c:
        yield c


AZURE_SAMPLE = {
    "RecognitionStatus": "Success", "DisplayText": "Wery well.",
    "NBest": [{
        "Display": "Wery well.",
        "PronunciationAssessment": {"AccuracyScore": 70, "FluencyScore": 88, "CompletenessScore": 100, "PronScore": 78},
        "Words": [
            {"Word": "very", "PronunciationAssessment": {"AccuracyScore": 40, "ErrorType": "Mispronunciation"},
             "Phonemes": [{"Phoneme": "v", "PronunciationAssessment": {"AccuracyScore": 10}},
                          {"Phoneme": "eh", "PronunciationAssessment": {"AccuracyScore": 90}},
                          {"Phoneme": "r", "PronunciationAssessment": {"AccuracyScore": 80}},
                          {"Phoneme": "iy", "PronunciationAssessment": {"AccuracyScore": 95}}]},
            {"Word": "well", "PronunciationAssessment": {"AccuracyScore": 95, "ErrorType": "None"},
             "Phonemes": [{"Phoneme": "w", "PronunciationAssessment": {"AccuracyScore": 95}},
                          {"Phoneme": "eh", "PronunciationAssessment": {"AccuracyScore": 95}},
                          {"Phoneme": "l", "PronunciationAssessment": {"AccuracyScore": 95}}]},
        ]}],
}


def test_health_and_phonemize(client):
    h = client.get("/health").json()
    assert h["local_model"] is True and h["azure"] is False
    p = client.get("/phonemize", params={"text": "very well zzqx"}).json()
    assert p["words"][0]["phones"] == ["V", "EH", "R", "IY"] and p["oov"] == ["zzqx"]


def test_local_assess_flags_v_w_confusion(client):
    # learner says "wery well" for "very well"
    said = ["W", "EH", "R", "IY", "W", "EH", "L"]
    r = client.post("/assess", files={"audio": ("a.wav", wav_bytes(said), "audio/wav")},
                    data={"reference_text": "Very well", "l1": "te"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "local"
    v = body["words"][0]["phones"][0]
    assert v["phone"] == "V" and v["score"] < 60
    assert body["words"][1]["score"] > 60
    assert body["feedback"][0]["phone"] == "V" and "lower lip" in body["feedback"][0]["tip"]


def test_rejects_bad_input(client):
    files = {"audio": ("a.wav", b"not audio", "audio/wav")}
    assert client.post("/assess", files=files, data={"reference_text": "hi"}).status_code == 422
    good = {"audio": ("a.wav", wav_bytes(["V", "EH"]), "audio/wav")}
    assert client.post("/assess", files=good, data={"reference_text": "  "}).status_code == 422
    assert client.post("/assess", files=good, data={"reference_text": "very", "mode": "azure"}).status_code == 503


def test_fused_assess(azure_client_on):
    said = ["W", "EH", "R", "IY", "W", "EH", "L"]
    r = azure_client_on.post("/assess", files={"audio": ("a.wav", wav_bytes(said), "audio/wav")},
                             data={"reference_text": "Very well", "l1": "hi"})
    body = r.json()
    assert body["source"] == "fused"
    assert body["overall"]["fluency"] == 88 and body["overall"]["azure_accuracy"] == 70
    assert body["words"][0]["azure_score"] == 40 and body["words"][0]["phones"][0]["score"] < 40
    assert body["recognized_text"] == "Wery well."
    assert "azure_ms" in body["timings"] and "local_ms" in body["timings"]


def test_azure_failure_falls_back_to_local(azure_client_on, monkeypatch):
    async def boom(*a, **k):
        raise azure_client.AzureError("HTTP 401")

    monkeypatch.setattr(azure_client, "assess", boom)
    r = azure_client_on.post("/assess", files={"audio": ("a.wav", wav_bytes(["V", "EH", "R", "IY"]), "audio/wav")},
                             data={"reference_text": "very"})
    assert r.status_code == 200 and r.json()["source"] == "local" and "401" in r.json()["errors"]["azure"]


def test_fuse_mismatched_phones_shifts_scores():
    local = {"ok": True, "overall": {"accuracy": 80.0},
             "words": [{"word": "data", "score": 80.0, "phones": [
                 {"phone": "D", "score": 80.0, "heard": None}, {"phone": "EY", "score": 80.0, "heard": None},
                 {"phone": "T", "score": 80.0, "heard": None}, {"phone": "AH", "score": 80.0, "heard": None}]}]}
    az = {"overall": {"accuracy": 60.0}, "words": [{"word": "data", "score": 60.0, "error_type": "None",
          "phones": [{"phone": "D", "score": 60.0}, {"phone": "AE", "score": 60.0},
                     {"phone": "T", "score": 60.0}, {"phone": "AH", "score": 60.0}]}]}
    out = fuse(local, az, w=0.5)
    assert out["words"][0]["score"] == 70.0
    out = attach_feedback(out, "te")
    assert out["l1"] == "te" and out["feedback"] == []


def test_normalize_handles_omission_and_insertion():
    resp = {"RecognitionStatus": "Success", "NBest": [{"AccuracyScore": 50, "Words": [
        {"Word": "the", "AccuracyScore": 0, "ErrorType": "Omission"},
        {"Word": "uh", "AccuracyScore": 0, "ErrorType": "Insertion"}]}]}
    out = azure_client.normalize_response(resp)
    assert [w["word"] for w in out["words"]] == ["the"] and out["overall"]["accuracy"] == 50
    with pytest.raises(azure_client.AzureError):
        azure_client.normalize_response({"RecognitionStatus": "NoMatch"})
