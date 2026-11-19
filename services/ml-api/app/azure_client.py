"""Azure AI Speech Pronunciation Assessment over the short-audio REST API.

Docs: https://learn.microsoft.com/azure/ai-services/speech-service/how-to-pronunciation-assessment
The REST API takes up to ~60 s of 16 kHz PCM WAV. The assessment parameters go
in a base64-encoded JSON header. Responses are normalised into the same shape
as the local GOP report so fusion.py can combine them.
"""
from __future__ import annotations

import base64
import json

import httpx

# Azure en-US "SAPI" phone set -> our ARPAbet inventory
SAPI_TO_ARPABET = {
    "aa": "AA", "ae": "AE", "ah": "AH", "ao": "AO", "aw": "AW", "ax": "AH", "ay": "AY", "b": "B",
    "ch": "CH", "d": "D", "dh": "DH", "eh": "EH", "er": "ER", "ey": "EY", "f": "F", "g": "G",
    "h": "HH", "hh": "HH", "ih": "IH", "iy": "IY", "jh": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "ng": "NG", "ow": "OW", "oy": "OY", "p": "P", "r": "R", "s": "S", "sh": "SH",
    "t": "T", "th": "TH", "uh": "UH", "uw": "UW", "v": "V", "w": "W", "y": "Y", "z": "Z", "zh": "ZH",
}


class AzureError(RuntimeError):
    pass


def _pa(obj: dict) -> dict:
    """Scores live under 'PronunciationAssessment' in newer responses, inline in older ones."""
    return obj.get("PronunciationAssessment", obj)


def normalize_response(resp: dict) -> dict:
    if resp.get("RecognitionStatus") not in (None, "Success"):
        raise AzureError(f"recognition failed: {resp.get('RecognitionStatus')}")
    nbest = (resp.get("NBest") or [None])[0]
    if not nbest:
        raise AzureError("no recognition result")
    pa = _pa(nbest)
    words = []
    for w in nbest.get("Words", []):
        wpa = _pa(w)
        if wpa.get("ErrorType") == "Insertion":
            continue
        phones = [{"phone": SAPI_TO_ARPABET.get(str(p.get("Phoneme", "")).lower(), str(p.get("Phoneme", "")).upper()),
                   "score": float(_pa(p).get("AccuracyScore", 0.0))} for p in w.get("Phonemes", [])]
        words.append({"word": w.get("Word", "").lower(), "score": float(wpa.get("AccuracyScore", 0.0)),
                      "error_type": wpa.get("ErrorType", "None"), "phones": phones})
    return {
        "ok": True, "source": "azure",
        "recognized_text": nbest.get("Display") or resp.get("DisplayText"),
        "overall": {k: float(pa[v]) for k, v in [("accuracy", "AccuracyScore"), ("fluency", "FluencyScore"),
                                                 ("completeness", "CompletenessScore"), ("pron", "PronScore")]
                    if v in pa},
        "words": words,
    }


async def assess(wav16k: bytes, reference_text: str, key: str, region: str, language: str = "en-US",
                 client: httpx.AsyncClient | None = None, timeout: float = 15.0) -> dict:
    params = {"ReferenceText": reference_text, "GradingSystem": "HundredMark",
              "Granularity": "Phoneme", "Dimension": "Comprehensive", "EnableMiscue": True}
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Accept": "application/json",
        "Pronunciation-Assessment": base64.b64encode(json.dumps(params).encode()).decode(),
    }
    url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        r = await client.post(url, params={"language": language, "format": "detailed"}, headers=headers, content=wav16k)
    except httpx.HTTPError as exc:
        raise AzureError(f"request failed: {exc}") from exc
    finally:
        if own:
            await client.aclose()
    if r.status_code != 200:
        raise AzureError(f"HTTP {r.status_code}: {r.text[:200]}")
    return normalize_response(r.json())
