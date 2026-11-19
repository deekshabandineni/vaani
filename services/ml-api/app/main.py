"""Vaani ML service.

POST /assess     audio + reference text -> pronunciation report (local, azure or fused)
GET  /phonemize  text -> words with ARPAbet phones (lets clients score custom sentences on-device)
GET  /health     which scorers are available
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from vaani_ml.infer import OnnxScorer
from vaani_ml.phonemes import text_to_phones

from . import azure_client
from .audio import AudioError, decode, to_wav16k
from .config import Settings
from .fusion import attach_feedback, fuse

log = logging.getLogger("vaani.ml")
Mode = Literal["auto", "local", "azure", "fused"]


class State:
    settings: Settings
    scorer: OnnxScorer | None = None


state = State()


def load(settings: Settings | None = None):
    state.settings = settings or Settings()
    state.scorer = None
    if Path(state.settings.model_path).exists():
        state.scorer = OnnxScorer(state.settings.model_path, state.settings.calibration_path, state.settings.threads)
        log.info("loaded local model %s", state.settings.model_path)
    else:
        log.warning("no local model at %s; local scoring disabled", state.settings.model_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not hasattr(state, "settings"):  # tests call load() themselves
        load()
    yield


app = FastAPI(title="Vaani ML service", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "local_model": state.scorer is not None,
            "azure": state.settings.azure_enabled,
            "model": Path(state.settings.model_path).name if state.scorer else None}


@app.get("/phonemize")
def phonemize(text: str):
    words, oov = text_to_phones(text)
    return {"words": words, "oov": oov}


@app.post("/assess")
async def assess(audio: UploadFile = File(...), reference_text: str = Form(...),
                 l1: str = Form("other"), mode: Mode = Form("auto")):
    s = state.settings
    if not reference_text.strip():
        raise HTTPException(422, "reference_text is empty")
    try:
        samples = decode(await audio.read(), s.max_audio_seconds)
    except AudioError as exc:
        raise HTTPException(422, str(exc)) from exc

    want_local = mode in ("auto", "local", "fused")
    want_azure = mode in ("auto", "azure", "fused") and s.azure_enabled
    if mode in ("local", "fused") and state.scorer is None:
        raise HTTPException(503, "local model not loaded")
    if mode in ("azure", "fused") and not s.azure_enabled:
        raise HTTPException(503, "Azure Speech is not configured")
    if state.scorer is None and not want_azure:
        raise HTTPException(503, "no scorer available: add a model or configure Azure")

    timings, errors = {}, {}

    async def run_local():
        t0 = time.perf_counter()
        r = await asyncio.to_thread(state.scorer.assess, samples, reference_text)
        timings["local_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return r

    async def run_azure():
        t0 = time.perf_counter()
        try:
            return await azure_client.assess(to_wav16k(samples), reference_text, s.azure_key,
                                             s.azure_region, s.azure_language)
        except azure_client.AzureError as exc:
            errors["azure"] = str(exc)
            log.warning("azure failed: %s", exc)
            return None
        finally:
            timings["azure_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    local_task = run_local() if want_local and state.scorer else asyncio.sleep(0, result=None)
    azure_task = run_azure() if want_azure else asyncio.sleep(0, result=None)
    local, az = await asyncio.gather(local_task, azure_task)

    if local and not local.get("ok"):
        if az is None:
            raise HTTPException(422, local.get("error", "could not score recording"))
        local = None
    if local and az:
        report = fuse(local, az, s.azure_weight)
    elif local:
        report = local
    elif az:
        report = az
    else:
        raise HTTPException(502, errors.get("azure", "scoring failed"))

    report = attach_feedback(report, l1)
    report["reference_text"] = reference_text
    report["timings"] = timings
    if errors:
        report["errors"] = errors
    return report
