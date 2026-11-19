"""Combine the local GOP report with Azure's assessment, and attach coaching tips.

Strategy
- Words are matched in order by text (both systems score the same reference).
- If Azure returns the same phone sequence for a word, each phone score is
  w * azure + (1 - w) * local. Otherwise, the local phone scores are shifted by
  w * (azure_word - local_word) so the word average agrees with the fused word score.
- The local model keeps ownership of "heard as" (substitution) hints, since
  Azure does not say which phone it heard instead.
- `agreement` (Pearson r of word scores) is logged so the two systems can be
  compared across many sessions. That comparison is the basis of the write-up.
"""
from __future__ import annotations

import copy

import numpy as np

from vaani_ml.gop import MISPRONOUNCED_BELOW
from vaani_ml.l1_profiles import PROFILES, feedback_for, prior_for


def _match_words(local_words: list[dict], azure_words: list[dict]) -> list[dict | None]:
    matched, j = [], 0
    for lw in local_words:
        found = None
        for k in range(j, min(j + 3, len(azure_words))):
            if azure_words[k]["word"].strip(".,!?'\"") == lw["word"]:
                found, j = azure_words[k], k + 1
                break
        matched.append(found)
    return matched


def fuse(local: dict, azure: dict, w: float) -> dict:
    out = copy.deepcopy(local)
    out["source"] = "fused"
    pairs = []
    for lw, aw in zip(out["words"], _match_words(local["words"], azure["words"])):
        if aw is None:
            continue
        lw["azure_score"] = aw["score"]
        lw["error_type"] = aw.get("error_type", "None")
        pairs.append((lw["score"], aw["score"]))
        a_phones = [p["phone"] for p in aw["phones"]]
        if a_phones == [p["phone"] for p in lw["phones"]]:
            for lp, ap in zip(lw["phones"], aw["phones"]):
                lp["score"] = round(w * ap["score"] + (1 - w) * lp["score"], 1)
        else:
            shift = w * (aw["score"] - lw["score"])
            for lp in lw["phones"]:
                lp["score"] = round(float(np.clip(lp["score"] + shift, 0, 100)), 1)
        if lw["error_type"] == "Omission":
            for lp in lw["phones"]:
                lp["score"] = 0.0
        for lp in lw["phones"]:
            if lp["score"] >= MISPRONOUNCED_BELOW:
                lp["heard"] = None
        lw["score"] = round(float(np.mean([p["score"] for p in lw["phones"]])), 1)

    phones = [p["score"] for wd in out["words"] for p in wd["phones"]]
    out["overall"] = {**azure.get("overall", {}), "accuracy": round(float(np.mean(phones)), 1),
                      "local_accuracy": local["overall"]["accuracy"],
                      "azure_accuracy": azure.get("overall", {}).get("accuracy")}
    if len(pairs) >= 3 and np.std([p[0] for p in pairs]) > 0 and np.std([p[1] for p in pairs]) > 0:
        out["agreement"] = round(float(np.corrcoef(np.array(pairs).T)[0, 1]), 3)
    out["recognized_text"] = azure.get("recognized_text")
    return out


def attach_feedback(report: dict, l1: str = "other", max_tips: int = 3) -> dict:
    """Pick the weakest phones and add L1-aware tips, weakest first."""
    weak = []
    for wd in report.get("words", []):
        for p in wd.get("phones", []):
            if p["score"] < MISPRONOUNCED_BELOW:
                weak.append((p["score"] - 20 * prior_for(p["phone"], l1), wd["word"], p))
    weak.sort(key=lambda x: x[0])
    tips, seen = [], set()
    for _, word, p in weak:
        if p["phone"] in seen:
            continue
        fb = feedback_for(p["phone"], p.get("heard"))
        if fb:
            seen.add(p["phone"])
            tips.append({**fb, "word": word, "score": p["score"]})
        if len(tips) >= max_tips:
            break
    report["feedback"] = tips
    report["l1"] = l1 if l1 in PROFILES else "other"
    return report
