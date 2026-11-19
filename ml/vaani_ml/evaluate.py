"""Evaluate GOP scores against human ratings and fit the score calibration.

    python -m vaani_ml.evaluate exported/vaani_int8.onnx data/so762_test.jsonl \
        --fit-on data/so762_train.jsonl --calibration-out exported/calibration.json

Reports (speechocean762 conventions):
- phone-level Pearson correlation (PCC) between GOP and expert phone accuracy (0-2)
- utterance-level PCC between our mean phone score and expert utterance accuracy (0-10)
- mispronunciation detection precision / recall / F1 (expert score < 1.5 = error)
The calibration is a 1-D logistic regression from GOP to the expert score / 2,
so a calibrated "80" means roughly what an expert rater would call 80%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

from .data import load_audio, read_manifest
from .gop import Calibration
from .infer import OnnxScorer


def collect(scorer: OnnxScorer, manifest: str, limit: int | None = None):
    gops, human, utt_pred, utt_human = [], [], [], []
    for rec in read_manifest(manifest)[:limit]:
        report = scorer.assess_words(load_audio(rec["audio"]), rec["words"])
        if not report["ok"]:
            continue
        phones = [p for w in report["words"] for p in w["phones"]]
        gops.extend(p["gop"] for p in phones)
        human.extend(rec["phone_scores"])
        if rec.get("utt_accuracy") is not None:
            utt_pred.append(report["overall"]["accuracy"])
            utt_human.append(rec["utt_accuracy"])
    return np.array(gops), np.array(human), np.array(utt_pred), np.array(utt_human)


def fit_calibration(gops: np.ndarray, human: np.ndarray) -> Calibration:
    y = np.clip(human / 2.0, 1e-3, 1 - 1e-3)

    def bce(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * gops + b)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(bce, x0=[3.0, 3.0], method="Nelder-Mead")
    return Calibration(a=float(res.x[0]), b=float(res.x[1]))


def detection_metrics(gops, human, cal: Calibration, threshold: float = 60.0):
    pred_err = np.array([cal.score(g) < threshold for g in gops])
    true_err = human < 1.5
    tp = int(np.sum(pred_err & true_err))
    precision = tp / max(1, int(pred_err.sum()))
    recall = tp / max(1, int(true_err.sum()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test_manifest")
    ap.add_argument("--fit-on", help="manifest used to fit calibration (use the train split)")
    ap.add_argument("--calibration-out")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    scorer = OnnxScorer(args.model)
    if args.fit_on:
        g_fit, h_fit, _, _ = collect(scorer, args.fit_on, args.limit)
        cal = fit_calibration(g_fit, h_fit)
        if args.calibration_out:
            Path(args.calibration_out).write_text(json.dumps({"a": cal.a, "b": cal.b}, indent=2))
        scorer.calibration = cal
    gops, human, utt_pred, utt_human = collect(scorer, args.test_manifest, args.limit)
    results = {
        "phones": int(gops.size),
        "phone_pcc": round(float(pearsonr(gops, human)[0]), 4),
        "utterance_pcc": round(float(pearsonr(utt_pred, utt_human)[0]), 4) if utt_pred.size > 2 else None,
        "detection": detection_metrics(gops, human, scorer.calibration),
        "calibration": {"a": scorer.calibration.a, "b": scorer.calibration.b},
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
