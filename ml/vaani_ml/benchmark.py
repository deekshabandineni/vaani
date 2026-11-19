"""Edge benchmark: FP32 vs INT8 ONNX models.

    python -m vaani_ml.benchmark exported/vaani_fp32.onnx exported/vaani_int8.onnx \
        [--manifest data/so762_test.jsonl] [--md docs/benchmark.md]

Measures model size, single-thread CPU latency for 3/6/10 s of audio, and how
much quantization changes the output (frame argmax agreement, mean absolute
phone-score difference). Without --manifest it uses synthetic utterances.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .data import SyntheticPhoneDataset, load_audio, read_manifest
from .features import SAMPLE_RATE, log_mel
from .infer import OnnxScorer


def latency_ms(scorer: OnnxScorer, seconds: float, runs: int = 20) -> float:
    feats = log_mel(np.random.default_rng(0).standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32) * 0.1)[None]
    scorer.session.run(["logits"], {"feats": feats})  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        scorer.session.run(["logits"], {"feats": feats})
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def samples(manifest: str | None, n: int):
    if manifest:
        for rec in read_manifest(manifest)[:n]:
            yield load_audio(rec["audio"]), [{"word": "utt", "phones": rec["phones"]}]
    else:
        ds = SyntheticPhoneDataset(n=n, seed=7)
        for i in range(n):
            phones, audio = ds.sample(i)
            yield audio, [{"word": "utt", "phones": phones}]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fp32")
    ap.add_argument("int8")
    ap.add_argument("--manifest")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--md")
    args = ap.parse_args(argv)

    a, b = OnnxScorer(args.fp32), OnnxScorer(args.int8)
    agree, diffs = [], []
    for audio, words in samples(args.manifest, args.n):
        la, lb = a.logits(audio), b.logits(audio)
        agree.append(float(np.mean(la.argmax(-1) == lb.argmax(-1))))
        ra, rb = a.assess_words(audio, words), b.assess_words(audio, words)
        if ra["ok"] and rb["ok"]:
            sa = [p["score"] for p in ra["words"][0]["phones"]]
            sb = [p["score"] for p in rb["words"][0]["phones"]]
            diffs.append(float(np.mean(np.abs(np.array(sa) - np.array(sb)))))

    res = {
        "size_mb": {"fp32": round(Path(args.fp32).stat().st_size / 1e6, 2),
                    "int8": round(Path(args.int8).stat().st_size / 1e6, 2)},
        "latency_ms": {f"{s}s": {"fp32": round(latency_ms(a, s), 1), "int8": round(latency_ms(b, s), 1)}
                       for s in (3, 6, 10)},
        "frame_argmax_agreement": round(float(np.mean(agree)), 4),
        "mean_abs_phone_score_diff": round(float(np.mean(diffs)), 2) if diffs else None,
        "data": args.manifest or "synthetic",
    }
    print(json.dumps(res, indent=2))
    if args.md:
        lines = ["| Metric | FP32 | INT8 |", "|---|---|---|",
                 f"| Model size (MB) | {res['size_mb']['fp32']} | {res['size_mb']['int8']} |"]
        for k, v in res["latency_ms"].items():
            lines.append(f"| Latency, {k} audio (ms, 1 CPU thread) | {v['fp32']} | {v['int8']} |")
        lines += ["", f"Frame argmax agreement: {res['frame_argmax_agreement']:.2%}  ",
                  f"Mean |phone score diff|: {res['mean_abs_phone_score_diff']}  ", f"Data: {res['data']}"]
        Path(args.md).write_text("\n".join(lines) + "\n")
    return res


if __name__ == "__main__":
    main()
