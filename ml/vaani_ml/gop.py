"""Goodness-of-Pronunciation scoring on top of a CTC phoneme recognizer.

Pipeline
1. The acoustic model gives frame-level log posteriors lp[T, C] (C = blank + 39).
2. CTC Viterbi forced alignment of the *expected* phoneme sequence assigns every
   expected phone a span of frames.
3. For each span we renormalise posteriors over non-blank classes and compute
       GOP = mean_t [ log P(expected | t) - max_q log P(q | t) ]
   which is 0 when the expected phone is the most likely phone and negative
   otherwise.
4. A logistic calibration maps GOP to a 0-100 score. Its parameters are fitted
   against human ratings in evaluate.py (speechocean762) and stored in
   calibration.json; the defaults below are only a sensible prior.

client/src/core/gop.js is a line-by-line port; keep them in sync.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phonemes import IPA, PHONE_TO_ID, TOKENS

NEG_INF = -1e30
MISPRONOUNCED_BELOW = 60.0


@dataclass
class Calibration:
    a: float = 3.0
    b: float = 3.0

    def score(self, gop: float) -> float:
        return float(100.0 / (1.0 + np.exp(-(self.a * gop + self.b))))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if path and Path(path).exists():
            d = json.loads(Path(path).read_text())
            return cls(a=float(d["a"]), b=float(d["b"]))
        return cls()


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def ctc_force_align(lp: np.ndarray, targets: list[int], blank: int = 0) -> np.ndarray | None:
    """Viterbi alignment. Returns for each frame the index of the target phone
    (0..N-1) it is assigned to, or -1 for blank frames. None if infeasible."""
    T = lp.shape[0]
    N = len(targets)
    if N == 0:
        return None
    ext = np.full(2 * N + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    S = ext.size
    repeats = sum(1 for i in range(1, N) if targets[i] == targets[i - 1])
    if T < N + repeats:
        return None

    # skip transition s-2 -> s allowed when ext[s] is non-blank and differs from ext[s-2]
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    alpha = np.full(S, NEG_INF)
    alpha[0] = lp[0, ext[0]]
    alpha[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate([[NEG_INF], alpha[:-1]])
        prev2 = np.where(can_skip, np.concatenate([[NEG_INF, NEG_INF], alpha[:-2]]), NEG_INF)
        cand = np.stack([stay, prev1, prev2])
        choice = cand.argmax(axis=0)
        alpha = cand[choice, np.arange(S)] + lp[t, ext]
        back[t] = choice

    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= NEG_INF / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= back[t, s]
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


def phone_spans(assign: np.ndarray, n: int) -> list[tuple[int, int]]:
    spans = []
    for i in range(n):
        idx = np.where(assign == i)[0]
        spans.append((int(idx[0]), int(idx[-1]) + 1))
    return spans


def score_utterance(
    logits_or_lp: np.ndarray,
    words: list[dict],
    calibration: Calibration | None = None,
) -> dict:
    """words: [{"word": str, "phones": [ARPAbet]}] (from phonemes.text_to_phones)."""
    cal = calibration or Calibration()
    lp = log_softmax(np.asarray(logits_or_lp, dtype=np.float64), axis=-1)
    flat = [p for w in words for p in w["phones"]]
    targets = [PHONE_TO_ID[p] for p in flat]
    assign = ctc_force_align(lp, targets)
    if assign is None:
        return {"ok": False, "error": "audio too short for the reference text", "words": []}

    nb = log_softmax(lp[:, 1:], axis=-1)  # renormalise over the 39 phones
    spans = phone_spans(assign, len(flat))

    out_words, k, all_scores = [], 0, []
    for w in words:
        phones = []
        for p in w["phones"]:
            s, e = spans[k]
            seg = nb[s:e]
            pid = PHONE_TO_ID[p] - 1
            gop = float(np.mean(seg[:, pid] - seg.max(axis=1)))
            best = int(seg.sum(axis=0).argmax())
            score = cal.score(gop)
            heard = TOKENS[best + 1] if (best != pid and score < MISPRONOUNCED_BELOW) else None
            phones.append({
                "phone": p, "ipa": IPA[p], "gop": round(gop, 4), "score": round(score, 1),
                "heard": heard, "start": s, "end": e,
            })
            all_scores.append(score)
            k += 1
        wscore = float(np.mean([ph["score"] for ph in phones]))
        out_words.append({"word": w["word"], "score": round(wscore, 1), "phones": phones})

    return {
        "ok": True,
        "source": "local",
        "overall": {"accuracy": round(float(np.mean(all_scores)), 1)},
        "words": out_words,
    }
