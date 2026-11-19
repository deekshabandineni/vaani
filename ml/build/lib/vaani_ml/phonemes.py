"""Phoneme inventory and text -> phoneme conversion.

We use the 39 ARPAbet phonemes from CMUdict with stress removed, plus a CTC
blank at index 0. The same inventory is exported to shared/phonemes.json so
the browser and the Node gateway agree on indices.
"""
from __future__ import annotations

import re
from functools import lru_cache

BLANK = "<b>"
PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
TOKENS = [BLANK] + PHONES
PHONE_TO_ID = {p: i for i, p in enumerate(TOKENS)}
ID_TO_PHONE = {i: p for p, i in PHONE_TO_ID.items()}
NUM_CLASSES = len(TOKENS)  # 40

# A few IPA symbols so the UI can show learners something familiar.
IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f",
    "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_WORD_RE = re.compile(r"[a-z']+")


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone).upper()


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, list[list[str]]]:
    import cmudict  # bundled data, no network needed

    return cmudict.dict()


def normalize_text(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("’", "'"))


def word_to_phones(word: str) -> list[str] | None:
    prons = _cmudict().get(word.lower())
    if not prons:
        return None
    return [strip_stress(p) for p in prons[0]]


def text_to_phones(text: str) -> tuple[list[dict], list[str]]:
    """Return ([{word, phones}], oov_words)."""
    words, oov = [], []
    for w in normalize_text(text):
        phones = word_to_phones(w)
        if phones is None:
            oov.append(w)
            continue
        words.append({"word": w, "phones": phones})
    return words, oov


def encode(phones: list[str]) -> list[int]:
    return [PHONE_TO_ID[p] for p in phones]
