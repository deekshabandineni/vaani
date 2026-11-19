"""Generate shared/*.json consumed by the gateway and the browser client.

    python scripts/build_shared.py
writes shared/phonemes.json, shared/l1_profiles.json and shared/drills.json.
Re-run after editing phonemes.py, l1_profiles.py or drill_sentences.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from vaani_ml import l1_profiles  # noqa: E402
from vaani_ml.phonemes import IPA, PHONES, TOKENS, text_to_phones  # noqa: E402


def main():
    out = ROOT / "shared"
    out.mkdir(exist_ok=True)
    (out / "phonemes.json").write_text(json.dumps({"tokens": TOKENS, "phones": PHONES, "ipa": IPA}, indent=2))
    (out / "l1_profiles.json").write_text(json.dumps({
        "contrasts": l1_profiles.CONTRASTS, "profiles": l1_profiles.PROFILES,
        "default_prior": l1_profiles.DEFAULT_PRIOR}, indent=2, ensure_ascii=False))

    drills = []
    src = Path(__file__).with_name("drill_sentences.txt")
    for line in src.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        focus, text = (s.strip() for s in line.split("|", 1))
        words, oov = text_to_phones(text)
        if oov:
            raise SystemExit(f"OOV words {oov} in: {text}")
        drills.append({"id": f"d{len(drills) + 1:03d}", "text": text,
                       "focus": [] if focus == "MIXED" else focus.split(), "words": words})
    (out / "drills.json").write_text(json.dumps(drills, indent=1))
    print(f"{len(drills)} drills written")


if __name__ == "__main__":
    main()
