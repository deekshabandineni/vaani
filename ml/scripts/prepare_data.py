"""Convert public corpora into Vaani JSONL manifests.

    python scripts/prepare_data.py librispeech  /data/LibriSpeech/train-clean-100 data/libri_train.jsonl
    python scripts/prepare_data.py speechocean  /data/speechocean762 train data/so762_train.jsonl
    python scripts/prepare_data.py speechocean  /data/speechocean762 test  data/so762_test.jsonl
    python scripts/prepare_data.py l2arctic     /data/l2arctic        data/l2arctic.jsonl [--speakers ASI RRBI SVBI TNI]

Sources
- LibriSpeech (native English, CC BY 4.0): https://www.openslr.org/12
- speechocean762 (non-native English with expert phone scores, CC BY 4.0):
  https://www.openslr.org/101 and https://github.com/jimbozhang/speechocean762
- L2-ARCTIC (non-native English incl. Hindi-L1 speakers; licence requires a
  request form): https://psi.engr.tamu.edu/l2-arctic-corpus/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani_ml.phonemes import strip_stress, text_to_phones  # noqa: E402


def to_record(uid: str, audio: Path, text: str) -> dict | None:
    words, oov = text_to_phones(text)
    if oov or not words:
        return None
    return {"id": uid, "audio": str(audio.resolve()), "text": text,
            "phones": [p for w in words for p in w["phones"]]}


def librispeech(root: Path):
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            rec = to_record(uid, trans.parent / f"{uid}.flac", text)
            if rec:
                yield rec


def speechocean(root: Path, split: str):
    scores = json.loads((root / "resource" / "scores.json").read_text())
    wavs = dict(line.split(maxsplit=1) for line in (root / split / "wav.scp").read_text().splitlines())
    for line in (root / split / "text").read_text().splitlines():
        uid, text = line.split(" ", 1)
        s = scores.get(uid)
        if s is None:
            continue
        words, phones, phone_scores = [], [], []
        for w in s["words"]:
            wp = [strip_stress(p) for p in w["phones"].split()] if isinstance(w["phones"], str) \
                else [strip_stress(p) for p in w["phones"]]
            # speechocean762 marks some phones with '*' / '<unk>'; skip those words
            if any(not p.isalpha() for p in wp):
                continue
            words.append({"word": w["text"].lower(), "phones": wp})
            phones.extend(wp)
            phone_scores.extend(float(x) for x in w["phones-accuracy"])
        if len(phone_scores) != len(phones) or not phones:
            continue
        yield {"id": uid, "audio": str((root / wavs[uid]).resolve()), "text": text,
               "phones": phones, "words": words, "phone_scores": phone_scores,
               "utt_accuracy": s.get("accuracy"), "utt_total": s.get("total")}


def l2arctic(root: Path, speakers: list[str] | None):
    for spk in sorted(p for p in root.iterdir() if p.is_dir()):
        if speakers and spk.name not in speakers:
            continue
        for txt in sorted((spk / "transcript").glob("*.txt")):
            wav = spk / "wav" / f"{txt.stem}.wav"
            rec = to_record(f"{spk.name}_{txt.stem}", wav, txt.read_text().strip())
            if rec and wav.exists():
                yield rec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="corpus", required=True)
    a = sub.add_parser("librispeech"); a.add_argument("root"); a.add_argument("out")
    b = sub.add_parser("speechocean"); b.add_argument("root"); b.add_argument("split"); b.add_argument("out")
    c = sub.add_parser("l2arctic"); c.add_argument("root"); c.add_argument("out")
    c.add_argument("--speakers", nargs="*", help="e.g. Hindi-L1 speakers ASI RRBI SVBI TNI")
    args = ap.parse_args()

    root = Path(args.root)
    gen = {"librispeech": lambda: librispeech(root),
           "speechocean": lambda: speechocean(root, args.split),
           "l2arctic": lambda: l2arctic(root, args.speakers)}[args.corpus]()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for rec in gen:
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} utterances to {args.out}")


if __name__ == "__main__":
    main()
