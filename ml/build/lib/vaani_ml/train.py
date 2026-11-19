"""Train the CTC phoneme recognizer.

Real data (GPU recommended):
    python -m vaani_ml.train --train data/libri_train.jsonl data/l2arctic.jsonl \
        --val data/so762_test.jsonl --epochs 40 --batch-size 32 --out runs/base

Smoke test on synthetic tones (CPU, ~1 min):
    python -m vaani_ml.train --synthetic 1024 --epochs 6 --d-model 96 --layers 2 --out runs/synth
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import ManifestDataset, SyntheticPhoneDataset, make_collate
from .model import PhonemeRecognizer, count_parameters


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    best = logits.argmax(-1)
    out = []
    for seq, n in zip(best, lengths):
        prev, hyp = -1, []
        for t in seq[: int(n)].tolist():
            if t != prev and t != 0:
                hyp.append(t)
            prev = t
        out.append(hyp)
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


@torch.no_grad()
def evaluate_per(model, loader, device) -> float:
    model.eval()
    errs = total = 0
    for feats, lengths, targets, tlens in loader:
        logits = model(feats.to(device), lengths.to(device))
        hyps = greedy_decode(logits.cpu(), model.output_lengths(lengths))
        offs = 0
        for hyp, tl in zip(hyps, tlens.tolist()):
            ref = targets[offs: offs + tl].tolist()
            offs += tl
            errs += edit_distance(hyp, ref)
            total += len(ref)
    model.train()
    return errs / max(1, total)


def build_datasets(args):
    if args.synthetic:
        train = SyntheticPhoneDataset(n=args.synthetic, seed=1)
        val = SyntheticPhoneDataset(n=max(64, args.synthetic // 8), seed=2)
        return train, val
    train = ConcatDataset([ManifestDataset(m) for m in args.train])
    val = ManifestDataset(args.val) if args.val else None
    return train, val


def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=make_collate(augment=not args.no_augment), num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=make_collate(False),
                        num_workers=args.workers) if val_ds else None

    model = PhonemeRecognizer(d_model=args.d_model, n_layers=args.layers, heads=args.heads,
                              dropout=args.dropout).to(device)
    print(f"parameters: {count_parameters(model):,}  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(train_dl)
    warmup = max(1, int(0.08 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    history, best_per, step = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for feats, lengths, targets, tlens in train_dl:
            feats, lengths = feats.to(device), lengths.to(device)
            logits = model(feats, lengths)
            lp = logits.log_softmax(-1).transpose(0, 1)  # [T, B, C]
            loss = ctc(lp, targets.to(device), model.output_lengths(lengths), tlens.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
        per = evaluate_per(model, val_dl, device) if val_dl else float("nan")
        rec = {"epoch": epoch, "loss": running / len(train_dl), "val_per": per, "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        if not val_dl or per < best_per:
            best_per = per
            torch.save({"state_dict": model.state_dict(), "config": model.config}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_per": best_per, "history": history, "checkpoint": str(out / "best.pt")}


def load_checkpoint(path: str | Path, device: str = "cpu") -> PhonemeRecognizer:
    ckpt = torch.load(path, map_location=device)
    model = PhonemeRecognizer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--val")
    ap.add_argument("--synthetic", type=int, default=0, help="train on N synthetic utterances instead")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/base")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
